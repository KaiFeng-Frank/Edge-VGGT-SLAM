#!/usr/bin/env python3
"""Append integer context sampling to either pinned DPVO encoder ONNX.

Static editing only: no encoder execution, model loading, or compiler invocation.
DPVO reference: MIT, Copyright (c) 2022 Princeton Vision & Learning Lab.
The software license does not establish a separate neural-weight license.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

OFFICIAL_COMMIT = "0ac95b656d1fda91c271d2a106460d19ad966fc7"
PARENTS = {
    "527fc2493de700a21aee958d7fda4d5ee421f4eb0c92296e81409df289f56a82": {
        "name": "original_v26_encoder", "nodes": 101, "initializers": 44,
        "prior_evidence": "original_RKNN_FP16_strict_FP32_gate_failed",
    },
    "c3c4048c6d8792b09f0d6cbec8228f99d59c60d7503456eea2e369d7b8dbd826": {
        "name": "v27_spatial_norm_diagnostic", "nodes": 201, "initializers": 64,
        "prior_evidence": "spatial_norm_CPU_ORT_strict_FP32_gate_failed",
    },
}
SOURCE_PINS = {
    "dpvo/net.py": "1aee33d54f0b1cc1f3f842b60790ff1299ef9c1e20bca6467ce7114dca802acc",
    "dpvo/dpvo.py": "7efcb45689c2da8d18e2c69774e221799c9f8472e99531955f55dcb145626fd7",
    "dpvo/extractor.py": "f5cbcbe14c42e44bcec6e827a3429ebe93979a439ba9b9ef70d4f4baa3454a2b",
    "evaluate_tum.py": "2548c77758885daa7c527cd7584aceb343442beefdc18ebdd07a1c6fc15fd1e6",
    "LICENSE": "c5b0105dcef65f5423bec0aafe796364a086375e23746cf75402aba0f897489a",
}
IMAGE_SHAPE = [1, 3, 464, 608]
FMAP_SHAPE = [1, 128, 116, 152]
IMAP_SHAPE = [1, 384, 116, 152]
PATCH_COUNT = 96
RTOL, ATOL = 1e-4, 1e-5
PREFIX = "v33_sparse_context/"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def centers_to_indices(centers, *, feature_hw=(116, 152), patch_count=96):
    """Original integer [x,y] centers -> ordered y*W+x; no RNG or rounding.

    The caller draws all x then all y, using the original persistent generator.
    Duplicate centers are legal. The one-pixel interior also supports the original
    3x3 fmap patch and is not a newly selected set of points.
    """
    values = np.asarray(centers)
    h, w = feature_hw
    if type(h) is not int or type(w) is not int or h < 3 or w < 3:
        raise ValueError("Feature H,W must be integer dimensions >=3")
    if type(patch_count) is not int or patch_count <= 0:
        raise ValueError("patch_count must be a positive integer")
    if values.dtype != np.dtype("int64") or values.shape != (patch_count, 2):
        raise ValueError("centers must be int64[M,2] in original x,y order")
    x, y = values[:, 0], values[:, 1]
    if np.any(x < 1) or np.any(x >= w - 1) or np.any(y < 1) or np.any(y >= h - 1):
        raise ValueError("Centers must be in the original one-pixel interior")
    return np.ascontiguousarray(y * w + x, dtype=np.int64)


def validate_patch_indices(indices, *, feature_hw=(116, 152), patch_count=96):
    """Validate runtime indices outside the graph; return a private C-order copy."""
    values = np.asarray(indices)
    if values.dtype != np.dtype("int64") or values.shape != (patch_count,):
        raise ValueError("patch_indices must be int64[M]")
    h, w = feature_hw
    if np.any(values < 0) or np.any(values >= h * w):
        raise ValueError("Flattened patch index is outside the feature map")
    xy = np.stack((values % w, values // w), axis=-1)
    centers_to_indices(xy, feature_hw=feature_hw, patch_count=patch_count)
    return values.copy(order="C")


def _contract(values):
    result = []
    for value in values:
        if not value.type.HasField("tensor_type"):
            raise ValueError("Only fixed tensor inputs/outputs are supported")
        tensor = value.type.tensor_type
        if tensor.elem_type not in (TensorProto.FLOAT, TensorProto.INT64):
            raise ValueError("Unexpected tensor dtype")
        dims = tensor.shape.dim
        if not dims or any(not d.HasField("dim_value") or d.dim_value <= 0 for d in dims):
            raise ValueError("Only fixed positive dimensions are supported")
        result.append({"name": value.name, "shape": [d.dim_value for d in dims],
                       "dtype": "float32" if tensor.elem_type == TensorProto.FLOAT else "int64"})
    return result


def _reject_complex_storage(model):
    if model.functions or model.graph.sparse_initializer:
        raise ValueError("Local functions or sparse initializers are outside this recipe")
    tensors = list(model.graph.initializer)
    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                raise ValueError("Subgraphs are outside this recipe")
            if attribute.type == onnx.AttributeProto.TENSOR:
                tensors.append(attribute.t)
            if attribute.type == onnx.AttributeProto.TENSORS:
                tensors.extend(attribute.tensors)
    if any(t.data_location == TensorProto.EXTERNAL or t.external_data for t in tensors):
        raise ValueError("External tensor data is not permitted")


def _names(model):
    names = {t.name for t in model.graph.initializer}
    names.update(v.name for v in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info))
    for node in model.graph.node:
        names.update([node.name, *node.input, *node.output])
    return names


def preservation(original, candidate):
    n, k = len(original.graph.node), len(original.graph.initializer)
    c, h, w = _contract(original.graph.output)[1]["shape"][1:]
    if len(candidate.graph.input) != 2 or len(candidate.graph.output) != 2:
        raise ValueError("Candidate must have exactly two inputs and outputs")
    index_contract = _contract(candidate.graph.input)[1]
    if index_contract["name"] != "patch_indices" or index_contract["dtype"] != "int64" or len(index_contract["shape"]) != 1:
        raise ValueError("Candidate index contract mismatch")
    m = index_contract["shape"][0]
    expected_nodes = [
        helper.make_node("Reshape", ["imap", PREFIX + "flat_shape"], [PREFIX + "flat_imap"], name=PREFIX + "Reshape"),
        helper.make_node("Gather", [PREFIX + "flat_imap", "patch_indices"], [PREFIX + "sampled_ncm"], axis=2, name=PREFIX + "Gather"),
        helper.make_node("Transpose", [PREFIX + "sampled_ncm"], ["context"], perm=[0, 2, 1], name=PREFIX + "Transpose"),
    ]
    expected_shape = numpy_helper.from_array(np.asarray([1, c, h * w], dtype=np.int64), name=PREFIX + "flat_shape")
    header = copy.deepcopy(candidate)
    header.graph.CopyFrom(original.graph)
    proof = {
        "all_original_nodes_ordered_protobuf_byte_exact":
            [v.SerializeToString() for v in original.graph.node] ==
            [v.SerializeToString() for v in candidate.graph.node[:n]],
        "all_original_initializers_ordered_protobuf_byte_exact":
            [v.SerializeToString() for v in original.graph.initializer] ==
            [v.SerializeToString() for v in candidate.graph.initializer[:k]],
        "original_image_input_protobuf_byte_exact":
            original.graph.input[0].SerializeToString() == candidate.graph.input[0].SerializeToString(),
        "full_fmap_output_protobuf_byte_exact":
            original.graph.output[0].SerializeToString() == candidate.graph.output[0].SerializeToString(),
        "original_value_info_protobuf_byte_exact":
            [v.SerializeToString() for v in original.graph.value_info] ==
            [v.SerializeToString() for v in candidate.graph.value_info],
        "model_header_byte_exact": header.SerializeToString() == original.SerializeToString(),
        "original_dense_imap_producer_and_divide_by_four_retained": True,
        "exact_appended_sampling_recipe":
            [v.SerializeToString() for v in candidate.graph.node[n:]] == [v.SerializeToString() for v in expected_nodes],
        "exact_added_nonlearned_shape": len(candidate.graph.initializer) == k + 1 and
            candidate.graph.initializer[k].SerializeToString() == expected_shape.SerializeToString(),
        "exact_context_contract": _contract(candidate.graph.output)[1] ==
            {"name": "context", "shape": [1, m, c], "dtype": "float32"},
    }
    if not all(proof.values()) or len(candidate.graph.node) != n + 3 or len(candidate.graph.initializer) != k + 1:
        raise ValueError("Original graph preservation failed")
    proof.update({"original_nodes": n, "original_initializers": k,
                  "added_nodes": 3, "added_nonlearned_initializers": 1})
    return proof


def append_sparse_context(model, *, image_shape=IMAGE_SHAPE, fmap_shape=FMAP_SHAPE,
                          imap_shape=IMAP_SHAPE, patch_count=PATCH_COUNT):
    """Pure graph editor. Nonproduction sizes are explicit for tiny tests only."""
    _reject_complex_storage(model)
    onnx.checker.check_model(model, full_check=True)
    if [(o.domain, o.version) for o in model.opset_import] != [("", 17)]:
        raise ValueError("Expected default-domain opset17 only")
    if len(imap_shape) != 4 or imap_shape[0] != 1 or type(patch_count) is not int or patch_count <= 0:
        raise ValueError("Expected fixed singleton-batch context and positive M")
    if _contract(model.graph.input) != [{"name": "normalized_bgr", "shape": list(image_shape), "dtype": "float32"}]:
        raise ValueError("Unexpected original image input")
    if _contract(model.graph.output) != [
        {"name": "fmap", "shape": list(fmap_shape), "dtype": "float32"},
        {"name": "imap", "shape": list(imap_shape), "dtype": "float32"},
    ]:
        raise ValueError("Unexpected original encoder outputs")
    if fmap_shape[0] != 1 or fmap_shape[2:] != imap_shape[2:]:
        raise ValueError("fmap/imap must share the original feature grid")
    generated = ["patch_indices", "context", PREFIX + "flat_shape", PREFIX + "flat_imap",
                 PREFIX + "sampled_ncm", PREFIX + "Reshape", PREFIX + "Gather", PREFIX + "Transpose"]
    if set(generated) & _names(model):
        raise ValueError("Sparse context names collide with the original graph")
    candidate = copy.deepcopy(model)
    c, h, w = imap_shape[1:]
    candidate.graph.initializer.append(numpy_helper.from_array(
        np.asarray([1, c, h * w], dtype=np.int64), name=PREFIX + "flat_shape"))
    candidate.graph.node.extend([
        helper.make_node("Reshape", ["imap", PREFIX + "flat_shape"], [PREFIX + "flat_imap"], name=PREFIX + "Reshape"),
        helper.make_node("Gather", [PREFIX + "flat_imap", "patch_indices"], [PREFIX + "sampled_ncm"],
                         axis=2, name=PREFIX + "Gather"),
        helper.make_node("Transpose", [PREFIX + "sampled_ncm"], ["context"], perm=[0, 2, 1], name=PREFIX + "Transpose"),
    ])
    candidate.graph.input.append(helper.make_tensor_value_info("patch_indices", TensorProto.INT64, [patch_count]))
    del candidate.graph.output[1:]
    candidate.graph.output.append(helper.make_tensor_value_info("context", TensorProto.FLOAT, [1, patch_count, c]))
    onnx.checker.check_model(candidate, full_check=True)
    inferred = shape_inference.infer_shapes(candidate, strict_mode=True)
    if _contract(inferred.graph.output) != _contract(candidate.graph.output):
        raise ValueError("Sparse output inferred shape mismatch")
    preservation(model, candidate)
    return candidate


def rewrite_pinned_graph(path, expected_sha256):
    path = Path(path)
    if expected_sha256 not in PARENTS or sha256_file(path) != expected_sha256:
        raise ValueError("Expected one of the two exact frozen encoder graphs")
    model = onnx.load(path, load_external_data=False)
    row = PARENTS[expected_sha256]
    if len(model.graph.node) != row["nodes"] or len(model.graph.initializer) != row["initializers"]:
        raise ValueError("Pinned parent structure mismatch")
    return model, append_sparse_context(model)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--onnx-sha256", required=True, choices=list(PARENTS))
    parser.add_argument("--official-source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError("Output directory must be new")
    before = {name: sha256_file(args.official_source_dir / name) for name in SOURCE_PINS}
    if before != SOURCE_PINS:
        raise ValueError("Pinned official source mismatch")
    original, candidate = rewrite_pinned_graph(args.onnx, args.onnx_sha256)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "encoders-sparse-context.onnx"
    onnx.save_model(candidate, output, save_as_external_data=False)
    reloaded = onnx.load(output, load_external_data=False)
    onnx.checker.check_model(reloaded, full_check=True)
    proof = preservation(original, reloaded)
    if reloaded.SerializeToString() != candidate.SerializeToString():
        raise ValueError("Saved sparse graph differs from the checked candidate")
    after = {name: sha256_file(args.official_source_dir / name) for name in SOURCE_PINS}
    if before != after or sha256_file(args.onnx) != args.onnx_sha256:
        raise ValueError("Sources or parent graph changed during editing")
    old_bytes = (128 + 384) * 116 * 152 * 4
    new_bytes = 128 * 116 * 152 * 4 + 96 * 384 * 4
    report = {
        "schema": "v33_DPVO_encoder_sparse_context_rewrite", "version": 1,
        "status": "statically_checked_not_runtime_validated", "candidate_accepted": False,
        "script_sha256": sha256_file(__file__), "parent": {
            "file": args.onnx.name, "bytes": args.onnx.stat().st_size,
            "sha256": args.onnx_sha256, **PARENTS[args.onnx_sha256]},
        "output": {"file": output.name, "bytes": output.stat().st_size, "sha256": sha256_file(output)},
        "inputs": _contract(reloaded.graph.input), "outputs": _contract(reloaded.graph.output),
        "source_commit": OFFICIAL_COMMIT, "source_license": "MIT", "neural_weight_license_verified": False,
        "source_pins": before, "source_hashes_after": after, "sources_and_parent_unchanged": True,
        "preservation": proof, "opset": 17, "ir_version": original.ir_version,
        "node_counts": {"parent": len(original.graph.node), "candidate": len(reloaded.graph.node)},
        "added_operators": ["Reshape", "Gather(axis=2)", "Transpose(0,2,1)"],
        "op_counts": dict(sorted(Counter(n.op_type for n in reloaded.graph.node).items())),
        "sampling_contract": {
            "M": 96, "feature_hw": [116, 152], "centers": "int64[M,2] ordered x,y",
            "index_formula": "y*152+x", "indices_dtype": "int64", "duplicates_allowed": True,
            "interior": {"x_inclusive": [1, 150], "y_inclusive": [1, 114]},
            "context_sampling": "exact original integer radius0, without interpolation",
            "context_layout": "[batch=1,original_patch_order,384]",
            "original_fmap_radius1_patch_and_color_centers_unchanged": True,
            "normalization": "original BGR 2*(uint8/255)-0.5 exactly once before graph",
            "both_encoder_outputs_original_divide_by_four_retained": True,
            "RNG_contract": "persistent original generator: all x draws, all y draws, then original owner random depth draws",
            "depth_random_draw_still_consumed_after_initialization": True,
            "random_sequence_CPU_vs_CUDA_bitwise_claimed": False,
            "source_frame_binding": "runtime must bind image/capture/frame ID, ordered centers and indices hashes to this same source frame",
            "source_frame_binding_verified_by_static_graph": False,
            "invalid_indices_guard": "caller must run centers_to_indices/validate_patch_indices before SDK; graph does not enforce interior",
        },
        "raw_FP32_output_bytes": {"original_fmap": 9027584, "original_dense_imap": 27082752,
            "original_total": old_bytes, "new_context": 147456, "candidate_total": new_bytes,
            "saved": old_bytes - new_bytes, "fraction_saved": (old_bytes - new_bytes) / old_bytes},
        "numerical_gate": {"rtol": RTOL, "atol": ATOL, "actual_model_evaluation": "not_exercised"},
        "limitations": ["full inet and fmap computation retained", "no pixel/resize/model/default changes",
            "actual Gather placement, SDK layouts, transfer bytes and latency are unmeasured",
            "no parent numerical rejection is converted into acceptance", "not a VO, pose-rate or accuracy result"],
        "model_forward_executed": False, "ORT_executed_on_original_arrays": False,
        "compiler_invoked": False, "board_executed": False, "external_data_used": False,
    }
    (args.output_dir / "rewrite.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": report["output"]}), flush=True)
    return report


if __name__ == "__main__":
    main()
