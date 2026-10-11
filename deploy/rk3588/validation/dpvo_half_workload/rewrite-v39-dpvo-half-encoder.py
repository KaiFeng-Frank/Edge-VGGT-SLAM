#!/usr/bin/env python3
"""Static fixed-size DPVO encoder rewrite: same weights, half-sized pixels.

MIT project helper; DPVO copyright (c) 2022 Princeton Vision & Learning Lab.
This edits ONNX metadata and four explicitly named INT64 shape constants only.
No neural/ORT/SDK execution or calibration is performed. The spatial/reciprocal
parent's historical FP32 rejection remains separate from new-size actual gates.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

ORIGINAL_SHA = "527fc2493de700a21aee958d7fda4d5ee421f4eb0c92296e81409df289f56a82"
RECIPROCAL_SHA = "d3422c9f258ab760f1c848f7a6859acd76b315f718e4e0687ee1ef0612ab1d70"
RECIPROCAL_RECIPE_SHA = "98c585fd19dbb41c8190f99fa13cd10868e522f6a29524fc3730cfe61c245764"
IMAGE = [1, 3, 232, 304]
FEATURE_HW = [58, 76]
M = 16
SHAPES = {
    "/original/fnet/Constant": ([1, 3, 464, 608], [1, 3, 232, 304], "/original/fnet/Reshape"),
    "/original/fnet/Constant_1": ([1, 1, 128, 116, 152], [1, 1, 128, 58, 76], "/original/fnet/Reshape_1"),
    "/original/inet/Constant": ([1, 1, 384, 116, 152], [1, 1, 384, 58, 76], "/original/inet/Reshape"),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract(values):
    return [dict(name=v.name, dtype=TensorProto.DataType.Name(v.type.tensor_type.elem_type),
                 shape=[d.dim_value for d in v.type.tensor_type.shape.dim]) for v in values]


def _shape(value, dims):
    require(len(value.type.tensor_type.shape.dim) == len(dims), "Unexpected fixed tensor rank")
    for d, n in zip(value.type.tensor_type.shape.dim, dims):
        d.dim_value = n


def _constant(node):
    require(node.op_type == "Constant" and len(node.attribute) == 1 and node.attribute[0].name == "value" and
            node.attribute[0].type == onnx.AttributeProto.TENSOR, "Exact shape Constant expected")
    t = node.attribute[0].t
    require(t.data_type == TensorProto.INT64 and not t.external_data, "Shape constant must be embedded INT64")
    return t


def half_graph(parent, method):
    require(method in ("original", "reciprocal"), "Only two declared parents")
    require(not parent.functions and not parent.graph.sparse_initializer and not parent.graph.value_info,
            "Expected simple frozen graph without additional value_info")
    require([(o.domain, o.version) for o in parent.opset_import] == [("", 17)], "Frozen opset17 required")
    expected_in = [dict(name="normalized_bgr", dtype="FLOAT", shape=[1, 3, 464, 608])]
    expected_out = [dict(name="fmap", dtype="FLOAT", shape=[1, 128, 116, 152])]
    if method == "reciprocal":
        expected_in.append(dict(name="patch_indices", dtype="INT64", shape=[96]))
        expected_out.append(dict(name="context", dtype="FLOAT", shape=[1, 96, 384]))
    else:
        expected_out.append(dict(name="imap", dtype="FLOAT", shape=[1, 384, 116, 152]))
    require(contract(parent.graph.input) == expected_in and contract(parent.graph.output) == expected_out, "Exact parent tensor contract")
    require(all(not t.external_data and t.data_location != TensorProto.EXTERNAL for t in parent.graph.initializer), "No external weights")
    candidate = copy.deepcopy(parent)
    nodes = {n.name: n for n in candidate.graph.node}
    require(len(nodes) == len(candidate.graph.node), "Unique node names required")
    changes = []
    for name, (old, new, consumer) in SHAPES.items():
        n = nodes[name]
        t = _constant(n)
        require(numpy_helper.to_array(t).tolist() == old and list(nodes[consumer].input)[1:] == list(n.output),
                "Only the exact original Reshape shape may change: " + name)
        replacement = numpy_helper.from_array(np.asarray(new, np.int64), name=t.name)
        replacement.doc_string = t.doc_string
        n.attribute[0].t.CopyFrom(replacement)
        changes.append(dict(storage="Constant tensor", name=name, old=old, new=new, consumer=consumer, dtype="INT64"))
    if method == "reciprocal":
        inits = {t.name: t for t in candidate.graph.initializer}
        name = "v33_sparse_context/flat_shape"
        t = inits[name]
        require(t.data_type == TensorProto.INT64 and numpy_helper.to_array(t).tolist() == [1, 384, 17632], "Exact row-major context shape")
        replacement = numpy_helper.from_array(np.asarray([1, 384, 4408], np.int64), name=name)
        replacement.doc_string = t.doc_string
        t.CopyFrom(replacement)
        changes.append(dict(storage="nonlearned initializer", name=name, old=[1, 384, 17632], new=[1, 384, 4408], dtype="INT64"))
        _shape(candidate.graph.input[1], [M])
        _shape(candidate.graph.output[1], [1, M, 384])
    else:
        _shape(candidate.graph.output[1], [1, 384, *FEATURE_HW])
    _shape(candidate.graph.input[0], IMAGE)
    _shape(candidate.graph.output[0], [1, 128, *FEATURE_HW])
    onnx.checker.check_model(candidate, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(candidate, strict_mode=True)
    require(contract(inferred.graph.output) == contract(candidate.graph.output), "New fixed output shape inference")
    return candidate, validate_preservation(parent, candidate, method, changes)


def validate_preservation(parent, candidate, method, declared_changes=None):
    require(method in ("original", "reciprocal"), "Declared parent method required")
    expected_inputs = copy.deepcopy(list(parent.graph.input))
    expected_outputs = copy.deepcopy(list(parent.graph.output))
    _shape(expected_inputs[0], IMAGE)
    _shape(expected_outputs[0], [1, 128, *FEATURE_HW])
    if method == "reciprocal":
        _shape(expected_inputs[1], [M])
        _shape(expected_outputs[1], [1, M, 384])
    else:
        _shape(expected_outputs[1], [1, 384, *FEATURE_HW])
    require([v.SerializeToString() for v in candidate.graph.input] == [v.SerializeToString() for v in expected_inputs] and
            [v.SerializeToString() for v in candidate.graph.output] == [v.SerializeToString() for v in expected_outputs],
            "Only exact declared IO dimensions may change")
    expected_changes = [dict(storage="Constant tensor", name=name, old=old, new=new, consumer=consumer, dtype="INT64")
                        for name, (old, new, consumer) in SHAPES.items()]
    if method == "reciprocal":
        expected_changes.append(dict(storage="nonlearned initializer", name="v33_sparse_context/flat_shape",
                                     old=[1, 384, 17632], new=[1, 384, 4408], dtype="INT64"))
    require(declared_changes is None or declared_changes == expected_changes, "Exact shape-change declaration required")
    restored = copy.deepcopy(candidate)
    selected = set(SHAPES)
    original_nodes = {n.name: n for n in parent.graph.node}
    require(len(parent.graph.node) == len(candidate.graph.node) and
            [n.name for n in parent.graph.node] == [n.name for n in candidate.graph.node], "Node order/count unchanged")
    for n in restored.graph.node:
        if n.name in selected:
            old, new, _ = SHAPES[n.name]
            require(numpy_helper.to_array(_constant(n)).tolist() == new, "Exact selected new shape")
            expected = copy.deepcopy(original_nodes[n.name])
            old_tensor = _constant(expected)
            replacement = numpy_helper.from_array(np.asarray(new, np.int64), name=old_tensor.name)
            replacement.doc_string = old_tensor.doc_string
            expected.attribute[0].t.CopyFrom(replacement)
            require(n.SerializeToString() == expected.SerializeToString(), "Only selected Constant shape values may change")
            n.CopyFrom(original_nodes[n.name])
    require(len(parent.graph.initializer) == len(candidate.graph.initializer) and
            [t.name for t in parent.graph.initializer] == [t.name for t in candidate.graph.initializer], "Initializer order/count unchanged")
    if method == "reciprocal":
        original = {t.name: t for t in parent.graph.initializer}
        for t in restored.graph.initializer:
            if t.name == "v33_sparse_context/flat_shape":
                require(t.data_type == TensorProto.INT64 and numpy_helper.to_array(t).tolist() == [1, 384, 4408], "Exact sparse flat shape")
                expected = numpy_helper.from_array(np.asarray([1, 384, 4408], np.int64), name=t.name)
                expected.doc_string = original[t.name].doc_string
                require(t.SerializeToString() == expected.SerializeToString(), "Only flat-shape values may change")
                t.CopyFrom(original[t.name])
    require(len(restored.graph.input) == len(parent.graph.input) and len(restored.graph.output) == len(parent.graph.output), "IO count unchanged")
    for a, b in zip(restored.graph.input, parent.graph.input):
        a.CopyFrom(b)
    for a, b in zip(restored.graph.output, parent.graph.output):
        a.CopyFrom(b)
    require(restored.SerializeToString() == parent.SerializeToString(), "Unlisted graph/weight/numeric/header mutation")
    learned = [t for t in parent.graph.initializer if t.data_type == TensorProto.FLOAT]
    return dict(all_learned_and_other_numeric_values_protobuf_byte_exact=True,
        all_other_nodes_ordered_protobuf_byte_exact=True, all_nonshape_initializers_ordered_protobuf_byte_exact=True,
        no_added_removed_nodes_or_initializers=True, model_header_and_graph_metadata_byte_exact=True,
        only_three_INT64_Reshape_Constants_changed=True, sparse_INT64_flat_shape_changed=(method == "reciprocal"),
        original_initializer_count=len(parent.graph.initializer), FLOAT_initializer_count=len(learned),
        FLOAT_initializer_manifest=[dict(name=t.name, shape=list(t.dims), protobuf_sha256=hashlib.sha256(t.SerializeToString()).hexdigest()) for t in learned],
        shape_constant_changes=expected_changes, original_inputs=contract(parent.graph.input), new_inputs=contract(candidate.graph.input),
        original_outputs=contract(parent.graph.output), new_outputs=contract(candidate.graph.output), original_pixels_preserved=False,
        resize_is_new_preprocessing=True, new_resolution_numerical_gate_measured=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("original-onnx", "reciprocal-onnx", "reciprocal-recipe", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args(argv)
    sources = {str(args.original_onnx.resolve()): ORIGINAL_SHA, str(args.reciprocal_onnx.resolve()): RECIPROCAL_SHA,
               str(args.reciprocal_recipe.resolve()): RECIPROCAL_RECIPE_SHA, str(Path(__file__).resolve()): sha(__file__)}
    for path, digest in sources.items():
        require(sha(path) == digest, "Pinned parent/source SHA: " + Path(path).name)
    recipe = json.loads(args.reciprocal_recipe.read_text())
    require(recipe["output"]["sha256"] == RECIPROCAL_SHA and recipe["historical_parent_FP32_gate_passed"] is False and
            recipe["historical_parent_rejection_retained"] is True, "Reciprocal historical rejection must be retained")
    out = args.output_dir.resolve()
    require(not out.exists() and not any(out == Path(p) or out.is_relative_to(Path(p)) for p in sources), "New separate output directory")
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema="v39_DPVO_half_resolution_encoder_shape_rewrite", status="statically_checked_not_numerically_validated",
        script_sha256=sha(__file__), checkpoint_sha256="30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12",
        patch_count=M, feature_hw=FEATURE_HW, image_shape=IMAGE, source_pins=sources, models={}, candidate_accepted=False,
        historical_parent_FP32_gate_passed=False, historical_parent_rejection_retained=True,
        original_weights_preserved=True, neural_ORT_SDK_execution=False, preprocessing_changed=True)
    for method, path in (("original", args.original_onnx), ("reciprocal", args.reciprocal_onnx)):
        model = onnx.load(path, load_external_data=False)
        new, proof = half_graph(model, method)
        target = out/f"encoder-half-{method}.onnx"
        onnx.save(new, target)
        report["models"][method] = dict(file=target.name, bytes=target.stat().st_size, sha256=sha(target), parent_onnx_sha256=sha(path),
            inputs=contract(new.graph.input), outputs=contract(new.graph.output), preservation=proof)
    report["source_hashes_after"] = {p: sha(p) for p in sources}
    require(report["source_hashes_after"] == sources, "Source changed during editing")
    report["sources_and_parents_unchanged"] = True
    (out/"rewrite.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "patch_count", "image_shape")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
