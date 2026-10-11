#!/usr/bin/env python3
"""Statically lower the pinned DPVO encoder InstanceNorms to NCHW spatial math.

This editor does not execute the original or rewritten encoder.  The expression
is equivalent over real numbers; FP32 accumulation and RKNN placement still
require an independent, board-only numerical check.  DPVO source is MIT.
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


PARENT_SHA256 = "527fc2493de700a21aee958d7fda4d5ee421f4eb0c92296e81409df289f56a82"
EXTRACTOR_SHA256 = "f5cbcbe14c42e44bcec6e827a3429ebe93979a439ba9b9ef70d4f4baa3454a2b"
OFFICIAL_COMMIT = "0ac95b656d1fda91c271d2a106460d19ad966fc7"
RTOL, ATOL = 1e-4, 1e-5
INPUT_CONTRACT = {"name": "normalized_bgr", "shape": [1, 3, 464, 608], "dtype": "float32"}
OUTPUT_CONTRACTS = [
    {"name": "fmap", "shape": [1, 128, 116, 152], "dtype": "float32"},
    {"name": "imap", "shape": [1, 384, 116, 152], "dtype": "float32"},
]
RECIPE = ["GlobalAveragePool", "Sub", "Mul", "GlobalAveragePool", "Add",
          "Sqrt", "Div", "Reshape", "Reshape", "Mul", "Add"]


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def array_info(value):
    value = np.asarray(value)
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "bytes": value.nbytes, "sha256": sha256_bytes(value.tobytes(order="C"))}


def _reject_external_or_subgraphs(model):
    if model.graph.sparse_initializer or model.functions:
        raise ValueError("Sparse initializers and local functions are outside this recipe")
    tensors = list(model.graph.initializer)
    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                raise ValueError("Subgraphs are outside this recipe")
            if attribute.type == onnx.AttributeProto.TENSOR:
                tensors.append(attribute.t)
            elif attribute.type == onnx.AttributeProto.TENSORS:
                tensors.extend(attribute.tensors)
    if any(t.data_location == TensorProto.EXTERNAL or t.external_data for t in tensors):
        raise ValueError("External tensor data is not permitted")


def _shape_table(model):
    inferred = shape_inference.infer_shapes(model, strict_mode=True)
    return {v.name: v for v in list(inferred.graph.input) +
            list(inferred.graph.value_info) + list(inferred.graph.output)}


def _fp32_shape(value, name):
    if value is None or not value.type.HasField("tensor_type"):
        raise ValueError("Missing inferred tensor type: " + name)
    tensor = value.type.tensor_type
    if tensor.elem_type != TensorProto.FLOAT:
        raise ValueError("Only FP32 normalization tensors are supported: " + name)
    dims = list(tensor.shape.dim)
    if not dims or any(not d.HasField("dim_value") or d.dim_value <= 0 for d in dims):
        raise ValueError("All normalization dimensions must be fixed and positive: " + name)
    return [d.dim_value for d in dims]


def _constant_tensors(model):
    constants = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Constant" and node.domain in ("", "ai.onnx"):
            values = [a for a in node.attribute if a.name == "value"]
            if len(node.output) == 1 and len(values) == 1 and values[0].type == onnx.AttributeProto.TENSOR:
                constants[node.output[0]] = numpy_helper.to_array(values[0].t)
    return constants


def _all_names(model):
    names = {t.name for t in model.graph.initializer}
    names.update(v.name for v in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info))
    for node in model.graph.node:
        names.add(node.name)
        names.update(node.input)
        names.update(node.output)
    return names


def rewrite_instance_norm(model, *, expected_count=None):
    """Return a new model and recipe rows; never run either graph.

    The generic entry point exists for tiny synthetic tests.  The CLI additionally
    pins the exact parent encoder, ten norms, identities, and original 1/0 affine.
    """
    _reject_external_or_subgraphs(model)
    onnx.checker.check_model(model, full_check=True)
    if [(o.domain, o.version) for o in model.opset_import] != [("", 17)]:
        raise ValueError("This recipe requires default-domain ONNX opset 17 only")
    norms = [n for n in model.graph.node if n.op_type == "InstanceNormalization"]
    if not norms or (expected_count is not None and len(norms) != expected_count):
        raise ValueError("Unexpected InstanceNormalization count")
    shapes, constants = _shape_table(model), _constant_tensors(model)
    occupied = _all_names(model)
    candidate = copy.deepcopy(model)
    nodes, rows, additions = [], [], []
    for node in model.graph.node:
        if node.op_type != "InstanceNormalization":
            nodes.append(copy.deepcopy(node))
            continue
        if node.domain not in ("", "ai.onnx") or len(node.input) != 3 or len(node.output) != 1:
            raise ValueError("Unexpected InstanceNormalization domain or arity")
        if any(a.name != "epsilon" or a.type != onnx.AttributeProto.FLOAT for a in node.attribute) or len(node.attribute) > 1:
            raise ValueError("Unexpected InstanceNormalization attributes")
        epsilon = np.float32(node.attribute[0].f if node.attribute else 1e-5)
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        shape = _fp32_shape(shapes.get(node.input[0]), node.input[0])
        if len(shape) != 4:
            raise ValueError("The statistical axes must be fixed NCHW H,W")
        scale, bias = (constants.get(name) for name in node.input[1:])
        for label, value in (("scale", scale), ("bias", bias)):
            if value is None or value.dtype != np.float32 or list(value.shape) != [shape[1]] or not np.isfinite(value).all():
                raise ValueError(label + " must be an original finite constant FP32 channel vector")
        prefix = "v27_in_%03d/" % len(rows)
        suffixes = ["mean", "centered", "squared", "variance", "variance_eps", "denominator", "normalized",
                    "scale_nchw", "bias_nchw", "scaled", "epsilon", "affine_shape"]
        new_names = [prefix + s for s in suffixes]
        op_names = [prefix + "node_%02d_%s" % (i, op) for i, op in enumerate(RECIPE)]
        if (set(new_names) | set(op_names)) & occupied:
            raise ValueError("Generated normalization name collides with the original graph")
        occupied.update(new_names + op_names)
        values = {s: prefix + s for s in suffixes}
        x, gamma, beta = node.input
        specs = [
            ("GlobalAveragePool", [x], [values["mean"]]),
            ("Sub", [x, values["mean"]], [values["centered"]]),
            ("Mul", [values["centered"], values["centered"]], [values["squared"]]),
            ("GlobalAveragePool", [values["squared"]], [values["variance"]]),
            ("Add", [values["variance"], values["epsilon"]], [values["variance_eps"]]),
            ("Sqrt", [values["variance_eps"]], [values["denominator"]]),
            ("Div", [values["centered"], values["denominator"]], [values["normalized"]]),
            ("Reshape", [gamma, values["affine_shape"]], [values["scale_nchw"]]),
            ("Reshape", [beta, values["affine_shape"]], [values["bias_nchw"]]),
            ("Mul", [values["normalized"], values["scale_nchw"]], [values["scaled"]]),
            ("Add", [values["scaled"], values["bias_nchw"]], list(node.output)),
        ]
        replacement = [helper.make_node(op, inputs, outputs, name=name)
                       for (op, inputs, outputs), name in zip(specs, op_names)]
        epsilon_tensor = numpy_helper.from_array(np.asarray(epsilon), name=values["epsilon"])
        affine_shape = numpy_helper.from_array(np.asarray([1, shape[1], 1, 1], dtype=np.int64), name=values["affine_shape"])
        additions.extend([epsilon_tensor, affine_shape])
        nodes.extend(replacement)
        rows.append({"original_node": node.name, "original_node_sha256": sha256_bytes(node.SerializeToString()),
                     "input": x, "output": node.output[0], "shape": shape, "statistical_axes": [2, 3],
                     "variance": "population_mean_of_centered_squares", "epsilon": float(epsilon),
                     "epsilon_fp32": array_info(np.asarray(epsilon)),
                     "scale": {"name": gamma, **array_info(scale)}, "bias": {"name": beta, **array_info(bias)},
                     "scale_all_one": bool(np.all(scale == 1)), "bias_all_zero": bool(np.all(bias == 0)),
                     "new_nodes": op_names, "new_ops": list(RECIPE),
                     "new_constants": [epsilon_tensor.name, affine_shape.name]})
    del candidate.graph.node[:]
    candidate.graph.node.extend(nodes)
    candidate.graph.initializer.extend(additions)
    onnx.checker.check_model(candidate, full_check=True)
    inferred = _shape_table(candidate)
    for row in rows:
        if _fp32_shape(inferred.get(row["output"]), row["output"]) != row["shape"]:
            raise ValueError("Rewritten normalization changed its output shape")
    validate_preservation(model, candidate, rows)
    return candidate, rows


def validate_preservation(original, candidate, rows):
    new_names = {name for row in rows for name in row["new_nodes"]}
    old_nodes = [n.SerializeToString() for n in original.graph.node if n.op_type != "InstanceNormalization"]
    kept_nodes = [n.SerializeToString() for n in candidate.graph.node if n.name not in new_names]
    old_initializers = [t.SerializeToString() for t in original.graph.initializer]
    new_initializers = [t.SerializeToString() for t in candidate.graph.initializer[:len(old_initializers)]]
    same_io = all([v.SerializeToString() for v in getattr(original.graph, name)] ==
                  [v.SerializeToString() for v in getattr(candidate.graph, name)]
                  for name in ("input", "output", "value_info"))
    same_header = copy.deepcopy(candidate)
    same_header.graph.CopyFrom(original.graph)
    header_equal = same_header.SerializeToString() == original.SerializeToString()
    proof = {"original_initializers_byte_exact": old_initializers == new_initializers,
             "non_instance_norm_nodes_byte_exact": old_nodes == kept_nodes,
             "input_output_protobuf_byte_exact": same_io, "model_header_byte_exact": header_equal,
             "original_initializer_count": len(old_initializers), "preserved_non_instance_norm_nodes": len(old_nodes),
             "added_nonlearned_constants": len(candidate.graph.initializer) - len(old_initializers),
             "original_weights_or_output_divide_by_four_changed": False}
    if not all(proof[k] for k in ("original_initializers_byte_exact", "non_instance_norm_nodes_byte_exact",
                                 "input_output_protobuf_byte_exact", "model_header_byte_exact")):
        raise ValueError("Original graph preservation check failed")
    if any(n.op_type == "InstanceNormalization" for n in candidate.graph.node):
        raise ValueError("InstanceNormalization remains in the candidate")
    if len(candidate.graph.initializer) != len(old_initializers) + 2 * len(rows):
        raise ValueError("Unexpected added initializer count")
    return proof


def _contract(values):
    return [{"name": v.name, "shape": _fp32_shape(v, v.name), "dtype": "float32"} for v in values]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--onnx-sha256", required=True)
    parser.add_argument("--extractor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.onnx_sha256 != PARENT_SHA256 or sha256_file(args.onnx) != PARENT_SHA256:
        raise ValueError("The input must be the frozen v26 encoder ONNX")
    if sha256_file(args.extractor) != EXTRACTOR_SHA256:
        raise ValueError("Official extractor source SHA mismatch")
    if args.output_dir.exists():
        raise FileExistsError("Output directory must be new")
    model = onnx.load(args.onnx, load_external_data=False)
    if _contract(model.graph.input) != [INPUT_CONTRACT] or _contract(model.graph.output) != OUTPUT_CONTRACTS:
        raise ValueError("Pinned encoder input/output contract mismatch")
    candidate, rows = rewrite_instance_norm(model, expected_count=10)
    expected_shapes = [[1, 32, 232, 304]] * 5 + [[1, 64, 116, 152]] * 5
    if [r["shape"] for r in rows] != expected_shapes or any(not r["scale_all_one"] or not r["bias_all_zero"] for r in rows):
        raise ValueError("Pinned InstanceNorm shape or affine identity mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "encoders.onnx"
    onnx.save_model(candidate, output, save_as_external_data=False)
    reloaded = onnx.load(output, load_external_data=False)
    onnx.checker.check_model(reloaded, full_check=True)
    preservation = validate_preservation(model, reloaded, rows)
    if sha256_file(args.onnx) != PARENT_SHA256 or sha256_file(args.extractor) != EXTRACTOR_SHA256:
        raise ValueError("Original graph or extractor changed during the rewrite")
    report = {
        "kind": "v27_dpvo_encoder_instance_norm_spatial_rewrite", "version": 1,
        "status": "rewritten_and_statically_checked_not_runtime_validated",
        "script_sha256": sha256_file(__file__),
        "input": {"file": str(args.onnx.resolve()), "bytes": args.onnx.stat().st_size, "sha256": PARENT_SHA256},
        "output": {"file": output.name, "bytes": output.stat().st_size, "sha256": sha256_file(output)},
        "source": {"extractor_sha256": EXTRACTOR_SHA256, "commit": OFFICIAL_COMMIT, "license": "MIT"},
        "input_contract": INPUT_CONTRACT, "output_contracts": OUTPUT_CONTRACTS,
        "opset": [{"domain": o.domain, "version": o.version} for o in model.opset_import],
        "ir_version": model.ir_version,
        "node_counts": {"original": len(model.graph.node), "candidate": len(candidate.graph.node)},
        "op_counts": {"original": dict(sorted(Counter(n.op_type for n in model.graph.node).items())),
                      "candidate": dict(sorted(Counter(n.op_type for n in candidate.graph.node).items()))},
        "preservation": preservation, "rewrites": rows,
        "formula": "gamma * ((x - GAP_HW(x)) / sqrt(GAP_HW((x - GAP_HW(x))^2) + original_epsilon)) + beta",
        "FP32_gate": {"rtol": RTOL, "atol": ATOL, "status": "pending_board_ORT"},
        "sources_unchanged_after": True, "external_data_used": False,
        "host_original_graph_executed": False, "board_executed": False, "RKNN_compiled": False,
        "limitations": ["Equivalent real-number arithmetic does not guarantee FP32 bitwise identity.",
                        "No measured speed, precision, or operator placement claim follows from this static rewrite.",
                        "RKNN may recognize the explicit statistics and fuse them again; inspect the actual compiler log/profile."]}
    (args.output_dir / "rewrite.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": report["output"],
                      "manifest": str(args.output_dir / "rewrite.json")}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
