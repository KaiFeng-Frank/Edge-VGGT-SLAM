#!/usr/bin/env python3
"""Static, diagnostic-only rewrite of ten pinned DPVO spatial-norm Div nodes.

The FP32 expression changes x/d to x*(1/d); real-arithmetic equivalence does
not imply bitwise FP32 equality or equivalent quantized execution. No model,
ORT, compiler or SDK is executed here. The prior CPU rejection remains true.

DPVO source provenance: princeton-vl/DPVO, commit
0ac95b656d1fda91c271d2a106460d19ad966fc7 (MIT). This graph editor does not
redistribute neural weights or assert a separate neural-weight license.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import sys

import onnx
from onnx import TensorProto, helper, numpy_helper


SCHEMA = "v38_DPVO_spatial_encoder_small_reciprocal_rewrite"
PARENT_SHA256 = "6c3b4de4d151813dc172d711b6a760ad8bdc17120f2f3faa76dbf32bd0c451ab"
PARENT_RECIPE_SHA256 = "005667f44cf98447d2b622882d7d40cbcce8a545985ddf2104eaa62740273ce2"
PARENT_REFERENCE_SHA256 = "24aef39822c86ae60e1cb6306d2d2966fa8a440320c4363f18c5db938ac4e168"
SELECTED_NAMES = tuple(f"v27_in_{i:03d}/node_06_Div" for i in range(10))
OTHER_DIV_NAMES = ("/original/Div", "/original/Div_1")
ONE_NAME = "v38_small_reciprocal/scalar_one"
EPSILON_BYTES = struct.pack("<f", 1e-5)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _proto(value) -> bytes:
    return value.SerializeToString()


def _header(model) -> bytes:
    clone = onnx.ModelProto()
    clone.CopyFrom(model)
    clone.ClearField("graph")
    return _proto(clone)


def _graph_metadata(graph) -> bytes:
    clone = onnx.GraphProto()
    clone.CopyFrom(graph)
    clone.ClearField("node")
    clone.ClearField("initializer")
    return _proto(clone)


def _one():
    return helper.make_tensor(ONE_NAME, TensorProto.FLOAT, [], struct.pack("<f", 1.0), raw=True)


def _replacement(i: int, old):
    prefix = f"v38_small_reciprocal/{i:03d}"
    inverse = f"{prefix}/inverse"
    return (
        helper.make_node("Div", [ONE_NAME, old.input[1]], [inverse], name=f"{prefix}/Div"),
        helper.make_node("Mul", [old.input[0], inverse], list(old.output), name=f"{prefix}/Mul"),
    )


def _shapes(model):
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    result = {}
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        tensor = value.type.tensor_type
        if tensor.HasField("shape") and all(d.HasField("dim_value") for d in tensor.shape.dim):
            result[value.name] = (tensor.elem_type, tuple(d.dim_value for d in tensor.shape.dim))
    return result


def selected_replacements(parent) -> list[dict]:
    """Check only graph structure, shape inference and existing FP32 constants."""
    onnx.checker.check_model(parent)
    _require(len(parent.graph.initializer) == 65, "Expected all 65 original initializers")
    _require(all(x.data_location != TensorProto.EXTERNAL and not x.external_data
                 for x in parent.graph.initializer), "External initializers are not supported")
    names = [n.name for n in parent.graph.node]
    _require(all(names.count(name) == 1 for name in SELECTED_NAMES + OTHER_DIV_NAMES),
             "Missing or duplicate selected/nonselected Div name")
    by_name = {n.name: n for n in parent.graph.node}
    producers = {}
    for node in parent.graph.node:
        for output in node.output:
            _require(output not in producers, "Duplicate graph tensor producer")
            producers[output] = node
    initializers = {x.name: x for x in parent.graph.initializer}
    _require(len(initializers) == 65, "Duplicate original initializer name")
    div_names = {n.name for n in parent.graph.node if n.op_type == "Div" and n.domain == ""}
    _require(div_names == set(SELECTED_NAMES + OTHER_DIV_NAMES), "Expected exactly the original 12 Div nodes")
    occupied = set(initializers) | set(names)
    for node in parent.graph.node:
        occupied.update(node.input)
        occupied.update(node.output)
    occupied.update(v.name for v in list(parent.graph.input) + list(parent.graph.output) + list(parent.graph.value_info))
    new_names = {ONE_NAME}
    for i in range(10):
        prefix = f"v38_small_reciprocal/{i:03d}"
        new_names.update((f"{prefix}/inverse", f"{prefix}/Div", f"{prefix}/Mul"))
    _require(not occupied.intersection(new_names), "New reciprocal names collide with parent graph")
    shapes = _shapes(parent)
    rows = []
    for i, name in enumerate(SELECTED_NAMES):
        prefix = f"v27_in_{i:03d}"
        node = by_name[name]
        _require(node.op_type == "Div" and node.domain == "" and len(node.attribute) == 0,
                 f"Unexpected selected Div operation: {name}")
        _require(list(node.input) == [f"{prefix}/centered", f"{prefix}/denominator"]
                 and list(node.output) == [f"{prefix}/normalized"], f"Unexpected operands: {name}")
        c, h, w = (32, 232, 304) if i < 5 else (64, 116, 152)
        _require(shapes.get(node.input[0]) == (TensorProto.FLOAT, (1, c, h, w))
                 and shapes.get(node.output[0]) == (TensorProto.FLOAT, (1, c, h, w))
                 and shapes.get(node.input[1]) == (TensorProto.FLOAT, (1, c, 1, 1)),
                 f"Unexpected centered/denominator shape or dtype: {name}")
        sqrt = producers.get(node.input[1])
        add = producers.get(f"{prefix}/variance_eps")
        _require(sqrt is not None and sqrt.name == f"{prefix}/node_05_Sqrt"
                 and sqrt.op_type == "Sqrt" and sqrt.domain == "" and not sqrt.attribute
                 and list(sqrt.input) == [f"{prefix}/variance_eps"], f"Unexpected denominator producer: {name}")
        _require(add is not None and add.name == f"{prefix}/node_04_Add" and add.op_type == "Add"
                 and add.domain == "" and not add.attribute
                 and list(add.input) == [f"{prefix}/variance", f"{prefix}/epsilon"],
                 f"Unexpected variance/epsilon producer: {name}")
        eps = initializers.get(f"{prefix}/epsilon")
        _require(eps is not None and eps.data_type == TensorProto.FLOAT and not eps.dims
                 and numpy_helper.to_array(eps).tobytes() == EPSILON_BYTES, f"Changed original epsilon: {name}")
        consumers = [n.name for n in parent.graph.node if node.input[1] in n.input]
        _require(consumers == [name], f"Unexpected denominator consumers: {name}")
        rows.append({"old_node": name, "numerator": node.input[0], "denominator": node.input[1],
                     "retained_output": node.output[0], "numerator_shape": [1, c, h, w],
                     "denominator_shape": [1, c, 1, 1], "epsilon_FP32_raw_sha256": hashlib.sha256(EPSILON_BYTES).hexdigest(),
                     "new_nodes": [n.name for n in _replacement(i, node)],
                     "new_inverse": f"v38_small_reciprocal/{i:03d}/inverse"})
    return rows


def validate_preservation(parent, candidate) -> dict:
    """Public pure-static verifier; it does not execute either graph."""
    selected_replacements(parent)
    onnx.checker.check_model(candidate)
    _require(len(candidate.graph.initializer) == 66, "Candidate must add exactly one initializer")
    _require([_proto(x) for x in candidate.graph.initializer[:65]]
             == [_proto(x) for x in parent.graph.initializer], "Original initializer bytes/order changed")
    _require(_proto(candidate.graph.initializer[65]) == _proto(_one()), "Added initializer is not unique FP32 scalar one")
    _require(_header(parent) == _header(candidate), "Model header changed")
    _require(_graph_metadata(parent.graph) == _graph_metadata(candidate.graph), "Original graph IO/value_info/metadata changed")
    selected = {name: i for i, name in enumerate(SELECTED_NAMES)}
    cursor = 0
    for old in parent.graph.node:
        expected = _replacement(selected[old.name], old) if old.name in selected else (old,)
        for node in expected:
            _require(cursor < len(candidate.graph.node) and _proto(candidate.graph.node[cursor]) == _proto(node),
                     f"Unexpected changed node or node order at {old.name}")
            cursor += 1
    _require(cursor == len(candidate.graph.node), "Additional candidate nodes found")
    _require(len(candidate.graph.node) == len(parent.graph.node) + 10, "Expected exactly ten extra nodes")
    return {
        "all_original_initializers_ordered_protobuf_byte_exact": True,
        "all_nonselected_nodes_ordered_protobuf_byte_exact": True,
        "original_IO_value_info_graph_metadata_protobuf_byte_exact": True,
        "model_header_protobuf_byte_exact": True,
        "exact_10_selected_Div_replacement": True,
        "original_2_nonselected_Div_protobuf_byte_exact": True,
        "one_added_nonlearned_FP32_scalar_one": True,
        "original_initializer_count": 65, "added_initializer_count": 1,
    }


def rewrite(parent):
    """Return a new ModelProto and a preservation proof; never mutate parent."""
    selected_replacements(parent)
    candidate = onnx.ModelProto()
    candidate.CopyFrom(parent)
    candidate.graph.ClearField("node")
    selected = {name: i for i, name in enumerate(SELECTED_NAMES)}
    for old in parent.graph.node:
        nodes = _replacement(selected[old.name], old) if old.name in selected else (old,)
        candidate.graph.node.extend(nodes)
    candidate.graph.initializer.append(_one())
    return candidate, validate_preservation(parent, candidate)


def validate_ancestors(recipe: dict, reference: dict) -> None:
    """Validate rejection/provenance fields after caller has checked file hashes."""
    _require(recipe.get("schema") == "v33_DPVO_encoder_sparse_context_rewrite"
             and recipe.get("status") == "statically_checked_not_runtime_validated"
             and recipe.get("candidate_accepted") is False
             and recipe.get("output", {}).get("sha256") == PARENT_SHA256,
             "Unexpected pinned parent recipe identity")
    _require(recipe.get("sources_and_parent_unchanged") is True
             and recipe.get("source_pins") == recipe.get("source_hashes_after"), "Parent recipe source pins are inconsistent")
    _require(reference.get("schema") == "v33_DPVO_sparse_context_board_CPU_ORT"
             and reference.get("status") == "completed_suffix_bitwise_gate_parent_rejection_retained_not_VO"
             and reference.get("receipt_sha256", {}).get("spatial_sparse") == PARENT_RECIPE_SHA256,
             "Unexpected pinned parent reference identity")
    _require(reference.get("original_CPU_strict_gate_each_method", {}).get("spatial") is False
             and reference.get("historical_spatial_rejection", {}).get("exact_FP32_gate_passed") is False
             and reference.get("historical_spatial_rejection", {}).get("retained") is True,
             "Parent CPU rejection must be retained")


def _initializer_manifest(model):
    return [{"name": x.name, "shape": list(x.dims), "dtype": TensorProto.DataType.Name(x.data_type),
             "raw_sha256": hashlib.sha256(numpy_helper.to_array(x).tobytes(order="C")).hexdigest(),
             "protobuf_sha256": hashlib.sha256(_proto(x)).hexdigest()} for x in model.graph.initializer]


def _io(values):
    return [{"name": v.name, "dtype": TensorProto.DataType.Name(v.type.tensor_type.elem_type),
             "shape": [d.dim_value if d.HasField("dim_value") else d.dim_param for d in v.type.tensor_type.shape.dim]}
            for v in values]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-recipe", type=Path, required=True)
    parser.add_argument("--parent-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New, absent directory")
    args = parser.parse_args(argv)
    paths = {"rewriter": Path(__file__).resolve(), "parent": args.parent.resolve(),
             "parent_recipe": args.parent_recipe.resolve(), "parent_reference": args.parent_reference.resolve()}
    output_dir = args.output_dir.resolve()
    _require(not output_dir.exists(), "Output directory must be absent; never overwrite prior evidence")
    _require(all(output_dir != p and output_dir not in p.parents for p in paths.values()), "Output overlaps a source")
    output_dir.mkdir(parents=True)
    report = {"schema": SCHEMA, "status": "failed_before_static_validation", "candidate_accepted": False,
              "historical_parent_FP32_gate_passed": False, "historical_parent_rejection_retained": True,
              "model_forward_executed": False, "ORT_executed": False, "compiler_invoked": False,
              "SDK_invoked": False, "board_model_execution": False}
    try:
        before = {key: sha256_file(path) for key, path in paths.items()}
        report["source_pins"] = before
        report["script_sha256"] = before["rewriter"]
        for key, expected in (("parent", PARENT_SHA256), ("parent_recipe", PARENT_RECIPE_SHA256),
                              ("parent_reference", PARENT_REFERENCE_SHA256)):
            _require(before[key] == expected, f"{key} SHA256 differs from frozen ancestor")
        recipe = json.loads(paths["parent_recipe"].read_text())
        reference = json.loads(paths["parent_reference"].read_text())
        validate_ancestors(recipe, reference)
        parent = onnx.load_model(paths["parent"], load_external_data=False)
        candidate, preservation = rewrite(parent)
        destination = output_dir / "encoders-reciprocal.onnx"
        onnx.save_model(candidate, destination)
        # Validate the serialized artifact, not just the in-memory candidate.
        _require(validate_preservation(parent, onnx.load_model(destination, load_external_data=False)) == preservation,
                 "Saved artifact preservation differs")
        report.update({
            "status": "statically_checked_parent_FP32_rejection_retained_not_runtime_validated",
            "parent": {"file": paths["parent"].name, "sha256": before["parent"], "bytes": paths["parent"].stat().st_size},
            "output": {"file": destination.name, "sha256": sha256_file(destination), "bytes": destination.stat().st_size},
            "parent_recipe_sha256": before["parent_recipe"], "parent_reference_sha256": before["parent_reference"],
            "preservation": preservation, "selected_replacements": selected_replacements(parent),
            "original_initializer_manifest": _initializer_manifest(parent),
            "parent_source_provenance": {k: recipe[k] for k in ("source_commit", "source_license", "source_pins", "source_hashes_after")},
            "prior_weight_bindings": {"pinned_parent_ONNX_sha256": before["parent"],
                                      "prior_capture_export_receipt_sha256": reference["receipt_sha256"],
                                      "neural_weight_license_verified": recipe.get("neural_weight_license_verified") is True},
            "inputs": _io(candidate.graph.input), "outputs": _io(candidate.graph.output),
            "op_counts": dict(Counter(n.op_type for n in candidate.graph.node)),
            "original_node_count": len(parent.graph.node), "candidate_node_count": len(candidate.graph.node),
            "small_denominator_elements_per_pass": 480, "large_normalized_elements_per_pass": 16926720,
            "numerical_gate": {"rtol": 1e-4, "atol": 1e-5, "passed_in_this_tool": None},
            "limitations": ["FP32 reciprocal multiplication can round differently from direct division.",
                            "No pixel, model, CPU ORT, compiler, SDK, runtime-placement or speed validation is performed.",
                            "Quantized fusion/placement and accuracy require separate actual board evidence.",
                            "Original parent CPU precision rejection is retained; this candidate is not adopted."],
        })
        after = {key: sha256_file(path) for key, path in paths.items()}
        report["source_hashes_after"] = after
        report["sources_and_parent_unchanged"] = before == after
        _require(before == after, "Source or ancestor artifact changed during static rewrite")
        (output_dir / "rewrite.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": report["status"], "output_sha256": report["output"]["sha256"],
                          "recipe_sha256": sha256_file(output_dir / "rewrite.json"), "candidate_accepted": False}))
        return 0
    except Exception as exc:
        report["status"] = "failed_static_rewrite_not_adopted"
        report["error"] = repr(exc)
        report["source_hashes_after"] = {key: sha256_file(path) for key, path in paths.items() if path.is_file()}
        report["sources_and_parent_unchanged"] = report.get("source_pins") == report["source_hashes_after"]
        (output_dir / "rewrite.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": report["status"], "error": report["error"]}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
