#!/usr/bin/env python3
"""Static, isolated Update graph candidate; no model or ORT execution.

Replace only the two -inf padding tensors used before grouped ReduceMax.
The existing v28 topology preflight requires true members before padding.
Consequently a nonempty group's first slot is a real member and can fill
its padding without changing its maximum. Existing empty-group guards,
membership masks, denominators, learned weights and Sigmoid are untouched.
Actual board CPU ORT and NPU gates remain mandatory after this static edit.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


EXPORT_SHA = "56f96b7bf80bde596de6a4008da2858a869a9f19a3162abb8db8ab1af0156cac"
CAPTURE_SHA = "87e18137a828496d623721f7748e288d02f9d8d6879278da1e5d3d9731cfa0a1"
ADAPTER_SHA = "3f755c8e24d092fa55fdf698d87c3d9ba8944487ee95d4e5afc47554d31d1380"
PARENT_GRAPH_SHAS = {
    "E96": "7c0dea7bae8f8dddd6d55ee87e7d927491be52d21d80e2f882a28e028379856f",
    "E864": "c7a51b1fa37342f8ee3ac1d0133dc2bf4e015d3b44579a4408e67585f63396b3",
}
TARGETS = (("/Constant_4", "/Where"), ("/Constant_11", "/Where_3"))
INPUT_NAMES = ("net", "inp", "corr", "previous_indices", "previous_valid",
               "following_indices", "following_valid", "kk_indices", "kk_valid",
               "kk_inverse", "ij_indices", "ij_valid", "ij_inverse")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protobuf_sha(value):
    return hashlib.sha256(value.SerializeToString(deterministic=True)).hexdigest()


def constants(model):
    values = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Constant":
            tensors = [value.t for value in node.attribute if value.name == "value"]
            require(len(tensors) == 1 and len(node.output) == 1,
                    "Only ordinary tensor Constants supported")
            values[node.output[0]] = numpy_helper.to_array(tensors[0])
    return values


def validate_prefix_mask(mask):
    """Small diagnostic guard; runtime still needs complete v28 preflight."""
    value = np.asarray(mask)
    require(value.ndim == 2 and min(value.shape) > 0,
            "Mask must be nonempty [groups,slots]")
    require(np.all((value == 0) | (value == 1)), "Mask must be exactly binary")
    require(np.all(value[:, :-1] >= value[:, 1:]),
            "Real members must precede padding; first slot must be real when nonempty")
    return True


def rewrite_model(model):
    """Pure graph surgery, also usable on tiny synthetic two-group fixtures."""
    onnx.checker.check_model(model, full_check=True)
    nodes = list(model.graph.node)
    require(len({node.name for node in nodes}) == len(nodes), "Unique node names required")
    named = {node.name: node for node in nodes}
    values = constants(model)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    shapes = {}
    for value in (*inferred.graph.input, *inferred.graph.output, *inferred.graph.value_info):
        if value.type.HasField("tensor_type"):
            dims = value.type.tensor_type.shape.dim
            if all(dim.HasField("dim_value") for dim in dims):
                shapes[value.name] = tuple(dim.dim_value for dim in dims)
    target_names = {name for name, _ in TARGETS}
    replacements, bindings, added = {}, [], []
    existing = {name for node in nodes for name in (*node.input, *node.output, node.name)}
    existing.update(value.name for value in model.graph.initializer)
    index_name = "v29_group_max_first_slot_index"
    require(index_name not in existing, "New index name collides")
    added.append(numpy_helper.from_array(np.array([0], dtype=np.int64), index_name))
    for number, (constant_name, where_name) in enumerate(TARGETS):
        require(constant_name in named and where_name in named, "Expected two padding bindings missing")
        source, where = named[constant_name], named[where_name]
        require(source.op_type == "Constant" and where.op_type == "Where"
                and len(where.input) == 3 and len(source.output) == 1
                and where.input[2] == source.output[0], "Unexpected Constant/Where binding")
        output = source.output[0]
        value = values[output]
        require(value.dtype == np.float32 and value.ndim == 4 and min(value.shape) > 0
                and np.isneginf(value).all(), "Expected exclusively FP32 -inf padding tensor")
        uses = [(node.name, position) for node in nodes for position, name in enumerate(node.input)
                if name == output]
        require(uses == [(where_name, 2)] and output not in {x.name for x in model.graph.output},
                "Padding tensor must have only the intended Where false-branch use")
        logits = where.input[1]
        require(shapes.get(logits) == value.shape, "Logits static shape differs from padding tensor")
        stem = "v29_group_max_" + str(number)
        gather_name, expand_name = stem + "/GatherFirstSlot", stem + "/ExpandFirstSlot"
        gathered, shape_name = stem + "/first_slot", stem + "/original_shape"
        require(not ({gather_name, expand_name, gathered, shape_name} & existing),
                "New node/tensor names collide")
        added.append(numpy_helper.from_array(np.array(value.shape, dtype=np.int64), shape_name))
        replacements[constant_name] = [
            helper.make_node("Gather", [logits, index_name], [gathered],
                             axis=2, name=gather_name),
            helper.make_node("Expand", [gathered, shape_name], [output], name=expand_name),
        ]
        bindings.append({"replaced_node": constant_name, "where_node_unchanged": where_name,
                         "output_name_preserved": output, "logits": logits,
                         "original_shape": list(value.shape), "gather_axis": 2,
                         "gather_indices": [0], "gather_node": gather_name,
                         "expand_node": expand_name,
                         "removed_padding_tensor_raw_sha256": hashlib.sha256(value.tobytes()).hexdigest()})
    nonfinite = {name for name, value in values.items()
                 if value.dtype.kind == "f" and not np.isfinite(value).all()}
    require(nonfinite == {named[name].output[0] for name in target_names},
            "Unexpected additional nonfinite parameter or Constant")
    candidate = copy.deepcopy(model)
    rewritten = [replacement for node in nodes
                 for replacement in replacements.get(node.name, [node])]
    del candidate.graph.node[:]
    candidate.graph.node.extend(rewritten)
    candidate.graph.initializer.extend(added)
    onnx.checker.check_model(candidate, full_check=True)
    kept_before = [node.SerializeToString(deterministic=True) for node in nodes
                   if node.name not in target_names]
    new_names = {node.name for items in replacements.values() for node in items}
    kept_after = [node.SerializeToString(deterministic=True) for node in candidate.graph.node
                  if node.name not in new_names]
    require(kept_before == kept_after, "Original non-target node bytes/order changed")
    require([x.SerializeToString(deterministic=True) for x in model.graph.initializer]
            == [x.SerializeToString(deterministic=True)
                for x in candidate.graph.initializer[:len(model.graph.initializer)]],
            "Original initializer bytes/order changed")
    before_header, after_header = copy.deepcopy(model), copy.deepcopy(candidate)
    for header in (before_header, after_header):
        del header.graph.node[:]
        del header.graph.initializer[:]
    require(before_header.SerializeToString(deterministic=True)
            == after_header.SerializeToString(deterministic=True), "Model/graph header or IO changed")
    require(all(value.dtype.kind != "f" or np.isfinite(value).all()
                for value in constants(candidate).values()), "Nonfinite Constant remains")
    report = {"bindings": bindings, "parent_nodes": len(nodes),
              "candidate_nodes": len(candidate.graph.node),
              "parent_initializers": len(model.graph.initializer),
              "candidate_initializers": len(candidate.graph.initializer),
              "original_non_target_nodes_protobuf_sha256": hashlib.sha256(b"".join(kept_before)).hexdigest(),
              "original_initializers_protobuf_sha256": hashlib.sha256(b"".join(
                  x.SerializeToString(deterministic=True) for x in model.graph.initializer)).hexdigest(),
              "original_non_target_nodes_and_order_unchanged": True,
              "all_original_initializers_bytes_and_order_unchanged": True,
              "model_header_inputs_outputs_value_info_unchanged": True,
              "remaining_nonfinite_constants": 0,
              "sigmoid_nodes_unchanged": True,
              "full_ONNX_checker_passed": True}
    return candidate, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--export-sha256", required=True)
    parser.add_argument("--adapter-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    require(args.adapter_source.name == "dpvo_update_export.py",
            "Frozen adapter basename required for unambiguous provenance")
    require(args.export_sha256 == EXPORT_SHA and sha(args.export_dir / "export.json") == EXPORT_SHA,
            "Frozen original export receipt differs")
    require(sha(args.adapter_source) == ADAPTER_SHA, "Frozen topology/adapter source differs")
    exported = json.loads((args.export_dir / "export.json").read_text())
    require(exported["capture_sha256"] == CAPTURE_SHA and exported["adapter_sha256"] == ADAPTER_SHA
            and exported["exact_wrapper_FP32_gate_passed"] and exported["original_weights_unchanged"],
            "Frozen original capture/weights/FP32 chain differs")
    pins = {Path(__file__).name: sha(__file__), args.adapter_source.name: ADAPTER_SHA,
            "export.json": EXPORT_SHA}
    paths = {Path(__file__).name: Path(__file__), args.adapter_source.name: args.adapter_source,
             "export.json": args.export_dir / "export.json"}
    for key, digest in PARENT_GRAPH_SHAS.items():
        item = exported["models"][key]
        require(item["file"] == key + ".onnx" and item["sha256"] == digest,
                "Frozen parent graph binding differs")
        path = args.export_dir / item["file"]
        require(sha(path) == digest and path.stat().st_size == item["bytes"], "Parent graph file differs")
        pins[path.name], paths[path.name] = digest, path
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema": "v29_DPVO_Update_finite_group_max_static_recipe",
              "status": "rewritten_not_runtime_validated", "script_sha256": sha(__file__),
              "capture_sha256": CAPTURE_SHA, "export_sha256": EXPORT_SHA,
              "adapter_sha256": ADAPTER_SHA, "models": {}, "source_pins": pins,
              "static_graph_edit_only": True, "model_forward_executed": False,
              "ORT_executed": False, "board_execution": False,
              "original_weights_changed": False, "production_changed": False,
              "real_CPU_ORT_gate_passed": None, "NPU_runtime_validated": False,
              "guard_contract": {"finite_logits_required": True,
                                 "existing_v28_full_topology_preflight_required": True,
                                 "nonempty_group_real_members_form_prefix": True,
                                 "empty_groups_not_referenced_by_inverse": True,
                                 "empty_group_max_zero_and_denominator_plus_one_unchanged": True},
              "limitations": ["No root cause is established by this rewrite.",
                              "Finite logits and v28 validated topology are required.",
                              "Original board CPU FP32 and NPU tests are still mandatory.",
                              "Sigmoid/codomain repair is a separate candidate, not included."]}
    for key in PARENT_GRAPH_SHAS:
        item = exported["models"][key]
        parent = onnx.load(str(args.export_dir / item["file"]), load_external_data=False)
        require(len(parent.graph.input) == 13 and tuple(x.name for x in parent.graph.input) == INPUT_NAMES
                and len(parent.graph.initializer) == 50 and len(parent.graph.node) == 133,
                "Original complete Update graph structure differs")
        candidate, checks = rewrite_model(parent)
        path = args.output_dir / item["file"]
        onnx.save_model(candidate, str(path), save_as_external_data=False)
        require(protobuf_sha(onnx.load(str(path))) == protobuf_sha(candidate), "Saved candidate differs")
        report["models"][key] = {"file": path.name, "bytes": path.stat().st_size,
                                  "sha256": sha(path), "parent_onnx_sha256": item["sha256"],
                                  "inputs": item["inputs"], "outputs": item["outputs"],
                                  "role": "finite_group_first_member_max_padding_candidate",
                                  "structural_checks": checks}
    report["source_hashes_after"] = {name: sha(path) for name, path in paths.items()}
    require(report["source_hashes_after"] == pins, "Frozen sources or parents changed")
    report["sources_and_inputs_unchanged"] = True
    output = args.output_dir / "rewrite.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "rewrite_sha256": sha(output),
                      "models": {key: value["sha256"] for key, value in report["models"].items()}}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
