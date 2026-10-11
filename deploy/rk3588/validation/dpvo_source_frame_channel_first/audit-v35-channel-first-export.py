#!/usr/bin/env python3
"""Saved-array/static-ONNX v35 audit; never executes Torch, ORT or an SDK.

Reuse only frozen v33 array/hash/numeric utilities. The physical CF contract
is new; inverse layout permutations and initializer reshapes are independently
checked here. Original full-graph gold and topology are inherited, not rerun.
Keep the adjacent original DPVO MIT LICENSE and copyright attribution.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

INPUTS = ("net", "inp", "corr")
OUTPUTS = ("net_out", "delta", "weight")
REFERENCE_SHA = "09c19fd17c28060f2011f13ad3e500ed05748ed8b5e40482ac5903e51a50934b"
HELPER_SHA = "c3cc443bee247bb7d3f8adbdb6ce153119ddf9a41a1a6316d2c1e149dfc1236d"
SOURCES = {
    "source_frame_update_channel_first.py": "7d89d91c64158d34f72b17091ebeaf06369ef71cb330495f6c3ec39092035c57",
    "export-v35-source-frame-channel-first-board.py": "ea3d4119d3cfd7b7a319b8e78bdf58a1c12c01a60dd3492178f1a1e3303d7dd8",
    "source_frame_update_softmax.py": "b89b9b84d3afb47fd926decddde0077222dc07930bf75ee08acb410dc4f6f3f4",
    "export-v33-source-frame-board.py": "1bb1df7dc4e61e6f3d60fcd1170b267c478fd9600de1f0481ec1a73a42fa1b73",
    "dpvo_probe_common.py": "ba47164a4273e0d392f8e23892f17be1aaf89b7ef42c3ebd746e3a7b3fd946a9",
    "dpvo_cpu_ops.py": "15191a0d481176e5be80bb6c41c3fb3f4d057d95b18ab18f6e0efcd73793dd32",
}
PERMITTED = ["identity", "conv1x1_kernel_reshape", "channel_affine_broadcast_reshape"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_check(computed, recorded, checks):
    checks.check(set(recorded) == set(computed) - {"max_normalized_tolerance"}, "Comparison keys differ")
    for name, value in recorded.items():
        if name == "relative_l2" and value is not None:
            checks.check(math.isclose(value, computed[name], rel_tol=1e-10, abs_tol=1e-14), "Relative L2 differs")
        else:
            checks.check(value == computed[name], "Recorded comparison differs: " + name)


def check_layout_pair(cf, canonical, *, inputs, checks):
    names, widths = (INPUTS, (384, 384, 882)) if inputs else (OUTPUTS, (384, 2, 2))
    checks.check(tuple(cf) == tuple(canonical) == names, "Ordered physical and canonical arrays differ")
    for name, width in zip(names, widths):
        a, b = cf[name], canonical[name]
        checks.check(a.dtype == b.dtype == np.float32 and a.shape == (1, width, 10, 96)
                     and b.shape == (1, 10, 96, width), "New CF/canonical FP32 contract differs: " + name)
        checks.check(a.flags.c_contiguous and b.flags.c_contiguous and np.isfinite(a).all()
                     and np.isfinite(b).all(), "Saved finite C-order arrays required")
        checks.check(a.transpose(0, 2, 3, 1).tobytes(order="C") == b.tobytes(order="C"),
                     "Exact inverse layout permutation failed: " + name)
    if not inputs:
        checks.check(np.all((cf["weight"] >= 0) & (cf["weight"] <= 1)), "Weight outside original domain")


def verify_initializers(model, originals, coverage, checks, helpers, onnx):
    checks.check(coverage["permitted_transforms"] == PERMITTED and coverage["parameter_values_unchanged"] is True,
                 "Initializer transform contract differs")
    checks.check(coverage["original_tensor_count"] == len(originals)
                 and len(model.graph.initializer) == len({v.name for v in model.graph.initializer}),
                 "Original or initializer tensor count differs")
    rows = coverage["mapping"]
    checks.check(len(rows) == coverage["initializer_count"] == len(model.graph.initializer), "Initializer mapping count differs")
    covered, transforms = set(), Counter()
    for tensor, row in zip(model.graph.initializer, rows):
        checks.check(tensor.data_location != onnx.TensorProto.EXTERNAL and not tensor.external_data,
                     "External initializer data prohibited")
        value = onnx.numpy_helper.to_array(tensor)
        checks.check(value.dtype == np.float32 and np.isfinite(value).all(), "Finite FP32 learned initializer required")
        checks.check(tensor.name == row["initializer"] and helpers.metadata(value) == row["array"],
                     "Actual initializer payload differs")
        candidates = [("identity", value)]
        if value.ndim == 4 and value.shape[2:] == (1, 1):
            candidates.append(("conv1x1_kernel_reshape", value.reshape(value.shape[:2])))
        if value.ndim == 4 and value.shape[0] == 1 and value.shape[2:] == (1, 1):
            candidates.append(("channel_affine_broadcast_reshape", value.reshape(-1)))
        candidates = [(transform, helpers.metadata(array)) for transform, array in candidates]
        matches = [dict(parameter=name, transform=transform)
                   for name, original in originals.items()
                   for transform, array in candidates if original == array]
        checks.check(bool(matches) and matches == row["original_matches"],
                     "Original bytes/shape/reshape mapping differs: " + tensor.name)
        covered.update(v["parameter"] for v in matches)
        transforms.update(v["transform"] for v in matches)
    checks.check(covered == set(originals) and coverage["all_original_parameters_covered"] is True,
                 "Original parameter payload coverage incomplete")
    return dict(parameters=len(originals), initializers=len(rows), all_covered=True,
                matched_transforms=dict(transforms), original_payload_reloaded=False,
                mapping_basis="Actual ONNX bytes reshaped back to original shape; raw SHA/dtype/bytes bound to frozen metadata")


def verify_channel_norm_graph(model, checks, onnx):
    producers = {name: node for node in model.graph.node for name in node.output}
    sqrts = [node for node in model.graph.node if node.op_type == "Sqrt"]
    checks.check(len(sqrts) == 4, "Four original channel LayerNorm denominators required")
    epsilons = []
    for node in sqrts:
        added = producers[node.input[0]]
        checks.check(added.op_type == "Add", "LayerNorm epsilon addition missing")
        children = [producers[name] for name in added.input]
        checks.check(sorted(v.op_type for v in children) == ["Constant", "ReduceMean"], "Population variance/epsilon graph differs")
        variance = next(v for v in children if v.op_type == "ReduceMean")
        eps = next(v for v in children if v.op_type == "Constant")
        tensor = next(onnx.helper.get_attribute_value(a) for a in eps.attribute if a.name == "value")
        value = onnx.numpy_helper.to_array(tensor)
        checks.check(value.dtype == np.float32 and value.size == 1 and value.item() == np.float32(1e-3).item(),
                     "Original FP32 LayerNorm epsilon changed")
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in variance.attribute}
        checks.check(attrs.get("axes") == [1] and attrs.get("keepdims", 1) == 1, "Channel population variance axes differ")
        square = producers[variance.input[0]]
        checks.check(square.op_type == "Mul" and len(square.input) == 2 and square.input[0] == square.input[1],
                     "Centered-square population variance required")
        centered = producers[square.input[0]]
        checks.check(centered.op_type == "Sub", "Channel variance is not centered")
        mean = producers[centered.input[1]]
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in mean.attribute}
        checks.check(mean.op_type == "ReduceMean" and mean.input[0] == centered.input[0]
                     and attrs.get("axes") == [1] and attrs.get("keepdims", 1) == 1,
                     "Centering must use mean over the same C axis")
        epsilons.append(float(value.item()))
    return dict(norms=4, channel_axis=1, FP32_epsilons=epsilons, centered_population_variance=True)


def load_helpers(source_dir):
    path = Path(source_dir) / "audit-v33-source-frame-export.py"
    if sha(path) != HELPER_SHA:
        raise ValueError("Frozen v33 array utilities differ")
    spec = importlib.util.spec_from_file_location("v35_saved_array_utilities", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def audit(run_dir, manifest_sha256, reference_dir, source_dir, v34_source_dir, v33_source_dir, v26_source_dir):
    import onnx  # Static protobuf/checker only; no runtime imported.
    run_dir, reference_dir, source_dir, v34_source_dir, v33_source_dir, v26_source_dir = map(
        Path, (run_dir, reference_dir, source_dir, v34_source_dir, v33_source_dir, v26_source_dir))
    helpers = load_helpers(v33_source_dir)
    checks = helpers.Checks()
    checks.check(sha(run_dir / "manifest.json") == manifest_sha256, "v35 manifest SHA differs")
    checks.check(sha(reference_dir / "manifest.json") == REFERENCE_SHA, "Frozen v34 manifest SHA differs")
    report = json.loads((run_dir / "manifest.json").read_text())
    old = json.loads((reference_dir / "manifest.json").read_text())
    checks.check(report["schema"] == "v35_DPVO_source_frame_channel_first_board_export"
                 and report["status"] == "completed_channel_first_export_diagnostic_not_adopted"
                 and report["experiment_version"] == 35
                 and report["expression_variant"] == "v35_channel_first_original_parameter_conv", "Explicit new CF experiment required")
    for name in ("candidate_accepted", "host_model_inference", "NPU_used", "BA_executed", "valid_pose_output", "realtime_validated",
                 "trajectory_accuracy_validated", "FP32_thresholds_relaxed", "output_clipping", "full_E23520_reference_reexecuted",
                 "source_plan_recomputed", "ORB_used", "added_LK_PnP_frontend", "SDK_or_NPU_placement_validated", "controlled_speedup_validated"):
        checks.check(report[name] is False, "Scope/acceptance flag overstates this audit: " + name)
    for name in ("actual_board_execution", "actual_inference_on_RK3588S", "original_VGGT_default_unchanged", "different_model_from_VGGT",
                 "outputs_finite", "weight_domain_valid", "original_parameters_unchanged", "sources_and_inputs_unchanged",
                 "affinity_restored", "ORT_session_reference_released"):
        checks.check(report[name] is True, "Execution/source/cleanup flag differs: " + name)
    checks.check(report["cleanup_errors"] == [] and report["terminal_sigmoid"] == "stable"
                 and report["source_pins"] == report["source_hashes_after"], "Method/source/cleanup differs")
    checks.check(report["input_layout"] == report["output_layout"] == "B,C,D,M"
                 and report["canonical_layout"] == "B,D,M,C", "New physical layout labels differ")
    checks.check(old["experiment_version"] == 34 and old["strict_FP32_gate_passed"] is True
                 and old["cleanup_errors"] == [], "Successful frozen v34 reference required")
    checks.check(report["script_sha256"] == SOURCES["export-v35-source-frame-channel-first-board.py"]
                 and report["wrapper_sha256"] == SOURCES["source_frame_update_channel_first.py"]
                 and report["parent_wrapper_sha256"] == SOURCES["source_frame_update_softmax.py"]
                 and report["parent_export_utility_sha256"] == SOURCES["export-v33-source-frame-board.py"]
                 and report["reference_manifest_sha256"] == REFERENCE_SHA, "Measured source ancestry differs")
    locations = {"source_frame_update_channel_first.py": source_dir, "export-v35-source-frame-channel-first-board.py": source_dir,
                 "source_frame_update_softmax.py": v34_source_dir, "export-v33-source-frame-board.py": v33_source_dir,
                 "dpvo_probe_common.py": v26_source_dir, "dpvo_cpu_ops.py": v26_source_dir}
    for name, expected in SOURCES.items():
        checks.check(sha(locations[name] / name) == expected, "Frozen current source differs: " + name)
        matches = [v for p, v in report["source_pins"].items() if Path(p).name == name]
        checks.check(matches == [expected], "Board source SHA differs: " + name)
    loader = report["loader"]
    checks.check(loader["commit"] == old["loader"]["commit"] == "0ac95b656d1fda91c271d2a106460d19ad966fc7"
                 and loader["checkpoint"] == old["loader"]["checkpoint"]
                 and loader["source_hashes"] == old["loader"]["source_hashes"], "Original source/checkpoint ancestry differs")
    for name, item in loader["source_hashes"].items():
        path = v26_source_dir / "official" / name
        checks.check(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], "Official current source differs: " + name)
        matches = [v for p, v in report["source_pins"].items() if p.endswith("/official/" + name)]
        checks.check(matches == [item["sha256"]], "Official measured source differs: " + name)
    for name, value in (("manifest.json", helpers.PINS["manifest.json"]), ("dpvo.pth", helpers.CHECKPOINT_SHA)):
        matches = [v for p, v in report["source_pins"].items() if Path(p).name == name]
        if name == "manifest.json":
            checks.check(sorted(matches) == sorted([REFERENCE_SHA, value])
                         and sha(v26_source_dir / name) == value, "Reference/source manifest pins differ")
        else:
            checks.check(matches == [value], "Original checkpoint pin differs")
    originals = report["original_Update_parameters"]
    checks.check(report["original_weights_before"] == report["original_weights_after"] == old["original_weights_before"]
                 and len(report["original_weights_before"]) == 94 and originals == old["original_Update_parameters"]
                 and len(originals) == 50, "Original 94/50 raw metadata differs")
    for name, item in originals.items():
        checks.check(item == report["original_weights_before"]["update." + name], "Update/full-model raw weight binding differs")
    for name in ("original_binding", "parent_original_binding"):
        binding = report[name]
        checks.check(binding["state_tensors"] == 50 and binding["original_parameter_objects_shared"] is True
                     and binding["original_module_objects_shared"] is True and binding["state_dict_names_preserved"] is True
                     and binding["terminal_sigmoid"] == "stable" and binding["stable_formula"] == "exp(min(x,0))/(1+exp(-abs(x)))",
                     "Original binding report differs")
    binding = report["original_binding"]
    checks.check(binding["input_output_layout"] == "B,C,D,M" and binding["channel_LayerNorm_axis"] == 1
                 and set(binding["LayerNorm_configs"]) == {"norm", "corr.3", "gru.0", "gru.2"}, "Channel LayerNorm configuration differs")
    for item in binding["LayerNorm_configs"].values():
        checks.check(item == dict(channels=384, eps=1e-3, elementwise_affine=True), "Original LayerNorm epsilon/affine changed")
    model_item = report["model"]
    checks.check(Path(model_item["file"]).name == model_item["file"] and model_item["file"] not in ("", ".", ".."), "Model basename required")
    path = run_dir / model_item["file"]
    checks.check(path.is_file() and not path.is_symlink() and path.stat().st_size == model_item["bytes"]
                 and sha(path) == model_item["sha256"], "ONNX payload differs")
    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model, full_check=True)
    inputs, outputs = [helpers.graph_contract(v) for v in model.graph.input], [helpers.graph_contract(v) for v in model.graph.output]
    expected_in = [dict(name=n, dtype="float32", shape=[1, c, 10, 96]) for n, c in zip(INPUTS, (384, 384, 882))]
    expected_out = [dict(name=n, dtype="float32", shape=[1, c, 10, 96]) for n, c in zip(OUTPUTS, (384, 2, 2))]
    checks.check(inputs == report["inputs"] == expected_in and outputs == report["outputs"] == expected_out,
                 "Three-input/output CF contracts differ")
    checks.check(report["canonical_inputs"] == old["inputs"] and report["canonical_outputs"] == old["outputs"], "Canonical bridge contract differs")
    counts = Counter(node.op_type for node in model.graph.node)
    softmax = [next(onnx.helper.get_attribute_value(a) for a in n.attribute if a.name == "axis")
               for n in model.graph.node if n.op_type == "Softmax"]
    reduce = [next(onnx.helper.get_attribute_value(a) for a in n.attribute if a.name == "axes")
              for n in model.graph.node if n.op_type == "ReduceMean"]
    checks.check(counts["Softmax"] == 2 and sorted(softmax) == [2, 3] and counts["Conv"] == 21
                 and counts["ReduceMean"] == 8 and all(v == [1] for v in reduce), "CF learned operations/axes differ")
    checks.check(not {"Transpose", "MatMul", "Gemm", "Gather", "GatherElements", "GatherND", "Where", "Pad"} & set(counts),
                 "Unexpected layout/group/arithmetic operations")
    checks.check(dict(counts) == report["ONNX_operator_counts"] and len(model.graph.node) == report["onnx_nodes"]
                 and sorted(counts) == report["onnx_operations"] and softmax == report["Softmax_axes"]
                 and reduce == report["channel_ReduceMean_axes"] and report["ONNX_Transpose_absent"] is True
                 and report["group_Gather_Where_absent"] is True, "Static graph summary differs")
    norms = verify_channel_norm_graph(model, checks, onnx)
    coverage = verify_initializers(model, originals, report["parameter_coverage"], checks, helpers, onnx)
    checks.check(len(report["cases"]) == len(old["cases"]) == 2, "Both fixed source-block states required")
    gates, results, layout_bytes, timings = [], [], 0, []
    for index, (case, prior) in enumerate(zip(report["cases"], old["cases"])):
        checks.check(case["index"] == index and case["role"] == prior["role"] and case["source_plan"] == prior["source_plan"]
                     and case["total_graph_edges"] == 23520 and case["all_source_blocks"] == prior["all_source_blocks"] == 17
                     and case["prior_inputs_sha256"] == prior["inputs_artifact"]["sha256"], "Frozen source/state identity differs")
        checks.check(case["source_plan"]["degree"] == 10 and case["source_plan"]["edges"] == 960
                     and case["source_plan"]["canonical_shape"] == [10, 96], "All original source members required")
        checks.check(case["input_layout"] == case["output_layout"] == "B,C,D,M"
                     and case["comparison_layout"] == "B,D,M,C", "Saved case layout differs")
        cf, canonical = {}, {}
        for label in ("inputs", "reference", "parent", "wrapper", "ort"):
            cf[label] = helpers.load_artifact(run_dir, case[label + "_artifact"], checks)
            canonical[label] = helpers.load_artifact(run_dir, case["canonical_" + label + "_artifact"], checks)
            check_layout_pair(cf[label], canonical[label], inputs=label == "inputs", checks=checks)
            layout_bytes += sum(v.nbytes for v in cf[label].values())
        old_inputs = helpers.load_artifact(reference_dir, prior["inputs_artifact"], checks)
        checks.check({k: helpers.metadata(v) for k, v in canonical["inputs"].items()} == prior["inputs_artifact"]["arrays"]
                     and all(np.array_equal(old_inputs[k], canonical["inputs"][k]) for k in INPUTS), "Input bytes not exact frozen v34 inputs")
        references = {label: helpers.load_artifact(reference_dir, prior[label + "_artifact"], checks)
                      for label in ("reference", "parent", "wrapper", "ort")}
        checks.check(case["historical_reference_bindings"] == {k: prior[k + "_artifact"] for k in references}, "Historical gold/comparison labels differ")
        checks.check(all(canonical["reference"][k].tobytes(order="C") == references["reference"][k].tobytes(order="C")
                         for k in OUTPUTS), "Inherited original gold changed")
        for method in ("parent", "wrapper", "ort"):
            row = dict(case_index=index, method=method, versus_saved={})
            field = "ORT_vs_saved" if method == "ort" else method + "_vs_saved"
            for label, reference in references.items():
                group = {}
                for name in OUTPUTS:
                    value = helpers.residuals(canonical[method][name], reference[name])
                    metric_check(value, case[field][label][name], checks)
                    group[name] = value
                    gates.append(value["gate_passed"])
                row["versus_saved"][label] = group
            results.append(row)
        for method, baseline, field in (("wrapper", "parent", "wrapper_vs_parent"), ("ort", "wrapper", "ORT_vs_wrapper")):
            group = {}
            for name in OUTPUTS:
                value = helpers.residuals(canonical[method][name], canonical[baseline][name])
                metric_check(value, case[field][name], checks)
                group[name] = value
                gates.append(value["gate_passed"])
            results.append(dict(case_index=index, method=method, versus_fresh=baseline, arrays=group))
        time_keys = ("CPU_input_pack_and_check_seconds", "parent_Torch_block_seconds", "Torch_block_seconds", "ORT_block_seconds",
                     "CPU_wrapper_restore_seconds", "CPU_ORT_restore_seconds")
        checks.check(all(type(case[k]) is float and math.isfinite(case[k]) and case[k] > 0 for k in time_keys), "Invalid timing boundary values")
        timings.append(dict(case_index=index, **{k: case[k] for k in time_keys}))
    checks.check(len(gates) == report["strict_comparison_count"] == 84
                 and type(report["strict_FP32_gate_passed"]) is bool and report["strict_FP32_gate_passed"] is all(gates),
                 "Fixed-gate numerical verdict/count differs")
    return dict(schema="v35_saved_array_static_ONNX_independent_review", status="passed_saved_export_evidence_audit",
                checks_passed=checks.passed, failed_checks=0, auditor_sha256=sha(__file__), reused_array_helper_sha256=HELPER_SHA,
                manifest_sha256=manifest_sha256, historical_v34_manifest_sha256=REFERENCE_SHA, source_sha256=SOURCES,
                model=report["model"], input_layout="B,C,D,M", canonical_layout="B,D,M,C", exact_layout_bridge_passed=True,
                CF_payload_bytes_inverse_checked=layout_bytes, original_state_tensors=94, original_Update_tensors=50,
                original_parameter_coverage=coverage, channel_LayerNorm_graph=norms, ONNX_operator_counts=dict(counts), Softmax_axes=softmax,
                comparisons=results, strict_comparisons=len(gates), strict_failed_comparisons=sum(not v for v in gates),
                strict_FP32_gate_passed=all(gates), candidate_accepted=False, timings=timings, cleanup_errors=[],
                sources_and_inputs_unchanged=True, host_model_or_ORT_execution=False, NPU_execution=False,
                full_graph_execution_verified_for_this_run=False, full_graph_source_plan_recomputed=False,
                trajectory_or_realtime_validated=False, controlled_speedup_validated=False,
                limits=["CF means physical B,C,D,M; layout permutations verified by exact bytes, not by shape alone.",
                        "Original 94/50 raw metadata is inherited and actual ONNX initializer payloads independently reversed to original shapes; no pretrained model is loaded by this audit.",
                        "Two complete D10/M96 source blocks only. Full E23520 topology and original CPU gold are inherited from frozen provenance, not freshly executed or reconstructed here.",
                        "An evidence audit pass preserves any fixed-gate numerical rejection and is never candidate adoption, SDK scheduling or 20Hz proof."])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "reference-dir", "source-dir", "v34-source-dir", "v33-source-dir", "v26-source-dir", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("New audit output required")
    try:
        result = audit(args.run_dir, args.manifest_sha256, args.reference_dir, args.source_dir,
                       args.v34_source_dir, args.v33_source_dir, args.v26_source_dir)
    except Exception as error:
        result = dict(schema="v35_saved_array_static_ONNX_independent_review", status="failed_saved_export_evidence_audit",
                      error=repr(error), auditor_sha256=sha(__file__), candidate_accepted=False,
                      host_model_or_ORT_execution=False, NPU_execution=False)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result.get(k) for k in ("status", "checks_passed", "error", "strict_FP32_gate_passed", "strict_failed_comparisons")}))
    return 0 if result["status"] == "passed_saved_export_evidence_audit" else 1


if __name__ == "__main__":
    raise SystemExit(main())
