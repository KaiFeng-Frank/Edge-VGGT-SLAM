#!/usr/bin/env python3
"""Saved-array/static-ONNX audit only; never runs Torch, ORT or an SDK.

Reuses pinned v33 artifact, numeric and initializer helpers. The original CPU
full-graph references remain historical v33 artifacts, not fresh gold execution.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path

REFERENCE_SHA = "3f45c586bcdfdd47cd81724fb4008e8dcac975d86257a1ec58b3fca97767ba13"
AUDITOR_SHA = "c3cc443bee247bb7d3f8adbdb6ce153119ddf9a41a1a6316d2c1e149dfc1236d"
SOURCES = {
    "source_frame_update_softmax.py": "b89b9b84d3afb47fd926decddde0077222dc07930bf75ee08acb410dc4f6f3f4",
    "export-v34-source-frame-softmax-board.py": "487e9c759eb11749902e39f9e96891e561d9f56fb606d79c0c555de331633e96",
    "source_frame_update.py": "cf1adb2224750cd0659139689f4b8a5a8107f2e733e8d1c6f86ea009c86a1bb2",
    "export-v33-source-frame-board.py": "1bb1df7dc4e61e6f3d60fcd1170b267c478fd9600de1f0481ec1a73a42fa1b73",
}
OUTPUTS = ("net_out", "delta", "weight")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_recorded_metric(computed, recorded, checks):
    # Unlike the old helper's successful-export assertion, rejection is legal.
    checks.check(set(recorded) == set(computed) - {"max_normalized_tolerance"}, "Comparison fields differ")
    for key, value in recorded.items():
        if key == "relative_l2" and value is not None:
            checks.check(math.isclose(value, computed[key], rel_tol=1e-10, abs_tol=1e-14), "Relative L2 differs")
        else:
            checks.check(value == computed[key], "Recorded comparison differs: " + key)


def check_scope(report, checks):
    checks.check(report["schema"] == "v33_DPVO_source_frame_Update_board_export"
                 and report["status"] == "completed_source_frame_export_diagnostic_not_adopted"
                 and report["experiment_version"] == 34 and report["expression_variant"] == "v34_direct_torch_softmax",
                 "Completed explicitly named v34 diagnostic required")
    for field in ("candidate_accepted", "host_model_inference", "NPU_used", "BA_executed", "valid_pose_output",
                  "realtime_validated", "trajectory_accuracy_validated", "FP32_thresholds_relaxed", "output_clipping",
                  "full_E23520_reference_reexecuted", "source_plan_recomputed"):
        checks.check(report[field] is False, "Unsubstantiated scope/acceptance claim: " + field)
    for field in ("actual_board_execution", "actual_inference_on_RK3588S", "original_VGGT_default_unchanged",
                  "different_model_from_VGGT", "outputs_finite", "weight_domain_valid", "original_parameters_unchanged",
                  "sources_and_inputs_unchanged", "affinity_restored", "ORT_session_reference_released"):
        checks.check(report[field] is True, "Execution/cleanup evidence differs: " + field)
    checks.check(report["ORB_used"] is False and report["added_LK_PnP_frontend"] is False
                 and report["cleanup_errors"] == [] and report["terminal_sigmoid"] == "stable"
                 and report["source_pins"] == report["source_hashes_after"], "Source/cleanup/method differs")


def audit(run_dir, manifest_sha256, reference_dir, source_dir, v33_source_dir):
    import onnx  # Static protobuf and checker only.
    run_dir, reference_dir, source_dir, v33_source_dir = map(Path, (run_dir, reference_dir, source_dir, v33_source_dir))
    helper_path = v33_source_dir / "audit-v33-source-frame-export.py"
    if sha(helper_path) != AUDITOR_SHA:
        raise ValueError("Frozen v33 auditor changed")
    spec = importlib.util.spec_from_file_location("v34_frozen_array_helpers", helper_path)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    checks = helpers.Checks()
    checks.check(sha(run_dir / "manifest.json") == manifest_sha256, "New manifest SHA differs")
    checks.check(sha(reference_dir / "manifest.json") == REFERENCE_SHA, "Historical reference manifest differs")
    report = json.loads((run_dir / "manifest.json").read_text())
    old = json.loads((reference_dir / "manifest.json").read_text())
    check_scope(report, checks)
    checks.check(report["script_sha256"] == SOURCES["export-v34-source-frame-softmax-board.py"]
                 and report["wrapper_sha256"] == SOURCES["source_frame_update_softmax.py"]
                 and report["parent_wrapper_sha256"] == SOURCES["source_frame_update.py"]
                 and report["parent_export_utility_sha256"] == SOURCES["export-v33-source-frame-board.py"]
                 and report["reference_manifest_sha256"] == REFERENCE_SHA, "Expression/source/reference ancestry differs")
    checks.check(report["original_weights_before"] == report["original_weights_after"] == old["original_weights_before"]
                 and len(report["original_weights_before"]) == 94
                 and report["original_Update_parameters"] == old["original_Update_parameters"]
                 and len(report["original_Update_parameters"]) == 50, "Original 94/50 parameter payload metadata differs")
    for name, value in report["original_Update_parameters"].items():
        checks.check(value == report["original_weights_before"]["update." + name], "Update/full-model parameter binding differs")
    for key in ("original_binding", "parent_original_binding"):
        binding = report[key]
        checks.check(binding["state_tensors"] == 50 and binding["state_dict_names_preserved"]
                     and binding["original_module_objects_shared"] and binding["original_parameter_objects_shared"],
                     "Original parameter identity binding differs")
    for name, expected in SOURCES.items():
        folder = source_dir if name in ("source_frame_update_softmax.py", "export-v34-source-frame-softmax-board.py") else v33_source_dir
        checks.check(sha(folder / name) == expected, "Current measured source changed: " + name)
        digests = [v for p, v in report["source_pins"].items() if Path(p).name == name]
        checks.check(digests == [expected], "Board source pin differs: " + name)
    model_item = report["model"]
    checks.check(Path(model_item["file"]).name == model_item["file"], "Model basename required")
    model_path = run_dir / model_item["file"]
    checks.check(not model_path.is_symlink() and model_path.stat().st_size == model_item["bytes"]
                 and sha(model_path) == model_item["sha256"], "ONNX artifact differs")
    model = onnx.load(model_path, load_external_data=False)
    onnx.checker.check_model(model, full_check=True)
    inputs, outputs = [helpers.graph_contract(v) for v in model.graph.input], [helpers.graph_contract(v) for v in model.graph.output]
    checks.check(inputs == report["inputs"] == old["inputs"] and outputs == report["outputs"] == old["outputs"],
                 "Stable three-input/output fixed contracts differ")
    counts = Counter(n.op_type for n in model.graph.node)
    axes = [next(onnx.helper.get_attribute_value(a) for a in n.attribute if a.name == "axis")
            for n in model.graph.node if n.op_type == "Softmax"]
    checks.check(counts["Softmax"] == 2 and sorted(axes) == [1, 2] and counts["ReduceMax"] == 0
                 and not {"Gather", "GatherElements", "GatherND", "Where"} & set(counts),
                 "Direct closed group Softmax graph differs")
    checks.check(dict(counts) == report["ONNX_operator_counts"] and len(model.graph.node) == report["onnx_nodes"]
                 and sorted(counts) == report["onnx_operations"] and report["group_Gather_Where_absent"] is True,
                 "Static graph summary differs")
    coverage = helpers.verify_initializers(model, report["original_Update_parameters"], report["parameter_coverage"], checks, onnx)
    gates, results = [], []
    checks.check(len(report["cases"]) == len(old["cases"]) == 2, "Two frozen real-state blocks required")
    for index, (case, prior) in enumerate(zip(report["cases"], old["cases"])):
        checks.check(case["index"] == index and case["role"] == prior["role"]
                     and case["source_plan"] == prior["source_plan"] and case["total_graph_edges"] == 23520
                     and case["all_source_blocks"] == prior["all_source_blocks"]
                     and case["prior_inputs_sha256"] == prior["inputs_artifact"]["sha256"], "Saved source/state identity differs")
        feeds = helpers.load_artifact(run_dir, case["inputs_artifact"], checks)
        original_feeds = helpers.load_artifact(reference_dir, prior["inputs_artifact"], checks)
        checks.check({k: helpers.metadata(v) for k, v in feeds.items()}
                     == {k: helpers.metadata(v) for k, v in original_feeds.items()}, "Actual block input bytes differ from v33")
        old_values = {key: helpers.load_artifact(reference_dir, prior[key + "_artifact"], checks)
                      for key in ("reference", "wrapper", "ort")}
        checks.check(case["historical_reference_bindings"] == {k: prior[k + "_artifact"] for k in old_values},
                     "Historical reference binding mislabelled")
        saved = {key: helpers.load_artifact(run_dir, case[key + "_artifact"], checks)
                 for key in ("reference", "parent", "wrapper", "ort")}
        checks.check({k: helpers.metadata(v) for k, v in saved["reference"].items()}
                     == prior["reference_artifact"]["arrays"], "Historical gold was changed")
        for label in ("parent", "wrapper", "ort"):
            values = saved[label]
            for name, width in zip(OUTPUTS, (384, 2, 2)):
                value = values[name]
                checks.check(list(value.shape) == [1, 10, 96, width] and str(value.dtype) == "float32",
                             "Saved output canonical contract differs")
            weight = values["weight"]
            checks.check((weight >= 0).all() and (weight <= 1).all(), "Weight outside domain")
            row = dict(case_index=index, method=label, versus_saved={})
            recorded_name = "ORT_vs_saved" if label == "ort" else label + "_vs_saved"
            for ref_name, reference in old_values.items():
                group = {}
                for name in OUTPUTS:
                    result = helpers.residuals(values[name], reference[name])
                    check_recorded_metric(result, case[recorded_name][ref_name][name], checks)
                    gates.append(result["gate_passed"])
                    group[name] = result
                row["versus_saved"][ref_name] = group
            results.append(row)
        for label, baseline, field in (("wrapper", "parent", "wrapper_vs_parent"), ("ort", "wrapper", "ORT_vs_wrapper")):
            for name in OUTPUTS:
                result = helpers.residuals(saved[label][name], saved[baseline][name])
                check_recorded_metric(result, case[field][name], checks)
                gates.append(result["gate_passed"])
    checks.check(type(report["strict_FP32_gate_passed"]) is bool and report["strict_FP32_gate_passed"] is all(gates),
                 "Overall strict gate status differs")
    return dict(status="passed_saved_export_evidence_audit", checks_passed=checks.passed, failed_checks=0,
                auditor_sha256=sha(__file__), reused_auditor_sha256=AUDITOR_SHA,
                manifest_sha256=manifest_sha256, historical_manifest_sha256=REFERENCE_SHA,
                source_sha256=SOURCES, ONNX=report["model"], ONNX_operator_counts=dict(counts), Softmax_axes=axes,
                original_parameter_coverage=coverage, comparisons=results, strict_comparisons=len(gates),
                strict_FP32_gate_passed=all(gates), candidate_accepted=False,
                full_graph_reference_is_historical=True, full_graph_execution_verified_for_this_run=False,
                host_model_or_ORT_execution=False, NPU_execution=False, trajectory_or_realtime_validated=False,
                limits=["Only two frozen D10/M96 blocks. Prior v33 gold is not fresh E23520 execution.",
                        "Original learned parameter values are bound to recorded original-oriented raw SHA and ONNX bytes; the host does not execute or reload the original pretrained model.",
                        "An evidence audit pass preserves any strict numerical rejection; it is not candidate adoption or a measured NPU speedup."])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "reference-dir", "source-dir", "v33-source-dir", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("New audit output required")
    try:
        report = audit(args.run_dir, args.manifest_sha256, args.reference_dir, args.source_dir, args.v33_source_dir)
    except Exception as error:
        report = dict(status="failed_saved_export_evidence_audit", error=repr(error), auditor_sha256=sha(__file__), candidate_accepted=False)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: report.get(k) for k in ("status", "checks_passed", "error", "strict_FP32_gate_passed")}))
    return 0 if report["status"] == "passed_saved_export_evidence_audit" else 1


if __name__ == "__main__":
    raise SystemExit(main())
