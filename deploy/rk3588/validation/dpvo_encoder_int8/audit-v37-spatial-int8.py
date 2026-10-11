#!/usr/bin/env python3
"""Independent saved NumPy/JSON spatial PTQ audit, with frozen audit math.

MIT project helper. No model/SDK/ORT, calibration, decoding or image processing.
The original INT8 review is a historical, non-controlled component comparison.
"""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re
import sys
import traceback

BASE_AUDITOR_SHA = "65acb24c78ebc3adbe59174086f9ec694baf07fbd9f859de5a11ca02e9700808"
ENTRY_SHA = "cc30f2bccf8b5b6178b62e8836eec5600504e927ac85b9d0f15814bcdb08df4a"
QUERY_SHA = "c61a377ede3d33b5a2412b1c7f526ea75deff2f18ed185f0f937792e41b1aca2"
COMPILER_SHA = "0fc013d5b6ed45cb97d592e8cd115304d971a000e2ab5353314413f56d41a3fa"
GRAPH_SHA = "6c3b4de4d151813dc172d711b6a760ad8bdc17120f2f3faa76dbf32bd0c451ab"
ORIGINAL_REVIEW_SHA = "d24faedb5f8eaabf2af0595fb3a6ef20a552ec3761e94bc6c874b98dedab0d2c"


def load_math(path):
    import hashlib
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != BASE_AUDITOR_SHA:
        raise ValueError("Frozen original saved-audit math SHA differs")
    spec = importlib.util.spec_from_file_location("_v37_frozen_saved_audit_math", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def spatial_declaration(value, checks):
    checks.check(value["schema"] == "v37_DPVO_spatial_sparse_encoder_PTQ_inference_diagnostic" and
                 value["status"] == "completed_spatial_encoder_PTQ_parent_rejection_retained_not_VO" and
                 value["script_sha256"] == ENTRY_SHA, "Frozen completed spatial runtime/schema")
    checks.check(value["diagnostic_allow_fp32_rejected"] is True and value["historical_parent_FP32_gate_passed"] is False and
                 value["historical_parent_rejection_retained"] is True and value["source_spatial_graph_sha256"] == GRAPH_SHA and
                 value["reference_method"] == "spatial" and value["reference_spatial_suffix_CPU_bitwise_gate_passed"] is True,
                 "Spatial suffix bound and parent precision rejection retained")
    checks.check(value["original_FP32_source_graph"] is False and value["spatial_FP32_graph_rewrite_used"] is True and
                 value["original_DPVO_weights_preserved"] is True and value["candidate_accepted"] is False,
                 "Rewritten numerical path is diagnostic, not original graph acceptance")
    checks.check(value["historical_FP16_receipt_sha256"] is None and
                 all("versus_historical_FP16" not in row for row in value["calls"]), "No unprovided historical FP16 reference invented")


def load_gold(args, base, checks, result):
    import numpy as np
    capture = base.pinned_json(args.capture_dir/"capture.json", base.CAPTURE_SHA, checks)
    reference = base.pinned_json(args.reference_dir/"ort.json", base.REFERENCE_SHA, checks)
    base.unchanged(capture, checks, "CPU gold")
    base.unchanged(reference, checks, "Spatial ORT gold")
    checks.check(capture["status"] == "completed_encoder_cpu_probe_not_VO" and capture["state_unchanged"] is True,
                 "Original CPU encoder and unchanged weights")
    checks.check(reference["status"] == "completed_suffix_bitwise_gate_parent_rejection_retained_not_VO" and
                 reference["suffix_bitwise_gate_passed"] is True and reference["original_CPU_strict_gate_each_method"]["spatial"] is False,
                 "Frozen suffix passes; spatial parent remains rejected")
    checks.equal(reference["preprocessing"], capture["preprocessing"], "Original preprocessing identity")
    rows = [row for row in reference["calls"] if row["method"] == "spatial"]
    checks.check([row["frame_index"] for row in rows] == [0, 1, 2], "All three spatial saved references, no outcome selection")
    gold = []
    result["references"] = []
    for frame, row in enumerate(rows):
        source = capture["inputs"][frame]
        checks.check(row["frame_id"] == source["frame_id"] and row["source_image"] == source["source"], "Exact RGB source identity")
        inputs = base.read_npz(args.reference_dir, row["input_artifact"],
            {"normalized_bgr": row["input_metadata"], "centers_xy": row["centers"], "patch_indices": row["indices"]}, checks)
        image, centers, indices = (inputs[k] for k in ("normalized_bgr", "centers_xy", "patch_indices"))
        original_image = np.load(base.artifact_path(args.capture_dir, source["input"], checks), allow_pickle=False)
        checks.equal(base.metadata(original_image), source["array"], "Original normalized input raw metadata")
        checks.check(image.dtype == np.float32 and image.shape == (1, 3, 464, 608) and
                     original_image.shape == (1, 1, 3, 464, 608) and image.tobytes(order="C") == original_image.tobytes(order="C"),
                     "Same already-normalized pixels; no reprocessing")
        checks.check(centers.dtype == indices.dtype == np.int64 and centers.shape == (96, 2) and indices.shape == (96,) and
                     np.array_equal(indices, centers[:, 1]*152 + centers[:, 0]), "Original ordered centers/index formula")
        cpu_row = next(item for item in capture["calls"] if item["frame_index"] == frame)
        dense = base.read_npz(args.capture_dir, cpu_row["artifact"], cpu_row["arrays"], checks)
        checks.check(set(dense) == {"fmap", "imap"} and dense["fmap"].shape == (1, 1, 128, 116, 152), "Complete original CPU feature gold")
        cpu = dict(fmap=np.ascontiguousarray(dense["fmap"][:, 0]), context=base.gather_context(dense["imap"], indices))
        saved = base.read_npz(args.reference_dir, row["artifact"], row["arrays"], checks)
        ort = {name: saved[name] for name in base.SHAPES}
        for name in ort:
            parent = saved["parent_" + ("fmap" if name == "fmap" else "context_gather")]
            checks.check(parent.tobytes(order="C") == ort[name].tobytes(order="C"), "Spatial suffix/parent actual saved bytes: " + name)
        parent_gates = {name: base.score(ort[name], cpu[name]) for name in cpu}
        checks.check(not all(value["gate_passed"] for value in parent_gates.values()), "Actual historical spatial-parent gate remains failed")
        result["references"].append(dict(frame_index=frame, frame_id=row["frame_id"], source_image=row["source_image"],
            input_metadata=row["input_metadata"], indices=row["indices"], CPU_gather_from_saved_full_imap_recomputed=True,
            original_CPU_arrays={k: base.metadata(v) for k, v in cpu.items()}, spatial_ORT_arrays={k: base.metadata(v) for k, v in ort.items()},
            spatial_parent_vs_original_CPU=parent_gates))
        gold.append(dict(identity=row, cpu=cpu, ort=ort))
    return gold, capture, reference


def check_run(value, profile, compiled, query, args, gold, base, checks):
    spatial_declaration(value, checks)
    base.unchanged(value, checks, "Spatial measured runtime")
    checks.check(value["compile_receipt_sha256"] == args.compile_sha256 and value["query_receipt_sha256"] == args.query_sha256 and
                 value["capture_sha256"] == base.CAPTURE_SHA and value["reference_sha256"] == base.REFERENCE_SHA and
                 value["collect_perf_detail"] is profile, "Measured model/references and normal/profile separation")
    for key in ("complete_VO", "realtime_validated", "Update_executed", "correlation_executed", "BA_executed", "images_decoded",
                "output_clipping", "manual_input_quantization", "production_changed", "ORB_used"):
        checks.check(value[key] is False, "No unsupported execution/quality claim: " + key)
    checks.check(value["actual_board_execution"] is True and value["NPU_used"] is True and value["normalization_applications"] == 0 and
                 value["affinity_restored"] is True and value["cpu_affinity"] == [4, 5, 6, 7], "Actual isolated execution and affinity restore")
    checks.equal(value["original_FP32_gate"], dict(rtol=base.RTOL, atol=base.ATOL), "Unchanged gate")
    checks.equal(value["configuration"], compiled["configuration"], "Actual PTQ configuration")
    checks.equal(value["model"], compiled["output"], "Actual spatial PTQ artifact")
    context = value["context"]
    for key in ("sdk_versions", "input_attrs", "output_attrs"):
        checks.equal(context[key], query[key], "Actual IO attr scale/zp/layout versus zero-run query: " + key)
    checks.check(context["core_mask"] == 7 and context["context_destroyed"] is True and context["destroy_return_code"] == 0,
                 "One actual context destroyed successfully")
    checks.check(value["parent_runtime_implementation_sha256"] == base.ENTRY_SHA and
                 value["parent_query_implementation_sha256"] == base.QUERY_SOURCE_SHA and
                 value["execution_arithmetic_delegate_sha256"] == base.TRANSPORT_SHA, "Exact original SDK delegate/source lineage")
    for basename, digest in (("run-v37-spatial-int8-board.py", ENTRY_SHA), ("query-v37-spatial-int8-io-board.py", QUERY_SHA),
                             ("run-v36-dpvo-int8-board.py", base.ENTRY_SHA), ("query-v36-int8-io-board.py", base.QUERY_SOURCE_SHA),
                             ("v34_sparse_rknn.py", base.TRANSPORT_SHA), ("block-int8.rknn", compiled["output"]["sha256"])):
        base.source_pin(value, basename, digest, checks)
    schedule = [(i, "profile") for i in range(3)] if profile else [(0, "cold"), (0, "warm"), (0, "warm"), (0, "warm"), (1, "other_frame"), (2, "other_frame")]
    checks.check([(row["frame_index"], row["timing_role"]) for row in value["calls"]] == schedule and
                 value["sdk_run_api_calls"] == len(schedule), "All six ordinary / three profile calls")
    checks.equal(value["last_sdk_attempt"], value["calls"][-1]["timing"], "Final SDK attempt completed")
    for row in value["calls"]:
        source = gold[row["frame_index"]]["identity"]
        checks.check(row["frame_id"] == source["frame_id"] and row["source_image"] == source["source_image"] and
                     row["input_metadata"] == source["input_metadata"] and row["indices"] == source["indices"], "Same exact input pixels, centers and original IDs")
        base.check_timing(row, checks)
    warm = [row["forward_seconds"] for row in value["calls"] if row["timing_role"] == "warm"]
    checks.equal(value["timing_summary"], dict(cold_seconds=None if profile else value["calls"][0]["forward_seconds"],
        warm_seconds=warm, warm_mean_seconds=sum(warm)/len(warm) if warm else None, profile_timings_not_normal_benchmark=profile), "Recorded exact timing means")
    checks.check(value["actual_driver_seconds"] >= sum(row["forward_seconds"] for row in value["calls"]), "Enclosing driver includes excluded checks/I/O")


def audit(args, base, checks, result):
    import numpy as np
    normal = base.pinned_json(args.normal_dir/"run.json", args.normal_sha256, checks)
    profile = base.pinned_json(args.profile_dir/"run.json", args.profile_sha256, checks)
    compiled = base.pinned_json(args.compile_json, args.compile_sha256, checks)
    query = base.pinned_json(args.query_json, args.query_sha256, checks)
    original = base.pinned_json(args.original_int8_review, ORIGINAL_REVIEW_SHA, checks)
    checks.check(original["status"] == "passed_evidence_consistent_candidate_FP32_rejected" and original["failed_checks"] == 0 and
                 original["audit_script_sha256"] == BASE_AUDITOR_SHA, "Previously independently audited original INT8 reference")
    checks.check(compiled["schema"] == "v37_board_only_DPVO_spatial_encoder_PTQ" and
                 compiled["status"] == "compiled_board_spatial_PTQ_parent_rejection_retained_not_runtime_validated" and
                 compiled["script_sha256"] == COMPILER_SHA and compiled["input"]["sha256"] == GRAPH_SHA and
                 compiled["ancestor"]["method"] == "spatial" and compiled["ancestor"]["parent_CPU_strict_gate_passed"] is False and
                 compiled["ancestor"]["suffix_CPU_bitwise_gate_passed"] is True and compiled["released"] is True and
                 compiled["diagnostic_allow_fp32_rejected"] is True and compiled["historical_parent_rejection_retained"] is True,
                 "Exact successful spatial PTQ compilation with parent rejection")
    base.unchanged(compiled, checks, "Spatial PTQ compile")
    checks.check(query["schema"] == "v37_DPVO_spatial_PTQ_native_IO_query" and
                 query["status"] == "queried_spatial_native_IO_parent_rejection_retained_not_inference_validated" and
                 query["script_sha256"] == QUERY_SHA and query["compile_receipt_sha256"] == args.compile_sha256 and
                 query["model_sha256"] == compiled["output"]["sha256"] and query["sdk_run_calls"] == 0 and
                 query["native_IO_contract_checked"] is True and query["context_destroyed"] is True and
                 query["historical_parent_FP32_gate_passed"] is False and query["historical_parent_rejection_retained"] is True,
                 "Same spatial quantized model's clean zero-run IO query")
    base.unchanged(query, checks, "Spatial zero-run query")
    gold, capture, reference = load_gold(args, base, checks, result)
    checks.equal(normal["source_pins"], profile["source_pins"], "Both actual runs share exact sources/inputs")
    first, result["runs"] = {}, {}
    for label, value, directory, is_profile in (("normal", normal, args.normal_dir, False), ("profile", profile, args.profile_dir, True)):
        check_run(value, is_profile, compiled, query, args, gold, base, checks)
        calls, local_first = [], {}
        for index, row in enumerate(value["calls"]):
            checks.check(row["index"] == index, "Exact measured call order")
            arrays = base.read_npz(directory, row["artifact"], row["arrays"], checks)
            checks.check(set(arrays) == set(base.SHAPES), "All actual outputs saved")
            frame = row["frame_index"]
            out = dict(index=index, frame_index=frame, frame_id=row["frame_id"], artifact=row["artifact"], arrays=row["arrays"], timing=row["timing"])
            for field, destination in (("versus_original_CPU", "cpu"), ("versus_v33_CPU_ORT", "ort")):
                out[field] = {}
                for name, array in arrays.items():
                    checks.check(array.dtype == np.float32 and array.shape == base.SHAPES[name] and np.any(array != 0), "Nonzero finite owned FP32 logical output")
                    computed = base.score(array, gold[frame][destination][name])
                    checks.equal(row[field][name], computed, "Independent saved-array metric: " + field + "." + name)
                    out[field][name] = computed
            checks.equal(row["finite_counts"], {k: dict(nonfinite=int(np.count_nonzero(~np.isfinite(v))), all_zero=bool(np.all(v == 0))) for k, v in arrays.items()}, "Actual nonfinite/zero counts")
            checks.check(row["finite_nonzero"] is True, "Finite/nonzero claim matches arrays")
            repeated = None if frame not in local_first else all(arrays[k].tobytes(order="C") == local_first[frame][k].tobytes(order="C") for k in arrays)
            checks.equal(row["repeat_raw_hash_equal"], repeated, "Ordinary same-input repeats raw bytes")
            local_first.setdefault(frame, arrays)
            if is_profile:
                checks.check(all(arrays[k].tobytes(order="C") == first[frame][k].tobytes(order="C") for k in arrays), "Profile and normal same-input raw bytes")
                out["profile"] = base.parse_profile(row["perf_detail"], checks)
            else:
                first.setdefault(frame, arrays)
                checks.check("perf_detail" not in row, "Ordinary timing without profile flag")
            calls.append(out)
        gate = all(v["gate_passed"] for row in calls for v in row["versus_original_CPU"].values())
        checks.equal(value["actual_original_CPU_strict_gate_passed"], gate, "Original gate independently recomputed")
        timed = calls if is_profile else [row for row in calls if value["calls"][row["index"]]["timing_role"] == "warm"]
        result["runs"][label] = dict(receipt_sha256=args.profile_sha256 if is_profile else args.normal_sha256,
            call_count=len(calls), saved_output_arrays=2*len(calls), calls=calls, original_CPU_strict_gate_passed=gate,
            v33_spatial_ORT_strict_gate_passed=all(v["gate_passed"] for row in calls for v in row["versus_v33_CPU_ORT"].values()),
            timing_summary=value["timing_summary"], mean_nested_stage_seconds={key: sum(row["timing"][key] for row in timed)/len(timed) for key in base.STAGES},
            release_records_checked=len(calls), successful_context_destroy_records=1, actual_driver_seconds=value["actual_driver_seconds"],
            peak_rss_mib=value["peak_rss_mib"], native_context=value["context"])
        log = directory/"launcher.log"
        text = log.read_text(errors="replace")
        errors = [line for line in text.splitlines() if re.search(r"(?:^|\s)E\s+RKNN\b|Traceback|normalize error|Unsupport CPU op|Error:", line)]
        checks.check(not errors and "error" not in value and value["cleanup_errors"] == [], "No error markers in pulled launcher logs/recorded cleanup")
        result["runs"][label]["launcher_log"] = dict(file=log.name, sha256=base.sha(log), error_markers=errors, kernel_system_logs_inspected=False)
    profiles = [row["profile"] for row in result["runs"]["profile"]["calls"]]
    old_profiles = [row["profile"] for row in original["runs"]["profile"]["calls"]]
    table = []
    for frame, (old, current) in enumerate(zip(old_profiles, profiles)):
        def times(p, target, op):
            return p["operator_type_count_and_time"].get(op, [0, 0, 0, 0, 0])[{"CPU": 1, "NPU": 3}[target]]
        table.append(dict(frame_index=frame, original_INT8_operator_us=old["total_operator_us"], spatial_INT8_operator_us=current["total_operator_us"],
            original_INT8_CPU_transpose_us=times(old, "CPU", "Transpose"), spatial_INT8_CPU_transpose_us=times(current, "CPU", "Transpose"),
            original_INT8_NPU_exNorm_us=times(old, "NPU", "exNorm"), spatial_INT8_NPU_exNorm_us=times(current, "NPU", "exNorm"),
            original_INT8_CPU_Div_us=times(old, "CPU", "Div"), spatial_INT8_CPU_Div_us=times(current, "CPU", "Div"),
            spatial_INT8_CPU_Sqrt_us=times(current, "CPU", "Sqrt"), spatial_INT8_CPU_Div_fraction_of_operator_time=times(current, "CPU", "Div")/current["total_operator_us"]))
    result["original_vs_spatial_INT8_profile_table"] = table
    result["ordinary_component_timings"] = dict(original_INT8_three_warm_mean_seconds=original["runs"]["normal"]["timing_summary"]["warm_mean_seconds"],
        spatial_INT8_three_warm_mean_seconds=normal["timing_summary"]["warm_mean_seconds"], controlled_AB_test=False,
        statement="Historical original and spatial runs are descriptive, not thermally controlled/interleaved. Separate actual profiles expose the CPU operation bottleneck.")
    result["observed_profile_fact"] = "Spatial INT8 removes CPU Transpose/exNorm rows but retains full-tensor CPU Div and small CPU Sqrt. Use the actual per-frame elapsed/type/target table; no unexecuted reciprocal/Mul rewrite is inferred to succeed."
    result["evidence_scope"] = dict(spatial_NPZ_rehashed=9, spatial_output_arrays_rehashed=18, CPU_full_imap_gather_independently_recomputed=True,
        original_INT8_comparison_source="Previously frozen independent review; original arrays not re-audited in this invocation.",
        board_source_before_after_SHA_maps_checked=True, unavailable_board_sources_checkpoint_libraries_model_host_rehashed=False,
        actual_SDK_profile_dtype_labels_parsed=True, kernel_precision_independently_measured=False, all_operations_on_NPU=False,
        original_CPU_strict_gate_passed=result["runs"]["normal"]["original_CPU_strict_gate_passed"], historical_spatial_parent_rejection_retained=True,
        candidate_accepted=False, complete_VO_or_trajectory_accuracy_or_20Hz_validated=False, SDK_or_neural_or_ORT_execution=False,
        unchanged_thresholds=dict(rtol=base.RTOL, atol=base.ATOL), timing_profile_inside_run_not_added_twice=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("normal-dir", "profile-dir", "capture-dir", "reference-dir", "compile-json", "query-json", "original-int8-review", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("normal", "profile", "compile", "query"):
        p.add_argument("--" + name + "-sha256", required=True)
    p.add_argument("--math-auditor", type=Path, default=Path(__file__).with_name("audit-v37-dpvo-int8.py"))
    args = p.parse_args(argv)
    result = dict(schema="v37_DPVO_spatial_INT8_saved_artifact_independent_review", status="failed", candidate_accepted=False,
                  bindings={k: v for k, v in vars(args).items() if k.endswith("sha256")}, original_audit_math_sha256=BASE_AUDITOR_SHA,
                  original_INT8_review_sha256=ORIGINAL_REVIEW_SHA)
    checks = None
    try:
        base = load_math(args.math_auditor)
        result["audit_script_sha256"] = base.sha(__file__)
        checks = base.Checks()
        audit(args, base, checks, result)
        checks.check(base.sha(args.math_auditor) == BASE_AUDITOR_SHA, "Frozen audit math unchanged after saved-only analysis")
        result.update(status="passed_evidence_consistent_parent_rejection_retained", failed_checks=0)
    except Exception as exc:
        result.update(error=repr(exc), traceback=traceback.format_exc(), failed_checks=1)
    result["checks"] = 0 if checks is None else checks.count
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "checks", "failed_checks")}), flush=True)
    return 0 if result["failed_checks"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
