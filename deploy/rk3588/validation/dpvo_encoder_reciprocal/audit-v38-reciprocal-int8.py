#!/usr/bin/env python3
"""Saved-only reciprocal encoder PTQ audit using frozen NumPy audit utilities.

MIT project helper; DPVO source attribution: Princeton Vision & Learning Lab.
No neural model, SDK, ORT, calibration or image decoding is executed. A passed
audit confirms evidence consistency, not the candidate's FP32 or VO acceptance.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import traceback

MATH_SHA = "65acb24c78ebc3adbe59174086f9ec694baf07fbd9f859de5a11ca02e9700808"
GOLD_SHA = "e42caab9534b0424ef3b55ed7d2b116a8bc6e9405abca58cbb73de7d78b896b3"
ENTRY_SHA = "a18045ee34d3e27ba4a28e07ef031abd69f1aef6a5c1af89173aded51cfcdadb"
QUERY_SHA = "78c5a8bafeca7686086ebe4fdaa7796af97f294b35e9c7d6c4ec8f90244bc3ed"
CPU_SHA = "a7f181015d4479856329460dc4099171d4bb0dce8cedf317c35e719f3f36548b"
PTQ_SHA = "72cad7e4a6a31d4860875c7d40cf2d672c1190ee3fedceb3271556fc52255f41"
REWRITER_SHA = "6ccda3f8c50dff5b36c8d65e9901df50fa28b8fb1624a141baedbeda34829d28"
PARENT_RUNTIME_SHA = "cc30f2bccf8b5b6178b62e8836eec5600504e927ac85b9d0f15814bcdb08df4a"
PARENT_QUERY_SHA = "c61a377ede3d33b5a2412b1c7f526ea75deff2f18ed185f0f937792e41b1aca2"
PARENT_GRAPH_SHA = "6c3b4de4d151813dc172d711b6a760ad8bdc17120f2f3faa76dbf32bd0c451ab"
SPATIAL_REVIEW_SHA = "4547fbf1863aa9b6a7e9fe9bcf023ecf720bcda90d763786044d6dfa141416e2"


def frozen(path, digest, name):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
        raise ValueError("Frozen NumPy audit source SHA differs: " + Path(path).name)
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def retained_rejection(value, checks):
    checks.check(value["diagnostic_allow_fp32_rejected"] is True and
                 value["historical_parent_FP32_gate_passed"] is False and
                 value["historical_parent_rejection_retained"] is True and value["candidate_accepted"] is False,
                 "Historical parent rejection is retained, not promoted")


def partial_score(actual, reference, recorded, base, checks):
    full = base.score(actual, reference)
    checks.check(set(recorded).issubset(full), "Recorded CPU comparison has known fields")
    checks.equal(recorded, {k: full[k] for k in recorded}, "Independent CPU reference comparison")
    return full


def reciprocal_gold(args, evidence, gold, base, checks, result):
    parents, candidates = {}, {}
    checks.check([(r["role"], r["frame_index"]) for r in evidence["calls"]] ==
                 [(role, frame) for frame in range(3) for role in ("parent", "candidate")], "All six paired CPU calls")
    for row in evidence["calls"]:
        frame, identity = row["frame_index"], gold[row["frame_index"]]["identity"]
        for key in ("frame_id", "source_image", "input_metadata", "centers", "indices"):
            checks.equal(row[key], identity[key], "New CPU reference uses original input identity: " + key)
        values = base.read_npz(args.cpu_dir, row["artifact"], row["arrays"], checks)
        checks.check(set(values) == set(base.SHAPES), "Two exact CPU feature outputs")
        for key, array in values.items():
            partial_score(array, gold[frame]["cpu"][key], row["versus_original_CPU"][key], base, checks)
            if row["role"] == "parent":
                checks.check(base.score(array, gold[frame]["ort"][key])["gate_passed"], "Current CPU parent reproduces saved v33 parent")
        (parents if row["role"] == "parent" else candidates)[frame] = values
        if row["role"] == "candidate":
            for key in values:
                partial_score(values[key], parents[frame][key], row["versus_current_parent_ORT"][key], base, checks)
            gold[frame]["reciprocal"] = values
    gates = {role: all(base.score(rows[i][k], gold[i]["cpu"][k])["gate_passed"] for i in range(3) for k in base.SHAPES)
             for role, rows in (("parent", parents), ("candidate", candidates))}
    same = all(base.score(candidates[i][k], parents[i][k])["gate_passed"] for i in range(3) for k in base.SHAPES)
    bitwise = all(candidates[i][k].tobytes(order="C") == parents[i][k].tobytes(order="C") for i in range(3) for k in base.SHAPES)
    checks.equal(evidence["original_CPU_strict_gate_each_role"], gates, "Both CPU original gates recomputed")
    checks.equal(evidence["candidate_vs_current_parent_strict_gate_passed"], same, "New CPU expression strict gate recomputed")
    checks.equal(evidence["candidate_vs_current_parent_bitwise_equal"], bitwise, "New CPU expression bytes recomputed")
    checks.check(same and not gates["parent"], "CPU expression diagnostic passes parent gate; historical rejection persists")
    result["CPU_expression_reference"] = dict(candidate_vs_parent_strict_gate_passed=same, candidate_vs_parent_bitwise_equal=bitwise,
        original_CPU_strict_gate_each_role=gates, six_CPU_NPZ_rehashed=True)


def check_run(value, profile, args, compiled, query, recipe, gold, base, checks):
    retained_rejection(value, checks)
    base.unchanged(value, checks, "Measured reciprocal runtime")
    checks.check(value["schema"] == "v38_DPVO_reciprocal_sparse_encoder_PTQ_inference_diagnostic" and
                 value["status"] == "completed_reciprocal_encoder_PTQ_parent_rejection_retained_not_VO" and
                 value["script_sha256"] == ENTRY_SHA, "Exact completed reciprocal entrypoint/schema")
    checks.check(value["compile_receipt_sha256"] == args.compile_sha256 and value["query_receipt_sha256"] == args.query_sha256 and
                 value["rewrite_sha256"] == args.rewrite_sha256 and value["CPU_ORT_receipt_sha256"] == args.cpu_sha256 and
                 value["capture_sha256"] == base.CAPTURE_SHA and value["reference_sha256"] == base.REFERENCE_SHA and
                 value["collect_perf_detail"] is profile, "All actual lineage receipts and normal/profile separation")
    for key in ("complete_VO", "realtime_validated", "Update_executed", "correlation_executed", "BA_executed", "images_decoded",
                "output_clipping", "manual_input_quantization", "production_changed", "ORB_used", "INT8_internal_operator_coverage_verified"):
        checks.check(value[key] is False, "No unsupported acceptance or execution: " + key)
    checks.check(value["actual_board_execution"] is True and value["NPU_used"] is True and value["normalization_applications"] == 0 and
                 value["affinity_restored"] is True and value["cpu_affinity"] == [4, 5, 6, 7], "Actual isolated execution/restore without preprocessing")
    checks.check(value["original_FP32_source_graph"] is False and value["reciprocal_and_spatial_FP32_graph_rewrites_used"] is True and
                 value["original_DPVO_weights_preserved"] is True and value["historical_FP16_receipt_sha256"] is None,
                 "Expression rewrite/unchanged weights/no fabricated FP16 reference")
    checks.check(value["source_reciprocal_graph_sha256"] == recipe["output"]["sha256"] and
                 value["source_parent_spatial_graph_sha256"] == PARENT_GRAPH_SHA and
                 value["reference_method"] == "spatial_parent_and_reciprocal_candidate", "Correct reciprocal graph and three-reference family")
    checks.equal(value["original_FP32_gate"], dict(rtol=base.RTOL, atol=base.ATOL), "Unchanged FP32 gate")
    checks.equal(value["configuration"], compiled["configuration"], "Actual PTQ configuration")
    checks.equal(value["model"], compiled["output"], "Actual quantized model")
    for key in ("sdk_versions", "input_attrs", "output_attrs"):
        checks.equal(value["context"][key], query[key], "Actual native IO including scale/zp/stride: " + key)
    checks.check(value["context"]["core_mask"] == 7 and value["context"]["context_destroyed"] is True and
                 value["context"]["destroy_return_code"] == 0, "Context actually destroyed")
    checks.check(value["parent_runtime_implementation_sha256"] == PARENT_RUNTIME_SHA and
                 value["parent_query_implementation_sha256"] == PARENT_QUERY_SHA and
                 value["execution_arithmetic_delegate_sha256"] == base.TRANSPORT_SHA, "Frozen SDK arithmetic and parent delegates")
    for name, digest in (("run-v38-reciprocal-int8-board.py", ENTRY_SHA), ("query-v38-reciprocal-int8-io-board.py", QUERY_SHA),
                         ("run-v37-spatial-int8-board.py", PARENT_RUNTIME_SHA), ("query-v37-spatial-int8-io-board.py", PARENT_QUERY_SHA),
                         ("v34_sparse_rknn.py", base.TRANSPORT_SHA), ("block-int8.rknn", compiled["output"]["sha256"])):
        base.source_pin(value, name, digest, checks)
    schedule = [(i, "profile") for i in range(3)] if profile else [(0, "cold"), (0, "warm"), (0, "warm"), (0, "warm"), (1, "other_frame"), (2, "other_frame")]
    checks.check([(r["frame_index"], r["timing_role"]) for r in value["calls"]] == schedule and
                 value["sdk_run_api_calls"] == len(schedule), "Complete prescribed six/three SDK calls")
    checks.equal(value["last_sdk_attempt"], value["calls"][-1]["timing"], "Final SDK attempt completed")
    for row in value["calls"]:
        identity = gold[row["frame_index"]]["identity"]
        for key in ("frame_id", "source_image", "input_metadata", "indices"):
            checks.equal(row[key], identity[key], "Same exact pixels/indices/source identity: " + key)
        base.check_timing(row, checks)
    warm = [r["forward_seconds"] for r in value["calls"] if r["timing_role"] == "warm"]
    checks.equal(value["timing_summary"], dict(cold_seconds=None if profile else value["calls"][0]["forward_seconds"],
        warm_seconds=warm, warm_mean_seconds=sum(warm)/len(warm) if warm else None, profile_timings_not_normal_benchmark=profile), "Actual timing mean")
    checks.check(value["actual_driver_seconds"] >= sum(r["forward_seconds"] for r in value["calls"]), "Enclosing driver covers checks/save")


def profile_shapes(detail, recipe, base, checks):
    summary = base.parse_profile(detail, checks)
    rows = []
    for line in detail["raw_text"].splitlines():
        if re.match(r"^\d+\s", line):
            f = line.split()
            rows.append(dict(index=int(f[0]), type=f[1], dtype=f[2], target=f[3], input_shape=f[4], output_shape=f[5], elapsed_us=int(f[7]), name=f[-1]))
    selected = []
    for replacement in recipe["selected_replacements"]:
        divs = [r for r in rows if r["name"] == "Div:" + replacement["new_nodes"][0]]
        muls = [r for r in rows if r["name"] in ("Mul:" + replacement["new_nodes"][1], "Mul:" + replacement["new_nodes"][1] + "_MulRelu")]
        checks.check(len(divs) == len(muls) == 1, "Every selected reciprocal Div/Mul appears exactly once")
        div, mul = divs[0], muls[0]
        shape = lambda dims: "(" + ",".join(map(str, dims)) + ")"
        small, full = shape(replacement["denominator_shape"]), shape(replacement["numerator_shape"])
        checks.check(div["target"] == "CPU" and div["type"] == "Div" and div["input_shape"] == "(1,1,1,1)," + small and
                     div["output_shape"] == small, "Actual CPU division only has channel-sized spatial dimensions")
        checks.check(mul["target"] == "NPU" and mul["type"] in ("Mul", "MulRelu") and mul["input_shape"] == full + "," + small and
                     mul["output_shape"] == full, "Actual full-size multiplication executes on NPU")
        selected.append(dict(original_node=replacement["old_node"], CPU_small_div=div, NPU_full_mul=mul))
    summary["selected_reciprocal_rows"] = selected
    summary["selected_CPU_Div_us"] = sum(r["CPU_small_div"]["elapsed_us"] for r in selected)
    summary["selected_NPU_Mul_us"] = sum(r["NPU_full_mul"]["elapsed_us"] for r in selected)
    return summary


def audit(args, base, gold_helper, checks, result):
    import numpy as np
    normal = base.pinned_json(args.normal_dir/"run.json", args.normal_sha256, checks)
    profile = base.pinned_json(args.profile_dir/"run.json", args.profile_sha256, checks)
    compiled = base.pinned_json(args.compile_json, args.compile_sha256, checks)
    query = base.pinned_json(args.query_json, args.query_sha256, checks)
    recipe = base.pinned_json(args.rewrite_json, args.rewrite_sha256, checks)
    cpu = base.pinned_json(args.cpu_dir/"ort.json", args.cpu_sha256, checks)
    for label, value in (("PTQ", compiled), ("Zero-run query", query), ("New CPU references", cpu)):
        retained_rejection(value, checks)
        base.unchanged(value, checks, label)
    checks.check(recipe["schema"] == "v38_DPVO_spatial_encoder_small_reciprocal_rewrite" and recipe["script_sha256"] == REWRITER_SHA and
                 recipe["parent"]["sha256"] == PARENT_GRAPH_SHA and recipe["sources_and_parent_unchanged"] is True and
                 recipe["source_pins"] == recipe["source_hashes_after"], "Frozen static recipe and parent unchanged")
    checks.check(compiled["schema"] == "v38_board_only_DPVO_reciprocal_spatial_encoder_PTQ" and
                 compiled["status"] == "compiled_board_reciprocal_PTQ_parent_rejection_retained_not_runtime_validated" and
                 compiled["script_sha256"] == PTQ_SHA and compiled["input"]["sha256"] == recipe["output"]["sha256"] and
                 compiled["rewrite_sha256"] == args.rewrite_sha256 and compiled["ORT_receipt_sha256"] == args.cpu_sha256 and
                 compiled["released"] is True and compiled["host_calibration_executed"] is False, "Successful board-only PTQ and exact reciprocal lineage")
    checks.equal(compiled["graph_preservation"], recipe["preservation"], "Recorded static preservation identity")
    checks.check(query["schema"] == "v38_DPVO_reciprocal_PTQ_native_IO_query" and
                 query["status"] == "queried_reciprocal_native_IO_parent_rejection_retained_not_inference_validated" and
                 query["script_sha256"] == QUERY_SHA and query["compile_receipt_sha256"] == args.compile_sha256 and
                 query["model_sha256"] == compiled["output"]["sha256"] and query["sdk_run_calls"] == 0 and
                 query["native_IO_contract_checked"] is True and query["context_destroyed"] is True and query["destroy_return_code"] == 0,
                 "Exact successful zero-run native-IO query")
    checks.check(cpu["schema"] == "v38_DPVO_spatial_reciprocal_board_CPU_ORT" and
                 cpu["status"] == "completed_CPU_ORT_reciprocal_diagnostic_parent_rejection_retained_not_VO" and
                 cpu["script_sha256"] == CPU_SHA and cpu["rewrite_sha256"] == args.rewrite_sha256 and
                 cpu["providers"] == ["CPUExecutionProvider"] and cpu["graph_optimization"] == "ORT_DISABLE_ALL",
                 "Actual fixed-input CPU evidence with disabled graph optimizations")
    gold, capture, reference = gold_helper.load_gold(args, base, checks, result)
    reciprocal_gold(args, cpu, gold, base, checks, result)
    checks.equal(normal["source_pins"], profile["source_pins"], "Normal/profile recorded source and input pins match")
    first, result["runs"] = {}, {}
    fields = {"versus_original_CPU": "cpu", "versus_v33_spatial_CPU_ORT": "ort", "versus_v38_reciprocal_CPU_ORT": "reciprocal"}
    for label, value, directory, is_profile in (("normal", normal, args.normal_dir, False), ("profile", profile, args.profile_dir, True)):
        check_run(value, is_profile, args, compiled, query, recipe, gold, base, checks)
        calls, local = [], {}
        for index, row in enumerate(value["calls"]):
            checks.check(row["index"] == index, "Actual SDK call order")
            arrays = base.read_npz(directory, row["artifact"], row["arrays"], checks)
            checks.check(set(arrays) == set(base.SHAPES), "Both actual feature outputs saved")
            frame = row["frame_index"]
            out = dict(index=index, frame_index=frame, frame_id=row["frame_id"], artifact=row["artifact"], arrays=row["arrays"], timing=row["timing"])
            for field, destination in fields.items():
                out[field] = {}
                for key, array in arrays.items():
                    checks.check(array.dtype == np.float32 and array.shape == base.SHAPES[key] and np.any(array != 0), "Owned nonzero FP32 logical feature shape")
                    computed = base.score(array, gold[frame][destination][key])
                    checks.equal(row[field][key], computed, "Independent saved-array error: " + field + "." + key)
                    out[field][key] = computed
            counts = {k: dict(nonfinite=int(np.count_nonzero(~np.isfinite(v))), all_zero=bool(np.all(v == 0))) for k, v in arrays.items()}
            checks.equal(row["finite_counts"], counts, "Actual nonfinite/allzero counts")
            checks.check(row["finite_nonzero"] is True, "Finite/nonzero claim matches saved arrays")
            repeated = None if frame not in local else all(arrays[k].tobytes(order="C") == local[frame][k].tobytes(order="C") for k in arrays)
            checks.equal(row["repeat_raw_hash_equal"], repeated, "Same-input repeats compare C-order bytes")
            local.setdefault(frame, arrays)
            if is_profile:
                checks.check(all(arrays[k].tobytes(order="C") == first[frame][k].tobytes(order="C") for k in arrays), "Normal/profile output bytes match")
                out["profile"] = profile_shapes(row["perf_detail"], recipe, base, checks)
            else:
                first.setdefault(frame, arrays)
                checks.check("perf_detail" not in row, "Normal timing does not include profiling")
            calls.append(out)
        gates = {field: all(v["gate_passed"] for row in calls for v in row[field].values()) for field in fields}
        checks.equal(value["actual_original_CPU_strict_gate_passed"], gates["versus_original_CPU"], "Original strict gate recomputed")
        checks.equal(value["actual_reciprocal_CPU_ORT_strict_gate_passed"], gates["versus_v38_reciprocal_CPU_ORT"], "Own CPU expression strict gate recomputed")
        timed = calls if is_profile else [calls[i] for i, row in enumerate(value["calls"]) if row["timing_role"] == "warm"]
        log = directory/"launcher.log"
        errors = [line for line in log.read_text(errors="replace").splitlines() if re.search(r"(?:^|\s)E\s+RKNN\b|Traceback|normalize error|Unsupport CPU op|Error:", line)]
        checks.check(not errors and "error" not in value, "No pulled launcher SDK error markers")
        result["runs"][label] = dict(receipt_sha256=args.profile_sha256 if is_profile else args.normal_sha256,
            call_count=len(calls), saved_output_arrays=2*len(calls), calls=calls, strict_gate_each_reference=gates,
            timing_summary=value["timing_summary"], mean_nested_stage_seconds={k: sum(r["timing"][k] for r in timed)/len(timed) for k in base.STAGES},
            release_records_checked=len(calls), successful_context_destroy_records=1, actual_driver_seconds=value["actual_driver_seconds"],
            peak_rss_mib=value["peak_rss_mib"], native_context=value["context"],
            launcher_log=dict(file=log.name, sha256=base.sha(log), error_markers=errors, kernel_system_logs_inspected=False))
    if args.spatial_int8_review is not None:
        old = base.pinned_json(args.spatial_int8_review, SPATIAL_REVIEW_SHA, checks)
        checks.check(old["status"] == "passed_evidence_consistent_parent_rejection_retained" and old["failed_checks"] == 0, "Historical saved-only spatial audit passed")
        result["historical_spatial_comparison"] = dict(controlled_AB_test=False, receipt_sha256=SPATIAL_REVIEW_SHA,
            prior_three_warm_mean_seconds=old["runs"]["normal"]["timing_summary"]["warm_mean_seconds"],
            reciprocal_three_warm_mean_seconds=normal["timing_summary"]["warm_mean_seconds"],
            frame_profiles=[dict(frame_index=i, old_operator_us=a["profile"]["total_operator_us"], new_operator_us=b["profile"]["total_operator_us"],
                old_CPU_Div_us=a["profile"]["operator_type_count_and_time"]["Div"][1], new_CPU_Div_us=b["profile"]["selected_CPU_Div_us"],
                new_selected_NPU_Mul_us=b["profile"]["selected_NPU_Mul_us"], old_CPU_Transpose_us=a["profile"]["CPU_transpose_us"],
                new_CPU_Transpose_us=b["profile"]["CPU_transpose_us"]) for i, (a, b) in enumerate(zip(old["runs"]["profile"]["calls"], result["runs"]["profile"]["calls"]))],
            interpretation="Historical sessions differ; observed profile placement/time supports the bottleneck diagnosis, not a controlled whole-VO speedup.")
    result["evidence_scope"] = dict(SDK_NPZ_rehashed=9, SDK_feature_arrays_rehashed=18, new_CPU_reference_NPZ_rehashed=6,
        original_CPU_full_imap_gather_independently_recomputed=True, normal_profile_and_repeats_C_order_bytes_equal=True,
        release_records_checked=9, successful_runtime_context_destroys=2, SDK_logs_and_actual_profile_parsed=True,
        board_recorded_source_before_after_maps_checked=True, unavailable_board_sources_weights_libraries_RKNN_host_rehashed=False,
        calibration_pixels_not_reread=True, original_graph_protobuf_preservation_recorded_not_reexecuted_here=True,
        actual_sdk_dtype_labels_not_independent_kernel_precision=True, historical_parent_FP32_rejection_retained=True,
        candidate_accepted=False, complete_VO_trajectory_accuracy_20Hz_not_validated=True, neural_ORT_SDK_execution_in_this_audit=False,
        unchanged_thresholds=dict(rtol=base.RTOL, atol=base.ATOL), profile_stage_timings_nested_not_added_twice=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("normal-dir", "profile-dir", "capture-dir", "reference-dir", "cpu-dir", "rewrite-json", "compile-json", "query-json", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("normal", "profile", "cpu", "rewrite", "compile", "query"):
        p.add_argument("--" + key + "-sha256", required=True)
    siblings = Path(__file__).resolve().parent.parent/"dpvo_encoder_int8"
    p.add_argument("--math-auditor", type=Path, default=siblings/"audit-v37-dpvo-int8.py")
    p.add_argument("--gold-auditor", type=Path, default=siblings/"audit-v37-spatial-int8.py")
    p.add_argument("--spatial-int8-review", type=Path)
    args = p.parse_args(argv)
    result = dict(schema="v38_DPVO_reciprocal_INT8_saved_artifact_independent_review", status="failed", candidate_accepted=False,
        audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frozen_NumPy_math_sha256=MATH_SHA, frozen_gold_loader_sha256=GOLD_SHA,
        bindings={k: v for k, v in vars(args).items() if k.endswith("sha256")})
    checks = None
    try:
        base = frozen(args.math_auditor, MATH_SHA, "_v38_frozen_numpy_math")
        helper = frozen(args.gold_auditor, GOLD_SHA, "_v38_frozen_numpy_gold")
        checks = base.Checks()
        audit(args, base, helper, checks, result)
        checks.check(base.sha(args.math_auditor) == MATH_SHA and base.sha(args.gold_auditor) == GOLD_SHA, "Frozen NumPy utilities unchanged")
        result.update(status="passed_evidence_consistent_parent_and_candidate_FP32_rejected", failed_checks=0)
    except Exception as exc:
        result.update(error=repr(exc), traceback=traceback.format_exc(), failed_checks=1)
    result["checks"] = checks.count if checks is not None else 0
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "checks", "failed_checks")}), flush=True)
    return 0 if result["failed_checks"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
