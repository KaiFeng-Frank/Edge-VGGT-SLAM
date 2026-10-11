#!/usr/bin/env python3
"""Independent saved-array/timing audit; stdlib + NumPy, no model or SDK."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROLES = ("ordinary", "flash")
SAMPLES = ("evaluation_00", "evaluation_01")
INPUT_SHAPE, OUTPUT_SHAPE = [2, 437, 1024], [1, 874, 1024]
ELEMENTS, FP32_BYTES = 894976, 3579904
RTOL, ATOL = 1e-4, 1e-5
NORMAL_SHA = "20cfcdc923c3634d505604bc59c44efac4d971a3db51999b700c1100e12ddc55"
PROFILE_SHA = "a4760f445d4d69790c67b09364e3408db0f9fdd7d8d493923aff524c2fe2ee1b"
CAPTURE_SHA = "bf39dfa65526a4135cbd2d7ee8b927ce2c4a40da75aebf64652ed2cd87a7e7fc"
CAPTURE_SCRIPT_SHA = "7a2279927a65e3740da933c1192c335d15f4f1803440ed84a27a2aebc8f02613"
CAPTURE_CONTRACT_SHA = "6ce1f87f1683e8082a4d567718acb853101e5ee6bbaeef1ced02993a8e3133c8"
CHECKPOINT_SHA = "d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0"
GRAPH_SHA = "2cf1f4de19c91b42b5bf685ab5e949bd98b08a64fc2e78b7f0c8f07992a009fc"
RUNNER_SHA = "10b553b7c2a005310c6b6d00a5f6442668e6597b9f1a0d7ee119f88fd349e9a6"
COMPILER_SHA = "709544077448718fe866e8e288810ac43110fb2afc98bd3a39cde5d6d3a965b0"
PARENT_SHA = "33fc5deb749c8b3ceaa47271e389f5fbbfc2dba6db93aa19887b2fc41e0e7882"
CONTRACT_SHA = "44c0a6a1dcca66ee09b2f15bb0e2f0ddc7b7eb3f6cb44fdc3b576e9f75463bc2"
COMPILE_SHA = "d19b34f90ee1c7d41724fd36fb4ba808a880863f57ce58f053cc292d7e705d34"
STAGES = ("input_set_seconds", "run_api_seconds", "outputs_get_seconds",
          "output_validation_and_copy_seconds", "pre_release_bookkeeping_seconds", "outputs_release_seconds")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_sha(value):
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def pin(path, expected, pins):
    path = Path(path).resolve(strict=True)
    require(sha(path) == expected, "SHA-pinned saved data/source changed")
    pins[path] = expected
    return path


def numeric_same(actual, declared, label):
    if actual is None or type(actual) in (bool, int, str):
        require(type(actual) is type(declared) and actual == declared, "Declared " + label + " differs")
    else:
        require(type(declared) in (int, float) and math.isfinite(declared)
                and math.isclose(actual, declared, rel_tol=2e-11, abs_tol=1e-12), "Declared " + label + " differs")


def ordered_sum(values):
    # Match the actual board CPython 3.11 order, independent of host sum changes.
    total = 0.0
    for value in values:
        total += value
    return total


def expected_calls():
    return [(sid, role, iteration) for sample_index, sid in enumerate(SAMPLES)
            for iteration in range(4) for role in (ROLES if (sample_index + iteration) % 2 == 0 else ROLES[::-1])]


def six_call_summary(calls, role):
    require(role in ROLES, "Unknown comparison role")
    expected = [(i, s, r) for i, (s, who, r) in enumerate(expected_calls()) if who == role and r > 0]
    selected = [x for x in calls if x["role"] == role and x["round_index"] > 0]
    require([(x["sdk_call_index"], x["sample_id"], x["round_index"]) for x in selected] == expected,
            "Exactly the six ordered subsequent calls, three per input, are required")
    values = [x["io_total_seconds"] for x in selected]
    require(all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in values), "Finite positive I/O time required")
    return {"mean_denominator": 6, "included_sdk_call_indices": [x[0] for x in expected],
            "subsequent_IO_mean_seconds": ordered_sum(values) / 6,
            "subsequent_IO_min_seconds": min(values), "subsequent_IO_max_seconds": max(values),
            "aggregation": "ordered left-to-right float addition, then division by exactly six",
            "first_for_each_input_excluded_even_if_context_already_warm": True}


def original_metrics(actual, reference, expected_shape=OUTPUT_SHAPE):
    import numpy as np
    require(actual.dtype == reference.dtype == np.float32 and list(actual.shape) == list(reference.shape) == expected_shape
            and np.isfinite(actual).all() and np.isfinite(reference).all(), "Same finite original FP32 tensor shape required")
    a, r = actual.astype(np.float64), reference.astype(np.float64)
    delta, tolerance = a - r, ATOL + RTOL * np.abs(r)
    absolute = np.abs(delta)
    rn, an = float(np.linalg.norm(r)), float(np.linalg.norm(a))
    return {"finite": True, "array_equal": bool(np.array_equal(actual, reference)),
            "max_abs_error": float(absolute.max()), "rmse": float(np.sqrt(np.mean(delta * delta))),
            "relative_l2": float(np.linalg.norm(delta) / rn) if rn else None,
            "cosine_similarity": float(np.sum(a * r) / (rn * an)) if rn and an else None,
            "rtol": RTOL, "atol": ATOL, "strict_FP32_gate_passed": bool(np.all(absolute <= tolerance)),
            "outside_tolerance_elements": int(np.count_nonzero(absolute > tolerance)),
            "max_tolerance_ratio": float(np.max(absolute / tolerance)), "output_raw_sha256": raw_sha(actual)}


def load_array(directory, item, expected_shape, pins):
    import numpy as np
    name = Path(item["file"])
    require(not name.is_absolute() and ".." not in name.parts and not any(x.isspace() for x in str(name)), "NPY path escaped")
    path = (directory / name).resolve(strict=True)
    require(path.is_relative_to(directory.resolve()) and path.is_file()
            and path.stat().st_size == item["bytes"] and item["bytes"] < 16 << 20, "NPY path/bytes changed")
    pin(path, item["sha256"], pins)
    value = np.load(path, allow_pickle=False)
    require(list(value.shape) == item["shape"] == expected_shape and str(value.dtype) == item["dtype"] == "float32"
            and value.flags.c_contiguous and np.isfinite(value).all()
            and value.nbytes == item["tensor_bytes"] == FP32_BYTES and raw_sha(value) == item["raw_sha256"],
            "NPY dtype/shape/layout/finite/raw SHA changed")
    return value


def audit_interval(sdk):
    require(sdk["status"] == "completed" and sdk["synchronous"] is True and sdk["input_api_type"] == "FP32"
            and sdk["input_api_type_enum"] == 0 and sdk["input_pass_through"] == 0 and sdk["input_format"] == 3
            and sdk["input_shape"] == INPUT_SHAPE and sdk["output_shape"] == OUTPUT_SHAPE
            and sdk["input_bytes"] == sdk["expected_FP32_output_bytes"] == sdk["output_bytes"] == FP32_BYTES
            and sdk["want_float"] == 1 and sdk["is_prealloc"] == 0 and sdk["manual_quantization_or_dequantization"] is False
            and sdk["output_buffer_released"] is True and sdk["output_owned_copy_before_release"] is True
            and sdk["npu_run_api_called"] is True, "Saved synchronous SDK get/copy/release contract failed")
    require(sdk["runtime_frame_id"] == sdk["output_frame_id"] and sdk["SDK_output_pointer"] > 0
            and sdk["owned_output_pointer"] > 0 and sdk["SDK_output_pointer"] != sdk["owned_output_pointer"]
            and all(type(sdk[k]) is int and sdk[k] == 0 for k in
                    ("inputs_set_code", "run_code", "outputs_get_code", "outputs_release_code")), "SDK ownership/success identity changed")
    begin, end = sdk["io_begin_monotonic_ns"], sdk["io_end_monotonic_ns"]
    require(type(begin) is type(end) is int and end > begin, "Positive monotonic integer interval required")
    stages = [sdk[k] for k in STAGES]
    require(all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in stages), "Finite nonnegative stages required")
    ns = [round(x * 1e9) for x in stages]
    require(all(math.isclose(n / 1e9, x, rel_tol=0, abs_tol=1e-15) for n, x in zip(ns, stages))
            and sum(ns) == end - begin, "SDK stage intervals must sum to complete I/O interval in integer nanoseconds")
    numeric_same((end - begin) / 1e9, sdk["io_total_seconds"], "complete monotonic I/O interval")


def check_descriptors(context, normal):
    require(context["init_flags"] == (0 if normal else 8) and context["core_mask"] == 7
            and context["context_destroyed"] is True and context["tensor_attributes_validated"] is True
            and all(context[k] == 0 for k in ("init_code", "core_mask_code", "destroy_code"))
            and context["sdk_version"].startswith("2.3.2 ") and context["driver_version"] == "0.7.2", "Context lifecycle changed")
    for direction, shape in (("input", INPUT_SHAPE), ("output", OUTPUT_SHAPE)):
        attr = context[direction + "_attr"]
        require(attr["index"] == 0 and attr["name"] == direction and attr["shape"] == shape
                and attr["n_elems"] == ELEMENTS and attr["size"] == ELEMENTS * 2 and attr["format"] == 3
                and attr["type"] == 1 and attr["quantization_type"] == 2 and attr["zero_point"] == 0
                and attr["scale"] == 1.0 and attr["pass_through"] == 0, "Actual original FP16 logical descriptor changed")
    return {d: context[d + "_attr"] for d in ("input", "output")}


def audit_cohort(directory, expected_sha, normal, evaluations, references, pins):
    import numpy as np
    run = json.loads(pin(directory / "run.json", expected_sha, pins).read_text())
    require(run["schema"] == "v50_original_pair_flash_SDK_AB" and run["status"] == "completed_diagnostic_not_adopted"
            and run["script_sha256"] == RUNNER_SHA and run["parent_source_sha256"] == PARENT_SHA
            and run["compile_sha256"] == COMPILE_SHA and run["original_graph_sha256"] == GRAPH_SHA
            and run["capture_sha256"] == CAPTURE_SHA and run["original_checkpoint"]["sha256"] == CHECKPOINT_SHA,
            "Fixed original source/capture/graph lineage changed")
    require(run["normal_benchmark"] is normal and run["profile_separate_fresh_contexts"] is (not normal)
            and run["strict_gate"] == {"rtol": RTOL, "atol": ATOL} and run["candidate_accepted"] is False
            and run["full_model_run"] is False and run["real_time_validated"] is False and run["whole_model_quality_validated"] is False
            and run["requested_sdk_call_count"] == run["successful_SDK_calls"] == len(run["calls"]) == len(run["sdk_calls"]) == 16
            and run["unmeasured_warmup_calls"] == 0 and run["contexts_live_together"] is True
            and run["cleanup_errors"] == [] and run["affinity_restored"] is True and run["environment_restored"] is True
            and run["source_and_inputs_unchanged"] is True
            and run["source_and_input_hashes"] == run["source_and_input_hashes_after"], "Run integrity/scope/gate changed")
    descriptions = {role: check_descriptors(run["context_records"][role], normal) for role in ROLES}
    arrays, calls, last_end = {}, [], 0
    for index, (sid, role, iteration) in enumerate(expected_calls()):
        row, sdk = run["calls"][index], run["sdk_calls"][index]
        sample_index = SAMPLES.index(sid)
        for value in (row, sdk):
            require((value["sdk_call_index"], value["sample_id"], value["role"], value["round_index"])
                    == (index, sid, role, iteration), "Exact AB/BA call ordering changed")
        require(sdk["sample_call_index"] == iteration and sdk["context_call_index"] == sample_index * 4 + iteration
                and sdk["first_in_context"] is (sample_index == 0 and iteration == 0), "Context/sample repeat identity changed")
        audit_interval(sdk)
        require(row["system_before"]["monotonic_ns"] <= sdk["io_begin_monotonic_ns"]
                and sdk["io_end_monotonic_ns"] <= row["system_after"]["monotonic_ns"]
                and sdk["io_begin_monotonic_ns"] > last_end, "Calls are not serial within saved platform intervals")
        last_end = sdk["io_end_monotonic_ns"]
        numeric_same(sdk["io_total_seconds"], row["io_total_seconds"], "ledger/call I/O interval")
        sample = evaluations[sid]
        require(sdk["input_raw_sha256_before"] == sdk["input_raw_sha256_after"] == row["input_raw_sha256"]
                == sample["artifacts"]["input.npy"]["raw_sha256"]
                and row["reference_raw_sha256"] == sample["artifacts"]["output.npy"]["raw_sha256"], "Original input/reference changed")
        value = load_array(directory, row["output"], OUTPUT_SHAPE, pins)
        metric = original_metrics(value, references[sid])
        for key, actual in metric.items():
            numeric_same(actual, row["original_FP32_comparison"][key], key)
        key = (sid, role, iteration)
        arrays[key] = value
        first = arrays[(sid, role, 0)]
        equal = raw_sha(value) == raw_sha(first) and np.array_equal(value, first)
        require(row["byte_equal_to_first_for_sample"] is equal and equal, "Saved repeats differ bitwise")
        calls.append({"sdk_call_index": index, "sample_id": sid, "role": role, "round_index": iteration,
                      "io_total_seconds": row["io_total_seconds"], "original_FP32_comparison": metric})
        if normal:
            require("separate_profile_detail" not in row, "Normal cohort contains profiling collection")
        else:
            detail = row["separate_profile_detail"]
            name = Path(detail["file"])
            require(not name.is_absolute() and name.name == str(name), "Profile detail path escaped")
            path = pin(directory / name, detail["saved_text_sha256"], pins)
            require(path.stat().st_size == detail["bytes"] and detail["raw_sha256"] == detail["saved_text_sha256"], "Saved profile byte count/hash changed")
    summaries = {role: six_call_summary(calls, role) for role in ROLES}
    for role, summary in summaries.items():
        declared = run["summaries"][role]
        require(declared["mean_denominator"] == 6 and declared["subsequent_IO_mean_seconds"] == summary["subsequent_IO_mean_seconds"],
                "Board-ordered six-call mean changed")
        passed = all(x["original_FP32_comparison"]["strict_FP32_gate_passed"] for x in calls if x["role"] == role)
        require(declared["strict_original_FP32_gate_passed"] is passed, "Aggregate original gate changed")
        summary["strict_original_FP32_gate_passed"] = passed
        summary["first_call_IO_seconds_by_input"] = {sid: next(x["io_total_seconds"] for x in calls if x["sample_id"] == sid and x["role"] == role and x["round_index"] == 0) for sid in SAMPLES}
        summary["subsequent_stage_mean_seconds"] = {stage: ordered_sum(x[stage] for x in run["sdk_calls"] if x["role"] == role and x["round_index"] > 0) / 6 for stage in STAGES}
    return run, summaries, arrays, descriptions, calls


def audit(normal_dir, profile_dir, capture_dir, flash_source_dir, runtime_source_dir):
    import numpy as np
    pins = {}
    source_files = ((flash_source_dir / "run-v50-flash-pair-board.py", RUNNER_SHA),
                    (flash_source_dir / "compile-v50-flash-pair-board.py", COMPILER_SHA),
                    (runtime_source_dir / "run-v49-original-pair-precision-board.py", PARENT_SHA),
                    (runtime_source_dir / "v49_quantized_pair_runtime_contracts.py", CONTRACT_SHA))
    for path, digest in source_files:
        pin(path, digest, pins)
    captured = json.loads(pin(capture_dir / "capture.json", CAPTURE_SHA, pins).read_text())
    require(captured["status"] == "completed_prefix_capture" and captured["script_sha256"] == CAPTURE_SCRIPT_SHA
            and captured["contracts_sha256"] == CAPTURE_CONTRACT_SHA and captured["checkpoint"]["sha256"] == CHECKPOINT_SHA
            and captured["frozen_pair"]["onnx_sha256"] == GRAPH_SHA and captured["original_model_only"] is True,
            "Exact original saved Torch capture provenance required")
    evaluations = {s["sample_id"]: s for s in captured["samples"] if s["split"] == "evaluation"}
    require(tuple(evaluations) == SAMPLES, "Exact independent calibration-excluded evaluation set required")
    references = {}
    for sid, sample in evaluations.items():
        load_array(capture_dir, sample["artifacts"]["input.npy"], INPUT_SHAPE, pins)
        references[sid] = load_array(capture_dir, sample["artifacts"]["output.npy"], OUTPUT_SHAPE, pins)
    n, nm, na, nd, nc = audit_cohort(normal_dir, NORMAL_SHA, True, evaluations, references, pins)
    p, pm, pa, pd, pc = audit_cohort(profile_dir, PROFILE_SHA, False, evaluations, references, pins)
    require(nd == pd and n["compiled_artifacts"] == p["compiled_artifacts"], "Normal/profile artifact/descriptor identity differs")
    for run in (n, p):
        for path, digest in source_files:
            matches = [v for k, v in run["source_and_input_hashes"].items() if Path(k).name == path.name]
            require(matches == [digest], "Saved actual runner source binding differs from independently read source")
    equalities = []
    for sid in SAMPLES:
        first = na[(sid, "ordinary", 0)]
        grouped = [array for table in (na, pa) for key, array in table.items() if key[0] == sid]
        require(len(grouped) == 16 and all(raw_sha(x) == raw_sha(first) and np.array_equal(x, first) for x in grouped),
                "Ordinary/Flash, repeated and normal/profile saved outputs are not all bitwise equal for one input")
        equalities.append({"sample_id": sid, "saved_outputs_compared": 16, "all_outputs_bitwise_equal": True,
                           "shared_raw_sha256": raw_sha(first), "original_FP32_comparison": original_metrics(first, references[sid])})
    for path, expected in pins.items():
        require(sha(path) == expected, "Source or saved input/output changed during audit")
    ordinary, flash = (nm[role]["subsequent_IO_mean_seconds"] for role in ROLES)
    return {"schema": "v50_original_pair_flash_saved_numeric_timing_audit", "audit_integrity_passed": True,
            "auditor_sha256": sha(__file__), "numpy": np.__version__, "normal_run_sha256": NORMAL_SHA,
            "profile_run_sha256": PROFILE_SHA, "original_capture_sha256": CAPTURE_SHA,
            "actual_compile_receipt_sha256_bound_via_runs": COMPILE_SHA, "original_graph_sha256": GRAPH_SHA,
            "original_checkpoint_sha256": CHECKPOINT_SHA, "source_files_independently_SHA_checked": 4,
            "saved_output_NPY_files_read_and_checked": 32, "saved_original_input_reference_NPY_files_checked": 4,
            "profile_text_file_SHA_and_bytes_checked": 16, "profile_operator_texts_parsed_by_this_auditor": False,
            "normal_profile_corresponding_bitwise_equal_calls": 16, "ordinary_flash_bitwise_equal_pairs_across_both_cohorts": 16,
            "within_role_repeat_bitwise_equal_comparisons_across_both_cohorts": 24,
            "same_input_outputs_across_both_cohorts": equalities, "normal_IO_summaries": nm, "profile_IO_summaries": pm,
            "normal_flash_minus_ordinary_mean_seconds": flash - ordinary,
            "normal_flash_mean_reduction_fraction": (ordinary - flash) / ordinary,
            "fixed_component_gate": {"rtol": RTOL, "atol": ATOL, "reference": "same-input original saved Torch FP32 output"},
            "normal_and_profile_exact_call_order": [{"sdk_call_index": i, "sample_id": s, "role": who, "round_index": r} for i, (s, who, r) in enumerate(expected_calls())],
            "logical_descriptors": nd, "all_saved_read_files_unchanged": True,
            "declared_affinity_environment_source_hashes_and_cleanup_checks_passed": True,
            "candidate_accepted": False, "full_model_ATE_RPE_or_fresh_pose_rate_or_P95_validated": False,
            "actual_model_ORT_SDK_compiler_calibration_or_board_execution_by_auditor": False,
            "no_raw_arrays_weights_or_SDK_binaries_published": True,
            "timing_scope": "SDK input set through synchronous run, outputs get, owned copy and release; excludes initialization, hashes, error metrics, snapshots, file saves and profile-table query",
            "limits": ["Two calibration-excluded evaluation inputs were reused for development; no new blind trajectory test.",
                       "All outputs still fail the unchanged original FP32 component gate; exact equality between variants is not reference equivalence.",
                       "Six normal subsequent calls per role only; small repeated-input diagnostic, not a statistically established speed change or full SLAM latency.",
                       "Profile calls use separately initialized contexts and are not pooled into normal performance means.",
                       "Operator/Flash execution claims require the separate operator audit; this saved-array audit proves no internal cause.",
                       "Original complete frame0/global0 pair only; no 20 Hz fresh-pose, P95 100 ms, ATE/RPE or full-model claim."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("normal-dir", "profile-dir", "capture-dir", "flash-source-dir", "runtime-source-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    dirs = [getattr(args, name) for name in ("normal_dir", "profile_dir", "capture_dir", "flash_source_dir", "runtime_source_dir")]
    require(not args.output.exists() and not any(args.output.resolve().is_relative_to(d.resolve()) for d in dirs),
            "New public-safe audit receipt outside all evidence/source directories required")
    result = audit(*(d.resolve() for d in dirs))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"audit_integrity_passed": True, "saved_output_arrays_checked": 32,
                      "all_same_input_outputs_bitwise_equal": True, "candidate_accepted": False,
                      "receipt_sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
