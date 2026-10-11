#!/usr/bin/env python3
"""Independent saved-output/SDK-receipt arithmetic; never calls a runtime API."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CAPTURE_SOURCE = HERE.parent / "capture"
CAPTURE_CONTRACTS_SHA = "6ce1f87f1683e8082a4d567718acb853101e5ee6bbaeef1ced02993a8e3133c8"
CAPTURE_AUDIT_SHA = "7d0debcb818ca2e01cb8a3e9f4122262f82ecf44969e85175f246666c617d97d"
ROLES = ("nonquant_fp16", "ptq_w8a8")
SCHEDULE = (ROLES, ROLES[::-1], ROLES, ROLES[::-1])
INPUT_SHAPE, OUTPUT_SHAPE = [2, 437, 1024], [1, 874, 1024]
ELEMENTS, FP32_BYTES = 894976, 3579904
RTOL, ATOL = 1e-4, 1e-5


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pin(path, expected, pins):
    path = Path(path).resolve(strict=True)
    require(sha(path) == expected, f"Runtime saved-data SHA mismatch: {path}")
    pins[str(path)] = expected
    return path


def source(path, expected, pins):
    path = pin(path, expected, pins)
    spec = importlib.util.spec_from_file_location("_v49_readonly_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def numeric_same(actual, declared, name):
    if actual is None:
        require(declared is None, f"Metric {name} must be null")
    elif type(actual) in (bool, int):
        require(type(declared) is type(actual) and declared == actual, f"Metric {name} mismatch")
    elif isinstance(actual, str):
        require(declared == actual, f"Metric {name} mismatch")
    else:
        require(type(declared) in (int, float) and math.isfinite(declared)
                and math.isclose(actual, declared, rel_tol=2e-11, abs_tol=1e-12), f"Metric {name} mismatch")


def original_metrics(actual, reference):
    import numpy as np
    require(actual.dtype == reference.dtype == np.float32 and list(actual.shape) == list(reference.shape) == OUTPUT_SHAPE
            and np.isfinite(actual).all() and np.isfinite(reference).all(), "Finite exact original FP32 shapes required")
    a, r = actual.astype(np.float64), reference.astype(np.float64)
    error = a - r
    absolute = np.abs(error)
    tolerance = ATOL + RTOL * np.abs(r)
    reference_norm, actual_norm = float(np.linalg.norm(r)), float(np.linalg.norm(a))
    return {"finite": True, "array_equal": bool(np.array_equal(actual, reference)),
            "max_abs_error": float(absolute.max()), "rmse": float(np.sqrt(np.mean(error * error))),
            "relative_l2": float(np.linalg.norm(error) / reference_norm) if reference_norm else None,
            "cosine_similarity": float(np.sum(a * r) / (reference_norm * actual_norm)) if reference_norm and actual_norm else None,
            "rtol": RTOL, "atol": ATOL, "strict_FP32_gate_passed": bool(np.all(absolute <= tolerance)),
            "outside_tolerance_elements": int(np.count_nonzero(absolute > tolerance)),
            "max_tolerance_ratio": float(np.max(absolute / tolerance)),
            "output_raw_sha256": hashlib.sha256(memoryview(actual).cast("B")).hexdigest()}


def descriptor(record, direction):
    """Independent enum/shape arithmetic; no native-stride access or conversions."""
    shape = INPUT_SHAPE if direction == "input" else OUTPUT_SHAPE
    require(direction in ("input", "output"), "Unsupported direction")
    require(record["name"] == direction and record["index"] == 0 and record["shape"] == shape
            and record["n_elems"] == ELEMENTS and record["format"] == 3 and record["pass_through"] == 0,
            "Actual SDK logical descriptor changed")
    for key in ("index", "n_elems", "size", "format", "type", "quantization_type", "zero_point", "pass_through"):
        require(type(record[key]) is int, "Actual descriptor integer field required")
    dtype, qnt, zp, scale = (record[key] for key in ("type", "quantization_type", "zero_point", "scale"))
    require(type(scale) in (int, float) and math.isfinite(scale), "Finite actual descriptor scale required")
    if dtype in (0, 1):
        require(qnt == 0 or (dtype == 1 and qnt == 2 and zp == 0 and scale == 1),
                "Floating descriptor requires NONE or exact FP16 identity affine")
        item_bytes, label = (4, "FP32") if dtype == 0 else (2, "FP16")
    elif dtype in (2, 3):
        low, high = (-128, 127) if dtype == 2 else (0, 255)
        require(qnt == 2 and scale > 0 and low <= zp <= high, "Integer descriptor requires valid positive affine scale/zero point")
        item_bytes, label = 1, "INT8" if dtype == 2 else "UINT8"
    else:
        raise ValueError("Unsupported actual logical SDK dtype")
    require(record["size"] == ELEMENTS * item_bytes, "Actual SDK size/type/element arithmetic mismatch")
    return {"direction": direction, "logical_shape": shape, "logical_dtype": label,
            "logical_bytes": ELEMENTS * item_bytes, "quantization_type": qnt, "zero_point": zp, "scale": scale,
            "API_FP32_bytes": FP32_BYTES, "SDK_boundary_conversion_included_in_quality_error": True,
            "native_stride_memory_used_by_audit": False,
            "all_internal_ops_INT8_or_NPU_proven": False}


def audit_sdk_interval(row):
    require(row["status"] == "completed" and row["synchronous"] is True
            and row["input_api_type"] == "FP32" and row["input_api_type_enum"] == 0
            and row["input_pass_through"] == 0 and row["input_format"] == 3
            and row["input_shape"] == INPUT_SHAPE and row["input_bytes"] == FP32_BYTES
            and row["output_shape"] == OUTPUT_SHAPE and row["expected_FP32_output_bytes"] == FP32_BYTES
            and row["want_float"] == 1 and row["is_prealloc"] == 0
            and row["manual_quantization_or_dequantization"] is False
            and row["output_bytes"] == FP32_BYTES and row["output_owned_copy_before_release"] is True
            and row["output_buffer_released"] is True and row["npu_run_api_called"] is True,
            "Actual SDK submission/get/copy/release contract failed")
    require(row["runtime_frame_id"] == row["output_frame_id"] and row["SDK_output_pointer"] > 0
            and row["owned_output_pointer"] > 0 and row["owned_output_pointer"] != row["SDK_output_pointer"],
            "Synchronous SDK identity or output ownership failed")
    require(all(type(row[key]) is int and row[key] == 0 for key in
                ("inputs_set_code", "run_code", "outputs_get_code", "outputs_release_code")), "Actual SDK API success codes required")
    begin, end = row["io_begin_monotonic_ns"], row["io_end_monotonic_ns"]
    require(type(begin) is type(end) is int and end > begin, "Actual SDK monotonic interval required")
    stages = {key: row[key] for key in ("input_set_seconds", "run_api_seconds", "outputs_get_seconds",
              "output_validation_and_copy_seconds", "pre_release_bookkeeping_seconds", "outputs_release_seconds")}
    require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in stages.values()), "SDK stages must be finite and nonnegative")
    numeric_same((end - begin) / 1e9, row["io_total_seconds"], "SDK monotonic interval")
    numeric_same(sum(stages.values()), row["io_total_seconds"], "SDK stage sum")
    return stages


def audit(runtime_dir, run_sha, runtime_source_dir, capture_dir, capture_sha):
    import numpy as np
    runtime_dir, runtime_source_dir, capture_dir = map(lambda p: Path(p).resolve(),
                                                       (runtime_dir, runtime_source_dir, capture_dir))
    pins = {}
    helper = source(CAPTURE_SOURCE / "v49_capture_contracts.py", CAPTURE_CONTRACTS_SHA, pins)
    saved = source(CAPTURE_SOURCE / "audit_saved_capture.py", CAPTURE_AUDIT_SHA, pins)
    capture, samples, bound = helper.load_capture(capture_dir, capture_sha)
    pins.update(bound)
    path = pin(runtime_dir / "run.json", run_sha, pins)
    report = json.loads(path.read_text())
    require(report["schema"] == "v49_original_VGGT_pair_precision_SDK_AB" and report["status"].startswith("completed_"),
            "Completed original pair diagnostic required")
    pin(runtime_source_dir / "run-v49-original-pair-precision-board.py", report["script_sha256"], pins)
    pin(runtime_source_dir / "v49_quantized_pair_runtime_contracts.py", report["contracts_sha256"], pins)
    require(report["capture_manifest_sha256"] == capture_sha and report["capture_contracts_sha256"] == CAPTURE_CONTRACTS_SHA
            and report["capture_script_sha256"] == capture["script_sha256"]
            and report["original_parameter_tensors"] == 36 and report["original_checkpoint"]["sha256"] == helper.CHECKPOINT_SHA,
            "Runtime no longer binds the exact original capture/model")
    require(report["inference_on_board_only"] is True and report["candidate_accepted"] is False
            and report["full_model_run"] is False and report["real_time_validated"] is False
            and report["whole_model_quality_validated"] is False and report["rtol"] == RTOL and report["atol"] == ATOL,
            "Runtime scope or fixed original FP32 gate changed")
    require(report["contexts_live_together"] is True and report["unmeasured_warmup_calls"] == 0
            and report["successful_synchronous_call_count"] == report["npu_run_api_call_count"]
            == report["completed_comparison_call_count"] == report["requested_sdk_call_count"] == 16
            and report["affinity_restored"] is True and report["environment_restored"] is True
            and report["source_and_inputs_unchanged"] is True and report["cleanup_errors"] == [],
            "Runtime counted calls or final integrity/cleanup failed")
    require(report["source_and_input_hashes"] == report["source_and_input_hashes_after"], "Runtime source hashes changed")
    evaluations = [s for s in samples if s["split"] == "evaluation"]
    calibrations = [s for s in samples if s["split"] == "calibration"]
    require(report["evaluation_sample_ids"] == [s["sample_id"] for s in evaluations]
            and report["calibration_sample_ids"] == [s["sample_id"] for s in calibrations], "Runtime evaluation/calibration identity mismatch")
    descriptors = {}
    for role in ROLES:
        context = report["context_records"][role]
        require(context["role"] == role and context["context_destroyed"] is True and context["destroy_code"] == 0
                and context["tensor_attributes_validated"] is True and context["core_mask"] == 7
                and context["sdk_version"].startswith("2.3.2") and context["driver_version"] == "0.7.2",
                "Actual runtime descriptor/context lifecycle required")
        descriptors[role] = {d: descriptor(context[d + "_attr"], d) for d in ("input", "output")}
        require(context["init_flags"] == (0 if report["normal_benchmark"] else 8), "Profile/normal context mode mismatch")
        require(context["all_internal_operators_INT8_verified"] is False and context["all_internal_operators_on_NPU_verified"] is False,
                "Unproved internal operator placement claim")
    expected_schedule = []
    for index, sample in enumerate(evaluations):
        for round_index, order in enumerate(SCHEDULE if index == 0 else tuple(v[::-1] for v in SCHEDULE)):
            expected_schedule.append({"sample_id": sample["sample_id"], "round_index": round_index, "order": list(order)})
    require(report["schedule"] == expected_schedule and len(report["calls"]) == len(report["sdk_calls"]) == 16,
            "Actual AB/BA schedule changed")
    expected_calls = [(item["sample_id"], item["round_index"], role)
                      for item in expected_schedule for role in item["order"]]
    references, inputs, first_output_raw, output_arrays = {}, {}, {}, {}
    for sample in evaluations:
        sid = sample["sample_id"]
        inputs[sid] = saved.read_artifact(capture_dir, sample["artifacts"]["input.npy"], INPUT_SHAPE, "float32", pins)
        references[sid] = saved.read_artifact(capture_dir, sample["artifacts"]["output.npy"], OUTPUT_SHAPE, "float32", pins)
        input_record = report["input_records"][sid]
        require(input_record["input_raw_sha256_before"] == input_record["input_raw_sha256_after"]
                == sample["artifacts"]["input.npy"]["raw_sha256"] and input_record["input_unchanged"] is True
                and input_record["original_reference_raw_sha256"] == sample["artifacts"]["output.npy"]["raw_sha256"],
                "Actual SDK input/reference lineage changed")
    calls = []
    for index, (sid, round_index, role) in enumerate(expected_calls):
        row, sdk = report["calls"][index], report["sdk_calls"][index]
        context_index = (0 if sid == evaluations[0]["sample_id"] else 4) + round_index
        for item in (row, sdk):
            require(item["sdk_call_index"] == index and item["sample_id"] == sid and item["role"] == role
                    and item["round_index"] == item["sample_call_index"] == round_index
                    and item["context_call_index"] == context_index and item["first_in_context"] is (context_index == 0),
                    "SDK/call identity schedule mismatch")
        require(row["first_for_sample"] is (round_index == 0), "First sample denominator changed")
        stages = audit_sdk_interval(sdk)
        numeric_same(sdk["io_total_seconds"], row["io_total_seconds"], "Call/SDK interval")
        require(sdk["input_raw_sha256_before"] == sdk["input_raw_sha256_after"] == saved.raw_sha(inputs[sid]),
                "SDK original input mutation")
        output = saved.read_artifact(runtime_dir, row["output"], OUTPUT_SHAPE, "float32", pins)
        metrics = original_metrics(output, references[sid])
        require(set(metrics) == set(row["original_FP32_comparison"]), "Original numerical report metric names changed")
        for name, value in metrics.items():
            numeric_same(value, row["original_FP32_comparison"][name], name)
        first_output_raw.setdefault((sid, role), metrics["output_raw_sha256"])
        require(row["byte_equal_to_first_for_sample"] is (metrics["output_raw_sha256"] == first_output_raw[(sid, role)]),
                "Saved repeat byte-equality receipt changed")
        output_arrays[(sid, round_index, role)] = output
        calls.append({"sdk_call_index": index, "sample_id": sid, "round_index": round_index, "role": role,
                      "io_total_seconds": row["io_total_seconds"], "sdk_stages_seconds": stages, "original_FP32_comparison": metrics})
    summaries = {}
    for role in ROLES:
        rows = [r for r in calls if r["role"] == role and r["round_index"] > 0]
        require(len(rows) == 6, "Six actual warm calls per implementation required")
        mean = sum(row["io_total_seconds"] for row in rows) / 6
        declared = report["summaries"]["implementations"][role]
        require(declared["mean_denominator"] == 6 and declared["included_sdk_call_indices"] == [r["sdk_call_index"] for r in rows],
                "Warm denominator or included calls changed")
        numeric_same(mean, declared["two_input_subsequent_io_mean_seconds"], "Warm mean")
        passes = all(row["original_FP32_comparison"]["strict_FP32_gate_passed"] for row in calls if row["role"] == role)
        require(declared["strict_original_FP32_gate_passed"] is passes, "Aggregate original FP32 gate differs")
        summaries[role] = {"subsequent_measured_call_denominator": 6, "included_sdk_call_indices": [r["sdk_call_index"] for r in rows],
                           "mean_io_seconds": mean, "all_saved_outputs_pass_fixed_original_FP32_gate": passes,
                           "per_input_first_metrics": {sample["sample_id"]: next(r["original_FP32_comparison"] for r in calls
                                if r["sample_id"] == sample["sample_id"] and r["role"] == role and r["round_index"] == 0)
                                for sample in evaluations}}
    require(len(report["paired_implementation_differences"]) == 8, "Eight paired implementation diagnostics required")
    for original, declared in zip(expected_schedule, report["paired_implementation_differences"]):
        sid, round_index = original["sample_id"], original["round_index"]
        require(all(declared[key] == value for key, value in original.items()), "Paired diagnostic schedule mismatch")
        a, b = (output_arrays[(sid, round_index, role)] for role in ROLES[::-1])
        difference = a.astype(np.float64) - b.astype(np.float64)
        actual = {"reference_for_original_FP32_gate": False, "array_equal": bool(np.array_equal(a, b)),
                  "max_abs_difference": float(np.max(np.abs(difference))), "difference_l2": float(np.linalg.norm(difference)),
                  "rmse_difference": float(np.sqrt(np.mean(difference * difference)))}
        for key, value in actual.items():
            numeric_same(value, declared["PTQ_minus_nonquant_implementation_diagnostic"][key], key)
    for path, expected in pins.items():
        require(sha(path) == expected, f"Saved output/source changed during audit: {path}")
    baseline, ptq = (summaries[role]["mean_io_seconds"] for role in ROLES)
    difference = {"nonquant_minus_PTQ_seconds": baseline - ptq,
                  "nonquant_minus_PTQ_percent_of_nonquant": 100 * (baseline - ptq) / baseline if baseline else None}
    if report["normal_benchmark"]:
        for key, value in difference.items():
            numeric_same(value, report["summaries"]["PTQ_IO_difference"][key], key)
    return {"schema": "v49_read_only_original_pair_runtime_saved_data_audit", "passed": True,
            "audit_script_sha256": sha(__file__), "run_receipt_sha256": run_sha, "capture_manifest_sha256": capture_sha,
            "normal_benchmark": report["normal_benchmark"], "profiled_IO_is_normal_speed_comparison": False,
            "SDK_outputs_physically_read": 16, "heldout_inputs_and_original_references_physically_read": 4,
            "descriptors": descriptors, "fixed_original_FP32_rtol": RTOL, "fixed_original_FP32_atol": ATOL,
            "summaries": summaries, "PTQ_IO_difference_diagnostic": difference,
            "sdk_interval_sum_and_monotonic_ns_agree": True, "calls": calls,
            "all_read_file_bytes_unchanged": True, "read_file_SHA_pins": pins,
            "actual_model_ORT_SDK_calibration_compilation_or_image_decode_by_audit": False,
            "full_model_or_SLAM_quality_or_realtime_proven_by_audit": False,
            "candidate_accepted_by_audit": False,
            "limits": ["Same two heldout inputs repeated four times each implementation; no fresh pose-rate evidence.",
                       "Six warm IO calls per implementation, excluding each input's first call; no robust latency percentile claim.",
                       "Original frame0/global0 component only; all remaining original computation outside this experiment.",
                       "Logical integer descriptors and w8a8 build do not prove all internal operators INT8 or NPU.",
                       "API FP32 submission/want_float retrieval includes SDK conversion error; no manual quantization is performed."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--run-sha256", required=True)
    parser.add_argument("--runtime-source-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--capture-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.resolve().is_relative_to(args.runtime_dir.resolve())
            and not args.output.resolve().is_relative_to(args.capture_dir.resolve()), "New audit receipt outside saved data required")
    report = audit(args.runtime_dir, args.run_sha256, args.runtime_source_dir, args.capture_dir, args.capture_sha256)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": True, "runtime_arrays": 16, "receipt_sha256": sha(args.output),
                      "normal_benchmark": report["normal_benchmark"], "summaries": report["summaries"]}))


if __name__ == "__main__":
    main()
