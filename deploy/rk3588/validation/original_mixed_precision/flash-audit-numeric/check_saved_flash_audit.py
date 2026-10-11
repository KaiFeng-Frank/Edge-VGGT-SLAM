#!/usr/bin/env python3
"""Generated checks for saved-data gates and timing denominators; no inference."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("_flash_saved_checks", HERE / "audit_saved_flash_pair.py")
    a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a)
    a.require(not args.output.exists(), "New check receipt required")
    checks = []
    def record(name):
        checks.append({"check": name, "passed": True})
    def rejects(fn):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError("Generated invalid case must reject")

    schedule = a.expected_calls()
    assert len(schedule) == 16
    assert schedule[:4] == [("evaluation_00", "ordinary", 0), ("evaluation_00", "flash", 0),
                            ("evaluation_00", "flash", 1), ("evaluation_00", "ordinary", 1)]
    assert schedule[8:12] == [("evaluation_01", "flash", 0), ("evaluation_01", "ordinary", 0),
                             ("evaluation_01", "ordinary", 1), ("evaluation_01", "flash", 1)]
    record("two_input_AB_BA_schedule_reverses_second_input_and_each_round")
    calls = [{"sdk_call_index": i, "sample_id": s, "role": role, "round_index": r,
              "io_total_seconds": 1000.0 if r == 0 else float((i + 1) ** 2)}
             for i, (s, role, r) in enumerate(schedule)]
    ordinary, flash = (a.six_call_summary(calls, role) for role in a.ROLES)
    assert ordinary["included_sdk_call_indices"] == [3, 4, 7, 10, 13, 14]
    assert flash["included_sdk_call_indices"] == [2, 5, 6, 11, 12, 15]
    assert ordinary["mean_denominator"] == flash["mean_denominator"] == 6
    assert ordinary["subsequent_IO_mean_seconds"] == 647 / 6
    assert flash["subsequent_IO_mean_seconds"] == 663 / 6
    record("six_call_means_exclude_both_per_input_first_calls_with_explicit_indices")
    rejects(lambda: a.six_call_summary([x for x in calls if x["sdk_call_index"] != 3], "ordinary"))
    rejects(lambda: a.six_call_summary(calls + [calls[3]], "ordinary"))
    record("missing_and_duplicate_repeat_cannot_keep_a_six_call_denominator")
    changed = copy.deepcopy(calls)
    changed[3]["io_total_seconds"] = float("nan")
    rejects(lambda: a.six_call_summary(changed, "ordinary"))
    record("nonfinite_IO_time_rejected")
    assert a.ordered_sum([1e16, 1, -1e16]) == 0.0
    record("ordered_addition_matches_board_float_order_without_host_sum_assumption")

    sdk = {"status": "completed", "synchronous": True, "input_api_type": "FP32", "input_api_type_enum": 0,
           "input_pass_through": 0, "input_format": 3, "input_shape": a.INPUT_SHAPE, "output_shape": a.OUTPUT_SHAPE,
           "input_bytes": a.FP32_BYTES, "expected_FP32_output_bytes": a.FP32_BYTES, "output_bytes": a.FP32_BYTES,
           "want_float": 1, "is_prealloc": 0, "manual_quantization_or_dequantization": False,
           "output_buffer_released": True, "output_owned_copy_before_release": True, "npu_run_api_called": True,
           "runtime_frame_id": 0, "output_frame_id": 0, "SDK_output_pointer": 1000, "owned_output_pointer": 2000,
           "inputs_set_code": 0, "run_code": 0, "outputs_get_code": 0, "outputs_release_code": 0,
           "io_begin_monotonic_ns": 100, "io_end_monotonic_ns": 121, "io_total_seconds": 21 / 1e9}
    sdk.update({key: (i + 1) / 1e9 for i, key in enumerate(a.STAGES)})
    a.audit_interval(sdk)
    record("integer_nanosecond_stage_sum_includes_input_copy_and_release")
    changed = dict(sdk, io_total_seconds=sdk["run_api_seconds"])
    rejects(lambda: a.audit_interval(changed))
    changed = dict(sdk, outputs_release_seconds=5 / 1e9)
    rejects(lambda: a.audit_interval(changed))
    record("run_only_time_or_missing_release_time_cannot_match_complete_IO_interval")
    rejects(lambda: a.audit_interval(dict(sdk, output_buffer_released=False)))
    rejects(lambda: a.audit_interval(dict(sdk, owned_output_pointer=sdk["SDK_output_pointer"])))
    record("unreleased_or_aliasing_output_storage_rejected")

    reference = np.array([0.031175846233963966], dtype=np.float32)
    candidate = np.array([0.03118896484375], dtype=np.float16).astype(np.float32)
    metric = a.original_metrics(candidate, reference, [1])
    error = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    assert np.all(error <= a.ATOL + a.RTOL * np.abs(candidate.astype(np.float64)))
    assert metric["outside_tolerance_elements"] == 1 and metric["strict_FP32_gate_passed"] is False
    record("strict_gate_uses_original_reference_magnitude_even_if_candidate_magnitude_would_pass")
    zero = np.array([0, 0], dtype=np.float32)
    actual = np.array([0, 0.0001], dtype=np.float32)
    metric = a.original_metrics(actual, zero, [2])
    assert metric["relative_l2"] is None and metric["cosine_similarity"] is None
    assert metric["outside_tolerance_elements"] == 1 and metric["strict_FP32_gate_passed"] is False
    rejects(lambda: a.original_metrics(np.array([np.inf], dtype=np.float32), reference, [1]))
    record("zero_reference_handles_undefined_norm_and_nonfinite_output_rejects")
    positive, negative = np.array([0.0], dtype=np.float32), np.array([-0.0], dtype=np.float32)
    assert np.array_equal(positive, negative) and a.raw_sha(positive) != a.raw_sha(negative)
    record("bitwise_check_distinguishes_signed_zero_when_array_equal_does_not")
    assert not any(name in sys.modules for name in ("torch", "onnxruntime", "rknn.api", "rknn_runtime"))
    record("no_model_ORT_or_SDK_modules_imported")
    result = {"schema": "v50_saved_flash_audit_generated_checks", "passed": True, "check_count": len(checks),
              "checks": checks, "auditor_source_sha256": a.sha(HERE / "audit_saved_flash_pair.py"),
              "check_source_sha256": a.sha(__file__), "numpy": np.__version__,
              "real_saved_arrays_or_actual_model_ORT_SDK_calibration_or_board_used": False}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": True, "checks": len(checks), "receipt_sha256": a.sha(args.output)}))


if __name__ == "__main__":
    main()
