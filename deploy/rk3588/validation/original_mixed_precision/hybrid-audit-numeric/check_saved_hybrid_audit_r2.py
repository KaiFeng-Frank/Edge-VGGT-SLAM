#!/usr/bin/env python3
"""Generated hybrid R2-role, mean and output-grid checks; no real inference."""
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
    parser.add_argument("--flash-auditor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("_hybrid_saved_generated", HERE / "audit_saved_hybrid_pair_r2.py")
    a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a)
    h = a.load_pure(args.flash_auditor, a.HELPER_SHA, "_pinned_flash_helpers_for_generated")
    a.require(not args.output.exists(), "New generated receipt required")
    checks = []
    def record(name):
        checks.append({"check": name, "passed": True})
    def rejects(fn):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError("Generated invalid case must reject")

    assert str(a.source_basename("run-v50-hybrid-pair-r2-board.py")) == "run-v50-hybrid-pair-r2-board.py"
    for value in ("/tmp/x.py", "../x.py", "x/y.py", "", ".", "..", "two names.py", 2):
        rejects(lambda value=value: a.source_basename(value))
    record("R2_source_basename_binding_rejects_path_escape_and_whitespace")

    schedule = a.expected_calls()
    assert len(schedule) == 16 and schedule[0] == ("evaluation_00", "ordinary", 0)
    assert schedule[2] == ("evaluation_00", "hybrid", 1)
    assert schedule[8] == ("evaluation_01", "hybrid", 0)
    assert schedule[10] == ("evaluation_01", "ordinary", 1)
    record("ordinary_hybrid_schedule_reverses_each_round_and_second_input")
    calls = [{"sdk_call_index": i, "sample_id": sid, "role": role, "round_index": r,
              "io_total_seconds": 1000.0 if r == 0 else float((i + 1) ** 2)}
             for i, (sid, role, r) in enumerate(schedule)]
    ordinary, hybrid = (a.six_summary(calls, role, h) for role in a.ROLES)
    assert ordinary["mean_denominator"] == hybrid["mean_denominator"] == 6
    assert ordinary["included_sdk_call_indices"] == [3, 4, 7, 10, 13, 14]
    assert hybrid["included_sdk_call_indices"] == [2, 5, 6, 11, 12, 15]
    assert ordinary["subsequent_IO_mean_seconds"] == 647 / 6
    assert hybrid["subsequent_IO_mean_seconds"] == 663 / 6
    record("six_call_means_use_three_repeats_per_input_and_exclude_each_first")
    rejects(lambda: a.six_summary(calls + [calls[2]], "hybrid", h))
    rejects(lambda: a.six_summary([x for x in calls if x["sdk_call_index"] != 2], "hybrid", h))
    record("duplicate_and_missing_hybrid_call_cannot_keep_mean_denominator")

    normal = {(sid, role, i): np.array([1, 2] if role == "ordinary" else [10, 20], dtype=np.float32)
              for sid in a.SAMPLES for role in a.ROLES for i in range(4)}
    profile = {key: value.copy() for key, value in normal.items()}
    equality = a.validate_role_arrays(normal, profile, h)
    assert len(equality) == 4 and all(x["outputs_compared"] == 8 for x in equality)
    assert not np.array_equal(normal[("evaluation_00", "ordinary", 0)], normal[("evaluation_00", "hybrid", 0)])
    record("different_ordinary_and_hybrid_outputs_are_allowed_if_each_role_is_repeatable")
    changed = {key: value.copy() for key, value in profile.items()}
    changed[("evaluation_00", "hybrid", 3)][0] += 1
    rejects(lambda: a.validate_role_arrays(normal, changed, h))
    record("one_different_profile_hybrid_repeat_rejects_same_role_equality")

    base = {"shape": h.OUTPUT_SHAPE, "name": "output", "index": 0, "n_elems": h.ELEMENTS,
            "format": 3, "pass_through": 0, "type": 1, "quantization_type": 2,
            "zero_point": 0, "scale": 1.0, "size": h.ELEMENTS * 2}
    fp16 = a.descriptor(base, "output", h)
    assert fp16["floating"] is True and fp16["logical_dtype"] == "FP16"
    integer = dict(base, type=2, size=h.ELEMENTS, zero_point=5, scale=float(np.float32(0.03759)))
    int8 = a.descriptor(integer, "output", h)
    assert int8["floating"] is False and int8["logical_dtype"] == "INT8"
    unsigned = a.descriptor(dict(integer, type=3, zero_point=128), "output", h)
    assert unsigned["logical_dtype"] == "UINT8"
    record("actual_FP16_identity_INT8_and_UINT8_boundaries_are_distinguished")
    rejects(lambda: a.descriptor(dict(integer, scale=0), "output", h))
    rejects(lambda: a.descriptor(dict(integer, size=h.ELEMENTS * 2), "output", h))
    rejects(lambda: a.descriptor(dict(base, zero_point=1), "output", h))
    record("invalid_affine_scale_byte_width_and_nonidentity_floating_metadata_reject")
    membership = a.output_membership(np.array([1, 1.0004], dtype=np.float32), fp16)
    assert membership["elements"] == 2 and membership["exact_value_members"] == 1
    record("FP16_membership_detects_finite_FP32_value_between_half_grid_points")
    scale = np.float32(integer["scale"])
    centered = np.arange(-128, 128, dtype=np.float64) - integer["zero_point"]
    levels = (centered * float(scale)).astype(np.float32)
    membership = a.output_membership(levels, int8)["hypotheses"]
    assert membership["FP32_scale_FP32_product"]["exact_value_members"] == 256
    assert membership["FP32_scale_FP32_product"]["max_distance_to_grid"] == 0
    rounded = (centered * float(np.float16(scale))).astype(np.float16).astype(np.float32)
    membership = a.output_membership(rounded, int8)["hypotheses"]
    assert membership["FP16_rounded_scale_FP16_rounded_product"]["exact_value_members"] == 256
    record("INT8_output_membership_allows_FP32_product_and_FP16_rounded_scale_product_hypotheses")
    offgrid = np.array([float(scale) * 0.3], dtype=np.float32)
    report = a.output_membership(offgrid, int8)["hypotheses"]["FP32_scale_FP32_product"]
    brute = float(np.min(np.abs(offgrid.astype(np.float64)[:, None] - levels.astype(np.float64)[None])))
    assert report["exact_value_members"] == 0 and report["max_distance_to_grid"] == brute
    record("offgrid_distance_matches_brute_search_over_all_256_affine_levels")
    assert not any(name in sys.modules for name in ("torch", "onnxruntime", "rknn.api", "rknn_runtime"))
    record("generated_checks_import_no_model_ORT_or_SDK_modules")
    result = {"schema": "v50_hybrid_saved_audit_generated_checks", "passed": True, "checks": checks,
              "check_count": len(checks), "auditor_source_sha256": a.sha(HERE / "audit_saved_hybrid_pair_r2.py"),
              "check_source_sha256": a.sha(__file__), "pure_saved_helper_sha256": a.HELPER_SHA,
              "real_saved_arrays_or_actual_model_ORT_SDK_calibration_or_board_used": False}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": True, "checks": len(checks), "receipt_sha256": a.sha(args.output)}))


if __name__ == "__main__":
    main()
