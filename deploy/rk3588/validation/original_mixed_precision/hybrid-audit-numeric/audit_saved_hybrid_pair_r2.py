#!/usr/bin/env python3
"""Saved original-pair hybrid R2 numerical/timing audit; no model or SDK."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HELPER_SHA = "d2ccb21f2a77a4e4671d410cffd754052618dae65fb790e32e50f64d9ef642af"
STEP1_SHA = "7c2a101be488dded5086c6b418480f459bc14a62f9269991c1d88a6892ec8133"
R1_CONTRACTS_SHA = "d92103b618fd186605269d151f4cd480429b26570aa5b466c5f0f90ca01075df"
ROLES, SAMPLES = ("ordinary", "hybrid"), ("evaluation_00", "evaluation_01")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_pure(path, expected, name):
    require(sha(path) == expected, "Frozen pure saved-data source SHA changed")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_basename(name):
    require(type(name) is str and name and not any(x.isspace() for x in name), "One source basename required")
    path = Path(name)
    require(path.name == name and not path.is_absolute() and name not in (".", ".."), "One source basename required")
    return path


def expected_calls():
    return [(sid, role, iteration) for index, sid in enumerate(SAMPLES) for iteration in range(4)
            for role in (ROLES if (index + iteration) % 2 == 0 else ROLES[::-1])]


def six_summary(calls, role, helper):
    require(role in ROLES, "Unknown role")
    expected = [(i, s, r) for i, (s, who, r) in enumerate(expected_calls()) if who == role and r > 0]
    rows = [x for x in calls if x["role"] == role and x["round_index"] > 0]
    require([(x["sdk_call_index"], x["sample_id"], x["round_index"]) for x in rows] == expected,
            "Exactly six ordered subsequent calls, three per input, are required")
    values = [x["io_total_seconds"] for x in rows]
    require(all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in values), "Finite positive IO required")
    return {"mean_denominator": 6, "included_sdk_call_indices": [i for i, _, _ in expected],
            "subsequent_IO_mean_seconds": helper.ordered_sum(values) / 6,
            "subsequent_IO_min_seconds": min(values), "subsequent_IO_max_seconds": max(values)}


def descriptor(attr, direction, helper):
    shape = helper.INPUT_SHAPE if direction == "input" else helper.OUTPUT_SHAPE
    require(direction in ("input", "output") and attr["shape"] == shape and attr["name"] == direction
            and attr["index"] == 0 and attr["n_elems"] == helper.ELEMENTS and attr["format"] == 3
            and attr["pass_through"] == 0, "Exact logical shape/index/format required")
    dtype, qnt, zp, scale = (attr[x] for x in ("type", "quantization_type", "zero_point", "scale"))
    require(type(dtype) is type(qnt) is type(zp) is int and type(scale) in (int, float)
            and math.isfinite(scale), "Typed finite logical descriptor required")
    if dtype in (0, 1):
        require(qnt == 0 or (dtype == 1 and qnt == 2 and zp == 0 and scale == 1), "Supported floating quantization metadata required")
        width, label = (4, "FP32") if dtype == 0 else (2, "FP16")
    elif dtype in (2, 3):
        low, high = (-128, 127) if dtype == 2 else (0, 255)
        require(qnt == 2 and scale > 0 and low <= zp <= high, "Positive affine integer descriptor required")
        width, label = 1, "INT8" if dtype == 2 else "UINT8"
    else:
        raise ValueError("Unsupported SDK logical dtype")
    require(attr["size"] == width * helper.ELEMENTS, "Descriptor dtype/size arithmetic mismatch")
    return {"logical_dtype": label, "dtype_enum": dtype, "shape": shape, "logical_bytes": attr["size"],
            "quantization_type": qnt, "zero_point": zp, "scale": scale,
            "API_FP32_bytes": helper.FP32_BYTES, "floating": dtype in (0, 1),
            "internal_arithmetic_precision_or_NPU_placement_proven": False}


def output_membership(value, desc):
    """Observed value membership only; no runtime inputs or quantizer fitting."""
    import numpy as np
    dtype = desc["dtype_enum"]
    if dtype == 0:
        return {"grid": "finite FP32", "elements": value.size, "exact_value_members": value.size}
    if dtype == 1:
        with np.errstate(over="ignore", under="ignore"):
            rounded = value.astype(np.float16).astype(np.float32)
        return {"grid": "finite FP16", "elements": value.size,
                "exact_value_members": int(np.count_nonzero(np.isfinite(rounded) & (value == rounded)))}
    low, high = (-128, 127) if dtype == 2 else (0, 255)
    centered = np.arange(low, high + 1, dtype=np.float64) - desc["zero_point"]
    scale = float(np.float32(desc["scale"]))
    require(scale == desc["scale"], "SDK affine scale must retain its recorded FP32 value")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        half_scale = float(np.float16(scale))
        grids = {"FP32_scale_FP32_product": (centered * scale).astype(np.float32),
                 "FP32_scale_FP16_rounded_product": (centered * scale).astype(np.float16).astype(np.float32),
                 "FP16_rounded_scale_FP32_product": (centered * half_scale).astype(np.float32),
                 "FP16_rounded_scale_FP16_rounded_product": (centered * half_scale).astype(np.float16).astype(np.float32)}
    results = {}
    for name, grid in grids.items():
        if not np.isfinite(grid).all():
            results[name] = {"finite_grid_defined": False}
            continue
        r, g = value.astype(np.float64), grid.astype(np.float64)
        pos = np.searchsorted(g, r)
        left, right = g[np.clip(pos - 1, 0, 255)], g[np.clip(pos, 0, 255)]
        distance = np.minimum(np.abs(r - left), np.abs(r - right))
        results[name] = {"finite_grid_defined": True, "elements": value.size,
                         "exact_value_members": int(np.count_nonzero(distance == 0)),
                         "max_distance_to_grid": float(distance.max()),
                         "grid_endpoints": [float(grid[0]), float(grid[-1])]}
    return {"grid": desc["logical_dtype"] + " affine hypotheses", "hypotheses": results,
            "SDK_rounding_rule_or_internal_error_cause_claimed": False}


def validate_role_arrays(normal, profile, helper):
    """Same-role determinism; cross-role equality is observed but not required."""
    import numpy as np
    result = []
    for sid in SAMPLES:
        for role in ROLES:
            first = normal[(sid, role, 0)]
            arrays = [table[(sid, role, i)] for table in (normal, profile) for i in range(4)]
            require(all(np.array_equal(first, x) and helper.raw_sha(first) == helper.raw_sha(x) for x in arrays),
                    "Same-role saved repeats or normal/profile output differs bitwise")
            result.append({"sample_id": sid, "role": role, "outputs_compared": 8,
                           "same_role_all_repeats_normal_profile_bitwise_equal": True,
                           "shared_raw_sha256": helper.raw_sha(first)})
    return result


def audit_cohort(directory, expected_sha, normal, bindings, samples, references, helper, pins):
    run = json.loads(helper.pin(directory / "run.json", expected_sha, pins).read_text())
    expected = {"schema": "v50_original_pair_hybrid_SDK_AB", "status": "completed_diagnostic_not_adopted",
                "script_sha256": bindings["runtime_script_sha256"], "parent_source_sha256": helper.PARENT_SHA,
                "ordinary_compile_sha256": helper.COMPILE_SHA, "capture_sha256": helper.CAPTURE_SHA,
                "original_graph_sha256": helper.GRAPH_SHA, "step1_sha256": STEP1_SHA,
                "normal_benchmark": normal, "profile_separate_fresh_contexts": not normal,
                "strict_gate": {"rtol": helper.RTOL, "atol": helper.ATOL}, "candidate_accepted": False,
                "full_model_run": False, "real_time_validated": False, "whole_model_quality_validated": False,
                "contexts_live_together": True, "requested_sdk_call_count": 16, "successful_SDK_calls": 16,
                "unmeasured_warmup_calls": 0, "cleanup_errors": [], "affinity_restored": True,
                "environment_restored": True, "source_and_inputs_unchanged": True}
    for key in ("hybrid_compile_sha256", "hybrid_source_sha256", "hybrid_contracts_sha256", "hybrid_policy_sha256"):
        expected[key] = bindings[key]
    require(all(type(run.get(k)) is type(v) and run[k] == v for k, v in expected.items())
            and run["source_and_input_hashes"] == run["source_and_input_hashes_after"]
            and run["original_checkpoint"]["sha256"] == helper.CHECKPOINT_SHA
            and len(run["calls"]) == len(run["sdk_calls"]) == 16, "Pinned hybrid run/source/gate/lifecycle changed")
    descriptions = {}
    for role in ROLES:
        context = run["context_records"][role]
        require(context["init_flags"] == (0 if normal else 8) and context["core_mask"] == 7
                and context["context_destroyed"] is True and context["tensor_attributes_validated"] is True
                and all(context[k] == 0 for k in ("init_code", "core_mask_code", "destroy_code"))
                and context["sdk_version"].startswith("2.3.2 ") and context["driver_version"] == "0.7.2", "Context lifecycle changed")
        descriptions[role] = {d: descriptor(context[d + "_attr"], d, helper) for d in ("input", "output")}
        floating = all(x["floating"] for x in descriptions[role].values())
        integer = any(not x["floating"] for x in descriptions[role].values())
        require(context["actual_logical_IO_floating"] is floating and context["integer_logical_IO_observed"] is integer
                and (role != "ordinary" or floating), "Queried boundary flags disagree with logical descriptors")
    require(run["hybrid_requested_floating_IO_observed"] is all(x["floating"] for x in descriptions["hybrid"].values()),
            "Hybrid floating-boundary observation flag differs from actual descriptors")
    arrays, calls, last_end = {}, [], 0
    for i, (sid, role, iteration) in enumerate(expected_calls()):
        row, sdk = run["calls"][i], run["sdk_calls"][i]
        for item in (row, sdk):
            require((item["sdk_call_index"], item["sample_id"], item["role"], item["round_index"])
                    == (i, sid, role, iteration), "Exact ordinary/hybrid AB/BA ordering changed")
        require(sdk["sample_call_index"] == iteration and sdk["context_call_index"] == SAMPLES.index(sid) * 4 + iteration
                and sdk["first_in_context"] is (sid == SAMPLES[0] and iteration == 0), "Per-sample/context repeat identity changed")
        helper.audit_interval(sdk)
        require(row["system_before"]["monotonic_ns"] <= sdk["io_begin_monotonic_ns"] > last_end
                and sdk["io_end_monotonic_ns"] <= row["system_after"]["monotonic_ns"], "Serial I/O intervals changed")
        last_end = sdk["io_end_monotonic_ns"]
        helper.numeric_same(sdk["io_total_seconds"], row["io_total_seconds"], "call IO interval")
        require(sdk["input_raw_sha256_before"] == sdk["input_raw_sha256_after"] == row["input_raw_sha256"]
                == samples[sid]["artifacts"]["input.npy"]["raw_sha256"]
                and row["reference_raw_sha256"] == samples[sid]["artifacts"]["output.npy"]["raw_sha256"], "Original input/reference identity changed")
        value = helper.load_array(directory, row["output"], helper.OUTPUT_SHAPE, pins)
        metric = helper.original_metrics(value, references[sid])
        for key, result in metric.items():
            helper.numeric_same(result, row["original_FP32_comparison"][key], key)
        arrays[(sid, role, iteration)] = value
        equal = helper.raw_sha(value) == helper.raw_sha(arrays[(sid, role, 0)])
        require(row["byte_equal_to_first_for_sample"] is equal and equal, "Same-role repeat raw bytes changed")
        calls.append({"sdk_call_index": i, "sample_id": sid, "role": role, "round_index": iteration,
                      "io_total_seconds": row["io_total_seconds"], "original_FP32_comparison": metric})
        if normal:
            require("separate_profile_detail" not in row, "Normal run contains profiling collection")
        else:
            detail, name = row["separate_profile_detail"], Path(row["separate_profile_detail"]["file"])
            require(not name.is_absolute() and name.name == str(name), "Profile text escaped directory")
            path = helper.pin(directory / name, detail["saved_text_sha256"], pins)
            require(path.stat().st_size == detail["bytes"] and detail["raw_sha256"] == detail["saved_text_sha256"], "Profile text bytes/hash changed")
    summaries = {role: six_summary(calls, role, helper) for role in ROLES}
    for role, summary in summaries.items():
        declared = run["summaries"][role]
        require(declared["mean_denominator"] == 6 and declared["subsequent_IO_mean_seconds"] == summary["subsequent_IO_mean_seconds"],
                "Exact board-ordered six-call IO mean changed")
        gate = all(x["original_FP32_comparison"]["strict_FP32_gate_passed"] for x in calls if x["role"] == role)
        require(declared["strict_original_FP32_gate_passed"] is gate, "Unchanged aggregate reference gate differs")
        summary["strict_original_FP32_gate_passed"] = gate
        summary["subsequent_stage_mean_seconds"] = {stage: helper.ordered_sum(x[stage] for x in run["sdk_calls"] if x["role"] == role and x["round_index"] > 0) / 6 for stage in helper.STAGES}
    return run, summaries, arrays, descriptions


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("normal-dir", "profile-dir", "capture-dir", "runtime-source-dir", "hybrid-source-dir",
                 "ordinary-compile-dir", "hybrid-compile-dir", "flash-auditor", "pins", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--pins-sha256", required=True)
    args = parser.parse_args()
    evidence_dirs = [getattr(args, x) for x in ("normal_dir", "profile_dir", "capture_dir", "runtime_source_dir",
                     "hybrid_source_dir", "ordinary_compile_dir", "hybrid_compile_dir")]
    require(not args.output.exists() and not any(args.output.resolve().is_relative_to(x.resolve()) for x in evidence_dirs), "New audit outside evidence directories required")
    helper = load_pure(args.flash_auditor, HELPER_SHA, "_frozen_flash_saved_helpers")
    pins = {}
    helper.pin(args.flash_auditor, HELPER_SHA, pins)
    bindings = json.loads(helper.pin(args.pins, args.pins_sha256, pins).read_text())
    require(bindings["schema"] == "v50_original_pair_hybrid_saved_audit_bindings", "Dedicated original experiment bindings required")
    required_hashes = ("normal_run_sha256", "profile_run_sha256", "runtime_script_sha256", "hybrid_compile_sha256",
                       "hybrid_source_sha256", "hybrid_contracts_sha256", "hybrid_policy_sha256", "hybrid_model_sha256")
    require(all(type(bindings[k]) is str and len(bindings[k]) == 64 and all(c in "0123456789abcdef" for c in bindings[k]) for k in required_hashes), "Complete actual SHA bindings required")
    source_files = [(args.runtime_source_dir / source_basename(bindings["runtime_source_filename"]), bindings["runtime_script_sha256"])]
    for filename_key, sha_key in (("hybrid_source_filename", "hybrid_source_sha256"), ("hybrid_contracts_filename", "hybrid_contracts_sha256")):
        source_files.append((args.hybrid_source_dir / source_basename(bindings[filename_key]), bindings[sha_key]))
    source_files.append((args.hybrid_source_dir / "v50_hybrid_step2_contracts.py", R1_CONTRACTS_SHA))
    for path, digest in source_files:
        helper.pin(path, digest, pins)
    contracts = load_pure(source_files[2][0], bindings["hybrid_contracts_sha256"], "_pure_hybrid_compile_receipt_contracts")
    require(contracts.R1_CONTRACTS_SHA == R1_CONTRACTS_SHA and contracts.R1_SOURCE == source_files[3][0].resolve(), "Actual adjacent frozen R1 contract dependency required")
    ordinary = json.loads(helper.pin(args.ordinary_compile_dir / "compile.json", helper.COMPILE_SHA, pins).read_text())
    hybrid = json.loads(helper.pin(args.hybrid_compile_dir / "compile.json", bindings["hybrid_compile_sha256"], pins).read_text())
    compile_gate = contracts.check_step2_compile(hybrid, bindings["hybrid_source_sha256"], bindings["hybrid_contracts_sha256"], bindings["hybrid_policy_sha256"], STEP1_SHA, bindings["hybrid_model_sha256"])
    captured = json.loads(helper.pin(args.capture_dir / "capture.json", helper.CAPTURE_SHA, pins).read_text())
    require(captured["status"] == "completed_prefix_capture" and captured["script_sha256"] == helper.CAPTURE_SCRIPT_SHA
            and captured["contracts_sha256"] == helper.CAPTURE_CONTRACT_SHA and captured["checkpoint"]["sha256"] == helper.CHECKPOINT_SHA
            and captured["original_model_only"] is True, "Original saved Torch capture required")
    samples = {s["sample_id"]: s for s in captured["samples"] if s["split"] == "evaluation"}
    require(tuple(samples) == SAMPLES, "Exact calibration-excluded evaluation set required")
    references = {}
    for sid, sample in samples.items():
        helper.load_array(args.capture_dir, sample["artifacts"]["input.npy"], helper.INPUT_SHAPE, pins)
        references[sid] = helper.load_array(args.capture_dir, sample["artifacts"]["output.npy"], helper.OUTPUT_SHAPE, pins)
    n, nm, na, nd = audit_cohort(args.normal_dir, bindings["normal_run_sha256"], True, bindings, samples, references, helper, pins)
    p, pm, pa, pd = audit_cohort(args.profile_dir, bindings["profile_run_sha256"], False, bindings, samples, references, helper, pins)
    require(nd == pd and n["compiled_artifacts"] == p["compiled_artifacts"]
            == {"ordinary": ordinary["variants"]["ordinary"]["output"], "hybrid": hybrid["output"]}, "Normal/profile compile identity/descriptors changed")
    for run in (n, p):
        for path, digest in source_files:
            require([value for key, value in run["source_and_input_hashes"].items() if Path(key).name == path.name] == [digest], "Independently read source differs from actual run binding")
    equality = validate_role_arrays(na, pa, helper)
    first = [{"sample_id": sid, "role": role, "original_FP32_comparison": helper.original_metrics(na[(sid, role, 0)], references[sid]),
              "observed_output_grid": output_membership(na[(sid, role, 0)], nd[role]["output"])} for sid in SAMPLES for role in ROLES]
    cross = [{"sample_id": sid, "ordinary_hybrid_bitwise_equal": helper.raw_sha(na[(sid, "ordinary", 0)]) == helper.raw_sha(na[(sid, "hybrid", 0)]),
              "max_abs_ordinary_hybrid_difference": float(np.max(np.abs(na[(sid, "ordinary", 0)].astype(np.float64) - na[(sid, "hybrid", 0)].astype(np.float64))))} for sid in SAMPLES]
    for path, expected in pins.items():
        require(helper.sha(path) == expected, "Saved data/source changed during audit")
    baseline, candidate = (nm[x]["subsequent_IO_mean_seconds"] for x in ROLES)
    result = {"schema": "v50_original_pair_hybrid_saved_numeric_timing_audit", "audit_integrity_passed": True,
              "auditor_sha256": sha(__file__), "frozen_saved_data_helper_sha256": HELPER_SHA, "bindings_sha256": args.pins_sha256,
              "adjacent_frozen_R1_compile_contracts_sha256": R1_CONTRACTS_SHA,
              "normal_run_sha256": bindings["normal_run_sha256"], "profile_run_sha256": bindings["profile_run_sha256"],
              "original_capture_sha256": helper.CAPTURE_SHA, "original_graph_sha256": helper.GRAPH_SHA,
              "original_checkpoint_sha256": helper.CHECKPOINT_SHA, "hybrid_compile_receipt_gate": compile_gate,
              "compiled_precision_request_scope": hybrid["diagnostic_requested_scope"],
              "exact_original_MLP_only_precision_proven": hybrid["precision_edit_gate"]["exact_original_MLP_only_precision_proven"],
              "saved_output_NPY_files_checked": 32, "original_input_reference_NPY_files_checked": 4,
              "profile_text_files_SHA_and_bytes_checked": 16, "profile_operator_texts_parsed": False,
              "normal_profile_corresponding_same_role_bitwise_equal_calls": 16,
              "same_role_repeat_bitwise_equal_comparisons_both_cohorts": 24, "same_role_equality_checks": equality,
              "cross_role_equality_observed_not_required": cross, "first_outputs_by_role_and_input": first,
              "logical_descriptors": nd, "hybrid_requested_floating_IO_observed": n["hybrid_requested_floating_IO_observed"],
              "normal_IO_summaries": nm, "profile_IO_summaries": pm,
              "normal_hybrid_mean_reduction_fraction": (baseline - candidate) / baseline,
              "normal_hybrid_minus_ordinary_mean_seconds": candidate - baseline,
              "fixed_component_gate": {"rtol": helper.RTOL, "atol": helper.ATOL, "reference": "same-input original saved Torch FP32 output"},
              "exact_call_order_both_cohorts": [{"sdk_call_index": i, "sample_id": s, "role": who, "round_index": r} for i, (s, who, r) in enumerate(expected_calls())],
              "all_read_evidence_files_unchanged": True, "declared_restoration_and_cleanup_passed": True, "candidate_accepted": False,
              "actual_model_ORT_SDK_compiler_calibration_or_board_execution_by_auditor": False,
              "raw_arrays_weights_exports_or_SDK_binaries_published": False,
              "limits": ["Exact same-role repeats and normal/profile output equality; cross-role output equality is not required.",
                         "Unchanged original FP32 gate retained separately for each role/input; no adoption or ATE/RPE inference.",
                         "Logical descriptor and observed output grid do not prove internal precision or error cause.",
                         "Six normal repeated-input calls per role only; profile means remain separate and do not establish full-model speed.",
                         "Two calibration-excluded inputs reused for development; original frame0/global0 pair only, no new blind trajectory test.",
                         "Full original-model 20 Hz fresh poses, P95 100 ms, ATE/RPE, stability and memory goals remain unvalidated."]}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"audit_integrity_passed": True, "saved_output_NPY_files_checked": 32, "candidate_accepted": False, "receipt_sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
