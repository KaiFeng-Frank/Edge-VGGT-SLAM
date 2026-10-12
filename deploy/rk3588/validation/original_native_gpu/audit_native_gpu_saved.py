#!/usr/bin/env python3
"""Saved-only v81 NumPy audit. No Torch, model, OpenCL or RKNN execution."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(actual, reference):
    actual, reference = actual.astype(np.float64), reference.astype(np.float64)
    delta = np.abs(actual - reference)
    return {"strict_pass": bool((delta <= 1e-5 + 1e-4 * np.abs(reference)).all()),
            "outside": int((delta > 1e-5 + 1e-4 * np.abs(reference)).sum()),
            "elements": actual.size, "max_abs": float(delta.max()),
            "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(reference))}


def ordered_half(value):
    bits = value.view(np.uint16).astype(np.int32)
    return np.where(bits & 0x8000, 0x8000 - (bits & 0x7fff), 0x8000 + bits)


def same_stat(actual, expected):
    # Float64 norm reduction may differ slightly between board and host BLAS.
    return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15) if isinstance(actual, float) else actual == expected


def audit(directory, reference_dir, control_path):
    files = {}

    def record(path):
        files[path] = sha(path)
        return files[path]

    result_path = directory / "result.json"
    result = json.loads(result_path.read_text())
    record(result_path)
    previous_path = reference_dir / "result.json"
    previous = json.loads(previous_path.read_text())
    require(record(previous_path) == result["reference"]["saved_receipt_sha256"], "reference receipt changed")
    control = json.loads(control_path.read_text())
    record(control_path)

    def load(parent, meta):
        path = (parent / meta["file"]).resolve(strict=True)
        require(path.is_relative_to(parent.resolve()) and not path.is_symlink(), "array outside saved directory")
        require(record(path) == meta["sha256"] and path.stat().st_size == meta["bytes"], "saved array identity changed")
        value = np.load(path, allow_pickle=False)
        require(value.flags.c_contiguous and np.isfinite(value).all() and list(value.shape) == meta["shape"] and
                str(value.dtype) == meta["dtype"] and hashlib.sha256(value.tobytes()).hexdigest() == meta["raw_sha256"],
                "saved array shape/dtype/value changed")
        return value

    reference = load(reference_dir, result["reference"]["original_CPU_output"])
    original_fc1 = load(reference_dir, previous["artifacts"]["torch_fc1"])
    original_gelu = load(reference_dir, previous["artifacts"]["torch_fc2_input"])
    intermediate = {key: load(directory, meta)
                    for key, meta in result["GPU_same_NPU_FC1_validation"]["artifacts"].items()}
    x, cpu, gpu = (intermediate[key] for key in ("FC1_biased", "CPU_GELU", "GPU_GELU_half"))
    require(x.shape == cpu.shape == gpu.shape == (106, 4096) and x.dtype == cpu.dtype == np.float32 and
            gpu.dtype == np.float16, "original intermediate contract changed")
    native_gpu = np.ascontiguousarray(intermediate["GPU_native_A8"].transpose(1, 0, 2).reshape(106, 4096))
    require(native_gpu.tobytes() == gpu.tobytes(), "saved native A8 unpack is wrong")
    rounded_cpu = cpu.astype(np.float16)
    ulp = np.abs(ordered_half(gpu) - ordered_half(rounded_cpu))
    keys, counts = np.unique(ulp, return_counts=True)
    strict = np.abs(gpu.astype(np.float64) - rounded_cpu.astype(np.float64)) > 1e-5 + 1e-4 * np.abs(rounded_cpu.astype(np.float64))
    disputed = gpu.view(np.uint16) != rounded_cpu.view(np.uint16)
    require(int(strict.sum()) == 637 and int(disputed.sum()) == 1685, "frozen bridge disagreement counts changed")
    require((x[disputed] < 0).all() and (ulp[strict] == 1).all(), "frozen negative/one-ULP finding changed")
    bridge_comparison = compare(gpu, rounded_cpu)
    for key, measured in bridge_comparison.items():
        require(same_stat(measured, result["GPU_same_NPU_FC1_validation"]["GPU_GELU_vs_same_FC1_CPU_GELU_once_FP16_rounded"][key]),
                "stored bridge comparison mismatch: " + key)

    # Numeric diagnostics for saved data, not a model or new inference. Only the
    # 1685 already-disputed elements call scalar math.erf. This is double libm
    # rounded to float32, not the board's float std::erf or a GPU implementation.
    scale_mul = x * np.float32(math.sqrt(.5))
    scale_div = x / np.float32(math.sqrt(2))
    samples = x[disputed]
    erm = np.fromiter((math.erf(float(v)) for v in scale_mul[disputed]), dtype=np.float64).astype(np.float32)
    erd = np.fromiter((math.erf(float(v)) for v in scale_div[disputed]), dtype=np.float64).astype(np.float32)
    torch_tree = ((samples * np.float32(.5)) * (np.float32(1) + erm)).astype(np.float16)
    gpu_tree = ((samples * (np.float32(1) + erd)) * np.float32(.5)).astype(np.float16)
    require(np.array_equal((samples * np.float32(.5)) * (np.float32(1) + erm),
                           (samples * (np.float32(1) + erm)) * np.float32(.5)), "half-multiply reordering differs")
    diagnostic = {}
    for name, value in (("Torch_scale_mul", torch_tree), ("GPU_scale_div", gpu_tree)):
        diagnostic[name] = {}
        for scope, selection in (("all_1685_disputed", np.ones(len(samples), dtype=bool)), ("strict_637", strict[disputed])):
            diagnostic[name][scope] = {
                "matches_saved_CPU_half": int((value[selection].view(np.uint16) == rounded_cpu[disputed][selection].view(np.uint16)).sum()),
                "matches_saved_GPU_half": int((value[selection].view(np.uint16) == gpu[disputed][selection].view(np.uint16)).sum())}

    summaries, first_outputs = {}, {}
    for route in "ABCD":
        calls = [row for row in result["calls"] if row["route"] == route]
        require(len(calls) == 13 and sum(row["first_for_route"] for row in calls) == 1, "route call accounting changed")
        values = [load(directory, row["output"]) for row in calls]
        require(all(value.shape == (106, 1024) and value.dtype == np.float32 and value.tobytes() == values[0].tobytes()
                    for value in values), "saved full output is unstable or invalid")
        measured = compare(values[0], reference)
        require(all(all(same_stat(value, row["vs_original_CPU_FP32"][key]) for key, value in measured.items()) for row in calls),
                "stored full-output comparison changed")
        warm = [row["complete_route_seconds"] for row in calls if not row["first_for_route"]]
        mean = float(np.mean(warm))
        require(abs(mean - result["summary_seconds"][route]["warm_mean"]) < 1e-12, "stored route timing mismatch")
        summaries[route] = {"calls": len(calls), "all_outputs_bitwise_stable": True,
                            "warm_mean_ms": 1000 * mean, "vs_original_FP32": measured,
                            "output_raw_sha256": hashlib.sha256(values[0].tobytes()).hexdigest()}
        first_outputs[route] = values[0]
    require(first_outputs["B"].tobytes() == first_outputs["C"].tobytes(), "B/C layout comparison changed")
    native = result["native_API"]
    require(result["native_lifecycle_balanced"] and native["calls"] == native["successes"] and
            native["successes"]["rknn_matmul_create"] == native["successes"]["rknn_matmul_destroy"] == 4 and
            native["successes"]["rknn_create_mem"] == native["successes"]["rknn_destroy_mem"] == 12 and
            native["successes"]["rknn_matmul_run"] == 78 and all(code == 0 for codes in native["return_codes"].values() for code in codes) and
            all(not row["live_context"] and not row["unresolved_memory_count"] and not row["cleanup_errors"]
                for row in result["native_contexts_after_close"]), "native resources/run accounting failed")
    compiled, cl = result["compiled_metrics_after_close"], result["GPU_metrics_after_close"]
    require(compiled["closed"] and not compiled["failed"] and compiled["live_contexts"] == 0 and
            compiled["contexts_created"] == compiled["contexts_destroyed"] == 1 and
            compiled["outputs_acquired"] == compiled["outputs_released"] == compiled["sdk_runs_succeeded"] == 13 and
            not compiled["cleanup_errors"] and cl["closed"] and not cl["failed"] and cl["fully_released"] and
            cl["live_OpenCL_objects"] == cl["live_owned_DMA_fds"] == 0 and
            cl["calls_completed"] == cl["events_created"] == cl["events_released"] == 13 and not cl["cleanup_errors"] and
            not result["cleanup_errors"] and result["affinity_restored"] and result["sources_and_inputs_unchanged"],
            "compiled/GPU cleanup failed")
    require(cl["original_bias_raw_sha256"] == result["original_parameter_metadata"]["mlp.fc1.bias"]["raw_sha256"],
            "bridge original bias identity changed")
    require(control["child_returncode"] == 0 and not control["restore_errors"] and control["frequency_governor_limits_restored"] and
            control["baseline"] == control["restored"] and control["restored"]["cur_freq"] == "300000000" and
            all(row["actual_frequency_hz"] == 1000000000 for row in control["observations"]), "control observations/restore failed")
    require(result["candidate_accepted"] is False and result["full_model_or_SLAM_validated"] is False, "unsupported acceptance claim")
    require(all(sha(path) == digest for path, digest in files.items()), "saved inputs changed during audit")
    gpu_warm = [row for row in result["calls"] if row["route"] == "D" and not row["first_for_route"]]
    return {
        "schema": "v81_original_MLP_GPU_saved_numeric_audit", "status": "saved_consistency_passed_precision_gate_failed",
        "scope": "Saved NumPy arrays and scalar arithmetic diagnostics only; no Torch/model/SDK/board execution",
        "same_FC1": {"elements": gpu.size, "vs_original_FP32_FC1": compare(x, original_fc1),
                     "CPU_GELU_vs_original_FP32_GELU": compare(cpu, original_gelu),
                     "GPU_half_vs_original_FP32_GELU": compare(gpu, original_gelu),
                     "GPU_vs_same_FC1_CPU_once_FP16": bridge_comparison,
                     "FP16_ULP_histogram": {str(int(k)): int(v) for k, v in zip(keys, counts)},
                     "FP16_numeric_and_bitwise_differences": int(disputed.sum()), "maximum_FP16_ULP_distance": int(ulp.max()),
                     "all_637_strict_failures_have_ULP_distance": 1, "all_differences_have_negative_FC1": True,
                     "disputed_FC1_range": [float(x[disputed].min()), float(x[disputed].max())],
                     "strict_fail_FC1_range": [float(x[strict].min()), float(x[strict].max())], "native_A8_unpacked_bits_match": True},
        "scalar_rounding_diagnostic": {"precision_caveat": "double math.erf then FP32; not the exact board float std::erf or GPU erf implementation",
                                       "scalar_inputs_checked": int(disputed.sum()), "comparisons": diagnostic,
                                       "FP32_mul_vs_div_scaled_input_differences": int((scale_mul.view(np.uint32) != scale_div.view(np.uint32)).sum()),
                                       "maximum_scaled_input_absolute_difference": float(np.abs(scale_mul - scale_div).max()),
                                       "scale_choice_half_changes_in_disputed_points": int((torch_tree.view(np.uint16) != gpu_tree.view(np.uint16)).sum()),
                                       "scale_choice_half_changes_in_637_strict_points": int((torch_tree[strict[disputed]].view(np.uint16) != gpu_tree[strict[disputed]].view(np.uint16)).sum()),
                                       "half_multiply_order_changes_in_disputed_points": 0},
        "full_outputs": summaries, "B_and_C_bitwise_equal": True,
        "D_vs_C": compare(first_outputs["D"], first_outputs["C"]),
        "D_over_compiled_warm_latency_ratio": summaries["D"]["warm_mean_ms"] / summaries["A"]["warm_mean_ms"],
        "D_warm_GPU_bridge_mean_ms": float(np.mean([row["stages_seconds"]["GPU_bridge_breakdown"]["complete_bridge_seconds"] for row in gpu_warm])) * 1000,
        "D_warm_GPU_kernel_mean_ms": float(np.mean([row["stages_seconds"]["GPU_bridge_breakdown"]["GPU_kernel_seconds"] for row in gpu_warm])) * 1000,
        "resources": {"native_contexts_created_destroyed": 4, "native_allocations_created_destroyed": 12, "native_runs": 78,
                      "compiled_contexts_created_destroyed": 1, "compiled_runs_outputs_acquired_released": 13,
                      "GPU_calls_events_created_released": 13, "live_contexts_OpenCL_objects_and_owned_DMA_fds": 0,
                      "cleanup_errors": [], "observed_1GHz_controller_samples": len(control["observations"]),
                      "maximum_observed_temperature_c": max(v for row in control["observations"] for v in row["temperatures_c"].values()),
                      "exact_baseline_restored": control["restored"]},
        "interpretation": "Inference: scale rounding accounts for a minority of disputed scalar points; standard GPU Erf vs CPU float Erf accuracy and negative 1+erf cancellation are consistent with the remaining half-boundary differences. GPU FP32 scaled/erf values were not saved, so the exact cause is unproven.",
        "next_source_fix": {"proposal_only_not_applied": True,
                            "first": "Match torch 2.3.1 scalar scaling: x * FP32(1/sqrt(2)), then (x * 0.5) * (1 + erf); keep original bias and one half_rte conversion.",
                            "second": "If needed capture GPU FP32 scaled input and Erf on the same saved diagnostic input before choosing an original-Erf implementation; do not assume the optional ARM NEON polynomial is the active wheel path.",
                            "limits": "Changing scale alone is not predicted to remove all 637 failures. This GPU native route is slower than compiled MLP and does not justify full-model expansion or accuracy/SLAM acceptance."},
        "primary_source_evidence": [
            {"url": "https://github.com/pytorch/pytorch/blob/v2.3.1/aten/src/ATen/native/cpu/Activation.cpp#L408-L425", "supports": "FP32 original GELU uses multiply-by-alpha scaling and x*half grouping"},
            {"url": "https://github.com/pytorch/pytorch/blob/v2.3.1/aten/src/ATen/cpu/vec/vec_base.h#L384-L386", "supports": "generic vector Erf maps std::erf"},
            {"url": "https://github.com/pytorch/pytorch/blob/v2.3.1/aten/src/ATen/cpu/vec/vec256/vec256_float_neon.h#L829-L858", "supports": "optional NEON Erf has a distinct polynomial/FMA implementation; active build path unverified"},
            {"url": "https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_C.html#relative-error-as-ulps", "supports": "standard single-precision Erf permits a 16-ULP error bound; no-relaxed-math is not CPU bit equality"}],
        "candidate_accepted": False, "general_model_trajectory_or_realtime_accuracy_validated": False,
        "input_files_count": len(files), "input_sha256": {str(path): digest for path, digest in files.items()},
        "audited_inputs_unchanged": True,
    }


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=root / "native-gpu-r1")
    parser.add_argument("--reference-dir", type=Path, default=Path("/home/frank/.local/share/edge-vggt-private/v68/native-compiled-rank3-ab-r1"))
    parser.add_argument("--control", type=Path, default=root / "native-gpu-control-r1/control.json")
    parser.add_argument("--output", type=Path, default=root / "native-gpu-saved-audit.json")
    args = parser.parse_args()
    require(not args.output.resolve().is_relative_to(args.directory.resolve()) and
            not args.output.resolve().is_relative_to(args.reference_dir.resolve()) and
            args.output.resolve() not in (args.control.resolve(), Path(__file__).resolve()), "output must be separate from saved inputs")
    report = audit(args.directory.resolve(), args.reference_dir.resolve(), args.control.resolve())
    report["audit_script_sha256"] = sha(Path(__file__).resolve())
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output_sha256": sha(args.output),
                      "script_sha256": report["audit_script_sha256"], "FP16_ULP_histogram": report["same_FC1"]["FP16_ULP_histogram"],
                      "D_warm_mean_ms": report["full_outputs"]["D"]["warm_mean_ms"]}, sort_keys=True))


if __name__ == "__main__":
    main()
