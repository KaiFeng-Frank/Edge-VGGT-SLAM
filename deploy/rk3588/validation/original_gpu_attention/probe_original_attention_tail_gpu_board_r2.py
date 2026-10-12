#!/usr/bin/env python3
"""Bounded board-only original VGGT QK-norm/RoPE/full-SDPA GPU component probe.

Uses two real original captures, frame0 and global0. No model, Torch, ORT or
RKNN SDK is loaded; actual OpenCL calls are restricted to the RK3588/S board.
One first diagnostic call and six/twelve normal warm calls are saved per case.
Complete warm timing includes FP32 QKV validation/packing/upload, all three
kernels, final consumer-layout download, owned finite validation and event
release. Initialization, references, hashes, comparisons and saves are separate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import statistics
import sys
import time

CAPTURE_SOURCE_SHA256 = "c307091ec55daa32d3cb1fd53cef3f931bd50e97feb6f1f63894a2eecf959a5c"
CAPTURE_SHA256 = "3b4eb0b24d320699dd7d198fd3158454a3043ed7244a0290b267cbfc450e8fee"
LEGACY_SOURCE_SHA256 = "f06486299101fb233c68abc9908e8c2629c4cb1f7dca6241133e6e222634acc2"
LIBRARY_SHA256 = "d57e53240b5078eee5cfbc79e103544c090a7825c09fdf3877fee45084897dda"
CHECKPOINT = dict(bytes=5026874952, sha256="d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0")
LOAD_KEYS = ("q_pre_norm", "k_pre_norm", "v", "q_norm_weight", "q_norm_bias",
             "k_norm_weight", "k_norm_bias", "cos_y", "sin_y", "cos_x", "sin_x",
             "q_post_norm", "k_post_norm", "q_post_rope", "k_post_rope", "proj_input", "attention_tail")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pin(path, digest):
    path = Path(path).resolve(strict=True)
    require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None and path.is_file() and sha(path) == digest,
            "Frozen file identity differs: " + str(path))
    return path


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def gpu_state():
    gpu = Path("/sys/class/devfreq/fb000000.gpu")
    return {name: (gpu / name).read_text().strip() for name in ("governor", "min_freq", "max_freq")}


def observation(expected_npu_hz, expected_gpu_state):
    npu = Path("/sys/class/devfreq/fdab0000.npu")
    gpu = Path("/sys/class/devfreq/fb000000.gpu")
    npu_hz, gpu_hz = int((npu / "cur_freq").read_text()), int((gpu / "cur_freq").read_text())
    temperatures = {p.parent.name: int(p.read_text()) / 1000.
                    for p in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp"))}
    state = gpu_state()
    require(npu_hz == expected_npu_hz and state == expected_gpu_state and
            int(state["min_freq"]) <= gpu_hz <= int(state["max_freq"]),
            "NPU frequency or GPU governor/limits changed; this probe never writes frequency")
    require(temperatures and max(temperatures.values()) < 75.0, "Observed temperature reached the 75C bound")
    return dict(npu_hz=npu_hz, gpu_hz=gpu_hz, gpu_governor=state["governor"], gpu_state=state,
                temperature_c=temperatures)


def compare(np, actual, reference):
    require(actual.shape == reference.shape and actual.dtype == reference.dtype == np.float32 and
            np.isfinite(actual).all() and np.isfinite(reference).all(), "Original finite FP32 comparison required")
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    error = a - b
    outside = int(np.count_nonzero(np.abs(error) > 1e-5 + 1e-4 * np.abs(b)))
    return dict(strict_rtol=1e-4, strict_atol=1e-5, strict_pass=outside == 0,
                outside_tolerance=outside, elements=actual.size,
                byte_equal=actual.tobytes() == reference.tobytes(), max_abs=float(np.max(np.abs(error))),
                relative_l2=float(np.linalg.norm(error.ravel()) / max(np.linalg.norm(b.ravel()), 1e-30)))


def read_artifacts(np, capture_dir, role, shape, watched):
    require(role["block_index"] == 0 and role["attention_tail_is_new_original_CPU_reference_not_preexisting_saved_reference"],
            "Original frame0/global0 capture required")
    result = {}
    for name in LOAD_KEYS:
        row = role["artifacts"][name]
        relative = Path(row["file"])
        require(not relative.is_absolute() and ".." not in relative.parts, "Relative original capture artifact required")
        path = pin(capture_dir / relative, row["sha256"])
        require(path.is_relative_to(capture_dir) and path.stat().st_size == row["bytes"], "Capture artifact escaped root or size changed")
        value = np.load(path, allow_pickle=False)
        expected = ([64] if name.endswith("_weight") or name.endswith("_bias") else
                    [shape[0], 1, shape[2], 32] if name in ("cos_y", "sin_y", "cos_x", "sin_x") else
                    [shape[0], shape[2], 1024] if name == "proj_input" else list(shape))
        require(row["shape"] == expected and list(value.shape) == expected and
                row["dtype"] == "float32" and value.dtype == np.float32 and value.flags.c_contiguous and
                row["tensor_bytes"] == value.nbytes and row["finite"] and np.isfinite(value).all() and
                hashlib.sha256(value.tobytes()).hexdigest() == row["raw_sha256"], "Original capture tensor metadata differs: " + name)
        value = np.array(value, copy=True, order="C"); value.flags.writeable = False
        result[name] = value
        watched[path] = row["sha256"]
    require(np.array_equal(result["attention_tail"].transpose(0, 2, 1, 3).reshape(shape[0], shape[2], 1024),
                           result["proj_input"]), "Original CPU tail/projection layout relation changed")
    return result


def save_npz(np, directory, name, values):
    path = directory / name
    np.savez(path, **values)
    return dict(file=name, bytes=path.stat().st_size, sha256=sha(path),
                arrays={key: dict(shape=list(value.shape), dtype=str(value.dtype), tensor_bytes=value.nbytes,
                    raw_sha256=hashlib.sha256(value.tobytes()).hexdigest(), finite=bool(np.isfinite(value).all()))
                    for key, value in values.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--capture-sha256", default=CAPTURE_SHA256)
    parser.add_argument("--bridge-source", type=Path, required=True)
    parser.add_argument("--bridge-sha256", required=True)
    parser.add_argument("--legacy-source", type=Path, required=True)
    parser.add_argument("--gpu-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-npu-hz", type=int, choices=(300_000_000, 1_000_000_000), default=1_000_000_000)
    parser.add_argument("--warm-calls", type=int, choices=(6, 12), default=6)
    args = parser.parse_args()
    compatible = next((p.read_bytes() for p in (Path("/proc/device-tree/compatible"),
        Path("/sys/firmware/devicetree/base/compatible")) if p.is_file()), b"")
    require(sys.platform == "linux" and platform.machine().lower() in ("aarch64", "arm64") and b"rk3588" in compatible,
            "Actual GPU component execution is RK3588/S-only; host GPU is prohibited")
    sys.dont_write_bytecode = True
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        require(os.environ.get(variable) == "1", variable + " must be externally limited to one")
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    require(args.capture_sha256 == CAPTURE_SHA256, "Use the frozen actual original capture")
    capture_path = pin(args.capture, args.capture_sha256)
    bridge_path = pin(args.bridge_source, args.bridge_sha256)
    legacy_path = pin(args.legacy_source, LEGACY_SOURCE_SHA256)
    library_path = pin(args.gpu_library, LIBRARY_SHA256)
    watched = {capture_path: args.capture_sha256, bridge_path: args.bridge_sha256,
               legacy_path: LEGACY_SOURCE_SHA256, library_path: LIBRARY_SHA256}
    capture = json.loads(capture_path.read_text())
    require(capture["schema"] == "v83_original_VGGT_attention_tail_capture" and
            capture["status"] == "completed_original_FP32_attention_tail_capture" and
            capture["script_sha256"] == CAPTURE_SOURCE_SHA256 and capture["original_model_only"] and
            capture["actual_inference_on_board_only"] and capture["original_parameters_verified"] == 36 and
            all(capture[key] for key in ("checkpoint_verified_before_deserialization", "hooks_removed", "model_state_unchanged",
                "checkpoint_stat_unchanged", "source_and_inputs_unchanged", "environment_restored", "affinity_restored")) and
            capture["cleanup_errors"] == [] and [row["tag"] for row in capture["cases"]] == ["v53", "v56-calib00"] and
            all(capture["checkpoint"][key] == value for key, value in CHECKPOINT.items()), "Clean original capture lineage required")
    contract = capture["attention_contract"]
    require(contract["heads"] == 16 and contract["head_dim"] == 64 and
            contract["q_norm_eps"] == contract["k_norm_eps"] == 1e-5 and contract["head_scale"] == .125 and
            contract["original_SDPA_scale_argument"] is None and contract["mask"] is None and
            contract["dropout_p"] == 0.0 and contract["is_causal"] is False and contract["tensor_dtype"] == "float32" and
            contract["rope_frequency"] == 100 and contract["rope_scaling_factor"] == 1.0 and
            contract["rope_component_dimension"] == 32 and
            contract["rope_rotate"] == "Within each 32-dimensional vertical/horizontal half: concatenate(-last16,first16)",
            "Original normalization/RoPE/complete SDPA contract required")
    import numpy as np
    legacy = load_source(legacy_path, "_v83_original_tail_legacy_opencl")
    module = load_source(bridge_path, "_v83_original_attention_tail_bridge")
    require(module.LEGACY_SOURCE_SHA256 == LEGACY_SOURCE_SHA256, "Frozen legacy loader contract differs")
    cls = module.make_bridge_class(legacy)
    baseline_gpu_state = gpu_state()
    report = dict(schema="v83_original_VGGT_QK_norm_RoPE_SDPA_GPU_component_probe",
        status="running", original_model_only=True, host_GPU_used=False,
        candidate_accepted=False, full_model_quality_or_realtime_validated=False,
        frequency_written=False, expected_npu_hz=args.expected_npu_hz,
        GPU_frequency_scope="Observed automatic DVFS under the unchanged baseline governor/min/max; no GPU clock/governor/limit writes and no continuous or fixed-frequency claim",
        GPU_baseline_governor_limits=baseline_gpu_state,
        checkpoint_identity_from_original_capture=CHECKPOINT, capture_sha256=args.capture_sha256,
        capture_source_sha256=CAPTURE_SOURCE_SHA256, script_sha256=sha(__file__), bridge_source_sha256=args.bridge_sha256,
        legacy_OpenCL_source_sha256=LEGACY_SOURCE_SHA256, library_sha256=LIBRARY_SHA256,
        original_attention_contract=contract, original_all_tokens_retained=True,
        CPU_Torch_ORT_model_RKNN_SDK_compiler_executed=False, onboard_Mali_OpenCL_execution=True,
        native_INT8_equivalence_verified=False, NPU_integration_tested=False,
        warm_calls_per_case_role=args.warm_calls,
        first_diagnostic_call_separate_from_warm_mean=True,
        timing_scope="Input finite/FP32 owned pack+upload, all three kernels, owned [B,S,1024] download+finite, event profiling/release/finish; init/hash/save/compare separate",
        cases=[], cleanup_errors=[])
    report_path = output / "result.json"
    started = time.perf_counter()
    try:
        for case in capture["cases"]:
            require(set(case["roles"]) == {"frame", "global"}, "Both original attention roles required")
            for role_name in ("frame", "global"):
                shape = (2, 16, 53, 64) if role_name == "frame" else (1, 16, 106, 64)
                original = read_artifacts(np, capture_path.parent, case["roles"][role_name], shape, watched)
                row = dict(case=case["tag"], role=role_name, shape=list(shape), consumer_shape=[shape[0], shape[2], 1024],
                    status="running", calls=[], original_capture_array_sha256={key: hashlib.sha256(value.tobytes()).hexdigest()
                        for key, value in original.items()})
                report["cases"].append(row)
                bridge = cls.__new__(cls)
                try:
                    bridge.__init__(shape=shape,
                        parameters=[original[key] for key in ("q_norm_weight", "q_norm_bias", "k_norm_weight", "k_norm_bias")],
                        tables=[original[key] for key in ("cos_y", "sin_y", "cos_x", "sin_x")],
                        q_eps=contract["q_norm_eps"], k_eps=contract["k_norm_eps"], scale=contract["head_scale"],
                        library=str(library_path))
                    row["initialization_metrics"] = bridge.metrics()
                    first = None
                    for index in range(args.warm_calls + 1):
                        before = observation(args.expected_npu_hz, baseline_gpu_state)
                        begin = time.perf_counter()
                        value, diagnostics, timing = bridge.run(original["q_pre_norm"], original["k_pre_norm"], original["v"],
                                                                 capture_intermediates=index == 0)
                        elapsed = time.perf_counter() - begin
                        after = observation(args.expected_npu_hz, baseline_gpu_state)
                        call = dict(index=index, diagnostic_first=index == 0, elapsed_seconds=elapsed,
                                    bridge_timing=timing, before=before, after=after,
                                    output_vs_original_CPU=compare(np, value, original["proj_input"]))
                        saved = dict(proj_input=value)
                        if index == 0:
                            first = value.copy()
                            tail = np.array(value.reshape(shape[0], shape[2], 16, 64).transpose(0, 2, 1, 3), copy=True, order="C")
                            saved.update(diagnostics, attention_tail=tail)
                            call["intermediates_vs_original_CPU"] = {key: compare(np, result, original[key])
                                                                      for key, result in diagnostics.items()}
                            call["tail_layout_vs_original_CPU"] = compare(np, tail, original["attention_tail"])
                        else:
                            call["output_vs_first_diagnostic"] = compare(np, value, first)
                        name = f"{case['tag']}-{role_name}-call{index:02d}.npz"
                        call["saved_output"] = save_npz(np, output, name, saved)
                        row["calls"].append(call)
                    warm = row["calls"][1:]
                    durations = [c["elapsed_seconds"] for c in warm]
                    row["summary"] = dict(diagnostic_first_seconds=row["calls"][0]["elapsed_seconds"],
                        warm_calls=len(warm), warm_complete_mean_seconds=statistics.mean(durations),
                        warm_complete_min_seconds=min(durations), warm_complete_max_seconds=max(durations),
                        warm_complete_p95_seconds=float(np.percentile(durations, 95)),
                        warm_kernel_mean_seconds=statistics.mean(c["bridge_timing"]["GPU_kernel_seconds_sum"] for c in warm),
                        all_warm_final_outputs_byte_equal_to_first=all(c["output_vs_first_diagnostic"]["byte_equal"] for c in warm),
                        all_final_outputs_strict_pass_original_CPU=all(c["output_vs_original_CPU"]["strict_pass"] for c in row["calls"]),
                        first_all_intermediates_strict_pass_original_CPU=all(c["strict_pass"] for c in
                            row["calls"][0]["intermediates_vs_original_CPU"].values()))
                    row["status"] = "completed_component_not_quality_acceptance"
                finally:
                    close_error = None
                    try:
                        bridge.close()
                    except BaseException as error:
                        close_error = type(error).__name__ + ": " + str(error)
                        report["cleanup_errors"].append(row["case"] + "/" + role_name + ": " + close_error)
                    metrics = bridge.metrics()
                    row["metrics_after_close"] = metrics
                    released = (close_error is None and metrics["closed"] and metrics["fully_released"] and
                        metrics["live_OpenCL_objects"] == metrics["live_owned_DMA_fds"] == metrics["pending_owned_host_uploads"] == 0 and
                        metrics["events_created"] == metrics["events_released"] and metrics["cleanup_errors"] == [])
                    if not released:
                        raise RuntimeError("GPU partial/full cleanup failed; retain live owner until process exits")
                atomic(report_path, report)
        require(all(sha(path) == digest for path, digest in watched.items()), "Frozen input/source/library changed during probe")
        gpu_frequencies = [sample["gpu_hz"] for row in report["cases"] for call in row["calls"]
                           for sample in (call["before"], call["after"])]
        require(gpu_state() == baseline_gpu_state, "GPU governor/limits did not remain unchanged")
        report.update(status="completed_component_not_quality_acceptance", source_and_inputs_unchanged=True,
                      GPU_final_governor_limits=gpu_state(), GPU_governor_limits_unchanged=True,
                      GPU_observed_hz_range=[min(gpu_frequencies), max(gpu_frequencies)],
                      GPU_unique_observed_hz=sorted(set(gpu_frequencies)),
                      watched_file_sha256={str(path): digest for path, digest in watched.items()},
                      GPU_calls=sum(len(row["calls"]) for row in report["cases"]),
                      events_created=sum(row["metrics_after_close"]["events_created"] for row in report["cases"]),
                      events_released=sum(row["metrics_after_close"]["events_released"] for row in report["cases"]))
    except BaseException as error:
        report.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        report["GPU_final_governor_limits"] = gpu_state()
        report["GPU_governor_limits_unchanged"] = report["GPU_final_governor_limits"] == baseline_gpu_state
        report["worker_seconds_including_init_comparison_and_saves"] = time.perf_counter() - started
        atomic(report_path, report)
    print(json.dumps(dict(status=report["status"], GPU_calls=report["GPU_calls"],
                         cleanup_errors=report["cleanup_errors"])), flush=True)


if __name__ == "__main__":
    main()
