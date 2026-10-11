#!/usr/bin/env python3
"""Portable saved-array-only audit of the v32 encoder research comparison.

Audit success means evidence is consistent, never acceptance of this known
FP32-rejected candidate. No Torch, ORT, SDK, image decoding or inference.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

sys.dont_write_bytecode = True
PARSER_SHA = "22b5b2ce0f6e86e9351f6337486f96d649964de6447d5556dd55fee5946d6d65"
COMPILER_SHA = "364d3493f9fe13789b449c0836c37d0eedca71cdbde71e10e28723ec88d1b003"
CAPTURE_SHA = "e0563efb962edb90684480be114b88bc2c7825015f3b13a163f6f47f06360088"
ORT_SHA = "153da765b1145d7bb74bc635c1982ae74c652dac28cf99be56cb40534c44b667"
ORIGINAL_COMPILE_SHA = "d6d13d829357c551223b8c620c48daf48c156430d376ca0578675fc2ad3bcec2"
REWRITE_SHA = "66eb8d135165f7538c97733d0c83198fa5a674e0cd2bdf64eb190c24819c5109"
FAILED_ORT_SHA = "a517f822a2e4afd01aff87df303183a3dc5850313134548891dc13c1e36b896b"
COMPLETE = "completed_encoder_diagnostic_FP32_rejection_retained_not_VO"
SHAPES = {"fmap": (1, 1, 128, 116, 152), "imap": (1, 1, 384, 116, 152)}
STEPS = ("input_packing_seconds", "input_set_seconds", "run_api_seconds",
         "outputs_get_seconds", "output_copy_seconds", "outputs_release_seconds")


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def load_module(path, expected, name):
    if sha(path) != expected:
        raise ValueError("Pinned pure helper SHA differs: " + Path(path).name)
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def schedule(profile=False):
    if profile:
        return [(i, "spatial", "profile") for i in range(3)]
    return [(0, "original", "first"), (0, "spatial", "first"),
            (0, "spatial", "warm"), (0, "original", "warm"),
            (0, "original", "warm"), (0, "spatial", "warm"),
            (1, "spatial", "other_frame"), (1, "original", "other_frame"),
            (2, "original", "other_frame"), (2, "spatial", "other_frame")]


def finite_stats(value):
    return dict(elements=int(value.size), finite=int(np.isfinite(value).sum()),
                nan=int(np.isnan(value).sum()), positive_inf=int(np.isposinf(value).sum()),
                negative_inf=int(np.isneginf(value).sum()), all_zero=bool(np.all(value == 0)))


def compare(actual, reference, parser):
    if actual.dtype != np.float32 or reference.dtype != np.float32 or actual.shape != reference.shape:
        raise ValueError("Matching FP32 shape required")
    if np.isfinite(actual).all() and np.isfinite(reference).all():
        return parser.errors(actual, reference)
    return dict(rtol=1e-4, atol=1e-5, gate_passed=False, finite=False,
                failed_elements=None, element_count=int(actual.size), max_abs=None,
                rms=None, relative_l2=None, cosine=None, finite_counts=finite_stats(actual),
                reason="Nonfinite raw output; no finite error norm is claimed")


def summary(calls):
    result = {}
    for method in sorted({row["method"] for row in calls}):
        rows = [row for row in calls if row["method"] == method]
        warm = [row["forward_seconds"] for row in rows if row["timing_role"] == "warm"]
        result[method] = dict(calls=len(rows), warm_seconds=warm,
                             warm_mean_seconds=sum(warm) / len(warm) if warm else None,
                             first_seconds=next((row["forward_seconds"] for row in rows
                                                 if row["timing_role"] == "first"), None))
    return result


def validate_descriptor(descriptor, index, name, shape, *, is_input):
    expected = {0: list(shape), 1: [shape[0], shape[2], shape[3], shape[1]]} if is_input else {0: list(shape)}
    okay = (descriptor["index"] == index and descriptor["name"] == name
            and descriptor["format"] in expected and descriptor["shape"] == expected[descriptor["format"]]
            and descriptor["type"] in (0, 1) and descriptor["n_elems"] == math.prod(shape)
            and descriptor["size"] == math.prod(shape) * (4 if descriptor["type"] == 0 else 2)
            and (descriptor["quantization_type"] == 0 or descriptor["type"] == 1
                 and descriptor["quantization_type"] == 2 and descriptor["zero_point"] == 0
                 and descriptor["scale"] == 1.0))
    if not okay:
        raise ValueError("SDK native descriptor/layout differs: " + name)


def summarize_profile(detail, parser):
    text = detail["raw_text"]
    raw = text.encode("utf-8")
    if len(raw) != detail["bytes"] or hashlib.sha256(raw).hexdigest() != detail["raw_sha256"]:
        raise ValueError("SDK profile raw text bytes/SHA differ")
    rows = parser.parse_profile(text)
    io = [row for row in rows if row["type"] in ("InputOperator", "OutputOperator")]
    internal = [row for row in rows if row not in io]
    cpu = [row for row in internal if row["target"] == "CPU"]
    transposes = [row for row in cpu if row["type"] == "Transpose"]
    return dict(raw_sha256=detail["raw_sha256"], bytes=len(raw), rows=len(rows),
                internal_target_counts=dict(sorted(Counter(row["target"] for row in internal).items())),
                internal_CPU_types=dict(sorted(Counter(row["type"] for row in cpu).items())),
                CPU_IO_rows=len([row for row in io if row["target"] == "CPU"]),
                target_elapsed_us={target: sum(row["elapsed_us"] for row in rows if row["target"] == target)
                                   for target in sorted({row["target"] for row in rows})},
                CPU_IO_elapsed_us=sum(row["elapsed_us"] for row in io if row["target"] == "CPU"),
                internal_CPU_transpose_elapsed_us=sum(row["elapsed_us"] for row in transposes),
                operator_elapsed_us=sum(row["elapsed_us"] for row in rows), CPU_transposes=transposes,
                all_internal_ops_NPU=bool(internal) and all(row["target"] == "NPU" for row in internal),
                largest_internal_ops=sorted(internal, key=lambda row: -row["elapsed_us"])[:5])


def audit(args):
    checks = []
    result = dict(schema="v32_independent_saved_encoder_diagnostic_review", status="failed",
                  analysis_script_sha256=sha(__file__), candidate_accepted=False,
                  historical_FP32_gate_passed=False, checks=checks,
                  scope=dict(saved_arrays_only=True, model_forward=False, ORT_run=False,
                             SDK_run=False, board_job=False, image_decoded=False),
                  limitations=["Audit passed means records agree; the historical strict rejection remains.",
                               "Unsaved warm outputs are checked by their recorded raw SHA metadata only.",
                               "SDK release/context destruction and board source rechecks are recorded evidence.",
                               "No correlation, learned Update, BA, valid pose, full VO or 20 Hz result.",
                               "Short AB/BA/AB calls are not a thermal-controlled or sustained benchmark.",
                               "Profile operator times are separate diagnostics and do not replace normal whole-IO timing."])

    def check(ok, name):
        checks.append(dict(name=name, passed=bool(ok)))
        if not ok:
            raise ValueError(name)

    def receipt(path, digest, role):
        check(path.is_file() and sha(path) == digest, role + " receipt SHA")
        value = json.loads(path.read_text(), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
        result.setdefault("receipts", {})[role] = dict(file=path.name, sha256=digest)
        return value

    def artifact(directory, row, role):
        item = row["artifact"]
        name = item["file"]
        check(isinstance(name, str) and Path(name).name == name, role + " simple artifact name")
        path = directory / name
        check(path.is_file() and not path.is_symlink() and path.stat().st_size == item["bytes"]
              and sha(path) == item["sha256"], role + " artifact bytes/SHA")
        with np.load(path, allow_pickle=False) as archive:
            check(set(archive.files) == set(SHAPES), role + " exact array keys")
            arrays = {name: archive[name] for name in SHAPES}
        for name, array in arrays.items():
            check(array.dtype == np.float32 and array.shape == SHAPES[name]
                  and parser.meta(array) == row["arrays"][name], role + " shape/dtype/raw SHA: " + name)
        return arrays

    def metric_equal(actual, recorded, label):
        check(actual.keys() == recorded.keys(), label + " metric fields")
        for key, value in actual.items():
            equal = (math.isclose(value, recorded[key], rel_tol=1e-10, abs_tol=1e-14)
                     if key not in ("rtol", "atol") and type(value) is float
                     and type(recorded[key]) in (float, int) else value == recorded[key])
            check(equal, label + ":" + key)

    try:
        parser = load_module(args.v26_auditor, PARSER_SHA, "_v32_profile_and_array_parser")
        compiler = load_module(args.compiler_source, COMPILER_SHA, "_v32_pure_compiler_contract")
        check(sha(args.runner_source) == args.runner_sha256, "frozen v32 runner source SHA")
        capture = receipt(args.capture_dir / "capture.json", CAPTURE_SHA, "capture")
        ort = receipt(args.ort_dir / "ort.json", ORT_SHA, "CPU_ORT")
        original_compile = receipt(args.compile_dir / "compile.json", ORIGINAL_COMPILE_SHA, "original_compile")
        compiled = receipt(args.candidate_compile_dir / "compile.json", args.candidate_compile_sha256, "candidate_compile")
        rewrite = receipt(args.rewrite_dir / "rewrite.json", REWRITE_SHA, "rewrite")
        failed_ort = receipt(args.failed_ort_dir / "ort.json", FAILED_ORT_SHA, "historical_failed_ORT")
        ancestor = compiler.validate_lineage(rewrite, failed_ort)
        compiler.validate_compilation(compiled, ancestor, expected_script_sha256=COMPILER_SHA)
        check(capture["status"] == "completed_encoder_cpu_probe_not_VO"
              and ort["status"] == "completed_CPU_ORT_gate" and ort["exact_reference_gate_passed"] is True
              and capture["state_unchanged"] is True and ort["state_before"] == capture["state_before"]
              and len(capture["state_before"]) == 94, "original weights and CPU/ORT prerequisites")
        model_records = {}
        for method, directory, comp in (("original", args.compile_dir, original_compile),
                                        ("spatial", args.candidate_compile_dir, compiled)):
            item = comp["output"]
            path = directory / Path(item["path"]).name
            check(path.is_file() and sha(path) == item["sha256"] and path.stat().st_size == item["size"],
                  method + " compiled artifact SHA/size")
            model_records[method] = dict(sha256=item["sha256"], bytes=item["size"])
        result["models"] = model_records
        result["ancestor"] = ancestor
        runs = {mode: receipt(getattr(args, mode + "_dir") / "run.json", getattr(args, mode + "_sha256"), mode)
                for mode in ("normal", "profile")}
        call_summaries, profiles, representatives = [], [], {}
        all_gates = {mode: [] for mode in runs}
        for mode, run in runs.items():
            complete = run["status"] == COMPLETE
            check(complete or run["status"] == "failed", mode + " terminal run status")
            check(run["schema"] == "v32_DPVO_encoder_RKNN_layout_diagnostic"
                  and run["script_sha256"] == args.runner_sha256
                  and run["candidate_compile_sha256"] == args.candidate_compile_sha256
                  and run["candidate_ancestor"] == ancestor
                  and run["original_parameters"] == capture["state_before"]
                  and run["inputs"] == capture["inputs"] and run["preprocessing"] == capture["preprocessing"],
                  mode + " runner/lineage/weights/input binding")
            check(run["candidate_accepted"] is False and run["historical_FP32_gate_passed"] is False
                  and run["FP32_thresholds_relaxed"] is False and run["output_clipping"] is False
                  and run["complete_VO_executed"] is False and run["realtime_validated"] is False
                  and run["collect_perf_detail"] is (mode == "profile"), mode + " rejected diagnostic scope")
            expected = schedule(mode == "profile")
            actual = [(r["frame_index"], r["method"], r["timing_role"]) for r in run["calls"]]
            check(actual == expected[:len(actual)] and len(actual) <= len(expected)
                  and (not complete or len(actual) == len(expected)), mode + " exact schedule/prefix")
            calls = run["calls"]
            check(run["sdk_run_api_calls"] == len(calls) if complete else
                  len(calls) <= run["sdk_run_api_calls"] <= len(calls) + 1, mode + " actual SDK call count")
            check(run["sources_and_inputs_unchanged"] is True
                  and run["source_hashes_after"] == run["source_pins"]
                  and {args.runner_sha256, COMPILER_SHA, parser.ABI_SHA, parser.HEADER_SHA, parser.LIBRARY_SHA,
                       CAPTURE_SHA, ORT_SHA, REWRITE_SHA, FAILED_ORT_SHA, args.candidate_compile_sha256}
                  <= set(run["source_pins"].values()), mode + " recorded before/after source pins")
            check(run["cleanup_errors"] == [] and run["affinity_restored"] is True, mode + " cleanup/affinity evidence")
            expected_methods = {"spatial"} if mode == "profile" else {"original", "spatial"}
            check(set(run["contexts"]) == expected_methods, mode + " context set")
            for method, context in run["contexts"].items():
                check(context["context_destroyed"] is True and context["core_mask"] == 7
                      and context["sdk_version"].startswith("2.3.2")
                      and context["init_flags"] == (8 if mode == "profile" else 0)
                      and context["model_sha256"] == model_records[method]["sha256"]
                      and context["model_bytes"] == model_records[method]["bytes"], mode + ":" + method + " context/model/destroy")
                validate_descriptor(context["input_attr"], 0, "normalized_bgr", (1, 3, 464, 608), is_input=True)
                check(len(context["output_attrs"]) == 2, mode + ":" + method + " two output descriptors")
                for i, (name, shape) in enumerate(SHAPES.items()):
                    validate_descriptor(context["output_attrs"][i], i, name, (shape[0], *shape[2:]), is_input=False)
            seen = {}
            for index, row in enumerate(calls):
                frame, method = row["frame_index"], row["method"]
                seed = capture["inputs"][frame]
                check(row["index"] == index and row["frame_id"] == seed["frame_id"]
                      and row["input_array"] == seed["array"], mode + " call source identity")
                times = row["timing"]
                native_format = run["contexts"][method]["input_attr"]["format"]
                check(times["input_api_format"] == native_format and times["input_api_dtype"] == "float32"
                      and times["input_logical_shape"] == [1, 3, 464, 608]
                      and times["input_buffer_shape"] == ([1, 464, 608, 3] if native_format == 1 else [1, 3, 464, 608]),
                      mode + " explicit logical/native input packing")
                check(times["synchronous"] is True and times["output_buffers_released"] is True
                      and times["output_api_want_float"] == 1 and times["input_pass_through"] == 0
                      and times["output_bytes"] == [math.prod(shape) * 4 for shape in SHAPES.values()],
                      mode + " copy/release synchronous call evidence")
                check(all(type(times[key]) in (int, float) and math.isfinite(times[key]) and times[key] >= 0
                          for key in STEPS) and row["forward_seconds"] == times["sdk_io_seconds"]
                      and row["forward_seconds"] >= sum(times[key] for key in STEPS) - 1e-6,
                      mode + " complete IO timing")
                key = (frame, method)
                expected_repeat = None if key not in seen else row["arrays"] == seen[key]
                check(row["repeat_raw_hash_equal"] is expected_repeat, mode + " repeat raw metadata claim")
                seen.setdefault(key, row["arrays"])
                if "artifact" in row:
                    values = artifact(getattr(args, mode + "_dir"), row, mode + ":call" + str(index))
                    valid = all(np.isfinite(array).all() and np.any(array != 0) for array in values.values())
                    check(row["finite_nonzero"] is bool(valid), mode + " raw finite/nonzero claim")
                    cpu_row = next(r for r in capture["calls"] if r["frame_index"] == frame)
                    cpu_values = artifact(args.capture_dir, cpu_row, "original CPU frame" + str(frame))
                    ort_values = artifact(args.ort_dir, ort["pairs"][frame], "original ORT frame" + str(frame))
                    metrics = {name: compare(value, cpu_values[name], parser) for name, value in values.items()}
                    ort_metrics = {name: compare(value, ort_values[name], parser) for name, value in values.items()}
                    if valid:
                        for name in SHAPES:
                            metric_equal(metrics[name], row["versus_CPU_reference"][name], mode + " recomputed CPU " + name)
                            metric_equal(ort_metrics[name], row["versus_CPU_ORT"][name], mode + " recomputed ORT " + name)
                    else:
                        check(not complete and index == len(calls) - 1, "Invalid raw output must terminate, not be accepted")
                    call_summaries.append(dict(mode=mode, call_index=index, frame_index=frame, method=method,
                                               arrays=row["arrays"], artifact=row["artifact"], finite_nonzero=bool(valid),
                                               versus_CPU_reference=metrics, versus_CPU_ORT=ort_metrics,
                                               forward_seconds=row["forward_seconds"], timing=times))
                    representatives[(mode, frame, method)] = values
                    all_gates[mode].extend(metric["gate_passed"] for metric in metrics.values())
                else:
                    check(expected_repeat is not None, "First method/frame output must be saved")
                    first = next(r for r in call_summaries if r["mode"] == mode and r["frame_index"] == frame and r["method"] == method)
                    if expected_repeat:
                        check(row["finite_nonzero"] == first["finite_nonzero"], "Repeat finite claim matches saved raw hash")
                        for name in SHAPES:
                            metric_equal(first["versus_CPU_reference"][name], row["versus_CPU_reference"][name], "metadata-only repeat CPU " + name)
                            metric_equal(first["versus_CPU_ORT"][name], row["versus_CPU_ORT"][name], "metadata-only repeat ORT " + name)
                        all_gates[mode].extend(m["gate_passed"] for m in first["versus_CPU_reference"].values())
                    else:
                        check(False, "Changed unsaved output cannot be independently numerically audited")
                if mode == "profile" and "perf_detail" in row:
                    profiles.append(dict(frame_index=frame, **summarize_profile(row["perf_detail"], parser)))
                else:
                    check(mode == "profile" or "perf_detail" not in row, "Normal timing has no SDK profile collection")
            if complete:
                check(run["actual_strict_FP32_gate_passed"] is all(all_gates[mode]), mode + " recorded overall strict gate")
                check(run["timing"] == summary(calls), mode + " exact two-warm timing summary")
            result.setdefault("execution", {})[mode] = dict(status=run["status"], error=run.get("error"), calls=len(calls),
                        complete_schedule=complete, contexts=run["contexts"], cleanup_errors=run["cleanup_errors"],
                        affinity_restored=run["affinity_restored"], source_rechecks_agree=True,
                        driver_seconds=run["actual_driver_seconds"], memory_final=run.get("memory_final"),
                        timing=summary(calls), timing_summary_complete=complete)
            log_path = getattr(args, mode + "_log", None) or getattr(args, mode + "_dir").parent / (getattr(args, mode + "_dir").name + ".log")
            if log_path.is_file():
                errors = [line for line in log_path.read_text(errors="replace").splitlines() if re.match(r"^\s*E\s+RKNN", line)]
                result.setdefault("SDK_logs", {})[mode] = dict(file=log_path.name, sha256=sha(log_path), E_RKNN_lines=errors)
                check(not complete or not errors, mode + " no SDK error hidden behind completed status")
            else:
                result.setdefault("SDK_logs", {})[mode] = dict(checked=False, reason="Companion log missing")
        ab_metrics = []
        for frame in range(3):
            a = representatives.get(("normal", frame, "original"))
            b = representatives.get(("normal", frame, "spatial"))
            if a is not None and b is not None:
                ab_metrics.append(dict(frame_index=frame, outputs={name: compare(b[name], a[name], parser) for name in SHAPES}))
            normal = representatives.get(("normal", frame, "spatial"))
            profiled = representatives.get(("profile", frame, "spatial"))
            if normal is not None and profiled is not None:
                result.setdefault("normal_profile_saved_output_comparison", []).append(dict(frame_index=frame,
                    outputs={name: compare(profiled[name], normal[name], parser) for name in SHAPES}))
        baseline_path = getattr(args, "baseline_profile_dir", None)
        if baseline_path is not None:
            original_profile = receipt(baseline_path / "run.json", args.baseline_profile_sha256, "historical_original_profile")
            check(original_profile["script_sha256"] == parser.DRIVER_SHA
                  and original_profile["collect_perf_detail"] is True, "historical original profile source/flags")
            result["historical_original_profile"] = [dict(frame_index=row["frame_index"], **summarize_profile(row["perf_detail"], parser))
                                                     for row in original_profile["calls"]]
            result["profile_comparison_limitation"] = "Original profile is historical, not a same-process/time hardware control."
        else:
            result["historical_original_profile"] = dict(status="unknown", reason="Not supplied; no layout-change numbers inferred")
        result.update(status="passed", saved_calls=call_summaries, profile=profiles, original_vs_spatial=ab_metrics,
                      independently_rehashed_saved_NPZ=len(call_summaries),
                      strict_FP32_gate_passed={mode: all(values) if values else None for mode, values in all_gates.items()},
                      recorded_repeat_scope="Four unsaved frame0 warm calls use recorded raw-array hashes; first outputs and all profile outputs are rehashed.")
    except Exception as exc:
        result["error"] = dict(type=type(exc).__name__, message=str(exc))
    result["check_count"] = len(checks)
    result["failed_check_count"] = sum(not row["passed"] for row in checks)
    return result


def main(argv=None):
    value = argparse.ArgumentParser(description=__doc__)
    for name in ("normal-dir", "profile-dir", "capture-dir", "ort-dir", "compile-dir", "candidate-compile-dir",
                 "rewrite-dir", "failed-ort-dir", "runner-source", "compiler-source", "v26-auditor"):
        value.add_argument("--" + name, type=Path, required=True)
    for name in ("normal-sha256", "profile-sha256", "candidate-compile-sha256", "runner-sha256"):
        value.add_argument("--" + name, required=True)
    for name in ("normal-log", "profile-log", "baseline-profile-dir"):
        value.add_argument("--" + name, type=Path)
    value.add_argument("--baseline-profile-sha256")
    value.add_argument("--output", type=Path, required=True)
    args = value.parse_args(argv)
    if args.output.exists() or bool(args.baseline_profile_dir) != bool(args.baseline_profile_sha256):
        value.error("Output must be new; optional baseline profile requires directory and SHA")
    report = audit(args)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=report["status"], candidate_accepted=False, checks=report["check_count"],
                         failures=report["failed_check_count"], error=report.get("error"), receipt_sha256=sha(args.output))))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
