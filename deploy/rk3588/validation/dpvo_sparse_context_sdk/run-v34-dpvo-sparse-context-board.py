#!/usr/bin/env python3
"""RK3588 dense/sparse complete SDK-I/O comparison; no CPU neural fallback.

DPVO MIT component research. No Update, correlation, BA, VO or default change.
"""
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np

BASE = Path("/mnt/vggt-data/vggt-slam2/experiments")
PARENT_SHA = "895ac2902a66c68048c03e98f353300c7065e7568047479db4fee2f9a80a33d3"
COMPILER_SHA = "26e639520692619ca7d69916d11d81237c535ee31f3ea9aa848101e4742744fa"
REFERENCE_SHA = "24aef39822c86ae60e1cb6306d2d2966fa8a440320c4363f18c5db938ac4e168"
DENSE_COMPILERS = {"original": "a40bf09e53ef9b378db4781be18f9e53f824afcc809c9009e0a65f4f0f91b4e5",
                   "spatial": "364d3493f9fe13789b449c0836c37d0eedca71cdbde71e10e28723ec88d1b003"}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def board_guard():
    require(platform.machine() == "aarch64" and platform.system() == "Linux" and
            b"rockchip,rk3588" in Path("/proc/device-tree/compatible").read_bytes(), "RK3588 board execution only")


def schedule(profile):
    if profile:
        return [(i, "sparse", "profile") for i in range(3)]
    return [(0, "dense", "first"), (0, "sparse", "first"), (0, "sparse", "warm"), (0, "dense", "warm"),
            (0, "dense", "warm"), (0, "sparse", "warm"), (1, "sparse", "other_frame"), (1, "dense", "other_frame"),
            (2, "dense", "other_frame"), (2, "sparse", "other_frame")]


def gather(imap, indices):
    y, x = np.divmod(indices, imap.shape[-1])
    return np.ascontiguousarray(imap[0][:, y, x].T[None])


def validate_dense_compile(value, method, expected_onnx):
    require(value["script_sha256"] == DENSE_COMPILERS[method] and value["input"]["sha256"] == expected_onnx and
            value["configuration"] == dict(target_platform="rk3588", float_dtype="float16", optimization_level=3, do_quantization=False) and
            value["released"] is True and not value.get("error") and not value.get("release_error") and
            value["versions"]["rknn-toolkit2"] == "2.3.2", "Exact measured dense compiler/config required")
    require([r["name"] for r in value["operations"]] == ["config", "load_onnx", "build", "export_rknn"] and
            all(r["code"] is None or type(r["code"]) is int and r["code"] == 0 for r in value["operations"]), "Dense compile operations")
    if method == "original":
        require(value["status"] == "compiled_not_runtime_validated", "Original dense compile status")
    else:
        require(value["status"] == "compiled_diagnostic_FP32_rejected_not_runtime_validated" and
                value["historical_FP32_gate"] == dict(rtol=1e-4, atol=1e-5, passed=False, status="rejected") and
                value["candidate_accepted"] is False and value["sources_and_inputs_unchanged"] is True and
                value["source_pins"] == value["source_hashes_after"] and value["cleanup_errors"] == [], "Preserved spatial dense rejection")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("original", "spatial"), default="original")
    for name in ("reference-dir", "rewrite-dir", "dense-compile-dir", "sparse-compile-dir", "parent-source",
                 "compiler-source", "sdk-helper", "abi-helper", "header", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("reference", "rewrite", "dense-compile", "sparse-compile", "sdk-helper"):
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--library", type=Path, default=Path("/usr/lib/librknnrt.so"))
    parser.add_argument("--collect-perf-detail", action="store_true")
    parser.add_argument("--save-full-dense-first", action="store_true",
                        help="Save first dense imap as well; other dense artifacts retain fmap/context only")
    parser.add_argument("--diagnostic-allow-fp32-rejected", action="store_true")
    args = parser.parse_args(argv)
    board_guard()  # Before imports or SDK initialization, including source loaders.
    require(args.method != "spatial" or args.diagnostic_allow_fp32_rejected, "Explicit spatial-parent diagnostic flag required")
    output = args.output_dir.resolve()
    protected = [v.resolve() for k, v in vars(args).items() if isinstance(v, Path) and k != "output_dir"]
    require(output.is_relative_to(BASE) and output != BASE and
            not any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected), "New isolated output must not overlap sources")
    output.mkdir(parents=True, exist_ok=False)
    pins, contexts, prior, abi = {}, {}, os.sched_getaffinity(0), None
    started = time.monotonic()
    report = dict(schema="v34_DPVO_sparse_context_RKNN_comparison", status="failed", script_sha256=sha(__file__),
        method=args.method, reference_sha256=args.reference_sha256, rewrite_sha256=args.rewrite_sha256,
        dense_compile_sha256=args.dense_compile_sha256, sparse_compile_sha256=args.sparse_compile_sha256,
        sdk_helper_sha256=args.sdk_helper_sha256, actual_board_execution=True, NPU_used=False, collect_perf_detail=args.collect_perf_detail,
        candidate_accepted=False, original_VGGT_default_unchanged=True, different_model_from_VGGT=True, ORB_used=False,
        added_LK_PnP_frontend=False, model_checkpoint_loaded=False, images_decoded=False, normalization_applications=0,
        Update_executed=False, correlation_executed=False, BA_executed=False, complete_VO_executed=False,
        realtime_validated=False, quality_validated=False, all_operators_on_npu_verified=False,
        FP32_thresholds_relaxed=False, output_clipping=False, calls=[], contexts={}, cleanup_errors=[], sdk_run_api_calls=0,
        timing_scope="native-layout input packing + SDK set/run/get + owned logical FP32 copies + release; dense context gather timed separately; references/hashes/saving/profile queries excluded",
        environment={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LD_LIBRARY_PATH")})

    def pin(path, expected):
        path = Path(path).resolve()
        require(sha(path) == expected, "SHA mismatch: " + path.name)
        require(str(path) not in pins or pins[str(path)] == expected, "Conflicting source SHA")
        pins[str(path)] = expected
        return path

    def artifact(directory, item):
        require(Path(item["file"]).name == item["file"], "Local simple artifact name required")
        path = pin(directory / item["file"], item["sha256"])
        require(path.stat().st_size == item["bytes"], "Artifact bytes differ")
        return path

    try:
        pin(__file__, report["script_sha256"])
        base = load(pin(args.parent_source, PARENT_SHA), "_v34_frozen_dense_RKNN")
        compiler = load(pin(args.compiler_source, COMPILER_SHA), "_v34_frozen_sparse_compiler")
        helper = load(pin(args.sdk_helper, args.sdk_helper_sha256), "_v34_sparse_SDK")
        abi = base.load_abi(args.abi_helper, pins)
        pin(args.header, base.HEADER_SHA)
        pin(args.library, base.LIBRARY_SHA)
        require({4, 5, 6, 7} <= prior, "Four A76 CPUs required")
        os.sched_setaffinity(0, {4, 5, 6, 7})
        report["cpu_affinity"] = sorted(os.sched_getaffinity(0))
        require(args.reference_sha256 == REFERENCE_SHA, "Frozen v33 board reference required")
        reference = json.loads(pin(args.reference_dir / "ort.json", args.reference_sha256).read_text())
        recipe = json.loads(pin(args.rewrite_dir / "rewrite.json", args.rewrite_sha256).read_text())
        ancestor = compiler.validate_lineage(recipe, reference, args.method, args.diagnostic_allow_fp32_rejected)
        require(args.rewrite_sha256 == compiler.MODELS[args.method]["recipe_sha"], "Exact recipe hash required")
        for path, digest in reference["source_pins"].items():
            pin(path, digest)  # Same board sources/input images; no decode or re-normalization.
        artifact(args.rewrite_dir, recipe["output"])
        dense = json.loads(pin(args.dense_compile_dir / "compile.json", args.dense_compile_sha256).read_text())
        sparse = json.loads(pin(args.sparse_compile_dir / "compile.json", args.sparse_compile_sha256).read_text())
        validate_dense_compile(dense, args.method, compiler.MODELS[args.method]["parent_sha"])
        compiler.validate_compilation(sparse, args.method, COMPILER_SHA)
        require(sparse["ancestor"] == ancestor and sparse["static_preservation"] == recipe["preservation"], "Sparse compile lineage differs")
        models = {}
        for role, directory, receipt in (("dense", args.dense_compile_dir, dense), ("sparse", args.sparse_compile_dir, sparse)):
            item = receipt["output"]
            model = pin(directory / Path(item["path"]).name, item["sha256"])
            require(model.stat().st_size == item["size"] and 0 < item["size"] <= 128 * 1024**2, "Compiled model size budget")
            models[role] = model
        rows = [r for r in reference["calls"] if r["method"] == args.method]
        require([r["frame_index"] for r in rows] == [0, 1, 2], "All three fixed inputs required")
        report.update(ancestor=ancestor, preprocessing=reference["preprocessing"], source_frames=[r["source_image"] for r in rows],
            inputs=[r["input_metadata"] for r in rows], memory_before_sdk=abi.process_memory())
        api = abi.bind_api(ct.CDLL(str(args.library.resolve())))
        for role, model in models.items():
            if args.collect_perf_detail and role != "sparse":
                continue
            record = dict(model_sha256=sha(model), model_bytes=model.stat().st_size)
            report["contexts"][role] = record
            contexts[role] = helper.Context(base, abi, api, model, role == "sparse", args.collect_perf_detail, report["cleanup_errors"], record)
        fingerprints, saved, reference_cache = {}, {}, {}
        for index, (frame, role, timing_role) in enumerate(schedule(args.collect_perf_detail)):
            row = rows[frame]
            if frame not in reference_cache:
                with np.load(artifact(args.reference_dir, row["input_artifact"]), allow_pickle=False) as loaded:
                    image, indices = loaded["normalized_bgr"], loaded["patch_indices"]
                require(base.metadata(image) == row["input_metadata"] and base.metadata(indices) == row["indices"] and
                        image.dtype == np.float32 and image.shape == base.INPUT_SHAPE and image.flags.c_contiguous,
                        "Exact once-normalized image/index bytes required")
                helper.validate_indices(indices)
                with np.load(artifact(args.reference_dir, row["artifact"]), allow_pickle=False) as loaded:
                    target = {name: loaded[name] for name in ("fmap", "context")}
                require(all(base.metadata(v) == row["arrays"][k] for k, v in target.items()), "Reference raw arrays differ")
                reference_cache[frame] = image, indices, target
            image, indices, target = reference_cache[frame]
            actual, timing = contexts[role].execute(image, indices, report)
            report["NPU_used"] = True
            finite = all(np.isfinite(v).all() and np.any(v != 0) for v in actual.values())
            gather_start = time.perf_counter()
            selected = actual if role == "sparse" else {"fmap": actual["fmap"], "context": gather(actual["imap"], indices)}
            gather_seconds = time.perf_counter() - gather_start if role == "dense" else 0.0
            record = dict(index=index, frame_index=frame, frame_id=row["frame_id"], source_image=row["source_image"],
                role=role, timing_role=timing_role, timing=timing, forward_seconds=timing["sdk_io_seconds"],
                dense_context_gather_seconds=gather_seconds, feature_payload_ready_seconds=timing["sdk_io_seconds"] + gather_seconds,
                input_array=row["input_metadata"], indices=row["indices"], finite_nonzero=bool(finite),
                arrays={k: base.metadata(v) for k, v in actual.items()}, selected_arrays={k: base.metadata(v) for k, v in selected.items()})
            report["calls"].append(record)
            key = frame, role
            record["repeat_raw_hash_equal"] = None if key not in fingerprints else record["arrays"] == fingerprints[key]
            if key not in fingerprints or not finite:
                path = output / f"call-{index:02d}-{role}-frame-{frame}.npz"
                payload = actual if role == "sparse" or not finite or args.save_full_dense_first and frame == 0 else selected
                with path.open("xb") as stream:
                    np.savez(stream, **payload)
                record["artifact"] = dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path),
                    arrays={k: base.metadata(v) for k, v in payload.items()}, includes_full_dense_imap="imap" in payload)
                saved[key] = selected
            fingerprints.setdefault(key, record["arrays"])
            require(finite, "Nonfinite or all-zero SDK output; raw arrays saved")
            record["versus_v33_CPU_ORT"] = {k: base.comparison(v, target[k]) for k, v in selected.items()}
            if (frame, "dense") in saved and (frame, "sparse") in saved:
                record["sparse_vs_dense_selected"] = {k: base.comparison(saved[(frame, "sparse")][k], saved[(frame, "dense")][k]) for k in target}
            if args.collect_perf_detail:
                record["perf_detail"] = abi.perf_detail(api, contexts[role].context)
            print(json.dumps(dict(event="sparse_context_call", index=index, frame=frame, role=role,
                seconds=record["forward_seconds"], strict=all(v["gate_passed"] for v in record["versus_v33_CPU_ORT"].values()))), flush=True)
        require(len(report["calls"]) == len(schedule(args.collect_perf_detail)), "Complete prescribed schedule required")
        report["actual_v33_CPU_ORT_strict_gate_passed"] = all(v["gate_passed"] for r in report["calls"] for v in r["versus_v33_CPU_ORT"].values())
        report["timing_summary"] = {role: dict(warm_seconds=[r["forward_seconds"] for r in report["calls"] if r["role"] == role and r["timing_role"] == "warm"],
            warm_mean_seconds=float(np.mean([r["forward_seconds"] for r in report["calls"] if r["role"] == role and r["timing_role"] == "warm"])) if not args.collect_perf_detail else None)
            for role in ("dense", "sparse")}
        report["status"] = "completed_sparse_context_SDK_comparison_not_VO"
    except Exception as exc:
        report.update(error=repr(exc), traceback=traceback.format_exc())
    finally:
        for context in contexts.values():
            context.close()
        report["source_pins"] = pins
        try:
            report["source_hashes_after"] = {path: sha(path) for path in pins}
            report["sources_and_inputs_unchanged"] = report["source_hashes_after"] == pins
            require(report["sources_and_inputs_unchanged"], "Pinned source/input changed")
        except Exception as exc:
            report["cleanup_errors"].append("source recheck: " + repr(exc))
        try:
            os.sched_setaffinity(0, prior)
            report["affinity_restored"] = os.sched_getaffinity(0) == prior
            require(report["affinity_restored"], "Affinity not restored")
        except Exception as exc:
            report["cleanup_errors"].append("affinity: " + repr(exc))
        if abi is not None:
            report["memory_final"] = abi.process_memory()
        if report["cleanup_errors"]:
            report["status"] = "failed"
        report["actual_driver_seconds"] = time.monotonic() - started
        (output / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report.get(k) for k in ("status", "error", "timing_summary")}), flush=True)
    return 0 if report["status"] == "completed_sparse_context_SDK_comparison_not_VO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
