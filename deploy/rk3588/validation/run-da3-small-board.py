#!/usr/bin/env python3
"""Isolated, board-only DA3-SMALL CPU FP32 fixed-input research probe.

No VGGT replacement, SLAM, GT conditioning, NPU, or online throughput is claimed.
Existing RGB FP32 pixels receive official ImageNet normalization exactly once.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback
import weakref

sys.dont_write_bytecode = True
from da3_probe_common import (
    CHECKPOINT_SHA256, MODEL_CONFIG_SHA256, OFFICIAL_COMMIT,
    canonical_arrays, check_import_origins, load_official_model,
    normalize_rgb_input, raw_metadata, require_board, sha256_file,
    verify_official_source, weights_metadata,
)

REFERENCE_CAPTURE_SHA256 = "8a5b9e8a7f94757fa906fbcc1b2d37ae3d4751076d69e966dd626a15a6c63598"


def rss():
    result = {"peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            result["current_rss_mib"] = int(line.split()[1]) / 1024.0
    return result


def platform_sample():
    result = {"monotonic": time.monotonic(), **rss(), "thermal": {}, "cpu_frequency_hz": {}}
    for path in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")):
        try:
            result["thermal"][path.parent.name] = int(path.read_text().strip()) / 1000.0
        except (OSError, ValueError):
            pass
    for path in sorted(Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_cur_freq")):
        try:
            result["cpu_frequency_hz"][path.parent.name] = int(path.read_text().strip()) * 1000
        except (OSError, ValueError):
            pass
    return result


def validate_input_manifest(path, expected_sha256):
    if sha256_file(path) != expected_sha256:
        raise ValueError("Input manifest SHA256 mismatch")
    data = json.loads(Path(path).read_text())
    if data.get("reference_sha256") != REFERENCE_CAPTURE_SHA256:
        raise ValueError("Expected the frozen three-pair original336 reference")
    pairs = data.get("pairs")
    if not isinstance(pairs, list) or [x.get("pair_index") for x in pairs] != [0, 1, 2]:
        raise ValueError("All three original pairs, in order 0/1/2, are required")
    import math
    for pair in pairs:
        ids = pair.get("frame_ids")
        sources = pair.get("source_images")
        if (not isinstance(ids, list) or len(ids) != 2 or
                any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in ids)
                or ids[0] >= ids[1] or not isinstance(sources, list) or len(sources) != 2):
            raise ValueError("Invalid fixed pair identities")
        for fid, source in zip(ids, sources):
            if source.get("frame_id") != fid or source.get("timestamp") != fid:
                raise ValueError("Source frame/timestamp identity mismatch")
            if sha256_file(source["path"]) != source["sha256"]:
                raise ValueError("Original source image SHA256 mismatch")
        images = Path(pair["images"])
        if images.stat().st_size != pair["bytes"] or sha256_file(images) != pair["sha256"]:
            raise ValueError("Raw RGB fixed-input NPY SHA256/size mismatch")
    return data


def save_call(output_dir, arrays, frame_ids, call_index, pair_index):
    import numpy as np
    file = f"call-{call_index:02d}-pair-{pair_index}.npz"
    values = dict(arrays, frame_ids=np.asarray(frame_ids, dtype=np.float64))
    np.savez(output_dir / file, **values)
    return {"artifact": {"file": file, "sha256": sha256_file(output_dir / file),
                         "bytes": (output_dir / file).stat().st_size},
            "arrays": {key: raw_metadata(value) for key, value in values.items()}}


def run_call(model, images, pose_capture):
    """Normal complete official core forward; hook/codec belong to this timer."""
    import torch
    pose_capture.clear()
    with torch.no_grad():
        start = time.perf_counter()
        output = model(images, extrinsics=None, intrinsics=None, export_feat_layers=[],
                       infer_gs=False, use_ray_pose=False)
        elapsed = time.perf_counter() - start
    if len(pose_capture) != 1:
        raise RuntimeError("Expected one original cam_dec invocation per forward")
    arrays = canonical_arrays(output, pose_capture[0], tuple(images.shape[-2:]))
    return elapsed, arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parent / "official")
    parser.add_argument("--source-manifest", type=Path, default=Path(__file__).resolve().parent / "manifest.json")
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", default=CHECKPOINT_SHA256)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    require_board()
    if args.output_dir.exists():
        raise ValueError("Output directory must be new")
    args.output_dir.mkdir(parents=True)
    script_dir = Path(__file__).resolve().parent
    helper = script_dir / "da3_probe_common.py"
    self_sha = sha256_file(__file__)
    helper_sha = sha256_file(helper)
    report = {"schema": "da3_small_fixed_input_cpu_probe_v24.1", "status": "running",
              "model_name": "da3-small", "official_commit": OFFICIAL_COMMIT,
              "config_sha256": MODEL_CONFIG_SHA256,
              "input_manifest_sha256": args.input_manifest_sha256,
              "source_manifest_sha256": args.source_manifest_sha256,
              "reference_capture_sha256": REFERENCE_CAPTURE_SHA256,
              "device": "cpu", "precision": "float32", "inference_on_board_only": True,
              "npu_used": False, "sdk_calls": 0, "ORB_used": False, "GT_used": False,
              "no_GT_conditions": True, "camera_conditions": {"extrinsics": None, "intrinsics": None},
              "use_ray_pose": False, "infer_gs": False, "graph_run": False, "online_pipeline_run": False,
              "quality_validated": False, "real_time_validated": False,
              "different_model_from_VGGT": True, "original_VGGT_weights_used": False,
              "source_code_sha256": {"run-da3-small-board.py": self_sha, "da3_probe_common.py": helper_sha},
              "normalization": {"method": "official InputProcessor.NORMALIZE once on existing FP32 RGB",
                                "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
                                "applications": 1, "uint8_roundtrip": False},
              "geometric_preprocessing_changed": False,
              "pose_encoding_semantics": "raw official DA3 cam_dec translation/quaternion/FOV; not VGGT codec",
              "extrinsic_semantics": "official affine_inverse(decoded c2w), world-to-camera 3x4",
              "depth_semantics": "official predicted Z-depth in each DA3 invocation's model gauge",
              "timing_semantics": {"forward_includes": ["official backbone", "complete DualDPT depth/ray heads",
                                                        "original cam_dec", "raw pose hook clone", "official camera codec"],
                                   "forward_excludes": ["imports", "source/checkpoint hashes", "weight load",
                                                        "RGB normalization", "output validation/copy/save"],
                                   "preprocessing_timed_separately": True,
                                   "not_online_FPS": True, "not_thermal_controlled_comparison": True},
              "records": [], "call_records": [], "cleanup_errors": []}
    model = None
    hook = None
    model_ref = None
    previous_affinity = set(os.sched_getaffinity(0))
    previous_threads = None
    previous_interop = None
    pose_capture = []
    try:
        requested = {4, 5, 6, 7}
        if not requested.issubset(previous_affinity):
            raise RuntimeError("The four A76 CPU cores 4/5/6/7 must be available")
        os.sched_setaffinity(0, requested)
        if set(os.sched_getaffinity(0)) != requested:
            raise RuntimeError("CPU affinity silently differs from the requested four A76 cores")
        import torch
        import numpy as np
        previous_threads = torch.get_num_threads()
        previous_interop = torch.get_num_interop_threads()
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        report["threads"] = 4
        report["interop_threads"] = 1
        report["cpu_affinity"] = sorted(requested)
        report["platform_start"] = platform_sample()
        source_pins = verify_official_source(args.source_dir, args.source_manifest, args.source_manifest_sha256)
        input_manifest = validate_input_manifest(args.input_manifest, args.input_manifest_sha256)
        report["selected_inputs"] = copy.deepcopy(input_manifest["pairs"])
        report["rectification"] = copy.deepcopy(input_manifest.get("rectification", {
            "source": "already rectified frozen original VGGT RGB input tensors",
            "reference_capture_sha256": REFERENCE_CAPTURE_SHA256,
            "performed_in_this_probe": False}))
        load_start = time.perf_counter()
        model, provenance = load_official_model(args.source_dir, args.source_manifest,
                                               args.source_manifest_sha256, args.checkpoint,
                                               args.checkpoint_sha256)
        report["model_load_seconds_including_verification"] = time.perf_counter() - load_start
        report["checkpoint"] = provenance["checkpoint"]
        report["load_provenance"] = provenance
        model_ref = weakref.ref(model)

        def capture_pose(_module, _inputs, output):
            pose_capture.append(output.detach().clone())

        hook = model.cam_dec.register_forward_hook(capture_pose)
        for pair in input_manifest["pairs"]:
            index = pair["pair_index"]
            raw = np.load(pair["images"], allow_pickle=False)
            if raw.shape != (2, 3, 252, 336) or raw.dtype != np.float32:
                raise ValueError("Expected existing RGB FP32 [2,3,252,336] pixels")
            prepare_start = time.perf_counter()
            images = normalize_rgb_input(raw)
            preprocessing_seconds = time.perf_counter() - prepare_start
            normalized_file = f"normalized-input-pair-{index}.npy"
            np.save(args.output_dir / normalized_file, images.numpy())
            input_record = raw_metadata(raw)
            normalized_record = raw_metadata(images.numpy())
            normalized_record.update({"file": normalized_file,
                                      "file_sha256": sha256_file(args.output_dir / normalized_file)})
            print(json.dumps({"pair_index": index, "event": "forward_start",
                              "shape": list(images.shape), "preprocessing_seconds": preprocessing_seconds}), flush=True)
            first_arrays = None
            calls = 4 if index == 0 else 1
            for repetition in range(calls):
                report["platform_before_forward"] = platform_sample()
                forward_seconds, arrays = run_call(model, images, pose_capture)
                call_index = len(report["call_records"])
                record = {"pair_index": index, "call_index": call_index,
                          "repetition": repetition, "timing_role": "cold" if call_index == 0 else "warm",
                          "frame_ids": list(pair["frame_ids"]), "input_tensor": input_record,
                          "normalized_input_tensor": normalized_record,
                          "preprocessing_seconds": preprocessing_seconds,
                          "forward_seconds": forward_seconds,
                          "platform_after_forward": platform_sample(),
                          **save_call(args.output_dir, arrays, pair["frame_ids"], call_index, index)}
                if first_arrays is None:
                    first_arrays = arrays
                    report["records"].append(copy.deepcopy(record))
                else:
                    record["same_pair_first_call_array_equal"] = {
                        key: bool(np.array_equal(value, first_arrays[key])) for key, value in arrays.items()}
                report["call_records"].append(record)
                print(json.dumps({"event": "forward_end", "pair_index": index,
                                  "call_index": call_index, "forward_seconds": forward_seconds,
                                  "peak_rss_mib": record["platform_after_forward"]["peak_rss_mib"]}), flush=True)
            del images, raw, arrays, first_arrays
        pose_capture.clear()
        report["weights_after"] = weights_metadata(model)
        report["weights_unchanged"] = report["weights_after"] == provenance["parameters"]
        if not report["weights_unchanged"]:
            raise RuntimeError("Original model state changed during the probe")
        report["import_origins"] = check_import_origins(args.source_dir, source_pins["source_hashes"])
        report["cold_seconds"] = report["call_records"][0]["forward_seconds"]
        report["first_pair_warm_mean_seconds"] = sum(x["forward_seconds"] for x in report["call_records"][1:4]) / 3
        report["status"] = "completed_fixed_input_diagnostic"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        if hook is not None:
            try:
                hook.remove()
                report["pose_hook_removed"] = True
            except Exception as exc:
                report["cleanup_errors"].append("pose hook: " + str(exc))
        pose_capture.clear()
        model = None
        gc.collect()
        report["model_released"] = model_ref is None or model_ref() is None
        try:
            if previous_threads is not None:
                torch.set_num_threads(previous_threads)
            # PyTorch permits changing inter-op threads only before parallel work;
            # this isolated process exits, so report the retained setting honestly.
            report["original_interop_threads"] = previous_interop
            os.sched_setaffinity(0, previous_affinity)
            report["affinity_restored"] = set(os.sched_getaffinity(0)) == previous_affinity
            if not report["affinity_restored"]:
                raise RuntimeError("Caller affinity restoration mismatch")
        except Exception as exc:
            report["cleanup_errors"].append("thread/affinity cleanup: " + str(exc))
        try:
            after = verify_official_source(args.source_dir, args.source_manifest, args.source_manifest_sha256)
            report["source_hashes_after"] = after["source_hashes"]
            report["source_unchanged"] = (sha256_file(__file__) == self_sha and sha256_file(helper) == helper_sha
                and sha256_file(args.input_manifest) == args.input_manifest_sha256
                and sha256_file(args.source_dir.parent / "model-metadata/config.json") == MODEL_CONFIG_SHA256)
            if not report["source_unchanged"]:
                raise RuntimeError("Probe/source/input-manifest changed")
        except Exception as exc:
            report["cleanup_errors"].append("final source validation: " + str(exc))
        report["platform_end"] = platform_sample()
        if report["cleanup_errors"] or not report["model_released"]:
            report["status"] = "failed"
        (args.output_dir / "capture.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["status"] == "completed_fixed_input_diagnostic" else 1


if __name__ == "__main__":
    raise SystemExit(main())
