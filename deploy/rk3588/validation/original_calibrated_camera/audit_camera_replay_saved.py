#!/usr/bin/env python3
"""Audit saved original VGGT INT8-camera timestamp replay; no model or SDK.

Counts and timing statistics come from the pinned completed replay, not from
the previous v80 result. The original 798-frame source and 13-context contract
remain fixed. This is an artifact consistency audit, not a quality acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


ORIGINAL_ENTRY_SHA256 = "ed7ae7ba73d4e54d17a1be9bcc0c9213776371cbe90c5921210779ca6d252d3e"
AGGREGATOR_ADAPTER_SHA256 = "f3f106034c6621a1e3ed07bdb265107b8522103f7d186ba4bbe0d0cefda16f2f"
AGGREGATOR_MANIFEST_SHA256 = "2d322f3afed5af4ab510a879960b941118c8a9ec585a7c4bd9bf06269fba1d97"
CAMERA_ADAPTER_SHA256 = "0a3db35965ffc7d17ff0855a5040ecb0f22f8be23a88605ec1a1e7793ad236ec"
CAMERA_COMPILE_SHA256 = "4d4094c76570dd497af3553b6dfcbc768ebd65147717db7aca8b20bc4c0efc89"
CAMERA_MODEL_SHA256 = "8bb5b37de0b552b78a1f6eec582c70e6d4e8622e363815a2a2287c76ecbd56ac"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(actual, expected, label, tolerance=1e-9):
    require(np.isfinite(actual) and abs(actual - expected) <= tolerance,
            f"{label}: {actual!r} != {expected!r}")


def vector(updates, name):
    values = np.asarray([row[name] for row in updates], dtype=np.float64)
    require(np.isfinite(values).all(), f"nonfinite update field: {name}")
    return values


def audit(replay_dir, control_path, wrapper_path, expected_results_sha256):
    inputs = {
        "results.json": replay_dir / "results.json",
        "map.json": replay_dir / "map.json",
        "poses.tum": replay_dir / "poses.tum",
        "control.json": control_path,
        "wrapper.json": wrapper_path,
    }
    inventory = {}

    def record(label, path):
        require(path.is_file() and not path.is_symlink(), f"not a regular saved file: {path}")
        inventory[label] = {"bytes": path.stat().st_size, "sha256": sha256(path)}

    for label, path in inputs.items():
        record(label, path)
    require(re.fullmatch(r"[0-9a-f]{64}", expected_results_sha256) is not None and
            inventory["results.json"]["sha256"] == expected_results_sha256,
            "results do not match the explicitly pinned replay SHA256")
    result = read_json(inputs["results.json"])
    mapping = read_json(inputs["map.json"])
    control = read_json(control_path)
    wrapper = read_json(wrapper_path)
    updates, source = result["updates"], result["source"]
    count = len(updates)
    require(result["status"] == "completed" and result["model"] == "original_VGGT",
            "replay did not complete with the original model")
    require(count == result["fresh_pose_count"] and count >= 2 and result["max_loops"] == 0,
            "fresh update count or loop contract changed")
    require(source["source"] == "original_timestamp_rgb_replay" and
            source["queue_capacity"] == 1 and source["error"] is None,
            "source contract or source completion failed")
    require(source["original_frames"] == result["original_frames"] == 798 and
            source["delivered"] == count + 1 and
            all(type(source[key]) is int and source[key] >= 0 for key in
                ("generated", "skipped_before_decode", "delivered", "overwritten")),
            "original source count or nonnegative integer accounting failed")
    require(source["original_frames"] == source["generated"] + source["skipped_before_decode"] and
            source["generated"] == source["delivered"] + source["overwritten"],
            "source frame conservation failed")
    sequence = vector(updates, "sequence")
    timestamps = vector(updates, "timestamp")
    captured = vector(updates, "captured_monotonic")
    published = vector(updates, "published_monotonic")
    latency = vector(updates, "input_to_pose_seconds")
    require(np.equal(sequence, np.floor(sequence)).all() and
            sequence[0] >= 1 and sequence[-1] == 797 and (np.diff(sequence) > 0).all(),
            "fresh source sequences are not unique and strictly increasing")
    require((np.diff(timestamps) > 0).all() and timestamps[0] > result["seed_timestamp"] and
            (np.diff(captured) > 0).all() and (np.diff(published) > 0).all(),
            "fresh clocks are not strictly increasing")
    require((latency >= 0).all() and np.array_equal(latency, published - captured),
            "input-to-pose is not exactly publication minus capture")
    phases = {name: vector(updates, name) for name in
              ("arrival_to_decode_seconds", "decode_seconds", "queue_seconds", "prediction_seconds",
               "full_model_seconds", "SL4_and_points_seconds", "publication_seconds", "processing_seconds")}
    require(all((values >= 0).all() for values in phases.values()), "negative saved phase duration")
    processing = phases["prediction_seconds"] + phases["SL4_and_points_seconds"] + phases["publication_seconds"]
    total = phases["arrival_to_decode_seconds"] + phases["decode_seconds"] + phases["queue_seconds"] + processing
    require(np.allclose(processing, phases["processing_seconds"], rtol=0, atol=1e-9) and
            np.allclose(total, latency, rtol=0, atol=1e-9) and
            (phases["full_model_seconds"] <= phases["prediction_seconds"] + 1e-9).all(),
            "saved latency phase equations or model-within-prediction scope failed")
    # Match timestamp_replay.py's actual left-associative clock expression.
    deadlines = np.asarray([source["start_monotonic"] + timestamp - result["seed_timestamp"]
                            for timestamp in timestamps])
    require(np.array_equal(captured, deadlines), "capture does not match original source deadline")
    require(source["last_capture_monotonic"] == captured[-1] and
            captured[-1] <= source["end_monotonic"] <= published[-1], "source EOF/capture clock ordering failed")
    anchors = np.r_[result["seed_timestamp"], timestamps[:-1]]
    require(np.array_equal(vector(updates, "anchor_timestamp"), anchors), "anchor chain changed")
    require(all(row["revision"] == i + 1 and row["graph_nodes"] == 2 * (i + 1)
                for i, row in enumerate(updates)), "publication revision or graph growth changed")
    wall_seconds = float(published[-1] - source["start_monotonic"])
    hz = count / wall_seconds
    p95_latency = float(np.percentile(latency, 95, method="linear"))
    gaps = np.diff(published)
    p95_gap, max_gap = float(np.percentile(gaps, 95, method="linear")), float(gaps.max())
    close(wall_seconds, result["fresh_wall_seconds_including_EOF_drain"], "fresh wall seconds")
    close(hz, result["fresh_pose_hz"], "fresh Hz", 1e-12)
    close(p95_latency, result["input_to_pose_p95_seconds"], "input-to-pose P95")

    poses = np.loadtxt(inputs["poses.tum"], dtype=np.float64, ndmin=2)
    require(poses.shape == (count + 1, 8) and np.isfinite(poses).all(), "invalid TUM pose shape/value")
    require(np.array_equal(poses[:, 0], np.r_[result["seed_timestamp"], timestamps]) and
            (np.diff(poses[:, 0]) > 0).all(), "TUM timestamps do not match seed plus fresh updates")
    quaternion_error = float(np.max(np.abs(np.linalg.norm(poses[:, 4:8], axis=1) - 1.0)))
    require(quaternion_error <= 1e-6, "saved TUM quaternion norms are invalid")
    require(mapping["format"] == "vggt_camera_local_dense_chunks_with_SL4_graph" and
            mapping["revision"] == count and len(mapping["chunks"]) == count and
            set(mapping["nodes"]) == {str(i) for i in range(2 * count)}, "map format or graph count changed")
    for key, matrix in mapping["nodes"].items():
        matrix = np.asarray(matrix, dtype=np.float64)
        require(matrix.shape == (4, 4) and np.isfinite(matrix).all(), f"invalid SL4 node {key}")
    point_counts, npz_paths = [], []
    array_count = 0
    for index, chunk in enumerate(mapping["chunks"]):
        require(chunk["submap_id"] == 2 * index and chunk["nodes"] == [2 * index, 2 * index + 1],
                "original submap/node layout changed")
        expected_name = f"map_chunks/submap_{2 * index:06d}.npz"
        require(chunk["path"] == expected_name, "unexpected map chunk path")
        path = replay_dir / expected_name
        npz_paths.append(path)
        record(expected_name, path)
        with np.load(path, allow_pickle=False) as saved:
            require(set(saved.files) == {"points_0", "colors_0", "points_1", "colors_1"},
                    f"unexpected arrays in {expected_name}")
            require(len(chunk["point_counts"]) == 2, "unexpected point-count metadata")
            for frame in (0, 1):
                size = chunk["point_counts"][frame]
                require(isinstance(size, int) and size > 0, "invalid saved point count")
                point_counts.append(size)
                for prefix, dtype in (("points", np.float32), ("colors", np.uint8)):
                    value = saved[f"{prefix}_{frame}"]
                    require(value.shape == (size, 3) and value.dtype == np.dtype(dtype) and
                            np.isfinite(value).all(), f"invalid {prefix}_{frame} in {expected_name}")
                    array_count += 1
    require(set(replay_dir.glob("map_chunks/*.npz")) == set(npz_paths), "extra or missing saved map chunks")

    contexts, runs = {}, {}
    for name in ("rknn", "dino_rknn", "camera_rknn", "depth_rknn"):
        metrics = result[name]
        require(metrics["closed"] and not metrics["failed"] and metrics["live_contexts"] == 0 and
                not metrics["cleanup_errors"] and metrics["contexts_created"] == metrics["contexts_destroyed"],
                f"SDK context cleanup failed: {name}")
        contexts[name] = metrics["contexts_created"]
        if name == "camera_rknn":
            runs[name] = metrics["npu_run_api_call_count"]
        else:
            require(metrics["sdk_run_attempts"] == metrics["sdk_runs_succeeded"], f"SDK run failed: {name}")
            runs[name] = metrics["sdk_runs_succeeded"]
    require(contexts == {"rknn": 4, "dino_rknn": 6, "camera_rknn": 2, "depth_rknn": 1} and
            runs == {"rknn": count * 4, "dino_rknn": (count + 1) * 6,
                     "camera_rknn": count * 4, "depth_rknn": count * 2},
            "SDK accounting changed")
    retained = result["rknn"]
    require(retained["outputs_acquired"] == retained["outputs_released"] == count * 9 and
            retained["completed_blocks"] == count * 48 and retained["completed_aggregator_calls"] == count and
            retained["completed_groups"] == count * 4 and retained["completed_pairs"] == count * 24 and
            retained["original_forward_restored"] and retained["all_original_48_blocks_retained"],
            "retained aggregator accounting or cleanup changed")
    require(result["dino_rknn"]["executed_original_layer_frames"] == (count + 1) * 24 and
            result["dino_rknn"]["completed_feature_calls"] == count and
            result["dino_rknn"]["completed_frame_groups"] == (count + 1) * 6 and
            result["dino_rknn"]["batch1_feature_calls"] == count - 1 and
            result["dino_rknn"]["batch2_feature_calls"] == 1 and
            result["camera_rknn"]["completed_head_calls"] == count and
            result["camera_rknn"]["completed_refinements"] == count * 4 and
            result["depth_rknn"]["completed_frames"] == count * 2, "original runtime frame accounting changed")
    session, cache = result["immutable_session"], result["cache"]
    require(session["closed"] and not session["failed"] and session["full_seal_validated"] and
            session["final_full_validated"] and session["invalidations"] == 0 and not session["cleanup_errors"] and
            cache["closed"] and cache["restored"] and cache["cache_entries"] == cache["cache_bytes"] == 0 and
            not cache["restore_errors"] and cache["last_error"] is None and not result["cleanup_errors"],
            "immutable session/cache cleanup failed")
    require(cache["requests"] == cache["original_dino_calls"] == cache["completed_original_dino_calls"] == count
            and cache["input_frames"] == count * 2 and cache["hit_frames"] == count - 1
            and cache["computed_frames"] == cache["miss_frames"] == count + 1 and
            cache["errors"] == cache["invalidations"] == cache["loop_bypass_calls"] == cache["loop_computed_frames"] == 0,
            "original previous-tail DINO cache conservation failed")

    observations = control["observations"]
    require(len(observations) > 0 and control["child_returncode"] == 0 and not control["restore_errors"] and
            control["requested_frequency_hz"] == 1_000_000_000 and
            control["timeout_seconds"] <= 180 and control["temperature_limit_c"] <= 75 and
            all(row["actual_frequency_hz"] == 1_000_000_000 for row in observations) and
            all(row["platform"]["npu"]["cur_freq"] == 1_000_000_000 for row in updates),
            "controller or saved update frequency observations failed")
    expected_baseline = {"governor": "userspace", "cur_freq": "300000000", "min_freq": "300000000",
                         "max_freq": "1000000000", "userspace/set_freq": "300000000"}
    require(control["baseline"] == control["restored"] == expected_baseline and
            control["frequency_governor_limits_restored"], "controller failed exact 300 MHz restoration")
    temperatures = np.asarray([value for row in observations for value in row["temperatures_c"].values()],
                              dtype=np.float64)
    require(temperatures.size and np.isfinite(temperatures).all() and
            float(temperatures.max()) < control["temperature_limit_c"], "temperature observation contract failed")
    require(wrapper["installer_calls"] == 1 and wrapper["installer_restored"] and wrapper["sources_unchanged"] and
            wrapper["camera_installer_calls"] == 1 and wrapper["camera_resources_closed"] and
            wrapper["camera_installation_failures"] == [] and wrapper["camera_rknn"] == [result["camera_rknn"]] and
            not wrapper["cleanup_errors"] and wrapper["original_report_status"] == "completed" and
            wrapper["status"] == "completed_original_timestamp_replay_pending_quality_review" and
            wrapper["annotated_report_sha256"] == inventory["results.json"]["sha256"] and
            wrapper["aggregator_metrics_after_original_cleanup"] == [retained] and
            wrapper["entrypoint_sha256"] == result["script_sha256"] == ORIGINAL_ENTRY_SHA256 and
            wrapper["adapter_source_sha256"] == result["runtime_source_sha256"]["private/rknn_calibrated_retained.py"] == AGGREGATOR_ADAPTER_SHA256 and
            wrapper["quantized_manifest_sha256"] == result["quantized_manifest_sha256"] == AGGREGATOR_MANIFEST_SHA256 and
            wrapper["camera_adapter_source_sha256"] == result["camera_adapter_source_sha256"] ==
                result["runtime_source_sha256"]["private/rknn_calibrated_camera.py"] == CAMERA_ADAPTER_SHA256 and
            wrapper["camera_compile_receipt_sha256"] == result["camera_compile_receipt_sha256"] == CAMERA_COMPILE_SHA256,
            "wrapper receipt mismatch")
    for path, digest in wrapper["frozen_runtime_source_sha256"].items():
        short_path = path.removeprefix("vggt_slam/")
        if short_path in result["runtime_source_sha256"]:
            require(result["runtime_source_sha256"][short_path] == digest, f"saved source digest mismatch: {path}")
    require(wrapper["original_model_only"] and wrapper["single_original_model_instance"] and
            wrapper["source_queue_and_pose_clocks_unchanged"] and
            wrapper["DINO_depth_and_SL4_unchanged"] and result["DINO_depth_and_SL4_unchanged"] and
            wrapper["original_aggregator_ONNX_unchanged"] and result["original_aggregator_ONNX_unchanged"] and
            result["requested_aggregator_quantized_dtype"] == wrapper["requested_aggregator_quantized_dtype"] == "w8a8" and
            result["calibrated_retained_aggregator"] and result["calibrated_camera_head"] and
            wrapper["original_camera_ONNX_unchanged"] and result["original_camera_ONNX_unchanged"] and
            wrapper["compiled_camera_internal_precision_changed"] and result["compiled_camera_internal_precision_changed"] and
            wrapper["requested_camera_quantized_dtype"] == result["requested_camera_quantized_dtype"] == "w8a8" and
            wrapper["half_IO_for_DINO_unchanged"] and wrapper["aggregator_half_IO_enabled"] is False,
            "original replay or quantization provenance changed")
    camera = result["camera_rknn"]
    require(camera["calibrated_camera"] and camera["requested_quantized_dtype"] == "w8a8" and
            camera["context_init_attempts"] == 2 and camera["original_ONNX_graph_unchanged"] and
            camera["original_four_iterations_FP32_accumulation_and_activation_unchanged"] and
            camera["API_FP32_conversion_not_internal_FP32"] and
            camera["compile_receipt_sha256"] == CAMERA_COMPILE_SHA256 and
            camera["model_sha256"] == CAMERA_MODEL_SHA256 and camera["verified_original_parameters"] == 69,
            "calibrated original CameraHead provenance changed")
    descriptors = [*camera["input_attrs"], camera["output_attr"]]
    expected_io = [("camera_tokens", [1, 2, 2048]), ("conditioning_pose", [1, 2, 9]), ("pose_delta", [1, 2, 9])]
    require(len(descriptors) == len(expected_io), "unexpected camera logical I/O count")
    for row, (name, shape) in zip(descriptors, expected_io):
        require(row["name"] == name and row["shape"] == shape and row["n_elems"] == int(np.prod(shape)) and
                row["type"] == 2 and row["quantization_type"] == 2 and row["format"] == 3 and
                type(row["zero_point"]) is int and -128 <= row["zero_point"] <= 127 and
                np.isfinite(row["scale"]) and row["scale"] > 0 and row["pass_through"] == 0,
                "camera logical affine INT8 descriptor differs")
    for flag in ("candidate_accepted", "trajectory_quality_validated", "real_time_validated"):
        require(result[flag] is False and wrapper[flag] is False, f"unexpected acceptance claim: {flag}")
    require(result["physical_sensor_latency_validated"] is False and result["eight_GB_hardware_validated"] is False and
            result["short_replay_not_30min"] is True, "saved validation scope changed")
    # Check all inputs again after reading; the audit writes only its separate report.
    for label, path in {**inputs, **{p.relative_to(replay_dir).as_posix(): p for p in npz_paths}}.items():
        require(inventory[label]["sha256"] == sha256(path) and inventory[label]["bytes"] == path.stat().st_size,
                f"input changed during saved-file audit: {label}")
    inventory_digest = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "schema": "v82_original_VGGT_INT8_camera_saved_timestamp_replay_audit",
        "status": "passed_saved_file_consistency_checks",
        "audit_scope": "Saved JSON/TUM/NumPy only; no model, SDK, or board execution",
        "fresh_pose_count": count, "fresh_wall_seconds_including_EOF_drain": wall_seconds, "fresh_pose_hz": hz,
        "input_to_pose_seconds": {"p95": p95_latency, "maximum": float(latency.max()),
                                  "exact_published_minus_captured_for_all_updates": True},
        "publication_gaps_seconds": {"count": len(gaps), "p95": p95_gap, "maximum": max_gap,
                                     "above_one_second": int((gaps > 1.0).sum()),
                                     "above_50ms": int((gaps > 0.05).sum())},
        "saved_phase_seconds": {key: {"mean": float(np.mean(values)), "p95": float(np.percentile(values, 95)),
                                       "maximum": float(values.max())} for key, values in phases.items()},
        "saved_latency_phase_equations_passed": True,
        "startup_seconds": result["startup_seconds"], "peak_RSS_MiB": result["peak_rss_mib"],
        "fresh_sequence_and_timestamp_unique_strictly_increasing": True,
        "source": {key: source[key] for key in ("queue_capacity", "original_frames", "generated", "delivered",
                                                "overwritten", "skipped_before_decode")},
        "seed_delivered_once": True, "source_frame_conservation_passed": True,
        "original_timestamp_deadline_matches_all_captures": True,
        "EOF_drain_seconds": float(published[-1] - source["end_monotonic"]),
        "poses": {"rows": len(poses), "columns": poses.shape[1], "all_finite": True,
                  "timestamps_match_seed_and_updates": True, "maximum_quaternion_norm_error": quaternion_error},
        "map": {"chunks": len(npz_paths), "SL4_nodes": len(mapping["nodes"]), "SL4_shape": [4, 4],
                "arrays_checked": array_count, "point_sets": len(point_counts), "points_total": sum(point_counts),
                "point_count_minimum": min(point_counts), "point_count_maximum": max(point_counts),
                "points_dtype": "float32", "colors_dtype": "uint8", "all_shapes_and_values_checked": True,
                "units": mapping["units"], "metric_scale_validated": mapping["metric_scale_validated"]},
        "runtime": {"contexts_created_and_destroyed": contexts, "total_contexts": sum(contexts.values()),
                    "SDK_run_count_by_backend": runs, "total_SDK_runs": sum(runs.values()),
                    "live_contexts": 0, "retained_outputs_acquired_and_released": retained["outputs_acquired"],
                    "other_backends_output_acquire_release_totals": "not recorded; not independently asserted",
                    "cleanup_errors": [], "immutable_session_and_cache_closed": True,
                    "wrapper_installer_restored": True},
        "control": {"observations": len(observations), "all_observed_frequencies_hz": 1_000_000_000,
                    "saved_update_frequency_observations": len(updates),
                    "maximum_observed_temperature_c": float(temperatures.max()),
                    "exact_baseline_state_restored": control["restored"], "restore_errors": [],
                    "frequency_scope": "saved controller and per-update observations, not continuous telemetry"},
        "provenance": {"original_model_only": True, "original_aggregator_ONNX_unchanged": True,
                    "DINO_depth_and_SL4_unchanged": True, "requested_aggregator_quantized_dtype": "w8a8",
                       "all_internal_operators_INT8_verified": retained["all_internal_operators_INT8_verified"],
                    "original_camera_ONNX_unchanged": True, "compiled_camera_internal_precision_changed": True,
                    "original_four_iterations_FP32_accumulation_and_activation_unchanged": True,
                    "requested_camera_quantized_dtype": "w8a8",
                    "camera_all_internal_operators_INT8_verified": camera["all_internal_operators_INT8_verified"],
                    "camera_adapter_source_sha256": wrapper["camera_adapter_source_sha256"],
                    "camera_compile_receipt_sha256": wrapper["camera_compile_receipt_sha256"],
                    "camera_logical_IO_descriptors": descriptors,
                       "entrypoint_sha256": wrapper["entrypoint_sha256"],
                       "adapter_source_sha256": wrapper["adapter_source_sha256"],
                       "quantized_manifest_sha256": wrapper["quantized_manifest_sha256"]},
        "acceptance": {"candidate_accepted": False, "trajectory_quality_validated": False,
                       "real_time_validated": False, "physical_sensor_latency_validated": False,
                       "eight_GB_hardware_validated": False, "thirty_minute_run_validated": False,
                       "average_fresh_pose_rate_at_least_1Hz": hz >= 1.0,
                       "all_publication_gaps_at_most_1s": bool((gaps <= 1.0).all()),
                       "average_fresh_pose_rate_at_least_20Hz": hz >= 20.0,
                       "complete_input_to_pose_P95_at_most_100ms": p95_latency <= 0.1,
                       "rate_and_P95_targets_both_met": bool(hz >= 20.0 and p95_latency <= 0.1),
                       "limitations": "Short recorded-dataset consistency audit; map finiteness is not trajectory/quantization accuracy"},
        "files": inventory, "input_inventory_sha256": inventory_digest, "all_audited_input_files_unchanged": True,
    }


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--results-sha256", required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=root / "camera-replay-saved-audit.json")
    args = parser.parse_args()
    replay_dir, output = args.replay_dir.resolve(), args.output.resolve()
    require(not output.is_relative_to(replay_dir) and
            output not in (args.control.resolve(), args.wrapper.resolve(), Path(__file__).resolve()),
            "audit report must be a separate file outside the saved replay inputs")
    require(not output.exists(), "Use a new saved audit report; do not overwrite existing evidence")
    report = audit(replay_dir, args.control.resolve(), args.wrapper.resolve(), args.results_sha256)
    report["audit_script_sha256"] = sha256(Path(__file__).resolve())
    output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output), "sha256": sha256(output),
                      "script_sha256": report["audit_script_sha256"], "fresh_pose_hz": report["fresh_pose_hz"],
                      "input_to_pose_p95_seconds": report["input_to_pose_seconds"]["p95"],
                      "map_chunks": report["map"]["chunks"], "points_total": report["map"]["points_total"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
