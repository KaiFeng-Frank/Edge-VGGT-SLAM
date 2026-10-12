#!/usr/bin/env python3
"""Audit only saved v80 replay files; never import the model or RKNN SDK."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_RESULTS_SHA256 = "32e811f045cb23141d4121e0af6bde33164efea0640bb031f2cc79f566445048"


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


def audit(replay_dir, control_path, wrapper_path):
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
    require(inventory["results.json"]["sha256"] == EXPECTED_RESULTS_SHA256,
            "results are not the frozen v80 INT8 replay")
    result = read_json(inputs["results.json"])
    mapping = read_json(inputs["map.json"])
    control = read_json(control_path)
    wrapper = read_json(wrapper_path)
    updates, source = result["updates"], result["source"]
    count = len(updates)
    require(result["status"] == "completed" and result["model"] == "original_VGGT",
            "replay did not complete with the original model")
    require(count == result["fresh_pose_count"] == 30 and result["max_loops"] == 0,
            "fresh update count or loop contract changed")
    require(source["source"] == "original_timestamp_rgb_replay" and
            source["queue_capacity"] == 1 and source["error"] is None,
            "source contract or source completion failed")
    require(source["original_frames"] == result["original_frames"] == 798 and
            source["generated"] == 797 and source["skipped_before_decode"] == 1 and
            source["delivered"] == count + 1 == 31 and source["overwritten"] == 766,
            "unexpected source accounting")
    require(source["original_frames"] == source["generated"] + source["skipped_before_decode"] and
            source["generated"] == source["delivered"] + source["overwritten"],
            "source frame conservation failed")
    sequence = vector(updates, "sequence")
    timestamps = vector(updates, "timestamp")
    captured = vector(updates, "captured_monotonic")
    published = vector(updates, "published_monotonic")
    latency = vector(updates, "input_to_pose_seconds")
    require(np.equal(sequence, np.floor(sequence)).all() and
            sequence[0] == 1 and sequence[-1] == 797 and (np.diff(sequence) > 0).all(),
            "fresh source sequences are not unique and strictly increasing")
    require((np.diff(timestamps) > 0).all() and timestamps[0] > result["seed_timestamp"] and
            (np.diff(captured) > 0).all() and (np.diff(published) > 0).all(),
            "fresh clocks are not strictly increasing")
    require((latency >= 0).all() and np.array_equal(latency, published - captured),
            "input-to-pose is not exactly publication minus capture")
    # Match timestamp_replay.py's actual left-associative clock expression.
    deadlines = np.asarray([source["start_monotonic"] + timestamp - result["seed_timestamp"]
                            for timestamp in timestamps])
    require(np.array_equal(captured, deadlines), "capture does not match original source deadline")
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
    close(p95_gap, 0.9469753618061076, "publication gap P95")
    close(max_gap, 1.1136119030124974, "publication gap max")
    require(int((gaps > 1.0).sum()) == 1, "unexpected count of publication gaps above one second")

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
    require(contexts == {"rknn": 4, "dino_rknn": 6, "camera_rknn": 1, "depth_rknn": 1} and
            runs == {"rknn": 120, "dino_rknn": 186, "camera_rknn": 120, "depth_rknn": 60},
            "SDK accounting changed")
    retained = result["rknn"]
    require(retained["outputs_acquired"] == retained["outputs_released"] == 270 and
            retained["completed_blocks"] == count * 48 and retained["completed_aggregator_calls"] == count and
            retained["original_forward_restored"] and retained["all_original_48_blocks_retained"],
            "retained aggregator accounting or cleanup changed")
    require(result["dino_rknn"]["executed_original_layer_frames"] == 31 * 24 and
            result["camera_rknn"]["completed_refinements"] == count * 4 and
            result["depth_rknn"]["completed_frames"] == count * 2, "original runtime frame accounting changed")
    session, cache = result["immutable_session"], result["cache"]
    require(session["closed"] and not session["failed"] and session["full_seal_validated"] and
            session["final_full_validated"] and session["invalidations"] == 0 and not session["cleanup_errors"] and
            cache["closed"] and cache["restored"] and cache["cache_entries"] == cache["cache_bytes"] == 0 and
            not cache["restore_errors"] and cache["last_error"] is None and not result["cleanup_errors"],
            "immutable session/cache cleanup failed")

    observations = control["observations"]
    require(len(observations) == 182 and control["child_returncode"] == 0 and not control["restore_errors"] and
            control["requested_frequency_hz"] == 1_000_000_000 and
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
            not wrapper["cleanup_errors"] and wrapper["original_report_status"] == "completed" and
            wrapper["annotated_report_sha256"] == inventory["results.json"]["sha256"] and
            wrapper["aggregator_metrics_after_original_cleanup"] == [retained] and
            wrapper["entrypoint_sha256"] == result["script_sha256"] and
            wrapper["adapter_source_sha256"] == result["runtime_source_sha256"]["private/rknn_calibrated_retained.py"] and
            wrapper["quantized_manifest_sha256"] == result["quantized_manifest_sha256"], "wrapper receipt mismatch")
    for path, digest in wrapper["frozen_runtime_source_sha256"].items():
        short_path = path.removeprefix("vggt_slam/")
        if short_path in result["runtime_source_sha256"]:
            require(result["runtime_source_sha256"][short_path] == digest, f"saved source digest mismatch: {path}")
    require(wrapper["original_model_only"] and wrapper["single_original_model_instance"] and
            wrapper["source_queue_and_pose_clocks_unchanged"] and
            wrapper["DINO_camera_depth_and_SL4_unchanged"] and
            result["DINO_camera_depth_and_SL4_unchanged"] and wrapper["original_aggregator_ONNX_unchanged"] and
            result["original_aggregator_ONNX_unchanged"] and result["requested_aggregator_quantized_dtype"] == "w8a8",
            "original replay or quantization provenance changed")
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
        "schema": "v80_original_VGGT_saved_timestamp_replay_audit",
        "status": "passed_saved_file_consistency_checks",
        "audit_scope": "Saved JSON/TUM/NumPy only; no model, SDK, or board execution",
        "fresh_pose_count": count, "fresh_wall_seconds_including_EOF_drain": wall_seconds, "fresh_pose_hz": hz,
        "input_to_pose_seconds": {"p95": p95_latency, "maximum": float(latency.max()),
                                  "exact_published_minus_captured_for_all_updates": True},
        "publication_gaps_seconds": {"count": len(gaps), "p95": p95_gap, "maximum": max_gap,
                                     "above_one_second": int((gaps > 1.0).sum())},
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
                       "DINO_camera_depth_and_SL4_unchanged": True, "requested_aggregator_quantized_dtype": "w8a8",
                       "all_internal_operators_INT8_verified": retained["all_internal_operators_INT8_verified"],
                       "entrypoint_sha256": wrapper["entrypoint_sha256"],
                       "adapter_source_sha256": wrapper["adapter_source_sha256"],
                       "quantized_manifest_sha256": wrapper["quantized_manifest_sha256"]},
        "acceptance": {"candidate_accepted": False, "trajectory_quality_validated": False,
                       "real_time_validated": False, "physical_sensor_latency_validated": False,
                       "eight_GB_hardware_validated": False, "thirty_minute_run_validated": False,
                       "average_fresh_pose_rate_at_least_1Hz": hz >= 1.0,
                       "all_publication_gaps_at_most_1s": bool((gaps <= 1.0).all()),
                       "limitations": "Short recorded-dataset consistency audit; map finiteness is not trajectory/quantization accuracy"},
        "files": inventory, "input_inventory_sha256": inventory_digest, "all_audited_input_files_unchanged": True,
    }


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=root / "replay-int8-r1")
    parser.add_argument("--control", type=Path, default=root / "replay-int8-control-r1/control.json")
    parser.add_argument("--wrapper", type=Path, default=root / "replay-int8-wrapper-r1.json")
    parser.add_argument("--output", type=Path, default=root / "replay-saved-audit.json")
    args = parser.parse_args()
    replay_dir, output = args.replay_dir.resolve(), args.output.resolve()
    require(not output.is_relative_to(replay_dir) and
            output not in (args.control.resolve(), args.wrapper.resolve(), Path(__file__).resolve()),
            "audit report must be a separate file outside the saved replay inputs")
    report = audit(replay_dir, args.control.resolve(), args.wrapper.resolve())
    report["audit_script_sha256"] = sha256(Path(__file__).resolve())
    output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output), "sha256": sha256(output),
                      "script_sha256": report["audit_script_sha256"], "fresh_pose_hz": report["fresh_pose_hz"],
                      "input_to_pose_p95_seconds": report["input_to_pose_seconds"]["p95"],
                      "map_chunks": report["map"]["chunks"], "points_total": report["map"]["points_total"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
