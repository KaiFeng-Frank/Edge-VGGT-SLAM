#!/usr/bin/env python3
"""Experimental board-only continuous tracking plus asynchronous VGGT mapping.

RGB-only replay follows original timestamps. The mapper owns its own process;
the frontend never waits for model inference, graph optimization or loop closure.
This entry point has no ORB frontend and does not imply real-time acceptance.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party" / "vggt")]


def cpu_list(value):
    values = tuple(int(x) for x in value.split(","))
    if not values or len(set(values)) != len(values) or min(values) < 0:
        raise argparse.ArgumentTypeError("Need distinct nonnegative CPU IDs")
    return values


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rgb-index", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--image-size", type=int, default=112)
    p.add_argument("--backend", choices=["cpu", "xnnpack-f16"], default="cpu")
    p.add_argument("--camera-backend", choices=["cpu", "rknn-step"], default="cpu")
    for name in ("model", "export-receipt", "compile-receipt", "capture-receipt"):
        p.add_argument(f"--camera-{name}", type=Path)
        p.add_argument(f"--camera-{name}-sha256")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--mapper-cpus", type=cpu_list, default=(4, 5))
    p.add_argument("--frontend-cpus", type=cpu_list, default=(6, 7))
    p.add_argument("--capture-cpus", type=cpu_list, default=(0, 1))
    p.add_argument("--candidate-cpus", type=cpu_list, default=(2, 3))
    p.add_argument("--map-period", type=float, default=3.)
    p.add_argument("--bootstrap-gap", type=float, default=.25)
    p.add_argument("--max-submaps", type=int, default=1024)
    p.add_argument("--mapper-memory-limit-mib", type=float, default=6500.)
    p.add_argument("--history-frames", type=int, default=300)
    p.add_argument("--history-seconds", type=float, default=10.)
    p.add_argument("--replay-stride", type=int, default=1,
                   help="Measured-observation stride for delayed depth replay; live frames remain unchanged")
    p.add_argument("--tracking-width", type=int, default=480)
    p.add_argument("--tracking-features", type=int, default=800)
    camera = p.add_mutually_exclusive_group()
    camera.add_argument("--local-calibration", type=Path,
                   help="Diagnostic frontend K/distortion JSON; nonzero distortion disables global map association")
    camera.add_argument("--rectify-calibration", type=Path,
                   help="Undistort both capture and mapper with the same K; remap time is included in replay latency")
    p.add_argument("--max-loops", type=int, choices=[0, 1], default=0)
    p.add_argument("--salad-checkpoint", type=str)
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--max-frames", type=int)
    a = p.parse_args()
    if (a.image_size < 14 or a.image_size % 14 or a.threads < 1 or a.map_period <= 0
            or a.bootstrap_gap <= 0 or a.max_submaps < 1 or a.mapper_memory_limit_mib <= 0
            or a.replay_stride < 1 or (a.max_frames is not None and a.max_frames < 2)):
        p.error("Invalid size, period or capacity")
    if not a.checkpoint.is_file():
        p.error("Checkpoint is missing")
    if a.max_loops and not a.salad_checkpoint:
        p.error("Loop closure requires an explicit local SALAD checkpoint")
    try:
        camera_configuration(a)
    except ValueError as exc:
        p.error(str(exc))
    return a


def camera_configuration(a):
    """Require a complete, explicit artifact chain for the fixed two-frame head."""
    import re
    enabled = getattr(a, "camera_backend", "cpu") == "rknn-step"
    config = {}
    for name in ("model", "export_receipt", "compile_receipt", "capture_receipt"):
        path = getattr(a, f"camera_{name}", None)
        digest = getattr(a, f"camera_{name}_sha256", None)
        if not enabled:
            if path is not None or digest is not None:
                raise ValueError("Camera artifacts require --camera-backend rknn-step")
            continue
        if path is None or not path.is_file() or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"RKNN camera requires an existing {name} and its explicit lowercase SHA256")
        config["model_path" if name == "model" else name] = str(path.resolve())
        config[{"model": "model_sha256", "export_receipt": "export_sha256",
                "compile_receipt": "compile_sha256", "capture_receipt": "capture_sha256"}[name]] = digest
    if enabled and a.max_loops:
        raise ValueError("RKNN camera currently supports two-frame mapping with --max-loops 0 only")
    return config if enabled else None


def json_value(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_row(stream, row):
    stream.write(json.dumps(row, default=json_value, allow_nan=False, separators=(",", ":")) + "\n")
    stream.flush()
    return time.monotonic()


def frame_token(item):
    return {key: item[key] for key in ("path", "frame_id", "timestamp", "captured_monotonic", "sequence")}


def summarize_pose_timing(rows, source):
    import numpy as np
    valid = [r for r in rows if r["valid"]]
    origin = source["start_monotonic"]
    ending = source["last_capture_monotonic"]
    duration = ending-origin if ending is not None and origin is not None else 0.
    full_seconds = int(duration)
    bins = [0] * full_seconds
    for row in valid:
        # Publication bins count available new-observation poses in real time.
        index = int(row["published"]-origin)
        if 0 <= index < full_seconds:
            bins[index] += 1
    ages = [(r["published"]-r["captured"])*1000 for r in valid]
    intervals = np.diff([r["published"] for r in valid])
    return dict(delivered_observations=len(rows), fresh_valid_poses=len(valid),
                valid_fraction_of_original_frames=len(valid)/source["original_frames"],
                valid_fraction_of_delivered_frames=len(valid)/len(rows) if rows else 0.,
                replay_duration_seconds=duration, valid_pose_publications_per_full_second=bins,
                minimum_full_second_pose_count=min(bins) if bins else None,
                first_valid_seconds_after_start=valid[0]["published"]-origin if valid else None,
                valid_pose_latency_p50_ms=float(np.percentile(ages, 50)) if ages else None,
                valid_pose_latency_p95_ms=float(np.percentile(ages, 95)) if ages else None,
                valid_pose_latency_max_ms=max(ages) if ages else None,
                maximum_valid_pose_publication_gap_seconds=float(max(intervals)) if len(intervals) else None,
                latency_origin="scheduled_RGB_replay_arrival_before_decode",
                latency_end="pose_JSONL_write_and_flush_return", physical_camera_latency_validated=False,
                startup_and_losses_included_in_per_second_counts=True)


def run(a):
    from vggt_slam.edge_mapping import atomic_json, mapping_worker, platform_sample
    from vggt_slam.timestamp_replay import TimestampReplay, read_rgb_index
    camera_rknn = camera_configuration(a)
    rows = read_rgb_index(a.rgb_index)
    if a.max_frames:
        rows = rows[:a.max_frames]
    rectifier = None
    rectification_calibration = None
    if getattr(a, "rectify_calibration", None) is not None:
        if getattr(a, "local_calibration", None) is not None:
            raise ValueError("Select either common rectification or frontend-only calibration")
        from vggt_slam.image_rectification import CalibrationRectifier
        rectification_calibration = json.loads(a.rectify_calibration.read_text())
        rectifier = CalibrationRectifier(rectification_calibration)
    a.output.mkdir(parents=True, exist_ok=False)
    config = dict(checkpoint=str(a.checkpoint.resolve()), image_size=a.image_size, threads=a.threads,
                  cpus=a.mapper_cpus, backend=a.backend, output=str(a.output.resolve()),
                  max_loops=a.max_loops, salad_checkpoint=a.salad_checkpoint,
                  max_submaps=a.max_submaps, memory_limit_mib=a.mapper_memory_limit_mib,
                  save_predictions=a.save_predictions,
                  rectification_calibration=rectification_calibration, camera_rknn=camera_rknn)
    stats = dict(status="starting", ORB_used=False, npu_used=False, model_backbone="original_VGGT",
                 mapper_config=config, command=sys.argv, queues=dict(capture=1, mapping_jobs=1,
                 mapping_results=1, pending_geometry=1, candidate_workers=1),
                 real_time_validated=False, accuracy_validated=False, endurance_validated=False,
                 eight_gb_board_validated=False, loop_closure_nonblocking_validated=False,
                 initial_platform=platform_sample())
    stats["camera_backend_requested"] = "rknn-step" if camera_rknn is not None else "cpu"
    if rectifier is not None:
        stats["common_rectification"] = rectifier.metadata()
        stats["rectification_timing"] = "capture remap after scheduled arrival; mapper remap inside map processing"
    ctx = mp.get_context("spawn")
    jobs, results = ctx.Queue(1), ctx.Queue(1)
    worker = ctx.Process(target=mapping_worker, args=(config, jobs, results), name="vggt-mapper")
    worker.start()
    source = None
    executor = None
    poses = events = receipts = None
    timings, map_timings = [], []
    maps = 0
    busy = False
    shutdown_sent = False
    future = None
    candidate_meta = None
    candidate_started = None
    candidate_rounds = 0
    pending_geometry = None
    latest = overlap = first = None
    last_submit = float("-inf")
    nodes = {}
    active_node = None
    tracker = None
    try:
        deadline = time.monotonic()+120
        while True:
            try:
                ready = results.get(timeout=.2)
                break
            except queue.Empty:
                if not worker.is_alive() or time.monotonic() > deadline:
                    raise RuntimeError("Mapper failed to become ready within 120 seconds")
        if ready["kind"] != "ready":
            raise RuntimeError(ready.get("traceback", str(ready)))
        stats["mapper_ready"] = ready
        if rectifier is not None and ready.get("rectification") != rectifier.metadata():
            raise RuntimeError("Mapper and capture rectification configurations differ")
        os.sched_setaffinity(0, a.frontend_cpus)
        import cv2
        import numpy as np
        from vggt_slam.continuous_tracking import (
            ContinuousDepthTracker, build_depth_candidate, advance_depth_candidate)
        from vggt_slam.tracking_geometry import associate_pose
        cv2.setNumThreads(1)
        calibration = None
        if rectifier is not None:
            info = rectifier.metadata()
            calibration = dict(K=info["output_K"], distortion=info["output_distortion"],
                               image_hw=info["image_hw"], source=info["source"])
            stats["local_frontend_calibration"] = calibration
            stats["calibration_scope"] = "capture and VGGT/SL4 use identical rectified RGB; frontend uses rectified K with zero distortion; mapper estimates its own K"
        elif getattr(a, "local_calibration", None) is not None:
            calibration = json.loads(a.local_calibration.read_text())
            if (set(calibration) != {"K", "distortion", "image_hw", "source"}
                    or len(calibration["image_hw"]) != 2
                    or any(type(v) is not int or v <= 0 for v in calibration["image_hw"])):
                raise ValueError("Calibration needs K, distortion, positive image_hw and source")
            stats["local_frontend_calibration"] = calibration
            stats["calibration_scope"] = "frontend_only; VGGT/SL4 receives original RGB and estimates its own intrinsics"
        tracker = ContinuousDepthTracker(max_width=a.tracking_width, max_features=a.tracking_features,
                                         history_frames=a.history_frames, history_seconds=a.history_seconds,
                                         replay_stride=a.replay_stride,
                                         intrinsic_raw=calibration["K"] if calibration else None,
                                         distortion=calibration["distortion"] if calibration else None)
        executor = ThreadPoolExecutor(max_workers=1, initializer=os.sched_setaffinity,
                                      initargs=(0, a.candidate_cpus), thread_name_prefix="depth-replay")
        poses = (a.output / "tracking.jsonl").open("w")
        receipts = (a.output / "publications.jsonl").open("w")
        events = (a.output / "events.jsonl").open("w")
        source = (TimestampReplay(rows, cpus=a.capture_cpus, decoder=rectifier.decode_bgr)
                  if rectifier is not None else TimestampReplay(rows, cpus=a.capture_cpus))
        source.start()
        stats["status"] = "running"
        eof_started = None
        while True:
            try:
                packet = results.get_nowait()
            except queue.Empty:
                packet = None
            if packet is not None:
                if packet.get("camera_runtime") is not None:
                    stats["camera_runtime"] = packet["camera_runtime"]
                    stats["npu_used"] = bool(packet["camera_runtime"].get("npu_used", False))
                if packet["kind"] == "error":
                    raise RuntimeError(packet["traceback"])
                if packet["kind"] == "stopped":
                    if not shutdown_sent or busy or future is not None or pending_geometry is not None:
                        raise RuntimeError("Mapper stopped before the requested work was drained")
                    stats["mapper_stopped"] = packet
                    break
                if packet["kind"] != "map":
                    raise RuntimeError(f"Unexpected mapper packet {packet['kind']}")
                busy = False
                maps += 1
                overlap = packet["anchor"]
                nodes[packet["graph_node_id"]] = (np.asarray(packet["homography"]), packet["revision"])
                for key, H in packet["node_updates"].items():
                    nodes[int(key)] = (np.asarray(H), packet["revision"])
                # Only a few recent anchor transforms are needed by the
                # frontend. Complete graph storage belongs to the mapper.
                protected = {active_node, packet["graph_node_id"]}
                if candidate_meta is not None:
                    protected.add(candidate_meta["graph_node_id"])
                for key in sorted(nodes)[:-8]:
                    if key not in protected:
                        del nodes[key]
                pending_geometry = packet
                record = {key: value for key, value in packet.items()
                          if key not in ("depth", "confidence", "intrinsic")}
                record.update(received_monotonic=time.monotonic(), after_source_eof=source.done)
                write_row(events, record)
                map_timings.append(dict(revision=packet["revision"], published=packet["published_monotonic"],
                                        processing_seconds=packet["processing_seconds"],
                                        age_seconds=packet["newest_frame_age_seconds"]))
            if future is not None and future.done():
                candidate = future.result()
                gap = tracker.state["sequence"]-candidate["last_sequence"]
                if candidate["available"] and gap > 3:
                    if candidate_rounds < 8 and time.monotonic()-candidate_started <= a.history_seconds:
                        snapshot = tracker.snapshot_candidate_catchup(candidate)
                        future = executor.submit(advance_depth_candidate, candidate, snapshot)
                        candidate_rounds += 1
                        write_row(events, dict(kind="depth_background_catchup", round=candidate_rounds,
                                               anchor=candidate_meta["anchor"], pending_observations=gap,
                                               snapshot_available=snapshot["available"], reason=snapshot["reason"],
                                               submitted_monotonic=time.monotonic()))
                        candidate = None
                    else:
                        candidate = dict(candidate, available=False, reason="background_catchup_budget_exhausted")
                if candidate is not None:
                    applied = tracker.apply_depth_candidate(candidate)
                    applied.update(kind="depth_adoption", graph_node_id=candidate_meta["graph_node_id"],
                                   revision=candidate_meta["revision"], completed_monotonic=time.monotonic(),
                                   anchor_age_seconds=time.monotonic()-candidate_meta["anchor"]["captured_monotonic"],
                                   background_rounds=candidate_rounds, after_source_eof=source.done)
                    if applied["adopted"]:
                        active_node = candidate_meta["graph_node_id"]
                    write_row(events, applied)
                    future = candidate_meta = None
            if pending_geometry is not None and future is None:
                packet, pending_geometry = pending_geometry, None
                try:
                    snapshot = tracker.snapshot_depth_history(packet["anchor"]["frame_id"],
                                                              packet["anchor"]["captured_monotonic"])
                except ValueError as exc:
                    write_row(events, dict(kind="depth_rejected", reason=str(exc),
                                           anchor=packet["anchor"], revision=packet["revision"]))
                else:
                    future = executor.submit(build_depth_candidate, snapshot, depth=packet["depth"],
                                             confidence=packet["confidence"], K_model=packet["intrinsic"],
                                             model_image_size=packet["model_image_size"])
                    candidate_meta = dict(anchor=packet["anchor"], graph_node_id=packet["graph_node_id"],
                                          revision=packet["revision"])
                    candidate_started = time.monotonic()
                    candidate_rounds = 0
            item = source.read()
            if item is not None:
                if calibration is not None and list(item["frame"].shape[:2]) != calibration["image_hw"]:
                    raise ValueError("RGB dimensions differ from the supplied calibration")
                begin = time.monotonic()
                result = tracker.process(item["frame"], item["frame_id"], item["captured_monotonic"])
                result.update(timestamp=item["timestamp"], graph_node_id=active_node,
                              decode_started_monotonic=item["decode_started_monotonic"],
                              ready_monotonic=item["ready_monotonic"], delivered_monotonic=item["delivered_monotonic"],
                              processing_started_monotonic=begin, graph_revision=None)
                if result["pose_valid"] and result["model_association_available"] and active_node in nodes:
                    H, revision = nodes[active_node]
                    result["global_association"] = associate_pose(result["T_current_from_anchor"], result["K_tracking"],
                                                                  H @ result["H_model_from_local"])
                    result["graph_revision"] = revision
                published = write_row(poses, result)
                write_row(receipts, dict(frame_id=item["frame_id"], pose_valid=result["pose_valid"],
                                         captured_monotonic=item["captured_monotonic"],
                                         pose_available_monotonic=published,
                                         latency_seconds=published-item["captured_monotonic"]))
                timings.append(dict(valid=result["pose_valid"], captured=item["captured_monotonic"], published=published))
                latest = frame_token(item)
                if first is None:
                    first = latest
            if not busy and latest is not None and not source.done:
                initial = overlap is None
                eligible = (initial and latest["captured_monotonic"]-first["captured_monotonic"] >= a.bootstrap_gap)
                eligible |= (not initial and latest["frame_id"] > overlap["frame_id"] and
                             time.monotonic()-last_submit >= a.map_period)
                if eligible:
                    batch = [first if initial else overlap, latest]
                    jobs.put_nowait(dict(batch=batch, requested_nodes=list(nodes)))
                    busy = True
                    last_submit = time.monotonic()
                    write_row(events, dict(kind="map_submitted", batch=batch, submitted_monotonic=last_submit))
            if source.exhausted:
                if eof_started is None:
                    eof_started = time.monotonic()
                    write_row(events, dict(kind="source_eof", completed_monotonic=eof_started))
                if time.monotonic()-eof_started > 120:
                    raise RuntimeError("Mapper/depth replay did not drain within 120 seconds")
                if not busy and future is None and pending_geometry is None and not shutdown_sent:
                    jobs.put_nowait(None)
                    shutdown_sent = True
            if not worker.is_alive() and not shutdown_sent:
                # Queue feeder publication can follow exit by a short interval;
                # the next iteration still consumes an explicit error packet.
                try:
                    failure = results.get(timeout=.2)
                except queue.Empty:
                    raise RuntimeError(f"Mapper exited without result, code {worker.exitcode}")
                raise RuntimeError(failure.get("traceback", str(failure)))
            if item is None and source.done:
                # Finished replay has no condition wait. Yield while draining
                # background work so this loop cannot starve the depth worker.
                time.sleep(.002)
        stats["status"] = "completed_experiment"
    except BaseException as exc:
        stats.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        cleanup_errors = []
        if source is not None:
            try:
                source.close()
            except Exception as exc:
                cleanup_errors.append(f"source: {exc}")
            stats["source"] = source.metrics()
            stats["pose_timing"] = summarize_pose_timing(timings, stats["source"])
        if executor is not None:
            try:
                executor.shutdown(wait=True, cancel_futures=True)
            except Exception as exc:
                cleanup_errors.append(f"depth worker: {exc}")
        if worker.is_alive():
            if not shutdown_sent:
                try:
                    jobs.put_nowait(None)
                except queue.Full:
                    pass
            worker.join(timeout=3)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
                stats["mapper_terminated_during_cleanup"] = True
        else:
            worker.join(timeout=1)
        if tracker is not None:
            stats["tracker"] = tracker.metrics()
        stats.update(map_updates=maps, map_timing=map_timings, final_platform=platform_sample(),
                     mapper_exitcode=worker.exitcode, cleanup_errors=cleanup_errors)
        if worker.exitcode not in (0, None) or cleanup_errors:
            stats["status"] = "failed"
        for stream in (poses, events, receipts):
            if stream is not None:
                stream.close()
        atomic_json(a.output / "metrics.json", stats)
        jobs.close()
        results.close()
    if stats["status"] == "failed":
        raise RuntimeError(f"Edge runtime cleanup failed; see {a.output / 'metrics.json'}")
    return stats


if __name__ == "__main__":
    print(json.dumps(run(arguments()), default=json_value, allow_nan=False))
