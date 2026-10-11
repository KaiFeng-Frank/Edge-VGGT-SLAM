"""Sole-owner VGGT/SL4 mapping process for the experimental edge runner."""
from pathlib import Path
import json
import math
import os
import queue
import resource
import time
import traceback


def prepare_rectified_images(image_names, rectifier, image_size):
    """Use the unchanged VGGT PIL preprocessing on identical rectified pixels.

    Lossless in-memory PNGs bridge BGR capture to PIL's RGB loader without
    writing a dataset, accumulating a frame cache, or changing input identities.
    This work belongs inside the timed mapping job.
    """
    from contextlib import ExitStack
    from io import BytesIO
    import cv2
    from vggt.utils.load_fn import load_and_preprocess_images
    with ExitStack() as stack:
        streams = []
        for path in image_names:
            frame = rectifier.decode_bgr(path)
            ok, encoded = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if not ok:
                raise RuntimeError("Cannot encode rectified mapper input")
            streams.append(stack.enter_context(BytesIO(encoded.tobytes())))
        return load_and_preprocess_images(streams, image_size=image_size)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")))
    temporary.replace(path)


def platform_sample():
    result = dict(temperature_c={}, cpu_frequency_khz={})
    try:
        fields = {"MemTotal", "MemAvailable", "MemFree", "Buffers", "Cached",
                  "SReclaimable", "Shmem", "CmaTotal", "CmaFree"}
        result["linux_memory_bytes"] = {
            parts[0].rstrip(":"): int(parts[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            for parts in [line.split()]
            if len(parts) == 3 and parts[0].rstrip(":") in fields and parts[2] == "kB"}
    except (OSError, ValueError):
        result["linux_memory_bytes"] = None
    for p in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            result["temperature_c"][p.joinpath("type").read_text().strip()] = int(p.joinpath("temp").read_text()) / 1000
        except (OSError, ValueError):
            pass
    for p in Path("/sys/devices/system/cpu/cpufreq").glob("policy*"):
        try:
            result["cpu_frequency_khz"][p.name] = int(p.joinpath("scaling_cur_freq").read_text())
        except (OSError, ValueError):
            pass
    result["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return result


def validate_mapping_config(config):
    """Reject auxiliary options before importing/loading a model or graph."""
    allowed = {"cpus", "threads", "checkpoint", "image_size", "camera_rknn", "backend",
               "max_loops", "output", "max_submaps", "memory_limit_mib", "save_predictions",
               "rectification_calibration"}
    if type(config.get("max_loops", 0)) is not int or config.get("max_loops", 0) != 0:
        raise ValueError("Only max_loops=0 is supported")
    if any(value is not None for key, value in config.items() if key not in allowed):
        raise ValueError("Unsupported mapping configuration; only original VGGT is permitted")


def mapping_worker(config, jobs, results):
    """Only this process may access or optimize the graph or the neural model.

    Input/output queues have capacity one and the caller admits one job at a
    time. Capacity exhaustion fails explicitly; it never freezes a stale map
    while reporting that dense mapping is continuing.
    """
    handle = None
    camera_handle = None
    terminal = None
    try:
        validate_mapping_config(config)
        if config["cpus"] is not None:
            os.sched_setaffinity(0, config["cpus"])
        import cv2
        import numpy as np
        import torch
        from vggt_slam.runtime import load_model
        from vggt_slam.solver import Solver
        cv2.setNumThreads(1)
        torch.set_num_threads(config["threads"])
        torch.set_num_interop_threads(1)
        started = time.monotonic()
        rectifier = None
        if config.get("rectification_calibration") is not None:
            from vggt_slam.image_rectification import CalibrationRectifier
            rectifier = CalibrationRectifier(config["rectification_calibration"])
        model = load_model(config["checkpoint"], device="cpu", image_size=config["image_size"],
                           prune_unused_heads=True, offline=True)
        if config.get("camera_rknn") is not None:
            from vggt_slam.rknn_camera import install_rknn_camera
            # Release the original CPU CameraHead before packing remaining
            # Linear weights; the NPU context owns its shared FP16 weights.
            camera_handle = install_rknn_camera(model, **config["camera_rknn"])
        if config["backend"] == "xnnpack-f16":
            from vggt_slam.xnnpack_linear import install_xnnpack_linear
            handle = install_xnnpack_linear(model, threads=config["threads"], dtype="float16",
                                           scope="all", cpu_affinity=config["cpus"])
        solver = Solver(init_conf_threshold=25., device="cpu", image_size=config["image_size"],
                        max_loops=0,
                        offline=True, enable_viewer=False)
        output = Path(config["output"])
        chunks = output / "map_chunks"
        chunks.mkdir()
        manifest = dict(format="vggt_camera_local_dense_chunks_with_SL4_graph", units="model_units",
                        chunks=[], nodes={}, revision=0, metric_scale_validated=False,
                        rectification=rectifier.metadata() if rectifier is not None else None)
        saved = set()
        results.put(dict(kind="ready", load_seconds=time.monotonic()-started,
                         platform=platform_sample(), cpus=sorted(os.sched_getaffinity(0)),
                         camera_runtime=camera_handle.metrics() if camera_handle is not None else None,
                         rectification=rectifier.metadata() if rectifier is not None else None))
        while True:
            job = jobs.get()
            if job is None:
                break
            if solver.map.get_num_submaps() >= config["max_submaps"]:
                raise RuntimeError("Configured map capacity exhausted; mapping stopped explicitly")
            begin = time.monotonic()
            batch = job["batch"]
            paths = [x["path"] for x in batch]
            if rectifier is None:
                prediction = solver.run_predictions(paths, model, max_loops=config["max_loops"])
            else:
                images = prepare_rectified_images(paths, rectifier, config["image_size"])
                prediction = solver.run_predictions(paths, model, max_loops=config["max_loops"],
                                                    prepared_images=images)
                del images
            predicted = time.monotonic()
            for key in ("depth", "depth_conf", "intrinsic", "extrinsic"):
                if not np.asarray(prediction[key]).size or not np.isfinite(prediction[key]).all():
                    raise RuntimeError(f"Invalid VGGT {key}")
            solver.add_points(prediction)
            solver.graph.optimize()
            graph_error = float(solver.graph.graph.error(solver.graph.values))
            if not math.isfinite(graph_error):
                raise RuntimeError("Nonfinite SL4 optimization error")
            optimized = time.monotonic()
            submap = solver.current_working_submap
            if list(submap.frame_ids) != [x["frame_id"] for x in batch]:
                raise RuntimeError("Mapper input identities differ from inserted submap")
            # New local chunks are immutable. An atomic graph manifest updates
            # every chunk's world placement after SL4 optimization without rewriting
            # the complete accumulated point cloud at every revision.
            for node, current in sorted(solver.map.submaps.items()):
                if node not in saved:
                    arrays = {}
                    counts = []
                    for i in range(len(current.pointclouds)):
                        mask = current.conf_masks[i] > current.conf_threshold
                        arrays[f"points_{i}"] = np.asarray(current.pointclouds[i][mask], np.float32)
                        arrays[f"colors_{i}"] = np.asarray(current.colors[i][mask], np.uint8)
                        counts.append(len(arrays[f"points_{i}"]))
                    name = f"submap_{node:06d}.npz"
                    np.savez(chunks / name, **arrays)
                    manifest["chunks"].append(dict(path=f"map_chunks/{name}", submap_id=node,
                                                   nodes=[node+i for i in range(len(counts))], point_counts=counts))
                    saved.add(node)
                for i in range(len(current.pointclouds)):
                    H = np.asarray(solver.graph.get_homography(node+i), np.float64)
                    if H.shape != (4, 4) or not np.isfinite(H).all():
                        raise RuntimeError("Invalid graph homography")
                    manifest["nodes"][str(node+i)] = H.tolist()
            manifest["revision"] += 1
            temporary_poses = output / "poses.tum.tmp"
            solver.map.write_poses_to_file(str(temporary_poses), solver.graph, kitti_format=False)
            temporary_poses.replace(output / "poses.tum")
            atomic_json(output / "map.json", manifest)
            published = time.monotonic()
            index = len(batch)-1
            node = int(submap.get_id()+index)
            sample = platform_sample()
            if sample["peak_rss_mib"] > config["memory_limit_mib"]:
                raise RuntimeError("Mapper exceeded the configured process memory budget")
            row = dict(kind="map", revision=manifest["revision"], anchor=batch[-1], graph_node_id=node,
                       homography=manifest["nodes"][str(node)],
                       node_updates={str(key): manifest["nodes"][str(key)] for key in job["requested_nodes"]
                                     if str(key) in manifest["nodes"]},
                       depth=np.array(prediction["depth"][index], copy=True),
                       confidence=np.array(prediction["depth_conf"][index], copy=True),
                       intrinsic=np.array(prediction["intrinsic"][index], copy=True),
                       model_image_size=config["image_size"],
                       processing_seconds=published-begin, prediction_seconds=predicted-begin,
                       optimization_seconds=optimized-predicted, publication_seconds=published-optimized,
                       published_monotonic=published, newest_frame_age_seconds=published-batch[-1]["captured_monotonic"],
                       graph_error=graph_error, graph_nodes=int(solver.graph.values.size()),
                       graph_factors=int(solver.graph.graph.size()), submaps=solver.map.get_num_submaps(),
                       accepted_loop_closures=int(solver.graph.get_num_loops()), platform=sample)
            if camera_handle is not None:
                row["camera_runtime"] = camera_handle.metrics()
            if config["save_predictions"]:
                np.savez(output / f"predictions_{manifest['revision']:04d}.npz",
                         **{k: prediction[k] for k in ("depth", "depth_conf", "intrinsic", "extrinsic")},
                         frame_ids=[x["frame_id"] for x in batch])
            results.put(row)
        terminal = dict(kind="stopped", platform=platform_sample())
    except BaseException as exc:
        terminal = dict(kind="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        cleanup_errors = []
        for backend in (handle, camera_handle):
            if backend is None:
                continue
            try:
                backend.close()
            except BaseException as exc:
                cleanup_errors.append(f"{type(exc).__name__}: {exc}")
        if cleanup_errors:
            terminal = dict(kind="error", error=f"Runtime cleanup failed: {cleanup_errors}",
                            traceback="; ".join(cleanup_errors), previous_terminal=terminal)
        if terminal is not None and camera_handle is not None:
            terminal["camera_runtime"] = camera_handle.metrics()
        if terminal is not None:
            try:
                results.put(terminal, timeout=2)
            except queue.Full:
                # The parent may already have failed; it has its own bounded
                # join/terminate cleanup and must not leave a blocked producer.
                pass
