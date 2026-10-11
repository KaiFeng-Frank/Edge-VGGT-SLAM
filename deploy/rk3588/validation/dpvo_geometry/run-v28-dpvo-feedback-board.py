#!/usr/bin/env python3
"""RK3588-only fixed-three-view neural/correlation/SE3-BA feedback probe.

This is neither the official eight-frame initializer nor online VO. Features
come from frozen real CPU captures; no encoder or image decoding is timed.
Copyright (c) 2022 Princeton Vision & Learning Lab (upstream mathematics,
MIT); see LICENSE. Newly written integration is also under the MIT license.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import resource
import sys
import time
import traceback
from pathlib import Path

BASE = Path("/mnt/vggt-data/vggt-slam2/experiments")
PINS = {
    "dpvo_cpu_ops.py": "15191a0d481176e5be80bb6c41c3fb3f4d057d95b18ab18f6e0efcd73793dd32",
    "dpvo_probe_common.py": "ba47164a4273e0d392f8e23892f17be1aaf89b7ef42c3ebd746e3a7b3fd946a9",
    "manifest.json": "ce2ecaf96b086f69ef4d90a2e07e7b81afe8ae094656eb41260182000a69e375",
    "input-manifest.json": "bffcd221ee1e7c4b17045bf9b642df7272747ee000e67d5e682610b635804539",
}
CAPTURE_SHA = "e0563efb962edb90684480be114b88bc2c7825015f3b13a163f6f47f06360088"
MOTION_SHA = "55a7db19d1a9bfbf986da62ed999fc8b122add0fca84f01681e6ad0b82ba4e86"
CHECKPOINT_SHA = "30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12"
GEOMETRY_SHA = "13eb1e03eb4a5ed1bd1533549f2b9f4994abfaa61e0b55304b99e6fe77d813df"
BUILD_SHA = "34fbf049852b351e7df1097cc806fbf83e1e1035e18b01ca4e9e8ba39499cd18"
LIBRARY_SHA = "829d1b42fa24d28b1e280b882c0a9e8556b0e7cdbe02f7489eeef15ba3581bca"
NATIVE_PINS = {
    "dpvo_corr.cpp": "3cf0c9cf79754fff996efc03c2ca2b162733f10b2b9aa580dee263e5df5b9990",
    "dpvo_corr.h": "1866c747265b401f8d29d896e5f634ac104f238704e5fe443aa683a7395ab932",
    "dpvo_corr.py": "a6c37bf4b840b492736d5c5916be6e2c0e8680693d01d0fe3a5fe62702ecb779",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def meta(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes,
                tensor_sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest())


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def require_board():
    require(platform.machine() == "aarch64" and platform.system() == "Linux",
            "RK3588 board only; host model or feature execution is prohibited")
    require(b"rockchip,rk3588" in Path("/proc/device-tree/compatible").read_bytes(),
            "RK3588 device-tree evidence required")


def validate_value(value, shape, name):
    import numpy as np
    require(isinstance(value, np.ndarray) and value.dtype == np.float32 and
            value.shape == shape and np.isfinite(value).all(),
            name + " requires finite original FP32 shape")


def residual_summary(linear, weight, common_mask=None):
    import numpy as np
    residual = linear["residual"][0].astype(np.float64)
    finite = np.isfinite(residual).all(axis=-1)
    selected = linear["active"] if common_mask is None else common_mask
    selected = selected & finite
    norm = np.linalg.norm(residual[finite], axis=-1)
    return dict(active_edges=int(linear["active"].sum()),
                finite_residual_edges=int(finite.sum()), nonfinite_residual_edges=int((~finite).sum()),
                all_finite_residual_norm_mean=float(norm.mean()) if norm.size else None,
                all_finite_residual_norm_max=float(norm.max()) if norm.size else None,
                selected_edges=int(selected.sum()),
                weighted_squared_residual=float(np.sum(weight[0, selected].astype(np.float64) * residual[selected] ** 2)),
                selection="current online gate" if common_mask is None else "fixed pre-BA active mask, finite rows only")


def feedback_round(geo, poses, patches, intrinsics, graph, net, correlation, update, on_stage=None):
    """One real feedback step, or a tiny injected synthetic check.

update(net,ctx,corr,ii,jj,kk) returns owned FP32 net/delta/weight.
correlation.correlate(coords,kk,jj,cache_key=...,radius=3) returns owned
corr and timing. Its version key is immutable for these three features.
The function never falls back to old geometry or a previous neural output.
"""
    import numpy as np
    ii, jj, kk, ctx = (graph[k] for k in ("ii", "jj", "kk", "ctx"))
    count = len(kk)
    clock = [time.monotonic()]
    pose_before, patch_before, net_before = poses.copy(), patches.copy(), net.copy()
    clock.append(time.monotonic())
    coords = geo.reproject(poses, patches, intrinsics, ii, jj, kk)
    validate_value(coords, (1, count, 2, 3, 3), "reprojected correlation coordinates")
    clock.append(time.monotonic())
    if on_stage is not None:
        on_stage("correlation_begin")
    corr, native_timing = correlation.correlate(
        coords, kk, jj, cache_key=correlation.cache_key, radius=3)
    validate_value(corr, (1, count, 882), "correlation")
    clock.append(time.monotonic())
    if on_stage is not None:
        on_stage("update_begin")
    new_net, delta, weight = update(net, ctx, corr, ii, jj, kk)
    for key, value, shape in (("net", new_net, (1, count, 384)),
                               ("delta", delta, (1, count, 2)),
                               ("weight", weight, (1, count, 2))):
        validate_value(value, shape, key)
    require(np.all((weight >= 0) & (weight <= 1)), "Original Update weights must be in [0,1]")
    clock.append(time.monotonic())
    target = np.ascontiguousarray(coords[:, :, :, 1, 1] + delta)
    clock.append(time.monotonic())
    if on_stage is not None:
        on_stage("BA_begin")
    ba = geo.BA(poses, patches, intrinsics, target, weight,
                np.array([1e-4], np.float32), ii, jj, kk,
                1, poses.shape[1], 96, iterations=2, eff_impl=False)
    if on_stage is not None:
        on_stage("BA_completed")
    clock.append(time.monotonic())
    coords_after = geo.reproject(poses, patches, intrinsics, ii, jj, kk)
    clock.append(time.monotonic())
    before = geo.fastba_linearize(pose_before, patch_before, intrinsics, target, weight, ii, jj, kk)
    after = geo.fastba_linearize(poses, patches, intrinsics, target, weight, ii, jj, kk)
    for name, value in (("poses", poses), ("patches", patches), ("coords after BA", coords_after)):
        require(np.isfinite(value).all(), "Nonfinite feedback geometry: " + name)
    require(np.array_equal(poses[0, 0], pose_before[0, 0]), "Fixed first pose changed")
    require(np.array_equal(patches[:, :, :2], patch_before[:, :, :2]), "BA changed patch x/y")
    require(np.all(patches[:, :, 2] > 0), "BA inverse depth is not positive")
    require(np.all(patches[:, :, 2] == patches[:, :, 2, :1, :1]), "BA did not preserve shared patch inverse depth")
    left_delta = geo.se3_log(geo.se3_compose(poses, geo.se3_inverse(pose_before)))
    depth_delta = patches[:, :, 2, 1, 1] - patch_before[:, :, 2, 1, 1]
    result = dict(
        ba=ba,
        pose_left_increment=left_delta.tolist(),
        pose_translation_change_norm=float(np.linalg.norm(poses[..., :3] - pose_before[..., :3])),
        pose_left_increment_norm=float(np.linalg.norm(left_delta)),
        inverse_depth_change_norm=float(np.linalg.norm(depth_delta)),
        inverse_depth_changed_patches=int(np.count_nonzero(depth_delta)),
        inverse_depth_range=[float(patches[:, :, 2].min()), float(patches[:, :, 2].max())],
        quaternion_norm_error_max=float(np.abs(np.linalg.norm(poses[..., 3:], axis=-1) - 1).max()),
        coords_change_norm=float(np.linalg.norm(coords_after - coords)),
        recurrent_net_change_norm=float(np.linalg.norm(new_net - net_before)),
        residual_before=residual_summary(before, weight),
        residual_after=residual_summary(after, weight),
        residual_after_using_pre_BA_active_mask=residual_summary(after, weight, before["active"]),
        native_correlation_timing=native_timing,
        all_feedback_state_finite=True, fixed_first_pose_preserved=True)
    clock.append(time.monotonic())
    labels = ("snapshot", "reproject", "correlation", "update", "target", "BA", "reproject_after", "diagnostics")
    result["seconds"] = {name:clock[i + 1] - clock[i] for i, name in enumerate(labels)}
    result.update(begin_monotonic=clock[0], end_monotonic=clock[-1], round_total_seconds=clock[-1] - clock[0])
    arrays = dict(poses_before=pose_before, poses_after=poses.copy(),
                  patches_before=patch_before, patches_after=patches.copy(),
                  coords_before=coords, coords_after=coords_after,
                  net_before=net_before, net_after=new_net.copy(), corr=corr,
                  delta=delta, weight=weight, target=target,
                  projection_before=before["projection"], projection_after=after["projection"],
                  residual_before=before["residual"], residual_after=after["residual"],
                  active_before=before["active"], active_after=after["active"])
    return new_net, arrays, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("v26-source-dir", "capture-dir", "motion-dir", "checkpoint",
                 "native-source-dir", "library", "build-receipt", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--geometry", type=Path, default=Path(__file__).with_name("dpvo_cpu_geometry.py"))
    args = parser.parse_args(argv)
    require_board()  # Before importing Torch or opening captured features.
    output = args.output_dir.resolve()
    require(output.is_relative_to(BASE), "New output inside isolated board experiments required")
    output.mkdir(exist_ok=False)
    prior_affinity = os.sched_getaffinity(0)
    pins = {}
    neural = cached = None
    state = None
    immutable = {}
    report = dict(schema="v28_DPVO_fixed_three_frame_feedback", status="failed",
                  script_sha256=sha(__file__), actual_board_execution=True,
                  original_DPVO_neural_weights=True, original_VGGT_weights_used=False,
                  production_changed=False, ORB_used=False, NPU_used=False, CUDA_used=False,
                  GT_or_sensor_depth_used=False, BA_executed=False, feedback_rounds_completed=0,
                  complete_VO_executed=False, official_eight_frame_initialization_executed=False,
                  streamed_or_online_input=False, trajectory_accuracy_validated=False,
                  realtime_validated=False, calls=[], stage_events=[], cleanup_errors=[])

    def pin(path, expected):
        path = Path(path).resolve()
        require(sha(path) == expected, "Pinned file differs: " + str(path))
        pins[str(path)] = expected
        return path

    def load_artifact(folder, artifact, metadata):
        name = artifact["file"]
        require(Path(name).name == name, "Safe artifact basename required")
        path = pin(folder / name, artifact["sha256"])
        require(path.stat().st_size == artifact["bytes"], "Artifact length differs")
        with np.load(path, allow_pickle=False) as store:
            values = {key:store[key].copy() for key in store.files}
        require(set(values) == set(metadata) and
                all(meta(value) == metadata[key] and np.isfinite(value).all() for key, value in values.items()),
                "Saved artifact raw metadata or finite gate failed")
        return values

    def save(name, values):
        path = output / name
        np.savez(path, **values)
        return dict(file=name, sha256=sha(path), bytes=path.stat().st_size,
                    arrays={key:meta(value) for key, value in values.items()})

    try:
        pin(__file__, report["script_sha256"])
        pin(args.geometry, GEOMETRY_SHA)
        for name, digest in PINS.items():
            pin(args.v26_source_dir / name, digest)
        build = json.loads(pin(args.build_receipt, BUILD_SHA).read_text())
        require(build["status"] == "compiled_and_synthetic_smoke_passed" and
                build["source_sha256"] == NATIVE_PINS and build["output"]["sha256"] == LIBRARY_SHA,
                "Frozen ARM native build chain differs")
        pin(args.library, LIBRARY_SHA)
        for name, digest in NATIVE_PINS.items():
            pin(args.native_source_dir / name, digest)
        pin(args.checkpoint, CHECKPOINT_SHA)
        capture = json.loads(pin(args.capture_dir / "capture.json", CAPTURE_SHA).read_text())
        motion = json.loads(pin(args.motion_dir / "motion.json", MOTION_SHA).read_text())
        manifest = json.loads((args.v26_source_dir / "input-manifest.json").read_text())
        require(capture["status"] == "completed_encoder_cpu_probe_not_VO" and
                motion["status"] == "completed_dynamic_neural_probe_not_VO" and
                motion["capture_sha256"] == CAPTURE_SHA and capture["input_manifest_sha256"] == PINS["input-manifest.json"],
                "Frozen real capture/motion input chain required")
        frame_ids = [row["frame_id"] for row in manifest["frames"]]
        require(frame_ids == motion["frame_ids"] == [row["frame_id"] for row in capture["inputs"]] and len(frame_ids) == 3,
                "Original three frame IDs/order differ")
        require({4, 5, 6, 7} <= prior_affinity, "All four A76 cores must be initially permitted")
        os.sched_setaffinity(0, {4, 5, 6, 7})
        require(os.sched_getaffinity(0) == {4, 5, 6, 7}, "Actual CPU affinity differs")
        import numpy as np
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        geo = module(args.geometry, "_v28_geometry")
        common = module(args.v26_source_dir / "dpvo_probe_common.py", "_v28_frozen_common")
        native = module(args.native_source_dir / "dpvo_corr.py", "_v28_frozen_corr")
        report.update(frame_ids=frame_ids, input_manifest_sha256=PINS["input-manifest.json"],
                      capture_sha256=CAPTURE_SHA, motion_sha256=MOTION_SHA,
                      geometry_sha256=GEOMETRY_SHA, library_sha256=LIBRARY_SHA, build_sha256=BUILD_SHA,
                      cpu_affinity=sorted(os.sched_getaffinity(0)), torch_threads=4, torch_interop_threads=1,
                      numpy_version=np.__version__, torch_version=torch.__version__,
                      OMP_WAIT_POLICY=os.environ.get("OMP_WAIT_POLICY"), GOMP_SPINCOUNT=os.environ.get("GOMP_SPINCOUNT"),
                      OPENBLAS_NUM_THREADS=os.environ.get("OPENBLAS_NUM_THREADS"), OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS"))
        start = time.monotonic()
        neural, loader = common.load_neural(args.v26_source_dir / "official", args.v26_source_dir / "manifest.json",
                                           args.checkpoint, CHECKPOINT_SHA, args.v26_source_dir / "dpvo_cpu_ops.py",
                                           cpu_ops_sha256=PINS["dpvo_cpu_ops.py"])
        report.update(neural_load_seconds=time.monotonic() - start, loader=loader)
        for relative, value in loader["source_hashes"].items():
            pin(args.v26_source_dir / "official" / relative, value["sha256"])
        state = {key:meta(value.detach().numpy()) for key, value in neural.state_dict().items()}
        require(state == capture["state_before"] == motion["original_weights_before"], "Original 94 tensor bytes differ")
        report["original_weights_before"] = state
        patch = load_artifact(args.motion_dir, motion["patches"], motion["patches"]["arrays"])
        parent = next(row for row in motion["scenarios"] if row["name"] == "three_frame_all_pairs")
        graph = load_artifact(args.motion_dir, parent["graph"], parent["graph"]["arrays"])
        feature_list = []
        for index in range(3):
            row = next(row for row in capture["calls"] if row["frame_index"] == index)
            feature_list.append(load_artifact(args.capture_dir, row["artifact"], row["arrays"])["fmap"])
        fmap = np.ascontiguousarray(np.concatenate(feature_list, axis=1))
        gmap = patch["gmap"]
        validate_value(fmap, (1, 3, 128, 116, 152), "saved fmap")
        validate_value(gmap, (1, 288, 128, 3, 3), "saved gmap")
        require(parent["edge_count"] == 864 and graph["kk"].shape == (864,) and
                np.array_equal(graph["kk"], np.repeat(np.arange(288, dtype=np.int64), 3)) and
                np.array_equal(graph["jj"], np.tile(np.arange(3, dtype=np.int64), 288)) and
                np.array_equal(graph["ii"], patch["source_frame"][graph["kk"]]) and
                np.array_equal(patch["source_frame"], np.repeat(np.arange(3, dtype=np.int64), 96)),
                "Frozen E864 all-pairs edge order/source ownership differs")
        require(np.array_equal(graph["ctx"], patch["contexts"][:, graph["kk"]]) and
                not np.any(graph["net0"]) and np.all(patch["patches"][:, :, 2] == 1),
                "Original context/zero net/unit inverse-depth input differs")
        intrinsics = np.asarray([row["pixel_intrinsics"] for row in capture["inputs"]], np.float32)[None] / np.float32(4)
        require(np.all(intrinsics == intrinsics[:, :1]), "Online CUDA BA common-K contract requires identical views")
        poses = np.zeros((1, 3, 7), np.float32)
        poses[..., 6] = 1
        patches = patch["patches"].copy()
        net = graph["net0"].copy()
        coords = geo.reproject(poses, patches, intrinsics, graph["ii"], graph["jj"], graph["kk"])
        require(np.allclose(coords, graph["coords"], rtol=1e-4, atol=1e-5), "Initial reprojection differs from saved identity graph")
        immutable = {"fmap":(fmap, meta(fmap)), "gmap":(gmap, meta(gmap))}
        immutable.update({"graph/" + key:(value, meta(value)) for key, value in graph.items()})
        immutable.update({"patch_input/" + key:(value, meta(value)) for key, value in patch.items()})
        feature_key = meta(fmap)["tensor_sha256"] + ":" + meta(gmap)["tensor_sha256"]
        cached = native.PreparedCorrelation(args.library, gmap, fmap, threads=4,
                                             cache_key=feature_key, library_sha256=LIBRARY_SHA)
        require(cached.preparation["native"]["neon"] == 1, "Actual ARM NEON build required")
        report.update(cache_preparation=cached.preparation,
                      initial_state=save("initial-state.npz", dict(poses=poses, patches=patches, intrinsics=intrinsics,
                                                                  ii=graph["ii"], jj=graph["jj"], kk=graph["kk"], net=net)),
                      initial_reprojection_vs_frozen_identity=dict(rtol=1e-4, atol=1e-5, passed=True,
                                                                  max_abs=float(np.abs(coords - graph["coords"]).max())),
                      graph_contract=dict(frames=3, patches_per_view=96, total_patches=288, edges=864,
                                          optimized_pose_interval=[1, 3], fixed_pose_ids=[0], feature_resolution=[116, 152],
                                          intrinsics=intrinsics.tolist(), pixel_intrinsics=[row["pixel_intrinsics"] for row in capture["inputs"]],
                                          initialization="identity poses and unit inverse depth; fixed captured graph, not official initialization"))

        def update(previous_net, context, correlation, ii, jj, kk):
            output_net, (delta, weight, unused) = neural.update(
                torch.from_numpy(previous_net), torch.from_numpy(context), torch.from_numpy(correlation),
                None, torch.from_numpy(ii), torch.from_numpy(jj), torch.from_numpy(kk))
            require(unused is None, "Unexpected original Update auxiliary output")
            return tuple(value.detach().cpu().numpy().copy() for value in (output_net, delta, weight))

        def on_stage(stage):
            report["active_stage"] = stage
            report["stage_events"].append(dict(iteration=report["active_round"], stage=stage,
                                               monotonic=time.monotonic()))
            if stage == "BA_completed":
                report["BA_executed"] = True

        with torch.inference_mode():
            for iteration in range(4):
                report["active_round"] = iteration
                print(json.dumps(dict(event="feedback_round_begin", iteration=iteration)), flush=True)
                net, arrays, row = feedback_round(geo, poses, patches, intrinsics, graph, net, cached, update, on_stage)
                row.update(iteration=iteration, recurrent_net_carried_from_previous_round=iteration > 0,
                           geometry_revision_before=iteration, geometry_revision_after=iteration + 1)
                report["calls"].append(row)
                row["artifact"] = save("feedback-round-%02d.npz" % iteration, arrays)
                report["feedback_rounds_completed"] = iteration + 1
                print(json.dumps(dict(event="feedback_round_complete", iteration=iteration,
                                      seconds=row["seconds"], pose_change=row["pose_left_increment_norm"],
                                      inverse_depth_change=row["inverse_depth_change_norm"])), flush=True)
        report.pop("active_round", None)
        report.pop("active_stage", None)
        require(cached.metrics()["completed_calls"] == 4 and cached.metrics()["failed_calls"] == 0,
                "Actual native-correlation call accounting differs")
        report.update(status="completed_fixed_graph_neural_geometry_feedback_not_VO",
                      expected_neural_Update_calls=4, completed_neural_Update_calls=len(report["calls"]),
                      completed_BA_iterations=sum(len(row["ba"]["iterations"]) for row in report["calls"]),
                      final_state=save("final-state.npz", dict(poses=poses, patches=patches, intrinsics=intrinsics, net=net)),
                      round_time_sum_seconds=sum(row["round_total_seconds"] for row in report["calls"]),
                      stage_time_sum_seconds={key:sum(row["seconds"][key] for row in report["calls"]) for key in report["calls"][0]["seconds"]},
                      timing_scope="Each feedback round includes snapshots, NumPy reprojection, full native-correlation wrapper, original Update incl Torch wrapping/output copies, target construction, dense CPU BA(two steps), post-BA reprojection and diagnostic math. Hashing/NPZ/JSON writes, model loading, captured CNN computation and one-time cache preparation are excluded and separately reported where executed.",
                      limitations=["Fixed three captured real views; no new observation admission, official eight-frame initialization, motion initialization, keyframe removal, normalization, loop closure or online lifecycle.",
                                   "Identity poses/unit inverse depth differ from official pre-initialization random depths. This demonstrates the true feedback computation only.",
                                   "NumPy dense fastBA mathematics is synthetically validated but not CUDA-bitwise or CUDA numerical equivalence tested.",
                                   "Learned pixel targets vary every round; their residuals are not GT error or trajectory accuracy. A changing validity mask cannot prove residual improvement.",
                                   "All compute in this probe is CPU, with unchanged original neural parameters. It is not an NPU or 20Hz claim."])
    except Exception as exc:
        report.update(error=repr(exc), traceback=traceback.format_exc())
    finally:
        report["actual_stage_attempts"] = {name:sum(event["stage"] == name + "_begin" for event in report["stage_events"])
                                           for name in ("correlation", "update", "BA")}
        report["completed_BA_calls"] = sum(event["stage"] == "BA_completed" for event in report["stage_events"])
        if cached is not None:
            report["cached_metrics_before_close"] = cached.metrics()
            try:
                start = time.monotonic()
                cached.close()
                report["cache_close_seconds"] = time.monotonic() - start
                report["cached_metrics_after_close"] = cached.metrics()
                require(report["cached_metrics_after_close"]["closed"] and
                        report["cached_metrics_after_close"]["current_context_bytes"] == 0,
                        "Native correlation cache did not close/release")
            except Exception as exc:
                report["cleanup_errors"].append(repr(exc))
        try:
            if neural is not None and state is not None:
                after = {key:meta(value.detach().numpy()) for key, value in neural.state_dict().items()}
                report["original_weights_after"] = after
                report["original_weights_unchanged"] = after == state
                require(after == state, "Original weights changed")
            report["immutable_features_graph_inputs_unchanged"] = all(meta(value) == before for value, before in immutable.values())
            require(report["immutable_features_graph_inputs_unchanged"], "Captured immutable input changed")
            after_pins = {path:sha(path) for path in pins}
            report.update(source_hashes_after=after_pins, sources_and_inputs_unchanged=after_pins == pins)
            require(after_pins == pins, "Pinned source/input changed")
        except Exception as exc:
            report["cleanup_errors"].append(repr(exc))
        neural = None
        gc.collect()
        report["neural_reference_released"] = True
        try:
            os.sched_setaffinity(0, prior_affinity)
            report["affinity_restored"] = os.sched_getaffinity(0) == prior_affinity
            require(report["affinity_restored"], "CPU affinity restoration failed")
        except Exception as exc:
            report["cleanup_errors"].append(repr(exc))
        if report["cleanup_errors"]:
            report["status"] = "failed"
        report.update(source_pins=pins, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        (output / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key:report.get(key) for key in ("status", "error", "feedback_rounds_completed", "cleanup_errors")}), flush=True)
    return 0 if report["status"] == "completed_fixed_graph_neural_geometry_feedback_not_VO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
