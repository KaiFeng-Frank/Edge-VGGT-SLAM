#!/usr/bin/env python3
"""Read completed v30 JSON/NPZ only: no model, image, ORT or BA execution.

This checks saved source identities, lifecycle/topology, current-pose binding,
counts, artifact bytes, reported source/weight immutability and cleanup. It does
not replay the neural network or independently validate trajectory accuracy.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

RGB_SHA = "532884d60bef0c9a05e14bf311998883baf59a2fa47009d1021f2c7937bd69a3"
CAPTURE_SHA = "e0563efb962edb90684480be114b88bc2c7825015f3b13a163f6f47f06360088"
CORE_SHA = "295233513c1b670c953db25bbf4bfad8bdb72b2001f40ac3e1bf5d17083f9bde"
FLOATS = ("poses", "patches", "intrinsics", "net", "target", "weight", "target_inactive", "weight_inactive")
IDS = ("ii", "jj", "kk", "ii_inactive", "jj_inactive", "kk_inactive")
DEFAULT = dict(patches_per_frame=96, removal_window=22, optimization_window=10,
    patch_lifetime=13, keyframe_index=4, keyframe_threshold=15., motion_model="DAMPED_LINEAR",
    motion_damping=.5, capacity=64, observation_capacity=4096, max_active_edges=98304,
    max_inactive_edges=393216, loop_closure=False, classic_loop_closure=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def meta(array):
    return dict(shape=list(array.shape), dtype=str(array.dtype), bytes=array.nbytes,
                tensor_sha256=hashlib.sha256(array.tobytes(order="C")).hexdigest())


def read(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON token: " + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def artifact(directory, item):
    name = item["file"]
    require(isinstance(name, str) and Path(name).name == name, "Unsafe artifact name")
    path = Path(directory) / name
    require(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], "Artifact file SHA/bytes differ")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    require({k: meta(v) for k, v in arrays.items()} == item["arrays"], "Artifact raw array metadata differs")
    return arrays


def exact(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes(order="C") == b.tobytes(order="C")


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def pose_array(value, count):
    a = np.asarray(value, np.float32)
    require(a.shape == (count, 7) and np.isfinite(a).all(), "Saved pose shape/finite differs")
    require(np.all(np.abs(np.linalg.norm(a[:, 3:], axis=-1) - 1) <= 1e-3), "Saved quaternion is not near unit")
    return a


def matrices(poses):
    """Independent NumPy SE3 value conversion, not a geometry update or BA."""
    p = np.asarray(poses, np.float64)
    q = p[:, 3:] / np.linalg.norm(p[:, 3:], axis=1, keepdims=True)
    x, y, z, w = q.T
    out = np.tile(np.eye(4), (len(p), 1, 1))
    out[:, 0, 0] = 1 - 2 * (y*y + z*z)
    out[:, 0, 1] = 2 * (x*y - z*w)
    out[:, 0, 2] = 2 * (x*z + y*w)
    out[:, 1, 0] = 2 * (x*y + z*w)
    out[:, 1, 1] = 1 - 2 * (x*x + z*z)
    out[:, 1, 2] = 2 * (y*z - x*w)
    out[:, 2, 0] = 2 * (x*z - y*w)
    out[:, 2, 1] = 2 * (y*z + x*w)
    out[:, 2, 2] = 1 - 2 * (x*x + y*y)
    out[:, :3, 3] = p[:, :3]
    return out


def source_rows(index_path, manifest, requested):
    require(sha(index_path) == RGB_SHA == manifest["rgb_index_sha256"], "Official RGB index SHA differs")
    expected = []
    for line in Path(index_path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        stamp, relative = line.split()
        require(relative.startswith("rgb/") and ".." not in Path(relative).parts, "RGB index path escaped")
        expected.append((stamp, float(stamp), relative))
        if len(expected) == requested:
            break
    rows = manifest["frames"]
    require(len(rows) == len(expected) == requested, "Requested contiguous source count differs")
    require(manifest["schema"] == "v30_first_contiguous_TUM_RGB_frames" and manifest["GT_read"] is False,
            "Source manifest contract differs")
    for row, (frame_id, stamp, path) in zip(rows, expected):
        require((row["frame_id"], row["timestamp"], row["relative_path"]) == (frame_id, stamp, path),
                "Input identity/order is not first contiguous RGB sequence")
        require(type(row["bytes"]) is int and row["bytes"] > 0 and len(row["sha256"]) == 64,
                "Source image metadata missing")
    require(len({r["frame_id"] for r in rows}) == requested, "Repeated source identity")
    return rows


def lifecycle(rows, inputs, config):
    """Independent integer graph evolution; no neural or geometric computation."""
    require(config == DEFAULT, "This audit pins original-default overlay and explicit research budgets")
    M, radius = config["patches_per_frame"], config["patch_lifetime"]
    require(len(rows) == len(inputs), "Output/source row count differs")
    require(all(b["timestamp"] > a["timestamp"] for a,b in zip(inputs,inputs[1:])),
            "Source timestamps do not strictly increase")
    retained, removed, skipped, fresh, source_updates = [], [], [], [], []
    relative_parents = {}
    ii = jj = kk = np.empty(0, np.int64)
    inactive = {name: np.empty(0, np.int64) for name in ("ii", "jj", "kk")}
    initialized = False
    init_index = None
    expected_calls = []
    offline_relative_ids = []
    previous_publish = None
    for index, (row, source) in enumerate(zip(rows, inputs)):
        result = row["result"]
        require(row["index"] == index and row["frame_id"] == source["frame_id"]
            and row["timestamp"] == source["timestamp"], "Output row source identity differs")
        require(result["frame_id"] == source["frame_id"] and result["timestamp"] == source["timestamp"]
            and result["observation_id"] == index and result["captured_monotonic"] == row["arrival_monotonic"],
            "Current pose source/capture binding differs")
        arrival, publish = row["arrival_monotonic"], row["pose_available_monotonic"]
        require(finite_number(arrival) and finite_number(publish) and publish >= arrival,
                "Arrival/publication time invalid")
        if previous_publish is not None:
            require(arrival >= previous_publish, "Serial source arrived before previous row returned")
        previous_publish = publish
        components = [row[k] for k in ("decode_undistort_crop_seconds", "encode_seconds", "stream_seconds")]
        require(all(finite_number(x) and x >= 0 for x in components), "Negative/nonfinite processing stage")
        require(math.isclose(sum(components), row["processing_seconds"], rel_tol=1e-9, abs_tol=1e-9)
            and math.isclose(publish-arrival, row["processing_seconds"], rel_tol=1e-9, abs_tol=1e-9),
            "Processing time components/arrival scope differ")
        require(finite_number(result["process_seconds"]) and 0 <= result["process_seconds"] <= row["stream_seconds"] + 1e-9,
                "Core timing outside owner stage")
        old_n = len(retained)
        has_probe = old_n > 0 and not initialized
        if has_probe:
            require(finite_number(result["motion_probe"]) and result["motion_probe"] >= 0, "Motion probe value missing")
            expected_calls.append(dict(edges=M, kind="motion_probe", observation_id=index,
                source_ids=[retained[-1]], target_ids=[index]))
        else:
            require(result["motion_probe"] is None, "Unexpected motion probe after initialization")
        admitted = not has_probe or result["motion_probe"] >= 2.
        require(result["admitted"] is admitted, "Motion admission does not match original threshold")
        if not admitted:
            require(not result["fresh_pose_valid"] and not result["initialized"] and result["pose_w2c"] is None
                and result["updates"] == 0 and result["reason"] == "initial_motion_below_2", "Skipped frame counted as fresh")
            skipped.append(index)
            offline_relative_ids.append(index)
            relative_parents[index] = index-1
            continue
        retained.append(index)
        n = len(retained)
        require(n < config["capacity"], "Original spare-slot capacity guard violated")
        # Construct by source-frame pairs, patch-major within each append block.
        forward_patches = np.arange(max(n-radius, 0)*M, (n-1)*M, dtype=np.int64)
        backward_patches = np.arange((n-1)*M, n*M, dtype=np.int64)
        backward_targets = np.arange(max(n-radius, 0), n, dtype=np.int64)
        added_kk = np.r_[forward_patches, np.repeat(backward_patches, len(backward_targets))]
        added_jj = np.r_[np.full(len(forward_patches), n-1, np.int64), np.tile(backward_targets, M)]
        ii, jj, kk = np.r_[ii, added_kk//M], np.r_[jj, added_jj], np.r_[kk, added_kk]
        require(len(ii) <= config["max_active_edges"], "Active edge budget exceeded")
        if n == 8 and not initialized:
            initialized, init_index, updates = True, index, 12
            require(len(ii) == 6144, "Default eight-view initialization is not E6144")
        else:
            updates = int(initialized)
        require(result["updates"] == updates and result["initialized"] is initialized
            and result["fresh_pose_valid"] is bool(updates), "Init12/subsequent1/fresh rule differs")
        require(len(result.get("update_diagnostics", [])) == updates, "Update diagnostic count differs")
        for diagnostic in result.get("update_diagnostics", []):
            interval = [max(n-config["optimization_window"] if initialized else 1, 1), n]
            require(diagnostic["edges"] == len(ii) and diagnostic["iterations"] == 2
                and diagnostic["optimized_pose_interval"] == interval, "BA edge/interval/two-step metadata differs")
            inside = diagnostic["diagnostic"]
            require(inside["optimized_pose_interval"] == interval and len(inside["iterations"]) == 2
                and inside["upstream_eff_impl_requested"] is False, "BA inner diagnostic differs")
            for iteration, detail in enumerate(inside["iterations"]):
                require(detail["iteration"] == iteration and 0 <= detail["active_edges"] <= len(ii)
                    and all(finite_number(detail[k]) and detail[k] >= 0 for k in
                        ("weighted_squared_residual_before", "pose_update_norm", "inverse_depth_update_norm")),
                    "BA numeric diagnostic invalid")
            expected_calls.append(dict(edges=len(ii), kind="fresh_update", observation_id=index,
                source_ids=[retained[x] for x in np.unique(ii)], target_ids=[retained[x] for x in np.unique(jj)]))
        source_updates.extend([index]*updates)
        if updates:
            pose_array([result["pose_w2c"]], 1)
            require(result["reason"] == "updated_current_observation", "Fresh pose reason differs")
            fresh.append(index)
        else:
            require(result["pose_w2c"] is None and result["reason"] == "initialization_wait", "Seed pose counted as current")
        keyframe = result["keyframe"]
        if initialized and index != init_index:
            require(isinstance(keyframe, dict) and finite_number(keyframe["mean_bidirectional_flow"]), "Keyframe decision missing")
            k = n - config["keyframe_index"]
            deleted = keyframe["mean_bidirectional_flow"] < config["keyframe_threshold"]
            require(keyframe["removed_observation_id"] == (retained[k] if deleted else None), "Wrong keyframe source removal")
            if deleted:
                relative_parents[retained[k]] = retained[k-1]
                removed.append(retained[k]); offline_relative_ids.append(retained[k]); retained.pop(k)
                keep = (ii != k) & (jj != k)
                ii, jj, kk = ii[keep], jj[keep], kk[keep]
                moved = ii > k
                kk = kk - M*moved.astype(np.int64)
                ii, jj = ii - moved.astype(np.int64), jj - (jj > k).astype(np.int64)
                require(all(not np.any(x >= k) for x in (inactive["ii"], inactive["jj"])), "Inactive edge intersects removed keyframe")
            aged = kk//M < len(retained)-config["removal_window"]
            for name, values in (("ii", ii), ("jj", jj), ("kk", kk)):
                inactive[name] = np.r_[inactive[name], values[aged]]
            require(len(inactive["ii"]) <= config["max_inactive_edges"], "Inactive edge budget exceeded")
            ii, jj, kk = ii[~aged], jj[~aged], kk[~aged]
        else:
            require(keyframe is None, "Keyframe removal before initialization completion")
        require(result["retained_frames"] == len(retained) and result["active_edges"] == len(ii), "Saved frame/edge counts differ")
        require(retained[-1] == index, "Latest pose slot no longer current source")
    return dict(retained=retained, removed=removed, skipped=skipped, fresh=fresh,
        ii=ii, jj=jj, kk=kk, inactive=inactive, init_index=init_index, initialized=initialized,
        expected_calls=expected_calls, source_updates=source_updates, relative_ids=sorted(offline_relative_ids),
        relative_parents=relative_parents)


def snapshot_check(saved, values, structure, inputs, *, status):
    metadata = saved["metadata"]
    N, M, E = len(structure["retained"]), 96, len(structure["ii"])
    require(set(values) == set(FLOATS+IDS), "Snapshot array fields differ")
    for name in FLOATS:
        require(values[name].dtype == np.float32 and np.isfinite(values[name]).all(), "Snapshot floating arrays nonfinite/type")
    for name in IDS:
        require(values[name].dtype == np.int64 and values[name].ndim == 1, "Snapshot index type/shape")
    require(values["poses"].shape == (1,N,7) and values["patches"].shape == (1,N*M,3,3,3)
        and values["intrinsics"].shape == (1,N,4), "Snapshot frame geometry shape")
    pose_array(values["poses"][0], N)
    require(values["net"].shape == (1,E,384) and values["target"].shape == values["weight"].shape == (1,E,2), "Snapshot recurrent/edge arrays")
    I = len(structure["inactive"]["ii"])
    require(values["target_inactive"].shape == values["weight_inactive"].shape == (1,I,2), "Inactive factor shape")
    for name in ("weight", "weight_inactive"):
        require(np.all((values[name] >= 0) & (values[name] <= 1)), "Snapshot weight codomain differs")
    for name in ("ii", "jj", "kk"):
        require(exact(values[name], structure[name]) and exact(values[name+"_inactive"], structure["inactive"][name]),
                "Final forward/back/removal graph order differs")
    for ii, jj, kk in ((values["ii"],values["jj"],values["kk"]),
                      (values["ii_inactive"],values["jj_inactive"],values["kk_inactive"])):
        require(np.array_equal(ii,kk//M) and np.all(ii>=0) and np.all(jj>=0) and np.all(kk>=0)
            and np.all(ii<N) and np.all(jj<N) and np.all(kk<N*M), "Snapshot graph source/range invariant")
    require(np.all(values["intrinsics"] == values["intrinsics"][:, :1]) and np.all(values["intrinsics"][..., :2]>0), "Common feature-grid K differs")
    require(metadata["status"] == status and metadata["initialized"] is structure["initialized"], "Snapshot lifecycle status")
    require(metadata["retained_observation_ids"] == structure["retained"]
        and metadata["retained_frame_ids"] == [inputs[i]["frame_id"] for i in structure["retained"]], "Snapshot retained source IDs differ")
    require(sorted(map(int,metadata["relative"])) == structure["relative_ids"], "Skipped/deleted relative map differs")
    for key,parent in structure["relative_parents"].items():
        relation = metadata["relative"][str(key)]
        require(relation[0] == parent, "Relative pose parent does not match original source lifecycle")
        pose_array([relation[1]],1)
        if key in structure["skipped"]:
            require(exact(np.asarray(relation[1],np.float32),np.array([0,0,0,0,0,0,1],np.float32)),
                    "Motion-skipped observation is not identity-relative")
    require(len(metadata["observations"]) == len(inputs), "Snapshot source history count differs")
    for i, (row, source) in enumerate(zip(metadata["observations"],inputs)):
        require(row["frame_id"]==source["frame_id"] and row["timestamp"]==source["timestamp"] and row["observation_id"]==i,
                "Snapshot observation identity differs")
    return metadata


def termination_check(result, before, final, structure, inputs, rows, relative):
    require(result["offline_updates"] == 12 and result["fresh_pose_valid"] is False and result["online_publication"] is False,
            "EOF updates counted as fresh or omitted")
    require(result["initialized"] is structure["initialized"] and result["trajectory_state"] ==
        ("initialized_offline_reconstruction" if structure["initialized"] else "initialization_incomplete"), "EOF initialization tag differs")
    require(result["reconstructed_poses_counted_as_fresh"] is False and result["reconstructed_observation_ids"] == structure["relative_ids"],
            "Reconstruction counted as fresh or mismatched")
    require(len(result["update_diagnostics"]) == 12, "EOF diagnostic count differs")
    N,E=len(structure["retained"]),len(structure["ii"])
    for diag in result["update_diagnostics"]:
        require(diag["edges"]==E and diag["iterations"]==2 and diag["optimized_pose_interval"] ==
            [max(N-10 if structure["initialized"] else 1,1),N], "EOF BA factor/interval differs")
        require(diag["diagnostic"]["optimized_pose_interval"] == diag["optimized_pose_interval"]
            and len(diag["diagnostic"]["iterations"]) == 2
            and diag["diagnostic"]["upstream_eff_impl_requested"] is False, "EOF BA inner metadata differs")
    require(len(result["observations"]) == len(inputs), "Offline source history count differs")
    for row, source in zip(result["observations"],inputs):
        require(row["frame_id"]==source["frame_id"] and row["timestamp"]==source["timestamp"], "Offline source identity differs")
    w2c=pose_array(result["pose_w2c"],len(inputs));c2w=pose_array(result["pose_c2w"],len(inputs))
    require(np.allclose(matrices(w2c) @ matrices(c2w), np.eye(4),rtol=1e-4,atol=1e-5), "Offline inverse-pose relation differs")
    for slot,index in enumerate(structure["retained"]):
        require(exact(w2c[index],final["poses"][0,slot]), "Retained offline pose bytes differ from final snapshot")
    w2c_matrix = matrices(w2c)
    for key,parent in structure["relative_parents"].items():
        relation = np.asarray(relative[str(key)][1],np.float32)
        require(np.allclose(w2c_matrix[key],matrices(relation[None])[0] @ w2c_matrix[parent],
                rtol=1e-4,atol=1e-5), "Offline relative-pose reconstruction order differs")
    require(exact(np.asarray(rows[-1]["result"]["pose_w2c"],np.float32),before["poses"][0,-1]) if structure["fresh"] else True,
            "Last fresh current-source pose is not before-EOF snapshot pose")
    return w2c


def correlation_check(corr, calls, inputs):
    require(corr["attempted_calls"]==corr["completed_calls"]==len(corr["calls"])==len(calls)
        and corr["current_context_open"] is False,"Native correlation count/close differs")
    require(len(corr["closed_contexts"])==len(corr["generations"]),"Not every native generation released")
    for item in corr["closed_contexts"]:
        require(item["closed"] is True and item["current_context_bytes"]==0 and item["failed_calls"]==0,"Native cache release/error differs")
    # Cache identity is the ordered members, not transient graph-slot numbers.
    for i,(actual,expected) in enumerate(zip(corr["calls"],calls)):
        generation=corr["generations"][actual["generation"]]
        source=[inputs[x]["frame_id"] for x in expected["source_ids"]]
        target=[inputs[x]["frame_id"] for x in expected["target_ids"]]
        require(actual["index"]==i and actual["edges"]==expected["edges"] and generation["source_frame_ids"]==source
            and generation["target_frame_ids"]==target,"Native subset source identities differ after graph remap")
        require(json.loads(generation["key"])==dict(source=source,target=target,patches_per_frame=96),"Native cache identity key differs")


def audit(args):
    run_dir=Path(args.run_dir);report=read(run_dir/"run.json")
    require(report["schema"]=="v30_DPVO_fresh_frame_CPU_replay" and report["status"] in
        ("completed_unpaced_fresh_frame_CPU_VO_research_not_realtime","completed_unpaced_initialization_incomplete_no_fresh_pose"),
        "Completed original-lifecycle run required")
    checks=[]
    require(report["frames_requested"]==report["frames_completed"]==len(report["frames"])==args.expected_frames,
            "Not all requested source observations completed")
    manifest=read(run_dir/"input-manifest.json")
    require(sha(run_dir/"input-manifest.json")==report["input_manifest_sha256"], "Input manifest SHA differs")
    inputs=source_rows(args.rgb_index,manifest,args.expected_frames);checks.append("official_contiguous_source_identity_and_order")
    structure=lifecycle(report["frames"],inputs,report["config"]);checks.append("all_current_rows_and_original_lifecycle_graph_evolution")
    snapshots={}
    for key,status in (("snapshot_before_termination","running"),("final_snapshot","terminated")):
        values=artifact(run_dir,report[key]["artifact"])
        snapshot_check(report[key],values,structure,inputs,status=status)
        snapshots[key]=values;checks.append(key+"_file_and_raw_metadata_topology")
    before,final=snapshots["snapshot_before_termination"],snapshots["final_snapshot"]
    for name in IDS+("intrinsics","target_inactive","weight_inactive"):
        require(exact(before[name],final[name]), "EOF changed factor membership or inactive frozen data")
    checks.append("EOF_factor_membership_and_inactive_values_unchanged")
    before_meta,final_meta = (report[k]["metadata"] for k in ("snapshot_before_termination","final_snapshot"))
    require(before_meta["relative"] == final_meta["relative"], "EOF altered frozen removed-frame relationships")
    termination_check(report["termination"],before,final,structure,inputs,report["frames"],final_meta["relative"])
    checks.append("EOF_reconstruction_separate_from_current_source_fresh_poses")
    counts=dict(observations=args.expected_frames,admitted=args.expected_frames-len(structure["skipped"]),
        motion_skipped=len(structure["skipped"]),fresh_poses=len(structure["fresh"]),
        update_calls=len(structure["source_updates"]),motion_probe_calls=sum(c["kind"]=="motion_probe" for c in structure["expected_calls"]),
        BA_calls=len(structure["source_updates"]),BA_iterations=2*len(structure["source_updates"]),
        deleted_keyframes=len(structure["removed"]),stored_inactive_edges=len(structure["inactive"]["ii"]),offline_updates=0)
    for key,value in counts.items():
        require(report["metrics_before_termination"][key]==value,"Before-EOF core counter differs: "+key)
        expected=value+(12 if key in ("update_calls","BA_calls","offline_updates") else 24 if key=="BA_iterations" else 0)
        require(report["stream_metrics"][key]==expected,"After-EOF core counter differs: "+key)
    require(report["stream_metrics_at_exit"]==report["stream_metrics"],"Core state changed at cleanup")
    require(report["random_depth_draw_calls"]==args.expected_frames,"Random depth draw omitted on initialized/rejected observations")
    checks.append("per_observation_Update_BA_random_and_fresh_counts")
    calls=list(structure["expected_calls"])+[dict(edges=len(structure["ii"]),kind="offline_update",
        source_ids=[structure["retained"][i] for i in np.unique(structure["ii"])],
        target_ids=[structure["retained"][i] for i in np.unique(structure["jj"])]) for _ in range(12)]
    require(len(report["update_calls"])==len(calls),"Original Update callback count differs")
    for i,(actual,expected) in enumerate(zip(report["update_calls"],calls)):
        require(actual["index"]==i and actual["edges"]==expected["edges"] and finite_number(actual["seconds"])
            and actual["seconds"]>=0 and 0<=actual["weight_min"]<=actual["weight_max"]<=1,"Update callback count/edge/domain/timing differs")
    checks.append("actual_callback_order_and_edge_counts")
    corr=report["correlation"]
    correlation_check(corr,calls,inputs)
    checks.append("compact_native_source_target_identity_and_all_contexts_released")
    capture_path=Path(args.encoder_capture_dir)/"capture.json"
    require(sha(capture_path)==CAPTURE_SHA,"Original encoder capture pin differs")
    capture=read(capture_path)
    require(len(report["original_weights_before"])==94 and report["original_weights_before"]==report["original_weights_after"]==capture["state_before"]
        and report["original_weights_unchanged"] is True,"Original 94 parameter metadata differs")
    checks.append("original_parameter_metadata_coverage_and_unchanged")
    pins=report["source_pins"]
    require(pins==report["source_hashes_after"] and report["sources_and_inputs_unchanged"] is True,"Board source/input before-after differs")
    for row in inputs:
        matches=[digest for path,digest in pins.items() if path.endswith('/'+row["relative_path"])]
        require(matches==[row["sha256"]],"Input image SHA missing from actual board source pins")
    local={}
    for name in ("dpvo_stream.py","v30_frame_encoder.py","run-v30-dpvo-stream-board.py","source-freeze.json"):
        path=Path(args.source_dir)/name;digest=sha(path);matches=[v for p,v in pins.items() if Path(p).name==name]
        require(matches==[digest],"Archived/local actual implementation pin differs: "+name)
        local[name]=digest
    require(local["dpvo_stream.py"]==CORE_SHA and report["script_sha256"]==local["run-v30-dpvo-stream-board.py"],"Frozen core or actual entrypoint differs")
    frozen=read(Path(args.source_dir)/"source-freeze.json")
    require(frozen["schema"]=="v30_stream_source_freeze" and all(frozen["sources"][n]==local[n] for n in local if n!='source-freeze.json'),"Frozen source manifest differs")
    checks.append("local_implementation_bytes_and_board_reported_source_immutability")
    encoder=report["encoder_metrics"]
    require(encoder["closed"] is True and encoder["completed_frames"]==args.expected_frames and encoder["source_recheck_passed"] is True,
            "Encoder execution count/close/source differs")
    require(all(report["encoder_sources_after"][p]==pins[p] for p in report["encoder_sources_after"]),"Encoder source pins differ")
    require(report["cleanup_errors"]==[] and report["affinity_restored"] is True,"Cleanup or affinity restore failed")
    for key,value in dict(ORB_used=False,added_LK_PnP_frontend=False,NPU_used=False,GT_or_sensor_depth_input=False,
        unpaced_file_replay=True,physical_camera=False,realtime_validated=False,accuracy_vs_VGGT_validated=False,
        different_model_from_VGGT=True,original_VGGT_default_unchanged=True).items():
        require(report[key] is value,"Provenance/validation flag differs: "+key)
    checks.append("encoder_native_cleanup_and_explicit_research_scope")
    for index in range(3):
        row=report["frames"][index]
        require(row["original_input_raw_bytes_match"] is True,"First-three input guard not exercised")
        require(set(row["first_three_original_capture_gate"])=={'fmap','gmap','context','patches'},"First-three feature gate coverage differs")
        for gate in row["first_three_original_capture_gate"].values():
            require(gate["rtol"]==1e-4 and gate["atol"]==1e-5 and gate["failed_elements"]==0,"First-three original gate failed or altered")
    checks.append("reported_first_three_original_input_feature_guards")
    fresh_times=[report["frames"][i]["processing_seconds"] for i in structure["fresh"]]
    return dict(schema='v30_DPVO_saved_stream_independent_audit',status='passed',checked=len(checks),failed=0,checks=checks,
        audit_sha256=sha(__file__),run_sha256=sha(run_dir/'run.json'),input_manifest_sha256=sha(run_dir/'input-manifest.json'),
        original_encoder_capture_sha256=CAPTURE_SHA,local_implementation_sha256=local,
        snapshot_sha256={k:report[k]['artifact']['sha256'] for k in snapshots},
        requested_inputs=args.expected_frames,admitted_inputs=counts['admitted'],motion_skipped=structure['skipped'],
        initialization_source_index=structure['init_index'],fresh_source_indices=structure['fresh'],fresh_poses=len(structure['fresh']),
        deleted_keyframes=len(structure['removed']),retained_observation_ids=structure['retained'],
        active_edges=len(structure['ii']),inactive_edges=len(structure['inactive']['ii']),
        online_Update_calls=len(structure['source_updates']),motion_probe_calls=counts['motion_probe_calls'],EOF_Update_calls=12,
        total_original_Update_calls=len(calls),BA_iterations=2*(len(structure['source_updates'])+12),
        native_generations=len(corr['generations']),all_native_contexts_released=True,
        original_parameter_tensors=94,source_and_weight_metadata_unchanged=True,
        fresh_processing_seconds={'p50':float(np.quantile(fresh_times,.5)),'p95':float(np.quantile(fresh_times,.95)),
            'max':float(np.max(fresh_times))} if fresh_times else None,
        host_operations='JSON and saved NPZ/NumPy values only; no model, image, ORT, correlation, geometry update or BA execution',
        board_only_source_hashes='Board before/after equality checked; host independently rehashed only specified local implementation, RGB index and saved artifacts',
        first_three_input_feature_checks='Board-reported original gate flags checked; this audit did not decode images or rerun encoders',
        per_frame_recurrent_values='Lifecycle topology/counts and final net verified; per-frame neural arrays are not independently replayed',
        fresh_pose_definition='Current admitted initialized source row with reported successful Update/BA; no independent metric accuracy claim',
        unpaced_file_replay=True,NPU_used=False,trajectory_accuracy_validated=False,realtime_20Hz_validated=False,
        physical_camera_latency_validated=False,EOF_or_reconstructed_counted_as_fresh=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir','rgb-index','encoder-capture-dir','source-dir','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-frames',type=int,default=192)
    args=p.parse_args(argv)
    require(not args.output.exists(),'Audit output must be new; preserve previous receipts')
    try:
        result=audit(args)
    except Exception as error:
        result=dict(schema='v30_DPVO_saved_stream_independent_audit',status='failed',audit_sha256=sha(__file__),
                    error=type(error).__name__+': '+str(error),host_model_execution=False)
    with args.output.open('x') as file:
        json.dump(result,file,indent=2,allow_nan=False);file.write('\n')
    print(json.dumps({k:result.get(k) for k in ('status','checked','failed','error','fresh_poses')}))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
