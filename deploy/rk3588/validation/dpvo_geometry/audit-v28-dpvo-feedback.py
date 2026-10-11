#!/usr/bin/env python3
"""NumPy-only saved-array audit; never calls a model, correlation or BA."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

GEOMETRY_SHA = "13eb1e03eb4a5ed1bd1533549f2b9f4994abfaa61e0b55304b99e6fe77d813df"
DRIVER_SHA = "e35a2bde3154f527a010b0cf7a2f3192bfe6e356ffd3f6b6259a9289e6bbed42"
CAPTURE_SHA = "e0563efb962edb90684480be114b88bc2c7825015f3b13a163f6f47f06360088"
MOTION_SHA = "55a7db19d1a9bfbf986da62ed999fc8b122add0fca84f01681e6ad0b82ba4e86"
BUILD_SHA = "34fbf049852b351e7df1097cc806fbf83e1e1035e18b01ca4e9e8ba39499cd18"
SELECTED_EDGES = [0, 1, 2, 143, 432, 861, 863]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def meta(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes,
                tensor_sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest())


def load_artifact(folder, artifact):
    name = artifact["file"]
    require(Path(name).name == name and name not in ("", ".", ".."), "Unsafe artifact basename")
    path = folder / name
    require(path.stat().st_size == artifact["bytes"] and sha(path) == artifact["sha256"], "Artifact file SHA/bytes differ")
    with np.load(path, allow_pickle=False) as store:
        arrays = {key:store[key].copy() for key in store.files}
    require(set(arrays) == set(artifact["arrays"]) and
            all(meta(value) == artifact["arrays"][key] for key, value in arrays.items()),
            "Artifact tensor SHA/shape/dtype differ")
    return arrays


def quat_product(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw*bx + ax*bw + ay*bz - az*by,
                     aw*by + ay*bw + az*bx - ax*bz,
                     aw*bz + az*bw + ax*by - ay*bx,
                     aw*bw - ax*bx - ay*by - az*bz])


def rotation(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def selected_projection(poses, patches, K, ii, jj, kk, edge, *, raw_fastba=False):
    """Independent FP64 arithmetic on one saved edge, not a BA solve.

Normalized mode is pinned LieTorch/projective_ops. Raw mode follows
CUDA relSE3's single relative-quaternion action, with no normalization.
Numerical differences versus the CPU reference's equivalent group
composition are exposed under the unchanged FP32 arithmetic gate.
"""
    source, destination, point = int(ii[edge]), int(jj[edge]), int(kk[edge])
    pi, pj = poses[0, source].astype(np.float64), poses[0, destination].astype(np.float64)
    qi, qj = pi[3:].copy(), pj[3:].copy()
    if not raw_fastba:
        qi /= np.linalg.norm(qi)
        qj /= np.linalg.norm(qj)
    inverse = qi.copy()
    inverse[:3] *= -1
    relative = quat_product(qj, inverse)
    if not raw_fastba:
        relative /= np.linalg.norm(relative)
    R = rotation(relative)
    translation = pj[:3] - R @ pi[:3]
    sample = patches[0, point].astype(np.float64)
    source_K = K[0, 0 if raw_fastba else source].astype(np.float64)
    target_K = K[0, 0 if raw_fastba else destination].astype(np.float64)
    rays = np.stack(((sample[0]-source_K[2])/source_K[0],
                     (sample[1]-source_K[3])/source_K[1], np.ones((3, 3))))
    camera = np.einsum("ij,jhw->ihw", R, rays) + translation[:, None, None] * sample[2]
    z = camera[2]
    denominator = z if raw_fastba else np.maximum(z, .1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = camera[:2] / denominator[None] * target_K[:2, None, None] + target_K[2:, None, None]
    return pixels, z


def residual_summary(residual, active, weight, selected=None):
    error = residual[0].astype(np.float64)
    finite = np.isfinite(error).all(axis=-1)
    chosen = active if selected is None else selected
    mask = chosen & finite
    norm = np.linalg.norm(error[finite], axis=-1)
    return dict(active_edges=int(active.sum()), finite_residual_edges=int(finite.sum()),
                nonfinite_residual_edges=int((~finite).sum()), selected_edges=int(mask.sum()),
                all_finite_residual_norm_mean=float(norm.mean()) if norm.size else None,
                all_finite_residual_norm_max=float(norm.max()) if norm.size else None,
                weighted_squared_residual=float(np.sum(weight[0, mask].astype(np.float64)*error[mask]**2)),
                selection="current online gate" if selected is None else "fixed pre-BA active mask, finite rows only")


def audit(run_dir, capture_dir, motion_dir, source_dir, build_path):
    run = json.loads((run_dir/"run.json").read_text())
    require(sha(capture_dir/"capture.json") == CAPTURE_SHA and
            sha(motion_dir/"motion.json") == MOTION_SHA and sha(build_path) == BUILD_SHA, "Historical reference chain differs")
    capture = json.loads((capture_dir/"capture.json").read_text())
    motion = json.loads((motion_dir/"motion.json").read_text())
    build = json.loads(build_path.read_text())
    require(sha(source_dir/"dpvo_cpu_geometry.py") == GEOMETRY_SHA and
            sha(source_dir/"run-v28-dpvo-feedback-board.py") == DRIVER_SHA, "Measured source pin differs")
    checks = []
    def check(name, condition):
        checks.append({"name":name, "passed":bool(condition)})
        require(condition, name)
    check("completed actual CPU feedback, not VO/NPU/trajectory", run["status"] == "completed_fixed_graph_neural_geometry_feedback_not_VO" and
          run["actual_board_execution"] and not run["NPU_used"] and not run["CUDA_used"] and not run["ORB_used"] and
          not run["complete_VO_executed"] and not run["realtime_validated"] and not run["trajectory_accuracy_validated"])
    check("measured source/capture/motion/native build binding", run["script_sha256"] == DRIVER_SHA and run["geometry_sha256"] == GEOMETRY_SHA and
          run["capture_sha256"] == CAPTURE_SHA and run["motion_sha256"] == MOTION_SHA and run["build_sha256"] == BUILD_SHA and
          run["library_sha256"] == build["output"]["sha256"])
    check("four calls and eight BA steps", len(run["calls"]) == run["feedback_rounds_completed"] == run["completed_neural_Update_calls"] == 4 and
          run["completed_BA_iterations"] == 8 and run["actual_stage_attempts"] == {"correlation":4,"update":4,"BA":4} and run["completed_BA_calls"] == 4)
    check("weights94 raw bytes unchanged", len(run["original_weights_before"]) == 94 and
          run["original_weights_before"] == run["original_weights_after"] == capture["state_before"] == motion["original_weights_before"])
    check("source/immutable inputs/affinity/cleanup", run["source_pins"] == run["source_hashes_after"] and
          run["sources_and_inputs_unchanged"] and run["immutable_features_graph_inputs_unchanged"] and
          run["original_weights_unchanged"] and run["affinity_restored"] and run["neural_reference_released"] and run["cleanup_errors"] == [])
    cache = run["cached_metrics_after_close"]
    check("actual four native queries, ownership and release", cache["completed_calls"] == cache["attempts"] == 4 and cache["failed_calls"] == 0 and
          cache["closed"] and cache["current_context_bytes"] == 0 and cache["preparation"]["feature_inputs_copied"] and
          cache["preparation"]["native"]["neon"] == 1 and cache["library_sha256"] == build["output"]["sha256"])
    initial, final = load_artifact(run_dir,run["initial_state"]), load_artifact(run_dir,run["final_state"])
    check("initial identity/unit depth/zero net", initial["poses"].shape == (1,3,7) and
          np.array_equal(initial["poses"], np.broadcast_to(np.array([0,0,0,0,0,0,1],np.float32),(1,3,7))) and
          np.all(initial["patches"][:,:,2] == 1) and not np.any(initial["net"]))
    ii,jj,kk = (initial[name] for name in ("ii","jj","kk"))
    K = initial["intrinsics"]
    expected_K = np.asarray([row["pixel_intrinsics"] for row in capture["inputs"]],np.float32)[None]/np.float32(4)
    check("E864 ordered allpairs and original RES4 common intrinsics", np.array_equal(ii,np.repeat(np.arange(3,dtype=np.int64),96)[kk]) and
          np.array_equal(kk,np.repeat(np.arange(288,dtype=np.int64),3)) and np.array_equal(jj,np.tile(np.arange(3,dtype=np.int64),288)) and
          np.array_equal(K,expected_K) and np.all(K == K[:,:1]))
    previous = {"poses_after":initial["poses"],"patches_after":initial["patches"],"net_after":initial["net"]}
    rows=[]
    max_projection_error=0.
    for iteration,row in enumerate(run["calls"]):
        arrays = load_artifact(run_dir,row["artifact"])
        check("round%d sequence/current state/net carry"%iteration, row["iteration"] == iteration and row["geometry_revision_before"] == iteration and
              row["geometry_revision_after"] == iteration+1 and row["recurrent_net_carried_from_previous_round"] == (iteration>0) and
              all(np.array_equal(arrays[key+"_before"],previous[key+"_after"]) for key in ("poses","patches","net")))
        check("round%d all state/neural arrays FP32 finite"%iteration, all(arrays[key].dtype == np.float32 and np.isfinite(arrays[key]).all()
              for key in ("poses_before","poses_after","patches_before","patches_after","net_before","net_after","coords_before","coords_after","corr","delta","weight","target")))
        check("round%d exact current target construction"%iteration, np.array_equal(arrays["target"],arrays["coords_before"][:,:,:,1,1]+arrays["delta"]))
        check("round%d weight range/fixed pose/xy and shared positive depths"%iteration,
              np.all((arrays["weight"]>=0)&(arrays["weight"]<=1)) and np.array_equal(arrays["poses_after"][0,0],arrays["poses_before"][0,0]) and
              np.array_equal(arrays["patches_after"][:,:,:2],arrays["patches_before"][:,:,:2]) and
              np.all(arrays["patches_after"][:,:,2] > 0) and np.all(arrays["patches_after"][:,:,2] == arrays["patches_after"][:,:,2,:1,:1]))
        check("round%d correct BA2/fixed interval and dense reference scope"%iteration, row["ba"]["optimized_pose_interval"] == [1,3] and
              len(row["ba"]["iterations"]) == 2 and row["ba"]["dense_reference"] and not row["ba"]["CUDA_bitwise_equivalence_validated"])
        selected_details=[]
        for suffix in ("before","after"):
            check("round%d exact saved residual from saved projection %s"%(iteration,suffix),
                  np.array_equal(arrays["residual_"+suffix],arrays["target"]-arrays["projection_"+suffix],equal_nan=True))
            for edge in SELECTED_EDGES:
                xy,z = selected_projection(arrays["poses_"+suffix],arrays["patches_"+suffix],K,ii,jj,kk,edge)
                raw,raw_z = selected_projection(arrays["poses_"+suffix],arrays["patches_"+suffix],K,ii,jj,kk,edge,raw_fastba=True)
                stored = arrays["coords_"+suffix][0,edge].astype(np.float64)
                projected = arrays["projection_"+suffix][0,edge].astype(np.float64)
                check("round%d independent 3x3 LieTorch reprojection %s edge%d"%(iteration,suffix,edge),np.allclose(stored,xy,rtol=1e-4,atol=1e-5))
                check("round%d independent CUDA raw center projection %s edge%d"%(iteration,suffix,edge),np.allclose(projected,raw[:,1,1],rtol=1e-4,atol=1e-5,equal_nan=True))
                residual = arrays["residual_"+suffix][0,edge].astype(np.float64)
                expected_active = np.linalg.norm(residual)<128 and raw_z[1,1]>.2 and projected[0]>-64 and projected[1]>-64 and projected[0]<2*K[0,0,2]+64 and projected[1]<2*K[0,0,3]+64
                check("round%d selected original BA validity gate %s edge%d"%(iteration,suffix,edge),bool(arrays["active_"+suffix][edge]) == bool(expected_active))
                error=float(np.max(np.abs(stored-xy)))
                max_projection_error=max(max_projection_error,error)
                selected_details.append(dict(state=suffix,edge=edge,correlation_reprojection_max_abs=error,
                                             raw_center_projection_max_abs=float(np.max(np.abs(projected-raw[:,1,1])))))
        before = residual_summary(arrays["residual_before"],arrays["active_before"],arrays["weight"])
        after = residual_summary(arrays["residual_after"],arrays["active_after"],arrays["weight"])
        fixed = residual_summary(arrays["residual_after"],arrays["active_after"],arrays["weight"],arrays["active_before"])
        check("round%d exact independent residual summaries/current versus fixed mask"%iteration,
              before == row["residual_before"] and after == row["residual_after"] and fixed == row["residual_after_using_pre_BA_active_mask"])
        check("round%d real monotonic stage boundaries"%iteration,row["end_monotonic"]>row["begin_monotonic"] and all(v>=0 for v in row["seconds"].values()) and
              abs(sum(row["seconds"].values())-row["round_total_seconds"])<1e-10 and abs(row["end_monotonic"]-row["begin_monotonic"]-row["round_total_seconds"])<1e-10)
        depth_change = arrays["patches_after"][:,:,2,1,1].astype(np.float64)-arrays["patches_before"][:,:,2,1,1].astype(np.float64)
        rows.append(dict(iteration=iteration,seconds=row["seconds"],round_total_seconds=row["round_total_seconds"],
                         pose_translation_change_norm=float(np.linalg.norm(arrays["poses_after"][...,:3].astype(np.float64)-arrays["poses_before"][...,:3].astype(np.float64))),
                         recurrent_net_change_norm=float(np.linalg.norm(arrays["net_after"].astype(np.float64)-arrays["net_before"].astype(np.float64))),
                         inverse_depth_change_norm=float(np.linalg.norm(depth_change)),changed_depths=int(np.count_nonzero(depth_change)),
                         residual_before=before,residual_after_current_gate=after,residual_after_fixed_pre_BA_mask=fixed,
                         pre_BA_active_edges_dropped_after_BA=int(np.count_nonzero(arrays["active_before"] & ~arrays["active_after"])),
                         selected_independent_geometry=selected_details))
        previous=arrays
    check("exact final state continuity",all(np.array_equal(final[key],previous[key+"_after"]) for key in ("poses","patches","net")) and np.array_equal(final["intrinsics"],K))
    sums={key:sum(row["seconds"][key] for row in rows) for key in rows[0]["seconds"]}
    check("all stage timing sums match source receipt", sums == run["stage_time_sum_seconds"] and sum(row["round_total_seconds"] for row in rows) == run["round_time_sum_seconds"])
    return dict(schema="v28_DPVO_saved_feedback_independent_review",status="passed",audit_sha256=sha(__file__),
                checked=len(checks),failed=0,checks=checks,
                source_sha256={"driver":DRIVER_SHA,"geometry":GEOMETRY_SHA},
                input_receipt_sha256={"run":sha(run_dir/"run.json"),"capture":CAPTURE_SHA,"motion":MOTION_SHA,"build":BUILD_SHA},
                host_operations="NumPy saved-array hashes/identities/stats and predeclared seven-edge FP64 projection arithmetic only; no model, native correlation or BA solve",
                selected_edges=SELECTED_EDGES,selected_reprojection_max_abs=max_projection_error,
                Jacobian_validation_scope="Frozen geometry's 15 synthetic checks include source/destination/inverse-depth finite differences and Schur-vs-joint solve. No full real BA or Jacobian execution is repeated by this saved-array auditor.",
                feedback_rounds=rows,stage_seconds_sum=sums,total_feedback_seconds=sum(row["round_total_seconds"] for row in rows),
                cache_preparation=run["cache_preparation"],cache_closed=cache["closed"],cache_bytes_after_close=cache["current_context_bytes"],
                peak_rss_mib=run["peak_rss_mib"],original_weight_tensor_count=94,source_and_weights_unchanged=True,
                whole_VO_or_trajectory_accuracy_validated=False,realtime_validated=False,GT_used=False,NPU_used=False,
                interpretation="True fixed-graph neural-to-geometry feedback is verified. Learned target changes and masks are reported, not GT quality. Neither decreasing gated residual nor this three-frame probe validates an online trajectory or 20Hz.")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir","capture-dir","motion-dir","source-dir","build-receipt","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args(argv)
    report=audit(args.run_dir,args.capture_dir,args.motion_dir,args.source_dir,args.build_receipt)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({key:report[key] for key in ("status","checked","failed","total_feedback_seconds")}))


if __name__=="__main__":
    main()
