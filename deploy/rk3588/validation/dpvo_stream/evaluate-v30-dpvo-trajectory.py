#!/usr/bin/env python3
"""Saved-array-only DPVO evaluation; fresh rows and EOF reconstruction are separate.

No image decoding, neural model, geometry solver, native library or board access.
This is post-run GT analysis, not a runtime acceptance test or a speed benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def require(ok, reason):
    if not bool(ok):
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matrices(pose):
    """Translation + xyzw quaternion -> transform. Normalize only quaternion math."""
    a = np.asarray(pose, np.float64)
    require(a.ndim == 2 and a.shape[1] == 7 and np.isfinite(a).all(), "Finite [N,7] pose required")
    norm = np.linalg.norm(a[:, 3:], axis=1)
    require(np.all(np.abs(norm-1) <= 1e-3), "Non-unit pose quaternion")
    x, y, z, w = (a[:, 3:]/norm[:, None]).T
    out = np.broadcast_to(np.eye(4), (len(a), 4, 4)).copy()
    out[:, :3, 3] = a[:, :3]
    out[:, :3, :3] = np.stack((
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=1).reshape(-1, 3, 3)
    return out


def inverse(T):
    result = np.broadcast_to(np.eye(4), T.shape).copy()
    result[..., :3, :3] = T[..., :3, :3].swapaxes(-1, -2)
    result[..., :3, 3] = -np.einsum("...ij,...j->...i", result[..., :3, :3], T[..., :3, 3])
    return result


def summary(values):
    a = np.asarray(values, np.float64)
    require(np.isfinite(a).all(), "Nonfinite metric")
    if not a.size:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None, "rmse": None}
    return {"count": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.quantile(a, .95)), "max": float(a.max()), "rmse": float(np.sqrt(np.mean(a*a)))}


def nearest_association(stamps, gt_stamps, max_difference=.02):
    """Explicit evaluator policy: nearest GT, <=20ms, one-to-one in source order."""
    stamps, gt_stamps = np.asarray(stamps), np.asarray(gt_stamps)
    require(np.all(np.diff(stamps) > 0) and np.all(np.diff(gt_stamps) > 0), "Strictly ordered timestamps required")
    result, used = {}, set()
    for i, stamp in enumerate(stamps):
        high = int(np.searchsorted(gt_stamps, stamp))
        candidates = [j for j in (high-1, high) if 0 <= j < len(gt_stamps)]
        if not candidates:
            continue
        chosen = min(candidates, key=lambda j: (abs(float(gt_stamps[j]-stamp)), j))
        if abs(float(gt_stamps[chosen]-stamp)) <= max_difference and chosen not in used:
            result[i] = chosen
            used.add(chosen)
    return result


def endpoint_plan(stamps, delta=1., tolerance=.02):
    """Plan on ALL requested source timestamps, before any success/GT filtering."""
    stamps = np.asarray(stamps, np.float64)
    require(stamps.ndim == 1 and np.isfinite(stamps).all() and np.all(np.diff(stamps) > 0), "Ordered source timestamps required")
    pairs = []
    for i, stamp in enumerate(stamps):
        wanted = stamp + delta
        if not len(stamps) or wanted > stamps[-1]:
            continue
        high = int(np.searchsorted(stamps, wanted))
        choices = [j for j in (high-1, high) if i < j < len(stamps)]
        if choices:
            j = min(choices, key=lambda k: (abs(float(stamps[k]-wanted)), k))
            if abs(float(stamps[j]-wanted)) <= tolerance:
                pairs.append((i, j))
    return pairs


def fit_positive_sim3_centers(source, target):
    """One whole-trajectory least-squares alignment for evaluation, never inference."""
    source, target = np.asarray(source), np.asarray(target)
    if len(source) < 3:
        return None, "fewer_than_three_matched_centers"
    x, y = source-source.mean(0), target-target.mean(0)
    sx, sy = np.linalg.svd(x, compute_uv=False), np.linalg.svd(y, compute_uv=False)
    if sx[0] <= 0 or sy[0] <= 0 or sx[1]/sx[0] < 1e-8 or sy[1]/sy[0] < 1e-8:
        return None, "degenerate_center_alignment_rank_below_two"
    U, singular, Vt = np.linalg.svd(y.T@x/len(x))
    signs = np.array([1., 1., 1. if np.linalg.det(U@Vt) > 0 else -1.])
    R = (U*signs)@Vt
    scale = float(np.sum(singular*signs)/np.mean(np.sum(x*x, axis=1)))
    if not np.isfinite(scale) or scale <= 0:
        return None, "nonpositive_alignment_scale"
    t = target.mean(0)-scale*(R@source.mean(0))
    return {"scale": scale, "rotation": R, "translation": t}, None


def evaluate_population(poses, association, gt_c2w, plan):
    matched = [i for i in sorted(poses) if i in association]
    pred_c2w = inverse(np.stack([poses[i] for i in matched])) if matched else np.empty((0, 4, 4))
    target = gt_c2w[[association[i] for i in matched]]
    alignment, reason = fit_positive_sim3_centers(pred_c2w[:, :3, 3], target[:, :3, 3])
    ate = []
    if alignment is not None:
        center = alignment["scale"]*(pred_c2w[:, :3, 3]@alignment["rotation"].T)+alignment["translation"]
        ate = np.linalg.norm(center-target[:, :3, 3], axis=1)
    rot, translation, direction, used_pairs = [], [], [], []
    for a, b in plan:
        if a not in poses or b not in poses or a not in association or b not in association:
            continue
        relative = poses[b]@inverse(poses[a])
        gt_relative = inverse(gt_c2w[association[b]])@gt_c2w[association[a]]
        error = relative[:3, :3]@gt_relative[:3, :3].T
        rot.append(float(np.degrees(np.arccos(np.clip((np.trace(error)-1)/2, -1, 1)))))
        predicted_translation, gt_translation = relative[:3, 3], gt_relative[:3, 3]
        lengths = np.linalg.norm(predicted_translation), np.linalg.norm(gt_translation)
        if min(lengths) > 1e-8:
            cosine = float(predicted_translation@gt_translation/(lengths[0]*lengths[1]))
            direction.append(float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))
        if alignment is not None:
            translation.append(float(np.linalg.norm(alignment["scale"]*relative[:3, 3]-gt_relative[:3, 3])))
        used_pairs.append([a, b])
    return {"poses": len(poses), "GT_matched_poses": len(matched),
            "ATE_alignment": alignment, "ATE_unavailable_reason": reason,
            "ATE_sim3_aligned_meters": summary(ate),
            "fixed_endpoint_plan_pairs": len(plan), "used_endpoint_pairs": used_pairs,
            "RPE_endpoint_coverage": len(used_pairs)/len(plan) if plan else None,
            "RPE_rotation_degrees": summary(rot),
            "RPE_translation_GT_scale_meters": summary(translation),
            "RPE_translation_direction_degrees_nonzero_vectors": summary(direction)}


def analyze(run_dir, groundtruth, run_sha256, gt_sha256):
    root, gt_path = Path(run_dir), Path(groundtruth)
    require(sha(root/"run.json") == run_sha256 and sha(gt_path) == gt_sha256, "Input receipt/GT SHA mismatch")
    run = json.loads((root/"run.json").read_text())
    require(run["schema"] == "v30_DPVO_fresh_frame_CPU_replay", "Wrong saved run schema")
    require(sha(root/"input-manifest.json") == run["input_manifest_sha256"], "Source schedule SHA mismatch")
    source = json.loads((root/"input-manifest.json").read_text())["frames"]
    stamps = np.array([x["timestamp"] for x in source], np.float64)
    plan = endpoint_plan(stamps)
    fresh, latencies, last_capture, last_reply = {}, [], -np.inf, -np.inf
    require(len(run["frames"]) == run["frames_completed"], "Delivered row count mismatch")
    for i, row in enumerate(run["frames"]):
        result, original = row["result"], source[i]
        require(row["index"] == result["observation_id"] == i and
                row["frame_id"] == result["frame_id"] == original["frame_id"] and
                row["timestamp"] == result["timestamp"] == original["timestamp"], "Source identity mismatch")
        capture, reply = row["arrival_monotonic"], row["pose_available_monotonic"]
        require(np.isfinite([capture, reply]).all() and capture > last_capture and reply > last_reply
                and reply >= capture and capture == result["captured_monotonic"], "Saved timing/identity mismatch")
        require(abs((reply-capture)-row["processing_seconds"]) < 1e-8, "Saved processing time mismatch")
        last_capture, last_reply = capture, reply
        require(type(result["fresh_pose_valid"]) is bool, "Fresh flag must be bool")
        if result["fresh_pose_valid"]:
            require(result["admitted"] is True and result["initialized"] is True and result["updates"] >= 1
                    and result["reason"] == "updated_current_observation", "Invalid fresh event")
            fresh[i] = matrices([result["pose_w2c"]])[0]
            latencies.append(reply-capture)
        else:
            require(result["pose_w2c"] is None, "Non-fresh row contains a published pose")
    metrics = run["metrics_before_termination"] if "metrics_before_termination" in run else run["stream_metrics_at_exit"]
    require(metrics["fresh_poses"] == len(fresh) and metrics["observations"] == len(run["frames"]), "Fresh/accounting mismatch")
    gt = np.loadtxt(gt_path, comments="#")
    require(gt.ndim == 2 and gt.shape[1] == 8 and np.isfinite(gt).all(), "TUM GT must be finite timestamp+xyz+xyzw")
    gt_c2w = matrices(gt[:, 1:])
    association = nearest_association(stamps, gt[:, 0])
    final_result = {"state": "not_present", "counted_as_fresh": False}
    terminated = run.get("termination")
    if terminated is not None:
        require(terminated["fresh_pose_valid"] is False and terminated["online_publication"] is False and
                terminated["reconstructed_poses_counted_as_fresh"] is False, "EOF pose counted as online")
        require(len(terminated["observations"]) == len(run["frames"]), "EOF source count differs")
        for i, observation in enumerate(terminated["observations"]):
            require(observation["observation_id"] == i and observation["frame_id"] == source[i]["frame_id"]
                    and observation["timestamp"] == source[i]["timestamp"], "EOF reconstruction identity differs")
        final_result = {"state": terminated["trajectory_state"], "counted_as_fresh": False,
                        "offline_updates": terminated["offline_updates"],
                        "relative_reconstruction_count": len(terminated["reconstructed_observation_ids"])}
        if terminated["initialized"]:
            final_w2c = matrices(terminated["pose_w2c"])
            final_c2w = matrices(terminated["pose_c2w"])
            require(len(final_w2c) == len(run["frames"]) and
                    np.allclose(inverse(final_w2c), final_c2w, rtol=1e-5, atol=1e-5), "EOF inverse pose mismatch")
            final_result["evaluation"] = evaluate_population(dict(enumerate(final_w2c)), association, gt_c2w, plan)
    require(sha(root/"run.json") == run_sha256 and sha(gt_path) == gt_sha256 and
            sha(root/"input-manifest.json") == run["input_manifest_sha256"], "Saved input changed during analysis")
    return {"status": "saved_numeric_evaluation_not_realtime_or_accuracy_acceptance",
            "run_sha256": run_sha256, "GT_sha256": gt_sha256,
            "evaluator_sha256": sha(__file__), "original_run_status": run["status"],
            "planned_source_observations": len(source), "delivered_observations": len(run["frames"]),
            "fixed_1second_source_endpoint_plan": plan,
            "fresh_pose_source_coverage": len(fresh)/len(source) if source else None,
            "fresh_online": evaluate_population(fresh, association, gt_c2w, plan),
            "EOF_reoptimized_and_reconstructed": final_result,
            "unpaced_software_processing_seconds": summary(latencies),
            "GT_policy": "Nearest timestamp <=20ms, one-to-one in source order; GT is camera-to-world xyzw. No image/model input uses GT.",
            "metric_policy": "One positive Sim3 center alignment per population for ATE and its RPE translation scale; rotation RPE uses Pb*inverse(Pa). Plans use all requested source timestamps before filtering. No per-segment alignment, no heldout/model selection.",
            "limits": "EOF and fresh are distinct populations and alignments. Endpoints are overlapping, not independent samples. Pose latency is serial software file-admission processing, not camera exposure/queue age. Not full original VGGT comparison, map quality, complete VO acceptance or20Hz.",
            "model_forward_calls": 0, "geometry_BA_calls": 0, "images_read": 0, "board_jobs": 0}


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--run-sha256", required=True)
    p.add_argument("--groundtruth", type=Path, required=True)
    p.add_argument("--groundtruth-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = analyze(a.run_dir, a.groundtruth, a.run_sha256, a.groundtruth_sha256)
    with a.output.open("x", encoding="utf-8") as stream:
        json.dump(plain(result), stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "fresh_poses": result["fresh_online"]["poses"]}))


if __name__ == "__main__":
    main()
