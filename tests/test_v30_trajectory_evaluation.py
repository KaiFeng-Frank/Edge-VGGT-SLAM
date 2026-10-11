"""Tiny saved-numeric/SE3 tests; no dataset, image, model or BA execution."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
ENTRY = ROOT / "evaluate-v30-dpvo-trajectory.py"
if not ENTRY.exists():
    ENTRY = ROOT.parent / "deploy/rk3588/validation/dpvo_stream/evaluate-v30-dpvo-trajectory.py"
spec = importlib.util.spec_from_file_location("v30_eval", ENTRY)
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def pose(angle, translation):
    return np.r_[translation, 0., 0., np.sin(angle/2), np.cos(angle/2)]


def test_xyzw_world_to_camera_inverse_optical_center():
    P = e.matrices([pose(np.pi/2, [1., 2., 3.])])[0]
    np.testing.assert_allclose(P[:3, :3], [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], atol=1e-15)
    np.testing.assert_allclose(e.inverse(P)[:3, 3], [-2., 1., -3.], atol=1e-15)
    np.testing.assert_allclose(P@e.inverse(P), np.eye(4), atol=1e-15)


def test_single_positive_sim3_alignment_and_relative_w2c_order():
    centers = np.array([[0., 0., 0.], [1., .2, 0.], [.2, .7, .3], [.8, .9, .5]])
    pred_c2w = e.matrices([pose(a, t) for a, t in zip([0., .2, -.4, .6], centers)])
    global_T = e.matrices([pose(.7, [.4, -.2, .8])])[0]
    gt = np.stack([global_T@p for p in pred_c2w])
    gt[:, :3, 3] = 2*(centers@global_T[:3, :3].T)+global_T[:3, 3]
    result = e.evaluate_population(dict(enumerate(e.inverse(pred_c2w))), dict(enumerate(range(4))), gt, [(0, 2), (1, 3)])
    assert result["ATE_alignment"]["scale"] == pytest.approx(2.)
    assert result["ATE_sim3_aligned_meters"]["max"] < 2e-15
    assert result["RPE_rotation_degrees"]["max"] < 2e-6
    assert result["RPE_translation_GT_scale_meters"]["max"] < 2e-15
    assert result["RPE_translation_direction_degrees_nonzero_vectors"]["max"] < 2e-6


def test_fixed_endpoint_coverage_does_not_choose_replacement_successes():
    stamps = np.arange(5, dtype=float)*.5
    plan = e.endpoint_plan(stamps)
    assert plan == [(0, 2), (1, 3), (2, 4)]
    matrices = e.matrices([pose(0., [x, x*x, 0.]) for x in stamps])
    result = e.evaluate_population({i: e.inverse(matrices[i]) for i in [0, 1, 3, 4]}, dict(enumerate(range(5))), matrices, plan)
    assert result["used_endpoint_pairs"] == [[1, 3]]
    assert result["RPE_endpoint_coverage"] == pytest.approx(1/3)


def test_nearest_gt_policy_tolerance_and_one_to_one():
    assert e.nearest_association([0., .01, 1., 2.], [0., .99, 2.1]) == {0: 0, 2: 1}


def test_degenerate_ate_does_not_invent_scale_but_rotation_rpe_available():
    centers = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    gt = e.matrices([pose(0., x) for x in centers])
    result = e.evaluate_population(dict(enumerate(e.inverse(gt))), dict(enumerate(range(3))), gt, [(0, 2)])
    assert result["ATE_alignment"] is None
    assert result["ATE_unavailable_reason"] == "degenerate_center_alignment_rank_below_two"
    assert result["RPE_translation_GT_scale_meters"]["count"] == 0
    assert result["RPE_rotation_degrees"]["count"] == 1


def fixture(root):
    root.mkdir()
    stamps = np.arange(10)*.5
    centers = np.array([[t, .2*t*t, .1*t] for t in stamps])
    source = [{"frame_id": str(i), "timestamp": float(t)} for i, t in enumerate(stamps)]
    (root/"input-manifest.json").write_text(json.dumps({"frames": source}))
    w2c = [pose(0., -x).tolist() for x in centers]
    c2w = [pose(0., x).tolist() for x in centers]
    gt = np.array([[t, *(2*x+[.3, -.7, .5]), 0., 0., 0., 1.] for t, x in zip(stamps, centers)])
    gt_path = root/"gt.txt"
    np.savetxt(gt_path, gt)
    frames = []
    for i, original in enumerate(source):
        fresh = i >= 7
        capture = 100.+i
        result = dict(original, observation_id=i, captured_monotonic=capture,
                      fresh_pose_valid=fresh, admitted=True, initialized=fresh,
                      updates=12 if i == 7 else 1 if fresh else 0,
                      reason="updated_current_observation" if fresh else "initialization_wait",
                      pose_w2c=w2c[i] if fresh else None)
        frames.append(dict(original, index=i, arrival_monotonic=capture,
                           pose_available_monotonic=capture+.25, processing_seconds=.25, result=result))
    observations = [dict(x, observation_id=i, captured_monotonic=100.+i) for i, x in enumerate(source)]
    run = {"schema": "v30_DPVO_fresh_frame_CPU_replay", "status": "completed_synthetic_only",
           "input_manifest_sha256": e.sha(root/"input-manifest.json"), "frames": frames,
           "frames_completed": 10, "metrics_before_termination": {"fresh_poses": 3, "observations": 10},
           "termination": {"fresh_pose_valid": False, "online_publication": False,
                "reconstructed_poses_counted_as_fresh": False, "observations": observations,
                "trajectory_state": "initialized_offline_reconstruction", "offline_updates": 12,
                "reconstructed_observation_ids": [], "initialized": True, "pose_w2c": w2c, "pose_c2w": c2w}}
    (root/"run.json").write_text(json.dumps(run))
    return run, gt_path


def test_saved_population_fresh_three_eof_ten_not_borrowed(tmp_path):
    root = tmp_path/"fixture"
    _, gt = fixture(root)
    result = e.analyze(root, gt, e.sha(root/"run.json"), e.sha(gt))
    assert result["fresh_online"]["poses"] == 3
    assert result["fresh_online"]["RPE_endpoint_coverage"] == pytest.approx(1/8)
    assert result["EOF_reoptimized_and_reconstructed"]["evaluation"]["poses"] == 10
    assert result["EOF_reoptimized_and_reconstructed"]["evaluation"]["RPE_endpoint_coverage"] == 1.
    assert result["fresh_pose_source_coverage"] == .3
    assert result["fresh_online"]["ATE_sim3_aligned_meters"]["max"] < 2e-14
    assert result["model_forward_calls"] == result["geometry_BA_calls"] == result["images_read"] == 0


@pytest.mark.parametrize("mutant", ["nonfresh_pose", "source_identity", "timing", "EOF_fresh", "EOF_inverse"])
def test_saved_mutations_rejected(tmp_path, mutant):
    root = tmp_path/"fixture"
    run, gt = fixture(root)
    if mutant == "nonfresh_pose": run["frames"][0]["result"]["pose_w2c"] = pose(0., [0., 0., 0.]).tolist()
    if mutant == "source_identity": run["frames"][7]["result"]["frame_id"] = "wrong"
    if mutant == "timing": run["frames"][7]["processing_seconds"] += .01
    if mutant == "EOF_fresh": run["termination"]["fresh_pose_valid"] = True
    if mutant == "EOF_inverse": run["termination"]["pose_c2w"][7][0] += .1
    (root/"run.json").write_text(json.dumps(run))
    with pytest.raises(ValueError):
        e.analyze(root, gt, e.sha(root/"run.json"), e.sha(gt))
