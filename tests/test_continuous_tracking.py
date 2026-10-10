"""Synthetic measured tracks + real PnP/triangulation; no model or dataset."""

import cv2
import numpy as np
import pytest

from vggt_slam.continuous_tracking import ContinuousDepthTracker, build_depth_candidate, _LocalMap
from vggt_slam.fast_tracking import model_pixel_affine


@pytest.fixture(autouse=True)
def deterministic_cv():
    threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(12)
    yield
    cv2.setNumThreads(threads)


class Scene:
    """Known 3D features appear and disappear; fake only image feature measures."""

    hw = (240, 320)
    K = np.array([[260., 0., 159.5], [0., 260., 119.5], [0., 0., 1.]])

    def __init__(self, monkeypatch, *, rotation_only=False, distortion=None):
        self.rotation_only = rotation_only
        self.distortion = distortion
        self.xyz, self.birth = [], []
        rng = np.random.default_rng(67)
        for generation in range(22):
            birth = generation * 4
            uv = rng.uniform([25., 25.], [295., 215.], (90, 2))
            normalized = cv2.undistortPoints(uv[:, None], self.K, distortion).reshape(-1, 2)
            camera_xyz = np.column_stack((normalized, np.ones(len(uv)))) * rng.uniform(3., 6., (90, 1))
            T = self.pose(birth)
            self.xyz.extend((camera_xyz - T[:3, 3]) @ T[:3, :3])
            self.birth.extend([birth] * len(uv))
        self.xyz, self.birth = np.asarray(self.xyz), np.asarray(self.birth)
        monkeypatch.setattr(cv2, "goodFeaturesToTrack", self.corners)
        monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", self.flow)

    def pose(self, index):
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(np.array([0., .0005 * index, .0002 * index]))[0]
        if not self.rotation_only:
            T[:3, 3] = [-.04 * index, -.003 * index, .001 * index]
        return T

    def measurements(self, index):
        rows = np.flatnonzero((self.birth <= index) & (index < self.birth + 16))
        T = self.pose(index)
        uv = cv2.projectPoints(self.xyz[rows], cv2.Rodrigues(T[:3, :3])[0],
                               T[:3, 3], self.K, self.distortion)[0].reshape(-1, 2)
        keep = (uv[:, 0] > 12) & (uv[:, 0] < 308) & (uv[:, 1] > 12) & (uv[:, 1] < 228)
        return rows[keep], uv[keep].astype(np.float32)

    def image(self, index):
        return np.full(self.hw, index, np.uint8)

    def corners(self, gray, *, maxCorners, mask, **kwargs):
        _, uv = self.measurements(int(gray[0, 0]))
        pixels = np.rint(uv).astype(int)
        keep = mask[pixels[:, 1], pixels[:, 0]] > 0
        return uv[keep][:maxCorners, None].copy() if keep.any() else None

    def flow(self, previous, current, points, unused, **kwargs):
        source, target = int(previous[0, 0]), int(current[0, 0])
        source_ids, source_uv = self.measurements(source)
        target_ids, target_uv = self.measurements(target)
        target_by_id = dict(zip(target_ids, target_uv))
        result, status = [], []
        for point in points.reshape(-1, 2):
            nearest = np.argmin(np.linalg.norm(source_uv - point, axis=1))
            key = source_ids[nearest]
            matched = np.linalg.norm(source_uv[nearest] - point) < .01 and key in target_by_id
            result.append(target_by_id[key] if matched else point)
            status.append(int(matched))
        return np.asarray(result, np.float32)[:, None], np.asarray(status, np.uint8)[:, None], None

    def depth(self, index, scale=1.):
        A, hw = model_pixel_affine(self.hw, 280)
        rows, uv = self.measurements(index)
        camera_xyz = self.xyz[rows] @ self.pose(index)[:3, :3].T + self.pose(index)[:3, 3]
        pixels = np.rint(np.column_stack((uv, np.ones(len(uv)))) @ A[:2].T).astype(int)
        depth = np.zeros(hw, np.float32)
        depth[pixels[:, 1], pixels[:, 0]] = camera_xyz[:, 2] * scale
        return dict(depth=depth, confidence=np.ones(hw, np.float32), K_model=A @ self.K,
                    model_image_size=280)


def tracker(**kwargs):
    return ContinuousDepthTracker(max_width=320, max_features=360, max_landmarks=300,
                                  max_keyframes=4, history_frames=24, **kwargs)


def observe(frontend, scene, index):
    return frontend.process(scene.image(index), index, 10. + index * .05)


def test_replenishment_tracks_long_after_all_seed_points_disappear(monkeypatch):
    scene = Scene(monkeypatch)
    frontend = tracker()
    first = observe(frontend, scene, 0)
    assert first["status"] == "waiting_geometry" and first["T_current_from_anchor"] is None
    initial = frontend.attach_depth(0, 10., **scene.depth(0))
    assert initial["adopted"] and not initial["emits_fresh_pose"]
    seed_ids = set(frontend._map.landmarks)
    poses = []
    for index in range(1, 65):
        result = observe(frontend, scene, index)
        assert result["pose_valid"], (index, result)
        assert result["inliers"] >= 20 and result["inlier_ratio"] >= .5
        np.testing.assert_allclose(result["T_current_from_anchor"], scene.pose(index), atol=2e-3)
        assert result["frame_id"] == index and result["captured_monotonic"] == 10. + index * .05
        assert not result["historical_replay"] and not result["metric_scale_validated"]
        assert frontend.state["landmarks"] <= 300 and frontend.state["tracked_features"] <= 360
        assert frontend.state["local_keyframes"] <= 4 and frontend.state["history_frames"] <= 24
        poses.append(result["T_current_from_anchor"])
    assert not seed_ids.intersection(frontend._map.landmarks)
    assert frontend.metrics()["triangulated_landmarks"] > 100
    assert frontend.metrics()["fresh_valid_poses"] == 64
    assert len({tuple(T[:3, 3]) for T in poses}) == 64
    assert frontend.state["retained_gray_frames"] == 1


def test_delayed_depth_replays_measured_history_and_only_next_capture_counts(monkeypatch):
    scene = Scene(monkeypatch)
    frontend = tracker()
    for index in range(11):
        assert not observe(frontend, scene, index)["pose_valid"]
    snapshot = frontend.snapshot_depth_history(0, 10.)
    assert not snapshot["records"][0]["uv"].flags.writeable
    candidate = build_depth_candidate(snapshot, **scene.depth(0))
    assert candidate["available"] and candidate["replay_valid_poses"] == 10
    assert frontend.metrics()["fresh_valid_poses"] == 0 and frontend.state["anchor_frame_id"] is None
    observe(frontend, scene, 11)
    observe(frontend, scene, 12)
    adopted = frontend.apply_depth_candidate(candidate)
    assert adopted["adopted"] and adopted["catchup_valid_poses"] == 2
    assert adopted["historical_replay"] and not adopted["emits_fresh_pose"]
    assert frontend.metrics()["fresh_valid_poses"] == 0
    assert frontend.metrics()["historical_replay_poses"] == 12
    current = observe(frontend, scene, 13)
    assert current["pose_valid"] and frontend.metrics()["fresh_valid_poses"] == 1
    np.testing.assert_allclose(current["T_current_from_anchor"], scene.pose(13), atol=2e-4)


@pytest.mark.parametrize("stride", [1, 3, 4, 100])
def test_replay_stride_uses_exact_records_and_always_reaches_latest(monkeypatch, stride):
    scene = Scene(monkeypatch)
    frontend = tracker(replay_stride=stride)
    for index in range(11):
        observe(frontend, scene, index)
    snapshot = frontend.snapshot_depth_history(0, 10.)
    measured = {record["sequence"]: record for record in snapshot["records"]}
    original_step = _LocalMap.step
    visited = []

    def checked_step(local_map, record):
        if record["sequence"] <= 10:
            expected = measured[record["sequence"]]
            assert record is expected
            np.testing.assert_array_equal(record["ids"], expected["ids"])
            np.testing.assert_array_equal(record["uv"], expected["uv"])
        visited.append(record["sequence"])
        return original_step(local_map, record)

    monkeypatch.setattr(_LocalMap, "step", checked_step)
    candidate = build_depth_candidate(snapshot, **scene.depth(0))
    expected = list(range(stride, 11, stride))
    if not expected or expected[-1] != 10:
        expected.append(10)
    assert visited == expected
    assert candidate["available"] and candidate["last_sequence"] == 10
    assert candidate["last_frame_id"] == 10 and candidate["last_captured_monotonic"] == 10.5
    assert candidate["replay_stride"] == stride
    assert candidate["replay_available_observations"] == 10
    assert candidate["replay_observations"] == candidate["replay_valid_poses"] == len(expected)
    assert candidate["replay_skipped_observations"] == 10 - len(expected)
    assert frontend.metrics()["fresh_valid_poses"] == 0

    observe(frontend, scene, 11)
    observe(frontend, scene, 12)
    visited.clear()
    applied = frontend.apply_depth_candidate(candidate)
    assert applied["adopted"] and not applied["emits_fresh_pose"]
    assert visited == ([11, 12] if stride == 1 else [12])
    assert frontend._map.last_sequence == 12
    assert frontend._map.last_frame_id == 12
    assert frontend._map.last_captured == 10.6
    assert applied["catchup_available_observations"] == 2
    assert applied["catchup_skipped_observations"] == (0 if stride == 1 else 1)
    assert frontend.metrics()["fresh_valid_poses"] == 0
    visited.clear()
    current = observe(frontend, scene, 13)
    assert current["pose_valid"] and visited == [13]
    assert frontend.metrics()["fresh_valid_poses"] == 1
    np.testing.assert_allclose(current["T_current_from_anchor"], scene.pose(13), atol=3e-3)


def test_strided_replay_still_rejects_lost_final_observation_and_keeps_map(monkeypatch):
    scene = Scene(monkeypatch)
    frontend = tracker(replay_stride=4)
    observe(frontend, scene, 0)
    frontend.attach_depth(0, 10., **scene.depth(0))
    for index in range(1, 7):
        assert observe(frontend, scene, index)["pose_valid"]
    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", lambda a, b, p, unused, **kwargs:
                        (p.copy(), np.zeros((len(p), 1), np.uint8), None))
    assert not observe(frontend, scene, 7)["pose_valid"]
    candidate = build_depth_candidate(frontend.snapshot_depth_history(1, 10.05), **scene.depth(1))
    assert candidate["replay_observations"] == 2  # Actual frames 5 and 7.
    assert not candidate["available"] and candidate["reason"] == "insufficient_landmarks"
    rejected = frontend.apply_depth_candidate(candidate)
    assert not rejected["adopted"] and not rejected["emits_fresh_pose"]
    assert frontend.state["anchor_frame_id"] == 0 and frontend.state["gauge_id"] == 0
    assert frontend.metrics()["fresh_valid_poses"] == 6


def test_strided_catchup_checks_latest_and_does_not_reuse_earlier_success(monkeypatch):
    scene = Scene(monkeypatch)
    frontend = tracker(replay_stride=3)
    observe(frontend, scene, 0)
    frontend.attach_depth(0, 10., **scene.depth(0))
    assert observe(frontend, scene, 1)["pose_valid"]
    candidate = build_depth_candidate(frontend.snapshot_depth_history(1, 10.05), **scene.depth(1))
    for index in range(2, 5):
        assert observe(frontend, scene, index)["pose_valid"]
    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", lambda a, b, p, unused, **kwargs:
                        (p.copy(), np.zeros((len(p), 1), np.uint8), None))
    assert not observe(frontend, scene, 5)["pose_valid"]
    rejected = frontend.apply_depth_candidate(candidate)
    assert rejected["catchup_observations"] == 2  # Frame 4 succeeds, frame 5 fails.
    assert rejected["catchup_valid_poses"] == 1
    assert rejected["catchup_available_observations"] == 4
    assert rejected["catchup_skipped_observations"] == 2
    assert not rejected["adopted"] and rejected["reason"] == "insufficient_landmarks"
    assert frontend.state["anchor_frame_id"] == 0 and frontend.state["gauge_id"] == 0


@pytest.mark.parametrize("stride", [0, -1, 1.5, True])
def test_replay_stride_requires_positive_integer(stride):
    with pytest.raises(ValueError, match="replay_stride"):
        tracker(replay_stride=stride)


def test_same_observation_depth_refresh_changes_gauge_and_units_explicitly(monkeypatch):
    scene = Scene(monkeypatch)
    frontend = tracker()
    observe(frontend, scene, 0)
    frontend.attach_depth(0, 10., **scene.depth(0))
    for index in range(1, 10):
        observe(frontend, scene, index)
    refresh = frontend.attach_depth(5, 10.25, **scene.depth(5, scale=2.))
    assert refresh["adopted"] and refresh["gauge_id"] == 1
    assert frontend.metrics()["fresh_valid_poses"] == 9
    result = observe(frontend, scene, 10)
    expected = scene.pose(10) @ np.linalg.inv(scene.pose(5))
    expected[:3, 3] *= 2.
    assert result["anchor_frame_id"] == 5 and result["anchor_captured_monotonic"] == 10.25
    assert result["gauge_id"] == 1 and not result["metric_scale_validated"]
    np.testing.assert_allclose(result["T_current_from_anchor"], expected, atol=3e-3)
    stale = frontend.attach_depth(3, 10.15, **scene.depth(3))
    assert not stale["adopted"] and stale["reason"] == "anchor_not_newer_than_active_seed"


def test_failed_depth_keeps_previous_geometry_and_never_emits_identity(monkeypatch):
    scene = Scene(monkeypatch)
    frontend = tracker()
    observe(frontend, scene, 0)
    frontend.attach_depth(0, 10., **scene.depth(0))
    observe(frontend, scene, 1)
    depth = scene.depth(1)
    depth["depth"][:] = 0
    rejected = frontend.attach_depth(1, 10.05, **depth)
    assert not rejected["adopted"] and rejected["reason"] == "insufficient_valid_anchor_depth"
    assert frontend.state["anchor_frame_id"] == 0 and frontend.state["gauge_id"] == 0
    result = observe(frontend, scene, 2)
    assert result["pose_valid"] and not np.allclose(result["T_current_from_anchor"], np.eye(4))


def test_flow_loss_is_null_and_new_depth_can_reinitialize(monkeypatch):
    scene = Scene(monkeypatch)
    frontend = tracker()
    observe(frontend, scene, 0)
    frontend.attach_depth(0, 10., **scene.depth(0))
    assert observe(frontend, scene, 1)["pose_valid"]
    native_flow = scene.flow
    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", lambda a, b, p, unused, **kwargs:
                        (p.copy(), np.zeros((len(p), 1), np.uint8), None))
    lost = observe(frontend, scene, 2)
    assert lost["status"] == "lost" and not lost["pose_valid"] and lost["T_current_from_anchor"] is None
    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", native_flow)
    adopted = frontend.attach_depth(2, 10.1, **scene.depth(2))
    assert adopted["adopted"] and adopted["gauge_id"] == 1
    assert observe(frontend, scene, 3)["pose_valid"]


def test_rotation_without_translation_does_not_create_unit_motion_landmarks(monkeypatch):
    scene = Scene(monkeypatch, rotation_only=True)
    frontend = tracker()
    observe(frontend, scene, 0)
    frontend.attach_depth(0, 10., **scene.depth(0))
    for index in range(1, 13):
        result = observe(frontend, scene, index)
        assert result["pose_valid"]
        assert result["triangulated_points"] == 0
        assert np.linalg.norm(result["T_current_from_anchor"][:3, 3]) < 1e-5
    assert frontend.metrics()["triangulated_landmarks"] == 0


def test_external_calibration_distortion_is_explicit_and_not_globally_linear(monkeypatch):
    D = np.array([.18, -.07, .001, -.002, .01])
    scene = Scene(monkeypatch, distortion=D)
    frontend = tracker(intrinsic_raw=scene.K, distortion=D)
    observe(frontend, scene, 0)
    seeded = frontend.attach_depth(0, 10., **scene.depth(0))
    assert seeded["adopted"] and seeded["intrinsic_source"] == "explicit_external"
    assert not seeded["model_association_available"] and seeded["H_model_from_local"] is None
    result = observe(frontend, scene, 1)
    assert result["pose_valid"] and result["model_association_reason"]
    np.testing.assert_allclose(result["T_current_from_anchor"], scene.pose(1), atol=3e-3)
    assert frontend.metrics()["distortion_compensated"]


def test_external_zero_distortion_exports_exact_model_from_local_matrix(monkeypatch):
    scene = Scene(monkeypatch)
    external_K = scene.K.copy()
    external_K[0, 0] *= 1.1
    frontend = tracker(intrinsic_raw=external_K)
    observe(frontend, scene, 0)
    info = frontend.attach_depth(0, 10., **scene.depth(0))
    expected = np.eye(4)
    expected[:3, :3] = np.linalg.inv(scene.K) @ external_K
    np.testing.assert_allclose(info["H_model_from_local"], expected, atol=1e-14)
    assert info["model_association_available"]


def test_old_depth_snapshot_expiry_and_foreign_owner_are_rejected(monkeypatch):
    scene = Scene(monkeypatch)
    frontend = tracker()
    observe(frontend, scene, 0)
    candidate = build_depth_candidate(frontend.snapshot_depth_history(0, 10.), **scene.depth(0))
    for index in range(1, 30):
        observe(frontend, scene, index)
    with pytest.raises(ValueError, match="history expired"):
        frontend.snapshot_depth_history(0, 10.)
    rejected = frontend.apply_depth_candidate(candidate)
    assert not rejected["adopted"] and rejected["reason"] == "candidate_catchup_history_expired"
    assert frontend.state["anchor_frame_id"] is None and frontend.metrics()["fresh_valid_poses"] == 0
    with pytest.raises(ValueError, match="different tracker"):
        tracker().apply_depth_candidate(candidate)


def test_failed_catchup_does_not_replace_existing_gauge(monkeypatch):
    scene = Scene(monkeypatch)
    frontend = tracker()
    observe(frontend, scene, 0)
    frontend.attach_depth(0, 10., **scene.depth(0))
    observe(frontend, scene, 1)
    candidate = build_depth_candidate(frontend.snapshot_depth_history(1, 10.05), **scene.depth(1))
    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", lambda a, b, p, unused, **kwargs:
                        (p.copy(), np.zeros((len(p), 1), np.uint8), None))
    lost = observe(frontend, scene, 2)
    assert not lost["pose_valid"]
    rejected = frontend.apply_depth_candidate(candidate)
    assert not rejected["adopted"] and rejected["reason"] == "insufficient_landmarks"
    assert frontend.state["anchor_frame_id"] == 0 and frontend.state["gauge_id"] == 0
    assert frontend.metrics()["fresh_valid_poses"] == 1


def test_candidate_identity_and_value_copy_are_not_aliased(monkeypatch):
    scene = Scene(monkeypatch)
    frontend = tracker()
    observe(frontend, scene, 0)
    candidate = build_depth_candidate(frontend.snapshot_depth_history(0, 10.), **scene.depth(0))
    wrong = dict(candidate, anchor_frame_id=1)
    with pytest.raises(ValueError, match="identity/geometry"):
        frontend.apply_depth_candidate(wrong)
    assert frontend.apply_depth_candidate(candidate)["adopted"]
    for point in candidate["_local_map"].landmarks.values():
        point[:] = 0
    result = observe(frontend, scene, 1)
    assert result["pose_valid"]
    np.testing.assert_allclose(result["T_current_from_anchor"], scene.pose(1), atol=2e-4)
    result["T_current_from_anchor"][:] = 0
    result["K_tracking"][:] = 0
    assert observe(frontend, scene, 2)["pose_valid"]


def test_real_pnp_uses_current_observations_and_rejects_outliers():
    rng = np.random.default_rng(16)
    K = Scene.K
    xyz = np.column_stack((rng.uniform(-1.4, 1.4, 80), rng.uniform(-.9, .9, 80), rng.uniform(3., 7., 80)))
    original = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), K, None)[0].reshape(-1, 2)
    seed = dict(ids=np.arange(80), uv=original, frame_id=0, captured_monotonic=1., sequence=0)
    geometry = _LocalMap(tracker().config, K, None, seed, xyz, seed["ids"])
    R = np.array([.015, -.02, .01])
    t = np.array([.12, -.03, .06])
    pixels = cv2.projectPoints(xyz, R, t, K, None)[0].reshape(-1, 2)
    pixels[:20] = rng.uniform([15., 15.], [305., 225.], (20, 2))
    record = dict(seed, uv=pixels, frame_id=1, captured_monotonic=1.1, sequence=1)
    result = geometry.step(record)
    assert result["pose_valid"] and result["inliers"] == 60 and result["inlier_ratio"] == .75
    np.testing.assert_allclose(result["T_current_from_anchor"][:3, :3], cv2.Rodrigues(R)[0], atol=1e-6)
    np.testing.assert_allclose(result["T_current_from_anchor"][:3, 3], t, atol=1e-6)
    lost = geometry.step(dict(record, uv=rng.uniform([0., 0.], [320., 240.], (80, 2)), sequence=2))
    assert not lost["pose_valid"] and lost["T_current_from_anchor"] is None


def test_real_lk_replenishes_features_while_waiting_and_bounds_time_history():
    texture = np.random.default_rng(17).integers(0, 256, (120, 160), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (5, 5), 1.)
    frontend = ContinuousDepthTracker(max_width=160, max_features=90, history_frames=6, history_seconds=.16)
    for index in range(10):
        image = cv2.warpAffine(texture, np.float32([[1, 0, index], [0, 1, 0]]), (160, 120))
        result = frontend.process(image, index, index * .05)
        assert result["status"] == "waiting_geometry" and result["T_current_from_anchor"] is None
        assert frontend.state["history_frames"] <= 4
        assert frontend.state["retained_history_seconds"] <= .16
        assert frontend.state["retained_gray_frames"] == 1
    assert frontend.state["tracked_features"] > 30
    assert frontend.metrics()["fresh_valid_poses"] == 0


@pytest.mark.parametrize("kwargs", [
    {"max_features": 19}, {"max_landmarks": 19}, {"history_frames": 1},
    {"history_seconds": float("inf")}, {"history_seconds": 0},
    {"min_inlier_ratio": 0}, {"reprojection_error": 0}, {"min_parallax_degrees": 0},
    {"distortion": [0.] * 5}, {"intrinsic_raw": np.zeros((3, 3))},
])
def test_invalid_configuration_is_explicit(kwargs):
    with pytest.raises(ValueError):
        ContinuousDepthTracker(**kwargs)


def test_metadata_resolution_and_depth_crop_mismatches_fail_without_new_pose(monkeypatch):
    scene = Scene(monkeypatch)
    frontend = tracker()
    observe(frontend, scene, 0)
    for frame_id, captured in ((0, 10.1), (1, 10.), (float("nan"), 10.1)):
        with pytest.raises(ValueError):
            frontend.process(scene.image(1), frame_id, captured)
    with pytest.raises(ValueError, match="Resolution changed"):
        frontend.process(np.zeros((120, 160), np.uint8), 1, 10.1)
    with pytest.raises(ValueError, match="Exact anchor"):
        frontend.snapshot_depth_history(0, 10.000001)
    arguments = scene.depth(0)
    arguments["depth"] = np.ones((280, 280))
    with pytest.raises(ValueError, match="model crop"):
        frontend.attach_depth(0, 10., **arguments)
    assert frontend.state["frame_id"] == 0 and frontend.state["anchor_frame_id"] is None
    assert frontend.metrics()["fresh_valid_poses"] == 0
