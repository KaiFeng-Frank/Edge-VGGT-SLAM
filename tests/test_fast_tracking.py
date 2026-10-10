"""Synthetic local-camera geometry and real LK tests; no Torch or model."""

import cv2
import numpy as np
import pytest

from vggt_slam.fast_tracking import (
    SparseDepthTracker, model_pixel_affine, tracking_pixel_affine,
)


@pytest.fixture(autouse=True)
def one_cv_thread():
    previous = cv2.getNumThreads()
    cv2.setNumThreads(1)
    yield
    cv2.setNumThreads(previous)


def pixels(points, affine):
    return np.column_stack((points, np.ones(len(points)))) @ affine[:2].T


@pytest.mark.parametrize("shape,size,expected_height,crop_y", [
    ((1080, 1920), 518, 294, 0), ((1920, 1080), 518, 924, 203),
    ((10, 2000), 112, 14, 0),
])
def test_model_affine_has_pixel_center_and_portrait_crop(shape, size, expected_height, crop_y):
    affine, model_hw = model_pixel_affine(shape, size)
    assert model_hw == (min(expected_height, size), size)
    origin = pixels(np.array([[0., 0.]]), affine)[0]
    expected_origin = [(size / shape[1] - 1) / 2,
                       (expected_height / shape[0] - 1) / 2 - crop_y]
    np.testing.assert_allclose(origin, expected_origin)
    center = pixels(np.array([[(shape[1] - 1) / 2, (shape[0] - 1) / 2]]), affine)[0]
    np.testing.assert_allclose(center, [(size - 1) / 2, (expected_height - 1) / 2 - crop_y])


def test_tracking_affine_uses_actual_rounded_resize_dimensions():
    affine = tracking_pixel_affine((101, 201), (50, 100))
    np.testing.assert_allclose(pixels(np.array([[100., 50.]]), affine), [[49.5, 24.5]])
    assert affine[0, 0] != affine[1, 1]


def synthetic_anchor(monkeypatch, *, count=80, outliers=16):
    rng = np.random.default_rng(31)
    xyz = np.column_stack((rng.uniform(-1.3, 1.3, count),
                           rng.uniform(-.9, .9, count), rng.uniform(3., 7., count)))
    original_hw, tracking_hw, model_size = (480, 640), (360, 480), 280
    intrinsic_raw = np.array([[420., 0., 319.5], [0., 420., 239.5], [0., 0., 1.]])
    affine_model, model_hw = model_pixel_affine(original_hw, model_size)
    affine_tracking = tracking_pixel_affine(original_hw, tracking_hw)
    intrinsic_model = affine_model @ intrinsic_raw
    intrinsic_tracking = affine_tracking @ intrinsic_raw
    reference_pixels = (xyz @ intrinsic_raw.T)[:, :2] / xyz[:, 2, None]
    tracking_reference = pixels(reference_pixels, affine_tracking).astype(np.float32)
    rotation = np.array([.025, -.035, .012])
    translation = np.array([.12, -.03, .08])
    current_pixels, _ = cv2.projectPoints(xyz, rotation, translation, intrinsic_tracking, None)
    current_pixels = current_pixels.reshape(-1, 2).astype(np.float32)
    current_pixels[:outliers] = rng.uniform([40., 40.], [440., 320.], (outliers, 2))
    monkeypatch.setattr(cv2, "goodFeaturesToTrack", lambda *args, **kwargs: tracking_reference[:, None].copy())
    calls = []

    def optical_flow(previous, current, pts, unused, **kwargs):
        calls.append(len(pts))
        values = current_pixels if len(calls) % 2 else tracking_reference
        return values[:, None].copy(), np.ones((count, 1), np.uint8), None

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", optical_flow)
    tracker = SparseDepthTracker(min_inliers=20, reprojection_error=2.)
    tracker.process(np.zeros(original_hw, np.uint8), 1, 10.)
    anchor_id = tracker.add_anchor()
    depth = np.zeros(model_hw, np.float32)
    sample = np.rint(pixels(reference_pixels, affine_model)).astype(int)
    assert len({tuple(point) for point in sample}) == count
    depth[sample[:, 1], sample[:, 0]] = xyz[:, 2]
    confidence = np.ones(model_hw, np.float32)
    truth = np.eye(4)
    truth[:3, :3] = cv2.Rodrigues(rotation)[0]
    truth[:3, 3] = translation
    return tracker, anchor_id, depth, confidence, intrinsic_model, truth, calls, xyz, current_pixels


def test_late_anchor_depth_and_real_pnp_reject_outliers_without_future_depth(monkeypatch):
    tracker, key, depth, confidence, intrinsic, truth, calls, xyz, _ = synthetic_anchor(monkeypatch)
    waiting = tracker.estimate(key)
    assert waiting["status"] == "waiting_depth" and waiting["T_current_from_anchor"] is None
    assert waiting["K_tracking"] is None
    tracker.process(np.zeros((480, 640), np.uint8), 2, 10.1)
    assert tracker.estimate(key)["status"] == "waiting_depth"
    info = tracker.attach_depth(key, depth[..., None], confidence, intrinsic, 280)
    assert calls == [80, 80]
    assert info["retained_depth_points"] == 80
    assert info["model_hw"] == (210, 280) and info["tracking_hw"] == (360, 480)
    np.testing.assert_allclose(info["K_tracking"], info["tracking_from_raw"] @
                               np.linalg.inv(info["model_from_raw"]) @ intrinsic)
    np.testing.assert_allclose(tracker._anchors[key].points3d, xyz, atol=3e-7)
    # Returned geometry is a snapshot, rather than a mutable internal camera.
    info["K_tracking"][:] = 0
    cv2.setRNGSeed(11)
    result = tracker.estimate(key)
    assert result["status"] == "tracking"
    np.testing.assert_allclose(result["T_current_from_anchor"], truth, atol=2e-5)
    assert 60 <= result["inliers"] <= 64 and result["inlier_ratio"] >= .75
    assert result["positive_depth_ratio"] == 1.
    assert result["median_reprojection_error"] < 1e-3
    assert result["frame_id"] == 2 and result["anchor_frame_id"] == 1
    assert result["pnp_seconds"] >= 0
    assert tracker.ready_anchor_ids == (key,)

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", lambda a, b, points, unused, **kw:
                        (points.copy(), np.zeros((len(points), 1), np.uint8), None))
    tracker.process(np.zeros((480, 640), np.uint8), 3, 10.2)
    lost = tracker.estimate(key)
    assert lost["status"] == "lost" and lost["T_current_from_anchor"] is None
    assert lost["matched_points"] == 0 and not tracker.ready_anchor_ids


def test_no_same_frame_identity_and_no_stale_pose_after_pnp_failure(monkeypatch):
    tracker, key, depth, confidence, intrinsic, _, _, _, _ = synthetic_anchor(monkeypatch, outliers=0)
    tracker.attach_depth(key, depth, confidence, intrinsic, 280)
    same_frame = tracker.estimate(key)
    assert same_frame["status"] == "lost" and same_frame["reason"] == "no_later_frame"
    assert same_frame["T_current_from_anchor"] is None
    tracker.process(np.zeros((480, 640), np.uint8), 2, 10.1)
    assert tracker.estimate(key)["status"] == "tracking"
    monkeypatch.setattr(cv2, "solvePnPRansac", lambda *a, **kw: (False, None, None, None))
    result = tracker.estimate(key)
    assert result["status"] == "lost" and result["reason"] == "pnp_failed"
    assert result["T_current_from_anchor"] is None


def test_refined_behind_camera_pose_is_rejected(monkeypatch):
    tracker, key, depth, confidence, intrinsic, _, _, _, _ = synthetic_anchor(monkeypatch, outliers=0)
    tracker.attach_depth(key, depth, confidence, intrinsic, 280)
    tracker.process(np.zeros((480, 640), np.uint8), 2, 10.1)
    monkeypatch.setattr(cv2, "solvePnPRefineLM", lambda *a, **kw: (np.zeros((3, 1)), np.array([[0.], [0.], [-20.]])))
    result = tracker.estimate(key)
    assert result["status"] == "lost" and result["T_current_from_anchor"] is None
    assert result["positive_depth_ratio"] == 0 and result["inliers"] == 0


def test_backward_inconsistent_tracks_prune_matching_depth_points(monkeypatch):
    tracker, key, depth, confidence, intrinsic, truth, _, xyz, current = synthetic_anchor(monkeypatch, outliers=0)
    tracker.attach_depth(key, depth, confidence, intrinsic, 280)
    reference = tracker._anchors[key].original_pixels.copy()
    backward = reference.copy()
    backward[:8, 0] += 2.  # Forward flow succeeds, but violates the 1px backward limit.
    outputs = iter((current, backward))
    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", lambda *a, **kw:
                        (next(outputs)[:, None].copy(), np.ones((80, 1), np.uint8), None))
    result = tracker.process(np.zeros((480, 640), np.uint8), 2, 10.1)
    assert result["anchors"][key]["tracked_points"] == 72
    np.testing.assert_allclose(tracker._anchors[key].points3d, xyz[8:], atol=3e-7)
    np.testing.assert_allclose(tracker._anchors[key].original_pixels, reference[8:])
    estimate = tracker.estimate(key)
    assert estimate["status"] == "tracking"
    np.testing.assert_allclose(estimate["T_current_from_anchor"], truth, atol=2e-5)


def test_real_lk_is_continuous_batched_and_measures_anchor_displacement(monkeypatch):
    texture = np.random.default_rng(12).integers(0, 256, (240, 320), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (5, 5), 1.)
    tracker = SparseDepthTracker(max_width=160, max_corners=80)
    tracker.process(texture, 0, 0.)
    tracker.add_anchor("first")
    tracker.add_anchor("second")
    native_lk = cv2.calcOpticalFlowPyrLK
    calls = []

    def recorded_lk(*args, **kwargs):
        calls.append(len(args[2]))
        return native_lk(*args, **kwargs)

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", recorded_lk)
    for frame_id in (1, 2):
        moved = cv2.warpAffine(texture, np.float32([[1, 0, 2 * frame_id], [0, 1, frame_id]]),
                               (320, 240), borderMode=cv2.BORDER_REFLECT)
        result = tracker.process(moved, frame_id, frame_id * .1)
        assert len(calls) == frame_id * 2
        for values in result["anchors"].values():
            assert values["tracked_points"] > 40
            assert values["mean_disparity_original_pixels"] == pytest.approx(frame_id * np.sqrt(5), abs=.12)
    assert calls[0] == 160
    assert tracker.state["retained_gray_frames"] == 1
    assert tracker.state["tracking_shape"] == (120, 160)


def test_portrait_crop_and_invalid_depth_confidence_remove_correspondences(monkeypatch):
    original_hw, tracking_hw = (1920, 1080), (960, 540)
    raw_points = np.array([[540., 0.], [540., 1919.], [200., 900.],
                           [400., 1000.], [600., 1100.], [800., 1200.]])
    affine_b = tracking_pixel_affine(original_hw, tracking_hw)
    points = pixels(raw_points, affine_b).astype(np.float32)
    monkeypatch.setattr(cv2, "goodFeaturesToTrack", lambda *a, **kw: points[:, None].copy())
    tracker = SparseDepthTracker(max_width=540, min_inliers=4)
    tracker.process(np.zeros(original_hw, np.uint8), 1, 1.)
    key = tracker.add_anchor()
    affine_a, model_hw = model_pixel_affine(original_hw, 280)
    assert model_hw == (280, 280)
    sample = np.rint(pixels(raw_points, affine_a)).astype(int)
    depth = np.full(model_hw, 4., np.float32)
    confidence = np.full(model_hw, 10., np.float32)
    depth[sample[2, 1], sample[2, 0]] = np.nan
    depth[sample[3, 1], sample[3, 0]] = -1
    confidence[sample[4, 1], sample[4, 0]] = .1
    info = tracker.attach_depth(key, depth, confidence, np.array([[200., 0., 139.5],
                               [0., 200., 139.5], [0., 0., 1.]]), 280)
    assert info["confidence_threshold"] == 10.
    assert info["retained_depth_points"] == 1
    assert tracker.state["anchors"][key]["tracked_points"] == 1
    assert tracker.state["anchors"][key]["depth_attached"] is True
    assert tracker.state["anchors"][key]["depth_ready"] is False
    assert tracker.estimate(key)["status"] == "lost"


@pytest.mark.parametrize("complex_field", ["depth", "confidence", "intrinsic"])
def test_complex_geometry_is_rejected_without_discarding_imaginary_values(complex_field):
    tracker = SparseDepthTracker()
    tracker.process(np.zeros((60, 80), np.uint8), 1, 1.)
    key = tracker.add_anchor()
    values = dict(depth=np.ones((84, 112)), confidence=np.ones((84, 112)),
                  intrinsic=np.array([[80., 0., 55.5], [0., 80., 41.5], [0., 0., 1.]]))
    values[complex_field] = values[complex_field].astype(np.complex64) + 1j
    with pytest.raises(ValueError, match="real numeric"):
        tracker.attach_depth(key, **values, model_image_size=112)
    assert tracker.state["anchors"][key]["depth_attached"] is False


def test_blank_frame_and_anchor_capacity_are_explicit_and_bounded():
    tracker = SparseDepthTracker(max_corners=12)
    with pytest.raises(RuntimeError, match="Process a frame"):
        tracker.add_anchor()
    blank = np.zeros((60, 80), np.uint8)
    tracker.process(blank, 0, 0.)
    for key in ("a", "b", "c"):
        tracker.add_anchor(key)
        assert tracker.estimate(key)["status"] == "lost"
    assert tracker.anchor_count == 3 and tracker.anchor_ids == ("a", "b", "c")
    with pytest.raises(ValueError, match="Anchor limit"):
        tracker.add_anchor("extra")
    for index in range(1, 15):
        tracker.process(blank, index, index * .1)
    assert tracker.state["retained_gray_frames"] == 1
    assert sum(row["tracked_points"] for row in tracker.state["anchors"].values()) <= 36
    tracker.drop_anchor("b")
    assert tracker.anchor_count == 2
    with pytest.raises(KeyError):
        tracker.estimate("b")
    with pytest.raises(KeyError):
        tracker.attach_depth("b", np.ones((14, 14)), np.ones((14, 14)), np.eye(3), 14)


@pytest.mark.parametrize("frame_id,captured", [(0, 1.), (1, 0.), (np.nan, 2.), (2, np.inf)])
def test_nonfinite_or_nonadvancing_frame_metadata_rejected_without_state_change(frame_id, captured):
    tracker = SparseDepthTracker()
    image = np.zeros((60, 80), np.uint8)
    tracker.process(image, 1, 1.)
    with pytest.raises(ValueError):
        tracker.process(image, frame_id, captured)
    assert tracker.state["frame_id"] == 1 and tracker.state["captured_monotonic"] == 1.


def test_resolution_change_and_wrong_model_shape_are_not_silently_reused():
    tracker = SparseDepthTracker()
    tracker.process(np.zeros((100, 200), np.uint8), 1, 1.)
    key = tracker.add_anchor()
    with pytest.raises(ValueError, match="resolution changed"):
        tracker.process(np.zeros((101, 200), np.uint8), 2, 2.)
    with pytest.raises(ValueError, match="model crop shape"):
        tracker.attach_depth(key, np.ones((112, 112)), np.ones((112, 112)), np.eye(3), 112)
    assert tracker.state["original_shape"] == (100, 200)


@pytest.mark.parametrize("options", [dict(max_anchors=4), dict(max_width=0),
    dict(min_inliers=3), dict(min_inlier_ratio=1.1), dict(fb_error=-1), dict(reprojection_error=np.nan)])
def test_invalid_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        SparseDepthTracker(**options)
