"""Bounded sparse RGB/depth tracking in an anchor's local camera frame.

This component does not update the VGGT map or export global poses. Depth
belongs to the original anchor image, including when it arrives later.
"""

import time

import cv2
import numpy as np


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value.item() if isinstance(value, np.generic) else value


def _shape(shape):
    if len(shape) != 2 or any(not isinstance(v, (int, np.integer)) or v < 1 for v in shape):
        raise ValueError("Image shape must be (positive height, positive width)")
    return tuple(int(v) for v in shape)


def tracking_pixel_affine(original_shape, tracking_shape):
    """Map original pixel centers to the resized tracking image (matrix B)."""
    height, width = _shape(original_shape)
    tracking_height, tracking_width = _shape(tracking_shape)
    sx, sy = tracking_width / width, tracking_height / height
    return np.array([[sx, 0., (sx - 1.) / 2.],
                     [0., sy, (sy - 1.) / 2.], [0., 0., 1.]])


def model_pixel_affine(original_shape, model_image_size):
    """Return original-to-VGGT-crop A and expected (height, width)."""
    height, width = _shape(original_shape)
    model_width = _positive_int(model_image_size, "model_image_size")
    if model_width % 14:
        raise ValueError("model_image_size must be divisible by 14")
    new_height = max(14, round(height * (model_width / width) / 14) * 14)
    crop_y = max(0, (new_height - model_width) // 2)
    affine = tracking_pixel_affine((height, width), (new_height, model_width))
    affine[1, 2] -= crop_y
    return affine, (min(new_height, model_width), model_width)


def _transform_pixels(points, affine):
    return np.column_stack((points, np.ones(len(points)))) @ affine[:2].T


class _Anchor:
    def __init__(self, anchor_id, frame_id, captured, pixels, affine):
        self.anchor_id = anchor_id
        self.frame_id = frame_id
        self.captured = captured
        self.original_pixels = pixels.copy()
        self.current_pixels = pixels.copy()
        self.tracking_affine = affine.copy()
        self.points3d = None
        self.intrinsic = None
        self.depth_info = None

    def keep(self, mask, current_pixels=None):
        self.original_pixels = self.original_pixels[mask]
        self.current_pixels = (self.current_pixels if current_pixels is None else current_pixels)[mask]
        if self.points3d is not None:
            self.points3d = self.points3d[mask]


class SparseDepthTracker:
    """Continuous LK tracking plus independently validated local PnP poses.

    One resized gray frame and at most three anchors are retained. All anchors
    share a single forward/backward LK pair per frame. Call from one thread.
    """

    def __init__(self, max_width=480, max_corners=400, max_anchors=3,
                 min_inliers=20, min_inlier_ratio=.5, reprojection_error=3., fb_error=1.0):
        self.max_width = _positive_int(max_width, "max_width")
        self.max_corners = _positive_int(max_corners, "max_corners")
        self.max_anchors = _positive_int(max_anchors, "max_anchors")
        self.min_inliers = _positive_int(min_inliers, "min_inliers")
        if self.max_anchors > 3 or self.min_inliers < 4:
            raise ValueError("max_anchors must be <=3 and min_inliers must be >=4")
        self.min_inlier_ratio = float(_finite_number(min_inlier_ratio, "min_inlier_ratio"))
        self.reprojection_error = float(_finite_number(reprojection_error, "reprojection_error"))
        self.fb_error = float(_finite_number(fb_error, "fb_error"))
        if not 0 <= self.min_inlier_ratio <= 1 or self.reprojection_error <= 0 or self.fb_error < 0:
            raise ValueError("Require 0<=min_inlier_ratio<=1, reprojection_error>0 and fb_error>=0")
        self._previous_gray = None
        self._original_shape = None
        self._tracking_affine = None
        self._frame_id = None
        self._captured = None
        self._anchors = {}

    @property
    def anchor_ids(self):
        return tuple(self._anchors)

    @property
    def anchor_count(self):
        return len(self._anchors)

    @property
    def ready_anchor_ids(self):
        return tuple(key for key, anchor in self._anchors.items()
                     if anchor.points3d is not None and len(anchor.points3d) >= self.min_inliers)

    @property
    def state(self):
        return dict(frame_id=self._frame_id, captured_monotonic=self._captured,
                    original_shape=self._original_shape,
                    tracking_shape=None if self._previous_gray is None else self._previous_gray.shape,
                    retained_gray_frames=int(self._previous_gray is not None),
                    anchor_count=self.anchor_count, ready_anchor_ids=self.ready_anchor_ids,
                    anchors={key: dict(frame_id=anchor.frame_id,
                                       tracked_points=len(anchor.current_pixels),
                                       depth_attached=anchor.points3d is not None,
                                       depth_ready=anchor.points3d is not None
                                       and len(anchor.points3d) >= self.min_inliers)
                             for key, anchor in self._anchors.items()})

    def process(self, frame, frame_id, captured_monotonic):
        started = time.perf_counter()
        frame_id = _finite_number(frame_id, "frame_id")
        captured = float(_finite_number(captured_monotonic, "captured_monotonic"))
        if self._frame_id is not None and (frame_id <= self._frame_id or captured <= self._captured):
            raise ValueError("frame_id and captured_monotonic must strictly increase")
        frame = np.asarray(frame)
        if (frame.dtype != np.uint8 or frame.ndim not in (2, 3) or not frame.size
                or (frame.ndim == 3 and frame.shape[2] != 3)):
            raise ValueError("frame must be a nonempty uint8 gray or BGR image")
        original_shape = frame.shape[:2]
        if self._original_shape is not None and original_shape != self._original_shape:
            raise ValueError("Source resolution changed; create a new SparseDepthTracker")
        width = min(original_shape[1], self.max_width)
        height = max(1, round(original_shape[0] * width / original_shape[1]))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        gray = (cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
                if gray.shape != (height, width) else gray.copy())
        affine = tracking_pixel_affine(original_shape, gray.shape)
        anchors = list(self._anchors.values())
        sizes = [len(anchor.current_pixels) for anchor in anchors]
        tracking_error = None
        flow_started = time.perf_counter()
        if sum(sizes):
            previous = np.concatenate([anchor.current_pixels for anchor in anchors]).astype(np.float32)
            current = previous.copy()
            keep = np.zeros(len(previous), dtype=bool)
            try:
                forward, status, _ = cv2.calcOpticalFlowPyrLK(
                    self._previous_gray, gray, previous.reshape(-1, 1, 2), None,
                    winSize=(21, 21), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
                if forward is not None and status is not None:
                    current = forward.reshape(-1, 2)
                    valid = (status.reshape(-1).astype(bool) & np.isfinite(current).all(axis=1)
                             & (current[:, 0] >= 0) & (current[:, 0] < width)
                             & (current[:, 1] >= 0) & (current[:, 1] < height))
                    indices = np.flatnonzero(valid)
                    if len(indices):
                        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
                            gray, self._previous_gray, current[indices].reshape(-1, 1, 2), None,
                            winSize=(21, 21), maxLevel=3,
                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
                        if backward is not None and back_status is not None:
                            backward = backward.reshape(-1, 2)
                            retained = (back_status.reshape(-1).astype(bool)
                                        & np.isfinite(backward).all(axis=1)
                                        & (np.linalg.norm(backward - previous[indices], axis=1) <= self.fb_error))
                            keep[indices[retained]] = True
            except cv2.error as exc:
                tracking_error = str(exc)
                keep[:] = False
            offset = 0
            for anchor, size in zip(anchors, sizes):
                anchor.keep(keep[offset:offset+size], current[offset:offset+size])
                offset += size
        flow_seconds = time.perf_counter() - flow_started
        self._previous_gray, self._original_shape = gray, original_shape
        self._tracking_affine, self._frame_id, self._captured = affine, frame_id, captured
        summary = {}
        scales = np.diag(affine)[:2]
        for key, anchor in self._anchors.items():
            motion = (anchor.current_pixels - anchor.original_pixels) / scales
            summary[key] = dict(tracked_points=len(motion), mean_disparity_original_pixels=
                                float(np.linalg.norm(motion, axis=1).mean()) if len(motion) else None)
        return dict(frame_id=frame_id, captured_monotonic=captured,
                    processing_seconds=time.perf_counter()-started,
                    optical_flow_seconds=flow_seconds, tracking_error=tracking_error,
                    anchors=summary)

    def add_anchor(self, anchor_id=None):
        if self._previous_gray is None:
            raise RuntimeError("Process a frame before adding an anchor")
        anchor_id = self._frame_id if anchor_id is None else anchor_id
        if anchor_id in self._anchors:
            raise ValueError(f"Anchor already exists: {anchor_id}")
        if len(self._anchors) >= self.max_anchors:
            raise ValueError(f"Anchor limit {self.max_anchors} reached; drop an anchor first")
        pixels = cv2.goodFeaturesToTrack(self._previous_gray, maxCorners=self.max_corners,
                                       qualityLevel=.01, minDistance=8, blockSize=7)
        pixels = np.empty((0, 2), dtype=np.float32) if pixels is None else pixels.reshape(-1, 2)
        self._anchors[anchor_id] = _Anchor(anchor_id, self._frame_id, self._captured,
                                         pixels, self._tracking_affine)
        return anchor_id

    def drop_anchor(self, anchor_id):
        del self._anchors[anchor_id]

    def attach_depth(self, anchor_id, depth, confidence, intrinsic, model_image_size,
                     confidence_percentile=25.):
        anchor = self._anchors[anchor_id]
        percentile = float(_finite_number(confidence_percentile, "confidence_percentile"))
        if not 0 <= percentile <= 100:
            raise ValueError("confidence_percentile must be between 0 and 100")
        affine, expected_shape = model_pixel_affine(self._original_shape, model_image_size)
        maps = []
        for name, value in (("depth", depth), ("confidence", confidence)):
            value = np.asarray(value)
            if value.shape == expected_shape + (1,):
                value = value[..., 0]
            real_numeric = (np.issubdtype(value.dtype, np.integer)
                            or np.issubdtype(value.dtype, np.floating))
            if value.shape != expected_shape or not real_numeric:
                raise ValueError(f"{name} must be real numeric with model crop shape "
                                 f"{expected_shape} or {expected_shape + (1,)}")
            maps.append(value)
        depth, confidence = maps
        intrinsic = np.asarray(intrinsic)
        if not (np.issubdtype(intrinsic.dtype, np.integer)
                or np.issubdtype(intrinsic.dtype, np.floating)):
            raise ValueError("intrinsic must be a real numeric camera matrix")
        intrinsic = intrinsic.astype(np.float64, copy=False)
        if (intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all()
                or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
                or not np.allclose(intrinsic[2], [0, 0, 1], rtol=0, atol=1e-8)):
            raise ValueError("intrinsic must be a finite 3x3 camera matrix with positive focal lengths")
        try:
            tracking_intrinsic = anchor.tracking_affine @ np.linalg.inv(affine) @ intrinsic
            inverse_intrinsic = np.linalg.inv(tracking_intrinsic)
        except np.linalg.LinAlgError as exc:
            raise ValueError("intrinsic must be invertible") from exc
        if not np.isfinite(tracking_intrinsic).all() or not np.isfinite(inverse_intrinsic).all():
            raise ValueError("intrinsic produces a non-finite tracking camera matrix")
        model_pixels = _transform_pixels(anchor.original_pixels,
                                         affine @ np.linalg.inv(anchor.tracking_affine))
        indices = np.rint(model_pixels).astype(np.int64)
        height, width = expected_shape
        roi = ((indices[:, 0] >= 0) & (indices[:, 0] < width)
               & (indices[:, 1] >= 0) & (indices[:, 1] < height))
        eligible = (np.isfinite(depth) & (depth > 0) & np.isfinite(confidence) & (confidence > 0))
        threshold = float(np.percentile(confidence[eligible], percentile)) if eligible.any() else None
        keep = np.zeros(len(model_pixels), dtype=bool)
        rows = np.flatnonzero(roi)
        if len(rows) and threshold is not None:
            x, y = indices[rows].T
            keep[rows] = eligible[y, x] & (confidence[y, x] >= threshold)
        points = anchor.original_pixels[keep]
        rays = np.column_stack((points, np.ones(len(points)))) @ inverse_intrinsic.T
        sample_indices = indices[keep]
        sampled_depth = depth[sample_indices[:, 1], sample_indices[:, 0]].astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            points3d = rays * sampled_depth[:, None]
        finite_points = np.isfinite(points3d).all(axis=1) & (points3d[:, 2] > 0)
        keep[np.flatnonzero(keep)[~finite_points]] = False
        points3d = points3d[finite_points]
        anchor.keep(keep)
        anchor.points3d, anchor.intrinsic = points3d, tracking_intrinsic
        anchor.depth_info = dict(depth_sampling="nearest_numpy_rint",
                                 confidence_percentile=percentile, confidence_threshold=threshold,
                                 model_shape=expected_shape, retained_depth_points=len(points3d))
        return dict(anchor_id=anchor_id, anchor_frame_id=anchor.frame_id,
                    K_tracking=tracking_intrinsic.copy(),
                    model_from_raw=affine.copy(), tracking_from_raw=anchor.tracking_affine.copy(),
                    model_hw=expected_shape, tracking_hw=self._previous_gray.shape,
                    **anchor.depth_info)

    def estimate(self, anchor_id):
        anchor = self._anchors[anchor_id]
        count = 0 if anchor.points3d is None else len(anchor.points3d)
        result = dict(status="lost", reason=None, frame_id=self._frame_id,
                      captured_monotonic=self._captured, anchor_id=anchor_id,
                      anchor_frame_id=anchor.frame_id, T_current_from_anchor=None,
                      matched_points=count, tracked_points=len(anchor.current_pixels),
                      inliers=0, inlier_ratio=0., median_reprojection_error=None,
                      positive_depth_ratio=None, pnp_seconds=0.,
                      reprojection_error_units="tracking_pixels", depth_sampling="nearest_numpy_rint")
        result["K_tracking"] = None if anchor.intrinsic is None else anchor.intrinsic.copy()
        if len(anchor.current_pixels) < self.min_inliers:
            result["reason"] = "insufficient_tracks"
            return result
        if anchor.points3d is None:
            result.update(status="waiting_depth", reason="anchor_depth_not_attached")
            return result
        if self._frame_id == anchor.frame_id:
            result["reason"] = "no_later_frame"
            return result
        if count < self.min_inliers:
            result["reason"] = "insufficient_depth_correspondences"
            return result
        started = time.perf_counter()
        try:
            ok, rotation, translation, inliers = cv2.solvePnPRansac(
                anchor.points3d, anchor.current_pixels.astype(np.float64), anchor.intrinsic, None,
                iterationsCount=100, reprojectionError=self.reprojection_error,
                confidence=.99, flags=cv2.SOLVEPNP_EPNP)
            if not ok or inliers is None or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
                result["reason"] = "pnp_failed"
                return result
            inliers = np.unique(inliers.reshape(-1))
            result.update(inliers=len(inliers), inlier_ratio=len(inliers)/count)
            if len(inliers) < self.min_inliers or len(inliers)/count < self.min_inlier_ratio:
                result["reason"] = "insufficient_pnp_inliers"
                return result
            rotation, translation = cv2.solvePnPRefineLM(
                anchor.points3d[inliers], anchor.current_pixels[inliers].astype(np.float64),
                anchor.intrinsic, None, rotation, translation)
            if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
                result["reason"] = "nonfinite_refined_pose"
                return result
            matrix, _ = cv2.Rodrigues(rotation)
            current3d = anchor.points3d @ matrix.T + translation.reshape(1, 3)
            positive = np.isfinite(current3d).all(axis=1) & (current3d[:, 2] > 1e-6)
            result["positive_depth_ratio"] = float(positive.mean())
            projected, _ = cv2.projectPoints(anchor.points3d, rotation, translation, anchor.intrinsic, None)
            errors = np.linalg.norm(projected.reshape(-1, 2) - anchor.current_pixels, axis=1)
            inliers = inliers[positive[inliers] & np.isfinite(errors[inliers])
                              & (errors[inliers] <= self.reprojection_error)]
            result.update(inliers=len(inliers), inlier_ratio=len(inliers)/count,
                          median_reprojection_error=float(np.median(errors[inliers])) if len(inliers) else None)
            if len(inliers) < self.min_inliers or len(inliers)/count < self.min_inlier_ratio:
                result["reason"] = "refined_pose_failed_validation"
                return result
            transform = np.eye(4)
            transform[:3, :3], transform[:3, 3] = matrix, translation.reshape(3)
            if not np.isfinite(transform).all():
                result["reason"] = "nonfinite_pose"
                return result
            result.update(status="tracking", reason=None, T_current_from_anchor=transform)
        except cv2.error as exc:
            result["reason"] = f"pnp_error: {exc}"
        finally:
            result["pnp_seconds"] = time.perf_counter()-started
        return result
