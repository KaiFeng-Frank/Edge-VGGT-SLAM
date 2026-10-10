"""Bounded LK/PnP tracking with replenished, triangulated local landmarks.

No model, graph optimizer, descriptor frontend or worker thread is owned here.
All live mutations belong to one caller. A detached observation snapshot can be
replayed in a worker when delayed VGGT depth arrives, then adopted by the owner.
Historical replay is never emitted or counted as a fresh observation pose.

Each accepted depth seed defines a new camera-local gauge. Units are VGGT depth
units, never established metres. A caller must associate that exact anchor with
its graph node; a projective VGGT homography is not a rigid global transform.
"""

from collections import deque
import copy
import time
import uuid

import cv2
import numpy as np

from .fast_tracking import model_pixel_affine, tracking_pixel_affine


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be finite and real")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite and real")
    return value.item() if isinstance(value, np.generic) else value


def _positive_int(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _camera(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    matrix = np.array(value, dtype=np.float64, copy=True)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0., 0., 1.], rtol=0, atol=1e-10)):
        raise ValueError(f"{name} must be a finite invertible pinhole camera matrix")
    if abs(matrix[0, 1]) > 1e-10 or abs(matrix[1, 0]) > 1e-10:
        raise ValueError(f"{name} must have zero skew for OpenCV pinhole operations")
    return matrix


def _rays(uv, K, distortion):
    return cv2.undistortPoints(np.asarray(uv, np.float64).reshape(-1, 1, 2), K, distortion).reshape(-1, 2)


def _project(xyz, Tcw, K, distortion):
    rv = cv2.Rodrigues(Tcw[:3, :3])[0]
    uv = cv2.projectPoints(np.asarray(xyz, np.float64), rv, Tcw[:3, 3], K, distortion)[0].reshape(-1, 2)
    z = (np.asarray(xyz) @ Tcw[:3, :3].T + Tcw[:3, 3])[:, 2]
    return uv, z


def _copy_record(record, readonly=False):
    result = {key: value for key, value in record.items() if key not in ("ids", "uv")}
    for key in ("ids", "uv"):
        result[key] = record[key].copy()
        if readonly:
            result[key].setflags(write=False)
    return result


def _replay_subset(records, stride):
    """Use real observation values only, always including the newest record."""
    selected = records[stride - 1::stride]
    if records and (not selected or selected[-1] is not records[-1]):
        selected.append(records[-1])
    return selected


def _candidate_geometry(candidate, owner_token, config):
    if candidate.get("owner_token") != owner_token:
        raise ValueError("Depth candidate belongs to a different tracker")
    geometry = candidate.get("_local_map")
    if (not isinstance(geometry, _LocalMap) or geometry.config != config
            or geometry.anchor_frame_id != candidate["anchor_frame_id"]
            or geometry.anchor_captured != candidate["anchor_captured_monotonic"]
            or geometry.last_sequence != candidate["last_sequence"]
            or geometry.last_frame_id != candidate["last_frame_id"]
            or geometry.last_captured != candidate["last_captured_monotonic"]
            or not np.array_equal(geometry.K, candidate["K_tracking"])):
        raise ValueError("Depth candidate identity/geometry does not match its measured history")
    return geometry


def _missing(reason, matched=0):
    return dict(pose_valid=False, reason=reason, T_current_from_anchor=None,
                matched_points=int(matched), inliers=0, inlier_ratio=0.,
                median_reprojection_error=None, positive_depth_ratio=None,
                inlier_feature_ids=[], inlier_pixel_bbox_area=None,
                pnp_seconds=0., triangulation_seconds=0., triangulated_points=0,
                triangulation_candidates=0, minimum_triangulation_parallax_degrees=None,
                new_keyframe=False)


class _LocalMap:
    def __init__(self, config, K, distortion, seed, xyz, ids):
        self.config = dict(config)
        self.K = K.copy()
        self.distortion = None if distortion is None else distortion.copy()
        self.anchor_frame_id = seed["frame_id"]
        self.anchor_captured = seed["captured_monotonic"]
        self.anchor_sequence = seed["sequence"]
        self.landmarks = {int(key): point.copy() for key, point in zip(ids, xyz)}
        self.last_seen = {int(key): seed["sequence"] for key in ids}
        self.keyframes = deque([dict(record=_copy_record(seed), Tcw=np.eye(4))],
                               maxlen=config["max_keyframes"])
        self.last_sequence = seed["sequence"]
        self.last_frame_id = seed["frame_id"]
        self.last_captured = seed["captured_monotonic"]
        self.last_result = _missing("anchor_frame_has_no_later_observation", len(ids))
        self.model_from_local = None
        self.model_association_reason = None

    def _triangulate(self, record, Tcw):
        started = time.perf_counter()
        unknown = {int(key): uv for key, uv in zip(record["ids"], record["uv"])
                   if int(key) not in self.landmarks}
        candidate_count, added, angles = 0, 0, []
        # Older baselines are tried first, then newer features can use more
        # recent keyframes. Each point ID is added once, only from real views.
        for keyframe in self.keyframes:
            if not unknown:
                break
            ref = keyframe["record"]
            ref_uv = {int(key): uv for key, uv in zip(ref["ids"], ref["uv"])}
            ids = [key for key in unknown if key in ref_uv]
            if not ids:
                continue
            reference_pose = keyframe["Tcw"]
            center_a = -reference_pose[:3, :3].T @ reference_pose[:3, 3]
            center_b = -Tcw[:3, :3].T @ Tcw[:3, 3]
            if np.linalg.norm(center_a - center_b) <= 1e-10:
                continue
            ua = np.asarray([ref_uv[key] for key in ids])
            ub = np.asarray([unknown[key] for key in ids])
            na, nb = _rays(ua, self.K, self.distortion), _rays(ub, self.K, self.distortion)
            ra = np.column_stack((na, np.ones(len(ids)))) @ reference_pose[:3, :3]
            rb = np.column_stack((nb, np.ones(len(ids)))) @ Tcw[:3, :3]
            cosine = np.sum(ra * rb, axis=1) / (np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1))
            parallax = np.degrees(np.arccos(np.clip(cosine, -1., 1.)))
            candidate_count += len(ids)
            eligible = np.isfinite(parallax) & (parallax >= self.config["min_parallax_degrees"])
            rows = np.flatnonzero(eligible)
            if not len(rows):
                continue
            homogeneous = cv2.triangulatePoints(reference_pose[:3], Tcw[:3], na[rows].T, nb[rows].T).T
            valid_w = np.isfinite(homogeneous).all(axis=1) & (np.abs(homogeneous[:, 3]) > 1e-12)
            xyz = np.zeros((len(rows), 3), np.float64)
            xyz[valid_w] = homogeneous[valid_w, :3] / homogeneous[valid_w, 3, None]
            pa, za = _project(xyz, reference_pose, self.K, self.distortion)
            pb, zb = _project(xyz, Tcw, self.K, self.distortion)
            error_a = np.linalg.norm(pa - ua[rows], axis=1)
            error_b = np.linalg.norm(pb - ub[rows], axis=1)
            keep = (valid_w & np.isfinite(xyz).all(axis=1) & (za > 0) & (zb > 0)
                    & np.isfinite(error_a) & np.isfinite(error_b)
                    & (error_a <= self.config["reprojection_error"])
                    & (error_b <= self.config["reprojection_error"]))
            for local_index in np.flatnonzero(keep):
                if len(self.landmarks) >= self.config["max_landmarks"]:
                    break
                key = ids[rows[local_index]]
                self.landmarks[key] = xyz[local_index].copy()
                self.last_seen[key] = record["sequence"]
                unknown.pop(key)
                added += 1
                angles.append(float(parallax[rows[local_index]]))
        # Culling affects only bounded local storage, not output frame counts.
        active = set(int(key) for key in record["ids"])
        excess = len(self.landmarks) - self.config["max_landmarks"]
        if excess > 0:
            order = sorted(self.landmarks, key=lambda key: (key in active, self.last_seen[key], key))
            for key in order[:excess]:
                del self.landmarks[key]
                del self.last_seen[key]
        return dict(triangulated_points=added, triangulation_candidates=candidate_count,
                    minimum_triangulation_parallax_degrees=min(angles) if angles else None,
                    triangulation_seconds=time.perf_counter() - started)

    def step(self, record):
        if record["sequence"] <= self.last_sequence:
            raise ValueError("Local geometry requires a strictly new observation")
        self.last_sequence = record["sequence"]
        self.last_frame_id, self.last_captured = record["frame_id"], record["captured_monotonic"]
        active = set(int(key) for key in record["ids"])
        # A failed LK track is never matched again under its old ID. Remove its
        # 3D slot now so new observed features can use the finite map budget.
        for key in set(self.landmarks) - active:
            del self.landmarks[key]
            del self.last_seen[key]
        keys = [int(key) for key in record["ids"] if int(key) in self.landmarks]
        result = _missing("insufficient_landmarks", len(keys))
        if len(keys) < self.config["min_inliers"]:
            self.last_result = result
            return result
        lookup = {int(key): uv for key, uv in zip(record["ids"], record["uv"])}
        xyz = np.asarray([self.landmarks[key] for key in keys], np.float64)
        uv = np.asarray([lookup[key] for key in keys], np.float64)
        started = time.perf_counter()
        try:
            ok, rv, tv, inliers = cv2.solvePnPRansac(
                xyz, uv, self.K, self.distortion, iterationsCount=100,
                reprojectionError=self.config["reprojection_error"], confidence=.99,
                flags=cv2.SOLVEPNP_EPNP)
            if not ok or inliers is None or len(inliers) < self.config["min_inliers"]:
                result["reason"] = "pnp_ransac_failed"
            elif len(inliers) / len(keys) < self.config["min_inlier_ratio"]:
                result["reason"] = "insufficient_pnp_inlier_ratio"
            else:
                rows = np.asarray(inliers).reshape(-1)
                rv, tv = cv2.solvePnPRefineLM(xyz[rows], uv[rows], self.K, self.distortion, rv, tv)
                Tcw = np.eye(4)
                Tcw[:3, :3], Tcw[:3, 3] = cv2.Rodrigues(rv)[0], np.asarray(tv).reshape(3)
                projected, positive = _project(xyz, Tcw, self.K, self.distortion)
                errors = np.linalg.norm(projected - uv, axis=1)
                keep = ((positive > 0) & np.isfinite(errors)
                        & (errors <= self.config["reprojection_error"]))
                count = int(keep.sum())
                result.update(inliers=count, inlier_ratio=count / len(keys),
                              median_reprojection_error=float(np.median(errors[keep])) if count else None,
                              positive_depth_ratio=float(np.mean(positive > 0)),
                              inlier_feature_ids=[keys[i] for i in np.flatnonzero(keep)],
                              inlier_pixel_bbox_area=float(np.prod(np.ptp(uv[keep], axis=0))) if count else None)
                if not np.isfinite(Tcw).all():
                    result["reason"] = "nonfinite_pnp_pose"
                elif count < self.config["min_inliers"] or count / len(keys) < self.config["min_inlier_ratio"]:
                    result["reason"] = "refined_pnp_quality_failed"
                else:
                    result.update(pose_valid=True, reason=None, T_current_from_anchor=Tcw)
                    for key in keys:
                        self.last_seen[key] = record["sequence"]
                    # A rejected 3D correspondence is not silently recycled
                    # into the next PnP denominator with its old coordinates.
                    for index in np.flatnonzero(~keep):
                        key = keys[index]
                        del self.landmarks[key]
                        del self.last_seen[key]
        except cv2.error as exc:
            result.update(reason="pnp_opencv_error", error=str(exc))
        result["pnp_seconds"] = time.perf_counter() - started
        if result["pose_valid"]:
            result.update(self._triangulate(record, result["T_current_from_anchor"]))
            previous_keyframe = self.keyframes[-1]["record"]
            if record["sequence"] - previous_keyframe["sequence"] >= self.config["keyframe_interval"]:
                self.keyframes.append(dict(record=_copy_record(record), Tcw=result["T_current_from_anchor"].copy()))
                result["new_keyframe"] = True
        self.last_result = result
        return result


def build_depth_candidate(history_snapshot, *, depth, confidence, K_model,
                          model_image_size, confidence_percentile=25.):
    """Replay detached measured feature observations using only anchor depth.

    Suitable for an independent worker. No live tracker, graph or image source
    is accessed. Failure returns an explicit unavailable candidate, never a
    fallback identity. Finite positive depth/confidence and exact crop are used.
    The configured stride applies only to historical geometry replay, not LK
    measurements or live poses. The exact final observation is always tested.
    """
    started = time.perf_counter()
    records = history_snapshot["records"]
    config = history_snapshot["config"]
    if not records:
        raise ValueError("Depth snapshot must contain the anchor observation")
    anchor = records[0]
    percentile = float(_number(confidence_percentile, "confidence_percentile"))
    if not 0 <= percentile <= 100:
        raise ValueError("confidence_percentile must be in [0,100]")
    A, model_hw = model_pixel_affine(history_snapshot["raw_hw"], model_image_size)
    B = tracking_pixel_affine(history_snapshot["raw_hw"], history_snapshot["tracking_hw"])
    maps = []
    for name, value in (("depth", depth), ("confidence", confidence)):
        value = np.asarray(value)
        if value.shape == model_hw + (1,):
            value = value[..., 0]
        if value.shape != model_hw or value.dtype.kind not in "fiu":
            raise ValueError(f"{name} must match model crop {model_hw}")
        maps.append(value)
    depth, confidence = maps
    estimated_K = B @ np.linalg.inv(A) @ _camera(K_model, "K_model")
    K = (estimated_K if history_snapshot["intrinsic_raw"] is None
         else B @ history_snapshot["intrinsic_raw"])
    K = _camera(K, "K_tracking")
    distortion = history_snapshot["distortion"]
    model_uv = np.column_stack((anchor["uv"], np.ones(len(anchor["uv"])))) @ (A @ np.linalg.inv(B))[:2].T
    samples = np.rint(model_uv).astype(np.int64)
    eligible = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence) & (confidence > 0)
    threshold = float(np.percentile(confidence[eligible], percentile)) if eligible.any() else None
    valid = ((samples[:, 0] >= 0) & (samples[:, 0] < model_hw[1])
             & (samples[:, 1] >= 0) & (samples[:, 1] < model_hw[0]))
    if threshold is None:
        valid[:] = False
    else:
        rows = np.flatnonzero(valid)
        x, y = samples[rows].T
        valid[rows] = eligible[y, x] & (confidence[y, x] >= threshold)
    indices = np.flatnonzero(valid)
    normalized = _rays(anchor["uv"][indices], K, distortion) if len(indices) else np.empty((0, 2))
    values = depth[samples[indices, 1], samples[indices, 0]].astype(np.float64)
    xyz = np.column_stack((normalized, np.ones(len(indices)))) * values[:, None]
    finite = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0)
    xyz, ids = xyz[finite], anchor["ids"][indices][finite]
    xyz, ids = xyz[:config["max_landmarks"]], ids[:config["max_landmarks"]]
    model_from_local = None
    association_reason = "nonlinear_raw_distortion_requires_common_undistorted_model_input"
    if distortion is None or not np.any(distortion):
        model_from_local = np.eye(4)
        model_from_local[:3, :3] = np.linalg.inv(estimated_K) @ K
        association_reason = None
    candidate = dict(owner_token=history_snapshot["owner_token"], available=False,
                     reason="insufficient_valid_anchor_depth", anchor_frame_id=anchor["frame_id"],
                     anchor_captured_monotonic=anchor["captured_monotonic"],
                     last_sequence=records[-1]["sequence"], last_frame_id=records[-1]["frame_id"],
                     last_captured_monotonic=records[-1]["captured_monotonic"],
                     replay_observations=0, replay_valid_poses=0,
                     replay_stride=config["replay_stride"],
                     replay_available_observations=len(records) - 1,
                     replay_skipped_observations=0,
                     background_catchup_rounds=0, background_catchup_observations=0,
                     background_catchup_valid_poses=0, background_catchup_available_observations=0,
                     background_catchup_skipped_observations=0, background_catchup_seconds=0.,
                     retained_depth_points=len(ids), model_hw=model_hw,
                     model_from_raw=A, tracking_from_raw=B, K_tracking=K,
                     K_model_tracking=estimated_K, depth_sampling="nearest_pixel_center",
                     H_model_from_local=model_from_local,
                     model_association_available=model_from_local is not None,
                     model_association_reason=association_reason,
                     intrinsic_source=("vggt_estimated" if history_snapshot["intrinsic_raw"] is None
                                       else "explicit_external"), confidence_threshold=threshold)
    if len(ids) >= config["min_inliers"]:
        local_map = _LocalMap(config, K, distortion, anchor, xyz, ids)
        local_map.model_from_local = model_from_local
        local_map.model_association_reason = association_reason
        for record in _replay_subset(records[1:], config["replay_stride"]):
            result = local_map.step(record)
            candidate["replay_observations"] += 1
            candidate["replay_valid_poses"] += int(result["pose_valid"])
        # It is legitimate to seed the present frame for the next observation;
        # that seed is not a fresh identity pose. A past seed must reach now.
        candidate["available"] = len(records) == 1 or local_map.last_result["pose_valid"]
        candidate["reason"] = None if candidate["available"] else local_map.last_result["reason"]
        if candidate["available"]:
            candidate["_local_map"] = local_map
    candidate["replay_skipped_observations"] = (candidate["replay_available_observations"]
                                                 - candidate["replay_observations"])
    candidate["replay_seconds"] = time.perf_counter() - started
    return candidate


def advance_depth_candidate(candidate, history_snapshot):
    """Advance a detached candidate using copied real tracks in a worker.

    This never accesses the owner or source and never emits historical poses.
    The input candidate is retained unchanged. Missing/expired history or a
    failed final PnP returns unavailable, with no fallback geometry to adopt.
    """
    started = time.perf_counter()
    geometry = _candidate_geometry(candidate, history_snapshot["owner_token"], history_snapshot["config"])
    for name in ("anchor_frame_id", "anchor_captured_monotonic", "last_sequence",
                 "last_frame_id", "last_captured_monotonic"):
        if history_snapshot[name] != candidate[name]:
            raise ValueError("Catchup snapshot does not match the candidate's exact source identity")
    result = {key: value.copy() if isinstance(value, np.ndarray) else value
              for key, value in candidate.items() if key != "_local_map"}
    result.update(available=False, candidate_observation_age_seconds=history_snapshot["candidate_observation_age_seconds"],
                  maximum_candidate_age_seconds=history_snapshot["maximum_candidate_age_seconds"],
                  advance_target_sequence=history_snapshot["target_sequence"],
                  historical_replay=True, emits_fresh_pose=False)
    if not history_snapshot["available"]:
        result["reason"] = history_snapshot["reason"]
    else:
        records = history_snapshot["records"]
        expected = candidate["last_sequence"]
        for record in records:
            expected += 1
            if record["sequence"] != expected:
                raise ValueError("Catchup snapshot must cover consecutive measured observations")
        last = records[-1] if records else dict(sequence=candidate["last_sequence"],
                                                frame_id=candidate["last_frame_id"],
                                                captured_monotonic=candidate["last_captured_monotonic"])
        if (last["sequence"] != history_snapshot["target_sequence"]
                or last["frame_id"] != history_snapshot["target_frame_id"]
                or last["captured_monotonic"] != history_snapshot["target_captured_monotonic"]):
            raise ValueError("Catchup snapshot endpoint differs from the exact newest observation")
        local_map = copy.deepcopy(geometry)
        selected = _replay_subset(records, history_snapshot["config"]["replay_stride"])
        valid = 0
        for record in selected:
            valid += int(local_map.step(record)["pose_valid"])
        result.update(last_sequence=local_map.last_sequence, last_frame_id=local_map.last_frame_id,
                      last_captured_monotonic=local_map.last_captured,
                      background_catchup_rounds=result["background_catchup_rounds"] + 1,
                      background_catchup_available_observations=result["background_catchup_available_observations"] + len(records),
                      background_catchup_observations=result["background_catchup_observations"] + len(selected),
                      background_catchup_valid_poses=result["background_catchup_valid_poses"] + valid,
                      background_catchup_skipped_observations=result["background_catchup_skipped_observations"] + len(records) - len(selected))
        result["available"] = (local_map.last_sequence == local_map.anchor_sequence
                               or local_map.last_result["pose_valid"])
        result["reason"] = None if result["available"] else local_map.last_result["reason"]
        if result["available"]:
            result["_local_map"] = local_map
    elapsed = time.perf_counter() - started
    result["advance_seconds"] = elapsed
    result["background_catchup_seconds"] += elapsed
    return result


class ContinuousDepthTracker:
    """One-owner, bounded feature frontend; all poses use the newest observation.

    PnP thresholds retain the existing sparse frontend's defaults. New points
    need real translation, >=1 degree parallax and positive/reprojecting depth
    in both views. There is no pure-rotation or unit-translation bootstrap.
    """

    def __init__(self, max_width=480, max_features=800, max_landmarks=1200,
                 max_keyframes=6, history_frames=120, history_seconds=6.,
                 min_inliers=20, min_inlier_ratio=.5, reprojection_error=3.,
                 fb_error=1., min_parallax_degrees=1., keyframe_interval=4,
                 intrinsic_raw=None, distortion=None, replay_stride=1):
        self.config = dict(max_width=_positive_int(max_width, "max_width"),
                           max_features=_positive_int(max_features, "max_features"),
                           max_landmarks=_positive_int(max_landmarks, "max_landmarks"),
                           max_keyframes=_positive_int(max_keyframes, "max_keyframes", 2),
                           history_frames=_positive_int(history_frames, "history_frames", 2),
                           min_inliers=_positive_int(min_inliers, "min_inliers", 4),
                           replay_stride=_positive_int(replay_stride, "replay_stride"),
                           keyframe_interval=_positive_int(keyframe_interval, "keyframe_interval"))
        for name, value in (("history_seconds", history_seconds), ("min_inlier_ratio", min_inlier_ratio),
                            ("reprojection_error", reprojection_error), ("fb_error", fb_error),
                            ("min_parallax_degrees", min_parallax_degrees)):
            self.config[name] = float(_number(value, name))
        if (history_seconds <= 0 or not 0 < min_inlier_ratio <= 1 or reprojection_error <= 0
                or fb_error < 0 or not 0 < min_parallax_degrees < 90
                or min(max_features, max_landmarks) < min_inliers):
            raise ValueError("Invalid tracking limits or geometric quality thresholds")
        self.intrinsic_raw = None if intrinsic_raw is None else _camera(intrinsic_raw, "intrinsic_raw")
        self.distortion = None
        if distortion is not None:
            coefficients = np.asarray(distortion, np.float64).reshape(-1)
            if self.intrinsic_raw is None or len(coefficients) not in (4, 5, 8, 12, 14) or not np.isfinite(coefficients).all():
                raise ValueError("Distortion needs explicit intrinsic_raw and finite OpenCV coefficients")
            self.distortion = coefficients.copy()
        self._owner_token = uuid.uuid4().hex
        self._gray = None
        self._raw_hw = self._tracking_hw = None
        self._frame_id = self._captured = None
        self._sequence = -1
        self._next_feature = 0
        self._ids = np.empty(0, np.int64)
        self._uv = np.empty((0, 2), np.float32)
        self._history = deque()
        self._map = None
        self._gauge_id = -1
        self._counts = dict(observations=0, fresh_valid_poses=0, lost_observations=0,
                            waiting_observations=0, features_detected=0,
                            triangulated_landmarks=0, adopted_depth_seeds=0,
                            rejected_depth_seeds=0, historical_replay_poses=0,
                            maximum_history_frames=0, maximum_landmarks=0)

    @property
    def state(self):
        return dict(frame_id=self._frame_id, captured_monotonic=self._captured,
                    sequence=self._sequence, gauge_id=self._gauge_id,
                    raw_hw=self._raw_hw, tracking_hw=self._tracking_hw,
                    retained_gray_frames=int(self._gray is not None),
                    tracked_features=len(self._ids), history_frames=len(self._history),
                    retained_history_seconds=(self._history[-1]["captured_monotonic"] -
                                              self._history[0]["captured_monotonic"]) if self._history else 0.,
                    oldest_history_frame_id=self._history[0]["frame_id"] if self._history else None,
                    landmarks=len(self._map.landmarks) if self._map else 0,
                    local_keyframes=len(self._map.keyframes) if self._map else 0,
                    anchor_frame_id=self._map.anchor_frame_id if self._map else None,
                    anchor_captured_monotonic=self._map.anchor_captured if self._map else None)

    def metrics(self):
        return dict(self._counts, **self.state, frontend="continuous_shi_tomasi_lk_depth_pnp",
                    replay_stride=self.config["replay_stride"],
                    camera_intrinsic_source="explicit_external" if self.intrinsic_raw is not None else "vggt_estimated",
                    distortion_compensated=self.distortion is not None and bool(np.any(self.distortion)),
                    metric_scale_validated=False, accuracy_validated=False, real_time_validated=False)

    def _flow(self, gray):
        if not len(self._uv):
            return None
        previous = self._uv
        current, status, _ = cv2.calcOpticalFlowPyrLK(
            self._gray, gray, previous[:, None], None, winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        if current is None or status is None:
            self._ids, self._uv = self._ids[:0], self._uv[:0]
            return "forward_lk_failed"
        current = current.reshape(-1, 2)
        h, w = gray.shape
        valid = (status.reshape(-1).astype(bool) & np.isfinite(current).all(axis=1)
                 & (current[:, 0] >= 0) & (current[:, 0] < w)
                 & (current[:, 1] >= 0) & (current[:, 1] < h))
        rows = np.flatnonzero(valid)
        keep = np.zeros(len(previous), bool)
        if len(rows):
            back, back_status, _ = cv2.calcOpticalFlowPyrLK(
                gray, self._gray, current[rows, None], None, winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
            if back is not None and back_status is not None:
                backward = back.reshape(-1, 2)
                good = (back_status.reshape(-1).astype(bool) & np.isfinite(backward).all(axis=1)
                        & (np.linalg.norm(backward - previous[rows], axis=1) <= self.config["fb_error"]))
                keep[rows[good]] = True
        self._ids, self._uv = self._ids[keep], current[keep].astype(np.float32)
        return None

    def _detect(self, gray):
        capacity = self.config["max_features"] - len(self._ids)
        if capacity <= 0:
            return 0
        mask = np.full(gray.shape, 255, np.uint8)
        for x, y in np.rint(self._uv).astype(int):
            cv2.circle(mask, (int(x), int(y)), 8, 0, -1)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=capacity, qualityLevel=.01,
                                         minDistance=8, blockSize=7, mask=mask)
        if corners is None:
            return 0
        uv = np.asarray(corners, np.float32).reshape(-1, 2)[:capacity]
        ids = np.arange(self._next_feature, self._next_feature + len(uv), dtype=np.int64)
        self._next_feature += len(uv)
        self._ids = np.concatenate((self._ids, ids))
        self._uv = np.concatenate((self._uv, uv))
        return len(uv)

    def process(self, frame, frame_id, captured_monotonic):
        started = time.perf_counter()
        frame_id = _number(frame_id, "frame_id")
        captured = float(_number(captured_monotonic, "captured_monotonic"))
        if self._frame_id is not None and (frame_id <= self._frame_id or captured <= self._captured):
            raise ValueError("Frame ID and capture timestamp must strictly increase")
        frame = np.asarray(frame)
        if (frame.dtype != np.uint8 or frame.ndim not in (2, 3) or not frame.size
                or (frame.ndim == 3 and frame.shape[2] != 3)):
            raise ValueError("Frame must be a nonempty uint8 gray or BGR image")
        raw_hw = tuple(frame.shape[:2])
        if self._raw_hw is not None and raw_hw != self._raw_hw:
            raise ValueError("Resolution changed; create a new continuous tracker")
        w = min(raw_hw[1], self.config["max_width"])
        h = max(1, round(raw_hw[0] * w / raw_hw[1]))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA) if gray.shape != (h, w) else gray.copy()
        flow_started = time.perf_counter()
        flow_error = None
        if self._gray is not None:
            try:
                flow_error = self._flow(gray)
            except cv2.error as exc:
                self._ids, self._uv = self._ids[:0], self._uv[:0]
                flow_error = str(exc)
        flow_seconds = time.perf_counter() - flow_started
        new_features = self._detect(gray)
        self._gray, self._raw_hw, self._tracking_hw = gray, raw_hw, gray.shape
        self._frame_id, self._captured = frame_id, captured
        self._sequence += 1
        record = dict(frame_id=frame_id, captured_monotonic=captured, sequence=self._sequence,
                      ids=self._ids.copy(), uv=self._uv.copy())
        self._history.append(record)
        while (len(self._history) > self.config["history_frames"] or
               captured - self._history[0]["captured_monotonic"] > self.config["history_seconds"]):
            self._history.popleft()
        result = self._map.step(record) if self._map is not None else _missing("no_vggt_geometry")
        result = dict(result)
        result.update(frame_id=frame_id, captured_monotonic=captured, sequence=self._sequence,
                      status="tracking" if result["pose_valid"] else "lost" if self._map is not None else "waiting_geometry",
                      anchor_frame_id=self._map.anchor_frame_id if self._map else None,
                      anchor_captured_monotonic=self._map.anchor_captured if self._map else None,
                      gauge_id=self._gauge_id, K_tracking=self._map.K.copy() if self._map else None,
                      H_model_from_local=(self._map.model_from_local.copy()
                                          if self._map is not None and self._map.model_from_local is not None else None),
                      model_association_available=self._map is not None and self._map.model_from_local is not None,
                      model_association_reason=self._map.model_association_reason if self._map else "no_vggt_geometry",
                      tracked_features=len(self._ids), new_features=new_features,
                      landmarks=len(self._map.landmarks) if self._map else 0,
                      optical_flow_seconds=flow_seconds, tracking_error=flow_error,
                      processing_seconds=time.perf_counter() - started,
                      translation_units="vggt_anchor_depth_units" if self._map else None,
                      metric_scale_validated=False, historical_replay=False)
        if result["T_current_from_anchor"] is not None:
            result["T_current_from_anchor"] = result["T_current_from_anchor"].copy()
        self._counts["observations"] += 1
        self._counts["fresh_valid_poses"] += int(result["pose_valid"])
        self._counts["lost_observations"] += int(result["status"] == "lost")
        self._counts["waiting_observations"] += int(result["status"] == "waiting_geometry")
        self._counts["features_detected"] += new_features
        self._counts["triangulated_landmarks"] += result["triangulated_points"]
        self._counts["maximum_history_frames"] = max(self._counts["maximum_history_frames"], len(self._history))
        self._counts["maximum_landmarks"] = max(self._counts["maximum_landmarks"], result["landmarks"])
        return result

    def snapshot_depth_history(self, anchor_frame_id, anchor_captured_monotonic):
        """Copy exact retained source observations, safe to hand to a worker."""
        frame_id = _number(anchor_frame_id, "anchor_frame_id")
        captured = float(_number(anchor_captured_monotonic, "anchor_captured_monotonic"))
        index = next((i for i, record in enumerate(self._history)
                      if record["frame_id"] == frame_id and record["captured_monotonic"] == captured), None)
        if index is None:
            raise ValueError("Exact anchor observation is absent or history expired")
        return dict(owner_token=self._owner_token, config=dict(self.config), raw_hw=self._raw_hw,
                    tracking_hw=self._tracking_hw,
                    intrinsic_raw=self.intrinsic_raw.copy() if self.intrinsic_raw is not None else None,
                    distortion=self.distortion.copy() if self.distortion is not None else None,
                    records=[_copy_record(record, readonly=True) for record in list(self._history)[index:]])

    def snapshot_candidate_catchup(self, candidate):
        """Owner-only value copy; candidate age is measured from real captures.

        The upper layer additionally limits wall time/rounds. The historical
        capture span may not exceed history_seconds, even if repeated worker
        advances keep the candidate close to the latest observation.
        """
        _candidate_geometry(candidate, self._owner_token, self.config)
        last_sequence = candidate["last_sequence"]
        if last_sequence > self._sequence:
            raise ValueError("Depth candidate refers to a future observation")
        age = self._captured - candidate["anchor_captured_monotonic"]
        snapshot = dict(owner_token=self._owner_token, config=dict(self.config), available=True, reason=None,
                        **{key: candidate[key] for key in ("anchor_frame_id", "anchor_captured_monotonic",
                           "last_sequence", "last_frame_id", "last_captured_monotonic")},
                        target_sequence=self._sequence, target_frame_id=self._frame_id,
                        target_captured_monotonic=self._captured, candidate_observation_age_seconds=age,
                        maximum_candidate_age_seconds=self.config["history_seconds"], records=[])
        records = [record for record in self._history if record["sequence"] > last_sequence]
        if self._map is not None and candidate["anchor_frame_id"] <= self._map.anchor_frame_id:
            snapshot.update(available=False, reason="anchor_not_newer_than_active_seed")
        elif age > self.config["history_seconds"]:
            snapshot.update(available=False, reason="candidate_anchor_age_exceeded")
        elif self._sequence > last_sequence and (not records or records[0]["sequence"] != last_sequence + 1):
            snapshot.update(available=False, reason="candidate_catchup_history_expired")
        else:
            snapshot["records"] = [_copy_record(record, readonly=True) for record in records]
        return snapshot

    def apply_depth_candidate(self, candidate):
        """Catch up a detached seed, then swap gauges without emitting old poses."""
        started = time.perf_counter()
        if candidate.get("owner_token") != self._owner_token:
            raise ValueError("Depth candidate belongs to a different tracker")
        info = {key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in candidate.items() if key != "_local_map"}
        info.update(adopted=False, catchup_observations=0, catchup_valid_poses=0,
                    catchup_available_observations=0, catchup_skipped_observations=0,
                    gauge_id=self._gauge_id, historical_replay=True, emits_fresh_pose=False)
        if not candidate["available"]:
            self._counts["rejected_depth_seeds"] += 1
            info["apply_seconds"] = time.perf_counter() - started
            return info
        geometry = _candidate_geometry(candidate, self._owner_token, self.config)
        snapshot = self.snapshot_candidate_catchup(candidate)
        records = snapshot["records"]
        info["catchup_available_observations"] = len(records)
        info["candidate_observation_age_seconds"] = snapshot["candidate_observation_age_seconds"]
        info["maximum_candidate_age_seconds"] = snapshot["maximum_candidate_age_seconds"]
        if not snapshot["available"]:
            info.update(reason=snapshot["reason"], apply_seconds=time.perf_counter() - started)
            self._counts["rejected_depth_seeds"] += 1
            return info
        local_map = copy.deepcopy(geometry)
        for record in _replay_subset(records, self.config["replay_stride"]):
            result = local_map.step(record)
            info["catchup_observations"] += 1
            info["catchup_valid_poses"] += int(result["pose_valid"])
        info["catchup_skipped_observations"] = len(records) - info["catchup_observations"]
        if local_map.last_sequence != local_map.anchor_sequence and not local_map.last_result["pose_valid"]:
            info.update(reason=local_map.last_result["reason"], apply_seconds=time.perf_counter() - started)
            self._counts["rejected_depth_seeds"] += 1
            return info
        self._map = local_map
        self._gauge_id += 1
        self._counts["adopted_depth_seeds"] += 1
        self._counts["historical_replay_poses"] += (info["replay_valid_poses"]
                                                  + info["background_catchup_valid_poses"]
                                                  + info["catchup_valid_poses"])
        self._counts["maximum_landmarks"] = max(self._counts["maximum_landmarks"], len(local_map.landmarks))
        info.update(adopted=True, reason=None, gauge_id=self._gauge_id,
                    apply_seconds=time.perf_counter() - started)
        return info

    def attach_depth(self, anchor_frame_id, anchor_captured_monotonic, **depth_arguments):
        """Synchronous convenience path; replay time is explicitly separate."""
        history = self.snapshot_depth_history(anchor_frame_id, anchor_captured_monotonic)
        return self.apply_depth_candidate(build_depth_candidate(history, **depth_arguments))
