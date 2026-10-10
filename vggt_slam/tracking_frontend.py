"""Online scheduling for the opt-in sparse frontend and delayed VGGT anchors.

The main acquisition thread owns all feature state. The backend only publishes
copied arrays after graph optimization; it never accesses the tracker. Local
SE3 and projective global geometry have distinct output fields. This component
bounds its feature/latency buffers, not the existing dense GraphMap.
"""
from collections import Counter, deque
import json
import math
from pathlib import Path
import queue
import threading
import time

import numpy as np


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


class OnlineTrackingFrontend:
    def __init__(self, output, min_disparity=30.0, max_width=480,
                 max_corners=400, tracker=None, clock=time.monotonic):
        if not math.isfinite(min_disparity) or min_disparity < 0:
            raise ValueError("min_disparity must be finite and nonnegative")
        if tracker is None:
            from vggt_slam.fast_tracking import SparseDepthTracker
            tracker = SparseDepthTracker(max_width=max_width, max_corners=max_corners,
                                         max_anchors=3)
        self.tracker, self.clock, self.min_disparity = tracker, clock, min_disparity
        self.stream = Path(output).open("x", buffering=1)
        self.publications = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.active = self.staged = self.inflight = self.ready_candidate = self.selection = None
        self.geometry = {}
        self.latest_graph_revision = 0
        self.first_capture = self.first_pose_emit = self.last_pose_emit = None
        self.pose_ages = deque(maxlen=4096)
        self.processing_times = deque(maxlen=4096)
        self.pose_age_sum = self.pose_age_max = 0.0
        self.reasons = Counter()
        self.rejected_anchor_reasons = Counter()
        self.counters = Counter()
        self.closed = False

    def _drop(self, anchor):
        if anchor is None:
            return
        self.tracker.drop_anchor(anchor)
        with self.lock:
            self.geometry.pop(anchor, None)
        if self.selection == anchor:
            self.selection = None

    def requested_nodes(self):
        """Backend may read this small copy; graph access stays in its thread."""
        with self.lock:
            return {key: value["graph_node_id"] for key, value in self.geometry.items()}

    def submitted(self, anchor_id):
        if self.closed:
            raise RuntimeError("Tracking frontend is closed")
        # At EOF there may be a completed publication but no next frame. It
        # can be consumed for bookkeeping without emitting a retrospective pose.
        self.accept_publications()
        if self.inflight is not None or self.staged != anchor_id:
            raise RuntimeError("Tracking anchor does not match the submitted VGGT batch")
        self.inflight, self.staged = anchor_id, None
        self.counters["submitted_anchor_jobs"] += 1

    def publish(self, *, anchor_id, depth, confidence, intrinsic, image_size,
                graph_node_id, homography, graph_revision, node_updates,
                backend_ready_monotonic):
        """Called by the backend only after a successful optimized map update."""
        if self.closed:
            raise RuntimeError("Tracking frontend is closed")
        publication = dict(anchor_id=anchor_id, depth=np.array(depth, copy=True),
                           confidence=np.array(confidence, copy=True),
                           intrinsic=np.array(intrinsic, copy=True), image_size=image_size,
                           graph_node_id=graph_node_id, homography=np.array(homography, copy=True),
                           graph_revision=int(graph_revision),
                           node_updates={key: np.array(value, copy=True)
                                         for key, value in node_updates.items()},
                           backend_ready_monotonic=float(backend_ready_monotonic))
        if not math.isfinite(publication["backend_ready_monotonic"]):
            raise ValueError("Nonfinite backend publication time")
        # There is only one inference job. Losing a second publication would
        # conceal a scheduling bug, so a full queue is an explicit failure.
        self.publications.put_nowait(publication)
        with self.lock:
            self.counters["backend_publications"] += 1

    def accept_publications(self):
        try:
            item = self.publications.get_nowait()
        except queue.Empty:
            return
        anchor = item["anchor_id"]
        if anchor != self.inflight or anchor not in self.tracker.anchor_ids:
            raise RuntimeError("Backend returned an unknown or expired tracking anchor")
        if item["backend_ready_monotonic"] > self.clock():
            raise RuntimeError("Backend publication has a future timestamp")
        if item["graph_revision"] <= self.latest_graph_revision:
            raise RuntimeError("Backend graph revision did not advance")
        self.latest_graph_revision = item["graph_revision"]
        self.tracker.attach_depth(anchor, item["depth"], item["confidence"],
                                  item["intrinsic"], item["image_size"])
        with self.lock:
            for key, value in self.geometry.items():
                if key in item["node_updates"]:
                    value["homography"] = item["node_updates"][key]
                    value["graph_revision"] = item["graph_revision"]
            self.geometry[anchor] = dict(graph_node_id=item["graph_node_id"],
                homography=item["homography"], graph_revision=item["graph_revision"],
                backend_ready_monotonic=item["backend_ready_monotonic"])
        if self.ready_candidate is not None and self.ready_candidate != self.active:
            self._drop(self.ready_candidate)
        self.ready_candidate, self.inflight = anchor, None
        self.counters["depth_publications_consumed"] += 1

    def observe(self, frame, frame_id, captured_monotonic):
        if self.closed:
            raise RuntimeError("Tracking frontend is closed")
        begin = self.clock()
        if not math.isfinite(captured_monotonic) or captured_monotonic > begin:
            raise ValueError("Frame has an invalid or future capture time")
        if self.first_capture is None:
            self.first_capture = captured_monotonic
        flow = self.tracker.process(frame, frame_id, captured_monotonic)
        self.accept_publications()
        estimate = None
        candidate_pose = None
        if self.ready_candidate is not None:
            candidate = self.ready_candidate
            candidate_pose = self.tracker.estimate(candidate)
            self.ready_candidate = None
            if candidate_pose["status"] == "tracking":
                old = self.active
                self.active, estimate = candidate, candidate_pose
                if old is not None and old != candidate:
                    self._drop(old)
                self.counters["anchor_activations"] += 1
            else:
                self.counters["rejected_depth_anchors"] += 1
                self.rejected_anchor_reasons[candidate_pose.get("reason") or candidate_pose["status"]] += 1
                self._drop(candidate)
        if estimate is None and self.active is not None:
            estimate = self.tracker.estimate(self.active)
        if estimate is None:
            estimate = dict(status="waiting_depth", reason="no_valid_depth_anchor",
                            frame_id=frame_id, anchor_id=None,
                            T_current_from_anchor=None, matched_points=0, inliers=0)
        if estimate["frame_id"] != frame_id:
            raise RuntimeError("Tracker returned a pose for a different frame")
        measurement = flow["anchors"].get(self.selection)
        selected = (measurement is None or measurement["tracked_points"] < 20 or
                    measurement["mean_disparity_original_pixels"] is None or
                    measurement["mean_disparity_original_pixels"] > self.min_disparity)
        if selected:
            if self.staged is not None:
                self._drop(self.staged)
            self.staged = self.tracker.add_anchor(frame_id)
            self.selection = self.staged
            self.counters["selected_keyframes"] += 1
        record = dict(frame_id=frame_id, captured_monotonic=captured_monotonic,
                      frontend="experimental_sparse_depth_pnp", keyframe_selected=selected,
                      estimate=estimate, global_geometry=None,
                      new_depth_anchor_estimate=candidate_pose,
                      latest_graph_revision=self.latest_graph_revision,
                      fixed_anchor_intrinsics_assumed=True, distortion_model="none",
                      flow_processing_seconds=flow["processing_seconds"])
        if estimate["status"] == "tracking":
            from vggt_slam.tracking_geometry import associate_pose
            geometry = self.geometry[self.active]
            record.update(anchor_id=self.active, graph_node_id=geometry["graph_node_id"],
                          graph_revision=geometry["graph_revision"],
                          backend_ready_monotonic=geometry["backend_ready_monotonic"])
            if geometry["graph_revision"] == self.latest_graph_revision:
                record["global_geometry"] = associate_pose(
                    estimate["T_current_from_anchor"], estimate["K_tracking"], geometry["homography"])
                self.counters["global_projection_frames"] += 1
                if record["global_geometry"]["rigid_global_pose"] is not None:
                    self.counters["rigid_global_pose_frames"] += 1
            else:
                record["global_geometry_reason"] = "anchor_graph_revision_stale"
        elif estimate.get("T_current_from_anchor") is not None:
            raise RuntimeError("Unsuccessful tracking estimate contains a pose")
        emitted = self.clock()
        record.update(emitted_monotonic=emitted, processing_seconds=emitted-begin,
                      frame_age_seconds=emitted-captured_monotonic)
        self.processing_times.append(emitted-begin)
        self.counters["delivered_frames"] += 1
        self.reasons[estimate.get("reason") or estimate["status"]] += 1
        if estimate["status"] == "tracking":
            self.counters["valid_relative_pose_frames"] += 1
            age = emitted-captured_monotonic
            self.pose_ages.append(age)
            self.pose_age_sum += age
            self.pose_age_max = max(self.pose_age_max, age)
            if self.first_pose_emit is None:
                self.first_pose_emit = emitted
            self.last_pose_emit = emitted
        self.stream.write(json.dumps(json_value(record), allow_nan=False)+"\n")
        return selected

    def metrics(self):
        count = self.counters["valid_relative_pose_frames"]
        delivered = self.counters["delivered_frames"]
        return dict(self.counters, frontend="experimental_sparse_depth_pnp",
                    valid_relative_pose_frames=count, delivered_frames=delivered,
                    valid_relative_pose_fraction_delivered=count/delivered if delivered else 0,
                    first_pose_wait_seconds=(self.first_pose_emit-self.first_capture
                                             if self.first_pose_emit is not None else None),
                    valid_pose_emission_hz=((count-1)/(self.last_pose_emit-self.first_pose_emit)
                        if count > 1 and self.last_pose_emit > self.first_pose_emit else None),
                    mean_valid_pose_age_seconds=self.pose_age_sum/count if count else None,
                    max_valid_pose_age_seconds=self.pose_age_max if count else None,
                    recent_pose_age_p95_seconds=float(np.percentile(self.pose_ages, 95)) if self.pose_ages else None,
                    recent_processing_p95_seconds=float(np.percentile(self.processing_times, 95))
                        if self.processing_times else None,
                    quantile_window_limit=4096, quantile_pose_samples=len(self.pose_ages),
                    statuses=dict(self.reasons), feature_anchor_limit=3,
                    rejected_depth_anchor_reasons=dict(self.rejected_anchor_reasons),
                    latest_graph_revision=self.latest_graph_revision,
                    dense_map_memory_bounded=False, real_time_validated=False,
                    accuracy_validated=False, closed=self.closed)

    def close(self):
        if self.closed:
            return
        # EOF publications update the map, but cannot retroactively count as
        # online tracking poses. Preserve their count for the audit.
        while True:
            try:
                self.publications.get_nowait()
                self.counters["publications_after_last_frame"] += 1
            except queue.Empty:
                break
        for anchor in tuple(self.tracker.anchor_ids):
            self._drop(anchor)
        self.stream.close()
        self.closed = True
