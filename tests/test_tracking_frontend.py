"""Causal frontend scheduling tests, without model weights or camera access."""

import json
import queue

import numpy as np
import pytest

from vggt_slam.tracking_frontend import OnlineTrackingFrontend


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeTracker:
    """Track source markers and enforce the actual three-anchor contract."""

    def __init__(self, clock):
        self.clock = clock
        self.anchors = {}
        self.events = []
        self.outcomes = {}
        self.disparity = 100.
        self.tracked_points = 40
        self.max_retained_anchors = 0
        self.frame_id = self.captured = self.marker = None

    @property
    def anchor_ids(self):
        return tuple(self.anchors)

    def process(self, frame, frame_id, captured_monotonic):
        assert self.frame_id is None or frame_id > self.frame_id
        assert self.captured is None or captured_monotonic > self.captured
        self.frame_id, self.captured = frame_id, captured_monotonic
        self.marker = int(frame[0, 0])
        self.events.append(("process", frame_id))
        self.clock.advance(.01)
        return dict(frame_id=frame_id, processing_seconds=.01, anchors={
            key: dict(tracked_points=self.tracked_points,
                      mean_disparity_original_pixels=self.disparity)
            for key in self.anchors})

    def add_anchor(self, anchor_id=None):
        key = self.frame_id if anchor_id is None else anchor_id
        assert key not in self.anchors
        assert len(self.anchors) < 3, "Feature anchor budget exceeded"
        self.anchors[key] = dict(frame_id=self.frame_id, original_marker=self.marker,
                                 depth=None, intrinsic=None)
        self.max_retained_anchors = max(self.max_retained_anchors, len(self.anchors))
        self.events.append(("add", key))
        return key

    def drop_anchor(self, anchor_id):
        del self.anchors[anchor_id]
        self.events.append(("drop", anchor_id))

    def attach_depth(self, anchor_id, depth, confidence, intrinsic, image_size):
        self.events.append(("attach", anchor_id, self.frame_id))
        anchor = self.anchors[anchor_id]
        anchor.update(depth=np.array(depth, copy=True), confidence=np.array(confidence, copy=True),
                      intrinsic=np.array(intrinsic, copy=True), image_size=image_size)

    def estimate(self, anchor_id):
        anchor = self.anchors[anchor_id]
        self.events.append(("estimate", anchor_id, self.frame_id))
        if anchor["depth"] is None:
            status, reason = "waiting_depth", "anchor_depth_not_attached"
        elif self.frame_id == anchor["frame_id"]:
            status, reason = "lost", "no_later_frame"
        else:
            status, reason = "tracking", None
        result = dict(status=status, reason=reason, frame_id=self.frame_id,
                      captured_monotonic=self.captured, anchor_id=anchor_id,
                      anchor_frame_id=anchor["frame_id"],
                      T_current_from_anchor=np.eye(4) if status == "tracking" else None,
                      K_tracking=None if anchor["intrinsic"] is None else anchor["intrinsic"].copy(),
                      matched_points=40 if anchor["depth"] is not None else 0,
                      inliers=40 if status == "tracking" else 0, pnp_seconds=.005)
        result.update(self.outcomes.get((anchor_id, self.frame_id), {}))
        self.clock.advance(.005)
        return result


@pytest.fixture
def frontend(tmp_path):
    clock = Clock()
    tracker = FakeTracker(clock)
    output = tmp_path / "online-poses.jsonl"
    instance = OnlineTrackingFrontend(output, tracker=tracker, clock=clock)
    yield instance, tracker, clock, output
    instance.close()


def observe(context, frame_id, marker=None):
    instance, _, clock, _ = context
    clock.now = frame_id + .02
    image = np.full((4, 5), frame_id if marker is None else marker, np.uint8)
    return instance.observe(image, frame_id, float(frame_id))


def publish(context, *, anchor_id=None, revision=None, node_updates=None, **changes):
    instance, _, clock, _ = context
    anchor_id = instance.inflight if anchor_id is None else anchor_id
    values = dict(anchor_id=anchor_id, depth=np.ones((1, 1)), confidence=np.ones((1, 1)),
                  intrinsic=np.eye(3), image_size=112, graph_node_id=100 + anchor_id,
                  homography=np.eye(4),
                  graph_revision=instance.latest_graph_revision + 1 if revision is None else revision,
                  node_updates={} if node_updates is None else node_updates,
                  backend_ready_monotonic=clock.now)
    values.update(changes)
    instance.publish(**values)


def records(context):
    return [json.loads(line) for line in context[3].read_text().splitlines()]


def activate_first_anchor(context):
    instance = context[0]
    observe(context, 1)
    instance.submitted(1)
    publish(context)
    observe(context, 2)
    assert instance.active == 1 and instance.staged == 2


def test_three_anchor_budget_including_ready_inflight_and_staged(frontend):
    instance, tracker, _, _ = frontend
    observe(frontend, 1)
    instance.submitted(1)
    observe(frontend, 2)
    observe(frontend, 3)
    assert tracker.anchor_ids == (1, 3)  # Latest staged keyframe replaces frame 2.
    publish(frontend)
    observe(frontend, 4)
    instance.submitted(4)
    observe(frontend, 5)
    assert set(tracker.anchor_ids) == {1, 4, 5}
    publish(frontend, node_updates={1: np.eye(4)})
    instance.submitted(5)  # Consume frame 4 without another camera delivery.
    assert (instance.active, instance.ready_candidate, instance.inflight) == (1, 4, 5)
    assert len(tracker.anchor_ids) == 3
    observe(frontend, 6)
    assert (instance.active, instance.inflight, instance.staged) == (4, 5, 6)
    assert set(tracker.anchor_ids) == {4, 5, 6}
    publish(frontend, node_updates={4: np.eye(4)})
    observe(frontend, 7)
    assert tracker.max_retained_anchors == 3
    assert set(tracker.anchor_ids) == {5, 7}
    assert instance.metrics()["anchor_activations"] == 3
    assert instance.metrics()["submitted_anchor_jobs"] == 3


def test_delayed_depth_uses_anchor_snapshot_after_current_frame_tracking(frontend):
    instance, tracker, clock, _ = frontend
    observe(frontend, 1, marker=10)
    instance.submitted(1)
    observe(frontend, 2, marker=20)
    depth, confidence, intrinsic, homography = (np.full((1, 1), 10.), np.ones((1, 1)),
                                               np.eye(3), np.eye(4))
    ready_time = clock.now
    publish(frontend, depth=depth, confidence=confidence, intrinsic=intrinsic, homography=homography)
    depth[:] = 30.
    confidence[:] = 0.
    intrinsic[:] = 0.
    homography[0, 3] = 99.
    assert not any(event[0] == "attach" for event in tracker.events)
    observe(frontend, 3, marker=30)
    assert tracker.events.index(("process", 3)) < tracker.events.index(("attach", 1, 3))
    assert tracker.events.index(("attach", 1, 3)) < tracker.events.index(("estimate", 1, 3))
    assert tracker.anchors[1]["original_marker"] == 10 and tracker.marker == 30
    assert tracker.anchors[1]["depth"][0, 0] == 10.
    assert tracker.anchors[1]["confidence"][0, 0] == 1.
    np.testing.assert_array_equal(tracker.anchors[1]["intrinsic"], np.eye(3))
    row = records(frontend)[-1]
    assert row["frame_id"] == 3 and row["estimate"]["anchor_frame_id"] == 1
    assert row["new_depth_anchor_estimate"]["status"] == "tracking"
    assert row["new_depth_anchor_estimate"]["frame_id"] == 3
    assert row["new_depth_anchor_estimate"]["anchor_id"] == 1
    assert row["emitted_monotonic"] >= row["backend_ready_monotonic"] == ready_time
    assert row["frame_age_seconds"] == pytest.approx(row["emitted_monotonic"] - 3.)
    np.testing.assert_array_equal(row["global_geometry"]["rigid_global_pose"], np.eye(4))
    metrics = instance.metrics()
    assert metrics["delivered_frames"] == 3 and metrics["valid_relative_pose_frames"] == 1
    assert metrics["first_pose_wait_seconds"] == pytest.approx(row["emitted_monotonic"] - 1.)
    assert metrics["real_time_validated"] is False and metrics["accuracy_validated"] is False


def test_failed_candidate_preserves_current_active_pose_then_lost_does_not_reuse_it(frontend):
    instance, tracker, _, _ = frontend
    activate_first_anchor(frontend)
    instance.submitted(2)
    updated = np.eye(4)
    updated[0, 3] = 4.
    publish(frontend, node_updates={1: updated})
    updated[0, 3] = 999.  # The backend's array is copied at publication.
    tracker.outcomes[(2, 3)] = dict(status="lost", reason="insufficient_tracks", T_current_from_anchor=None)
    observe(frontend, 3)
    row = records(frontend)[-1]
    assert ("estimate", 2, 3) in tracker.events and ("estimate", 1, 3) in tracker.events
    assert instance.active == 1 and 2 not in tracker.anchor_ids
    assert row["estimate"]["anchor_id"] == 1 and row["estimate"]["frame_id"] == 3
    rejected = row["new_depth_anchor_estimate"]
    assert rejected["anchor_id"] == 2 and rejected["frame_id"] == 3
    assert rejected["status"] == "lost" and rejected["reason"] == "insufficient_tracks"
    assert rejected["T_current_from_anchor"] is None
    assert row["graph_revision"] == row["latest_graph_revision"] == 2
    assert row["global_geometry"]["rigid_global_pose"][0][3] == 4.
    assert instance.requested_nodes() == {1: 101}
    tracker.outcomes[(1, 4)] = dict(status="lost", reason="insufficient_tracks", T_current_from_anchor=None)
    observe(frontend, 4)
    lost = records(frontend)[-1]
    assert lost["estimate"]["status"] == "lost" and lost["estimate"]["T_current_from_anchor"] is None
    assert lost["global_geometry"] is None
    assert lost["new_depth_anchor_estimate"] is None
    metrics = instance.metrics()
    assert metrics["rejected_depth_anchors"] == 1 and metrics["valid_relative_pose_frames"] == 2
    assert metrics["rejected_depth_anchor_reasons"] == {"insufficient_tracks": 1}
    assert metrics["delivered_frames"] == 4


def test_rejected_anchor_reason_survives_fallback_waiting_depth_without_stale_candidate(frontend):
    instance, tracker, _, _ = frontend
    observe(frontend, 1)
    instance.submitted(1)
    publish(frontend)
    tracker.outcomes[(1, 2)] = dict(status="lost", reason="insufficient_pnp_inliers",
                                  T_current_from_anchor=None, inliers=10)
    observe(frontend, 2)
    row = records(frontend)[-1]
    assert instance.active is None
    assert row["estimate"]["status"] == "waiting_depth"
    assert row["estimate"]["reason"] == "no_valid_depth_anchor"
    assert row["estimate"]["T_current_from_anchor"] is None
    rejected = row["new_depth_anchor_estimate"]
    assert rejected["anchor_id"] == 1 and rejected["frame_id"] == 2
    assert rejected["status"] == "lost" and rejected["reason"] == "insufficient_pnp_inliers"
    assert rejected["T_current_from_anchor"] is None and rejected["inliers"] == 10
    assert row["global_geometry"] is None
    metrics = instance.metrics()
    assert metrics["valid_relative_pose_frames"] == 0 and metrics["rejected_depth_anchors"] == 1
    assert metrics["rejected_depth_anchor_reasons"] == {"insufficient_pnp_inliers": 1}
    observe(frontend, 3)
    assert records(frontend)[-1]["new_depth_anchor_estimate"] is None
    assert instance.metrics()["rejected_depth_anchor_reasons"] == {"insufficient_pnp_inliers": 1}


def test_missing_active_revision_update_keeps_local_pose_but_suppresses_global_geometry(frontend):
    instance, tracker, _, _ = frontend
    activate_first_anchor(frontend)
    instance.submitted(2)
    publish(frontend, node_updates={})
    tracker.outcomes[(2, 3)] = dict(status="lost", reason="pnp_failed", T_current_from_anchor=None)
    observe(frontend, 3)
    row = records(frontend)[-1]
    assert row["estimate"]["status"] == "tracking"
    assert row["latest_graph_revision"] == 2 and row["graph_revision"] == 1
    assert row["global_geometry"] is None
    assert row["global_geometry_reason"] == "anchor_graph_revision_stale"
    assert instance.metrics()["valid_relative_pose_frames"] == 2
    assert instance.metrics()["global_projection_frames"] == 1


@pytest.mark.parametrize("bad_kind", ["stale_pose", "wrong_frame"])
def test_invalid_active_estimate_fails_without_writing_or_counting_an_online_pose(frontend, bad_kind):
    instance, tracker, _, _ = frontend
    activate_first_anchor(frontend)
    tracker.outcomes[(1, 3)] = (dict(status="lost", reason="lost", T_current_from_anchor=np.eye(4))
                               if bad_kind == "stale_pose" else dict(frame_id=2))
    expected = "contains a pose" if bad_kind == "stale_pose" else "different frame"
    with pytest.raises(RuntimeError, match=expected):
        observe(frontend, 3)
    assert len(records(frontend)) == 2
    assert instance.metrics()["delivered_frames"] == 2
    assert instance.metrics()["valid_relative_pose_frames"] == 1


@pytest.mark.parametrize("invalid_publication", ["unknown_anchor", "future_time", "same_revision"])
def test_bad_backend_publication_cannot_create_a_tracking_pose(frontend, invalid_publication):
    instance = frontend[0]
    activate_first_anchor(frontend)
    instance.submitted(2)
    changes = {"anchor_id": 99} if invalid_publication == "unknown_anchor" else (
        {"backend_ready_monotonic": 100.} if invalid_publication == "future_time" else {"revision": 1})
    publish(frontend, **changes)
    with pytest.raises(RuntimeError):
        observe(frontend, 3)
    assert len(records(frontend)) == 2
    assert instance.metrics()["depth_publications_consumed"] == 1
    assert instance.metrics()["valid_relative_pose_frames"] == 1


def test_publication_queue_is_bounded_and_does_not_overwrite_depth(frontend):
    instance = frontend[0]
    observe(frontend, 1)
    instance.submitted(1)
    publish(frontend, depth=np.full((1, 1), 7.))
    with pytest.raises(queue.Full):
        publish(frontend, depth=np.full((1, 1), 99.))
    observe(frontend, 2)
    assert frontend[1].anchors[1]["depth"][0, 0] == 7.
    assert instance.metrics()["backend_publications"] == 1


@pytest.mark.parametrize("consume_before_tail", [False, True])
def test_eof_publications_never_emit_retrospective_online_poses(frontend, consume_before_tail):
    instance, tracker, _, _ = frontend
    observe(frontend, 1)
    instance.submitted(1)
    observe(frontend, 2)
    publish(frontend)
    if consume_before_tail:
        instance.submitted(2)  # Consume late first depth before submitting EOF tail.
        assert instance.ready_candidate == 1 and instance.active is None
        assert instance.metrics()["depth_publications_consumed"] == 1
        publish(frontend)
    instance.close()
    metrics = instance.metrics()
    assert metrics["publications_after_last_frame"] == 1
    assert metrics["backend_publications"] == (2 if consume_before_tail else 1)
    assert metrics.get("depth_publications_consumed", 0) == int(consume_before_tail)
    assert metrics["delivered_frames"] == 2 and metrics["valid_relative_pose_frames"] == 0
    assert metrics["first_pose_wait_seconds"] is None and metrics["mean_valid_pose_age_seconds"] is None
    assert len(records(frontend)) == 2 and all(row["global_geometry"] is None for row in records(frontend))
    assert all(row["new_depth_anchor_estimate"] is None for row in records(frontend))
    assert not tracker.anchor_ids and instance.requested_nodes() == {} and metrics["closed"] is True
    instance.close()  # Idempotent cleanup must not create another publication count.
    assert instance.metrics()["publications_after_last_frame"] == 1
    with pytest.raises(RuntimeError, match="closed"):
        observe(frontend, 3)


@pytest.mark.parametrize("operation", ["submitted", "publish"])
def test_closed_frontend_rejects_backend_calls_without_queue_or_counter_changes(frontend, operation):
    instance, tracker, _, _ = frontend
    observe(frontend, 1)
    instance.submitted(1)
    instance.close()
    before = instance.metrics()
    with pytest.raises(RuntimeError, match="closed"):
        if operation == "submitted":
            instance.submitted(1)
        else:
            publish(frontend, anchor_id=1)
    assert instance.metrics() == before
    assert instance.publications.empty() and not tracker.anchor_ids
    assert len(records(frontend)) == 1


def test_eof_after_online_tracking_preserves_only_previously_emitted_pose_counts(frontend):
    instance = frontend[0]
    activate_first_anchor(frontend)
    before = instance.metrics()
    instance.submitted(2)
    publish(frontend)
    instance.close()
    after = instance.metrics()
    for name in ("delivered_frames", "valid_relative_pose_frames", "first_pose_wait_seconds",
                 "mean_valid_pose_age_seconds", "global_projection_frames"):
        assert after[name] == before[name]
    assert after["publications_after_last_frame"] == 1 and len(records(frontend)) == 2


def test_backend_submission_must_match_staged_anchor(frontend):
    instance = frontend[0]
    observe(frontend, 1)
    with pytest.raises(RuntimeError, match="submitted VGGT batch"):
        instance.submitted(99)
    instance.submitted(1)
    observe(frontend, 2)
    with pytest.raises(RuntimeError, match="submitted VGGT batch"):
        instance.submitted(2)
    assert instance.inflight == 1 and instance.staged == 2


def test_keyframe_measurement_belongs_to_latest_selected_anchor(frontend):
    instance, tracker, _, _ = frontend
    observe(frontend, 1)
    tracker.disparity = instance.min_disparity
    assert observe(frontend, 2) is False
    assert instance.staged == instance.selection == 1
    tracker.disparity += .1
    assert observe(frontend, 3) is True
    assert instance.staged == instance.selection == 3 and tracker.anchor_ids == (3,)
    tracker.disparity = 0.
    tracker.tracked_points = 19
    assert observe(frontend, 4) is True
    assert instance.staged == instance.selection == 4 and tracker.anchor_ids == (4,)
