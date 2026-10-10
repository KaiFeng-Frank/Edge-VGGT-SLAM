"""Synthetic source/queue/clock checks; no dataset images or neural inference."""

import importlib.util
import io
import json
from pathlib import Path
import queue
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from vggt_slam.timestamp_replay import TimestampReplay, read_rgb_index


@pytest.fixture
def edge():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_edge.py"
    spec = importlib.util.spec_from_file_location("edge_runtime_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rgb_index_keeps_original_timestamps_and_resolved_identity(tmp_path):
    (tmp_path / "rgb").mkdir()
    paths = [tmp_path / "rgb" / "first image.png", tmp_path / "rgb" / "second.png"]
    for path in paths:
        path.write_bytes(b"synthetic placeholder, not an image")
    index = tmp_path / "rgb.txt"
    index.write_text("# sensor timestamps\n1305031102.175304 rgb/first image.png\n"
                     "1305031102.211214 rgb/second.png\n")
    rows = read_rgb_index(index)
    assert [r["timestamp"] for r in rows] == [1305031102.175304, 1305031102.211214]
    assert [r["path"] for r in rows] == [str(p.resolve()) for p in paths]


@pytest.mark.parametrize("text", ["nan a\n2 b\n", "1 a\ninf b\n", "1 a\n1 b\n",
                                 "2 a\n1 b\n", "1 a\n", "1 a\n2 missing\n"])
def test_rgb_index_rejects_invalid_source_without_decoding(tmp_path, text):
    for name in ("a", "b"):
        (tmp_path / name).write_bytes(b"not decoded")
    index = tmp_path / "rgb.txt"
    index.write_text(text)
    with pytest.raises(ValueError):
        read_rgb_index(index)


@pytest.mark.parametrize("symlink", [False, True])
def test_rgb_index_rejects_paths_outside_dataset(tmp_path, symlink):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "a").write_bytes(b"placeholder")
    outside = tmp_path / "outside"
    outside.write_bytes(b"placeholder")
    if symlink:
        (dataset / "b").symlink_to(outside)
    index = dataset / "rgb.txt"
    index.write_text(f"1 a\n2 {'b' if symlink else '../outside'}\n")
    with pytest.raises(ValueError, match="outside dataset"):
        read_rgb_index(index)


def test_timestamp_replay_single_slot_overwrites_and_retains_last_eof_frame():
    stamps = [1305031102.175304 + i * .00001 for i in range(4)]
    rows = [dict(timestamp=stamp, path=str(i)) for i, stamp in enumerate(stamps)]
    source = TimestampReplay(rows, decoder=lambda path: np.full((2, 3, 3), int(path), np.uint8))
    source.start()
    source._thread.join(timeout=2)
    assert not source._thread.is_alive() and source.done and not source.exhausted
    assert source.generated == 4 and source.overwritten == 3 and source.delivered == 0
    last = source.read(timeout=0)
    assert last["sequence"] == 3 and last["frame_id"] == last["timestamp"] == stamps[-1]
    np.testing.assert_array_equal(last["frame"], np.full((2, 3, 3), 3, np.uint8))
    assert last["captured_monotonic"] == pytest.approx(source.start_monotonic + stamps[-1] - stamps[0])
    assert last["captured_monotonic"] <= last["ready_monotonic"] <= last["delivered_monotonic"]
    assert source.exhausted and source.read(timeout=0) is None
    metrics = source.metrics()
    assert metrics["generated"] == metrics["delivered"] + metrics["overwritten"]
    assert metrics["queue_capacity"] == 1 and not metrics["physical_sensor_latency_validated"]
    source.close()


def test_timestamp_latency_starts_before_decode_and_preserves_irregular_cadence():
    entered = [threading.Event(), threading.Event()]
    proceed = [threading.Event(), threading.Event()]

    def decoder(path):
        index = int(path)
        entered[index].set()
        assert proceed[index].wait(timeout=2)
        return np.full((2, 2, 3), index, np.uint8)

    source = TimestampReplay([dict(timestamp=1000., path="0"), dict(timestamp=1000.017, path="1")],
                             decoder=decoder)
    source.start()
    try:
        assert entered[0].wait(timeout=1)
        assert source.read(timeout=0) is None  # Arrival alone is not a decoded observation.
        before_decode_completed = time.monotonic()
        proceed[0].set()
        first = source.read(timeout=1)
        assert first["captured_monotonic"] <= first["decode_started_monotonic"]
        assert first["decode_started_monotonic"] <= before_decode_completed <= first["ready_monotonic"]
        assert entered[1].wait(timeout=1)
        proceed[1].set()
        second = source.read(timeout=1)
        assert second["captured_monotonic"] - first["captured_monotonic"] == pytest.approx(.017, abs=1e-9)
        assert second["ready_monotonic"] <= second["delivered_monotonic"]
        assert [first["frame_id"], second["frame_id"]] == [1000., 1000.017]
    finally:
        for event in proceed:
            event.set()
        source.close()


def test_timestamp_source_error_is_propagated_and_reader_stops():
    source = TimestampReplay([dict(timestamp=1., path="bad"), dict(timestamp=1.01, path="unused")],
                             decoder=lambda unused: None)
    source.start()
    source._thread.join(timeout=1)
    with pytest.raises(RuntimeError, match="Cannot decode RGB image bad"):
        source.read(timeout=0)
    source.close()
    assert source.done and not source._thread.is_alive() and source.generated == 0
    assert source.metrics()["error"]


def test_timestamp_close_interrupts_long_due_wait_and_keeps_existing_slot():
    source = TimestampReplay([dict(timestamp=1., path="a"), dict(timestamp=60., path="b")],
                             decoder=lambda unused: np.zeros((2, 2, 3), np.uint8))
    source.start()
    with source._condition:
        assert source._condition.wait_for(lambda: source.generated == 1, timeout=1)
    source.close()
    assert source.generated == 1 and source.done and not source._thread.is_alive()
    assert source.read(timeout=0)["frame_id"] == 1.
    assert source.exhausted


def test_timing_summary_includes_startup_and_losses_and_decode_queue_age(edge):
    source = dict(start_monotonic=100., last_capture_monotonic=103.4, original_frames=100)
    rows = [dict(valid=False, captured=100.1, published=100.2),
            dict(valid=True, captured=101.1, published=101.18),
            dict(valid=False, captured=101.2, published=101.25),
            dict(valid=True, captured=102.0, published=102.12)]
    summary = edge.summarize_pose_timing(rows, source)
    assert summary["valid_pose_publications_per_full_second"] == [0, 1, 1]
    assert summary["minimum_full_second_pose_count"] == 0
    assert summary["fresh_valid_poses"] == 2 and summary["delivered_observations"] == 4
    assert summary["valid_fraction_of_original_frames"] == .02
    assert summary["valid_fraction_of_delivered_frames"] == .5
    assert summary["valid_pose_latency_p95_ms"] == pytest.approx(118.)
    assert summary["maximum_valid_pose_publication_gap_seconds"] == pytest.approx(.94)
    assert summary["latency_origin"] == "scheduled_RGB_replay_arrival_before_decode"
    assert summary["startup_and_losses_included_in_per_second_counts"]


def test_json_publication_clock_follows_flush_and_frame_token_is_exact(edge, monkeypatch):
    clock = SimpleNamespace(value=10.)

    class Stream(io.StringIO):
        def write(self, text):
            clock.value += .02
            return super().write(text)

        def flush(self):
            clock.value += .03

    monkeypatch.setattr(edge, "time", SimpleNamespace(monotonic=lambda: clock.value))
    stream = Stream()
    assert edge.write_row(stream, dict(value=np.array([1., 2.]))) == pytest.approx(10.05)
    assert json.loads(stream.getvalue()) == {"value": [1., 2.]}
    item = dict(path="original/rgb/1305031102.175304.png", frame_id=1305031102.175304,
                timestamp=1305031102.175304, captured_monotonic=12.123456789123,
                sequence=7, frame=np.zeros((1, 1)), ready_monotonic=13.)
    token = edge.frame_token(item)
    assert token["frame_id"] == item["frame_id"] and token["captured_monotonic"] == item["captured_monotonic"]
    assert token["sequence"] == 7 and "frame" not in token


class RunnerService:
    """Finite fake source/model transport around the real main coordinator."""

    def __init__(self, edge, monkeypatch, tmp_path, mode="normal"):
        import vggt_slam.continuous_tracking as tracking
        import vggt_slam.edge_mapping as mapping
        import vggt_slam.timestamp_replay as replay

        self.mode = mode
        self.clock = 100.
        self.rows = [dict(timestamp=1000. + i * .2, path=f"original-{i}.png") for i in range(8)]
        self.results = queue.Queue(1)
        self.batch_jobs, self.snapshots, self.applies, self.executors = [], [], [], []
        self.background_advances = []
        self.joins = 0
        self.source_closed = False
        service = self

        class Worker:
            exitcode = None
            alive = True

            def start(self):
                self.alive = mode != "startup_failure"
                if self.alive:
                    service.results.put(dict(kind="ready", load_seconds=.1))
                else:
                    self.exitcode = 2
                    service.results.put(dict(kind="error", traceback="synthetic mapper startup failure"))

            def is_alive(self):
                return self.alive

            def join(self, timeout):
                service.joins += 1

            def terminate(self):
                self.alive, self.exitcode = False, -15

        self.worker = Worker()

        class Jobs:
            def put_nowait(self, job):
                if job is None:
                    service.worker.alive = False
                    service.worker.exitcode = 9 if mode == "exit_failure" else 0
                    service.results.put_nowait(dict(kind="stopped"))
                    return
                service.batch_jobs.append(job)
                if mode == "mapper_failure":
                    service.results.put_nowait(dict(kind="error", traceback="synthetic inference failure"))
                    return
                revision = len(service.batch_jobs)
                anchor = job["batch"][-1]
                H = np.eye(4)
                H[0, 3] = revision
                service.results.put_nowait(dict(kind="map", revision=revision, anchor=anchor,
                    graph_node_id=revision * 2 - 1, homography=H.tolist(),
                    node_updates={str(node): H.tolist() for node in job["requested_nodes"]},
                    depth=np.ones((2, 2)), confidence=np.ones((2, 2)), intrinsic=np.eye(3),
                    model_image_size=14, published_monotonic=service.now(),
                    processing_seconds=.03, newest_frame_age_seconds=.05))

            def close(self):
                pass

        self.jobs = Jobs()
        results = self.results
        results.close = lambda: None

        class Context:
            def __init__(self):
                self.queues = 0

            def Queue(self, capacity):
                assert capacity == 1
                self.queues += 1
                return service.jobs if self.queues == 1 else results

            def Process(self, **kwargs):
                return service.worker

        class Source:
            def __init__(self, rows, cpus):
                self.rows, self.index, self.done = rows, 0, False

            def start(self):
                if mode == "early_stopped":
                    service.worker.alive, service.worker.exitcode = False, 0
                    service.results.put_nowait(dict(kind="stopped"))

            def read(self):
                if mode == "source_failure" and self.index == 2:
                    raise RuntimeError("synthetic source failure")
                if self.done:
                    return None
                sequence = self.index
                row = self.rows[sequence]
                self.index += 1
                self.done = self.index == len(self.rows)
                service.clock = max(service.clock, 100. + sequence * .2)
                return dict(row, frame_id=row["timestamp"], sequence=sequence,
                            captured_monotonic=100. + sequence * .2,
                            decode_started_monotonic=service.now(), ready_monotonic=service.now(),
                            delivered_monotonic=service.now(), frame=np.full((2, 2, 3), sequence, np.uint8))

            @property
            def exhausted(self):
                return self.done

            def close(self):
                service.source_closed = True
                if mode == "cleanup_failure":
                    raise RuntimeError("synthetic source close failure")

            def metrics(self):
                return dict(original_frames=len(self.rows), delivered=self.index, generated=self.index,
                            overwritten=0, start_monotonic=100.,
                            last_capture_monotonic=100. + (self.index - 1) * .2 if self.index else None,
                            end_monotonic=service.now())

        class Tracker:
            def __init__(self, **kwargs):
                self.observations = []
                self.active = False

            @property
            def state(self):
                return dict(sequence=len(self.observations) - 1)

            def process(self, frame, frame_id, captured):
                self.observations.append((frame_id, captured))
                return dict(frame_id=frame_id, captured_monotonic=captured, pose_valid=self.active,
                            T_current_from_anchor=np.eye(4) if self.active else None,
                            K_tracking=np.eye(3), H_model_from_local=np.eye(4),
                            model_association_available=self.active)

            def snapshot_depth_history(self, frame_id, captured):
                assert (frame_id, captured) in self.observations
                snapshot = dict(anchor=(frame_id, captured), observations=list(self.observations))
                service.snapshots.append(snapshot)
                return snapshot

            def snapshot_candidate_catchup(self, candidate):
                assert candidate["last_sequence"] <= self.state["sequence"]
                return dict(available=True, reason=None, anchor=candidate["snapshot"]["anchor"],
                            last_sequence=candidate["last_sequence"], target_sequence=self.state["sequence"],
                            observations=list(self.observations))

            def apply_depth_candidate(self, candidate):
                catchup = self.state["sequence"] - candidate["last_sequence"] if candidate["available"] else 0
                service.applies.append(catchup)
                adopted = candidate["available"]
                if adopted:
                    self.active = True
                return dict(adopted=adopted, catchup_observations=catchup,
                            reason=candidate["reason"], historical_replay=True, emits_fresh_pose=False)

            def metrics(self):
                return dict(observations=len(self.observations))

        class Future:
            def __init__(self, fn, args, kwargs):
                self.polls = 0
                self.threshold = (5 if fn is build and mode in ("background_advance", "background_budget",
                                                               "advance_failure")
                                  else 1 if fn is advance_candidate else 3)
                try:
                    self.value, self.error = fn(*args, **kwargs), None
                except Exception as exc:
                    self.value, self.error = None, exc

            def done(self):
                self.polls += 1
                return self.polls >= self.threshold

            def result(self):
                if self.error:
                    raise self.error
                return self.value

        class Executor:
            def __init__(self, max_workers, **kwargs):
                assert max_workers == 1
                self.futures, self.closed = [], False
                service.executors.append(self)

            def submit(self, fn, *args, **kwargs):
                assert not self.futures or self.futures[-1].polls >= self.futures[-1].threshold
                future = Future(fn, args, kwargs)
                self.futures.append(future)
                return future

            def shutdown(self, wait, cancel_futures):
                assert wait and cancel_futures
                self.closed = True

        def build(snapshot, **kwargs):
            if mode == "candidate_failure":
                raise RuntimeError("synthetic depth worker failure")
            return dict(snapshot=snapshot, last_sequence=len(snapshot["observations"]) - 1,
                        available=mode != "candidate_rejected",
                        reason="synthetic quality rejection" if mode == "candidate_rejected" else None)

        def advance_candidate(candidate, snapshot):
            if mode == "advance_failure":
                raise RuntimeError("synthetic advance worker failure")
            assert snapshot["anchor"] == candidate["snapshot"]["anchor"]
            service.background_advances.append(dict(base=candidate["last_sequence"],
                                                    target=snapshot["target_sequence"]))
            return dict(candidate, snapshot=snapshot, last_sequence=snapshot["target_sequence"])

        monkeypatch.setattr(edge, "time", SimpleNamespace(monotonic=self.now, sleep=lambda d: self.advance(d)))
        monkeypatch.setattr(edge.mp, "get_context", lambda method: Context())
        monkeypatch.setattr(edge.os, "sched_setaffinity", lambda pid, cpus: None)
        monkeypatch.setattr(edge, "ThreadPoolExecutor", Executor)
        monkeypatch.setattr(replay, "TimestampReplay", Source)
        monkeypatch.setattr(replay, "read_rgb_index", lambda path: list(self.rows))
        monkeypatch.setattr(mapping, "platform_sample", lambda: {"peak_rss_mib": 1.})
        monkeypatch.setattr(tracking, "ContinuousDepthTracker", Tracker)
        monkeypatch.setattr(tracking, "build_depth_candidate", build)
        monkeypatch.setattr(tracking, "advance_depth_candidate", advance_candidate)
        self.args = SimpleNamespace(rgb_index=tmp_path / "rgb.txt", checkpoint=tmp_path / "model.pt",
            output=tmp_path / "run", max_frames=None, image_size=14, threads=2, mapper_cpus=(4, 5),
            backend="cpu", max_loops=0, salad_checkpoint=None, max_submaps=10,
            mapper_memory_limit_mib=100., save_predictions=False, frontend_cpus=(6, 7),
            candidate_cpus=(2, 3), capture_cpus=(0, 1), tracking_width=320, tracking_features=400,
            history_frames=30, history_seconds=10., bootstrap_gap=.1, map_period=.01, replay_stride=3)
        if mode == "background_budget":
            self.args.history_seconds = .2

    def now(self):
        self.clock += .01
        return self.clock

    def advance(self, duration):
        self.clock += duration


def test_edge_main_preserves_exact_batch_identity_and_replays_pending_candidate(edge, monkeypatch, tmp_path):
    service = RunnerService(edge, monkeypatch, tmp_path)
    metrics = edge.run(service.args)
    assert metrics["status"] == "completed_experiment" and not metrics["ORB_used"]
    assert metrics["source"]["delivered"] == len(service.rows)
    assert metrics["map_updates"] >= 2
    assert service.batch_jobs[0]["batch"][0]["frame_id"] == service.rows[0]["timestamp"]
    for job in service.batch_jobs:
        previous, latest = job["batch"]
        assert previous["sequence"] < latest["sequence"]
        for item in job["batch"]:
            assert item["path"] == service.rows[item["sequence"]]["path"]
            assert item["frame_id"] == item["timestamp"] == service.rows[item["sequence"]]["timestamp"]
            assert item["captured_monotonic"] == 100. + item["sequence"] * .2
    assert max(service.applies) >= 2  # Main advanced while detached history was replayed.
    assert service.source_closed and service.joins >= 1 and not service.worker.is_alive()
    assert all(executor.closed for executor in service.executors)
    events = [json.loads(line) for line in (service.args.output / "events.jsonl").read_text().splitlines()]
    adopted = [event for event in events if event["kind"] == "depth_adoption"]
    assert adopted and all(not event["emits_fresh_pose"] for event in adopted)
    assert len(adopted) < metrics["map_updates"]  # Single pending slot replaces an older result.
    for publication in map(json.loads, (service.args.output / "publications.jsonl").read_text().splitlines()):
        assert publication["latency_seconds"] == pytest.approx(
            publication["pose_available_monotonic"] - publication["captured_monotonic"])


@pytest.mark.parametrize("mode", ["startup_failure", "early_stopped", "mapper_failure", "source_failure",
                                 "candidate_failure", "advance_failure", "cleanup_failure", "exit_failure"])
def test_edge_failures_cannot_report_success_and_cleanup_is_bounded(edge, monkeypatch, tmp_path, mode):
    service = RunnerService(edge, monkeypatch, tmp_path, mode)
    with pytest.raises(RuntimeError):
        edge.run(service.args)
    metrics = json.loads((service.args.output / "metrics.json").read_text())
    assert metrics["status"] == "failed"
    assert service.joins >= 1 and not service.worker.is_alive()
    assert all(executor.closed for executor in service.executors)
    if mode != "startup_failure":
        assert service.source_closed


def test_rejected_candidate_never_creates_a_fresh_pose(edge, monkeypatch, tmp_path):
    service = RunnerService(edge, monkeypatch, tmp_path, "candidate_rejected")
    metrics = edge.run(service.args)
    assert metrics["status"] == "completed_experiment"
    assert metrics["pose_timing"]["fresh_valid_poses"] == 0
    assert service.applies and service.source_closed


def test_coordinator_advances_large_gap_before_owner_adoption(edge, monkeypatch, tmp_path):
    service = RunnerService(edge, monkeypatch, tmp_path, "background_advance")
    metrics = edge.run(service.args)
    assert metrics["status"] == "completed_experiment"
    assert service.background_advances and service.background_advances[0] == dict(base=1, target=6)
    assert all(gap <= 3 for gap in service.applies)
    events = [json.loads(line) for line in (service.args.output / "events.jsonl").read_text().splitlines()]
    advance = next(event for event in events if event["kind"] == "depth_background_catchup")
    adopted = next(event for event in events if event["kind"] == "depth_adoption")
    assert advance["pending_observations"] == 5 and advance["round"] == 1
    assert adopted["adopted"] and adopted["background_rounds"] == 1
    assert advance["anchor"]["frame_id"] == service.rows[1]["timestamp"]
    assert advance["submitted_monotonic"] < adopted["completed_monotonic"]
    assert metrics["source"]["delivered"] == len(service.rows)
    assert service.source_closed and all(executor.closed for executor in service.executors)


def test_coordinator_budget_exhaustion_rejects_without_owner_replay_or_hidden_retry(edge, monkeypatch, tmp_path):
    service = RunnerService(edge, monkeypatch, tmp_path, "background_budget")
    metrics = edge.run(service.args)
    assert metrics["status"] == "completed_experiment" and not service.background_advances
    events = [json.loads(line) for line in (service.args.output / "events.jsonl").read_text().splitlines()]
    rejected = next(event for event in events if event["kind"] == "depth_adoption")
    assert not rejected["adopted"] and rejected["reason"] == "background_catchup_budget_exhausted"
    assert rejected["background_rounds"] == 0
    assert not rejected["emits_fresh_pose"] and rejected["catchup_observations"] == 0
    assert not any(event["kind"] == "depth_background_catchup" for event in events)
    assert metrics["source"]["delivered"] == len(service.rows)
    assert metrics["pose_timing"]["fresh_valid_poses"] == 0
    assert service.source_closed and all(executor.closed for executor in service.executors)


@pytest.mark.parametrize("close_failure", [False, True])
def test_mapping_terminal_is_emitted_after_runtime_cleanup(monkeypatch, tmp_path, close_failure):
    import vggt_slam.edge_mapping as mapping

    order = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda n: None,
                                                            set_num_interop_threads=lambda n: None))
    monkeypatch.setitem(sys.modules, "vggt_slam.runtime", SimpleNamespace(load_model=lambda *a, **k: object()))
    monkeypatch.setitem(sys.modules, "vggt_slam.solver", SimpleNamespace(Solver=lambda **kwargs: object()))

    def close():
        order.append("cleanup")
        if close_failure:
            raise RuntimeError("synthetic backend close failure")

    monkeypatch.setitem(sys.modules, "vggt_slam.xnnpack_linear",
                        SimpleNamespace(install_xnnpack_linear=lambda *a, **k: SimpleNamespace(close=close)))
    monkeypatch.setattr(mapping.os, "sched_getaffinity", lambda pid: {4, 5})
    monkeypatch.setattr(mapping, "platform_sample", lambda: {"peak_rss_mib": 1.})
    packets = []

    class Results:
        def put(self, value, **kwargs):
            order.append(value["kind"])
            packets.append(value)

    config = dict(cpus=None, threads=2, checkpoint="fake.pt", image_size=14, backend="xnnpack-f16",
                  max_loops=0, salad_checkpoint=None, output=str(tmp_path))
    mapping.mapping_worker(config, SimpleNamespace(get=lambda: None), Results())
    assert order == ["ready", "cleanup", "error" if close_failure else "stopped"]
    if close_failure:
        assert packets[-1]["previous_terminal"]["kind"] == "stopped"
        assert "Runtime cleanup failed" in packets[-1]["error"]
