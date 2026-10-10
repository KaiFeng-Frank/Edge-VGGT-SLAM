"""Camera tests use fake drivers and sysfs; they never open real hardware."""

import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import probe_camera
from vggt_slam import cameras


class FakeCapture:
    def __init__(self):
        self.frames = queue.Queue()
        self.released = False
        self.properties = {}

    def isOpened(self):
        return True

    def set(self, key, value):
        self.properties[key] = value
        return True

    def get(self, key):
        return self.properties.get(key, 0)

    def read(self):
        try:
            return True, self.frames.get(timeout=0.01)
        except queue.Empty:
            return False, None

    def release(self):
        self.released = True


def install_fake_capture(monkeypatch, capture):
    cv2 = SimpleNamespace(
        VideoCapture=lambda *args: capture,
        CAP_V4L2=200, CAP_PROP_FOURCC=6,
        CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4,
        CAP_PROP_FPS=5, CAP_PROP_CONVERT_RGB=16, CAP_PROP_BUFFERSIZE=38,
        VideoWriter_fourcc=lambda *args: 1196444237,
    )
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setattr(cameras.OpenCVCamera, "_resolve_device", lambda self: "/dev/video8")


def test_sysfs_distinguishes_internal_nodes_and_usb_products(tmp_path):
    root = tmp_path / "sys/class/video4linux"
    root.mkdir(parents=True)
    internal = root / "video0"
    internal.mkdir()
    (internal / "name").write_text("rkisp_mainpath")
    usb = tmp_path / "sys/devices/usb1/1-1"
    (usb / "1-1:1.0").mkdir(parents=True)
    (usb / "idVendor").write_text("1234")
    (usb / "idProduct").write_text("5678")
    (usb / "product").write_text("Test UVC Camera")
    external = root / "video17"
    external.mkdir()
    (external / "name").write_text("Test Video Capture")
    (external / "device").symlink_to(usb / "1-1:1.0", target_is_directory=True)
    rows = cameras.enumerate_video_devices(str(root))
    assert rows[0]["is_internal"] and not rows[0]["is_usb"]
    assert rows[1]["is_usb"] and not rows[1]["is_internal"]
    assert rows[1]["usb_product"] == "Test UVC Camera"
    assert rows[1]["usb_vendor_id"] == "1234"


def test_auto_selection_never_queries_internal_nodes_and_skips_metadata(monkeypatch):
    devices = [
        dict(device="/dev/video0", name="rkisp_mainpath", is_usb=False, is_internal=True),
        dict(device="/dev/video1", name="UVC metadata", is_usb=True, is_internal=False),
        dict(device="/dev/video2", name="UVC capture", is_usb=True, is_internal=False),
    ]
    monkeypatch.setattr(cameras, "enumerate_video_devices", lambda: devices)
    queried = []

    def query(device):
        queried.append(device)
        return device == "/dev/video2"

    monkeypatch.setattr(cameras, "_is_capture_node", query)
    assert cameras.OpenCVCamera()._resolve_device() == "/dev/video2"
    assert queried == ["/dev/video1", "/dev/video2"]
    monkeypatch.setattr(cameras, "enumerate_video_devices", lambda: devices[:1])
    with pytest.raises(RuntimeError, match="No usable USB"):
        cameras.OpenCVCamera()._resolve_device()
    assert queried == ["/dev/video1", "/dev/video2"]
    monkeypatch.setattr(Path, "is_char_device", lambda self: True)
    with pytest.raises(RuntimeError, match="internal Rockchip"):
        cameras.OpenCVCamera(device="/dev/video0")._resolve_device()


def test_delivery_is_fresh_and_missing_frames_time_out(monkeypatch):
    capture = FakeCapture()
    install_fake_capture(monkeypatch, capture)
    camera = cameras.OpenCVCamera(timeout=0.1)
    try:
        camera.start()
        for value in (1, 2):
            expected = np.full((6, 8, 3), value, dtype=np.uint8)
            capture.frames.put(expected)
            assert np.array_equal(camera.capture(), expected)
            assert camera.frame_timestamp <= time.monotonic()
        begin = time.monotonic()
        with pytest.raises((TimeoutError, RuntimeError)):
            camera.capture()
        assert time.monotonic() - begin < 0.8
        assert camera.acquisition_stats["frames_delivered"] == 2
    finally:
        camera.stop()
    assert capture.released
    assert camera.acquisition_stats["frames_read"] == 2
    assert camera.acquisition_stats["last_frame_monotonic"] <= camera.acquisition_stats["reader_end_monotonic"] <= time.monotonic()


def test_stale_unread_frame_is_rejected_and_blocked_driver_has_bounded_shutdown(monkeypatch):
    class BlockedCapture(FakeCapture):
        def __init__(self):
            super().__init__()
            self.blocked = threading.Event()
            self.unblock = threading.Event()
            self.first = True

        def read(self):
            if self.first:
                self.first = False
                return True, np.ones((6, 8, 3), dtype=np.uint8)
            self.blocked.set()
            self.unblock.wait()
            return False, None

    capture = BlockedCapture()
    install_fake_capture(monkeypatch, capture)
    camera = cameras.OpenCVCamera(timeout=0.1)
    try:
        camera.start()
        assert capture.blocked.wait(0.5)
        time.sleep(0.12)
        with pytest.raises(TimeoutError, match="older"):
            camera.capture()
        begin = time.monotonic()
        camera.stop()
        assert time.monotonic() - begin < 0.8
        with pytest.raises(RuntimeError, match="shutting down"):
            camera.start()
    finally:
        capture.unblock.set()
        if camera._session is not None:
            camera._session.thread.join(0.5)
        camera.stop()
    assert capture.released


def test_driver_open_is_also_bounded(monkeypatch):
    capture = FakeCapture()
    install_fake_capture(monkeypatch, capture)
    unblock = threading.Event()

    def blocked_open(*args):
        unblock.wait()
        return capture

    monkeypatch.setattr(sys.modules["cv2"], "VideoCapture", blocked_open)
    camera = cameras.OpenCVCamera(timeout=0.1)
    begin = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="Opening"):
            camera.start()
        assert time.monotonic() - begin < 0.8
    finally:
        unblock.set()
        if camera._session is not None:
            camera._session.thread.join(0.5)
        camera.stop()
    assert capture.released


def test_realsense_software_timestamp_follows_sdk_wait(monkeypatch):
    frame = np.zeros((6, 8, 3), dtype=np.uint8)
    pipeline = SimpleNamespace(
        start=lambda config: None,
        stop=lambda: None,
        wait_for_frames=lambda: SimpleNamespace(
            get_color_frame=lambda: SimpleNamespace(get_data=lambda: frame)
        ),
    )
    rs = SimpleNamespace(
        pipeline=lambda: pipeline,
        config=lambda: SimpleNamespace(enable_stream=lambda *args: None),
        stream=SimpleNamespace(color="color"), format=SimpleNamespace(bgr8="bgr8"),
    )
    monkeypatch.setitem(sys.modules, "pyrealsense2", rs)
    camera = cameras.RealSenseCamera()
    assert camera.frame_timestamp is None
    camera.start()
    begin = time.monotonic()
    assert camera.capture() is frame
    assert begin <= camera.frame_timestamp <= time.monotonic()
    camera.stop()


def test_inventory_does_not_claim_verified_camera(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(probe_camera, "enumerate_video_devices", lambda: [{"is_usb": True}])
    monkeypatch.setattr(probe_camera, "enumerate_usb_devices", lambda: [{"product": "Test UVC"}])
    output = tmp_path / "inventory.json"
    monkeypatch.setattr(sys, "argv", ["probe_camera.py", "--output", str(output)])
    assert probe_camera.main() == 0
    report = json.loads(output.read_text())
    assert not report["capture_verified"]
    assert report["capture"]["status"] == "not_requested"
    assert json.loads(capsys.readouterr().out) == report


def test_missing_device_reports_failure_and_json(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(probe_camera, "enumerate_video_devices", lambda: [])
    monkeypatch.setattr(probe_camera, "enumerate_usb_devices", lambda: [])
    output = tmp_path / "missing.json"
    monkeypatch.setattr(sys, "argv", [
        "probe_camera.py", "--device", "/dev/video999999", "--output", str(output)
    ])
    assert probe_camera.main() == 1
    report = json.loads(output.read_text())
    assert not report["capture_verified"]
    assert report["capture"]["status"] == "error"
    assert report["capture"]["frames_captured"] == 0
    assert "Camera device does not exist" in report["capture"]["error"]
    assert json.loads(capsys.readouterr().out) == report


def test_probe_success_measures_distinct_frame_deliveries(monkeypatch, tmp_path, capsys):
    class ProbeMock:
        stream_info = dict(device="/dev/video8", width=8, height=6, fps=50)

        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def capture(self):
            time.sleep(0.005)
            return np.zeros((6, 8, 3), dtype=np.uint8)

        def stop(self):
            pass

    monkeypatch.setattr(probe_camera, "OpenCVCamera", ProbeMock)
    monkeypatch.setattr(probe_camera, "enumerate_video_devices", lambda: [])
    monkeypatch.setattr(probe_camera, "enumerate_usb_devices", lambda: [])
    output = tmp_path / "capture.json"
    monkeypatch.setattr(sys, "argv", [
        "probe_camera.py", "--device", "/dev/video8", "--frames", "4",
        "--warmup", "1", "--output", str(output),
    ])
    assert probe_camera.main() == 0
    report = json.loads(output.read_text())
    assert report["capture_verified"]
    assert report["capture"]["frames_captured"] == 4
    assert 0 < report["capture"]["measured_fps"] <= 250
    assert report["capture"]["frame_shape"] == [6, 8, 3]
    assert json.loads(capsys.readouterr().out) == report


def test_replay_keeps_one_latest_frame_and_drains_it_before_eof(monkeypatch, tmp_path):
    paths = [tmp_path / f"{index}.png" for index in range(5)]
    frames = {str(path): np.full((6, 8, 3), index, dtype=np.uint8)
              for index, path in enumerate(paths)}
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        IMREAD_COLOR=1, imread=lambda path, mode: frames[path]))
    camera = cameras.ReplayCamera(paths, fps=500, timeout=0.2)
    try:
        camera.start()
        camera._session.thread.join(1.0)
        assert not camera._session.thread.is_alive()
        assert not camera.exhausted  # The last image still has to be delivered.
        before_drain = camera.acquisition_stats
        assert before_drain["reader_exhausted"] and not before_drain["exhausted"]
        assert before_drain["last_frame_monotonic"] <= before_drain["reader_end_monotonic"] <= time.monotonic()
        assert np.array_equal(camera.capture(), frames[str(paths[-1])])
        assert camera.frame_path == str(paths[-1])
        assert camera.frame_timestamp <= time.monotonic()
        assert camera.exhausted
        assert camera.capture() is None
        stats = camera.acquisition_stats
        assert stats["frames_read"] == 5
        assert stats["frames_delivered"] == 1
        assert stats["frames_overwritten"] == 4
        assert stats["exhausted"]
        assert stats["reader_end_monotonic"] == before_drain["reader_end_monotonic"]
        assert camera.stream_info["buffer_capacity_frames"] == 1
    finally:
        camera.stop()
    assert camera.exhausted
    assert camera.acquisition_stats == stats


def test_replay_paces_reads_independently_of_consumers(monkeypatch):
    arrivals = []

    def imread(*args):
        arrivals.append(time.monotonic())
        return np.zeros((6, 8, 3), dtype=np.uint8)

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(IMREAD_COLOR=1, imread=imread))
    camera = cameras.ReplayCamera(["0.png", "1.png", "2.png"], fps=20, timeout=0.2)
    try:
        camera.start()
        camera._session.thread.join(1.0)
        assert not camera._session.thread.is_alive()
        assert len(arrivals) == 3
        assert arrivals[-1] - arrivals[0] >= 0.09
        assert camera.acquisition_stats["frames_overwritten"] == 2
    finally:
        camera.stop()


@pytest.mark.parametrize("failure", ["decode", "exception"])
def test_replay_reader_failures_reach_capture(monkeypatch, failure):
    def imread(*args):
        if failure == "exception":
            raise OSError("injected read failure")
        return None

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(IMREAD_COLOR=1, imread=imread))
    camera = cameras.ReplayCamera(["0.png"], timeout=0.2)
    try:
        camera.start()
        with pytest.raises(RuntimeError, match="Image replay failed"):
            camera.capture()
        assert not camera.exhausted
        assert camera.acquisition_stats["frames_read"] == 0
        camera._session.thread.join(1.)
        assert not camera.acquisition_stats["reader_exhausted"]
        assert camera.acquisition_stats["reader_end_monotonic"] <= time.monotonic()
    finally:
        camera.stop()


def test_replay_blocked_decode_times_out_and_stops_boundedly(monkeypatch):
    unblock = threading.Event()
    entered = threading.Event()

    def imread(*args):
        entered.set()
        unblock.wait()
        return np.zeros((6, 8, 3), dtype=np.uint8)

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(IMREAD_COLOR=1, imread=imread))
    camera = cameras.ReplayCamera(["0.png"], timeout=0.05)
    try:
        camera.start()
        assert entered.wait(0.5)
        with pytest.raises(TimeoutError, match="replay image timed out"):
            camera.capture()
        begin = time.monotonic()
        camera.stop()
        assert time.monotonic() - begin < 0.5
        with pytest.raises(RuntimeError, match="shutting down"):
            camera.start()
    finally:
        unblock.set()
        if camera._session is not None:
            camera._session.thread.join(0.5)
        camera.stop()


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity"), reason="Linux CPU affinity required")
def test_replay_reader_binds_real_thread_without_changing_caller(monkeypatch):
    original = os.sched_getaffinity(0)
    requested = (min(original),)
    observed = []

    def imread(*args):
        observed.append((threading.get_native_id(), os.sched_getaffinity(0)))
        return np.zeros((6, 8, 3), dtype=np.uint8)

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(IMREAD_COLOR=1, imread=imread))
    camera = cameras.ReplayCamera(["0.png"], timeout=0.2, capture_cpus=requested)
    try:
        camera.start()
        assert camera.capture() is not None
        config = camera.acquisition_stats["reader_thread"]
        assert config["cpu_affinity"] == list(requested)
        assert config["native_thread_id"] == observed[0][0] != threading.get_native_id()
        assert observed[0][1] == set(requested)
        assert os.sched_getaffinity(0) == original
    finally:
        camera.stop()


def test_capture_cpu_validation_does_not_expand_callers_allowed_set(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {4, 5, 6, 7})
    for factory in (lambda cpus: cameras.ReplayCamera(["0.png"], capture_cpus=cpus),
                    lambda cpus: cameras.OpenCVCamera(capture_cpus=cpus)):
        with pytest.raises(ValueError, match="outside the caller's allowed CPUs"):
            factory([0])
        with pytest.raises(ValueError, match="distinct nonnegative"):
            factory([4, 4])


def test_replay_affinity_failure_propagates_before_image_decode(monkeypatch):
    decoded = []
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1})

    def fail_bind(*args):
        raise OSError("injected reader binding failure")

    monkeypatch.setattr(os, "sched_setaffinity", fail_bind)
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        IMREAD_COLOR=1, imread=lambda *args: decoded.append(True)))
    camera = cameras.ReplayCamera(["0.png"], timeout=0.2, capture_cpus=[0])
    try:
        camera.start()
        with pytest.raises(RuntimeError, match="injected reader binding failure"):
            camera.capture()
        assert not decoded
        assert camera.acquisition_stats["frames_read"] == 0
    finally:
        camera.stop()


def test_v4l2_reader_affinity_failure_propagates_before_open(monkeypatch):
    opened = []
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1})
    monkeypatch.setattr(cameras.OpenCVCamera, "_resolve_device", lambda self: "/dev/video8")

    def fail_bind(*args):
        raise OSError("injected V4L2 binding failure")

    monkeypatch.setattr(os, "sched_setaffinity", fail_bind)
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(VideoCapture=lambda *args: opened.append(True)))
    camera = cameras.OpenCVCamera(timeout=0.2, capture_cpus=[0])
    with pytest.raises(RuntimeError, match="injected V4L2 binding failure"):
        camera.start()
    assert not opened


def test_observer_detach_waits_for_inflight_callback():
    camera = cameras.ReplayCamera(["1.jpg"], timeout=1)
    entered, release, detached = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def observe(frame, timestamp, path):
        calls.append(path)
        entered.set()
        assert release.wait(2)

    camera.set_frame_observer(observe)
    worker = threading.Thread(target=camera._observe_frame,
                              args=(np.zeros((2, 2, 3), np.uint8), time.monotonic(), "1.jpg"))
    worker.start()
    assert entered.wait(1)
    def detach():
        camera.set_frame_observer(None)
        detached.set()
    detach_worker = threading.Thread(target=detach)
    detach_worker.start()
    try:
        assert not detached.wait(.05)
    finally:
        release.set()
        worker.join(2)
        detach_worker.join(2)
    assert detached.is_set() and not worker.is_alive() and not detach_worker.is_alive()
    camera._observe_frame(np.zeros((2, 2, 3), np.uint8), time.monotonic(), "2.jpg")
    assert calls == ["1.jpg"]


def test_replay_observer_failure_propagates_to_capture(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        IMREAD_COLOR=1, imread=lambda *args: np.zeros((2, 2, 3), np.uint8)))
    camera = cameras.ReplayCamera(["1.jpg", "2.jpg"], fps=25, timeout=1)
    def fail(*args):
        raise RuntimeError("injected observer failure")
    camera.set_frame_observer(fail)
    try:
        camera.start()
        with pytest.raises(RuntimeError, match="injected observer failure"):
            camera.capture()
    finally:
        camera.set_frame_observer(None)
        camera.stop()
    assert camera.acquisition_stats["frames_delivered"] == 0


def test_v4l2_raw_observer_is_independent_of_latest_consumer(monkeypatch):
    capture = FakeCapture()
    install_fake_capture(monkeypatch, capture)
    camera = cameras.OpenCVCamera(timeout=1)
    received = []
    camera.set_frame_observer(lambda frame, stamp, path: received.append((frame, stamp, path)))
    frames = [np.full((6, 8, 3), i, np.uint8) for i in (1, 2, 3)]
    try:
        camera.start()
        for frame in frames:
            capture.frames.put(frame)
        with camera._session.condition:
            assert camera._session.condition.wait_for(lambda: camera._session.sequence == 3, timeout=1)
        assert len(received) == 3
        assert all(item[0] is frame and item[2] is None for item, frame in zip(received, frames))
        assert all(a[1] < b[1] for a, b in zip(received, received[1:]))
        assert camera.capture() is frames[-1]
        assert camera.acquisition_stats["frames_overwritten"] == 2
        camera.set_frame_observer(None)
        capture.frames.put(np.zeros((6, 8, 3), np.uint8))
        camera.capture()
        assert len(received) == 3
    finally:
        camera.set_frame_observer(None)
        camera.stop()


def test_realsense_observer_owns_retained_pixels(monkeypatch):
    frame = np.ones((6, 8, 3), dtype=np.uint8)
    pipeline = SimpleNamespace(start=lambda config: None, stop=lambda: None,
        wait_for_frames=lambda: SimpleNamespace(get_color_frame=lambda:
            SimpleNamespace(get_data=lambda: frame)))
    monkeypatch.setitem(sys.modules, "pyrealsense2", SimpleNamespace(
        pipeline=lambda: pipeline, config=lambda: SimpleNamespace(enable_stream=lambda *args: None),
        stream=SimpleNamespace(color="color"), format=SimpleNamespace(bgr8="bgr8")))
    camera = cameras.RealSenseCamera()
    received = []
    camera.set_frame_observer(lambda image, timestamp, path: received.append((image, timestamp, path)))
    try:
        camera.start()
        assert camera.capture() is frame
        assert received[0][1] == camera.frame_timestamp and received[0][2] is None
        frame[:] = 0
        assert np.all(received[0][0] == 1)
    finally:
        camera.set_frame_observer(None)
        camera.stop()


@pytest.mark.parametrize("backend", ["replay", "v4l2"])
def test_reader_check_reports_observer_failure_without_next_capture(monkeypatch, backend):
    first = np.zeros((6, 8, 3), np.uint8)
    changed = np.zeros((12, 16, 3), np.uint8)
    failure = ValueError("injected raw resolution change")

    def observe(frame, timestamp, path):
        if frame.shape != first.shape:
            raise failure

    if backend == "replay":
        frames = {"1.jpg": first, "2.jpg": changed}
        monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
            IMREAD_COLOR=1, imread=lambda path, mode: frames[path]))
        camera = cameras.ReplayCamera(frames, fps=500, timeout=1)
    else:
        capture = FakeCapture()
        install_fake_capture(monkeypatch, capture)
        camera = cameras.OpenCVCamera(timeout=1)
    camera.set_frame_observer(observe)
    try:
        camera.start()
        if backend == "v4l2":
            capture.frames.put(first)
            capture.frames.put(changed)
        session = camera._session
        session.thread.join(1)
        assert not session.thread.is_alive()
        # A time/frame limit can leave this old slot unread. No next capture()
        # may be relied on to discover the reader's failure.
        assert session.sequence == 1 and session.delivered_sequence == 0
        with pytest.raises(RuntimeError, match="injected raw resolution change") as exc:
            camera.check()
        assert exc.value.__cause__ is failure
    finally:
        camera.set_frame_observer(None)
        camera.stop()
    assert camera._session is None
    with pytest.raises(RuntimeError, match="injected raw resolution change") as exc:
        camera.check()
    assert exc.value.__cause__ is failure


@pytest.mark.parametrize("backend", ["replay", "v4l2"])
def test_reader_check_keeps_nonobserver_error_after_stop(monkeypatch, backend):
    first = np.zeros((6, 8, 3), np.uint8)
    failure = OSError("injected independent reader failure")
    release = threading.Event()
    if backend == "replay":
        def imread(path, mode):
            if path == "1.jpg":
                return first
            raise failure

        monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(IMREAD_COLOR=1, imread=imread))
        camera = cameras.ReplayCamera(["1.jpg", "2.jpg"], fps=500, timeout=1)
        prefix = "Image replay failed"
    else:
        class FailedCapture(FakeCapture):
            def read(self):
                assert release.wait(2)
                raise failure

        capture = FailedCapture()
        install_fake_capture(monkeypatch, capture)
        camera = cameras.OpenCVCamera(timeout=1)
        prefix = "V4L2 capture failed"
    try:
        camera.start()
        release.set()
        session = camera._session
        session.thread.join(1)
        assert not session.thread.is_alive()
        reader_end = camera.acquisition_stats["reader_end_monotonic"]
        assert reader_end is not None and reader_end <= time.monotonic()
        with pytest.raises(RuntimeError, match=prefix) as exc:
            camera.check()
        assert exc.value.__cause__ is failure
    finally:
        release.set()
        camera.stop()
    assert camera._session is None
    assert camera.acquisition_stats["reader_end_monotonic"] == reader_end
    camera.stop()  # Repeated stop must not erase the failure.
    with pytest.raises(RuntimeError, match=prefix) as exc:
        camera.check()
    assert exc.value.__cause__ is failure


def test_observer_check_closes_detach_and_reader_error_publication_race():
    camera = cameras.ReplayCamera(["1.jpg"])
    failure = ValueError("injected callback failure before reader catches it")
    caught, publish = threading.Event(), threading.Event()
    reader_errors = []

    def fail(*args):
        raise failure

    def reader():
        try:
            camera._observe_frame(np.zeros((2, 2, 3), np.uint8), time.monotonic())
        except Exception as exc:
            caught.set()
            assert publish.wait(2)
            reader_errors.append(exc)

    camera.set_frame_observer(fail)
    worker = threading.Thread(target=reader)
    worker.start()
    try:
        assert caught.wait(1)
        camera.set_frame_observer(None)
        # Detach has completed, but the outer reader except has not yet
        # published its error. The observer lock must already protect a latch.
        assert reader_errors == []
        with pytest.raises(RuntimeError, match="callback failure") as exc:
            camera.check()
        assert exc.value.__cause__ is failure
    finally:
        publish.set()
        worker.join(2)
    assert not worker.is_alive() and reader_errors == [failure]


@pytest.mark.parametrize("backend", ["replay", "v4l2"])
def test_reader_check_clears_previous_error_only_on_new_start(monkeypatch, backend):
    frame = np.zeros((6, 8, 3), np.uint8)
    if backend == "replay":
        monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
            IMREAD_COLOR=1, imread=lambda *args: frame))
        camera = cameras.ReplayCamera(["1.jpg"], timeout=1)
    else:
        capture = FakeCapture()
        install_fake_capture(monkeypatch, capture)
        camera = cameras.OpenCVCamera(timeout=1)

    def fail(*args):
        raise ValueError("injected previous-session observer failure")

    camera.set_frame_observer(fail)
    try:
        camera.start()
        if backend == "v4l2":
            capture.frames.put(frame)
        camera._session.thread.join(1)
        assert not camera._session.thread.is_alive()
        camera.set_frame_observer(None)
        camera.stop()
        with pytest.raises(RuntimeError, match="previous-session"):
            camera.check()
        if backend == "v4l2":
            capture = FakeCapture()
            install_fake_capture(monkeypatch, capture)
        camera.start()
        if backend == "v4l2":
            capture.frames.put(frame)
        assert camera.capture() is frame
        assert camera.check() is None
    finally:
        camera.set_frame_observer(None)
        camera.stop()
    assert camera.check() is None


def test_camera_default_check_is_noop_for_healthy_synchronous_backend():
    camera = cameras.RealSenseCamera()
    assert camera.check() is None
