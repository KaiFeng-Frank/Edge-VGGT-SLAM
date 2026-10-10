"""Camera backends for real-time SLAM."""

from abc import ABC, abstractmethod
import math
import os
from pathlib import Path
import re
import struct
import threading
import time
from typing import Optional, Union

import numpy as np


def _validate_capture_cpus(cpus):
    if cpus is None:
        return None
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise ValueError("Capture CPU affinity requires Linux sched_getaffinity/sched_setaffinity.")
    requested = tuple(cpus)
    if (not requested or any(type(cpu) is not int or cpu < 0 for cpu in requested)
            or len(set(requested)) != len(requested)):
        raise ValueError("Capture CPUs must be distinct nonnegative integer CPU IDs.")
    allowed = os.sched_getaffinity(0)
    unavailable = set(requested) - allowed
    if unavailable:
        raise ValueError(f"Capture CPUs {sorted(unavailable)} are outside the caller's allowed CPUs {sorted(allowed)}.")
    return tuple(sorted(requested))


def _capture_thread_config(cpus):
    """Bind this reader before native decoding opens or creates child threads."""
    if cpus is not None:
        try:
            os.sched_setaffinity(0, cpus)
        except OSError as exc:
            raise RuntimeError(f"Could not bind capture reader to CPUs {list(cpus)}: {exc}") from exc
    actual = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    if cpus is not None and actual != list(cpus):
        raise RuntimeError(f"Capture reader requested CPUs {list(cpus)} but received {actual}.")
    return dict(native_thread_id=threading.get_native_id(), requested_cpus=cpus,
                cpu_affinity=actual)


class Camera(ABC):
    """Abstract camera that produces BGR uint8 frames."""

    def __init__(self):
        self._observer_lock = threading.Lock()
        self._frame_observer = None
        self._observer_error = None

    def _reset_observer_error(self):
        with self._observer_lock:
            self._observer_error = None

    def check(self) -> None:
        """Report a latched source failure even if no further capture is needed.

        Detaching the observer before checking closes the callback boundary.
        Healthy synchronous backends need no additional check. Errors remain
        available after stop and are reset by a new start.
        """
        with self._observer_lock:
            error = self._observer_error
        if error is not None:
            raise RuntimeError(f"Frame observer failed: {error}") from error

    def set_frame_observer(self, callback):
        """Attach a lightweight raw-frame sink, or synchronously detach it.

        The sink receives (BGR array, software arrival time, source path or
        None). It must not modify the array or block on inference. Replay and
        V4L2 invoke it on their reader, independently of the latest-frame
        consumer. Detach waits for the current callback to return. Exceptions
        fail the reader and are propagated by capture() or check().
        """
        if callback is not None and not callable(callback):
            raise TypeError("Frame observer must be callable or None")
        with self._observer_lock:
            self._frame_observer = callback

    def _observe_frame(self, frame, timestamp, path=None, stop=None, copy_for_observer=False):
        with self._observer_lock:
            if self._frame_observer is not None and (stop is None or not stop.is_set()):
                try:
                    self._frame_observer(frame.copy() if copy_for_observer else frame, timestamp, path)
                except Exception as exc:
                    # Publish before releasing the detach lock. The reader's
                    # outer except may not yet have stored session.error.
                    if self._observer_error is None:
                        self._observer_error = exc
                    raise

    @abstractmethod
    def start(self) -> None:
        """Open the device and begin streaming."""
        ...

    @abstractmethod
    def capture(self) -> Optional[np.ndarray]:
        """Return a BGR uint8 (H, W, 3) frame, or None if unavailable."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Release the device."""
        ...


class RealSenseCamera(Camera):
    """Intel RealSense color stream via pyrealsense2."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        super().__init__()
        self._width = width
        self._height = height
        self._fps = fps
        self._pipeline = None
        self._last_frame_timestamp = None

    @property
    def frame_timestamp(self) -> Optional[float]:
        """Monotonic software arrival after the SDK wait; not exposure time."""
        return self._last_frame_timestamp

    def start(self) -> None:
        import pyrealsense2 as rs

        self._reset_observer_error()
        self._last_frame_timestamp = None
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps)
        self._pipeline.start(config)

    def capture(self) -> Optional[np.ndarray]:
        frames = self._pipeline.wait_for_frames()
        received_at = time.monotonic()
        color = frames.get_color_frame()
        if color:
            self._last_frame_timestamp = received_at
            frame = np.asanyarray(color.get_data())
            # Only the asynchronous sink retains data beyond this SDK frame.
            self._observe_frame(frame, received_at, copy_for_observer=True)
            return frame
        return None

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None


def _read_sysfs(path: Path) -> Optional[str]:
    try:
        return path.read_text().strip()
    except (OSError, UnicodeError):
        return None


def enumerate_video_devices(sysfs_root: str = "/sys/class/video4linux") -> list:
    """Read camera identity from sysfs without starting any video stream.

    Rockchip exposes many ISP/CIF nodes even when no camera is connected.
    Those nodes must not be treated as evidence of an attached USB camera.
    """
    devices = []
    entries = Path(sysfs_root).glob("video*")
    for entry in sorted(entries, key=lambda p: int(p.name[5:]) if p.name[5:].isdigit() else -1):
        if not re.fullmatch(r"video\d+", entry.name):
            continue
        name = _read_sysfs(entry / "name") or entry.name
        usb_product = usb_vendor_id = usb_product_id = driver = None
        device_path = (entry / "device").resolve()
        for ancestor in (device_path, *device_path.parents):
            if driver is None and (ancestor / "driver").is_symlink():
                driver = (ancestor / "driver").resolve().name
            vendor = _read_sysfs(ancestor / "idVendor")
            product_id = _read_sysfs(ancestor / "idProduct")
            if vendor is not None and product_id is not None:
                usb_vendor_id, usb_product_id = vendor, product_id
                usb_product = _read_sysfs(ancestor / "product")
                break
        is_usb = usb_vendor_id is not None
        identity = f"{name} {driver or ''}".lower()
        is_internal = not is_usb and any(
            token in identity for token in ("rkisp", "rkispp", "rkcif", "rockchip", "rk-dvp")
        )
        devices.append({
            "device": f"/dev/{entry.name}",
            "name": name,
            "driver": driver,
            "is_usb": is_usb,
            "is_internal": is_internal,
            "usb_product": usb_product,
            "usb_vendor_id": usb_vendor_id,
            "usb_product_id": usb_product_id,
        })
    return devices


def _is_capture_node(device: str) -> bool:
    """Use VIDIOC_QUERYCAP to exclude UVC metadata-only nodes."""
    import fcntl

    fd = os.open(device, os.O_RDONLY | os.O_NONBLOCK)
    try:
        capability = bytearray(104)  # struct v4l2_capability
        fcntl.ioctl(fd, 0x80685600, capability, True)  # VIDIOC_QUERYCAP
        _, capabilities, device_caps = struct.unpack_from("=III", capability, 80)
        effective_caps = device_caps if capabilities & 0x80000000 else capabilities
        return bool(effective_caps & (0x00000001 | 0x00001000))
    finally:
        os.close(fd)


class _CaptureSession:
    def __init__(self):
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.condition = threading.Condition()
        self.thread = None
        self.error = None
        self.frame = None
        self.first_frame_time = None
        self.frame_time = None
        self.sequence = 0
        self.delivered_sequence = 0
        self.delivered_count = 0
        self.overwritten_count = 0
        self.reader_thread = None
        self.reader_end_monotonic = None


class ReplayCamera(Camera):
    """Paced image input for testing the asynchronous pipeline without hardware.

    The reader has one latest-frame slot. Slow consumers lose unread frames;
    they never cause an image queue to grow. Arrival times come from software
    read completion, and the requested rate is synthetic rather than a camera
    measurement. The final unread frame remains available before ``exhausted``.
    """

    def __init__(self, image_paths, fps: float = 30, timeout: float = 5.0,
                 capture_cpus=None):
        super().__init__()
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Replay fps must be positive and finite.")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Replay timeout must be positive and finite.")
        self._image_paths = tuple(str(path) for path in image_paths)
        if not self._image_paths:
            raise ValueError("Replay requires at least one image.")
        self._fps, self._timeout = fps, timeout
        self._capture_cpus = _validate_capture_cpus(capture_cpus)
        self._session = None
        self._last_frame_timestamp = None
        self._last_frame_path = None
        self._last_acquisition_stats = {}
        self._last_error = None

    @property
    def stream_info(self) -> dict:
        return {"backend": "paced_image_replay", "fps": self._fps,
                "input_frames": len(self._image_paths), "buffer_capacity_frames": 1}

    @property
    def frame_timestamp(self) -> Optional[float]:
        """Monotonic software read completion of the last delivered image."""
        return self._last_frame_timestamp

    @property
    def frame_path(self) -> Optional[str]:
        return self._last_frame_path

    @property
    def exhausted(self) -> bool:
        session = self._session
        if session is None:
            return bool(self._last_acquisition_stats.get("exhausted", False))
        with session.condition:
            return bool(session.exhausted and session.sequence <= session.delivered_sequence)

    @property
    def acquisition_stats(self) -> dict:
        session = self._session
        if session is None:
            return dict(self._last_acquisition_stats)
        with session.condition:
            return {"input_frames": len(self._image_paths), "frames_read": session.sequence,
                    "frames_delivered": session.delivered_count,
                    "frames_overwritten": session.overwritten_count,
                    "first_frame_monotonic": session.first_frame_time,
                    "last_frame_monotonic": session.frame_time,
                    "reader_thread": session.reader_thread,
                    "reader_exhausted": session.exhausted,
                    "reader_end_monotonic": session.reader_end_monotonic,
                    "exhausted": session.exhausted and session.sequence <= session.delivered_sequence,
                    "timestamp_basis": "monotonic software image read completion; synthetic replay timing"}

    def start(self) -> None:
        if self._session is not None and self._session.thread.is_alive():
            if not self._session.stop.is_set():
                return
            raise RuntimeError("Previous image replay is still shutting down.")
        self._reset_observer_error()
        self._last_error = None
        session = _CaptureSession()
        session.exhausted = False
        session.frame_path = None
        session.thread = threading.Thread(target=self._read_frames, args=(session,),
                                          name="paced-image-replay", daemon=True)
        self._session = session
        self._last_frame_timestamp = self._last_frame_path = None
        self._last_acquisition_stats = {}
        session.thread.start()

    def _read_frames(self, session: _CaptureSession) -> None:
        try:
            with session.condition:
                session.reader_thread = _capture_thread_config(self._capture_cpus)
            import cv2

            next_read = time.monotonic()
            for path in self._image_paths:
                if session.stop.wait(max(0.0, next_read - time.monotonic())):
                    return
                frame = cv2.imread(path, cv2.IMREAD_COLOR)
                received_at = time.monotonic()
                if frame is None or not frame.size:
                    raise RuntimeError(f"Could not decode replay image: {path}")
                if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                    raise RuntimeError(f"Expected a BGR uint8 replay image: {path}")
                self._observe_frame(frame, received_at, path, session.stop)
                with session.condition:
                    if session.stop.is_set():
                        return
                    if session.sequence > session.delivered_sequence:
                        session.overwritten_count += 1
                    session.frame = frame
                    session.frame_path = path
                    session.frame_time = received_at
                    if session.first_frame_time is None:
                        session.first_frame_time = received_at
                    session.sequence += 1
                    session.condition.notify_all()
                next_read = max(next_read + 1.0 / self._fps, received_at)
            with session.condition:
                session.exhausted = True
                session.reader_end_monotonic = time.monotonic()
        except Exception as exc:
            with session.condition:
                session.error = exc
        finally:
            with session.condition:
                if session.reader_end_monotonic is None:
                    session.reader_end_monotonic = time.monotonic()
                session.condition.notify_all()

    def capture(self) -> Optional[np.ndarray]:
        session = self._session
        if session is None or session.stop.is_set():
            raise RuntimeError("Replay is not started.")
        deadline = time.monotonic() + self._timeout
        with session.condition:
            while True:
                if session.error is not None:
                    raise RuntimeError(f"Image replay failed: {session.error}") from session.error
                if session.sequence > session.delivered_sequence:
                    session.delivered_sequence = session.sequence
                    session.delivered_count += 1
                    self._last_frame_timestamp = session.frame_time
                    self._last_frame_path = session.frame_path
                    return session.frame
                if session.exhausted:
                    return None
                if session.stop.is_set():
                    raise RuntimeError("Replay has been stopped.")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Waiting for a replay image timed out after {self._timeout:g} seconds.")
                session.condition.wait(remaining)

    def check(self) -> None:
        super().check()
        session = self._session
        if session is None:
            error = self._last_error
        else:
            with session.condition:
                error = session.error
        if error is not None:
            raise RuntimeError(f"Image replay failed: {error}") from error

    def stop(self) -> None:
        session = self._session
        if session is None:
            return
        session.stop.set()
        with session.condition:
            session.condition.notify_all()
        session.thread.join(timeout=min(self._timeout, 1.0))
        if not session.thread.is_alive():
            self._last_acquisition_stats = self.acquisition_stats
            self._last_error = session.error
            self._session = None


class OpenCVCamera(Camera):
    """Linux UVC/V4L2 camera, with bounded waits and a latest-frame buffer.

    An omitted device selects only USB capture nodes. Pass a device path (or
    index) to select one explicitly. Internal Rockchip ISP/CIF nodes require
    ``allow_internal=True`` and an independently configured sensor pipeline.

    Resolution and frame rate are requested from the driver; ``stream_info``
    reports the negotiated values. Native V4L2 reads run on a daemon thread,
    because OpenCV's read-timeout property does not cover V4L2. If a driver
    stays blocked during shutdown, its thread releases the device when the
    read returns; another start is refused until that thread exits.
    """

    def __init__(
        self,
        device: Optional[Union[str, int]] = None,
        width: int = 640,
        height: int = 480,
        fps: float = 30,
        timeout: float = 5.0,
        fourcc: Optional[str] = "MJPG",
        allow_internal: bool = False,
        capture_cpus=None,
    ):
        super().__init__()
        if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError("Camera width, height and fps must be positive.")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Camera timeout must be positive and finite.")
        if fourcc is not None and len(fourcc) != 4:
            raise ValueError("Camera fourcc must contain exactly four characters.")
        self._device = device
        self._width, self._height, self._fps = width, height, fps
        self._timeout, self._fourcc = timeout, fourcc
        self._allow_internal = allow_internal
        self._capture_cpus = _validate_capture_cpus(capture_cpus)
        self._session = None
        self._stream_info = {}
        self._last_frame_timestamp = None
        self._last_acquisition_stats = {}
        self._last_error = None

    @property
    def stream_info(self) -> dict:
        return dict(self._stream_info)

    @property
    def frame_timestamp(self) -> Optional[float]:
        """Monotonic software arrival time of the last delivered frame.

        This is taken after OpenCV's read, rather than when the caller consumes
        the buffer. It is not the camera's hardware exposure timestamp.
        """
        return self._last_frame_timestamp

    @property
    def acquisition_stats(self) -> dict:
        """Source reads and consumer deliveries, independent of SLAM rate."""
        session = self._session
        if session is None:
            return dict(self._last_acquisition_stats)
        with session.condition:
            return {
                "frames_read": session.sequence,
                "frames_delivered": session.delivered_count,
                "frames_overwritten": session.overwritten_count,
                "first_frame_monotonic": session.first_frame_time,
                "last_frame_monotonic": session.frame_time,
                "reader_thread": session.reader_thread,
                "reader_end_monotonic": session.reader_end_monotonic,
                "timestamp_basis": "monotonic time immediately after OpenCV read; no hardware exposure timestamp",
            }

    def _resolve_device(self) -> str:
        devices = enumerate_video_devices()
        if self._device is None:
            candidates = [device for device in devices if device["is_usb"]]
            query_errors = []
            for candidate in candidates:
                try:
                    if _is_capture_node(candidate["device"]):
                        return candidate["device"]
                except OSError as exc:
                    query_errors.append(f"{candidate['device']}: {exc}")
            detail = f" ({'; '.join(query_errors)})" if query_errors else ""
            raise RuntimeError(
                "No usable USB V4L2 capture device found. Connect a UVC camera "
                "or select a configured device explicitly." + detail
            )

        value = str(self._device)
        path = Path(f"/dev/video{value}" if value.isdigit() else value).resolve()
        if not re.fullmatch(r"video\d+", path.name) or path.parent != Path("/dev"):
            raise ValueError("Camera device must be /dev/videoN, its symlink, or a nonnegative index.")
        if not path.is_char_device():
            raise RuntimeError(f"Camera device does not exist or is not a character device: {path}")
        identity = next((device for device in devices if device["device"] == str(path)), None)
        if identity and identity["is_internal"] and not self._allow_internal:
            raise RuntimeError(
                f"{path} ({identity['name']}) is an internal Rockchip ISP/CIF node. "
                "It does not establish that a camera is connected. Configure the "
                "sensor pipeline before opting in with allow_internal=True."
            )
        return str(path)

    def start(self) -> None:
        if self._session is not None and self._session.thread.is_alive():
            if not self._session.stop.is_set():
                return
            raise RuntimeError("Previous V4L2 capture is still shutting down; its driver read has not returned.")
        device = self._resolve_device()
        self._reset_observer_error()
        self._last_error = None
        session = _CaptureSession()
        session.thread = threading.Thread(
            target=self._read_frames, args=(session, device), name="v4l2-camera", daemon=True
        )
        self._session = session
        self._stream_info = {}
        self._last_frame_timestamp = None
        self._last_acquisition_stats = {}
        session.thread.start()
        if not session.ready.wait(self._timeout):
            self.stop()
            raise TimeoutError(f"Opening {device} timed out after {self._timeout:g} seconds.")
        if session.error is not None:
            self.stop()
            raise RuntimeError(f"Could not start {device}: {session.error}") from session.error

    def _read_frames(self, session: _CaptureSession, device: str) -> None:
        capture = None
        try:
            with session.condition:
                session.reader_thread = _capture_thread_config(self._capture_cpus)
            import cv2

            capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
            if not capture.isOpened():
                raise RuntimeError("OpenCV could not open the V4L2 device (check permissions and device usage).")
            if self._fourcc is not None:
                capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self._fourcc))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            capture.set(cv2.CAP_PROP_FPS, self._fps)
            capture.set(cv2.CAP_PROP_CONVERT_RGB, 1)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._stream_info = {
                "device": device,
                "backend": "V4L2",
                "width": capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                "height": capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                "fps": capture.get(cv2.CAP_PROP_FPS),
                "fourcc": capture.get(cv2.CAP_PROP_FOURCC),
            }
            for key in ("width", "height", "fps", "fourcc"):
                if not math.isfinite(self._stream_info[key]):
                    self._stream_info[key] = None
            session.ready.set()
            last_success = time.monotonic()
            while not session.stop.is_set():
                ok, frame = capture.read()
                received_at = time.monotonic()
                if not ok or frame is None or frame.size == 0:
                    if time.monotonic() - last_success >= self._timeout:
                        raise TimeoutError(f"No frames from {device} for {self._timeout:g} seconds.")
                    session.stop.wait(0.02)
                    continue
                if frame.dtype != np.uint8:
                    raise RuntimeError(f"Expected a uint8 camera frame, received {frame.dtype}.")
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.ndim == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                if frame.ndim != 3 or frame.shape[2] != 3:
                    raise RuntimeError(f"Expected a BGR frame, received shape {frame.shape}.")
                last_success = received_at
                self._observe_frame(frame, received_at, stop=session.stop)
                with session.condition:
                    if session.stop.is_set():
                        break
                    if session.sequence > session.delivered_sequence:
                        session.overwritten_count += 1
                    session.frame = frame
                    session.frame_time = last_success
                    if session.first_frame_time is None:
                        session.first_frame_time = last_success
                    session.sequence += 1
                    session.condition.notify_all()
        except Exception as exc:
            with session.condition:
                session.error = exc
                session.condition.notify_all()
        finally:
            # Acquisition is over even if the driver takes time to release.
            # Consumers must not treat the final unread frame as an active source.
            with session.condition:
                session.reader_end_monotonic = time.monotonic()
            session.ready.set()
            if capture is not None:
                try:
                    capture.release()
                except Exception as exc:
                    if session.error is None:
                        session.error = exc
            with session.condition:
                session.condition.notify_all()

    def capture(self) -> Optional[np.ndarray]:
        session = self._session
        if session is None or session.stop.is_set():
            raise RuntimeError("Camera is not started.")
        deadline = time.monotonic() + self._timeout
        with session.condition:
            while session.sequence <= session.delivered_sequence:
                if session.error is not None:
                    raise RuntimeError(f"V4L2 capture failed: {session.error}") from session.error
                if session.stop.is_set():
                    raise RuntimeError("Camera has been stopped.")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Waiting for a new camera frame timed out after {self._timeout:g} seconds.")
                session.condition.wait(remaining)
            if time.monotonic() - session.frame_time > self._timeout:
                raise TimeoutError(f"The latest camera frame is older than {self._timeout:g} seconds; the stream stalled.")
            session.delivered_sequence = session.sequence
            session.delivered_count += 1
            self._last_frame_timestamp = session.frame_time
            return session.frame

    def check(self) -> None:
        super().check()
        session = self._session
        if session is None:
            error = self._last_error
        else:
            with session.condition:
                error = session.error
        if error is not None:
            raise RuntimeError(f"V4L2 capture failed: {error}") from error

    def stop(self) -> None:
        session = self._session
        if session is None:
            return
        session.stop.set()
        with session.condition:
            session.condition.notify_all()
        session.thread.join(timeout=min(self._timeout, 1.0))
        if not session.thread.is_alive():
            self._last_acquisition_stats = self.acquisition_stats
            self._last_error = session.error
            self._session = None


BACKENDS = {
    "realsense": RealSenseCamera,
    "opencv": OpenCVCamera,
    "v4l2": OpenCVCamera,
    "uvc": OpenCVCamera,
}
