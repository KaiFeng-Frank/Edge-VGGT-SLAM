"""One-slot image replay using a dataset's original RGB timestamps.

Scheduled arrival is a software replay timestamp, not an exposure timestamp.
Decode and queuing occur after arrival and are included in replay pose latency.
"""
from pathlib import Path
from bisect import bisect_right
import math
import threading
import time


def read_rgb_index(path):
    path = Path(path).resolve()
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        stamp, relative = line.split(maxsplit=1)
        stamp = float(stamp)
        image = (path.parent / relative).resolve()
        if not math.isfinite(stamp) or (rows and stamp <= rows[-1]["timestamp"]):
            raise ValueError("RGB timestamps must be finite and strictly increasing")
        if not image.is_relative_to(path.parent) or not image.is_file():
            raise ValueError(f"RGB image missing or outside dataset: {relative}")
        rows.append(dict(timestamp=stamp, path=str(image)))
    if len(rows) < 2:
        raise ValueError("Need at least two timestamped RGB images")
    return rows


class TimestampReplay:
    def __init__(self, rows, *, cpus=None, decoder=None):
        self.rows = rows
        self.cpus = cpus
        self.decoder = decoder
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._slot = None
        self._thread = None
        self.error = None
        self.done = False
        self.generated = self.delivered = self.overwritten = 0
        self.skipped_before_decode = 0
        self.start_monotonic = self.end_monotonic = None
        self.last_capture_monotonic = None

    def start(self):
        if self._thread is not None:
            raise RuntimeError("Replay already started")
        self._thread = threading.Thread(target=self._run, name="timestamp-replay", daemon=True)
        self._thread.start()

    def _run(self):
        try:
            if self.cpus is not None:
                import os
                os.sched_setaffinity(0, self.cpus)
            if self.decoder is None:
                import cv2
                self.decoder = cv2.imread
            self.start_monotonic = time.monotonic()
            first = self.rows[0]["timestamp"]
            deadlines = [self.start_monotonic + row["timestamp"] - first for row in self.rows]
            sequence = 0
            while sequence < len(self.rows):
                captured = deadlines[sequence]
                if self._stop.wait(max(0., captured - time.monotonic())):
                    break
                # The decoder itself must not become an unbounded queue of
                # expired arrivals. Preserve the first observation, then decode
                # only the newest frame already due; never retimestamp a frame.
                if sequence:
                    newest = max(sequence, bisect_right(deadlines, time.monotonic()) - 1)
                    self.skipped_before_decode += newest - sequence
                    sequence = newest
                    captured = deadlines[sequence]
                row = self.rows[sequence]
                decode_started = time.monotonic()
                frame = self.decoder(row["path"])
                ready = time.monotonic()
                if frame is None:
                    raise RuntimeError(f"Cannot decode RGB image {row['path']}")
                item = dict(row, frame_id=row["timestamp"], sequence=sequence,
                            captured_monotonic=captured, decode_started_monotonic=decode_started,
                            ready_monotonic=ready, frame=frame)
                with self._condition:
                    self.overwritten += int(self._slot is not None)
                    self._slot = item
                    self.generated += 1
                    self.last_capture_monotonic = captured
                    self._condition.notify_all()
                sequence += 1
        except BaseException as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._condition:
                self.done = True
                self.end_monotonic = time.monotonic()
                self._condition.notify_all()

    def read(self, timeout=.005):
        with self._condition:
            if self._slot is None and not self.done:
                self._condition.wait(timeout)
            if self.error:
                raise RuntimeError(self.error)
            item, self._slot = self._slot, None
            if item is not None:
                self.delivered += 1
                item["delivered_monotonic"] = time.monotonic()
            return item

    @property
    def exhausted(self):
        with self._condition:
            return self.done and self._slot is None

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError("Image decoder did not stop within five seconds")

    def metrics(self):
        return dict(source="original_timestamp_rgb_replay", queue_capacity=1,
                    original_frames=len(self.rows), generated=self.generated, delivered=self.delivered,
                    overwritten=self.overwritten, skipped_before_decode=self.skipped_before_decode,
                    decode_policy="first_then_latest_due_original_timestamp", start_monotonic=self.start_monotonic,
                    end_monotonic=self.end_monotonic, last_capture_monotonic=self.last_capture_monotonic,
                    physical_sensor_latency_validated=False, error=self.error)
