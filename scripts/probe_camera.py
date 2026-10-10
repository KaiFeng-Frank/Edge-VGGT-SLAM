#!/usr/bin/env python3
"""List camera identities or verify actual V4L2 frames without loading SLAM.

Examples:
    python3 scripts/probe_camera.py --output camera_inventory.json
    python3 scripts/probe_camera.py --device /dev/video17 --frames 120 --output camera_probe.json
    python3 scripts/probe_camera.py --device auto --save-frame camera.jpg
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vggt_slam.cameras import OpenCVCamera, enumerate_video_devices


def enumerate_usb_devices(sysfs_root: str = "/sys/bus/usb/devices") -> list:
    devices = []
    for entry in sorted(Path(sysfs_root).glob("*")):
        try:
            vendor_id = (entry / "idVendor").read_text().strip()
            product_id = (entry / "idProduct").read_text().strip()
        except (OSError, UnicodeError):
            continue
        try:
            product = (entry / "product").read_text().strip()
        except (OSError, UnicodeError):
            product = None
        devices.append({
            "sysfs_device": entry.name,
            "product": product,
            "vendor_id": vendor_id,
            "product_id": product_id,
        })
    return devices


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", help="/dev/videoN, index N, or 'auto'; omitted means inventory only")
    parser.add_argument("--width", type=_positive_int, default=640)
    parser.add_argument("--height", type=_positive_int, default=480)
    parser.add_argument("--fps", type=_positive_float, default=30)
    parser.add_argument("--frames", type=_positive_int, default=120, help="number of fresh frames to measure")
    parser.add_argument("--warmup", type=int, default=5, help="number of initial frames to discard")
    parser.add_argument("--timeout", type=_positive_float, default=5.0, help="open/read timeout in seconds")
    parser.add_argument("--fourcc", default="MJPG", help="requested pixel format, e.g. MJPG or YUYV")
    parser.add_argument("--allow-internal", action="store_true", help="opt in to an explicitly selected, configured ISP/CIF node")
    parser.add_argument("--output", type=Path, help="also save the JSON report here")
    parser.add_argument("--save-frame", type=Path, help="save the last verified frame as an image")
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if len(args.fourcc) != 4:
        parser.error("--fourcc must contain exactly four characters")
    if args.save_frame and args.device is None:
        parser.error("--save-frame requires --device")

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "video_devices": enumerate_video_devices(),
        "usb_devices": enumerate_usb_devices(),
        "capture_verified": False,
        "capture": {"status": "not_requested"},
    }
    exit_code = 0
    if args.device is not None:
        camera = None
        report["capture"] = {
            "status": "pending",
            "requested": {
                "device": args.device,
                "width": args.width,
                "height": args.height,
                "fps": args.fps,
                "fourcc": args.fourcc,
                "frames": args.frames,
                "warmup_frames": args.warmup,
                "timeout_seconds": args.timeout,
            },
            "frames_captured": 0,
        }
        try:
            camera = OpenCVCamera(
                device=None if args.device == "auto" else args.device,
                width=args.width, height=args.height, fps=args.fps,
                timeout=args.timeout, fourcc=args.fourcc, allow_internal=args.allow_internal,
            )
            camera.start()
            report["capture"]["negotiated"] = camera.stream_info
            for _ in range(args.warmup):
                camera.capture()
            first_time = last_time = None
            start_time = time.monotonic()
            for _ in range(args.frames):
                frame = camera.capture()
                last_time = time.monotonic()
                if first_time is None:
                    first_time = last_time
                report["capture"]["frames_captured"] += 1
                report["capture"]["frame_shape"] = list(frame.shape)
            elapsed = last_time - start_time
            report["capture"].update({
                "status": "ok",
                "elapsed_seconds": elapsed,
                "measured_fps": (args.frames - 1) / (last_time - first_time) if args.frames > 1 else None,
                "fps_basis": "fresh frame delivery intervals; latest frames may be skipped if the consumer is slower",
            })
            report["capture_verified"] = True
            if args.save_frame:
                import cv2

                args.save_frame.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(args.save_frame), frame):
                    raise RuntimeError(f"Failed to save the verified frame to {args.save_frame}")
                report["capture"]["saved_frame"] = str(args.save_frame.resolve())
        except Exception as exc:
            report["capture"].update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            exit_code = 1
        finally:
            if camera is not None:
                camera.stop()

    encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded)
        except OSError as exc:
            print(f"Could not save camera report: {exc}", file=sys.stderr)
            exit_code = 1
    print(encoded, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
