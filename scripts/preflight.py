#!/usr/bin/env python3
"""Record actual platform capabilities, without claiming SLAM was tested."""
import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party" / "vggt")]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", type=Path)
a = p.parse_args()
report = {"machine": platform.machine(), "python": sys.version,
          "kernel": platform.release(), "modules": {}, "slam_validated": False,
          "npu_inference_executed": False,
          "npu_runtime_present": Path("/usr/lib/librknnrt.so").is_file()}
report["cpu_frequency_policies"] = {}
for policy in sorted(Path("/sys/devices/system/cpu/cpufreq").glob("policy*")):
    values = {}
    for key in ("affected_cpus", "scaling_governor", "scaling_min_freq", "scaling_max_freq", "scaling_cur_freq"):
        try:
            values[key] = (policy / key).read_text().strip()
        except OSError:
            pass
    report["cpu_frequency_policies"][policy.name] = values
report["devfreq_at_preflight"] = {}
for device in Path("/sys/class/devfreq").glob("*"):
    values = {}
    for key in ("governor", "min_freq", "max_freq", "cur_freq"):
        try:
            values[key] = (device / key).read_text().strip()
        except OSError:
            pass
    report["devfreq_at_preflight"][device.name] = values
errors = []
for name in ("numpy", "torch", "torchvision", "cv2", "scipy", "open3d",
             "viser", "gtsam", "vggt_slam.solver", "vggt.models.vggt"):
    try:
        m = importlib.import_module(name)
        report["modules"][name] = getattr(m, "__version__", "imported")
        if name == "gtsam":
            missing = [x for x in ("SL4", "PriorFactorSL4", "BetweenFactorSL4")
                       if not hasattr(m, x)]
            if missing:
                raise RuntimeError(f"Missing required symbols: {missing}")
        if name == "torch":
            report["cuda_available"] = m.cuda.is_available()
    except Exception as exc:
        report["modules"][name] = f"ERROR: {type(exc).__name__}: {exc}"
        errors.append(name)
from vggt_slam.cameras import enumerate_video_devices
report["video_devices"] = enumerate_video_devices()
report["dependencies_pass"] = not errors
encoded = json.dumps(report, indent=2)
if a.output:
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(encoded + "\n")
print(encoded)
raise SystemExit(1 if errors else 0)
