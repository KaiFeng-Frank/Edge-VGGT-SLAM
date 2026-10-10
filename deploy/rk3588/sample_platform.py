#!/usr/bin/env python3
"""Read platform counters until one process exits; never change frequencies."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import threading
import time


STOP = threading.Event()


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def process_identity(pid):
    value = read_text(f"/proc/{pid}/stat")
    if value is None:
        return None
    fields = value.rsplit(") ", 1)[1].split()
    return None if fields[0] == "Z" else fields[19]


def read_number(path, divisor=1):
    value = read_text(path)
    try:
        return int(value) if divisor == 1 else int(value) / divisor
    except (TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    if args.pid < 1 or args.interval <= 0:
        parser.error("PID and interval must be positive")
    identity = process_identity(args.pid)
    if identity is None:
        parser.error("Target process has already exited")
    if args.output.exists() or args.summary.exists():
        parser.error("Output/summary already exists; nothing was overwritten")
    command = read_text(f"/proc/{args.pid}/cmdline")
    zones = {}
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        kind = read_text(zone / "type")
        if kind:
            zones[kind] = zone / "temp"
    frequency_counts = Counter()
    active_frequency_counts = Counter()
    ranges = {}
    samples = active_samples = 0
    began = time.monotonic()
    last = began
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda _sig, _frame: STOP.set())

    def update_range(name, value):
        if value is None:
            return
        row = ranges.setdefault(name, {"min": value, "max": value, "last": value})
        row["min"], row["max"], row["last"] = min(row["min"], value), max(row["max"], value), value

    with args.output.open("x") as stream:
        while not STOP.is_set() and process_identity(args.pid) == identity:
            last = time.monotonic()
            raw_load = read_text("/sys/kernel/debug/rknpu/load")
            load = {core: int(value) for core, value in re.findall(
                r"Core\s*(\d+)\s*:\s*(\d+)\s*%", raw_load or "", re.IGNORECASE)}
            npu_frequency = read_number("/sys/class/devfreq/fdab0000.npu/cur_freq")
            cpu_frequencies = {str(policy): read_number(
                f"/sys/devices/system/cpu/cpufreq/policy{policy}/scaling_cur_freq") for policy in (4, 6)}
            temperatures = {name: read_number(path, 1000) for name, path in zones.items()}
            row = dict(monotonic=last, wall_time_utc=datetime.now(timezone.utc).isoformat(),
                       target_pid=args.pid, npu_load_percent=load, npu_load_raw=raw_load,
                       npu_frequency_hz=npu_frequency, cpu_frequency_khz=cpu_frequencies,
                       temperature_c=temperatures)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            samples += 1
            if npu_frequency is not None:
                frequency_counts[str(npu_frequency)] += 1
            if any(value > 0 for value in load.values()):
                active_samples += 1
                if npu_frequency is not None:
                    active_frequency_counts[str(npu_frequency)] += 1
            for core, value in load.items():
                update_range("npu_load_core" + core + "_percent", value)
            for policy, value in cpu_frequencies.items():
                update_range("cpu_policy" + policy + "_khz", value)
            for name, value in temperatures.items():
                update_range("temperature_" + name + "_c", value)
            STOP.wait(max(0.0, args.interval - (time.monotonic() - last)))
        os.fsync(stream.fileno())
    summary = dict(status="interrupted" if STOP.is_set() else "target_exited",
                   target_pid=args.pid, target_start_ticks=identity,
                   target_command=command.replace("\0", " ") if command else None,
                   sampler_cpu_affinity=sorted(os.sched_getaffinity(0)),
                   interval_seconds=args.interval, samples=samples,
                   first_monotonic=began, last_monotonic=last,
                   duration_seconds=time.monotonic() - began,
                   npu_active_samples=active_samples,
                   npu_frequency_sample_counts=dict(frequency_counts),
                   npu_active_frequency_sample_counts=dict(active_frequency_counts),
                   ranges=ranges,
                   frequency_or_thermal_causality_validated=False,
                   limitations="Driver-reported load and instantaneous frequency sampled sequentially; load averaging period not independently verified; no hardware kernel timing or thermal-throttling cause inferred")
    with args.summary.open("x") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
