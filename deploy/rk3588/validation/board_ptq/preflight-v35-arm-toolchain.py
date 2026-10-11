#!/usr/bin/env python3
"""Read-only ARM toolchain preflight; stdlib only, no package/model imports.

The RKNN PTQ calibration pass must run on the board, not on the x86 host.
This file does not install, compile, calibrate, decode images, or run inference.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import sys
import sysconfig


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def version_tuple(value):
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", value)
    if not match:
        raise ValueError("Unrecognized version: " + value)
    return tuple(int(x or 0) for x in match.groups())


def evaluate_environment(plan, *, machine, python_version, prefix, base_prefix,
                         config_text, libc, installed, system, compatible):
    """Pure metadata checks; injectable arguments support tiny host tests."""
    problems = []
    if plan.get("schema") != "v35_ARM64_RKNN_2_3_2_dependency_plan":
        problems.append("Unknown dependency plan schema")
    if machine != "aarch64":
        problems.append("Board aarch64 required; x86 PTQ forbidden")
    if system != "Linux" or not any(t in (b"rockchip,rk3588", b"rockchip,rk3588s")
                                   for t in compatible.split(b"\0")):
        problems.append("Linux RK3588 device-tree identity required; other ARM hosts prohibited")
    if tuple(python_version[:2]) != (3, 11):
        problems.append("CPython 3.11 wheel ABI required")
    if Path(prefix).resolve() == Path(base_prefix).resolve():
        problems.append("A separate venv is required")
    fields = {}
    for line in config_text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip().lower()] = value.strip().lower()
    if fields.get("include-system-site-packages") != "false":
        problems.append("venv must exclude system-site-packages")
    try:
        libc_supported = libc[0] == "glibc" and version_tuple(libc[1]) >= (2, 24, 0)
    except (ValueError, IndexError):
        libc_supported = False
    if not libc_supported:
        problems.append("Selected ruamel.yaml.clib wheel requires glibc >=2.24")
    expected = {r["name"]: r["version"] for r in plan.get("packages", [])}
    expected["rknn-toolkit2"] = "2.3.2"
    rows = []
    for name, wanted in sorted(expected.items()):
        actual = installed.get(name)
        # A local CPU build suffix does not change the upstream release version.
        matched = actual is not None and actual.split("+", 1)[0] == wanted
        rows.append(dict(name=name, expected=wanted, installed=actual, matched=matched))
        if not matched:
            problems.append("Pinned package mismatch: " + name)
    return dict(status="ready_for_board_PTQ_diagnostic" if not problems else "blocked",
                problems=problems, packages=rows, host_neural_execution_allowed=False,
                calibration_location="board CPU", installation_executed=False,
                model_inference_executed=False, quantization_executed=False)


def inspect_environment(plan):
    config_path = Path(sys.prefix) / "pyvenv.cfg"
    installed = {}
    for name in [r["name"] for r in plan["packages"]] + ["rknn-toolkit2"]:
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = None
    dt_path = Path("/proc/device-tree/compatible")
    compatible = dt_path.read_bytes() if dt_path.is_file() else b""
    result = evaluate_environment(
        plan, machine=platform.machine(), python_version=sys.version_info,
        prefix=sys.prefix, base_prefix=sys.base_prefix,
        config_text=config_path.read_text() if config_path.is_file() else "",
        libc=platform.libc_ver(), installed=installed, system=platform.system(), compatible=compatible)
    result.update(python_executable=sys.executable, prefix=sys.prefix,
                  base_prefix=sys.base_prefix, machine=platform.machine(),
                  python_version=platform.python_version(), libc=list(platform.libc_ver()),
                  pyvenv_config_file=str(config_path), system=platform.system(),
                  device_tree_compatible=compatible.decode("ascii", errors="replace").split("\0"),
                  shared_library_imports_verified=False)
    include = Path(sysconfig.get_path("include"))
    result["optimizer_build_prerequisites"] = dict(
        python_include=str(include), Python_h_exists=(include / "Python.h").is_file(),
        sysconfig_CC=sysconfig.get_config_var("CC"),
        sysconfig_CXX=sysconfig.get_config_var("CXX"),
        executables={n: shutil.which(n) for n in ("cmake", "ninja", "gcc", "g++", "protoc", "make")},
        note="Optimizer source can build bundled protoc; gcc may be supplied by isolated v27 sysroot")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependency-plan", type=Path, required=True)
    parser.add_argument("--dependency-plan-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    if sha(args.dependency_plan) != args.dependency_plan_sha256:
        raise ValueError("Dependency plan SHA mismatch")
    if args.output_json.exists():
        raise ValueError("Preflight output must be new")
    plan = json.loads(args.dependency_plan.read_text())
    result = inspect_environment(plan)
    result.update(schema="v35_board_toolchain_preflight", script_sha256=sha(__file__),
                  dependency_plan_sha256=args.dependency_plan_sha256)
    result["dependency_plan_unchanged"] = sha(args.dependency_plan) == args.dependency_plan_sha256
    args.output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "problems")}), flush=True)
    return 0 if result["status"] == "ready_for_board_PTQ_diagnostic" else 1


if __name__ == "__main__":
    raise SystemExit(main())
