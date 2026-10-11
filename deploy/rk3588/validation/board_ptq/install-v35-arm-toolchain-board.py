#!/usr/bin/env python3
"""Prepare an isolated RK3588 ARM64 Toolkit 2.3.2 environment.

Preparation candidate, not an installation success receipt. This script is
executed only by the board operator. It installs pinned packages and builds
onnxoptimizer from its unchanged source archive; it never calls model forward,
RKNN load/build/inference, or calibration. The existing Python venv is untouched.
Our installer: MIT. Third-party packages keep their respective licences.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import urllib.parse
import urllib.request
import zipfile

PLAN_SHA = "a32e1a00c8bdbf74241de3de236cdaef050567b26b9d591a7f57c65ab247e2b4"
SDIST_SHA = "158bf9fda2e33bfe8ba1484f295cf25447921822f840de90831f7007af1295d3"
CXX_SHA = "2aafdb1e153fc490fbd510d352572eee55ecdd29dd95eca239b4460c1afe3a12"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path, algorithm="sha256", prefix=b""):
    digest = hashlib.new(algorithm)
    digest.update(prefix)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def board_guard(machine=None, system=None, compatible=None):
    require((platform.machine() if machine is None else machine) == "aarch64", "RK3588 ARM64 board required")
    require((platform.system() if system is None else system) == "Linux", "Linux board required")
    if compatible is None:
        path = Path("/proc/device-tree/compatible")
        compatible = path.read_bytes() if path.is_file() else b""
    require(any(t in (b"rockchip,rk3588", b"rockchip,rk3588s") for t in compatible.split(b"\0")),
            "RK3588 device-tree identity required; other ARM hosts prohibited")


def verify_file(path, row):
    path = Path(path)
    require(path.is_file() and path.stat().st_size == row["size"], "Package size mismatch: " + path.name)
    require(sha(path) == row["sha256"], "Package SHA mismatch: " + path.name)
    return path


def verify_sdk(path, expected_sha, toolkit):
    path = Path(path)
    require(path.name == toolkit["file"] and path.stat().st_size == toolkit["bytes"], "Wrong ARM SDK wheel")
    require(sha(path) == expected_sha, "Operator-provided SDK SHA mismatch")
    blob_prefix = ("blob " + str(path.stat().st_size) + "\0").encode()
    require(sha(path, "sha1", blob_prefix) == toolkit["git_blob_sha1"], "Official SDK Git blob mismatch")
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        require(len(names) == 1, "Ambiguous SDK wheel metadata")
        require(hashlib.sha256(archive.read(names[0])).hexdigest() == toolkit["metadata_sha256"],
                "SDK metadata does not match the dependency audit")


def safe_extract(archive_path, destination):
    """Extract the pinned source distribution without links or path traversal."""
    destination = Path(destination).resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        require(sum(m.size for m in members if m.isfile()) <= 80 * 1024**2, "Unexpected source archive size")
        for member in members:
            name = Path(member.name)
            require(not name.is_absolute() and ".." not in name.parts and
                    name.parts and name.parts[0] == "onnxoptimizer-0.3.8", "Unsafe source archive path")
            require(member.isdir() or member.isfile(), "Source archive links/devices prohibited")
            target = destination / name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("xb") as output:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(chunk)
    return destination / "onnxoptimizer-0.3.8"


def build_environment(venv, sysroot, work, inherited):
    """Process-local compiler settings; no global PATH or system changes."""
    venv, sysroot, work = [Path(p).resolve() for p in (venv, sysroot, work)]
    environment = dict(inherited)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    flags = ["--sysroot=" + str(sysroot), "-B" + str(sysroot / "usr/bin"),
             "-B" + str(sysroot / "usr/lib/gcc/aarch64-linux-gnu/11")]
    pybind = venv / "lib/python3.11/site-packages/pybind11/share/cmake/pybind11"
    cmake_args = ["-G", "Ninja", "-DCMAKE_MAKE_PROGRAM=" + str(venv / "bin/ninja"),
                  "-Dpybind11_DIR=" + str(pybind), "-DONNX_OPT_USE_SYSTEM_PROTOBUF=OFF"]
    environment.update(
        PATH=str(venv / "bin") + os.pathsep + str(sysroot / "usr/bin") + os.pathsep + inherited.get("PATH", ""),
        CC=str(sysroot / "usr/bin/aarch64-linux-gnu-gcc-11"),
        CXX=str(sysroot / "usr/bin/aarch64-linux-gnu-g++-11"),
        AR=str(sysroot / "usr/bin/aarch64-linux-gnu-ar"),
        CFLAGS=shlex.join(flags), CXXFLAGS=shlex.join(flags), LDFLAGS=shlex.join(flags),
        CMAKE_ARGS=shlex.join(cmake_args), MAX_JOBS="2", TMPDIR=str(work / "tmp"),
        PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PIP_NO_INDEX="1",
        PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_NO_CACHE_DIR="1",
        XDG_CACHE_HOME=str(work / "cache"), ONNX_OPT_USE_SYSTEM_PROTOBUF="0")
    return environment


def run_command(command, *, cwd, env, report):
    index = len(report["commands"])
    log = Path(report["work_directory"]) / ("command-%02d.log" % index)
    print("RUN", index, command[0], command[1:4], flush=True)
    start = time.monotonic()
    with log.open("wb") as stream:
        process = subprocess.run(list(map(str, command)), cwd=cwd, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, check=False)
    report["commands"].append(dict(argv=list(map(str, command)), cwd=str(cwd),
                                    exit_code=process.returncode, seconds=time.monotonic() - start,
                                    log_file=log.name, log_sha256=sha(log)))
    require(process.returncode == 0, "Command failed; see " + log.name)


def fetch_or_find(row, wheelhouse, destination, allow_download):
    if wheelhouse is not None:
        return verify_file(Path(wheelhouse) / row["filename"], row)
    require(allow_download, "Supply --wheelhouse or explicitly --download-wheels on the board")
    url = urllib.parse.urlparse(row["url"])
    require(url.scheme == "https" and url.hostname == "files.pythonhosted.org", "Non-PyPI download prohibited")
    destination = Path(destination) / row["filename"]
    with urllib.request.urlopen(row["url"], timeout=60) as source, destination.open("xb") as output:
        total = 0
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            total += len(chunk)
            require(total <= row["size"], "Download exceeds declared package size")
            output.write(chunk)
    return verify_file(destination, row)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dependency-plan", "base-python", "sdk-wheel", "gcc-sysroot", "venv-path", "output-dir",
                 "preflight-script"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sdk-wheel-sha256", required=True,
                        help="SHA256 computed by the operator; official Git blob is verified separately")
    parser.add_argument("--preflight-script-sha256", required=True)
    parser.add_argument("--dependency-plan-sha256", default=PLAN_SHA)
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--download-wheels", action="store_true")
    args = parser.parse_args(argv)
    board_guard()  # Before network, venv creation, package build, or subprocess.
    require(sha(args.dependency_plan) == args.dependency_plan_sha256 == PLAN_SHA, "Frozen dependency plan mismatch")
    plan = json.loads(args.dependency_plan.read_text())
    require(plan["schema"] == "v35_ARM64_RKNN_2_3_2_dependency_plan", "Unknown dependency plan")
    require(sha(args.preflight_script) == args.preflight_script_sha256, "Preflight script SHA mismatch")
    verify_sdk(args.sdk_wheel, args.sdk_wheel_sha256, plan["toolkit"])
    compiler = args.gcc_sysroot / "usr/bin/aarch64-linux-gnu-g++-11"
    require(sha(compiler) == CXX_SHA, "Pinned v27 board GCC SHA mismatch")
    for tool in ("aarch64-linux-gnu-gcc-11", "aarch64-linux-gnu-ar"):
        require((args.gcc_sysroot / "usr/bin" / tool).is_file(), "Isolated compiler tool missing: " + tool)
    output, venv = args.output_dir.resolve(), args.venv_path.resolve()
    require(not output.exists() and not venv.exists(), "Work directory and venv must both be new")
    protected = [args.base_python.resolve(), Path(sys.prefix).resolve(), args.dependency_plan.resolve(),
                 args.sdk_wheel.resolve(), args.preflight_script.resolve(), args.gcc_sysroot.resolve()]
    if args.wheelhouse is not None:
        protected.append(args.wheelhouse.resolve())
    require(not any(new == p or new.is_relative_to(p) or p.is_relative_to(new)
                    for new in (output, venv) for p in protected), "New installation/source paths overlap")
    require(not (output.is_relative_to(venv) or venv.is_relative_to(output)), "Work directory/venv overlap")
    require(platform.libc_ver()[0] == "glibc" and
            tuple(int(v) for v in platform.libc_ver()[1].split(".")[:2]) >= (2, 24), "glibc >=2.24 required")
    output.mkdir(parents=True, exist_ok=False)
    (output / "tmp").mkdir()
    pins = {str(p.resolve()): sha(p) for p in (Path(__file__), args.dependency_plan, args.sdk_wheel,
                                               args.preflight_script, compiler)}
    report = dict(schema="v35_isolated_ARM64_installation", status="failed", preparation_candidate=True,
                  script_sha256=sha(__file__), work_directory=str(output), venv_path=str(venv),
                  original_venv_modified=False, system_site_packages=False, commands=[], packages=[],
                  source_pins=pins, calibration_executed=False, model_forward_executed=False,
                  RKNN_build_executed=False, runtime_validated=False, partial_venv_preserved_on_failure=True)
    try:
        environment = build_environment(venv, args.gcc_sysroot, output, os.environ)
        probe = subprocess.run([str(args.base_python), "-I", "-c",
            "import json,sys,sysconfig;print(json.dumps({'version':list(sys.version_info[:2]),"
            "'include':sysconfig.get_path('include')}))"], env=environment, capture_output=True, text=True, check=True)
        metadata = json.loads(probe.stdout)
        require(metadata["version"] == [3, 11], "Base CPython 3.11 required")
        require((Path(metadata["include"]) / "Python.h").is_file(), "Matching Python.h required for source build")
        report["base_python_metadata"] = metadata
        run_command([args.base_python, "-I", "-m", "venv", str(venv)], cwd=output, env=environment, report=report)
        require("include-system-site-packages = false" in (venv / "pyvenv.cfg").read_text().lower(),
                "venv isolation not established")
        python = venv / "bin/python"
        pip = [python, "-I", "-m", "pip", "install", "--no-index", "--no-deps", "--no-cache-dir",
               "--disable-pip-version-check"]
        for row in plan["packages"]:
            if row["packagetype"] != "bdist_wheel":
                continue
            with tempfile.TemporaryDirectory(prefix="one-wheel-", dir=output / "tmp") as scratch:
                package = fetch_or_find(row, args.wheelhouse, scratch, args.download_wheels)
                run_command(pip + [package], cwd=output, env=environment, report=report)
                report["packages"].append(dict(name=row["name"], version=row["version"],
                                                bytes=row["size"], sha256=row["sha256"]))
        optimizer = next(r for r in plan["packages"] if r["name"] == "onnxoptimizer")
        require(optimizer["version"] == "0.3.8" and optimizer["sha256"] == SDIST_SHA, "Optimizer source pin mismatch")
        with tempfile.TemporaryDirectory(prefix="optimizer-archive-", dir=output / "tmp") as scratch:
            package = fetch_or_find(optimizer, args.wheelhouse, scratch, args.download_wheels)
            source = safe_extract(package, output / "optimizer-source")
        pybind = venv / "lib/python3.11/site-packages/pybind11/share/cmake/pybind11"
        require((pybind / "pybind11Config.cmake").is_file(), "External CPython-3.11-compatible pybind11 missing")
        run_command([python, "-I", "setup.py", "bdist_wheel"], cwd=source, env=environment, report=report)
        wheels = list((source / "dist").glob("onnxoptimizer-0.3.8-*.whl"))
        require(len(wheels) == 1, "Source build must produce exactly one optimizer wheel")
        run_command(pip + wheels, cwd=output, env=environment, report=report)
        report["optimizer_build"] = dict(source_sha256=SDIST_SHA, wheel_sha256=sha(wheels[0]),
            wheel_file=str(wheels[0].relative_to(output)), wheel_bytes=wheels[0].stat().st_size,
            MAX_JOBS=2, external_pybind11="2.11.1", protobuf_protoc="built from bundled source")
        run_command(pip + [args.sdk_wheel.resolve()], cwd=output, env=environment, report=report)
        run_command([python, "-I", args.preflight_script.resolve(), "--dependency-plan", args.dependency_plan.resolve(),
                     "--dependency-plan-sha256", PLAN_SHA, "--output-json", output / "preflight.json"],
                    cwd=output, env=environment, report=report)
        report["preflight"] = json.loads((output / "preflight.json").read_text())
        require(report["preflight"]["status"] == "ready_for_board_PTQ_diagnostic", "Post-install metadata preflight failed")
        run_command([python, "-I", "-m", "pip", "check"], cwd=output, env=environment, report=report)
        report["status"] = "installed_metadata_checked_not_calibrated_or_runtime_validated"
    except BaseException as exc:
        report.update(error=repr(exc), traceback=traceback.format_exc())
    finally:
        report["source_hashes_after"] = {p: sha(p) if Path(p).is_file() else None for p in pins}
        report["sources_unchanged"] = report["source_hashes_after"] == pins
        if not report["sources_unchanged"]:
            report["status"] = "failed"
        (output / "install.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report.get(k) for k in ("status", "error")}), flush=True)
    return 0 if report["status"].startswith("installed_metadata_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
