#!/usr/bin/env python3
"""Board-only build of the dot4 candidate; no feature loading or inference."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

sys.dont_write_bytecode = True
PARENT_BUILDER_SHA = "b1f2553aa0e7786dd1343288ea67dd05d5b6e9f470c88693a6134d5d82756d2d"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, why):
    if not condition:
        raise ValueError(why)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source-dir", "parent-builder", "toolchain-dir", "output-dir"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("source-freeze-sha256", "bootstrap-sha256"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    require(sha(a.parent_builder) == PARENT_BUILDER_SHA, "Frozen v36 builder differs")
    spec = importlib.util.spec_from_file_location("_v45_frozen_builder", a.parent_builder)
    parent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent)
    parent.board_guard()  # No output/compiler/library access before ARM guard.
    out = a.output_dir.resolve()
    require(out.is_relative_to(parent.BASE), "New board experiments directory required")
    out.mkdir(exist_ok=False)
    pins = {}
    report = dict(schema="v45_horizontal_dot4_native_build", status="failed",
        script_sha256=sha(__file__), parent_builder_sha256=PARENT_BUILDER_SHA,
        cleanup_errors=[], real_features_loaded=False, actual_correlation_executed=False,
        neural_or_BA_executed=False, candidate_accepted=False)
    def pin(path, digest):
        path = Path(path).resolve()
        require(sha(path) == digest, "Pinned source/toolchain differs: " + str(path))
        pins[str(path)] = digest
        return path
    try:
        pin(__file__, report["script_sha256"])
        pin(a.parent_builder, PARENT_BUILDER_SHA)
        frozen = json.loads(pin(a.source_dir / "dot4-source-freeze.json", a.source_freeze_sha256).read_text())
        require(frozen["schema"] == "v45_horizontal_dot4_source_freeze"
                and frozen["build_script_sha256"] == report["script_sha256"], "Candidate freeze differs")
        for name, digest in frozen["source_sha256"].items():
            require(Path(name).name == name, "Source basename required")
            pin(a.source_dir / name, digest)
        require(frozen["parent_source_sha256"] == parent.PARENT_PINS, "Frozen v27 include pins differ")
        t = a.toolchain_dir.resolve()
        bootstrap = json.loads(pin(t / "bootstrap-r2.json", a.bootstrap_sha256).read_text())
        require(bootstrap["status"] == "isolated_compiler_and_synthetic_smoke_passed", "Verified toolchain required")
        root = t / "root"
        compiler = pin(root / "usr/bin/aarch64-linux-gnu-g++-11", bootstrap["compiler_sha256"])
        pin(t / "manifest.json", sha(t / "manifest.json"))
        library = out / "libdpvo_frame_corr_dot4.so"
        command = parent.build_command(a.source_dir.resolve(), a.source_dir.resolve(), library, compiler, root)
        index = command.index(str(a.source_dir.resolve() / "dpvo_frame_corr.cpp"))
        command[index] = str(a.source_dir.resolve() / "dpvo_frame_corr_dot4.cpp")
        env = os.environ.copy()
        env["TMPDIR"] = str(out)
        env["LD_LIBRARY_PATH"] = str(root / "usr/lib/aarch64-linux-gnu") + ":" + env.get("LD_LIBRARY_PATH", "")
        result = subprocess.run(command, cwd=out, env=env, capture_output=True, text=True)
        report.update(command=command, cwd=str(out), TMPDIR=env["TMPDIR"],
            exit_code=result.returncode, output_log=result.stdout + result.stderr,
            source_freeze_sha256=a.source_freeze_sha256, source_sha256=frozen["source_sha256"],
            parent_source_sha256=parent.PARENT_PINS, compiler_sha256=bootstrap["compiler_sha256"],
            toolchain_bootstrap_sha256=a.bootstrap_sha256)
        require(result.returncode == 0 and library.is_file(), "Candidate compilation failed")
        report.update(status="compiled_dot4_candidate_not_numerically_validated",
            output=dict(file=library.name, bytes=library.stat().st_size, sha256=sha(library)))
    except Exception as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc())
    finally:
        try:
            after = {path: sha(path) for path in pins}
            report.update(source_pins=pins, source_hashes_after=after, sources_and_inputs_unchanged=after == pins)
            require(after == pins, "Build input changed")
        except Exception as error:
            report["cleanup_errors"].append(repr(error))
            report["status"] = "failed"
        (out / "build.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report.get(k) for k in ("status", "output", "error", "cleanup_errors")}))
    return 0 if report["status"] == "compiled_dot4_candidate_not_numerically_validated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
