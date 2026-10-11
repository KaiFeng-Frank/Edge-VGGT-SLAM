#!/usr/bin/env python3
"""Compile a pinned diagnostic original-weight source-frame Update; no inference.

Only RKNN graph compilation is executed. Board provenance is declared/pinned
through the export manifest; its /mnt source files are not present on the host.
FP32-rejected exports need explicit diagnostic authorization and remain rejected.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

sys.dont_write_bytecode = True
CONFIGURATION = dict(target_platform="rk3588", float_dtype="float16",
                     optimization_level=3, do_quantization=False)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compile_graph(toolkit, onnx_path, output_path, operations):
    """Exactly the existing config/load/build/export sequence; fake-testable."""
    calls = [
        ("config", lambda: toolkit.config(target_platform="rk3588", float_dtype="float16",
                                          optimization_level=3)),
        ("load_onnx", lambda: toolkit.load_onnx(model=str(Path(onnx_path).resolve()))),
        ("build", lambda: toolkit.build(do_quantization=False)),
        ("export_rknn", lambda: toolkit.export_rknn(str(Path(output_path).resolve()))),
    ]
    for name, call in calls:
        start = time.monotonic()
        code = call()
        operations.append(dict(name=name, code=code, seconds=time.monotonic()-start))
        if code not in (0, None):
            raise RuntimeError(name + " returned " + str(code))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--io-helper", type=Path,
                   default=Path(__file__).with_name("v35_rknn_source_frame_cf.py"))
    p.add_argument("--io-helper-sha256", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--allow-fp32-rejected-diagnostic", action="store_true")
    a = p.parse_args(argv)
    # A new output directory never overlays a source or previous receipt.
    a.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(schema="v35_channel_first_Update_RKNN_compilation", experiment_version=35,
        status="starting", script_sha256=sha256(__file__), configuration=CONFIGURATION,
        model_inference_executed=False, production_changed=False, candidate_accepted=False,
        realtime_validated=False, strict_thresholds_relaxed=False, operations=[],
        board_source_files_rehashed_on_host=False,
        provenance_boundary="Board absolute source files are declared by the pinned manifest; host rehashes manifest, ONNX, compiler and IO helper only.",
        allow_fp32_rejected_diagnostic=a.allow_fp32_rejected_diagnostic,
        machine=platform.machine(), python=sys.version)
    toolkit = None
    pins = {str(Path(__file__).resolve()): report["script_sha256"]}
    error = None
    try:
        if sha256(a.io_helper) != a.io_helper_sha256:
            raise ValueError("IO helper SHA256 mismatch")
        pins[str(a.io_helper.resolve())] = a.io_helper_sha256
        spec = importlib.util.spec_from_file_location("v33_compiler_IO_contract", a.io_helper)
        io = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = io
        spec.loader.exec_module(io)
        manifest, onnx_path, checked = io.validate_manifest(a.manifest, a.manifest_sha256,
            allow_fp32_rejected_diagnostic=a.allow_fp32_rejected_diagnostic,
            verify_source_files=False)
        pins.update(checked)
        report.update(manifest_sha256=a.manifest_sha256, io_helper_sha256=a.io_helper_sha256,
            input=dict(file=onnx_path.name, bytes=onnx_path.stat().st_size, sha256=sha256(onnx_path)),
            inputs=manifest["inputs"], outputs=manifest["outputs"],
            export_strict_FP32_gate_passed=manifest["strict_FP32_gate_passed"],
            original_parameters_unchanged=manifest["original_parameters_unchanged"],
            declared_board_source_pins=manifest["source_pins"],
            declared_board_source_hashes_after=manifest["source_hashes_after"])
        import torch
        import numpy as np
        import onnx
        from rknn.api import RKNN
        report["versions"] = {n: importlib.metadata.version(n) for n in
            ("rknn-toolkit2", "torch", "numpy", "onnx", "onnxruntime", "protobuf")}
        if (report["versions"]["rknn-toolkit2"] != "2.3.2"
                or not torch.__version__.startswith("2.4.0") or np.__version__ != "1.26.4"):
            raise RuntimeError("Use frozen Toolkit2 2.3.2 / Torch2.4.0 / NumPy1.26.4 toolchain")
        graph = onnx.load(str(onnx_path), load_external_data=False)
        if any(t.data_location == onnx.TensorProto.EXTERNAL for t in graph.graph.initializer):
            raise ValueError("External ONNX tensor data is unsupported")
        onnx.checker.check_model(graph, full_check=True)
        for field in ("inputs", "outputs"):
            values = graph.graph.input if field == "inputs" else graph.graph.output
            logical = [dict(name=v.name, dtype="float32" if v.type.tensor_type.elem_type == 1 else "unsupported",
                            shape=[d.dim_value if d.HasField("dim_value") else None
                                   for d in v.type.tensor_type.shape.dim]) for v in values]
            declared = [dict(name=r["name"], dtype="float32", shape=r["shape"])
                        for r in manifest[field]]
            if logical != declared:
                raise ValueError("ONNX " + field + " differs from pinned manifest")
        report["ONNX_operator_counts"] = dict(Counter(n.op_type for n in graph.graph.node))
        toolkit = RKNN(verbose=True)
        output = a.output_dir / "block.rknn"
        compile_graph(toolkit, onnx_path, output, report["operations"])
        report["output"] = dict(file=output.name, bytes=output.stat().st_size, sha256=sha256(output))
        report["status"] = "compiled_source_frame_diagnostic_not_runtime_validated"
    except BaseException as exc:
        error = exc
        report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
    finally:
        if toolkit is not None:
            try:
                toolkit.release()
                report["released"] = True
            except Exception as exc:
                report["release_error"] = repr(exc)
                report["released"] = False
                report["status"] = "failed"
                error = error or exc
        try:
            report["source_hashes_after"] = {p: sha256(p) for p in pins}
            report["sources_and_inputs_unchanged"] = report["source_hashes_after"] == pins
            if not report["sources_and_inputs_unchanged"]:
                raise ValueError("Compiler source or pinned input changed")
        except Exception as exc:
            report.update(status="failed", source_recheck_error=repr(exc))
            error = error or exc
        report["source_pins"] = pins
        report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        (a.output_dir / "compile.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:report.get(k) for k in ("status","error","manifest_sha256","output")}), flush=True)
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())

