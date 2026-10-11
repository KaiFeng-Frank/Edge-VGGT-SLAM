#!/usr/bin/env python3
"""Static RKNN compilation only; no model forward or graph numerical execution.

DPVO reference: MIT, Copyright (c) 2022 Princeton Vision & Learning Lab.
The spatial parent's rejected FP32 gate is mandatory diagnostic provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import time
import traceback

SCHEMA = "v34_DPVO_sparse_context_compile"
SUCCESS = "compiled_sparse_context_not_runtime_validated"
CONFIGURATION = dict(target_platform="rk3588", float_dtype="float16", optimization_level=3, do_quantization=False)
REFERENCE_SHA = "24aef39822c86ae60e1cb6306d2d2966fa8a440320c4363f18c5db938ac4e168"
EDITOR_SHA = "39004857559e27e3f9271ab413fc4498365370eac8862c56b16ba44b93f109ec"
FROZEN_COMPILER_SHA = "a40bf09e53ef9b378db4781be18f9e53f824afcc809c9009e0a65f4f0f91b4e5"
MODELS = {
    "original": {"recipe_sha": "368ec845f14655bf71e2fc9d80d435197c5ab6b8dd807c7c9af92ef9e33d13e0",
        "parent_sha": "527fc2493de700a21aee958d7fda4d5ee421f4eb0c92296e81409df289f56a82",
        "graph_sha": "c52cecd383e60ba564dcb5100be6c7e975fa6ecc051ed89e2c09bff73bc071dc"},
    "spatial": {"recipe_sha": "005667f44cf98447d2b622882d7d40cbcce8a545985ddf2104eaa62740273ce2",
        "parent_sha": "c3c4048c6d8792b09f0d6cbec8228f99d59c60d7503456eea2e369d7b8dbd826",
        "graph_sha": "6c3b4de4d151813dc172d711b6a760ad8bdc17120f2f3faa76dbf32bd0c451ab"},
}
INPUTS = [{"name": "normalized_bgr", "shape": [1, 3, 464, 608], "dtype": "float32"},
          {"name": "patch_indices", "shape": [96], "dtype": "int64"}]
OUTPUTS = [{"name": "fmap", "shape": [1, 128, 116, 152], "dtype": "float32"},
           {"name": "context", "shape": [1, 96, 384], "dtype": "float32"}]


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def pin(path, expected, pins):
    path = Path(path).resolve()
    require(sha(path) == expected, "Pinned SHA mismatch: " + path.name)
    require(str(path) not in pins or pins[str(path)] == expected, "Conflicting SHA")
    pins[str(path)] = expected
    return path


def validate_lineage(recipe, reference, method, diagnostic):
    require(method in MODELS, "Known sparse method required")
    spec = MODELS[method]
    require(recipe["schema"] == "v33_DPVO_encoder_sparse_context_rewrite" and
            recipe["status"] == "statically_checked_not_runtime_validated" and recipe["candidate_accepted"] is False and
            recipe["script_sha256"] == EDITOR_SHA and recipe["parent"]["sha256"] == spec["parent_sha"] and
            recipe["output"]["sha256"] == spec["graph_sha"] and recipe["inputs"] == INPUTS and recipe["outputs"] == OUTPUTS and
            recipe["source_pins"] == recipe["source_hashes_after"] and recipe["sources_and_parent_unchanged"] is True,
            "Exact sparse recipe/preservation/source identities required")
    require(reference["schema"] == "v33_DPVO_sparse_context_board_CPU_ORT" and
            reference["status"] == "completed_suffix_bitwise_gate_parent_rejection_retained_not_VO" and
            reference["suffix_bitwise_gate_passed"] is True and reference["suffix_strict_gate_passed"] is True and
            reference["source_pins"] == reference["source_hashes_after"] and
            reference["sources_and_inputs_unchanged"] is True and reference["cleanup_errors"] == [] and
            reference["receipt_sha256"][method + "_sparse"] == spec["recipe_sha"] and
            reference["graph_preservation"][method] == recipe["preservation"], "Exact accepted CPU suffix evidence required")
    require(reference["gate"]["rtol"] == 1e-4 and reference["gate"]["atol"] == 1e-5 and
            reference["original_CPU_strict_gate_each_method"] == {"original": True, "spatial": False} and
            reference["historical_spatial_rejection"]["exact_FP32_gate_passed"] is False and
            reference["historical_spatial_rejection"]["retained"] is True, "Historical parent rejection must not be promoted")
    require(method != "spatial" or diagnostic is True, "Spatial requires explicit --diagnostic-allow-fp32-rejected")
    rows = [row for row in reference["calls"] if row["method"] == method]
    require([row["frame_index"] for row in rows] == [0, 1, 2] and all(
        value["bitwise_equal"] is True and value["gate_passed"] is True
        for row in rows for value in row["versus_current_parent_ORT"].values()), "All three suffix frames required")
    return {"method": method, "reference_receipt_sha256": REFERENCE_SHA, "recipe_sha256": spec["recipe_sha"],
            "parent_onnx_sha256": spec["parent_sha"], "sparse_onnx_sha256": spec["graph_sha"],
            "suffix_CPU_bitwise_gate_passed": True, "parent_CPU_strict_gate_passed": method == "original",
            "spatial_parent_rejection_retained": True, "preservation": recipe["preservation"]}


def validate_compilation(report, method, expected_script):
    require(report["schema"] == SCHEMA and report["status"] == SUCCESS and report["script_sha256"] == expected_script and
            report["method"] == method and report["configuration"] == CONFIGURATION and
            report["input"]["sha256"] == MODELS[method]["graph_sha"] and
            report["ancestor"]["reference_receipt_sha256"] == REFERENCE_SHA and
            report["ancestor"]["recipe_sha256"] == MODELS[method]["recipe_sha"] and
            report["ancestor"]["parent_CPU_strict_gate_passed"] is (method == "original") and
            report["candidate_accepted"] is False and report["model_inference_executed"] is False and
            report["released"] is True and report["cleanup_errors"] == [] and
            report["sources_and_inputs_unchanged"] is True and report["source_pins"] == report["source_hashes_after"],
            "Successful hash-bound sparse compile required; not numerical runtime acceptance")
    require(report["versions"]["rknn-toolkit2"] == "2.3.2", "Pinned compiler version required")
    require([r["name"] for r in report["operations"]] == ["config", "load_onnx", "build", "export_rknn"] and all(
        (r["code"] is None or type(r["code"]) is int and r["code"] == 0) and
        math.isfinite(r["seconds"]) and r["seconds"] >= 0 for r in report["operations"]), "Compile operations invalid")
    require(0 < report["output"]["size"] <= 128 * 1024**2, "Compiled sparse size budget")


def perform_compile(factory, graph, output, report):
    toolkit = None
    try:
        toolkit = factory(verbose=True)
        for name, operation in [
            ("config", lambda: toolkit.config(target_platform="rk3588", float_dtype="float16", optimization_level=3)),
            ("load_onnx", lambda: toolkit.load_onnx(model=str(graph))),
            ("build", lambda: toolkit.build(do_quantization=False)),
            ("export_rknn", lambda: toolkit.export_rknn(str(output / "block.rknn")))]:
            print("START", name, flush=True)
            begin = time.monotonic()
            code = operation()
            report["operations"].append(dict(name=name, code=code, seconds=time.monotonic() - begin))
            require(code is None or type(code) is int and code == 0, "Failed compiler operation " + name)
        model = output / "block.rknn"
        report["output"] = dict(path=str(model), size=model.stat().st_size, sha256=sha(model))
    finally:
        if toolkit is not None:
            primary = sys.exc_info()[0] is not None
            try:
                toolkit.release()
                report["released"] = True
            except Exception as exc:
                report["cleanup_errors"].append("RKNN release: " + repr(exc))
                if not primary:
                    raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(MODELS), default="original")
    for name in ("rewrite-dir", "reference-dir", "parent-onnx", "editor", "frozen-compiler", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--rewrite-sha256", required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--diagnostic-allow-fp32-rejected", action="store_true")
    args = parser.parse_args(argv)
    require(args.reference_sha256 == REFERENCE_SHA and args.rewrite_sha256 == MODELS[args.method]["recipe_sha"], "Fixed receipts required")
    require(args.method != "spatial" or args.diagnostic_allow_fp32_rejected, "Explicit rejected-parent diagnostic required")
    output = args.output_dir.resolve()
    protected = [v.resolve() for k, v in vars(args).items() if isinstance(v, Path) and k != "output_dir"]
    require(not any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected), "Output/source overlap")
    output.mkdir(parents=True, exist_ok=False)
    pins = {}
    report = dict(schema=SCHEMA, status="failed", script_sha256=sha(__file__), method=args.method,
        configuration=CONFIGURATION, candidate_accepted=False, model_inference_executed=False,
        production_changed=False, realtime_validated=False, diagnostic_only=True, operations=[],
        released=False, cleanup_errors=[], cwd=str(Path.cwd()), source_pins=pins)
    try:
        pin(__file__, report["script_sha256"], pins)
        recipe = json.loads(pin(args.rewrite_dir / "rewrite.json", args.rewrite_sha256, pins).read_text())
        reference = json.loads(pin(args.reference_dir / "ort.json", args.reference_sha256, pins).read_text())
        report["ancestor"] = validate_lineage(recipe, reference, args.method, args.diagnostic_allow_fp32_rejected)
        graph = pin(args.rewrite_dir / recipe["output"]["file"], MODELS[args.method]["graph_sha"], pins)
        parent_path = pin(args.parent_onnx, MODELS[args.method]["parent_sha"], pins)
        report["input"] = dict(path=str(graph), sha256=sha(graph))
        require(graph.stat().st_size == recipe["output"]["bytes"], "Graph size differs")
        pin(args.frozen_compiler, FROZEN_COMPILER_SHA, pins)
        editor_path = pin(args.editor, EDITOR_SHA, pins)
        spec = importlib.util.spec_from_file_location("_v34_frozen_editor", editor_path)
        editor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(editor)
        import onnx
        parent = onnx.load(parent_path, load_external_data=False)
        candidate = onnx.load(graph, load_external_data=False)
        editor._reject_complex_storage(candidate)
        onnx.checker.check_model(candidate, full_check=True)
        proof = editor.preservation(parent, candidate)
        require(proof == recipe["preservation"] and editor._contract(candidate.graph.input) == INPUTS and
                editor._contract(candidate.graph.output) == OUTPUTS, "Actual graph preservation/IO differs")
        report["static_preservation"] = proof
        report["versions"] = {n: importlib.metadata.version(n) for n in ("rknn-toolkit2", "torch", "numpy", "onnx")}
        require(report["versions"]["rknn-toolkit2"] == "2.3.2" and report["versions"]["torch"].startswith("2.4.0") and
                report["versions"]["numpy"] == "1.26.4", "Frozen compiler environment required")
        from rknn.api import RKNN
        perform_compile(RKNN, graph, output, report)
        report["status"] = SUCCESS
    except BaseException as exc:
        report.update(error=repr(exc), traceback=traceback.format_exc())
    finally:
        report["source_hashes_after"] = {p: sha(p) if Path(p).is_file() else None for p in pins}
        report["sources_and_inputs_unchanged"] = report["source_hashes_after"] == pins
        if not report["sources_and_inputs_unchanged"] or report["cleanup_errors"]:
            report["status"] = "failed"
        if report["status"] == SUCCESS:
            try:
                validate_compilation(report, args.method, report["script_sha256"])
            except Exception as exc:
                report.update(status="failed", error=repr(exc))
        report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        (output / "compile.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report.get(k) for k in ("status", "error", "released")}), flush=True)
    return 0 if report["status"] == SUCCESS else 1


if __name__ == "__main__":
    raise SystemExit(main())
