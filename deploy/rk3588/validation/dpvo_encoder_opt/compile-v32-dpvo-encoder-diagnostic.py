#!/usr/bin/env python3
"""Compile only the pinned, FP32-rejected v27 encoder for isolated research.

This is not an acceptance path. The historical 1e-4/1e-5 failure remains a
mandatory ancestor, and no inference or production replacement occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

sys.dont_write_bytecode = True
SCHEMA = "v32_DPVO_rejected_encoder_diagnostic_compile"
SUCCESS = "compiled_diagnostic_FP32_rejected_not_runtime_validated"
CONFIGURATION = dict(target_platform="rk3588", float_dtype="float16",
                     optimization_level=3, do_quantization=False)
PARENT_SHA = "527fc2493de700a21aee958d7fda4d5ee421f4eb0c92296e81409df289f56a82"
CANDIDATE_SHA = "c3c4048c6d8792b09f0d6cbec8228f99d59c60d7503456eea2e369d7b8dbd826"
REWRITE_SHA = "66eb8d135165f7538c97733d0c83198fa5a674e0cd2bdf64eb190c24819c5109"
REWRITER_SHA = "671ceca472d104cedc318b380cd6dfaeb8852e4207383bf533fd4c98d0937f11"
FAILED_ORT_SHA = "a517f822a2e4afd01aff87df303183a3dc5850313134548891dc13c1e36b896b"
ORT_CHECKER_SHA = "f2ce1b906a58343925007ad4f3429f690ede07f5f230b4e2c2d7b743aa9e91cf"
COMPILER_SHA = "a40bf09e53ef9b378db4781be18f9e53f824afcc809c9009e0a65f4f0f91b4e5"
EXTRACTOR_SHA = "f5cbcbe14c42e44bcec6e827a3429ebe93979a439ba9b9ef70d4f4baa3454a2b"
CAPTURE_SHA = "e0563efb962edb90684480be114b88bc2c7825015f3b13a163f6f47f06360088"
EXPORT_SHA = "45fa589fa3a25a9722dc0832f5d14ffcec662e70d1dff135b86be641eeeba4b2"
ORIGINAL_ORT_SHA = "153da765b1145d7bb74bc635c1982ae74c652dac28cf99be56cb40534c44b667"
GATE = dict(rtol=1e-4, atol=1e-5, passed=False, status="rejected")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_lineage(rewrite, failed_ort):
    """Pure metadata check; callers must also hash receipts against fixed pins."""
    require(rewrite.get("kind") == "v27_dpvo_encoder_instance_norm_spatial_rewrite"
            and rewrite.get("version") == 1
            and rewrite.get("status") == "rewritten_and_statically_checked_not_runtime_validated",
            "Frozen v27 static rewrite required")
    require(rewrite.get("script_sha256") == REWRITER_SHA
            and rewrite["input"]["sha256"] == PARENT_SHA
            and rewrite["output"]["sha256"] == CANDIDATE_SHA
            and rewrite.get("external_data_used") is False
            and rewrite.get("sources_unchanged_after") is True,
            "Rewrite source/graph identity or lifecycle differs")
    proof = rewrite["preservation"]
    require(all(proof.get(key) is True for key in (
                "original_initializers_byte_exact", "non_instance_norm_nodes_byte_exact",
                "input_output_protobuf_byte_exact", "model_header_byte_exact"))
            and proof.get("original_initializer_count") == 44
            and proof.get("preserved_non_instance_norm_nodes") == 91
            and proof.get("added_nonlearned_constants") == 20
            and proof.get("original_weights_or_output_divide_by_four_changed") is False
            and len(rewrite["rewrites"]) == 10
            and rewrite["node_counts"] == {"original": 101, "candidate": 201}
            and rewrite["FP32_gate"] == dict(rtol=1e-4, atol=1e-5, status="pending_board_ORT"),
            "Only the ten pinned norms may change; historical gate must remain unchanged")
    require(failed_ort.get("schema") == "v27_DPVO_InstanceNorm_rewrite_board_ORT"
            and failed_ort.get("status") == "failed"
            and failed_ort.get("script_sha256") == ORT_CHECKER_SHA
            and failed_ort.get("actual_board_execution") is True
            and failed_ort.get("exact_FP32_gate_passed") is False
            and failed_ort.get("error") == "ValueError('Original FP32 gate failed; do not compile as validated')",
            "The exact historical FP32 numerical rejection is required")
    require(failed_ort.get("providers") == ["CPUExecutionProvider"]
            and failed_ort.get("graph_optimization") == "ORT_DISABLE_ALL"
            and failed_ort.get("affinity_restored") is True
            and failed_ort.get("sources_and_inputs_unchanged") is True
            and failed_ort.get("cleanup_errors") == []
            and failed_ort.get("source_hashes_after") == failed_ort.get("source_pins")
            and isinstance(failed_ort.get("source_pins"), dict) and failed_ort["source_pins"],
            "Historical ORT failure must be numerical, with clean source/lifecycle checks")
    require((failed_ort.get("capture_sha256"), failed_ort.get("export_sha256"),
             failed_ort.get("old_ort_sha256"), failed_ort.get("rewrite_sha256"))
            == (CAPTURE_SHA, EXPORT_SHA, ORIGINAL_ORT_SHA, REWRITE_SHA)
            and failed_ort.get("onnx") == rewrite["output"], "Historical ancestor hashes differ")
    calls = failed_ort["calls"]
    require(len(calls) == 3, "All three historical ORT frames are required")
    counts = {name: {key: [] for key in ("fmap", "imap")}
              for name in ("versus_original_CPU", "versus_original_ORT")}
    for index, call in enumerate(calls):
        require(call["frame_index"] == index, "Historical frame ordering differs")
        for comparison, outputs in counts.items():
            for name, failures in outputs.items():
                metric = call[comparison][name]
                count = metric["failed_elements"]
                require(type(count) is int and count >= 0
                        and metric.get("rtol") == 1e-4 and metric.get("atol") == 1e-5
                        and metric.get("gate_passed") is (count == 0), "Historical numerical gate is inconsistent")
                failures.append(count)
    require(counts["versus_original_CPU"] == dict(fmap=[4, 5, 53], imap=[0, 0, 0])
            and counts["versus_original_ORT"] == dict(fmap=[10, 2, 69], imap=[0, 0, 0]),
            "Historical rejected element counts differ")
    return dict(original_onnx_sha256=PARENT_SHA, candidate_onnx_sha256=CANDIDATE_SHA,
                rewrite_receipt_sha256=REWRITE_SHA, rewrite_script_sha256=REWRITER_SHA,
                failed_ORT_receipt_sha256=FAILED_ORT_SHA, failed_ORT_script_sha256=ORT_CHECKER_SHA,
                original_capture_sha256=CAPTURE_SHA, original_export_sha256=EXPORT_SHA,
                original_ORT_sha256=ORIGINAL_ORT_SHA, original_compiler_script_sha256=COMPILER_SHA,
                official_extractor_sha256=EXTRACTOR_SHA, preservation=dict(proof),
                historical_failed_elements=counts)


def validate_compilation(compiled, ancestor, *, expected_script_sha256):
    """Pure runtime receipt gate; no numerical acceptance is granted."""
    require(compiled.get("schema") == SCHEMA and compiled.get("status") == SUCCESS
            and compiled.get("script_sha256") == expected_script_sha256
            and compiled.get("configuration") == CONFIGURATION
            and compiled.get("ancestor") == ancestor
            and compiled.get("historical_FP32_gate") == GATE
            and compiled.get("diagnostic_only") is True
            and compiled.get("candidate_accepted") is False
            and compiled.get("model_inference_executed") is False
            and compiled.get("production_changed") is False
            and compiled.get("realtime_validated") is False, "Successful isolated diagnostic compile required")
    require(compiled.get("released") is True and compiled.get("cleanup_errors") == []
            and not compiled.get("release_error") and not compiled.get("error")
            and compiled.get("sources_and_inputs_unchanged") is True
            and compiled.get("source_pins") == compiled.get("source_hashes_after")
            and compiled.get("static_graph_preservation") == ancestor["preservation"],
            "Compiler cleanup, source, or original-parameter checks failed")
    require(compiled["input"]["sha256"] == CANDIDATE_SHA
            and compiled["versions"]["rknn-toolkit2"] == "2.3.2"
            and compiled["versions"]["torch"].startswith("2.4.0")
            and compiled["versions"]["numpy"] == "1.26.4", "Compiler input/environment differs")
    operations = compiled["operations"]
    require([row["name"] for row in operations] == ["config", "load_onnx", "build", "export_rknn"]
            and all((row["code"] is None or type(row["code"]) is int and row["code"] == 0)
                    and type(row["seconds"]) in (int, float) and math.isfinite(row["seconds"])
                    and row["seconds"] >= 0 for row in operations), "Compiler operation schedule/codes differ")
    output = compiled["output"]
    require(type(output["size"]) is int and 0 < output["size"] <= 128 << 20
            and isinstance(output["sha256"], str) and len(output["sha256"]) == 64,
            "Compiled model artifact metadata is invalid")


def perform_compile(factory, model_path, output_dir, report):
    """Original four SDK calls with unconditional release, including failures."""
    toolkit = None
    try:
        toolkit = factory(verbose=True)
        calls = [
            ("config", lambda: toolkit.config(target_platform="rk3588", float_dtype="float16", optimization_level=3)),
            ("load_onnx", lambda: toolkit.load_onnx(model=str(model_path))),
            ("build", lambda: toolkit.build(do_quantization=False)),
            ("export_rknn", lambda: toolkit.export_rknn(str(output_dir / "block.rknn"))),
        ]
        for name, call in calls:
            print("START", name, flush=True)
            start = time.monotonic()
            code = call()
            report["operations"].append(dict(name=name, code=code, seconds=time.monotonic() - start))
            require(code is None or type(code) is int and code == 0, f"{name} returned {code}")
        output = output_dir / "block.rknn"
        report["output"] = dict(path=str(output), size=output.stat().st_size, sha256=sha(output))
    finally:
        if toolkit is not None:
            try:
                toolkit.release()
                report["released"] = True
            except Exception as exc:
                report["release_error"] = repr(exc)
                report["cleanup_errors"].append("RKNN release: " + repr(exc))
                raise


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    for name in ("onnx", "parent-onnx", "rewrite", "failed-ort", "rewrite-helper",
                 "frozen-compiler", "extractor", "output-dir"):
        value.add_argument("--" + name, type=Path, required=True)
    for name in ("onnx", "rewrite", "failed-ort"):
        value.add_argument("--" + name + "-sha256", required=True)
    value.add_argument("--diagnostic-allow-fp32-rejected", action="store_true")
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    require(args.diagnostic_allow_fp32_rejected, "Explicit --diagnostic-allow-fp32-rejected is required")
    output_dir = args.output_dir.resolve()
    expected = {Path(__file__).resolve(): sha(__file__), args.onnx.resolve(): CANDIDATE_SHA,
                args.parent_onnx.resolve(): PARENT_SHA, args.rewrite.resolve(): REWRITE_SHA,
                args.failed_ort.resolve(): FAILED_ORT_SHA, args.rewrite_helper.resolve(): REWRITER_SHA,
                args.frozen_compiler.resolve(): COMPILER_SHA, args.extractor.resolve(): EXTRACTOR_SHA}
    require(all(not output_dir.is_relative_to(path) and not path.is_relative_to(output_dir)
                for path in expected), "Output overlaps a protected source/input")
    require((args.onnx_sha256, args.rewrite_sha256, args.failed_ort_sha256)
            == (CANDIDATE_SHA, REWRITE_SHA, FAILED_ORT_SHA), "Only the fixed historical candidate is allowed")
    output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(schema=SCHEMA, status="starting", script_sha256=expected[Path(__file__).resolve()],
                  diagnostic_only=True, candidate_accepted=False, historical_FP32_gate=dict(GATE),
                  model_inference_executed=False, production_changed=False, realtime_validated=False,
                  input=dict(path=str(args.onnx.resolve()), sha256=CANDIDATE_SHA),
                  configuration=dict(CONFIGURATION), operations=[], released=False, cleanup_errors=[],
                  machine=platform.machine(), python=sys.version,
                  source_pins={str(path): digest for path, digest in expected.items()})
    try:
        require(all(sha(path) == digest for path, digest in expected.items()), "Pinned source/input SHA mismatch")
        rewrite = json.loads(args.rewrite.read_text())
        failed_ort = json.loads(args.failed_ort.read_text())
        report["ancestor"] = validate_lineage(rewrite, failed_ort)
        import onnx
        import numpy as np
        import torch
        report["versions"] = {name: importlib.metadata.version(name) for name in
                              ("rknn-toolkit2", "torch", "numpy", "onnx", "onnxruntime", "protobuf")}
        require(torch.__version__.startswith("2.4.0") and np.__version__ == "1.26.4"
                and report["versions"]["rknn-toolkit2"] == "2.3.2", "Frozen compiler environment required")
        spec = importlib.util.spec_from_file_location("_v32_frozen_norm_rewriter", args.rewrite_helper)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        original = onnx.load(args.parent_onnx, load_external_data=False)
        candidate = onnx.load(args.onnx, load_external_data=False)
        helper._reject_external_or_subgraphs(original)
        helper._reject_external_or_subgraphs(candidate)
        onnx.checker.check_model(candidate, full_check=True)
        reconstructed, rows = helper.rewrite_instance_norm(original, expected_count=10)
        require(reconstructed.SerializeToString() == candidate.SerializeToString()
                and rows == rewrite["rewrites"], "Candidate differs from the frozen norm-only graph rewrite")
        report["static_graph_preservation"] = helper.validate_preservation(original, candidate, rows)
        require(report["static_graph_preservation"] == report["ancestor"]["preservation"],
                "Original parameter/graph preservation differs")
        from rknn.api import RKNN
        perform_compile(RKNN, args.onnx.resolve(), output_dir, report)
        report["status"] = SUCCESS
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
    finally:
        after = {str(path): sha(path) if path.is_file() else None for path in expected}
        report["source_hashes_after"] = after
        report["sources_and_inputs_unchanged"] = after == report["source_pins"]
        if not report["sources_and_inputs_unchanged"]:
            report["status"] = "failed"
            report["cleanup_errors"].append("Pinned source/input changed")
        if report["status"] == SUCCESS:
            try:
                validate_compilation(report, report["ancestor"], expected_script_sha256=report["script_sha256"])
            except Exception as exc:
                report.update(status="failed", error=repr(exc))
        report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        (output_dir / "compile.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=report["status"], candidate_accepted=False,
                         historical_FP32_gate_passed=False, error=report.get("error"),
                         receipt=str(output_dir / "compile.json")), sort_keys=True), flush=True)
    return 0 if report["status"] == SUCCESS else 1


if __name__ == "__main__":
    raise SystemExit(main())
