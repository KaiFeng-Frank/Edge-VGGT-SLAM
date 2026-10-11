#!/usr/bin/env python3
"""Board-only PTQ of a CPU-checked small reciprocal/Mul diagnostic.

MIT project helper; DPVO reference is MIT (Princeton Vision & Learning Lab).
CPU calibration is real neural execution: never run this on a host or other ARM
device. Original spatial FP32 rejection is retained; build success is not INT8
coverage, runtime precision, VO, or deployment acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import resource
import time
import traceback

PTQ_HELPER_SHA = "0fc013d5b6ed45cb97d592e8cd115304d971a000e2ab5353314413f56d41a3fa"
PARENT_PTQ_SHA = "4d660a6d48d296fda930adee7e192571263df717431a208daa63ae43e5f9427d"
DATASET_SHA = "5c4c4712a2241fd0dd3cf83de2f454386f22bcdff4d1695077a8cd545b018372"
BASE = Path("/mnt/vggt-data/vggt-slam2/experiments")
SUCCESS = "compiled_board_reciprocal_PTQ_parent_rejection_retained_not_runtime_validated"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def board_guard(machine=None, system=None, compatible=None):
    require((platform.machine() if machine is None else machine) == "aarch64", "Host PTQ prohibited; calibration is neural execution")
    require((platform.system() if system is None else system) == "Linux", "Linux board required")
    if compatible is None:
        p = Path("/proc/device-tree/compatible")
        compatible = p.read_bytes() if p.is_file() else b""
    require(any(t in (b"rockchip,rk3588", b"rockchip,rk3588s") for t in compatible.split(b"\0")), "RK3588 device-tree required")


def load_pinned(path, expected, name, pins):
    path = Path(path).resolve()
    require(isinstance(expected, str) and len(expected) == 64 and all(c in "0123456789abcdef" for c in expected) and sha(path) == expected, "Pinned source/receipt SHA mismatch")
    pins[str(path)] = expected
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def json_pin(path, expected, pins):
    path = Path(path).resolve()
    require(sha(path) == expected, "Receipt SHA mismatch: " + path.name)
    pins[str(path)] = expected
    return json.loads(path.read_text())


def cpu_gate(value, driver_sha, recipe_sha, rewriter_sha, parent_sha, candidate_sha):
    require(value["schema"] == "v38_DPVO_spatial_reciprocal_board_CPU_ORT" and
            value["status"] == "completed_CPU_ORT_reciprocal_diagnostic_parent_rejection_retained_not_VO" and
            value["script_sha256"] == driver_sha and value["rewrite_sha256"] == recipe_sha and
            value["rewriter_sha256"] == rewriter_sha and value["parent_graph_sha256"] == parent_sha and
            value["candidate_graph"]["sha256"] == candidate_sha, "Actual same-candidate CPU diagnostic chain required")
    require(value["candidate_vs_current_parent_strict_gate_passed"] is True and
            value["parent_reproduced_saved_v33_strict_gate_passed"] is True and
            value["original_CPU_strict_gate_each_role"]["parent"] is False and
            value["historical_parent_FP32_gate_passed"] is False and value["historical_parent_rejection_retained"] is True and
            value["candidate_accepted"] is False, "Candidate must pass original parent gate without promoting historical parent")
    require(value["gate"] == {"rtol":1e-4, "atol":1e-5} and value["graph_optimization"] == "ORT_DISABLE_ALL" and
            value["providers"] == ["CPUExecutionProvider"] and value["cpu_affinity"] == [4, 5, 6, 7] and
            value["affinity_restored"] is True and value["session_lifecycle"]["ORT_session_references_dropped"] is True,
            "Original CPU gate/provider/affinity and session cleanup required")
    require(value["sources_and_inputs_unchanged"] is True and value["source_pins"] == value["source_hashes_after"] and value["cleanup_errors"] == [], "CPU sources/cleanup failed")
    require([(r["frame_index"],r["role"]) for r in value["calls"]] == [(i,r) for i in range(3) for r in ("parent","candidate")], "Exactly three fixed parent/candidate CPU pairs required")
    require(all(v["gate_passed"] is True for r in value["calls"] if r["role"] == "candidate" for v in r["versus_current_parent_ORT"].values()), "Recorded individual parent gates failed")


def same_calibration(rows, parent):
    require(parent["calibration"]["samples"] == 32 and len(rows) == 32 and rows == parent["calibration"]["rows"], "Same 32 calibration rows/all 64 file hashes and shapes required")
    return dict(samples=32, input_files=64, same_order_and_paths=True, all_file_hashes_byte_counts_and_raw_guards_same=True,
                image_content_or_GT_selection_used=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dependency-plan", "preflight-script", "frozen-ptq-helper", "ort-driver", "parent-ptq-json", "parent-onnx",
                 "rewriter", "rewrite-dir", "ort-dir", "dataset", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("dependency-plan", "preflight-script", "ort-driver", "rewriter", "rewrite", "ort"):
        parser.add_argument("--"+name+"-sha256", required=True)
    parser.add_argument("--diagnostic-allow-fp32-rejected", action="store_true")
    args = parser.parse_args(argv)
    board_guard()  # before reading arrays, output creation, helper/ONNX/SDK imports
    require(args.diagnostic_allow_fp32_rejected, "Explicit rejected-parent PTQ diagnostic required")
    output = args.output_dir.resolve()
    require(output.is_relative_to(BASE) and output != BASE and not output.exists(), "New board experiment output required")
    protected = [v.resolve() for k,v in vars(args).items() if isinstance(v,Path) and k != "output_dir"]
    require(not any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected), "Output/source overlap")
    output.mkdir(parents=True, exist_ok=False)
    pins = {str(Path(__file__).resolve()):sha(__file__)}
    report = dict(schema="v38_board_only_DPVO_reciprocal_spatial_encoder_PTQ", status="failed", script_sha256=sha(__file__),
        parent_PTQ_source_sha256=PTQ_HELPER_SHA, parent_PTQ_receipt_sha256=PARENT_PTQ_SHA,
        diagnostic_only=True, diagnostic_allow_fp32_rejected=True, historical_parent_FP32_gate_passed=False,
        historical_parent_rejection_retained=True, candidate_accepted=False, production_changed=False,
        host_calibration_executed=False, calibration_location="RK3588 aarch64 board CPU", PTQ_requested=True,
        deployment_inference_executed=False, INT8_operator_coverage_validated=False, source_pins=pins, operations=[], released=False, cleanup_errors=[])
    begin = time.monotonic()
    try:
        base = load_pinned(args.frozen_ptq_helper, PTQ_HELPER_SHA, "_v37_frozen_PTQ_math", pins)
        cpu = load_pinned(args.ort_driver, args.ort_driver_sha256, "_v38_frozen_CPU_recipe_guards", pins)
        parent = json_pin(args.parent_ptq_json, PARENT_PTQ_SHA, pins)
        require(parent["schema"] == "v37_board_only_DPVO_spatial_encoder_PTQ" and
                parent["status"] == "compiled_board_spatial_PTQ_parent_rejection_retained_not_runtime_validated" and
                parent["script_sha256"] == PTQ_HELPER_SHA and parent["released"] is True and
                parent["historical_parent_FP32_gate_passed"] is False and parent["historical_parent_rejection_retained"] is True,
                "Frozen successful spatial PTQ, with rejected FP32 parent, required")
        cpu.unchanged(parent)
        recipe = json_pin(args.rewrite_dir/"rewrite.json", args.rewrite_sha256, pins)
        cpu.recipe_guard(recipe, args.rewriter_sha256)
        graph = (args.rewrite_dir/recipe["output"]["file"]).resolve()
        require(Path(recipe["output"]["file"]).name == recipe["output"]["file"] and sha(graph) == recipe["output"]["sha256"] and graph.stat().st_size == recipe["output"]["bytes"], "Candidate graph artifact mismatch")
        pins[str(graph)] = recipe["output"]["sha256"]
        require(parent["input"]["sha256"] == cpu.PARENT_SHA and sha(args.parent_onnx) == cpu.PARENT_SHA, "Same frozen spatial parent required")
        pins[str(args.parent_onnx.resolve())] = cpu.PARENT_SHA
        evidence = json_pin(args.ort_dir/"ort.json", args.ort_sha256, pins)
        cpu_gate(evidence, args.ort_driver_sha256, args.rewrite_sha256, args.rewriter_sha256, cpu.PARENT_SHA, recipe["output"]["sha256"])
        require(evidence["capture_sha256"] == cpu.CAPTURE_SHA and evidence["reference_sha256"] == cpu.REFERENCE_SHA and
                evidence["parent_recipe_sha256"] == cpu.PARENT_RECIPE_SHA, "Same original CPU/sampling reference chain required")
        report.update(ORT_receipt_sha256=args.ort_sha256, ORT_driver_sha256=args.ort_driver_sha256,
            candidate_vs_current_parent_CPU_strict_gate_passed=True, candidate_vs_original_CPU_strict_gate_passed=evidence["original_CPU_strict_gate_each_role"]["candidate"],
            rewrite_sha256=args.rewrite_sha256, rewriter_sha256=args.rewriter_sha256,
            input=dict(file=str(graph), sha256=recipe["output"]["sha256"], original_graph_weights_preserved=True), ancestor=parent["ancestor"])
        editor = load_pinned(args.rewriter, args.rewriter_sha256, "_v38_frozen_reciprocal_editor", pins)
        import onnx
        proof = editor.validate_preservation(onnx.load(args.parent_onnx,load_external_data=False), onnx.load(graph,load_external_data=False))
        require(proof == recipe["preservation"], "Static weight/node/IO preservation mismatch")
        report["graph_preservation"] = proof
        plan = json_pin(args.dependency_plan, args.dependency_plan_sha256, pins)
        preflight = load_pinned(args.preflight_script, args.preflight_script_sha256, "_v35_frozen_preflight", pins)
        report["environment"] = preflight.inspect_environment(plan)
        require(report["environment"]["status"] == "ready_for_board_PTQ_diagnostic", "Isolated ARM environment not ready")
        report["configuration"] = {**base.CONFIGURATION,"do_quantization":True,"auto_hybrid":False}
        require(report["configuration"] == parent["configuration"], "Only graph expression may change; same PTQ configuration required")
        dataset,rows = base.prepare_dataset(args.dataset, DATASET_SHA, output,pins)
        agreement = same_calibration(rows,parent)
        require(base.sha(dataset) == parent["calibration"]["dataset_sha256"], "Same resolved calibration list required")
        report["calibration"] = {**parent["calibration"],"rows":rows,"same_frozen_parent_calibration":agreement,
            "source_dataset_sha256":DATASET_SHA,"dataset_sha256":base.sha(dataset)}
        from rknn.api import RKNN
        base.perform_ptq(RKNN,graph,dataset,output,report)
        report["status"] = SUCCESS
    except BaseException as exc:
        report.update(error=repr(exc),traceback=traceback.format_exc())
    finally:
        report["source_hashes_after"] = {p:sha(p) if Path(p).is_file() else None for p in pins}
        report["sources_and_inputs_unchanged"] = report["source_hashes_after"] == pins
        report.update(peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,actual_driver_seconds=time.monotonic()-begin)
        if not report["sources_and_inputs_unchanged"] or report["cleanup_errors"]:
            report["status"] = "failed"
        (output/"compile.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:report.get(k) for k in ("status","error","released")}),flush=True)
    return 0 if report["status"] == SUCCESS else 1


if __name__ == "__main__":
    raise SystemExit(main())
