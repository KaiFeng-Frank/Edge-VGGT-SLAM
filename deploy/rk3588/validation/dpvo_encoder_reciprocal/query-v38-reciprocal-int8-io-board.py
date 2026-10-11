#!/usr/bin/env python3
"""Board-only ZERO-inference IO query for reciprocal/Mul spatial PTQ.

MIT project helper; DPVO reference MIT, Princeton Vision & Learning Lab.
SDK query/attribute arithmetic is delegated unchanged to the frozen v37 query.
New model hashes and quantization descriptors must come from actual receipts.
This tool does not submit inputs, run inference or prove INT8 coverage/quality.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import traceback

BASE = Path("/mnt/vggt-data/vggt-slam2/experiments")
PARENT_QUERY_SHA = "c61a377ede3d33b5a2412b1c7f526ea75deff2f18ed185f0f937792e41b1aca2"
PTQ_SHA = "72cad7e4a6a31d4860875c7d40cf2d672c1190ee3fedceb3271556fc52255f41"
CPU_DRIVER_SHA = "a7f181015d4479856329460dc4099171d4bb0dce8cedf317c35e719f3f36548b"
REWRITER_SHA = "6ccda3f8c50dff5b36c8d65e9901df50fa28b8fb1624a141baedbeda34829d28"
PARENT_PTQ_SHA = "4d660a6d48d296fda930adee7e192571263df717431a208daa63ae43e5f9427d"
ABI_SHA = "37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608"
HEADER_SHA = "c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a"
LIBRARY_SHA = "d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8"
SUCCESS = "queried_reciprocal_native_IO_parent_rejection_retained_not_inference_validated"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def board_guard(machine=None, system=None, compatible=None):
    require((platform.machine() if machine is None else machine) == "aarch64" and
            (platform.system() if system is None else system) == "Linux", "Linux aarch64 RK3588/S board only")
    if compatible is None:
        p = Path("/proc/device-tree/compatible")
        compatible = p.read_bytes() if p.is_file() else b""
    require(any(t in (b"rockchip,rk3588", b"rockchip,rk3588s") for t in compatible.split(b"\0")), "RK3588/S device identity required")


def module(path, digest, name, pin):
    spec = importlib.util.spec_from_file_location(name, pin(Path(path), digest))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def load_bundle(args, pin):
    """Validate only actual saved lineage; no RKNN/ORT execution in this API."""
    cpu = module(args.frozen_cpu_driver, CPU_DRIVER_SHA, "_v38_pinned_CPU_guard", pin)
    compiler = module(args.frozen_ptq_helper, PTQ_SHA, "_v38_pinned_PTQ_guard", pin)
    recipe = json.loads(pin(args.rewrite_dir/"rewrite.json", args.rewrite_sha256).read_text())
    cpu.recipe_guard(recipe, REWRITER_SHA)
    evidence = json.loads(pin(args.ort_dir/"ort.json", args.ort_sha256).read_text())
    compiler.cpu_gate(evidence, CPU_DRIVER_SHA, args.rewrite_sha256, REWRITER_SHA, cpu.PARENT_SHA, recipe["output"]["sha256"])
    require(evidence["capture_sha256"] == cpu.CAPTURE_SHA and evidence["reference_sha256"] == cpu.REFERENCE_SHA,
            "Original input/reference receipts required")
    parent = json.loads(pin(args.parent_ptq_json, PARENT_PTQ_SHA).read_text())
    cpu.unchanged(parent)
    compiled = json.loads(pin(args.compile_json, args.compile_sha256).read_text())
    validate_compile(compiled, recipe, evidence, parent, args)
    return compiled, recipe, evidence


def validate_compile(value, recipe, evidence, parent, args):
    require(value["schema"] == "v38_board_only_DPVO_reciprocal_spatial_encoder_PTQ" and
            value["status"] == "compiled_board_reciprocal_PTQ_parent_rejection_retained_not_runtime_validated" and
            value["script_sha256"] == PTQ_SHA, "Successful frozen reciprocal board PTQ required")
    require(value["parent_PTQ_source_sha256"] == "0fc013d5b6ed45cb97d592e8cd115304d971a000e2ab5353314413f56d41a3fa" and
            value["parent_PTQ_receipt_sha256"] == PARENT_PTQ_SHA and value["ancestor"] == parent["ancestor"], "Frozen spatial PTQ lineage required")
    require(value["configuration"] == parent["configuration"] and value["configuration"]["do_quantization"] is True and
            value["configuration"]["auto_hybrid"] is False, "Same frozen PTQ configuration required")
    require(value["released"] is True and value["cleanup_errors"] == [] and value["sources_and_inputs_unchanged"] is True and
            value["source_pins"] == value["source_hashes_after"], "Clean PTQ/source evidence required")
    require(value["diagnostic_only"] is True and value["candidate_accepted"] is False and value["production_changed"] is False and
            value["diagnostic_allow_fp32_rejected"] is True and value["historical_parent_FP32_gate_passed"] is False and
            value["historical_parent_rejection_retained"] is True and value["host_calibration_executed"] is False and
            value["calibration_location"] == "RK3588 aarch64 board CPU" and value["PTQ_requested"] is True and
            value["INT8_operator_coverage_validated"] is False and value["deployment_inference_executed"] is False,
            "Only board PTQ diagnostic; do not promote parent or runtime precision")
    require(value["input"]["sha256"] == recipe["output"]["sha256"] and value["input"]["original_graph_weights_preserved"] is True and
            value["graph_preservation"] == recipe["preservation"] and value["rewrite_sha256"] == args.rewrite_sha256 and
            value["rewriter_sha256"] == REWRITER_SHA and value["ORT_receipt_sha256"] == args.ort_sha256 and
            value["ORT_driver_sha256"] == CPU_DRIVER_SHA and value["candidate_vs_current_parent_CPU_strict_gate_passed"] is True and
            value["candidate_vs_original_CPU_strict_gate_passed"] == evidence["original_CPU_strict_gate_each_role"]["candidate"],
            "Same rewritten graph and unchanged CPU-gate evidence required")
    require(value["calibration"]["samples"] == 32 and len(value["calibration"]["rows"]) == 32 and
            value["calibration"]["rows"] == parent["calibration"]["rows"] and
            value["calibration"]["dataset_sha256"] == parent["calibration"]["dataset_sha256"] and
            value["calibration"]["source_dataset_sha256"] == "5c4c4712a2241fd0dd3cf83de2f454386f22bcdff4d1695077a8cd545b018372",
            "Same 32 rows/all64 file hashes and ordered calibration lists required")
    require([r["name"] for r in value["operations"]] == ["config", "load_onnx", "board_CPU_calibration_and_build", "export_rknn"] and
            all(type(r["code"]) is int and r["code"] == 0 for r in value["operations"]), "Actual board PTQ operations failed/changed")
    require(isinstance(value["output"]["sha256"], str) and len(value["output"]["sha256"]) == 64 and
            0 < value["output"]["bytes"] <= 128*1024**2, "Actual model output SHA/byte budget required")


def add_lineage_arguments(parser):
    for name in ("rewrite-dir", "ort-dir", "parent-ptq-json", "frozen-cpu-driver", "frozen-ptq-helper"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("rewrite", "ort"):
        parser.add_argument("--"+name+"-sha256", required=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "compile-json", "abi-helper", "header", "output-dir", "frozen-query-helper"):
        p.add_argument("--"+name, type=Path, required=True)
    for name in ("model", "compile"):
        p.add_argument("--"+name+"-sha256", required=True)
    add_lineage_arguments(p)
    p.add_argument("--library",type=Path,default=Path("/usr/lib/librknnrt.so"))
    p.add_argument("--diagnostic-allow-fp32-rejected",action="store_true")
    args = p.parse_args(argv)
    board_guard()  # prior to model/source reads, SDK loading, or directory writes
    require(args.diagnostic_allow_fp32_rejected,"Explicit rejected-parent diagnostic required")
    output=args.output_dir.resolve()
    require(output.is_relative_to(BASE) and output!=BASE and not output.exists(),"New isolated board experiment required")
    paths=[v.resolve() for k,v in vars(args).items() if isinstance(v,Path) and k!="output_dir"]
    require(not any(output==v or output.is_relative_to(v) or v.is_relative_to(output) for v in paths),"Output/source overlap")
    output.mkdir(parents=True,exist_ok=False)
    pins={}
    def pin(path,digest):
        path=Path(path).resolve()
        require(isinstance(digest,str) and len(digest)==64 and all(c in "0123456789abcdef" for c in digest) and sha(path)==digest,"Pinned file SHA mismatch: "+path.name)
        require(str(path) not in pins or pins[str(path)]==digest,"Conflicting source SHA")
        pins[str(path)]=digest
        return path
    report=dict(schema="v38_DPVO_reciprocal_PTQ_native_IO_query",status="failed",script_sha256=sha(__file__),
        compile_receipt_sha256=args.compile_sha256,model_sha256=args.model_sha256,parent_query_implementation_sha256=PARENT_QUERY_SHA,
        rewrite_sha256=args.rewrite_sha256,CPU_ORT_receipt_sha256=args.ort_sha256,diagnostic_allow_fp32_rejected=True,
        historical_parent_FP32_gate_passed=False,historical_parent_rejection_retained=True,original_DPVO_weights_preserved=True,
        initialization_attempted=False,initialized=False,context_destroyed=None,cleanup_errors=[],source_pins=pins,
        actual_board_execution=True,sdk_run_calls=0,model_inference_executed=False,NPU_used=False,NPU_operator_coverage_validated=False,
        candidate_accepted=False,precision_gate_measured=False,performance_measured=False,production_changed=False,ORB_used=False,
        native_IO_contract_checked=False,queried_command_ids=[5,0,1,2,6],bound_functions=["rknn_init","rknn_query","rknn_destroy"])
    try:
        pin(__file__,report["script_sha256"])
        base=module(args.frozen_query_helper,PARENT_QUERY_SHA,"_v37_frozen_query",pin)
        compiled,recipe,evidence=load_bundle(args,pin)
        pin(args.model,args.model_sha256)
        require(compiled["output"]["sha256"]==args.model_sha256 and compiled["output"]["bytes"]==args.model.stat().st_size,"Actual model differs from successful PTQ artifact")
        require(args.model.name==compiled["output"]["file"] and args.model.parent.resolve()==args.compile_json.parent.resolve(),"Same local compiled artifact required")
        for path,digest in ((args.abi_helper,ABI_SHA),(args.header,HEADER_SHA),(args.library,LIBRARY_SHA)):
            pin(path,digest)
        report.update(source_reciprocal_graph_sha256=recipe["output"]["sha256"],source_parent_spatial_graph_sha256=recipe["parent"]["sha256"],
            compile_binding=dict(configuration=compiled["configuration"],output=compiled["output"],compiler_sha256=PTQ_SHA,
                calibration_dataset_sha256=compiled["calibration"]["dataset_sha256"],rewrite_sha256=args.rewrite_sha256,
                CPU_ORT_receipt_sha256=args.ort_sha256,parent_PTQ_receipt_sha256=PARENT_PTQ_SHA))
        abi=base.load_abi(args.abi_helper)
        report["abi"]=abi.abi_record()
        require(report["abi"]["context_size"]==report["abi"]["pointer_size"]==8,"Pinned 64-bit ABI required")
        api=base.bind_query_api(ct.CDLL(str(args.library.resolve())),abi)
        base.probe_sdk(api,abi,args.model,report)
        report["status"]=SUCCESS
    except BaseException as exc:
        report.update(error=repr(exc),traceback=traceback.format_exc())
    finally:
        report["source_hashes_after"]={path:sha(path) if Path(path).is_file() else None for path in pins}
        report["sources_and_inputs_unchanged"]=report["source_hashes_after"]==pins
        if report["cleanup_errors"] or not report["sources_and_inputs_unchanged"]:report["status"]="failed"
        (output/"query.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:report.get(k) for k in ("status","error","sdk_run_calls","context_destroyed")}),flush=True)
    return 0 if report["status"]==SUCCESS else 1


if __name__=="__main__":
    raise SystemExit(main())
