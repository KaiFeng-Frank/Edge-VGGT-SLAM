#!/usr/bin/env python3
"""Board-only reciprocal spatial encoder PTQ inference diagnostic, not VO.

MIT project helper, referencing MIT DPVO (Princeton Vision & Learning Lab).
Frozen v37 PTQContext and v34 transport perform SDK ownership/layout/timing.
FP32 NHWC inputs use pass_through=0; INT64 indices are untouched. Outputs are
owned logical FP32 copies before release. Parent rejection is never promoted.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
from types import SimpleNamespace

BASE = Path("/mnt/vggt-data/vggt-slam2/experiments")
QUERY_SHA = "78c5a8bafeca7686086ebe4fdaa7796af97f294b35e9c7d6c4ec8f90244bc3ed"
RUNTIME_SHA = "cc30f2bccf8b5b6178b62e8836eec5600504e927ac85b9d0f15814bcdb08df4a"
PARENT_QUERY_SHA = "c61a377ede3d33b5a2412b1c7f526ea75deff2f18ed185f0f937792e41b1aca2"
TRANSPORT_SHA = "646f875d1b5eea561c0d6aa4d1b9078351014a39a62976ec895a814cd466b387"
ABI_SHA = "37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608"
HEADER_SHA = "c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a"
LIBRARY_SHA = "d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8"
CAPTURE_SHA = "e0563efb962edb90684480be114b88bc2c7825015f3b13a163f6f47f06360088"
REFERENCE_SHA = "24aef39822c86ae60e1cb6306d2d2966fa8a440320c4363f18c5db938ac4e168"
SUCCESS = "completed_reciprocal_encoder_PTQ_parent_rejection_retained_not_VO"


def require(value,message):
    if not value:
        raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):
            h.update(chunk)
    return h.hexdigest()


def module(path,digest,name,pin):
    spec=importlib.util.spec_from_file_location(name,pin(path,digest))
    value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value
    spec.loader.exec_module(value)
    return value


def board_guard(machine=None,system=None,compatible=None):
    require((platform.machine() if machine is None else machine)=="aarch64" and (platform.system() if system is None else system)=="Linux","Linux aarch64 RK3588/S board only")
    if compatible is None:
        p=Path("/proc/device-tree/compatible")
        compatible=p.read_bytes() if p.is_file() else b""
    require(any(t in (b"rockchip,rk3588",b"rockchip,rk3588s") for t in compatible.split(b"\0")),"RK3588/S identity required")


def validate_query(value,compile_sha,model_sha,recipe_sha,ort_sha,graph_sha):
    require(value["schema"]=="v38_DPVO_reciprocal_PTQ_native_IO_query" and
            value["status"]=="queried_reciprocal_native_IO_parent_rejection_retained_not_inference_validated" and
            value["script_sha256"]==QUERY_SHA,"Actual frozen reciprocal zero-run query required")
    require(value["compile_receipt_sha256"]==compile_sha and value["model_sha256"]==model_sha and
            value["rewrite_sha256"]==recipe_sha and value["CPU_ORT_receipt_sha256"]==ort_sha and
            value["source_reciprocal_graph_sha256"]==graph_sha and
            value["source_parent_spatial_graph_sha256"]=="6c3b4de4d151813dc172d711b6a760ad8bdc17120f2f3faa76dbf32bd0c451ab",
            "Query must bind this exact PTQ/CPU/rewrite chain")
    require(value["sdk_run_calls"]==0 and value["model_inference_executed"] is False and value["native_IO_contract_checked"] is True and
            value["initialized"] is True and value["context_destroyed"] is True and value["cleanup_errors"]==[] and
            value["sources_and_inputs_unchanged"] is True and value["source_pins"]==value["source_hashes_after"],"Clean zero-run IO/ownership/source contract")
    require(value["diagnostic_allow_fp32_rejected"] is True and value["historical_parent_FP32_gate_passed"] is False and
            value["historical_parent_rejection_retained"] is True and value["candidate_accepted"] is False and
            value["parent_query_implementation_sha256"]==PARENT_QUERY_SHA,"Historical parent rejection must remain explicit")


def load_candidate_gold(args,evidence,rows,base,pin,artifact):
    import numpy as np
    calls=[r for r in evidence["calls"] if r["role"]=="candidate"]
    require([r["frame_index"] for r in calls]==[0,1,2],"All three reciprocal CPU reference calls required")
    for i,(row,call) in enumerate(zip(rows,calls)):
        identity=row["identity"]
        require(call["frame_index"]==i and call["frame_id"]==identity["frame_id"] and call["source_image"]==identity["source_image"] and
                call["input_metadata"]==identity["input_metadata"] and call["indices"]==identity["indices"] and call["centers"]==identity["centers"],"CPU candidate must use the same original pixels/INT64 centers/order")
        with np.load(artifact(args.ort_dir,call["artifact"]),allow_pickle=False) as archive:
            require(set(archive.files)=={"fmap","context"},"Exactly two CPU candidate features")
            saved={k:archive[k].copy(order="C") for k in archive.files}
        require(all(v.dtype==np.float32 and tuple(v.shape)==dict(base.OUTPUTS)[k] and np.isfinite(v).all() and
                    base.metadata(v)==call["arrays"][k] for k,v in saved.items()),"CPU candidate artifact/shape/raw SHA mismatch")
        row["reciprocal_ort"]=saved
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("compile-dir","query-dir","query-helper","capture-dir","reference-dir","transport-helper","abi-helper","header",
                 "frozen-runtime-helper","frozen-query-helper","rewrite-dir","ort-dir","parent-ptq-json","frozen-cpu-driver","frozen-ptq-helper","output-dir"):
        parser.add_argument("--"+name,type=Path,required=True)
    for name in ("compile","query","rewrite","ort"):
        parser.add_argument("--"+name+"-sha256",required=True)
    parser.add_argument("--capture-sha256",default=CAPTURE_SHA)
    parser.add_argument("--reference-sha256",default=REFERENCE_SHA)
    parser.add_argument("--library",type=Path,default=Path("/usr/lib/librknnrt.so"))
    parser.add_argument("--collect-perf-detail",action="store_true")
    parser.add_argument("--diagnostic-allow-fp32-rejected",action="store_true")
    args=parser.parse_args(argv)
    board_guard()  # before NumPy/imports/model/SDK or directory creation
    require(args.diagnostic_allow_fp32_rejected,"Explicit rejected-parent diagnostic required")
    output=args.output_dir.resolve()
    require(output.is_relative_to(BASE) and output!=BASE and not output.exists(),"New isolated board experiment required")
    paths=[v.resolve() for k,v in vars(args).items() if isinstance(v,Path) and k!="output_dir"]
    require(not any(output==v or output.is_relative_to(v) or v.is_relative_to(output) for v in paths),"Output/source overlap")
    output.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
    prior,pins,context=os.sched_getaffinity(0),{},None
    started=time.monotonic()
    report=dict(schema="v38_DPVO_reciprocal_sparse_encoder_PTQ_inference_diagnostic",status="failed",script_sha256=sha(__file__),
        compile_receipt_sha256=args.compile_sha256,query_receipt_sha256=args.query_sha256,rewrite_sha256=args.rewrite_sha256,
        CPU_ORT_receipt_sha256=args.ort_sha256,capture_sha256=args.capture_sha256,reference_sha256=args.reference_sha256,
        collect_perf_detail=args.collect_perf_detail,context={},calls=[],cleanup_errors=[],sdk_run_api_calls=0,source_pins=pins,
        actual_board_execution=True,NPU_used=False,candidate_accepted=False,production_changed=False,ORB_used=False,
        different_model_from_VGGT=True,original_VGGT_default_unchanged=True,original_FP32_source_graph=False,
        reciprocal_and_spatial_FP32_graph_rewrites_used=True,original_DPVO_weights_preserved=True,
        diagnostic_allow_fp32_rejected=True,historical_parent_FP32_gate_passed=False,historical_parent_rejection_retained=True,
        parent_runtime_implementation_sha256=RUNTIME_SHA,parent_query_implementation_sha256=PARENT_QUERY_SHA,
        execution_arithmetic_delegate_sha256=TRANSPORT_SHA,PTQ_changes_numeric_execution=True,
        INT8_internal_operator_coverage_verified=False,complete_VO=False,realtime_validated=False,Update_executed=False,
        correlation_executed=False,BA_executed=False,images_decoded=False,normalization_applications=0,
        manual_input_quantization=False,output_clipping=False,original_FP32_gate=dict(rtol=1e-4,atol=1e-5),
        expected_output_FP32_logical_bytes=9175040,
        timing_scope="Frozen transport: FP32 NHWC/INT64 packing + SDK set/run/get + owned logical FP32 copies + release; all comparisons/hashes/save/profile queries excluded",
        limitations=["Sparse encoder, three fixed saved inputs, not fresh-frame VO or quality/performance acceptance.",
            "Spatial parent failed unchanged original CPU gate; its rejection remains retained.",
            "Actual INT8 native IO and SDK profile labels are separate evidence from kernel precision and end-to-end accuracy.",
            "Nonfinite or all-zero outputs are fatal with raw arrays retained. Strict FP32 mismatch is reported without threshold relaxation."])
    def pin(path,digest):
        path=Path(path).resolve()
        require(isinstance(digest,str) and len(digest)==64 and all(c in "0123456789abcdef" for c in digest) and sha(path)==digest,"Pinned source/input SHA mismatch: "+path.name)
        require(str(path) not in pins or pins[str(path)]==digest,"Conflicting source SHA")
        pins[str(path)]=digest
        return path
    def artifact(directory,item):
        require(Path(item["file"]).name==item["file"],"Simple artifact name required")
        path=pin(directory/item["file"],item["sha256"])
        require(path.stat().st_size==item["bytes"],"Artifact byte count mismatch")
        return path
    try:
        import numpy as np
        pin(__file__,report["script_sha256"])
        base=module(args.frozen_runtime_helper,RUNTIME_SHA,"_v37_frozen_runtime",pin)
        query=module(args.query_helper,QUERY_SHA,"_v38_frozen_query",pin)
        query_delegate=module(args.frozen_query_helper,PARENT_QUERY_SHA,"_v37_frozen_attr_query",pin)
        transport=module(args.transport_helper,TRANSPORT_SHA,"_v34_frozen_typed_transport",pin)
        abi=module(args.abi_helper,ABI_SHA,"_v12_frozen_abi",pin)
        pin(args.header,HEADER_SHA);pin(args.library,LIBRARY_SHA)
        require({4,5,6,7}<=prior,"Four A76 CPUs required")
        os.sched_setaffinity(0,{4,5,6,7})
        report["cpu_affinity"]=sorted(os.sched_getaffinity(0))
        args.compile_json=args.compile_dir/"compile.json"
        compiled,recipe,evidence=query.load_bundle(args,pin)
        model=artifact(args.compile_dir,compiled["output"])
        proof=json.loads(pin(args.query_dir/"query.json",args.query_sha256).read_text())
        validate_query(proof,args.compile_sha256,compiled["output"]["sha256"],args.rewrite_sha256,args.ort_sha256,recipe["output"]["sha256"])
        rows,capture,reference=base.load_rows(args,pin,artifact)
        rows=load_candidate_gold(args,evidence,rows,base,pin,artifact)
        report.update(preprocessing=capture["preprocessing"],inputs=[r["identity"]["input_metadata"] for r in rows],
            source_frames=[r["identity"]["source_image"] for r in rows],sampling_rule=reference["sampling_rule"],model=compiled["output"],
            configuration=compiled["configuration"],abi=abi.abi_record(),source_reciprocal_graph_sha256=recipe["output"]["sha256"],
            source_parent_spatial_graph_sha256=recipe["parent"]["sha256"],reference_method="spatial_parent_and_reciprocal_candidate",
            reference_spatial_suffix_CPU_bitwise_gate_passed=True,historical_FP16_receipt_sha256=None,
            query_arithmetic_delegate_sha256=PARENT_QUERY_SHA,query_helper_sha256=QUERY_SHA)
        require(report["abi"]["context_size"]==report["abi"]["pointer_size"]==8,"Pinned 64-bit ABI required")
        api=abi.bind_api(ct.CDLL(str(args.library.resolve())))
        context=base.PTQContext(api,abi,query_delegate,proof,model,args.collect_perf_detail,report)
        fingerprints={}
        for index,(frame,role) in enumerate(base.schedule(args.collect_perf_detail)):
            row=rows[frame]
            actual,timing=context.execute(transport,row["image"],row["indices"],report)
            report["NPU_used"]=True
            finite=bool(all(np.isfinite(v).all() and np.any(v!=0) for v in actual.values()))
            record=dict(index=index,frame_index=frame,frame_id=row["identity"]["frame_id"],timing_role=role,source_image=row["identity"]["source_image"],
                input_metadata=row["identity"]["input_metadata"],indices=row["identity"]["indices"],timing=timing,forward_seconds=timing["sdk_io_seconds"],
                arrays={k:base.metadata(v) for k,v in actual.items()},finite_nonzero=finite,
                finite_counts={k:dict(nonfinite=int(np.count_nonzero(~np.isfinite(v))),all_zero=bool(np.all(v==0))) for k,v in actual.items()})
            path=output/f"call-{index:02d}-frame-{frame}.npz"
            with path.open("xb") as stream:np.savez(stream,**actual)
            record["artifact"]=dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path))
            report["calls"].append(record)
            require(finite,"Nonfinite/all-zero SDK output; raw arrays saved")
            for name,key in (("versus_original_CPU","cpu"),("versus_v33_spatial_CPU_ORT","ort"),("versus_v38_reciprocal_CPU_ORT","reciprocal_ort")):
                record[name]={k:base.comparison(v,row[key][k]) for k,v in actual.items()}
            record["repeat_raw_hash_equal"]=None if frame not in fingerprints else fingerprints[frame]==record["arrays"]
            fingerprints.setdefault(frame,record["arrays"])
            if args.collect_perf_detail:record["perf_detail"]=abi.perf_detail(api,context.context)
            print(json.dumps(dict(event="reciprocal_PTQ_encoder_call",index=index,frame=frame,seconds=record["forward_seconds"],
                strict=all(v["gate_passed"] for v in record["versus_original_CPU"].values()))),flush=True)
        require(len(report["calls"])==len(base.schedule(args.collect_perf_detail)),"Complete prescribed schedule required")
        report["actual_original_CPU_strict_gate_passed"]=all(v["gate_passed"] for r in report["calls"] for v in r["versus_original_CPU"].values())
        report["actual_reciprocal_CPU_ORT_strict_gate_passed"]=all(v["gate_passed"] for r in report["calls"] for v in r["versus_v38_reciprocal_CPU_ORT"].values())
        warm=[r["forward_seconds"] for r in report["calls"] if r["timing_role"]=="warm"]
        report["timing_summary"]=dict(cold_seconds=None if args.collect_perf_detail else report["calls"][0]["forward_seconds"],
            warm_seconds=warm,warm_mean_seconds=sum(warm)/len(warm) if warm else None,profile_timings_not_normal_benchmark=args.collect_perf_detail)
        report["status"]=SUCCESS
    except BaseException as exc:
        report.update(error=repr(exc),traceback=traceback.format_exc())
    finally:
        if context is not None:context.close()
        report["source_hashes_after"]={p:sha(p) if Path(p).is_file() else None for p in pins}
        report["sources_and_inputs_unchanged"]=report["source_hashes_after"]==pins
        if not report["sources_and_inputs_unchanged"]:report["cleanup_errors"].append("source/input rehash changed")
        try:
            os.sched_setaffinity(0,prior)
            report["affinity_restored"]=os.sched_getaffinity(0)==prior
            require(report["affinity_restored"],"Affinity restoration failed")
        except Exception as exc:report["cleanup_errors"].append("affinity: "+repr(exc))
        if report["cleanup_errors"]:report["status"]="failed"
        report.update(actual_driver_seconds=time.monotonic()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        (output/"run.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:report.get(k) for k in ("status","error","actual_original_CPU_strict_gate_passed","timing_summary")}),flush=True)
    return 0 if report["status"]==SUCCESS else 1


if __name__=="__main__":
    raise SystemExit(main())
