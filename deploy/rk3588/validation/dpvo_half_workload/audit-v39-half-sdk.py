#!/usr/bin/env python3
"""Saved-array-only half-M16 SDK evidence review; NumPy, no inference.

MIT project helper. Operator-table parser follows the frozen v26 saved audit.
Successful audit means evidence is consistent; numerical rejection stays false.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

DRIVER_SHA = "8850b1c10572ecc5e8255b1c5c195bf2a8aceb467cc7250c0ef41f1bb2932bb6"
POLICY_SHA = "cc6a31f89bdcff0b43c0d51266ad62c48d0e297346845a98d42bad5c8874ef4a"
PROBE_SHA = "bdf0feea251e478d761a7bf5c9635e46fee39cb0e9a9720b0949bb54b18e7c8a"
COMPILE_SHA = "5fb3cc31867c1bb59905e5fe0dccd77d60a1586087dd96c8f12e06ce59d0127e"
NORMAL_SHA = "13408c3aebb9dd41b38889594fe7fccc909bc1248f1172ab3b23d4152b8bafdb"
PROFILE_SHA = "c219e215ed398871eacd68725e32b3123c5f393b88c3c96a7b9669354752974e"
SHAPES = {"fmap": (1, 128, 58, 76), "context": (1, 16, 384)}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):h.update(chunk)
    return h.hexdigest()


def meta(a):
    return dict(shape=list(a.shape), dtype=str(a.dtype), bytes=int(a.nbytes),
                tensor_sha256=hashlib.sha256(a.tobytes(order="C")).hexdigest())


def errors(a, b):
    if a.dtype != np.float32 or b.dtype != np.float32 or a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Matching finite FP32 arrays required")
    x, y = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    d = x-y
    n = float(np.linalg.norm(y));e = float(np.linalg.norm(d))
    count = int(np.count_nonzero(np.abs(d) > 1e-5+1e-4*np.abs(y)))
    return dict(rtol=1e-4, atol=1e-5, gate_passed=count == 0, failed_elements=count, element_count=int(a.size),
        bitwise_equal=a.tobytes(order="C") == b.tobytes(order="C"), max_abs=float(np.max(np.abs(d))),
        relative_l2=e/n if n else 0.0 if e == 0 else None, rmse=float(np.sqrt(np.mean(d*d))),
        finite=True, actual_all_zero=bool(np.all(a == 0)))


def parse_profile(detail):
    text = detail["raw_text"]
    if detail["query_command"] != 3 or len(text.encode()) != detail["bytes"] or hashlib.sha256(text.encode()).hexdigest() != detail["raw_sha256"]:
        raise ValueError("Actual performance buffer hash/length")
    rows = []
    for line in text.splitlines():
        if not re.match(r"^\d+\s", line):continue
        p = line.split()
        if len(p) < 11 or p[3] not in ("NPU", "CPU", "GPU") or not re.fullmatch(r"\d+/\d+/\d+", p[6]):
            raise ValueError("Unrecognized actual operator row")
        workload = [v for v in p[8:] if re.fullmatch(r"\d+(?:\.\d+)?%/\d+(?:\.\d+)?%/\d+(?:\.\d+)?%", v)]
        if len(workload) != 1:raise ValueError("Actual three-core workload missing")
        rows.append(dict(index=int(p[0]), op=p[1], dtype=p[2], target=p[3], input_shapes=p[4], output_shape=p[5],
            elapsed_us=int(p[7]), name=p[-1]))
    total = re.search(r"Total Operator Elapsed Per Frame Time\(us\):\s*(\d+)", text)
    if not rows or [r["index"] for r in rows] != list(range(1, len(rows)+1)) or not total or sum(r["elapsed_us"] for r in rows) != int(total[1]):
        raise ValueError("Profile indices/total elapsed mismatch")
    return rows


def audit(args):
    checks = []
    result = dict(schema="v39_saved_half_M16_SDK_independent_review", status="failed", checks=checks,
        analysis_script_sha256=sha(__file__), scope=dict(saved_arrays_only=True, images_decoded=False,
            model_ORT_SDK_BA_run=False, board_job=False, board_library_independently_rehashed=False),
        limitations=["Three fixed inputs and short repeats; encoder component only, not new-observation VO frequency",
            "Actual profile includes internal CPU operators; this is mixed inference, not full NPU",
            "All original FP32 gates remain rejected; no trajectory quality acceptance or production adoption",
            "Copy-before-release is bound by frozen typed transport and recorded SDK lifecycle, not a second physical observation",
            "Board source/library hashes are recorded before/after evidence; host only rehashes locally supplied driver and artifacts"],
        FP64_summary_tolerance=dict(rtol=1e-10, atol=1e-14, purpose="Reduction ordering only; element gate and byte comparisons exact"))
    def check(ok, name):
        checks.append(dict(name=name, passed=bool(ok)))
        if not ok:raise ValueError(name)
    def receipt(directory, filename, digest):
        path=Path(directory)/filename
        check(sha(path)==digest, filename+" receipt SHA "+Path(directory).name)
        result.setdefault("receipts", {})[Path(directory).name]=dict(file=filename,sha256=digest)
        return json.loads(path.read_text())
    def artifact(directory, row):
        check(Path(row["file"]).name==row["file"], "Relative artifact filename")
        path=Path(directory)/row["file"]
        check(path.is_file() and not path.is_symlink() and path.stat().st_size==row["bytes"] and sha(path)==row["sha256"], path.name+" bytes/file SHA")
        with np.load(path,allow_pickle=False) as z:values={k:z[k] for k in z.files}
        check(set(values)==set(row["arrays"]), path.name+" keys")
        for k,a in values.items():check(meta(a)==row["arrays"][k], path.name+" raw metadata "+k)
        return values
    try:
        check(sha(args.driver)==DRIVER_SHA,"Frozen actual SDK driver local SHA")
        probe=receipt(args.probe_dir,"probe.json",PROBE_SHA)
        compiled=receipt(args.compile_dir,"compile.json",COMPILE_SHA)
        normal=receipt(args.normal_dir,"run.json",NORMAL_SHA)
        profile=receipt(args.profile_dir,"run.json",PROFILE_SHA)
        model=Path(args.compile_dir)/compiled["output"]["file"]
        check(sha(model)==compiled["output"]["sha256"] and model.stat().st_size==compiled["output"]["bytes"],"Compiled model bytes/SHA")
        check(compiled["released"] is True and compiled["cleanup_errors"]==[] and compiled["new_size_CPU_strict_gate_each_method"]=={"original":True,"reciprocal":True},"Successful PTQ and distinct new-size CPU gates")
        check(probe["strict_gate_each_method"]=={"original":True,"reciprocal":True} and probe["state_unchanged"] is True,"Original94 new-size teacher gates")
        gold=[]
        for i,row in enumerate(probe["calls"]):
            check(row["index"]==row["source_index"]==i,"Ordered source reference "+str(i))
            inp=artifact(args.probe_dir,row["input_artifact"])
            teacher=artifact(args.probe_dir,row["teacher_artifact"])
            original=artifact(args.probe_dir,row["original"]["artifact"])
            reciprocal=artifact(args.probe_dir,row["reciprocal"]["artifact"])
            image,ix=inp["normalized_bgr"],inp["patch_indices"]
            check(image.shape==(1,3,232,304) and image.dtype==np.float32 and np.isfinite(image).all() and ix.shape==(16,) and ix.dtype==np.int64,"New-size input contract "+str(i))
            y,x=np.divmod(ix,76)
            check(np.all((x>=1)&(x<75)&(y>=1)&(y<57)) and np.array_equal(ix,inp["centers_xy"][:,1]*76+inp["centers_xy"][:,0]),"Ordered M16 index/center binding "+str(i))
            gather=lambda a:np.ascontiguousarray(a[0,:,y,x][None])
            check(np.array_equal(gather(teacher["imap"]),teacher["context"]),"Original CPU context gather "+str(i))
            refs={"original_CPU":{k:teacher[k] for k in SHAPES},"original_ORT":{"fmap":original["fmap"],"context":gather(original["imap"])},"reciprocal_ORT":reciprocal}
            for name,values in refs.items():
                for k,shape in SHAPES.items():
                    check(values[k].shape==shape and values[k].dtype==np.float32 and np.isfinite(values[k]).all(),"Reference canonical shape "+name+" "+k)
                    check(errors(values[k],refs["original_CPU"][k])["gate_passed"],"New-size reference CPU gate "+name+" "+k)
            gold.append((row,refs))
        check(len(gold)==3,"Exactly three reference source images")
        result["runs"]={}
        saved={}
        for mode,run,directory in (("normal",normal,args.normal_dir),("profile",profile,args.profile_dir)):
            profiling=mode=="profile";count=3 if profiling else 6
            check(run["schema"]=="v39_DPVO_half_M16_encoder_SDK_diagnostic" and run["status"]=="completed_half_M16_SDK_diagnostic_not_VO_not_adopted","Completed diagnostic schema "+mode)
            check(run["script_sha256"]==DRIVER_SHA and run["compile_receipt_sha256"]==COMPILE_SHA and run["probe_sha256"]==PROBE_SHA and run["policy_sha256"]==POLICY_SHA and run["model_sha256"]==sha(model),"Source/model/teacher chain "+mode)
            check(run["source_pins"]==run["source_hashes_after"] and run["sources_and_inputs_unchanged"] is True and run["cleanup_errors"]==[] and run["affinity_restored"] is True,"Source and cleanup evidence "+mode)
            check(run["collect_perf_detail"] is profiling and run["normal_benchmark"] is not profiling and len(run["calls"])==count and run["sdk_run_api_calls"]==run["activity"]["sdk_run_api_calls"]==count,"Actual call count and separate profiling "+mode)
            check(run["candidate_accepted"] is False and run["historical_parent_FP32_gate_passed"] is False and run["historical_parent_rejection_retained"] is True and run["strict_FP32_gate_passed"] is False and run["SDK_inference_executed"] is True,"Successful execution does not change rejection "+mode)
            check(run["complete_VO"] is False and run["realtime_validated"] is False and run["GT_read"] is False and run["CPU_fallback_used"] is False,"Scope retained "+mode)
            check(run["references"]==probe["calls"],"Exact source and saved reference rows "+mode)
            c=run["context"]
            check(c["init_return_code"]==c["destroy_return_code"]==0 and c["context_destroyed"] is True and c["closed"] is True and c["completed_output_copy_release_calls"]==count and c["init_flags"]==(8 if profiling else 0) and c["core_mask"]==7,"Context init/copy/release/destroy evidence "+mode)
            check(c["input_count"]==c["output_count"]==2 and c["native_IO_contract_checked"] is True,"Exact actual IO counts "+mode)
            for a,n,shape,dtype,fmt in [(c["input_attrs"][0],"normalized_bgr",[1,232,304,3],2,1),(c["input_attrs"][1],"patch_indices",[16],8,3),(c["output_attrs"][0],"fmap",[1,128,58,76],2,0),(c["output_attrs"][1],"context",[1,16,384],2,3)]:
                check(a["name"]==n and a["shape"]==shape and a["type"]==dtype and a["format"]==fmt and a["n_elems"]==math.prod(shape) and a["size"]==math.prod(shape)*(8 if dtype==8 else 1) and a["quantization_type"]==2 and a["scale"]>0,"Actual native descriptor "+mode+" "+n)
            check(c["input_attrs"][1]["scale"]==1 and c["input_attrs"][1]["zero_point"]==0,"Neutral INT64 indices "+mode)
            summaries=[]
            for i,row in enumerate(run["calls"]):
                frame=i if profiling else 0 if i<4 else i-3
                source,refs=gold[frame]
                check(row["call_index"]==i and row["frame_index"]==frame and row["frame_id"]==source["frame_id"] and row["source_index"]==source["source_index"] and row["source_image_sha256"]==source["source_image_sha256"] and row["input_array"]==source["image_array"] and row["indices"]==source["index_array"],"Actual call source identity "+mode+str(i))
                values=artifact(directory,row["artifact"])
                check(set(values)==set(SHAPES),"Exact canonical SDK output keys")
                actual={}
                for name,ref in refs.items():
                    actual[name]={}
                    for k in SHAPES:
                        check(values[k].shape==SHAPES[k] and values[k].dtype==np.float32 and np.isfinite(values[k]).all() and np.any(values[k]!=0),"Finite nonzero SDK output "+mode+str(i)+k)
                        e=errors(values[k],ref[k]);recorded=row["comparison"][name][k]
                        check(set(e)==set(recorded),"Exact metric keys")
                        for field,v in e.items():
                            equal=math.isclose(v,recorded[field],rel_tol=1e-10,abs_tol=1e-14) if isinstance(v,float) else v==recorded[field]
                            check(equal,"Metric "+mode+str(i)+name+k+field)
                        actual[name][k]=e
                t=row["timing"]
                check(t["input_api_types"]==[0,8] and t["input_api_formats"]==[1,3] and t["input_pass_through"]==[0,0] and t["input_buffer_shapes"]==[[1,232,304,3],[16]] and t["input_buffer_bytes"]==[846336,128] and t["output_want_float"]==1 and t["output_is_prealloc"]==0,"Actual typed input/output transport evidence "+mode+str(i))
                check(t["output_bytes"]==[2256896,24576] and t["output_buffers_released"] is True and t["synchronous"] is True,"FP32 logical copied bytes and release "+mode+str(i))
                pieces=[t[k] for k in ("input_packing_seconds","input_set_seconds","run_api_seconds","outputs_get_seconds","output_copy_seconds","outputs_release_seconds")]
                check(all(v>=0 for v in pieces) and sum(pieces)<=t["sdk_io_seconds"]+1e-8,"Inclusive IO timer decomposition "+mode+str(i))
                summary=dict(call_index=i,frame_index=frame,phase=row["phase"],sdk_io_seconds=t["sdk_io_seconds"],errors=actual)
                if profiling:
                    ops=parse_profile(row["perf_detail"])
                    counts={target:sum(r["target"]==target for r in ops) for target in ("NPU","CPU","GPU")}
                    io=[r for r in ops if r["op"] in ("InputOperator","OutputOperator")]
                    cpu=[r for r in ops if r["target"]=="CPU" and r not in io]
                    summary["profile"]=dict(rows=len(ops),target_counts=counts,internal_CPU_op_counts={k:sum(r["op"]==k for r in cpu) for k in sorted({r["op"] for r in cpu})},
                        CPU_IO_rows=len(io),CPU_Div_rows=[r for r in cpu if r["op"]=="Div"],CPU_Transpose_rows=[r for r in cpu if r["op"]=="Transpose"],
                        NPU_Transpose_rows=[r for r in ops if r["target"]=="NPU" and r["op"]=="Transpose"],
                        CPU_Div_elapsed_us=sum(r["elapsed_us"] for r in cpu if r["op"]=="Div"),operator_elapsed_us=sum(r["elapsed_us"] for r in ops))
                    check(len(io)==4 and all(r["target"]=="CPU" for r in io) and counts["GPU"]==0,"Actual profile IO/targets")
                summaries.append(summary)
                if (mode,frame) in saved:check(all(np.array_equal(values[k],saved[(mode,frame)][k]) and values[k].tobytes()==saved[(mode,frame)][k].tobytes() for k in SHAPES),"Repeated normal frame0 bytes")
                saved[(mode,frame)]=values
            warm=[r["sdk_io_seconds"] for r in summaries if r["phase"]=="warm"]
            mean=sum(warm)/len(warm) if warm else None
            check(run["warm_sdk_io_seconds"]==dict(samples=len(warm),mean=mean),"Independent ordinary warm mean "+mode)
            flags={name:all(r["errors"][name][k]["gate_passed"] for r in summaries for k in SHAPES) for name in gold[0][1]}
            check(flags==run["strict_gate_each_reference"]==dict.fromkeys(gold[0][1],False),"Three reference strict failures retained "+mode)
            result["runs"][mode]=dict(calls=summaries,warm_sdk_io_seconds=dict(samples=len(warm),mean=mean),context_destroyed=True,release_calls=count,peak_rss_mib=run["peak_rss_mib"],native_input_attrs=c["input_attrs"],native_output_attrs=c["output_attrs"])
        for frame in range(3):check(all(saved[("normal",frame)][k].tobytes()==saved[("profile",frame)][k].tobytes() for k in SHAPES),"Normal/profile same-frame output bytes "+str(frame))
        check(normal["context"]["input_attrs"]==profile["context"]["input_attrs"] and normal["context"]["output_attrs"]==profile["context"]["output_attrs"],"Same native model IO in both contexts")
        result.update(status="passed_saved_evidence_consistent_numeric_rejected",SDK_calls=9,SDK_saved_NPZ=9,SDK_saved_output_arrays=18,reference_NPZ=12,release_calls=9,destroy_calls=2,strict_FP32_gate_passed=False,candidate_accepted=False,normal_profile_bytes_equal=True)
    except Exception as exc:
        result["error"]=repr(exc)
    result["check_count"]=len(checks);result["failed_checks"]=sum(not c["passed"] for c in checks)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("normal-dir","profile-dir","probe-dir","compile-dir","driver","output"):p.add_argument("--"+n,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError("New output only; frozen receipts must not be overwritten")
    out=audit(args);args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:out.get(k) for k in ("status","check_count","failed_checks","error")},ensure_ascii=False))
    return 0 if out["status"]=="passed_saved_evidence_consistent_numeric_rejected" else 1


if __name__=="__main__":raise SystemExit(main())
