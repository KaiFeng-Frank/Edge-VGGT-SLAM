#!/usr/bin/env python3
"""Standalone saved normal/profile output audit; stdlib + NumPy, no model or SDK."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROLES = ("nonquant_fp16", "ptq_w8a8")
SCHEDULE = (ROLES, ROLES[::-1], ROLES, ROLES[::-1])
INPUT_SHAPE, OUTPUT_SHAPE = [2,437,1024], [1,874,1024]
ELEMENTS, FP32_BYTES = 894976, 3579904
RTOL, ATOL = 1e-4, 1e-5
CAPTURE_SHA = "bf39dfa65526a4135cbd2d7ee8b927ce2c4a40da75aebf64652ed2cd87a7e7fc"
CAPTURE_SOURCE_SHA = "7a2279927a65e3740da933c1192c335d15f4f1803440ed84a27a2aebc8f02613"
CAPTURE_CONTRACTS_SHA = "6ce1f87f1683e8082a4d567718acb853101e5ee6bbaeef1ced02993a8e3133c8"
CHECKPOINT_SHA = "d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0"
PAIR_GRAPH_SHA = "2cf1f4de19c91b42b5bf685ab5e949bd98b08a64fc2e78b7f0c8f07992a009fc"
RUNNER_SHA = "33fc5deb749c8b3ceaa47271e389f5fbbfc2dba6db93aa19887b2fc41e0e7882"
CONTRACTS_SHA = "44c0a6a1dcca66ee09b2f15bb0e2f0ddc7b7eb3f6cb44fdc3b576e9f75463bc2"


def require(value, message):
    if not value:
        raise ValueError(message)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def pin(path, expected, pins):
    path = Path(path).resolve(strict=True)
    require(sha(path) == expected, f"Runtime saved-data SHA mismatch: {path}")
    pins[str(path)] = expected
    return path

def numeric_same(actual, declared, name):
    if actual is None:
        require(declared is None, f"Metric {name} must be null")
    elif type(actual) in (bool, int):
        require(type(declared) is type(actual) and declared == actual, f"Metric {name} mismatch")
    elif isinstance(actual, str):
        require(declared == actual, f"Metric {name} mismatch")
    else:
        require(type(declared) in (int, float) and math.isfinite(declared)
                and math.isclose(actual, declared, rel_tol=2e-11, abs_tol=1e-12), f"Metric {name} mismatch")

def original_metrics(actual, reference):
    import numpy as np
    require(actual.dtype == reference.dtype == np.float32 and list(actual.shape) == list(reference.shape) == OUTPUT_SHAPE
            and np.isfinite(actual).all() and np.isfinite(reference).all(), "Finite exact original FP32 shapes required")
    a, r = actual.astype(np.float64), reference.astype(np.float64)
    error = a - r
    absolute = np.abs(error)
    tolerance = ATOL + RTOL * np.abs(r)
    reference_norm, actual_norm = float(np.linalg.norm(r)), float(np.linalg.norm(a))
    return {"finite": True, "array_equal": bool(np.array_equal(actual, reference)),
            "max_abs_error": float(absolute.max()), "rmse": float(np.sqrt(np.mean(error * error))),
            "relative_l2": float(np.linalg.norm(error) / reference_norm) if reference_norm else None,
            "cosine_similarity": float(np.sum(a * r) / (reference_norm * actual_norm)) if reference_norm and actual_norm else None,
            "rtol": RTOL, "atol": ATOL, "strict_FP32_gate_passed": bool(np.all(absolute <= tolerance)),
            "outside_tolerance_elements": int(np.count_nonzero(absolute > tolerance)),
            "max_tolerance_ratio": float(np.max(absolute / tolerance)),
            "output_raw_sha256": hashlib.sha256(memoryview(actual).cast("B")).hexdigest()}

def descriptor(record, direction):
    """Independent enum/shape arithmetic; no native-stride access or conversions."""
    shape = INPUT_SHAPE if direction == "input" else OUTPUT_SHAPE
    require(direction in ("input", "output"), "Unsupported direction")
    require(record["name"] == direction and record["index"] == 0 and record["shape"] == shape
            and record["n_elems"] == ELEMENTS and record["format"] == 3 and record["pass_through"] == 0,
            "Actual SDK logical descriptor changed")
    for key in ("index", "n_elems", "size", "format", "type", "quantization_type", "zero_point", "pass_through"):
        require(type(record[key]) is int, "Actual descriptor integer field required")
    dtype, qnt, zp, scale = (record[key] for key in ("type", "quantization_type", "zero_point", "scale"))
    require(type(scale) in (int, float) and math.isfinite(scale), "Finite actual descriptor scale required")
    if dtype in (0, 1):
        require(qnt == 0 or (dtype == 1 and qnt == 2 and zp == 0 and scale == 1),
                "Floating descriptor requires NONE or exact FP16 identity affine")
        item_bytes, label = (4, "FP32") if dtype == 0 else (2, "FP16")
    elif dtype in (2, 3):
        low, high = (-128, 127) if dtype == 2 else (0, 255)
        require(qnt == 2 and scale > 0 and low <= zp <= high, "Integer descriptor requires valid positive affine scale/zero point")
        item_bytes, label = 1, "INT8" if dtype == 2 else "UINT8"
    else:
        raise ValueError("Unsupported actual logical SDK dtype")
    require(record["size"] == ELEMENTS * item_bytes, "Actual SDK size/type/element arithmetic mismatch")
    return {"direction": direction, "logical_shape": shape, "logical_dtype": label,
            "logical_bytes": ELEMENTS * item_bytes, "quantization_type": qnt, "zero_point": zp, "scale": scale,
            "API_FP32_bytes": FP32_BYTES, "SDK_boundary_conversion_included_in_quality_error": True,
            "native_stride_memory_used_by_audit": False,
            "all_internal_ops_INT8_or_NPU_proven": False}

def audit_sdk_interval(row):
    require(row["status"] == "completed" and row["synchronous"] is True
            and row["input_api_type"] == "FP32" and row["input_api_type_enum"] == 0
            and row["input_pass_through"] == 0 and row["input_format"] == 3
            and row["input_shape"] == INPUT_SHAPE and row["input_bytes"] == FP32_BYTES
            and row["output_shape"] == OUTPUT_SHAPE and row["expected_FP32_output_bytes"] == FP32_BYTES
            and row["want_float"] == 1 and row["is_prealloc"] == 0
            and row["manual_quantization_or_dequantization"] is False
            and row["output_bytes"] == FP32_BYTES and row["output_owned_copy_before_release"] is True
            and row["output_buffer_released"] is True and row["npu_run_api_called"] is True,
            "Actual SDK submission/get/copy/release contract failed")
    require(row["runtime_frame_id"] == row["output_frame_id"] and row["SDK_output_pointer"] > 0
            and row["owned_output_pointer"] > 0 and row["owned_output_pointer"] != row["SDK_output_pointer"],
            "Synchronous SDK identity or output ownership failed")
    require(all(type(row[key]) is int and row[key] == 0 for key in
                ("inputs_set_code", "run_code", "outputs_get_code", "outputs_release_code")), "Actual SDK API success codes required")
    begin, end = row["io_begin_monotonic_ns"], row["io_end_monotonic_ns"]
    require(type(begin) is type(end) is int and end > begin, "Actual SDK monotonic interval required")
    stages = {key: row[key] for key in ("input_set_seconds", "run_api_seconds", "outputs_get_seconds",
              "output_validation_and_copy_seconds", "pre_release_bookkeeping_seconds", "outputs_release_seconds")}
    require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in stages.values()), "SDK stages must be finite and nonnegative")
    numeric_same((end - begin) / 1e9, row["io_total_seconds"], "SDK monotonic interval")
    numeric_same(sum(stages.values()), row["io_total_seconds"], "SDK stage sum")
    return stages

def load_array(directory, item, shape, pins):
    import numpy as np
    relative = Path(item["file"])
    require(not relative.is_absolute() and ".." not in relative.parts
            and not any(c.isspace() for c in str(relative)), "Saved output path escaped its directory")
    path = (directory / relative).resolve(strict=True)
    require(path.is_relative_to(directory.resolve()) and path.is_file()
            and path.stat().st_size == item["bytes"] and item["bytes"] < 16*1024*1024,
            "Saved output path/size changed")
    pin(path,item["sha256"],pins)
    value = np.load(path,allow_pickle=False)
    require(list(value.shape)==shape and value.dtype==np.float32 and value.flags.c_contiguous
            and np.isfinite(value).all() and item["shape"]==shape and item["dtype"]=="float32"
            and value.nbytes==item["tensor_bytes"]==FP32_BYTES
            and hashlib.sha256(memoryview(value).cast("B")).hexdigest()==item["raw_sha256"],
            "Saved output dtype/shape/layout/finiteness/raw SHA changed")
    return value


def audit_cohort(directory, expected_sha, normal, evaluations, references, pins):
    report_path=pin(directory/"run.json",expected_sha,pins)
    run=json.loads(report_path.read_text())
    require(run["schema"]=="v49_original_VGGT_pair_precision_SDK_AB"
            and run["status"].startswith("completed_") and run["script_sha256"]==RUNNER_SHA
            and run["contracts_sha256"]==CONTRACTS_SHA
            and run["normal_benchmark"] is normal and run["perf_detail_collection_enabled"] is (not normal)
            and run["capture_manifest_sha256"]==CAPTURE_SHA and run["capture_script_sha256"]==CAPTURE_SOURCE_SHA
            and run["capture_contracts_sha256"]==CAPTURE_CONTRACTS_SHA
            and run["original_checkpoint"]["sha256"]==CHECKPOINT_SHA
            and run["pair_graph_sha256"]==PAIR_GRAPH_SHA and run["original_parameter_tensors"]==36,
            "Runtime scope/source/capture lineage changed")
    require(run["candidate_accepted"] is False and run["full_model_run"] is False
            and run["real_time_validated"] is False and run["whole_model_quality_validated"] is False
            and run["contexts_live_together"] is True and run["unmeasured_warmup_calls"]==0
            and run["successful_synchronous_call_count"]==run["npu_run_api_call_count"]
            ==run["completed_comparison_call_count"]==16 and len(run["calls"])==len(run["sdk_calls"])==16
            and run["cleanup_errors"]==[] and run["affinity_restored"] is True
            and run["environment_restored"] is True and run["source_and_inputs_unchanged"] is True
            and run["source_and_input_hashes"]==run["source_and_input_hashes_after"]
            and run["rtol"]==RTOL and run["atol"]==ATOL, "Runtime integrity/gate/call scope changed")
    require(run["evaluation_sample_ids"]==[s["sample_id"] for s in evaluations], "Different heldout split")
    descriptions={}
    for role in ROLES:
        context=run["context_records"][role]
        require(context["init_flags"]==(0 if normal else 8) and context["context_destroyed"] is True
                and context["destroy_code"]==0 and context["tensor_attributes_validated"] is True
                and context["core_mask"]==7 and context["sdk_version"].startswith("2.3.2")
                and context["driver_version"]=="0.7.2", "Context descriptor/lifecycle changed")
        descriptions[role]={d:descriptor(context[d+"_attr"],d) for d in ("input","output")}
    calls,arrays,counts=[],{},{role:0 for role in ROLES}
    expected_schedule=[]
    for sample_index,sample in enumerate(evaluations):
        sid=sample["sample_id"]
        for round_index,order in enumerate(SCHEDULE if sample_index==0 else tuple(v[::-1] for v in SCHEDULE)):
            expected_schedule.append({"sample_id":sid,"round_index":round_index,"order":list(order)})
            for role in order:
                index=len(calls);row,sdk=run["calls"][index],run["sdk_calls"][index]
                for item in (row,sdk):
                    require(item["sdk_call_index"]==index and item["sample_id"]==sid and item["role"]==role
                            and item["round_index"]==item["sample_call_index"]==round_index
                            and item["context_call_index"]==counts[role]
                            and item["first_in_context"] is (counts[role]==0), "AB/BA SDK identity changed")
                require(row["first_for_sample"] is (round_index==0), "Sample-first denominator changed")
                stages=audit_sdk_interval(sdk)
                numeric_same(sdk["io_total_seconds"],row["io_total_seconds"],"SDK/call interval")
                require(sdk["input_raw_sha256_before"]==sdk["input_raw_sha256_after"]
                        ==sample["artifacts"]["input.npy"]["raw_sha256"], "Original SDK input changed")
                value=load_array(directory,row["output"],OUTPUT_SHAPE,pins)
                metric=original_metrics(value,references[sid])
                for key,val in metric.items():
                    numeric_same(val,row["original_FP32_comparison"][key],key)
                arrays[(sid,round_index,role)]=value
                calls.append({"sdk_call_index":index,"sample_id":sid,"round_index":round_index,"role":role,
                              "io_seconds":row["io_total_seconds"],"original_metrics":metric})
                if not normal:
                    detail=row["separate_profile_detail"]
                    pin(directory/detail["file"],detail["saved_text_sha256"],pins)
                counts[role]+=1
    require(run["schedule"]==expected_schedule,"Run schedule changed")
    summary={}
    for role in ROLES:
        rows=[r for r in calls if r["role"]==role and r["round_index"]>0]
        require(len(rows)==6,"Exactly six subsequent calls per implementation")
        mean=sum(r["io_seconds"] for r in rows)/6
        declared=run["summaries"]["implementations"][role]
        require(declared["mean_denominator"]==6
                and declared["included_sdk_call_indices"]==[r["sdk_call_index"] for r in rows],"Mean denominator changed")
        numeric_same(mean,declared["two_input_subsequent_io_mean_seconds"],"Six-call mean")
        passes=all(r["original_metrics"]["strict_FP32_gate_passed"] for r in calls if r["role"]==role)
        require(declared["strict_original_FP32_gate_passed"] is passes,"Aggregate FP32 gate changed")
        summary[role]={"mean_io_seconds":mean,"subsequent_denominator":6,
                       "strict_original_FP32_gate_passed":passes,
                       "first_metrics_by_sample":{s["sample_id"]:next(r["original_metrics"] for r in calls
                            if r["role"]==role and r["sample_id"]==s["sample_id"] and r["round_index"]==0)
                            for s in evaluations}}
    return run,summary,arrays,descriptions


def main():
    import numpy as np
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("normal-dir","profile-dir","capture-dir","runtime-source-dir","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--normal-sha256",required=True)
    parser.add_argument("--profile-sha256",required=True)
    args=parser.parse_args()
    require(not args.output.exists() and not any(args.output.resolve().is_relative_to(d.resolve())
            for d in (args.normal_dir,args.profile_dir,args.capture_dir)),"New audit receipt outside source data required")
    pins={}
    pin(args.runtime_source_dir/"run-v49-original-pair-precision-board.py",RUNNER_SHA,pins)
    pin(args.runtime_source_dir/"v49_quantized_pair_runtime_contracts.py",CONTRACTS_SHA,pins)
    pin(args.capture_dir/"capture.json",CAPTURE_SHA,pins)
    capture=json.loads((args.capture_dir/"capture.json").read_text())
    require(capture["script_sha256"]==CAPTURE_SOURCE_SHA and capture["contracts_sha256"]==CAPTURE_CONTRACTS_SHA
            and capture["checkpoint"]["sha256"]==CHECKPOINT_SHA and capture["status"]=="completed_prefix_capture",
            "Exact original prefix capture provenance required")
    evaluations=[s for s in capture["samples"] if s["split"]=="evaluation"]
    require([s["sample_id"] for s in evaluations]==["evaluation_00","evaluation_01"],"Exact two heldout inputs required")
    references={}
    for s in evaluations:
        load_array(args.capture_dir,s["artifacts"]["input.npy"],INPUT_SHAPE,pins)
        references[s["sample_id"]]=load_array(args.capture_dir,s["artifacts"]["output.npy"],OUTPUT_SHAPE,pins)
    normal,nmean,narray,ndesc=audit_cohort(args.normal_dir,args.normal_sha256,True,evaluations,references,pins)
    profile,pmean,parray,pdesc=audit_cohort(args.profile_dir,args.profile_sha256,False,evaluations,references,pins)
    require(ndesc==pdesc,"Profile/normal logical descriptors differ")
    same=[]
    for key,n in narray.items():
        p=parray[key]
        require(np.array_equal(n,p) and hashlib.sha256(memoryview(n).cast("B")).hexdigest()
                ==hashlib.sha256(memoryview(p).cast("B")).hexdigest(),"Profile output differs bitwise from normal")
        same.append({"sample_id":key[0],"round_index":key[1],"role":key[2],"bitwise_equal":True,
                     "shared_raw_sha256":hashlib.sha256(memoryview(n).cast("B")).hexdigest()})
    for path,digest in pins.items():
        require(sha(path)==digest,"Saved data/source changed during audit")
    result={"schema":"v49_standalone_saved_normal_profile_numerical_audit","passed":True,
            "auditor_sha256":sha(__file__),"normal_run_sha256":args.normal_sha256,"profile_run_sha256":args.profile_sha256,
            "original_capture_manifest_sha256":CAPTURE_SHA,"saved_runtime_output_arrays_read":32,
            "saved_original_input_reference_arrays_read":4,"profile_text_files_SHA_checked":16,
            "profile_vs_normal_bitwise_equal_calls":16,"profile_normal_descriptor_equal":True,
            "fixed_original_FP32_rtol":RTOL,"fixed_original_FP32_atol":ATOL,
            "normal_IO_summaries":nmean,"profile_IO_summaries":pmean,
            "profile_IO_used_for_normal_speed_comparison":False,"original_model_only":True,"candidate_accepted":False,
            "all_read_file_bytes_unchanged":True,"actual_model_ORT_SDK_compile_calibration_or_image_decode_by_auditor":False,
            "full_model_or_SLAM_quality_or_realtime_validated":False,"bitwise_checks":same,"read_file_SHA_pins":pins,
            "limits":["All SDK outputs still fail the same original Torch FP32 element-wise gate; no tolerance change or adoption.",
                      "Repeated two heldout saved inputs, original first frame/global component only; no fresh poses or full model.",
                      "Profile timing belongs to separate fresh profiling contexts and is not a normal speed measurement.",
                      "SHA-pinned operator texts are not parsed here; internal precision/placement requires separate operator audit."]}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"passed":True,"profile_normal_bitwise_equal_calls":16,"receipt_sha256":sha(args.output)}))

if __name__=="__main__":
    main()
