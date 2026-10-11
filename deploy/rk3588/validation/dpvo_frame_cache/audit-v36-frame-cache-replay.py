#!/usr/bin/env python3
"""Saved JSON/NumPy-only exact trajectory and timing audit, not inference.

Snapshot tensor hashes are compared as recorded metadata: the corresponding
NPZ payloads are not required or claimed independently inspected. DPVO MIT.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

PARENT_SHA256 = "84b8f02e14c0695c79d5c47319ff47ec4c7a8dcf53a3b38d905e03bc1efde6a8"
WRAPPER_SHA256 = "bf34cbb2e4396f18e8160a9eab1dae2f482b3f0fda0cba8c5942e7ede0e85c7e"
WRAPPER_ADDED = ("wrapper_script_sha256", "native_correlation_implementation", "wrapper_source_pins",
                 "wrapper_sources_unchanged", "parent_raw_receipt_sha256")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def semantic(value):
    """Remove only run-specific software clock fields, keeping sensor stamps."""
    if isinstance(value, dict):
        return {k: semantic(v) for k, v in value.items() if k not in ("captured_monotonic", "process_seconds")}
    if isinstance(value, list): return [semantic(v) for v in value]
    return value


def exact_pose(first, second):
    if first is None or second is None: return first is second
    a, b = np.asarray(first, np.float32), np.asarray(second, np.float32)
    return a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all() and a.tobytes() == b.tobytes()


def validate_observation(row, source, index):
    result = row["result"]
    if not (row["index"] == result["observation_id"] == index
            and row["frame_id"] == result["frame_id"] == source["frame_id"]
            and row["timestamp"] == result["timestamp"] == source["timestamp"]
            and row["arrival_monotonic"] == result["captured_monotonic"]):
        raise ValueError("original observation/source timestamp/admission identity")
    intervals = [row[k] for k in ("decode_undistort_crop_seconds", "encode_seconds", "stream_seconds", "processing_seconds")]
    if not (np.isfinite(intervals).all() and min(intervals) >= 0
            and row["pose_available_monotonic"] >= row["arrival_monotonic"]
            and np.isclose(row["pose_available_monotonic"] - row["arrival_monotonic"], row["processing_seconds"], rtol=0, atol=1e-9)
            and np.isclose(sum(intervals[:3]), intervals[3], rtol=0, atol=1e-9)):
        raise ValueError("actual sequential per-observation timing arithmetic")
    pose, fresh = result["pose_w2c"], result["fresh_pose_valid"]
    if fresh:
        if not result["initialized"] or not result["admitted"] or np.asarray(pose).shape != (7,) or not exact_pose(pose, pose):
            raise ValueError("fresh result needs this admitted initialized finite pose")
    elif pose is not None:
        raise ValueError("nonfresh observation must not contain a reused pose")


def timing_summary(run):
    rows, before = run["frames"], run["metrics_before_termination"]
    fresh = [v for v in rows if v["result"]["fresh_pose_valid"]]
    online_updates = before["update_calls"] + before["motion_probe_calls"]
    online_BA = before["BA_calls"]
    def total(items, stop=None): return float(sum(v["seconds"] for v in items[:stop]))
    corr = run["correlation"]
    def correlation_totals(stop):
        calls = corr["calls"][:stop]
        if "preparations" in corr:
            prepare = sum(v["preparation_seconds"] for v in calls)
        else:
            generations = {v["generation"] for v in calls}
            prepare = sum(v["preparation"]["total_seconds"] for v in corr["generations"] if v["index"] in generations)
        return dict(total_seconds=float(sum(v["total_seconds"] for v in calls)),
                    native_query_seconds=float(sum(v["native_query"]["total_seconds"] for v in calls)),
                    preparation_seconds=float(prepare),
                    native_kernel_seconds=float(sum(v["native_query"]["native"]["kernel_seconds"] for v in calls)))
    per_fresh = [v["processing_seconds"] for v in fresh]
    return dict(replay_wall_seconds=run["replay_elapsed_seconds"],
        source_sensor_span_seconds=rows[-1]["timestamp"]-rows[0]["timestamp"],
        first_fresh_from_replay_start_seconds=fresh[0]["pose_available_monotonic"]-run["replay_begin_monotonic"] if fresh else None,
        fresh_count=len(fresh), first_fresh_observation=fresh[0]["index"] if fresh else None,
        last_fresh_observation=fresh[-1]["index"] if fresh else None,
        fresh_processing_p50_seconds=float(np.percentile(per_fresh,50)) if fresh else None,
        fresh_processing_p95_seconds=float(np.percentile(per_fresh,95)) if fresh else None,
        online=dict(neural_calls=online_updates, BA_calls=online_BA,
            decode_seconds=float(sum(v["decode_undistort_crop_seconds"] for v in rows)),
            encode_outer_seconds=float(sum(v["encode_seconds"] for v in rows)),
            stream_seconds=float(sum(v["stream_seconds"] for v in rows)),
            original_Update_seconds=total(run["update_calls"],online_updates),
            BA_seconds=total(run["geometry_BA_calls"],online_BA),
            correlation=correlation_totals(online_updates)),
        including_offline_EOF=dict(original_Update_seconds=total(run["update_calls"]),
            BA_seconds=total(run["geometry_BA_calls"]),correlation=correlation_totals(None)),
        offline_termination_seconds=run["termination_seconds"],
        peak_RSS_MiB=run["peak_rss_mib"],
        timing_scope="Unpaced software file-read admission to pose availability; first three include saved FP32 validation; no capture instrumentation; report/print flush outside per-frame clock, inside replay wall. Nested SDK/native timings must not be added twice.")


def audit(run_dir, baseline_dir, *, baseline_sha256):
    folder, oldfolder = Path(run_dir), Path(baseline_dir)
    result = dict(schema="v36_frame_cache_replay_independent_JSON_audit", status="failed", checks=[],
        auditor_sha256=sha(__file__), model_native_ORT_BA_or_board_executed=False,
        snapshot_payloads_independently_read=False, trajectory_accuracy_or_20Hz_validated=False)
    def check(value, label):
        result["checks"].append(dict(name=label,passed=bool(value)))
        if not value: raise ValueError(label)
    try:
        paths = {"run":folder/"run.json", "parent":folder/"parent-run.json", "wrapper":folder/"wrapper.json",
                 "manifest":folder/"input-manifest.json", "baseline":oldfolder/"run.json", "baseline_manifest":oldfolder/"input-manifest.json"}
        check(sha(paths["baseline"]) == baseline_sha256, "externally pinned v35 baseline receipt")
        values = {k:json.loads(p.read_text()) for k,p in paths.items()}
        result["input_sha256"] = {k:sha(p) for k,p in paths.items()}
        new, old, parent, wrapper = (values[k] for k in ("run","baseline","parent","wrapper"))
        check(wrapper["status"]=="completed_full_replay_with_per_frame_cache" and wrapper["script_sha256"]==WRAPPER_SHA256
              and wrapper["parent_driver_sha256"]==PARENT_SHA256, "completed frozen cache wrapper and parent identities")
        check(wrapper["parent_raw_receipt_sha256"] == sha(paths["parent"])
              and wrapper["replay_receipt_sha256"] == sha(paths["run"]), "exact wrapper raw/final receipt SHA chain")
        check(wrapper["parent_constructor_restored"] and wrapper["sources_unchanged"]
              and wrapper["source_pins"] == wrapper["source_hashes_after"], "candidate finally restore and source chain unchanged")
        check(new["schema"]=="v36_DPVO_FAST_NPU_encoder_per_frame_native_cache_diagnostic_replay"
              and new["status"]=="completed_unpaced_FAST_incremental_correlation_diagnostic_not_realtime", "v36 real scope and terminal identity")
        normalized = dict(new)
        for k in WRAPPER_ADDED: normalized.pop(k)
        normalized.update(schema=parent["schema"],status=parent["status"])
        check(normalized==parent, "wrapper changes only declared provenance/schema labels")
        check(new["wrapper_script_sha256"]==wrapper["script_sha256"]
              and new["wrapper_source_pins"]==wrapper["source_pins"]
              and new["parent_raw_receipt_sha256"]==sha(paths["parent"]), "final run bound to exact wrapper and unedited parent body")
        check(paths["manifest"].read_bytes()==paths["baseline_manifest"].read_bytes(), "all original source images/sensor timestamps same frozen manifest")
        manifest=values["manifest"]["frames"]
        for label,run in (("cache",new),("v35",old)):
            check(run["script_sha256"]==PARENT_SHA256 and run["cleanup_errors"]==[]
                  and run["source_pins"]==run["source_hashes_after"] and run["sources_and_inputs_unchanged"],label+" original source/inputs and cleanup")
            check(run["original_weights_unchanged"] and run["original_weights_before"]==run["original_weights_after"]
                  and len(run["original_weights_before"])==94,label+" original 94 weight metadata stable")
            check(run["historical_encoder_FP32_gate_passed"] is False and run["first_three_FP32_gate_passed"] is False
                  and not run["candidate_accepted"] and run["diagnostic_allow_fp32_rejected"]
                  and not run["FP32_thresholds_relaxed"] and not run["output_clipping"] and not run["CPU_encoder_fallback"],label+" original FP32 rejection retained without clipping/fallback")
            check(not run["physical_camera"] and run["unpaced_file_replay"] and not run["realtime_validated"]
                  and not run["GT_or_sensor_depth_input"] and not run["ORB_used"] and not run["added_LK_PnP_frontend"],label+" experimental source and acceptance scope")
            check(run["input_manifest_sha256"]==sha(paths["manifest"])
                  and run["frames_completed"]==run["frames_requested"]==len(run["frames"])==len(manifest),label+" all requested observations retained")
            for i,row in enumerate(run["frames"]):
                validate_observation(row,manifest[i],i)
                check(True,label+" source/pose/timing observation "+str(i))
                if i:
                    check(row["timestamp"]>run["frames"][i-1]["timestamp"]
                          and row["arrival_monotonic"]>=run["frames"][i-1]["pose_available_monotonic"],label+" serial fresh capture order "+str(i))
            counts=run["stream_metrics"];before=run["metrics_before_termination"]
            check(sum(r["result"]["fresh_pose_valid"] for r in run["frames"])==before["fresh_poses"]==counts["fresh_poses"],label+" fresh count excludes EOF")
            check(run["termination"]["fresh_pose_valid"] is False and run["termination"]["online_publication"] is False
                  and run["termination"]["reconstructed_poses_counted_as_fresh"] is False
                  and run["termination"]["offline_updates"]==counts["offline_updates"]==12,label+" EOF reconstructed trajectory is separate offline work")
            check(counts["update_calls"]+counts["motion_probe_calls"]==len(run["update_calls"])
                  and counts["BA_calls"]==len(run["geometry_BA_calls"])
                  and counts["BA_iterations"]==2*len(run["geometry_BA_calls"]),label+" original neural/BA/motion accounting")
            enc=run["encoder_metrics"]
            check(enc["completed_frames"]==enc["sdk_run_api_calls"]==enc["successful_sdk_frames"]==len(manifest)
                  and enc["contexts_created"]==1 and enc["context"]["context_destroyed"]
                  and len(enc["calls"])==len(manifest),label+" actual encoder SDK calls and final destroy")
            for i,c in enumerate(enc["calls"]):
                check(c["index"]==i and c["completed"] and c["SDK_timing"]["synchronous"]
                      and c["SDK_timing"]["output_buffers_released"] and not c["native_log"]["RKNN_error_detected"],label+" synchronous copied/released SDK call "+str(i))
        for key in ("config","FAST_config_source","original_weights_before","original_weights_after","random_generator_before","random_generator_after","random_depth_draw_calls","online_lifecycle_source_sha256","stream_metrics","metrics_before_termination"):
            check(new[key]==old[key],"exact original config/weights/RNG/lifecycle "+key)
        for i,(a,b) in enumerate(zip(new["frames"],old["frames"])):
            check(semantic(a["result"])==semantic(b["result"]) and exact_pose(a["result"]["pose_w2c"],b["result"]["pose_w2c"]),
                  "exact original observation pose/lifecycle/BA diagnostics "+str(i))
        check(semantic(new["termination"])==semantic(old["termination"])
              and exact_pose(new["termination"]["pose_w2c"],old["termination"]["pose_w2c"])
              and exact_pose(new["termination"]["pose_c2w"],old["termination"]["pose_c2w"]), "exact separate EOF reconstructed trajectory and diagnostics")
        for name in ("snapshot_before_termination","final_snapshot"):
            check(semantic(new[name]["metadata"])==semantic(old[name]["metadata"])
                  and new[name]["artifact"]["arrays"]==old[name]["artifact"]["arrays"],"recorded complete state tensor hashes same "+name)
        for i in range(min(3,len(manifest))):
            check(new["frames"][i]["first_three_dense_encoder_artifact"]==old["frames"][i]["first_three_dense_encoder_artifact"],"recorded original dense feature hashes same "+str(i))
        cache=new["correlation"];calls=cache["calls"]
        check(cache["attempted_calls"]==cache["completed_calls"]==len(calls)==len(new["update_calls"])
              and not cache["failed_calls"] and not cache["call_history_dropped"] and not cache["preparation_history_dropped"], "complete successful native cache histories")
        check(cache["closed"] and cache["current_cached_frames"]==cache["current_context_bytes"]==0
              and cache["destroyed_frames"]==cache["prepared_frames"]==len(cache["preparations"]), "all owned per-frame native contexts destroyed")
        check(cache["max_frames"]==64 and cache["max_bytes"]==536870912 and cache["peak_context_bytes"]<=cache["max_bytes"]
              and cache["feature_input_arrays_strongly_retained"] and cache["native_feature_snapshots_owned"]
              and not cache["historical_features_hashed_per_query"], "bounded stable feature ownership without historical rehash")
        for i,c in enumerate(calls):
            check(c["index"]==i and c["edges"]==new["update_calls"][i]["edges"]==old["update_calls"][i]["edges"]
                  and c["edge_order_preserved"] and c["no_historical_feature_concatenation"]
                  and c["native_query"]["output_finite_validated_by_native"]
                  and c["native_query"]["native"]["output_elements"]==c["edges"]*882, "ordered real native query shape "+str(i))
        for name,path in (("total_seconds",("total_seconds",)),("preparation_seconds",("preparation_seconds",)),
                          ("query_seconds",("native_query","total_seconds")),("kernel_seconds",("native_query","native","kernel_seconds"))):
            total=0.
            for c in calls:
                value=c
                for key in path:value=value[key]
                total+=value
            check(np.isclose(total,cache["timing_sums"][name],rtol=0,atol=1e-8),"independent native timing sum "+name)
        check(sum(c["prepare_misses"] for c in calls)==cache["prepared_frames"]
              and sum(c["cache_hits"] for c in calls)==cache["cache_hits"],"native prepare/hit counter sums")
        result.update(status="passed_saved_JSON_exact_baseline_reproduction_not_accuracy_or_realtime",
            cache_timing=timing_summary(new),v35_timing=timing_summary(old),
            cache_ownership={k:cache[k] for k in ("prepared_frames","cache_hits","evicted_frames","destroyed_frames","peak_context_bytes","current_context_bytes","closed")},
            all_192_or_actual_source_result_numeric_fields_exact=True,
            offline_reconstructed_trajectory_exact=True,unchanged_recorded_state_tensor_hashes=True,
            comparison_is_same_process_controlled_AB=False,
            limitations=["Independent JSON pose values and clocks; snapshot/dense tensor payloads not provided here, only recorded hashes compared", "No GT or full-original VGGT accuracy comparison", "FP16 encoder remains rejected at original FP32 gate", "Separate runs/thermal histories; numerical reproduction does not establish causal performance gain", "Unpaced software admission, no camera queue or 20Hz/100ms validation"])
    except Exception as e:result["error"]=repr(e)
    result["check_count"]=len(result["checks"])
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",type=Path,required=True);p.add_argument("--baseline-dir",type=Path,required=True)
    p.add_argument("--baseline-sha256",required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv)
    report=audit(a.run_dir,a.baseline_dir,baseline_sha256=a.baseline_sha256)
    with a.output.open("x") as f:f.write(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:report.get(k) for k in ("status","check_count","error")}))
    return 0 if report["status"].startswith("passed_") else 1


if __name__=="__main__":raise SystemExit(main())
