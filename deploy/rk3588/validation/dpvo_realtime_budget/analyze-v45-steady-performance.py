#!/usr/bin/env python3
"""Saved JSON timing analysis only: never imports a model, SDK or native library."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

RUNS = {
    "v41": ("78f15e36e5444dfe17a70610eb4410f0605c262eb7102bc2fee39ba5cfdf427b",
             "v41_cached_undistort_online_RKNN_research_replay"),
    "v43": ("bf058a164f27031b26929637df87f34aab533ccfe527c46728a2f0150171ccfa",
             "v43_cached_undistort_metadata_only_online_RKNN_research_replay"),
    "v44": ("27528993e2c310a7c2b3623cdd1d0a388789d010b762214a2a8553f6efa99db8",
             "v44_cached_undistort_native_FP16_online_RKNN_research_replay"),
}
AB_SHA = "1a2132b70849fe823e3d85617d588f2d1dd5626326adfa4c14c0e718d2f7faf4"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seconds(value):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0, "Invalid saved duration")
    return float(value)


def statistics(values, scale=1000):
    x = np.asarray(values, dtype=np.float64)
    require(x.ndim == 1 and len(x) > 0 and np.isfinite(x).all(), "Invalid timing sample")
    x = x * scale
    return dict(count=len(x), mean=float(x.mean()), p50=float(np.percentile(x, 50)),
                p95=float(np.percentile(x, 95)), min=float(x.min()), max=float(x.max()),
                sum=float(x.sum()), percentile_method="linear")


def one(rows, label):
    require(len(rows) == 1, "Expected exactly one " + label)
    return rows[0]


def steady_rows(run, first=35, last=191):
    """Join by explicit Update index; _neural serially correlates before Update."""
    frames = run["frames"]
    require(len(frames) == last + 1, "Original complete observation count differs")
    require([f["index"] for f in frames] == list(range(len(frames))), "Frame order differs")
    require(run["correlation"]["call_history_dropped"] == 0, "Correlation history is incomplete")
    corr_by_id = {c["index"]: c for c in run["correlation"]["calls"]}
    records = {c["index"]: c for c in run["online_update_records"]}
    require(len(corr_by_id) == len(run["correlation"]["calls"]), "Duplicate correlation index")
    require(len(records) == len(run["online_update_records"]), "Duplicate callback index")
    rows = []
    for i in range(first, last + 1):
        f = frames[i]
        result = f["result"]
        require(result["fresh_pose_valid"] is True and result["observation_id"] == i
                and result["frame_id"] == f["frame_id"]
                and result["timestamp"] == f["timestamp"] and result["updates"] == 1,
                "Steady fresh observation identity/count differs")
        u = one([u for u in run["update_calls"] if u["frame_index"] == i
                 and u["stage"] == "stream" and u["route"] == "full"], "full Update")
        b = one([b for b in run["BA_calls"] if b["frame_index"] == i
                 and b["stage"] == "stream"], "two-step BA")
        require(u["index"] == u["SDK_record_index"], "Callback/correlation serial index differs")
        c, x = corr_by_id[u["index"]], records[u["SDK_record_index"]]
        require(u["edges"] == c["edges"] == x["edges"] and x["frame_index"] == i
                and x["stage"] == "stream" and x["route"] == "full", "Full graph identity differs")
        sdk = x["SDK_execution"]
        groups = sdk["groups"]
        require(len(groups) == sdk["sequential_groups"] and sdk["full_graph_returned"]
                and not sdk["padding"] and not sdk["dropped_edges"], "Incomplete SDK graph")
        timing_keys = ("forward_seconds", "pack_seconds", "run_api_seconds",
                       "input_set_seconds", "outputs_get_seconds", "owned_copy_seconds",
                       "outputs_release_seconds")
        group_sums = {k: sum(seconds(g["timing"][k]) for g in groups) for k in timing_keys}
        for g in groups:
            require(g["outputs_released"] and g["outputs_finite"] and g["weight_domain_valid"],
                    "SDK release/finite/domain evidence differs")
        row = dict(observation_id=i, frame_id=f["frame_id"], timestamp=f["timestamp"],
                   edges=u["edges"], SDK_calls=len(groups), model_keys=[g["model_key"] for g in groups])
        for dst, src in (("decode", "decode_undistort_crop_resize_seconds"),
                         ("encoder", "encode_seconds"), ("stream", "stream_seconds"),
                         ("publication", "publication_seconds"),
                         ("admission_to_publication", "admission_to_publication_seconds")):
            row[dst] = seconds(f[src])
        row.update(correlation=seconds(c["total_seconds"]),
                   correlation_kernel=seconds(c["native_query"]["native"]["kernel_seconds"]),
                   correlation_prepare=seconds(c["preparation_seconds"]), BA=seconds(b["seconds"]),
                   Update=seconds(u["seconds"]), callback=seconds(x["callback_seconds"]),
                   full_graph=seconds(sdk["full_graph_seconds"]),
                   capture_metadata=seconds(u["capture_and_metadata_seconds"]))
        row.update({"SDK_" + k.removesuffix("_seconds"): v for k, v in group_sums.items()})
        encoder = run["encoder_metrics"]["calls"][i]
        require(encoder["index"] == i, "Encoder observation order differs")
        row["encoder_SDK_IO"] = seconds(encoder["SDK_timing"]["sdk_io_seconds"])
        row["encoder_SDK_run"] = seconds(encoder["SDK_timing"]["run_api_seconds"])
        row["stream_remainder"] = row["stream"] - sum(row[k] for k in
            ("correlation", "Update", "BA", "capture_metadata"))
        epsilon = 1e-7
        require(row["stream_remainder"] >= -epsilon, "Overlapping stream partition")
        require(row["SDK_run_api"] <= row["SDK_forward"] + epsilon
                and row["SDK_forward"] <= row["full_graph"] + epsilon
                and row["full_graph"] <= row["callback"] + epsilon
                and row["callback"] <= row["Update"] + epsilon,
                "Saved nested Update clocks are inconsistent")
        require(row["correlation_kernel"] + row["correlation_prepare"]
                <= row["correlation"] + epsilon, "Saved nested correlation clocks are inconsistent")
        require(row["encoder_SDK_run"] <= row["encoder_SDK_IO"] + epsilon
                and row["encoder_SDK_IO"] <= row["encoder"] + epsilon,
                "Saved nested encoder clocks are inconsistent")
        require(abs(row["admission_to_publication"] - sum(row[k] for k in
                    ("decode", "encoder", "stream", "publication"))) < 1e-5,
                "Saved outer clock partition differs")
        require(abs(f["pose_available_monotonic"] - f["arrival_monotonic"]
                    - row["admission_to_publication"]) < 1e-7, "Publication clock identity differs")
        rows.append(row)
    return rows


def summarize(run, rows):
    counts = {"edges", "SDK_calls"}
    keys = [k for k in rows[0] if k not in {"observation_id", "frame_id", "timestamp", "model_keys"}]
    return dict(status=run["status"], replay_elapsed_seconds=run["replay_elapsed_seconds"],
                peak_rss_mib=run["peak_rss_mib"], termination_seconds=run.get("termination_seconds"),
                fresh_observations=sum(f["result"]["fresh_pose_valid"] for f in run["frames"]),
                cohort_count=len(rows), timing_milliseconds={k: statistics([r[k] for r in rows])
                    for k in keys if k not in counts}, count_statistics={k: statistics(
                        [r[k] for r in rows], 1) for k in counts},
                model_call_histogram=dict(Counter(k for r in rows for k in r["model_keys"])), rows=rows)


def closed_AB(run):
    require(run["schema"] == "v44_native_cache_closed_flag_three_saved_state_ABBA_diagnostic"
            and len(run["calls"]) == 24 and run["A_B_all_saved_outputs_bitwise_equal"] is True
            and run["cleanup_errors"] == [], "Closed-cache AB scope differs")
    summary = []
    for obs in (34, 127, 191):
        row = dict(observation_id=obs, methods={})
        for method in ("A", "B"):
            calls = [c for c in run["calls"] if c["observation_id"] == obs and c["method"] == method]
            require(len(calls) == 4 and [c["state_method_repeat_index"] for c in calls] == list(range(4)),
                    "AB schedule differs")
            warm = [c for c in calls if not c["excluded_first_state_method_call"]]
            require(len(warm) == 3, "AB cold exclusion differs")
            row["methods"][method] = dict(callback_ms=statistics([c["callback_record"]["callback_seconds"] for c in warm]),
                full_graph_ms=statistics([c["callback_record"]["SDK_execution"]["full_graph_seconds"] for c in warm]),
                SDK_run_ms=statistics([sum(g["timing"]["run_api_seconds"] for g in
                    c["callback_record"]["SDK_execution"]["groups"]) for c in warm]))
        summary.append(row)
    return dict(scope="same-three-saved-states component AB; no new encoder/correlation/BA/trajectory",
                candidate_accepted=False, historical_SDK_strict_FP32_rejected=True,
                independent_saved_array_review_repeated=False, summary=summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in RUNS:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--closed-ab", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    paths = {k: getattr(a, k) for k in RUNS} | {"closed_AB": a.closed_ab}
    before = {k: sha(path) for k, path in paths.items()}
    all_runs, summaries = {}, {}
    identities = None
    for name, (digest, schema) in RUNS.items():
        require(before[name] == digest, "Frozen run SHA differs: " + name)
        run = json.loads(paths[name].read_text())
        require(run["schema"] == schema and run["frames_completed"] == run["frames_requested"] == 192,
                "Completed observation prefix/schema differs")
        require(run["cleanup_errors"] == [] and run["sources_and_inputs_unchanged"]
                and run["original_weights_unchanged"] and not run["candidate_accepted"]
                and not run["SDK_FP32_equivalence_validated"], "Historical scope/cleanup changed")
        rows = steady_rows(run)
        current = [(r["observation_id"], r["frame_id"], r["timestamp"], r["edges"], r["model_keys"]) for r in rows]
        if identities is None:
            identities = current
        require(current == identities, "Cross-run steady identity/shape cohort differs")
        summaries[name] = summarize(run, rows)
        all_runs[name] = run
    require(before["closed_AB"] == AB_SHA, "Frozen closed-cache AB SHA differs")
    ab = closed_AB(json.loads(a.closed_ab.read_text()))
    native = summaries["v44"]["timing_milliseconds"]
    latency, kernel = native["admission_to_publication"]["mean"], native["correlation_kernel"]["mean"]
    report = dict(schema="v45_saved_JSON_steady_performance_analysis", status="passed_saved_timing_analysis",
        script_sha256=sha(__file__), artifact_sha256=before, actual_forward_executed=False,
        cohort=dict(first=35, last=191, count=157, initialization_fresh_observation_excluded=34,
                    EOF_excluded=True, cold_context_outliers_excluded=False, capture_IO_outliers_excluded=False),
        timing_scope=dict(outer="decode+encoder+stream+publication equals admission-to-publication",
            stream="correlation+Update+BA+capture_metadata+remainder (disjoint saved spans)",
            nested="SDK_run <= SDK_forward <= full_graph <= callback <= Update <= stream",
            remainder="Includes reprojection, original context gather, topology/keyframe work, copies and reporting; not separately measured.",
            correlation="Native kernel and preparation are nested inside total; reporting/deepcopies after total belong to remainder.",
            denominator="Progress writes/health samples between publications are in replay time but outside per-frame admission latency.",
            comparison="Different runs/heat states; descriptive comparison, no controlled speedup or causal attribution.",
            v41="Whole run failed at first EOF unknown B1D3; only its completed 192-observation prefix is analyzed."),
        runs=summaries, closed_cache_AB=ab,
        budget=dict(target_ms=50, native_correlation_kernel_fraction_of_corr=kernel/native["correlation"]["mean"],
            arithmetic_zero_correlation_kernel_counterfactual_ms=latency-kernel,
            decode_plus_encoder_ms=native["decode"]["mean"]+native["encoder"]["mean"],
            summed_Update_SDK_run_ms=native["SDK_run_api"]["mean"],
            counterfactual_is_not_measured_or_a_performance_prediction=True),
        candidate_accepted=False, realtime_validated=False, default_VGGT_unchanged=True)
    after = {k: sha(path) for k, path in paths.items()}
    require(before == after, "Saved input changed during analysis")
    report["sources_unchanged"] = True
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=report["status"], output_sha256=sha(a.output),
                         steady_count=157, native_latency_mean_ms=latency,
                         native_correlation_kernel_mean_ms=kernel)))


if __name__ == "__main__":
    main()
