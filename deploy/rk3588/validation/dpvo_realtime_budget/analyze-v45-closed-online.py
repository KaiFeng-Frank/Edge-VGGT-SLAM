#!/usr/bin/env python3
"""Add the completed closed-cache replay to the frozen saved-clock analysis."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

V45_SHA = "213eb55d28325cdc8ddbc18fb561a62da3c055482d2b9915c293d4c5154f7776"
BASE_REVIEW_SHA = "01b99a93578c3bb4910d5323f92d6d88f20b3c18a9777995e883cec22780be16"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analyzer", type=Path, default=Path(__file__).with_name("analyze-v45-steady-performance.py"))
    p.add_argument("--base-review", type=Path, required=True)
    p.add_argument("--v45", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if sha(a.base_review) != BASE_REVIEW_SHA or sha(a.v45) != V45_SHA:
        raise ValueError("Frozen saved evidence SHA differs")
    base = json.loads(a.base_review.read_text())
    if sha(a.analyzer) != base["script_sha256"]:
        raise ValueError("Frozen original timing analyzer differs")
    spec = importlib.util.spec_from_file_location("_v45_saved_clock_math", a.analyzer)
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)
    before = {str(p): sha(p) for p in (a.analyzer, a.base_review, a.v45)}
    run = json.loads(a.v45.read_text())
    H.require(run["schema"] == "v45_cached_undistort_native_FP16_closed_flag_online_RKNN_research_replay"
        and run["status"] == "completed_unpaced_half_online_RKNN_research_not_realtime"
        and run["frames_completed"] == run["frames_requested"] == 192
        and run["closed_flag_optimization_added"] is True, "New completed experiment contract differs")
    H.require(run["cleanup_errors"] == [] and run["sources_and_inputs_unchanged"]
        and run["original_weights_unchanged"] and not run["candidate_accepted"]
        and not run["SDK_FP32_equivalence_validated"] and not run["FP32_thresholds_relaxed"],
        "Historical rejection/cleanup boundary changed")
    rows = H.steady_rows(run)
    oldrows = base["runs"]["v44"]["rows"]
    for old, new in zip(oldrows, rows):
        for key in ("observation_id", "frame_id", "timestamp", "edges", "model_keys"):
            H.require(old[key] == new[key], "v44/v45 cohort identity differs")
    summary = H.summarize(run, rows)
    timing = summary["timing_milliseconds"]
    result = dict(schema="v45_closed_online_saved_JSON_performance_extension",
        status="passed_saved_timing_analysis", script_sha256=sha(__file__),
        base_analyzer_sha256=sha(a.analyzer), base_review_sha256=BASE_REVIEW_SHA,
        new_run_sha256=V45_SHA, actual_forward_executed=False, cohort=base["cohort"],
        all_previous_measurements_reused_from_frozen_review=True,
        comparison="Distinct actual replays and thermal states, descriptive comparison only; not controlled causality.",
        prior_v44={k: v for k, v in base["runs"]["v44"].items() if k != "rows"},
        completed_v45=summary, full_replay_fresh_Hz=summary["fresh_observations"] / summary["replay_elapsed_seconds"],
        budget=dict(target_ms=50,
            correlation_kernel_fraction_of_correlation=timing["correlation_kernel"]["mean"] / timing["correlation"]["mean"],
            arithmetic_zero_correlation_kernel_counterfactual_ms=timing["admission_to_publication"]["mean"] - timing["correlation_kernel"]["mean"],
            decode_plus_encoder_mean_ms=timing["decode"]["mean"] + timing["encoder"]["mean"],
            summed_Update_SDK_run_mean_ms=timing["SDK_run_api"]["mean"],
            condition="Serial replay with all other measured phases unchanged. Not a measured optimized result, not a hard lower bound for future asynchronous/pipelined architecture."),
        prefix_trajectory_arrays_rechecked=False,
        trajectory_scope="This helper checks saved observation/shape identities and clocks only; separate root parity audit verifies saved trajectories/arrays.",
        candidate_accepted=False, realtime_validated=False, default_VGGT_unchanged=True)
    H.require(before == {p: sha(p) for p in before}, "Analysis inputs changed")
    result["source_pins"] = before
    result["sources_unchanged"] = True
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=result["status"], output_sha256=sha(a.output),
        steady_mean_ms=timing["admission_to_publication"]["mean"],
        steady_P95_ms=timing["admission_to_publication"]["p95"],
        corr_kernel_mean_ms=timing["correlation_kernel"]["mean"],
        Update_mean_ms=timing["Update"]["mean"], SDK_run_mean_ms=timing["SDK_run_api"]["mean"])))


if __name__ == "__main__":
    main()
