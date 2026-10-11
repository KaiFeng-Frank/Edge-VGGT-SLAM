#!/usr/bin/env python3
"""Audit saved v31 diagnostic14 failure arrays; NumPy only, no inference.

This auditor checks consistency of the expected failed experiment. Its successful
exit never accepts the candidate model or validates a trajectory. The reference
array is the unchanged rtol/atol denominator. Calls without saved arrays are
explicitly distinguished from independently recomputed calls.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

RTOL, ATOL = 1e-4, 1e-5
SAVED_CALLS = (0, 13, 14)
ARRAY_NAMES = ("net", "delta", "weight")


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def residual_statistics(reference, candidate):
    """Compute fixed original-reference gate using saved float32 values."""
    if reference.dtype != np.float32 or candidate.dtype != np.float32:
        raise ValueError("Expected original and candidate FP32 arrays")
    if reference.shape != candidate.shape or not reference.size:
        raise ValueError("Expected identical nonempty array shapes")
    if not (np.isfinite(reference).all() and np.isfinite(candidate).all()):
        raise ValueError("Nonfinite saved values")
    original = reference.astype(np.float64)
    difference = candidate.astype(np.float64) - original
    error = np.abs(difference)
    tolerance = ATOL + RTOL * np.abs(original)
    ratio = error / tolerance
    failed = error > tolerance
    original_norm = float(np.linalg.norm(original.ravel()))
    relative_l2 = float(np.linalg.norm(difference.ravel())) / original_norm if original_norm else None
    failed_details = []
    for index in np.argwhere(failed)[:32]:
        coordinate = tuple(int(i) for i in index)
        failed_details.append({
            "index": list(coordinate),
            "reference": float(original[coordinate]),
            "candidate": float(candidate[coordinate]),
            "signed_error": float(difference[coordinate]),
            "absolute_error": float(error[coordinate]),
            "allowed_tolerance": float(tolerance[coordinate]),
            "normalized_tolerance_ratio": float(ratio[coordinate]),
            "excess_over_tolerance": float(error[coordinate] - tolerance[coordinate]),
        })
    return {
        "shape": list(reference.shape), "elements": int(reference.size),
        "finite": True, "rtol": RTOL, "atol": ATOL,
        "failed_elements": int(failed.sum()), "gate_passed": not bool(failed.any()),
        "max_abs": float(error.max()), "relative_l2": relative_l2,
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
        "max_normalized_tolerance_ratio": float(ratio.max()),
        "failed_coordinates": failed_details,
        "failed_coordinates_truncated": bool(failed.sum() > 32),
    }


def audit_run(run_dir, run_sha256):
    root = Path(run_dir)
    checks = 0

    def require(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            raise ValueError(message)

    run_path = root / "run.json"
    require(file_sha(run_path) == run_sha256, "run.json SHA mismatch")
    run = json.loads(run_path.read_text())
    rows = run["original_reference_calls"]
    require(len(rows) == 15, "Expected 15 original-reference comparisons")
    require([row["index"] for row in rows] == list(range(15)), "Call indices mismatch")
    require([row["edges"] for row in rows] == [96] * 13 + [6144] * 2, "Edge cohort mismatch")
    for row in rows:
        for name in ARRAY_NAMES:
            recorded = row["checks"][name]
            expected_failures = 1 if row["index"] == 14 and name != "weight" else 0
            require(recorded["rtol"] == RTOL and recorded["atol"] == ATOL, "Changed original gate")
            require(recorded["failed_elements"] == expected_failures, "Unexpected recorded failure count")
            require(recorded["gate_passed"] is (expected_failures == 0), "Recorded gate flag mismatch")

    saved_results, artifacts = [], []
    for call_index in SAVED_CALLS:
        row = rows[call_index]
        arrays = {}
        for role, metadata_field in (("original", "original_arrays"), ("candidate", "candidate_arrays")):
            artifact = row[role]
            name = artifact["file"]
            require(isinstance(name, str) and Path(name).name == name, "Unsafe artifact filename")
            path = root / name
            require(not path.is_symlink() and path.is_file(), "Expected regular artifact file")
            require(path.stat().st_size == artifact["bytes"], "NPZ bytes mismatch")
            require(file_sha(path) == artifact["sha256"], "NPZ SHA mismatch")
            require(artifact["arrays"] == row[metadata_field], "Conflicting array metadata")
            with np.load(path, allow_pickle=False) as npz:
                require(set(npz.files) == set(ARRAY_NAMES), "NPZ keys mismatch")
                arrays[role] = {name: npz[name].copy() for name in ARRAY_NAMES}
            for name, value in arrays[role].items():
                metadata = artifact["arrays"][name]
                expected_channels = 384 if name == "net" else 2
                require(value.shape == (1, row["edges"], expected_channels), "Array contract shape mismatch")
                require(list(value.shape) == metadata["shape"] and str(value.dtype) == metadata["dtype"], "Array metadata mismatch")
                require(value.nbytes == metadata["bytes"] and array_sha(value) == metadata["tensor_sha256"], "Raw array SHA/bytes mismatch")
                require(value.dtype == np.float32 and np.isfinite(value).all(), "Nonfinite/non-FP32 array")
                if name == "weight":
                    require(bool(((value >= 0) & (value <= 1)).all()), "Saved weight outside probability domain")
            artifacts.append({"call_index": call_index, "role": role, **artifact})
        results = {}
        for name in ARRAY_NAMES:
            value = residual_statistics(arrays["original"][name], arrays["candidate"][name])
            recorded = row["checks"][name]
            require(all(value[key] == recorded[key] for key in ("rtol", "atol", "failed_elements", "gate_passed", "max_abs")), "Independent statistics disagree with run")
            results[name] = value
        saved_results.append({"index": call_index, "edges": row["edges"], "arrays": results})

    require(run["status"] == "failed", "Expected failed experiment")
    require("unchanged original FP32 gate" in run["error"], "Unexpected failure reason")
    require(run["frames_requested"] == 14 and run["frames_completed"] == 13 and run["active_frame"] == 13, "Frame failclosed accounting mismatch")
    require(len(run["frames"]) == 13, "Frame rows mismatch")
    for index, frame in enumerate(run["frames"]):
        result = frame["result"]
        require(frame["index"] == index and result["observation_id"] == index, "Frame order mismatch")
        require(result["fresh_pose_valid"] is False and result["pose_w2c"] is None and result["initialized"] is False, "Failed initialization must not emit a fresh pose")
    metrics = run["stream_metrics_at_exit"]
    require(metrics["status"] == "failed" and metrics["fresh_poses"] == 0 and metrics["offline_updates"] == 0, "Fresh/EOF failure accounting mismatch")
    require(metrics["observations"] == 13 and metrics["admitted"] == 7 and metrics["retained_frames"] == 7, "Candidate frame was incorrectly committed")
    require(metrics["motion_probe_calls"] == 13 and metrics["update_calls"] == 2 and metrics["BA_calls"] == 1 and metrics["BA_iterations"] == 2, "Failure stage counters mismatch")
    require("termination" not in run and "trajectory" not in run, "EOF reconstruction unexpected in failed run")
    dynamic = run["dynamic_Update_metrics"]
    require(dynamic["attempted_calls"] == dynamic["completed_calls"] == 15 and dynamic["failed_calls"] == 0, "Runtime versus numerical rejection counters mismatch")
    require(dynamic["calls"] == run["update_calls"] and len(dynamic["calls"]) == 15, "Dynamic call record mismatch")
    require(dynamic["closed"] is True and run["encoder_metrics"]["closed"] is True, "Runtime/encoder not closed")
    cache = dynamic["topology_cache"]
    require(cache["closed"] is True and cache["current_entries"] == cache["current_payload_bytes"] == 0, "Topology cache retained values")
    corr = run["correlation"]
    require(corr["attempted_calls"] == corr["completed_calls"] == 15 and corr["current_context_open"] is False, "Native correlation lifecycle mismatch")
    require(len(corr["closed_contexts"]) == len(corr["generations"]), "Missing correlation context closure")
    require(all(x["closed"] is True and x["current_context_bytes"] == 0 for x in corr["closed_contexts"]), "Correlation context not freed")
    require(run["cleanup_errors"] == [] and run["affinity_restored"] is True, "Cleanup/affinity failed")
    require(run["original_weights_before"] == run["original_weights_after"] and run["original_weights_unchanged"] is True, "Recorded weights changed")
    require(run["source_pins"] == run["source_hashes_after"] and run["sources_and_inputs_unchanged"] is True, "Recorded sources changed")
    require(dynamic["providers"] == ["CPUExecutionProvider"] and dynamic["NPU_used"] is False and run["NPU_used"] is False, "Unexpected NPU/provider claim")
    require(run["realtime_validated"] is False and run["accuracy_vs_VGGT_validated"] is False, "Unsupported acceptance claim")
    require(dynamic["model_sha256"] in run["dynamic_sources_after"].values() and dynamic["manifest_sha256"] in run["dynamic_sources_after"].values(), "Dynamic artifact pin chain mismatch")
    require(file_sha(run_path) == run_sha256, "run.json changed during audit")
    return {
        "schema": "v31_saved_array_diagnostic14_independent_review",
        "status": "passed_evidence_consistency_candidate_FP32_rejected",
        "checks_passed": checks, "candidate_FP32_equivalent": False,
        "strict_gate": {"rtol": RTOL, "atol": ATOL, "formula": "abs(candidate-reference) <= atol+rtol*abs(reference)", "reference_is_denominator": True, "changed": False},
        "auditor_sha256": file_sha(__file__), "run_sha256": run_sha256,
        "script_sha256": run["script_sha256"], "dynamic_model_sha256": dynamic["model_sha256"],
        "dynamic_manifest_sha256": dynamic["manifest_sha256"],
        "input_manifest_sha256": run["input_manifest_sha256"],
        "saved_artifacts": artifacts, "independently_recomputed_calls": saved_results,
        "recorded_only_calls_not_independently_recomputed": list(range(1, 13)),
        "reported_gate_cohort": {"passed_calls": list(range(14)), "rejected_call": 14, "net_failed_elements": 1, "delta_failed_elements": 1, "weight_failed_elements": 0},
        "failure_lifecycle": {
            "frames_requested": 14, "frames_completed": 13, "failed_observation_index": 13,
            "fresh_poses": 0, "initialized": False, "candidate_observation_committed": False,
            "offline_updates": 0, "termination_run": False,
            "completed_finite_ORT_calls": 15, "strict_comparison_failed_calls": 1,
            "dynamic_runtime_closed": True, "topology_cache_closed_and_empty": True,
            "encoder_closed": True, "correlation_contexts_closed": len(corr["closed_contexts"]),
            "cleanup_errors": [], "affinity_restored": True,
        },
        "source_and_weight_evidence": {
            "board_reported_before_after_equal": True,
            "board_source_pin_count": len(run["source_pins"]),
            "board_original_parameter_count": len(run["original_weights_before"]),
            "all_board_source_files_independently_rehashed_on_host": False,
            "reported_source_hashes_by_basename": {Path(k).name: v for k, v in run["source_pins"].items() if Path(k).suffix in (".py", ".so")},
        },
        "scope": {
            "saved_NPZ_files_read": 6, "raw_arrays_verified": 18,
            "host_model_forward_calls": 0, "host_ORT_calls": 0, "host_BA_calls": 0,
            "images_read": 0, "board_jobs": 0,
            "trajectory_accuracy_validated": False, "realtime_validated": False,
            "small_relative_L2_does_not_override_elementwise_rejection": True,
        },
        "next_separate_trajectory_diagnostic": {
            "permitted_research_interpretation": "A separately named approximate-output research run may continue only while keeping strict_FP32_gate_passed=false and recording every strict failure; this is not equivalence or production acceptance.",
            "required_controls": [
                "Retain this failed receipt and immutable historical tolerances.",
                "Use identical frozen observation IDs, sensor timestamps, input hashes, RNG seeds, lifecycle config and CPU geometry for independent original and candidate rollouts.",
                "Each rollout must carry its own net, poses, depth, motion decisions and graph topology; a Torch evaluation on the candidate current state is only a local diagnostic.",
                "Keep nonfinite/invalid-domain/capacity/source/lifecycle failures fatal; do not replace poses or skip rejection records to improve coverage.",
                "Use preplanned sensor-time RPE endpoints, disclosed common and individual coverage, one stated scale convention and GT association; separate fresh and reconstructed EOF populations.",
                "Measure accuracy and throughput separately; diagnostic original-reference overhead is not ordinary replay performance.",
            ],
            "production_adoption_or_gate_relaxation_authorized_by_this_audit": False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = audit_run(args.run_dir, args.run_sha256)
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an audit receipt")
    args.output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": receipt["status"], "checks_passed": receipt["checks_passed"], "auditor_sha256": receipt["auditor_sha256"], "receipt_sha256": file_sha(args.output)}))


if __name__ == "__main__":
    main()
