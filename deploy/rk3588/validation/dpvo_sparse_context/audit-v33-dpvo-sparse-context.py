#!/usr/bin/env python3
"""NumPy-only saved-artifact audit; never creates a model or ORT session.

DPVO reference: MIT, Copyright (c) 2022 Princeton Vision & Learning Lab.
Passing this audit means evidence agrees, not acceptance of a rejected parent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback
from pathlib import Path

import numpy as np

DRIVER_SHA = "b184dcc440771730cce13f9e7a72c065b619f469cab5b754a43a89c6bb0314cd"
EDITOR_SHA = "39004857559e27e3f9271ab413fc4498365370eac8862c56b16ba44b93f109ec"
FROZEN = {
    "capture": "e0563efb962edb90684480be114b88bc2c7825015f3b13a163f6f47f06360088",
    "export": "45fa589fa3a25a9722dc0832f5d14ffcec662e70d1dff135b86be641eeeba4b2",
    "old_ort": "153da765b1145d7bb74bc635c1982ae74c652dac28cf99be56cb40534c44b667",
    "spatial_ort": "a517f822a2e4afd01aff87df303183a3dc5850313134548891dc13c1e36b896b",
    "spatial_rewrite": "66eb8d135165f7538c97733d0c83198fa5a674e0cd2bdf64eb190c24819c5109",
    "original_sparse": "368ec845f14655bf71e2fc9d80d435197c5ab6b8dd807c7c9af92ef9e33d13e0",
    "spatial_sparse": "005667f44cf98447d2b622882d7d40cbcce8a545985ddf2104eaa62740273ce2",
}
RTOL, ATOL = 1e-4, 1e-5
SUCCESS = "completed_suffix_bitwise_gate_parent_rejection_retained_not_VO"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def array_info(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype), "bytes": value.nbytes,
            "tensor_sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}


def score(actual, expected):
    require(actual.dtype == expected.dtype == np.float32 and actual.shape == expected.shape,
            "FP32 comparison shape/dtype mismatch")
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), "Nonfinite comparison")
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    magnitude = np.abs(difference)
    failed = int(np.count_nonzero(magnitude > ATOL + RTOL * np.abs(expected.astype(np.float64))))
    norm = float(np.linalg.norm(expected.astype(np.float64).ravel()))
    err = float(np.linalg.norm(difference.ravel()))
    return {"rtol": RTOL, "atol": ATOL, "failed_elements": failed, "element_count": actual.size,
            "gate_passed": failed == 0, "array_equal": bool(np.array_equal(actual, expected)),
            "bitwise_equal": actual.tobytes(order="C") == expected.tobytes(order="C"),
            "max_abs": float(magnitude.max()), "relative_l2": err / norm if norm else (0.0 if err == 0 else None)}


def diagnostic_indices(frame):
    # Independent rendition of the predeclared formula; no feature-dependent choice.
    require(type(frame) is int and 0 <= frame < 3, "Unplanned diagnostic frame")
    centers = np.asarray([[1 + (37 * i + 17 * frame) % 150,
                           1 + (19 * i + 11 * frame) % 114] for i in range(96)], dtype=np.int64)
    centers[:6] = [(1, 1), (150, 114), (1, 114), (150, 1), (1, 1), (150, 114)]
    if frame == 1:
        centers = centers[::-1].copy()
    if frame == 2:
        centers = np.roll(centers, 13, axis=0)
    indices = centers[:, 1] * 152 + centers[:, 0]
    return centers, indices


def gather(dense, indices):
    require(dense.dtype == np.float32 and dense.ndim == 4 and dense.shape[0] == 1,
            "NCHW singleton FP32 dense reference required")
    require(indices.dtype == np.int64 and indices.ndim == 1 and
            np.all(indices >= 0) and np.all(indices < dense.shape[2] * dense.shape[3]), "Index bounds/type")
    # Explicit y/x advanced indexing, independently of the graph's flattened Gather.
    y, x = np.divmod(indices, dense.shape[3])
    return np.ascontiguousarray(dense[0][:, y, x].T[None])


class Checks:
    def __init__(self):
        self.count = 0

    def check(self, condition, message):
        self.count += 1
        require(condition, message)

    def equal(self, actual, expected, message):
        if isinstance(expected, dict):
            self.check(isinstance(actual, dict) and set(actual) == set(expected), message + " fields")
            for key, value in expected.items():
                self.equal(actual[key], value, message + "." + key)
        elif isinstance(expected, float):
            self.check(type(actual) in (float, int) and math.isfinite(actual) and
                       math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-14), message)
        else:
            self.check(actual == expected and (type(actual) is bool if type(expected) is bool else True), message)


def pinned_json(path, expected, checks):
    checks.check(sha(path) == expected, "Receipt SHA: " + Path(path).name)
    return json.loads(Path(path).read_text())


def artifact(directory, item, checks):
    name = item["file"]
    checks.check(isinstance(name, str) and Path(name).name == name, "Local simple artifact name")
    path = Path(directory) / name
    checks.check(path.stat().st_size == item["bytes"], "Artifact byte count " + name)
    checks.check(sha(path) == item["sha256"], "Artifact file SHA " + name)
    return path


def read_npz(directory, item, declared, checks):
    path = artifact(directory, item, checks)
    with np.load(path, allow_pickle=False) as saved:
        checks.check(set(saved.files) == set(declared), "NPZ exact arrays " + path.name)
        values = {key: saved[key] for key in saved.files}
    for key, value in values.items():
        checks.equal(array_info(value), declared[key], "NPZ raw metadata " + path.name + "." + key)
        checks.check(np.isfinite(value).all(), "Saved finite " + key)
    return values


def reference(directory, row, checks):
    values = read_npz(directory, row["artifact"], row["arrays"], checks)
    for key, channels in [("fmap", 128), ("imap", 384)]:
        checks.check(values[key].dtype == np.float32 and values[key].shape == (1, 1, channels, 116, 152),
                     "Frozen dense reference shape " + key)
    return {key: value[:, 0] for key, value in values.items()}


def check_global(report, history, checks):
    checks.check(report["schema"] == "v33_DPVO_sparse_context_board_CPU_ORT" and
                 report["script_sha256"] == DRIVER_SHA, "Measured source/schema")
    checks.check(report["providers"] == ["CPUExecutionProvider"] and report["graph_optimization"] == "ORT_DISABLE_ALL" and
                 report["cpu_affinity"] == [4, 5, 6, 7] and report["intra_threads"] == 4 and report["inter_threads"] == 1,
                 "Measured CPU ORT configuration")
    checks.check(report["actual_board_execution"] is True and report["NPU_used"] is False, "Actual scope")
    for key in ["complete_VO", "original_VGGT_used", "realtime_validated", "production_changed",
                "candidate_accepted_as_original_model", "model_checkpoint_loaded", "images_decoded"]:
        checks.check(report[key] is False, "No unsupported claim " + key)
    checks.check(report["normalization_applications_in_this_driver"] == 0, "No re-normalization")
    checks.equal(report["receipt_sha256"], FROZEN, "All measured input/recipe receipt identities")
    checks.check(report["sampling_rule"]["M"] == 96 and report["sampling_rule"]["original_RNG_observations"] is False and
                 report["sampling_rule"]["used_source_RGB_for_selection"] is False, "Explicit diagnostic sampling")
    checks.check(history["status"] == "failed" and history["exact_FP32_gate_passed"] is False,
                 "Frozen rejected spatial ancestor")
    rejection = report["historical_spatial_rejection"]
    checks.check(rejection["receipt_sha256"] == FROZEN["spatial_ort"] and rejection["status"] == "failed" and
                 rejection["exact_FP32_gate_passed"] is False and rejection["retained"] is True, "Rejection retained")
    checks.check(report["source_pins"] == report["source_hashes_after"] and report["sources_and_inputs_unchanged"] is True,
                 "Board-recorded provenance coherence; not host rehash of unavailable board paths")
    checks.check(DRIVER_SHA in report["source_pins"].values() and EDITOR_SHA in report["source_pins"].values(), "Implementation pins")
    checks.check(report["cleanup_errors"] == [] and report["affinity_restored"] is True, "Recorded cleanup")
    for method in ("original", "spatial"):
        life = report["session_lifecycle"][method]
        checks.check([row["role"] for row in life["session_initializations"]] == ["parent", "sparse"], "Session order")
        checks.check(life["ORT_session_references_dropped"] is True and life["gc_collect_performed"] is True and
                     life["native_explicit_destroy_API_available"] is False, "ORT ownership scope")


def audit(args):
    checks, output = Checks(), {"schema": "v33_DPVO_sparse_context_saved_array_review", "status": "failed",
        "audit_script_sha256": sha(__file__), "host_model_or_ORT_execution": False, "saved_arrays_only": True,
        "NPU_or_complete_VO_validated": False, "production_changed": False, "calls": []}
    try:
        report = pinned_json(args.run_dir / "ort.json", args.run_sha256, checks)
        capture = pinned_json(args.capture_dir / "capture.json", FROZEN["capture"], checks)
        old = pinned_json(args.old_ort_dir / "ort.json", FROZEN["old_ort"], checks)
        history = pinned_json(args.spatial_ort_dir / "ort.json", FROZEN["spatial_ort"], checks)
        checks.check(sha(args.driver) == DRIVER_SHA and sha(args.editor) == EDITOR_SHA, "Local source copies match measured pins")
        check_global(report, history, checks)
        checks.check(report["preprocessing"] == capture["preprocessing"] and
                     report["preprocessing"]["normalization"] == "2 * (image / 255.0) - 0.5, FP32", "Input preprocessing provenance")
        checks.check(report["gate"]["rtol"] == RTOL and report["gate"]["atol"] == ATOL, "No tolerance change")
        checks.check([(r["method"], r["frame_index"]) for r in report["calls"]] ==
                     [(method, i) for method in ("original", "spatial") for i in range(3)], "Six calls, fixed order, no selection")
        for method in ("original", "spatial"):
            directory = getattr(args, method + "_sparse_dir")
            recipe = pinned_json(directory / "rewrite.json", FROZEN[method + "_sparse"], checks)
            checks.check(recipe["candidate_accepted"] is False and recipe["script_sha256"] == EDITOR_SHA and
                         recipe["source_pins"] == recipe["source_hashes_after"], "Static lineage retains pending numerical state")
            artifact(directory, recipe["output"], checks)
            checks.equal(report["graph_preservation"][method], recipe["preservation"], "Recorded unchanged graph evidence")
        calculated = {"suffix_strict": True, "suffix_bitwise": True, "original_CPU": {"original": True, "spatial": True}}
        unique_inputs, artifact_shas = set(), {}
        for row in report["calls"]:
            i, method = row["frame_index"], row["method"]
            source = capture["inputs"][i]
            checks.check(row["frame_id"] == source["frame_id"] == source["source"]["frame_id"] and
                         row["source_image"] == source["source"], "Original frame identity/order")
            values = read_npz(args.run_dir, row["artifact"], row["arrays"], checks)
            checks.check(values["fmap"].shape == values["parent_fmap"].shape == (1, 128, 116, 152) and
                         values["context"].shape == values["parent_context_gather"].shape == (1, 96, 384), "Sparse shapes")
            for key in ("fmap", "parent_fmap", "context", "parent_context_gather"):
                checks.check(values[key].dtype == np.float32, "Output FP32 dtype " + key)
            centers, indices = diagnostic_indices(i)
            checks.check(values["centers_xy"].dtype == values["patch_indices"].dtype == np.int64 and
                         values["centers_xy"].tobytes() == centers.tobytes() and values["patch_indices"].tobytes() == indices.tobytes(),
                         "Predeclared order, duplicates and row-major integer indices")
            inputs = read_npz(args.run_dir, row["input_artifact"], {"normalized_bgr": row["input_metadata"],
                "centers_xy": row["centers"], "patch_indices": row["indices"]}, checks)
            original_path = artifact(args.capture_dir, source["input"], checks)
            original_input = np.load(original_path, allow_pickle=False)
            checks.equal(array_info(original_input), source["array"], "Original raw normalized input")
            checks.check(original_input.shape == (1, 1, 3, 464, 608) and inputs["normalized_bgr"].shape == (1, 3, 464, 608) and
                         inputs["normalized_bgr"].dtype == np.float32 and
                         inputs["normalized_bgr"].tobytes() == original_input[:, 0].tobytes(), "Same pixels, singleton view only removed")
            checks.check(inputs["centers_xy"].tobytes() == centers.tobytes() and inputs["patch_indices"].tobytes() == indices.tobytes(),
                         "Shared input and output indices match")
            unique_inputs.add(row["input_artifact"]["file"])
            cpu_row = next(r for r in capture["calls"] if r["frame_index"] == i)
            cpu = reference(args.capture_dir, cpu_row, checks)
            parent_row = old["pairs"][i] if method == "original" else history["calls"][i]
            parent_dir = args.old_ort_dir if method == "original" else args.spatial_ort_dir
            checks.check(parent_row["frame_index"] == i, "Frozen parent frame index")
            saved_parent = reference(parent_dir, parent_row, checks)
            selected_cpu = {"fmap": cpu["fmap"], "context": gather(cpu["imap"], indices)}
            selected_parent = {"fmap": saved_parent["fmap"], "context": gather(saved_parent["imap"], indices)}
            current_parent = {"fmap": values["parent_fmap"], "context": values["parent_context_gather"]}
            results = {}
            for label, expected in [("versus_current_parent_ORT", current_parent), ("versus_original_CPU", selected_cpu),
                                    ("versus_saved_corresponding_parent_ORT", selected_parent)]:
                results[label] = {key: score(values[key], expected[key]) for key in ("fmap", "context")}
                checks.equal(row[label], results[label], "Independently recomputed " + label)
            checks.equal(row["parent_repeated_vs_saved"]["fmap"], score(values["parent_fmap"], saved_parent["fmap"]),
                         "Saved complete current parent fmap vs historical parent")
            checks.equal(row["parent_dense_outputs_metadata"]["fmap"], array_info(values["parent_fmap"]), "Current fmap actual raw metadata")
            dense_matches = {}
            for key in ("fmap", "imap"):
                expected_meta = array_info(saved_parent[key])
                dense_matches[key] = row["parent_dense_outputs_metadata"][key] == expected_meta
                checks.check(dense_matches[key], "Board-recorded current dense parent hash matches host-rehashed historical " + key)
            checks.check(current_parent["context"].tobytes() == selected_parent["context"].tobytes(), "Saved current parent context equals historical dense gather")
            for key in ("parent_forward_seconds", "sparse_forward_seconds"):
                checks.check(type(row[key]) in (float, int) and math.isfinite(row[key]) and row[key] > 0, "Recorded positive session timing")
            calculated["suffix_strict"] &= all(v["gate_passed"] for v in results["versus_current_parent_ORT"].values())
            calculated["suffix_bitwise"] &= all(v["bitwise_equal"] for v in results["versus_current_parent_ORT"].values())
            calculated["original_CPU"][method] &= all(v["gate_passed"] for v in results["versus_original_CPU"].values())
            artifact_shas[row["artifact"]["file"]] = row["artifact"]["sha256"]
            artifact_shas[row["input_artifact"]["file"]] = row["input_artifact"]["sha256"]
            output["calls"].append({"method": method, "frame_index": i, "frame_id": row["frame_id"],
                "input_raw_sha256": row["input_metadata"]["tensor_sha256"], "indices_raw_sha256": row["indices"]["tensor_sha256"],
                "comparisons": results, "current_parent_context_bitwise_historical_gather": True,
                "current_dense_parent_recorded_metadata_matches_host_rehashed_history": dense_matches,
                "parent_session_run_seconds": row["parent_forward_seconds"], "sparse_session_run_seconds": row["sparse_forward_seconds"]})
            del cpu, saved_parent, selected_cpu, selected_parent, values, inputs, original_input
        checks.check(len(unique_inputs) == 3 and len(artifact_shas) == 9, "Only three input snapshots, six output snapshots")
        checks.equal(report["suffix_strict_gate_passed"], bool(calculated["suffix_strict"]), "Suffix strict summary")
        checks.equal(report["suffix_bitwise_gate_passed"], bool(calculated["suffix_bitwise"]), "Suffix bitwise summary")
        checks.equal(report["original_CPU_strict_gate_each_method"], calculated["original_CPU"], "Original CPU summary with spatial rejection")
        checks.check(report["status"] == (SUCCESS if calculated["suffix_strict"] and calculated["suffix_bitwise"] else "failed"), "Measured result status")
        output.update(status="passed", numerical_evidence=calculated, candidate_accepted_as_original_model=False,
            raw_receipt_sha256=args.run_sha256, source_implementation_sha256={"driver": DRIVER_SHA, "editor": EDITOR_SHA},
            local_artifact_sha256=artifact_shas, historical_spatial_rejection_retained=True,
            dense_parent_scope={"fmap": "Complete current parent array saved and byte-compared with historical array",
                "imap": "Current complete imap not saved; board-recorded raw SHA matched independently rehashed historical imap. Saved current context also byte-matched its historical gather.",
                "full_current_imap_independently_available": False},
            source_scope="Board before/after equality checked as recorded; unavailable original board paths not independently rehashed on host",
            timing_scope=report["timing_scope"], measured_samples_per_method_and_frame=1,
            limits=["CPU ORT isolated suffix only; no RKNN suffix placement or I/O measured", "Diagnostic centers, not original RNG tracking observations",
                    "All full CNN calculations retained", "Spatial parent still fails original CPU FP32 gate", "No VO, trajectory accuracy, sustained 20 Hz, camera latency or default change"],
            recorded_cleanup_errors=report["cleanup_errors"], raw_artifacts_modified=False)
    except Exception as exc:
        output.update(error=repr(exc), traceback=traceback.format_exc())
    output.update(checks=checks.count, failures=0 if output["status"] == "passed" else 1)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "capture", "old-ort", "spatial-ort", "original-sparse", "spatial-sparse"):
        parser.add_argument("--" + name + "-dir", type=Path, required=True)
    parser.add_argument("--run-sha256", required=True)
    parser.add_argument("--driver", type=Path, default=Path(__file__).with_name("run-v33-dpvo-sparse-context-ort-board.py"))
    parser.add_argument("--editor", type=Path, default=Path(__file__).with_name("rewrite-v33-dpvo-sparse-context.py"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "New audit output required; preserve prior failure receipts")
    result = audit(args)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "checks", "failures")}), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
