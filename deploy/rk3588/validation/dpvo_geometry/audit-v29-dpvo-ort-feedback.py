#!/usr/bin/env python3
"""Independent NumPy saved-array audit, with no model/correlation/BA execution.

DPVO/v28 mathematical provenance: Princeton Vision & Learning Lab, MIT.
This audit checks saved identities, targets, residual statistics, topology and
timing, not a second BA solve or a claimed online trajectory.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

DRIVER_SHA = "c89152b8192d193d0d81483c814fac873d474c09adc7ad6cfe9050c46cb08af1"
PARENT_SHA = "e35a2bde3154f527a010b0cf7a2f3192bfe6e356ffd3f6b6259a9289e6bbed42"
GEOMETRY_SHA = "c893f12d188baced438504cfc592a8900e18e40ef8223af0a7e944e4484b0c3b"
ADAPTER_SHA = "3f755c8e24d092fa55fdf698d87c3d9ba8944487ee95d4e5afc47554d31d1380"
CAPTURE_SHA = "e0563efb962edb90684480be114b88bc2c7825015f3b13a163f6f47f06360088"
MOTION_SHA = "55a7db19d1a9bfbf986da62ed999fc8b122add0fca84f01681e6ad0b82ba4e86"
UPDATE_CAPTURE_SHA = "87e18137a828496d623721f7748e288d02f9d8d6879278da1e5d3d9731cfa0a1"
UPDATE_EXPORT_SHA = "56f96b7bf80bde596de6a4008da2858a869a9f19a3162abb8db8ab1af0156cac"
MANIFEST_SHA = "27c8ec9c3e68f65a1f7648fe11031967eb37cad588a83351ce0bd1f9393765dc"
BUILD_SHA = "34fbf049852b351e7df1097cc806fbf83e1e1035e18b01ca4e9e8ba39499cd18"
OUTPUTS = ("net_out", "delta", "weight")
STAGES = ("snapshot", "reproject", "correlation", "update", "target", "BA", "reproject_after", "diagnostics")
TOPOLOGY_NAMES = ("previous_indices", "previous_valid", "following_indices", "following_valid",
                  "kk_indices", "kk_valid", "kk_inverse", "ij_indices", "ij_valid", "ij_inverse")
RTOL, ATOL = 1e-4, 1e-5


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def meta(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes,
                tensor_sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest())


def same_bytes(left, right):
    return left.shape == right.shape and left.dtype == right.dtype and left.tobytes(order="C") == right.tobytes(order="C")


def artifact(folder, item):
    name = item["file"]
    require(isinstance(name, str) and name not in ("", ".", "..") and Path(name).name == name,
            "Unsafe artifact basename")
    path = Path(folder) / name
    require(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], "Artifact file SHA/bytes differ: " + name)
    with np.load(path, allow_pickle=False) as store:
        arrays = {key: store[key].copy() for key in store.files}
    require({key: meta(value) for key, value in arrays.items()} == item["arrays"], "Artifact raw metadata differs: " + name)
    return arrays


def receipt(path, expected):
    require(sha(path) == expected, "Historical receipt SHA differs: " + Path(path).name)
    return json.loads(Path(path).read_text())


def strict_compare(actual, original):
    require(actual.dtype == original.dtype == np.float32 and actual.shape == original.shape and
            np.isfinite(actual).all() and np.isfinite(original).all(), "Comparison needs same finite FP32 shape")
    difference = actual.astype(np.float64) - original.astype(np.float64)
    failures = int(np.count_nonzero(np.abs(difference) > ATOL + RTOL * np.abs(original.astype(np.float64))))
    return dict(rtol=RTOL, atol=ATOL, gate_passed=failures == 0, failed_elements=failures,
                max_abs=float(np.abs(difference).max()),
                relative_l2=float(np.linalg.norm(difference.ravel()) /
                                  max(float(np.linalg.norm(original.astype(np.float64).ravel())), 1e-30)),
                array_equal=bool(np.array_equal(actual, original)), bitwise_equal=same_bytes(actual, original))


def reference_checks(row, arrays, saved_reference, graph):
    expected_inputs = {name: meta(value) for name, value in
                       (("net", arrays["net_before"]), ("inp", graph["ctx"]), ("corr", arrays["corr"]),
                        ("ii", graph["ii"]), ("jj", graph["jj"]), ("kk", graph["kk"]))}
    require(row["inputs"] == expected_inputs, "Torch reference is not bound to the current ORT net/corr and original graph")
    require(tuple(saved_reference) == OUTPUTS, "Saved Torch reference output order differs")
    checks = {name: strict_compare(arrays[key], saved_reference[name]) for name, key in
              zip(OUTPUTS, ("net_after", "delta", "weight"))}
    # Scalar BLAS norm-reduction roundoff is not the elementwise FP32 gate.
    # Keep thresholds/counts/max_abs/byte decisions exact; only relative_l2
    # summary reduction is allowed this much smaller portability tolerance.
    require(set(row["comparisons"]) == set(checks) and all(
        set(checks[name]) == set(row["comparisons"][name]) and all(
            math.isclose(value, row["comparisons"][name][key], rel_tol=1e-12, abs_tol=1e-15)
            if key == "relative_l2" else value == row["comparisons"][name][key]
            for key, value in checks[name].items()) for name in checks) and
            row["same_current_net_before_and_correlation"] is True and
            row["strict_FP32_gate_passed"] is True and all(x["gate_passed"] for x in checks.values()),
            "Current-input original/ORT strict gate failed or recorded statistics differ")
    require(np.all((saved_reference["weight"] >= 0) & (saved_reference["weight"] <= 1)),
            "Original Torch reference weight outside [0,1]")
    return checks


def expected_topology(ii, jj, kk):
    require(all(x.dtype == np.int64 and x.ndim == 1 and x.shape == kk.shape for x in (ii, jj, kk)), "Int64 graph IDs required")
    count = len(kk)
    order = np.argsort(jj, kind="stable")
    order = order[np.argsort(kk[order], kind="stable")]
    previous, following = np.full(count, -1, np.int64), np.full(count, -1, np.int64)
    for left, right in zip(order[:-1], order[1:]):
        if kk[left] == kk[right]:
            previous[right], following[left] = left, right
    result = dict(previous_indices=np.maximum(previous, 0), previous_valid=(previous >= 0).astype(np.float32)[None, :, None],
                  following_indices=np.maximum(following, 0), following_valid=(following >= 0).astype(np.float32)[None, :, None])
    for name, labels in (("kk", kk), ("ij", ii * np.int64(12345) + jj)):
        unique, inverse = np.unique(labels, return_inverse=True)
        members = [np.flatnonzero(labels == value) for value in unique]
        capacity = max(len(value) for value in members)
        indices = np.zeros((len(unique), capacity), np.int64)
        valid = np.zeros(indices.shape, np.float32)
        for index, value in enumerate(members):
            indices[index, :len(value)], valid[index, :len(value)] = value, 1
        result.update({name+"_indices": indices, name+"_valid": valid,
                       name+"_inverse": inverse.astype(np.int64), name+"_labels": unique})
    result.update(ii=ii, jj=jj, kk=kk)
    return result


def residual_summary(residual, active, weight, selected=None):
    error = residual[0].astype(np.float64)
    finite = np.isfinite(error).all(axis=-1)
    chosen = active if selected is None else selected
    mask = chosen & finite
    norm = np.linalg.norm(error[finite], axis=-1)
    return dict(active_edges=int(active.sum()), finite_residual_edges=int(finite.sum()),
                nonfinite_residual_edges=int((~finite).sum()), selected_edges=int(mask.sum()),
                all_finite_residual_norm_mean=float(norm.mean()) if norm.size else None,
                all_finite_residual_norm_max=float(norm.max()) if norm.size else None,
                weighted_squared_residual=float(np.sum(weight[0, mask].astype(np.float64) * error[mask]**2)),
                selection="current online gate" if selected is None else "fixed pre-BA active mask, finite rows only")


def timing_checks(row, previous_reference_end=None):
    require(tuple(row["seconds"]) == STAGES and all(math.isfinite(x) and x >= 0 for x in row["seconds"].values()),
            "Feedback stage times must be finite/nonnegative and ordered")
    total = sum(row["seconds"].values())
    begin, end = row["begin_monotonic"], row["end_monotonic"]
    require(math.isfinite(begin) and math.isfinite(end) and end > begin and
            abs(total - row["round_total_seconds"]) < 1e-9 and abs(end-begin-total) < 1e-9,
            "Feedback timer sum/boundaries differ")
    ref = row["original_Torch_reference"]
    rb, re = ref["begin_monotonic"], ref["end_monotonic"]
    require(math.isfinite(rb) and math.isfinite(re) and end <= rb <= re and
            abs(re-rb-ref["reference_forward_seconds"]) < 1e-9,
            "Original reference timer overlaps feedback or has incorrect duration")
    if previous_reference_end is not None:
        require(begin >= previous_reference_end, "Next round began before the prior golden call finished")
    ort = row["ORT_update_timing"]
    update_begin = begin + sum(row["seconds"][name] for name in STAGES[:3])
    update_end = update_begin + row["seconds"]["update"]
    require(update_begin-1e-9 <= ort["begin_monotonic"] <= ort["end_monotonic"] <= update_end+1e-9,
            "ORT timing lies outside the Update stage")
    parts = ("input_check_and_pack_seconds", "run_seconds", "owned_copy_seconds", "output_check_seconds")
    require(all(math.isfinite(ort[name]) and ort[name] >= 0 for name in parts) and
            abs(sum(ort[name] for name in parts)-ort["total_seconds"]) < 1e-9 and
            abs(ort["end_monotonic"]-ort["begin_monotonic"]-ort["total_seconds"]) < 1e-9,
            "ORT checks/pack/run/copy timer accounting differs")
    return re


def audit(args):
    run_path = args.run_dir / "run.json"
    run = json.loads(run_path.read_text())
    capture = receipt(args.capture_dir/"capture.json", CAPTURE_SHA)
    motion = receipt(args.motion_dir/"motion.json", MOTION_SHA)
    update_capture = receipt(args.update_capture_dir/"capture.json", UPDATE_CAPTURE_SHA)
    build = receipt(args.build_receipt, BUILD_SHA)
    manifest = receipt(args.candidate_manifest, MANIFEST_SHA)
    candidate = receipt(args.candidate_evidence_dir/"run.json", run["candidate_evidence_sha256"])
    geometry = receipt(args.geometry_evidence_dir/"run.json", run["prior_geometry_acceptance"]["receipt_sha256"])
    source_files = {
        "driver": (args.v29_source_dir/"run-v29-dpvo-ort-feedback-board.py", DRIVER_SHA),
        "vectorized_geometry": (args.v29_source_dir/"dpvo_cpu_geometry_vectorized.py", GEOMETRY_SHA),
        "frozen_feedback": (args.v28_source_dir/"run-v28-dpvo-feedback-board.py", PARENT_SHA),
        "external_topology_adapter": (args.v28_source_dir/"dpvo_update_export.py", ADAPTER_SHA),
    }
    for name, (path, expected) in source_files.items():
        require(sha(path) == expected, "Measured local source differs: " + name)
    checks = []

    def check(name, ok):
        checks.append(dict(name=name, passed=bool(ok)))
        require(ok, name)

    check("completed CPU-only fixed graph research", run["schema"] == "v29_DPVO_fixed_three_frame_ORT_feedback" and
          run["status"] == "completed_fixed_graph_ORT_neural_geometry_feedback_not_VO" and
          run["actual_board_execution"] is True and run["candidate_accepted"] is True and
          all(run[key] is False for key in ("NPU_used", "CUDA_used", "ORB_used", "production_changed", "complete_VO_executed",
              "official_eight_frame_initialization_executed", "streamed_or_online_input", "trajectory_accuracy_validated", "realtime_validated", "GT_or_sensor_depth_used")))
    check("measured source and original/native input binding", run["script_sha256"] == DRIVER_SHA and run["geometry_sha256"] == GEOMETRY_SHA and
          run["frozen_feedback_sha256"] == PARENT_SHA and run["adapter_sha256"] == ADAPTER_SHA and
          run["capture_sha256"] == CAPTURE_SHA and run["motion_sha256"] == MOTION_SHA and run["build_sha256"] == BUILD_SHA and
          run["library_sha256"] == build["output"]["sha256"] and run["candidate_manifest_sha256"] == MANIFEST_SHA and
          run["prior_candidate_acceptance"]["receipt_sha256"] == run["candidate_evidence_sha256"])
    check("prior sixteen-state CPU ORT candidate acceptance/source/cleanup", candidate["status"] == "completed_numerical_accepted" and
          candidate["execution_status"] == "completed" and candidate["candidate_accepted"] is True and candidate["mode"] == "ort" and
          candidate["manifest_sha256"] == MANIFEST_SHA and candidate["capture_sha256"] == UPDATE_CAPTURE_SHA and candidate["export_sha256"] == UPDATE_EXPORT_SHA and
          candidate["first_only"] is False and candidate["selected_saved_states"] == len(candidate["calls"]) == 16 and
          candidate["model_keys"] == ["E864", "E96"] and sum(row["model_key"] == "E864" for row in candidate["calls"]) == 8 and
          candidate["providers"] == ["CPUExecutionProvider"] and candidate["ORT_optimization"] == "disabled" and
          all(row["core_strict_FP32_gate_passed"] and row["weight_domain_passed"] and row["all_outputs_finite"] for row in candidate["calls"]) and
          candidate["source_pins"] == candidate["source_hashes_after"] and candidate["sources_and_inputs_unchanged"] and not candidate["cleanup_errors"])
    check("prior scalar/vectorized geometry strict gate/source/cleanup", geometry["status"] == "completed_geometry_only_benchmark" and
          geometry["candidate_accepted"] and geometry["strict_FP32_gate_passed"] and geometry["vectorized_geometry_sha256"] == GEOMETRY_SHA and
          geometry["sources_and_artifacts_unchanged"] and geometry["immutable_arrays_unchanged"] and
          geometry["source_pins"] == geometry["source_hashes_after"] and not geometry["cleanup_errors"])
    check("four ORT plus four current-state references and eight BA steps", run["feedback_rounds_completed"] == run["feedback_rounds_executed"] == len(run["calls"]) == 4 and
          run["ORT_Update_calls_attempted"] == run["ORT_Update_calls_completed"] == len(run["ORT_Update_calls"]) == run["expected_ORT_Update_calls"] == 4 and
          run["original_Torch_reference_calls_attempted"] == run["original_Torch_reference_calls_completed"] == run["expected_original_Torch_reference_calls"] == 4 and
          run["completed_BA_iterations"] == 8 and run["completed_BA_calls"] == 4 and
          run["actual_stage_attempts"] == dict(correlation=4, update=4, BA=4) and run["current_state_strict_FP32_gate_passed"])
    check("94 original parameter raw-metadata hashes unchanged", len(run["original_weights_before"]) == 94 and
          run["original_weights_before"] == run["original_weights_after"] == capture["state_before"] == motion["original_weights_before"] == update_capture["original_weights_before"] and
          run["original_weights_unchanged"] and run["original_DPVO_neural_weights"] and not run["original_VGGT_weights_used"])
    check("reported source/immutable input/lifecycle clean", run["source_pins"] == run["source_hashes_after"] and
          run["sources_and_inputs_unchanged"] and run["immutable_features_graph_topology_unchanged"] and
          run["affinity_restored"] and run["ORT_session_released"] and run["neural_reference_released"] and not run["cleanup_errors"])
    check("CPU/ORT providers and explicit threads", run["cpu_affinity"] == [4, 5, 6, 7] and run["torch_threads"] == run["ORT_threads"] == 4 and
          run["torch_interop_threads"] == run["ORT_interop_threads"] == 1 and run["providers"] == ["CPUExecutionProvider"] and
          run["ORT_optimization"] == "disabled" and run["ORT_version"] == candidate["ORT_version"])
    cache = run["cached_metrics_after_close"]
    check("four native correlation calls and released owned cache", cache["attempts"] == cache["completed_calls"] == 4 and cache["failed_calls"] == 0 and
          cache["closed"] and cache["current_context_bytes"] == 0 and cache["preparation"]["feature_inputs_copied"] and
          cache["preparation"]["native"]["neon"] == 1 and cache["library_sha256"] == build["output"]["sha256"])
    for model in manifest["models"].values():
        path = args.candidate_manifest.parent / model["file"]
        check("candidate graph " + model["file"] + " actual file hash", path.stat().st_size == model["bytes"] and sha(path) == model["sha256"])
    graph_item = next(row for row in motion["scenarios"] if row["name"] == "three_frame_all_pairs")["graph"]
    graph = artifact(args.motion_dir, graph_item)
    patch_item = motion["patches"]
    original_patch = artifact(args.motion_dir, patch_item)
    initial, final = artifact(args.run_dir, run["initial_state"]), artifact(args.run_dir, run["final_state"])
    topology = artifact(args.run_dir, run["topology_artifact"])
    check("original motion graph has six exact fields and no corr", set(graph) == {"ii", "jj", "kk", "coords", "ctx", "net0"})
    check("three actual capture/source frame IDs in order", run["frame_ids"] == motion["frame_ids"] == [row["frame_id"] for row in capture["inputs"]] and len(run["frame_ids"]) == 3)
    ii, jj, kk, K = (initial[name] for name in ("ii", "jj", "kk", "intrinsics"))
    expected_K = np.asarray([row["pixel_intrinsics"] for row in capture["inputs"]], np.float32)[None] / np.float32(4)
    check("initial identity/zero net/unit inverse-depth and original patches", initial["poses"].dtype == np.float32 and
          same_bytes(initial["poses"], np.broadcast_to(np.array([0,0,0,0,0,0,1], np.float32), (1,3,7))) and
          not np.any(initial["net"]) and same_bytes(initial["net"], graph["net0"]) and
          np.all(initial["patches"][:, :, 2] == 1) and same_bytes(initial["patches"], original_patch["patches"]))
    check("E864 original edge order/common RES4 intrinsics", same_bytes(ii, graph["ii"]) and same_bytes(jj, graph["jj"]) and same_bytes(kk, graph["kk"]) and
          np.array_equal(kk, np.repeat(np.arange(288, dtype=np.int64), 3)) and np.array_equal(jj, np.tile(np.arange(3, dtype=np.int64), 288)) and
          np.array_equal(ii, np.repeat(np.arange(3, dtype=np.int64), 96)[kk]) and same_bytes(K, expected_K) and np.all(K == K[:, :1]))
    computed_topology = expected_topology(ii, jj, kk)
    check("independent stable neighbors and complete grouped membership", set(topology) == set(computed_topology) and
          all(same_bytes(topology[name], computed_topology[name]) for name in computed_topology))
    for name, value in topology.items():
        item = meta(value)
        item["sha256"] = item.pop("tensor_sha256")
        check("topology summary raw metadata " + name, item == run["topology"]["arrays"][name])
    check("topology prepared once, correct capacities and no truncation", run["topology_built_once_outside_feedback"] and
          run["topology"]["edge_count"] == 864 and topology["kk_indices"].shape == (288,3) and topology["ij_indices"].shape == (9,96) and
          all(run["topology"]["groups"][key]["padding_slots"] == 0 for key in ("kk", "ij")))
    identity = next(row for row in update_capture["scenarios"] if row["name"] == "three_frame_all_pairs_identity")["states"][0]
    preflight = artifact(args.update_capture_dir, identity["inputs"])
    check("actual saved state0 preflight and no fake/stale inference corr", run["topology_preflight_input"]["sha256"] == identity["inputs"]["sha256"] and
          run["topology_preflight_input"]["arrays"] == identity["inputs"]["arrays"] and
          run["topology_preflight_input"]["used_only_for_input_validation_not_feedback"] and
          all(same_bytes(preflight[name], graph[key]) for name, key in (("net","net0"),("ctx","ctx"),("ii","ii"),("jj","jj"),("kk","kk"))))
    shapes = {"poses_before": (1,3,7), "poses_after": (1,3,7), "patches_before": (1,288,3,3,3), "patches_after": (1,288,3,3,3),
              "coords_before": (1,864,2,3,3), "coords_after": (1,864,2,3,3), "net_before": (1,864,384), "net_after": (1,864,384),
              "corr": (1,864,882), "delta": (1,864,2), "weight": (1,864,2), "target": (1,864,2),
              "projection_before": (1,864,2), "projection_after": (1,864,2), "residual_before": (1,864,2), "residual_after": (1,864,2)}
    previous = {"poses_after": initial["poses"], "patches_after": initial["patches"], "net_after": initial["net"]}
    previous_reference_end, summaries = None, []
    for iteration, row in enumerate(run["calls"]):
        arrays = artifact(args.run_dir, row["artifact"])
        saved_reference = artifact(args.run_dir, row["original_Torch_reference"]["artifact"])
        check("round%d pose/patch/ORT-net continuity" % iteration, row["iteration"] == iteration and
              row["geometry_revision_before"] == iteration and row["geometry_revision_after"] == iteration+1 and
              row["recurrent_net_carried_from_previous_ORT_round"] == (iteration > 0) and
              all(same_bytes(arrays[key+"_before"], previous[key+"_after"]) for key in ("poses","patches","net")))
        check("round%d exact shapes and finite FP32 outputs" % iteration, set(arrays) == set(shapes) | {"active_before","active_after"} and
              all(arrays[key].dtype == np.float32 and arrays[key].shape == shape and np.isfinite(arrays[key]).all() for key, shape in shapes.items()) and
              all(arrays[key].dtype == np.bool_ and arrays[key].shape == (864,) for key in ("active_before", "active_after")))
        check("round%d current target and saved projection/residual identities" % iteration,
              same_bytes(arrays["target"], arrays["coords_before"][:, :, :, 1, 1]+arrays["delta"]) and
              all(same_bytes(arrays["residual_"+key], arrays["target"]-arrays["projection_"+key]) for key in ("before","after")))
        check("round%d weights/fixed pose/patchxy/shared positive inverse-depth" % iteration, np.all((arrays["weight"] >= 0) & (arrays["weight"] <= 1)) and
              same_bytes(arrays["poses_after"][0,0], arrays["poses_before"][0,0]) and
              same_bytes(arrays["patches_after"][:, :, :2], arrays["patches_before"][:, :, :2]) and
              np.all(arrays["patches_after"][:, :, 2] > 0) and np.all(arrays["patches_after"][:, :, 2] == arrays["patches_after"][:, :, 2, :1, :1]))
        check("round%d BA two steps with original fixed interval" % iteration, row["ba"]["optimized_pose_interval"] == [1,3] and
              [x["iteration"] for x in row["ba"]["iterations"]] == [0,1] and row["ba"]["dense_reference"] and
              not row["ba"]["upstream_eff_impl_requested"] and not row["ba"]["CUDA_bitwise_equivalence_validated"] and
              not row["ba"]["VO_or_trajectory_accuracy_validated"] and row["all_feedback_state_finite"] and row["fixed_first_pose_preserved"])
        independent = reference_checks(row["original_Torch_reference"], arrays, saved_reference, graph)
        check("round%d same-current-input golden metadata and independent strict gates" % iteration, all(x["gate_passed"] for x in independent.values()))
        previous_reference_end = timing_checks(row, previous_reference_end)
        check("round%d ORT/reference timers separated and copied totals truthful" % iteration,
              row["ORT_update_timing"] == run["ORT_Update_calls"][iteration] and row["ORT_update_timing"]["index"] == iteration)
        events = [event for event in run["stage_events"] if event["iteration"] == iteration]
        check("round%d actual correlation/update/BA begin+completion events" % iteration,
              [e["stage"] for e in events] == ["correlation_begin", "update_begin", "BA_begin", "BA_completed"] and
              row["begin_monotonic"] <= events[0]["monotonic"] <= events[1]["monotonic"] <= events[2]["monotonic"] <= events[3]["monotonic"] <= row["end_monotonic"])
        before = residual_summary(arrays["residual_before"], arrays["active_before"], arrays["weight"])
        after = residual_summary(arrays["residual_after"], arrays["active_after"], arrays["weight"])
        fixed = residual_summary(arrays["residual_after"], arrays["active_after"], arrays["weight"], arrays["active_before"])
        check("round%d independent residual totals with both masks" % iteration, before == row["residual_before"] and
              after == row["residual_after"] and fixed == row["residual_after_using_pre_BA_active_mask"])
        summaries.append(dict(iteration=iteration, artifact_sha256=row["artifact"]["sha256"], reference_artifact_sha256=row["original_Torch_reference"]["artifact"]["sha256"],
                              current_net_before_sha256=meta(arrays["net_before"])["tensor_sha256"], current_corr_sha256=meta(arrays["corr"])["tensor_sha256"],
                              reference_checks=independent, seconds=row["seconds"], round_total_seconds=row["round_total_seconds"],
                              original_Torch_reference_seconds=row["original_Torch_reference"]["reference_forward_seconds"],
                              weight_range=[float(arrays["weight"].min()), float(arrays["weight"].max())],
                              inverse_depth_range=[float(arrays["patches_after"][:,:,2].min()),float(arrays["patches_after"][:,:,2].max())],
                              residual_before=before, residual_after_using_current_mask=after, residual_after_using_pre_BA_mask=fixed))
        previous = arrays
    check("exact final ORT net/pose/depth state", all(same_bytes(final[key], previous[key+"_after"]) for key in ("poses","patches","net")) and same_bytes(final["intrinsics"], K))
    totals = {key: sum(row["seconds"][key] for row in run["calls"]) for key in STAGES}
    feedback_seconds = sum(row["round_total_seconds"] for row in run["calls"])
    reference_seconds = sum(row["original_Torch_reference"]["reference_forward_seconds"] for row in run["calls"])
    check("feedback/reference independent totals, actual driver includes reference work", totals == run["stage_time_sum_seconds"] and
          feedback_seconds == run["round_time_sum_seconds"] and reference_seconds == run["original_Torch_reference_time_sum_seconds"] and
          run["actual_elapsed_includes_original_Torch_reference"] and run["actual_driver_elapsed_seconds"] >= feedback_seconds+reference_seconds)
    return dict(schema="v29_DPVO_ORT_feedback_saved_array_independent_review", status="passed", audit_sha256=sha(__file__),
                checked=len(checks), failed=0, checks=checks, source_sha256={name: digest for name, (_, digest) in source_files.items()},
                input_receipt_sha256=dict(run=sha(run_path), capture=CAPTURE_SHA, motion=MOTION_SHA, original_update_capture=UPDATE_CAPTURE_SHA,
                                         candidate_manifest=MANIFEST_SHA, candidate_ORT=run["candidate_evidence_sha256"],
                                         vectorized_geometry=run["prior_geometry_acceptance"]["receipt_sha256"], native_build=BUILD_SHA),
                host_operations="Only NumPy saved-array file/raw SHA, equality, topology ordering, target/residual algebra and FP32 threshold/statistics. No Torch/ORT/model import or execution, no native correlation, no reprojection/geometry execution or BA solve.",
                local_source_hashes_independently_checked=True,
                remaining_board_source_before_after="Board-reported hash equality; not all board-only library/source paths were rehashed on host.",
                parameter_scope="94 saved parameter raw-metadata hashes agree with frozen original CPU captures; no checkpoint tensors were loaded by this audit.",
                statistical_summary_portability=dict(relative_L2_scalar_only_rtol=1e-12, relative_L2_scalar_only_atol=1e-15,
                                                     elementwise_FP32_thresholds_and_failure_counts_unchanged=True),
                current_reference_scope="Saved input metadata and measured pinned source bind each golden call to the current ORT net_before/corr, original ctx/edge IDs. This is not a golden independent recurrent trajectory.",
                verified_run_NPZ_artifacts=11, feedback_rounds=4, ORT_Update_calls=4, original_Torch_reference_calls=4, BA_iterations=8,
                rounds=summaries, stage_seconds_sum=totals, total_feedback_seconds=feedback_seconds,
                total_original_Torch_reference_seconds=reference_seconds, actual_driver_elapsed_seconds=run["actual_driver_elapsed_seconds"],
                feedback_mean_seconds=feedback_seconds/4, CNN_executed=False, NPU_used=False,
                original_weight_tensor_count=94, peak_rss_mib=run["peak_rss_mib"], cache_closed=True, cache_bytes_after_close=0,
                ORT_session_released=True, neural_reference_released=True, no_cleanup_errors=True,
                full_VO_validated=False, trajectory_accuracy_validated=False, realtime_or_online_latency_validated=False,
                interpretation="Four fixed-capture recurrent ORT-to-geometry rounds pass the current-input original Torch strict gates. Feedback mean excludes CNN, loading/I/O and indispensable separate research reference calls; it is not online latency or a 20Hz/full-VO result.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "capture-dir", "motion-dir", "update-capture-dir", "v29-source-dir", "v28-source-dir",
                 "candidate-manifest", "candidate-evidence-dir", "geometry-evidence-dir", "build-receipt", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "New audit output required; immutable earlier receipts are not overwritten")
    result = audit(args)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: result[key] for key in ("status","checked","failed","total_feedback_seconds","total_original_Torch_reference_seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
