#!/usr/bin/env python3
"""NumPy-only saved M48 inputs/gold/topology audit; no model or SDK execution.

The original learned gold is bound to the once-called frozen parent, not
independently re-executed. Source-derived layout arithmetic retains MIT LICENSE.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
CAPTURE_SHA256 = "51b9d830762c1c1f701ca38a13378b874240a04f3d7e98181fa397ae1aa8a5d2"
PARENT_SHA256 = "84b8f02e14c0695c79d5c47319ff47ec4c7a8dcf53a3b38d905e03bc1efde6a8"
ADAPTER_SHA256 = "0f427739037e2178dddc4631a32fb602ac0a2ddc259e48e147028604ae6733f1"
STREAM_SHA256 = "295233513c1b670c953db25bbf4bfad8bdb72b2001f40ac3e1bf5d17083f9bde"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for value in iter(lambda: f.read(1 << 20), b""):
            h.update(value)
    return h.hexdigest()


def meta(x):
    return dict(shape=list(x.shape), dtype=str(x.dtype), bytes=x.nbytes,
                tensor_sha256=hashlib.sha256(x.tobytes(order="C")).hexdigest())


def digest(values):
    return hashlib.sha256(json.dumps({k: meta(v) for k, v in values.items()}, sort_keys=True).encode()).hexdigest()


def audit_capture(capture_dir, parent_dir, *, adapter_path=None):
    capdir, parentdir = Path(capture_dir), Path(parent_dir)
    adapter_path = Path(adapter_path) if adapter_path else Path(__file__).with_name("source_frame_batches.py")
    report = dict(schema="v36_saved_M48_capture_independent_audit", status="failed", checks=[], cases=[],
        auditor_sha256=sha(__file__), model_OR_BA_OR_ORT_OR_SDK_executed=False,
        learned_gold_independently_recomputed=False, fresh_trajectory_or_20Hz_validated=False,
        group_layout_identity_only=True, candidate_accepted=False)
    def check(condition, name):
        report["checks"].append(dict(name=name, passed=bool(condition)))
        if not condition:
            raise ValueError(name)
    def load_artifact(item):
        name = item["file"]
        check(Path(name).name == name and not (capdir / name).is_symlink(), "safe artifact basename")
        f = capdir / name
        check(f.stat().st_size == item["bytes"] and sha(f) == item["sha256"], "artifact file bytes and SHA")
        with np.load(f, allow_pickle=False) as z:
            values = {k: z[k].copy() for k in z.files}
        check({k: meta(v) for k, v in values.items()} == item["arrays"], "all raw array metadata and SHA")
        check(all(np.isfinite(v).all() for v in values.values()), "all saved values finite")
        return values
    try:
        check(sha(adapter_path) == ADAPTER_SHA256, "frozen degree adapter SHA")
        spec = importlib.util.spec_from_file_location("_v36_capture_audit_batches", adapter_path)
        adapter = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = adapter
        spec.loader.exec_module(adapter)
        adapter.load_pinned_plan()  # Exact frozen independent topology contract.
        cpath, ppath, mpath = capdir / "capture.json", parentdir / "run.json", parentdir / "input-manifest.json"
        cap, parent, manifest = (json.loads(v.read_text()) for v in (cpath, ppath, mpath))
        report["input_bindings"] = dict(capture_sha256=sha(cpath), parent_run_sha256=sha(ppath),
            parent_input_manifest_sha256=sha(mpath), adapter_sha256=sha(adapter_path))
        check(cap["schema"] == "v36_full_M48_original_Update_capture"
              and cap["status"] == "completed_original_M48_capture_not_performance_or_candidate", "complete capture identity")
        check(cap["script_sha256"] == CAPTURE_SHA256 and cap["parent_runner_sha256"] == PARENT_SHA256
              and cap["parent_stream_sha256"] == STREAM_SHA256, "capture and frozen parent source chain")
        check(cap["cleanup_errors"] == [] and cap["original_loader_and_stream_class_restored"] is True
              and cap["parent_runner_source_unchanged"] is True, "capture finally restore and source unchanged")
        check(cap["original_Update_replaced"] is False and cap["original_weights_or_RNG_changed"] is False
              and cap["upstream_revision_counters_exposed"] is False
              and cap["timing_is_ordinary_performance"] is False
              and cap["full_grouped_Update_or_NPU_candidate_executed"] is False, "unchanged callback and scope flags")
        check(parent["schema"] == "v35_DPVO_original_FAST_dense_spatial_RKNN_encoder_diagnostic_replay"
              and parent["script_sha256"] == PARENT_SHA256 and parent["status"].startswith("completed_"), "actual completed original parent")
        check(parent["cleanup_errors"] == [] and parent["sources_and_inputs_unchanged"] is True
              and parent["source_pins"] == parent["source_hashes_after"], "original parent source/inputs and cleanup")
        check(parent["original_weights_unchanged"] is True
              and len(parent["original_weights_before"]) == 94
              and parent["original_weights_before"] == parent["original_weights_after"], "original 94 weights metadata unchanged")
        check(parent["config"]["patches_per_frame"] == 48
              and parent["loop_closure"] is False, "M48 loop-disabled original configuration")
        check(cap["original_parent_receipt"]["run_sha256"] == sha(ppath)
              and cap["original_parent_receipt"]["run_bytes"] == ppath.stat().st_size
              and cap["original_parent_receipt"]["input_manifest_sha256"] == sha(mpath)
              and parent["input_manifest_sha256"] == sha(mpath), "exact original receipt and source manifest binding")
        calls = parent["update_calls"]
        check(cap["original_call_attempts"] == cap["original_call_successes"] == len(calls)
              == cap["original_parent_receipt"]["original_update_calls"], "once-called original callback accounting")
        check(all(c["index"] == i for i, c in enumerate(calls)), "ordered original callback indices")
        records = cap["records"]
        check(cap["captured"] == len(records) == len(cap["capture_observations"])
              and len({r["observation_id"] for r in records}) == len(records)
              and {r["observation_id"] for r in records} == set(cap["capture_observations"]), "all declared unique observations captured")
        check([r["actual_original_call_index"] for r in records] == sorted({r["actual_original_call_index"] for r in records}),
              "ordered unique gold callback indices")
        source_frames = manifest["frames"]
        check(parent["frames_completed"] == len(parent["frames"]) == len(source_frames), "complete original observation accounting")
        saved_bytes, used_files, previous_identity = 0, set(), None
        for record in records:
            check(record["phase"] == "online" and record["step_within_observation"] == 0
                  and record["before_BA"] is True and record["original_called_once"] is True
                  and record["callback_outputs_returned_unmodified"] is True
                  and record["fresh_pose_or_trajectory_validated"] is False, "selected first complete online pre-BA callback, no EOF substitute")
            index, observation, edges = record["actual_original_call_index"], record["observation_id"], record["edges"]
            check(0 <= index < len(calls) and calls[index]["edges"] == edges,
                  "gold bound to actual original callback and edge count")
            before, gold = load_artifact(record["inputs_artifact"]), load_artifact(record["original_gold_artifact"])
            for artifact in (record["inputs_artifact"], record["original_gold_artifact"]):
                check(artifact["file"] not in used_files, "unique saved case artifact")
                used_files.add(artifact["file"]); saved_bytes += artifact["bytes"]
            check(set(before) == {"net", "inp", "corr", "ii", "jj", "kk", "poses", "patches", "intrinsics", "frame_contexts", "retained_observation_ids"}
                  and set(gold) == {"net", "delta", "weight"}, "exact pre-Update and original gold keys")
            frames = record["retained_frames"]; n = len(frames)
            check(n > 0 and record["patches_per_frame"] == 48 and record["source"] == frames[-1]
                  and frames[-1]["observation_id"] == observation, "actual retained last source and M48")
            check(all(v.dtype == np.float32 for k, v in before.items() if k not in ("ii", "jj", "kk", "retained_observation_ids"))
                  and all(before[k].dtype == np.int64 for k in ("ii", "jj", "kk", "retained_observation_ids")), "saved original FP32/int64 dtypes")
            check(before["net"].shape == before["inp"].shape == gold["net"].shape == (1, edges, 384)
                  and before["corr"].shape == (1, edges, 882)
                  and gold["delta"].shape == gold["weight"].shape == (1, edges, 2)
                  and all(v.dtype == np.float32 for v in gold.values())
                  and np.all((gold["weight"] >= 0) & (gold["weight"] <= 1)), "full original gold shape/finite/weight domain")
            check(before["poses"].shape == (1, n, 7) and before["patches"].shape == (1, n * 48, 3, 3, 3)
                  and before["intrinsics"].shape == (1, n, 4)
                  and before["frame_contexts"].shape == (n, 48, 384), "actual geometry/context storage shapes")
            ids = [f["observation_id"] for f in frames]
            check(ids == sorted(set(ids)) and np.array_equal(before["retained_observation_ids"], np.array(ids, np.int64)),
                  "retained observation order and raw IDs")
            for frame in frames:
                i = frame["observation_id"]
                source, processed = source_frames[i], parent["frames"][i]
                check(frame["frame_id"] == source["frame_id"] == processed["frame_id"]
                      and frame["timestamp"] == source["timestamp"] == processed["timestamp"]
                      and frame["captured_monotonic"] == processed["arrival_monotonic"], "source timestamp and actual software admission identity")
            ii, jj, kk = (before[k] for k in ("ii", "jj", "kk"))
            check(ii.shape == jj.shape == kk.shape == (edges,)
                  and min(ii.min(), jj.min(), kk.min()) >= 0
                  and max(ii.max(), jj.max()) < n and kk.max() < n * 48, "actual graph index domain")
            expected = before["frame_contexts"].reshape(-1, 384)[kk][None]
            check(meta(expected) == meta(before["inp"]), "independent original ctx[kk] byte equality")
            identity = record["snapshot_identity"]
            check(identity["graph"]["sha256"] == digest(dict(ii=ii, jj=jj, kk=kk))
                  and identity["geometry"]["sha256"] == digest({k: before[k] for k in ("poses", "patches", "intrinsics")}), "actual topology/geometry revision content digests")
            feature_sha = hashlib.sha256((digest(dict(contexts=before["frame_contexts"]))
                + json.dumps(frames, sort_keys=True)).encode()).hexdigest()
            check(identity["features"]["sha256"] == feature_sha and identity["recurrent_revision"] == index,
                  "actual feature identities and recurrent callback revision")
            check(all(type(identity[k]["revision"]) is int and identity[k]["revision"] > 0
                      for k in ("graph", "geometry", "features")), "explicit capture-derived serials")
            if previous_identity is not None:
                check(all(identity[k]["revision"] >= previous_identity[k]["revision"]
                          and (identity[k]["sha256"] == previous_identity[k]["sha256"]
                               or identity[k]["revision"] > previous_identity[k]["revision"])
                          for k in ("graph", "geometry", "features")), "capture serial monotonicity across selected cases")
            previous_identity = identity
            plan = adapter.build_degree_plan(ii, jj, kk, patches_per_frame=48)
            budget = adapter.static_budget(plan)
            for name, value in [("input_" + k, before[k]) for k in ("net", "inp", "corr")] + [("gold_" + k, gold[k]) for k in ("net", "delta", "weight")]:
                restored = np.empty_like(value)
                for batch in plan.batches:
                    packed = adapter.pack_batch(value, batch)
                    check(packed.shape == batch.shape(value.shape[-1]), "actual CF pack shape " + name)
                    restored[0, batch.canonical_edge_indices.ravel()] = packed.transpose(0, 2, 3, 1).reshape(-1, value.shape[-1])
                check(meta(restored) == meta(value), "full edge pack/scatter byte identity " + name)
                del restored
            check(budget["FP32_full_inputs_bytes"] == sum(before[k].nbytes for k in ("net", "inp", "corr"))
                  and budget["FP32_full_outputs_bytes"] == sum(v.nbytes for v in gold.values()), "actual full input/output bytes independently summed")
            report["cases"].append(dict(observation_id=observation, original_call_index=index,
                edges=edges, retained_sources=n, topology=plan.metadata(), static_original_Linear_budget=budget,
                original_callback_seconds=record["original_callback_seconds"],
                capture_extra_seconds=record["selected_capture_overhead_seconds"],
                original_gold_reexecuted=False, grouped_neural_execution=False))
            del before, gold, expected
        check(saved_bytes == cap["saved_artifact_bytes"], "all selected artifact byte accounting")
        report.update(status="passed_saved_capture_identity_not_neural_equivalence", check_count=len(report["checks"]),
            original_callback_calls=len(calls), captured=len(records), saved_artifact_bytes=saved_bytes,
            actual_neural_grouped_numerics_or_timing_measured=False,
            parent_encoder_FP32_gate_passed=parent["historical_encoder_FP32_gate_passed"],
            timing_is_capture_diagnostic_not_ordinary_performance=True,
            revision_scope="Derived serial/content hashes; hidden intermediate serial increments not independently reconstructed")
    except Exception as e:
        report.update(error=repr(e), check_count=len(report["checks"]))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--parent-run-dir", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists(): raise ValueError("New audit receipt required")
    report = audit_capture(args.capture_dir, args.parent_run_dir, adapter_path=args.adapter)
    with args.output.open("x") as f: f.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report.get(k) for k in ("status", "check_count", "captured", "error")}))
    return 0 if report["status"].startswith("passed_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
