#!/usr/bin/env python3
"""Offline NumPy audit of saved original DPVO Update, wrapper and CPU ORT.

No Torch/ONNX/ORT import, checkpoint tensor load, neural execution or board
operation. A consistent failed numerical gate can pass this audit; numerical
acceptance is independently computed and reported separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import traceback

import numpy as np


RTOL, ATOL = 1e-4, 1e-5
COMMIT = "0ac95b656d1fda91c271d2a106460d19ad966fc7"
CAPTURE_SHA = "87e18137a828496d623721f7748e288d02f9d8d6879278da1e5d3d9731cfa0a1"
CORR_SHA = "0733a1544f2692af826fa22871a78cbafbe9fbfc1b828e53a6a1849a90c22804"
ADAPTER_SHA = "3f755c8e24d092fa55fdf698d87c3d9ba8944487ee95d4e5afc47554d31d1380"
CHECKPOINT_SHA = "30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12"
SOURCE_PINS = {
    "dpvo_probe_common.py": "ba47164a4273e0d392f8e23892f17be1aaf89b7ef42c3ebd746e3a7b3fd946a9",
    "dpvo_cpu_ops.py": "15191a0d481176e5be80bb6c41c3fb3f4d057d95b18ab18f6e0efcd73793dd32",
    "manifest.json": "ce2ecaf96b086f69ef4d90a2e07e7b81afe8ae094656eb41260182000a69e375",
}
OFFICIAL_SOURCE_PINS = {
    "dpvo/net.py": "1aee33d54f0b1cc1f3f842b60790ff1299ef9c1e20bca6467ce7114dca802acc",
    "dpvo/blocks.py": "9c9e2848da524829374e56bb1b14a891e9e2c19bc683af8e958e8982ba33d661",
    "dpvo/extractor.py": "f5cbcbe14c42e44bcec6e827a3429ebe93979a439ba9b9ef70d4f4baa3454a2b",
}
INPUT_NAMES = ("net", "inp", "corr", "previous_indices", "previous_valid",
               "following_indices", "following_valid", "kk_indices", "kk_valid", "kk_inverse",
               "ij_indices", "ij_valid", "ij_inverse")
OUTPUT_NAMES = ("net_out", "delta", "weight")
SCENARIOS = ("first_motion_0_to_1_identity", "first_motion_0_to_1_shifted",
             "three_frame_all_pairs_identity", "three_frame_all_pairs_shifted")
OUTPUT_KEYS = ("net", "delta", "weight")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def meta(a):
    return {"shape": list(a.shape), "dtype": str(a.dtype), "bytes": a.nbytes,
            "tensor_sha256": hashlib.sha256(a.tobytes(order="C")).hexdigest()}


def byte_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes(order="C") == b.tobytes(order="C")


def compare(a, b):
    if a.shape != b.shape or a.dtype != b.dtype or a.dtype != np.float32:
        raise ValueError("matching FP32 arrays required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("matching finite arrays required")
    difference = a.astype(np.float64) - b.astype(np.float64)
    outside = np.abs(difference) > ATOL + RTOL * np.abs(b.astype(np.float64))
    count = int(np.count_nonzero(outside))
    return {"rtol": RTOL, "atol": ATOL, "failed_elements": count, "gate_passed": count == 0,
            "max_abs": float(np.max(np.abs(difference))),
            "relative_l2": float(np.linalg.norm(difference.ravel()) /
                                 max(np.linalg.norm(b.astype(np.float64).ravel()), 1e-30)),
            "bitwise_equal": byte_equal(a, b)}


def expected_topology(ii, jj, kk, capacities):
    """NumPy oracle of actual IDs, stable ties and original int64 group hash.

    No scatter/softmax/Update evaluation. Padding is not membership: all E real
    edges must be partitioned once; capacities cannot truncate the graph.
    """
    if not (ii.dtype == jj.dtype == kk.dtype == np.int64 and ii.ndim == 1
            and ii.shape == jj.shape == kk.shape and ii.size > 0):
        raise ValueError("same nonempty int64 topology required")
    e = ii.size
    order = np.lexsort((np.arange(e), jj, kk))
    previous, following = np.full(e, -1, np.int64), np.full(e, -1, np.int64)
    for left, right in zip(order[:-1], order[1:]):
        if kk[left] == kk[right]:
            previous[right], following[left] = left, right
    arrays = {
        "previous_indices": np.maximum(previous, 0),
        "previous_valid": (previous >= 0).astype(np.float32).reshape(1, e, 1),
        "following_indices": np.maximum(following, 0),
        "following_valid": (following >= 0).astype(np.float32).reshape(1, e, 1),
    }
    groups = {}
    hashed = np.add(np.multiply(ii, np.int64(12345), dtype=np.int64), jj, dtype=np.int64)
    for name, labels in (("kk", kk), ("ij", hashed)):
        unique, inverse = np.unique(labels, return_inverse=True)
        g, l = capacities[name]
        if type(g) is not int or type(l) is not int or min(g, l) <= 0:
            raise ValueError("positive group capacity required")
        counts = np.bincount(inverse, minlength=unique.size)
        if g < unique.size or l < counts.max():
            raise ValueError("group capacity overflow")
        index, valid = np.zeros((g, l), np.int64), np.zeros((g, l), np.float32)
        for group in range(unique.size):
            members = np.flatnonzero(inverse == group)
            index[group, :members.size] = members
            valid[group, :members.size] = 1
        arrays[name + "_indices"], arrays[name + "_valid"] = index, valid
        arrays[name + "_inverse"] = inverse.astype(np.int64)
        groups[name] = {"labels": unique.tolist(), "group_sizes": counts.tolist() + [0] * (g - unique.size),
                        "actual_groups": int(unique.size), "capacity": [g, l],
                        "padding_slots": int(g * l - e)}
    return arrays, groups


class Audit:
    def __init__(self):
        self.checks, self.failures, self.files = 0, [], {}

    def check(self, condition, label):
        self.checks += 1
        if not bool(condition):
            self.failures.append(label)
        return bool(condition)

    def require(self, condition, label):
        if not self.check(condition, label):
            raise ValueError(label)

    def number(self, value, label):
        self.require(type(value) in (int, float) and math.isfinite(value) and value >= 0, label)

    def receipt(self, directory, name, digest):
        path = Path(directory) / name
        self.require(sha(path) == digest, name + " receipt SHA")
        self.files[str(path.resolve())] = digest
        row = json.loads(path.read_text())
        self.require(isinstance(row, dict), name + " object schema")
        return row

    def artifact(self, directory, item):
        name = item["file"]
        self.require(isinstance(name, str) and Path(name).name == name, "artifact basename")
        path = (Path(directory) / name).resolve(strict=True)
        self.require(path.is_relative_to(Path(directory).resolve()), name + " does not escape directory")
        self.require(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], name + " file size/SHA")
        self.files[str(path)] = item["sha256"]
        with np.load(path, allow_pickle=False) as loaded:
            arrays = {k: loaded[k].copy() for k in loaded.files}
        self.require({k: meta(v) for k, v in arrays.items()} == item["arrays"], name + " all raw tensor SHAs")
        for key, value in arrays.items():
            self.require(value.dtype.kind in "fiu" and np.isfinite(value).all(), name + "/" + key + " finite numeric")
        return arrays

    def reported_compare(self, got, reference, reported, label):
        calculated = compare(got, reference)
        self.require(set(reported) == set(calculated), label + " comparison fields")
        for key, value in calculated.items():
            if key in ("max_abs", "relative_l2"):
                self.check(type(reported[key]) in (int, float) and
                           math.isclose(reported[key], value, rel_tol=1e-12, abs_tol=1e-15), label + "/" + key)
            else:
                self.check(type(reported[key]) == type(value) and reported[key] == value, label + "/" + key)
        return calculated


def _unchanged(audit, row, label):
    audit.require(row.get("source_pins") == row.get("source_hashes_after")
                  and row.get("sources_and_inputs_unchanged") is True, label + " board-reported before/after pins")
    audit.require(row.get("cleanup_errors") == [] and row.get("affinity_restored") is True,
                  label + " board-reported clean cleanup/affinity")
    audit.require(row.get("actual_board_execution") is True and row.get("ORB_used") is False
                  and row.get("BA_executed") is False and row.get("valid_pose_output") is False
                  and row.get("production_changed") is False and row.get("realtime_validated") is False,
                  label + " execution/scope flags")


def _source_bindings(audit, row, label):
    pins = row["source_pins"]
    for name, expected in SOURCE_PINS.items():
        audit.require(any(Path(p).name == name and s == expected for p, s in pins.items()),
                      label + " source " + name)
    audit.require(any(Path(p).name == "dpvo.pth" and s == CHECKPOINT_SHA for p, s in pins.items()),
                  label + " official checkpoint pin")
    loader = row["loader"]
    audit.require(loader["repo"] == "princeton-vl/DPVO" and loader["commit"] == COMMIT
                  and loader["manifest_sha256"] == SOURCE_PINS["manifest.json"], label + " official source identity")
    for name, expected in OFFICIAL_SOURCE_PINS.items():
        audit.require(loader["source_hashes"][name]["sha256"] == expected, label + " original source " + name)
    audit.require(loader["original_Update_class_AST_preserved"] is True and
                  loader["sources_and_checkpoint_unchanged"] is True, label + " original source/AST before-after claims")
    audit.require(loader["checkpoint"] == {"sha256": CHECKPOINT_SHA, "bytes": 14167743}
                  and loader["weights_only"] is True and loader["map_location"] == "cpu", label + " loading provenance")
    state = loader["load_state"]
    audit.require(state["strict"] is True and state["inference_parameter_coverage"] == 1.0
                  and state["all_loaded_parameter_bytes_equal"] is True and state["released_state_keys"] == 98
                  and state["inference_state_keys"] == 94 and state["inference_parameters"] == 3384324
                  and state["exact_dropped_legacy_keys"] == ["update.lmbda.0.weight", "update.lmbda.0.bias",
                                                            "update.lmbda.2.weight", "update.lmbda.2.bias"],
                  label + " strict 98-to-94 state coverage")
    weights = row["original_weights_before"]
    audit.require(len(weights) == 94 and sum(np.prod(v["shape"]) for v in weights.values()) == 3384324
                  and sum(k.startswith("update.") for k in weights) == 50,
                  label + " complete parameter manifest dimensions")
    for name, info in weights.items():
        loaded = state["loaded_tensors"][name]
        audit.require(info["dtype"] == "float32" and info["bytes"] == int(np.prod(info["shape"])) * 4
                      and info["shape"] == loaded["shape"] and info["tensor_sha256"] == loaded["tensor_sha256"],
                      label + " loaded parameter manifest " + name)
    audit.require(row["original_weights_unchanged"] is True, label + " board-reported unchanged parameters")


def _outputs(audit, values, edges, label, *, exact_sigmoid_range=True):
    audit.require(set(values) == set(OUTPUT_KEYS), label + " output keys")
    for key in OUTPUT_KEYS:
        audit.require(values[key].shape == (1, edges, 384 if key == "net" else 2)
                      and values[key].dtype == np.float32, label + "/" + key + " shape/dtype")
    weight = values["weight"]
    bounds = {"min": float(weight.min()), "max": float(weight.max()),
              "negative_elements": int(np.count_nonzero(weight < 0)),
              "above_one_elements": int(np.count_nonzero(weight > 1)),
              "exact_mathematical_range": bool(((weight >= 0) & (weight <= 1)).all())}
    if exact_sigmoid_range:
        audit.require(bounds["exact_mathematical_range"], label + " original sigmoid weight range")
    return bounds


def _call_times(audit, calls, key, reported_mean, label):
    audit.require(len(calls) == 7, label + " four zero-state plus three recurrent calls")
    for call in calls:
        audit.number(call[key], label + " nonnegative finite call time")
    mean = sum(c[key] for c in calls[1:4]) / 3
    audit.check(math.isclose(mean, reported_mean, rel_tol=1e-12, abs_tol=1e-15), label + " three warm-call mean")
    return mean


def audit_capture(audit, capture_dir, capture, corr_dir, corr):
    audit.require(capture["schema"] == "v28_DPVO_original_Update_capture" and
                  capture["status"] == "captured_original_Update_states_not_VO", "original capture terminal status")
    _unchanged(audit, capture, "capture")
    _source_bindings(audit, capture, "capture")
    audit.require(corr["status"] == "completed_native_corr_probe_not_VO" and
                  corr["exact_FP32_gate_passed"] is True and capture["corr_receipt_sha256"] == CORR_SHA,
                  "v27 original reference link")
    _unchanged(audit, corr, "v27 reference")
    audit.require(capture["original_weights_before"] == corr["original_weights_before"],
                  "v27-to-v28 exact parameter manifest")
    audit.require([s["name"] for s in capture["scenarios"]] == list(SCENARIOS)
                  and [s["name"] for s in corr["scenarios"]] == list(SCENARIOS), "four exact original scenarios")
    results, all_states = [], {}
    for case, old in zip(capture["scenarios"], corr["scenarios"]):
        name, edges = case["name"], case["edges"]
        audit.require(edges == old["edge_count"] and edges == (96 if name.startswith("first") else 864), name + " E bucket")
        graph = audit.artifact(corr_dir, old["graph"])
        reference_row = next(c for c in old["calls"] if c["mode"] == "reference" and c["iteration"] == 0)
        old_values = audit.artifact(corr_dir, reference_row["artifact"])
        audit.require([s["step"] for s in case["states"]] == [0, 1, 2, 3], name + " all four captured states")
        states = []
        for state in case["states"]:
            step = state["step"]
            inputs, outputs = audit.artifact(capture_dir, state["inputs"]), audit.artifact(capture_dir, state["outputs"])
            audit.require(set(inputs) == {"net", "ctx", "corr", "ii", "jj", "kk"}, name + " input keys")
            for key in ("net", "ctx", "corr"):
                audit.require(inputs[key].shape == (1, edges, 882 if key == "corr" else 384)
                              and inputs[key].dtype == np.float32, name + "/" + key + " input shape/dtype")
            for key in ("ii", "jj", "kk"):
                audit.require(inputs[key].shape == (edges,) and inputs[key].dtype == np.int64
                              and byte_equal(inputs[key], graph[key]), name + "/" + key + " original topology binding")
            audit.require(byte_equal(inputs["ctx"], graph["ctx"]) and byte_equal(inputs["corr"], old_values["corr"]),
                          name + " frozen original context/correlation")
            _outputs(audit, outputs, edges, name + "/state" + str(step))
            if step == 0:
                audit.require(byte_equal(inputs["net"], graph["net0"]) and not np.count_nonzero(inputs["net"]),
                              name + " true original zero-state input")
                for key in OUTPUT_KEYS:
                    audit.require(byte_equal(outputs[key], old_values[key]), name + "/" + key + " bitwise original v27 baseline")
            else:
                audit.require(byte_equal(inputs["net"], states[-1]["outputs"]["net"]),
                              name + "/state" + str(step) + " true previous original net feedback")
                audit.require(np.count_nonzero(inputs["net"]) > 0, name + " nonzero recurrent state")
            states.append({"inputs": inputs, "outputs": outputs})
        calls = case["calls"]
        mean = _call_times(audit, calls, "seconds", case["warm_mean_seconds"], name + "/original")
        for index, call in enumerate(calls):
            step, repetition = (0, index) if index < 4 else (index - 3, 0)
            role = "cold" if index == 0 else "warm_same_state" if index < 4 else "recurrent_same_corr_single"
            audit.require((call["state_step"], call["repetition"], call["timing_role"]) == (step, repetition, role),
                          name + " original call identity/role")
            audit.require(call["arrays"] == {k: meta(v) for k, v in states[step]["outputs"].items()},
                          name + " original repeat/state output raw metadata")
        profile = case["profile"]
        stages = profile["nonoverlapping_stage_seconds"]
        audit.require(set(stages) == {"corr", "norm", "c1", "c2", "agg_kk", "agg_ij", "gru", "d", "w"},
                      name + " nine nonnested profile components")
        for value in [profile["total_seconds"], profile["other_and_hook_overhead_seconds"], *stages.values()]:
            audit.number(value, name + " profile durations")
        audit.check(math.isclose(sum(stages.values()) + profile["other_and_hook_overhead_seconds"],
                                 profile["total_seconds"], rel_tol=1e-12, abs_tol=1e-12), name + " profile partition sum")
        audit.require(profile["bitwise_equal_uninstrumented"] is True and profile["instrumented_not_in_warm_mean"] is True,
                      name + " separate instrumented call (no separately saved profile array)")
        results.append({"name": name, "edges": edges, "states": 4, "normal_calls": 7,
                        "cold_seconds": calls[0]["seconds"], "warm_mean_seconds": mean,
                        "recurrent_single_seconds": [c["seconds"] for c in calls[4:]], "separate_profile": profile})
        all_states[name] = states
    return results, all_states


def audit_export(audit, export_dir, exported, capture, capture_sha, states):
    audit.require(exported["schema"] == "v28_DPVO_full_Update_export"
                  and exported["status"] == "exported_Update_board_ORT_pending", "export terminal diagnostic status")
    _unchanged(audit, exported, "export")
    _source_bindings(audit, exported, "export")
    audit.require(exported["original_weights_before"] == capture["original_weights_before"], "export exact original parameters")
    audit.require(exported["capture_sha256"] == capture_sha and exported["adapter_sha256"] == ADAPTER_SHA
                  and exported["input_names"] == list(INPUT_NAMES) and exported["output_names"] == list(OUTPUT_NAMES),
                  "frozen adapter/capture/13input contract")
    audit.require(set(exported["models"]) == {"E96", "E864"}, "two fixed E artifacts")
    models = {}
    for key, model in exported["models"].items():
        path = Path(export_dir) / model["file"]
        audit.require(Path(model["file"]).name == model["file"] and path.stat().st_size == model["bytes"]
                      and sha(path) == model["sha256"], key + " ONNX file size/SHA (no host graph execution)")
        audit.files[str(path.resolve())] = model["sha256"]
        audit.require([v["name"] for v in model["inputs"]] == list(INPUT_NAMES)
                      and [v["name"] for v in model["outputs"]] == list(OUTPUT_NAMES)
                      and model["runtime_topology_inputs_preserved"] is True, key + " reported external topology contract")
        for item in model["inputs"]:
            expected_dtype = 7 if item["name"].endswith(("_indices", "_inverse")) else 1
            audit.require(item["dtype"] == expected_dtype, key + "/" + item["name"] + " reported FP32/int64 type")
        models[key] = {"sha256": model["sha256"], "bytes": model["bytes"], "nodes": model["nodes"],
                       "initializers": model["initializers"], "input_dtype_counts": {"FP32": 7, "int64": 6}}
    audit.require([s["name"] for s in exported["scenarios"]] == list(SCENARIOS), "four export scenarios")
    results, all_wrapped, all_gates = [], {}, []
    for row in exported["scenarios"]:
        name, edges = row["name"], row["edges"]
        audit.require(row["model_key"] == "E" + str(edges), name + " shared model bucket")
        topo = audit.artifact(export_dir, row["topology"])
        audit.require(set(topo) == set(INPUT_NAMES[3:]), name + " exactly ten external topology arrays")
        cap = {k: list(topo[k + "_indices"].shape) for k in ("kk", "ij")}
        source = states[name][0]["inputs"]
        expected, groups = expected_topology(source["ii"], source["jj"], source["kk"], cap)
        for key in expected:
            audit.require(byte_equal(topo[key], expected[key]), name + "/" + key + " independent original topology oracle")
        audit.require(row["topology_bytes"] == sum(v.nbytes for v in topo.values()), name + " topology byte count")
        for group in groups:
            recorded = row["topology_summary"]["groups"][group]
            audit.require(all(recorded[k] == v for k, v in groups[group].items()), name + "/" + group + " membership summary")
        model = exported["models"][row["model_key"]]
        shapes = {"net": list(source["net"].shape), "inp": list(source["ctx"].shape), "corr": list(source["corr"].shape),
                  **{k: list(v.shape) for k, v in topo.items()}}
        audit.require(all(v["shape"] == shapes[v["name"]] for v in model["inputs"]), name + " exact static capacity shapes")
        audit.number(row["topology_prepare_seconds"], name + " separately timed CPU topology generation")
        audit.require([s["step"] for s in row["states"]] == [0, 1, 2, 3], name + " wrapper states")
        values, checks = [], []
        for state in row["states"]:
            step = state["step"]
            outputs = audit.artifact(export_dir, state["outputs"])
            _outputs(audit, outputs, edges, name + "/wrapper")
            audit.number(state["forward_seconds"], name + " wrapper single-call time")
            calculated = {k: audit.reported_compare(outputs[k], states[name][step]["outputs"][k], state["checks"][k],
                                                   name + "/wrapper/state" + str(step) + "/" + k) for k in OUTPUT_KEYS}
            all_gates.extend(c["gate_passed"] for c in calculated.values())
            checks.append({"step": step, "forward_seconds": state["forward_seconds"], "versus_original": calculated})
            values.append(outputs)
        results.append({"name": name, "edges": edges, "topology_bytes": row["topology_bytes"],
                        "topology_prepare_seconds": row["topology_prepare_seconds"], "groups": groups, "states": checks})
        all_wrapped[name] = values
    gate = all(all_gates)
    audit.check(exported["exact_wrapper_FP32_gate_passed"] is gate, "independent aggregate wrapper gate agrees")
    audit.check(exported["numeric_validation"] == ("wrapper_passed" if gate else "wrapper_failed_preserved_as_diagnostic"),
                "wrapper diagnostic classification matches real gate")
    return {"scenarios": results, "models": models, "exact_original_FP32_gate_passed": gate}, all_wrapped


def audit_ort(audit, ort_dir, ort, exported, export_sha, states, wrapped):
    audit.require(ort["schema"] == "v28_DPVO_full_Update_board_ORT"
                  and ort["status"] == "completed_Update_ORT_probe_not_VO", "ORT completed diagnostic status")
    _unchanged(audit, ort, "ORT")
    audit.require(ort["capture_sha256"] == CAPTURE_SHA and ort["export_sha256"] == export_sha
                  and ort["wrapper_FP32_gate_passed"] is exported["exact_wrapper_FP32_gate_passed"], "ORT chain/gate bindings")
    audit.require([m["name"] for m in ort["modes"]] == ["disabled", "enabled"], "both CPU ORT optimization modes")
    results, aggregate = [], []
    for mode in ort["modes"]:
        original_gates, wrapper_gates, cases = [], [], []
        audit.require(set(mode["sessions"]) == {"E96", "E864"}, mode["name"] + " two sessions")
        for key, session in mode["sessions"].items():
            audit.require(session["providers"] == ["CPUExecutionProvider"]
                          and session["model_sha256"] == exported["models"][key]["sha256"], mode["name"] + "/" + key + " model/provider")
            audit.require([s["name"] for s in session["inputs"]] == list(INPUT_NAMES), mode["name"] + " complete input names")
            for item, expected in zip(session["inputs"], exported["models"][key]["inputs"]):
                typ = "tensor(int64)" if expected["dtype"] == 7 else "tensor(float)"
                audit.require(item["shape"] == expected["shape"] and item["type"] == typ, mode["name"] + " actual ORT input shape/type")
            audit.number(session["init_seconds"], mode["name"] + " initialization separately reported")
        audit.require([s["name"] for s in mode["scenarios"]] == list(SCENARIOS), mode["name"] + " four cases")
        for case in mode["scenarios"]:
            name, edges = case["name"], case["edges"]
            mean = _call_times(audit, case["calls"], "forward_seconds", case["warm_mean_seconds"], mode["name"] + "/" + name)
            stored, checks, ranges = {}, [], []
            for index, call in enumerate(case["calls"]):
                step, repetition = (0, index) if index < 4 else (index - 3, 0)
                role = "cold_case" if index == 0 else "warm_same_state" if index < 4 else "nonzero_captured_state_single"
                audit.require((call["step"], call["repetition"], call["timing_role"]) == (step, repetition, role), name + " ORT call identity")
                audit.number(call["feed_prepare_seconds"], name + " feed preparation excluded/separate")
                if "outputs" in call:
                    audit.require(repetition == 0, name + " actual raw array is first repeat only")
                    stored[step] = audit.artifact(ort_dir, call["outputs"])
                    bounds = _outputs(audit, stored[step], edges, name + "/ORT", exact_sigmoid_range=False)
                    ranges.append({"step": step, **bounds})
                audit.require(step in stored and call["repeat_bitwise_equal"] is True
                              and call["arrays"] == {k: meta(v) for k, v in stored[step].items()},
                              name + " repeated metadata tied to saved exact array")
                a = stored[step]
                original = {k: audit.reported_compare(a[k], states[name][step]["outputs"][k], call["versus_original"][k],
                                                      mode["name"] + "/" + name + "/" + k + "/original")
                            for k in OUTPUT_KEYS}
                wrapper = {k: audit.reported_compare(a[k], wrapped[name][step][k], call["versus_wrapper"][k],
                                                     mode["name"] + "/" + name + "/" + k + "/wrapper")
                           for k in OUTPUT_KEYS}
                original_gates.extend(x["gate_passed"] for x in original.values())
                wrapper_gates.extend(x["gate_passed"] for x in wrapper.values())
                if repetition == 0:
                    checks.append({"step": step, "versus_original": original, "versus_wrapper": wrapper})
            audit.require(set(stored) == {0, 1, 2, 3}, name + " four independently saved ORT states")
            cases.append({"name": name, "edges": edges, "calls": 7, "raw_saved_calls": 4,
                          "same-state_extra_repeats_validated_via_recorded_rawSHA": 3,
                          "cold_seconds": case["calls"][0]["forward_seconds"], "warm_mean_seconds": mean,
                          "warm_seconds": [c["forward_seconds"] for c in case["calls"][1:4]],
                          "nonzero_single_seconds": [c["forward_seconds"] for c in case["calls"][4:]],
                          "output_weight_range": ranges, "checks": checks})
        original_gate, wrapper_gate = all(original_gates), all(wrapper_gates)
        audit.check(mode["exact_original_FP32_gate_passed"] is original_gate, mode["name"] + " original aggregate gate")
        audit.check(mode["exact_wrapper_FP32_gate_passed"] is wrapper_gate, mode["name"] + " wrapper aggregate gate")
        aggregate.append(original_gate)
        results.append({"name": mode["name"], "scenarios": cases, "normal_calls": 28, "raw_saved_calls": 16,
                        "exact_original_FP32_gate_passed": original_gate, "exact_wrapper_FP32_gate_passed": wrapper_gate})
    overall = all(aggregate)
    audit.check(ort["exact_original_FP32_gate_passed"] is overall, "independent total ORT gate agrees")
    return {"modes": results, "exact_original_FP32_gate_passed": overall,
            "feedback_drift": "not_exercised; every nonzero input is an original CPU captured state"}


def run(args):
    audit = Audit()
    report = {"schema": "v28_saved_DPVO_Update_independent_NumPy_audit", "status": "failed",
              "auditor_sha256": sha(__file__), "host_model_OR_ORT_execution": False, "board_operations": False,
              "raw_artifacts_modified": False, "predeclared_gate": {"rtol": RTOL, "atol": ATOL},
              "NPU_executed": False, "BA_or_pose_or_complete_VO_validated": False, "realtime_validated": False}
    try:
        audit.require(args.capture_sha256 == CAPTURE_SHA and args.corr_sha256 == CORR_SHA, "frozen original capture/corr receipt pins")
        capture = audit.receipt(args.capture_dir, "capture.json", args.capture_sha256)
        corr = audit.receipt(args.corr_dir, "run.json", args.corr_sha256)
        report["receipt_hashes"] = {"capture": args.capture_sha256, "v27_corr_reference": args.corr_sha256}
        report["original"], states = audit_capture(audit, args.capture_dir, capture, args.corr_dir, corr)
        if args.export_dir is not None:
            exported = audit.receipt(args.export_dir, "export.json", args.export_sha256)
            report["receipt_hashes"]["export"] = args.export_sha256
            report["wrapper"], wrapped = audit_export(audit, args.export_dir, exported, capture, args.capture_sha256, states)
            if args.ort_dir is not None:
                ort = audit.receipt(args.ort_dir, "ort.json", args.ort_sha256)
                report["receipt_hashes"]["ORT"] = args.ort_sha256
                report["ORT"] = audit_ort(audit, args.ort_dir, ort, exported, args.export_sha256, states, wrapped)
        if args.local_source_dir is not None:
            source_dir = Path(args.local_source_dir)
            for receipt in (capture, exported if args.export_dir is not None else None,
                            ort if args.ort_dir is not None else None):
                if receipt is None:
                    continue
                candidates = {Path(p).name: s for p, s in receipt["source_pins"].items()}
                for name in ("run-v28-dpvo-update-capture-board.py", "run-v28-dpvo-update-export-board.py",
                             "run-v28-dpvo-update-ort-board.py", "dpvo_update_export.py"):
                    if name in candidates:
                        audit.require(sha(source_dir / name) == candidates[name], "local source rehash " + name)
        for path, digest in audit.files.items():
            audit.check(sha(path) == digest, "audited input unchanged after analysis: " + Path(path).name)
        report["status"] = "passed" if not audit.failures else "failed"
    except Exception as exc:
        report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
    report.update(checks=audit.checks, failures=audit.failures,
                  independently_rehashed_files=[{"file": Path(p).name, "sha256": h} for p, h in audit.files.items()])
    report["limitations"] = [
        "Original parameter manifests/board before-after source pins are checked for consistency; no checkpoint tensor load/network rerun.",
        "All saved arrays independently hashed. Extra identical-call repetitions have metadata only; no extra raw arrays saved. Instrumented capture profile equality is board-reported.",
        "Original profile has separate nonnested hooks. Partition arithmetic verified; timestamps/events proving each interval were not recorded.",
        "Warm means exclude external topology, graph initialization, file I/O and checks; not full DPVO/pose/VO/camera latency or sustained throughput.",
        "Independent numerical false is preserved even if audit checks pass; completion is not numerical acceptance.",
        "ORT may have small negative sigmoid weights within the unchanged FP32 comparison tolerance. Exact [0,1] bounds are reported separately; no clipping or full-VO/BA acceptance is claimed."
    ]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "corr"):
        parser.add_argument("--" + name + "-dir", type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    for name in ("export", "ort"):
        parser.add_argument("--" + name + "-dir", type=Path)
        parser.add_argument("--" + name + "-sha256")
    parser.add_argument("--local-source-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    for name in ("export", "ort"):
        if (getattr(args, name + "_dir") is None) != (getattr(args, name + "_sha256") is None):
            parser.error("--" + name + "-dir and --" + name + "-sha256 required together")
    if args.ort_dir is not None and args.export_dir is None:
        parser.error("ORT audit requires export receipt/artifacts")
    if args.output.exists():
        parser.error("Output must be a new JSON file")
    report = run(args)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: report.get(k) for k in ("status", "checks", "failures", "error")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
