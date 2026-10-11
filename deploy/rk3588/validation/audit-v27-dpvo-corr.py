#!/usr/bin/env python3
"""Audit saved DPVO native-correlation probes using NumPy only.

No model, shared library, image, Torch or device is executed. Whole saved
correlation/Update arrays are compared, and seven declared edges per graph
are checked against an independently implemented mathematical oracle.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
PINS = {
    "capture.json": "e0563efb962edb90684480be114b88bc2c7825015f3b13a163f6f47f06360088",
    "motion.json": "55a7db19d1a9bfbf986da62ed999fc8b122add0fca84f01681e6ad0b82ba4e86",
    "run-v27-dpvo-corr-board.py": "16c1f6c77b20710a0689f026aef44bc1f96482dac66158bb38148542398b0183",
    "corr_numpy_oracle.py": "738a5e860910ba38d30c9d36b4245530e2ab303627877f761b0c07d0e1beef3b",
    "dpvo_corr.cpp": "3cf0c9cf79754fff996efc03c2ca2b162733f10b2b9aa580dee263e5df5b9990",
    "dpvo_corr.h": "1866c747265b401f8d29d896e5f634ac104f238704e5fe443aa683a7395ab932",
    "dpvo_corr.py": "a6c37bf4b840b492736d5c5916be6e2c0e8680693d01d0fe3a5fe62702ecb779",
}
MODES = ("reference", "native_one_shot", "native_cached")
KEYS = ("corr", "net", "delta", "weight")
checks = 0
files = {}


def require(condition, label):
    global checks
    if not condition:
        raise ValueError(label)
    checks += 1


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verified(path, expected=None, size=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "regular file: " + path.name)
    sha = digest(path)
    if expected is not None:
        require(sha == expected, "file SHA: " + path.name)
    if size is not None:
        require(path.stat().st_size == size, "file bytes: " + path.name)
    files[str(path.resolve())] = sha
    return path


def read_json(path, expected=None):
    return json.loads(verified(path, expected).read_text(encoding="utf-8"))


def metadata(array):
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "bytes": array.nbytes,
            "tensor_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}


def artifact(folder, record, array_metadata=None):
    name = record["file"]
    require(Path(name).name == name and name not in ("", ".", ".."), "safe artifact name")
    path = verified(Path(folder) / name, record["sha256"], record["bytes"])
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    expected = record.get("arrays") if array_metadata is None else array_metadata
    require(set(arrays) == set(expected), "artifact keys: " + name)
    for key, value in arrays.items():
        require(metadata(value) == expected[key], "array metadata: " + name + ":" + key)
        require(np.isfinite(value).all(), "finite array: " + name + ":" + key)
    return arrays


def same_bytes(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def close_number(a, b, label):
    require(math.isfinite(a) and math.isfinite(b) and
            math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12), label)


def comparison(a, reference):
    require(a.shape == reference.shape, "comparison shape")
    x, y = a.astype(np.float64), reference.astype(np.float64)
    delta = x - y
    denominator = float(np.linalg.norm(y.ravel()))
    return {"rtol": 1e-4, "atol": 1e-5,
            "failed_elements": int(np.count_nonzero(np.abs(delta) > 1e-5 + 1e-4 * np.abs(y))),
            "gate_passed": bool(np.all(np.abs(delta) <= 1e-5 + 1e-4 * np.abs(y))),
            "max_abs": float(np.max(np.abs(delta), initial=0)),
            "relative_l2": float(np.linalg.norm(delta.ravel()) / denominator) if denominator else None,
            "bitwise_equal": same_bytes(a, reference)}


def recorded_comparison(actual, record):
    for key, value in actual.items():
        if key in ("max_abs", "relative_l2") and value is not None:
            close_number(value, record[key], "recorded " + key)
        else:
            require(value == record[key], "recorded " + key)


def audit(args):
    run = read_json(args.run_dir / "run.json")
    capture = read_json(args.capture_dir / "capture.json", PINS["capture.json"])
    motion = read_json(args.motion_dir / "motion.json", PINS["motion.json"])
    build = read_json(args.build_receipt, run["build_sha256"])
    oracle_path = verified(args.native_source_dir / "corr_numpy_oracle.py", PINS["corr_numpy_oracle.py"])
    spec = importlib.util.spec_from_file_location("independent_corr_oracle", oracle_path)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    require(run["status"] == "completed_native_corr_probe_not_VO", "completed probe")
    require(run["schema"] == "v27_DPVO_native_corr_board_probe", "run schema")
    require(run["script_sha256"] == PINS["run-v27-dpvo-corr-board.py"], "measured driver")
    require(run["capture_sha256"] == PINS["capture.json"] and run["motion_sha256"] == PINS["motion.json"], "baseline bindings")
    require(run["actual_board_execution"] is True, "actual board record")
    for key in ("production_changed", "ORB_used", "NPU_used", "BA_executed", "valid_pose_output", "tracked_trajectory", "realtime_validated"):
        require(run[key] is False, "scope: " + key)
    require(run["cleanup_errors"] == [] and run["affinity_restored"] is True, "terminal cleanup")
    require(run["cpu_affinity"] == [4, 5, 6, 7] and run["torch_threads"] == 4 and run["torch_interop_threads"] == 1, "CPU configuration")
    require(run["sources_and_inputs_unchanged"] is True and run["source_pins"] == run["source_hashes_after"], "source after boundary")
    require(run["original_weights_before"] == capture["state_before"] == motion["original_weights_before"], "94 original parameter metadata")
    require(len(run["original_weights_before"]) == 94 and run["original_weights_unchanged"] is True, "all original weights unchanged")
    require(run["loader"]["original_Update_class_AST_preserved"] is True, "original Update definition")
    require(build["status"] == "compiled_and_synthetic_smoke_passed" and build["exit_code"] == 0, "native build completed")
    require(build["source_sha256"] == build["source_sha256_after"], "native build source after")
    require(build["output"]["sha256"] == run["library_sha256"], "native library binding")
    for flag in ("-fopenmp", "-fno-fast-math", "-ffp-contract=off"):
        require(flag in build["command"], "native compiler flag: " + flag)
    verified(args.build_receipt.parent / build["output"]["file"], run["library_sha256"], build["output"]["bytes"])
    for name, sha in build["source_sha256"].items():
        require(sha == PINS[name], "frozen native source: " + name)
        verified(args.native_source_dir / name, sha)
    local_sources = {
        "run-v27-dpvo-corr-board.py": args.driver,
        "build.json": args.build_receipt,
        "libdpvo_corr.so": args.build_receipt.parent / "libdpvo_corr.so",
        "dpvo.pth": args.checkpoint,
    }
    for board_path, sha in run["source_pins"].items():
        name = Path(board_path).name
        local = local_sources.get(name)
        if local is None:
            candidates = [folder / name for folder in (args.native_source_dir, args.v26_source_dir, args.capture_dir, args.motion_dir)]
            matches = [p for p in candidates if p.is_file()]
            require(len(matches) == 1, "unique local measured source: " + name)
            local = matches[0]
        verified(local, sha)

    feature_parts = []
    for call_index, frame_index in ((0, 0), (4, 1), (5, 2)):
        call = capture["calls"][call_index]
        require(call["call_index"] == call_index and call["frame_index"] == frame_index, "captured feature order")
        feature_parts.append(artifact(args.capture_dir, call["artifact"], call["arrays"])["fmap"])
    fmap = np.concatenate(feature_parts, axis=1)
    patches = artifact(args.motion_dir, motion["patches"])
    gmap = patches["gmap"]
    require(metadata(fmap) == run["features"]["fmap"] and metadata(gmap) == run["features"]["gmap"], "original features raw SHA")
    key = metadata(fmap)["tensor_sha256"] + ":" + metadata(gmap)["tensor_sha256"]
    require(key == run["features"]["cache_key"] and run["features"]["frame_ids"] == motion["frame_ids"], "owned cache feature identity")
    context_bytes = gmap.nbytes + fmap.nbytes + math.prod((*fmap.shape[:-2], fmap.shape[-2] // 4, fmap.shape[-1] // 4)) * 4
    expected_names = [parent + suffix for parent in ("first_motion_0_to_1", "three_frame_all_pairs") for suffix in ("_identity", "_shifted")]
    require([s["name"] for s in run["scenarios"]] == expected_names, "four graph coverage/order")
    results = []
    total_elements = 0
    for scenario in run["scenarios"]:
        parent = next(s for s in motion["scenarios"] if s["name"] == scenario["parent"])
        original_graph = artifact(args.motion_dir, parent["graph"])
        graph = artifact(args.run_dir, scenario["graph"])
        edges = scenario["edge_count"]
        require(edges == parent["edge_count"] and edges in (96, 864), "edge count")
        expected_coords = original_graph["coords"].copy()
        if scenario["shifted"]:
            indices = np.arange(edges, dtype=np.int64)
            expected_coords[0, :, 0] += (np.float32(.17) * (indices % 7 - 3).astype(np.float32))[:, None, None]
            expected_coords[0, :, 1] += (np.float32(.23) * (indices % 5 - 2).astype(np.float32))[:, None, None]
        require(same_bytes(graph["coords"], expected_coords), "fixed graph-coordinate shift")
        for name in set(graph) - {"coords"}:
            require(same_bytes(graph[name], original_graph[name]), "unchanged graph array: " + name)
        require(not np.any(graph["net0"]) and scenario["fresh_zero_net_each_call"] is True and scenario["feedback_iterations"] == 0, "fresh state without feedback")
        calls = scenario["calls"]
        require([(c["iteration"], c["mode"]) for c in calls] == [(iteration, mode) for iteration in range(4) for mode in MODES], "12 interleaved calls")
        representatives = {}
        for call in calls[:3]:
            mode = call["mode"]
            representatives[mode] = artifact(args.run_dir, call["artifact"])
            require(call["artifact"]["arrays"] == call["arrays"], "representative raw SHA binding")
            for name, shape in {"corr": (1, edges, 882), "net": (1, edges, 384), "delta": (1, edges, 2), "weight": (1, edges, 2)}.items():
                array = representatives[mode][name]
                require(array.shape == shape and array.dtype == np.float32, "original output shape/type")
                total_elements += array.size
            require(np.all((representatives[mode]["weight"] >= 0) & (representatives[mode]["weight"] <= 1)), "Update weight range")
        reference = representatives["reference"]
        if not scenario["shifted"]:
            old = artifact(args.motion_dir, parent["calls"][0]["artifact"])
            require(all(same_bytes(reference[name], old[name]) for name in KEYS), "fresh identity reference equals v26")
        errors = {mode: {name: comparison(arrays[name], reference[name]) for name in KEYS}
                  for mode, arrays in representatives.items()}
        for call in calls:
            mode = call["mode"]
            require(call["timing_role"] == ("cold" if call["iteration"] == 0 else "warm"), "timing role")
            require(call["arrays"] == calls[MODES.index(mode)]["arrays"] and call["repeat_bitwise_equal"] is True, "all repeat tensor hashes")
            for name in KEYS:
                recorded_comparison(errors[mode][name], call["versus_reference"][name])
                require(errors[mode][name]["gate_passed"], "fixed full-array numerical gate")
            if "versus_v26_saved" in call:
                require(not scenario["shifted"] and mode == "reference", "v26 gate scope")
                for name in KEYS:
                    recorded_comparison(errors[mode][name], call["versus_v26_saved"][name])
            for field in ("corr_seconds", "update_seconds", "total_dynamic_seconds"):
                require(math.isfinite(call[field]) and call[field] > 0, "positive timing")
            close_number(call["total_dynamic_seconds"], call["corr_seconds"] + call["update_seconds"], "dynamic timing sum")
            timing = call["native_timing"]
            if mode == "reference":
                require(timing is None, "reference has no native timing")
                continue
            query = timing["correlation"] if mode == "native_one_shot" else timing
            native = query["native"]
            require(native["abi_version"] == 1 and native["neon"] == 1 and native["threads"] == 4, "actual ARM native contract")
            require(native["context_bytes"] == context_bytes and native["output_elements"] == edges * 882, "native buffer dimensions")
            require(query["cache_key"] == key and query["includes_prepare"] is False, "query cache identity")
            require(0 <= native["kernel_seconds"] <= native["total_seconds"] <= query["native_call_seconds"] <= query["total_seconds"] <= call["corr_seconds"], "nested query timing")
            if mode == "native_one_shot":
                prep = timing["preparation"]
                require(timing["closed"] is True and timing["includes_prepare_and_close"] is True, "one-shot cleanup and scope")
                require(prep["feature_inputs_copied"] is True and prep["source_array_mutations_do_not_update_cache"] is True, "owned one-shot preparation")
                require(prep["context_bytes"] == context_bytes and prep["cache_key"] == key, "one-shot feature identity")
                require(prep["total_seconds"] + query["total_seconds"] <= timing["total_seconds"] <= call["corr_seconds"], "one-shot includes prepare/query/close")
        prep = scenario["cached_preparation"]
        before, after = scenario["cached_metrics_before_close"], scenario["cached_metrics_after_close"]
        require(prep == before["preparation"] == after["preparation"], "cached preparation recorded unchanged")
        require(prep["feature_inputs_copied"] is True and prep["source_array_mutations_do_not_update_cache"] is True and prep["cache_key"] == key, "owned fixed-version cached features")
        require(prep["context_bytes"] == context_bytes and prep["native"]["neon"] == 1, "cached preparation bytes and ARM path")
        require(before["closed"] is False and before["current_context_bytes"] == context_bytes and after["closed"] is True and after["current_context_bytes"] == 0, "cached close releases owned context")
        require({k: v for k, v in before.items() if k not in ("closed", "current_context_bytes")} == {k: v for k, v in after.items() if k not in ("closed", "current_context_bytes")}, "history survives close")
        require(after["attempts"] == after["completed_calls"] == 4 and after["failed_calls"] == 0 and after["library_sha256"] == run["library_sha256"], "cached call and library accounting")
        cached = [c["native_timing"] for c in calls if c["mode"] == "native_cached"]
        close_number(after["correlate_total_seconds_sum"], sum(t["total_seconds"] for t in cached), "cached query timing sum")
        close_number(after["kernel_seconds_sum"], sum(t["native"]["kernel_seconds"] for t in cached), "cached kernel timing sum")
        close_number(after["correlate_total_seconds_max"], max(t["total_seconds"] for t in cached), "cached maximum timing")
        require(math.isfinite(scenario["cached_close_seconds"]) and scenario["cached_close_seconds"] >= 0, "close timing")
        warm = {}
        for mode in MODES:
            subset = [c for c in calls if c["mode"] == mode and c["iteration"] > 0]
            warm[mode] = {field.replace("_seconds", "_ms"): sum(c[field] for c in subset) / 3 * 1000
                          for field in ("corr_seconds", "update_seconds", "total_dynamic_seconds")}
            for field, value in scenario["warm_means"][mode].items():
                close_number(value * 1000, warm[mode][field.replace("_seconds", "_ms")], "three-warm mean")
        selected_edges = [0, 1, 2, edges // 2, edges // 2 + 1, edges - 2, edges - 1]
        sample_indices = [(0, edge, ox, oy, py, px, level) for edge in selected_edges
                          for ox in range(7) for oy in range(7) for py in range(3)
                          for px in range(3) for level in range(2)]
        expected = oracle.selected_elements(gmap, fmap, graph["coords"], graph["kk"], graph["jj"], sample_indices)
        oracle_errors = {}
        for mode, arrays in representatives.items():
            actual = arrays["corr"][0, selected_edges, :].reshape(-1)
            oracle_errors[mode] = oracle.error_stats(actual, expected)
            require(oracle_errors[mode]["outside_fixed_FP32_gate"] == 0, "declared independent oracle sample")
        results.append({"name": scenario["name"], "edge_count": edges,
                        "representative_arrays_vs_reference": errors,
                        "warm_means_ms": warm,
                        "cached_preparation_ms": prep["total_seconds"] * 1000,
                        "cached_close_ms": scenario["cached_close_seconds"] * 1000,
                        "one_shot_warm_preparation_ms": sum(c["native_timing"]["preparation"]["total_seconds"] for c in calls if c["mode"] == "native_one_shot" and c["iteration"] > 0) / 3 * 1000,
                        "oracle_selected_edges": selected_edges, "oracle_elements_per_mode": len(sample_indices),
                        "independent_oracle": oracle_errors})
    require(run["exact_FP32_gate_passed"] is True, "recorded aggregate gate")
    for path, sha in files.items():
        require(digest(Path(path)) == sha, "audit input source-after: " + Path(path).name)
    return {
        "schema": "v27_DPVO_saved_array_independent_review",
        "status": "passed_saved_array_audit_not_VO",
        "checks_passed": checks, "checks_failed": 0,
        "auditor_sha256": digest(Path(__file__)), "oracle_sha256": PINS["corr_numpy_oracle.py"],
        "input_receipts_sha256": {"run": digest(args.run_dir / "run.json"), "capture": PINS["capture.json"], "motion": PINS["motion.json"], "build": run["build_sha256"]},
        "native_sources_sha256": build["source_sha256"], "native_library_sha256": run["library_sha256"],
        "native_compiler_sha256": build["compiler_sha256"],
        "call_count": 48, "graph_count": 4, "saved_representative_artifacts": 12,
        "saved_representative_elements_compared": total_elements,
        "repeat_arrays_checked_via_raw_sha": True,
        "oracle_independent_elements": 4 * 7 * 882, "oracle_comparisons_across_three_modes": 4 * 7 * 882 * 3,
        "FP32_gate": {"rtol": 1e-4, "atol": 1e-5, "passed": True},
        "original_Update_saved_outputs_gate_passed": True,
        "original_weights_retained_and_unchanged": True,
        "owned_cache_context_bytes": context_bytes, "terminal_cache_context_bytes": 0,
        "cleanup_errors": [], "peak_board_process_rss_mib": run["peak_rss_mib"],
        "cpu_affinity": run["cpu_affinity"], "torch_threads": run["torch_threads"],
        "native_requested_threads": 4, "actual_OpenMP_team_size_measured": False,
        "native_ARM_NEON_reported_by_executed_calls": True,
        "source_and_saved_inputs_unchanged_during_audit": True,
        "scenarios": results,
        "scope": {"saved_array_math_only_on_host": True, "host_model_forward": False,
                  "board_tasks_started_by_auditor": False, "NPU_used": False,
                  "BA_executed": False, "trajectory_accuracy_validated": False,
                  "realtime_20Hz_validated": False},
        "limitations": ["Twelve saved representative outputs bind the remaining repetitions by exact raw tensor hashes; repeated arrays were not all independently saved.",
                        "All saved corr/net/delta/weight elements use the original fixed FP32 gate; the independent float64 mathematical oracle covers seven predeclared edges per graph, not all edges.",
                        "Reference and one-shot include feature pyramid preparation on each call; cached query timing excludes its separately reported owned-feature preparation.",
                        "Timing is a short interleaved CPU primitive/Update probe, not a sustained, controlled-clock or end-to-end VO benchmark.",
                        "Ownership flags and release accounting are checked against frozen source and receipts; saved arrays alone cannot establish every in-process alias relationship."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "capture-dir", "motion-dir", "build-receipt", "native-source-dir", "v26-source-dir", "driver", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new file")
    try:
        result = audit(args)
    except Exception as error:
        result = {"schema": "v27_DPVO_saved_array_independent_review", "status": "failed",
                  "auditor_sha256": digest(Path(__file__)), "checks_passed": checks,
                  "checks_failed": 1, "error": str(error)}
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks_passed": result["checks_passed"],
                      "review_sha256": digest(args.output)}, ensure_ascii=False))
    return 0 if result["status"].startswith("passed_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
