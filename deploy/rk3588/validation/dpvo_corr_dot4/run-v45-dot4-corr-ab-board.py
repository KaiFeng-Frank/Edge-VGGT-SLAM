#!/usr/bin/env python3
"""Board-only v36 cached dot versus dot4: identical saved features/graphs, no neural/BA."""
import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

import numpy as np

sys.dont_write_bytecode = True
PARENT_AB_SHA = "531777cd463d4ace4299960a7034cfe42b108db68391c3241cbc6d616a560c9d"
HELPER_SHA = "9743962325ba1c4ada689769eaf800a617ba1f46f4602062a6dab3b23b5bdf48"
OLD_LIBRARY_SHA = "23ffa5b4b3e9251b8a41e149da2610c9db5818fe5fe148ab18f8b45b41486582"
NEW_LIBRARY_SHA = "bd324375294722faeaf02b71645082e9b11e46f64bbdedee7eb52e185d0bb9cc"
NEW_BUILD_SHA = "bf74468f6f365204c0ed7e23a8bd7d2f95c9d1316de4c78f21c68aa01e353e2e"
CPP_SHA = "8e174eb55e0b114a89b6d61aef9e56e32312633d0b0be44d3ffbca9df67c15e9"
BUILD_SCRIPT_SHA = "aea69d5e614efeb7b80ab309c04dc821379b6eb96c048be7bc590a5d63e0472f"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-ab-helper", "frame-helper", "capture-dir", "motion-dir",
                 "parent-library", "parent-build", "candidate-library", "candidate-build", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--parent-build-sha256", required=True)
    a = p.parse_args(argv)
    if platform.machine() != "aarch64" or platform.system() != "Linux":
        raise ValueError("RK3588 board only")
    import hashlib
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if sha(a.parent_ab_helper) != PARENT_AB_SHA:
        raise ValueError("Frozen v36 saved-array loader differs")
    spec = importlib.util.spec_from_file_location("_v45_frozen_AB_loader", a.parent_ab_helper)
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)
    H.board_guard()  # Before real saved arrays, libraries or output creation.
    out = a.output_dir.resolve()
    H.require(out.is_relative_to(H.BASE), "New isolated board output required")
    out.mkdir(exist_ok=False)
    pins, caches = {}, {}
    affinity = os.sched_getaffinity(0)
    report = dict(schema="v45_saved_features_cached_corr_dot4_ABBA", status="failed_execution",
        script_sha256=sha(__file__), candidate_accepted=False, realtime_validated=False,
        neural_forward_executed=False, Update_or_BA_executed=False, actual_board_native_correlation=True,
        methods=dict(A="frozen_v36_per_frame_cache_original_dot", B="same_per_frame_cache_horizontal4_dot"),
        cleanup_errors=[], cases=[])
    def pin(path, digest):
        path = Path(path).resolve()
        H.require(sha(path) == digest, "Pinned file differs: " + str(path))
        pins[str(path)] = digest
        return path
    def save(name, values):
        path = out / name
        np.savez(path, **values)
        return dict(file=name, sha256=sha(path), bytes=path.stat().st_size,
                    arrays={k: H.meta(v) for k, v in values.items()})
    try:
        pin(__file__, report["script_sha256"])
        pin(a.parent_ab_helper, PARENT_AB_SHA)
        pin(a.frame_helper, HELPER_SHA)
        old = json.loads(pin(a.parent_build, a.parent_build_sha256).read_text())
        H.require(old["schema"] == "v36_immutable_frame_correlation_native_build"
            and old["status"] == "compiled_and_synthetic_smoke_passed"
            and old["output"]["sha256"] == OLD_LIBRARY_SHA and old["cleanup_errors"] == []
            and old["sources_and_inputs_unchanged"], "Frozen original cached build differs")
        new = json.loads(pin(a.candidate_build, NEW_BUILD_SHA).read_text())
        H.require(new["schema"] == "v45_horizontal_dot4_native_build"
            and new["status"] == "compiled_dot4_candidate_not_numerically_validated"
            and new["script_sha256"] == BUILD_SCRIPT_SHA and new["cleanup_errors"] == []
            and new["sources_and_inputs_unchanged"] and new["output"]["sha256"] == NEW_LIBRARY_SHA
            and new["source_sha256"]["dpvo_frame_corr_dot4.cpp"] == CPP_SHA
            and new["source_sha256"]["dpvo_frame_corr.py"] == HELPER_SHA,
            "Candidate build/source identity differs")
        pin(a.parent_library, OLD_LIBRARY_SHA)
        pin(a.candidate_library, NEW_LIBRARY_SHA)
        variant = ctypes.CDLL(str(a.candidate_library)).dpvo_corr_v45_math_variant
        variant.restype = ctypes.c_char_p
        H.require(variant() == b"horizontal4_independent_original_channel_lane_order_no_FMA", "Candidate variant differs")
        helper = H.module(a.frame_helper, "_v45_frozen_frame_cache")
        captured = json.loads(pin(a.capture_dir / "capture.json", H.CAPTURE_SHA).read_text())
        motion = json.loads(pin(a.motion_dir / "motion.json", H.MOTION_SHA).read_text())
        H.require(captured["status"] == "completed_encoder_cpu_probe_not_VO"
            and motion["status"] == "completed_dynamic_neural_probe_not_VO"
            and motion["capture_sha256"] == H.CAPTURE_SHA
            and captured["sources_and_inputs_unchanged"] and motion["sources_and_inputs_unchanged"],
            "Frozen captured feature/motion lineage differs")
        patch = H.load_artifact(a.motion_dir, motion["patches"], motion["patches"]["arrays"], pin)
        H.require(patch["gmap"].shape == (1, 288, 128, 3, 3), "Original three M96 frames required")
        frames = []
        for i in range(3):
            row = next(r for r in captured["calls"] if r["frame_index"] == i)
            values = H.load_artifact(a.capture_dir, row["artifact"], row["arrays"], pin)
            H.require(values["fmap"].shape == (1, 1, 128, 116, 152), "Original fmap geometry differs")
            frames.append(dict(frame_id=motion["frame_ids"][i], gmap=patch["gmap"][0, i*96:(i+1)*96], fmap=values["fmap"][0, 0]))
        scenarios = []
        for row in motion["scenarios"]:
            graph = H.load_artifact(a.motion_dir, row["graph"], row["graph"]["arrays"], pin)
            gold = H.load_artifact(a.motion_dir, row["calls"][0]["artifact"], row["calls"][0]["artifact"]["arrays"], pin)["corr"]
            H.require(np.array_equal(graph["ii"], graph["kk"] // 96), "Source ownership differs")
            scenarios.append((graph, gold))
        first = next(s for s in scenarios if len(s[0]["kk"]) == 96)
        full = next(s for s in scenarios if len(s[0]["kk"]) == 864)
        reorder, subset1 = H.remap_graph(full[0], [2, 0, 1], 96)
        delete, subset2 = H.remap_graph(full[0], [0, 2], 96, remove=1)
        cases = [("first_two_frames", frames[:2], first[0], first[1], 2, 2),
                 ("grow_three_frames", frames, full[0], full[1], 3, 3),
                 ("reorder_slots", [frames[i] for i in (2, 0, 1)], reorder, full[1][:, subset1], 3, 3),
                 ("delete_middle", [frames[0], frames[2]], delete, full[1][:, subset2], 3, 2)]
        H.require({4, 5, 6, 7} <= affinity, "Four A76 CPUs required")
        os.sched_setaffinity(0, {4, 5, 6, 7})
        report.update(cpu_affinity=sorted(os.sched_getaffinity(0)), native_threads=4,
            parent_build_sha256=a.parent_build_sha256, candidate_build_sha256=NEW_BUILD_SHA,
            candidate_library_sha256=NEW_LIBRARY_SHA, parent_library_sha256=OLD_LIBRARY_SHA,
            OMP_WAIT_POLICY=os.environ.get("OMP_WAIT_POLICY"), GOMP_SPINCOUNT=os.environ.get("GOMP_SPINCOUNT"))
        for method, library, digest in (("A", a.parent_library, OLD_LIBRARY_SHA), ("B", a.candidate_library, NEW_LIBRARY_SHA)):
            caches[method] = helper.PerFrameCorrelation(library, library_sha256=digest, patch_count=96, threads=4)
        for name, current, graph, gold, prepared, resident in cases:
            graph = {k: graph[k] for k in ("ii", "jj", "kk", "coords")}
            row = dict(name=name, frame_ids=[f["frame_id"] for f in current], edges=len(graph["kk"]),
                graph=save(name + "-graph.npz", graph), original_CPU_reference=save(name + "-reference.npz", dict(corr=gold)), calls=[])
            report["cases"].append(row)
            originals = {}
            for repetition in range(4):
                outputs = {}
                for method in (("A", "B") if repetition % 2 == 0 else ("B", "A")):
                    start = time.monotonic()
                    result = caches[method](graph["coords"], ii=graph["ii"], jj=graph["jj"], kk=graph["kk"], frames=tuple(current))
                    end = time.monotonic()
                    record = dict(method=method, round=repetition, cold=repetition == 0,
                        full_callback_seconds=end-start, array=H.meta(result), versus_original_CPU=H.comparison(result, gold))
                    if method not in originals:
                        originals[method] = result.copy()
                        record["artifact"] = save(name + "-" + method + ".npz", dict(corr=result))
                    record["repeat_bitwise_equal"] = result.tobytes() == originals[method].tobytes()
                    metrics = caches[method].metrics()
                    record["internal_callback_record"] = metrics["calls"][-1]
                    H.require(metrics["prepared_frames"] == prepared and metrics["current_cached_frames"] == resident,
                        "Frame cache lifecycle differs")
                    row["calls"].append(record)
                    outputs[method] = result
                    H.require(record["repeat_bitwise_equal"], "Repeated identical input differs")
                compare = H.comparison(outputs["B"], outputs["A"])
                for record in row["calls"][-2:]:
                    record["B_vs_A"] = compare
                print(json.dumps(dict(case=name, round=repetition, bitwise=compare["bitwise_equal"], strict_gate=compare["gate_passed"])), flush=True)
            row["warm_mean_seconds"] = {m: sum(c["full_callback_seconds"] for c in row["calls"] if c["method"] == m and not c["cold"])/3 for m in ("A", "B")}
        report["all_FP32_gates_passed"] = all(c["B_vs_A"]["gate_passed"] and c["versus_original_CPU"]["gate_passed"] for r in report["cases"] for c in r["calls"])
        report["A_B_bitwise_equal"] = all(c["B_vs_A"]["bitwise_equal"] for r in report["cases"] for c in r["calls"])
        report["status"] = "completed_dot4_AB_bitwise_not_adopted" if report["all_FP32_gates_passed"] and report["A_B_bitwise_equal"] else "completed_numerical_rejected"
        report["timing_scope"] = "ABBA twice per case. Full callback includes native scan/kernel/preparation/cache reporting. Hashes, saves and metrics reads after timer; kernel nested. Same threads and immutable features, no CNN/Update/BA/new observation, no sustained 20Hz claim."
    except Exception as error:
        report.update(status="failed_execution", error=repr(error), traceback=traceback.format_exc())
    finally:
        report["lifecycle"] = {}
        for method, cache in caches.items():
            try:
                before = cache.metrics()
                cache.close()
                after = cache.metrics()
                report["lifecycle"][method] = dict(before_close=before, after_close=after)
                H.require(after["closed"] and after["current_context_bytes"] == after["current_cached_frames"] == 0, "Cache cleanup failed")
            except Exception as error:
                report["cleanup_errors"].append(repr(error))
        try:
            report["source_pins"] = pins
            report["source_hashes_after"] = {p: sha(p) for p in pins}
            report["sources_and_inputs_unchanged"] = report["source_hashes_after"] == pins
            H.require(report["sources_and_inputs_unchanged"], "Source/input changed")
            os.sched_setaffinity(0, affinity)
            report["affinity_restored"] = os.sched_getaffinity(0) == affinity
            H.require(report["affinity_restored"], "Affinity restoration failed")
        except Exception as error:
            report["cleanup_errors"].append(repr(error))
        if report["cleanup_errors"]:
            report["status"] = "failed_execution"
        report["peak_RSS_MiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        (out / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["status"] == "completed_dot4_AB_bitwise_not_adopted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
