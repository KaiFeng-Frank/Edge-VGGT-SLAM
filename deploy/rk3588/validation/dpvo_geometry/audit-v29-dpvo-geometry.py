#!/usr/bin/env python3
"""Audit v29 saved arrays; no geometry imports, solves, inference or board access.

Saved Schur identities follow DPVO; MIT notice is in the adjacent LICENSE.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


RUN_SHA = "c870b31ad5b5ed96e8f178b92f3b7badb710b023eb31a845f934868e0b5fa2b4"
FEEDBACK_SHA = "241c802270cf6be171dff5e56388c8094bd6277f621622afc766270db9a1b967"
SOURCE_SHAS = {
    "run-v29-dpvo-geometry-board.py": "8a2a7c8408e234fb5e43899b63519e7dd33c2290beb8c4b9935fc46b86b35a26",
    "dpvo_cpu_geometry_vectorized.py": "c893f12d188baced438504cfc592a8900e18e40ef8223af0a7e944e4484b0c3b",
    "dpvo_cpu_geometry.py": "13eb1e03eb4a5ed1bd1533549f2b9f4994abfaa61e0b55304b99e6fe77d813df",
}
OUTPUT_KEYS = ("poses_after", "patches_after", "coords_after", "projection_after", "residual_after", "active_after")
SYSTEM_SHAPES = {
    "B": (12, 12), "E": (12, 288), "C": (288,), "v": (12,),
    "u": (288,), "Q": (288,), "S": (12, 12), "y": (12,),
    "dx": (2, 6), "dz": (288,), "unique_patch_ids": (288,),
}
RTOL, ATOL = 1e-4, 1e-5


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def meta(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes,
                tensor_sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest())


class Checks:
    def __init__(self):
        self.count = 0
        self.failures = []

    def check(self, ok, label):
        self.count += 1
        if not bool(ok):
            self.failures.append(label)

    def require(self, ok, label):
        self.check(ok, label)
        if not bool(ok):
            raise ValueError(label)


def compare(actual, expected, checks, label):
    a, b = np.asarray(actual), np.asarray(expected)
    checks.require(a.shape == b.shape and a.dtype == b.dtype, label + ": shape/dtype")
    checks.require(np.isfinite(a).all() and np.isfinite(b).all(), label + ": finite")
    failed = a != b if a.dtype.kind in "biu" else ~np.isclose(a, b, rtol=RTOL, atol=ATOL)
    delta = a.astype(np.float64) - b.astype(np.float64)
    checks.check(not failed.any(), label + ": fixed FP32 gate")
    return dict(passed=not bool(failed.any()), failed_elements=int(failed.sum()),
                elements=int(a.size), max_abs=float(np.abs(delta).max()) if a.size else 0.,
                bitwise_equal=a.tobytes(order="C") == b.tobytes(order="C"))


def load_artifact(folder, artifact, checks, label):
    name = artifact["file"]
    checks.require(Path(name).name == name and name not in ("", ".", ".."), label + ": safe basename")
    path = folder / name
    checks.require(path.stat().st_size == artifact["bytes"], label + ": file bytes")
    checks.require(sha(path) == artifact["sha256"], label + ": file SHA256")
    with np.load(path, allow_pickle=False) as store:
        values = {key: store[key].copy(order="C") for key in store.files}
    checks.require(set(values) == set(artifact["arrays"]), label + ": key set")
    for key, value in values.items():
        checks.require(value.dtype.kind in "bfiu" and np.isfinite(value).all(), label + "/" + key + ": finite numeric")
        checks.require(meta(value) == artifact["arrays"][key], label + "/" + key + ": raw metadata SHA256")
        value.flags.writeable = False
    return values


def summarize_calls(calls, recorded, input_hashes, output_hashes, checks, label):
    checks.require(len(calls) == 8, label + ": eight ordinary calls")
    for position, call in enumerate(calls):
        repetition, order = divmod(position, 2)
        backend = ("scalar", "vectorized")[order if repetition % 2 == 0 else 1 - order]
        checks.require(call["backend"] == backend, label + ": backend order")
        checks.check(call["repetition"] == repetition and call["order_in_pair"] == order and call["BA_iterations"] == 2, label + ": AB/BA call identity")
        checks.check(call["timing_role"] == ("first_per_case" if repetition == 0 else "warm_same_input"), label + ": timing role")
        checks.check(call["input_sha256"] == input_hashes, label + ": input raw SHA")
        checks.check(call["output_sha256"] == output_hashes[backend], label + ": repeat output raw SHA")
        for key in ("BA_seconds", "external_copy_seconds", "input_hash_seconds", "post_validation_seconds"):
            value = call[key]
            checks.require(isinstance(value, (int, float)) and np.isfinite(value) and (value > 0 if key == "BA_seconds" else value >= 0), label + ": " + key)
    summary = {}
    for backend in ("scalar", "vectorized"):
        times = [x["BA_seconds"] for x in calls if x["backend"] == backend]
        rec, mean = recorded[backend], float(np.mean(times[1:]))
        checks.check(rec["first_seconds"] == times[0] and rec["warm_seconds"] == times[1:], label + ": summary members")
        checks.check(abs(rec["warm_mean_seconds"] - mean) <= 1e-12, label + ": recomputed warm mean")
        summary[backend] = dict(first_ms=1000 * times[0], warm_ms=[1000 * x for x in times[1:]], warm_mean_ms=1000 * mean)
    summary["scalar_over_vectorized_warm_mean"] = summary["scalar"]["warm_mean_ms"] / summary["vectorized"]["warm_mean_ms"]
    return summary


def saved_system_algebra(system, checks, label):
    """Check identities using saved updates; never factorize/solve/update geometry."""
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        q = 1 / (system["C"] + np.array([1e-4], np.float32))
        schur = system["B"] - (system["E"] * q[None]) @ system["E"].T
        diagonal = np.diag_indices(schur.shape[0])
        schur[diagonal] += 1 + 1e-4 * schur[diagonal]
        y = system["v"] - system["E"] @ (q * system["u"])
        dz = q * (system["u"] - system["E"].T @ system["dx"].reshape(-1))
    return {key: compare(value, system[key], checks, label + "/" + key)
            for key, value in dict(Q=q, S=schur, y=y, dz=dz).items()}


def verify_events(run, checks):
    expected, ordinary = [], []
    for case, row in enumerate(run["cases"]):
        for call in row["calls"]:
            for stage in ("begin", "completed"):
                expected.append(dict(case=case, kind="ordinary", backend=call["backend"], repetition=call["repetition"], stage=stage))
            ordinary.append(call["BA_seconds"])
        for backend in ("scalar", "vectorized"):
            for stage in ("begin", "completed"):
                expected.append(dict(case=case, kind="instrumented", backend=backend, repetition=None, stage=stage))
    events = run["BA_events"]
    checks.require(len(events) == len(expected) == 80, "80 begin/completed events")
    previous = -np.inf
    for event, identity in zip(events, expected):
        checks.check({k: event[k] for k in identity} == identity, "BA event identity/order")
        now = event["monotonic"]
        checks.require(np.isfinite(now) and now > previous, "BA event monotonic timestamp")
        previous = now
    j = 0
    for position in range(0, len(events), 2):
        begin, end = events[position:position + 2]
        if begin["kind"] == "ordinary":
            checks.check(end["monotonic"] - begin["monotonic"] + 1e-8 >= ordinary[j], "BA stopwatch within event interval")
            j += 1
    for kind, expected_count in (("ordinary", 32), ("instrumented", 8)):
        for stage, field in (("begin", "attempted"), ("completed", "completed")):
            count = sum(e["kind"] == kind and e["stage"] == stage for e in events)
            checks.check(count == expected_count == run[f"{kind}_BA_calls_{field}"], "Recomputed " + kind + " " + field)
    checks.check(run["BA_iterations_in_completed_calls"] == 80, "80 completed BA steps")


def audit(run_dir, feedback_dir, source_dir, scalar_geometry):
    checks = Checks()
    report = dict(schema="v29_geometry_saved_array_independent_review", auditor_sha256=sha(__file__),
                  status="failed", model_inference_executed=False, BA_executed=False,
                  geometry_modules_imported=False, linear_system_solve_executed=False, board_accessed=False,
                  numpy_version=np.__version__, cases=[], source_hashes={})
    try:
        run_path, feedback_path = run_dir / "run.json", feedback_dir / "run.json"
        checks.require(sha(run_path) == RUN_SHA, "Frozen v29 run SHA256")
        checks.require(sha(feedback_path) == FEEDBACK_SHA, "Frozen v28 feedback SHA256")
        run, feedback = json.loads(run_path.read_text()), json.loads(feedback_path.read_text())
        report.update(run_sha256=sha(run_path), reference_feedback_sha256=sha(feedback_path))
        for name, expected in SOURCE_SHAS.items():
            path = scalar_geometry if name == "dpvo_cpu_geometry.py" else source_dir / name
            actual = sha(path)
            checks.require(actual == expected, name + ": source SHA256")
            report["source_hashes"][name] = actual
        checks.require(run["status"] == "completed_geometry_only_benchmark" and len(run["cases"]) == 4, "Completed four-case benchmark")
        checks.check(run["actual_board_execution"] is True and run["candidate_accepted"] is True and run["strict_FP32_gate_passed"] is True, "Board acceptance flags")
        for key in ("NPU_used", "neural_Update_executed", "correlation_executed", "complete_feedback_executed", "online_VO_executed", "trajectory_accuracy_validated", "realtime_validated"):
            checks.check(run[key] is False, "Scope false: " + key)
        checks.check(run["FP32_gate"] == dict(rtol=RTOL, atol=ATOL, masks_and_integer_IDs_exact=True), "Unchanged FP32 gate")
        checks.check(run["OPENBLAS_NUM_THREADS"] == "1" and run["cpu_affinity"] == [4, 5, 6, 7], "Board BLAS/affinity")
        checks.check(run["source_feedback_sha256"] == FEEDBACK_SHA and run["scalar_geometry_sha256"] == SOURCE_SHAS["dpvo_cpu_geometry.py"] and
                     run["vectorized_geometry_sha256"] == SOURCE_SHAS["dpvo_cpu_geometry_vectorized.py"], "Source role binding")
        initial = load_artifact(feedback_dir, feedback["initial_state"], checks, "v28 initial")
        rounds = [load_artifact(feedback_dir, row["artifact"], checks, "v28 case" + str(i)) for i, row in enumerate(feedback["calls"])]
        checks.require(len(rounds) == 4, "Four reference feedback states")
        raw_meta = [{key: meta(value) for key, value in x.items()} for x in [initial] + rounds]
        checks.check(run["input_array_hashes_before"] == raw_meta == run["input_array_hashes_after"], "Recomputed input raw metadata before/after")
        checks.check(np.array_equal(initial["intrinsics"], np.array(feedback["graph_contract"]["intrinsics"], np.float32)), "Frozen K")
        checks.check(np.array_equal(initial["kk"], np.repeat(np.arange(288, dtype=np.int64), 3)) and
                     np.array_equal(initial["jj"], np.tile(np.arange(3, dtype=np.int64), 288)) and
                     np.array_equal(initial["ii"], np.repeat(np.arange(3, dtype=np.int64), 96)[initial["kk"]]), "Exact E864 graph order")
        expected_pins = {**report["source_hashes"], "run.json": FEEDBACK_SHA,
                         feedback["initial_state"]["file"]: feedback["initial_state"]["sha256"],
                         **{row["artifact"]["file"]: row["artifact"]["sha256"] for row in feedback["calls"]}}
        pins = run["source_pins"]
        checks.check(len(pins) == len(expected_pins) and {Path(p).name: digest for p, digest in pins.items()} == expected_pins, "All source/artifact pin roles")
        checks.check(run["source_hashes_after"] == pins and run["sources_and_artifacts_unchanged"] is True, "Board after-source hashes")
        required = set(OUTPUT_KEYS) | {f"system{step}_{key}" for step in range(2) for key in SYSTEM_SHAPES}
        for i, (row, source) in enumerate(zip(run["cases"], rounds)):
            label, previous = "case" + str(i), initial if i == 0 else rounds[i - 1]
            checks.check(np.array_equal(source["poses_before"], previous["poses" if i == 0 else "poses_after"]) and
                         np.array_equal(source["patches_before"], previous["patches" if i == 0 else "patches_after"]), label + ": source continuity")
            checks.check(np.array_equal(source["target"], source["coords_before"][..., 1, 1] + source["delta"]), label + ": target=center+delta")
            inputs = dict(poses=source["poses_before"], patches=source["patches_before"], intrinsics=initial["intrinsics"], target=source["target"], weight=source["weight"],
                          ii=initial["ii"], jj=initial["jj"], kk=initial["kk"], **{"lambda": np.array([1e-4], np.float32)})
            input_hashes = {key: meta(value)["tensor_sha256"] for key, value in inputs.items()}
            checks.check(row["case_index"] == i and row["input_sha256"] == input_hashes and row["source_artifact_sha256"] == feedback["calls"][i]["artifact"]["sha256"], label + ": source identity")
            stores = {backend: load_artifact(run_dir, row["artifacts"][backend], checks, label + "/" + backend) for backend in ("scalar", "vectorized")}
            refs, algebra = {}, {}
            for backend, store in stores.items():
                checks.require(set(store) == required, label + ": 28 saved arrays")
                refs[backend] = [compare(store[key], source[key], checks, label + "/" + backend + "/reference/" + key) for key in OUTPUT_KEYS]
                checks.check(store["poses_after"][0, 0].tobytes() == source["poses_before"][0, 0].tobytes(), label + ": fixed first pose")
                checks.check(np.all(store["patches_after"][..., 2, :, :] >= 1e-4), label + ": positive inverse depth")
                compare(store["residual_after"], source["target"] - store["projection_after"], checks, label + ": residual identity")
                for step in range(2):
                    system = {key: store[f"system{step}_{key}"] for key in SYSTEM_SHAPES}
                    for key, shape in SYSTEM_SHAPES.items():
                        checks.require(system[key].shape == shape and system[key].dtype == ("int64" if key == "unique_patch_ids" else "float32"), label + ": system shape/dtype " + key)
                    checks.check(np.array_equal(system["unique_patch_ids"], np.arange(288, dtype=np.int64)), label + ": sorted patch IDs")
                    checks.check(np.all(system["C"] >= 0) and np.all(system["Q"] > 0), label + ": positive diagonal inverse")
                    norms = feedback["calls"][i]["ba"]["iterations"][step]
                    checks.check(np.isclose(float(np.linalg.norm(system["dx"])), norms["pose_update_norm"], rtol=RTOL, atol=ATOL) and
                                 np.isclose(float(np.linalg.norm(system["dz"])), norms["inverse_depth_update_norm"], rtol=RTOL, atol=ATOL), label + ": saved update norms")
                    algebra[backend + "/step" + str(step)] = saved_system_algebra(system, checks, label + "/" + backend + "/algebra" + str(step))
            compared = [compare(stores["vectorized"][key], stores["scalar"][key], checks, label + "/vectorized_vs_scalar/" + key) for key in sorted(required)]
            output_hashes = {backend: {key: meta(store[key])["tensor_sha256"] for key in OUTPUT_KEYS} for backend, store in stores.items()}
            timing = summarize_calls(row["calls"], row["summary"], input_hashes, output_hashes, checks, label)
            checks.check(row["ordinary_BA_calls"] == 8 and row["instrumented_validation_BA_calls"] == 2 and row["strict_FP32_gate_passed"] is True, label + ": counters/gate")
            checks.check(abs(row["outside_ordinary_BA_seconds"] - (row["case_wall_seconds"] - sum(x["BA_seconds"] for x in row["calls"]))) <= 1e-12, label + ": external timing boundary")
            checks.require(bool(row["checks"]), label + ": board checks present")
            for name, item in row["checks"].items():
                checks.check(item["passed"] is True and item["failed_elements"] == 0 and item["bitwise_equal"] is True and item["max_abs"] == 0, label + ": recorded " + name)
            report["cases"].append(dict(case_index=i, timing=timing,
                artifacts={backend: {key: row["artifacts"][backend][key] for key in ("file", "bytes", "sha256")} for backend in stores},
                scalar_vectorized_28_arrays_bitwise_equal=all(x["bitwise_equal"] for x in compared),
                six_outputs_bitwise_equal_frozen_v28=all(x["bitwise_equal"] for x in refs["vectorized"]),
                raw_tensor_metadata_verified=56, saved_system_algebra=algebra))
        verify_events(run, checks)
        checks.check(run["cleanup_errors"] == [] and run["affinity_restored"] is True and run["immutable_arrays_unchanged"] is True, "Board cleanup/affinity/immutable flags")
        checks.check(np.isfinite(run["peak_rss_mib"]) and run["peak_rss_mib"] > 0, "Finite peak RSS")
        report.update(status="passed" if not checks.failures else "failed", ordinary_BA_calls_verified=32, instrumented_BA_calls_verified=8,
                      total_completed_BA_steps_verified=80, saved_output_artifacts_verified=8, saved_output_tensors_verified=224,
                      cleanup_verified_from_board_receipt=not checks.failures, peak_rss_mib=run["peak_rss_mib"],
                      timing_scope=run["timing_scope"], first_call_semantics=run["first_call_semantics"],
                      limitations=["BA component only, excludes CNN/Update/correlation/new observations/full feedback.",
                                   "Four same-process frozen snapshots, interleaved; not temperature-controlled or online VO.",
                                   "Representative output arrays and separately instrumented systems independently read. Other repetitions bound by recorded raw hashes, not separately saved arrays.",
                                   "Lifecycle and after-hashes recorded on board; transferred files/source copies rehashed here.",
                                   "Saved Schur identities under fixed FP32 gate; no factorization, SE3 updates or actual-data BA execution here.",
                                   "Observed cohort byte equality is not a general floating-point or trajectory-quality guarantee."],
                      online_VO_validated=False, trajectory_accuracy_validated=False, realtime_validated=False)
    except Exception as exc:
        report["error"] = str(exc)
    report.update(checks=checks.count, failed_checks=len(checks.failures), failures=checks.failures)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--feedback-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--scalar-geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit(args.run_dir, args.feedback_dir, args.source_dir, args.scalar_geometry)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report.get(key) for key in ("status", "checks", "failed_checks", "error")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
