#!/usr/bin/env python3
"""RK3588-only v28 versus v29 BA on four frozen feedback snapshots.

No neural network, correlation, new observations, trajectory or NPU execution.
Adapted DPVO mathematics retain Princeton Vision & Learning Lab's MIT notice
in the adjacent LICENSE. Ordinary BA timing excludes outer copies/checks/I/O.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback


BASE = Path("/mnt/vggt-data/vggt-slam2/experiments")
RUN_SHA = "241c802270cf6be171dff5e56388c8094bd6277f621622afc766270db9a1b967"
SCALAR_SHA = "13eb1e03eb4a5ed1bd1533549f2b9f4994abfaa61e0b55304b99e6fe77d813df"
VECTOR_SHA = "c893f12d188baced438504cfc592a8900e18e40ef8223af0a7e944e4484b0c3b"
RTOL, ATOL = 1e-4, 1e-5
MODULE_NAMES = ("_v29_private_scalar_geometry", "_v29_private_vectorized_geometry")
SYSTEM_KEYS = ("B", "E", "C", "v", "u", "Q", "S", "y", "dx", "dz", "unique_patch_ids")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def meta(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes,
                tensor_sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest())


def require_board():
    require(platform.machine() == "aarch64" and platform.system() == "Linux",
            "RK3588 board only; actual-data BA on the host is prohibited")
    require(b"rockchip,rk3588" in Path("/proc/device-tree/compatible").read_bytes(),
            "RK3588 device-tree evidence required")
    require(os.environ.get("OPENBLAS_NUM_THREADS") == "1",
            "Start this process with OPENBLAS_NUM_THREADS=1 before NumPy import")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_artifact(folder, artifact, pin):
    import numpy as np
    name = artifact["file"]
    require(Path(name).name == name and name not in ("", ".", ".."), "Safe artifact basename required")
    path = pin(folder / name, artifact["sha256"])
    require(path.stat().st_size == artifact["bytes"], "Artifact byte count differs")
    with np.load(path, allow_pickle=False) as store:
        arrays = {key: store[key].copy(order="C") for key in store.files}
    require(set(arrays) == set(artifact["arrays"]), "Artifact tensor key set differs")
    require(all(meta(v) == artifact["arrays"][k] and np.isfinite(v).all()
                for k, v in arrays.items()), "Artifact raw metadata/finite check failed")
    for value in arrays.values():
        value.flags.writeable = False
    return arrays


def comparison(actual, expected):
    import numpy as np
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and a.dtype == b.dtype, "Comparison shape/dtype mismatch")
    require(np.isfinite(a).all() and np.isfinite(b).all(), "Nonfinite comparison value")
    if a.dtype.kind in "biu":
        failed = a != b
    else:
        failed = ~np.isclose(a, b, rtol=RTOL, atol=ATOL)
    delta = a.astype(np.float64) - b.astype(np.float64)
    denominator = float(np.linalg.norm(b.astype(np.float64)))
    return dict(passed=not bool(failed.any()), failed_elements=int(failed.sum()),
                elements=a.size, max_abs=float(np.abs(delta).max()) if a.size else 0.0,
                relative_l2=float(np.linalg.norm(delta) / denominator) if denominator else
                    (0.0 if not np.any(delta) else None),
                bitwise_equal=a.tobytes(order="C") == b.tobytes(order="C"))


def compare_tree(actual, expected, prefix=""):
    checks = {}
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), "Diagnostic keys differ")
        for key in expected:
            checks.update(compare_tree(actual[key], expected[key], prefix + "/" + str(key)))
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), "Diagnostic list differs")
        for index, value in enumerate(expected):
            checks.update(compare_tree(actual[index], value, prefix + "/" + str(index)))
    else:
        checks[prefix] = comparison(actual, expected)
    return checks


def copy_inputs(inputs):
    return {key: value.copy(order="C") for key, value in inputs.items()}


def arguments(inputs):
    return tuple(inputs[key] for key in ("poses", "patches", "intrinsics", "target", "weight",
                                         "lambda", "ii", "jj", "kk")) + (1, 3, 96)


def collect_trace(geometry, inputs):
    """Separate instrumented validation; never enters the ordinary timing mean."""
    systems = []
    original = geometry.fastba_system
    def traced(*args, **kwargs):
        value = original(*args, **kwargs)
        systems.append(value)  # Owned local system arrays, no copy in the BA body.
        return value
    geometry.fastba_system = traced
    try:
        output = copy_inputs(inputs)
        diagnostic = geometry.BA(*arguments(output), iterations=2, eff_impl=False)
    finally:
        geometry.fastba_system = original
    require(len(systems) == 2, "Exactly two BA validation systems required")
    return output, diagnostic, systems


def post_geometry(geometry, outputs):
    p, d, K = (outputs[k] for k in ("poses", "patches", "intrinsics"))
    target, weight, ii, jj, kk = (outputs[k] for k in ("target", "weight", "ii", "jj", "kk"))
    linear = geometry.fastba_linearize(p, d, K, target, weight, ii, jj, kk)
    return dict(poses_after=p, patches_after=d,
                coords_after=geometry.reproject(p, d, K, ii, jj, kk),
                projection_after=linear["projection"], residual_after=linear["residual"],
                active_after=linear["active"])


def run_case(scalar, vectorized, inputs, reference, saved_diagnostics, on_BA_event=None):
    """Callable with tiny injected/generated arrays for host-only tests."""
    import numpy as np
    case_begin = time.perf_counter()
    backends = {"scalar": scalar, "vectorized": vectorized}
    first = {}
    calls, checks, pair_validation_seconds = [], {}, 0.0
    for repetition in range(4):
        order = ("scalar", "vectorized") if repetition % 2 == 0 else ("vectorized", "scalar")
        pair = {}
        for backend in order:
            geometry = backends[backend]
            begin = time.perf_counter()
            copied = copy_inputs(inputs)
            ba_arguments = arguments(copied)
            copy_seconds = time.perf_counter() - begin
            begin = time.perf_counter()
            immutable_before = {k: meta(v) for k, v in copied.items() if k not in ("poses", "patches")}
            input_checks_seconds = time.perf_counter() - begin
            if on_BA_event is not None:
                on_BA_event("ordinary", backend, repetition, "begin")
            begin = time.perf_counter()
            diagnostic = geometry.BA(*ba_arguments, iterations=2, eff_impl=False)
            seconds = time.perf_counter() - begin
            if on_BA_event is not None:
                on_BA_event("ordinary", backend, repetition, "completed")
            begin = time.perf_counter()
            require(immutable_before == {k: meta(v) for k, v in copied.items()
                                          if k not in ("poses", "patches")}, "BA mutated immutable input")
            result = post_geometry(geometry, copied)
            require(np.array_equal(copied["poses"][0, 0], inputs["poses"][0, 0]), "Fixed first pose changed")
            for key in result:
                checks[f"repeat{repetition}/{backend}/saved/{key}"] = comparison(result[key], reference[key])
            checks.update(compare_tree(diagnostic, saved_diagnostics,
                                       f"repeat{repetition}/{backend}/saved/BA"))
            if repetition:
                for key in result:
                    checks[f"repeat{repetition}/{backend}/first/{key}"] = comparison(result[key], first[backend][key])
            else:
                first[backend] = {k: v.copy() for k, v in result.items()}
            pair[backend] = result
            validation_seconds = time.perf_counter() - begin
            calls.append(dict(backend=backend, repetition=repetition, timing_role="first_per_case" if repetition == 0 else "warm_same_input",
                              order_in_pair=order.index(backend), BA_iterations=2,
                              BA_seconds=seconds, external_copy_seconds=copy_seconds,
                              input_hash_seconds=input_checks_seconds, post_validation_seconds=validation_seconds,
                              input_sha256={k: meta(v)["tensor_sha256"] for k, v in inputs.items()},
                              output_sha256={k: meta(v)["tensor_sha256"] for k, v in result.items()}))
        begin = time.perf_counter()
        for key in pair["scalar"]:
            checks[f"repeat{repetition}/vectorized_vs_scalar/{key}"] = comparison(pair["vectorized"][key], pair["scalar"][key])
        pair_validation_seconds += time.perf_counter() - begin

    begin = time.perf_counter()
    traces = {}
    for backend, geometry in backends.items():
        if on_BA_event is not None:
            on_BA_event("instrumented", backend, None, "begin")
        output, diagnostic, systems = collect_trace(geometry, inputs)
        if on_BA_event is not None:
            on_BA_event("instrumented", backend, None, "completed")
        for key, value in post_geometry(geometry, output).items():
            checks[f"trace/{backend}/ordinary/{key}"] = comparison(value, first[backend][key])
        checks.update(compare_tree(diagnostic, saved_diagnostics, f"trace/{backend}/saved/BA"))
        for iteration, system in enumerate(systems):
            for key in SYSTEM_KEYS:
                first[backend][f"system{iteration}_{key}"] = system[key].copy()
        traces[backend] = systems
    for iteration in range(2):
        checks.update(compare_tree(traces["vectorized"][iteration], traces["scalar"][iteration],
                                   f"system{iteration}/vectorized_vs_scalar"))
    instrumented_seconds = time.perf_counter() - begin
    summary = {key:dict(first_seconds=next(r["BA_seconds"] for r in calls if r["backend"] == key),
                        warm_seconds=[r["BA_seconds"] for r in calls if r["backend"] == key and r["repetition"]],
                        warm_mean_seconds=float(np.mean([r["BA_seconds"] for r in calls if r["backend"] == key and r["repetition"]])))
               for key in backends}
    case_wall_seconds = time.perf_counter() - case_begin
    return dict(calls=calls, checks=checks, strict_FP32_gate_passed=all(x["passed"] for x in checks.values()),
                ordinary_BA_calls=8, instrumented_validation_BA_calls=2,
                instrumented_validation_seconds=instrumented_seconds,
                pairwise_validation_seconds=pair_validation_seconds,
                case_wall_seconds=case_wall_seconds,
                outside_ordinary_BA_seconds=case_wall_seconds - sum(x["BA_seconds"] for x in calls),
                summary=summary), first


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback-dir", type=Path, required=True)
    parser.add_argument("--scalar-geometry", type=Path, required=True)
    parser.add_argument("--vectorized-geometry", type=Path,
                        default=Path(__file__).with_name("dpvo_cpu_geometry_vectorized.py"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    require_board()  # Before NumPy import or opening actual saved arrays.
    output = args.output_dir.resolve()
    require(output.is_relative_to(BASE), "New isolated board experiment output required")
    output.mkdir(exist_ok=False)
    prior = os.sched_getaffinity(0)
    pins, loaded, modules, immutable = {}, [], {}, []
    report = dict(schema="v29_DPVO_frozen_snapshot_geometry_benchmark", status="failed",
                  script_sha256=sha(__file__), actual_board_execution=True, NPU_used=False,
                  neural_Update_executed=False, correlation_executed=False, complete_feedback_executed=False,
                  online_VO_executed=False, trajectory_accuracy_validated=False, realtime_validated=False,
                  cases=[], BA_events=[], cleanup_errors=[])
    def pin(path, expected):
        path = Path(path).resolve()
        require(sha(path) == expected, "Pinned source/artifact differs: " + str(path))
        pins[str(path)] = expected
        return path
    try:
        pin(__file__, report["script_sha256"])
        pin(args.scalar_geometry, SCALAR_SHA)
        pin(args.vectorized_geometry, VECTOR_SHA)
        frozen = json.loads(pin(args.feedback_dir / "run.json", RUN_SHA).read_text())
        require(frozen["status"] == "completed_fixed_graph_neural_geometry_feedback_not_VO" and
                frozen["feedback_rounds_completed"] == 4 and frozen["completed_BA_iterations"] == 8,
                "Original complete four-round feedback required")
        require({4, 5, 6, 7} <= prior, "Four A76 CPUs must initially be permitted")
        os.sched_setaffinity(0, {4, 5, 6, 7})
        require(os.sched_getaffinity(0) == {4, 5, 6, 7}, "Actual CPU affinity differs")
        import numpy as np
        report.update(numpy_version=np.__version__, OPENBLAS_NUM_THREADS=os.environ["OPENBLAS_NUM_THREADS"],
                      cpu_affinity=sorted(os.sched_getaffinity(0)), source_feedback_sha256=RUN_SHA,
                      scalar_geometry_sha256=SCALAR_SHA, vectorized_geometry_sha256=VECTOR_SHA,
                      FP32_gate=dict(rtol=RTOL, atol=ATOL, masks_and_integer_IDs_exact=True))
        load_begin = time.perf_counter()
        initial = load_artifact(args.feedback_dir, frozen["initial_state"], pin)
        rounds = [load_artifact(args.feedback_dir, row["artifact"], pin) for row in frozen["calls"]]
        loaded = [initial] + rounds
        immutable = [{k: meta(v) for k, v in x.items()} for x in loaded]
        report["input_array_hashes_before"] = immutable
        require(initial["poses"].shape == (1, 3, 7) and initial["patches"].shape == (1, 288, 3, 3, 3), "Frozen geometry shape differs")
        require(np.array_equal(initial["intrinsics"], np.asarray(frozen["graph_contract"]["intrinsics"], np.float32)), "Initial K binding differs")
        require(np.array_equal(initial["kk"], np.repeat(np.arange(288, dtype=np.int64), 3)) and
                np.array_equal(initial["jj"], np.tile(np.arange(3, dtype=np.int64), 288)) and
                np.array_equal(initial["ii"], np.repeat(np.arange(3, dtype=np.int64), 96)[initial["kk"]]), "Frozen graph edge order differs")
        for index, row in enumerate(rounds):
            require(frozen["calls"][index]["iteration"] == index, "Case order differs")
            previous = initial if index == 0 else rounds[index - 1]
            require(np.array_equal(row["poses_before"], previous["poses" if index == 0 else "poses_after"]) and
                    np.array_equal(row["patches_before"], previous["patches" if index == 0 else "patches_after"]), "Saved geometry continuity differs")
            require(np.array_equal(row["target"], row["coords_before"][..., 1, 1] + row["delta"]), "Frozen target binding differs")
        report["artifact_load_and_validation_seconds"] = time.perf_counter() - load_begin
        load_begin = time.perf_counter()
        modules = {"scalar": load_module(args.scalar_geometry, MODULE_NAMES[0]),
                   "vectorized": load_module(args.vectorized_geometry, MODULE_NAMES[1])}
        report["geometry_module_load_seconds"] = time.perf_counter() - load_begin
        for index, source in enumerate(rounds):
            print(json.dumps(dict(event="geometry_case_begin", case=index)), flush=True)
            inputs = dict(poses=source["poses_before"], patches=source["patches_before"],
                          intrinsics=initial["intrinsics"], target=source["target"], weight=source["weight"],
                          ii=initial["ii"], jj=initial["jj"], kk=initial["kk"], **{"lambda": np.array([1e-4], np.float32)})
            reference = {k: source[k] for k in ("poses_after", "patches_after", "coords_after", "projection_after", "residual_after", "active_after")}
            def event(kind, backend, repetition, stage):
                report["BA_events"].append(dict(case=index, kind=kind, backend=backend,
                                               repetition=repetition, stage=stage, monotonic=time.monotonic()))
            row, saved = run_case(modules["scalar"], modules["vectorized"], inputs, reference,
                                  frozen["calls"][index]["ba"], event)
            row.update(case_index=index, input_sha256={k: meta(v)["tensor_sha256"] for k, v in inputs.items()},
                       source_artifact_sha256=frozen["calls"][index]["artifact"]["sha256"], artifacts={})
            start = time.perf_counter()
            for backend, values in saved.items():
                path = output / f"case-{index:02d}-{backend}.npz"
                np.savez(path, **values)
                row["artifacts"][backend] = dict(file=path.name, sha256=sha(path), bytes=path.stat().st_size,
                                                 arrays={k: meta(v) for k, v in values.items()})
            row["artifact_save_and_hash_seconds"] = time.perf_counter() - start
            report["cases"].append(row)
            print(json.dumps(dict(event="geometry_case_complete", case=index, gate=row["strict_FP32_gate_passed"], summary=row["summary"])), flush=True)
        report.update(status="completed_geometry_only_benchmark", strict_FP32_gate_passed=all(r["strict_FP32_gate_passed"] for r in report["cases"]),
                      candidate_accepted=all(r["strict_FP32_gate_passed"] for r in report["cases"]),
                      timing_scope="Only BA(...iterations=2) is in ordinary BA_seconds, including its internal copy-on-success, linearization, assembly, Schur solve, SE3/depth updates and diagnostics. External copies/hashes/checks/projections/save are outside. Separate instrumented validations excluded from mean.",
                      first_call_semantics="First per case/backend, no BA warmup. Modules/process/BLAS may already be warm from earlier cases; not independent process-cold measurements.",
                      order="Each case repeats AB, BA, AB, BA with unchanged copied input; checks occur between calls. Temperature/frequency are not controlled.",
                      limitations=["Frozen snapshot BA only, not continuous feedback, neural/correlation latency, trajectory or 20Hz.",
                                   "Eight extra instrumented BA validations expose both iteration systems; they are excluded from ordinary latency and explicitly counted.",
                                   "Fixed FP32 per-element gate and exact masks/IDs; no clipping, fallback or tolerance changes.",
                                   "Four-CPU affinity does not imply four BLAS threads; OPENBLAS_NUM_THREADS is explicitly1."])
    except Exception as exc:
        report.update(error=repr(exc), traceback=traceback.format_exc())
    finally:
        for kind in ("ordinary", "instrumented"):
            for stage, field in (("begin", "attempted"), ("completed", "completed")):
                report[f"{kind}_BA_calls_{field}"] = sum(e["kind"] == kind and e["stage"] == stage for e in report["BA_events"])
        report["BA_iterations_in_completed_calls"] = 2 * (report["ordinary_BA_calls_completed"] + report["instrumented_BA_calls_completed"])
        final_hash_begin = time.perf_counter()
        try:
            after = {path: sha(path) for path in pins}
            report.update(source_hashes_after=after, sources_and_artifacts_unchanged=after == pins)
            require(after == pins, "Source or captured artifact changed")
        except Exception as exc:
            report["cleanup_errors"].append(repr(exc))
        try:
            if loaded:
                report["input_array_hashes_after"] = [{k: meta(v) for k, v in x.items()} for x in loaded]
                report["immutable_arrays_unchanged"] = immutable == report["input_array_hashes_after"]
                require(report["immutable_arrays_unchanged"], "Frozen source arrays mutated")
        except Exception as exc:
            report["cleanup_errors"].append(repr(exc))
        report["final_source_and_array_hash_seconds"] = time.perf_counter() - final_hash_begin
        modules.clear()
        for name in MODULE_NAMES:
            sys.modules.pop(name, None)
        loaded.clear()
        initial = rounds = inputs = reference = saved = source = None
        gc.collect()
        try:
            os.sched_setaffinity(0, prior)
            report["affinity_restored"] = os.sched_getaffinity(0) == prior
            require(report["affinity_restored"], "Affinity restoration failed")
        except Exception as exc:
            report["cleanup_errors"].append(repr(exc))
        if report["cleanup_errors"]:
            report["status"] = "failed"
            report["candidate_accepted"] = False
        report.update(source_pins=pins, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        (output / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report.get(k) for k in ("status", "strict_FP32_gate_passed", "candidate_accepted", "cleanup_errors", "error")}), flush=True)
    return 0 if report["status"] == "completed_geometry_only_benchmark" and report.get("candidate_accepted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
