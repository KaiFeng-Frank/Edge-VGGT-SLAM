#!/usr/bin/env python3
"""RK3588-only four-round ORT Update / CPU correlation / SE3-BA research.

The fixed three captured views and identity/unit-depth initialization are the
frozen v28 experiment, not the official initializer or a complete VO. Every
ORT update is checked against the original Torch Update on that SAME current
input, outside the feedback timer. Those reference calls are not online work
and are disclosed separately. No image decoding, encoder, NPU or GT is used.

DPVO and the reused v28 mathematics: Copyright (c) 2022 Princeton Vision &
Learning Lab, MIT; see LICENSE. This integration is also MIT licensed.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import resource
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
BASE = Path("/mnt/vggt-data/vggt-slam2/experiments")
FEEDBACK_SHA = "e35a2bde3154f527a010b0cf7a2f3192bfe6e356ffd3f6b6259a9289e6bbed42"
PREFLIGHT_DRAFT_SHA = "d39da4902a23088908dd0ffcf7b6a49123b7b62264db3c69f93c3ff8c4f74cc8"
GEOMETRY_SHA = "c893f12d188baced438504cfc592a8900e18e40ef8223af0a7e944e4484b0c3b"
ADAPTER_SHA = "3f755c8e24d092fa55fdf698d87c3d9ba8944487ee95d4e5afc47554d31d1380"
UPDATE_CAPTURE_SHA = "87e18137a828496d623721f7748e288d02f9d8d6879278da1e5d3d9731cfa0a1"
UPDATE_EXPORT_SHA = "56f96b7bf80bde596de6a4008da2858a869a9f19a3162abb8db8ab1af0156cac"
MANIFEST_SHA = "27c8ec9c3e68f65a1f7648fe11031967eb37cad588a83351ce0bd1f9393765dc"
INPUT_NAMES = (
    "net", "inp", "corr", "previous_indices", "previous_valid",
    "following_indices", "following_valid", "kk_indices", "kk_valid",
    "kk_inverse", "ij_indices", "ij_valid", "ij_inverse",
)
OUTPUT_NAMES = ("net_out", "delta", "weight")
RTOL, ATOL = 1e-4, 1e-5
SUCCESS = "completed_fixed_graph_ORT_neural_geometry_feedback_not_VO"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def meta(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes,
                tensor_sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest())


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def require_board():
    require(platform.machine() == "aarch64" and platform.system() == "Linux",
            "RK3588 board only; host model/feature execution is prohibited")
    require(b"rockchip,rk3588" in Path("/proc/device-tree/compatible").read_bytes(),
            "RK3588 device-tree evidence required")


def safe_child(folder, name):
    require(isinstance(name, str) and name not in ("", ".", "..") and
            Path(name).name == name, "Artifact must have a safe basename")
    return Path(folder) / name


def validate_acceptance(receipt, manifest, captured, exported, manifest_sha):
    """Fail closed on saved-state coverage before any Torch/ORT model loading."""
    require(manifest_sha == MANIFEST_SHA and
            manifest["schema"] == "v29_DPVO_Update_candidate_manifest" and
            manifest["source_export_sha256"] == UPDATE_EXPORT_SHA and
            manifest["source_capture_sha256"] == UPDATE_CAPTURE_SHA,
            "Only the pinned sigmoid-only candidate is accepted")
    require(captured["status"] == "captured_original_Update_states_not_VO" and
            exported["status"] == "exported_Update_board_ORT_pending" and
            exported["capture_sha256"] == UPDATE_CAPTURE_SHA and
            exported["adapter_sha256"] == ADAPTER_SHA and
            exported["exact_wrapper_FP32_gate_passed"] is True,
            "Frozen original Update capture/export contract differs")
    for ancestor in (captured, exported):
        require(ancestor["original_weights_unchanged"] is True and
                ancestor["sources_and_inputs_unchanged"] is True and
                ancestor["source_pins"] == ancestor["source_hashes_after"] and
                not ancestor["cleanup_errors"], "Ancestor weights/source/cleanup check failed")
    require(receipt["schema"] == "v29_Update_candidate_board_diagnostic" and
            receipt["status"] == "completed_numerical_accepted" and
            receipt["execution_status"] == "completed" and
            receipt["candidate_accepted"] is True and receipt["mode"] == "ort" and
            receipt["manifest_sha256"] == manifest_sha and
            receipt["capture_sha256"] == UPDATE_CAPTURE_SHA and
            receipt["export_sha256"] == UPDATE_EXPORT_SHA,
            "A completed, numerically accepted same-candidate CPU ORT receipt is required")
    require(receipt["first_only"] is False and receipt["model_keys"] == ["E864", "E96"] and
            receipt["selected_saved_states"] == 16 and len(receipt["calls"]) == 16 and
            receipt["providers"] == ["CPUExecutionProvider"] and
            receipt["ORT_optimization"] == "disabled" and
            receipt["sources_and_inputs_unchanged"] is True and
            receipt["source_pins"] == receipt["source_hashes_after"] and
            not receipt["cleanup_errors"], "All sixteen states, CPU-only provider and clean sources are required")
    require(set(manifest["models"]) == {"E96", "E864"}, "Both original model shapes are required")
    planned = []
    require(len(captured["scenarios"]) == len(exported["scenarios"]) == 4,
            "Exactly four original scenarios are required")
    for case, rewritten in zip(captured["scenarios"], exported["scenarios"]):
        require(case["name"] == rewritten["name"] and
                [s["step"] for s in case["states"]] == [0, 1, 2, 3] and
                [s["step"] for s in rewritten["states"]] == [0, 1, 2, 3],
                "Original scenario order/four states differ")
        key = rewritten["model_key"]
        require(key in ("E96", "E864"), "Unknown original model shape")
        planned.extend((key, case["name"], step) for step in range(4))
    require(sum(key == "E864" for key, _, _ in planned) == 8 and
            sum(key == "E96" for key, _, _ in planned) == 8 and len(set(planned)) == 16,
            "Exact eight-state coverage for both shapes required")
    require([(r["model_key"], r["case"], r["step"]) for r in receipt["calls"]] == planned,
            "Acceptance calls must cover original states in exact order, without selection")
    for index, call in enumerate(receipt["calls"]):
        require(call["index"] == index and call["all_outputs_finite"] is True and
                call["weight_domain_passed"] is True and
                call["core_strict_FP32_gate_passed"] is True,
                "Every accepted state must pass finite, weight domain and original FP32 gates")
        for group in ("versus_original", "versus_wrapper"):
            require(set(call[group]) == {"net", "delta", "weight"} and
                    all(check["gate_passed"] is True and check["rtol"] == RTOL and
                        check["atol"] == ATOL for check in call[group].values()),
                    "Every state must preserve the predeclared original/wrapper tolerances")
    for key, candidate in manifest["models"].items():
        original = exported["models"][key]
        require(candidate["source_onnx_sha256"] == original["sha256"] and
                candidate["inputs"] == original["inputs"] and candidate["outputs"] == original["outputs"] and
                candidate["role"] == "stable_weight_sigmoid" and
                candidate["original_initializers_byte_unchanged"] is True and
                candidate["other_nodes_byte_unchanged"] is True,
                "Candidate may only change the final Sigmoid expression")
    return planned


def check_session(session, model):
    require(session.get_providers() == ["CPUExecutionProvider"], "CPU ORT only")
    for actual, wanted in ((session.get_inputs(), model["inputs"]),
                           (session.get_outputs(), model["outputs"])):
        require([v.name for v in actual] == [v["name"] for v in wanted], "ORT I/O names/order differ")
        for value, item in zip(actual, wanted):
            expected_type = "tensor(int64)" if item["dtype"] == 7 else "tensor(float)"
            require(item["dtype"] in (1, 7) and value.shape == item["shape"] and value.type == expected_type,
                    "ORT I/O shape/type differs")
    require(tuple(x["name"] for x in model["inputs"]) == INPUT_NAMES and
            tuple(x["name"] for x in model["outputs"]) == OUTPUT_NAMES,
            "Exactly the original thirteen inputs and three outputs required")


def validate_geometry_evidence(receipt):
    require(receipt["status"] == "completed_geometry_only_benchmark" and
            receipt["candidate_accepted"] is True and receipt["strict_FP32_gate_passed"] is True and
            receipt["vectorized_geometry_sha256"] == GEOMETRY_SHA and
            receipt["scalar_geometry_sha256"] == "13eb1e03eb4a5ed1bd1533549f2b9f4994abfaa61e0b55304b99e6fe77d813df" and
            receipt["sources_and_artifacts_unchanged"] is True and
            receipt["immutable_arrays_unchanged"] is True and
            receipt["source_pins"] == receipt["source_hashes_after"] and
            GEOMETRY_SHA in receipt["source_pins"].values() and not receipt["cleanup_errors"],
            "Accepted same-source vectorized geometry evidence required")
    require(receipt["FP32_gate"] == dict(rtol=RTOL, atol=ATOL, masks_and_integer_IDs_exact=True) and
            receipt["ordinary_BA_calls_attempted"] == receipt["ordinary_BA_calls_completed"] == 32 and
            receipt["instrumented_BA_calls_attempted"] == receipt["instrumented_BA_calls_completed"] == 8 and
            len(receipt["cases"]) == 4 and
            all(case["strict_FP32_gate_passed"] is True for case in receipt["cases"]),
            "Complete original geometry gate/call accounting required")


class ORTUpdate:
    """Fixed, once-validated topology; each call owns its feed and output copies."""
    def __init__(self, session, model, topology, graph):
        import numpy as np
        check_session(session, model)
        require(tuple(topology) == INPUT_NAMES[3:], "Exact ten external topology arrays required")
        self.session, self.model = session, model
        self.topology = {name: np.array(value, copy=True, order="C") for name, value in topology.items()}
        self.graph = {name: np.array(graph[name], copy=True, order="C") for name in ("ii", "jj", "kk")}
        self.attempted_calls = 0
        self.calls = []

    def __call__(self, net, context, corr, ii, jj, kk):
        import numpy as np
        require(self.session is not None, "ORT session is closed")
        self.attempted_calls += 1
        begin = time.monotonic()
        require(all(np.array_equal(value, self.graph[name]) for name, value in
                    (("ii", ii), ("jj", jj), ("kk", kk))), "Fixed feedback graph IDs changed")
        feed = {"net": net, "inp": context, "corr": corr, **self.topology}
        for item in self.model["inputs"]:
            value = feed[item["name"]]
            dtype = np.int64 if item["dtype"] == 7 else np.float32
            require(isinstance(value, np.ndarray) and value.dtype == dtype and
                    list(value.shape) == item["shape"] and np.isfinite(value).all(),
                    "Current ORT feed shape/dtype/finite gate failed: " + item["name"])
        prepared = {key: np.array(value, copy=True, order="C") for key, value in feed.items()}
        packed = time.monotonic()
        results = self.session.run(list(OUTPUT_NAMES), prepared)
        ran = time.monotonic()
        require(len(results) == 3, "ORT output count differs")
        owned = tuple(np.array(value, copy=True, order="C") for value in results)
        copied = time.monotonic()
        for value, item in zip(owned, self.model["outputs"]):
            require(value.dtype == np.float32 and list(value.shape) == item["shape"] and
                    np.isfinite(value).all(), "Nonfinite or malformed ORT output: " + item["name"])
        require(np.all((owned[2] >= 0) & (owned[2] <= 1)), "ORT weight is outside [0,1]; no clipping allowed")
        checked = time.monotonic()
        self.calls.append(dict(index=len(self.calls), begin_monotonic=begin, end_monotonic=checked,
                               input_check_and_pack_seconds=packed-begin, run_seconds=ran-packed,
                               owned_copy_seconds=copied-ran, output_check_seconds=checked-copied,
                               total_seconds=checked-begin))
        return owned

    def close(self):
        self.session = None


def strict_compare(actual, expected):
    import numpy as np
    require(actual.dtype == expected.dtype == np.float32 and actual.shape == expected.shape,
            "Strict reference comparison requires identical FP32 shapes")
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), "Nonfinite reference comparison")
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    failed = int(np.count_nonzero(np.abs(difference) > ATOL + RTOL * np.abs(expected.astype(np.float64))))
    return dict(rtol=RTOL, atol=ATOL, gate_passed=failed == 0, failed_elements=failed,
                max_abs=float(np.abs(difference).max()),
                relative_l2=float(np.linalg.norm(difference.ravel()) /
                                  max(np.linalg.norm(expected.astype(np.float64).ravel()), 1e-30)),
                array_equal=bool(np.array_equal(actual, expected)),
                bitwise_equal=actual.tobytes(order="C") == expected.tobytes(order="C"))


def reference_round(reference, arrays, graph):
    """Exactly one golden call, on the saved ORT-current state, outside feedback."""
    started = time.monotonic()
    expected = reference(arrays["net_before"], graph["ctx"], arrays["corr"],
                         graph["ii"], graph["jj"], graph["kk"])
    ended = time.monotonic()
    require(len(expected) == 3, "Original Update returned incorrect reference count")
    checks = {name: strict_compare(arrays[key], value) for name, key, value in
              zip(OUTPUT_NAMES, ("net_after", "delta", "weight"), expected)}
    return expected, dict(begin_monotonic=started, end_monotonic=ended,
                          reference_forward_seconds=ended-started, comparisons=checks,
                          strict_FP32_gate_passed=all(row["gate_passed"] for row in checks.values()),
                          same_current_net_before_and_correlation=True,
                          inputs={name: meta(value) for name, value in
                                  (("net", arrays["net_before"]), ("inp", graph["ctx"]),
                                   ("corr", arrays["corr"]), ("ii", graph["ii"]),
                                   ("jj", graph["jj"]), ("kk", graph["kk"]))})


def preflight_correlation(saved, graph):
    """Use captured identity-state corr only for the once-only input validator.

    The frozen v26 motion graph has no corr. Real feedback still recomputes
    correlation each round; this value is never passed to an inference call.
    """
    import numpy as np
    require(set(saved) == {"net", "ctx", "corr", "ii", "jj", "kk"},
            "Original Update preflight input schema differs")
    for name, key in (("net", "net0"), ("ctx", "ctx"), ("ii", "ii"), ("jj", "jj"), ("kk", "kk")):
        value, original = saved[name], graph[key]
        require(value.dtype == original.dtype and value.shape == original.shape and
                value.tobytes(order="C") == original.tobytes(order="C"),
                "Original captured preflight " + name + " differs from fixed motion graph")
    corr = saved["corr"]
    require(corr.dtype == np.float32 and corr.shape == (1, len(graph["kk"]), 882) and
            np.isfinite(corr).all(), "Captured preflight correlation requires finite original FP32 shape")
    return corr


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("v26-source-dir", "capture-dir", "motion-dir", "checkpoint",
                 "native-source-dir", "library", "build-receipt", "output-dir",
                 "frozen-feedback", "adapter", "candidate-manifest", "candidate-evidence-dir",
                 "update-capture-dir", "update-export-dir", "geometry-evidence-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--candidate-manifest-sha256", required=True)
    parser.add_argument("--candidate-evidence-sha256", required=True)
    parser.add_argument("--geometry-evidence-sha256", required=True)
    parser.add_argument("--geometry", type=Path, default=Path(__file__).with_name("dpvo_cpu_geometry_vectorized.py"))
    args = parser.parse_args(argv)
    require_board()  # Before captured arrays, checkpoint, Torch and ORT.
    output = args.output_dir.resolve()
    require(output.is_relative_to(BASE), "New output inside isolated board experiments required")
    output.mkdir(exist_ok=False)
    prior_affinity = os.sched_getaffinity(0)
    pins, immutable = {}, {}
    neural = cached = updater = session = state = topology_adapter = None
    drive_begin = time.monotonic()
    report = dict(schema="v29_DPVO_fixed_three_frame_ORT_feedback", status="failed",
                  script_sha256=sha(__file__), actual_board_execution=True,
                  preflight_draft_sha256=PREFLIGHT_DRAFT_SHA,
                  preflight_revision="Read original captured E864 identity state0 corr; v26 motion graph has no corr field",
                  original_DPVO_neural_weights=True, original_VGGT_weights_used=False,
                  production_changed=False, ORB_used=False, NPU_used=False, CUDA_used=False,
                  GT_or_sensor_depth_used=False, BA_executed=False, feedback_rounds_completed=0,
                  feedback_rounds_executed=0, expected_ORT_Update_calls=4,
                  expected_original_Torch_reference_calls=4, original_Torch_reference_calls_attempted=0,
                  original_Torch_reference_calls_completed=0,
                  complete_VO_executed=False, official_eight_frame_initialization_executed=False,
                  streamed_or_online_input=False, trajectory_accuracy_validated=False,
                  realtime_validated=False, candidate_accepted=False,
                  calls=[], stage_events=[], cleanup_errors=[])

    def pin(path, expected):
        path = Path(path).resolve()
        require(sha(path) == expected, "Pinned file differs: " + str(path))
        pins[str(path)] = expected
        return path

    def load_artifact(folder, item, metadata=None):
        path = pin(safe_child(folder, item["file"]), item["sha256"])
        require(path.stat().st_size == item["bytes"], "Artifact length differs")
        with np.load(path, allow_pickle=False) as store:
            values = {key: store[key].copy() for key in store.files}
        wanted = item["arrays"] if metadata is None else metadata
        require({key: meta(value) for key, value in values.items()} == wanted and
                all(np.isfinite(value).all() for value in values.values()),
                "Saved artifact raw metadata/finite gate failed")
        return values

    def save(name, values):
        path = output / name
        np.savez(path, **values)
        return dict(file=name, sha256=sha(path), bytes=path.stat().st_size,
                    arrays={key: meta(value) for key, value in values.items()})

    try:
        pin(__file__, report["script_sha256"])
        parent = module(pin(args.frozen_feedback, FEEDBACK_SHA), "_v29_pinned_feedback")
        pin(args.geometry, GEOMETRY_SHA)
        pin(args.adapter, ADAPTER_SHA)
        original_update_capture = json.loads(pin(args.update_capture_dir / "capture.json", UPDATE_CAPTURE_SHA).read_text())
        original_export = json.loads(pin(args.update_export_dir / "export.json", UPDATE_EXPORT_SHA).read_text())
        manifest = json.loads(pin(args.candidate_manifest, args.candidate_manifest_sha256).read_text())
        evidence = json.loads(pin(args.candidate_evidence_dir / "run.json", args.candidate_evidence_sha256).read_text())
        planned = validate_acceptance(evidence, manifest, original_update_capture, original_export,
                                      args.candidate_manifest_sha256)
        for path, digest in evidence["source_pins"].items():
            pin(path, digest)
        geometry_evidence = json.loads(pin(args.geometry_evidence_dir / "run.json",
                                          args.geometry_evidence_sha256).read_text())
        validate_geometry_evidence(geometry_evidence)
        for path, digest in geometry_evidence["source_pins"].items():
            pin(path, digest)
        for key, row in manifest["models"].items():
            for folder, item in ((args.candidate_manifest.parent, row),
                                 (args.update_export_dir, original_export["models"][key])):
                path = pin(safe_child(folder, item["file"]), item["sha256"])
                require(path.stat().st_size == item["bytes"], "Original/candidate ONNX length differs")
        # Only NumPy / saved arrays so far; no executable inference session.
        import numpy as np
        for call in evidence["calls"]:
            values = load_artifact(args.candidate_evidence_dir, call["outputs"])
            require(tuple(values) == OUTPUT_NAMES, "Accepted artifact output order differs")
            for name, value in values.items():
                row = next(r for r in manifest["models"][call["model_key"]]["outputs"] if r["name"] == name)
                require(value.dtype == np.float32 and list(value.shape) == row["shape"],
                        "Accepted output shape/dtype differs")
            require(np.all((values["weight"] >= 0) & (values["weight"] <= 1)),
                    "Accepted saved-state weight violates [0,1]")
        report["prior_candidate_acceptance"] = dict(manifest_sha256=args.candidate_manifest_sha256,
                                                     receipt_sha256=args.candidate_evidence_sha256,
                                                     states=planned, strict_FP32_gate_passed=True,
                                                     original_weight_domain_passed=True,
                                                     inference_not_run_until_acceptance_checked=True)
        report["prior_geometry_acceptance"] = dict(receipt_sha256=args.geometry_evidence_sha256,
                                                    vectorized_geometry_sha256=GEOMETRY_SHA,
                                                    strict_FP32_gate_passed=True,
                                                    ordinary_BA_calls_completed=32,
                                                    instrumented_BA_calls_completed=8)
        for name, digest in parent.PINS.items():
            pin(args.v26_source_dir / name, digest)
        build = json.loads(pin(args.build_receipt, parent.BUILD_SHA).read_text())
        require(build["status"] == "compiled_and_synthetic_smoke_passed" and
                build["source_sha256"] == parent.NATIVE_PINS and build["output"]["sha256"] == parent.LIBRARY_SHA,
                "Frozen ARM native build chain differs")
        pin(args.library, parent.LIBRARY_SHA)
        for name, digest in parent.NATIVE_PINS.items():
            pin(args.native_source_dir / name, digest)
        pin(args.checkpoint, parent.CHECKPOINT_SHA)
        capture = json.loads(pin(args.capture_dir / "capture.json", parent.CAPTURE_SHA).read_text())
        motion = json.loads(pin(args.motion_dir / "motion.json", parent.MOTION_SHA).read_text())
        input_manifest = json.loads((args.v26_source_dir / "input-manifest.json").read_text())
        require(capture["status"] == "completed_encoder_cpu_probe_not_VO" and
                motion["status"] == "completed_dynamic_neural_probe_not_VO" and
                motion["capture_sha256"] == parent.CAPTURE_SHA and
                capture["input_manifest_sha256"] == parent.PINS["input-manifest.json"],
                "Frozen real capture/motion chain required")
        frame_ids = [row["frame_id"] for row in input_manifest["frames"]]
        require(frame_ids == motion["frame_ids"] == [row["frame_id"] for row in capture["inputs"]] and len(frame_ids) == 3,
                "Original three frame IDs/order differ")
        require({4, 5, 6, 7} <= prior_affinity, "All four A76 cores must be initially permitted")
        os.sched_setaffinity(0, {4, 5, 6, 7})
        require(os.sched_getaffinity(0) == {4, 5, 6, 7}, "Actual CPU affinity differs")
        import torch
        import onnxruntime as ort
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        require(ort.__version__ == evidence["ORT_version"], "ORT version differs from accepted candidate evidence")
        geo = module(args.geometry, "_v29_vectorized_geometry")
        common = module(args.v26_source_dir / "dpvo_probe_common.py", "_v29_frozen_common")
        native = module(args.native_source_dir / "dpvo_corr.py", "_v29_frozen_corr")
        adapter = module(args.adapter, "_v29_frozen_external_topology")
        require(adapter.INPUT_NAMES == INPUT_NAMES and adapter.OUTPUT_NAMES == OUTPUT_NAMES,
                "Frozen adapter I/O contract differs")
        report.update(frame_ids=frame_ids, input_manifest_sha256=parent.PINS["input-manifest.json"],
                      capture_sha256=parent.CAPTURE_SHA, motion_sha256=parent.MOTION_SHA,
                      geometry_sha256=GEOMETRY_SHA, frozen_feedback_sha256=FEEDBACK_SHA, adapter_sha256=ADAPTER_SHA,
                      candidate_manifest_sha256=args.candidate_manifest_sha256,
                      candidate_evidence_sha256=args.candidate_evidence_sha256,
                      library_sha256=parent.LIBRARY_SHA, build_sha256=parent.BUILD_SHA,
                      cpu_affinity=sorted(os.sched_getaffinity(0)), torch_threads=4, torch_interop_threads=1,
                      ORT_version=ort.__version__, ORT_threads=4, ORT_interop_threads=1,
                      ORT_optimization="disabled", providers=["CPUExecutionProvider"],
                      numpy_version=np.__version__, torch_version=torch.__version__,
                      OMP_WAIT_POLICY=os.environ.get("OMP_WAIT_POLICY"), GOMP_SPINCOUNT=os.environ.get("GOMP_SPINCOUNT"),
                      OPENBLAS_NUM_THREADS=os.environ.get("OPENBLAS_NUM_THREADS"), OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS"))
        start = time.monotonic()
        neural, loader = common.load_neural(args.v26_source_dir / "official", args.v26_source_dir / "manifest.json",
                                           args.checkpoint, parent.CHECKPOINT_SHA, args.v26_source_dir / "dpvo_cpu_ops.py",
                                           cpu_ops_sha256=parent.PINS["dpvo_cpu_ops.py"])
        report.update(neural_load_seconds=time.monotonic()-start, loader=loader)
        for relative, value in loader["source_hashes"].items():
            pin(args.v26_source_dir / "official" / relative, value["sha256"])
        state = {key: meta(value.detach().numpy()) for key, value in neural.state_dict().items()}
        require(len(state) == 94 and state == capture["state_before"] == motion["original_weights_before"] ==
                original_update_capture["original_weights_before"] == original_export["original_weights_before"],
                "Original 94 parameter bytes differ")
        report["original_weights_before"] = state
        patch = load_artifact(args.motion_dir, motion["patches"])
        scenario = next(row for row in motion["scenarios"] if row["name"] == "three_frame_all_pairs")
        graph = load_artifact(args.motion_dir, scenario["graph"])
        feature_list = []
        for index in range(3):
            row = next(row for row in capture["calls"] if row["frame_index"] == index)
            feature_list.append(load_artifact(args.capture_dir, row["artifact"], row["arrays"])["fmap"])
        fmap = np.ascontiguousarray(np.concatenate(feature_list, axis=1))
        gmap = patch["gmap"]
        parent.validate_value(fmap, (1, 3, 128, 116, 152), "saved fmap")
        parent.validate_value(gmap, (1, 288, 128, 3, 3), "saved gmap")
        require(scenario["edge_count"] == 864 and graph["kk"].shape == (864,) and
                np.array_equal(graph["kk"], np.repeat(np.arange(288, dtype=np.int64), 3)) and
                np.array_equal(graph["jj"], np.tile(np.arange(3, dtype=np.int64), 288)) and
                np.array_equal(graph["ii"], patch["source_frame"][graph["kk"]]) and
                np.array_equal(patch["source_frame"], np.repeat(np.arange(3, dtype=np.int64), 96)),
                "Frozen E864 all-pairs edge ownership/order differs")
        require(np.array_equal(graph["ctx"], patch["contexts"][:, graph["kk"]]) and
                not np.any(graph["net0"]) and np.all(patch["patches"][:, :, 2] == 1),
                "Original context/zero net/unit inverse-depth differs")
        intrinsics = np.asarray([row["pixel_intrinsics"] for row in capture["inputs"]], np.float32)[None] / np.float32(4)
        require(np.all(intrinsics == intrinsics[:, :1]), "Online CUDA BA common-K contract requires identical views")
        poses = np.zeros((1, 3, 7), np.float32)
        poses[..., 6] = 1
        patches, net = patch["patches"].copy(), graph["net0"].copy()
        coords = geo.reproject(poses, patches, intrinsics, graph["ii"], graph["jj"], graph["kk"])
        require(np.allclose(coords, graph["coords"], rtol=RTOL, atol=ATOL), "Initial reprojection differs from frozen identity graph")
        original_identity = next(case for case in original_update_capture["scenarios"]
                                 if case["name"] == "three_frame_all_pairs_identity")
        require(original_identity["edges"] == 864 and original_identity["states"][0]["step"] == 0,
                "Original E864 identity state0 preflight required")
        preflight_item = original_identity["states"][0]["inputs"]
        preflight_values = load_artifact(args.update_capture_dir, preflight_item)
        preflight_corr = preflight_correlation(preflight_values, graph)
        report["topology_preflight_input"] = dict(case=original_identity["name"], step=0,
                                                  file=preflight_item["file"], sha256=preflight_item["sha256"],
                                                  arrays=preflight_item["arrays"],
                                                  graph_net_context_edge_ids_bitwise_equal=True,
                                                  used_only_for_input_validation_not_feedback=True)
        topology_begin = time.monotonic()
        topology = adapter.build_topology(graph["ii"], graph["jj"], graph["kk"],
                                           group_caps={"kk": (288, 3), "ij": (9, 96)})
        topology_adapter = adapter.ExportableUpdate(neural.update).eval()
        adapter.validate_inputs(topology_adapter, (torch.from_numpy(net), torch.from_numpy(graph["ctx"]),
                                                  torch.from_numpy(preflight_corr), *topology.tensors()))
        topology_values = topology.numpy_dict()
        feed_topology = {name: topology_values[name] for name in INPUT_NAMES[3:]}
        require(set(topology_values) == set(INPUT_NAMES[3:]) | {"ii", "jj", "kk", "kk_labels", "ij_labels"},
                "Topology export metadata differs")
        model = manifest["models"]["E864"]
        for item in model["inputs"][3:]:
            value = feed_topology[item["name"]]
            require(list(value.shape) == item["shape"] and value.dtype == (np.int64 if item["dtype"] == 7 else np.float32),
                    "Once-built topology differs from candidate shape/dtype")
        report.update(topology_build_and_validate_seconds=time.monotonic()-topology_begin,
                      topology_built_once_outside_feedback=True, topology=topology.summary(),
                      topology_artifact=save("fixed-topology.npz", topology_values))
        immutable = {"fmap": (fmap, meta(fmap)), "gmap": (gmap, meta(gmap))}
        immutable.update({"graph/"+key: (value, meta(value)) for key, value in graph.items()})
        immutable.update({"patch_input/"+key: (value, meta(value)) for key, value in patch.items()})
        immutable.update({"topology/"+key: (value, meta(value)) for key, value in topology_values.items()})
        immutable.update({"preflight/"+key: (value, meta(value)) for key, value in preflight_values.items()})
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        start = time.monotonic()
        session = ort.InferenceSession(str(args.candidate_manifest.parent / model["file"]),
                                       sess_options=options, providers=["CPUExecutionProvider"])
        updater = ORTUpdate(session, model, feed_topology, graph)
        report["ORT_session_load_seconds"] = time.monotonic()-start
        feature_key = meta(fmap)["tensor_sha256"]+":"+meta(gmap)["tensor_sha256"]
        cached = native.PreparedCorrelation(args.library, gmap, fmap, threads=4,
                                             cache_key=feature_key, library_sha256=parent.LIBRARY_SHA)
        require(cached.preparation["native"]["neon"] == 1, "Actual ARM NEON build required")
        report.update(cache_preparation=cached.preparation,
                      initial_state=save("initial-state.npz", dict(poses=poses, patches=patches, intrinsics=intrinsics,
                                                                  ii=graph["ii"], jj=graph["jj"], kk=graph["kk"], net=net)),
                      initial_reprojection_vs_frozen_identity=dict(rtol=RTOL, atol=ATOL, passed=True,
                                                                  max_abs=float(np.abs(coords-graph["coords"]).max())),
                      graph_contract=dict(frames=3, patches_per_view=96, total_patches=288, edges=864,
                                          optimized_pose_interval=[1, 3], fixed_pose_ids=[0], feature_resolution=[116, 152],
                                          intrinsics=intrinsics.tolist(),
                                          pixel_intrinsics=[row["pixel_intrinsics"] for row in capture["inputs"]],
                                          initialization="identity poses and unit inverse depth; frozen graph, not official initialization"))

        def golden_update(previous_net, context, correlation, ii, jj, kk):
            output_net, (delta, weight, unused) = neural.update(
                torch.from_numpy(previous_net), torch.from_numpy(context), torch.from_numpy(correlation),
                None, torch.from_numpy(ii), torch.from_numpy(jj), torch.from_numpy(kk))
            require(unused is None, "Unexpected original Update auxiliary output")
            return tuple(value.detach().cpu().numpy().copy() for value in (output_net, delta, weight))

        def on_stage(stage):
            report["active_stage"] = stage
            report["stage_events"].append(dict(iteration=report["active_round"], stage=stage, monotonic=time.monotonic()))
            if stage == "BA_completed":
                report["BA_executed"] = True

        with torch.inference_mode():
            for iteration in range(4):
                report["active_round"] = iteration
                print(json.dumps(dict(event="feedback_round_begin", iteration=iteration)), flush=True)
                net, arrays, row = parent.feedback_round(geo, poses, patches, intrinsics, graph, net, cached, updater, on_stage)
                row.update(iteration=iteration, recurrent_net_carried_from_previous_ORT_round=iteration > 0,
                           geometry_revision_before=iteration, geometry_revision_after=iteration+1,
                           ORT_update_timing=updater.calls[-1])
                report["calls"].append(row)
                report["feedback_rounds_executed"] = iteration+1
                row["artifact"] = save("feedback-round-%02d.npz" % iteration, arrays)
                report["active_stage"] = "original_Torch_reference"
                report["original_Torch_reference_calls_attempted"] += 1
                expected, check = reference_round(golden_update, arrays, graph)
                report["original_Torch_reference_calls_completed"] += 1
                check["artifact"] = save("reference-round-%02d.npz" % iteration,
                                         dict(zip(OUTPUT_NAMES, expected)))
                require(check["begin_monotonic"] >= row["end_monotonic"], "Reference overlapped the feedback timer")
                row["original_Torch_reference"] = check
                require(check["strict_FP32_gate_passed"], "Current-state ORT/original Update FP32 gate failed; no continuation")
                report["feedback_rounds_completed"] = iteration+1
                print(json.dumps(dict(event="feedback_round_complete", iteration=iteration, seconds=row["seconds"],
                                      reference_seconds=check["reference_forward_seconds"], strict_FP32_gate_passed=True)), flush=True)
        report.pop("active_round", None)
        report.pop("active_stage", None)
        require(updater.attempted_calls == len(updater.calls) == 4 and
                report["original_Torch_reference_calls_attempted"] == report["original_Torch_reference_calls_completed"] == 4 and
                cached.metrics()["completed_calls"] == 4 and cached.metrics()["failed_calls"] == 0,
                "Actual ORT/reference/native correlation call accounting differs")
        report.update(status=SUCCESS, candidate_accepted=True, current_state_strict_FP32_gate_passed=True,
                      completed_BA_iterations=sum(len(row["ba"]["iterations"]) for row in report["calls"]),
                      final_state=save("final-state.npz", dict(poses=poses, patches=patches, intrinsics=intrinsics, net=net)),
                      round_time_sum_seconds=sum(row["round_total_seconds"] for row in report["calls"]),
                      stage_time_sum_seconds={key: sum(row["seconds"][key] for row in report["calls"])
                                              for key in report["calls"][0]["seconds"]},
                      original_Torch_reference_time_sum_seconds=sum(row["original_Torch_reference"]["reference_forward_seconds"]
                                                                    for row in report["calls"]),
                      timing_scope="Feedback: unchanged v28 snapshot/reprojection/native correlation, current-feed checks/copies + CPU ORT + owned outputs/domain checks, target, vectorized CPU BA(two steps), post-BA reprojection and diagnostics. Every round then separately executes one original Torch reference including wrapping/copies. References, topology/session/model loading, cache preparation, hashing/artifact I/O are excluded from feedback time; actual driver elapsed time includes them. No online latency is inferred.",
                      limitations=["Fixed three captured views; no live observations, official eight-frame initializer, keyframe management, normalization, loop closure or online lifecycle.",
                                   "Identity/unit inverse-depth initialization is the frozen v28 research condition, not the official motion/depth initialization.",
                                   "Every original reference consumes the ORT recurrent net and current correlation; it is not a separate golden recurrent trajectory.",
                                   "Final Sigmoid has an algebraically stable equivalent expression; no weights, tolerances or weight clipping changed.",
                                   "Geometry/native correlation are CPU implementations; CUDA arithmetic/trajectory accuracy and full VO are unvalidated.",
                                   "Changing learned targets/residual masks are not GT accuracy; timing excludes indispensable research reference calls and is not an online or 20Hz claim."])
    except Exception as exc:
        report.update(status="failed", candidate_accepted=False, error=repr(exc), traceback=traceback.format_exc())
    finally:
        report["actual_stage_attempts"] = {name: sum(e["stage"] == name+"_begin" for e in report["stage_events"])
                                           for name in ("correlation", "update", "BA")}
        report["completed_BA_calls"] = sum(e["stage"] == "BA_completed" for e in report["stage_events"])
        if updater is not None:
            report.update(ORT_Update_calls_attempted=updater.attempted_calls,
                          ORT_Update_calls_completed=len(updater.calls), ORT_Update_calls=updater.calls)
            updater.close()
        session = None
        gc.collect()
        report["ORT_session_released"] = True
        if cached is not None:
            report["cached_metrics_before_close"] = cached.metrics()
            try:
                begin = time.monotonic()
                cached.close()
                report["cache_close_seconds"] = time.monotonic()-begin
                report["cached_metrics_after_close"] = cached.metrics()
                require(report["cached_metrics_after_close"]["closed"] and
                        report["cached_metrics_after_close"]["current_context_bytes"] == 0,
                        "Native correlation cache did not close/release")
            except Exception as exc:
                report["cleanup_errors"].append(repr(exc))
        try:
            if neural is not None and state is not None:
                after = {key: meta(value.detach().numpy()) for key, value in neural.state_dict().items()}
                report.update(original_weights_after=after, original_weights_unchanged=after == state)
                require(after == state, "Original weights changed")
            report["immutable_features_graph_topology_unchanged"] = all(meta(value) == before for value, before in immutable.values())
            require(report["immutable_features_graph_topology_unchanged"], "Captured immutable input changed")
            after_pins = {path: sha(path) for path in pins}
            report.update(source_hashes_after=after_pins, sources_and_inputs_unchanged=after_pins == pins)
            require(after_pins == pins, "Pinned source/input changed")
        except Exception as exc:
            report["cleanup_errors"].append(repr(exc))
        topology_adapter = neural = None
        gc.collect()
        report["neural_reference_released"] = True
        try:
            os.sched_setaffinity(0, prior_affinity)
            report["affinity_restored"] = os.sched_getaffinity(0) == prior_affinity
            require(report["affinity_restored"], "CPU affinity restoration failed")
        except Exception as exc:
            report["cleanup_errors"].append(repr(exc))
        if report["cleanup_errors"]:
            report.update(status="failed", candidate_accepted=False)
        report.update(source_pins=pins, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                      actual_driver_elapsed_seconds=time.monotonic()-drive_begin,
                      actual_elapsed_includes_original_Torch_reference=True)
        (output / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: report.get(key) for key in
                      ("status", "error", "feedback_rounds_completed", "candidate_accepted", "cleanup_errors")}), flush=True)
    return 0 if report["status"] == SUCCESS else 1


if __name__ == "__main__":
    raise SystemExit(main())
