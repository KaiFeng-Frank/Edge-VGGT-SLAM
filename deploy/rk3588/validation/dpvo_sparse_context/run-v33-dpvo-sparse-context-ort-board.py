#!/usr/bin/env python3
"""Board-only CPU ORT check of the isolated DPVO sparse-context suffix.

Same frozen normalized inputs, weights and pixels. The spatial parent's failed
FP32 gate stays rejected even if sampling itself is exact. No neural checkpoint
loading, image decoding, RNG replacement, NPU execution, or VO is performed here.
DPVO reference is MIT, Copyright (c) 2022 Princeton Vision & Learning Lab.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import resource
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np

BASE = Path("/mnt/vggt-data/vggt-slam2")
CAPTURE_SHA = "e0563efb962edb90684480be114b88bc2c7825015f3b13a163f6f47f06360088"
EXPORT_SHA = "45fa589fa3a25a9722dc0832f5d14ffcec662e70d1dff135b86be641eeeba4b2"
OLD_ORT_SHA = "153da765b1145d7bb74bc635c1982ae74c652dac28cf99be56cb40534c44b667"
SPATIAL_ORT_SHA = "a517f822a2e4afd01aff87df303183a3dc5850313134548891dc13c1e36b896b"
EDITOR_SHA = "39004857559e27e3f9271ab413fc4498365370eac8862c56b16ba44b93f109ec"
ORIGINAL_SHA = "527fc2493de700a21aee958d7fda4d5ee421f4eb0c92296e81409df289f56a82"
SPATIAL_SHA = "c3c4048c6d8792b09f0d6cbec8228f99d59c60d7503456eea2e369d7b8dbd826"
IMAGE = {"name": "normalized_bgr", "shape": [1, 3, 464, 608], "dtype": "float32"}
INDEX = {"name": "patch_indices", "shape": [96], "dtype": "int64"}
DENSE_OUTPUTS = [{"name": "fmap", "shape": [1, 128, 116, 152], "dtype": "float32"},
                 {"name": "imap", "shape": [1, 384, 116, 152], "dtype": "float32"}]
SPARSE_OUTPUTS = [DENSE_OUTPUTS[0], {"name": "context", "shape": [1, 96, 384], "dtype": "float32"}]
RTOL, ATOL = 1e-4, 1e-5


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metadata(array):
    a = np.asarray(array)
    return {"shape": list(a.shape), "dtype": str(a.dtype), "bytes": a.nbytes,
            "tensor_sha256": hashlib.sha256(a.tobytes(order="C")).hexdigest()}


def compare(actual, reference):
    require(actual.dtype == reference.dtype == np.float32 and actual.shape == reference.shape,
            "Matching FP32 output shapes/dtypes required")
    require(np.isfinite(actual).all() and np.isfinite(reference).all(), "Nonfinite output/reference")
    x, y = actual.astype(np.float64), reference.astype(np.float64)
    difference = x - y
    failures = int(np.count_nonzero(np.abs(difference) > ATOL + RTOL * np.abs(y)))
    norm = float(np.linalg.norm(y.ravel()))
    error_norm = float(np.linalg.norm(difference.ravel()))
    return {"rtol": RTOL, "atol": ATOL, "failed_elements": failures, "element_count": actual.size,
            "gate_passed": failures == 0, "array_equal": bool(np.array_equal(actual, reference)),
            "bitwise_equal": actual.tobytes(order="C") == reference.tobytes(order="C"),
            "max_abs": float(np.abs(difference).max()),
            "relative_l2": error_norm / norm if norm else (0.0 if not error_norm else None)}


class Pins:
    def __init__(self):
        self.before = {}

    def pin(self, path, expected):
        path = Path(path).resolve()
        require(isinstance(expected, str) and len(expected) == 64 and
                all(c in "0123456789abcdef" for c in expected), "Invalid SHA256")
        require(sha256_file(path) == expected, "SHA mismatch: " + str(path))
        require(str(path) not in self.before or self.before[str(path)] == expected, "Conflicting source SHA")
        self.before[str(path)] = expected
        return path

    def receipt(self, directory, name, expected):
        return json.loads(self.pin(Path(directory) / name, expected).read_text())

    def artifact(self, directory, item):
        name = item["file"]
        require(isinstance(name, str) and Path(name).name == name, "Simple local artifact name required")
        path = self.pin(Path(directory) / name, item["sha256"])
        require(path.stat().st_size == item["bytes"], "Artifact byte count mismatch")
        return path

    def finish(self):
        after = {p: sha256_file(p) for p in self.before}
        require(after == self.before, "Pinned sources or inputs changed")
        return after


def deterministic_centers(frame_index):
    """Predeclared diagnostic indices, not original random observations.

    Formula is independent of image/feature values; corners/repeats plus reverse
    and rotated order exercise selection. No pass/fail-dependent point choice.
    """
    require(type(frame_index) is int and 0 <= frame_index < 3, "Exactly three fixed diagnostic frames")
    i = np.arange(96, dtype=np.int64)
    centers = np.stack((1 + (37 * i + 17 * frame_index) % 150,
                        1 + (19 * i + 11 * frame_index) % 114), axis=-1)
    centers[:6] = [[1, 1], [150, 114], [1, 114], [150, 1], [1, 1], [150, 114]]
    if frame_index == 1:
        centers = centers[::-1]
    if frame_index == 2:
        centers = np.roll(centers, 13, axis=0)
    return centers.copy(order="C")


def gather_context(imap, indices):
    require(imap.dtype == np.float32 and imap.ndim == 4 and imap.shape[0] == 1,
            "Singleton NCHW FP32 imap required")
    require(indices.dtype == np.int64 and indices.ndim == 1 and
            np.all(indices >= 0) and np.all(indices < imap.shape[2] * imap.shape[3]),
            "Valid original flat integer indices required")
    sampled = np.take(imap.reshape(1, imap.shape[1], -1), indices, axis=2)
    return np.ascontiguousarray(sampled.transpose(0, 2, 1))


def check_session(session, inputs, outputs):
    require(session.get_providers() == ["CPUExecutionProvider"], "CPU-only ORT provider required")
    types = {"float32": "tensor(float)", "int64": "tensor(int64)"}
    for actual, expected in ((session.get_inputs(), inputs), (session.get_outputs(), outputs)):
        require(len(actual) == len(expected), "ORT input/output count mismatch")
        for a, e in zip(actual, expected):
            require(a.name == e["name"] and list(a.shape) == e["shape"] and a.type == types[e["dtype"]],
                    "ORT name/order/shape/type mismatch: " + e["name"])


def run_checked(session, feeds, inputs, outputs):
    require(list(feeds) == [r["name"] for r in inputs], "Exact ordered feed names required")
    for row in inputs:
        value = feeds[row["name"]]
        require(isinstance(value, np.ndarray) and value.dtype == np.dtype(row["dtype"]) and
                list(value.shape) == row["shape"] and value.flags.c_contiguous and np.isfinite(value).all(),
                "ORT input dtype/shape/layout/finite mismatch")
    begin = time.monotonic()
    values = session.run([r["name"] for r in outputs], feeds)
    seconds = time.monotonic() - begin
    require(len(values) == len(outputs), "ORT output count mismatch")
    result = {}
    for row, value in zip(outputs, values):
        require(isinstance(value, np.ndarray) and value.dtype == np.float32 and
                list(value.shape) == row["shape"] and np.isfinite(value).all(), "ORT nonfinite or malformed output")
        # Own snapshots after timing; no ORT buffer can mutate a saved comparison.
        result[row["name"]] = value.copy(order="C")
    return result, seconds


@contextmanager
def session_pair(ort, parent_graph, sparse_graph, lifecycle, *, image=IMAGE,
                 dense_outputs=DENSE_OUTPUTS, index=INDEX, sparse_outputs=SPARSE_OUTPUTS):
    sessions = {}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    try:
        for role, graph, inputs, outputs in [("parent", parent_graph, [image], dense_outputs),
                                            ("sparse", sparse_graph, [image, index], sparse_outputs)]:
            started = time.monotonic()
            sessions[role] = ort.InferenceSession(str(graph), sess_options=options, providers=["CPUExecutionProvider"])
            lifecycle.setdefault("session_initializations", []).append({"role": role,
                "seconds": time.monotonic() - started})
            check_session(sessions[role], inputs, outputs)
        yield sessions
    finally:
        # Python ORT exposes no public explicit destroy call. Dropping both session
        # references and collecting is the release boundary, not a native-SDK claim.
        sessions.clear()
        gc.collect()
        lifecycle["ORT_session_references_dropped"] = True
        lifecycle["gc_collect_performed"] = True
        lifecycle["native_explicit_destroy_API_available"] = False


def load_editor(path, expected, pins):
    require(expected == EDITOR_SHA, "Frozen sparse editor SHA required")
    path = pins.pin(path, expected)
    spec = importlib.util.spec_from_file_location("v33_frozen_sparse_editor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _historical_state(receipt, expected_status):
    require(receipt["status"] == expected_status and receipt["sources_and_inputs_unchanged"] is True and
            receipt["source_pins"] == receipt["source_hashes_after"] and receipt["cleanup_errors"] == [],
            "Historical source/cleanup/status mismatch")


def load_inputs(args, pins):
    require(args.capture_sha256 == CAPTURE_SHA and args.export_sha256 == EXPORT_SHA and
            args.old_ort_sha256 == OLD_ORT_SHA and args.spatial_ort_sha256 == SPATIAL_ORT_SHA,
            "Frozen original capture/export/ORT and spatial rejection receipts required")
    capture = pins.receipt(args.capture_dir, "capture.json", args.capture_sha256)
    exported = pins.receipt(args.export_dir, "export.json", args.export_sha256)
    old = pins.receipt(args.old_ort_dir, "ort.json", args.old_ort_sha256)
    spatial = pins.receipt(args.spatial_rewrite_dir, "rewrite.json", args.spatial_rewrite_sha256)
    history = pins.receipt(args.spatial_ort_dir, "ort.json", args.spatial_ort_sha256)
    for receipt, status in [(capture, "completed_encoder_cpu_probe_not_VO"),
                            (exported, "exported_not_runtime_validated"), (old, "completed_CPU_ORT_gate"),
                            (history, "failed")]:
        _historical_state(receipt, status)
    require(old["exact_reference_gate_passed"] is True and history["exact_FP32_gate_passed"] is False,
            "Historical original acceptance/spatial rejection must remain explicit")
    require(capture["checkpoint_sha256"] == exported["checkpoint_sha256"] == old["checkpoint_sha256"] ==
            "30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12", "Checkpoint identity differs")
    require(exported["capture_sha256"] == old["capture_sha256"] == history["capture_sha256"] == CAPTURE_SHA and
            old["export_sha256"] == history["export_sha256"] == EXPORT_SHA and
            history["old_ort_sha256"] == OLD_ORT_SHA and history["rewrite_sha256"] == args.spatial_rewrite_sha256,
            "Parent source/reference chain differs")
    require(exported["onnx"]["sha256"] == old["onnx"]["sha256"] == spatial["input"]["sha256"] == ORIGINAL_SHA and
            spatial["output"]["sha256"] == history["onnx"]["sha256"] == SPATIAL_SHA,
            "Parent encoder graph chain differs")
    require(spatial["status"] == "rewritten_and_statically_checked_not_runtime_validated" and
            all(spatial["preservation"][k] is True for k in ["original_initializers_byte_exact",
                "non_instance_norm_nodes_byte_exact", "input_output_protobuf_byte_exact", "model_header_byte_exact"]),
            "Spatial static preservation evidence missing")
    editor = load_editor(args.editor, args.editor_sha256, pins)
    for relative, digest in editor.SOURCE_PINS.items():
        pins.pin(args.official_source_dir / relative, digest)
    # Rehash original sources, checkpoint and source RGB files without decoding.
    for path, digest in capture["source_pins"].items():
        pins.pin(path, digest)
    originals = {"original": pins.artifact(args.export_dir, exported["onnx"]),
                 "spatial": pins.artifact(args.spatial_rewrite_dir, spatial["output"])}
    candidates, recipes, proofs = {}, {}, {}
    import onnx
    for name, directory, expected in [("original", args.original_sparse_dir, args.original_sparse_sha256),
                                       ("spatial", args.spatial_sparse_dir, args.spatial_sparse_sha256)]:
        recipe = pins.receipt(directory, "rewrite.json", expected)
        parent_sha = ORIGINAL_SHA if name == "original" else SPATIAL_SHA
        require(recipe["schema"] == "v33_DPVO_encoder_sparse_context_rewrite" and
                recipe["status"] == "statically_checked_not_runtime_validated" and
                recipe["candidate_accepted"] is False and recipe["parent"]["sha256"] == parent_sha and
                recipe["script_sha256"] == EDITOR_SHA and recipe["inputs"] == [IMAGE, INDEX] and
                recipe["outputs"] == SPARSE_OUTPUTS and recipe["source_pins"] == recipe["source_hashes_after"] == editor.SOURCE_PINS,
                "Sparse recipe/source/contract mismatch")
        graph = pins.artifact(directory, recipe["output"])
        parent = onnx.load(originals[name], load_external_data=False)
        candidate = onnx.load(graph, load_external_data=False)
        editor._reject_complex_storage(candidate)
        onnx.checker.check_model(candidate, full_check=True)
        proof = editor.preservation(parent, candidate)
        require(proof == recipe["preservation"] and editor._contract(candidate.graph.input) == [IMAGE, INDEX] and
                editor._contract(candidate.graph.output) == SPARSE_OUTPUTS, "Actual sparse graph preservation mismatch")
        recipes[name], candidates[name], proofs[name] = recipe, graph, proof
    require(len(capture["inputs"]) == len(old["pairs"]) == len(history["calls"]) == 3, "Exactly original three frames required")
    rows = []
    for i, item in enumerate(capture["inputs"]):
        ref = next(row for row in capture["calls"] if row["frame_index"] == i)
        older, spatial_ref = old["pairs"][i], history["calls"][i]
        require(item["frame_index"] == older["frame_index"] == spatial_ref["frame_index"] == i and
                item["frame_id"] == ref["frame_id"] == spatial_ref["frame_id"] == item["source"]["frame_id"],
                "Original source frame identity/order differs")
        require(item["array"]["shape"] == [1, 1, 3, 464, 608] and item["array"]["dtype"] == "float32", "Normalized source shape differs")
        rows.append({"index": i, "frame_id": item["frame_id"], "source_image": item["source"],
                     "input_path": pins.artifact(args.capture_dir, item["input"]), "input_metadata": item["array"],
                     "CPU_reference": {"path": pins.artifact(args.capture_dir, ref["artifact"]), "arrays": ref["arrays"]},
                     "original_ORT_reference": {"path": pins.artifact(args.old_ort_dir, older["artifact"]), "arrays": older["arrays"]},
                     "spatial_ORT_reference": {"path": pins.artifact(args.spatial_ort_dir, spatial_ref["artifact"]), "arrays": spatial_ref["arrays"]}})
    require(capture["preprocessing"] == old["preprocessing"] == exported["preprocessing"] and
            capture["preprocessing"]["normalization"] == "2 * (image / 255.0) - 0.5, FP32", "Once-normalized input provenance differs")
    return {"editor": editor, "parents": originals, "candidates": candidates, "recipes": recipes,
            "proofs": proofs, "rows": rows, "preprocessing": capture["preprocessing"],
            "historical_spatial_rejection": {"receipt_sha256": args.spatial_ort_sha256,
                "status": history["status"], "exact_FP32_gate_passed": False,
                "error": history.get("error"), "retained": True}}


def load_reference(row, label):
    item = row[label]
    with np.load(item["path"], allow_pickle=False) as saved:
        result = {key: saved[key].copy(order="C") for key in ("fmap", "imap")}
    for key, value in result.items():
        require(metadata(value) == item["arrays"][key] and value.dtype == np.float32 and
                list(value.shape) == [1, 1, *DENSE_OUTPUTS[0 if key == "fmap" else 1]["shape"][1:]] and
                np.isfinite(value).all(), "Reference raw SHA/shape/finite mismatch")
    return {key: value[:, 0] for key, value in result.items()}


def import_ort():
    import onnxruntime
    return onnxruntime


def board_guard(output):
    require(platform.machine() == "aarch64", "RK3588 aarch64 board only")
    require(b"rockchip,rk3588" in Path("/proc/device-tree/compatible").read_bytes(), "RK3588 board only")
    require(output.resolve().is_relative_to(BASE / "experiments"), "New isolated board experiment output required")
    require(not output.exists(), "Output directory must be new")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "export", "old-ort", "spatial-rewrite", "spatial-ort", "original-sparse", "spatial-sparse"):
        parser.add_argument("--" + name + "-dir", type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--editor", type=Path, required=True)
    parser.add_argument("--editor-sha256", required=True)
    parser.add_argument("--official-source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    board_guard(args.output_dir)  # Before imports, model/session construction or output writes.
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    pins, prior = Pins(), os.sched_getaffinity(0)
    report = {"schema": "v33_DPVO_sparse_context_board_CPU_ORT", "status": "failed",
              "script_sha256": sha256_file(__file__), "calls": [], "cleanup_errors": [],
              "actual_board_execution": True, "NPU_used": False, "complete_VO": False,
              "original_VGGT_used": False, "realtime_validated": False, "production_changed": False,
              "candidate_accepted_as_original_model": False, "model_checkpoint_loaded": False,
              "images_decoded": False, "normalization_applications_in_this_driver": 0,
              "sampling_rule": {"method": "predeclared_integer_formula_with_corner_repeat_and_order_stress",
                  "M": 96, "original_RNG_observations": False, "used_source_RGB_for_selection": False,
                  "formula": "x=1+(37*i+17*frame)%150,y=1+(19*i+11*frame)%114; first6 corner/repeats; frame1 reverse; frame2 roll13"},
              "gate": {"suffix_target": "parent fmap/context gather bitwise equality", "rtol": RTOL, "atol": ATOL,
                  "strict_original_CPU_reported_separately": True, "spatial_historical_rejection_retained": True},
              "session_lifecycle": {}, "timing_scope": "each ORT session.run only, including its returned arrays; metadata, copies, reference loads and saving excluded; not end-to-end or SDK latency"}
    try:
        pins.pin(__file__, report["script_sha256"])
        inputs = load_inputs(args, pins)
        os.sched_setaffinity(0, {4, 5, 6, 7})
        ort = import_ort()
        report.update(ORT_version=ort.__version__, numpy_version=np.__version__, providers=["CPUExecutionProvider"],
                      graph_optimization="ORT_DISABLE_ALL", cpu_affinity=sorted(os.sched_getaffinity(0)),
                      intra_threads=4, inter_threads=1, graph_preservation=inputs["proofs"],
                      historical_spatial_rejection=inputs["historical_spatial_rejection"], preprocessing=inputs["preprocessing"],
                      receipt_sha256={name: getattr(args, name + "_sha256") for name in
                          ("capture", "export", "old_ort", "spatial_rewrite", "spatial_ort", "original_sparse", "spatial_sparse")})
        for method in ("original", "spatial"):
            lifecycle = report["session_lifecycle"].setdefault(method, {})
            with session_pair(ort, inputs["parents"][method], inputs["candidates"][method], lifecycle) as sessions:
                for row in inputs["rows"]:
                    i = row["index"]
                    x5 = np.load(row["input_path"], allow_pickle=False)
                    require(metadata(x5) == row["input_metadata"] and np.isfinite(x5).all() and
                            float(x5.min()) >= -0.5 and float(x5.max()) <= 1.5, "Frozen normalized input mismatch")
                    x = x5[:, 0]
                    centers = deterministic_centers(i)
                    index = inputs["editor"].centers_to_indices(centers)
                    index = inputs["editor"].validate_patch_indices(index)
                    CPU = load_reference(row, "CPU_reference")
                    stored = load_reference(row, "original_ORT_reference" if method == "original" else "spatial_ORT_reference")
                    parent, parent_seconds = run_checked(sessions["parent"], {"normalized_bgr": x}, [IMAGE], DENSE_OUTPUTS)
                    sparse, sparse_seconds = run_checked(sessions["sparse"], {"normalized_bgr": x, "patch_indices": index},
                                                        [IMAGE, INDEX], SPARSE_OUTPUTS)
                    expected = {"fmap": parent["fmap"], "context": gather_context(parent["imap"], index)}
                    CPU_selected = {"fmap": CPU["fmap"], "context": gather_context(CPU["imap"], index)}
                    stored_selected = {"fmap": stored["fmap"], "context": gather_context(stored["imap"], index)}
                    versus_parent = {key: compare(sparse[key], expected[key]) for key in sparse}
                    versus_CPU = {key: compare(sparse[key], CPU_selected[key]) for key in sparse}
                    versus_stored = {key: compare(sparse[key], stored_selected[key]) for key in sparse}
                    parent_repeat = {key: compare(parent[key], stored[key]) for key in parent}
                    artifact = out / (method + f"-frame-{i}.npz")
                    payload = {**sparse, "parent_fmap": parent["fmap"], "parent_context_gather": expected["context"],
                               "centers_xy": centers, "patch_indices": index}
                    np.savez(artifact, **payload)
                    if method == "original":
                        input_path = out / f"input-frame-{i}.npz"
                        np.savez(input_path, normalized_bgr=x, centers_xy=centers, patch_indices=index)
                    input_path = out / f"input-frame-{i}.npz"
                    report["calls"].append({"method": method, "frame_index": i, "frame_id": row["frame_id"],
                        "source_image": row["source_image"], "input_metadata": metadata(x), "centers": metadata(centers),
                        "indices": metadata(index), "parent_forward_seconds": parent_seconds,
                        "sparse_forward_seconds": sparse_seconds, "versus_current_parent_ORT": versus_parent,
                        "versus_original_CPU": versus_CPU, "versus_saved_corresponding_parent_ORT": versus_stored,
                        "parent_repeated_vs_saved": parent_repeat,
                        "parent_dense_outputs_metadata": {key: metadata(value) for key, value in parent.items()},
                        "arrays": {key: metadata(value) for key, value in payload.items()},
                        "artifact": {"file": artifact.name, "bytes": artifact.stat().st_size, "sha256": sha256_file(artifact)},
                        "input_artifact": {"file": input_path.name, "bytes": input_path.stat().st_size, "sha256": sha256_file(input_path)}})
                    print(json.dumps({"method": method, "frame": i, "parent_seconds": parent_seconds,
                        "sparse_seconds": sparse_seconds, "suffix_bitwise": all(v["bitwise_equal"] for v in versus_parent.values())}), flush=True)
                    # Free dense references before the next real image/model.
                    del CPU, stored, parent, sparse, expected, CPU_selected, stored_selected, payload
        report["suffix_strict_gate_passed"] = len(report["calls"]) == 6 and all(
            v["gate_passed"] for row in report["calls"] for v in row["versus_current_parent_ORT"].values())
        report["suffix_bitwise_gate_passed"] = len(report["calls"]) == 6 and all(
            v["bitwise_equal"] for row in report["calls"] for v in row["versus_current_parent_ORT"].values())
        report["original_CPU_strict_gate_each_method"] = {method: all(v["gate_passed"]
            for row in report["calls"] if row["method"] == method for v in row["versus_original_CPU"].values())
            for method in ("original", "spatial")}
        require(report["suffix_strict_gate_passed"] and report["suffix_bitwise_gate_passed"],
                "Predeclared isolated suffix bitwise/strict gate failed")
        report["status"] = "completed_suffix_bitwise_gate_parent_rejection_retained_not_VO"
    except Exception as exc:
        report.update(error=repr(exc), traceback=traceback.format_exc())
    finally:
        try:
            report["source_hashes_after"] = pins.finish()
            report["sources_and_inputs_unchanged"] = True
        except Exception as exc:
            report["cleanup_errors"].append(repr(exc))
            report["sources_and_inputs_unchanged"] = False
        try:
            os.sched_setaffinity(0, prior)
            report["affinity_restored"] = os.sched_getaffinity(0) == prior
            require(report["affinity_restored"], "Affinity restoration failed")
        except Exception as exc:
            report["cleanup_errors"].append(repr(exc))
        report.update(source_pins=pins.before, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        if report["cleanup_errors"]:
            report["status"] = "failed"
        (out / "ort.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report.get(key) for key in ("status", "error", "suffix_bitwise_gate_passed")}), flush=True)
    return 0 if report["status"] == "completed_suffix_bitwise_gate_parent_rejection_retained_not_VO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
