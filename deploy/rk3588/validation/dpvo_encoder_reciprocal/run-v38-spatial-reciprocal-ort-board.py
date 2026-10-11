#!/usr/bin/env python3
"""Board-only saved-input CPU ORT diagnostic of small reciprocal/Mul.

MIT project helper; DPVO reference is MIT, Copyright (c) 2022 Princeton
Vision & Learning Lab. No decoder, GT, RNG, checkpoint loading or NPU.
The spatial parent's original-FP32 rejection remains explicit. These six
session.run calls are a numerical diagnostic, not a performance benchmark/VO.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import time
import traceback

BASE = Path("/mnt/vggt-data/vggt-slam2/experiments")
CAPTURE_SHA = "e0563efb962edb90684480be114b88bc2c7825015f3b13a163f6f47f06360088"
REFERENCE_SHA = "24aef39822c86ae60e1cb6306d2d2966fa8a440320c4363f18c5db938ac4e168"
PARENT_SHA = "6c3b4de4d151813dc172d711b6a760ad8bdc17120f2f3faa76dbf32bd0c451ab"
PARENT_RECIPE_SHA = "005667f44cf98447d2b622882d7d40cbcce8a545985ddf2104eaa62740273ce2"
ORT_HELPER_SHA = "b184dcc440771730cce13f9e7a72c065b619f469cab5b754a43a89c6bb0314cd"
CHECKPOINT_SHA = "30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12"
IMAGE = {"name": "normalized_bgr", "shape": [1, 3, 464, 608], "dtype": "float32"}
INDEX = {"name": "patch_indices", "shape": [96], "dtype": "int64"}
OUTPUTS = [{"name": "fmap", "shape": [1, 128, 116, 152], "dtype": "float32"},
           {"name": "context", "shape": [1, 96, 384], "dtype": "float32"}]
SUCCESS = "completed_CPU_ORT_reciprocal_diagnostic_parent_rejection_retained_not_VO"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def board_guard(machine=None, system=None, compatible=None):
    require((platform.machine() if machine is None else machine) == "aarch64", "aarch64 board only; host neural execution prohibited")
    require((platform.system() if system is None else system) == "Linux", "Linux board required")
    if compatible is None:
        p = Path("/proc/device-tree/compatible")
        compatible = p.read_bytes() if p.is_file() else b""
    require(any(t in (b"rockchip,rk3588", b"rockchip,rk3588s") for t in compatible.split(b"\0")), "RK3588 device-tree identity required")


def load_pinned_module(path, expected, name, pins=None):
    require(isinstance(expected, str) and len(expected) == 64 and all(c in "0123456789abcdef" for c in expected), "Invalid expected source SHA")
    path = Path(path).resolve()
    require(sha(path) == expected, "Source SHA mismatch: " + path.name)
    if pins is not None:
        pins.pin(path, expected)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unchanged(value, flag="sources_and_inputs_unchanged"):
    require(value[flag] is True and value["source_pins"] == value["source_hashes_after"] and value.get("cleanup_errors", []) == [], "Historical source/cleanup mismatch")


def recipe_guard(recipe, expected_rewriter):
    require(recipe["schema"] == "v38_DPVO_spatial_encoder_small_reciprocal_rewrite" and
            recipe["status"] == "statically_checked_parent_FP32_rejection_retained_not_runtime_validated" and
            recipe["script_sha256"] == expected_rewriter, "Reciprocal recipe/source mismatch")
    require(recipe["parent"]["sha256"] == PARENT_SHA and recipe["parent_recipe_sha256"] == PARENT_RECIPE_SHA and
            recipe["parent_reference_sha256"] == REFERENCE_SHA, "Frozen spatial parent/recipe/reference required")
    require(recipe["historical_parent_FP32_gate_passed"] is False and recipe["historical_parent_rejection_retained"] is True and
            recipe["candidate_accepted"] is False, "Historical parent rejection may not be promoted")
    unchanged(recipe, "sources_and_parent_unchanged")
    expected = ["all_original_initializers_ordered_protobuf_byte_exact", "all_nonselected_nodes_ordered_protobuf_byte_exact",
        "original_IO_value_info_graph_metadata_protobuf_byte_exact", "model_header_protobuf_byte_exact",
        "exact_10_selected_Div_replacement", "original_2_nonselected_Div_protobuf_byte_exact", "one_added_nonlearned_FP32_scalar_one"]
    require(all(recipe["preservation"][k] is True for k in expected) and
            recipe["preservation"]["original_initializer_count"] == 65 and recipe["preservation"]["added_initializer_count"] == 1,
            "Only ten selected Div substitutions and nonlearned scalar allowed")


def load_inputs(args, base, pins):
    capture = pins.receipt(args.capture_dir, "capture.json", CAPTURE_SHA)
    reference = pins.receipt(args.reference_dir, "ort.json", REFERENCE_SHA)
    parent_recipe = json.loads(pins.pin(args.parent_recipe, PARENT_RECIPE_SHA).read_text())
    recipe = pins.receipt(args.rewrite_dir, "rewrite.json", args.rewrite_sha256)
    unchanged(capture)
    unchanged(reference)
    require(capture["status"] == "completed_encoder_cpu_probe_not_VO" and capture["state_unchanged"] is True and
            capture["checkpoint_sha256"] == CHECKPOINT_SHA, "Frozen original CPU encoder/weights required")
    require(reference["status"] == "completed_suffix_bitwise_gate_parent_rejection_retained_not_VO" and
            reference["suffix_bitwise_gate_passed"] is True and reference["original_CPU_strict_gate_each_method"]["spatial"] is False,
            "Spatial suffix evidence and rejected parent required")
    require(parent_recipe["schema"] == "v33_DPVO_encoder_sparse_context_rewrite" and parent_recipe["output"]["sha256"] == PARENT_SHA,
            "Frozen spatial-sparse graph recipe required")
    recipe_guard(recipe, args.rewriter_sha256)
    editor = load_pinned_module(args.rewriter, args.rewriter_sha256, "_v38_reciprocal_editor", pins)
    parent_path = pins.pin(args.parent_onnx, PARENT_SHA)
    require(parent_path.stat().st_size == recipe["parent"]["bytes"], "Parent byte count mismatch")
    graph = pins.artifact(args.rewrite_dir, recipe["output"])
    import onnx  # static validation only; main's board guard already passed
    parent = onnx.load(parent_path, load_external_data=False)
    candidate = onnx.load(graph, load_external_data=False)
    onnx.checker.check_model(candidate, full_check=True)
    proof = editor.validate_preservation(parent, candidate)
    require(proof == recipe["preservation"], "Actual initializer/node/IO/header preservation differs")
    require(len(recipe["original_initializer_manifest"]) == 65 and len(recipe["selected_replacements"]) == 10,
            "Full original initializer and selected-node manifests required")
    for path, digest in capture["source_pins"].items():
        pins.pin(path, digest)  # includes original RGB SHA, without decoding
    rows = [r for r in reference["calls"] if r["method"] == "spatial"]
    require([r["frame_index"] for r in rows] == [0, 1, 2] and len(capture["inputs"]) == 3, "Exactly the three frozen inputs, no selection")
    require(reference["preprocessing"] == capture["preprocessing"], "Original preprocessing identity differs")
    result = []
    for i, row in enumerate(rows):
        original = capture["inputs"][i]
        cpu_row = next(r for r in capture["calls"] if r["frame_index"] == i)
        require(row["frame_id"] == original["frame_id"] == cpu_row["frame_id"] and row["source_image"] == original["source"], "Exact ordered RGB IDs/source SHA required")
        result.append(dict(identity=row,
            input_path=pins.artifact(args.reference_dir, row["input_artifact"]),
            original_input_path=pins.artifact(args.capture_dir, original["input"]), original_input_metadata=original["array"],
            cpu_path=pins.artifact(args.capture_dir, cpu_row["artifact"]), cpu_arrays=cpu_row["arrays"],
            saved_spatial_path=pins.artifact(args.reference_dir, row["artifact"])))
    return dict(parent=parent_path, candidate=graph, recipe=recipe, proof=proof, rows=result,
                preprocessing=capture["preprocessing"], sampling_rule=reference["sampling_rule"])


def read_arrays(path, declarations, base):
    import numpy as np
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == set(declarations), "Saved NPZ keys mismatch")
        values = {k: saved[k].copy(order="C") for k in declarations}
    require(all(base.metadata(v) == declarations[k] and np.isfinite(v).all() for k, v in values.items()), "Saved array raw SHA/shape/dtype/finite mismatch")
    return values


def load_frame(row, base):
    import numpy as np
    ref = row["identity"]
    inputs = read_arrays(row["input_path"], {"normalized_bgr": ref["input_metadata"], "centers_xy": ref["centers"], "patch_indices": ref["indices"]}, base)
    image, centers, indices = (inputs[k] for k in ("normalized_bgr", "centers_xy", "patch_indices"))
    original = np.load(row["original_input_path"], allow_pickle=False)
    require(base.metadata(original) == row["original_input_metadata"] and original.shape == (1, 1, 3, 464, 608) and original.dtype == np.float32 and
            image.shape == (1, 3, 464, 608) and image.dtype == np.float32 and image.tobytes(order="C") == original.tobytes(order="C") and
            image.min() >= -.5 and image.max() <= 1.5, "Exact once-normalized saved pixels required; no uint8 conversion")
    require(centers.dtype == indices.dtype == np.int64 and centers.shape == (96, 2) and indices.shape == (96,) and
            np.array_equal(indices, centers[:, 1]*152+centers[:, 0]) and np.all((centers[:, 0] >= 1) & (centers[:, 0] < 151)) and
            np.all((centers[:, 1] >= 1) & (centers[:, 1] < 115)), "Original ordered INT64 interior centers required")
    cpu = read_arrays(row["cpu_path"], row["cpu_arrays"], base)
    require(cpu["fmap"].shape == (1, 1, 128, 116, 152) and cpu["imap"].shape == (1, 1, 384, 116, 152) and
            all(v.dtype == np.float32 for v in cpu.values()), "Original CPU full-imap/fmap gold shape differs")
    gold = dict(fmap=cpu["fmap"][:, 0].copy(order="C"), context=base.gather_context(cpu["imap"][:, 0], indices))
    saved = read_arrays(row["saved_spatial_path"], ref["arrays"], base)
    require(saved["patch_indices"].tobytes(order="C") == indices.tobytes(order="C") and saved["centers_xy"].tobytes(order="C") == centers.tobytes(order="C"), "Reference indices/order changed")
    return inputs, gold, {k: saved[k] for k in ("fmap", "context")}


@contextmanager
def paired_sessions(ort, base, parent, candidate, lifecycle, *, inputs=None, outputs=None):
    inputs, outputs = ([IMAGE, INDEX] if inputs is None else inputs), (OUTPUTS if outputs is None else outputs)
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sessions = {}
    try:
        for role, graph in (("parent", parent), ("candidate", candidate)):
            begin = time.monotonic()
            sessions[role] = ort.InferenceSession(str(graph), sess_options=options, providers=["CPUExecutionProvider"])
            lifecycle.setdefault("session_initializations", []).append(dict(role=role, seconds=time.monotonic()-begin))
            base.check_session(sessions[role], inputs, outputs)
        yield sessions
    finally:
        sessions.clear()
        gc.collect()
        lifecycle.update(ORT_session_references_dropped=True, gc_collect_performed=True, native_explicit_destroy_API_available=False)


def run_frame_pair(sessions, base, inputs, gold, saved, output, frame, report, *, contracts=None):
    ports, outputs = ([IMAGE, INDEX], OUTPUTS) if contracts is None else contracts
    feeds = {k["name"]: inputs[k["name"]] for k in ports}
    actual = {}
    for role in ("parent", "candidate"):
        values, seconds = base.run_checked(sessions[role], feeds, ports, outputs)
        artifact = output / (role + f"-frame-{frame}.npz")
        import numpy as np
        np.savez(artifact, **values)
        comparisons = {key: base.compare(value, gold[key]) for key, value in values.items()}
        row = dict(role=role, frame_index=frame, forward_seconds=seconds, versus_original_CPU=comparisons,
            arrays={key: base.metadata(value) for key, value in values.items()},
            artifact=dict(file=artifact.name, bytes=artifact.stat().st_size, sha256=sha(artifact)))
        if role == "parent":
            row["versus_saved_v33_spatial_ORT"] = {k: base.compare(v, saved[k]) for k, v in values.items()}
        else:
            row["versus_current_parent_ORT"] = {k: base.compare(v, actual["parent"][k]) for k, v in values.items()}
        report["calls"].append(row)
        actual[role] = values
        print(json.dumps(dict(role=role, frame=frame, forward_seconds=seconds,
            original_CPU_strict_gate_passed=all(v["gate_passed"] for v in comparisons.values()))), flush=True)
    return actual


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("capture-dir", "reference-dir", "parent-onnx", "parent-recipe", "rewrite-dir", "rewriter", "frozen-ort-helper", "output-dir"):
        parser.add_argument("--"+key, type=Path, required=True)
    for key in ("rewrite", "rewriter"):
        parser.add_argument("--"+key+"-sha256", required=True)
    parser.add_argument("--diagnostic-allow-fp32-rejected", action="store_true")
    args = parser.parse_args(argv)
    board_guard()  # before output, graph/session construction, or NumPy/ONNX/ORT imports
    require(args.diagnostic_allow_fp32_rejected, "Explicit parent-rejected diagnostic required")
    output = args.output_dir.resolve()
    require(output.is_relative_to(BASE) and output != BASE and not output.exists(), "New project board experiment output required")
    protected = [v.resolve() for k, v in vars(args).items() if isinstance(v, Path) and k != "output_dir"]
    require(not any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected), "Output/source overlap")
    base = load_pinned_module(args.frozen_ort_helper, ORT_HELPER_SHA, "_v33_frozen_ort_math")
    output.mkdir(parents=True, exist_ok=False)
    pins, prior = base.Pins(), os.sched_getaffinity(0)
    report = dict(schema="v38_DPVO_spatial_reciprocal_board_CPU_ORT", status="failed", script_sha256=sha(__file__),
        frozen_ORT_helper_sha256=ORT_HELPER_SHA, calls=[], cleanup_errors=[], actual_board_execution=True, NPU_used=False,
        diagnostic_allow_fp32_rejected=True, historical_parent_FP32_gate_passed=False, historical_parent_rejection_retained=True,
        candidate_accepted=False, complete_VO=False, realtime_validated=False, production_changed=False, images_decoded=False,
        GT_read=False, RNG_executed=False, model_checkpoint_loaded=False, normalization_applications_in_this_driver=0,
        gate=dict(rtol=1e-4, atol=1e-5), session_lifecycle={},
        timing_scope="Single ORT session.run per role/frame; owned copies, hashes, saving and CPU gold excluded. Six numerical checks, no performance acceptance.")
    begin = time.monotonic()
    try:
        pins.pin(__file__, report["script_sha256"])
        pins.pin(args.frozen_ort_helper, ORT_HELPER_SHA)
        loaded = load_inputs(args, base, pins)
        os.sched_setaffinity(0, {4, 5, 6, 7})
        import onnxruntime as ort
        report.update(ORT_version=ort.__version__, providers=["CPUExecutionProvider"], graph_optimization="ORT_DISABLE_ALL",
            cpu_affinity=sorted(os.sched_getaffinity(0)), intra_threads=4, inter_threads=1,
            capture_sha256=CAPTURE_SHA, reference_sha256=REFERENCE_SHA, parent_graph_sha256=PARENT_SHA,
            parent_recipe_sha256=PARENT_RECIPE_SHA, rewrite_sha256=args.rewrite_sha256, rewriter_sha256=args.rewriter_sha256,
            candidate_graph=loaded["recipe"]["output"], graph_preservation=loaded["proof"],
            preprocessing=loaded["preprocessing"], sampling_rule=loaded["sampling_rule"])
        with paired_sessions(ort, base, loaded["parent"], loaded["candidate"], report["session_lifecycle"]) as sessions:
            for frame, row in enumerate(loaded["rows"]):
                inputs, gold, saved = load_frame(row, base)
                start = len(report["calls"])
                actual = run_frame_pair(sessions, base, inputs, gold, saved, output, frame, report)
                for call in report["calls"][start:]:
                    ref = row["identity"]
                    call.update(frame_id=ref["frame_id"], source_image=ref["source_image"],
                        input_metadata=ref["input_metadata"], centers=ref["centers"], indices=ref["indices"],
                        input_source_artifact=ref["input_artifact"], CPU_full_imap_gather_recomputed=True)
                del inputs, gold, saved, actual
        parent = [r for r in report["calls"] if r["role"] == "parent"]
        candidate = [r for r in report["calls"] if r["role"] == "candidate"]
        require(len(parent) == len(candidate) == 3, "Exactly six fixed diagnostic calls required")
        report["parent_reproduced_saved_v33_strict_gate_passed"] = all(v["gate_passed"] for r in parent for v in r["versus_saved_v33_spatial_ORT"].values())
        report["candidate_vs_current_parent_strict_gate_passed"] = all(v["gate_passed"] for r in candidate for v in r["versus_current_parent_ORT"].values())
        report["candidate_vs_current_parent_bitwise_equal"] = all(v["bitwise_equal"] for r in candidate for v in r["versus_current_parent_ORT"].values())
        report["original_CPU_strict_gate_each_role"] = {role: all(v["gate_passed"] for r in rows for v in r["versus_original_CPU"].values()) for role, rows in (("parent", parent), ("candidate", candidate))}
        require(report["parent_reproduced_saved_v33_strict_gate_passed"], "Parent ORT reproduction differs from frozen v33 reference")
        report["status"] = SUCCESS
    except BaseException as exc:
        report.update(error=repr(exc), traceback=traceback.format_exc())
    finally:
        try:
            report["source_hashes_after"] = pins.finish()
            report["sources_and_inputs_unchanged"] = True
        except Exception as exc:
            report["cleanup_errors"].append(repr(exc))
            report["source_hashes_after"] = {p: sha(p) if Path(p).is_file() else None for p in pins.before}
            report["sources_and_inputs_unchanged"] = False
        try:
            os.sched_setaffinity(0, prior)
            report["affinity_restored"] = os.sched_getaffinity(0) == prior
            require(report["affinity_restored"], "Affinity restoration failed")
        except Exception as exc:
            report["cleanup_errors"].append(repr(exc))
        report.update(source_pins=pins.before, actual_driver_seconds=time.monotonic()-begin,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        if report["cleanup_errors"]:
            report["status"] = "failed"
        (output/"ort.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report.get(k) for k in ("status", "error", "candidate_vs_current_parent_strict_gate_passed")}), flush=True)
    return 0 if report["status"] == SUCCESS else 1


if __name__ == "__main__":
    raise SystemExit(main())
