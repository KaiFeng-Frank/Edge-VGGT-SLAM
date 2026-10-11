#!/usr/bin/env python3
"""Saved NumPy arrays, JSON and static ONNX audit only; never run a model."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import onnx

PARENT_SHA = "6c3b4de4d151813dc172d711b6a760ad8bdc17120f2f3faa76dbf32bd0c451ab"
PARENT_RECIPE_SHA = "005667f44cf98447d2b622882d7d40cbcce8a545985ddf2104eaa62740273ce2"
REFERENCE_SHA = "24aef39822c86ae60e1cb6306d2d2966fa8a440320c4363f18c5db938ac4e168"
CAPTURE_SHA = "e0563efb962edb90684480be114b88bc2c7825015f3b13a163f6f47f06360088"
EDITOR_SHA = "6ccda3f8c50dff5b36c8d65e9901df50fa28b8fb1624a141baedbeda34829d28"
CPU_DRIVER_SHA = "a7f181015d4479856329460dc4099171d4bb0dce8cedf317c35e719f3f36548b"
PTQ_DRIVER_SHA = "72cad7e4a6a31d4860875c7d40cf2d672c1190ee3fedceb3271556fc52255f41"
PARENT_PTQ_SHA = "4d660a6d48d296fda930adee7e192571263df717431a208daa63ae43e5f9427d"


class Checks:
    def __init__(self):
        self.count = 0

    def check(self, value, message):
        self.count += 1
        if not value:
            raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes,
                tensor_sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest())


def saved(directory, declaration, arrays, c):
    c.check(Path(declaration["file"]).name == declaration["file"], "Artifact path changed")
    path = directory / declaration["file"]
    c.check(sha(path) == declaration["sha256"] and path.stat().st_size == declaration["bytes"], "Saved file SHA/size differs")
    with np.load(path, allow_pickle=False) as archive:
        c.check(set(archive.files) == set(arrays), "Saved keys differ")
        values = {k: archive[k].copy(order="C") for k in archive.files}
    for key, value in values.items():
        c.check(metadata(value) == arrays[key] and np.isfinite(value).all(), "Saved tensor bytes/shape/dtype differ")
    return values


def comparison(actual, reference):
    if actual.dtype != reference.dtype or actual.dtype != np.float32 or actual.shape != reference.shape:
        raise ValueError("Equal FP32 arrays required")
    x, y = actual.astype(np.float64), reference.astype(np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Finite outputs required")
    diff = x - y
    failures = int(np.count_nonzero(np.abs(diff) > 1e-5 + 1e-4 * np.abs(y)))
    norm = float(np.linalg.norm(y.ravel()))
    diff_norm = float(np.linalg.norm(diff.ravel()))
    return dict(rtol=1e-4, atol=1e-5, failed_elements=failures, element_count=actual.size,
                gate_passed=failures == 0, array_equal=bool(np.array_equal(actual, reference)),
                bitwise_equal=actual.tobytes(order="C") == reference.tobytes(order="C"),
                max_abs=float(np.abs(diff).max()), relative_l2=diff_norm / norm if norm else (0.0 if not diff_norm else None))


def compare_record(actual, recorded, c):
    for key, value in actual.items():
        other = recorded[key]
        same = math.isclose(value, other, rel_tol=1e-12, abs_tol=1e-14) if type(value) is float else value == other
        c.check(same, "Recorded comparison differs: " + key)


def audit(args):
    c = Checks()
    recipe_path = args.rewrite_dir / "rewrite.json"
    recipe = json.loads(recipe_path.read_text())
    cpu_path = args.cpu_dir / "ort.json"
    cpu = json.loads(cpu_path.read_text())
    ptq_path = args.ptq_dir / "compile.json"
    ptq = json.loads(ptq_path.read_text())
    parent_ptq = json.loads(args.parent_ptq.read_text())
    reference_path, capture_path = args.reference_dir / "ort.json", args.capture_dir / "capture.json"
    c.check(sha(args.parent_onnx) == PARENT_SHA and sha(args.parent_recipe) == PARENT_RECIPE_SHA
            and sha(reference_path) == REFERENCE_SHA and sha(capture_path) == CAPTURE_SHA and sha(args.parent_ptq) == PARENT_PTQ_SHA,
            "Frozen ancestor hashes changed")
    reference, capture = json.loads(reference_path.read_text()), json.loads(capture_path.read_text())
    c.check(sha(args.editor) == EDITOR_SHA == recipe["script_sha256"] == cpu["rewriter_sha256"] == ptq["rewriter_sha256"], "Editor/source chain changed")
    c.check(sha(args.board_source_dir / "run-v38-spatial-reciprocal-ort-board.py") == cpu["script_sha256"] == CPU_DRIVER_SHA
            and sha(args.board_source_dir / "compile-v38-reciprocal-int8-board.py") == ptq["script_sha256"] == PTQ_DRIVER_SHA, "Actual drivers changed")
    candidate_path = args.rewrite_dir / recipe["output"]["file"]
    c.check(sha(candidate_path) == recipe["output"]["sha256"] == cpu["candidate_graph"]["sha256"] == ptq["input"]["sha256"]
            and candidate_path.stat().st_size == recipe["output"]["bytes"] == cpu["candidate_graph"]["bytes"], "Candidate graph identity changed")
    c.check(sha(recipe_path) == cpu["rewrite_sha256"] == ptq["rewrite_sha256"] and sha(cpu_path) == ptq["ORT_receipt_sha256"], "Recipe/CPU/PTQ chain changed")
    spec = importlib.util.spec_from_file_location("frozen_pure_static_editor", args.editor)
    editor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(editor)
    parent = onnx.load_model(args.parent_onnx, load_external_data=False)
    candidate = onnx.load_model(candidate_path, load_external_data=False)
    proof = editor.validate_preservation(parent, candidate)  # protobuf/checker/shape inference only
    c.check(proof == recipe["preservation"] == cpu["graph_preservation"] == ptq["graph_preservation"], "Static preservation proof differs")
    c.check(len(parent.graph.initializer) == 65 and len(candidate.graph.initializer) == 66, "Original/new initializer counts differ")
    for old, new, declaration in zip(parent.graph.initializer, candidate.graph.initializer, recipe["original_initializer_manifest"]):
        c.check(old.SerializeToString() == new.SerializeToString(), "Original initializer protobuf changed")
        c.check(declaration["name"] == old.name and declaration["shape"] == list(old.dims)
                and declaration["protobuf_sha256"] == hashlib.sha256(old.SerializeToString()).hexdigest()
                and declaration["raw_sha256"] == hashlib.sha256(onnx.numpy_helper.to_array(old).tobytes()).hexdigest(), "Initializer manifest differs")
    c.check(len(candidate.graph.node) == len(parent.graph.node) + 10 and
            sum(n.op_type == "Div" for n in candidate.graph.node) == 12, "Ten substitutions/two other Div count differs")
    for document, flag in ((recipe, "sources_and_parent_unchanged"), (cpu, "sources_and_inputs_unchanged"), (ptq, "sources_and_inputs_unchanged")):
        c.check(document[flag] is True and document["source_pins"] == document["source_hashes_after"], "Source before/after maps differ")
        c.check(document["candidate_accepted"] is False and document["historical_parent_FP32_gate_passed"] is False
                and document["historical_parent_rejection_retained"] is True, "Original rejection promoted")
    c.check(cpu["status"] == "completed_CPU_ORT_reciprocal_diagnostic_parent_rejection_retained_not_VO"
            and cpu["NPU_used"] is False and cpu["gate"] == {"rtol":1e-4, "atol":1e-5}
            and cpu["providers"] == ["CPUExecutionProvider"] and cpu["graph_optimization"] == "ORT_DISABLE_ALL", "Actual CPU scope/gate differs")
    c.check(cpu["cleanup_errors"] == [] and cpu["affinity_restored"] is True
            and cpu["session_lifecycle"]["ORT_session_references_dropped"] is True, "CPU cleanup incomplete")
    c.check([(r["frame_index"], r["role"]) for r in cpu["calls"]] == [(i, role) for i in range(3) for role in ("parent", "candidate")], "Six fixed CPU call order differs")
    rows = []
    for frame in range(3):
        parent_row, candidate_row = cpu["calls"][frame*2:frame*2+2]
        old_row = next(r for r in reference["calls"] if r["frame_index"] == frame and r["method"] == "spatial")
        original_input = capture["inputs"][frame]
        c.check(parent_row["frame_id"] == candidate_row["frame_id"] == old_row["frame_id"] == original_input["frame_id"]
                and parent_row["source_image"] == candidate_row["source_image"] == old_row["source_image"], "RGB identity/source differs")
        inputs = saved(args.reference_dir, old_row["input_artifact"],
                       {"normalized_bgr":old_row["input_metadata"], "centers_xy":old_row["centers"], "patch_indices":old_row["indices"]}, c)
        for row in (parent_row, candidate_row):
            c.check(row["input_metadata"] == old_row["input_metadata"] and row["centers"] == old_row["centers"]
                    and row["indices"] == old_row["indices"] and row["input_source_artifact"] == old_row["input_artifact"], "Input tensor/index provenance differs")
        original_npy = args.capture_dir / original_input["input"]["file"]
        c.check(sha(original_npy) == original_input["input"]["sha256"], "Original saved input SHA differs")
        raw_input = np.load(original_npy, allow_pickle=False)
        c.check(metadata(raw_input) == original_input["array"] and raw_input.tobytes() == inputs["normalized_bgr"].tobytes(), "Same once-normalized saved pixels required")
        xy, indices = inputs["centers_xy"], inputs["patch_indices"]
        c.check(xy.dtype == indices.dtype == np.int64 and xy.shape == (96,2) and indices.shape == (96,)
                and np.array_equal(indices, xy[:,1]*152+xy[:,0]) and np.all((xy[:,0]>=1)&(xy[:,0]<151))
                and np.all((xy[:,1]>=1)&(xy[:,1]<115)), "Ordered integer sampling differs")
        original_row = next(r for r in capture["calls"] if r["frame_index"] == frame)
        original = saved(args.capture_dir, original_row["artifact"], original_row["arrays"], c)
        gold = dict(fmap=original["fmap"][:,0], context=np.take(original["imap"][:,0].reshape(1,384,-1),indices,axis=2).transpose(0,2,1).copy(order="C"))
        old = saved(args.reference_dir, old_row["artifact"], old_row["arrays"], c)
        actual_parent = saved(args.cpu_dir, parent_row["artifact"], parent_row["arrays"], c)
        actual_candidate = saved(args.cpu_dir, candidate_row["artifact"], candidate_row["arrays"], c)
        summary = dict(frame_index=frame, frame_id=parent_row["frame_id"], candidate_vs_parent={}, parent_vs_original={}, candidate_vs_original={})
        for key in ("fmap", "context"):
            for actual, recorded in ((comparison(actual_parent[key],old[key]),parent_row["versus_saved_v33_spatial_ORT"][key]),
                                     (comparison(actual_parent[key],gold[key]),parent_row["versus_original_CPU"][key]),
                                     (comparison(actual_candidate[key],gold[key]),candidate_row["versus_original_CPU"][key]),
                                     (comparison(actual_candidate[key],actual_parent[key]),candidate_row["versus_current_parent_ORT"][key])):
                compare_record(actual, recorded, c)
            summary["candidate_vs_parent"][key] = comparison(actual_candidate[key],actual_parent[key])
            summary["parent_vs_original"][key] = comparison(actual_parent[key],gold[key])
            summary["candidate_vs_original"][key] = comparison(actual_candidate[key],gold[key])
        rows.append(summary)
    parent_gate = all(v["gate_passed"] for r in rows for v in r["candidate_vs_parent"].values())
    bitwise = all(v["bitwise_equal"] for r in rows for v in r["candidate_vs_parent"].values())
    original_gates = {role:all(v["gate_passed"] for r in rows for v in r[role+"_vs_original"].values()) for role in ("parent", "candidate")}
    c.check(parent_gate is True and cpu["candidate_vs_current_parent_strict_gate_passed"] is True
            and bitwise is False and cpu["candidate_vs_current_parent_bitwise_equal"] is False
            and original_gates == cpu["original_CPU_strict_gate_each_role"] == {"parent":False,"candidate":False}, "Numerical summary promoted or changed")
    c.check(ptq["status"] == "compiled_board_reciprocal_PTQ_parent_rejection_retained_not_runtime_validated"
            and ptq["parent_PTQ_receipt_sha256"] == PARENT_PTQ_SHA and ptq["configuration"] == parent_ptq["configuration"], "PTQ configuration/ancestor differs")
    parent_raw_dataset_hashes = [digest for path, digest in parent_ptq["source_pins"].items() if Path(path).name == "dataset.txt"]
    c.check(len(parent_raw_dataset_hashes) == 1, "Exactly one raw parent calibration list pin required")
    c.check(ptq["calibration"]["rows"] == parent_ptq["calibration"]["rows"] and len(ptq["calibration"]["rows"]) == 32
            and ptq["calibration"]["source_dataset_sha256"] == parent_raw_dataset_hashes[0]
            and ptq["calibration"]["dataset_sha256"] == parent_ptq["calibration"]["dataset_sha256"], "Raw/resolved lists or 32 calibration rows differ")
    c.check(sum(len(r["input_files"]) for r in ptq["calibration"]["rows"]) == 64, "All 64 file pins required")
    c.check([r["name"] for r in ptq["operations"]] == ["config","load_onnx","board_CPU_calibration_and_build","export_rknn"]
            and all(r["code"] == 0 and math.isfinite(r["seconds"]) and r["seconds"] >= 0 for r in ptq["operations"]), "Actual PTQ operations failed")
    c.check(ptq["released"] is True and ptq["cleanup_errors"] == [] and ptq["deployment_inference_executed"] is False
            and ptq["INT8_operator_coverage_validated"] is False and ptq["candidate_vs_original_CPU_strict_gate_passed"] is False, "PTQ cleanup/scope/rejection differs")
    return dict(schema="v38_reciprocal_saved_CPU_static_ONNX_PTQ_independent_review", status="passed_evidence_checks_original_precision_rejection_retained",
                checks_passed=c.count, checks_failed=0, auditor_sha256=sha(__file__), recipe_sha256=sha(recipe_path),
                CPU_receipt_sha256=sha(cpu_path), PTQ_receipt_sha256=sha(ptq_path), candidate_ONNX_sha256=sha(candidate_path),
                rewriter_sha256=EDITOR_SHA, original_initializers_byte_exact=65, new_nonlearned_initializers=1, Div_replacements=10,
                actual_CPU_saved_artifacts_checked=6, actual_CPU_saved_arrays_checked=12, candidate_parent_fixed_gate_passed=True,
                candidate_parent_bitwise_equal=False, original_CPU_gate_passed_each_role=original_gates, actual_frame_errors=rows,
                CPU_call_seconds=[r["forward_seconds"] for r in cpu["calls"]], CPU_cleanup_errors=[],
                PTQ=dict(same_32_rows_64_input_file_pins_and_order=True, same_raw_and_resolved_dataset_hashes=True,
                         same_configuration=True, build_seconds=ptq["operations"][2]["seconds"], released=True,
                         output_model=ptq["output"], output_model_file_rehashed=False,
                         calibration_content_verification="Metadata equality only; no calibration NPY/pixel reads or PTQ execution on host."),
                source_scope="Local frozen rewriter/CPU/PTQ helpers and supplied saved artifacts rehashed; remaining board source maps checked before/after consistency.",
                candidate_accepted=False, original_parent_rejection_retained=True, no_model_OR_ORT_SDK_compiler_BA_execution_by_auditor=True,
                limits=["This audit proves recorded CPU differences and static preservation, not original CPU acceptance.",
                        "PTQ build is not NPU runtime placement, speed or quantized quality evidence; the compiled file was not present for host rehash.",
                        "No VO, fresh observation, trajectory or realtime acceptance."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rewrite-dir","cpu-dir","ptq-dir","parent-ptq","parent-onnx","parent-recipe","reference-dir","capture-dir","editor","board-source-dir","output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Never overwrite a prior audit receipt")
    result = audit(args)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"status":result["status"],"checks":result["checks_passed"],"receipt_sha256":sha(args.output)}))


if __name__ == "__main__":
    main()
