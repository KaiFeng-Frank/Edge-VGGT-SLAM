#!/usr/bin/env python3
"""Link saved board PTQ file records to fixed calibration provenance.

MIT project helper. This reads JSON, dataset text and saved sampling arrays;
it never decodes images, imports a neural runtime, or executes calibration.
Unavailable image/index NPY bodies remain explicitly unverified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

COMPILER_SHA = "0f66e0ead40ab6eff546a47367ee93ec044b2b84e064eb8e9876352e0ca05fb0"
GENERATOR_SHA = "6da7358beea87520676083d52fa5171034cd10f8eaf31e47b9e753ab0aec4bb4"
POLICY_SHA = "7fde7e1635e8310b8782ddd6881a28aefa570eceafca0cb62e0809c3c5e2dcab"
SPARSE_GRAPH_SHA = "c52cecd383e60ba564dcb5100be6c7e975fa6ecc051ed89e2c09bff73bc071dc"
CONFIGURATION = dict(target_platform="rk3588", float_dtype="float16", optimization_level=3,
    quantized_dtype="w8a8", quantized_algorithm="normal", quantized_method="channel",
    mean_values=None, std_values=None, quant_img_RGB2BGR=False,
    do_quantization=True, auto_hybrid=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


class Checks:
    def __init__(self):
        self.count, self.failures = 0, []

    def check(self, condition, label):
        self.count += 1
        if not bool(condition):
            self.failures.append(label)


def check_rows(checks, compiled, generated, centers, source_pins):
    checks.check(len(compiled) == len(generated) == 32, "Exactly 32 calibration rows")
    if len(compiled) != 32 or len(generated) != 32:
        raise ValueError("Incomplete calibration population")
    checks.check(centers.dtype == np.int64 and centers.shape == (32, 96, 2), "Saved integer center contract")
    total = 0
    bindings = []
    for i, (row, sample) in enumerate(zip(compiled, generated)):
        checks.check(row["index"] == sample["calibration_index"] == i, f"row {i}: order")
        checks.check(sample["source_index"] == 256 + 16*i, f"row {i}: fixed source selection")
        checks.check(sample["source_index"] not in range(192), f"row {i}: evaluation disjoint")
        files = row["input_files"]
        checks.check(len(files) == len(sample["input_files"]) == 2, f"row {i}: two graph inputs")
        if len(files) != 2:
            raise ValueError("Expected image then integer indices")
        for j, (actual, expected) in enumerate(zip(files, sample["input_files"])):
            wanted = f"{'image' if j == 0 else 'indices'}-{i:02d}.npy"
            checks.check(Path(actual["file"]).is_absolute(), f"row {i}/{j}: resolved absolute path")
            checks.check(Path(actual["file"]).name == expected["file"] == wanted, f"row {i}/{j}: input role/order")
            checks.check(actual["sha256"] == expected["sha256"], f"row {i}/{j}: generator/compiler file SHA")
            checks.check(actual["bytes"] == expected["bytes"], f"row {i}/{j}: generator/compiler file bytes")
            checks.check(source_pins.get(actual["file"]) == expected["sha256"], f"row {i}/{j}: before/after pinned input")
            total += actual["bytes"]
        checks.check(row["image_dtype"] == sample["image_array"]["dtype"] == "float32" and
                     row["image_shape"] == sample["image_array"]["shape"] == [1, 3, 464, 608], f"row {i}: normalized image contract")
        checks.check(row["indices_dtype"] == sample["index_array"]["dtype"] == "int64" and
                     row["indices_shape"] == sample["index_array"]["shape"] == [96], f"row {i}: INT64 index contract")
        checks.check(row["index_order_preserved"] is True and row["index_cast_performed"] is False, f"row {i}: no cast/reorder")
        checks.check(math.isfinite(row["image_min"]) and math.isfinite(row["image_max"]) and
                     -.500001 <= row["image_min"] <= row["image_max"] <= 1.500001, f"row {i}: recorded normalized range")
        indices = np.ascontiguousarray(centers[i, :, 1]*np.int64(152) + centers[i, :, 0])
        checks.check(hashlib.sha256(indices.tobytes(order="C")).hexdigest() == sample["index_array"]["tensor_sha256"],
                     f"row {i}: saved centers derive original index raw SHA")
        checks.check(row["index_min"] == int(indices.min()) and row["index_max"] == int(indices.max()),
                     f"row {i}: PTQ recorded index extrema")
        bindings.append(dict(calibration_index=i, source_index=sample["source_index"],
            timestamp_text=sample["timestamp_text"], source_image_sha256=sample["source_image_sha256"],
            input_files=[{k: v for k, v in item.items() if k != "file"} | {"file": Path(item["file"]).name} for item in files]))
    return total, bindings


def audit(compile_path, compile_sha, calibration_dir, calibration_sha, review_path, review_sha):
    c = Checks()
    compile_path, directory, review_path = Path(compile_path), Path(calibration_dir), Path(review_path)
    generated_path = directory / "calibration.json"
    c.check(sha(compile_path) == compile_sha, "Expected PTQ receipt bytes")
    c.check(sha(generated_path) == calibration_sha, "Expected generator receipt bytes")
    c.check(sha(review_path) == review_sha, "Frozen calibration independent review bytes")
    ptq, generated, prior = (json.loads(p.read_text()) for p in (compile_path, generated_path, review_path))
    c.check(prior["status"] == "passed" and prior["failures"] == [] and prior["input_receipt_sha256"] == calibration_sha,
            "Prior sampling/metadata audit passed and binds generator receipt")
    c.check(ptq["schema"] == "v35_board_only_DPVO_encoder_PTQ" and
            ptq["status"] == "compiled_board_PTQ_not_runtime_or_trajectory_validated", "PTQ build-only status")
    c.check(ptq["script_sha256"] == COMPILER_SHA and ptq["configuration"] == CONFIGURATION, "Frozen compiler/configuration")
    c.check(generated["script_sha256"] == GENERATOR_SHA and generated["policy_sha256"] == POLICY_SHA, "Frozen calibration policy/generator")
    c.check(ptq["released"] is True and ptq["cleanup_errors"] == [], "Toolkit release and cleanup")
    c.check(ptq["source_pins"] == ptq["source_hashes_after"] and ptq["sources_and_inputs_unchanged"] is True,
            "Board-recorded PTQ source/input hashes unchanged")
    c.check(generated["source_pins"] == generated["source_hashes_after"] and generated["sources_and_inputs_unchanged"] is True,
            "Board-recorded generator hashes unchanged")
    c.check(ptq["PTQ_requested"] is True and ptq["host_calibration_executed"] is False and
            ptq["calibration_location"] == "aarch64 board CPU", "Actual calibration location remains board")
    for field in ("candidate_accepted", "production_changed", "INT8_operator_coverage_validated", "deployment_inference_executed"):
        c.check(ptq[field] is False, field + " remains unvalidated/unchanged")
    c.check(ptq["diagnostic_only"] is True and ptq["original_FP32_gate"] == dict(rtol=1e-4, atol=1e-5, measured=False),
            "Original precision gate is preserved and unmeasured")
    ops = ptq["operations"]
    c.check([r["name"] for r in ops] == ["config", "load_onnx", "board_CPU_calibration_and_build", "export_rknn"], "Build operation order")
    c.check(all(type(r["code"]) is int and r["code"] == 0 and math.isfinite(r["seconds"]) and r["seconds"] >= 0 for r in ops),
            "Actual build operations succeeded with finite timings")
    c.check(ptq["input"]["sha256"] == SPARSE_GRAPH_SHA and ptq["ancestor"]["sparse_onnx_sha256"] == SPARSE_GRAPH_SHA,
            "Original sparse graph retained")
    c.check(ptq["ancestor"]["method"] == "original" and ptq["ancestor"]["suffix_CPU_bitwise_gate_passed"] is True and
            ptq["ancestor"]["parent_CPU_strict_gate_passed"] is True, "Original parent/suffix gate provenance")
    c.check(ptq["calibration"]["samples"] == 32 and ptq["calibration"]["mean_std_identity"] is True, "Fixed sample count, no hidden normalization")
    dataset_path = directory / "dataset.txt"
    c.check(sha(dataset_path) == generated["dataset"]["sha256"] == prior["dataset_sha256"], "Original relative dataset bytes")
    c.check(dataset_path.read_text() == "".join(f"image-{i:02d}.npy indices-{i:02d}.npy\n" for i in range(32)), "Relative dataset input order")
    dataset_pins = [v for k, v in ptq["source_pins"].items() if Path(k).name == "dataset.txt"]
    c.check(dataset_pins == [generated["dataset"]["sha256"]], "PTQ pins same original dataset")
    sampling_path = directory / generated["sampling_artifact"]["file"]
    c.check(sha(sampling_path) == generated["sampling_artifact"]["sha256"] == prior["sampling_sha256"], "Same saved sampling archive")
    with np.load(sampling_path, allow_pickle=False) as archive:
        centers = archive["centers"]
    total, bindings = check_rows(c, ptq["calibration"]["rows"], generated["samples"], centers, ptq["source_pins"])
    resolved_text = "".join(" ".join(item["file"] for item in row["input_files"]) + "\n" for row in ptq["calibration"]["rows"])
    reconstructed = hashlib.sha256(resolved_text.encode()).hexdigest()
    c.check(reconstructed == ptq["calibration"]["dataset_sha256"], "Resolved absolute dataset reconstructed from recorded paths")
    c.check([v for k, v in ptq["source_pins"].items() if Path(k).name == "dataset-resolved.txt"] == [reconstructed], "Resolved dataset pin")
    c.check(total == 108363776 and total <= generated["policy"]["output_file_budget_bytes"], "All 64 recorded NPY file sizes within budget")
    return dict(schema="v36_PTQ_calibration_provenance_independent_review", status="passed" if not c.failures else "failed",
        checks=c.count, failures=c.failures, audit_script_sha256=sha(__file__), compile_receipt_sha256=compile_sha,
        calibration_receipt_sha256=calibration_sha, prior_calibration_review_sha256=review_sha,
        sampling_sha256=sha(sampling_path), relative_dataset_sha256=sha(dataset_path), resolved_dataset_sha256=reconstructed,
        samples=32, input_file_metadata_bindings=64, derived_index_rows=32,
        selection_disjoint=True, same_sequence_not_independent_scene=True, bindings=bindings,
        recorded_input_file_bytes=total, recorded_model=ptq["output"],
        recorded_board_calibration_and_build_seconds=ops[2]["seconds"], recorded_peak_rss_mib=ptq["peak_rss_mib"],
        input_NPY_bodies_independently_rehashed=False, model_binary_independently_rehashed=False,
        source_before_after_equality_scope="board-recorded", audit_execution="saved_JSON_text_sampling_only",
        SDK_import=False, model_forward=False, board_execution=False, GT_read=False, raw_inputs_modified=False,
        INT8_operator_coverage_validated=False, original_FP32_gate_measured=False, trajectory_accuracy_validated=False,
        limitations=["The 64 NPY bodies and RKNN binary are board-only; this audit binds their recorded hashes, not their host bytes.",
            "The relative and resolved dataset lists have different bytes; the resolved list is independently reconstructed from recorded paths.",
            "PTQ build success proves neither internal INT8 coverage nor deployment precision, throughput or trajectory accuracy."])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("compile-json", "calibration-dir", "calibration-review", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("compile", "calibration", "calibration-review"):
        p.add_argument("--" + name + "-sha256", required=True)
    a = p.parse_args(argv)
    if a.output.exists():
        raise ValueError("Preserve prior audit; output must be new")
    result = audit(a.compile_json, a.compile_sha256, a.calibration_dir, a.calibration_sha256,
                   a.calibration_review, a.calibration_review_sha256)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "checks", "failures", "samples", "input_file_metadata_bindings")}), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
