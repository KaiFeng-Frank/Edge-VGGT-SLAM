#!/usr/bin/env python3
"""Audit saved calibration metadata/arrays without image decoding or inference.

MIT project helper. DPVO preprocessing provenance: Princeton Vision & Learning
Lab, MIT, commit 0ac95b656d1fda91c271d2a106460d19ad966fc7.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

GENERATOR_SHA = "6da7358beea87520676083d52fa5171034cd10f8eaf31e47b9e753ab0aec4bb4"
POLICY_SHA = "7fde7e1635e8310b8782ddd6881a28aefa570eceafca0cb62e0809c3c5e2dcab"
ENCODER_SHA = "fc6fc662613a720cfb779a360a63906a593a96ff08addecb9d01cfd8767ca7a8"
RGB_SHA = "532884d60bef0c9a05e14bf311998883baf59a2fa47009d1021f2c7937bd69a3"
CALIBRATION_INDICES = list(range(256, 753, 16))
SAMPLING_KEYS = {"centers", "consumed_inverse_depth", "state_before", "state_after_centers", "state_after_depth"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def info(a):
    return dict(shape=list(a.shape), dtype=str(a.dtype), bytes=int(a.nbytes),
                tensor_sha256=hashlib.sha256(a.tobytes(order="C")).hexdigest())


class Checks:
    def __init__(self):
        self.count = 0
        self.failures = []

    def check(self, condition, label):
        self.count += 1
        if not bool(condition):
            self.failures.append(label)

    def array(self, a, metadata, label):
        actual = info(a)
        for field in actual:
            self.check(metadata.get(field) == actual[field], f"{label}: {field}")


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def array_metadata(checks, metadata, shape, dtype, label):
    checks.check(metadata.get("shape") == list(shape), label + ": shape")
    checks.check(metadata.get("dtype") == dtype, label + ": dtype")
    checks.check(metadata.get("bytes") == math.prod(shape) * np.dtype(dtype).itemsize, label + ": bytes")
    checks.check(valid_sha(metadata.get("tensor_sha256")), label + ": SHA syntax")


def rgb_rows(text):
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        stamp, name = line.split()
        value = float(stamp)
        if not math.isfinite(value) or (rows and value <= rows[-1]["timestamp"]):
            raise ValueError("Non-monotonic original RGB index")
        if name != "rgb/" + stamp + ".png":
            raise ValueError("Original RGB filename/timestamp mismatch")
        rows.append(dict(source_index=len(rows), timestamp_text=stamp, timestamp=value, relative_path=name))
    return rows


def check_dataset(checks, text):
    expected = [f"image-{i:02d}.npy indices-{i:02d}.npy" for i in range(32)]
    checks.check(text == "\n".join(expected) + "\n", "Exact dataset rows, relative filenames and input order")


def check_sampling(checks, arrays, sample_records, archive_metadata):
    checks.check(set(arrays) == SAMPLING_KEYS, "Sampling archive keys")
    state_length = arrays["state_before"].shape[-1]
    shapes = {"centers": ((32, 96, 2), "int64"), "consumed_inverse_depth": ((32, 96), "float32")}
    shapes.update({key: ((32, state_length), "uint8") for key in SAMPLING_KEYS if key.startswith("state_")})
    checks.check(state_length > 0, "Nonempty saved RNG state")
    derived = []
    for key, (shape, dtype) in shapes.items():
        a = arrays[key]
        checks.check(a.shape == shape and str(a.dtype) == dtype, key + ": exact aggregate contract")
        checks.array(a, archive_metadata[key], "archive " + key)
        for i in range(32):
            checks.array(a[i], sample_records[i]["RNG"]["arrays"][key], f"sample {i} {key}")
    centers = arrays["centers"]
    checks.check(np.all((centers[..., 0] >= 1) & (centers[..., 0] <= 150) &
                        (centers[..., 1] >= 1) & (centers[..., 1] <= 114)), "All original interior centers")
    depth = arrays["consumed_inverse_depth"]
    checks.check(np.isfinite(depth).all() and np.all((depth >= 0) & (depth < 1)), "Consumed depth finite in [0,1)")
    for i in range(32):
        indices = np.ascontiguousarray(centers[i, :, 1] * np.int64(152) + centers[i, :, 0])
        checks.array(indices, sample_records[i]["index_array"], f"sample {i} derived indices")
        derived.append(indices)
        before, middle, after = (arrays[k][i] for k in ("state_before", "state_after_centers", "state_after_depth"))
        checks.check(not np.array_equal(before, middle), f"sample {i}: centers consume RNG state")
        checks.check(not np.array_equal(middle, after), f"sample {i}: depth consumes RNG state")
        if i:
            checks.check(np.array_equal(arrays["state_after_depth"][i-1], before), f"RNG continuity {i-1}->{i}")
    return derived


def audit(run_dir, original_rgb_index, expected_receipt_sha):
    directory = Path(run_dir)
    checks = Checks()
    receipt_path = directory / "calibration.json"
    checks.check(sha(receipt_path) == expected_receipt_sha, "Expected calibration receipt SHA")
    report = json.loads(receipt_path.read_text())
    policy_path = directory / "policy.json"
    policy = json.loads(policy_path.read_text())
    checks.check(sha(policy_path) == POLICY_SHA, "Frozen policy bytes")
    checks.check(report["policy_sha256"] == POLICY_SHA and report["policy"] == policy, "Receipt policy copy")
    checks.check(report["schema"] == "v36_DPVO_board_calibration_data", "Receipt schema")
    checks.check(report["status"] == "generated_fixed_calibration_data_not_quantized_or_validated", "Generation completed")
    checks.check(report["script_sha256"] == GENERATOR_SHA, "Frozen generator SHA")
    checks.check(report["source_pins"] == report["source_hashes_after"] and report["sources_and_inputs_unchanged"] is True,
                 "Board-recorded source pins before/after unchanged")
    checks.check(report["cleanup_errors"] == [], "No reported cleanup errors")
    for field in ("GT_read", "model_forward_executed", "RKNN_build_executed", "online_graph_equivalence_claimed", "accuracy_or_realtime_validated"):
        checks.check(report[field] is False, field + " remains false")
    samples = report["samples"]
    checks.check(len(samples) == 32, "Exactly 32 samples")
    if len(samples) != 32:
        raise ValueError("Cannot audit incomplete sample population")
    checks.check(policy["calibration_source_indices"] == CALIBRATION_INDICES, "Fixed 256+16*i policy")
    checks.check(policy["evaluation_source_indices"] == list(range(192)), "First192 evaluation policy")
    checks.check(not set(CALIBRATION_INDICES).intersection(policy["evaluation_source_indices"]), "Calibration/evaluation frame disjointness")
    checks.check(policy["independent_scene_claim"] is False, "Same sequence remains explicit")
    checks.check(sha(original_rgb_index) == RGB_SHA, "Independent original RGB metadata SHA")
    rows = rgb_rows(Path(original_rgb_index).read_text())
    checks.check(len(rows) == 798, "Original source has 798 rows")
    if len(rows) != 798:
        raise ValueError("Original RGB population changed")
    pins = report["source_pins"]
    for name, expected in (("generate-v36-dpvo-calibration-board.py", GENERATOR_SHA),
                           ("calibration-policy.json", POLICY_SHA), ("rgb.txt", RGB_SHA), ("v30_frame_encoder.py", ENCODER_SHA)):
        matches = [v for k, v in pins.items() if Path(k).name == name]
        checks.check(matches == [expected], "Recorded source pin: " + name)
    checks.check(len(pins) == 36 and all(valid_sha(v) for v in pins.values()), "Exactly four source pins plus 32 image pins")
    dataset = directory / "dataset.txt"
    checks.check(sha(dataset) == report["dataset"]["sha256"] and dataset.stat().st_size == report["dataset"]["bytes"], "Dataset file SHA/bytes")
    check_dataset(checks, dataset.read_text())
    sampling_path = directory / "sampling.npz"
    checks.check(sha(sampling_path) == report["sampling_artifact"]["sha256"] and
                 sampling_path.stat().st_size == report["sampling_artifact"]["bytes"], "Sampling file SHA/bytes")
    with np.load(sampling_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    derived = check_sampling(checks, arrays, samples, report["sampling_artifact"]["arrays"])
    body_counts = {"image": 0, "indices": 0}
    missing = []
    recorded_input_bytes = 0
    for i, sample in enumerate(samples):
        checks.check(sample["calibration_index"] == i, f"sample {i}: calibration order")
        for field, value in rows[CALIBRATION_INDICES[i]].items():
            checks.check(sample[field] == value, f"sample {i}: original {field}")
        image_pins = [v for k, v in pins.items() if Path(k).as_posix().endswith("/" + sample["relative_path"])]
        checks.check(image_pins == [sample["source_image_sha256"]], f"sample {i}: source image pin")
        checks.check(type(sample["source_image_bytes"]) is int and sample["source_image_bytes"] > 0, f"sample {i}: source image bytes")
        for key, shape, dtype in (("decoded_BGR", (480, 640, 3), "uint8"), ("cropped_BGR", (464, 608, 3), "uint8"),
                                  ("image_array", (1, 3, 464, 608), "float32"), ("index_array", (96,), "int64")):
            array_metadata(checks, sample[key], shape, dtype, f"sample {i} {key} recorded metadata")
        checks.check(sample["RNG"]["sequence"] == policy["random_consumption_per_selected_frame"], f"sample {i}: recorded draw sequence")
        inputs = sample["input_files"]
        checks.check([v["file"] for v in inputs] == [f"image-{i:02d}.npy", f"indices-{i:02d}.npy"], f"sample {i}: exact input files/order")
        for item, key, kind in zip(inputs, ("image_array", "index_array"), ("image", "indices")):
            checks.check(item["bytes"] == sample[key]["bytes"] + 128 and valid_sha(item["sha256"]), f"sample {i} {kind}: recorded NPY header/bytes/SHA")
            recorded_input_bytes += item["bytes"]
            path = directory / item["file"]
            if not path.is_file():
                missing.append(item["file"])
                continue
            checks.check(sha(path) == item["sha256"] and path.stat().st_size == item["bytes"], f"sample {i} {kind}: available file bytes/SHA")
            a = np.load(path, allow_pickle=False)
            checks.array(a, sample[key], f"sample {i} {kind}: available raw array")
            checks.check(np.isfinite(a).all(), f"sample {i} {kind}: available array finite")
            if kind == "indices":
                checks.check(np.array_equal(a, derived[i]), f"sample {i}: available indices equal centers-derived values")
            else:
                checks.check(a.min() >= -.5 and a.max() <= 1.5, f"sample {i}: available normalized range")
            body_counts[kind] += 1
    computed_before = recorded_input_bytes + sampling_path.stat().st_size + dataset.stat().st_size + policy_path.stat().st_size
    checks.check(computed_before == report["output_bytes_before_receipt"], "Independent sum of recorded input sizes plus available metadata")
    total_bytes = computed_before + receipt_path.stat().st_size
    checks.check(total_bytes <= policy["output_file_budget_bytes"] == 256 * 1024**2, "Full generator-output budget including receipt")
    checks.check(math.isfinite(report["generation_seconds"]) and report["generation_seconds"] > 0, "Recorded generation timing finite positive")
    complete = not missing
    return dict(schema="v36_DPVO_calibration_independent_review", status="passed" if not checks.failures else "failed",
                checks=checks.count, failures=checks.failures, audit_script_sha256=sha(__file__),
                input_receipt_sha256=sha(receipt_path), policy_sha256=sha(policy_path), sampling_sha256=sha(sampling_path),
                dataset_sha256=sha(dataset), original_rgb_index_sha256=sha(original_rgb_index),
                audit_scope="saved_sampling_and_available_input_arrays" if complete else "saved_sampling_and_metadata_only",
                complete_input_body_verification=complete, independent_input_body_counts=body_counts,
                unavailable_input_files=missing, selected_source_indices=CALIBRATION_INDICES,
                calibration_frames=32, evaluation_frames=192, selection_disjoint=True, same_sequence_not_independent_scene=True,
                sampling_arrays_verified=5, sampling_rows_verified=160, RNG_state_continuity_joins=31,
                derived_INT64_index_rows_verified=32, RNG_draws_replayed=False,
                preprocessing_reexecuted=False, real_image_decoding=False, GT_read=False, model_forward=False,
                ORT_execution=False, RKNN_build=False, board_execution=False, raw_inputs_modified=False,
                board_recorded_generation_seconds=report["generation_seconds"], board_recorded_peak_rss_mib=report["peak_rss_mib"],
                file_budget=dict(recorded_input_file_bytes=recorded_input_bytes, available_metadata_before_receipt_bytes=computed_before-recorded_input_bytes,
                                 independently_summed_total_bytes=total_bytes, maximum_bytes=policy["output_file_budget_bytes"],
                                 all_output_file_bytes_independently_read=complete),
                limitations=["Unavailable NPY file hashes and image pixel metadata are board-recorded, not independently rehashed.",
                             "RNG states are independently hashed and joined; seed/draw algorithm semantics are source/policy evidence, not replayed RNG.",
                             "Original source PNG hashes and source before/after equality are board-recorded; original rgb.txt metadata is independently hashed.",
                             "Preprocessing, quantization, tensor transport, accuracy and realtime acceptance are not established by this audit."])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rgb-index", type=Path, required=True)
    parser.add_argument("--calibration-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    protected = [args.run_dir.resolve(), args.rgb_index.resolve(), Path(__file__).resolve()]
    output = args.output.resolve()
    if output.exists() or any(output == p or output.is_relative_to(p) for p in protected):
        raise ValueError("New audit receipt outside raw input paths required")
    try:
        result = audit(args.run_dir, args.rgb_index, args.calibration_sha256)
    except Exception as exc:
        result = dict(schema="v36_DPVO_calibration_independent_review", status="failed", error=repr(exc),
                      audit_script_sha256=sha(__file__), model_forward=False, raw_inputs_modified=False)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result.get(k) for k in ("status", "checks", "failures", "error")}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
