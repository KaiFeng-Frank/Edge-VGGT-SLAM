"""Original VGGT prefix-capture provenance and saved-data contracts.

No model, tracing, compiler or calibration execution is provided here.
NumPy is imported only while reading saved NPY arrays.
"""
from __future__ import annotations

from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "v49_original_VGGT_pair_prefix_capture"
PLAN_SCHEMA = "v49_original_VGGT_real_image_split_plan"
PLAN_SHA = "61b14e0e0033a49059bf52c7dd9cbeaae8bd4393423b2bce827cc451f22caf37"
CAPTURE_SOURCE_SHA = "7a2279927a65e3740da933c1192c335d15f4f1803440ed84a27a2aebc8f02613"
CHECKPOINT_SHA = "d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0"
CHECKPOINT_BYTES = 5026874952
PAIR_EXPORT_SHA = "149450ffc675ce59950c6f89cf84adb41e388cd050eeb731b6d09a6fe16e9fe1"
PAIR_ONNX_SHA = "2cf1f4de19c91b42b5bf685ab5e949bd98b08a64fc2e78b7f0c8f07992a009fc"
FROZEN_INPUT_RAW_SHA = "85fbd0dc87e6576f1c54304f1fc7077b2625f01ff458baa931a10e0ce5d1d9f9"
FROZEN_POSITIONS_SHA = "8a01a56f84a95c19d3894590d4bcc05da80c1eb71a43801dd4abcd953322e064"
FROZEN_POSITIONS_RAW_SHA = "3387fc89d04b96637c9497fd2b4394c8d0ececf22867c74d358734fbc4e674fb"
MODEL_SOURCE_PINS_SHA = "ff50f23b7fd1e728a2757ae9796f03263b8392e1328f53fc6b82fe5e2c37d283"
CALIBRATION_FILE_SHA = "0043286979ae6abc29385eeba1abd441f0299fe25325301332a28d4a438a51db"
PREPROCESS_FUNCTION_SHA = "0353b56642d35e0f9f9752419287b0c111e3926357f5e09b6d5738f768f474df"
RUNTIME_SOURCE_PINS = {
    "vggt_slam/runtime.py": "baf65a5afcfa98584e25b1ef85d7cae31507a7da2b4cc03f97d2015776796e8c",
    "vggt_slam/image_rectification.py": "e8c8472ac4dc3b84c132f20afd35be4dbeb3c23a9491ae1299ae5b1938e7ffdd",
    "vggt_slam/edge_mapping.py": "73de3126ef20ea4a7bea8a44954ae8d0f88c77f1a77536cd60bf772038d06818",
}
LEGACY_IMAGES = [
    {"index": 0, "name": "1305031102.175304.png", "timestamp": 1305031102.175304,
     "sha256": "331c3fd7a3d5eefb19e3da2026625aa221ca1991f5ae30ff96060e2e3e45e368"},
    {"index": 8, "name": "1305031102.443271.png", "timestamp": 1305031102.443271,
     "sha256": "49deecb97bbb0ef1b8d14212b10d5cdb9a2ce7b4c852ad4838bc8e2a872d82df"},
]
INPUT_SHAPE = [2, 437, 1024]
POSITIONS_SHAPE = [2, 437, 2]
OUTPUT_SHAPE = [1, 874, 1024]
RGB_SHAPE = [2, 3, 252, 336]


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_text(value):
    require(isinstance(value, str) and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value), "Explicit lowercase SHA256 required")
    return value


def pinned(path, expected):
    require(sha(path) == sha_text(expected), f"SHA256 mismatch: {path}")
    return expected


def json_file(path):
    return json.loads(Path(path).read_text())


def timestamp_identity(image):
    value = Decimal(Path(image["name"]).stem)
    require(value.is_finite() and value > 0, "Invalid timestamp filename")
    require(type(image["timestamp"]) in (int, float) and math.isfinite(image["timestamp"])
            and float(value) == image["timestamp"], "Timestamp and PNG filename disagree")
    return value


def validate_plan(plan):
    require(plan.get("schema") == PLAN_SCHEMA and plan.get("dataset") == "rgbd_dataset_freiburg1_xyz"
            and plan.get("dataset_image_count") == 798 and plan.get("requested_image_size") == 336
            and plan.get("expected_tensor_shape") == RGB_SHAPE, "Wrong fixed original-image plan")
    require(plan.get("legacy_evaluation_images") == LEGACY_IMAGES, "Legacy evaluation exclusion changed")
    samples = plan.get("samples")
    require(isinstance(samples, list) and len(samples) == 10, "Exactly eight calibration and two evaluation samples required")
    counts, ids, all_indices, all_timestamps, all_hashes = {}, set(), set(), set(), set()
    legacy_indices = {v["index"] for v in LEGACY_IMAGES}
    legacy_timestamps = {timestamp_identity(v) for v in LEGACY_IMAGES}
    legacy_hashes = {v["sha256"] for v in LEGACY_IMAGES}
    for sample in samples:
        require(set(sample) == {"sample_id", "split", "images"}, "Unexpected plan sample fields")
        split, sample_id = sample["split"], sample["sample_id"]
        require(split in ("calibration", "evaluation") and isinstance(sample_id, str)
                and sample_id and all(c.isalnum() or c == "_" for c in sample_id)
                and sample_id not in ids, "Invalid or duplicate sample identity")
        ids.add(sample_id)
        counts[split] = counts.get(split, 0) + 1
        images = sample["images"]
        require(isinstance(images, list) and len(images) == 2, "Original pair must have two ordered images")
        pair_timestamps = []
        for image in images:
            require(set(image) == {"index", "name", "path", "timestamp", "bytes", "sha256"},
                    "Unexpected planned image fields")
            index = image["index"]
            require(type(index) is int and 0 <= index < 798 and index not in all_indices
                    and index not in legacy_indices, "Repeated or excluded image index")
            require(type(image["bytes"]) is int and image["bytes"] > 0, "Invalid PNG byte count")
            name = image["name"]
            require(isinstance(name, str) and Path(name).name == name and name.endswith(".png"), "Invalid PNG name")
            expected_path = "/mnt/vggt-data/vggt-slam2/datasets/rgbd_dataset_freiburg1_xyz/rgb/" + name
            require(image["path"] == expected_path, "Image is not the original TUM RGB path")
            stamp = timestamp_identity(image)
            digest = sha_text(image["sha256"])
            require(stamp not in all_timestamps and stamp not in legacy_timestamps and
                    digest not in all_hashes and digest not in legacy_hashes,
                    "Image timestamps or PNG SHA overlap across samples or legacy evaluation")
            all_indices.add(index)
            all_timestamps.add(stamp)
            all_hashes.add(digest)
            pair_timestamps.append(stamp)
        require(images[0]["index"] < images[1]["index"] and pair_timestamps[0] < pair_timestamps[1],
                "First/other image role or chronological order changed")
    require(counts == {"calibration": 8, "evaluation": 2}, "Wrong split cardinality")
    return samples


def parse_rgb_index(text):
    rows, stamps, names = [], set(), set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        require(len(fields) == 2, "Invalid RGB index row")
        stamp, relative = fields
        value = Decimal(stamp)
        require(value.is_finite() and value > 0 and value not in stamps, "Invalid or repeated RGB timestamp")
        name = Path(relative).name
        require(relative == "rgb/" + name and name.endswith(".png") and name not in names
                and Decimal(Path(name).stem) == value, "RGB filename and timestamp disagree")
        require(not rows or value > Decimal(rows[-1]["timestamp_string"]), "RGB timestamps must increase")
        rows.append({"index": len(rows), "timestamp_string": stamp, "name": name, "relative_path": relative})
        stamps.add(value)
        names.add(name)
    require(len(rows) == 798, "Original RGB index must have 798 images")
    return rows


def verify_plan_rows(plan, rows):
    for sample in validate_plan(plan):
        for image in sample["images"]:
            row = rows[image["index"]]
            require(row["name"] == image["name"] and Decimal(row["timestamp_string"]) == timestamp_identity(image),
                    "Plan index does not identify the same original image")
    return True


def safe_file(directory, relative):
    require(isinstance(relative, str) and relative and not any(c.isspace() for c in relative)
            and "\x00" not in relative, "Artifact path must have no whitespace or NUL")
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, "Artifact path escapes capture directory")
    directory = Path(directory).resolve()
    resolved = (directory / path).resolve()
    require(resolved.is_relative_to(directory) and resolved.is_file(), "Artifact must be a capture-owned file")
    return resolved


def load_array(directory, item, shape, dtype):
    """Read and re-pin saved data only; returns an owned NumPy copy."""
    import numpy as np
    path = safe_file(directory, item["file"])
    require(item["shape"] == list(shape) and item["dtype"] == dtype,
            "Saved artifact declared shape/dtype mismatch")
    require(type(item["bytes"]) is int and item["bytes"] == path.stat().st_size,
            "Saved artifact file byte count mismatch")
    require(item["bytes"] < 16 * 1024 * 1024, "NPY exceeds fixed capture tensor limit")
    pinned(path, item["sha256"])
    value = np.load(path, allow_pickle=False)
    require(isinstance(value, np.ndarray) and list(value.shape) == list(shape)
            and str(value.dtype) == dtype and value.flags.c_contiguous
            and np.isfinite(value).all(), "Saved NPY shape/dtype/layout/finiteness mismatch")
    require(value.nbytes == item["tensor_bytes"] and hashlib.sha256(memoryview(value).cast("B")).hexdigest()
            == sha_text(item["raw_sha256"]), "Saved NPY tensor bytes/raw SHA mismatch")
    return np.array(value, copy=True, order="C")


def require_call_counts(sample):
    counts = sample["call_counts"]
    evaluation = sample["split"] == "evaluation"
    require(counts["dino_completed"] == [1] * 24 and counts["dino_entered"] == [1] * 24,
            "Complete original DINO prefix did not run once")
    require(counts["frame_completed"] == ([1] + [0] * 23 if evaluation else [0] * 24)
            and counts["frame_entered"] == [1] + [0] * 23,
            "Frame prefix completion/controlled pre-hook stop mismatch")
    require(counts["global_completed"] == ([1] + [0] * 23 if evaluation else [0] * 24)
            and counts["global_entered"] == ([1] + [0] * 23 if evaluation else [0] * 24),
            "Global prefix completion mismatch")
    require(counts["camera_head_entered"] == counts["depth_head_entered"] == 0,
            "Capture unexpectedly ran downstream heads")
    require(sample["controlled_stop"] == ("after_original_global0" if evaluation else "before_original_frame0")
            and sample["hooks_removed"] is True and sample["hook_registry_restored"] is True
            and sample["forward_functions_unchanged"] is True, "Controlled prefix exit or cleanup failed")


def load_capture(directory: Path, expected_sha256: str):
    """Validate the portable capture and NPY data; source-board paths are metadata.

    Returns (report, report['samples'], pins). Pins are resolved capture-owned
    artifact paths. Physical PNG/checkpoint/source hashes were checked by the
    board capture; this reader does not claim to rehash unavailable board paths.
    """
    directory = Path(directory).resolve()
    report_path = safe_file(directory, "capture.json")
    pinned(report_path, expected_sha256)
    report = json_file(report_path)
    require(report.get("schema") == SCHEMA and report.get("status") == "completed_prefix_capture"
            and report.get("script_sha256") == CAPTURE_SOURCE_SHA
            and report.get("contracts_sha256") == sha(__file__), "Unrecognized frozen capture source or status")
    require(report.get("checkpoint", {}).get("sha256") == CHECKPOINT_SHA and
            report["checkpoint"]["bytes"] == CHECKPOINT_BYTES, "Wrong original checkpoint provenance")
    require(report.get("source_and_inputs_unchanged") is True and report.get("parameter_state_unchanged") is True
            and report.get("CPU_affinity_restored") is True and report.get("environment_restored") is True
            and not report.get("cleanup_errors"),
            "Capture integrity/cleanup gate failed")
    require(report.get("actual_inference_on_RK3588S_only") is True and report.get("actual_ORT_or_compiler_run") is False
            and report.get("full_model_forward_completed") is False and report.get("real_time_validated") is False,
            "Capture execution scope changed")
    pins = {str(report_path): expected_sha256}
    provenance = report["provenance_artifacts"]
    for key, item in provenance.items():
        path = safe_file(directory, item["file"])
        pinned(path, item["sha256"])
        require(path.stat().st_size == item["bytes"], "Provenance artifact file size mismatch")
        pins[str(path)] = item["sha256"]
    require(provenance["image-plan.json"]["sha256"] == PLAN_SHA and
            provenance["original-vggt-source-pins.json"]["sha256"] == MODEL_SOURCE_PINS_SHA
            and provenance["frozen-pair-export.json"]["sha256"] == PAIR_EXPORT_SHA
            and provenance["rectification-calibration.json"]["sha256"] == CALIBRATION_FILE_SHA,
            "Fixed source/preprocessing/plan/pair lineage changed")
    plan = json_file(safe_file(directory, provenance["image-plan.json"]["file"]))
    planned = validate_plan(plan)
    rows = parse_rgb_index(safe_file(directory, provenance["rgb-index.txt"]["file"]).read_text())
    verify_plan_rows(plan, rows)
    closure = json_file(safe_file(directory, provenance["original-vggt-source-pins.json"]["file"]))
    expected_sources = {"third_party/vggt/" + name: digest for name, digest in closure.items()}
    expected_sources.update(RUNTIME_SOURCE_PINS)
    require(report["source_hashes"] == expected_sources, "Captured original source closure mismatch")
    require(report["preprocessing"]["requested_image_size"] == 336 and
            report["preprocessing"]["expected_RGB_tensor_shape"] == RGB_SHAPE and
            report["preprocessing"]["function_sha256"] == PREPROCESS_FUNCTION_SHA,
            "Original v17 preprocessing changed")
    frozen = report["frozen_pair"]
    require(frozen == {"export_sha256": PAIR_EXPORT_SHA, "onnx_sha256": PAIR_ONNX_SHA,
            "original_input_raw_sha256": FROZEN_INPUT_RAW_SHA,
            "original_positions_raw_sha256": FROZEN_POSITIONS_RAW_SHA}, "Wrong fixed original pair")
    identity = report["parameter_identity"]
    require(identity["full_model_parameter_count"] == 1157941492 and
            identity["named_parameter_tensors"] == 1341 and
            identity["all_36_pair_parameter_values_match_frozen_graph"] is True,
            "Original model parameter identity mismatch")
    pair_export = json_file(safe_file(directory, provenance["frozen-pair-export.json"]["file"]))
    expected_bindings = {group: {name: {"shape": item["original"]["shape"],
                                     "dtype": item["original"]["dtype"],
                                     "raw_sha256": item["original"]["sha256"]}
                                for name, item in entries.items()}
                         for group, entries in pair_export["original_parameter_bindings"].items()}
    require(identity["pair_bindings"] == expected_bindings,
            "Captured pair parameter identities differ from frozen original graph")
    samples = report["samples"]
    require(len(samples) == len(planned), "Capture did not finish every planned sample")
    raw_split_sets = {"calibration": set(), "evaluation": set()}
    for sample, original in zip(samples, planned):
        require(sample["sample_id"] == original["sample_id"] and sample["split"] == original["split"],
                "Captured sample split/order differs from pre-inference plan")
        for index, (captured_image, planned_image) in enumerate(zip(sample["images"], original["images"])):
            require(all(captured_image[key] == value for key, value in planned_image.items())
                    and captured_image["ordinal_in_pair"] == index
                    and captured_image["model_role"] == ("first" if index == 0 else "other")
                    and captured_image["timestamp_string"] == rows[planned_image["index"]]["timestamp_string"],
                    "Original image order, source identity or first/other role changed")
        require(len(sample["images"]) == 2 and sample["prepared_RGB"]["shape"] == RGB_SHAPE
                and sample["prepared_RGB"]["dtype"] == "float32"
                and sample["images_tensor_unchanged"] is True and sample["parameter_state_unchanged"] is True
                and sample["original_camera_register_roles_equal"] is True,
                "Original pixel input/model state or special-token roles changed")
        require_call_counts(sample)
        artifacts = sample["artifacts"]
        expected_names = {"input.npy", "positions.npy"}
        if sample["split"] == "evaluation":
            expected_names |= {"frame-reference.npy", "output.npy"}
            require(sample["frame_to_global_view_bitwise_equal"] is True,
                    "Evaluation did not use original contiguous frame/global chain")
        require(set(artifacts) == expected_names, "Unexpected or missing captured tensors")
        for name in sorted(expected_names):
            dtype = "int64" if name == "positions.npy" else "float32"
            shape = POSITIONS_SHAPE if name == "positions.npy" else (OUTPUT_SHAPE if name == "output.npy" else INPUT_SHAPE)
            value = load_array(directory, artifacts[name], shape, dtype)
            if name == "positions.npy":
                require(artifacts[name]["raw_sha256"] == FROZEN_POSITIONS_RAW_SHA,
                        "Captured RoPE positions differ from fixed graph")
            pins[str(safe_file(directory, artifacts[name]["file"]))] = artifacts[name]["sha256"]
            del value
        digest = artifacts["input.npy"]["raw_sha256"]
        require(digest != FROZEN_INPUT_RAW_SHA, "Old evaluation input entered new capture split")
        require(digest not in raw_split_sets[sample["split"]], "Repeated tensor input within split")
        raw_split_sets[sample["split"]].add(digest)
    require(not (raw_split_sets["calibration"] & raw_split_sets["evaluation"]),
            "Calibration tensor input overlaps held-out evaluation")
    return report, samples, pins
