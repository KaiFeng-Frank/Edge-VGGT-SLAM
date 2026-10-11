#!/usr/bin/env python3
"""Read-only audit of saved original prefix data; no model/decoder/SDK execution."""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CAPTURE_SHA = "7a2279927a65e3740da933c1192c335d15f4f1803440ed84a27a2aebc8f02613"
CONTRACTS_SHA = "6ce1f87f1683e8082a4d567718acb853101e5ee6bbaeef1ced02993a8e3133c8"
SOURCE_FREEZE_SHA = "fe1bd10b3986c9a0c2d965f8f0fa59fcfa58883cc892450bc7c43c90ad40f4f2"
INPUT_SHAPE = [2, 437, 1024]
POSITIONS_SHAPE = [2, 437, 2]
OUTPUT_SHAPE = [1, 874, 1024]


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pin(path, expected, pins):
    path = Path(path).resolve()
    require(sha(path) == expected, f"Read-only audit SHA mismatch: {path}")
    pins[str(path)] = expected
    return path


def raw_sha(value):
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def artifact_path(root, item):
    relative = item["file"]
    require(isinstance(relative, str) and relative and not any(c.isspace() for c in relative),
            "Artifact path contains whitespace")
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, "Artifact path escapes dataset")
    resolved = (root / path).resolve(strict=True)
    require(resolved.is_relative_to(root.resolve()) and resolved.is_file(), "Artifact path must be inside dataset")
    return resolved


def read_artifact(root, item, shape, dtype, pins):
    import numpy as np
    path = artifact_path(Path(root), item)
    require(list(item["shape"]) == shape and item["dtype"] == dtype and path.stat().st_size == item["bytes"],
            "Declared artifact shape/dtype/file size changed")
    require(item["bytes"] < 16 * 1024 * 1024, "Artifact exceeds known capture limit")
    pin(path, item["sha256"], pins)
    value = np.load(path, allow_pickle=False)
    require(isinstance(value, np.ndarray) and list(value.shape) == shape and str(value.dtype) == dtype
            and value.flags.c_contiguous and np.isfinite(value).all(),
            "Actual NPY shape/dtype/layout/finiteness changed")
    require(value.nbytes == item["tensor_bytes"] and raw_sha(value) == item["raw_sha256"],
            "Actual NPY raw SHA or tensor bytes changed")
    return value


def array_stats(value):
    import numpy as np
    # Descriptive saved-data statistics only; not a numerical-quality threshold.
    double = value.astype(np.float64)
    return {"min": float(double.min()), "max": float(double.max()), "mean": float(double.mean()),
            "std": float(double.std()), "RMS": float(np.sqrt(np.mean(double * double))),
            "nonzero_elements": int(np.count_nonzero(value)), "elements": value.size}


def expected_counts(split):
    evaluation = split == "evaluation"
    first = [1] + [0] * 23
    return {"dino_entered": [1] * 24, "dino_completed": [1] * 24,
            "frame_entered": first, "frame_completed": first if evaluation else [0] * 24,
            "global_entered": first if evaluation else [0] * 24,
            "global_completed": first if evaluation else [0] * 24,
            "camera_head_entered": 0, "depth_head_entered": 0}


def audit(capture_dir, capture_sha, frozen_pair_dir, original_source_root, runtime_source_root):
    import numpy as np
    pins = {}
    pin(HERE / "capture-v49-pair-prefix-board.py", CAPTURE_SHA, pins)
    helper = pin(HERE / "v49_capture_contracts.py", CONTRACTS_SHA, pins)
    pin(HERE / "source-freeze-r1.json", SOURCE_FREEZE_SHA, pins)
    spec = importlib.util.spec_from_file_location("_v49_saved_capture_audit_contracts", helper)
    c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c)
    capture_dir, frozen_pair_dir = Path(capture_dir).resolve(), Path(frozen_pair_dir).resolve()
    captured, samples, contract_pins = c.load_capture(capture_dir, capture_sha)
    pins.update(contract_pins)
    pair_path = pin(frozen_pair_dir / "export.json", c.PAIR_EXPORT_SHA, pins)
    pair = json.loads(pair_path.read_text())
    old_input = read_artifact(frozen_pair_dir, pair["artifacts"]["input.npy"], INPUT_SHAPE, "float32", pins)
    old_pos = read_artifact(frozen_pair_dir, pair["artifacts"]["positions.npy"], POSITIONS_SHAPE, "int64", pins)
    require(raw_sha(old_input) == c.FROZEN_INPUT_RAW_SHA and raw_sha(old_pos) == c.FROZEN_POSITIONS_RAW_SHA,
            "Original frozen pair input/positions changed")
    special_roles = np.array(old_input[:, :5], copy=True, order="C")
    require(not np.array_equal(special_roles[0], special_roles[1]), "Original first/other special tokens are indistinguishable")
    # Original position_getter uses Cartesian y,x order, then the aggregator
    # adds 1 and prepends exactly five zero positions; this is an arithmetic
    # cross-check of saved data, not a model execution or replacement input.
    grid = np.array([(y + 1, x + 1) for y in range(18) for x in range(24)], dtype=np.int64)
    one_frame_pos = np.concatenate((np.zeros((5, 2), dtype=np.int64), grid), axis=0)
    expected_pos = np.repeat(one_frame_pos[None], 2, axis=0)
    require(np.array_equal(old_pos, expected_pos), "Frozen positions do not match original 18x24 y,x formula")
    source_closure = json.loads((capture_dir / "original-vggt-source-pins.json").read_text())
    require(len(source_closure) == 41, "Original source closure cardinality changed")
    source_root = Path(original_source_root).resolve()
    source_files = {}
    for relative, digest in source_closure.items():
        path = (source_root / relative).resolve()
        require(path.is_relative_to(source_root) and relative.startswith("vggt/"), "Original source path escaped package")
        pin(path, digest, pins)
        source_files[relative] = digest
    runtime_root = Path(runtime_source_root).resolve()
    for relative, digest in c.RUNTIME_SOURCE_PINS.items():
        path = (runtime_root / relative).resolve()
        require(path.is_relative_to(runtime_root), "Runtime source path escaped frozen snapshot")
        pin(path, digest, pins)
    expected_binding = {group: {name: {"shape": item["original"]["shape"], "dtype": item["original"]["dtype"],
                                               "raw_sha256": item["original"]["sha256"]}
                               for name, item in entries.items()}
                        for group, entries in pair["original_parameter_bindings"].items()}
    require(sum(map(len, expected_binding.values())) == 36 and expected_binding == captured["parameter_identity"]["pair_bindings"],
            "Saved 36 pair parameter identities differ from original frozen graph")
    ids, image_sha, image_index, image_timestamp, input_raw = set(), set(), set(), set(), set()
    legacy_hash = {row["sha256"] for row in c.LEGACY_IMAGES}
    audit_rows, split_counts, tensors_read = [], {"calibration": 0, "evaluation": 0}, 2
    patch_raw_by_split = {"calibration": set(), "evaluation": set()}
    for sample in samples:
        sid, split = sample["sample_id"], sample["split"]
        require(sid not in ids and split in split_counts, "Repeated sample identity or invalid split")
        ids.add(sid)
        split_counts[split] += 1
        require(sample["call_counts"] == expected_counts(split), "Independent original-prefix call-count check failed")
        require(all(sample[key] is True for key in ("hooks_removed", "hook_registry_restored", "forward_functions_unchanged",
                    "images_tensor_unchanged", "parameter_state_unchanged", "original_camera_register_roles_equal")),
                "Prefix cleanup or input-integrity receipt failed")
        for ordinal, image in enumerate(sample["images"]):
            stamp = Decimal(image["timestamp_string"])
            require(image["sha256"] not in image_sha and image["sha256"] not in legacy_hash
                    and image["index"] not in image_index and image["index"] not in (0, 8)
                    and stamp not in image_timestamp and stamp == Decimal(Path(image["name"]).stem)
                    and image["ordinal_in_pair"] == ordinal and image["model_role"] == ("first" if ordinal == 0 else "other"),
                    "Independent timestamp/PNG/index/role split exclusion failed")
            image_sha.add(image["sha256"])
            image_index.add(image["index"])
            image_timestamp.add(stamp)
        value = read_artifact(capture_dir, sample["artifacts"]["input.npy"], INPUT_SHAPE, "float32", pins)
        position = read_artifact(capture_dir, sample["artifacts"]["positions.npy"], POSITIONS_SHAPE, "int64", pins)
        tensors_read += 2
        digest = raw_sha(value)
        require(digest not in input_raw and digest != c.FROZEN_INPUT_RAW_SHA, "New split reused another input or old evaluation input")
        input_raw.add(digest)
        require(np.array_equal(position, old_pos) and np.array_equal(position, expected_pos),
                "Saved original positions differ from frozen and arithmetic references")
        require(np.array_equal(value[:, :5], special_roles), "Saved first/other camera/register roles differ from original frozen input")
        for frame in range(2):
            patch_raw_by_split[split].add(raw_sha(np.ascontiguousarray(value[frame, 5:])))
        row = {"sample_id": sid, "split": split, "input_raw_sha256": digest,
               "position_bitwise_equal_to_original_and_formula": True,
               "special_roles_bitwise_equal_to_original": True, "input_stats": array_stats(value),
               "prefix_seconds_with_hooks_from_board": sample["prefix_seconds_including_copy_hooks_and_validation"],
               "artifacts": {name: {key: item[key] for key in ("sha256", "raw_sha256", "shape", "dtype", "bytes", "tensor_bytes")}
                             for name, item in sample["artifacts"].items()}}
        if split == "evaluation":
            require(sample["frame_to_global_view_bitwise_equal"] is True, "Original contiguous frame/global link failed")
            frame_value = read_artifact(capture_dir, sample["artifacts"]["frame-reference.npy"], INPUT_SHAPE, "float32", pins)
            output = read_artifact(capture_dir, sample["artifacts"]["output.npy"], OUTPUT_SHAPE, "float32", pins)
            tensors_read += 2
            row.update(frame_reference_stats=array_stats(frame_value), output_stats=array_stats(output),
                       original_frame_global_bridge_checked_by_board_capture=True)
            del frame_value, output
        audit_rows.append(row)
        del value, position
    require(split_counts == {"calibration": 8, "evaluation": 2} and len(image_sha) == len(image_index) == len(image_timestamp) == 20,
            "Independent capture split cardinality failed")
    require(captured["CPU_affinity_restored"] is True and captured["environment_restored"] is True
            and captured["source_and_inputs_unchanged"] is True and captured["parameter_state_unchanged"] is True,
            "Final process/source/parameter restoration failed")
    # Re-pin every file actually read. No capture/source artifact is written.
    for path, digest in pins.items():
        require(sha(path) == digest, f"A file changed during read-only audit: {path}")
    return {"schema": "v49_read_only_original_prefix_saved_data_audit", "passed": True,
            "script_sha256": sha(__file__), "capture_manifest_sha256": capture_sha,
            "capture_script_sha256": CAPTURE_SHA, "capture_contracts_sha256": CONTRACTS_SHA,
            "saved_arrays_read": tensors_read, "capture_arrays_read": tensors_read - 2,
            "source_model_files_physically_read": len(source_files), "runtime_files_physically_read": len(c.RUNTIME_SOURCE_PINS),
            "pair_parameter_metadata_bound": 36, "split_counts": split_counts,
            "image_index_timestamp_png_sha_mutually_disjoint": True,
            "original_special_roles_checked_independently_from_frozen_input": True,
            "original_rope_checked_independently_from_frozen_positions_and_yx_formula": True,
            "unique_patch_feature_hashes_per_split": {k: len(v) for k, v in patch_raw_by_split.items()},
            "patch_feature_hash_overlap_between_splits": len(patch_raw_by_split["calibration"] & patch_raw_by_split["evaluation"]),
            "board_full_checkpoint_SHA_before_capture": captured["checkpoint"]["sha256"],
            "host_checkpoint_or_source_PNG_rehashed": False,
            "actual_model_or_ORT_or_SDK_execution_by_audit": False,
            "actual_calibration_or_compilation_by_audit": False,
            "real_image_decoding_by_audit": False,
            "board_capture_wall_seconds_including_hash_load": captured["total_seconds_including_weight_hash_load"],
            "original_model_quality_or_realtime_proven_by_audit": False,
            "all_read_file_bytes_unchanged": True, "read_file_SHA_pins": pins, "samples": audit_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--capture-sha256", required=True)
    parser.add_argument("--frozen-pair-dir", type=Path, required=True)
    parser.add_argument("--original-source-root", type=Path, required=True,
                        help="Pinned original third_party/vggt package root; source reads only")
    parser.add_argument("--runtime-source-root", type=Path, required=True,
                        help="Current frozen public snapshot root; source reads only")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists() and not output.is_relative_to(args.capture_dir.resolve())
            and not output.is_relative_to(args.frozen_pair_dir.resolve()), "Audit must write a new receipt outside saved data")
    report = audit(args.capture_dir, args.capture_sha256, args.frozen_pair_dir,
                   args.original_source_root, args.runtime_source_root)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": True, "capture_arrays_read": report["capture_arrays_read"],
                      "original_source_files": report["source_model_files_physically_read"],
                      "receipt_sha256": sha(output)}))


if __name__ == "__main__":
    main()
