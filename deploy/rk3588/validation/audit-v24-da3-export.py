#!/usr/bin/env python3
"""NumPy-only saved-reference/ORT review; no model construction or execution."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
R2_SHA = "a4cc8bf99dfea17e054c9838c55daeaa6b8cf6c686c245bc78af5166dfe79c60"
R3_SHA = "dbff9f1a2562e65f3584795636c0820959334ea838a1fde45cce47fcdc50bd2f"
PARENT_SHA = "4e503b58f4072d5bc819abbe35e24b06535f6d4fab842b34f418684288d13b13"
COMMON_SHA = "3c54586b77af2b8f07d32f886f19dbaee2a57b0d587be84513f8e5bea54cb4d4"
CPU_R2_CAPTURE_SHA = "6f22341da390def7fc54584eb41500b7367bf1b43249a7fbdc7b6e4b866855d3"
KEYS = {"depth": "depth", "conf": "depth_conf", "pose": "pose_enc",
        "extrinsic": "extrinsic", "intrinsic": "intrinsic"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_metadata(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype), "bytes": value.nbytes,
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}


def compare(actual, expected):
    if actual.shape != expected.shape or actual.dtype != np.float32 or expected.dtype != np.float32:
        raise ValueError("Fixed reference shape/dtype mismatch")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Nonfinite saved values")
    error = actual.astype(np.float64) - expected.astype(np.float64)
    norm = np.linalg.norm(expected.astype(np.float64).ravel())
    close = np.isclose(actual, expected, rtol=1e-4, atol=1e-5)
    return {"max_abs": float(np.max(np.abs(error))), "rms": float(np.sqrt(np.mean(error**2))),
            "relative_l2": float(np.linalg.norm(error.ravel()) / norm) if norm else None,
            "array_equal": bool(np.array_equal(actual, expected)), "finite": True,
            "element_count": int(close.size), "failed_elements": int(np.count_nonzero(~close)),
            "elementwise_gate_passed": bool(close.all()), "rtol": 1e-4, "atol": 1e-5}


def statistics_equal(actual, declared):
    if set(actual) != set(declared):
        return False
    for key, value in actual.items():
        other = declared[key]
        if key in ("max_abs", "rms", "relative_l2") and value is not None and other is not None:
            # Float64 reductions may round differently across ARM/x86 BLAS.
            # This compares reported statistics, not the fixed per-element gate.
            if not math.isclose(value, other, rel_tol=1e-12, abs_tol=1e-15):
                return False
        elif value != other:
            return False
    return True


def artifact(directory, exported, name):
    entry = exported["artifacts"][name]
    path = directory / name
    if Path(name).name != name or sha(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
        raise ValueError("Saved artifact SHA/size mismatch: " + name)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--cpu-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--ort-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"schema": "da3_fixed_grid_export_saved_array_review_v24.1", "status": "running",
              "auditor_sha256": sha(__file__), "checks": [], "pairs": [],
              "host_model_constructed": False, "host_model_forward_executed": False,
              "host_ORT_inference_executed": False, "board_jobs": False,
              "NPU_execution_verified": False, "GT_quality_or_realtime_verified": False}

    def check(value, name):
        report["checks"].append({"name": name, "passed": bool(value)})
        if not value:
            raise ValueError(name)

    try:
        cpu = json.loads((args.cpu_dir / "capture.json").read_text())
        exported = json.loads((args.export_dir / "export.json").read_text())
        report["CPU_capture_sha256"] = sha(args.cpu_dir / "capture.json")
        report["export_receipt_sha256"] = sha(args.export_dir / "export.json")
        check(report["CPU_capture_sha256"] == CPU_R2_CAPTURE_SHA,
              "exact_measured_CPU_r2_capture_SHA256")
        check(cpu["status"] == "completed_fixed_input_diagnostic" and
              exported["status"] == "exported_not_runtime_validated", "complete_CPU_and_export_receipts")
        check(sha(args.source_package / "export-v24-da3-board-r3.py") == exported["script_sha256"] == R3_SHA and
              sha(args.source_package / "export-v24-da3-board-r2.py") == R2_SHA and
              sha(args.source_package / "export-v24-da3-board.py") == PARENT_SHA and
              sha(args.source_package / "da3_probe_common.py") == COMMON_SHA,
              "exact_R3_R2_parent_and_common_source_SHA256")
        check(exported["parent_implementation"]["sha256"] == PARENT_SHA and
              exported["parent_position_implementation"]["sha256"] == R2_SHA and
              exported["position_getter_restored"] and exported["affinity_restored"] and
              exported["DPT_position_embedding_restored"] and
              exported["sources_and_inputs_unchanged"] and exported["cleanup_errors"] == [] and
              exported["original_weights_unchanged"], "fixed_getter_restore_weights_and_source_cleanup")
        report["source_sha256"] = {"export-v24-da3-board-r3.py": R3_SHA,
                                   "export-v24-da3-board-r2.py": R2_SHA,
                                   "export-v24-da3-board.py": PARENT_SHA,
                                   "da3_probe_common.py": COMMON_SHA}
        yy, xx = np.meshgrid(np.arange(18, dtype=np.int64), np.arange(24, dtype=np.int64), indexing="ij")
        grid = np.tile(np.stack((yy, xx), axis=-1).reshape(1, 432, 2), (2, 1, 1))
        grid_metadata = array_metadata(grid)
        fixed = exported["fixed_position_grid"]
        check({key: fixed[key] for key in grid_metadata} == grid_metadata and
              fixed["batch"] == 2 and fixed["patch_height"] == 18 and fixed["patch_width"] == 24 and
              fixed["y_x_row_major_verified"] and fixed["exactly_equal_original_getter"],
              "independent_integer_y_x_grid_shape_order_SHA256")
        report["fixed_position_grid"] = grid_metadata
        embeddings = exported["fixed_DPT_position_embeddings"]
        expected_keys = [(48, 18, 24), (96, 18, 24), (192, 18, 24),
                         (384, 18, 24), (32, 252, 336), (32, 144, 192)]
        check([(pe["channels"], pe["patch_height"], pe["patch_width"]) for pe in embeddings] == expected_keys,
              "six_fixed_DPT_position_cache_keys")
        for i, pe in enumerate(embeddings):
            channels, height, width = expected_keys[i]
            constant = pe["constant"]
            check(pe["batch"] == 2 and pe["original_width"] == 336 and pe["original_height"] == 252 and
                  pe["ratio"] == 0.1 and pe["dtype"] == "torch.float32" and pe["device"] == "cpu" and
                  constant["shape"] == [1, channels, height, width] and constant["dtype"] == "float32" and
                  constant["bytes"] == channels*height*width*4 and
                  len(constant["sha256"]) == 64 and
                  pe["original_expanded_shape"] == [2, channels, height, width],
                  "fixed_DPT_position_constant_metadata_" + str(i))
        check(exported["fixed_DPT_position_calls"] == 12, "six_DPT_positions_used_in_wrapper_and_trace")
        report["DPT_position_constants"] = embeddings
        report["DPT_position_arithmetic_scope"] = (
            "Static source review: original official FP32 UV/sincos arithmetic, then ratio/permutation/addition; "
            "singleton broadcasting replaces expand. No host model execution or independent Torch PE calculation.")
        check(exported["official_model_provenance"]["parameters"] == cpu["load_provenance"]["parameters"] and
              exported["official_model_provenance"]["checkpoint"] == cpu["load_provenance"]["checkpoint"] and
              exported["official_model_provenance"]["source_hashes"] == cpu["load_provenance"]["source_hashes"],
              "same_complete_original_weight_and_official_source_binding")
        check(len(exported["pairs"]) == 3 and [x["index"] for x in exported["pairs"]] == [0, 1, 2],
              "complete_three_pair_order")
        for pair in exported["pairs"]:
            index = pair["index"]
            old = cpu["records"][index]
            check(pair["rgb_file_sha256"] == cpu["selected_inputs"][index]["sha256"],
                  "original_RGB_pair_identity_" + str(index))
            input_path = artifact(args.export_dir, exported, pair["input_artifact"])
            new_input = np.load(input_path, allow_pickle=False)
            old_input_path = args.cpu_dir / old["normalized_input_tensor"]["file"]
            check(sha(old_input_path) == old["normalized_input_tensor"]["file_sha256"],
                  "original_CPU_normalized_file_SHA256_pair_" + str(index))
            old_input = np.load(old_input_path, allow_pickle=False)
            check(new_input.shape == (2, 3, 252, 336) and new_input.dtype == np.float32 and
                  array_metadata(new_input) == array_metadata(old_input[0]) and
                  array_metadata(new_input) == exported["artifacts"][pair["input_artifact"]]["array"],
                  "normalized_pixels_exact_remove_B1_only_pair_" + str(index))
            path = artifact(args.export_dir, exported, pair["reference_artifact"])
            entry = exported["artifacts"][pair["reference_artifact"]]
            original_path = args.cpu_dir / old["artifact"]["file"]
            check(sha(original_path) == old["artifact"]["sha256"] and
                  original_path.stat().st_size == old["artifact"]["bytes"],
                  "original_CPU_reference_file_SHA256_pair_" + str(index))
            with np.load(path, allow_pickle=False) as saved, np.load(original_path, allow_pickle=False) as original:
                check(set(saved.files) == set(KEYS), "five_original_reference_arrays_pair_" + str(index))
                results = {}
                for name, canonical in KEYS.items():
                    actual = saved[name]
                    expected = original[canonical][0]
                    if name == "depth":
                        expected = expected[..., 0]
                    check(array_metadata(actual) == entry["arrays"][name],
                          "reference_raw_metadata_pair_" + str(index) + "_" + name)
                    result = compare(actual, expected)
                    check(result["array_equal"] and array_metadata(actual) == array_metadata(expected),
                          "CPU_reference_byte_exact_pair_" + str(index) + "_" + name)
                    results[name] = result
            report["pairs"].append({"pair_index": index, "CPU_reference_comparisons": results})
        check(all(value["elementwise_gate_passed"] for value in exported["wrapper_CPU_comparisons"].values()),
              "fixed_wrapper_original_CPU_reference_gate")
        check(sha(args.export_dir / exported["onnx"]["file"]) == exported["onnx"]["sha256"],
              "exported_ONNX_SHA256")
        report["onnx"] = exported["onnx"]
        if args.ort_dir is not None:
            ort_path = args.ort_dir / "ort.json"
            ort = json.loads(ort_path.read_text())
            report["ORT_receipt_sha256"] = sha(ort_path)
            check(ort["status"] == "completed" and ort["CPUExecutionProvider_only"] and
                  ort["exact_graph_reference_gate_passed"] and
                  ort["script_sha256"] == R3_SHA and
                  ort["export_sha256"] == report["export_receipt_sha256"] and
                  ort["onnx"] == exported["onnx"] and
                  [pair["index"] for pair in ort["pairs"]] == [0, 1, 2],
                  "actual_board_ORT_three_pairs_same_export_model")
            check(ort["affinity_restored"] and ort["sources_and_inputs_unchanged"] and
                  ort["cleanup_errors"] == [], "actual_board_ORT_cleanup")
            all_gate = True
            for pair in ort["pairs"]:
                index = pair["index"]
                name = f"actual-pair-{index}.npz"
                actual_path = artifact(args.ort_dir, ort, name)
                expected_path = artifact(args.export_dir, exported, exported["pairs"][index]["reference_artifact"])
                checks = {}
                with np.load(actual_path, allow_pickle=False) as saved, np.load(expected_path, allow_pickle=False) as expected:
                    check(set(saved.files) == set(KEYS), "actual_ORT_five_arrays_pair_" + str(index))
                    for key in KEYS:
                        check(array_metadata(saved[key]) == ort["artifacts"][name]["arrays"][key],
                              "actual_ORT_raw_metadata_pair_" + str(index) + "_" + key)
                        checks[key] = compare(saved[key], expected[key])
                        declared = pair["comparisons"][{"depth": "depth", "conf": "depth_conf", "pose": "pose_enc"}[key]] if key in ("depth", "conf", "pose") else pair["codec_comparisons"][key]
                        check(statistics_equal(checks[key], declared), "independent_ORT_error_statistics_pair_" + str(index) + "_" + key)
                        all_gate = all_gate and checks[key]["elementwise_gate_passed"]
                check(pair["ORT_forward_seconds"] > 0 and pair["official_codec_seconds"] >= 0,
                      "actual_ORT_positive_timing_pair_" + str(index))
                report["pairs"][index]["ORT_vs_official_reference"] = checks
            check(all_gate, "all_saved_ORT_outputs_fixed_strict_numerical_gate")
            report["actual_board_ORT_gate_verified"] = True
        else:
            report["actual_board_ORT_gate_verified"] = False
        report.update(status="passed", check_count=len(report["checks"]),
                      all_15_original_reference_arrays_byte_exact_CPU_r2=True,
                      outputs_semantics="official DA3 depth/conf/raw pose; CPU codec w2c/K; no VGGT/GT/trajectory claim")
    except Exception as error:
        report.update(status="failed", error=type(error).__name__ + ": " + str(error), check_count=len(report["checks"]))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "checks": report["check_count"],
                      "receipt_sha256": sha(args.output), "error": report.get("error")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
