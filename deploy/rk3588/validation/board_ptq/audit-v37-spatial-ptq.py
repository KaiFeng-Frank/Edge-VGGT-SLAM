#!/usr/bin/env python3
"""Review saved PTQ JSON/text and source AST only; never read calibration NPY.

No compiler, model, SDK, ONNX, NumPy or device module is imported or executed.
SPDX-License-Identifier: MIT
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import shlex

SOURCE_SHA = "0fc013d5b6ed45cb97d592e8cd115304d971a000e2ab5353314413f56d41a3fa"
PARENT_SHA = "0f66e0ead40ab6eff546a47367ee93ec044b2b84e064eb8e9876352e0ca05fb0"
SUCCESS_SHA = "4d660a6d48d296fda930adee7e192571263df717431a208daa63ae43e5f9427d"
FAILURE_SHA = "9dae9f1318e898bb19c932a2a7d0667f256d69a291fde765773cf533a82beb57"
ORIGINAL_SHA = "680373c1e605c10f660781988e74b65e07ea8d31a550fcd93022487978bac4a8"
CALIBRATION_SHA = "61566b1769ab3831a81935e37ea6fcdc4f603a21a988a64ae0aa9e7f7441f0dd"
RAW_DATASET_SHA = "5c4c4712a2241fd0dd3cf83de2f454386f22bcdff4d1695077a8cd545b018372"
RESOLVED_DATASET_SHA = "00b58af6d303a441a878c6252966a0de3d745e33d20370e26af6b1f6c63bfc1a"
UNCHANGED = ("require", "sha", "board_guard", "load_pinned",
             "validate_calibration_arrays", "prepare_dataset", "perform_ptq")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def source_ast(path):
    return ast.parse(Path(path).read_text())


def ast_dump(node):
    return ast.dump(node, include_attributes=False)


def compare_helpers(parent, candidate):
    old = {n.name: n for n in parent.body if isinstance(n, ast.FunctionDef)}
    new = {n.name: n for n in candidate.body if isinstance(n, ast.FunctionDef)}
    if old.keys() != new.keys() or set(old) != set(UNCHANGED) | {"main"}:
        raise ValueError("Unexpected top-level function set")
    result = {}
    for name in UNCHANGED:
        a, b = ast_dump(old[name]), ast_dump(new[name])
        if a != b:
            raise ValueError("Frozen helper AST changed: " + name)
        result[name] = digest(b.encode())
    config = lambda tree: next(n for n in tree.body if isinstance(n, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == "CONFIGURATION" for t in n.targets))
    if ast_dump(config(parent)) != ast_dump(config(candidate)):
        raise ValueError("PTQ configuration AST changed")
    if ast_dump(old["main"]) == ast_dump(new["main"]):
        raise ValueError("Expected explicit spatial-diagnostic main change")
    return result


def launch_argument(path, name):
    fields = shlex.split(Path(path).read_text())
    if fields.count(name) != 1:
        raise ValueError("Exactly one launcher argument required: " + name)
    return fields[fields.index(name) + 1]


def review_query(query, success, check):
    check("query.status", query["schema"] == "v37_DPVO_spatial_PTQ_native_IO_query" and
          query["status"] == "queried_spatial_native_IO_parent_rejection_retained_not_inference_validated")
    check("query.exact_compilation_binding", query["compile_receipt_sha256"] == SUCCESS_SHA and
          query["model_sha256"] == success["output"]["sha256"] and
          query["compile_binding"]["configuration"] == success["configuration"] and
          query["compile_binding"]["output"] == success["output"] and
          query["compile_binding"]["ancestor"] == success["ancestor"] and
          query["compile_binding"]["calibration_dataset_sha256"] == RESOLVED_DATASET_SHA)
    check("query.original_rejection_retained", query["diagnostic_allow_fp32_rejected"] is True and
          query["historical_parent_FP32_gate_passed"] is False and
          query["historical_parent_rejection_retained"] is True and query["candidate_accepted"] is False)
    check("query.init_destroy_success", query["actual_board_execution"] is True and
          query["initialized"] is True and query["context_destroyed"] is True and
          query["init_return_code"] == query["destroy_return_code"] == 0 and query["cleanup_errors"] == [])
    check("query.zero_runs_not_inference", query["sdk_run_calls"] == 0 and
          query["model_inference_executed"] is False and query["NPU_used"] is False and
          query["NPU_operator_coverage_validated"] is False and query["precision_gate_measured"] is False and
          query["performance_measured"] is False and query["production_changed"] is False and
          query["bound_functions"] == ["rknn_init", "rknn_query", "rknn_destroy"])
    check("query.source_map_unchanged", query["sources_and_inputs_unchanged"] is True and
          query["source_pins"] == query["source_hashes_after"])
    check("query.two_inputs_outputs", query["io_count"] == {"inputs": 2, "outputs": 2} and
          len(query["input_attrs"]) == len(query["output_attrs"]) == 2)
    contracts = [("normalized_bgr", [1, 464, 608, 3], 2, 1, 1),
                 ("patch_indices", [96], 8, 3, 8),
                 ("fmap", [1, 128, 116, 152], 2, 0, 1),
                 ("context", [1, 96, 384], 2, 3, 1)]
    for attr, (name, shape, dtype, layout, itemsize) in zip(query["input_attrs"] + query["output_attrs"], contracts):
        check("query.native_attr." + name, attr["name"] == name and attr["shape"] == shape and
              attr["type"] == dtype and attr["format"] == layout and attr["n_elems"] == math.prod(shape) and
              attr["size"] == math.prod(shape) * itemsize and attr["size_with_stride"] >= attr["size"] and
              attr["quantization_type"] == 2 and math.isfinite(attr["scale"]) and attr["scale"] > 0)
    indices = query["input_attrs"][1]
    check("query.indices_not_quantized_to_float", indices["scale"] == 1 and indices["zero_point"] == 0 and
          query["validated_io"]["patch_indices"]["integer_index_cast_performed"] is False)
    return {"recorded_board_init_query_destroy": True, "sdk_run_calls": 0,
            "compiled_model_sha256": query["model_sha256"], "input_attrs": query["input_attrs"],
            "output_attrs": query["output_attrs"], "recorded_model_memory_bytes": query["model_memory_bytes"],
            "NPU_internal_placement_or_precision_verified": False,
            "limitation": "Saved native descriptor/init/query/destroy metadata only; no inference or SDK call by this auditor."}


def audit(args):
    checks, bindings = [], {}

    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})
        if not condition:
            raise ValueError("Failed metadata/source check: " + name)

    def read(label, path, expected=None, text=False):
        path = Path(path)
        if path.suffix == ".npy":
            raise ValueError("Calibration pixels/indices must not be read")
        raw = path.read_bytes()
        actual = digest(raw)
        if expected:
            check(label + ".file_sha256", actual == expected)
        bindings[label] = {"file": path.name, "sha256": actual, "bytes": len(raw)}
        return raw.decode() if text else json.loads(raw)

    source = read("spatial_compiler", args.source, SOURCE_SHA, text=True)
    parent = read("original_compiler", args.parent_source, PARENT_SHA, text=True)
    unchanged = compare_helpers(ast.parse(parent), ast.parse(source))
    for name in unchanged:
        check("unchanged_helper_AST." + name, True)
    check("configuration_AST_unchanged", True)
    derivation = read("source_derivation", args.derivation)
    check("derivation_matches_actual_sources", derivation["parent_sha256"] == PARENT_SHA
          and derivation["source_sha256"] == SOURCE_SHA)
    check("derivation_helper_set", derivation["unchanged_function_bodies"] == list(UNCHANGED)
          and derivation["changed_functions"] == ["main"] and derivation["configuration_unchanged"] is True)
    success = read("spatial_r2", args.success_dir / "compile.json", SUCCESS_SHA)
    failure = read("spatial_r1", args.failure_dir / "compile.json", FAILURE_SHA)
    original = read("original_PTQ", args.original_compile, ORIGINAL_SHA)
    generator = read("calibration_generation", args.calibration_dir / "calibration.json", CALIBRATION_SHA)
    dataset = read("raw_dataset", args.calibration_dir / "dataset.txt", RAW_DATASET_SHA, text=True)
    for name, report in (("spatial_r2", success), ("spatial_r1", failure), ("original_PTQ", original)):
        check(name + ".source_map_before_after", report["sources_and_inputs_unchanged"] is True
              and report["source_pins"] == report["source_hashes_after"])
        check(name + ".cleanup_empty", report["cleanup_errors"] == [])
        check(name + ".diagnostic_not_adopted", report["candidate_accepted"] is False
              and report["production_changed"] is False and report["host_calibration_executed"] is False
              and report["deployment_inference_executed"] is False
              and report["INT8_operator_coverage_validated"] is False)
    check("same_spatial_source_r1_r2", failure["script_sha256"] == success["script_sha256"] == SOURCE_SHA)
    check("success_exact_status", success["schema"] == "v37_board_only_DPVO_spatial_encoder_PTQ"
          and success["status"] == "compiled_board_spatial_PTQ_parent_rejection_retained_not_runtime_validated")
    check("same_configuration", success["configuration"] == original["configuration"] == failure["configuration"])
    check("fixed_original_gate_not_measured", success["original_FP32_gate"] ==
          {"rtol": 1e-4, "atol": 1e-5, "measured": False})
    for name, report in (("spatial_r2", success), ("spatial_r1", failure)):
        ancestor = report["ancestor"]
        check(name + ".parent_rejection_retained", report["diagnostic_allow_fp32_rejected"] is True
              and report["historical_parent_FP32_gate_passed"] is False
              and report["historical_parent_rejection_retained"] is True
              and ancestor["method"] == "spatial" and ancestor["parent_CPU_strict_gate_passed"] is False
              and ancestor["suffix_CPU_bitwise_gate_passed"] is True
              and ancestor["spatial_parent_rejection_retained"] is True)
        check(name + ".ancestor_chain", ancestor["parent_onnx_sha256"] ==
              "c3c4048c6d8792b09f0d6cbec8228f99d59c60d7503456eea2e369d7b8dbd826"
              and report["input"]["sha256"] == ancestor["sparse_onnx_sha256"] ==
              "6c3b4de4d151813dc172d711b6a760ad8bdc17120f2f3faa76dbf32bd0c451ab"
              and ancestor["reference_receipt_sha256"] == original["ancestor"]["reference_receipt_sha256"])
    check("success_operations", [v["name"] for v in success["operations"]] ==
          ["config", "load_onnx", "board_CPU_calibration_and_build", "export_rknn"]
          and all(type(v["code"]) is int and v["code"] == 0 and math.isfinite(v["seconds"])
                  and v["seconds"] >= 0 for v in success["operations"]))
    check("success_released", success["released"] is True)
    check("successful_artifact_metadata", success["output"] == {"file": "block-int8.rknn", "bytes": 2250354,
          "sha256": "84f4894400c996ac026fe97865d4be7e9e9265164d6859a2ebc97fe1bcfbdcb1"})
    check("successful_environment", success["environment"]["status"] == "ready_for_board_PTQ_diagnostic"
          and success["environment"]["problems"] == [] and success["environment"]["machine"] == "aarch64")
    check("all_calibration_metadata_equal_original", success["calibration"] == original["calibration"])
    rows = success["calibration"]["rows"]
    lines = dataset.splitlines()
    check("32_rows_and_generator_samples", success["calibration"]["samples"] == len(rows) == len(lines)
          == len(generator["samples"]) == 32)
    check("generator_raw_dataset_identity", generator["dataset"] ==
          {"file": "dataset.txt", "bytes": len(dataset.encode()), "sha256": RAW_DATASET_SHA})
    check("generator_inputs_recorded_unchanged", generator["sources_and_inputs_unchanged"] is True
          and generator["source_pins"] == generator["source_hashes_after"] and generator["cleanup_errors"] == [])
    calibration_paths = []
    for index, row in enumerate(rows):
        prefix = "calibration.row_" + str(index)
        record = generator["samples"][index]
        check(prefix + ".index", row["index"] == record["calibration_index"] == index
              and record["source_index"] == 256 + 16 * index)
        check(prefix + ".shape_dtype_metadata", row["image_dtype"] == "float32"
              and row["image_shape"] == [1, 3, 464, 608] and row["indices_dtype"] == "int64"
              and row["indices_shape"] == [96] and row["index_order_preserved"] is True
              and row["index_cast_performed"] is False)
        check(prefix + ".range_metadata", -.5 <= row["image_min"] <= row["image_max"] <= 1.5
              and 153 <= row["index_min"] <= row["index_max"] <= 17478)
        names = lines[index].split()
        check(prefix + ".two_ordered_files", len(row["input_files"]) == len(names) == len(record["input_files"]) == 2)
        for position, (entry, gen, name) in enumerate(zip(row["input_files"], record["input_files"], names)):
            label = prefix + ".input_" + str(position)
            check(label + ".file_metadata", Path(entry["file"]).name == gen["file"] == name
                  and entry["sha256"] == gen["sha256"] and entry["bytes"] == gen["bytes"])
            check(label + ".both_PTQ_source_pins", success["source_pins"].get(entry["file"]) ==
                  original["source_pins"].get(entry["file"]) == entry["sha256"])
            check(label + ".both_PTQ_after_pins", success["source_hashes_after"].get(entry["file"]) ==
                  original["source_hashes_after"].get(entry["file"]) == entry["sha256"])
            calibration_paths.append(entry["file"])
    check("64_unique_file_pins", len(set(calibration_paths)) == 64 and
          {p for p in success["source_pins"] if p.endswith(".npy")} == set(calibration_paths))
    resolved_text = "\n".join(" ".join(entry["file"] for entry in r["input_files"]) for r in rows) + "\n"
    check("resolved_text_SHA_from_recorded_paths", digest(resolved_text.encode()) ==
          success["calibration"]["dataset_sha256"] == RESOLVED_DATASET_SHA)
    raw_pins = {p: value for p, value in success["source_pins"].items() if p.endswith("/dataset.txt")}
    resolved_pins = {p: value for p, value in success["source_pins"].items() if p.endswith("/dataset-resolved.txt")}
    check("raw_and_resolved_distinct_source_pins", list(raw_pins.values()) == [RAW_DATASET_SHA]
          and list(resolved_pins.values()) == [RESOLVED_DATASET_SHA] and RAW_DATASET_SHA != RESOLVED_DATASET_SHA)
    for name, directory in (("r1", args.failure_dir), ("r2", args.success_dir)):
        read(name + "_launcher", directory / "launch-board.sh", text=True)
    check("r1_wrong_resolved_SHA_as_raw_argument", launch_argument(args.failure_dir / "launch-board.sh",
          "--dataset-sha256") == RESOLVED_DATASET_SHA)
    check("r2_correct_raw_SHA_argument", launch_argument(args.success_dir / "launch-board.sh",
          "--dataset-sha256") == RAW_DATASET_SHA)
    check("r1_failed_before_toolkit", failure["status"] == "failed" and failure["error"] ==
          "ValueError('Calibration dataset list SHA mismatch')" and failure["operations"] == []
          and failure["released"] is False and "output" not in failure and "calibration" not in failure)
    query_review = None
    if args.query_dir is not None:
        query = read("spatial_native_IO_query", args.query_dir / "query.json",
                     "7420da38f9f3867a1e9e39f164dbb09fe1608175c5a0c9d2c5f4b96ed491c566")
        read("spatial_query_source", args.query_source,
             "c61a377ede3d33b5a2412b1c7f526ea75deff2f18ed185f0f937792e41b1aca2", text=True)
        check("query.own_source_pin", query["script_sha256"] == bindings["spatial_query_source"]["sha256"] and
              query["script_sha256"] in query["source_pins"].values())
        query_review = review_query(query, success, check)
    return {"schema": "v37_spatial_PTQ_independent_source_and_metadata_review",
            "status": "passed_source_metadata_only_parent_rejection_retained", "candidate_accepted": False,
            "audit_script_sha256": digest(Path(__file__).read_bytes()), "bindings": bindings,
            "AST_unchanged_helpers": unchanged, "unchanged_helper_count": len(unchanged),
            "calibration_rows": 32, "calibration_input_file_pins": 64,
            "calibration_NPY_read": False, "pixel_values_independently_verified": False,
            "preprocessing_independently_reexecuted": False, "model_or_SDK_executed_by_audit": False,
            "compiled_RKNN_independently_host_rehashed": False, "all_board_source_pins_host_rehashed": False,
            "scope": "JSON/source AST/raw dataset text only. Calibration NPY, parent ONNX, compiled RKNN and board deployment were not executed or reread; source/input before-after equality is saved board evidence.",
            "parent_FP32_gate_passed": False, "historical_parent_rejection_retained": True,
            "success": {"board_CPU_calibration_build_seconds": success["operations"][2]["seconds"],
                        "compiled_artifact_metadata": success["output"], "recorded_peak_rss_mib": success["peak_rss_mib"],
                        "recorded_release": True, "cleanup_errors": [], "deployment_inference_validated": False,
                        "INT8_operator_coverage_validated": False, "trajectory_or_realtime_validated": False},
            "retained_r1_failure": {"classification": "launcher_input_identity_failure_before_RKNN_import_not_algorithm_failure",
                                    "incorrect_dataset_argument_sha256": RESOLVED_DATASET_SHA,
                                    "correct_raw_dataset_sha256": RAW_DATASET_SHA,
                                    "operations_executed": 0, "same_compiler_source_as_r2": True},
            "native_IO_zero_run_query": query_review,
            "checks_passed": len(checks), "failed_checks": 0, "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "parent-source", "derivation", "success-dir", "failure-dir",
                 "original-compile", "calibration-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--query-dir", type=Path)
    parser.add_argument("--query-source", type=Path)
    args = parser.parse_args()
    if (args.query_dir is None) != (args.query_source is None):
        raise ValueError("Query directory and source must be supplied together")
    if args.output.exists():
        raise ValueError("New output required; preserve prior review")
    result = audit(args)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "checks_passed", "failed_checks", "calibration_NPY_read")}))


if __name__ == "__main__":
    main()
