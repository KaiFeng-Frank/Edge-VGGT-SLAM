#!/usr/bin/env python3
"""Board-only spatial PTQ native IO query; ZERO inference calls.

MIT project helper. DPVO (Princeton Vision & Learning Lab) is MIT licensed.
Native IO quantization metadata is not proof of internal INT8 operator coverage.
Only rknn_init, rknn_query and rknn_destroy are bound by this entrypoint.
The spatial parent's historical FP32 rejection is mandatory and retained.
An explicit diagnostic flag is required before creating the output directory.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback

sys.dont_write_bytecode = True

BASE = Path("/mnt/vggt-data/vggt-slam2")
ABI_SHA = "37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608"
HEADER_SHA = "c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a"
LIBRARY_SHA = "d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8"
COMPILER_SHA = "0fc013d5b6ed45cb97d592e8cd115304d971a000e2ab5353314413f56d41a3fa"
PARENT_QUERY_SHA = "871f67c5a37ff23b703d173bb09234504a58fb16b8ace65e57991afe6f7e3d90"
PARENT_PTQ_SHA = "0f66e0ead40ab6eff546a47367ee93ec044b2b84e064eb8e9876352e0ca05fb0"
GRAPH_SHA = "6c3b4de4d151813dc172d711b6a760ad8bdc17120f2f3faa76dbf32bd0c451ab"
RECIPE_SHA = "005667f44cf98447d2b622882d7d40cbcce8a545985ddf2104eaa62740273ce2"
PARENT_GRAPH_SHA = "c3c4048c6d8792b09f0d6cbec8228f99d59c60d7503456eea2e369d7b8dbd826"
CONFIGURATION = dict(target_platform="rk3588", float_dtype="float16", optimization_level=3,
    quantized_dtype="w8a8", quantized_algorithm="normal", quantized_method="channel",
    mean_values=None, std_values=None, quant_img_RGB2BGR=False, do_quantization=True, auto_hybrid=False)
DTYPES = {0: ("float32", 4), 1: ("float16", 2), 2: ("int8", 1), 3: ("uint8", 1), 8: ("int64", 8)}
FORMATS = {0: "NCHW", 1: "NHWC", 3: "UNDEFINED"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def require_sha(path, expected):
    require(isinstance(expected, str) and len(expected) == 64 and
            all(c in "0123456789abcdef" for c in expected), "Invalid expected SHA256")
    actual = sha(path)
    require(actual == expected, "SHA256 mismatch: " + str(path))
    return actual


def board_guard(machine=None, system=None, compatible=None):
    machine = platform.machine() if machine is None else machine
    system = platform.system() if system is None else system
    require(machine == "aarch64" and system == "Linux", "Linux aarch64 RK3588 board only")
    compatible = Path("/proc/device-tree/compatible").read_bytes() if compatible is None else compatible
    require(b"rockchip,rk3588" in compatible.split(b"\0") or b"rockchip,rk3588s" in compatible.split(b"\0"),
            "RK3588/RK3588S compatible device required")


def validate_compile(value, model_sha, model_bytes):
    require(value["schema"] == "v37_board_only_DPVO_spatial_encoder_PTQ" and
            value["status"] == "compiled_board_spatial_PTQ_parent_rejection_retained_not_runtime_validated", "Completed spatial board PTQ receipt required")
    require(value["script_sha256"] == COMPILER_SHA and value["configuration"] == CONFIGURATION, "Exact frozen board PTQ compiler/config")
    require(value["released"] is True and value["cleanup_errors"] == [] and
            value["sources_and_inputs_unchanged"] is True and value["source_pins"] == value["source_hashes_after"],
            "Clean toolkit release and unchanged source/input evidence required")
    require(value["diagnostic_only"] is True and value["candidate_accepted"] is False and value["production_changed"] is False and
            value["host_calibration_executed"] is False and value["calibration_location"] == "aarch64 board CPU" and
            value["PTQ_requested"] is True, "Board-only diagnostic provenance required")
    require(value["INT8_operator_coverage_validated"] is False and value["deployment_inference_executed"] is False and
            value["original_FP32_gate"] == dict(rtol=1e-4, atol=1e-5, measured=False), "Do not promote unmeasured coverage/precision")
    require(value["input"]["sha256"] == GRAPH_SHA and value["ancestor"]["method"] == "spatial" and
            value["ancestor"]["sparse_onnx_sha256"] == GRAPH_SHA and value["ancestor"]["recipe_sha256"] == RECIPE_SHA and
            value["ancestor"]["parent_onnx_sha256"] == PARENT_GRAPH_SHA and
            value["ancestor"]["parent_CPU_strict_gate_passed"] is False and
            value["ancestor"]["suffix_CPU_bitwise_gate_passed"] is True and value["ancestor"]["spatial_parent_rejection_retained"] is True,
            "Exact spatial graph/suffix evidence with original parent FP32 rejection retained")
    require(value["diagnostic_allow_fp32_rejected"] is True and value["historical_parent_FP32_gate_passed"] is False and
            value["historical_parent_rejection_retained"] is True and value["parent_PTQ_source_sha256"] == PARENT_PTQ_SHA and
            value["input"]["original_graph_weights_preserved"] is True,
            "Explicit rejected-parent diagnostic and frozen original weights/compiler lineage required")
    operations = value["operations"]
    require([r["name"] for r in operations] == ["config", "load_onnx", "board_CPU_calibration_and_build", "export_rknn"] and
            all(type(r["code"]) is int and r["code"] == 0 for r in operations), "PTQ operations failed or changed")
    require(value["calibration"]["samples"] == 32 and len(value["calibration"]["rows"]) == 32, "Fixed 32 calibration samples required")
    require(0 < model_bytes <= 256 * 1024**2 and value["output"]["sha256"] == model_sha and
            value["output"]["bytes"] == model_bytes, "Model bytes/SHA must match compile output")


def validate_attr(row, index, name, logical_shape, *, spatial=False, integer=False):
    require(row["index"] == index and row["name"] == name, "Unexpected IO port identity: " + name)
    fmt = row["format"]
    require(fmt in (0, 1) if spatial else fmt in (0, 3) or integer and fmt == 1, "Unsupported/packed native layout: " + name)
    if spatial and fmt == 1:
        n, channels, height, width = logical_shape
        native_shape = (n, height, width, channels)
    else:
        native_shape = logical_shape
    require(row["shape"] == list(native_shape) and row["n_elems"] == math.prod(logical_shape), "Unexpected native shape/elements: " + name)
    dtype = row["type"]
    require(dtype == 8 if integer else dtype in (0, 1, 2, 3), "Unexpected native dtype: " + name)
    require(row["size"] == row["n_elems"] * DTYPES[dtype][1] and
            (row["size_with_stride"] == 0 or row["size_with_stride"] >= row["size"]), "Logical/stride byte contract: " + name)
    qnt, scale, zp = row["quantization_type"], row["scale"], row["zero_point"]
    require(math.isfinite(scale), "Nonfinite quantization scale: " + name)
    if integer:
        require(qnt == 0 or qnt == 2 and zp == 0 and scale == 1.0, "INT64 indices require neutral quantization; no numeric cast")
    elif dtype in (0, 1):
        require(qnt == 0 or dtype == 1 and qnt == 2 and zp == 0 and scale == 1.0, "Non-neutral floating descriptor: " + name)
    else:
        lower, upper = (-128, 127) if dtype == 2 else (0, 255)
        require(qnt == 2 and scale > 0 and lower <= zp <= upper, "8-bit IO requires positive affine scale and in-range zero point: " + name)
    return dict(logical_shape=list(logical_shape), native_shape=row["shape"], dtype=DTYPES[dtype][0],
        layout=FORMATS[fmt], quantization_type=qnt, scale=scale, zero_point=zp,
        integer_index_cast_performed=False if integer else None)


def bind_query_api(library, abi):
    contracts = {
        "rknn_init": [ct.POINTER(abi.CONTEXT), ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_void_p],
        "rknn_query": [abi.CONTEXT, ct.c_int, ct.c_void_p, ct.c_uint32],
        "rknn_destroy": [abi.CONTEXT],
    }
    for name, arguments in contracts.items():
        function = getattr(library, name)
        function.argtypes, function.restype = arguments, ct.c_int
    return library


def probe_sdk(api, abi, model, report):
    """No input submission/run/output retrieval functions are accessed here."""
    context = abi.CONTEXT()
    primary_failure = False
    try:
        filename = ct.create_string_buffer(os.fsencode(model))
        start = time.monotonic()
        report["initialization_attempted"] = True
        code = int(api.rknn_init(ct.byref(context), ct.cast(filename, ct.c_void_p), 0, 0, None))
        report.update(init_return_code=code, init_seconds=time.monotonic() - start)
        abi.check_call(code, "rknn_init")
        require(context.value != 0, "Successful init returned null context")
        report["initialized"] = True
        start = time.monotonic()
        version = abi.query(api, context, 5, abi.SDKVersion())
        report["sdk_versions"] = dict(api=bytes(version.api_version).decode(errors="replace"),
                                      driver=bytes(version.drv_version).decode(errors="replace"))
        require(report["sdk_versions"]["api"].startswith("2.3.2"), "Pinned RKNN runtime version 2.3.2 required")
        counts = abi.query(api, context, 0, abi.InputOutputNum())
        report["io_count"] = dict(inputs=int(counts.n_input), outputs=int(counts.n_output))
        require(counts.n_input == 2 and counts.n_output == 2, "Original sparse model has exactly 2 inputs and 2 outputs")
        report["input_attrs"] = [abi.attr_record(abi.query(api, context, 1, abi.TensorAttr(index=i))) for i in range(2)]
        report["output_attrs"] = [abi.attr_record(abi.query(api, context, 2, abi.TensorAttr(index=i))) for i in range(2)]
        memory = abi.query(api, context, 6, abi.MemSize())
        report["model_memory_bytes"] = {name: int(getattr(memory, name)) for name, _ in memory._fields_ if name != "reserved"}
        report["query_seconds"] = time.monotonic() - start
        require(memory.total_dma_allocated_size <= 1024**3, "Context DMA exceeds isolated 1GiB budget")
        report["validated_io"] = dict(
            normalized_bgr=validate_attr(report["input_attrs"][0], 0, "normalized_bgr", (1, 3, 464, 608), spatial=True),
            patch_indices=validate_attr(report["input_attrs"][1], 1, "patch_indices", (96,), integer=True),
            fmap=validate_attr(report["output_attrs"][0], 0, "fmap", (1, 128, 116, 152), spatial=True),
            context=validate_attr(report["output_attrs"][1], 1, "context", (1, 96, 384)))
        report["native_IO_contract_checked"] = True
    except Exception:
        primary_failure = True
        raise
    finally:
        if context.value:
            start = time.monotonic()
            try:
                code = int(api.rknn_destroy(context))
                report.update(destroy_return_code=code, context_destroyed=code == 0)
                abi.check_call(code, "rknn_destroy")
            except Exception as exc:
                report["context_destroyed"] = False
                report["cleanup_errors"].append("rknn_destroy: " + repr(exc))
                if not primary_failure:
                    raise
            finally:
                report["destroy_seconds"] = time.monotonic() - start


def load_abi(path):
    spec = importlib.util.spec_from_file_location("_v36_query_pinned_abi", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "compile-json", "abi-helper", "header", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("model", "compile"):
        p.add_argument("--" + name + "-sha256", required=True)
    p.add_argument("--frozen-query-helper", type=Path, required=True)
    p.add_argument("--diagnostic-allow-fp32-rejected", action="store_true")
    p.add_argument("--library", type=Path, default=Path("/usr/lib/librknnrt.so"))
    a = p.parse_args(argv)
    board_guard()  # Before model bytes, source module import, native library or directory creation.
    require(a.diagnostic_allow_fp32_rejected, "Explicit --diagnostic-allow-fp32-rejected required before query output creation")
    output = a.output_dir.resolve()
    require(output.is_relative_to(BASE / "experiments") and output != BASE / "experiments", "New output must be in project experiments")
    for path in (a.model.parent, a.compile_json.parent):
        require(not output.is_relative_to(path.resolve()), "Separate query output from source model/compile directory")
    output.mkdir(parents=True, exist_ok=False)
    pins = {}
    report = dict(schema="v37_DPVO_spatial_PTQ_native_IO_query", status="failed", script_sha256=sha(__file__),
        compile_receipt_sha256=a.compile_sha256, model_sha256=a.model_sha256, diagnostic_only=True,
        parent_query_implementation_sha256=PARENT_QUERY_SHA, diagnostic_allow_fp32_rejected=True,
        historical_parent_FP32_gate_passed=False, historical_parent_rejection_retained=True,
        original_DPVO_weights_preserved=True, source_spatial_graph_sha256=GRAPH_SHA,
        initialization_attempted=False, initialized=False, context_destroyed=None, cleanup_errors=[],
        actual_board_execution=True, sdk_run_calls=0, model_inference_executed=False, NPU_used=False,
        NPU_operator_coverage_validated=False, native_IO_contract_checked=False, candidate_accepted=False,
        precision_gate_measured=False, performance_measured=False, production_changed=False, ORB_used=False,
        queried_command_ids=[5, 0, 1, 2, 6], bound_functions=["rknn_init", "rknn_query", "rknn_destroy"],
        source_pins=pins, limitations=["No inference, input submission or output retrieval is performed.",
            "Native port dtype/scale/zero-point do not prove internal INT8 operator placement or model accuracy.",
            "The spatial FP32 parent previously failed the unchanged original gate; this initialization query does not promote it.",
            "Any later execution must account for input packing, SDK conversion, owned output copy and release."])
    try:
        for path, expected in ((Path(__file__), report["script_sha256"]), (a.compile_json, a.compile_sha256),
                               (a.model, a.model_sha256), (a.abi_helper, ABI_SHA), (a.header, HEADER_SHA), (a.library, LIBRARY_SHA),
                               (a.frozen_query_helper, PARENT_QUERY_SHA)):
            pins[str(path.resolve())] = require_sha(path, expected)
        compiled = json.loads(a.compile_json.read_text())
        validate_compile(compiled, a.model_sha256, a.model.stat().st_size)
        report["compile_binding"] = dict(configuration=compiled["configuration"], output=compiled["output"],
            calibration_dataset_sha256=compiled["calibration"]["dataset_sha256"], compiler_sha256=compiled["script_sha256"],
            ancestor=compiled["ancestor"], historical_parent_FP32_gate_passed=False, historical_parent_rejection_retained=True)
        abi = load_abi(a.abi_helper)
        report["abi"] = abi.abi_record()
        require(report["abi"]["context_size"] == 8 and report["abi"]["pointer_size"] == 8, "64-bit pinned ABI required")
        api = bind_query_api(ct.CDLL(str(a.library)), abi)
        probe_sdk(api, abi, a.model, report)
        report["status"] = "queried_spatial_native_IO_parent_rejection_retained_not_inference_validated"
    except Exception as exc:
        report.update(error=repr(exc), traceback=traceback.format_exc())
    finally:
        after = {}
        for path in pins:
            try:
                after[path] = sha(path)
            except Exception as exc:
                report["cleanup_errors"].append("source rehash: " + repr(exc))
        report["source_hashes_after"] = after
        report["sources_and_inputs_unchanged"] = after == pins and len(pins) == 7
        if report["cleanup_errors"] or not report["sources_and_inputs_unchanged"]:
            report["status"] = "failed"
        (output / "query.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "sdk_run_calls", "context_destroyed", "cleanup_errors")}), flush=True)
    return 0 if report["status"] == "queried_spatial_native_IO_parent_rejection_retained_not_inference_validated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
