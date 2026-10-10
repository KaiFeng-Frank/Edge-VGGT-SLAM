#!/usr/bin/env python3
"""Board-only synchronous RKNN graph probe for the captured original DINO block.

This measures one compiled block, never the full VGGT model or SLAM accuracy.
The first call and three subsequent calls include input submission, synchronous
run/output retrieval, an owned FP32 output copy, and SDK output release. Error
metrics and hashing are outside that interval. No hidden warmup is performed.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

sys.dont_write_bytecode = True

HEADER_SHA256 = "c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a"
SHAPE = (1, 782, 1024)
CORE_MASK = 1
CPUS = {6, 7}
THREADS = 2
MAX_MODEL_BYTES = 256 * 1024 * 1024
CONTEXT = ct.c_uint64


class InputOutputNum(ct.Structure):
    _fields_ = [("n_input", ct.c_uint32), ("n_output", ct.c_uint32)]


class TensorAttr(ct.Structure):
    _fields_ = [
        ("index", ct.c_uint32), ("n_dims", ct.c_uint32),
        ("dims", ct.c_uint32 * 16), ("name", ct.c_char * 256),
        ("n_elems", ct.c_uint32), ("size", ct.c_uint32),
        ("fmt", ct.c_int), ("type", ct.c_int), ("qnt_type", ct.c_int),
        ("fl", ct.c_int8), ("zp", ct.c_int32), ("scale", ct.c_float),
        ("w_stride", ct.c_uint32), ("size_with_stride", ct.c_uint32),
        ("pass_through", ct.c_uint8), ("h_stride", ct.c_uint32),
    ]


class Input(ct.Structure):
    _fields_ = [
        ("index", ct.c_uint32), ("buf", ct.c_void_p), ("size", ct.c_uint32),
        ("pass_through", ct.c_uint8), ("type", ct.c_int), ("fmt", ct.c_int),
    ]


class Output(ct.Structure):
    _fields_ = [
        ("want_float", ct.c_uint8), ("is_prealloc", ct.c_uint8),
        ("index", ct.c_uint32), ("buf", ct.c_void_p), ("size", ct.c_uint32),
    ]


class SDKVersion(ct.Structure):
    _fields_ = [("api_version", ct.c_char * 256), ("drv_version", ct.c_char * 256)]


class MemSize(ct.Structure):
    _fields_ = [
        ("total_weight_size", ct.c_uint32), ("total_internal_size", ct.c_uint32),
        ("total_dma_allocated_size", ct.c_uint64), ("total_sram_size", ct.c_uint32),
        ("free_sram_size", ct.c_uint32), ("reserved", ct.c_uint32 * 10),
    ]


class PerfRun(ct.Structure):
    _fields_ = [("run_duration", ct.c_int64)]


class PerfDetail(ct.Structure):
    _fields_ = [("perf_data", ct.c_void_p), ("data_len", ct.c_uint64)]


class RunExtend(ct.Structure):
    _fields_ = [
        ("frame_id", ct.c_uint64), ("non_block", ct.c_int32),
        ("timeout_ms", ct.c_int32), ("fence_fd", ct.c_int32),
    ]


class OutputExtend(ct.Structure):
    _fields_ = [("frame_id", ct.c_uint64)]


STRUCTS = {
    "rknn_input_output_num": InputOutputNum, "rknn_tensor_attr": TensorAttr,
    "rknn_input": Input, "rknn_output": Output, "rknn_sdk_version": SDKVersion,
    "rknn_mem_size": MemSize, "rknn_perf_run": PerfRun,
    "rknn_perf_detail": PerfDetail,
    "rknn_run_extend": RunExtend, "rknn_output_extend": OutputExtend,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or any(
        c not in "0123456789abcdef" for c in expected
    ):
        raise ValueError(f"Invalid required SHA256 for {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {path}: {actual} != {expected}")
    return actual


def abi_record():
    return {
        "context_size": ct.sizeof(CONTEXT), "pointer_size": ct.sizeof(ct.c_void_p),
        "structs": {
            name: {"size": ct.sizeof(kind), "alignment": ct.alignment(kind),
                   "offsets": {field: getattr(kind, field).offset for field, _ in kind._fields_}}
            for name, kind in STRUCTS.items()
        },
    }


def check_call(code, operation):
    if code != 0:
        raise RuntimeError(f"{operation} returned {code}")


def bind_api(library):
    functions = {
        "rknn_init": [ct.POINTER(CONTEXT), ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.c_void_p],
        "rknn_query": [CONTEXT, ct.c_int, ct.c_void_p, ct.c_uint32],
        "rknn_set_core_mask": [CONTEXT, ct.c_int],
        "rknn_inputs_set": [CONTEXT, ct.c_uint32, ct.POINTER(Input)],
        "rknn_run": [CONTEXT, ct.POINTER(RunExtend)],
        "rknn_outputs_get": [CONTEXT, ct.c_uint32, ct.POINTER(Output), ct.POINTER(OutputExtend)],
        "rknn_outputs_release": [CONTEXT, ct.c_uint32, ct.POINTER(Output)],
        "rknn_destroy": [CONTEXT],
    }
    for name, arguments in functions.items():
        function = getattr(library, name)
        function.argtypes = arguments
        function.restype = ct.c_int
    return library


def query(api, context, command, value):
    check_call(api.rknn_query(context, command, ct.byref(value), ct.sizeof(value)),
               f"rknn_query({command})")
    return value


def attr_record(attr):
    if not 1 <= attr.n_dims <= 16:
        raise ValueError("Invalid RKNN tensor rank")
    return {
        "index": int(attr.index), "name": bytes(attr.name).decode("utf-8", errors="replace"),
        "shape": list(attr.dims[:attr.n_dims]), "n_elems": int(attr.n_elems),
        "size": int(attr.size), "format": int(attr.fmt), "type": int(attr.type),
        "quantization_type": int(attr.qnt_type), "fractional_length": int(attr.fl),
        "zero_point": int(attr.zp), "pass_through": int(attr.pass_through),
        "scale": float(attr.scale), "w_stride": int(attr.w_stride),
        "h_stride": int(attr.h_stride), "size_with_stride": int(attr.size_with_stride),
    }


def validate_attr(attr):
    record = attr_record(attr)
    if tuple(record["shape"]) != SHAPE or record["n_elems"] != math.prod(SHAPE):
        raise ValueError(f"RKNN tensor shape/element mismatch: {record}")
    identity_fp16_affine = (
        record["type"] == 1 and record["quantization_type"] == 2
        and record["zero_point"] == 0 and record["scale"] == 1.0
    )
    if record["type"] not in (0, 1) or (
        record["quantization_type"] != 0 and not identity_fp16_affine
    ):
        raise ValueError(f"Expected FP32/FP16 with NONE or the exact FP16 identity-affine descriptor: {record}")
    item_bytes = 4 if record["type"] == 0 else 2
    if record["size"] != record["n_elems"] * item_bytes:
        raise ValueError(f"RKNN logical tensor byte size does not match its shape/type: {record}")
    if record["format"] not in (0, 1, 3):
        raise ValueError(f"Packed native input/output layout is unsupported by this probe: {record}")
    if not math.isfinite(record["scale"]):
        raise ValueError(f"Nonfinite tensor quantization scale: {record}")
    return record


def load_capture(directory, expected_sha):
    import numpy as np
    directory = Path(directory).resolve(strict=True)
    receipt = directory / "capture.json"
    require_sha(receipt, expected_sha)
    metadata = json.loads(receipt.read_text())
    cpu_reference = metadata.get("cpu_reference", {})
    if metadata.get("status") != "completed" or any(
        cpu_reference.get(key) != value for key, value in {
            "device": "cpu", "dtype": "float32", "no_grad": True, "eval": True,
            "finite": True, "all_elementwise_equal": True,
        }.items()
    ):
        raise ValueError("Capture must be completed with the original CPU block")
    arrays, files = [], {}
    for name in ("input.npy", "output.npy"):
        path = directory / name
        item = metadata["artifacts"][name]
        digest = require_sha(path, item["sha256"])
        if path.stat().st_size != item["bytes"]:
            raise ValueError(f"Capture artifact byte size mismatch: {name}")
        array = np.load(path, allow_pickle=False, mmap_mode="r")
        if array.dtype != np.dtype("float32") or tuple(array.shape) != SHAPE:
            raise ValueError(f"Expected native FP32 {SHAPE} in {name}")
        if item["shape"] != list(SHAPE) or item["dtype"] != "float32":
            raise ValueError(f"Capture artifact metadata mismatch: {name}")
        if not array.flags.c_contiguous or not bool(np.isfinite(array).all()):
            raise ValueError(f"Nonfinite or noncontiguous capture artifact: {name}")
        arrays.append(array)
        files[str(path)] = digest
    return arrays[0], arrays[1], metadata, {str(receipt): expected_sha, **files}


def comparison(actual, expected):
    import numpy as np
    if actual.shape != expected.shape or actual.dtype != np.float32:
        raise ValueError("Output does not match the reference shape/dtype")
    if not bool(np.isfinite(actual).all()) or not bool(np.isfinite(expected).all()):
        raise ValueError("Nonfinite output/reference")
    actual64 = actual.astype(np.float64).reshape(-1)
    expected64 = expected.astype(np.float64).reshape(-1)
    difference = actual64 - expected64
    reference_norm = float(np.linalg.norm(expected64))
    actual_norm = float(np.linalg.norm(actual64))
    error_norm = float(np.linalg.norm(difference))
    cosine = None if reference_norm == 0 or actual_norm == 0 else float(
        np.dot(actual64, expected64) / (reference_norm * actual_norm)
    )
    relative = error_norm / reference_norm if reference_norm else (0.0 if error_norm == 0 else None)
    return {
        "finite": True, "array_equal": bool(np.array_equal(actual, expected)),
        "max_abs_error": float(np.max(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "relative_l2": relative, "cosine_similarity": cosine,
        "relative_l2_defined": relative is not None, "cosine_defined": cosine is not None,
        "output_raw_sha256": hashlib.sha256(memoryview(actual).cast("B")).hexdigest(),
    }


def execute_call(api, context, inputs, input_format, timeout_ms, cleanup_errors, activity=None):
    """Copy the float output while SDK-owned memory is live; always release it."""
    import numpy as np
    request = (Input * 1)()
    request[0] = Input(0, inputs.ctypes.data, inputs.nbytes, 0, 0, input_format)
    outputs = (Output * 1)()
    outputs[0].want_float = 1
    outputs[0].is_prealloc = 0
    outputs[0].index = 0
    run = RunExtend(0, 0, timeout_ms, -1)
    received = OutputExtend()
    row = {"want_float": 1, "is_prealloc": 0, "input_type": "FP32",
           "input_pass_through": 0, "input_format": input_format,
           "synchronous": True, "run_timeout_ms": timeout_ms}
    begin = time.perf_counter()
    got = False
    try:
        check_call(api.rknn_inputs_set(context, 1, request), "rknn_inputs_set")
        set_end = time.perf_counter()
        if activity is not None:
            activity["npu_run_api_called"] = True
            activity["npu_run_api_call_count"] = activity.get("npu_run_api_call_count", 0) + 1
        check_call(api.rknn_run(context, ct.byref(run)), "rknn_run")
        run_end = time.perf_counter()
        check_call(api.rknn_outputs_get(context, 1, outputs, ct.byref(received)), "rknn_outputs_get")
        got = True
        get_end = time.perf_counter()
        if run.frame_id != received.frame_id:
            raise ValueError("Synchronous run/output frame identity mismatch")
        if not outputs[0].buf or outputs[0].size != inputs.nbytes:
            raise ValueError("Invalid FP32 output pointer/byte size")
        view = np.ctypeslib.as_array(ct.cast(outputs[0].buf, ct.POINTER(ct.c_float)),
                                   shape=(math.prod(SHAPE),))
        owned = view.copy().reshape(SHAPE)
        copy_end = time.perf_counter()
        row.update(input_set_seconds=set_end - begin, run_api_seconds=run_end - set_end,
                   outputs_get_seconds=get_end - run_end, output_copy_seconds=copy_end - get_end,
                   runtime_frame_id=int(run.frame_id), output_bytes=int(outputs[0].size))
    finally:
        # Some SDK errors may return a partially allocated buffer.
        if got or outputs[0].buf:
            release_begin = time.perf_counter()
            code = api.rknn_outputs_release(context, 1, outputs)
            release_end = time.perf_counter()
            row["outputs_release_seconds"] = release_end - release_begin
            row["output_buffer_released"] = code == 0
            row["io_total_seconds"] = release_end - begin
            if code:
                cleanup_errors.append(f"rknn_outputs_release returned {code}")
            check_call(code, "rknn_outputs_release")
    # SDK run_duration is informational and outside the IO interval.
    perf = PerfRun()
    code = api.rknn_query(context, 4, ct.byref(perf), ct.sizeof(perf))
    row["perf_run_query_code"] = int(code)
    row["sdk_run_duration_us"] = int(perf.run_duration) if code == 0 else None
    return owned, row


def process_memory():
    result = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, size, unit = line.split()
            if unit != "kB":
                raise ValueError("Unexpected /proc memory unit")
            result[key.rstrip(":") + "_MiB"] = int(size) / 1024.0
    return result


def perf_detail(api, context):
    detail = query(api, context, 3, PerfDetail())
    if not detail.perf_data or not 0 < detail.data_len <= 8 * 1024 * 1024:
        raise ValueError("Invalid or oversized SDK performance detail buffer")
    raw = ct.string_at(detail.perf_data, detail.data_len)
    return {"query_command": 3, "bytes": int(detail.data_len),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "raw_text": raw.decode("utf-8", errors="replace"),
            "operator_targets_interpreted_by_probe": False}


def require_board():
    if sys.platform != "linux" or platform.machine().lower() not in ("aarch64", "arm64"):
        raise RuntimeError("Board-only probe: Linux aarch64 RK3588 is required")
    if ct.sizeof(ct.c_void_p) != 8 or sys.byteorder != "little":
        raise RuntimeError("This ABI requires little-endian LP64")
    for path in (Path("/proc/device-tree/compatible"),
                 Path("/sys/firmware/devicetree/base/compatible")):
        if path.exists():
            compatible = path.read_bytes().replace(b"\0", b",").decode(errors="replace")
            if "rk3588" not in compatible:
                raise RuntimeError(f"Unexpected board: {compatible}")
            return compatible
    raise RuntimeError("RK3588 device-tree identity is unavailable")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--capture-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON receipt, never overwritten")
    parser.add_argument("--library", type=Path, default=Path("/usr/lib/librknnrt.so"))
    parser.add_argument("--core-mask", type=int, choices=(1, 7), default=CORE_MASK,
                        help="Whole-graph API: core0 (1, default) or cores0/1/2 (7)")
    parser.add_argument("--run-timeout-ms", type=int, default=30000)
    parser.add_argument("--collect-perf-detail", action="store_true",
                        help="Separate profiling run; collected timings are not a normal benchmark")
    args = parser.parse_args(argv)
    if not 1 <= args.run_timeout_ms <= 60000:
        parser.error("run-timeout-ms must be within 1..60000")
    output = args.output.resolve()
    if output.exists():
        parser.error("output already exists")
    if not output.parent.is_dir():
        parser.error("output parent directory must already exist")
    protected = [args.model.resolve(), args.capture_dir.resolve() / "capture.json",
                 args.capture_dir.resolve() / "input.npy", args.capture_dir.resolve() / "output.npy",
                 args.library.resolve(), Path(__file__).resolve()]
    if output in protected:
        parser.error("output must differ from all inputs, runtime and source")
    # Claim the new receipt exclusively before touching an SDK context.
    stream = output.open("x", encoding="utf-8")
    started = time.monotonic()
    report = {
        "kind": "original_DINO_block_compiled_RKNN_independent_board_probe",
        "status": "failed", "full_model_quality_validated": False,
        "real_time_validated": False, "production_modified": False,
        "all_operators_on_npu_verified": False, "npu_run_api_called": False,
        "npu_run_api_call_count": 0,
        "core_mask": args.core_mask, "core_mask_set_successfully": False,
        "cpus_requested": sorted(CPUS), "threads": THREADS,
        "first_call_count": 1, "subsequent_measured_call_count": 3,
        "unmeasured_warmup_calls": 0, "init_flags": 8 if args.collect_perf_detail else 0,
        "async_enabled": False, "perf_detail_collection_enabled": args.collect_perf_detail,
        "normal_benchmark_timing": not args.collect_perf_detail,
        "tensor_metadata_policy": {
            "accepted_types": {"0": "FP32", "1": "FP16"},
            "accepted_quantization_descriptors": [
                {"quantization_type": 0, "meaning": "NONE"},
                {"type": 1, "quantization_type": 2, "scale": 1.0, "zero_point": 0,
                 "meaning": "FP16 with exact identity AFFINE_ASYMMETRIC metadata"},
            ],
            "integer_tensors_and_nonidentity_quantization_rejected": True,
            "input_api_type": "FP32, pass_through=0, SDK converts to declared floating type",
            "output_api_type": "want_float=1, SDK returns FP32",
        },
        "io_interval": "inputs_set + blocking run + blocking outputs_get + owned copy + release",
        "io_excludes": ["init", "queries", "error metrics", "hashing"],
        "timing_limitations": ["Four sequential calls are not statistical performance validation",
                               "This is a single block, not whole-model or camera timing"],
        "abi": abi_record(), "calls": [], "cleanup_errors": [],
    }
    context, api, pins = CONTEXT(), None, {}
    if args.collect_perf_detail:
        report["timing_limitations"].append("SDK performance collection changes execution cost; use a separate normal run")
    try:
        report["device_tree_compatible"] = require_board()
        allowed = os.sched_getaffinity(0)
        if not CPUS <= allowed:
            raise RuntimeError(f"CPUs 6,7 are not allowed: {sorted(allowed)}")
        os.sched_setaffinity(0, CPUS)
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = str(THREADS)
        report["cpu_affinity"] = sorted(os.sched_getaffinity(0))
        report["python"] = sys.version
        import numpy as np
        report["numpy_version"] = np.__version__
        model = args.model.resolve(strict=True)
        if not 0 < model.stat().st_size <= MAX_MODEL_BYTES:
            raise ValueError("Compiled single-block model size is outside 1..256MiB")
        model_sha = require_sha(model, args.model_sha256)
        inputs, expected, capture, pins = load_capture(args.capture_dir, args.capture_sha256)
        header = Path(__file__).resolve().parents[1] / "reference" / "rknn_api_v2.3.2.h"
        pins.update({str(model): model_sha, str(header): require_sha(header, HEADER_SHA256),
                     str(Path(__file__).resolve()): sha256_file(__file__)})
        library = args.library.resolve(strict=True)
        pins[str(library)] = sha256_file(library)
        report.update(source_and_input_hashes=pins.copy(), capture_metadata=capture,
                      input_shape=list(inputs.shape), input_dtype=str(inputs.dtype),
                      compiled_model_bytes=model.stat().st_size,
                      reference_raw_sha256=hashlib.sha256(memoryview(expected).cast("B")).hexdigest(),
                      memory_before_sdk=process_memory())
        api = bind_api(ct.CDLL(str(library)))
        tick = time.perf_counter()
        filename = ct.create_string_buffer(os.fsencode(model))
        check_call(api.rknn_init(ct.byref(context), ct.cast(filename, ct.c_void_p),
                                0, report["init_flags"], None), "rknn_init")
        report["init_seconds"] = time.perf_counter() - tick
        check_call(api.rknn_set_core_mask(context, args.core_mask), "rknn_set_core_mask")
        report["core_mask_set_successfully"] = True
        version = query(api, context, 5, SDKVersion())
        report["sdk_version"] = bytes(version.api_version).decode(errors="replace")
        report["driver_version"] = bytes(version.drv_version).decode(errors="replace")
        if not report["sdk_version"].startswith("2.3.2"):
            raise ValueError("Probe ABI is pinned to RKNN runtime 2.3.2")
        counts = query(api, context, 0, InputOutputNum())
        if counts.n_input != 1 or counts.n_output != 1:
            raise ValueError("Expected exactly one input and one output")
        input_attr = query(api, context, 1, TensorAttr(index=0))
        output_attr = query(api, context, 2, TensorAttr(index=0))
        # Preserve both SDK descriptors even when the following gate rejects.
        report["input_attr"] = attr_record(input_attr)
        report["output_attr"] = attr_record(output_attr)
        report["tensor_attributes_validated"] = False
        validate_attr(input_attr)
        validate_attr(output_attr)
        report["tensor_attributes_validated"] = True
        memory = MemSize()
        code = api.rknn_query(context, 6, ct.byref(memory), ct.sizeof(memory))
        report["model_memory_query_code"] = int(code)
        report["model_memory_bytes"] = {
            name: int(getattr(memory, name)) for name, _ in memory._fields_ if name != "reserved"
        } if code == 0 else None
        first = None
        for index in range(4):
            actual, row = execute_call(api, context, inputs, int(input_attr.fmt),
                                       args.run_timeout_ms, report["cleanup_errors"], report)
            row.update(call_index=index, phase="first" if index == 0 else "subsequent",
                       versus_cpu_reference=comparison(actual, expected))
            if first is None:
                first = actual
            row["versus_first_call"] = comparison(actual, first)
            if args.collect_perf_detail:
                row["perf_detail"] = perf_detail(api, context)
            report["calls"].append(row)
        warm = [row["io_total_seconds"] for row in report["calls"][1:]]
        report["subsequent_io_seconds"] = {
            "values": warm, "mean": float(np.mean(warm)), "median": float(np.median(warm)),
            "min": min(warm), "max": max(warm),
        }
        report["status"] = "completed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if api is not None and context.value:
            try:
                check_call(api.rknn_destroy(context), "rknn_destroy")
                report["context_destroyed"] = True
            except Exception as error:
                report["cleanup_errors"].append(f"{type(error).__name__}: {error}")
        else:
            report["context_destroyed"] = None
        if pins:
            try:
                report["source_and_input_hashes_after"] = {path: sha256_file(path) for path in pins}
                report["source_and_inputs_unchanged"] = report["source_and_input_hashes_after"] == pins
                if not report["source_and_inputs_unchanged"]:
                    report["cleanup_errors"].append("Source/input/runtime hash changed during probe")
            except Exception as error:
                report["cleanup_errors"].append(f"Source recheck: {type(error).__name__}: {error}")
        report["memory_final"] = process_memory()
        report["successful_synchronous_call_count"] = len(report["calls"])
        report["total_seconds"] = time.monotonic() - started
        if report["cleanup_errors"]:
            report["status"] = "failed"
        try:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        finally:
            stream.close()
    print(json.dumps({"status": report["status"], "error": report.get("error"),
                      "output": str(output), "sha256": sha256_file(output)},
                     allow_nan=False), flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
