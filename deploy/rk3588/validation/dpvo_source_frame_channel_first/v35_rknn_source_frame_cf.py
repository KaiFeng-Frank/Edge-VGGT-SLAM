"""Channel-first [B,C,D,M] complete-source-frame SDK contract (research).

Derived from the hash-pinned v33 padded-output adapter. Only tensor contracts,
manifest identities and ancestry bindings change. Native rank4 packing and
returned FP32 byte/ownership guards remain. CPU gate failure can be observed
only through explicit diagnostic entrypoints; no production adoption or VO.
DPVO arithmetic attribution: Princeton-VL/DPVO, MIT. SDK files remain external.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import copy
import ctypes as ct
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import time

import numpy as np


INPUT_NAMES = ("net", "inp", "corr")
OUTPUT_NAMES = ("net_out", "delta", "weight")
MANIFEST_SCHEMA = "v35_DPVO_source_frame_channel_first_board_export"
MANIFEST_STATUS = "completed_channel_first_export_diagnostic_not_adopted"
CONFIGURATION = dict(target_platform="rk3588", float_dtype="float16",
                     optimization_level=3, do_quantization=False)
ABI_SHA256 = "37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608"
HEADER_SHA256 = "c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a"
LIBRARY_SHA256 = "d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8"
PARENT_NAME = "v33_rknn_source_frame_padded.py"
PARENT_SHA256 = "a5af6d39b2d19471255fc86aa42c9aa9b0247e428715c23ffd4ba79b184ad865"
PADDING_RATIONALE = ("Accept output-only internal size_with_stride >= native logical size; "
                     "retain outputs_get FP32 logical 4*N bytes, pointer and owned-copy guards. "
                     "Zero stride retains the parent's unspecified-stride meaning.")
RTOL, ATOL = 1e-4, 1e-5
UINT32_MAX = 2**32 - 1
_FD_LOCK = threading.Lock()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pin(path, expected, pins):
    path = Path(path).resolve()
    require(isinstance(expected, str) and re.fullmatch("[0-9a-f]{64}", expected),
            "Required lowercase SHA256 is invalid")
    require(sha256(path) == expected, "SHA256 mismatch: " + str(path))
    pins[str(path)] = expected
    return path


def recheck_pins(pins):
    after = {p: sha256(p) for p in pins}
    require(after == pins, "Source or input bytes changed")
    return after


def pin_parent_source(pins):
    """Bind the byte-identical compiler I/O ancestor, without changing it."""
    return pin(Path(__file__).resolve().with_name(PARENT_NAME), PARENT_SHA256, pins)


def implementation_provenance():
    return dict(runtime_helper_sha256=sha256(__file__), parent_file=PARENT_NAME,
                parent_sha256=PARENT_SHA256, header_sha256=HEADER_SHA256,
                logical_layout="BCDM_channel_first", compiler_uses_this_helper=True,
                rationale=PADDING_RATIONALE,
                returned_buffer_bytes_verified_at_each_call=True,
                packed_FP32_output_not_guaranteed_by_attribute_alone=True)


def contracts(inputs, outputs):
    def read(rows, names):
        require(isinstance(rows, (list, tuple)) and len(rows) == 3,
                "Expected exactly three logical operands")
        result = {}
        for row in rows:
            name, shape = row["name"], tuple(row["shape"])
            require(name in names and name not in result, "Unknown/duplicate operand")
            require(type(row["dtype"]) in (int, str) and row["dtype"] in (1, "float32"),
                    "Only logical FP32 is supported")
            require(len(shape) == 4 and all(type(x) is int and 0 < x <= UINT32_MAX
                                           for x in shape), "Expected positive fixed rank4")
            require(math.prod(shape) * 4 <= UINT32_MAX, "Operand exceeds ABI byte size")
            result[name] = dict(name=name, shape=shape, logical_shape=list(shape),
                                logical_dtype="float32")
        return result
    ins, outs = read(inputs, INPUT_NAMES), read(outputs, OUTPUT_NAMES)
    b, width, degree, patches = ins["net"]["shape"]
    require(b == 1 and 1 <= degree <= 25 and 1 <= patches <= 96 and width == 384,
            "Expected fixed B1, D1..25, M1..96, C384")
    require(ins["inp"]["shape"] == (b, 384, degree, patches)
            and ins["corr"]["shape"] == (b, 882, degree, patches),
            "Source-frame input shapes differ")
    require(outs["net_out"]["shape"] == (b, 384, degree, patches)
            and outs["delta"]["shape"] == (b, 2, degree, patches)
            and outs["weight"]["shape"] == (b, 2, degree, patches),
            "Source-frame output shapes differ")
    return ins, outs


def validate_manifest(path, expected_sha256, *, allow_fp32_rejected_diagnostic=False,
                      verify_source_files=False):
    """Verify declared export/provenance; does not independently run its gates.

    Source paths are manifest-relative or absolute, but model.file is strictly
    relative to the manifest directory. Host compilation cannot rehash board
    absolute sources: verify_source_files=False records only that declaration;
    board callers must pass True. Numeric gates are declared board evidence,
    not newly executed or independently recomputed by this metadata validator.
    """
    pins = {}
    path = pin(path, expected_sha256, pins)
    manifest = json.loads(path.read_text())
    require(manifest["schema"] == MANIFEST_SCHEMA and manifest["status"] == MANIFEST_STATUS,
            "Unexpected source-frame export schema/status")
    require(manifest["original_parameters_unchanged"] is True
            and manifest["sources_and_inputs_unchanged"] is True
            and manifest["cleanup_errors"] == []
            and manifest["candidate_accepted"] is False,
            "Original-parameter/source/cleanup diagnostic contract failed")
    strict = manifest["strict_FP32_gate_passed"]
    require(type(strict) is bool, "FP32 gate must be an explicit boolean")
    require(strict or allow_fp32_rejected_diagnostic,
            "FP32 rejection requires an explicit diagnostic flag")
    cases = manifest["cases"]
    require(isinstance(cases, list) and len(cases) > 0 and all(isinstance(c, dict) for c in cases),
            "No source-frame reference cases")
    for flag in ("outputs_finite", "weight_domain_valid"):
        quality = manifest.get(flag)
        require(quality is True or (quality is None and all(c.get(flag) is True for c in cases)),
                "Mandatory finite/domain evidence missing: " + flag)
        require(all(flag not in c or c[flag] is True for c in cases),
                "A case contradicts declared finite/domain evidence: " + flag)
    contracts(manifest["inputs"], manifest["outputs"])
    sources = manifest["source_pins"]
    require(isinstance(sources, dict) and len(sources) > 0
            and manifest["source_hashes_after"] == sources, "Source provenance is incomplete")
    for source, expected in sources.items():
        require(isinstance(expected, str) and re.fullmatch("[0-9a-f]{64}", expected),
                "Invalid declared source SHA256")
        source = Path(source)
        if verify_source_files:
            pin(source if source.is_absolute() else path.parent / source, expected, pins)
    item = manifest["model"]
    relative = Path(item["file"])
    require(not relative.is_absolute() and ".." not in relative.parts,
            "ONNX file must remain within manifest directory")
    model = pin(path.parent / relative, item["sha256"], pins)
    require(model.is_relative_to(path.parent) and type(item["bytes"]) is int
            and model.stat().st_size == item["bytes"] > 0, "ONNX artifact bytes/path differ")
    return manifest, model, pins


def validate_compilation(path, expected_sha256, manifest, manifest_sha256, *,
                         compiler_path, compiler_sha256):
    """Bind the actual RKNN bytes to this export and unchanged compiler config.

    Host compiler source paths in the receipt need not exist on the board;
    explicit compiler_path is the relocated, byte-identical script to pin.
    The compiler uses this channel-first contract; the frozen parent is an
    separately pinned output adapter and is not falsely recorded as that source.
    """
    pins = {}
    path = pin(path, expected_sha256, pins)
    pin(compiler_path, compiler_sha256, pins)
    pin_parent_source(pins)
    pins[str(Path(__file__).resolve())] = sha256(__file__)
    report = json.loads(path.read_text())
    require(report["schema"] == "v35_channel_first_Update_RKNN_compilation"
            and report["status"] == "compiled_source_frame_diagnostic_not_runtime_validated",
            "Unexpected compilation schema/status")
    require(report["script_sha256"] == compiler_sha256
            and report["io_helper_sha256"] == sha256(__file__)
            and report["manifest_sha256"] == manifest_sha256
            and report["input"]["sha256"] == manifest["model"]["sha256"],
            "Compile/export/implementation chain differs")
    require(report["configuration"] == CONFIGURATION and report["released"] is True
            and not report.get("release_error") and report["sources_and_inputs_unchanged"] is True
            and report["source_pins"] == report["source_hashes_after"],
            "Compiler config/cleanup/source contract failed")
    require(report["export_strict_FP32_gate_passed"] is manifest["strict_FP32_gate_passed"]
            and report["candidate_accepted"] is False
            and (manifest["strict_FP32_gate_passed"] or report["allow_fp32_rejected_diagnostic"] is True),
            "Diagnostic numerical rejection was not preserved")
    item = report["output"]
    relative = Path(item["file"])
    require(not relative.is_absolute() and ".." not in relative.parts,
            "RKNN must stay within compilation directory")
    model = pin(path.parent / relative, item["sha256"], pins)
    require(model.is_relative_to(path.parent) and model.stat().st_size == item["bytes"] > 0,
            "Compiled RKNN artifact bytes/path differ")
    return report, model, pins


def _bind(rows, logical, *, output):
    require(len(rows) == 3 and {r["name"] for r in rows} == set(logical),
            "SDK name/count binding mismatch")
    result = []
    for index, row in enumerate(rows):
        require(type(row["index"]) is int and row["index"] == index,
                "SDK index/order binding mismatch")
        operand = logical[row["name"]]
        shape = operand["shape"]
        native = tuple(row["shape"])
        fmt = row["format"]
        if fmt in (0, 3):
            require(native == shape, "SDK original shape differs")
            permutation = None
        elif fmt == 1 and not output:
            require(native == (shape[0], shape[2], shape[3], shape[1]),
                    "SDK NHWC mapping differs from explicit logical rank4")
            permutation = (0, 2, 3, 1)
        else:
            raise ValueError("Unimplemented SDK layout; no implicit output permutation")
        require(row["type"] in (0, 1), "Integer/packed output or input is unsupported")
        quant = row["quantization_type"]
        neutral = (quant == 2 and row["scale"] == 1.0 and row["zero_point"] == 0
                   and row["fractional_length"] == 0)
        require(quant == 0 or neutral, "Nonidentity quantization is unsupported")
        require(row["n_elems"] == math.prod(shape)
                and row["size"] == math.prod(shape) * (4 if row["type"] == 0 else 2),
                "SDK elements/native bytes differ")
        stride_bytes = row.get("size_with_stride", 0)
        if output:
            require(type(stride_bytes) is int and (stride_bytes == 0
                    or row["size"] <= stride_bytes <= UINT32_MAX),
                    "Output stride footprint is smaller than native logical size or invalid")
        else:
            require(row.get("size_with_stride", 0) in (0, row["size"]),
                    "Padded native byte layout is unimplemented")
        result.append(dict(operand, index=index, sdk_shape=list(native), sdk_format=fmt,
                           sdk_type=row["type"], sdk_native_bytes=row["size"],
                           sdk_size_with_stride=stride_bytes,
                           native_output_padding_bytes=max(0, stride_bytes-row["size"]) if output else 0,
                           sdk_quantization_type=quant, neutral_affine_metadata=neutral,
                           sdk_fractional_length=row["fractional_length"],
                           sdk_scale=row["scale"], sdk_zero_point=row["zero_point"],
                           permutation=permutation, transport_type=0, transport_format=fmt,
                           transport_bytes=math.prod(shape) * 4, pass_through=0,
                           want_float=1, returned_FP32_bytes=math.prod(shape) * 4))
    return result


def validate_input_attrs(inputs, outputs, attrs):
    ins, _ = contracts(inputs, outputs)
    return _bind(attrs, ins, output=False)


def validate_output_attrs(inputs, outputs, attrs):
    _, outs = contracts(inputs, outputs)
    return _bind(attrs, outs, output=True)


@dataclass(frozen=True)
class PackedInputs:
    requests: object
    owned_buffers: tuple
    records: tuple


def pack_inputs(abi, arrays, bindings):
    require(set(arrays) == set(INPUT_NAMES) and len(bindings) == 3,
            "Expected exactly three source arrays/bindings")
    require({r["name"] for r in bindings} == set(INPUT_NAMES)
            and [r["index"] for r in bindings] == list(range(3)), "Invalid SDK bindings")
    requests = (abi.Input * 3)()
    owned, records = [], []
    for row in bindings:
        a = arrays[row["name"]]
        require(isinstance(a, np.ndarray) and a.dtype == np.dtype("float32")
                and a.shape == row["shape"] and bool(np.isfinite(a).all()),
                "Source shape/dtype/nonfinite mismatch: " + row["name"])
        view = a.transpose(row["permutation"]) if row["permutation"] else a
        buffer = np.array(view, dtype=np.float32, order="C", copy=True)
        require(buffer.shape == tuple(row["sdk_shape"])
                and buffer.nbytes == row["transport_bytes"], "Packed SDK bytes differ")
        requests[row["index"]] = abi.Input(row["index"], buffer.ctypes.data,
                                          buffer.nbytes, 0, 0, row["transport_format"])
        owned.append(buffer)
        records.append(dict(index=row["index"], name=row["name"],
                            logical_shape=list(a.shape), buffer_shape=list(buffer.shape),
                            source_dtype="float32", transport_dtype="float32",
                            native_attribute_format=row["sdk_format"],
                            request_format=row["transport_format"], pass_through=0,
                            bytes=buffer.nbytes, owned=True))
    return PackedInputs(requests, tuple(owned), tuple(records))


def compare(actual, reference):
    """Fixed strict gate; never emit NaN/Infinity in JSON metadata."""
    require(actual.shape == reference.shape and actual.dtype == reference.dtype == np.float32,
            "Reference shape/dtype differs")
    finite = bool(np.isfinite(actual).all() and np.isfinite(reference).all())
    if not finite:
        return dict(finite=False, gate_passed=False, rtol=RTOL, atol=ATOL,
                    nonfinite_elements=int((~np.isfinite(actual)).sum()),
                    failed_elements=None, max_abs=None, relative_l2=None)
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    diff = a - b
    failures = int((np.abs(diff) > ATOL + RTOL * np.abs(b)).sum())
    norm = float(np.linalg.norm(b.ravel()))
    return dict(finite=True, gate_passed=failures == 0, rtol=RTOL, atol=ATOL,
                failed_elements=failures, nonfinite_elements=0,
                max_abs=float(np.abs(diff).max()),
                relative_l2=float(np.linalg.norm(diff.ravel()) / norm) if norm else None)


@contextmanager
def _native_log(directory):
    """Serial diagnostic ownership of native fd1/2; no background SDK calls."""
    with _FD_LOCK, tempfile.TemporaryFile(dir=directory) as log:
        record = {}
        libc = ct.CDLL(None)
        saved = []
        try:
            for stream in (sys.stdout, sys.stderr):
                stream.flush()
            libc.fflush(None)
            for fd in (1, 2):
                saved.append(os.dup(fd))
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            yield record
        finally:
            libc.fflush(None)
            for fd, original in zip((1, 2), saved):
                os.dup2(original, fd)
                os.close(original)
            log.seek(0, os.SEEK_END)
            size = log.tell()
            log.seek(0)
            raw = log.read(4 << 20)
            digest = hashlib.sha256(raw)
            for block in iter(lambda: log.read(1 << 20), b""):
                digest.update(block)
            record.update(bytes=size, truncated=size > len(raw),
                          raw_sha256=digest.hexdigest(),
                          raw_text=raw.decode("utf-8", errors="replace"),
                          RKNN_error_detected=bool(re.search(rb"\bE\s+RKNN\b", raw)))


class SourceFrameSession:
    """One context, synchronous serial calls; outputs own memory after release.

    For host tests construct with an injected fake API and hardware=False. Real
    construction must use open_board(), which checks board and binary identity.
    Finite/domain/strict rejection is returned for raw-array preservation; API,
    frame-ID, native error log or release failure is an execution failure.
    """
    def __init__(self, abi, api, context, inputs, outputs, input_attrs, output_attrs,
                 *, pins=None, hardware=False, capture_logs=False, log_directory=None,
                 collect_perf_detail=False):
        require(not hardware, "Real SDK must be constructed through open_board")
        self.abi, self.api, self.context = abi, api, context
        self._hardware = False
        self.input_bindings = validate_input_attrs(inputs, outputs, input_attrs)
        self.output_bindings = validate_output_attrs(inputs, outputs, output_attrs)
        self._pins = dict(pins or {})
        self._capture_logs = capture_logs
        self._log_directory = log_directory
        self._profile = collect_perf_detail
        self._lock = threading.RLock()
        self._metrics = dict(synthetic=True, sdk_run_api_calls=0, completed_calls=0,
                             SDK_outputs_releases=0, npu_used=False, closed=False,
                             context_destroyed=False, cleanup_errors=[],
                             implementation_provenance=implementation_provenance(),
                             output_bindings=copy.deepcopy(self.output_bindings))

    @classmethod
    def open_board(cls, model_path, model_sha256, inputs, outputs, *, abi_path,
                   header_path, library_path, pins=None, collect_perf_detail=False,
                   log_directory=None):
        """Hardware guard and all file SHA gates precede SDK load/context init."""
        sources = dict(pins or {})
        pin_parent_source(sources)
        abi_file = pin(abi_path, ABI_SHA256, sources)
        spec = importlib.util.spec_from_file_location("v33_frozen_sdk_abi", abi_file)
        abi = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(abi)
        abi.require_board()
        pin(header_path, HEADER_SHA256, sources)
        pin(library_path, LIBRARY_SHA256, sources)
        model_path = pin(model_path, model_sha256, sources)
        sources[str(Path(__file__).resolve())] = sha256(__file__)
        contracts(inputs, outputs)
        recheck_pins(sources)
        if log_directory is not None:
            require(Path(log_directory).is_dir(), "Log directory does not exist")
        api = abi.bind_api(ct.CDLL(str(library_path)))
        context = abi.CONTEXT()
        instance = None
        try:
            data = model_path.read_bytes()
            require(0 < len(data) <= UINT32_MAX, "RKNN model bytes exceed ABI")
            buffer = ct.create_string_buffer(data)
            abi.check_call(api.rknn_init(ct.byref(context), ct.cast(buffer, ct.c_void_p),
                                         len(data), 8 if collect_perf_detail else 0, None), "rknn_init")
            abi.check_call(api.rknn_set_core_mask(context, 7), "rknn_set_core_mask")
            version = abi.query(api, context, 5, abi.SDKVersion())
            require(bytes(version.api_version).decode().startswith("2.3.2"), "Unexpected SDK version")
            counts = abi.query(api, context, 0, abi.InputOutputNum())
            require(counts.n_input == counts.n_output == 3, "SDK operand count differs")
            ins = [abi.attr_record(abi.query(api, context, 1, abi.TensorAttr(index=i))) for i in range(3)]
            outs = [abi.attr_record(abi.query(api, context, 2, abi.TensorAttr(index=i))) for i in range(3)]
            instance = cls(abi, api, context, inputs, outputs, ins, outs, pins=sources,
                           capture_logs=True, log_directory=log_directory,
                           collect_perf_detail=collect_perf_detail)
            instance._hardware = True
            instance._model_buffer = buffer
            instance._metrics.update(synthetic=False, sdk_version=bytes(version.api_version).decode(),
                                     driver_version=bytes(version.drv_version).decode(), core_mask=7,
                                     init_flags=8 if collect_perf_detail else 0,
                                     input_attributes=ins, output_attributes=outs,
                                     source_pins=sources, abi=abi.abi_record())
            memory = abi.query(api, context, 6, abi.MemSize())
            instance._metrics["memory"] = dict(total_dma_allocated_size=int(memory.total_dma_allocated_size),
                                               total_weight_size=int(memory.total_weight_size),
                                               total_internal_size=int(memory.total_internal_size))
            return instance
        except BaseException:
            if instance is not None:
                instance.close()
            elif context.value:
                abi.check_call(api.rknn_destroy(context), "partial_init_destroy")
            raise

    def execute(self, feed, reference=None, *, timeout_ms=60000):
        with self._lock:
            total_begin = time.monotonic()
            require(not self._metrics["closed"], "Context is closed")
            require(type(timeout_ms) is int and 1 <= timeout_ms <= 2**31-1, "Invalid timeout")
            record = dict(candidate_accepted=False, execution_status="starting")
            values = {}
            def run():
                abi, api, ctx = self.abi, self.api, self.context
                begin = time.monotonic()
                packed = pack_inputs(abi, feed, self.input_bindings)
                packed_end = time.monotonic()
                out = (abi.Output * 3)()
                for i in range(3):
                    out[i].want_float = 1
                    out[i].index = i
                ext = abi.RunExtend(non_block=0, timeout_ms=timeout_ms, fence_fd=-1)
                received = abi.OutputExtend()
                got = False
                try:
                    abi.check_call(api.rknn_inputs_set(ctx, 3, packed.requests), "rknn_inputs_set")
                    set_end = time.monotonic()
                    self._metrics["sdk_run_api_calls"] += 1
                    self._metrics["npu_used"] = self._hardware
                    abi.check_call(api.rknn_run(ctx, ct.byref(ext)), "rknn_run")
                    run_end = time.monotonic()
                    abi.check_call(api.rknn_outputs_get(ctx, 3, out, ct.byref(received)), "rknn_outputs_get")
                    got = True
                    get_end = time.monotonic()
                    require(ext.frame_id == received.frame_id, "SDK frame-ID mismatch")
                    for row in self.output_bindings:
                        output = out[row["index"]]
                        require(bool(output.buf) and output.size == row["returned_FP32_bytes"],
                                "SDK returned output bytes/pointer differ")
                        view = np.ctypeslib.as_array(ct.cast(output.buf, ct.POINTER(ct.c_float)),
                                                    shape=(math.prod(row["shape"]),))
                        values[row["name"]] = view.copy().reshape(row["shape"])
                    copy_end = time.monotonic()
                    record["timing"] = dict(pack_seconds=packed_end-begin,
                        input_set_seconds=set_end-packed_end, run_api_seconds=run_end-set_end,
                        outputs_get_seconds=get_end-run_end, owned_copy_seconds=copy_end-get_end,
                        frame_id=int(ext.frame_id), input_records=list(packed.records), synchronous=True)
                finally:
                    if got:
                        release_begin = time.monotonic()
                        code = int(api.rknn_outputs_release(ctx, 3, out))
                        record.setdefault("timing", {})["outputs_release_seconds"] = time.monotonic()-release_begin
                        self._metrics["SDK_outputs_releases"] += 1
                        record["outputs_released"] = code == 0
                        if code:
                            self._metrics["cleanup_errors"].append("rknn_outputs_release " + str(code))
                            raise RuntimeError("rknn_outputs_release returned " + str(code))
                    record.setdefault("timing", {})["forward_seconds"] = time.monotonic()-begin
            try:
                if self._capture_logs:
                    try:
                        with _native_log(self._log_directory) as native:
                            run()
                    finally:
                        record["native_log"] = native
                    require(not native["truncated"] and not native["RKNN_error_detected"],
                            "Native SDK emitted E RKNN or an oversized log")
                else:
                    run()
                verification_begin = time.monotonic()
                recheck_pins(self._pins)
                record.update(execution_status="completed", outputs_finite=all(np.isfinite(x).all() for x in values.values()),
                              weight_domain_valid=bool(((values["weight"] >= 0) & (values["weight"] <= 1)).all()))
                record["outputs_finite"] = bool(record["outputs_finite"])
                if reference is not None:
                    require(set(reference) == set(OUTPUT_NAMES), "Reference operands differ")
                    record["versus_reference"] = {k: compare(v, reference[k]) for k, v in values.items()}
                    record["strict_FP32_gate_passed"] = all(x["gate_passed"] for x in record["versus_reference"].values())
                else:
                    record["strict_FP32_gate_passed"] = None
                record["all_outputs_zero"] = bool(all(not np.any(x) for x in values.values()))
                record["strict_component_gate_passed"] = (record["outputs_finite"] and record["weight_domain_valid"]
                    and record["strict_FP32_gate_passed"] is True and not record["all_outputs_zero"])
                record["numerical_status"] = ("strict_component_gate_passed_not_adopted_not_VO" if record["strict_component_gate_passed"] else "numerical_rejected_or_reference_missing")
                record["verification_seconds"] = time.monotonic()-verification_begin
                if self._profile:
                    record["profile"] = self.abi.perf_detail(self.api, self.context)
                self._metrics["completed_calls"] += 1
            except BaseException as exc:
                record.update(execution_status="failed", error=repr(exc))
                self.close()
            # Callers always receive copied raw values, including invalid output,
            # so they can preserve failed NPZ without putting NaN into JSON.
            record["total_execute_seconds"] = time.monotonic()-total_begin
            record["timing_scope"] = ("forward_seconds includes owned input packing/value checks, SDK set/run/get, owned FP32 copy and release; "
                "total_execute_seconds additionally includes native log handling, source rechecks, numerical/reference checks and profile query or failure cleanup. "
                "Large external provenance may be checked once before/after the experiment by the caller; only session pins are rechecked per call.")
            return values, record

    def close(self):
        with self._lock:
            if self._metrics["closed"]:
                return
            self._metrics["closed"] = True
            try:
                code = int(self.api.rknn_destroy(self.context))
                self._metrics["context_destroyed"] = code == 0
                if code:
                    self._metrics["cleanup_errors"].append("rknn_destroy " + str(code))
            except Exception as exc:
                self._metrics["cleanup_errors"].append(repr(exc))
            try:
                self._metrics["source_hashes_after"] = recheck_pins(self._pins)
                self._metrics["sources_and_inputs_unchanged"] = True
            except Exception as exc:
                self._metrics["sources_and_inputs_unchanged"] = False
                self._metrics["cleanup_errors"].append(repr(exc))

    def metrics(self):
        with self._lock:
            return copy.deepcopy(self._metrics)

