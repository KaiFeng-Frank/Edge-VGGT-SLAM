"""Explicit shared-weight RKNN backend for the original four-step CameraHead.

The fixed graph accepts CPU FP32 raw camera tokens [1,2,2048] and an
unactivated conditioning pose [1,2,9]. One context executes all four original
iterations; FP32 recurrence and the original linear/linear/ReLU activation
remain on the CPU. Unsupported inputs and SDK errors never fall back to the
old CPU head. Installation verifies all 69 original parameter tensors before
replacing the head, and deliberately retains no reference to its CPU weights.

Timing covers synchronous SDK IO, including output copy and release. Successful
SDK calls establish NPU use, not that every internal operator executes there
or that the whole mapping/tracking pipeline meets a latency target.
"""
from __future__ import annotations

import copy
import ctypes as ct
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import threading
import time

import numpy as np
import torch
from torch import nn


class RKNNCameraError(RuntimeError):
    """An explicit CameraHead deployment requirement or SDK operation failed."""


_RUNTIME_SHA = "dbe4fc933891581099084a569e2e847e15c582082680bf31bcf5dacc86210109"
_ABI_SHA = "37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608"
_EXPORT_SHA = "b26efe9c6cd62a2b7977e6e9c096cf0bd9bbf63076a063b9a332ef59565af258"
_COMPILE_SHA = "1fd295bc2b42bdbf51fddb5a277b93a77d72b6131868763412e0d588ce427690"
_HEADER_SHA = "c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a"
_LIBRARY_SHA = "d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8"
_ROOT = Path(__file__).resolve().parents[1]
_MAX_MODEL_BYTES = 1 << 30


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(path, expected):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"Explicit lowercase SHA256 required: {path}")
    if _sha(path) != expected:
        raise ValueError(f"SHA256 mismatch: {path}")
    return expected


def _load_helpers():
    directory = _ROOT / "deploy/rk3588/validation"
    path = directory / "run-v16-camera-step-board.py"
    _require_sha(path, _RUNTIME_SHA)
    spec = importlib.util.spec_from_file_location("_rknn_camera_frozen_step_runtime", path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    base, abi_path = runtime.load_frozen("run-v12-rknn-block-board.py", _ABI_SHA,
                                       "_rknn_camera_frozen_ABI")
    step, step_path = runtime.load_frozen("export-v16-camera-step-board.py", _EXPORT_SHA,
                                        "_rknn_camera_frozen_step_contract")
    _, old_path, capture, capture_path = step.load_frozen()
    original, original_path = capture.base_capture()
    header = directory.parent / "reference/rknn_api_v2.3.2.h"
    _require_sha(header, _HEADER_SHA)
    pins = {str(p): _sha(p) for p in
            (path, abi_path, step_path, old_path, capture_path, original_path, header)}
    return runtime, base, step, capture, original, pins


def _tensor_info(value):
    if value.device.type != "cpu" or value.dtype != torch.float32 or value.layout != torch.strided:
        raise ValueError("Original camera parameters must be dense CPU FP32 tensors")
    if not value.is_contiguous():
        raise ValueError("Original camera parameter storage must be contiguous")
    array = value.detach().numpy()
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite original camera parameter")
    return {"shape": list(array.shape), "dtype": "float32", "finite": True,
            "bytes": int(array.nbytes),
            "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest()}


def _verify_original_head(head, cap, exported, capture):
    if not isinstance(head, nn.Module) or head.training:
        raise ValueError("Install on the original eval CameraHead before any CPU Linear replacement")
    if capture.camera_contract(head) != cap.get("architecture") or exported.get("architecture") != cap.get("architecture"):
        raise ValueError("Original CameraHead architecture/epsilon mismatch")
    expected = cap["artifacts"]["camera-state.npz"]["tensors"]
    state = head.state_dict()
    if len(state) != 69 or set(state) != set(expected) or "empty_pose_tokens" not in state:
        raise ValueError("All 69 original CameraHead parameters are required")
    if any(_tensor_info(value) != expected[name] for name, value in state.items()):
        raise ValueError("Original CameraHead parameter raw SHA/shape mismatch")
    if sum(item["bytes"] for item in expected.values()) != cap.get("unique_state_bytes"):
        raise ValueError("Original CameraHead parameter byte count mismatch")
    bindings = exported["weight_sharing"]["parameters"]
    names = [item["parameter"] for item in bindings]
    if len(names) != 68 or len(set(names)) != 68 or set(names) != set(state) - {"empty_pose_tokens"}:
        raise ValueError("Exactly 68 unique original graph parameter bindings required")
    for item in bindings:
        original = expected[item["parameter"]]
        if item["shape"] != original["shape"] or item["raw_sha256"] != original["sha256"]:
            raise ValueError("Shared graph parameter identity differs from the original CPU head")
    empty = expected["empty_pose_tokens"]
    if (empty["shape"] != [1, 1, 9] or
            exported["outer_iteration_contract"]["empty_weight"] != empty or
            exported["weight_sharing"].get("original_unique_parameter_bytes") != cap["unique_state_bytes"] - 36):
        raise ValueError("Original learned empty-pose seed or shared-weight bytes mismatch")
    # This 72-byte expanded seed is the only original parameter value retained.
    return np.broadcast_to(state["empty_pose_tokens"].detach().numpy(), (1, 2, 9)).copy()


def _read_contract(model_path, export_receipt, compile_receipt, capture_receipt,
                   model_sha256, export_sha256, compile_sha256, capture_sha256,
                   head, helpers):
    runtime, base, step, capture, original, helper_pins = helpers
    paths = [Path(p).resolve(strict=True) for p in
             (model_path, export_receipt, compile_receipt, capture_receipt)]
    hashes = (model_sha256, export_sha256, compile_sha256, capture_sha256)
    for path, digest in zip(paths, hashes):
        _require_sha(path, digest)
    if not 0 < paths[0].stat().st_size <= _MAX_MODEL_BYTES:
        raise ValueError("Shared camera RKNN model exceeds the 1 GiB file budget")
    exported, compiled, cap = [json.loads(path.read_text()) for path in paths[1:]]
    if (cap.get("kind") != "original_complete_CameraHead_CPU_capture" or
            cap.get("status") != "completed" or cap.get("script_sha256") != step.CAPTURE_SHA or
            cap.get("camera_iterations") != 4 or cap.get("CPU_camera_forward_call_count") != 3 or
            cap.get("CPU_trunk_call_count") != 48 or cap.get("npu_used") is not False or
            cap.get("cleanup_errors") or cap.get("source_and_inputs_unchanged") is not True or
            cap.get("checkpoint_stat_unchanged") is not True or
            cap.get("checkpoint", {}).get("sha256") != capture.CHECKPOINT_SHA or
            cap.get("cpu_reference", {}).get("all_elementwise_equal") is not True or
            exported.get("status") != "exported_and_structurally_checked" or
            exported.get("script_sha256") != _EXPORT_SHA or exported.get("cleanup_errors") or
            exported.get("capture", {}).get("sha256") != capture_sha256 or
            exported.get("source_and_inputs_unchanged") is not True or
            exported.get("weight_sharing", {}).get("original_parameters_verified") != 68 or
            exported.get("weight_sharing", {}).get("weights_not_duplicated_per_iteration") is not True or
            exported.get("outer_iteration_contract", {}).get("iterations") != 4 or
            exported.get("outer_iteration_contract", {}).get("all_four_original_iterations_array_equal") is not True or
            exported.get("onnx", {}).get("both_dynamic_inputs_reach_output") is not True or
            compiled.get("status") != "compiled_not_runtime_validated" or
            compiled.get("wrapper_sha256") != _COMPILE_SHA or
            compiled.get("export_receipt_sha256") != export_sha256 or
            compiled.get("exporter_script_sha256") != _EXPORT_SHA or
            compiled.get("input", {}).get("sha256") != exported.get("onnx", {}).get("sha256") or
            compiled.get("output", {}).get("sha256") != model_sha256 or
            compiled.get("output", {}).get("size") != paths[0].stat().st_size or
            compiled.get("wrapper_and_frozen_compiler_unchanged") is not True or
            compiled.get("required_outer_iterations") != 4 or
            compiled.get("original_parameter_tensors_per_graph") != 68 or
            compiled.get("configuration", {}).get("target_platform") != "rk3588" or
            compiled.get("configuration", {}).get("float_dtype") != "float16" or
            compiled.get("configuration", {}).get("do_quantization") is not False):
        raise ValueError("Original capture/export/compiled camera SHA/provenance mismatch")
    sources = {**original.SOURCE_PINS, **capture.EXTRA_PINS}
    if cap.get("source_hashes") != sources:
        raise ValueError("Original camera/model source identity mismatch")
    for name, digest in sources.items():
        _require_sha(_ROOT / name, digest)
    seed = _verify_original_head(head, cap, exported, capture)
    provenance = {"files": {str(path): digest for path, digest in zip(paths, hashes)},
                  "helpers": helper_pins, "original_source_hashes": sources,
                  "checkpoint_sha256": capture.CHECKPOINT_SHA,
                  "verified_parameter_tensors": 69, "graph_parameter_tensors": 68,
                  "original_parameter_bytes": cap["unique_state_bytes"]}
    return paths[0], seed, provenance


class _RunTracker:
    """Count actual SDK success even if subsequent output retrieval fails."""
    def __init__(self, api, counters, hardware):
        self.api, self.counters, self.hardware = api, counters, hardware

    def __getattr__(self, name):
        return getattr(self.api, name)

    def rknn_run(self, *args):
        self.counters["sdk_run_attempts"] += 1
        begin = time.perf_counter()
        try:
            code = self.api.rknn_run(*args)
            if code == 0:
                self.counters["sdk_calls"] += 1
                if self.hardware:
                    self.counters["npu_used"] = True
            return code
        finally:
            self.counters["sdk_run_api_seconds"] += time.perf_counter() - begin


class RKNNCameraHead(nn.Module):
    """Installed inference-only head and resource handle; close does not restore CPU weights."""
    def __init__(self, model_path, seed, provenance, helpers, core_mask, library_path,
                 run_timeout_ms, max_dma_bytes, library):
        super().__init__()
        self._lock = threading.RLock()
        self._runtime, self._base, self._step, _, _, _ = helpers
        self._context = self._base.CONTEXT()
        self._api = None
        self._seed = seed
        self._timeout = run_timeout_ms
        self._metrics = {"backend": "rknn-camera-step", "camera_iterations": 4,
                         "input_shape": [1, 2, 2048], "pose_shape": [1, 2, 9],
                         "core_mask": core_mask, "provenance": provenance,
                         "library_is_mock": library is not None,
                         "hardware_verified": False, "npu_used": False,
                         "all_operators_on_npu_verified": False,
                         "CPU_recurrence_activation_explicit": True,
                         "closed": False, "failed": False, "last_error": None,
                         "cleanup_errors": [], "model_memory_bytes": None,
                         "current_dma_bytes": 0, "peak_dma_bytes": 0,
                         "npu_run_api_called": False, "npu_run_api_call_count": 0}
        self._metrics.update({name: 0 for name in
            ("attempted_heads", "completed_heads", "failed_heads", "sdk_run_attempts",
             "sdk_calls", "sdk_outputs_copied", "context_creates", "context_destroys",
             "context_destroy_attempts")})
        self._metrics.update({name: 0.0 for name in
            ("sdk_io_seconds", "sdk_run_api_seconds", "head_seconds",
             "cpu_recurrence_activation_seconds")})
        self.eval()  # Assigning a new child to an eval parent does not set its mode.
        try:
            if library is None:
                self._metrics["device_tree_compatible"] = self._base.require_board()
                path = Path(library_path).resolve(strict=True)
                self._metrics["runtime_library_sha256"] = _require_sha(path, _LIBRARY_SHA)
                library = ct.CDLL(str(path))
            raw_api = self._base.bind_api(library)
            self._api = _RunTracker(raw_api, self._metrics, not self._metrics["library_is_mock"])
            filename = ct.create_string_buffer(bytes(str(model_path), "utf-8"))
            code = self._api.rknn_init(ct.byref(self._context), ct.cast(filename, ct.c_void_p), 0, 0, None)
            if self._context.value:
                self._metrics["context_creates"] += 1
            self._base.check_call(code, "rknn_init")
            if not self._context.value:
                raise ValueError("RKNN returned an empty context")
            self._base.check_call(self._api.rknn_set_core_mask(self._context, core_mask), "rknn_set_core_mask")
            version = self._base.query(self._api, self._context, 5, self._base.SDKVersion())
            sdk_version = bytes(version.api_version).decode("utf-8")
            self._metrics.update(sdk_version=sdk_version,
                                 driver_version=bytes(version.drv_version).decode("utf-8"))
            if not re.match(r"^2\.3\.2(?:\D|$)", sdk_version):
                raise ValueError("RKNN runtime 2.3.2 is required by the frozen ABI")
            counts = self._base.query(self._api, self._context, 0, self._base.InputOutputNum())
            if (counts.n_input, counts.n_output) != (2, 1):
                raise ValueError("Camera step requires exactly two inputs and one output")
            inputs = [self._base.query(self._api, self._context, 1, self._base.TensorAttr(index=i)) for i in range(2)]
            output = self._base.query(self._api, self._context, 2, self._base.TensorAttr(index=0))
            self._metrics.update(input_attrs=[self._base.attr_record(a) for a in inputs],
                                 output_attr=self._base.attr_record(output), abi=self._base.abi_record())
            self._bindings = self._runtime.validate_bindings(self._base, inputs, output)
            self._metrics["actual_SDK_input_bindings"] = self._bindings
            memory = self._base.query(self._api, self._context, 6, self._base.MemSize())
            record = {name: int(getattr(memory, name)) for name, _ in memory._fields_ if name != "reserved"}
            self._metrics["model_memory_bytes"] = record
            dma = record["total_dma_allocated_size"]
            self._metrics.update(current_dma_bytes=dma, peak_dma_bytes=dma)
            if not 0 < dma <= max_dma_bytes:
                raise ValueError("Camera SDK DMA allocation exceeds its explicit budget or is unavailable")
            self._metrics["hardware_verified"] = not self._metrics["library_is_mock"]
        except Exception as exc:
            self._metrics.update(failed=True, last_error=f"{type(exc).__name__}: {exc}")
            self._close_locked()
            raise RKNNCameraError(f"Cannot install RKNN CameraHead: {exc}") from exc

    def forward(self, aggregated_tokens_list, num_iterations=4):
        with self._lock:
            if self._metrics["closed"]:
                raise RKNNCameraError("RKNN CameraHead is closed")
            self._metrics["attempted_heads"] += 1
            begin = time.perf_counter()
            try:
                if self.training or torch.is_grad_enabled():
                    raise ValueError("RKNN CameraHead requires eval and no_grad/inference_mode")
                if type(num_iterations) is not int or num_iterations != 4:
                    raise ValueError("All four original camera iterations are required")
                if not isinstance(aggregated_tokens_list, (list, tuple)) or not aggregated_tokens_list:
                    raise ValueError("Original aggregated token list is required")
                last = aggregated_tokens_list[-1]
                if (not isinstance(last, torch.Tensor) or last.device.type != "cpu" or
                        last.dtype != torch.float32 or last.layout != torch.strided or last.requires_grad or
                        last.ndim != 4 or tuple(last.shape[:2]) != (1, 2) or
                        last.shape[2] < 1 or last.shape[3] != 2048):
                    raise ValueError("Camera tokens require dense CPU FP32 [1,2,P>=1,2048]")
                raw = last[:, :, 0].detach().contiguous().numpy()
                if not np.isfinite(raw).all():
                    raise ValueError("Nonfinite raw camera tokens")
                prediction, poses = None, []
                for _ in range(4):
                    conditioning = self._seed if prediction is None else prediction
                    delta, row = self._runtime.execute_step(
                        self._base, self._api, self._context, raw, conditioning,
                        self._bindings, self._timeout, self._metrics["cleanup_errors"], self._metrics)
                    self._metrics["sdk_outputs_copied"] += 1
                    self._metrics["sdk_io_seconds"] += row["io_total_seconds"]
                    tick = time.perf_counter()
                    prediction = self._step.accumulate_delta(prediction, delta)
                    activated = self._step.activate_numpy_pose(prediction)
                    self._step.pose_array(activated)
                    poses.append(torch.from_numpy(activated))
                    self._metrics["cpu_recurrence_activation_seconds"] += time.perf_counter() - tick
                self._metrics["completed_heads"] += 1
                return poses
            except Exception as exc:
                self._metrics["failed_heads"] += 1
                self._metrics.update(failed=True, last_error=f"{type(exc).__name__}: {exc}")
                self._close_locked()
                raise RKNNCameraError(f"RKNN CameraHead aborted without CPU fallback: {exc}") from exc
            finally:
                self._metrics["head_seconds"] += time.perf_counter() - begin

    def _close_locked(self):
        if self._metrics["closed"]:
            return
        self._metrics["closed"] = True
        if self._context.value and self._api is not None:
            self._metrics["context_destroy_attempts"] += 1
            try:
                code = self._api.rknn_destroy(self._context)
                self._base.check_call(code, "rknn_destroy")
                self._metrics["context_destroys"] += 1
                self._metrics["current_dma_bytes"] = 0
            except Exception as exc:
                self._metrics["cleanup_errors"].append(f"{type(exc).__name__}: {exc}")
                self._metrics["failed"] = True
            finally:
                self._context = self._base.CONTEXT()
        self._seed = None

    def close(self):
        with self._lock:
            already_closed = self._metrics["closed"]
            self._close_locked()
            if not already_closed and self._metrics["cleanup_errors"]:
                raise RKNNCameraError("RKNN CameraHead cleanup failed: " + "; ".join(self._metrics["cleanup_errors"]))

    def metrics(self):
        with self._lock:
            return copy.deepcopy(self._metrics)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            if hasattr(self, "_lock"):
                with self._lock:
                    self._close_locked()
        except Exception:
            pass


def install_rknn_camera(model, model_path, export_receipt, compile_receipt, *,
                        model_sha256, export_sha256, compile_sha256,
                        capture_receipt, capture_sha256, core_mask=7,
                        library_path="/usr/lib/librknnrt.so", run_timeout_ms=30000,
                        max_dma_bytes=1 << 30, _library=None):
    """Verify original weights, create one context, then replace ``camera_head``.

    Call after CPU model loading/eval and before installing another Linear
    backend. ``close`` releases the context and leaves a closed replacement;
    the original CPU head is not kept for restoration. ``_library`` is solely
    for host fake-ABI tests and never reports NPU or hardware verification.
    """
    if type(core_mask) is not int or core_mask not in (1, 7):
        raise ValueError("Full-graph core_mask must be 1 or 7")
    if type(run_timeout_ms) is not int or not 0 < run_timeout_ms <= 600000:
        raise ValueError("Finite positive RKNN run timeout required (at most 600000 ms)")
    if type(max_dma_bytes) is not int or max_dma_bytes <= 0:
        raise ValueError("Positive explicit SDK DMA budget required")
    if not isinstance(model, nn.Module) or isinstance(getattr(model, "camera_head", None), RKNNCameraHead):
        raise ValueError("Install once on a CPU model with its original CameraHead")
    helpers = _load_helpers()
    path, seed, provenance = _read_contract(
        model_path, export_receipt, compile_receipt, capture_receipt,
        model_sha256, export_sha256, compile_sha256, capture_sha256,
        getattr(model, "camera_head", None), helpers)
    handle = RKNNCameraHead(path, seed, provenance, helpers, core_mask, library_path,
                            run_timeout_ms, max_dma_bytes, _library)
    try:
        model.camera_head = handle
    except Exception:
        handle.close()
        raise
    return handle
