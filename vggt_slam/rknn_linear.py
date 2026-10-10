"""Experimental RK3588 Linear offload using the RKNN 2.3.2 MatMul ABI.

This is an inference-only, explicit opt-in backend. It transfers CPU FP32
activations and FP16 weights to the NPU, runs FP16 x FP16 -> FP32 MatMul,
and returns CPU FP32 tensors. Other operators still execute in PyTorch.
The automated tests use an emulated library on the host. API symbols or a
successful context create alone do not establish correct hardware results,
whole-model conformance, or useful speed.

ABI sources are the official v2.3.2 headers in deploy/rk3588/reference.
Use K aligned to 32 elements and N to 16 elements for FP16. The header
comments describe byte alignment, but the deployment runtime rejected
K=16 and explicitly required K to align with 32. Normal weight layout is
the default; optional native weights are prepacked once on the CPU using
the documented RK3588 FP16 [N/16, K/32, 16, 32] ordering. A/C stay normal.
A compatible create still does not validate a run.
Only auto or individual-core masks are accepted. The target v2.3.2 runtime
rejected combined-core FP16 MatMul masks despite their enum declarations.
Contexts and their buffers are shared by shape, bounded, and serialized.
The byte limit and byte metrics cover reported IO buffers; SDK-internal
packing allocations are instead bounded by the number of live contexts.
No failed NPU operation falls back to a CPU Linear.

Usage::

    model.eval()
    with install_rknn_linear(model) as handle, torch.inference_mode():
        result = model(images)
        print(handle.metrics())

Do not mutate weights, move the model, or train it while the handle is active.
``close()`` restores the original modules and releases all NPU resources.
"""

from __future__ import annotations

from collections import OrderedDict
import ctypes as ct
import math
import platform
import threading
import time
from typing import Callable

import numpy as np
import torch
from torch import nn


class RKNNError(RuntimeError):
    """A required RKNN operation failed; CPU fallback is deliberately absent."""


class RKNNUnavailable(RKNNError):
    """The target architecture, runtime library, or required ABI is unavailable."""


# rknn_context is uint64_t on aarch64 (only 32-bit __arm__ uses uint32_t).
RKNNContext = ct.c_uint64
RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32 = 1
RKNN_TENSOR_FLOAT32 = 0
RKNN_TENSOR_FLOAT16 = 1
RKNN_MEMORY_SYNC_TO_DEVICE = 1
RKNN_MEMORY_SYNC_FROM_DEVICE = 2


class RKNNMatmulTensorAttr(ct.Structure):
    _fields_ = [
        ("name", ct.c_char * 256),
        ("n_dims", ct.c_uint32),
        ("dims", ct.c_uint32 * 16),
        ("size", ct.c_uint32),
        ("type", ct.c_int),
    ]


class RKNNMatmulIOAttr(ct.Structure):
    _fields_ = [("A", RKNNMatmulTensorAttr), ("B", RKNNMatmulTensorAttr),
                ("C", RKNNMatmulTensorAttr)]


class RKNNMatmulInfo(ct.Structure):
    _fields_ = [
        ("M", ct.c_int32), ("K", ct.c_int32), ("N", ct.c_int32),
        ("type", ct.c_int),
        ("B_layout", ct.c_int16), ("B_quant_type", ct.c_int16),
        ("AC_layout", ct.c_int16), ("AC_quant_type", ct.c_int16),
        ("iommu_domain_id", ct.c_int32), ("group_size", ct.c_int16),
        ("reserved", ct.c_int8 * 34),
    ]


class RKNNTensorMem(ct.Structure):
    _fields_ = [
        ("virt_addr", ct.c_void_p), ("phys_addr", ct.c_uint64),
        ("fd", ct.c_int32), ("offset", ct.c_int32),
        ("size", ct.c_uint32), ("flags", ct.c_uint32),
        ("priv_data", ct.c_void_p),
    ]


def _check(result: int, operation: str) -> None:
    if result != 0:
        raise RKNNError(f"{operation} returned {result}; NPU execution aborted")


class _API:
    def __init__(self, library_path: str, library=None):
        if ct.sizeof(ct.c_void_p) != 8:
            raise RKNNUnavailable("RKNN Linear requires the 64-bit aarch64 ABI")
        if library is None:
            if platform.system() != "Linux" or platform.machine().lower() not in {"aarch64", "arm64"}:
                raise RKNNUnavailable("RKNN Linear requires Linux aarch64 on RK3588; this host cannot execute it")
            try:
                library = ct.CDLL(library_path)
            except OSError as exc:
                raise RKNNUnavailable(f"Cannot load RKNN runtime {library_path}: {exc}") from exc
        self.library = library
        mem = ct.POINTER(RKNNTensorMem)
        signatures = {
            "rknn_matmul_create": ([ct.POINTER(RKNNContext), ct.POINTER(RKNNMatmulInfo),
                                    ct.POINTER(RKNNMatmulIOAttr)], ct.c_int),
            "rknn_matmul_destroy": ([RKNNContext], ct.c_int),
            "rknn_matmul_run": ([RKNNContext], ct.c_int),
            "rknn_matmul_set_io_mem": ([RKNNContext, mem, ct.POINTER(RKNNMatmulTensorAttr)], ct.c_int),
            "rknn_matmul_set_core_mask": ([RKNNContext, ct.c_int], ct.c_int),
            "rknn_create_mem": ([RKNNContext, ct.c_uint32], mem),
            "rknn_destroy_mem": ([RKNNContext, mem], ct.c_int),
            "rknn_mem_sync": ([RKNNContext, mem, ct.c_int], ct.c_int),
        }
        for name, (argtypes, restype) in signatures.items():
            try:
                function = getattr(library, name)
            except AttributeError as exc:
                raise RKNNUnavailable(f"RKNN runtime lacks required v2.3.2 symbol {name}") from exc
            function.argtypes = argtypes
            function.restype = restype
            setattr(self, name, function)


class _MatmulContext:
    def __init__(self, api: _API, shape: tuple[int, int, int], core_mask: int, weight_layout: str):
        self.api, self.shape = api, shape
        self.ctx = RKNNContext()
        self.attrs = RKNNMatmulIOAttr()
        self.memories = []
        self.views = {}
        self.loaded_weight = None
        info = RKNNMatmulInfo(M=shape[0], K=shape[1], N=shape[2],
                              type=RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32,
                              B_layout=1 if weight_layout == "native" else 0)
        try:
            _check(api.rknn_matmul_create(ct.byref(self.ctx), ct.byref(info), ct.byref(self.attrs)),
                   f"rknn_matmul_create(M={shape[0]}, K={shape[1]}, N={shape[2]}, FP16->FP32)")
            if not self.ctx.value:
                raise RKNNError("rknn_matmul_create returned a null context")
            _check(api.rknn_matmul_set_core_mask(self.ctx, core_mask), "rknn_matmul_set_core_mask")
            m, k, n = shape
            b_dims = (n // 16, k // 32, 16, 32) if weight_layout == "native" else (k, n)
            for name, dims, dtype in (("A", (m, k), RKNN_TENSOR_FLOAT16),
                                      ("B", b_dims, RKNN_TENSOR_FLOAT16),
                                      ("C", (m, n), RKNN_TENSOR_FLOAT32)):
                attr = getattr(self.attrs, name)
                expected_bytes = math.prod(dims) * (4 if dtype == RKNN_TENSOR_FLOAT32 else 2)
                if attr.n_dims != len(dims) or tuple(attr.dims[:attr.n_dims]) != dims or attr.type != dtype:
                    raise RKNNError(f"RKNN returned unsupported {weight_layout}-weight {name} attributes: "
                                    f"dims={list(attr.dims[:attr.n_dims])}, dtype={attr.type}")
                if attr.size < expected_bytes:
                    raise RKNNError(f"RKNN {name} buffer size {attr.size} is below {expected_bytes}")
            self.device_bytes = sum(getattr(self.attrs, name).size for name in "ABC")
        except BaseException as exc:
            try:
                self.close()
            except RKNNError as cleanup:
                exc.add_note(str(cleanup))
            raise

    def allocate(self) -> None:
        for name in "ABC":
            attr = getattr(self.attrs, name)
            pointer = self.api.rknn_create_mem(self.ctx, attr.size)
            if not pointer:
                raise RKNNError(f"rknn_create_mem({name}, {attr.size} bytes) returned null")
            self.memories.append((name, pointer))
            memory = pointer.contents
            if not memory.virt_addr or memory.size < attr.size:
                raise RKNNError(f"rknn_create_mem({name}) returned an invalid or undersized allocation")
            # C is written by the NPU. Do not dirty its CPU cache before
            # the first device write: old cache lines can overwrite output
            # on a noncoherent target. The official demo leaves C untouched.
            if name != "C":
                ct.memset(memory.virt_addr, 0, attr.size)
            raw = (ct.c_uint8 * attr.size).from_address(memory.virt_addr)
            dtype = np.float32 if attr.type == RKNN_TENSOR_FLOAT32 else np.float16
            dims = tuple(attr.dims[:attr.n_dims])
            self.views[name] = np.frombuffer(raw, dtype=dtype, count=math.prod(dims)).reshape(dims)
            self.bind(name)

    def bind(self, name: str) -> None:
        pointer = next(pointer for key, pointer in self.memories if key == name)
        attr = getattr(self.attrs, name)
        _check(self.api.rknn_matmul_set_io_mem(self.ctx, pointer, ct.byref(attr)),
               f"rknn_matmul_set_io_mem({name})")

    def sync(self, name: str, direction: int) -> None:
        pointer = next(pointer for key, pointer in self.memories if key == name)
        _check(self.api.rknn_mem_sync(self.ctx, pointer, direction), f"rknn_mem_sync({name}, {direction})")

    def close(self) -> None:
        errors = []
        self.views.clear()
        for name, pointer in reversed(self.memories):
            result = self.api.rknn_destroy_mem(self.ctx, pointer)
            if result != 0:
                errors.append(f"rknn_destroy_mem({name}) returned {result}")
        self.memories.clear()
        if self.ctx.value:
            result = self.api.rknn_matmul_destroy(self.ctx)
            if result != 0:
                errors.append(f"rknn_matmul_destroy returned {result}")
            self.ctx.value = 0
        if errors:
            raise RKNNError("; ".join(errors))


class _Pool:
    def __init__(self, api: _API, *, max_contexts: int, max_cache_bytes: int,
                 max_rows: int, core_mask: int, weight_layout: str):
        self.api = api
        self.max_contexts, self.max_cache_bytes = max_contexts, max_cache_bytes
        self.max_rows, self.core_mask = max_rows, core_mask
        self.weight_layout = weight_layout
        self.contexts = OrderedDict()
        self.lock = threading.RLock()
        self.closed = False
        self.stats = {"linear_calls": 0, "npu_calls": 0, "npu_run_attempts": 0,
                      "npu_run_seconds": 0.0, "linear_seconds": 0.0, "npu_errors": 0,
                      "context_creates": 0, "context_evictions": 0,
                      "peak_contexts": 0, "peak_device_bytes": 0, "weight_upload_bytes": 0,
                      "input_rows": 0, "padded_rows": 0}

    def _device_bytes(self) -> int:
        return sum(context.device_bytes for context in self.contexts.values())

    def _evict(self) -> None:
        _, context = self.contexts.popitem(last=False)
        context.close()
        self.stats["context_evictions"] += 1

    def _context(self, shape: tuple[int, int, int]) -> _MatmulContext:
        if shape in self.contexts:
            self.contexts.move_to_end(shape)
            return self.contexts[shape]
        m, k, n = shape
        expected = 2 * m * k + 2 * k * n + 4 * m * n
        if expected > self.max_cache_bytes:
            raise RKNNError(f"MatMul {shape} requires at least {expected} device bytes, "
                            f"above max_cache_bytes={self.max_cache_bytes}")
        while self.contexts and (len(self.contexts) >= self.max_contexts or
                                  self._device_bytes() + expected > self.max_cache_bytes):
            self._evict()
        context = _MatmulContext(self.api, shape, self.core_mask, self.weight_layout)
        try:
            if context.device_bytes > self.max_cache_bytes:
                raise RKNNError(f"RKNN allocations exceed max_cache_bytes={self.max_cache_bytes}")
            while self.contexts and self._device_bytes() + context.device_bytes > self.max_cache_bytes:
                self._evict()
            context.allocate()
        except BaseException as exc:
            try:
                context.close()
            except RKNNError as cleanup:
                exc.add_note(str(cleanup))
            raise
        self.contexts[shape] = context
        self.stats["context_creates"] += 1
        self.stats["peak_contexts"] = max(self.stats["peak_contexts"], len(self.contexts))
        self.stats["peak_device_bytes"] = max(self.stats["peak_device_bytes"], self._device_bytes())
        return context

    def linear(self, inputs: torch.Tensor, layer: "_RKNNLinear") -> torch.Tensor:
        start = time.perf_counter()
        with self.lock:
            if self.closed:
                raise RKNNError("RKNN Linear handle is closed")
            self.stats["linear_calls"] += 1
            try:
                flat = inputs.detach().reshape(-1, layer.in_features).numpy()
                output = np.empty((flat.shape[0], layer.out_features), dtype=np.float32)
                for offset in range(0, len(flat), self.max_rows):
                    block = flat[offset:offset + self.max_rows]
                    rows = 1 << (len(block) - 1).bit_length()
                    context = self._context((rows, layer.padded_k, layer.padded_n))
                    if context.loaded_weight is not layer.weight_token:
                        np.copyto(context.views["B"], layer.packed_weight)
                        context.sync("B", RKNN_MEMORY_SYNC_TO_DEVICE)
                        # Normal B is packed/cached at bind time. Retain a
                        # rebind for native B too: the static official demo
                        # does not establish that dynamic B can skip it.
                        context.bind("B")
                        context.loaded_weight = layer.weight_token
                        self.stats["weight_upload_bytes"] += layer.packed_weight.nbytes
                    context.views["A"].fill(0)
                    with np.errstate(over="raise", invalid="raise"):
                        np.copyto(context.views["A"][:len(block), :layer.in_features], block, casting="unsafe")
                    context.sync("A", RKNN_MEMORY_SYNC_TO_DEVICE)
                    # Match the official demo's copy-before-bind sequence;
                    # refresh normal-layout input conversion every block.
                    context.bind("A")
                    self.stats["npu_run_attempts"] += 1
                    run_start = time.perf_counter()
                    try:
                        result = self.api.rknn_matmul_run(context.ctx)
                    finally:
                        self.stats["npu_run_seconds"] += time.perf_counter() - run_start
                    _check(result, "rknn_matmul_run")
                    self.stats["npu_calls"] += 1
                    context.sync("C", RKNN_MEMORY_SYNC_FROM_DEVICE)
                    output[offset:offset + len(block)] = context.views["C"][:len(block), :layer.out_features]
                    self.stats["input_rows"] += len(block)
                    self.stats["padded_rows"] += rows - len(block)
                result_tensor = torch.from_numpy(output).reshape(*inputs.shape[:-1], layer.out_features)
                if layer.bias is not None:
                    result_tensor.add_(layer.bias.detach())
                return result_tensor
            except BaseException:
                self.stats["npu_errors"] += 1
                raise
            finally:
                self.stats["linear_seconds"] += time.perf_counter() - start

    def metrics(self) -> dict:
        with self.lock:
            return {**self.stats, "npu_used": self.stats["npu_calls"] > 0,
                    "active_contexts": len(self.contexts), "device_bytes": self._device_bytes(),
                    "max_contexts": self.max_contexts, "max_cache_bytes": self.max_cache_bytes,
                    "max_rows": self.max_rows, "core_mask": self.core_mask,
                    "weight_layout": self.weight_layout, "closed": self.closed}

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
            errors = []
            for context in self.contexts.values():
                try:
                    context.close()
                except RKNNError as exc:
                    errors.append(str(exc))
            self.contexts.clear()
            if errors:
                raise RKNNError("; ".join(errors))


class _RKNNLinear(nn.Module):
    def __init__(self, original: nn.Linear, pool: _Pool):
        super().__init__()
        self.weight = original.weight
        self.bias = original.bias
        self.in_features, self.out_features = original.in_features, original.out_features
        self.padded_k = (self.in_features + 31) // 32 * 32
        self.padded_n = (self.out_features + 15) // 16 * 16
        if self.padded_k > 10240:
            raise ValueError(f"Linear K={self.padded_k} exceeds the RKNN MatMul limit 10240")
        if pool.weight_layout == "native" and self.padded_k > 8192:
            raise ValueError("Native RK3588 weight packing supports K<=8192; larger K requires SDK segmented packing")
        self.pool = pool
        self.weight_token = object()
        self.versions = (self.weight._version, self.bias._version if self.bias is not None else None)
        normal_weight = np.zeros((self.padded_k, self.padded_n), dtype=np.float16)
        with np.errstate(over="raise", invalid="raise"):
            np.copyto(normal_weight[:self.in_features, :self.out_features],
                      self.weight.detach().numpy().T, casting="unsafe")
        if pool.weight_layout == "native":
            self.packed_weight = np.ascontiguousarray(
                normal_weight.reshape(self.padded_k // 32, 32, self.padded_n // 16, 16).transpose(2, 0, 3, 1))
        else:
            self.packed_weight = normal_weight
        self.train(original.training)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.training:
            raise RKNNError("RKNN Linear is inference-only; call model.eval()")
        if torch.is_grad_enabled() and (inputs.requires_grad or self.weight.requires_grad or
                                       (self.bias is not None and self.bias.requires_grad)):
            raise RKNNError("RKNN Linear does not support autograd; use torch.inference_mode() or no_grad()")
        if inputs.device.type != "cpu" or inputs.dtype != torch.float32:
            raise RKNNError("RKNN Linear requires CPU float32 input tensors")
        if self.weight.device.type != "cpu" or self.weight.dtype != torch.float32:
            raise RKNNError("RKNN Linear weights moved or changed dtype; restore and reinstall the backend")
        if self.bias is not None and (self.bias.device.type != "cpu" or self.bias.dtype != torch.float32):
            raise RKNNError("RKNN Linear bias moved or changed dtype; restore and reinstall the backend")
        if inputs.ndim == 0 or inputs.shape[-1] != self.in_features:
            raise RKNNError(f"RKNN Linear expects a final dimension of {self.in_features}")
        versions = (self.weight._version, self.bias._version if self.bias is not None else None)
        if versions != self.versions:
            raise RKNNError("RKNN Linear weights changed after FP16 packing; restore and reinstall the backend")
        return self.pool.linear(inputs, self)


class RKNNLinearHandle:
    """Owns replacements and a bounded NPU context pool. Close explicitly."""

    def __init__(self, pool: _Pool, replacements: list, skipped: list[str]):
        self.pool, self.replacements, self.skipped = pool, replacements, skipped
        self.replaced_names = [name for _, _, _, _, name in replacements]
        self.host_weight_bytes = sum(wrapper.packed_weight.nbytes for wrapper in
                                     {id(item[3]): item[3] for item in replacements}.values())
        self.restored = False

    def restore(self) -> None:
        with self.pool.lock:
            if self.restored:
                return
            for parent, key, original, wrapper, _ in self.replacements:
                if parent._modules.get(key) is not wrapper:
                    raise RKNNError(f"Cannot restore replaced module {key}: it was changed by the caller")
            for parent, key, original, _, _ in self.replacements:
                parent._modules[key] = original
            self.restored = True

    def close(self) -> None:
        errors = []
        try:
            self.restore()
        except RKNNError as exc:
            errors.append(str(exc))
        try:
            self.pool.close()
        except RKNNError as exc:
            errors.append(str(exc))
        for _, _, _, wrapper, _ in self.replacements:
            wrapper.packed_weight = None
        self.replacements.clear()
        self.host_weight_bytes = 0
        if errors:
            raise RKNNError("; ".join(errors))

    def metrics(self) -> dict:
        with self.pool.lock:
            return {**self.pool.metrics(), "experimental": True, "backend": "rknn_matmul_fp16",
                    "npu_seconds": self.pool.stats["npu_run_seconds"],
                    "replaced_modules": len(self.replaced_names), "replaced_module_names": list(self.replaced_names),
                    "skipped_modules": list(self.skipped), "fp16_weight_bytes": self.host_weight_bytes,
                    "context_cache_bytes": self.pool._device_bytes(), "restored": self.restored}

    def __enter__(self) -> "RKNNLinearHandle":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            self.close()
        except RKNNError as cleanup:
            if exc is None:
                raise
            exc.add_note(str(cleanup))
        return False


def install_rknn_linear(
    model: nn.Module, *, library_path: str = "/usr/lib/librknnrt.so",
    min_weight_elements: int = 1024 * 1024,
    max_contexts: int = 4, max_cache_bytes: int = 256 * 1024 * 1024,
    max_rows: int = 1024, core_mask: int = 0, weight_layout: str = "normal",
    predicate: Callable[[str, nn.Linear], bool] | None = None,
    _library=None,
) -> RKNNLinearHandle:
    """Replace selected Linear children, returning a restore/close handle.

    ``predicate(name, layer)`` can further restrict selection. The explicit
    weight threshold leaves small Linear modules executing their original
    PyTorch implementation; skipped names are included in metrics. Large M
    is split into ``max_rows`` blocks, with the last block padded to a power
    of two. ``max_rows`` must itself be a power of two. Weights are packed
    once on the host; at most ``max_contexts`` NPU contexts share them by
    (M, K, N) shape instead of retaining NPU buffers for every layer.
    ``weight_layout='native'`` opts into documented RK3588 FP16 4D weight
    packing, avoiding SDK normal-weight conversion on each layer switch.
    Native K>8192 is explicitly unsupported by this NumPy packing route.

    ``_library`` is a private injection point for host ABI/memory tests. Do
    not use it to report real NPU execution. This function does not download
    models, compile RKNN graphs, or initialize a MatMul context until a call.
    """
    if model.training:
        raise ValueError("Set model.eval() before installing RKNN Linear")
    if min_weight_elements < 1 or max_contexts < 1 or max_cache_bytes < 1:
        raise ValueError("Linear threshold and context limits must be positive")
    if max_rows < 1 or max_rows & (max_rows - 1):
        raise ValueError("max_rows must be a positive power of two")
    if core_mask not in {0, 1, 2, 4}:
        raise ValueError("RKNN Linear supports core masks 0 (auto), 1, 2, or 4 only; "
                         "the target v2.3.2 FP16 MatMul runtime rejected combined-core masks")
    if weight_layout not in {"normal", "native"}:
        raise ValueError("weight_layout must be 'normal' or 'native'")
    selected, skipped = [], []
    for name, layer in model.named_modules(remove_duplicate=False):
        if not isinstance(layer, nn.Linear):
            continue
        if not name:
            raise ValueError("Wrap a root nn.Linear in a containing module before installing RKNN Linear")
        if layer.weight.numel() < min_weight_elements or (predicate is not None and not predicate(name, layer)):
            skipped.append(name)
            continue
        if layer.training or layer.weight.device.type != "cpu" or layer.weight.dtype != torch.float32:
            raise ValueError(f"Selected Linear {name} must be in eval mode with CPU float32 weights")
        if layer.bias is not None and (layer.bias.device.type != "cpu" or layer.bias.dtype != torch.float32):
            raise ValueError(f"Selected Linear {name} must have CPU float32 bias")
        parent_name, _, key = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        selected.append((parent, key, layer, name))
    if not selected:
        raise ValueError("No Linear modules match the requested RKNN selection")
    api = _API(library_path, _library)
    pool = _Pool(api, max_contexts=max_contexts, max_cache_bytes=max_cache_bytes,
                 max_rows=max_rows, core_mask=core_mask, weight_layout=weight_layout)
    replacements, wrappers = [], {}
    # Prepare every host pack before mutating the model, so a failed install
    # leaves all original modules in place and creates no NPU resources.
    for parent, key, original, name in selected:
        if id(original) not in wrappers:
            wrappers[id(original)] = _RKNNLinear(original, pool)
        replacements.append((parent, key, original, wrappers[id(original)], name))
    handle = RKNNLinearHandle(pool, replacements, skipped)
    for parent, key, _, wrapper, _ in replacements:
        parent._modules[key] = wrapper
    return handle
