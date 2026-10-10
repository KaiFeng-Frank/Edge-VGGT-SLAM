"""Opt-in CPU Linear backend for the XNNPACK pinned by Torch 2.3.1.

Each selected layer owns one static-weight operator; all operators share one
serialized pthreadpool and reusable IO buffers. Inputs, returned tensors, and
bias addition are CPU float32. The optional float16 path converts activations
and packed weights to FP16 and uses ARM NEON FP16 *accumulation*, not FP32
accumulation. It needs numerical validation on the intended model/data.

The public ABI is checked against the exact installed Torch 2.3.1 headers.
Sources are retained in deploy/rk3588/reference. Host tests inject an emulated
library; those calls do not establish native execution or numerical accuracy.
There is no fallback after a selected layer fails. Use inference_mode/no_grad,
do not change parameters while installed, and close the returned handle.
Closing restores original modules and releases only our operators/threadpool;
it never deinitializes Torch's shared XNNPACK global state.
"""

from __future__ import annotations

import ctypes as ct
import hashlib
import math
import os
from pathlib import Path
import threading
import time

import numpy as np
import torch
from torch import nn


XNNPACK_COMMIT = "fcbf55af6cf28a4627bcd1f703ab7ad843f0f3a2"
PTHREADPOOL_COMMIT = "4fe0e1e183925bf8cfa6aae24237e724a96479b8"
HEADER_SHA256 = {
    "xnnpack.h": "62a48ebae3f2d77c41989eadf4a6a3c97a68af9126777323da73e9059e728b02",
    "pthreadpool.h": "afae65ccebb810dde2a3370875df42f89ef2e43502a3430208d7b6e980f4b74c",
}
XNN_EXTRA_BYTES = 16
XNN_FLAG_FP32_STATIC_WEIGHTS = 0x8
XNN_FLAG_YIELD_WORKERS = 0x10
STATUS_NAMES = {0: "success", 1: "uninitialized", 2: "invalid_parameter",
                3: "invalid_state", 4: "unsupported_parameter", 5: "unsupported_hardware",
                6: "out_of_memory", 7: "reallocation_required"}


class XNNPACKError(RuntimeError):
    """A selected XNNPACK operation failed; no PyTorch fallback is performed."""


class XNNPACKUnavailable(XNNPACKError):
    """The pinned ABI, library, or required native functionality is unavailable."""


class XNNPACKFiniteError(XNNPACKError):
    """An activation/parameter/output is nonfinite or cannot fit in FP16."""


def _verify_torch_abi() -> dict:
    if torch.__version__.split("+")[0] != "2.3.1" or ct.sizeof(ct.c_void_p) != 8:
        raise XNNPACKUnavailable("XNNPACK Linear requires 64-bit Torch 2.3.1 with the pinned headers")
    root = Path(torch.__file__).resolve().parent
    headers = {}
    for name, expected in HEADER_SHA256.items():
        path = root / "include" / name
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise XNNPACKUnavailable(f"Cannot verify XNNPACK ABI header {path}: {exc}") from exc
        if actual != expected:
            raise XNNPACKUnavailable(f"Unverified ABI: {name} SHA256 {actual} != {expected}")
        headers[name] = {"path": str(path), "sha256": actual}
    return {"abi_verified": True, "headers": headers, "torch_root": str(root)}


def _verify_library_file(path, torch_root: str) -> dict:
    """Only the verified Torch binary or a byte-identical copy is accepted."""
    canonical = Path(torch_root) / "lib" / "libtorch_cpu.so"
    try:
        candidate = Path(os.fspath(path)).resolve()
        if candidate.samefile(canonical):
            return {"library": str(candidate), "library_binary_verified": "installed_torch_file"}
        if candidate.stat().st_size != canonical.stat().st_size:
            raise XNNPACKUnavailable("Custom XNNPACK library differs from installed Torch libtorch_cpu.so")

        def sha256(file):
            digest = hashlib.sha256()
            with file.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()

        expected, actual = sha256(canonical), sha256(candidate)
        if actual != expected:
            raise XNNPACKUnavailable("Custom XNNPACK library is not a byte-identical Torch libtorch_cpu.so copy")
        return {"library": str(candidate), "library_binary_verified": "byte_identical_copy", "library_sha256": actual}
    except (OSError, TypeError) as exc:
        raise XNNPACKUnavailable(f"Cannot verify XNNPACK library file against installed Torch: {exc}") from exc


class _API:
    def __init__(self, library):
        self.mock = library is not None and not isinstance(library, (str, os.PathLike, ct.CDLL))
        if self.mock:
            self.metadata = {"abi_verified": False, "library_is_mock": True,
                             "library": type(library).__name__, "headers": {}}
        else:
            self.metadata = _verify_torch_abi()
            if library is None:
                library = Path(self.metadata["torch_root"]) / "lib" / "libtorch_cpu.so"
            path = library._name if isinstance(library, ct.CDLL) else library
            self.metadata.update(_verify_library_file(path, self.metadata["torch_root"]))
            if not isinstance(library, ct.CDLL):
                try:
                    library = ct.CDLL(self.metadata["library"])
                except OSError as exc:
                    raise XNNPACKUnavailable(f"Cannot load Torch XNNPACK library: {exc}") from exc
            self.metadata.update(library=str(library._name), library_is_mock=False)
        self.library = library
        void, size, fp = ct.c_void_p, ct.c_size_t, ct.POINTER(ct.c_float)
        signatures = {
            "xnn_initialize": ([void], ct.c_int),
            "xnn_create_fully_connected_nc_f16": (
                [size, size, size, size, void, void, ct.c_float, ct.c_float,
                 ct.c_uint32, void, void, ct.POINTER(void)], ct.c_int),
            "xnn_reshape_fully_connected_nc_f16": ([void, size, void], ct.c_int),
            "xnn_setup_fully_connected_nc_f16": ([void, void, void], ct.c_int),
            "xnn_create_fully_connected_nc_f32": (
                [size, size, size, size, fp, fp, ct.c_float, ct.c_float,
                 ct.c_uint32, void, void, ct.POINTER(void)], ct.c_int),
            "xnn_reshape_fully_connected_nc_f32": ([void, size, void], ct.c_int),
            "xnn_setup_fully_connected_nc_f32": ([void, fp, fp], ct.c_int),
            "xnn_run_operator": ([void, void], ct.c_int),
            "xnn_delete_operator": ([void], ct.c_int),
            "pthreadpool_create": ([size], void),
            "pthreadpool_get_threads_count": ([void], size),
            "pthreadpool_destroy": ([void], None),
        }
        for name, (arguments, result) in signatures.items():
            try:
                function = getattr(library, name)
                function.argtypes, function.restype = arguments, result
            except (AttributeError, TypeError) as exc:
                raise XNNPACKUnavailable(f"XNNPACK library lacks pinned ABI symbol {name}") from exc
            setattr(self, name, function)

    def check(self, name: str, *args) -> None:
        status = getattr(self, name)(*args)
        if status != 0:
            raise XNNPACKError(f"{name} returned {status} ({STATUS_NAMES.get(status, 'unknown')}); "
                               "XNNPACK execution aborted")


def _pointer(array: np.ndarray, dtype: str):
    return array.ctypes.data_as(ct.POINTER(ct.c_float) if dtype == "float32" else ct.c_void_p)


class _Engine:
    def __init__(self, api: _API, threads: int, dtype: str, cpu_affinity):
        self.api, self.dtype = api, dtype
        self.numpy_dtype = np.dtype(dtype)
        self.lock = threading.RLock()
        self.closed = False
        self.pool = ct.c_void_p()
        self.operators = []
        self.input_buffer = self.output_buffer = None
        self.threads_requested, self.threads_actual = threads, 0
        self.cpu_affinity = cpu_affinity
        self.stats = {"linear_calls": 0, "xnnpack_calls": 0, "xnnpack_run_attempts": 0,
                      "xnnpack_seconds": 0.0, "linear_seconds": 0.0, "packing_seconds": 0.0,
                      "input_copy_seconds": 0.0, "reshape_setup_seconds": 0.0,
                      "output_copy_bias_seconds": 0.0, "operator_creates": 0,
                      "operator_deletes": 0, "errors": 0, "finite_failures": 0,
                      "input_rows": 0, "packed_weight_bytes_estimated": 0,
                      "peak_packed_weight_bytes_estimated": 0, "peak_scratch_bytes": 0,
                      "last_error": None, "pool_create_seconds": 0.0}
        try:
            api.check("xnn_initialize", None)
            start = time.perf_counter()
            original_affinity = None
            try:
                if cpu_affinity is not None:
                    original_affinity = os.sched_getaffinity(0)
                    os.sched_setaffinity(0, cpu_affinity)
                    actual_affinity = os.sched_getaffinity(0)
                    if actual_affinity != set(cpu_affinity):
                        raise XNNPACKUnavailable(f"pthreadpool CPU affinity was restricted to {sorted(actual_affinity)}, "
                                                f"requested {list(cpu_affinity)}")
                pointer = api.pthreadpool_create(threads)
                self.pool = pointer if isinstance(pointer, ct.c_void_p) else ct.c_void_p(pointer)
            finally:
                if original_affinity is not None:
                    os.sched_setaffinity(0, original_affinity)
                self.stats["pool_create_seconds"] += time.perf_counter() - start
            if not self.pool.value:
                raise XNNPACKUnavailable("pthreadpool_create returned NULL")
            self.threads_actual = int(api.pthreadpool_get_threads_count(self.pool))
            if self.threads_actual != threads:
                raise XNNPACKUnavailable(f"pthreadpool has {self.threads_actual} threads, requested {threads}")
        except BaseException as exc:
            try:
                self.close()
            except XNNPACKError as cleanup:
                exc.add_note(str(cleanup))
            raise

    def record_error(self, exc: BaseException) -> None:
        self.stats["errors"] += 1
        if isinstance(exc, XNNPACKFiniteError):
            self.stats["finite_failures"] += 1
        self.stats["last_error"] = f"{type(exc).__name__}: {exc}"

    def create_operator(self, layer: nn.Linear, name: str) -> ct.c_void_p:
        start, operator = time.perf_counter(), ct.c_void_p()
        try:
            # [N,K] is PyTorch's default weight ordering. Flags never request
            # transposition. A real 16-byte tail is provided for source reads.
            count = layer.weight.numel()
            storage = np.zeros(count + XNN_EXTRA_BYTES // 4, dtype=np.float32)
            source = layer.weight.detach().numpy()
            np.copyto(storage[:count].reshape(layer.out_features, layer.in_features), source)
            if not np.isfinite(storage[:count]).all():
                raise XNNPACKFiniteError(f"Linear {name} has nonfinite weights")
            if layer.bias is not None and not np.isfinite(layer.bias.detach().numpy()).all():
                raise XNNPACKFiniteError(f"Linear {name} has nonfinite bias")
            if self.dtype == "float16" and np.max(np.abs(storage[:count])) > np.finfo(np.float16).max:
                raise XNNPACKFiniteError(f"Linear {name} weights overflow FP16")
            flags = XNN_FLAG_YIELD_WORKERS
            if self.dtype == "float16":
                flags |= XNN_FLAG_FP32_STATIC_WEIGHTS
            suffix = "f16" if self.dtype == "float16" else "f32"
            kernel = storage.ctypes.data_as(ct.c_void_p if suffix == "f16" else ct.POINTER(ct.c_float))
            # Flag 8 specifies FP32 source weights (and source bias if given).
            # Bias is deliberately NULL in both paths and added later in FP32.
            self.api.check(f"xnn_create_fully_connected_nc_{suffix}",
                           layer.in_features, layer.out_features, layer.in_features, layer.out_features,
                           kernel, None, -math.inf, math.inf, flags, None, None, ct.byref(operator))
            if not operator.value:
                raise XNNPACKError(f"XNNPACK returned NULL operator for {name} on success")
            self.operators.append(operator)
            self.stats["operator_creates"] += 1
            estimate = (count + layer.out_features) * self.numpy_dtype.itemsize
            self.stats["packed_weight_bytes_estimated"] += estimate
            self.stats["peak_packed_weight_bytes_estimated"] = self.stats["packed_weight_bytes_estimated"]
            return operator
        except BaseException as exc:
            self.record_error(exc)
            if operator.value and operator not in self.operators:
                try:
                    self.api.check("xnn_delete_operator", operator)
                except XNNPACKError as cleanup:
                    exc.add_note(str(cleanup))
            raise
        finally:
            self.stats["packing_seconds"] += time.perf_counter() - start

    def _scratch(self, input_count: int, output_count: int) -> None:
        extra = XNN_EXTRA_BYTES // self.numpy_dtype.itemsize
        for name, count in (("input_buffer", input_count), ("output_buffer", output_count)):
            buffer = getattr(self, name)
            if buffer is None or buffer.size < count + extra:
                setattr(self, name, np.empty(count + extra, dtype=self.numpy_dtype))
        self.stats["peak_scratch_bytes"] = max(self.stats["peak_scratch_bytes"], self.scratch_bytes())

    def scratch_bytes(self) -> int:
        return sum(buffer.nbytes for buffer in (self.input_buffer, self.output_buffer) if buffer is not None)

    def linear(self, inputs: torch.Tensor, wrapper: "_XNNPACKLinear") -> torch.Tensor:
        with self.lock:
            start = time.perf_counter()
            self.stats["linear_calls"] += 1
            try:
                if self.closed:
                    raise XNNPACKError("XNNPACK Linear handle is closed")
                wrapper.validate(inputs)
                rows = math.prod(inputs.shape[:-1])
                out_shape = (*inputs.shape[:-1], wrapper.out_features)
                if rows == 0:
                    return torch.empty(out_shape, dtype=torch.float32, device="cpu")
                a_count, c_count = rows * wrapper.in_features, rows * wrapper.out_features
                self._scratch(a_count, c_count)
                phase = time.perf_counter()
                source = inputs.detach().numpy().reshape(rows, wrapper.in_features)
                if not np.isfinite(source).all():
                    raise XNNPACKFiniteError("XNNPACK Linear input has nonfinite values")
                try:
                    with np.errstate(over="raise", invalid="raise"):
                        np.copyto(self.input_buffer[:a_count].reshape(source.shape), source, casting="unsafe")
                except FloatingPointError as exc:
                    raise XNNPACKFiniteError("XNNPACK Linear input overflows FP16") from exc
                extra = XNN_EXTRA_BYTES // self.numpy_dtype.itemsize
                self.input_buffer[a_count:a_count + extra] = 0
                self.output_buffer[c_count:c_count + extra] = 123.5
                self.stats["input_copy_seconds"] += time.perf_counter() - phase
                suffix = "f16" if self.dtype == "float16" else "f32"
                phase = time.perf_counter()
                self.api.check(f"xnn_reshape_fully_connected_nc_{suffix}", wrapper.operator, rows, self.pool)
                self.api.check(f"xnn_setup_fully_connected_nc_{suffix}", wrapper.operator,
                               _pointer(self.input_buffer, self.dtype), _pointer(self.output_buffer, self.dtype))
                self.stats["reshape_setup_seconds"] += time.perf_counter() - phase
                self.stats["xnnpack_run_attempts"] += 1
                phase = time.perf_counter()
                try:
                    self.api.check("xnn_run_operator", wrapper.operator, self.pool)
                finally:
                    self.stats["xnnpack_seconds"] += time.perf_counter() - phase
                self.stats["xnnpack_calls"] += 1
                self.stats["input_rows"] += rows
                phase = time.perf_counter()
                if not np.all(self.output_buffer[c_count:c_count + extra] == 123.5):
                    raise XNNPACKError("XNNPACK wrote beyond the output buffer")
                # Always copy, even for FP32: subsequent layers/calls reuse C.
                result = np.array(self.output_buffer[:c_count], dtype=np.float32, copy=True).reshape(out_shape)
                with np.errstate(over="ignore", invalid="ignore"):
                    if wrapper.bias is not None:
                        result += wrapper.bias.detach().numpy()
                if not np.isfinite(result).all():
                    raise XNNPACKFiniteError("XNNPACK Linear produced nonfinite output")
                self.stats["output_copy_bias_seconds"] += time.perf_counter() - phase
                return torch.from_numpy(result)
            except BaseException as exc:
                self.record_error(exc)
                raise
            finally:
                self.stats["linear_seconds"] += time.perf_counter() - start

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
            errors = []
            for operator in reversed(self.operators):
                try:
                    self.api.check("xnn_delete_operator", operator)
                    self.stats["operator_deletes"] += 1
                except BaseException as exc:
                    self.record_error(exc)
                    errors.append(str(exc))
                finally:
                    operator.value = None
            self.operators.clear()
            if self.pool.value:
                try:
                    self.api.pthreadpool_destroy(self.pool)
                except BaseException as exc:
                    self.record_error(exc)
                    errors.append(str(exc))
                finally:
                    self.pool.value = None
            self.input_buffer = self.output_buffer = None
            self.stats["packed_weight_bytes_estimated"] = 0
            if errors:
                raise XNNPACKError("; ".join(errors))


class _XNNPACKLinear(nn.Module):
    def __init__(self, original: nn.Linear, engine: _Engine, name: str):
        super().__init__()
        # Keep the exact original for restore without registering it twice.
        object.__setattr__(self, "original", original)
        self.weight, self.bias = original.weight, original.bias
        self.in_features, self.out_features = original.in_features, original.out_features
        self.engine, self.name = engine, name
        self.versions = (self.weight._version, self.bias._version if self.bias is not None else None)
        self.operator = engine.create_operator(original, name)
        self.train(False)
        self.validate_parameters()

    def validate_parameters(self) -> None:
        if self.original.weight is not self.weight or self.original.bias is not self.bias:
            raise XNNPACKError(f"Linear {self.name} parameter identity changed; restore and reinstall")
        for name, parameter in (("weight", self.weight), ("bias", self.bias)):
            if parameter is not None and (parameter.device.type != "cpu" or parameter.dtype != torch.float32):
                raise XNNPACKError(f"Linear {self.name} {name} must remain CPU float32; restore and reinstall")
        versions = (self.weight._version, self.bias._version if self.bias is not None else None)
        if versions != self.versions:
            raise XNNPACKError(f"Linear {self.name} parameters changed after packing; restore and reinstall")

    def validate(self, inputs: torch.Tensor) -> None:
        if self.training or self.original.training:
            raise XNNPACKError("XNNPACK Linear is inference-only; keep the model in eval mode")
        if not isinstance(inputs, torch.Tensor) or inputs.device.type != "cpu" or inputs.dtype != torch.float32:
            raise XNNPACKError("XNNPACK Linear requires CPU float32 input tensors")
        if inputs.ndim == 0 or inputs.shape[-1] != self.in_features:
            raise XNNPACKError(f"Linear {self.name} expects a final dimension of {self.in_features}")
        if torch.is_grad_enabled() and (inputs.requires_grad or self.weight.requires_grad or
                                       (self.bias is not None and self.bias.requires_grad)):
            raise XNNPACKError("XNNPACK Linear does not support autograd; use inference_mode() or no_grad()")
        self.validate_parameters()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.engine.linear(inputs, self)


class XNNPACKLinearHandle:
    """Owns installed modules, static operators, and one pthreadpool."""

    def __init__(self, engine: _Engine, replacements: list, skipped: list[str], scope: str):
        self.engine, self.replacements, self.skipped, self.scope = engine, replacements, skipped, scope
        self.replaced_names = [name for _, _, _, _, name in replacements]
        originals = {id(item[2]): item[2] for item in replacements}.values()
        self.original_parameter_bytes = sum((layer.weight.numel() +
            (layer.bias.numel() if layer.bias is not None else 0)) * 4 for layer in originals)
        self.restored = False

    def restore(self) -> None:
        with self.engine.lock:
            if self.restored:
                return
            for parent, key, _, wrapper, name in self.replacements:
                if parent._modules.get(key) is not wrapper:
                    exc = XNNPACKError(f"Cannot restore module {name}: it was changed by the caller")
                    self.engine.record_error(exc)
                    raise exc
            for parent, key, original, _, _ in self.replacements:
                parent._modules[key] = original
            self.restored = True

    def close(self) -> None:
        errors = []
        try:
            self.restore()
        except XNNPACKError as exc:
            errors.append(str(exc))
        try:
            self.engine.close()
        except XNNPACKError as exc:
            errors.append(str(exc))
        if errors:
            raise XNNPACKError("; ".join(errors))

    def metrics(self) -> dict:
        with self.engine.lock:
            stats = dict(self.engine.stats)
            scratch = self.engine.scratch_bytes()
            native_used = stats["xnnpack_calls"] > 0 and not self.engine.api.mock
            return {**stats, **self.engine.api.metadata, "backend": "xnnpack", "experimental": True,
                    "precision": self.engine.dtype, "accumulator": self.engine.dtype,
                    "accumulation_note": ("ARM NEON FP16 accumulation; weights, inputs, and GEMM output "
                                          "are FP16; bias is added in FP32" if self.engine.dtype == "float16"
                                          else "FP32 GEMM and bias addition"),
                    "denormals_note": "The pinned xnn_run_operator requests DISABLE_DENORMALS",
                    "xnnpack_used": native_used, "npu_used": False,
                    "threads_requested": self.engine.threads_requested,
                    "threads_actual": self.engine.threads_actual,
                    "cpu_affinity": list(self.engine.cpu_affinity) if self.engine.cpu_affinity is not None else None,
                    "scope": self.scope, "replaced_modules": len(self.replaced_names),
                    "unique_operators": stats["operator_creates"], "replaced_module_names": list(self.replaced_names),
                    "skipped_modules": list(self.skipped), "scratch_bytes": scratch,
                    "memory_bytes_estimated": stats["packed_weight_bytes_estimated"] + scratch,
                    "memory_estimate_note": "Logical packed weight+bias bytes and allocated IO scratch only; "
                                            "excludes SDK alignment, operator/threadpool overhead, original parameters",
                    "original_parameter_bytes": self.original_parameter_bytes,
                    "closed": self.engine.closed, "restored": self.restored,
                    "xnnpack_commit": XNNPACK_COMMIT, "pthreadpool_commit": PTHREADPOOL_COMMIT}

    def __enter__(self) -> "XNNPACKLinearHandle":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            self.close()
        except XNNPACKError as cleanup:
            if exc is None:
                raise
            exc.add_note(str(cleanup))
        return False


def install_xnnpack_linear(model: nn.Module, library=None, threads: int = 4,
                          dtype: str = "float32", scope: str = "all",
                          cpu_affinity=None) -> XNNPACKLinearHandle:
    """Replace CPU FP32 Linear children with both channel dimensions >=128.

    ``scope='aggregator'`` selects only children of ``model.aggregator``;
    ``all`` selects the entire model. Other names are recorded as skipped.
    Both precisions pack the original FP32 [N,K] weights once per unique
    module, use NULL XNN bias/cache pointers, and add original bias in FP32.
    Mutation/version checks protect static packs; untracked ``.data`` writes
    are unsupported. Caller affinity is restored immediately after pool
    creation; worker threads inherit only the optional temporary affinity.

    ``library=None`` loads Torch's libtorch_cpu.so after strict version/header
    checks. A path/CDLL must refer to that binary or a byte-identical copy;
    arbitrary custom libraries are rejected before any XNN call. A non-CDLL object is solely a
    host-test injection point, visibly marked unverified/mock in metrics.
    Install is atomic with respect to module replacement: prepare all owned
    operators first, and clean up on failure without changing the model.
    """
    if not isinstance(model, nn.Module) or model.training:
        raise ValueError("Set model.eval() before installing XNNPACK Linear")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    if dtype not in {"float32", "float16"}:
        raise ValueError("dtype must be 'float32' or 'float16'")
    if scope not in {"all", "aggregator"}:
        raise ValueError("scope must be 'all' or 'aggregator'")
    if cpu_affinity is not None:
        try:
            affinity = tuple(cpu_affinity)
        except (TypeError, ValueError) as exc:
            raise ValueError("cpu_affinity must be a nonempty collection of nonnegative CPU integers") from exc
        if not affinity or any(isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0 for cpu in affinity):
            raise ValueError("cpu_affinity must be a nonempty collection of nonnegative CPU integers")
        if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
            raise XNNPACKUnavailable("Temporary pthreadpool CPU affinity requires Linux sched affinity support")
        cpu_affinity = tuple(sorted(set(affinity)))
    selected, skipped = [], []
    for name, layer in model.named_modules(remove_duplicate=False):
        if not isinstance(layer, nn.Linear):
            continue
        if layer.in_features < 128 or layer.out_features < 128 or (scope == "aggregator" and
                                                               not name.startswith("aggregator.")):
            skipped.append(name)
            continue
        if not name:
            raise ValueError("Wrap a root nn.Linear in a containing module before installing XNNPACK Linear")
        if type(layer).forward is not nn.Linear.forward:
            raise ValueError(f"Selected Linear {name} overrides forward and cannot be replaced safely")
        if layer.training or layer.weight.device.type != "cpu" or layer.weight.dtype != torch.float32:
            raise ValueError(f"Selected Linear {name} must be in eval mode with CPU float32 weights")
        if layer.bias is not None and (layer.bias.device.type != "cpu" or layer.bias.dtype != torch.float32):
            raise ValueError(f"Selected Linear {name} must have CPU float32 bias")
        parent_name, _, key = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        selected.append((parent, key, layer, name))
    if not selected:
        raise ValueError("No Linear modules match the requested XNNPACK scope and channel threshold")
    engine = _Engine(_API(library), threads, dtype, cpu_affinity)
    replacements, wrappers = [], {}
    try:
        for parent, key, original, name in selected:
            if id(original) not in wrappers:
                wrappers[id(original)] = _XNNPACKLinear(original, engine, name)
            replacements.append((parent, key, original, wrappers[id(original)], name))
        handle = XNNPACKLinearHandle(engine, replacements, skipped, scope)
        for parent, key, _, wrapper, _ in replacements:
            parent._modules[key] = wrapper
        return handle
    except BaseException as exc:
        for parent, key, original, wrapper, _ in replacements:
            if parent._modules.get(key) is wrapper:
                parent._modules[key] = original
        try:
            engine.close()
        except XNNPACKError as cleanup:
            exc.add_note(str(cleanup))
        raise
