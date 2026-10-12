"""Optional ctypes OpenCL bridge for the original VGGT frame-0 MLP diagnostic.

No libraries are loaded on import. Construct only on the RK3588/S board.
Use original bias once: NPU FC1 is a bias-free MatMul; this bridge adds FC1
bias, evaluates standard FP32 Erf GELU, and writes native FP16 FC2 A.
Keep both RKNN allocations alive, serialize device/CPU access, and close this
bridge BEFORE destroying RKNN memories or contexts. There is no CPU fallback.

Primary ABI/ownership sources (not copied SDK implementations):
https://registry.khronos.org/OpenCL/extensions/arm/cl_arm_import_memory.txt
https://github.com/KhronosGroup/OpenCL-Headers/blob/main/CL/cl_ext.h
https://github.com/airockchip/rknn-toolkit2/blob/v2.3.2/rknpu2/runtime/Linux/librknn_api/include/rknn_matmul_api.h
RKNN SDK 2.3.2 User Guide, sections 3.5.4 and 5.5.4.1.3.
"""
from __future__ import annotations

import ctypes as ct
import hashlib
import os
from pathlib import Path
import platform
import threading
import time


KERNEL_SOURCE = r"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL FP_CONTRACT OFF
__kernel void original_bias_erf_gelu_c4_to_a8(
    __global const float *src, const ulong src_offset_elements,
    __global const float *bias,
    __global half *dst, const ulong dst_offset_elements,
    const uint rows, const uint channels)
{
    const size_t i = get_global_id(0);
    if (i >= (size_t)rows * (size_t)channels) return;
    // Traverse destination A8 storage; every original row/channel is written.
    const uint lane = (uint)(i % 8);
    const uint row = (uint)((i / 8) % rows);
    const uint channel = (uint)((i / 8 / rows) * 8 + lane);
    const size_t c4 = ((size_t)(channel / 4) * rows + row) * 4 + channel % 4;
    const float x = src[src_offset_elements + c4] + bias[channel];
    const float scaled = x / 0x1.6a09e6p+0f; // FP32 sqrt(2), original Erf equation
    const float plus_one = 1.0f + erf(scaled);
    const float product = x * plus_one;
    const float gelu = product * 0.5f;
    dst[dst_offset_elements + i] = convert_half_rte(gelu);
}
"""
BUILD_OPTIONS = b"-cl-std=CL1.2"  # no relaxed math, native_erf, tanh approximation, or contraction


class OpenCLBridgeError(RuntimeError):
    pass


def _check(code, operation):
    if code:
        raise OpenCLBridgeError(f"{operation} returned {int(code)}")


def _board_guard():
    compatibles = (Path("/proc/device-tree/compatible"), Path("/sys/firmware/devicetree/base/compatible"))
    compatible = next((path.read_bytes() for path in compatibles if path.is_file()), b"")
    if platform.machine().lower() not in ("aarch64", "arm64") or b"rk3588" not in compatible:
        raise OpenCLBridgeError("OpenCL execution is restricted to the connected RK3588/S board")


def _mem_fields(memory):
    value = memory.contents if hasattr(memory, "contents") else memory
    fields = {key: int(getattr(value, key)) for key in ("fd", "offset", "size")}
    if fields["fd"] < 0 or fields["offset"] < 0 or not 0 < fields["size"] <= 1 << 30:
        raise OpenCLBridgeError("invalid RKNN DMA fd/offset/size")
    return fields


def pack_native_a(value):
    """Source/NumPy helper: normal [M,K] -> owned FP16 [K/8,M,8]."""
    import numpy as np
    value = np.asarray(value)
    if value.ndim != 2 or value.shape[1] % 8 or not np.isfinite(value).all():
        raise ValueError("native A requires finite [M,K], K aligned to eight")
    with np.errstate(over="raise", invalid="raise"):
        half = value.astype(np.float16)
    return np.ascontiguousarray(half.reshape(len(value), -1, 8).transpose(1, 0, 2))


def unpack_native_c(value):
    """Source/NumPy helper: FP32 [N/4,M,4] -> owned FP32 [M,N]."""
    import numpy as np
    value = np.asarray(value)
    if value.ndim != 3 or value.shape[-1] != 4 or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError("native C requires finite FP32 [N/4,M,4]")
    return np.array(value.transpose(1, 0, 2).reshape(value.shape[1], -1), copy=True, order="C")


class NativeErfGeluBridge:
    """One persistent imported C1->A2 bridge; synchronous run returns timings.

    run callbacks are required, not optional no-ops:
      before_input(): FC1 C FROM_DEVICE after successful blocking FC1 run.
      before_output(): FC2 A TO_DEVICE before GPU writes, to flush old CPU dirt.
      after_output(): FC2 A FROM_DEVICE after clFinish, to invalidate stale CPU A.
    Caller then explicitly performs FC2 A TO_DEVICE + bind + blocking FC2 run.
    All callbacks and enqueue/finish/event-release costs are inside run timings.
    This conservative protocol still needs same-input board numerical validation;
    extension presence and clFinish alone do not establish NPU/GPU coherence.
    """

    def __init__(self, fc1_mem, fc2_mem, bias, *, rows=106, channels=4096,
                 library="libOpenCL.so.1", platform_index=0, device_index=0,
                 managed_host_consistency=True):
        _board_guard()
        import numpy as np
        bias = np.asarray(bias)
        if rows != 106 or channels != 4096 or bias.shape != (4096,) or bias.dtype != np.float32 or not np.isfinite(bias).all():
            raise ValueError("retain all original 106 rows and original FP32 4096-channel bias")
        if not isinstance(managed_host_consistency, bool):
            raise TypeError("managed_host_consistency must be explicit bool")
        self._lock = threading.RLock()
        self._owners = (fc1_mem, fc2_mem)  # strong references; caller owns actual RKNN allocations
        self._source, self._destination = _mem_fields(fc1_mem), _mem_fields(fc2_mem)
        if (self._source["size"] < rows * channels * 4 or self._source["offset"] % 4 or
                self._destination["size"] < rows * channels * 2 or self._destination["offset"] % 2):
            raise OpenCLBridgeError("RKNN native C/A allocation is undersized or has a misaligned offset")
        self.rows, self.channels = rows, channels
        self._bias = np.array(bias, copy=True, order="C")
        self._bias.flags.writeable = False
        self._fds, self._memory, self._objects = [], [], []
        self._context = self._queue = self._program = self._kernel = None
        self._closed, self._failed = False, False
        self._stats = {"calls_attempted": 0, "calls_completed": 0, "events_created": 0, "events_released": 0,
                       "cleanup_errors": [], "original_bias_raw_sha256": hashlib.sha256(self._bias.tobytes()).hexdigest(),
                       "kernel_sha256": hashlib.sha256(KERNEL_SOURCE.encode()).hexdigest(),
                       "build_options": BUILD_OPTIONS.decode(), "library": str(library),
                       "rows": rows, "channels": channels, "input_layout": [channels // 4, rows, 4],
                       "output_layout": [channels // 8, rows, 8], "output_dtype": "float16",
                       "managed_host_consistency": managed_host_consistency,
                       "model_or_trajectory_accuracy_validated": False, "zero_copy_performance_validated": False}
        started = time.perf_counter()
        try:
            self._lib = ct.CDLL(str(library))
            self._bind_api()
            self._platform = self._select_handles("platform", platform_index)
            self._device = self._select_handles("device", device_index)
            extensions = set(self._device_string(0x1030).split())
            required = {"cl_khr_fp16", "cl_arm_import_memory", "cl_arm_import_memory_dma_buf"}
            if not required <= extensions:
                raise OpenCLBridgeError("missing required OpenCL extensions: " + ",".join(sorted(required - extensions)))
            self._stats["device_name"] = self._device_string(0x102B)
            self._stats["driver_version"] = self._device_string(0x102D)
            self._stats["required_extensions"] = sorted(required)
            self._devices = (ct.c_void_p * 1)(self._device)
            properties = (ct.c_ssize_t * 3)(0x1084, self._platform, 0)  # CL_CONTEXT_PLATFORM
            self._context = self._create("clCreateContext", properties, 1, self._devices, None, None)
            self._queue = self._create("clCreateCommandQueue", self._context, self._device, 2)  # profiling
            source = (ct.c_char_p * 1)(KERNEL_SOURCE.encode())
            lengths = (ct.c_size_t * 1)(len(KERNEL_SOURCE.encode()))
            self._program = self._create("clCreateProgramWithSource", self._context, 1, source, lengths)
            code = self._lib.clBuildProgram(self._program, 1, self._devices, BUILD_OPTIONS, None, None)
            if code:
                raise OpenCLBridgeError(f"clBuildProgram returned {code}: {self._build_log()}")
            self._kernel = self._create("clCreateKernel", self._program, b"original_bias_erf_gelu_c4_to_a8")
            self._import = self._import_function()
            src = self._import_dma(self._source, 4, managed_host_consistency, "FC1_C")  # READ_ONLY
            dst = self._import_dma(self._destination, 2, managed_host_consistency, "FC2_A")  # WRITE_ONLY
            bias_mem = self._create("clCreateBuffer", self._context, 4 | 32, self._bias.nbytes,
                                    ct.c_void_p(self._bias.ctypes.data))  # READ_ONLY | COPY_HOST_PTR
            self._memory.append(bias_mem)
            args = (ct.c_void_p(src), ct.c_uint64(self._source["offset"] // 4), ct.c_void_p(bias_mem),
                    ct.c_void_p(dst), ct.c_uint64(self._destination["offset"] // 2), ct.c_uint32(rows), ct.c_uint32(channels))
            for index, value in enumerate(args):
                _check(self._lib.clSetKernelArg(self._kernel, index, ct.sizeof(value), ct.byref(value)), f"clSetKernelArg({index})")
            self._global = (ct.c_size_t * 1)(rows * channels)
            self._stats["initialization_seconds"] = time.perf_counter() - started
        except BaseException as error:
            self._failed = True
            try:
                self.close()
            except BaseException as cleanup:
                error.add_note(str(cleanup))
            raise

    def _bind_api(self):
        ptr, uint, size, integer, ulong = ct.c_void_p, ct.c_uint32, ct.c_size_t, ct.c_int32, ct.c_uint64
        pp, up, sp, ip = ct.POINTER(ptr), ct.POINTER(uint), ct.POINTER(size), ct.POINTER(integer)
        signatures = {
            "clGetPlatformIDs": ([uint, pp, up], integer),
            "clGetDeviceIDs": ([ptr, ulong, uint, pp, up], integer),
            "clGetDeviceInfo": ([ptr, uint, size, ptr, sp], integer),
            "clCreateContext": ([ct.POINTER(ct.c_ssize_t), uint, pp, ptr, ptr, ip], ptr),
            "clCreateCommandQueue": ([ptr, ptr, ulong, ip], ptr),
            "clCreateProgramWithSource": ([ptr, uint, ct.POINTER(ct.c_char_p), sp, ip], ptr),
            "clBuildProgram": ([ptr, uint, pp, ct.c_char_p, ptr, ptr], integer),
            "clGetProgramBuildInfo": ([ptr, ptr, uint, size, ptr, sp], integer),
            "clCreateKernel": ([ptr, ct.c_char_p, ip], ptr),
            "clCreateBuffer": ([ptr, ulong, size, ptr, ip], ptr),
            "clSetKernelArg": ([ptr, uint, size, ptr], integer),
            "clEnqueueNDRangeKernel": ([ptr, ptr, uint, sp, sp, sp, uint, pp, pp], integer),
            "clGetEventProfilingInfo": ([ptr, uint, size, ptr, sp], integer),
            "clFinish": ([ptr], integer),
        }
        for suffix in ("Event", "MemObject", "Kernel", "Program", "CommandQueue", "Context"):
            signatures["clRelease" + suffix] = ([ptr], integer)
        for name, (args, result) in signatures.items():
            function = getattr(self._lib, name)
            function.argtypes, function.restype = args, result

    def _select_handles(self, kind, index):
        if not isinstance(index, int) or index < 0:
            raise ValueError(f"invalid {kind} index")
        count = ct.c_uint32()
        function = self._lib.clGetPlatformIDs if kind == "platform" else self._lib.clGetDeviceIDs
        prefix = () if kind == "platform" else (self._platform, 4)  # GPU only
        _check(function(*prefix, 0, None, ct.byref(count)), f"query {kind} count")
        if not 0 < count.value <= 32 or index >= count.value:
            raise OpenCLBridgeError(f"unavailable {kind} index {index}, count {count.value}")
        handles = (ct.c_void_p * count.value)()
        _check(function(*prefix, count.value, handles, None), f"query {kind} handles")
        return handles[index]

    def _string(self, function, prefix):
        count = ct.c_size_t()
        _check(function(*prefix, 0, None, ct.byref(count)), "OpenCL string size")
        if not 0 < count.value <= 65536:
            raise OpenCLBridgeError(f"unexpected OpenCL string size {count.value}")
        value = ct.create_string_buffer(count.value)
        _check(function(*prefix, count.value, value, None), "OpenCL string value")
        return value.value.decode(errors="replace")

    def _device_string(self, key):
        return self._string(self._lib.clGetDeviceInfo, (self._device, key))

    def _build_log(self):
        return self._string(self._lib.clGetProgramBuildInfo, (self._program, self._device, 0x1183))

    def _create(self, name, *args):
        error = ct.c_int32()
        handle = getattr(self._lib, name)(*args, ct.byref(error))
        # Preserve an unexpectedly non-null failed allocation so close can release it.
        if handle:
            suffix = {"clCreateContext": "Context", "clCreateCommandQueue": "CommandQueue",
                      "clCreateProgramWithSource": "Program", "clCreateKernel": "Kernel",
                      "clCreateBuffer": "MemObject"}[name]
            self._objects.append((suffix, handle))
        _check(error.value, name)
        if not handle:
            raise OpenCLBridgeError(name + " returned a null object")
        return handle

    def _import_function(self):
        signature = ct.CFUNCTYPE(ct.c_void_p, ct.c_void_p, ct.c_uint64, ct.POINTER(ct.c_ssize_t),
                                 ct.c_void_p, ct.c_size_t, ct.POINTER(ct.c_int32))
        get_address = getattr(self._lib, "clGetExtensionFunctionAddressForPlatform", None)
        if get_address is not None:
            get_address.argtypes, get_address.restype = [ct.c_void_p, ct.c_char_p], ct.c_void_p
            address = get_address(self._platform, b"clImportMemoryARM")
            if address:
                return signature(address)
        symbol = getattr(self._lib, "clImportMemoryARM", None)
        if symbol is None:
            raise OpenCLBridgeError("clImportMemoryARM is not callable")
        symbol.argtypes, symbol.restype = signature._argtypes_, signature._restype_
        return symbol

    def _import_dma(self, fields, flags, managed_host_consistency, label):
        fd = os.dup(fields["fd"])
        self._fds.append(fd)  # own duplicated fd until imported cl_mem is released
        capacity = os.lseek(fd, 0, os.SEEK_END)  # documented DMA-BUF size query
        os.lseek(fd, 0, os.SEEK_SET)
        span = fields["offset"] + fields["size"]
        if span > capacity or not capacity:
            raise OpenCLBridgeError(f"{label}: offset+size {span} exceeds DMA-BUF capacity {capacity}")
        properties = [0x40B2, 0x40B4]  # CL_IMPORT_TYPE_ARM, CL_IMPORT_TYPE_DMA_BUF_ARM
        if managed_host_consistency:
            properties += [0x41E3, 1]  # CL_IMPORT_DMA_BUF_DATA_CONSISTENCY_WITH_HOST_ARM, CL_TRUE
        properties += [0]
        properties = (ct.c_ssize_t * len(properties))(*properties)
        fd_argument, error = ct.c_int32(fd), ct.c_int32()
        handle = self._import(self._context, flags, properties, ct.byref(fd_argument), span, ct.byref(error))
        if handle:
            self._objects.append(("MemObject", handle))
            self._memory.append(handle)
        _check(error.value, f"clImportMemoryARM({label})")
        if not handle:
            raise OpenCLBridgeError(f"clImportMemoryARM({label}) returned null")
        self._stats[label] = {"offset_bytes": fields["offset"], "RKNN_size_bytes": fields["size"],
                              "DMA_allocation_bytes": capacity, "import_span_bytes": span,
                              "imported_via_dma_fd": True, "copied_to_GPU_staging_buffer": False}
        return handle

    def run(self, *, before_input, before_output, after_output):
        if not all(callable(callback) for callback in (before_input, before_output, after_output)):
            raise TypeError("all three explicit RKNN synchronization callbacks are required")
        with self._lock:
            if self._closed or self._failed:
                raise OpenCLBridgeError("bridge is closed or failed")
            self._stats["calls_attempted"] += 1
            event, stages = ct.c_void_p(), {}
            started = last = time.perf_counter()
            try:
                before_input()
                now = time.perf_counter(); stages["FC1_C_FROM_DEVICE_seconds"] = now - last; last = now
                before_output()
                now = time.perf_counter(); stages["FC2_A_prewrite_TO_DEVICE_seconds"] = now - last; last = now
                code = self._lib.clEnqueueNDRangeKernel(self._queue, self._kernel, 1, None, self._global,
                                                       None, 0, None, ct.byref(event))
                if event.value:
                    self._stats["events_created"] += 1
                _check(code, "clEnqueueNDRangeKernel")
                now = time.perf_counter(); stages["enqueue_seconds"] = now - last; last = now
                _check(self._lib.clFinish(self._queue), "clFinish(GELU)")
                now = time.perf_counter(); stages["finish_seconds"] = now - last; last = now
                after_output()
                now = time.perf_counter(); stages["FC2_A_postwrite_FROM_DEVICE_seconds"] = now - last; last = now
                profiling = {}
                for name, key in (("queued", 0x1280), ("submit", 0x1281), ("start", 0x1282), ("end", 0x1283)):
                    value = ct.c_uint64()
                    _check(self._lib.clGetEventProfilingInfo(event, key, ct.sizeof(value), ct.byref(value), None), "event profiling " + name)
                    profiling[name] = value.value
                if not profiling["queued"] <= profiling["submit"] <= profiling["start"] <= profiling["end"]:
                    raise OpenCLBridgeError("invalid OpenCL event profiling clock order")
                stages["GPU_kernel_seconds"] = (profiling["end"] - profiling["start"]) / 1e9
                self._stats["calls_completed"] += 1
            except BaseException:
                self._failed = True
                raise
            finally:
                if event.value:
                    code = self._lib.clReleaseEvent(event)
                    if code == 0:
                        self._stats["events_released"] += 1
                    else:
                        self._failed = True
                        self._stats["cleanup_errors"].append(f"clReleaseEvent returned {code}")
                stages["profiling_and_event_release_seconds"] = time.perf_counter() - last
                stages["complete_bridge_seconds"] = time.perf_counter() - started
                self._stats["last_timing"] = dict(stages)
            if self._failed:
                raise OpenCLBridgeError("OpenCL event cleanup failed")
            return stages

    def metrics(self):
        return {**self._stats, "closed": self._closed, "failed": self._failed,
                "live_OpenCL_objects": len(self._objects), "live_owned_DMA_fds": len(self._fds),
                "fully_released": not self._objects and not self._fds}

    def close(self):
        with self._lock:
            if self._closed and not self._objects and not self._fds:
                return
            errors = []
            if self._queue:
                code = self._lib.clFinish(self._queue)
                if code:
                    errors.append(f"clFinish(close) returned {code}")
            remaining = []
            for suffix, handle in reversed(self._objects):
                code = getattr(self._lib, "clRelease" + suffix)(handle)
                if code:
                    errors.append(f"clRelease{suffix} returned {code}")
                    remaining.append((suffix, handle))
            # A release failure must remain visible, not turn into a false zero-live count.
            self._objects = list(reversed(remaining))
            if ("CommandQueue", self._queue) not in self._objects:
                self._queue = None
            if not self._objects:
                self._memory.clear()
                remaining_fds = []
                for fd in reversed(self._fds):
                    try:
                        os.close(fd)
                    except OSError as error:
                        errors.append(f"close owned DMA fd: {error}")
                        remaining_fds.append(fd)
                self._fds = list(reversed(remaining_fds))
            self._closed = True
            self._stats["cleanup_errors"].extend(errors)
            if errors:
                self._failed = True
                raise OpenCLBridgeError("; ".join(errors))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
