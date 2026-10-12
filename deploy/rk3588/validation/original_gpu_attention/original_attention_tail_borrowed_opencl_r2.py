"""Borrow the RKNN GPU custom-op context/queue/kernel for the original tail.

No libraries or SDK/model modules are loaded on import. This source adapts the
frozen, board-validated component kernels only for independent DMA inputs and
offsets. All original FP32 arithmetic and complete consumer layout are retained.
SDK-owned context, queue, program and initial kernel are NEVER released here.
Only locally created kernels/buffers and per-call duplicated DMA fds/imports are
released. The caller must keep RKNN tensors/context alive through successful close.

The required synchronization callbacks are explicit: input TO_DEVICE for CPU-ready
custom-op inputs; output prewrite TO_DEVICE; after clFinish, output FROM_DEVICE and
owned finite validation. No HOST import of a DMA mapping and no silent copy path.
This is not proof of NPU/GPU coherence, numerical equivalence or performance.
"""
from __future__ import annotations

import ctypes as ct
import hashlib
import os
import re
import threading
import time


COMPONENT_SOURCE_SHA256 = "9dc9e66b9b0521ff168e31063c47bd21c6dc4cbccc9bd339a4ecf28b647b33e7"
LEGACY_SOURCE_SHA256 = "f06486299101fb233c68abc9908e8c2629c4cb1f7dca6241133e6e222634acc2"


def kernel_source(component):
    """Change only memory bindings/offsets of the original validated kernels."""
    source = component.KERNEL_SOURCE
    changes = (
        ("__global const float *qkv, __global const float *parameters,\n"
         "    __global const float *tables, __global float *norm_debug,",
         "__global const float *query, const ulong query_offset,\n"
         "    __global const float *key_input, const ulong key_offset,\n"
         "    __global const float *parameters, __global const float *tables, __global float *norm_debug,"),
        ("const float2 original = (float2)(qkv[offset], qkv[VALUES + offset]);",
         "const float2 original = (float2)(query[query_offset + offset], key_input[key_offset + offset]);"),
        ("__global const float *qkv, __global const float *scores, __global float *proj_input)",
         "__global const float *value_input, const ulong value_offset,\n"
         "    __global const float *scores, __global float *proj_input, const ulong output_offset)"),
        ("qkv[2 * VALUES + (batch_head * TOKENS + key) * DIM + lane]",
         "value_input[value_offset + (batch_head * TOKENS + key) * DIM + lane]"),
        ("proj_input[((batch * TOKENS + query) * HEADS + head) * DIM + lane]",
         "proj_input[output_offset + ((batch * TOKENS + query) * HEADS + head) * DIM + lane]"),
    )
    for original, replacement in changes:
        if source.count(original) != 1:
            raise ValueError("Frozen component kernel binding no longer matches")
        source = source.replace(original, replacement)
    return source


def build_options(shape):
    shape = tuple(shape)
    if shape not in ((2, 16, 53, 64), (1, 16, 106, 64)):
        raise ValueError("Original full frame/global attention shape required")
    return f"-cl-std=CL1.2 -DBATCH={shape[0]} -DTOKENS={shape[2]} -DLOCAL_KEYS={64 if shape[2] == 53 else 128}".encode()


def make_borrowed_bridge_class(legacy, component):
    """Create a board-only bridge class; creation of this class runs no kernels."""
    source = kernel_source(component)

    class BorrowedOriginalAttentionTail(legacy.NativeErfGeluBridge):
        def __init__(self, *, context, queue, kernel, shape, parameters, tables,
                     epsilon, scale, library, managed_host_consistency=True):
            legacy._board_guard()
            import numpy as np
            self._lock = threading.RLock()
            self._fds, self._memory, self._objects = [], [], []
            self._context = self._queue = self._program = self._kernel = None
            self._closed, self._failed = False, False
            self._api_bound = False
            self._owners = ()
            self._stats = dict(calls_attempted=0, calls_completed=0, events_created=0,
                               events_released=0, cleanup_errors=[])
            self.shape = tuple(shape)
            build_options(self.shape)
            batch, heads, tokens, width = self.shape
            if epsilon not in (1e-5, ct.c_float(1e-5).value) or scale != .125:
                raise ValueError("Original ONNX LayerNorm epsilon and full SDPA scale required")
            if type(managed_host_consistency) is not bool:
                raise TypeError("Explicit DMA host consistency boolean required")
            if not all(int(value) for value in (context, queue, kernel)):
                raise ValueError("All three SDK-owned GPU handles are required")
            if len(parameters) != 4 or len(tables) != 4:
                raise ValueError("Four original affine arrays and original indexed RoPE tables required")
            for value in parameters:
                if value.dtype != np.float32 or value.shape != (64,) or not np.isfinite(value).all():
                    raise ValueError("Original finite FP32 norm parameters required")
            for value in tables:
                if value.dtype != np.float32 or value.shape != (batch, 1, tokens, 32) or not np.isfinite(value).all():
                    raise ValueError("Original finite indexed [B,1,S,32] RoPE tables required")
            self._parameters = np.ascontiguousarray(np.stack(parameters))
            self._tables = np.ascontiguousarray(np.stack(tables))
            self._parameters.flags.writeable = self._tables.flags.writeable = False
            self._owners = (self._parameters, self._tables)
            self.elements = batch * heads * tokens * width
            self.rows = batch * heads * tokens
            self._managed_host_consistency = managed_host_consistency
            self._borrowed = dict(context=int(context), queue=int(queue), kernel=int(kernel))
            self._queue = int(queue)
            self._context = int(context)
            self._kernel = int(kernel)
            self._stats.update(shape=list(self.shape), consumer_shape=[batch, tokens, 1024],
                kernel_sha256=hashlib.sha256(source.encode()).hexdigest(), build_options=build_options(shape).decode(),
                library=str(library), original_all_tokens_retained=True, original_two_axis_RoPE_tables_used=True,
                original_FP32_kernel_math_same_as_component=True,
                borrowed_SDK_context_queue_program_initial_kernel_not_owned=True,
                SDK_input_output_allocations_not_owned=True, per_compute_import_lifetime=True,
                GPU_staging_copies=False, DMA_imported=True, zero_copy_performance_validated=False,
                native_INT8_equivalence_verified=False, model_or_trajectory_accuracy_validated=False,
                epsilon=epsilon, scale=scale, managed_host_consistency=managed_host_consistency,
                fixed_parameter_raw_sha256=hashlib.sha256(self._parameters.tobytes()).hexdigest(),
                fixed_RoPE_tables_raw_sha256=hashlib.sha256(self._tables.tobytes()).hexdigest())
            started = time.perf_counter()
            try:
                self._lib = ct.CDLL(str(library))
                self._bind_api()
                self._api_bound = True
                ptr, uint, size, integer = ct.c_void_p, ct.c_uint32, ct.c_size_t, ct.c_int32
                for name in ("clGetKernelInfo", "clGetCommandQueueInfo", "clGetContextInfo"):
                    function = getattr(self._lib, name)
                    function.argtypes, function.restype = [ptr, uint, size, ptr, ct.POINTER(size)], integer
                self._lib.clGetProgramInfo.argtypes = [ptr, uint, size, ptr, ct.POINTER(size)]
                self._lib.clGetProgramInfo.restype = integer
                self._lib.clGetKernelWorkGroupInfo.argtypes = [ptr, ptr, uint, size, ptr, ct.POINTER(size)]
                self._lib.clGetKernelWorkGroupInfo.restype = integer
                self._lib.clEnqueueReadBuffer.argtypes = [ptr, ptr, uint, size, size, ptr, uint,
                                                         ct.POINTER(ptr), ct.POINTER(ptr)]
                self._lib.clEnqueueReadBuffer.restype = integer
                queue_context = self._info(self._lib.clGetCommandQueueInfo, self._queue, 0x1090, ptr, "queue context")
                self._device = self._info(self._lib.clGetCommandQueueInfo, self._queue, 0x1091, ptr, "queue device")
                properties = self._info(self._lib.clGetCommandQueueInfo, self._queue, 0x1093, ct.c_uint64, "queue properties")
                kernel_context = self._info(self._lib.clGetKernelInfo, self._kernel, 0x1193, ptr, "kernel context")
                if queue_context != self._context or kernel_context != self._context:
                    raise legacy.OpenCLBridgeError("Borrowed kernel/queue do not share the SDK GPU context")
                self._platform = self._info(self._lib.clGetDeviceInfo, self._device, 0x1031, ptr, "device platform")
                self._program = self._info(self._lib.clGetKernelInfo, self._kernel, 0x1194, ptr, "kernel program")
                self._record_program_specialization()
                argument_count = self._info(self._lib.clGetKernelInfo, self._kernel, 0x1191, uint, "norm kernel argument count")
                if argument_count != 11:
                    raise legacy.OpenCLBridgeError("SDK norm kernel does not use the borrowed FP32 DMA argument contract")
                self._profile_events = bool(properties & 2)
                extensions = set(self._device_string(0x1030).split())
                required = {"cl_arm_import_memory", "cl_arm_import_memory_dma_buf"}
                if not required <= extensions:
                    raise legacy.OpenCLBridgeError("SDK GPU device lacks ARM DMA import")
                self._stats.update(device_name=self._device_string(0x102B), driver_version=self._device_string(0x102D),
                                   SDK_queue_properties=properties, kernel_profiling_available=self._profile_events,
                                   required_extensions=sorted(required))
                self._kernels = [self._kernel,
                    self._create("clCreateKernel", self._program, b"original_scaled_qk"),
                    self._create("clCreateKernel", self._program, b"original_softmax_v_to_proj")]
                self._buffers = {}
                for name, count, flags, value in (
                    ("parameters", 256, 4 | 32, self._parameters),
                    ("tables", 4 * batch * tokens * 32, 4 | 32, self._tables),
                    ("norm_debug", 2 * self.elements, 1, None),
                    ("qk_rope", 2 * self.elements, 1, None),
                    ("scores", batch * heads * tokens * tokens, 1, None)):
                    handle = self._create("clCreateBuffer", self._context, flags, count * 4,
                                          None if value is None else ct.c_void_p(value.ctypes.data))
                    self._buffers[name] = handle
                    self._memory.append(handle)
                self._import = self._import_function()
                tiles = (tokens + 7) // 8
                self._launches = [(self.rows * 64, 64), (batch * heads * tiles * tiles * 64, 64),
                                  (self.rows * (64 if tokens == 53 else 128), 64 if tokens == 53 else 128)]
                self._stats["initialization_seconds"] = time.perf_counter() - started
            except BaseException as error:
                self._failed = True
                # A failed constructor is not assigned to the runtime owner's state.
                # Retain it explicitly so failed cleanup can be audited/retried before SDK destroy.
                error.borrowed_bridge = self
                try:
                    self.close()
                except BaseException as cleanup:
                    error.add_note(str(cleanup))
                raise

        def _record_program_specialization(self):
            options = self._string(self._lib.clGetProgramBuildInfo, (self._program, self._device, 0x1182))
            actual_source = self._string(self._lib.clGetProgramInfo, (self._program, 0x1164))
            function_name = self._string(self._lib.clGetKernelInfo, (self._kernel, 0x1190))
            expected = dict(BATCH=self.shape[0], TOKENS=self.shape[2],
                            LOCAL_KEYS=64 if self.shape[2] == 53 else 128)
            source_defines, option_defines = {}, {}
            for name in expected:
                source_defines[name] = [int(x) for x in re.findall(
                    r"(?m)^\s*#\s*define\s+" + name + r"\s+(\d+)\s*$", actual_source)]
                option_defines[name] = [int(x) for x in re.findall(
                    r"(?:^|\s)-D\s*" + name + r"=(\d+)(?=\s|$)", options)]
            self._stats["actual_SDK_program"] = dict(context=self._context, queue=self._queue,
                initial_kernel=self._kernel, program=self._program,
                function_name=function_name, build_options=options,
                source_bytes=len(actual_source.encode()), source_sha256=hashlib.sha256(actual_source.encode()).hexdigest(),
                expected_defines=expected, source_defines=source_defines, option_defines=option_defines)
            # A cache hit with the other role's macro specialization is an execution blocker.
            for name, value in expected.items():
                values = source_defines[name] + option_defines[name]
                if not values or any(x != value for x in values):
                    raise legacy.OpenCLBridgeError("SDK program specialization differs for " + name + ": " + repr(values))
            group = (ct.c_size_t * 3)()
            legacy._check(self._lib.clGetKernelWorkGroupInfo(self._kernel, self._device, 0x11B1,
                ct.sizeof(group), group, None), "SDK kernel compile work group")
            self._stats["actual_SDK_program"]["compile_work_group_size"] = list(group)
            if list(group) != [64, 1, 1]:
                raise legacy.OpenCLBridgeError("SDK norm kernel compiled work group differs")
            # Binary introspection is evidence only; implementations may not expose a binary.
            try:
                n = self._info(self._lib.clGetProgramInfo, self._program, 0x1162, ct.c_uint32, "program devices count")
                if not 1 <= n <= 32:
                    raise legacy.OpenCLBridgeError("Unexpected SDK program devices count")
                devices, sizes = (ct.c_void_p * n)(), (ct.c_size_t * n)()
                legacy._check(self._lib.clGetProgramInfo(self._program, 0x1163, ct.sizeof(devices), devices, None), "program devices")
                legacy._check(self._lib.clGetProgramInfo(self._program, 0x1165, ct.sizeof(sizes), sizes, None), "program binary sizes")
                if any(size > 64 * 1024 * 1024 for size in sizes):
                    raise legacy.OpenCLBridgeError("Unbounded SDK program binary")
                owners = [ct.create_string_buffer(size) if size else None for size in sizes]
                pointers = (ct.c_void_p * n)(*[ct.addressof(value) if value is not None else 0 for value in owners])
                legacy._check(self._lib.clGetProgramInfo(self._program, 0x1166, ct.sizeof(pointers), pointers, None), "program binaries")
                index = list(devices).index(self._device)
                self._stats["actual_SDK_program"].update(binary_bytes=sizes[index],
                    binary_sha256=hashlib.sha256(owners[index].raw).hexdigest() if owners[index] is not None else None)
            except BaseException as error:
                self._stats["actual_SDK_program"]["binary_query_error"] = str(error)

        def _diagnostic_read(self, handle, offset, shape):
            import numpy as np
            value = np.empty(shape, np.float32)
            legacy._check(self._lib.clEnqueueReadBuffer(self._queue, handle, 1, offset, value.nbytes,
                ct.c_void_p(value.ctypes.data), 0, None, None), "borrowed diagnostic readback")
            if not value.flags.owndata or not np.isfinite(value).all():
                raise legacy.OpenCLBridgeError("Borrowed diagnostic readback is not owned finite FP32")
            return value

        def _info(self, function, handle, name, type_, label):
            value = type_()
            legacy._check(function(handle, name, ct.sizeof(value), ct.byref(value), None), label)
            if value.value is None:
                raise legacy.OpenCLBridgeError(label + " returned a null handle")
            return value.value

        def _set_args(self, kernel, values):
            for index, value in enumerate(values):
                legacy._check(self._lib.clSetKernelArg(kernel, index, ct.sizeof(value), ct.byref(value)),
                              f"borrowed clSetKernelArg({index})")

        def run(self, q_mem, k_mem, v_mem, output_mem, *, before_input, before_output, after_output,
                diagnostic_consumer=None):
            if not all(callable(callback) for callback in (before_input, before_output, after_output)):
                raise TypeError("Explicit input/output synchronization and output-finite callbacks required")
            if diagnostic_consumer is not None and not callable(diagnostic_consumer):
                raise TypeError("Diagnostic consumer must be callable or None")
            with self._lock:
                if self._closed or self._failed:
                    raise legacy.OpenCLBridgeError("Borrowed original tail is closed or failed")
                self._stats["calls_attempted"] += 1
                events, imported, imported_fds, timings = [], [], [], {}
                started = last = time.perf_counter()
                try:
                    before_input()
                    now = time.perf_counter(); timings["input_TO_DEVICE_and_validation_seconds"] = now - last; last = now
                    before_output()
                    now = time.perf_counter(); timings["output_prewrite_TO_DEVICE_seconds"] = now - last; last = now
                    fields = [legacy._mem_fields(memory) for memory in (q_mem, k_mem, v_mem, output_mem)]
                    if any(row["size"] < self.elements * 4 or row["offset"] % 4 for row in fields):
                        raise legacy.OpenCLBridgeError("Original FP32 DMA input/output span is undersized or misaligned")
                    for name, row, flags in zip(("query", "key", "value", "proj_input"), fields, (4, 4, 4, 2)):
                        fd_start = len(self._fds)
                        handle = self._import_dma(row, flags, self._managed_host_consistency, name)
                        imported.append(handle)
                        imported_fds.extend(self._fds[fd_start:])
                    # Input offsets are relative to the imported allocation, never the CPU mapping.
                    self._set_args(self._kernels[0], [ct.c_void_p(imported[0]), ct.c_uint64(fields[0]["offset"] // 4),
                        ct.c_void_p(imported[1]), ct.c_uint64(fields[1]["offset"] // 4),
                        ct.c_void_p(self._buffers["parameters"]), ct.c_void_p(self._buffers["tables"]),
                        ct.c_void_p(self._buffers["norm_debug"]), ct.c_void_p(self._buffers["qk_rope"]),
                        ct.c_float(self._stats["epsilon"]), ct.c_float(self._stats["epsilon"]),
                        ct.c_uint32(int(diagnostic_consumer is not None))])
                    self._set_args(self._kernels[1], [ct.c_void_p(self._buffers["qk_rope"]),
                        ct.c_void_p(self._buffers["scores"]), ct.c_float(self._stats["scale"])])
                    self._set_args(self._kernels[2], [ct.c_void_p(imported[2]), ct.c_uint64(fields[2]["offset"] // 4),
                        ct.c_void_p(self._buffers["scores"]), ct.c_void_p(imported[3]), ct.c_uint64(fields[3]["offset"] // 4)])
                    now = time.perf_counter(); timings["DMA_import_and_args_seconds"] = now - last; last = now
                    for kernel, (global_size, local_size), name in zip(self._kernels, self._launches,
                            ("QK_norm_RoPE", "scaled_QK", "softmax_V_proj_layout")):
                        event = ct.c_void_p()
                        global_arg, local_arg = (ct.c_size_t * 1)(global_size), (ct.c_size_t * 1)(local_size)
                        waits = (ct.c_void_p * 1)(events[-1][1]) if events else None
                        code = self._lib.clEnqueueNDRangeKernel(self._queue, kernel, 1, None, global_arg,
                            local_arg, int(bool(events)), waits, ct.byref(event))
                        if event.value:
                            self._objects.append(("Event", event.value))
                            self._stats["events_created"] += 1
                            events.append((name, event.value))
                        legacy._check(code, "borrowed enqueue " + name)
                        if not event.value:
                            raise legacy.OpenCLBridgeError("Borrowed kernel enqueue returned no event")
                    legacy._check(self._lib.clFinish(self._queue), "borrowed original attention clFinish")
                    now = time.perf_counter(); timings["three_kernel_enqueue_and_finish_seconds"] = now - last; last = now
                    if diagnostic_consumer is not None:
                        normal = self._diagnostic_read(self._buffers["norm_debug"], 0, (2,) + self.shape)
                        rope = self._diagnostic_read(self._buffers["qk_rope"], 0, (2,) + self.shape)
                        b, h, s, _ = self.shape
                        diagnostics = {"GPU_q": self._diagnostic_read(imported[0], fields[0]["offset"], self.shape),
                            "GPU_k": self._diagnostic_read(imported[1], fields[1]["offset"], self.shape),
                            "GPU_v": self._diagnostic_read(imported[2], fields[2]["offset"], self.shape),
                            "GPU_proj_input": self._diagnostic_read(imported[3], fields[3]["offset"], (b, s, 1024)),
                            "GPU_q_post_norm": normal[0].copy(), "GPU_k_post_norm": normal[1].copy(),
                            "GPU_q_post_rope": rope[0].copy(), "GPU_k_post_rope": rope[1].copy(),
                            "GPU_scores": self._diagnostic_read(self._buffers["scores"], 0, (b, h, s, s)),
                            "GPU_parameters": self._diagnostic_read(self._buffers["parameters"], 0, (4, 64)),
                            "GPU_tables": self._diagnostic_read(self._buffers["tables"], 0, (4, b, 1, s, 32))}
                        legacy._check(self._lib.clFinish(self._queue), "borrowed diagnostic finish")
                        diagnostic_consumer(diagnostics)
                    now = time.perf_counter(); timings["optional_diagnostic_readback_and_consumer_seconds"] = now - last; last = now
                    timings["diagnostic_capture"] = diagnostic_consumer is not None
                    after_output()
                    now = time.perf_counter(); timings["output_FROM_DEVICE_owned_finite_seconds"] = now - last; last = now
                    device_timings = {}
                    if self._profile_events:
                        for name, event in events:
                            clocks = {}
                            for clock, value_name in (("queued", 0x1280), ("submit", 0x1281), ("start", 0x1282), ("end", 0x1283)):
                                value = ct.c_uint64()
                                legacy._check(self._lib.clGetEventProfilingInfo(event, value_name, ct.sizeof(value),
                                    ct.byref(value), None), "borrowed event profiling " + name)
                                clocks[clock] = value.value
                            if not clocks["queued"] <= clocks["submit"] <= clocks["start"] <= clocks["end"]:
                                raise legacy.OpenCLBridgeError("Borrowed event clocks are not ordered")
                            device_timings[name] = (clocks["end"] - clocks["start"]) / 1e9
                    timings["GPU_kernel_seconds_sum"] = sum(device_timings.values()) if self._profile_events else None
                    timings["device_kernel_seconds"] = device_timings
                    self._stats["calls_completed"] += 1
                except BaseException:
                    self._failed = True
                    raise
                finally:
                    # Blocking completion before releasing imported buffers or borrowed SDK tensors.
                    finish_code = self._lib.clFinish(self._queue)
                    if finish_code:
                        self._failed = True
                        self._stats["cleanup_errors"].append(f"Borrowed finish cleanup returned {finish_code}")
                    for _, event in reversed(events):
                        code = self._lib.clReleaseEvent(event)
                        if code == 0:
                            self._objects.remove(("Event", event))
                            self._stats["events_released"] += 1
                        else:
                            self._failed = True
                            self._stats["cleanup_errors"].append(f"Borrowed release event returned {code}")
                    released_all = finish_code == 0
                    if finish_code == 0:
                        for handle in reversed(imported):
                            code = self._lib.clReleaseMemObject(handle)
                            if code == 0:
                                self._objects.remove(("MemObject", handle))
                                self._memory.remove(handle)
                            else:
                                released_all = False
                                self._failed = True
                                self._stats["cleanup_errors"].append(f"Borrowed release DMA import returned {code}")
                    if released_all:
                        for fd in reversed(imported_fds):
                            try:
                                os.close(fd)
                                self._fds.remove(fd)
                            except OSError as error:
                                self._failed = True
                                self._stats["cleanup_errors"].append("Borrowed close DMA duplicate: " + str(error))
                    timings["profiling_event_release_DMA_release_seconds"] = time.perf_counter() - last
                    timings["complete_borrowed_tail_seconds"] = time.perf_counter() - started
                    self._stats["last_timing"] = timings
                if self._failed:
                    raise legacy.OpenCLBridgeError("Borrowed attention resources did not cleanly finish/release")
                return timings

        def close(self):
            # The inherited close would clear a queue it owns. Never run it on SDK handles.
            with self._lock:
                if self._closed and not self._objects and not self._fds:
                    return
                errors = []
                finished = True
                if self._queue and self._api_bound:
                    code = self._lib.clFinish(self._queue)
                    finished = code == 0
                    if code:
                        errors.append(f"Borrowed clFinish(close) returned {code}")
                if finished:
                    remaining = []
                    for kind, handle in reversed(self._objects):
                        code = getattr(self._lib, "clRelease" + kind)(handle)
                        if code:
                            errors.append(f"Borrowed clRelease{kind} returned {code}")
                            remaining.append((kind, handle))
                        elif kind == "Event":
                            self._stats["events_released"] += 1
                    self._objects = list(reversed(remaining))
                    if not self._objects:
                        self._memory.clear()
                        remaining_fds = []
                        for fd in reversed(self._fds):
                            try:
                                os.close(fd)
                            except OSError as error:
                                errors.append("Borrowed close duplicate DMA fd: " + str(error))
                                remaining_fds.append(fd)
                        self._fds = list(reversed(remaining_fds))
                self._closed = True
                self._stats["cleanup_errors"].extend(errors)
                if errors:
                    self._failed = True
                    raise legacy.OpenCLBridgeError("; ".join(errors))

        def metrics(self):
            result = super().metrics()
            result["live_owned_OpenCL_objects_excluding_borrowed_SDK_handles"] = len(self._objects)
            result["borrowed_SDK_handles_released_by_bridge"] = False
            return result

    return BorrowedOriginalAttentionTail
