"""Source-only optional OpenCL implementation of the original VGGT attention tail.

Instantiate only on RK3588/S. The caller supplies the frozen v81 OpenCL loader
module; its proven ctypes ABI, object tracking and release logic are reused.
No library is loaded on import. No model, SDK, Torch or alternative model is
imported. This component copies original captured FP32 operands; it does not
claim native INT8 equivalence, NPU integration or zero-copy execution.

The three kernels retain Q/K LayerNorm, the original two-axis RoPE tables, all
queries/keys/values, scale=.125, stable softmax and the original projection input
layout. No relaxed/native math, causal mask, attention omission or approximation.
FP32 reduction order can differ from the original Torch implementation, so every
intermediate and final output requires comparison against its original capture.
"""
from __future__ import annotations

import ctypes as ct
import hashlib
import threading
import time


LEGACY_SOURCE_SHA256 = "f06486299101fb233c68abc9908e8c2629c4cb1f7dca6241133e6e222634acc2"
BUILD_OPTIONS = b"-cl-std=CL1.2"

KERNEL_SOURCE = r"""
#pragma OPENCL FP_CONTRACT OFF
#ifndef TOKENS
#error TOKENS must be the complete original sequence length
#endif
#ifndef BATCH
#error BATCH must be the original batch
#endif
#ifndef LOCAL_KEYS
#error LOCAL_KEYS must cover all original keys
#endif
#define HEADS 16
#define DIM 64
#define ROWS (BATCH * HEADS * TOKENS)
#define VALUES (ROWS * DIM)
#define ROPE_VALUES (BATCH * TOKENS * 32)
#define TILES ((TOKENS + 7) / 8)

// One 64-lane group computes both Q and K for a complete original head row.
// parameters=[q_weight,q_bias,k_weight,k_bias], each [64].
// tables=[cos_y,sin_y,cos_x,sin_x], each [B,1,S,32], from original RoPE.
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void original_qk_norm_rope(
    __global const float *qkv, __global const float *parameters,
    __global const float *tables, __global float *norm_debug,
    __global float *qk_rope, const float q_eps, const float k_eps,
    const uint save_norm)
{
    const uint row = (uint)get_group_id(0), d = (uint)get_local_id(0);
    const uint offset = row * DIM + d;
    const float2 original = (float2)(qkv[offset], qkv[VALUES + offset]);
    __local float2 scratch[64];
    __local float2 normalized[64];
    scratch[d] = original;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = 32; stride; stride >>= 1) {
        if (d < stride) scratch[d] = scratch[d] + scratch[d + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float2 mean = scratch[0] * 0.015625f;
    barrier(CLK_LOCAL_MEM_FENCE); // every lane must read mean before scratch reuse
    const float2 centered = original - mean;
    scratch[d] = centered * centered;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = 32; stride; stride >>= 1) {
        if (d < stride) scratch[d] = scratch[d] + scratch[d + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float2 variance = scratch[0] * 0.015625f;
    const float2 inverse_std = (float2)(1.0f) / sqrt(variance + (float2)(q_eps, k_eps));
    const float2 weight = (float2)(parameters[d], parameters[2 * DIM + d]);
    const float2 bias = (float2)(parameters[DIM + d], parameters[3 * DIM + d]);
    const float2 affine = (centered * inverse_std) * weight + bias;
    normalized[d] = affine;
    if (save_norm) {
        norm_debug[offset] = affine.x;
        norm_debug[VALUES + offset] = affine.y;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const uint batch = row / (HEADS * TOKENS), sequence = row % TOKENS;
    const uint within_axis = d & 31u;
    const uint table_offset = (batch * TOKENS + sequence) * 32 + within_axis;
    const uint axis_offset = d < 32 ? 0 : 2 * ROPE_VALUES;
    const float cosine = tables[axis_offset + table_offset];
    const float sine = tables[axis_offset + ROPE_VALUES + table_offset];
    // Rotate separately within the original vertical/horizontal 32-dim blocks.
    const float2 partner = normalized[d ^ 16u];
    const float2 rotated = within_axis < 16 ? -partner : partner;
    const float2 rope = affine * cosine + rotated * sine;
    qk_rope[offset] = rope.x;
    qk_rope[VALUES + offset] = rope.y;
}

// One 8x8 tile per 64-lane group. Q and K rows are shared through local memory;
// the dot reduction is private, avoiding a workgroup barrier for every key.
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void original_scaled_qk(__global const float *qk_rope,
                                 __global float *scores, const float scale)
{
    const uint group = (uint)get_group_id(0), lane = (uint)get_local_id(0);
    const uint key_tile = group % TILES;
    const uint query_tile = (group / TILES) % TILES;
    const uint batch_head = group / (TILES * TILES);
    const uint query_start = query_tile * 8, key_start = key_tile * 8;
    __local float queries[8 * DIM];
    __local float keys[8 * DIM];
    for (uint i = lane; i < 8 * DIM; i += 64) {
        const uint row = i / DIM, d = i % DIM;
        const uint query = query_start + row, key = key_start + row;
        queries[i] = query < TOKENS ? qk_rope[(batch_head * TOKENS + query) * DIM + d] : 0.0f;
        keys[i] = key < TOKENS ? qk_rope[VALUES + (batch_head * TOKENS + key) * DIM + d] : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const uint query = query_start + lane / 8, key = key_start + lane % 8;
    if (query < TOKENS && key < TOKENS) {
        const uint query_offset = (lane / 8) * DIM, key_offset = (lane % 8) * DIM;
        float4 partial = (float4)(0.0f);
        for (uint i = 0; i < DIM / 4; ++i)
            partial = partial + vload4(i, queries + query_offset) * vload4(i, keys + key_offset);
        const float dot_product = (partial.x + partial.y) + (partial.z + partial.w);
        scores[(batch_head * TOKENS + query) * TOKENS + key] = dot_product * scale;
    }
}

// Stable softmax over EVERY original key, followed by the complete V product.
// Writes the actual original consumer layout [B,S,H*D], with no CPU transpose.
__attribute__((reqd_work_group_size(LOCAL_KEYS, 1, 1)))
__kernel void original_softmax_v_to_proj(
    __global const float *qkv, __global const float *scores, __global float *proj_input)
{
    const uint row = (uint)get_group_id(0), lane = (uint)get_local_id(0);
    const uint batch_head = row / TOKENS, query = row % TOKENS;
    const uint batch = batch_head / HEADS, head = batch_head % HEADS;
    const float score = lane < TOKENS ? scores[row * TOKENS + lane] : -INFINITY;
    __local float scratch[LOCAL_KEYS];
    __local float weights[LOCAL_KEYS];
    scratch[lane] = score;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = LOCAL_KEYS / 2; stride; stride >>= 1) {
        if (lane < stride) scratch[lane] = fmax(scratch[lane], scratch[lane + stride]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float maximum = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE); // every lane must read max before scratch reuse
    const float exponent = lane < TOKENS ? exp(score - maximum) : 0.0f;
    weights[lane] = exponent;
    scratch[lane] = exponent;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = LOCAL_KEYS / 2; stride; stride >>= 1) {
        if (lane < stride) scratch[lane] = scratch[lane] + scratch[lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float denominator = scratch[0];
    weights[lane] = weights[lane] / denominator;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lane < DIM) {
        float output = 0.0f;
        for (uint key = 0; key < TOKENS; ++key)
            output = output + weights[key] * qkv[2 * VALUES + (batch_head * TOKENS + key) * DIM + lane];
        proj_input[((batch * TOKENS + query) * HEADS + head) * DIM + lane] = output;
    }
}
"""


def make_bridge_class(legacy):
    """Reuse the frozen v81 loader without importing it or loading a GPU library."""
    class OriginalAttentionTailBridge(legacy.NativeErfGeluBridge):
        def __init__(self, *, shape, parameters, tables, q_eps, k_eps, scale,
                     library, platform_index=0, device_index=0):
            legacy._board_guard()
            import numpy as np
            self.np = np
            self._lock = threading.RLock()
            self._fds, self._memory, self._objects = [], [], []
            self._context = self._queue = self._program = self._kernel = None
            self._closed, self._failed = False, False
            self._owners = ()
            self._pending_host_uploads = []
            self._stats = dict(calls_attempted=0, calls_completed=0, events_created=0,
                               events_released=0, cleanup_errors=[])
            self.shape = tuple(shape)
            if self.shape not in ((2, 16, 53, 64), (1, 16, 106, 64)):
                raise ValueError("Only complete original frame0/global0 attention shapes are allowed")
            batch, heads, tokens, dim = self.shape
            if q_eps != 1e-5 or k_eps != 1e-5 or scale != .125:
                raise ValueError("Original QK normalization eps and SDPA scale must be retained")
            if len(parameters) != 4 or len(tables) != 4:
                raise ValueError("Four original affine parameters and four original indexed RoPE tables required")
            for value in parameters:
                if value.dtype != np.float32 or value.shape != (64,) or not np.isfinite(value).all():
                    raise ValueError("Original finite FP32 norm weight/bias [64] required")
            for value in tables:
                if value.dtype != np.float32 or value.shape != (batch, 1, tokens, 32) or not np.isfinite(value).all():
                    raise ValueError("Original indexed RoPE tables [B,1,S,32] required")
            self._parameters = np.stack(parameters).copy(order="C")
            self._tables = np.stack(tables).copy(order="C")
            self._parameters.flags.writeable = self._tables.flags.writeable = False
            self._owners = (self._parameters, self._tables)
            self.elements = batch * heads * tokens * dim
            self.rows = batch * heads * tokens
            local_keys = 64 if tokens == 53 else 128
            options = BUILD_OPTIONS + f" -DBATCH={batch} -DTOKENS={tokens} -DLOCAL_KEYS={local_keys}".encode()
            self._stats = dict(calls_attempted=0, calls_completed=0, events_created=0, events_released=0,
                cleanup_errors=[], shape=list(self.shape), consumer_shape=[batch, tokens, 1024],
                kernel_sha256=hashlib.sha256(KERNEL_SOURCE.encode()).hexdigest(),
                build_options=options.decode(), library=str(library), original_model_only=True,
                original_all_tokens_retained=True, original_two_axis_RoPE_tables_used=True,
                standard_FP32_sqrt_exp=True, contraction_disabled=True,
                q_eps=q_eps, k_eps=k_eps, scale=scale, GPU_staging_copies=True,
                DMA_imported=False, zero_copy_performance_validated=False,
                native_INT8_equivalence_verified=False, model_or_trajectory_accuracy_validated=False,
                fixed_parameter_raw_sha256=hashlib.sha256(self._parameters.tobytes()).hexdigest(),
                fixed_RoPE_tables_raw_sha256=hashlib.sha256(self._tables.tobytes()).hexdigest())
            started = time.perf_counter()
            try:
                self._lib = ct.CDLL(str(library))
                self._bind_api()
                ptr, uint, size, integer = ct.c_void_p, ct.c_uint32, ct.c_size_t, ct.c_int32
                transfer_signature = [ptr, ptr, uint, size, size, ptr, uint, ct.POINTER(ptr), ct.POINTER(ptr)]
                for name in ("clEnqueueWriteBuffer", "clEnqueueReadBuffer"):
                    function = getattr(self._lib, name)
                    function.argtypes, function.restype = transfer_signature, integer
                self._platform = self._select_handles("platform", platform_index)
                self._device = self._select_handles("device", device_index)
                self._stats.update(device_name=self._device_string(0x102B),
                                   driver_version=self._device_string(0x102D))
                limit = ct.c_size_t()
                legacy._check(self._lib.clGetDeviceInfo(self._device, 0x1004, ct.sizeof(limit),
                                                       ct.byref(limit), None), "max workgroup size")
                if limit.value < local_keys:
                    raise legacy.OpenCLBridgeError("OpenCL device cannot cover every original attention key")
                self._devices = (ct.c_void_p * 1)(self._device)
                properties = (ct.c_ssize_t * 3)(0x1084, self._platform, 0)
                self._context = self._create("clCreateContext", properties, 1, self._devices, None, None)
                self._queue = self._create("clCreateCommandQueue", self._context, self._device, 2)
                encoded = KERNEL_SOURCE.encode()
                source, lengths = (ct.c_char_p * 1)(encoded), (ct.c_size_t * 1)(len(encoded))
                self._program = self._create("clCreateProgramWithSource", self._context, 1, source, lengths)
                code = self._lib.clBuildProgram(self._program, 1, self._devices, options, None, None)
                if code:
                    raise legacy.OpenCLBridgeError(f"clBuildProgram returned {code}: {self._build_log()}")
                self._kernels = [self._create("clCreateKernel", self._program, name) for name in
                    (b"original_qk_norm_rope", b"original_scaled_qk", b"original_softmax_v_to_proj")]
                self._buffers = {}
                for name, count, flags, value in (
                    ("qkv", 3 * self.elements, 4, None),
                    ("parameters", 256, 4 | 32, self._parameters),
                    ("tables", 4 * batch * tokens * 32, 4 | 32, self._tables),
                    ("norm_debug", 2 * self.elements, 1, None),
                    ("qk_rope", 2 * self.elements, 1, None),
                    ("scores", batch * heads * tokens * tokens, 1, None),
                    ("proj_input", self.elements, 2, None)):
                    handle = self._create("clCreateBuffer", self._context, flags, count * 4,
                                          None if value is None else ct.c_void_p(value.ctypes.data))
                    self._memory.append(handle)
                    self._buffers[name] = handle
                self._set_args(self._kernels[0], [ct.c_void_p(self._buffers[name]) for name in
                    ("qkv", "parameters", "tables", "norm_debug", "qk_rope")] +
                    [ct.c_float(q_eps), ct.c_float(k_eps), ct.c_uint32(0)])
                self._set_args(self._kernels[1], [ct.c_void_p(self._buffers["qk_rope"]),
                    ct.c_void_p(self._buffers["scores"]), ct.c_float(scale)])
                self._set_args(self._kernels[2], [ct.c_void_p(self._buffers[name]) for name in
                    ("qkv", "scores", "proj_input")])
                tiles = (tokens + 7) // 8
                self._launches = [(self.rows * 64, 64), (batch * heads * tiles * tiles * 64, 64),
                                  (self.rows * local_keys, local_keys)]
                self._stats["OpenCL_buffer_bytes"] = (3 * self.elements + 256 + 4 * batch * tokens * 32 +
                    2 * self.elements + 2 * self.elements + batch * heads * tokens * tokens + self.elements) * 4
                self._stats["initialization_seconds"] = time.perf_counter() - started
            except BaseException as error:
                self._failed = True
                try:
                    self.close()
                except BaseException as cleanup:
                    error.add_note(str(cleanup))
                raise

        def _set_args(self, kernel, values):
            for index, value in enumerate(values):
                legacy._check(self._lib.clSetKernelArg(kernel, index, ct.sizeof(value), ct.byref(value)),
                              f"clSetKernelArg({index})")

        def _event_call(self, function, args, events, label):
            event = ct.c_void_p()
            code = function(*args, ct.byref(event))
            if event.value:
                self._objects.append(("Event", event.value))
                self._stats["events_created"] += 1
                events.append((label, event.value))
            legacy._check(code, label)
            if not event.value:
                raise legacy.OpenCLBridgeError(label + " returned no profiling event")

        def _read_buffer(self, name, value, events, label):
            self._event_call(self._lib.clEnqueueReadBuffer,
                (self._queue, self._buffers[name], 1, 0, value.nbytes, ct.c_void_p(value.ctypes.data), 0, None),
                events, label)

        def run(self, q, k, v, *, capture_intermediates=False):
            np = self.np
            with self._lock:
                if self._closed or self._failed:
                    raise legacy.OpenCLBridgeError("attention bridge is closed or failed")
                self._stats["calls_attempted"] += 1
                events, timings, diagnostics = [], {}, {}
                started = last = time.perf_counter()
                try:
                    for value in (q, k, v):
                        if value.shape != self.shape or value.dtype != np.float32 or not np.isfinite(value).all():
                            raise ValueError("Complete original finite FP32 Q/K/V operands required")
                    if type(capture_intermediates) is not bool:
                        raise TypeError("capture_intermediates must be bool")
                    packed = np.ascontiguousarray(np.stack((q, k, v)))
                    output = np.empty((self.shape[0], self.shape[2], 1024), np.float32)
                    flag = ct.c_uint32(int(capture_intermediates))
                    legacy._check(self._lib.clSetKernelArg(self._kernels[0], 7, ct.sizeof(flag), ct.byref(flag)),
                                  "set diagnostic norm output flag")
                    now = time.perf_counter(); timings["input_finite_pack_owned_allocation_seconds"] = now - last; last = now
                    self._pending_host_uploads.append(packed)
                    self._event_call(self._lib.clEnqueueWriteBuffer,
                        (self._queue, self._buffers["qkv"], 0, 0, packed.nbytes,
                         ct.c_void_p(packed.ctypes.data), 0, None), events, "QKV_upload")
                    for kernel, (global_size, local_size), name in zip(self._kernels, self._launches,
                            ("QK_norm_RoPE", "scaled_QK", "softmax_V_proj_layout")):
                        global_arg, local_arg = (ct.c_size_t * 1)(global_size), (ct.c_size_t * 1)(local_size)
                        self._event_call(self._lib.clEnqueueNDRangeKernel,
                            (self._queue, kernel, 1, None, global_arg, local_arg, 0, None), events, name)
                    now = time.perf_counter(); timings["upload_and_three_kernel_enqueue_seconds"] = now - last; last = now
                    self._read_buffer("proj_input", output, events, "proj_input_download")
                    legacy._check(self._lib.clFinish(self._queue), "clFinish(original attention tail)")
                    now = time.perf_counter(); timings["consumer_download_and_finish_seconds"] = now - last; last = now
                    if capture_intermediates:
                        normal = np.empty((2,) + self.shape, np.float32)
                        rope = np.empty((2,) + self.shape, np.float32)
                        self._read_buffer("norm_debug", normal, events, "norm_debug_download")
                        self._read_buffer("qk_rope", rope, events, "qk_RoPE_debug_download")
                        diagnostics = dict(q_post_norm=normal[0].copy(), k_post_norm=normal[1].copy(),
                                           q_post_rope=rope[0].copy(), k_post_rope=rope[1].copy())
                    now = time.perf_counter(); timings["optional_owned_diagnostic_download_seconds"] = now - last; last = now
                    if not output.flags.owndata or not output.flags.c_contiguous or not np.isfinite(output).all():
                        raise ValueError("Owned finite original projection input was not produced")
                    if any(not value.flags.owndata or not np.isfinite(value).all() for value in diagnostics.values()):
                        raise ValueError("Nonfinite or unowned GPU intermediate")
                    now = time.perf_counter(); timings["owned_output_finite_seconds"] = now - last; last = now
                    device_timings = {}
                    for name, event in events:
                        clocks = {}
                        for clock, key in (("queued", 0x1280), ("submit", 0x1281), ("start", 0x1282), ("end", 0x1283)):
                            value = ct.c_uint64()
                            legacy._check(self._lib.clGetEventProfilingInfo(event, key, ct.sizeof(value),
                                                                           ct.byref(value), None), "profiling " + name)
                            clocks[clock] = value.value
                        if not clocks["queued"] <= clocks["submit"] <= clocks["start"] <= clocks["end"]:
                            raise legacy.OpenCLBridgeError("Invalid profiling clock order: " + name)
                        device_timings[name] = {"seconds": (clocks["end"] - clocks["start"]) / 1e9,
                                                "clocks_ns": clocks}
                    timings["device_events"] = device_timings
                    timings["GPU_kernel_seconds_sum"] = sum(device_timings[name]["seconds"] for name in
                        ("QK_norm_RoPE", "scaled_QK", "softmax_V_proj_layout"))
                    self._stats["calls_completed"] += 1
                except BaseException:
                    self._failed = True
                    raise
                finally:
                    # Complete any nonblocking host upload before its owned packed array dies.
                    finish_code = self._lib.clFinish(self._queue)
                    if finish_code:
                        self._failed = True
                        self._stats["cleanup_errors"].append(f"clFinish(run cleanup) returned {finish_code}")
                    else:
                        self._pending_host_uploads.clear()
                    for _, event in reversed(events):
                        code = self._lib.clReleaseEvent(event)
                        if code == 0:
                            self._stats["events_released"] += 1
                            self._objects.remove(("Event", event))
                        else:
                            self._failed = True
                            self._stats["cleanup_errors"].append(f"clReleaseEvent returned {code}")
                    timings["profiling_event_release_and_final_finish_seconds"] = time.perf_counter() - last
                    timings["complete_tail_seconds"] = time.perf_counter() - started
                    timings["capture_intermediates"] = capture_intermediates
                    timings["host_upload_bytes"] = 3 * self.elements * 4
                    timings["host_download_bytes"] = self.elements * 4 * (5 if capture_intermediates else 1)
                    self._stats["last_timing"] = timings
                if self._failed:
                    raise legacy.OpenCLBridgeError("attention profiling/event cleanup failed")
                return output, diagnostics, timings

        def close(self):
            before = {handle for kind, handle in self._objects if kind == "Event"}
            try:
                if self._queue and self._lib.clFinish(self._queue) == 0:
                    self._pending_host_uploads.clear()
                super().close()
            finally:
                remaining = {handle for kind, handle in self._objects if kind == "Event"}
                self._stats["events_released"] += len(before - remaining)

        def metrics(self):
            result = super().metrics()
            result["pending_owned_host_uploads"] = len(self._pending_host_uploads)
            result["fully_released"] = result["fully_released"] and not self._pending_host_uploads
            return result

    return OriginalAttentionTailBridge
