"""Board-only original softmax from native FP32 QK C4 to native Half PV A8.

Reuse the SHA-pinned v81 OpenCL ABI and DMA import implementation, never its
shape-specific constructor or failure cleanup. No library is loaded on import.
Caller owns the two RKNN allocations and must close this bridge successfully
before freeing them. No staging matrix, SDK GPU callback, or CPU fallback.
"""
from __future__ import annotations

import ctypes as ct
import hashlib
import importlib.util
from pathlib import Path
import threading
import time

LEGACY_SHA = 'f06486299101fb233c68abc9908e8c2629c4cb1f7dca6241133e6e222634acc2'
LIBRARY_SHA = 'd57e53240b5078eee5cfbc79e103544c090a7825c09fdf3877fee45084897dda'
KERNEL_SHA = '5cbb36e03e09c38db9e6769f067e3168e1cf7f67d719009601c4456e9f2b150a'
BUILD_OPTIONS = b'-cl-std=CL1.2'
ROWS, KEYS, PAD_KEYS, WG_SIZE = 1041, 1041, 1056, 256


class SoftmaxBridgeError(RuntimeError):
    pass


def _check(code, operation):
    if code:
        raise SoftmaxBridgeError(f'{operation} returned {int(code)}')


def _require(ok, message):
    if not ok:
        raise SoftmaxBridgeError(message)


def _pin(path, expected):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(4 << 20), b''):
            digest.update(data)
    _require(digest.hexdigest() == expected, 'Pinned source/library differs: ' + str(path))
    return path


class NativeAttentionSoftmaxBridge:
    def __init__(self, qk_c_mem, pv_a_mem, *, legacy_source_path, library_path,
                 kernel_source_path=None, rows=ROWS, keys=KEYS, padded_keys=PAD_KEYS):
        # Initialize owners before any import/library/device allocation. On error
        # error.softmax_bridge exposes this object if cleanup remains incomplete.
        self._lock = threading.RLock()
        self._owners = (qk_c_mem, pv_a_mem)
        self._fds, self._memory, self._objects, self._events = [], [], [], []
        self._context = self._queue = self._program = self._kernel = None
        self._lib = self._legacy_class = None
        self._status = ct.c_uint32(0)  # Strong host owner for any pending scalar IO.
        self._closed, self._failed, self._queue_idle = False, False, True
        self.last_timings = {}
        self._stats = dict(calls_attempted=0, calls_completed=0, events_created=0, events_released=0,
            objects_created={}, objects_released={}, cleanup_errors=[], rows=ROWS, keys=KEYS,
            padded_keys=PAD_KEYS, input_layout=[PAD_KEYS // 4, ROWS, 4],
            output_layout=[PAD_KEYS // 8, ROWS, 8], output_dtype='float16',
            original_scale=0.125, build_options=BUILD_OPTIONS.decode(),
            standard_FP32_exp=True, FP_CONTRACT_OFF=True, reduction_order_changed=True,
            padding_excluded_from_softmax=True, padding_written_zero=True,
            GPU_matrix_staging_copies=False, SDK_GPU_callback=False,
            managed_host_consistency=True, frequency_or_driver_writes=False,
            original_model_only=True, model_or_trajectory_or_performance_accepted=False,
            finite_error_bits={'nonfinite_score': 1, 'nonfinite_exp': 2, 'invalid_sum': 4, 'invalid_probability': 8})
        started = time.perf_counter()
        try:
            _require((rows, keys, padded_keys) == (ROWS, KEYS, PAD_KEYS), 'Only the complete original1041 shape is supported')
            legacy_path = _pin(legacy_source_path, LEGACY_SHA)
            spec = importlib.util.spec_from_file_location('_v119_frozen_v81_opencl_abi', legacy_path)
            _require(spec is not None and spec.loader is not None, 'Cannot load pinned OpenCL source')
            legacy = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(legacy)
            legacy._board_guard()
            self._legacy_module, self._legacy_class = legacy, legacy.NativeErfGeluBridge
            # Reuse only the existing typed loader/queries/import resolver.
            for name in ('_bind_api', '_select_handles', '_string', '_device_string', '_build_log', '_import_function'):
                setattr(self, name, getattr(self._legacy_class, name).__get__(self, type(self)))
            self._source, self._destination = legacy._mem_fields(qk_c_mem), legacy._mem_fields(pv_a_mem)
            _require(self._source['size'] >= ROWS * PAD_KEYS * 4 and self._source['offset'] % 4 == 0,
                'QK C4 allocation is undersized or misaligned')
            _require(self._destination['size'] >= ROWS * PAD_KEYS * 2 and self._destination['offset'] % 2 == 0,
                'PV A8 allocation is undersized or misaligned')
            _require(self._source['fd'] != self._destination['fd'], 'Use distinct original QK C and PV A allocations')
            kernel_path = _pin(kernel_source_path or Path(__file__).with_name('attention_softmax.cl'), KERNEL_SHA)
            library = _pin(library_path, LIBRARY_SHA)
            self._stats.update(legacy_source=str(legacy_path), legacy_source_sha256=LEGACY_SHA,
                library=str(library), library_sha256=LIBRARY_SHA,
                kernel_source=str(kernel_path), kernel_source_sha256=KERNEL_SHA)
            self._lib = ct.CDLL(str(library))
            self._bind_api()
            ptr, uint, size = ct.c_void_p, ct.c_uint32, ct.c_size_t
            for name in ('clEnqueueReadBuffer', 'clEnqueueWriteBuffer'):
                f = getattr(self._lib, name)
                f.argtypes = [ptr, ptr, uint, size, size, ptr, uint, ct.POINTER(ptr), ct.POINTER(ptr)]
                f.restype = ct.c_int32
            self._lib.clGetKernelWorkGroupInfo.argtypes = [ptr, ptr, uint, size, ptr, ct.POINTER(size)]
            self._lib.clGetKernelWorkGroupInfo.restype = ct.c_int32
            self._platform = self._select_handles('platform', 0)
            self._device = self._select_handles('device', 0)
            required = {'cl_khr_fp16', 'cl_arm_import_memory', 'cl_arm_import_memory_dma_buf'}
            _require(required <= set(self._device_string(0x1030).split()), 'Required Mali FP16/DMA-import extension missing')
            self._stats.update(device_name=self._device_string(0x102B), driver_version=self._device_string(0x102D),
                required_extensions=sorted(required))
            self._devices = (ptr * 1)(self._device)
            properties = (ct.c_ssize_t * 3)(0x1084, self._platform, 0)
            self._context = self._create('clCreateContext', properties, 1, self._devices, None, None)
            self._queue = self._create('clCreateCommandQueue', self._context, self._device, 2)
            source_bytes = kernel_path.read_bytes()
            source = (ct.c_char_p * 1)(source_bytes)
            lengths = (size * 1)(len(source_bytes))
            self._program = self._create('clCreateProgramWithSource', self._context, 1, source, lengths)
            code = self._lib.clBuildProgram(self._program, 1, self._devices, BUILD_OPTIONS, None, None)
            if code:
                raise SoftmaxBridgeError(f'clBuildProgram returned {code}: {self._build_log()}')
            actual_options = self._string(self._lib.clGetProgramBuildInfo, (self._program, self._device, 0x1182))
            _require(actual_options.split() == BUILD_OPTIONS.decode().split(), 'Unexpected OpenCL compilation options: ' + actual_options)
            self._stats['actual_build_options'] = actual_options
            self._kernel = self._create('clCreateKernel', self._program, b'original_softmax_c4_to_half_a8')
            max_group, local_bytes = ct.c_size_t(), ct.c_uint64()
            _check(self._lib.clGetKernelWorkGroupInfo(self._kernel, self._device, 0x11B0,
                ct.sizeof(max_group), ct.byref(max_group), None), 'kernel workgroup size')
            _check(self._lib.clGetKernelWorkGroupInfo(self._kernel, self._device, 0x11B2,
                ct.sizeof(local_bytes), ct.byref(local_bytes), None), 'kernel local memory')
            device_local = ct.c_uint64()
            _check(self._lib.clGetDeviceInfo(self._device, 0x1023, ct.sizeof(device_local), ct.byref(device_local), None), 'device local memory')
            _require(max_group.value >= WG_SIZE and local_bytes.value <= device_local.value, 'Kernel exceeds reported Mali workgroup/local-memory limits')
            self._stats.update(kernel_max_workgroup_size=max_group.value, kernel_local_memory_bytes=local_bytes.value,
                device_local_memory_bytes=device_local.value, workgroup_size=WG_SIZE)
            self._import = self._import_function()
            src = self._import_dma(self._source, 4, True, 'QK_C')
            dst = self._import_dma(self._destination, 2, True, 'PV_A')
            self._flag_mem = self._create('clCreateBuffer', self._context, 1, ct.sizeof(self._status), None)
            self._memory.append(self._flag_mem)
            args = (ptr(src), ct.c_uint64(self._source['offset'] // 4),
                ptr(dst), ct.c_uint64(self._destination['offset'] // 2), ptr(self._flag_mem))
            for index, value in enumerate(args):
                _check(self._lib.clSetKernelArg(self._kernel, index, ct.sizeof(value), ct.byref(value)), 'kernel argument ' + str(index))
            self._global, self._local = (size * 1)(ROWS * WG_SIZE), (size * 1)(WG_SIZE)
            self._stats['initialization_seconds'] = time.perf_counter() - started
        except BaseException as error:
            self._failed = True
            error.softmax_bridge = self
            try:
                self.close()
            except BaseException as cleanup:
                error.add_note('Softmax bridge cleanup: ' + str(cleanup))
            raise

    def _remember_new_objects(self, before):
        for suffix, handle in self._objects[before:]:
            counts = self._stats['objects_created']
            counts[suffix] = counts.get(suffix, 0) + 1

    def _create(self, name, *args):
        before = len(self._objects)
        try:
            return self._legacy_class._create(self, name, *args)
        finally:
            self._remember_new_objects(before)

    def _import_dma(self, *args):
        before = len(self._objects)
        try:
            return self._legacy_class._import_dma(self, *args)
        finally:
            self._remember_new_objects(before)

    def _finish(self, operation):
        _check(self._lib.clFinish(self._queue), operation)
        self._queue_idle = True

    def _release_events(self):
        remaining = []
        for event in reversed(self._events):
            code = self._lib.clReleaseEvent(event)
            if code:
                remaining.append(event)
                self._stats['cleanup_errors'].append('clReleaseEvent returned ' + str(code))
            else:
                self._stats['events_released'] += 1
        self._events = list(reversed(remaining))

    def run(self, *, before_input, before_output, after_output):
        _require(all(callable(f) for f in (before_input, before_output, after_output)), 'Three explicit RKNN synchronization callbacks required')
        with self._lock:
            _require(not self._closed and not self._failed and not self._events, 'Bridge closed/failed or unreleased event')
            self._stats['calls_attempted'] += 1
            started = last = time.perf_counter()
            timing, event = {}, ct.c_void_p()
            try:
                before_input()
                now = time.perf_counter(); timing['QK_C_FROM_DEVICE_seconds'] = now-last; last=now
                before_output()
                now = time.perf_counter(); timing['PV_A_prewrite_TO_DEVICE_seconds'] = now-last; last=now
                self._status.value = 0
                self._queue_idle = False
                _check(self._lib.clEnqueueWriteBuffer(self._queue, self._flag_mem, 1, 0,
                    ct.sizeof(self._status), ct.byref(self._status), 0, None, None), 'reset finite flag')
                now = time.perf_counter(); timing['finite_flag_reset_seconds'] = now-last; last=now
                code = self._lib.clEnqueueNDRangeKernel(self._queue, self._kernel, 1, None,
                    self._global, self._local, 0, None, ct.byref(event))
                if event.value:
                    self._events.append(event.value)
                    self._stats['events_created'] += 1
                _check(code, 'enqueue original softmax')
                _require(event.value is not None, 'Kernel returned no profiling event')
                now = time.perf_counter(); timing['kernel_enqueue_seconds'] = now-last; last=now
                self._finish('finish original softmax')
                now = time.perf_counter(); timing['kernel_finish_seconds'] = now-last; last=now
                self._queue_idle = False
                _check(self._lib.clEnqueueReadBuffer(self._queue, self._flag_mem, 1, 0,
                    ct.sizeof(self._status), ct.byref(self._status), 0, None, None), 'read finite flag')
                self._finish('finish finite flag read')
                timing['finite_error_bits'] = self._status.value
                _require(self._status.value == 0, 'GPU softmax finite/error flag: ' + str(self._status.value))
                now = time.perf_counter(); timing['finite_flag_read_check_seconds'] = now-last; last=now
                after_output()
                now = time.perf_counter(); timing['PV_A_postwrite_FROM_DEVICE_seconds'] = now-last; last=now
                clock = {}
                for name, key in (('queued',0x1280),('submit',0x1281),('start',0x1282),('end',0x1283)):
                    value = ct.c_uint64()
                    _check(self._lib.clGetEventProfilingInfo(event, key, ct.sizeof(value), ct.byref(value), None), 'event ' + name)
                    clock[name] = value.value
                _require(clock['queued'] <= clock['submit'] <= clock['start'] <= clock['end'], 'GPU profiling clock order differs')
                timing.update(GPU_kernel_seconds=(clock['end']-clock['start'])/1e9, GPU_event_clocks_ns=clock)
            except BaseException:
                self._failed = True
                raise
            finally:
                # clFinish failure leaves event/import/FD/NPU owners live. The
                # caller must never free imported RKNN storage in that state.
                if self._queue_idle:
                    self._release_events()
                timing['profiling_and_event_release_seconds'] = time.perf_counter()-last
                timing['complete_bridge_seconds'] = time.perf_counter()-started
                self.last_timings = dict(timing)
                self._stats['last_timing'] = dict(timing)
            if self._events or self._stats['cleanup_errors']:
                self._failed = True
                raise SoftmaxBridgeError('OpenCL event cleanup incomplete')
            self._stats['calls_completed'] += 1
            return timing

    def metrics(self):
        with self._lock:
            released = not self._objects and not self._fds and not self._events and self._queue_idle
            return {**self._stats, 'closed': self._closed, 'failed': self._failed,
                'queue_idle': self._queue_idle, 'live_OpenCL_objects': len(self._objects),
                'live_owned_DMA_fds': len(self._fds), 'live_events': len(self._events),
                'fully_released': released}

    def close(self):
        with self._lock:
            if self._closed and self.metrics()['fully_released']:
                return
            self._closed = True
            if self._queue:
                try:
                    self._finish('finish before imported-memory release')
                except BaseException as error:
                    self._queue_idle = False
                    self._failed = True
                    self._stats['cleanup_errors'].append(str(error))
                    raise SoftmaxBridgeError('Queue finish failed; all imports/FDs/NPU owners retained') from error
            self._release_events()
            if self._events:
                self._failed = True
                raise SoftmaxBridgeError('Events remain live; imported storage retained')
            errors, remaining = [], []
            for suffix, handle in reversed(self._objects):
                code = getattr(self._lib, 'clRelease' + suffix)(handle)
                if code:
                    errors.append('clRelease' + suffix + ' returned ' + str(code))
                    remaining.append((suffix, handle))
                else:
                    counts = self._stats['objects_released']
                    counts[suffix] = counts.get(suffix, 0) + 1
            self._objects = list(reversed(remaining))
            if ('CommandQueue', self._queue) not in self._objects:
                self._queue = None
            if not self._objects:
                import os
                self._memory.clear()
                remaining_fds = []
                for fd in reversed(self._fds):
                    try:
                        os.close(fd)
                    except OSError as error:
                        errors.append('close DMA fd: ' + str(error))
                        remaining_fds.append(fd)
                self._fds = list(reversed(remaining_fds))
            if not self._objects and not self._fds:
                self._owners = ()
            self._stats['cleanup_errors'].extend(errors)
            if errors:
                self._failed = True
                raise SoftmaxBridgeError('; '.join(errors))
