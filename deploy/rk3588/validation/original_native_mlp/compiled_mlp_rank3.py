"""Fixed original FP16 compiled-MLP component comparator; RK3588S only."""
import ctypes as ct
import importlib.util
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

ABI_SHA = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
RUNTIME_SHA = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
MODEL_SHA = 'b5ab086ce279ac1543d5e9a3070c10c6e8ca7ab795db0dd8610d407411f38e8e'
NEON_SHA = 'd08147c438461669ad8ebd3c6483e0a7908053e95cc6f83d0fe7eb7a5cd668c9'
NEON_SOURCE_SHA = '77707b735a6291c417a997ed64acbe4e1cd60cc5f9dd150d83fe1836868bd5e3'
EXPERIMENTS = Path('/mnt/vggt-data/vggt-slam2/experiments')
SHAPE = (106, 1024)


def descriptor(base, attr):
    row = base.attr_record(attr)
    identity = row['quantization_type'] == 2 and row['zero_point'] == 0 and row['scale'] == 1.
    if (row['index'] != 0 or row['shape'] != [2, 53, 1024] or row['type'] != 1 or row['format'] not in (0, 1, 3)
            or row['pass_through'] != 0 or row['n_elems'] != math.prod(SHAPE)
            or row['size'] != 2*math.prod(SHAPE) or not math.isfinite(row['scale'])
            or (row['quantization_type'] != 0 and not identity)):
        raise ValueError('Expected dense fixed FP16 NONE/identity-affine MLP descriptor: '+str(row))
    return row


class CompiledMLP:
    def __init__(self, repo, core_mask=1):
        compatible = next((p.read_bytes() for p in (Path('/proc/device-tree/compatible'),
            Path('/sys/firmware/devicetree/base/compatible')) if p.is_file()), b'')
        if sys.platform != 'linux' or platform.machine().lower() not in ('aarch64', 'arm64') or b'rk3588' not in compatible:
            raise RuntimeError('Actual compiled MLP execution is board-only RK3588S')
        if core_mask not in (1, 7):
            raise ValueError('Only fixed core masks 1 and 7 are supported')
        repo = Path(repo).resolve(strict=True)
        abi = repo/'deploy/rk3588/validation/run-v12-rknn-block-board.py'
        import hashlib
        if hashlib.sha256(abi.read_bytes()).hexdigest() != ABI_SHA:
            raise ValueError('Frozen typed ABI source differs')
        spec = importlib.util.spec_from_file_location('v68_compiled_mlp_abi', abi)
        self.base = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.base)
        model = EXPERIMENTS/'vggt-v68-compiled-mlp-r2/original-frame0-mlp.rknn'
        library = Path('/usr/lib/librknnrt.so')
        neon = EXPERIMENTS/'vggt-v56-neon-build-r1/fp16_neon.so'
        for path, digest in ((model, MODEL_SHA), (library, RUNTIME_SHA), (neon, NEON_SHA),
                             (repo/'vggt_slam/native/fp16_neon.c', NEON_SOURCE_SHA)):
            self.base.require_sha(path, digest)
        self.api = self.base.bind_api(ct.CDLL(str(library)))
        self.neon = ct.CDLL(str(neon))
        for name in ('f32_to_f16', 'f16_to_f32'):
            f = getattr(self.neon, name)
            f.argtypes, f.restype = [ct.c_void_p, ct.c_void_p, ct.c_size_t], None
        self.context, self.closed, self.failed = self.base.CONTEXT(), False, False
        self.stats = dict(core_mask=core_mask, init_flags=0, input_api_type=1, input_pass_through=0,
            output_want_float=0, model_sha256=MODEL_SHA, runtime_sha256=RUNTIME_SHA, abi_sha256=ABI_SHA,
            neon_library_sha256=NEON_SHA, neon_source_sha256=NEON_SOURCE_SHA,
            contexts_created=0, contexts_destroyed=0, outputs_acquired=0, outputs_released=0,
            calls_attempted=0, calls_completed=0, sdk_run_attempts=0, sdk_runs_succeeded=0, cleanup_errors=[],
            precision='original_FP16_compiled_MLP_explicit_FP16_IO', full_model_or_SLAM_acceptance=False,
            seconds={key: 0. for key in ('input_conversion', 'rknn_inputs_set', 'rknn_run',
                'rknn_outputs_get', 'rknn_outputs_release', 'owned_half_copy', 'output_conversion', 'complete_route')})
        try:
            filename = ct.create_string_buffer(os.fsencode(model))
            code = self.api.rknn_init(ct.byref(self.context), ct.cast(filename, ct.c_void_p), 0, 0, None)
            self.stats['contexts_created'] = int(bool(self.context.value))
            self.base.check_call(code, 'rknn_init')
            self.base.check_call(self.api.rknn_set_core_mask(self.context, core_mask), 'rknn_set_core_mask')
            number = self.base.query(self.api, self.context, 0, self.base.InputOutputNum())
            if number.n_input != 1 or number.n_output != 1:
                raise ValueError('Exactly one input/output required')
            self.stats['input'] = descriptor(self.base, self.base.query(self.api, self.context, 1, self.base.TensorAttr(index=0)))
            self.stats['output'] = descriptor(self.base, self.base.query(self.api, self.context, 2, self.base.TensorAttr(index=0)))
            memory = self.base.query(self.api, self.context, 6, self.base.MemSize())
            self.stats['sdk_dma_bytes'] = int(memory.total_dma_allocated_size)
            version = self.base.query(self.api, self.context, 5, self.base.SDKVersion())
            self.stats['SDK_version'] = bytes(version.api_version).decode()
            self.stats['driver_version'] = bytes(version.drv_version).decode()
        except BaseException:
            self.failed = True
            self.close()
            raise

    def run(self, array):
        if self.closed or self.failed:
            raise RuntimeError('Compiled MLP is closed or failed')
        self.stats['calls_attempted'] += 1
        begin = time.perf_counter()
        times = {key: 0. for key in self.stats['seconds']}
        def call(name, function, *args):
            tick = time.perf_counter()
            try:
                return function(*args)
            finally:
                times[name] += time.perf_counter() - tick
        try:
            if array.dtype != np.float32 or array.shape != SHAPE or not array.flags.c_contiguous or not np.isfinite(array).all():
                raise ValueError('Finite contiguous FP32 [106,1024] required')
            half = np.empty(SHAPE, np.float16)
            call('input_conversion', self.neon.f32_to_f16, array.ctypes.data, half.ctypes.data, array.size)
            if not np.isfinite(half).all():
                raise ValueError('FP16 input overflow/nonfinite')
            base, api = self.base, self.api
            inputs = (base.Input*1)(base.Input(0, half.ctypes.data, half.nbytes, 0, 1, self.stats['input']['format']))
            outputs = (base.Output*1)()
            outputs[0].index, outputs[0].want_float = 0, 0
            run, received, got = base.RunExtend(0, 0, 30000, -1), base.OutputExtend(), False
            try:
                base.check_call(call('rknn_inputs_set', api.rknn_inputs_set, self.context, 1, inputs), 'rknn_inputs_set')
                self.stats['sdk_run_attempts'] += 1
                code = call('rknn_run', api.rknn_run, self.context, ct.byref(run))
                self.stats['sdk_runs_succeeded'] += int(code == 0)
                base.check_call(code, 'rknn_run')
                code = call('rknn_outputs_get', api.rknn_outputs_get, self.context, 1, outputs, ct.byref(received))
                self.stats['outputs_acquired'] += int(bool(outputs[0].buf))
                base.check_call(code, 'rknn_outputs_get')
                got = True
                if run.frame_id != received.frame_id or not outputs[0].buf or outputs[0].size != half.nbytes:
                    raise RuntimeError('Synchronous FP16 output identity/bytes differ')
                tick = time.perf_counter()
                view = np.ctypeslib.as_array(ct.cast(outputs[0].buf, ct.POINTER(ct.c_uint16)), shape=(array.size,)).view(np.float16)
                owned_half = np.empty(SHAPE, np.float16)
                np.copyto(owned_half.reshape(-1), view)
                times['owned_half_copy'] += time.perf_counter() - tick
            finally:
                if got or outputs[0].buf:
                    acquired = int(bool(outputs[0].buf))
                    code = call('rknn_outputs_release', api.rknn_outputs_release, self.context, 1, outputs)
                    self.stats['outputs_released'] += acquired * int(code == 0)
                    if code:
                        self.stats['cleanup_errors'].append('rknn_outputs_release returned '+str(code))
                    base.check_call(code, 'rknn_outputs_release')
            value = np.empty(SHAPE, np.float32)
            call('output_conversion', self.neon.f16_to_f32, owned_half.ctypes.data, value.ctypes.data, value.size)
            if not np.isfinite(value).all():
                raise ValueError('Finite owned FP32 output required')
            self.stats['calls_completed'] += 1
            return value
        except BaseException:
            self.failed = True
            raise
        finally:
            times['complete_route'] = time.perf_counter() - begin
            for key, seconds in times.items():
                self.stats['seconds'][key] += seconds
            self.stats['last_call_seconds'] = times

    def close(self):
        if self.context.value:
            code = self.api.rknn_destroy(self.context)
            if code:
                self.stats['cleanup_errors'].append('rknn_destroy returned '+str(code))
            self.base.check_call(code, 'rknn_destroy')
            self.context.value = 0
            self.stats['contexts_destroyed'] += 1
        self.closed = True

    def metrics(self):
        import copy
        return copy.deepcopy(dict(self.stats, closed=self.closed, failed=self.failed, live_contexts=int(bool(self.context.value))))
