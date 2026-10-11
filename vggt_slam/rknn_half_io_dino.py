"""Explicit FP16 I/O for the unchanged original 24-block DINO backend.

Only the SDK route differs. Original token preparation, batch/frame loop, final
norm, cache guards and cleanup remain inherited. Install before the patch cache;
close that cache first. FP32 equivalence or trajectory acceptance is not implied.
"""
from __future__ import annotations

import ctypes as ct
import math
from pathlib import Path
import time

from vggt_slam import rknn_dino_blocks as original
from vggt_slam.rknn_aggregator_blocks import pin
from vggt_slam.rknn_half_io_pairs import (
    NEON_LIBRARY_SHA, NEON_SOURCE_SHA, require_half_descriptor,
)

ORIGINAL_DINO_SHA = '6cf230ea3ae6c65b9a3c7ec0b31e9abb346edb2fc0413b6b77a3a3b510b495de'
SHAPE = original.SHAPE


class RKNNHalfIOOriginalDino(original.RKNNOriginalDinoBlocks):
    def __init__(self, model, manifest_path, manifest_sha256, *, half_io_library,
                 half_io_library_sha256, **options):
        if half_io_library_sha256 != NEON_LIBRARY_SHA:
            raise ValueError('Only the frozen board-validated NEON library is supported')
        library = pin(half_io_library, NEON_LIBRARY_SHA)
        pin(Path(original.__file__), ORIGINAL_DINO_SHA)
        source = pin(Path(original.__file__).parent / 'native/fp16_neon.c', NEON_SOURCE_SHA)
        super().__init__(model, manifest_path, manifest_sha256, **options)
        try:
            if len(self.stats['model_memory']) != 24:
                raise ValueError('Exactly 24 queried original DINO contexts are required')
            for row in self.stats['model_memory']:
                require_half_descriptor(row['input'], SHAPE, 'input', 0)
                require_half_descriptor(row['output'], SHAPE, 'output', 0)
            # Base initialization enforces the board guard before any native load.
            self.neon = ct.CDLL(str(library))
            for name in ('f32_to_f16', 'f16_to_f32'):
                function = getattr(self.neon, name)
                function.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t]
                function.restype = None
            self.stats.update(precision='original_FP16_DINO_models_unchanged_explicit_FP16_IO',
                half_io_experiment=True, internal_precision_changed=False,
                half_io_library_sha256=NEON_LIBRARY_SHA, half_io_source_sha256=NEON_SOURCE_SHA,
                half_io_library=str(library), half_io_source=str(source),
                input_api_type=1, input_pass_through=0, output_want_float=0, init_flags=0,
                numerical_or_trajectory_quality_accepted=False,
                half_io=dict(calls_attempted=0, calls_completed=0, input_conversion_calls=0,
                    output_conversion_calls=0, input_conversion_seconds=0., output_conversion_seconds=0.,
                    owned_half_copy_seconds=0., sdk_api_seconds=0., route_seconds=0.,
                    rknn_inputs_set_seconds=0., rknn_run_seconds=0.,
                    rknn_outputs_get_seconds=0., rknn_outputs_release_seconds=0.,
                    timing_scope='single-frame DINO route including checks/allocations, conversion, SDK calls, owned copy and widening'))
        except BaseException:
            self.failed = True
            self.close()
            raise

    def _run(self, context, array, fmt, shape):
        base, api, np = self.base, self.api, self.np
        timing = self.stats['half_io']
        timing['calls_attempted'] += 1
        started = time.perf_counter()

        def call(name, function, *args):
            tick = time.perf_counter()
            try:
                return function(*args)
            finally:
                elapsed = time.perf_counter() - tick
                timing[name + '_seconds'] += elapsed
                timing['sdk_api_seconds'] += elapsed

        try:
            if (fmt != 3 or tuple(shape) != SHAPE or array.shape != SHAPE
                    or array.dtype != np.float32 or not array.flags.c_contiguous
                    or not np.isfinite(array).all()):
                raise ValueError('Half I/O requires the dense finite original FP32 single-frame DINO tensor')
            feed = np.empty(SHAPE, dtype=np.float16)
            tick = time.perf_counter()
            try:
                self.neon.f32_to_f16(array.ctypes.data, feed.ctypes.data, array.size)
                timing['input_conversion_calls'] += 1
            finally:
                timing['input_conversion_seconds'] += time.perf_counter() - tick
            if not np.isfinite(feed).all():
                raise ValueError('Explicit FP16 DINO input conversion overflowed or became nonfinite')
            inputs = (base.Input * 1)(base.Input(0, feed.ctypes.data, feed.nbytes, 0, 1, fmt))
            outputs = (base.Output * 1)()
            outputs[0].index, outputs[0].want_float = 0, 0
            run, received = base.RunExtend(0, 0, self.timeout, -1), base.OutputExtend()
            got, begin = False, time.perf_counter()
            try:
                base.check_call(call('rknn_inputs_set', api.rknn_inputs_set, context, 1, inputs), 'rknn_inputs_set')
                self.stats['sdk_run_attempts'] += 1
                code = call('rknn_run', api.rknn_run, context, ct.byref(run))
                self.stats['sdk_runs_succeeded'] += int(code == 0)
                base.check_call(code, 'rknn_run')
                base.check_call(call('rknn_outputs_get', api.rknn_outputs_get, context, 1,
                    outputs, ct.byref(received)), 'rknn_outputs_get')
                got = True
                if run.frame_id != received.frame_id:
                    raise RuntimeError('Synchronous DINO output frame ID mismatch')
                count = math.prod(SHAPE)
                if not outputs[0].buf or outputs[0].size != count * 2:
                    raise RuntimeError('DINO output pointer/FP16 logical byte-size mismatch')
                tick = time.perf_counter()
                try:
                    view = np.ctypeslib.as_array(ct.cast(outputs[0].buf,
                        ct.POINTER(ct.c_uint16)), shape=(count,)).view(np.float16)
                    owned_half = np.empty(SHAPE, dtype=np.float16)
                    np.copyto(owned_half.reshape(-1), view)
                finally:
                    timing['owned_half_copy_seconds'] += time.perf_counter() - tick
            finally:
                try:
                    if got or outputs[0].buf:
                        code = call('rknn_outputs_release', api.rknn_outputs_release, context, 1, outputs)
                        if code:
                            self.stats['cleanup_errors'].append(f'rknn_outputs_release returned {code}')
                        base.check_call(code, 'rknn_outputs_release')
                finally:
                    self.stats['sdk_io_seconds'] += time.perf_counter() - begin
            # The half copy owns its storage and survives release/SDK buffer reuse.
            value = np.empty(SHAPE, dtype=np.float32)
            tick = time.perf_counter()
            try:
                self.neon.f16_to_f32(owned_half.ctypes.data, value.ctypes.data, owned_half.size)
                timing['output_conversion_calls'] += 1
            finally:
                timing['output_conversion_seconds'] += time.perf_counter() - tick
            timing['calls_completed'] += 1
            return value
        finally:
            timing['route_seconds'] += time.perf_counter() - started


def install_half_io_dino(model, manifest_path, manifest_sha256, **options):
    return RKNNHalfIOOriginalDino(model, manifest_path, manifest_sha256, **options)
