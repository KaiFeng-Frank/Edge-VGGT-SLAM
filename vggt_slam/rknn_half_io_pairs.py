"""Explicit FP16 I/O experiment; original FP16 RKNN models are unchanged.

All original parameter/source/position checks, guards, frame/global bridging
and cleanup are inherited from the pinned public pair backend. SDK half outputs
are copied before release and widened afterwards with the pinned NEON helper.
This adapter implies neither FP32 equivalence nor trajectory/real-time acceptance.
"""
from __future__ import annotations

import ctypes as ct
import math
from pathlib import Path
import time

from vggt_slam import rknn_aggregator_pairs as original
from vggt_slam.rknn_aggregator_blocks import pin, SHAPES

ORIGINAL_PAIRS_SHA = '08270da911213d0d22c21c9a7f001aaa7e16d7f7b32220f8a2b0c1ddc4b09b88'
NEON_LIBRARY_SHA = 'd08147c438461669ad8ebd3c6483e0a7908053e95cc6f83d0fe7eb7a5cd668c9'
NEON_SOURCE_SHA = '77707b735a6291c417a997ed64acbe4e1cd60cc5f9dd150d83fe1836868bd5e3'


def require_half_descriptor(row, shape, name, index):
    identity = row['quantization_type'] == 2 and row['zero_point'] == 0 and row['scale'] == 1.
    if (row['index'] != index or row['name'] != name or row['shape'] != list(shape)
            or row['type'] != 1 or row['format'] != 3 or row['pass_through'] != 0
            or row['n_elems'] != math.prod(shape) or row['size'] != 2 * math.prod(shape)
            or not math.isfinite(row['scale'])
            or (row['quantization_type'] != 0 and not identity)):
        raise ValueError(f'Half I/O requires dense fixed FP16 NONE/identity-affine descriptors: {row}')


class RKNNHalfIOOriginalPairs(original.RKNNOriginalAggregatorPairs):
    def __init__(self, model, manifest_path, manifest_sha256, *, half_io_library,
                 half_io_library_sha256, **options):
        if half_io_library_sha256 != NEON_LIBRARY_SHA:
            raise ValueError('Only the frozen finite-conversion-checked NEON library is supported')
        library = pin(half_io_library, half_io_library_sha256)
        source = pin(Path(__file__).parent / 'native/fp16_neon.c', NEON_SOURCE_SHA)
        pin(Path(original.__file__), ORIGINAL_PAIRS_SHA)
        super().__init__(model, manifest_path, manifest_sha256, **options)
        try:
            if len(self.stats['model_memory']) != 24:
                raise ValueError('Exactly 24 queried original pair contexts are required')
            for row in self.stats['model_memory']:
                require_half_descriptor(row['input'], SHAPES['frame'], 'input', 0)
                if len(row['outputs']) != 2:
                    raise ValueError('Both original pair output descriptors are required')
                for attr, binding in zip(row['outputs'], original.OUTPUT_BINDINGS):
                    require_half_descriptor(attr, binding['shape'], binding['name'], binding['index'])
            # The inherited board guard has run before loading any conversion code.
            self.neon = ct.CDLL(str(library))
            for name in ('f32_to_f16', 'f16_to_f32'):
                function = getattr(self.neon, name)
                function.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t]
                function.restype = None
            self.stats.update(precision='original_FP16_RKNN_models_unchanged_explicit_FP16_IO',
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
                    timing_scope='synchronous per-pair conversion, SDK calls and owned output copies'))
        except BaseException:
            self.failed = True
            self.close()
            raise

    def _run_pair(self, context, array, fmt):
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
            if (fmt != 3 or array.shape != SHAPES['frame'] or array.dtype != np.float32
                    or not array.flags.c_contiguous or not np.isfinite(array).all()):
                raise ValueError('Half I/O input must remain the dense finite original FP32 frame tensor')
            feed = np.empty(array.shape, dtype=np.float16)
            tick = time.perf_counter()
            try:
                self.neon.f32_to_f16(array.ctypes.data, feed.ctypes.data, array.size)
                timing['input_conversion_calls'] += 1
            finally:
                timing['input_conversion_seconds'] += time.perf_counter() - tick
            if not np.isfinite(feed).all():
                raise ValueError('Explicit FP16 I/O input conversion overflowed or became nonfinite')
            inputs = (base.Input * 1)(base.Input(0, feed.ctypes.data, feed.nbytes, 0, 1, fmt))
            outputs = (base.Output * 2)()
            for index in range(2):
                outputs[index].index, outputs[index].want_float = index, 0
            run, received = base.RunExtend(0, 0, self.timeout, -1), base.OutputExtend()
            got, owned_half, begin = False, [], time.perf_counter()
            try:
                base.check_call(call('rknn_inputs_set', api.rknn_inputs_set, context, 1, inputs), 'rknn_inputs_set')
                self.stats['sdk_run_attempts'] += 1
                code = call('rknn_run', api.rknn_run, context, ct.byref(run))
                self.stats['sdk_runs_succeeded'] += int(code == 0)
                base.check_call(code, 'rknn_run')
                base.check_call(call('rknn_outputs_get', api.rknn_outputs_get, context, 2,
                    outputs, ct.byref(received)), 'rknn_outputs_get')
                got = True
                if run.frame_id != received.frame_id:
                    raise RuntimeError('Synchronous pair output frame ID mismatch')
                tick = time.perf_counter()
                try:
                    for index, role in enumerate(('frame', 'global')):
                        shape, count = SHAPES[role], math.prod(SHAPES[role])
                        if not outputs[index].buf or outputs[index].size != count * 2:
                            raise RuntimeError('Pair output pointer/FP16 logical byte-size mismatch')
                        view = np.ctypeslib.as_array(ct.cast(outputs[index].buf,
                            ct.POINTER(ct.c_uint16)), shape=(count,)).view(np.float16)
                        owned = np.empty(shape, dtype=np.float16)
                        np.copyto(owned.reshape(-1), view)
                        owned_half.append(owned)
                finally:
                    timing['owned_half_copy_seconds'] += time.perf_counter() - tick
            finally:
                try:
                    if got or any(output.buf for output in outputs):
                        code = call('rknn_outputs_release', api.rknn_outputs_release, context, 2, outputs)
                        if code:
                            self.stats['cleanup_errors'].append(f'rknn_outputs_release returned {code}')
                        base.check_call(code, 'rknn_outputs_release')
                finally:
                    self.stats['sdk_io_seconds'] += time.perf_counter() - begin
            # Neither half copy aliases SDK memory; widening occurs after release.
            owned_float = []
            for half in owned_half:
                value = np.empty(half.shape, dtype=np.float32)
                tick = time.perf_counter()
                try:
                    self.neon.f16_to_f32(half.ctypes.data, value.ctypes.data, half.size)
                    timing['output_conversion_calls'] += 1
                finally:
                    timing['output_conversion_seconds'] += time.perf_counter() - tick
                owned_float.append(value)
            timing['calls_completed'] += 1
            return tuple(owned_float)
        finally:
            timing['route_seconds'] += time.perf_counter() - started


def install_half_io_pairs(model, manifest_path, manifest_sha256, **options):
    return RKNNHalfIOOriginalPairs(model, manifest_path, manifest_sha256, **options)
