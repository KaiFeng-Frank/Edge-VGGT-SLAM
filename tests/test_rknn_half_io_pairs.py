"""Generated SDK/converter doubles only; no native library, Torch or model run."""
import contextlib
import ctypes as ct
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from vggt_slam import rknn_half_io_pairs as runtime
import test_rknn_aggregator_pairs as pairs
from test_rknn_aggregator_blocks import FakeTensor, RunExtend, OutputExtend, original_forwards


class Function:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args):
        return self.function(*args)


class Converter:
    def __init__(self, api):
        def convert(source, destination, count, src_type, dst_type, dtype, event):
            api.events.append((event,))
            source = np.ctypeslib.as_array(ct.cast(source, ct.POINTER(src_type)), shape=(count,))
            if src_type is ct.c_uint16:
                source = source.view(np.float16)
            destination = np.ctypeslib.as_array(ct.cast(destination, ct.POINTER(dst_type)), shape=(count,))
            with np.errstate(over='ignore', invalid='ignore'):
                converted = source.astype(dtype)
            destination[:] = converted.view(np.uint16) if dst_type is ct.c_uint16 else converted
        self.f32_to_f16 = Function(lambda s, d, n: convert(
            s, d, n, ct.c_float, ct.c_uint16, np.float16, 'narrow'))
        self.f16_to_f32 = Function(lambda s, d, n: convert(
            s, d, n, ct.c_uint16, ct.c_float, np.float32, 'widen'))


class HalfSDK(pairs.PairSDK):
    def __init__(self, *, fault=None):
        super().__init__(partial_get=fault == 'partial_get')
        self.fault = fault

    def query(self, api, context, command, attr):
        row = super().query(api, context, command, attr)
        if self.fault == 'fp32_descriptor' and command == 1 and context.value == 3:
            row.update(type=0, size=row['n_elems'] * 4)
        return row

    def rknn_inputs_set(self, context, count, inputs):
        assert count == 1 and inputs[0].type == 1 and inputs[0].pass_through == 0 and inputs[0].fmt == 3
        assert inputs[0].size == 2 * int(np.prod(runtime.SHAPES['frame']))
        self.input = np.ctypeslib.as_array(ct.cast(inputs[0].buf, ct.POINTER(ct.c_uint16)),
            shape=(int(np.prod(runtime.SHAPES['frame'])),)).view(np.float16).copy()
        self.events.append(('input', context.value))
        return 0

    def rknn_run(self, context, pointer):
        run = ct.cast(pointer, ct.POINTER(RunExtend)).contents
        assert run.non_block == 0 and run.timeout_ms == 30000 and run.fence_fd == -1
        self.buffers = [self.input + np.float16(context.value), self.input + np.float16(100 + context.value)]
        run.frame_id = 42
        self.events.append(('run', context.value))
        return 0

    def rknn_outputs_get(self, context, count, outputs, pointer):
        assert count == 2 and [o.index for o in outputs] == [0, 1]
        assert all(o.want_float == 0 and o.is_prealloc == 0 for o in outputs)
        for index in range(1 if self.partial_get else 2):
            outputs[index].buf, outputs[index].size = self.buffers[index].ctypes.data, self.buffers[index].nbytes
        if self.fault == 'size':
            outputs[1].size -= 2
        ct.cast(pointer, ct.POINTER(OutputExtend)).contents.frame_id = 43 if self.fault == 'frame_id' else 42
        return -17 if self.partial_get else 0

    def rknn_outputs_release(self, context, count, outputs):
        self.events.append(('release',))
        super().rknn_outputs_release(context, count, outputs)
        return -18 if self.fault == 'release' else 0


@contextlib.contextmanager
def fixture(api):
    with pairs.fixture(api) as (model, args, options):
        library = options['source_root'] / 'generated-neon.placeholder'
        library.write_bytes(b'generated converter double; not a native library')
        digest = pairs.original.runtime.sha(library)
        converter = Converter(api)
        with patch.object(runtime, 'NEON_LIBRARY_SHA', digest), \
                patch.object(runtime.ct, 'CDLL', return_value=converter):
            yield model, args, dict(options, half_io_library=library, half_io_library_sha256=digest), converter


def assert_restored(model, before, api):
    for block, had, value, forward in before:
        assert ('forward' in block.__dict__) is had
        assert block.__dict__.get('forward') is value and block.forward == forward
    assert api.destroyed == list(range(1, 25))
    assert model.hooks[0].removed == 1
    assert not hasattr(model.aggregator, '_vggt_rknn_blocks_owner')


def test_owned_outputs_survive_sdk_release_and_later_calls():
    api = HalfSDK()
    with fixture(api) as (model, args, options, converter):
        before = original_forwards(model)
        handle = runtime.install_half_io_pairs(model, *args, **options)
        try:
            for function in (converter.f32_to_f16, converter.f16_to_f32):
                assert function.argtypes == [ct.c_void_p, ct.c_void_p, ct.c_size_t] and function.restype is None
            x = FakeTensor(np.full(runtime.SHAPES['frame'], .125, np.float32))
            frame = model.aggregator.frame_blocks[0].forward(x, pairs.positions(handle, 0, 'frame'))
            bridge = FakeTensor(frame.numpy().reshape(runtime.SHAPES['global']))
            global_ = model.aggregator.global_blocks[0].forward(bridge, pairs.positions(handle, 0, 'global'))
            np.testing.assert_array_equal(frame.numpy(), np.full(runtime.SHAPES['frame'], 1.125, np.float32))
            np.testing.assert_array_equal(global_.numpy(), np.full(runtime.SHAPES['global'], 101.125, np.float32))
            snapshot = frame.numpy().copy(), global_.numpy().copy()
            next_frame = model.aggregator.frame_blocks[1].forward(
                FakeTensor(global_.numpy().reshape(runtime.SHAPES['frame'])), pairs.positions(handle, 1, 'frame'))
            assert not np.shares_memory(frame.numpy(), next_frame.numpy())
            for value, saved in zip((frame.numpy(), global_.numpy()), snapshot):
                assert value.dtype == np.float32 and value.flags.owndata
                np.testing.assert_array_equal(value, saved)
            assert all(np.all(value == -999.) for value in api.buffers)
            events = [event[0] for event in api.events]
            assert events == ['narrow', 'input', 'run', 'release', 'widen', 'widen'] * 2
            stats = handle.metrics()
            assert stats['half_io']['input_conversion_calls'] == 2
            assert stats['half_io']['output_conversion_calls'] == 4
            assert stats['half_io']['calls_completed'] == 2 and stats['sdk_runs_succeeded'] == 2
            assert stats['candidate_accepted'] is False and stats['internal_precision_changed'] is False
        finally:
            handle.close()
        assert_restored(model, before, api)
        handle.close()
        assert api.destroyed == list(range(1, 25))


@pytest.mark.parametrize('fault,message', [
    ('partial_get', 'rknn_outputs_get returned -17'), ('size', 'FP16 logical byte-size'),
    ('frame_id', 'frame ID mismatch'), ('release', 'rknn_outputs_release returned -18'),
])
def test_failed_sdk_route_releases_outputs_restores_and_never_falls_back(fault, message):
    api = HalfSDK(fault=fault)
    with fixture(api) as (model, args, options, _):
        before = original_forwards(model)
        handle = runtime.install_half_io_pairs(model, *args, **options)
        try:
            forward = model.aggregator.frame_blocks[0].forward
            x = FakeTensor(np.zeros(runtime.SHAPES['frame'], np.float32))
            with pytest.raises(RuntimeError, match=message):
                forward(x, pairs.positions(handle, 0, 'frame'))
            assert api.releases == 1 and handle.failed and handle.metrics()['pending_pairs'] == 0
            assert not any(event[0] == 'widen' for event in api.events)
            with pytest.raises(RuntimeError, match='no fallback'):
                forward(x, pairs.positions(handle, 0, 'frame'))
            assert handle.metrics()['sdk_run_attempts'] == 1
        finally:
            if fault == 'release':
                with pytest.raises(RuntimeError, match='cleanup failed: rknn_outputs_release returned -18'):
                    handle.close()
            else:
                handle.close()
        assert_restored(model, before, api)


def test_post_initialization_nonhalf_descriptor_restores_all_contexts_and_forwards():
    api = HalfSDK(fault='fp32_descriptor')
    with fixture(api) as (model, args, options, _):
        before = original_forwards(model)
        with pytest.raises(ValueError, match='Half I/O requires'):
            runtime.install_half_io_pairs(model, *args, **options)
        assert_restored(model, before, api)
        assert api.events == []


def test_conversion_overflow_rejects_before_sdk_submission():
    api = HalfSDK()
    with fixture(api) as (model, args, options, _):
        handle = runtime.install_half_io_pairs(model, *args, **options)
        try:
            x = FakeTensor(np.full(runtime.SHAPES['frame'], 100000., np.float32))
            with pytest.raises(ValueError, match='conversion overflowed'):
                model.aggregator.frame_blocks[0].forward(x, pairs.positions(handle, 0, 'frame'))
            assert handle.failed and handle.metrics()['sdk_run_attempts'] == 0
            assert api.events == [('narrow',)]
        finally:
            handle.close()


def test_unpinned_conversion_library_is_rejected_before_any_context_creation():
    api = HalfSDK()
    with fixture(api) as (model, args, options, _):
        options['half_io_library_sha256'] = '0' * 64
        with pytest.raises(ValueError, match='frozen finite-conversion-checked'):
            runtime.install_half_io_pairs(model, *args, **options)
        assert api.initialized == [] and api.destroyed == [] and model.hooks == []
