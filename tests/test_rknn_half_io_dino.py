"""Generated SDK/converter doubles only: no Torch, model or native library load."""
import contextlib
import ctypes as ct
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

PUBLIC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PUBLIC), str(PUBLIC / 'tests')]
from vggt_slam import rknn_half_io_dino as runtime
from test_rknn_dino_blocks import DinoSDK, fixture as dino_fixture
from test_rknn_aggregator_blocks import FakeTensor, RunExtend, OutputExtend
from test_rknn_half_io_pairs import Converter


class HalfDinoSDK(DinoSDK):
    def __init__(self, fault=None):
        super().__init__()
        self.fault = fault

    def query(self, api, context, command, attr):
        row = super().query(api, context, command, attr)
        if self.fault == 'fp32_descriptor' and context.value == 3 and command == 2:
            row.update(type=0, size=row['n_elems'] * 4)
        return row

    def rknn_inputs_set(self, context, count, inputs):
        assert count == 1 and inputs[0].type == 1 and inputs[0].pass_through == 0 and inputs[0].fmt == 3
        assert inputs[0].size == 2 * int(np.prod(runtime.SHAPE))
        self.input = np.ctypeslib.as_array(ct.cast(inputs[0].buf, ct.POINTER(ct.c_uint16)),
            shape=(int(np.prod(runtime.SHAPE)),)).view(np.float16).copy()
        self.events.append(('input', context.value, float(self.input[0])))
        return 0

    def rknn_run(self, context, pointer):
        self.buffer = self.input + np.float16(context.value)
        ct.cast(pointer, ct.POINTER(RunExtend)).contents.frame_id = 42
        self.events.append(('run', context.value))
        return 0

    def rknn_outputs_get(self, context, count, outputs, pointer):
        assert count == 1 and outputs[0].index == 0 and outputs[0].want_float == 0 and outputs[0].is_prealloc == 0
        outputs[0].buf, outputs[0].size = self.buffer.ctypes.data, self.buffer.nbytes
        if self.fault == 'size':
            outputs[0].size -= 2
        ct.cast(pointer, ct.POINTER(OutputExtend)).contents.frame_id = 43 if self.fault == 'frame_id' else 42
        return -17 if self.fault == 'partial_get' else 0

    def rknn_outputs_release(self, context, count, outputs):
        self.events.append(('release',))
        super().rknn_outputs_release(context, count, outputs)  # Poisons the SDK-owned memory.
        return -18 if self.fault == 'release' else 0


@contextlib.contextmanager
def fixture(api):
    with dino_fixture(api) as (model, args, options):
        library = options['source_root'] / 'generated-neon.placeholder'
        library.write_bytes(b'fake converter, not executable native code')
        digest = hashlib.sha256(library.read_bytes()).hexdigest()
        converter = Converter(api)
        with patch.object(runtime, 'NEON_LIBRARY_SHA', digest), \
                patch.object(runtime.ct, 'CDLL', return_value=converter):
            yield model, args, dict(options, half_io_library=library, half_io_library_sha256=digest)


class HalfDinoTests(unittest.TestCase):
    def assert_restored(self, model, forwards, api):
        self.assertEqual([b.forward for b in model.aggregator.patch_embed.blocks], forwards)
        self.assertEqual(api.destroyed, list(range(1, 25)))
        self.assertFalse(hasattr(model.aggregator.patch_embed, '_vggt_rknn_dino_owner'))
        self.assertEqual(model.aggregator.patch_embed.hooks[0].removed, 1)

    def test_cold_two_frames_then_warm_one_frame_owned_after_release(self):
        api = HalfDinoSDK()
        with fixture(api) as (model, args, options):
            dino = model.aggregator.patch_embed
            forwards = [b.forward for b in dino.blocks]
            unchanged = (dino.forward, dino.forward_features, dino.prepare_tokens_with_masks, dino.norm)
            handle = runtime.install_half_io_dino(model, *args, **options)
            try:
                cold = FakeTensor(np.concatenate((np.full(runtime.SHAPE, 1., np.float32),
                                                   np.full(runtime.SHAPE, 10., np.float32))))
                for b in dino.blocks:
                    cold = b.forward(cold)
                cold_saved = cold.numpy().copy()
                warm = FakeTensor(np.full(runtime.SHAPE, 100., np.float32))
                for b in dino.blocks:
                    warm = b.forward(warm)
                np.testing.assert_array_equal(cold.numpy()[0], np.full(runtime.SHAPE[1:], 301., np.float32))
                np.testing.assert_array_equal(cold.numpy()[1], np.full(runtime.SHAPE[1:], 310., np.float32))
                np.testing.assert_array_equal(cold.numpy(), cold_saved)
                np.testing.assert_array_equal(warm.numpy(), np.full(runtime.SHAPE, 400., np.float32))
                self.assertTrue(warm.numpy().flags.owndata)
                self.assertEqual(warm.numpy().dtype, np.float32)
                self.assertFalse(np.shares_memory(warm.numpy(), api.buffer))
                self.assertTrue(np.all(api.buffer == -999.))
                self.assertEqual([e[0] for e in api.events], ['narrow', 'input', 'run', 'release', 'widen'] * 72)
                self.assertEqual([e[1] for e in api.events if e[0] == 'run'],
                                 [i for i in range(1, 25) for _ in range(2)] + list(range(1, 25)))
                self.assertEqual((dino.forward, dino.forward_features, dino.prepare_tokens_with_masks, dino.norm), unchanged)
                s = handle.metrics()
                self.assertEqual((s['completed_blocks'], s['completed_frame_blocks'], s['sdk_runs_succeeded']), (48, 72, 72))
                self.assertEqual((s['batch1_blocks'], s['batch2_blocks']), (24, 24))
                self.assertEqual(s['half_io']['calls_completed'], 72)
                self.assertEqual(s['half_io']['input_conversion_calls'], 72)
                self.assertEqual(s['half_io']['output_conversion_calls'], 72)
                self.assertGreaterEqual(s['half_io']['route_seconds'], s['half_io']['sdk_api_seconds'])
                self.assertFalse(s['internal_precision_changed'])
            finally:
                handle.close()
            self.assert_restored(model, forwards, api)
            handle.close()
            self.assertEqual(len(api.destroyed), 24)

    def test_partial_get_size_frame_id_and_release_failures_never_widen_or_fallback(self):
        for fault in ('partial_get', 'size', 'frame_id', 'release'):
            with self.subTest(fault=fault):
                api = HalfDinoSDK(fault)
                with fixture(api) as (model, args, options):
                    forwards = [b.forward for b in model.aggregator.patch_embed.blocks]
                    handle = runtime.install_half_io_dino(model, *args, **options)
                    try:
                        block = model.aggregator.patch_embed.blocks[0]
                        x = FakeTensor(np.zeros(runtime.SHAPE, np.float32))
                        with self.assertRaises(RuntimeError):
                            block.forward(x)
                        self.assertEqual(api.releases, 1)
                        self.assertTrue(handle.failed)
                        self.assertFalse(any(e[0] == 'widen' for e in api.events))
                        with self.assertRaisesRegex(RuntimeError, 'no fallback'):
                            block.forward(x)
                        self.assertEqual(handle.stats['sdk_run_attempts'], 1)
                    finally:
                        if fault == 'release':
                            with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
                                handle.close()
                        else:
                            handle.close()
                    self.assert_restored(model, forwards, api)

    def test_fp32_descriptor_rejects_and_restores_all_contexts(self):
        api = HalfDinoSDK('fp32_descriptor')
        with fixture(api) as (model, args, options):
            forwards = [b.forward for b in model.aggregator.patch_embed.blocks]
            with self.assertRaisesRegex(ValueError, 'Half I/O requires'):
                runtime.install_half_io_dino(model, *args, **options)
            self.assert_restored(model, forwards, api)
            self.assertEqual(api.events, [])

    def test_conversion_overflow_rejects_before_sdk_submission(self):
        api = HalfDinoSDK()
        with fixture(api) as (model, args, options):
            handle = runtime.install_half_io_dino(model, *args, **options)
            try:
                with self.assertRaisesRegex(ValueError, 'conversion overflowed'):
                    model.aggregator.patch_embed.blocks[0].forward(FakeTensor(np.full(runtime.SHAPE, 100000., np.float32)))
                self.assertTrue(handle.failed)
                self.assertEqual(handle.stats['sdk_run_attempts'], 0)
                self.assertEqual(api.events, [('narrow',)])
            finally:
                handle.close()

    def test_unpinned_library_rejects_before_context_initialization(self):
        api = HalfDinoSDK()
        with fixture(api) as (model, args, options):
            options['half_io_library_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'frozen board-validated'):
                runtime.install_half_io_dino(model, *args, **options)
            self.assertEqual(api.initialized, [])


if __name__ == '__main__':
    unittest.main()
