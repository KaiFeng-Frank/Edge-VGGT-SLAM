"""Small generated lifecycle checks; no real Torch, model, SDK or board calls.

Run: python -B -m unittest discover -s tests -p test_rknn_aggregator_blocks.py -v
The fake manifest/files/parameters exercise installation ordering and cleanup;
they are not numerical equivalence or board validation evidence.
"""
import contextlib
import ctypes as ct
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'vggt_slam/rknn_aggregator_blocks.py'
if not SOURCE.is_file():  # Same candidate source in the private experiment layout.
    SOURCE = ROOT / 'rknn_aggregator_blocks.py'
spec = importlib.util.spec_from_file_location('_generated_original_blocks', SOURCE)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array)
        self.device = SimpleNamespace(type='cpu')
        self.dtype, self.shape = self.array.dtype, self.array.shape
        self.requires_grad, self.layout = False, 'strided'

    def is_contiguous(self):
        return self.array.flags.c_contiguous

    def detach(self):
        return self

    def contiguous(self):
        return FakeTensor(np.ascontiguousarray(self.array))

    def numpy(self):
        return self.array


def fake_torch():
    module = ModuleType('torch')
    module.Tensor, module.float32, module.int64 = FakeTensor, np.dtype('float32'), np.dtype('int64')
    module.strided, module.is_grad_enabled, module.from_numpy = 'strided', lambda: False, FakeTensor
    return module


class FakeBlock:
    training = False

    def __init__(self, parameter, instance_forward=False):
        self.parameter = parameter
        self.norm1, self.norm2 = SimpleNamespace(eps=1e-6), SimpleNamespace(eps=1e-6)
        self.attn = SimpleNamespace(num_heads=16, q_norm=SimpleNamespace(eps=1e-6),
            k_norm=SimpleNamespace(eps=1e-6), rope=SimpleNamespace(base_frequency=100, scaling_factor=1.))
        self.mlp = SimpleNamespace(act=SimpleNamespace(approximate='none'))
        if instance_forward:
            self.forward = lambda *args, **kwargs: 'generated instance forward'

    def named_parameters(self):
        return [(name, self.parameter) for name in sorted(runtime.STATE_KEYS)]

    def forward(self, *args, **kwargs):
        return 'generated class forward'


class FakeHook:
    def __init__(self):
        self.removed = 0

    def remove(self):
        self.removed += 1


class FakeModel:
    training = False
    inference_image_size = 112

    def __init__(self, parameter):
        self.aggregator = SimpleNamespace(training=False, aa_order=['frame', 'global'],
            aa_block_size=1, aa_block_num=24, patch_start_idx=5, retained_layer_indices=None,
            frame_blocks=[FakeBlock(parameter, i % 2 == 0) for i in range(24)],
            global_blocks=[FakeBlock(parameter, i % 2 == 1) for i in range(24)])
        self.hooks = []

        def register(hook, *, with_kwargs):
            assert with_kwargs
            handle = FakeHook()
            self.hooks.append(handle)
            return handle
        self.aggregator.register_forward_pre_hook = register

    def modules(self):
        return [self, self.aggregator, *self.aggregator.frame_blocks, *self.aggregator.global_blocks]


# Only the structures consumed by the adapter are represented in this fake ABI.
class Input(ct.Structure):
    _fields_ = [('index', ct.c_uint32), ('buf', ct.c_void_p), ('size', ct.c_uint32),
                ('pass_through', ct.c_uint8), ('type', ct.c_int), ('fmt', ct.c_int)]


class Output(ct.Structure):
    _fields_ = [('want_float', ct.c_uint8), ('is_prealloc', ct.c_uint8),
                ('index', ct.c_uint32), ('buf', ct.c_void_p), ('size', ct.c_uint32)]


class RunExtend(ct.Structure):
    _fields_ = [('frame_id', ct.c_uint64), ('non_block', ct.c_int32),
                ('timeout_ms', ct.c_int32), ('fence_fd', ct.c_int32)]


class OutputExtend(ct.Structure):
    _fields_ = [('frame_id', ct.c_uint64)]


class FakeSDK:
    def __init__(self, *, fail_init_at=None, bad_attr_at=None, partial_get=False):
        self.fail_init_at, self.bad_attr_at, self.partial_get = fail_init_at, bad_attr_at, partial_get
        self.initialized, self.destroyed, self.shapes, self.events = [], [], {}, []
        self.releases = 0
        self.values = np.array([.25, -.5, 1., 2., -3., 4.], dtype=np.float32)
        self.buffer = (ct.c_float * 6)(*self.values)

    def rknn_init(self, pointer, filename, *unused):
        index = len(self.initialized) + 1
        ct.cast(pointer, ct.POINTER(ct.c_uint64))[0] = index
        self.initialized.append(index)
        role = Path(ct.string_at(filename).decode()).name.split('-')[0]
        self.shapes[index] = runtime.SHAPES[role]
        return -9 if index == self.fail_init_at else 0  # Nonzero context on failed init.

    def rknn_set_core_mask(self, *args):
        return 0

    def query(self, api, context, command, unused):
        if command == 5:
            return SimpleNamespace(api_version=b'2.3.2-generated')
        if command == 0:
            return SimpleNamespace(n_input=1, n_output=1)
        if command == 6:
            return SimpleNamespace(total_dma_allocated_size=64)
        shape = self.shapes[context.value]
        return dict(index=0, name='input' if command == 1 else 'output', shape=list(shape),
                    n_elems=int(np.prod(shape)), type=2 if context.value == self.bad_attr_at else 1,
                    quantization_type=0, zero_point=0, scale=1., format=3, pass_through=0,
                    size=int(np.prod(shape))*2)

    def rknn_destroy(self, context):
        self.destroyed.append(context.value)
        return 0

    def rknn_inputs_set(self, context, count, requests):
        assert count == 1 and requests[0].type == 0 and requests[0].pass_through == 0
        assert requests[0].size == 24 and requests[0].fmt == 3
        self.events.append('inputs_set')
        return 0

    def rknn_run(self, context, pointer):
        ct.cast(pointer, ct.POINTER(RunExtend)).contents.frame_id = 42
        self.events.append('run')
        return 0

    def rknn_outputs_get(self, context, count, outputs, pointer):
        assert count == 1 and outputs[0].want_float == 1 and outputs[0].is_prealloc == 0
        outputs[0].buf, outputs[0].size = ct.addressof(self.buffer), ct.sizeof(self.buffer)
        ct.cast(pointer, ct.POINTER(OutputExtend)).contents.frame_id = 42
        self.events.append('outputs_get')
        return -17 if self.partial_get else 0

    def rknn_outputs_release(self, context, count, outputs):
        assert count == 1 and outputs[0].buf == ct.addressof(self.buffer)
        self.releases += 1
        self.events.append('outputs_release')
        for i in range(6):
            self.buffer[i] = -999.  # Invalidating SDK memory exposes an unowned result.
        outputs[0].buf = None
        return 0


def fake_abi(api):
    def check_call(code, name):
        if code:
            raise RuntimeError(f'{name} returned {code}')
    return SimpleNamespace(CONTEXT=ct.c_uint64, Input=Input, Output=Output,
        RunExtend=RunExtend, OutputExtend=OutputExtend, SDKVersion=SimpleNamespace,
        InputOutputNum=SimpleNamespace, TensorAttr=SimpleNamespace, MemSize=SimpleNamespace,
        require_board=lambda: {'generated_board_double': True}, bind_api=lambda library: api,
        query=api.query, attr_record=lambda row: dict(row), check_call=check_call)


@contextlib.contextmanager
def installation_fixture(api):
    with tempfile.TemporaryDirectory(prefix='generated-original-blocks-') as temporary:
        directory = Path(temporary)
        sdk_files = {}
        for name, relative in (
            ('ABI_SHA', 'deploy/rk3588/validation/run-v12-rknn-block-board.py'),
            ('HEADER_SHA', 'deploy/rk3588/reference/rknn_api_v2.3.2.h'),
            ('LIBRARY_SHA', 'generated-library.placeholder')):
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'generated service placeholder')
            sdk_files[name] = runtime.sha(path)
        parameter = FakeTensor(np.array([.125], dtype=np.float32))
        parameter_sha = hashlib.sha256(parameter.numpy().tobytes()).hexdigest()
        model = FakeModel(parameter)
        rows = []
        for index in range(24):
            for role, shape in runtime.SHAPES.items():
                positions = np.zeros((*shape[:2], 2), dtype=np.int64)
                pos_path = directory / f'{role}-positions.npy'
                np.save(pos_path, positions, allow_pickle=False)
                graph = directory / f'{role}-{index:02d}.onnx'
                compiled = directory / f'{role}-{index:02d}.rknn'
                graph.write_bytes(b'generated graph placeholder')
                compiled.write_bytes(b'generated compiled placeholder')
                def item(path):
                    return dict(file=path.name, bytes=path.stat().st_size, sha256=runtime.sha(path))
                row = dict(item(graph), role=role, index=index, input_shape=list(shape),
                    output_shape=list(shape), input_name='input', output_name='output', heads=16,
                    rknn=item(compiled), positions=dict(item(pos_path), shape=list(positions.shape),
                        dtype='int64', tensor_bytes=positions.nbytes,
                        raw_sha256=hashlib.sha256(positions.tobytes()).hexdigest()),
                    parameter_raw_sha256=dict.fromkeys(runtime.STATE_KEYS, parameter_sha),
                    epsilons=dict.fromkeys(runtime.EPS, 1e-6), gelu_approximate='none',
                    rope_frequency=100, rope_scaling_factor=1.,
                    all_18_caller_parameter_values_preserved_in_graph=True)
                rows.append(row)
        manifest = directory / 'manifest.json'
        manifest.write_text(json.dumps(dict(schema='v53_original_VGGT_48_block_RKNN_manifest',
            image_shape=[2, 3, 84, 112], checkpoint={'sha256': runtime.CHECKPOINT_SHA, 'bytes': 5026874952},
            depth=24, aa_order=['frame', 'global'], aa_block_size=1, source_hashes={}, blocks=rows)))
        base = fake_abi(api)
        fake_spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
        with contextlib.ExitStack() as stack:
            for name, value in sdk_files.items():
                stack.enter_context(patch.object(runtime, name, value))
            stack.enter_context(patch.dict(sys.modules, {'torch': fake_torch()}))
            stack.enter_context(patch.object(runtime.importlib.util, 'spec_from_file_location', return_value=fake_spec))
            stack.enter_context(patch.object(runtime.importlib.util, 'module_from_spec', return_value=base))
            stack.enter_context(patch.object(runtime.ct, 'CDLL', return_value=object()))
            yield model, (manifest, runtime.sha(manifest)), dict(source_root=directory,
                library_path=directory/'generated-library.placeholder')


def original_forwards(model):
    return [(block, 'forward' in block.__dict__, block.__dict__.get('forward'), block.forward)
            for role in runtime.SHAPES for block in getattr(model.aggregator, role + '_blocks')]


def run_handle(api):
    handle = runtime.RKNNOriginalAggregatorBlocks.__new__(runtime.RKNNOriginalAggregatorBlocks)
    handle.base, handle.api, handle.np, handle.torch = fake_abi(api), api, np, fake_torch()
    handle.timeout, handle.lock, handle.closed, handle.failed = 30000, threading.RLock(), False, False
    handle.stats = dict(sdk_run_attempts=0, sdk_runs_succeeded=0, sdk_io_seconds=0.,
                        completed_blocks=0, block_calls={}, cleanup_errors=[])
    return handle


class OriginalBlockLifecycleTests(unittest.TestCase):
    def assert_restored(self, originals):
        for block, had, value, forward in originals:
            self.assertEqual('forward' in block.__dict__, had)
            self.assertIs(block.__dict__.get('forward'), value)
            self.assertEqual(block.forward, forward)

    def test_partial_initialization_failure_destroys_contexts_without_patching(self):
        api = FakeSDK(fail_init_at=3)
        with installation_fixture(api) as (model, args, options):
            originals = original_forwards(model)
            handle = runtime.RKNNOriginalAggregatorBlocks.__new__(runtime.RKNNOriginalAggregatorBlocks)
            with self.assertRaisesRegex(RuntimeError, 'rknn_init returned -9'):
                handle.__init__(model, *args, **options)
            self.assert_restored(originals)
            self.assertFalse(model.hooks)
            self.assertFalse(hasattr(model.aggregator, '_vggt_rknn_blocks_owner'))
            self.assertEqual(api.destroyed, [1, 2, 3])
            metrics = handle.metrics()
            self.assertTrue(metrics['closed'] and metrics['failed'])
            self.assertEqual(metrics['live_contexts'], 0)
            self.assertEqual(metrics['contexts_created'], 3)
            self.assertEqual(metrics['contexts_destroyed'], 3)
            handle.close()
            self.assertEqual(api.destroyed, [1, 2, 3])

    def test_query_failure_does_not_apply_prepared_patches(self):
        api = FakeSDK(bad_attr_at=4)
        with installation_fixture(api) as (model, args, options):
            originals = original_forwards(model)
            with self.assertRaisesRegex(ValueError, 'Expected fixed logical FP32/FP16'):
                runtime.install_rknn_aggregator_blocks(model, *args, **options)
            self.assert_restored(originals)
            self.assertFalse(model.hooks)
            self.assertEqual(api.destroyed, [1, 2, 3, 4])

    def test_close_restores_instance_and_class_forwards_and_is_idempotent(self):
        api = FakeSDK()
        with installation_fixture(api) as (model, args, options):
            originals = original_forwards(model)
            handle = runtime.install_rknn_aggregator_blocks(model, *args, **options)
            try:
                self.assertEqual(handle.metrics()['live_contexts'], 48)
                self.assertIs(model.aggregator._vggt_rknn_blocks_owner, handle)
                for block, _, _, forward in originals:
                    self.assertNotEqual(block.forward, forward)
                    self.assertIn('forward', block.__dict__)
            finally:
                handle.close()
            self.assert_restored(originals)
            self.assertFalse(hasattr(model.aggregator, '_vggt_rknn_blocks_owner'))
            self.assertEqual(model.hooks[0].removed, 1)
            self.assertEqual(api.destroyed, list(range(1, 49)))
            self.assertEqual(handle.metrics()['live_contexts'], 0)
            self.assertFalse(handle.metrics()['cleanup_errors'])
            handle.close()
            self.assertEqual(model.hooks[0].removed, 1)
            self.assertEqual(api.destroyed, list(range(1, 49)))

    def test_output_is_owned_before_sdk_buffer_release(self):
        api = FakeSDK()
        handle = run_handle(api)
        array = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
        result = handle._run(ct.c_uint64(1), array, 3, array.shape)
        np.testing.assert_array_equal(result, api.values.reshape(array.shape))
        np.testing.assert_array_equal(np.ctypeslib.as_array(api.buffer), np.full(6, -999., np.float32))
        self.assertFalse(np.shares_memory(result, np.ctypeslib.as_array(api.buffer)))
        self.assertEqual(api.releases, 1)
        self.assertEqual(api.events, ['inputs_set', 'run', 'outputs_get', 'outputs_release'])
        self.assertEqual(handle.stats['sdk_run_attempts'], 1)
        self.assertEqual(handle.stats['sdk_runs_succeeded'], 1)
        self.assertGreaterEqual(handle.stats['sdk_io_seconds'], 0.)

    def test_failed_outputs_get_releases_partial_buffer(self):
        api = FakeSDK(partial_get=True)
        handle = run_handle(api)
        array = np.zeros((1, 2, 3), dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, 'rknn_outputs_get returned -17'):
            handle._run(ct.c_uint64(1), array, 3, array.shape)
        self.assertEqual(api.releases, 1)
        self.assertEqual(api.events[-1], 'outputs_release')
        self.assertFalse(handle.stats['cleanup_errors'])
        self.assertGreaterEqual(handle.stats['sdk_io_seconds'], 0.)

    def test_failed_forward_blocks_subsequent_sdk_calls(self):
        api = FakeSDK(partial_get=True)
        handle = run_handle(api)
        array = np.zeros((1, 2, 3), dtype=np.float32)
        positions = np.zeros((1, 2, 2), dtype=np.int64)
        forward = handle._forward(SimpleNamespace(training=False), 'frame0', ct.c_uint64(1),
                                  array.shape, positions, 3)
        with self.assertRaisesRegex(RuntimeError, 'rknn_outputs_get returned -17'):
            forward(FakeTensor(array), FakeTensor(positions))
        self.assertTrue(handle.failed)
        self.assertEqual(api.releases, 1)
        with self.assertRaisesRegex(RuntimeError, 'no fallback'):
            forward(FakeTensor(array), FakeTensor(positions))
        self.assertEqual(api.events.count('run'), 1)
        self.assertEqual(handle.stats['completed_blocks'], 0)


if __name__ == '__main__':
    unittest.main()
