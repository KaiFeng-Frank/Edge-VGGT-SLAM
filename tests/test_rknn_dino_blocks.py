"""Generated original-DINO lifecycle/order checks; no real Torch or SDK load."""
import contextlib
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from vggt_slam import rknn_dino_blocks as runtime
from test_rknn_aggregator_blocks import FakeTensor, FakeHook, FakeSDK, fake_torch, fake_abi


class FakeIdentity:
    pass


class FakeDinoBlock:
    training = False

    def __init__(self, parameter, instance=False):
        self.parameter = parameter
        self.norm1, self.norm2 = SimpleNamespace(eps=1e-6), SimpleNamespace(eps=1e-6)
        self.attn = SimpleNamespace(num_heads=16, rope=None, q_norm=FakeIdentity(), k_norm=FakeIdentity())
        self.mlp = SimpleNamespace(act=SimpleNamespace(approximate='none'))
        if instance:
            self.forward = lambda x: x

    def named_parameters(self):
        return [(name, self.parameter) for name in sorted(runtime.STATE_KEYS)]

    def forward(self, x):
        return x


class FakeDino:
    training, chunked_blocks = False, False
    n_blocks, embed_dim, num_heads, patch_size, num_register_tokens = 24, 1024, 16, 14, 4

    def __init__(self, parameter):
        self.blocks = [FakeDinoBlock(parameter, i % 2 == 0) for i in range(24)]
        self.norm, self.hooks = object(), []

    def prepare_tokens_with_masks(self, *args):
        return 'generated preparation placeholder'

    def forward_features(self, *args):
        return 'generated loop placeholder'

    def forward(self, *args):
        return 'generated forward placeholder'

    def register_forward_pre_hook(self, hook, *, with_kwargs):
        assert with_kwargs
        handle = FakeHook()
        self.hooks.append(handle)
        return handle


class DinoSDK(FakeSDK):
    """Each fake context adds its index to generated tokens, then poisons released memory."""
    def rknn_init(self, pointer, filename, *unused):
        context = len(self.initialized) + 1
        ct.cast(pointer, ct.POINTER(ct.c_uint64))[0] = context
        self.initialized.append(context)
        self.shapes[context] = runtime.SHAPE
        return -9 if context == self.fail_init_at else 0

    def rknn_inputs_set(self, context, count, inputs):
        assert count == 1 and inputs[0].pass_through == 0 and inputs[0].type == 0 and inputs[0].fmt == 3
        assert inputs[0].size == int(np.prod(runtime.SHAPE))*4
        self.input = np.ctypeslib.as_array(ct.cast(inputs[0].buf, ct.POINTER(ct.c_float)),
                                         shape=(int(np.prod(runtime.SHAPE)),)).copy()
        self.events.append(('input', context.value, float(self.input[0])))
        return 0

    def rknn_run(self, context, pointer):
        self.buffer = self.input + np.float32(context.value)
        ct.cast(pointer, ct.POINTER(fake_abi(self).RunExtend)).contents.frame_id = 42
        self.events.append(('run', context.value))
        return 0

    def rknn_outputs_get(self, context, count, outputs, pointer):
        assert count == 1 and outputs[0].want_float == 1
        outputs[0].buf, outputs[0].size = self.buffer.ctypes.data, self.buffer.nbytes
        ct.cast(pointer, ct.POINTER(fake_abi(self).OutputExtend)).contents.frame_id = 42
        return -17 if self.partial_get else 0

    def rknn_outputs_release(self, context, count, outputs):
        assert count == 1 and outputs[0].buf == self.buffer.ctypes.data
        self.releases += 1
        self.buffer.fill(-999.)
        outputs[0].buf = None
        return 0


@contextlib.contextmanager
def fixture(api):
    with tempfile.TemporaryDirectory(prefix='generated-original-dino-') as temporary:
        directory = Path(temporary)
        sdk_paths = {}
        for key, relative in (
            ('ABI_SHA', 'deploy/rk3588/validation/run-v12-rknn-block-board.py'),
            ('HEADER_SHA', 'deploy/rk3588/reference/rknn_api_v2.3.2.h'),
            ('LIBRARY_SHA', 'generated-library.placeholder')):
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'generated original-DINO service placeholder')
            sdk_paths[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        parameter = FakeTensor(np.array([.125], dtype=np.float32))
        parameter_sha = hashlib.sha256(parameter.numpy().tobytes()).hexdigest()
        dino = FakeDino(parameter)
        aggregator_owner = object()
        model = SimpleNamespace(inference_image_size=112,
            aggregator=SimpleNamespace(training=False, patch_embed=dino, _vggt_rknn_blocks_owner=aggregator_owner))
        model.modules = lambda: [model.aggregator, dino, *dino.blocks]
        rows = []
        def item(path):
            return dict(file=path.name, bytes=path.stat().st_size,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for index in range(24):
            onnx, rknn = directory/f'dino-{index:02d}.onnx', directory/f'dino-{index:02d}.rknn'
            onnx.write_bytes(b'generated graph placeholder')
            rknn.write_bytes(b'generated compiled placeholder')
            rows.append(dict(item(onnx), role='dino', index=index, input_shape=list(runtime.SHAPE),
                output_shape=list(runtime.SHAPE), input_name='input', output_name='output',
                heads=16, rknn=item(rknn), parameter_raw_sha256=dict.fromkeys(runtime.STATE_KEYS, parameter_sha),
                epsilons={'norm1': 1e-6, 'norm2': 1e-6}, gelu_approximate='none',
                all_14_caller_parameter_values_preserved_in_graph=True))
        manifest = directory/'manifest.json'
        manifest.write_text(json.dumps(dict(schema='v54_original_VGGT_24_DINO_block_RKNN_manifest',
            image_shape=[1,3,84,112], depth=24, source_hashes={}, blocks=rows,
            checkpoint={'sha256': runtime.CHECKPOINT_SHA, 'bytes': 5026874952})))
        torch = fake_torch()
        torch.nn = SimpleNamespace(Identity=FakeIdentity)
        base = fake_abi(api)
        fake_spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
        with contextlib.ExitStack() as stack:
            for key, value in sdk_paths.items():
                stack.enter_context(patch.object(runtime, key, value))
            stack.enter_context(patch.dict(sys.modules, {'torch': torch}))
            stack.enter_context(patch.object(runtime.importlib.util, 'spec_from_file_location', return_value=fake_spec))
            stack.enter_context(patch.object(runtime.importlib.util, 'module_from_spec', return_value=base))
            stack.enter_context(patch.object(runtime.ct, 'CDLL', return_value=object()))
            yield model, (manifest, item(manifest)['sha256']), dict(source_root=directory,
                library_path=directory/'generated-library.placeholder')


class OriginalDinoTests(unittest.TestCase):
    def test_all_24_blocks_cold_two_frames_then_warm_one_frame_preserve_order(self):
        api = DinoSDK()
        with fixture(api) as (model, args, options):
            dino = model.aggregator.patch_embed
            methods = (dino.forward, dino.forward_features, dino.prepare_tokens_with_masks, dino.norm)
            originals = [(b, 'forward' in b.__dict__, b.__dict__.get('forward'), b.forward) for b in dino.blocks]
            owner = model.aggregator._vggt_rknn_blocks_owner
            handle = runtime.install_rknn_dino_blocks(model, *args, **options)
            try:
                installed = [b.forward for b in dino.blocks]
                cold = FakeTensor(np.concatenate((np.full(runtime.SHAPE, 1., np.float32),
                                                  np.full(runtime.SHAPE, 10., np.float32)), axis=0))
                for block in dino.blocks:
                    cold = block.forward(cold)  # Pure service doubles, mirroring the unchanged loop.
                np.testing.assert_array_equal(cold.numpy()[0], np.full(runtime.SHAPE[1:], 301., np.float32))
                np.testing.assert_array_equal(cold.numpy()[1], np.full(runtime.SHAPE[1:], 310., np.float32))
                warm = FakeTensor(np.full(runtime.SHAPE, 100., np.float32))
                for block in dino.blocks:
                    warm = block.forward(warm)
                np.testing.assert_array_equal(warm.numpy(), np.full(runtime.SHAPE, 400., np.float32))
                self.assertEqual([event[1] for event in api.events if event[0]=='run'],
                                 [i for i in range(1,25) for _ in range(2)] + list(range(1,25)))
                self.assertEqual([b.forward for b in dino.blocks], installed)
                self.assertEqual((dino.forward, dino.forward_features, dino.prepare_tokens_with_masks, dino.norm), methods)
                self.assertIs(model.aggregator._vggt_rknn_blocks_owner, owner)
                s=handle.metrics()
                self.assertEqual((s['completed_blocks'],s['completed_frame_blocks'],s['sdk_runs_succeeded']), (48,72,72))
                self.assertEqual((s['batch1_blocks'],s['batch2_blocks'],api.releases), (24,24,72))
                self.assertEqual(s['block_calls'],{f'dino{i}':2 for i in range(24)})
                self.assertEqual(s['block_frame_calls'],{f'dino{i}':3 for i in range(24)})
            finally:
                handle.close()
            for block,had,value,forward in originals:
                self.assertEqual('forward' in block.__dict__,had)
                self.assertIs(block.__dict__.get('forward'),value)
                self.assertEqual(block.forward,forward)
            self.assertEqual(api.destroyed,list(range(1,25)))
            self.assertEqual(handle.metrics()['live_contexts'],0)
            self.assertEqual(dino.hooks[0].removed,1)
            handle.close()
            self.assertEqual(api.destroyed,list(range(1,25)))

    def test_partial_init_failure_releases_contexts_and_leaves_forwards_unchanged(self):
        api = DinoSDK(fail_init_at=3)
        with fixture(api) as (model,args,options):
            dino=model.aggregator.patch_embed
            originals=[b.forward for b in dino.blocks]
            with self.assertRaisesRegex(RuntimeError,'rknn_init returned -9'):
                runtime.install_rknn_dino_blocks(model,*args,**options)
            self.assertEqual([b.forward for b in dino.blocks],originals)
            self.assertEqual(api.destroyed,[1,2,3])
            self.assertFalse(dino.hooks)
            self.assertFalse(hasattr(dino,'_vggt_rknn_dino_owner'))

    def test_install_rejects_an_already_installed_patch_cache(self):
        api = DinoSDK()
        with fixture(api) as (model,args,options):
            model.aggregator._vggt_patch_cache_owner=object()
            with self.assertRaisesRegex(ValueError,'before its tail cache'):
                runtime.install_rknn_dino_blocks(model,*args,**options)
            self.assertFalse(api.initialized)
            self.assertFalse(model.aggregator.patch_embed.hooks)

    def test_partial_output_failure_releases_buffer_and_rejects_next_forward(self):
        api = DinoSDK(partial_get=True)
        with fixture(api) as (model,args,options):
            handle=runtime.install_rknn_dino_blocks(model,*args,**options)
            try:
                block=model.aggregator.patch_embed.blocks[0]
                x=FakeTensor(np.zeros(runtime.SHAPE,np.float32))
                with self.assertRaisesRegex(RuntimeError,'rknn_outputs_get returned -17'):
                    block.forward(x)
                self.assertEqual(api.releases,1)
                self.assertTrue(handle.failed)
                with self.assertRaisesRegex(RuntimeError,'no fallback'):
                    block.forward(x)
                self.assertEqual(handle.metrics()['sdk_run_attempts'],1)
            finally:
                handle.close()
            self.assertEqual(len(api.destroyed),24)

    def test_fixed_shape_and_mode_guards_reject_without_sdk_calls(self):
        api = DinoSDK()
        with fixture(api) as (model,args,options):
            handle=runtime.install_rknn_dino_blocks(model,*args,**options)
            try:
                dino=model.aggregator.patch_embed
                for batch in (1,2):
                    handle._guard(dino,(FakeTensor(np.zeros((batch,3,84,112),np.float32)),),{})
                with self.assertRaisesRegex(RuntimeError,'no fallback'):
                    handle._guard(dino,(FakeTensor(np.zeros((3,3,84,112),np.float32)),),{})
                with self.assertRaisesRegex(RuntimeError,'no fallback'):
                    dino.blocks[0].forward(FakeTensor(np.zeros((1,52,1024),np.float32)))
                self.assertEqual(handle.metrics()['sdk_run_attempts'],0)
            finally:
                handle.close()


if __name__=='__main__':
    unittest.main()
