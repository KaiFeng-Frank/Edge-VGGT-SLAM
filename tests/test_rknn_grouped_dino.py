"""Generated service fakes only: no Torch model, native library or real SDK."""
import ast
from pathlib import Path
import threading
from types import SimpleNamespace as NS
import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'vggt_slam/rknn_grouped_dino.py'
tree = ast.parse(SOURCE.read_bytes())
tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.module.startswith('vggt_slam'))]
class Half:
    pass
scope = dict(RKNNHalfIOOriginalDino=Half, BASE=Path('/unused/generated'))
exec(compile(tree, str(SOURCE), 'exec'), scope)
Backend = scope['RKNNGroupedOriginalDino']


class Tensor:
    def __init__(self, data):
        self.data = data
        self.shape, self.dtype = data.shape, data.dtype
        self.device, self.layout, self.requires_grad = NS(type='cpu'), 'strided', False
    def __len__(self): return len(self.data)
    def __getitem__(self, key): return Tensor(self.data[key])
    def detach(self): return self
    def contiguous(self): return Tensor(np.ascontiguousarray(self.data))
    def numpy(self): return self.data


class Module:
    def __init__(self):
        self.training = False
        for field in scope['HOOK_FIELDS']: setattr(self, field, {})
    def modules(self): return [self]
    def forward(self, *args): raise AssertionError('Original individual block modules must not be called')


class Norm(Module):
    def __init__(self): super().__init__(); self.calls = 0
    def __call__(self, value):
        self.calls += 1
        for hook in self._forward_hooks.values(): hook()
        return Tensor(value.data * np.float32(2))


def fixture(batch=1):
    handle = Backend.__new__(Backend)
    handle.lock, handle.closed, handle.failed = threading.RLock(), False, False
    handle.np = np
    global_hooks = NS(**{'_global' + field: {} for field in scope['HOOK_FIELDS']})
    handle.torch = NS(Tensor=Tensor, float32=np.float32, strided='strided', from_numpy=Tensor,
        is_grad_enabled=lambda: False, nn=NS(modules=NS(module=global_hooks)))
    dino = NS(training=False, blocks=[Module() for _ in range(24)], num_register_tokens=4)
    dino.norm, handle.prepares = Norm(), []
    def prepare(images, masks):
        handle.prepares.append(masks)
        values = images.data[:, :1, :1, :1].reshape(batch, 1, 1) + np.float32(1)
        return Tensor(np.broadcast_to(values, (batch, 53, 1024)).copy())
    dino.prepare_tokens_with_masks = prepare
    dino.named_modules = lambda: [('norm', dino.norm)] + [(f'blocks.{i}', block) for i, block in enumerate(dino.blocks)]
    dino.named_parameters, dino.named_buffers = lambda: [], lambda: []
    handle.dino, handle.aggregator = dino, NS(patch_embed=dino)
    handle.original_prepare, handle.original_norm = prepare, dino.norm
    handle.original_feature_value = lambda *args: 'original feature method'
    handle.original_feature_had = True
    handle.feature_forward = handle._features
    dino.forward_features, dino._vggt_rknn_dino_owner = handle.feature_forward, handle
    handle.module_state, handle.tensor_state = handle._module_state(), handle._tensor_state()
    handle.config_state = tuple((name, getattr(dino, name, None)) for name in scope['CONFIG_FIELDS'])
    handle.contexts = [NS(value=i+1) for i in range(6)]
    handle.groups = list(enumerate(handle.contexts)); handle.sdk_calls = []
    def run(context, array, fmt, shape):
        handle.sdk_calls.append(context.value)
        output = array.copy(); output += np.float32(4)
        return output
    handle._run = run
    handle.stats = dict(completed_feature_calls=0, completed_groups=0, completed_frame_groups=0,
        executed_original_layer_frames=0, group_calls={}, group_frame_calls={}, batch1_feature_calls=0,
        batch2_feature_calls=0, contexts_destroyed=0, cleanup_errors=[])
    handle.removed, handle.destroyed = [], []
    handle.guard = NS(remove=lambda: handle.removed.append('dino'))
    handle.aggregate_guard = NS(remove=lambda: handle.removed.append('aggregator'))
    handle.api = NS(rknn_destroy=lambda context: handle.destroyed.append(context.value) or 0)
    handle.base = NS(check_call=lambda code, name: None if code == 0 else (_ for _ in ()).throw(RuntimeError(name)))
    images = np.zeros((batch, 3, 84, 112), np.float32)
    if batch == 2: images[1] = 1
    return handle, Tensor(images), global_hooks


@pytest.mark.parametrize('batch', [1, 2])
def test_complete_group_loop_original_prefix_norm_and_output_contract(batch):
    h, images, _ = fixture(batch); observations = []
    h.dino._forward_hooks = {'outer_timing': lambda: None}  # Outer hooks are not bypassed.
    h.original_norm._forward_hooks['observed'] = lambda: observations.append('norm')
    result = h._features(images)
    assert h.sdk_calls == [i for i in range(1, 7) for _ in range(batch)]
    assert h.prepares == [None] and h.original_norm.calls == 1 and observations == ['norm']
    assert set(result) == {'x_norm_clstoken', 'x_norm_regtokens', 'x_norm_patchtokens', 'x_prenorm', 'masks'}
    assert result['x_norm_clstoken'].shape == (batch, 1024)
    assert result['x_norm_regtokens'].shape == (batch, 4, 1024)
    assert result['x_norm_patchtokens'].shape == (batch, 48, 1024)
    for frame in range(batch):
        assert np.all(result['x_prenorm'].data[frame] == 25 + frame)
        assert np.all(result['x_norm_patchtokens'].data[frame] == 2 * (25 + frame))
    assert h.stats['completed_frame_groups'] == 6 * batch and h.stats['executed_original_layer_frames'] == 24 * batch


def test_bypassed_hooks_masks_and_method_mutation_refused_before_runs():
    for change in ('block_hook', 'global_hook', 'masks', 'feature_method'):
        h, images, global_hooks = fixture()
        if change == 'block_hook': h.dino.blocks[7]._forward_pre_hooks[1] = lambda: None
        if change == 'global_hook': global_hooks._global_forward_hooks[1] = lambda: None
        if change == 'feature_method': h.dino.forward_features = lambda *args: None
        with pytest.raises(RuntimeError):
            if change == 'feature_method': h._aggregate_guard(h.aggregator, (), {})  # Cache-hit path.
            else: h._features(images, masks=True if change == 'masks' else None)
        assert h.failed and h.sdk_calls == []


def test_cleanup_restores_method_removes_guards_and_destroys_all_contexts():
    h, _, _ = fixture(); original = h.original_feature_value
    h.dino._vggt_patch_cache_owner = object()
    with pytest.raises(RuntimeError): h.close()
    assert not h.closed and h.destroyed == []
    del h.dino._vggt_patch_cache_owner
    h.close(); h.close()
    assert h.dino.forward_features is original and not hasattr(h.dino, '_vggt_rknn_dino_owner')
    assert h.removed == ['dino', 'aggregator'] and h.destroyed == list(range(1, 7))
    assert h.stats['contexts_destroyed'] == 6 and all(context.value == 0 for context in h.contexts)


def test_sdk_failure_marks_failed_without_fallback():
    h, images, _ = fixture()
    def failure(*args): raise RuntimeError('generated SDK failure')
    h._run = failure
    with pytest.raises(RuntimeError, match='generated SDK failure'): h._features(images)
    assert h.failed and h.prepares == [None] and h.original_norm.calls == 0
    h.close(); assert len(h.destroyed) == 6
