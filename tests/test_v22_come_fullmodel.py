"""Tiny synthetic lifecycle checks; no images or released/model weights used."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
import types

import numpy as np
import pytest

torch = pytest.importorskip('torch')


@pytest.fixture(scope='module')
def research():
    path = Path(__file__).resolve().parents[1] / 'deploy/rk3588/validation/v22_come_fullmodel.py'
    spec = importlib.util.spec_from_file_location('v22_synthetic_fullmodel', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module', autouse=True)
def tiny_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class TinyMLP(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc1 = torch.nn.Linear(channels, 2 * channels)
        self.act = torch.nn.GELU(approximate='none')
        self.fc2 = torch.nn.Linear(2 * channels, channels)
        self.fail = False

    def forward(self, x):
        if self.fail:
            raise RuntimeError('synthetic original MLP failure')
        return self.fc2(self.act(self.fc1(x)))


class TinyBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm2 = torch.nn.LayerNorm(channels, eps=1e-6)
        self.mlp = TinyMLP(channels)
        self.ls2 = torch.nn.Parameter(torch.linspace(.01, .04, channels))
        self.attention_calls = 0
        self.last_residual = self.last_normalized = self.last_mlp = None

    def forward(self, x):
        # Synthetic per-token attention surrogate, intentionally outside MLP.
        self.attention_calls += 1
        residual = x + .02 * torch.sin(x)
        normalized = self.norm2(residual)
        mlp = self.mlp(normalized)
        self.last_residual, self.last_normalized, self.last_mlp = residual, normalized, mlp
        return residual + self.ls2 * mlp


class TinyDINO(torch.nn.Module):
    def __init__(self, channels, views, grid):
        super().__init__()
        self.blocks = torch.nn.ModuleList(TinyBlock(channels) for _ in range(24))
        self.channels, self.views, self.grid = channels, views, grid
        self.skip_plan_layer = False

    def forward(self, images):
        rows = 5 + self.grid[0] * self.grid[1]
        a = torch.arange(self.views * rows * self.channels, dtype=torch.float32)
        tokens = (torch.sin(a * .19) + .03 * a).reshape(self.views, rows, self.channels)
        tokens = tokens + images[0, :, 0, 0, 0, None, None]
        for index, block in enumerate(self.blocks):
            if index != 14 or not self.skip_plan_layer:
                tokens = block(tokens)
        return tokens


class TinyAggregator(torch.nn.Module):
    def __init__(self, channels=4, views=2, grid=(3, 6)):
        super().__init__()
        self.patch_start_idx = 5
        self.patch_embed = TinyDINO(channels, views, grid)
        self.frame_blocks = torch.nn.ModuleList(TinyBlock(channels) for _ in range(24))
        self.global_blocks = torch.nn.ModuleList(TinyBlock(channels) for _ in range(24))
        self.skip_last_global = False

    def forward(self, images):
        tokens = self.patch_embed(images)
        views, rows, channels = tokens.shape
        for index in range(24):
            tokens = self.frame_blocks[index](tokens.reshape(views, rows, channels))
            if index != 23 or not self.skip_last_global:
                tokens = self.global_blocks[index](tokens.reshape(1, views * rows, channels))
        return tokens


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aggregator = TinyAggregator()

    def forward(self, images):
        if images.ndim == 4:
            images = images.unsqueeze(0)
        return self.aggregator(images)


class TinyPredictor:
    def __init__(self):
        self.inputs = []
        self.fail = False

    def __call__(self, features, *, BPHWC):
        assert BPHWC == (1, 2, 3, 6, 4)
        assert features.dtype == torch.bfloat16 and features.device.type == 'cpu'
        self.inputs.append(features.detach().clone())
        if self.fail:
            raise RuntimeError('synthetic predictor failure')
        return torch.arange(36).reshape(2, 18, 1).bfloat16() + 1


def fixture(research, *, mask=None):
    torch.manual_seed(120)
    model = TinyModel().eval()
    predictor = TinyPredictor()
    callback = None if mask is None else lambda confidence, **kwargs: mask.copy()
    handle = research.ComeMLPResearch(model, predictor, expected_grid_hw=(3, 6),
                                     expected_views=2, channels=4, plan_callback=callback)
    images = torch.zeros(2, 3, 42, 84)
    return model, predictor, handle, images


def target_modules(model):
    a = model.aggregator
    return [b.mlp for b in a.patch_embed.blocks[15:]] + [b.mlp for b in a.frame_blocks] + [b.mlp for b in a.global_blocks]


def test_all_57_branches_one_predictor_original_weights_and_after_forward_artifacts(research):
    model, predictor, handle, images = fixture(research)
    params = {name: id(value) for name, value in model.named_parameters()}
    captured = []
    hook = model.aggregator.patch_embed.blocks[14].register_forward_hook(lambda m, a, out: captured.append(out.clone()))
    with handle, torch.no_grad():
        result = model(images)
        assert torch.isfinite(result).all()
        scene = handle.last_scene
        assert scene['mlp_calls'] == {'dino': 9, 'frame': 24, 'global': 24}
        assert len(scene['called_layers']) == len(set(map(tuple, scene['called_layers']))) == 57
        expected_features = captured[0][:, 5:].reshape(1, 2, 3, 6, 4)
        torch.testing.assert_close(scene['features'], expected_features, rtol=0, atol=0)
        torch.testing.assert_close(predictor.inputs[0], expected_features.bfloat16(), rtol=0, atol=0)
        record = handle.scene_record()
        assert record['feature_hashes_computed_after_forward']
        assert record['arrays']['predictor_features']['dtype'] == 'torch.bfloat16'
        assert len(record['arrays']['features']['sha256']) == 64
        json.dumps(record, allow_nan=False)
    hook.remove()
    metrics = handle.metrics()
    assert metrics['predictor_calls'] == metrics['plan_builds'] == metrics['scenes_completed'] == 1
    assert metrics['full_mlp_rows'] == 57 * 46
    assert metrics['compact_mlp_rows'] < metrics['full_mlp_rows']
    assert metrics['original_weight_hashes_unchanged'] and metrics['closed']
    assert metrics['cleanup_errors'] == []
    assert params == {name: id(value) for name, value in model.named_parameters()}
    assert all(block.attention_calls == 1 for block in model.aggregator.patch_embed.blocks)
    assert not metrics['attention_merged'] and not metrics['full_residual_merged']
    assert metrics['synthetic_predictor_callback_used']


def test_all_false_mask_is_identity_and_restores_all_class_forwards(research):
    model, _, handle, images = fixture(research, mask=np.zeros((2, 1, 2), np.bool_))
    originals = [module.forward.__func__ for module in target_modules(model)]
    with torch.no_grad():
        expected = model(images)
        with handle:
            actual = model(images)
        restored = model(images)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)
    for module, original in zip(target_modules(model), originals):
        assert 'forward' not in module.__dict__ and module.forward.__func__ is original
    assert not model.aggregator._forward_pre_hooks and not model.aggregator._forward_hooks
    assert not model.aggregator.patch_embed.blocks[14]._forward_hooks


def test_layer_scale_and_full_residual_not_averaged_or_scaled_by_token_weights(research):
    model, _, handle, images = fixture(research, mask=np.ones((2, 1, 2), np.bool_))
    final = model.aggregator.global_blocks[-1]
    original_forward = final.mlp.forward
    with handle, torch.no_grad():
        actual = model(images)
        plan = handle.last_scene['plan']
        compact = handle.merge.merge_tokens(final.last_normalized, plan)
        expanded = handle.merge.restore_tokens(original_forward(compact), plan, layout='global')
        expected = final.last_residual + final.ls2 * expanded
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert not torch.equal(actual, final.last_residual + final.ls2 * expanded * 9)
        for view in range(2):
            start = view * plan.tokens_per_view
            torch.testing.assert_close(final.last_mlp[:, start:start + 5],
                                      original_forward(final.last_normalized[:, start:start + 5]),
                                      rtol=1e-5, atol=1e-6)


def test_each_scene_rebuilds_predictor_plan_and_retains_only_newest_scene(research):
    model, predictor, handle, images = fixture(research)
    masks = [np.zeros((2, 1, 2), np.bool_), np.ones((2, 1, 2), np.bool_)]
    handle.plan_callback = lambda confidence, **kwargs: masks.pop(0)
    with handle, torch.no_grad():
        model(images)
        first = handle.last_scene
        model(images + .5)
        second = handle.last_scene
        assert first['scene_id'] == 1 and second['scene_id'] == 2
        assert first['plan'] is not second['plan']
        assert first['plan'].compact_rows == 46 and second['plan'].compact_rows == 14
        assert not torch.equal(predictor.inputs[0], predictor.inputs[1])
        assert handle._last_scene is not first and handle._active is None
        assert handle.metrics()['predictor_calls'] == handle.metrics()['plan_builds'] == 2
    assert handle.metrics()['mlp_calls'] == {'dino': 18, 'frame': 48, 'global': 48}


@pytest.mark.parametrize('failure', ['predictor', 'mlp'])
def test_original_failures_restore_all_hooks_forwards_and_weights(research, failure):
    model, predictor, handle, images = fixture(research)
    if failure == 'predictor': predictor.fail = True
    else: model.aggregator.frame_blocks[8].mlp.fail = True
    with pytest.raises(RuntimeError, match='synthetic .* failure'), torch.no_grad():
        with handle:
            model(images)
    assert handle.metrics()['closed'] and not handle.metrics()['active_scene']
    assert handle.metrics()['scenes_failed'] == 1
    assert handle.metrics()['original_weight_hashes_unchanged']
    assert all('forward' not in m.__dict__ for m in target_modules(model))
    assert not model.aggregator._forward_pre_hooks and not model.aggregator._forward_hooks
    handle.close()


def test_failed_scene_cannot_leave_old_plan_and_can_retry_new_scene(research):
    model, predictor, handle, images = fixture(research)
    with handle, torch.no_grad():
        model(images)
        predictor.fail = True
        with pytest.raises(RuntimeError, match='predictor failure'): model(images + 1)
        assert handle.last_scene['status'] == 'failed' and handle.last_scene['plan'] is None
        predictor.fail = False
        model(images + 2)
        assert handle.last_scene['scene_id'] == 3 and handle.last_scene['status'] == 'completed'
        assert handle.metrics()['scenes_completed'] == 2 and handle.metrics()['scenes_failed'] == 1


def test_hashes_never_computed_inside_forward_and_explicit_record_afterward(research, monkeypatch):
    model, _, handle, images = fixture(research)
    with handle, torch.no_grad():
        original = research._tensor_record
        def forbid_during_forward(tensor):
            assert handle._active is None, 'Hash/statistics entered the timed scene'
            return original(tensor)
        monkeypatch.setattr(research, '_tensor_record', forbid_during_forward)
        model(images)
        assert handle.scene_record()['arrays']['confidence']['dtype'] == 'torch.bfloat16'


def test_preexisting_instance_forward_is_restored_exactly(research):
    model, _, _, images = fixture(research)
    module = model.aggregator.frame_blocks[0].mlp
    original = module.forward
    def custom(self, tokens): return original(tokens) + .01
    installed = types.MethodType(custom, module)
    module.forward = installed
    handle = research.ComeMLPResearch(model, TinyPredictor(), expected_grid_hw=(3, 6), channels=4)
    with handle, torch.no_grad(): model(images)
    assert module.__dict__['forward'] is installed


def test_recursive_rejection_does_not_clear_outer_scene(research):
    model, predictor, handle, images = fixture(research)
    original_predict = predictor.__call__
    def nested_predict(features, **kwargs):
        with pytest.raises(RuntimeError, match='recursive'):
            model(images)
        assert handle._active is not None
        return original_predict(features, **kwargs)
    handle.predictor = nested_predict
    with handle, torch.no_grad(): model(images)
    assert handle.metrics()['scenes_completed'] == 1 and handle.metrics()['scenes_failed'] == 0
    assert handle._invocations == {}


def test_concurrent_scene_rejection_and_close_preserve_inflight_owner(research):
    model, predictor, handle, images = fixture(research)
    entered, resume = threading.Event(), threading.Event()
    errors = []
    def paused_predict(features, **kwargs):
        entered.set()
        assert resume.wait(3)
        return predictor(features, **kwargs)
    handle.predictor = paused_predict
    handle.__enter__()
    def owner():
        try:
            with torch.no_grad(): model(images)
        except BaseException as error: errors.append(error)
    thread = threading.Thread(target=owner)
    thread.start()
    try:
        assert entered.wait(3)
        with pytest.raises(RuntimeError, match='recursive'), torch.no_grad(): model(images)
        with pytest.raises(RuntimeError, match='active'): handle.close()
        with pytest.raises(RuntimeError, match='after its forward'): _ = handle.last_scene
        assert handle._active is not None
    finally:
        resume.set()
        thread.join(3)
        assert not thread.is_alive()
        handle.close()
    assert not errors and handle.metrics()['scenes_completed'] == 1


@pytest.mark.parametrize('bad', ['grad', 'dtype', 'shape', 'views'])
def test_wrong_scene_inputs_cannot_start_or_reuse_plan(research, bad):
    model, _, handle, images = fixture(research)
    if bad == 'dtype': images = images.double()
    if bad == 'shape': images = images[:, :, :-1]
    if bad == 'views': images = images[:1]
    with handle:
        if bad == 'grad':
            with pytest.raises(ValueError, match='fixed B1'): model(images)
        else:
            with pytest.raises(ValueError, match='fixed B1'), torch.no_grad(): model(images)
        assert handle.metrics()['scenes_started'] == 0 and handle._active is None
        assert handle._invocations == {}


@pytest.mark.parametrize('bad', ['missing_dino14', 'missing_global23'])
def test_missing_required_calls_fail_loudly_without_previous_mask(research, bad):
    model, _, handle, images = fixture(research)
    if bad == 'missing_dino14': model.aggregator.patch_embed.skip_plan_layer = True
    else: model.aggregator.skip_last_global = True
    with pytest.raises(RuntimeError, match='current scene|exactly 57'), torch.no_grad():
        with handle: model(images)
    assert handle.metrics()['scenes_failed'] == 1 and handle.metrics()['closed']


@pytest.mark.parametrize('bad', ['float32_confidence', 'wrong_shape', 'nonfinite'])
def test_bad_predictor_output_rejected_without_precision_fallback(research, bad):
    model, _, handle, images = fixture(research)
    conf = torch.ones(2, 18, 1, dtype=torch.bfloat16)
    if bad == 'float32_confidence': conf = conf.float()
    if bad == 'wrong_shape': conf = conf[:1]
    if bad == 'nonfinite': conf[0, 0, 0] = float('nan')
    handle.predictor = lambda features, **kwargs: conf
    with pytest.raises(ValueError, match='BF16|[Ff]inite'), torch.no_grad():
        with handle: model(images)
    assert handle.metrics()['closed'] and handle.metrics()['scenes_failed'] == 1


@pytest.mark.parametrize('bad', ['model_training', 'shared_mlp', 'fewer_blocks', 'predictor_training'])
def test_constructor_rejects_changed_model_structure_and_training(research, bad):
    model = TinyModel().eval()
    predictor = TinyPredictor()
    if bad == 'model_training': model.train()
    if bad == 'shared_mlp': model.aggregator.global_blocks[1].mlp = model.aggregator.global_blocks[0].mlp
    if bad == 'fewer_blocks': model.aggregator.frame_blocks = model.aggregator.frame_blocks[:-1]
    if bad == 'predictor_training': predictor = torch.nn.Identity().train()
    with pytest.raises(ValueError):
        research.ComeMLPResearch(model, predictor, expected_grid_hw=(3, 6), channels=4)


def test_changed_original_weight_fails_audit_but_restores_modules(research):
    model, _, handle, _ = fixture(research)
    handle.__enter__()
    with torch.no_grad(): next(model.parameters()).add_(1)
    with pytest.raises(RuntimeError, match='weights changed'): handle.close()
    assert handle.metrics()['closed'] and not handle.metrics()['original_weight_hashes_unchanged']
    assert all('forward' not in m.__dict__ for m in target_modules(model))


def test_frozen_merge_source_pin_and_research_license_are_explicit(research):
    path = Path(research.__file__).with_name('v22_token_merge.py')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == research.MERGE_HELPER_SHA256
    source = Path(research.__file__).read_text()
    assert 'CC BY-NC 4.0' in source and '5c7cfbc14da37f5dbb991e3a6f9e2c4a5efb8833' in source
    assert 'torch.cuda' not in source and 'nvtx' not in source.lower().replace('nvtx imports', '')
