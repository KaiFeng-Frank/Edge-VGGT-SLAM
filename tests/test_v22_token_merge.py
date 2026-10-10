"""Pure synthetic Co-Me port checks; no real-image or released-weight forward."""
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


@pytest.fixture(scope='module')
def merge():
    path = Path(__file__).resolve().parents[1] / 'deploy/rk3588/validation/v22_token_merge.py'
    spec = importlib.util.spec_from_file_location('v22_synthetic_token_merge', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def oracle(tokens, mask, grid=(18, 24), group=(3, 3), special=5):
    """Independent slow spatial-coordinate construction, without plan indices."""
    h, w = grid
    gh, gw = group
    compact, reverse, weights, offsets = [], [], [], [0]
    for view, source in enumerate(tokens):
        restore = list(range(len(compact), len(compact) + special))
        compact.extend(source[:special])
        weights.extend([1.] * special)
        restore.extend([-1] * (h * w))
        for row in range(0, h, gh):
            for col in range(0, w, gw):
                positions = [special + r * w + c for r in range(row, row + gh) for c in range(col, col + gw)]
                if mask[view, row // gh, col // gw]:
                    for pos in positions: restore[pos] = len(compact)
                    compact.append(source[positions].mean(axis=0))
                    weights.append(float(gh * gw))
                else:
                    for pos in positions:
                        restore[pos] = len(compact)
                        compact.append(source[pos])
                        weights.append(1.)
        reverse.append(restore)
        offsets.append(len(compact))
    return np.stack(compact)[None], np.asarray(reverse), np.asarray(weights), offsets


@pytest.mark.parametrize('kind', ['none', 'all', 'mixed', 'different_views'])
def test_spatial_indices_special_tokens_weights_and_restore_match_independent_oracle(merge, kind):
    tokens = np.arange(2 * 437 * 3, dtype=np.float32).reshape(2, 437, 3)
    mask = np.zeros((2, 6, 8), np.bool_)
    if kind == 'all': mask[:] = True
    if kind == 'mixed': mask[:, ::2, 1::2] = True
    if kind == 'different_views': mask[1] = True
    plan = merge.build_plan(mask)
    expected, reverse, weights, offsets = oracle(tokens, mask)
    actual = merge.merge_tokens(tokens, plan)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(plan.reverse_indices, reverse)
    np.testing.assert_array_equal(plan.token_weights, weights)
    assert plan.view_offsets.tolist() == offsets
    np.testing.assert_array_equal(merge.restore_tokens(actual, plan), expected[0][reverse])
    for view in range(2):
        np.testing.assert_array_equal(actual[0, offsets[view]:offsets[view] + 5], tokens[view, :5])
    assert plan.compact_rows == 874 - 8 * int(mask.sum())
    assert plan.summary()['attention_merged'] is False
    assert plan.summary()['residual_merged'] is False


def test_global_and_per_view_shapes_share_same_fixed_scene_plan(merge):
    mask = np.zeros((2, 6, 8), np.bool_)
    mask[0, 0, 0] = True
    plan = merge.build_plan(mask)
    tokens = np.arange(874 * 2, dtype=np.float32).reshape(2, 437, 2)
    per_view = merge.merge_tokens(tokens, plan)
    global_form = merge.merge_tokens(tokens.reshape(1, 874, 2), plan)
    np.testing.assert_array_equal(global_form, per_view)
    np.testing.assert_array_equal(merge.restore_tokens(per_view, plan, layout='global'),
                                  merge.restore_tokens(per_view, plan).reshape(1, 874, 2))
    # A 3x3 spatial block is not nine consecutive row-major patch tokens.
    np.testing.assert_array_equal(plan.mean_input_indices[0], [5, 6, 7, 29, 30, 31, 53, 54, 55])


def test_no_merge_still_reorders_blocks_but_restore_is_exact_and_numpy_needs_no_torch(merge, monkeypatch):
    original_import = builtins.__import__
    def guard(name, *args, **kwargs):
        if name == 'torch' or 'cuda' in name:
            raise AssertionError('NumPy merge must not import a hardware backend')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guard)
    tokens = np.arange(874, dtype=np.float32).reshape(2, 437, 1)
    plan = merge.build_plan(np.zeros((2, 6, 8), np.bool_))
    compact = merge.merge_tokens(tokens, plan)
    assert not np.array_equal(compact, tokens.reshape(1, 874, 1))
    np.testing.assert_array_equal(merge.restore_tokens(compact, plan), tokens)


def test_plan_copies_mask_and_keeps_indices_immutable(merge):
    mask = np.zeros((2, 6, 8), np.bool_)
    plan = merge.build_plan(mask)
    mask[:] = True
    assert not plan.merge_mask.any()
    for value in (plan.merge_mask, plan.reverse_indices, plan.view_offsets, plan.token_weights):
        assert not value.flags.writeable
    with pytest.raises(ValueError): plan.reverse_indices[0, 0] = 9


@pytest.mark.parametrize('bad', ['dtype', 'flat', 'groups', 'empty', 'grid', 'special'])
def test_bad_plans_are_rejected(merge, bad):
    mask = np.zeros((2, 6, 8), np.bool_)
    kwargs = {}
    if bad == 'dtype': mask = mask.astype(np.float32)
    if bad == 'flat': mask = mask.reshape(2, 48)
    if bad == 'groups': mask = mask[:, :, :7]
    if bad == 'empty': mask = mask[:0]
    if bad == 'grid': kwargs['grid_hw'] = (17, 24)
    if bad == 'special': kwargs['special_tokens'] = -1
    with pytest.raises(ValueError): merge.build_plan(mask, **kwargs)


@pytest.mark.parametrize('bad', ['integer', 'wrong_rows', 'wrong_views', 'empty_channels'])
def test_bad_tokens_cannot_be_associated_with_a_plan(merge, bad):
    plan = merge.build_plan(np.zeros((2, 6, 8), np.bool_))
    tokens = np.ones((2, 437, 3), np.float32)
    if bad == 'integer': tokens = tokens.astype(np.int64)
    if bad == 'wrong_rows': tokens = tokens[:, :436]
    if bad == 'wrong_views': tokens = np.ones((3, 437, 3), np.float32)
    if bad == 'empty_channels': tokens = tokens[:, :, :0]
    with pytest.raises(ValueError): merge.merge_tokens(tokens, plan)


@pytest.mark.parametrize('dtype', ['float32', 'bfloat16'])
def test_cpu_torch_merge_uses_original_dtype_means_and_split(merge, dtype):
    torch = pytest.importorskip('torch')
    values = torch.arange(2 * 23 * 3).reshape(2, 23, 3).to(getattr(torch, dtype))
    mask = np.array([[[True, False]], [[False, True]]], np.bool_)
    plan = merge.build_plan(mask, grid_hw=(3, 6))
    compact = merge.merge_tokens(values, plan)
    assert compact.dtype == values.dtype and compact.device.type == 'cpu'
    source_rows = torch.tensor(plan.mean_input_indices)
    expected_means = values.reshape(-1, 3)[source_rows].mean(dim=1)
    torch.testing.assert_close(compact[0, torch.tensor(plan.mean_output_indices)], expected_means, rtol=0, atol=0)
    restored = merge.restore_tokens(compact, plan)
    for view in range(2): torch.testing.assert_close(restored[view, :5], values[view, :5], rtol=0, atol=0)
    assert merge.merge_tokens(values.reshape(1, 46, 3), plan).shape == compact.shape
    assert len(plan._torch_cache) == 4


def test_nonlinear_mlp_approximation_does_not_merge_residual_or_scale_by_nine(merge):
    torch = pytest.importorskip('torch')
    torch.manual_seed(42)
    inputs = torch.randn(2, 23, 4)
    residual = torch.randn_like(inputs)
    gamma = torch.tensor([.5, 1., 1.5, 2.])
    mlp = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.GELU(), torch.nn.Linear(8, 4)).eval()
    mask = np.array([[[True, False]], [[False, True]]], np.bool_)
    plan = merge.build_plan(mask, grid_hw=(3, 6))
    with torch.no_grad():
        compact_result = mlp(merge.merge_tokens(inputs, plan))
        expanded = merge.restore_tokens(compact_result, plan)
        baseline = mlp(inputs)
        approximation = residual + gamma * expanded
    assert not torch.allclose(expanded, baseline)
    for view in range(2):
        unaffected = torch.cat((torch.arange(5), torch.where(~torch.tensor(plan.merged_patch_mask[view].reshape(-1)))[0] + 5))
        torch.testing.assert_close(expanded[view, unaffected], baseline[view, unaffected], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(approximation - gamma * expanded, residual, rtol=1e-5, atol=1e-6)


def test_scene_wide_quantile_is_strict_and_ties_do_not_force_quota(merge):
    torch = pytest.importorskip('torch')
    confidence = torch.stack((torch.ones((3, 6)), torch.full((3, 6), 3.))).bfloat16()
    mask = merge.merge_mask_from_confidence(confidence, grid_hw=(3, 6))
    np.testing.assert_array_equal(mask, np.array([[[True, True]], [[False, False]]]))
    tied = torch.ones((2, 3, 6), dtype=torch.bfloat16)
    assert not merge.merge_mask_from_confidence(tied, grid_hw=(3, 6)).any()


def test_bfloat16_means_precede_float_quantile_without_hidden_precision_change(merge):
    torch = pytest.importorskip('torch')
    torch.manual_seed(8)
    confidence = (torch.rand((2, 18, 24)) * 4 + 1).bfloat16()
    blocks = confidence.reshape(2, 6, 3, 8, 3).permute(0, 1, 3, 2, 4).reshape(2, 48, 9)
    means = blocks.mean(dim=-1)
    threshold = torch.quantile(means.float().reshape(1, -1), .5, dim=-1, keepdim=True)
    expected = (means.reshape(1, -1) < threshold).reshape(2, 6, 8).numpy()
    np.testing.assert_array_equal(merge.merge_mask_from_confidence(confidence), expected)
    np.testing.assert_array_equal(merge.merge_mask_from_confidence(confidence.reshape(2, 432, 1)), expected)


@pytest.mark.parametrize('bad', ['nan', 'shape', 'threshold'])
def test_invalid_confidence_is_not_silently_used(merge, bad):
    torch = pytest.importorskip('torch')
    confidence = torch.ones((2, 18, 24))
    kwargs = {}
    if bad == 'nan': confidence[0, 0, 0] = float('nan')
    if bad == 'shape': confidence = confidence[:, :17]
    if bad == 'threshold': kwargs['threshold'] = 1.01
    with pytest.raises(ValueError): merge.merge_mask_from_confidence(confidence, **kwargs)


def test_predictor_released_key_shapes_defaults_and_no_cuda_imports(merge):
    torch = pytest.importorskip('torch')
    model = merge.build_confidence_predictor()
    state = model.state_dict()
    assert len(state) == 18 and sum(v.numel() for v in state.values()) == 412417
    assert state['proj1.weight'].shape == (128, 1024)
    assert state['block.attn.qkv.weight'].shape == (384, 128)
    assert state['block.mlp.fc1.weight'].shape == (256, 128)
    assert state['proj2.0.weight'].shape == (128, 128, 3, 3)
    assert state['proj2.2.weight'].shape == (1, 128, 3, 3)
    assert model.block.norm1.eps == model.block.norm2.eps == 1e-5
    assert model.proj2[0].padding_mode == model.proj2[2].padding_mode == 'reflect'
    assert all(v.device.type == 'cpu' for v in state.values())
    source = Path(merge.__file__).read_text()
    assert 'import torch.cuda' not in source and 'nvtx' not in source


def test_small_random_predictor_matches_independent_equations_and_exp_plus_one(merge):
    torch = pytest.importorskip('torch')
    F = torch.nn.functional
    torch.manual_seed(11)
    model = merge.build_confidence_predictor(hidden_dims=(4, 8, 1))
    state = model.state_dict()
    tokens = torch.randn(1, 2, 3, 3, 4)
    def linear(x, name): return F.linear(x, state[name + '.weight'], state[name + '.bias'])
    with torch.no_grad():
        x = linear(tokens.reshape(1, 18, 4), 'proj1')
        normalized = F.layer_norm(x, (8,), state['block.norm1.weight'], state['block.norm1.bias'], eps=1e-5)
        q, k, v = linear(normalized, 'block.attn.qkv').reshape(1, 18, 3, 1, 8).permute(2, 0, 3, 1, 4).unbind(0)
        attention = F.scaled_dot_product_attention(q, k, v, dropout_p=0.).transpose(1, 2).reshape(1, 18, 8)
        x = x + linear(attention, 'block.attn.proj')
        normalized = F.layer_norm(x, (8,), state['block.norm2.weight'], state['block.norm2.bias'], eps=1e-5)
        x = x + linear(F.gelu(linear(normalized, 'block.mlp.fc1'), approximate='none'), 'block.mlp.fc2')
        x = x.reshape(2, 3, 3, 8).permute(0, 3, 1, 2)
        x = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), state['proj2.0.weight'], state['proj2.0.bias'])
        x = F.gelu(x, approximate='none')
        x = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), state['proj2.2.weight'], state['proj2.2.bias'])
        expected = torch.exp(x.permute(0, 2, 3, 1).flatten(1, 2)) + 1.
        actual = model(tokens, BPHWC=(1, 2, 3, 3, 4))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (2, 9, 1)


def test_predictor_rejects_dtype_or_layout_instead_of_promoting(merge):
    torch = pytest.importorskip('torch')
    model = merge.build_confidence_predictor(hidden_dims=(4, 8, 1)).bfloat16()
    with pytest.raises(ValueError, match='dtype'):
        model(torch.zeros((1, 2, 3, 3, 4)), BPHWC=(1, 2, 3, 3, 4))
    with pytest.raises(ValueError, match='shape'):
        model(torch.zeros((2, 9, 4), dtype=torch.bfloat16), BPHWC=(1, 2, 3, 3, 4))


def test_checkpoint_loader_strict_metadata_hash_dtype_and_state_keys_without_forward(merge, tmp_path):
    torch = pytest.importorskip('torch')
    config = {'predictor_dims': [1024, 128, 1], 'predictor_active': 'expp1',
              'predictor_num_register': 0, 'loss_function': 'pairwise-rank',
              'inject_block_idx': 14, 'train_dtype': 'torch.bfloat16'}
    configuration = tmp_path / 'config.json'
    configuration.write_text(json.dumps(config))
    checkpoint = tmp_path / 'synthetic-checkpoint.pth'
    torch.save(merge.build_confidence_predictor().bfloat16().state_dict(), checkpoint)
    kwargs = dict(expected_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  expected_config_sha256=hashlib.sha256(configuration.read_bytes()).hexdigest())
    loaded = merge.load_released_predictor(checkpoint, configuration, **kwargs)
    assert not loaded.training and all(not p.requires_grad and p.dtype == torch.bfloat16 for p in loaded.parameters())
    with pytest.raises(ValueError, match='SHA256'):
        merge.load_released_predictor(checkpoint, configuration, **{**kwargs, 'expected_checkpoint_sha256': '0' * 64})
    config['inject_block_idx'] = 13
    configuration.write_text(json.dumps(config))
    kwargs['expected_config_sha256'] = hashlib.sha256(configuration.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='metadata'):
        merge.load_released_predictor(checkpoint, configuration, **kwargs)
