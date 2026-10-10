"""Synthetic fixed DPT-PE tests; no released weights/images or real model.

The small FP32 reference follows the pinned Apache-2.0 official head_utils
formula at DA3 commit 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4.
"""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


HELPER_ROOT = Path(os.environ.get('V24_HELPER_ROOT', Path(__file__).resolve().parents[1]/'deploy/rk3588/validation'))
PATH = HELPER_ROOT/'export-v24-da3-board-r3.py'
if not PATH.is_file():
    PATH = Path(__file__).with_name('export-v24-da3-board-r3.py')
spec = importlib.util.spec_from_file_location('v24_export_r3_synthetic', PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def create_grid(width, height, aspect_ratio, dtype, device):
    diagonal = (aspect_ratio**2+1.)**.5
    span_x, span_y = aspect_ratio/diagonal, 1./diagonal
    xs = torch.linspace(-span_x*(width-1)/width, span_x*(width-1)/width, width, dtype=dtype, device=device)
    ys = torch.linspace(-span_y*(height-1)/height, span_y*(height-1)/height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(xs, ys, indexing='xy')
    return torch.stack((u, v), dim=-1)


def grid_to_embed(grid, channels):
    def axis_embed(pos):
        omega = torch.arange(channels//4, dtype=torch.float32, device=pos.device)
        omega /= channels/4.
        omega = 1./100**omega
        phase = torch.einsum('m,d->md', pos.reshape(-1), omega)
        return torch.cat((torch.sin(phase), torch.cos(phase)), dim=1).float()
    flattened = grid.reshape(-1, 2)
    return torch.cat((axis_embed(flattened[:, 0]), axis_embed(flattened[:, 1])), dim=-1).view(*grid.shape[:2], channels)


def original_add(x, W, H, ratio=.1):
    grid = create_grid(x.shape[-1], x.shape[-2], W/H, x.dtype, x.device)
    pe = grid_to_embed(grid, x.shape[1])*ratio
    return x+pe.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)


@pytest.mark.parametrize('height,width,channels', [(2, 3, 8), (3, 2, 12), (18, 24, 48)])
def test_non_square_original_fp32_position_addition_exact(height, width, channels):
    x = torch.arange(2*channels*height*width, dtype=torch.float32).reshape(2, channels, height, width)/91.
    fixed = probe.FixedHeadPositionEmbedding(create_grid, grid_to_embed)
    actual = fixed(x, 336, 252)
    assert torch.equal(actual, original_add(x, 336, 252))
    assert torch.equal(fixed(x.flip(-1), 336, 252), original_add(x.flip(-1), 336, 252))
    assert fixed.calls == 2 and len(fixed.cache) == 1
    metadata = fixed.records()[0]
    assert metadata['constant']['shape'] == [1, channels, height, width]
    assert metadata['original_expanded_shape'] == list(x.shape)
    assert metadata['constant']['dtype'] == 'float32'
    grid = create_grid(width, height, 336/252, torch.float32, torch.device('cpu'))
    assert torch.equal(grid[0, :, 1], grid[0, 0, 1].expand(width))
    assert torch.equal(grid[:, 0, 0], grid[0, 0, 0].expand(height))
    assert grid[-1, -1, 0] > grid[0, 0, 0] and grid[-1, -1, 1] > grid[0, 0, 1]


def test_cached_keys_work_in_trace_but_unknown_key_rejected(monkeypatch):
    fixed = probe.FixedHeadPositionEmbedding(create_grid, grid_to_embed)
    first = torch.zeros((2, 8, 2, 3), dtype=torch.float32)
    second = torch.zeros((2, 12, 3, 2), dtype=torch.float32)
    fixed(first, 336, 252)
    fixed(second, 336, 252)
    assert len(fixed.cache) == 2
    monkeypatch.setattr(torch.jit, 'is_tracing', lambda: True)
    assert torch.equal(fixed(first, 336, 252), original_add(first, 336, 252))
    with pytest.raises(ValueError, match='Unseen.*key'):
        fixed(torch.zeros((2, 8, 3, 4), dtype=torch.float32), 336, 252)
    assert len(fixed.cache) == 2


@pytest.mark.parametrize('change', ['dtype', 'geometry', 'ratio', 'channels'])
def test_fixed_geometry_and_dtype_are_not_silently_changed(change):
    fixed = probe.FixedHeadPositionEmbedding(create_grid, grid_to_embed)
    x, W, H, ratio = torch.zeros((2, 8, 2, 3)), 336, 252, .1
    if change == 'dtype':
        x = x.double()
    elif change == 'geometry':
        W = 252
    elif change == 'ratio':
        ratio = .2
    else:
        x = torch.zeros((2, 6, 2, 3))
    with pytest.raises(ValueError):
        fixed(x, W, H, ratio)
    assert not fixed.cache


class SyntheticHead(torch.nn.Module):
    def _add_pos_embed(self, x, W, H, ratio=.1):
        return original_add(x, W, H, ratio)


class SyntheticGetter:
    def __call__(self, batch, height, width, device):
        grid = torch.cartesian_prod(torch.arange(height, device=device), torch.arange(width, device=device))
        return grid[None].expand(batch, -1, -1).clone()


@pytest.mark.parametrize('instance_method', [False, True])
@pytest.mark.parametrize('raises', [False, True])
def test_original_head_getter_and_wrapper_restore_even_on_trace_error(monkeypatch, instance_method, raises):
    net = torch.nn.Module()
    net.head = SyntheticHead()
    if instance_method:
        net.head._add_pos_embed = net.head._add_pos_embed
    saved_method = net.head._add_pos_embed
    net.backbone = torch.nn.Module()
    net.backbone.pretrained = torch.nn.Module()
    getter = SyntheticGetter()
    net.backbone.pretrained.position_getter = getter
    monkeypatch.setitem(sys.modules, 'depth_anything_3.model.utils.head_utils',
                        SimpleNamespace(create_uv_grid=create_grid, position_grid_to_embed=grid_to_embed))
    factory = lambda model: model
    monkeypatch.setattr(probe.r2, 'ORIGINAL_MAKE_WRAPPER', factory)
    parent_factory = probe.parent.make_wrapper
    def fake_export(_args, report, _pins):
        assert net.head._add_pos_embed == saved_method  # References precede wrapper patching.
        assert net.backbone.pretrained.position_getter is getter
        wrapped = probe.parent.make_wrapper(net)
        x = torch.arange(96, dtype=torch.float32).reshape(2, 8, 2, 3)/11.
        assert torch.equal(wrapped.head._add_pos_embed(x, 336, 252), original_add(x, 336, 252))
        assert isinstance(net.backbone.pretrained.position_getter, probe.r2.FixedPositionGetter)
        if raises:
            raise RuntimeError('synthetic trace failure')
        report['status'] = 'exported_not_runtime_validated'
    monkeypatch.setattr(probe.r2, 'ORIGINAL_EXPORT', fake_export)
    report, pins = {'cleanup_errors': []}, {}
    if raises:
        with pytest.raises(RuntimeError, match='synthetic trace'):
            probe.export_model(SimpleNamespace(), report, pins)
    else:
        probe.export_model(SimpleNamespace(), report, pins)
    assert net.head._add_pos_embed == saved_method
    assert ('_add_pos_embed' in net.head.__dict__) == instance_method
    assert net.backbone.pretrained.position_getter is getter
    assert probe.parent.make_wrapper is parent_factory
    assert probe.r2.ORIGINAL_MAKE_WRAPPER is factory
    assert report['DPT_position_embedding_restored'] and report['position_getter_restored']
    assert len(report['fixed_DPT_position_embeddings']) == 1
    assert report['fixed_DPT_position_calls'] == 1
    assert report['export_entrypoint'] == 'export-v24-da3-board-r3.py'
    assert pins[str(probe.R2_PATH.resolve())] == probe.R2_SHA256
    assert pins[str(probe.r2.PARENT_PATH.resolve())] == probe.r2.PARENT_SHA256
    assert not report['cleanup_errors']
