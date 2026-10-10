"""Synthetic fixed-grid tests; no released weights/images or model forward."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


HELPER_ROOT = Path(os.environ.get('V24_HELPER_ROOT', Path(__file__).resolve().parents[1]/'deploy/rk3588/validation'))
PATH = HELPER_ROOT/'export-v24-da3-board-r2.py'
if not PATH.is_file():
    PATH = Path(__file__).with_name('export-v24-da3-board-r2.py')
spec = importlib.util.spec_from_file_location('v24_export_r2_synthetic', PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class SyntheticOriginalGetter:
    def __call__(self, batch_size, height, width, device):
        positions = torch.cartesian_prod(torch.arange(height, device=device), torch.arange(width, device=device))
        return positions.view(1, height*width, 2).expand(batch_size, -1, -1).clone()


@pytest.mark.parametrize('height,width', [(2, 3), (3, 2), (18, 24)])
def test_non_square_original_order_exact(height, width):
    original = SyntheticOriginalGetter()
    fixed = probe.FixedPositionGetter(original, 2, height, width)
    actual = fixed(2, height, width, torch.device('cpu'))
    assert torch.equal(actual, original(2, height, width, torch.device('cpu')))
    expected = np.array([(y, x) for y in range(height) for x in range(width)], dtype=np.int64)
    assert np.array_equal(actual[0].numpy(), expected)
    assert fixed.metadata['shape'] == [2, height*width, 2]


def test_fixed_grid_owns_values_and_rejects_changed_shape():
    fixed = probe.FixedPositionGetter(SyntheticOriginalGetter(), 2, 2, 3)
    output = fixed(2, 2, 3, 'cpu')
    output.fill_(99)
    assert not torch.equal(fixed.grid, output)
    with pytest.raises(ValueError, match='fixed shape'):
        fixed(2, 3, 2, 'cpu')
    with pytest.raises(ValueError, match='CPU only'):
        fixed(2, 2, 3, 'cuda')


@pytest.mark.parametrize('raises', [False, True])
def test_getter_and_wrapper_restore_even_after_export_failure(monkeypatch, raises):
    net = torch.nn.Module()
    net.backbone = torch.nn.Module()
    net.backbone.pretrained = torch.nn.Module()
    original = SyntheticOriginalGetter()
    net.backbone.pretrained.position_getter = original
    original_wrapper = probe.parent.make_wrapper
    def fake_export(_args, report, _pins):
        probe.parent.make_wrapper(net)
        assert isinstance(net.backbone.pretrained.position_getter, probe.FixedPositionGetter)
        if raises:
            raise RuntimeError('synthetic trace failure')
        report['status'] = 'exported_not_runtime_validated'
    monkeypatch.setattr(probe, 'ORIGINAL_EXPORT', fake_export)
    report, pins = {'cleanup_errors': []}, {}
    if raises:
        with pytest.raises(RuntimeError, match='synthetic trace'):
            probe.export_model(SimpleNamespace(), report, pins)
    else:
        probe.export_model(SimpleNamespace(), report, pins)
    assert net.backbone.pretrained.position_getter is original
    assert probe.parent.make_wrapper is original_wrapper
    assert report['position_getter_restored'] and not report['cleanup_errors']
    assert report['fixed_position_grid']['exactly_equal_original_getter']
    assert pins[str(probe.PARENT_PATH.resolve())] == probe.PARENT_SHA256


def test_incorrect_original_grid_rejected():
    def wrong(batch, height, width, device):
        return torch.ones((batch, height*width, 2), dtype=torch.int64)
    with pytest.raises(ValueError, match='grid/order'):
        probe.FixedPositionGetter(wrong, 2, 2, 3)


def test_parent_implementation_and_entrypoint_are_separate():
    report, pins = {}, {}
    probe.bind_parent(report, pins)
    assert report['export_entrypoint'] == 'export-v24-da3-board-r2.py'
    assert report['parent_implementation']['file'] == 'export-v24-da3-board.py'
    assert report['parent_implementation']['sha256'] == probe.PARENT_SHA256
