"""Synthetic-only checks for diagnostic capture identity and hook behavior."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / 'deploy/rk3588/validation/profile-v17-rectified-board.py'
spec = importlib.util.spec_from_file_location('stage_capture_v17', PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def reference():
    return dict(status='completed_fixed_input_diagnostic',
                script_sha256=m.PINS['capture-v17-resolution-pairs-board.py'],
                image_size=336, original_weights=True, precision='float32', npu_used=False, ORB_used=False,
                pairs=[dict(revision=1, frame_ids=[1., 2.])],
                selected_inputs=[dict(revision=1, frame_ids=[1., 2.])])


def test_valid_identity():
    pair, selected = m.validate_reference(reference(), 336)
    assert pair['frame_ids'] == selected['frame_ids'] == [1., 2.]


@pytest.mark.parametrize('key,value', [('status', 'failed'), ('script_sha256', 'bad'), ('image_size', 224),
                                      ('original_weights', False), ('precision', 'float16'),
                                      ('npu_used', True), ('ORB_used', True)])
def test_reject_wrong_reference(key, value):
    r = reference()
    r[key] = value
    with pytest.raises(ValueError):
        m.validate_reference(r, 336)


def test_reject_different_pair():
    r = reference()
    r['selected_inputs'][0]['frame_ids'][1] = 3.
    with pytest.raises(ValueError, match='identity'):
        m.validate_reference(r, 336)


def test_array_metadata():
    a = np.arange(8, dtype=np.float32).reshape(2, 4)
    row = m.array_record(a)
    assert row['shape'] == [2, 4] and row['bytes'] == 32
    assert m.array_record(a.copy()) == row
    a[0, 0] = 9
    assert m.array_record(a)['sha256'] != row['sha256']


@pytest.mark.parametrize('a', [np.zeros(4, np.float64), np.array([np.nan], np.float32),
                              np.empty(0, np.float32), np.zeros((4, 4), np.float32)[:, ::2]])
def test_reject_invalid_array(a):
    with pytest.raises(ValueError):
        m.array_record(a)


class SyntheticBlock(torch.nn.Module):
    def forward(self, x, pos=None, viz_attention=False):
        return x + 3


def test_hooks_capture_without_changing_values_and_remove():
    module = SyntheticBlock()
    x = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    pos = torch.arange(6, dtype=torch.int64).reshape(1, 3, 2)
    expected = module(x, pos=pos)
    cap = m.BlockCapture({'frame0': module})
    try:
        actual = module(x, pos=pos)
        assert torch.equal(actual, expected)
        assert cap.calls == {'frame0': 1}
        assert torch.equal(cap.values['frame0']['pos'], pos)
        assert torch.equal(cap.values['frame0']['input'], x)
        assert torch.equal(cap.values['frame0']['output'], actual)
        x.add_(8)
        assert not torch.equal(cap.values['frame0']['input'], x)
        with pytest.raises(ValueError, match='more than once'):
            module(x, pos=pos)
    finally:
        cap.close()
    assert not cap.handles and not module._forward_hooks and not module._forward_pre_hooks
    assert torch.equal(module(x), x + 3)


def test_dino_no_pos():
    module = SyntheticBlock()
    cap = m.BlockCapture({'dino0': module})
    try:
        module(torch.zeros((1, 3, 4), dtype=torch.float32))
        assert set(cap.values['dino0']) == {'input', 'output'}
    finally:
        cap.close()


def test_guard_rejects_host_before_any_inference_or_output(tmp_path):
    with pytest.raises(RuntimeError, match='requires Linux aarch64'):
        m.main(['--reference', str(tmp_path / 'absent.json'), '--reference-sha256', '0' * 64,
                '--output', str(tmp_path / 'output')])
    assert not (tmp_path / 'output').exists()
