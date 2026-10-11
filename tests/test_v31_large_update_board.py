"""Tiny synthetic protocol/geometry tests; no images, checkpoint or inference."""
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
HELPER = HERE / 'run-v31-large-update-board.py'
if not HELPER.exists():
    HELPER = HERE.parent / 'deploy' / 'rk3588' / 'validation' / 'dpvo_dynamic_update' / 'run-v31-large-update-board.py'
spec = importlib.util.spec_from_file_location('_v31_large_update_test', HELPER)
driver = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = driver
spec.loader.exec_module(driver)


def synthetic_state():
    centers = np.array([[[2, 4], [2, 4]], [[10, 11], [150, 114]]], np.int64)
    patches = np.ones((1, 4, 3, 3, 3), np.float32)
    xy = centers.reshape(-1, 2).astype(np.float32)
    d = np.arange(-1, 2, dtype=np.float32)
    patches[0, :, 0] = xy[:, 0, None, None] + d[None, None, :]
    patches[0, :, 1] = xy[:, 1, None, None] + d[None, :, None]
    state = dict(poses=np.zeros((1, 2, 7), np.float32), patches=patches,
                 intrinsics=np.ones((1, 2, 4), np.float32), net=np.ones((1, 2, 384), np.float32),
                 ii=np.array([0, 1], np.int64), jj=np.array([1, 0], np.int64),
                 kk=np.array([0, 3], np.int64))
    return state, centers


def outputs(edges=2):
    return dict(net=np.ones((1, edges, 384), np.float32),
                delta=np.full((1, edges, 2), .2, np.float32),
                weight=np.full((1, edges, 2), .5, np.float32))


def inputs():
    s, _ = synthetic_state()
    return dict(net=s['net'], context=np.ones((1, 2, 384), np.float32),
                corr=np.ones((1, 2, 882), np.float32), **{k: s[k] for k in driver.GRAPH_NAMES})


def test_centers_keep_view_patch_order_duplicates_and_boundary():
    state, expected = synthetic_state()
    actual = driver.validate_snapshot(state, views=2, count=2, edges=2)
    assert np.array_equal(actual, expected)
    assert actual.flags.owndata and actual.flags.c_contiguous
    actual[0, 0, 0] = 50
    assert state['patches'][0, 0, 0, 1, 1] == 2


@pytest.mark.parametrize('kind', ['fractional', 'grid', 'interior', 'nonfinite', 'dtype'])
def test_invalid_saved_patch_grid_rejected(kind):
    state, _ = synthetic_state()
    p = state['patches']
    if kind == 'fractional':
        p[0, 0, 0] += np.float32(.25)
    elif kind == 'grid':
        p[0, 0, 0, 0, 0] += 1
    elif kind == 'interior':
        p[0, 0, 0] -= 2
    elif kind == 'nonfinite':
        p[0, 0, 2, 0, 0] = np.inf
    else:
        p = p.astype(np.float64)
    with pytest.raises(ValueError):
        driver.centers_from_patches(p, views=2, count=2)


@pytest.mark.parametrize('kind', ['owner', 'target_range', 'index_dtype', 'net_nonfinite'])
def test_snapshot_graph_and_neural_contract_rejected(kind):
    state, _ = synthetic_state()
    if kind == 'owner':
        state['ii'][0] = 1
    elif kind == 'target_range':
        state['jj'][0] = 2
    elif kind == 'index_dtype':
        state['kk'] = state['kk'].astype(np.int32)
    else:
        state['net'][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        driver.validate_snapshot(state, views=2, count=2, edges=2)


def test_strict_gate_counts_failure_without_relaxing_threshold():
    expected, actual = outputs(), outputs()
    assert driver.compare_outputs(actual, expected)['gate_passed']
    actual['delta'][0, 0, 0] += np.float32(.001)
    result = driver.compare_outputs(actual, expected)
    assert not result['gate_passed']
    assert result['outputs']['delta']['failed_elements'] == 1
    assert result['outputs']['delta']['rtol'] == 1e-4
    assert result['outputs']['delta']['atol'] == 1e-5


def test_weight_domain_failure_can_be_within_element_tolerance():
    expected, actual = outputs(), outputs()
    expected['weight'].fill(0)
    actual['weight'].fill(-1e-7)
    result = driver.compare_outputs(actual, expected)
    assert result['outputs']['weight']['failed_elements'] == 0
    assert not result['actual']['weight_domain_passed']
    assert not result['gate_passed']


def test_nonfinite_diagnostic_is_json_safe_and_rejected():
    expected, actual = outputs(), outputs()
    actual['weight'][0, 0, 0] = np.nan
    actual['net'][0, 0, 0] = np.inf
    result = driver.compare_outputs(actual, expected)
    assert not result['gate_passed']
    assert result['actual']['arrays']['net']['nonfinite_elements'] == 1
    assert result['actual']['weight_min'] is None
    json.dumps(result, allow_nan=False)


class Tensor:
    def __init__(self, array):
        self.array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class TorchFake:
    from_numpy = staticmethod(Tensor)
    inference_mode = staticmethod(contextlib.nullcontext)


def test_original_timed_call_copies_saved_inputs_and_owned_outputs():
    saved = inputs()
    before = {k: v.copy() for k, v in saved.items()}
    retained = {}

    def update(net, context, corr, flow, ii, jj, kk):
        assert flow is None
        assert np.array_equal(ii.array, before['ii'])
        net.array.fill(3)
        value = outputs()
        value['net'] = net.array
        retained.update(value)
        return Tensor(value['net']), (Tensor(value['delta']), Tensor(value['weight']), None)

    result, timing = driver.timed_torch(TorchFake, update, saved)
    assert all(np.array_equal(saved[k], before[k]) for k in saved)
    assert all(not np.shares_memory(result[k], retained[k]) for k in driver.RESULT_NAMES)
    assert result['net'][0, 0, 0] == 3
    assert timing['total_seconds'] >= timing['original_Update_seconds'] >= 0


def test_alternating_scheduler_resets_input_and_saves_each_method_once():
    saved = inputs()
    seen, files = [], []

    def execute(method, current):
        assert current is saved
        assert current['net'][0, 0, 0] == 1  # Not previous output's hidden state.
        seen.append(method)
        value = outputs()
        value['net'].fill(9)
        return value, dict(total_seconds=.25 if method == 'Torch' else .1)

    def save(method, values):
        files.append(method)
        return dict(file=method + '.npz', arrays={k: driver.meta(v) for k, v in values.items()})

    result = driver.alternating_comparison(saved, execute, save)
    assert seen == ['Torch', 'ORT', 'ORT', 'Torch', 'Torch', 'ORT']
    assert files == ['Torch', 'ORT']
    assert result['numerical_gate_passed']
    assert len(result['calls']) == 6 and len(result['comparisons']) == 3
    assert result['timing']['Torch']['warm_seconds'] == [.25, .25]


def test_scheduler_strict_failure_kept_and_never_promoted():
    def execute(method, current):
        value = outputs()
        if method == 'ORT':
            value['net'][0, 0, 0] += .001
        return value, dict(total_seconds=.1)

    result = driver.alternating_comparison(inputs(), execute, lambda method, v: {'file': method})
    assert not result['numerical_gate_passed']
    assert all(g['comparison']['outputs']['net']['failed_elements'] == 1 for g in result['comparisons'])


def test_partial_failure_preserves_first_outputs_and_call_evidence():
    record, count = {}, {'ORT': 0}

    def execute(method, current):
        if method == 'ORT':
            count['ORT'] += 1
            if count['ORT'] == 2:
                raise RuntimeError('synthetic session failure')
        return outputs(), dict(total_seconds=.1)

    with pytest.raises(RuntimeError, match='session failure'):
        driver.alternating_comparison(inputs(), execute, lambda method, v: {'file': method}, record)
    assert set(record['representative_outputs']) == {'Torch', 'ORT'}
    assert record['calls'][-1]['execution_failed']
    assert record['calls'][-1]['repeat'] == 1
    assert len(record['comparisons']) == 1


def test_board_guard_precedes_output_and_model_side_effects(monkeypatch, tmp_path):
    def reject():
        raise ValueError('synthetic non-board')

    monkeypatch.setattr(driver, 'require_board', reject)
    output = tmp_path / 'must-not-exist'
    args = []
    for name in ('snapshot-dir', 'dataset-dir', 'v30-runner', 'encoder-module', 'source-dir',
                 'checkpoint', 'geometry', 'native-helper', 'library', 'build-receipt',
                 'encoder-model', 'encoder-export', 'encoder-ort', 'update-manifest'):
        args += ['--' + name, str(tmp_path / name)]
    args += ['--output-dir', str(output), '--update-manifest-sha256', '0' * 64]
    with pytest.raises(ValueError, match='non-board'):
        driver.main(args)
    assert not output.exists()
