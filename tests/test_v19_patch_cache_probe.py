"""Synthetic reference/provenance and numerical drift checks; no image/model."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / 'deploy/rk3588/validation/probe-v19-patch-cache-board.py'
spec = importlib.util.spec_from_file_location('v19_cache_probe', PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def reference():
    r = dict(status='completed_fixed_input_diagnostic',
             script_sha256=probe.HELPERS['capture-v17-resolution-pairs-board.py'],
             original_weights=True, precision='float32', npu_used=False,
             ORB_used=False, inference_on_board_only=True, image_size=336,
             pairs=[], selected_inputs=[])
    for i in range(1, 4):
        ids = [float(i), float(i + 1)]
        r['pairs'].append(dict(revision=i, frame_ids=ids, file=f'prediction_{i}.npz',
                               input_tensor=dict(shape=[2, 3, 252, 336])))
        r['selected_inputs'].append(dict(revision=i, frame_ids=ids,
            images=[dict(frame_id=f, path=f'image_{f}', sha256=str(f), samefile_validated=True) for f in ids]))
    return r


def test_exact_overlap_from_previous_final_to_next_first():
    pairs, selected = probe.validate_reference(reference())
    assert len(pairs) == len(selected) == 3


@pytest.mark.parametrize('field,value', [('status', 'failed'), ('precision', 'float16'),
    ('original_weights', False), ('image_size', 112), ('inference_on_board_only', False)])
def test_wrong_reference_identity_rejected(field, value):
    r = reference()
    r[field] = value
    with pytest.raises(ValueError):
        probe.validate_reference(r)


@pytest.mark.parametrize('change', ['pixels', 'no_overlap', 'source_id', 'order', 'shape', 'path_escape', 'missing_pair'])
def test_ambiguous_or_changed_pair_sources_rejected(change):
    r = reference()
    if change == 'pixels':
        r['selected_inputs'][1]['images'][0]['sha256'] = 'changed'
    elif change == 'no_overlap':
        r['selected_inputs'][1]['images'][0]['path'] = 'different'
    elif change == 'source_id':
        r['selected_inputs'][0]['images'][0]['frame_id'] = 0.
    elif change == 'order':
        r['pairs'][0]['frame_ids'] = [2., 1.]
    elif change == 'shape':
        r['pairs'][0]['input_tensor']['shape'] = [2, 3, 336, 252]
    elif change == 'path_escape':
        r['pairs'][0]['file'] = '../prediction.npz'
    else:
        r['pairs'].pop()
    with pytest.raises(ValueError):
        probe.validate_reference(r)


def test_drift_is_reported_without_claiming_bit_equality():
    expected = np.asarray([1., 2.], np.float32)
    close = expected.copy()
    close[0] += 1e-5
    row = probe.compare_arrays(close, expected, np)
    assert row['allclose'] and not row['array_equal'] and row['relative_l2'] > 0
    wrong = expected.copy()
    wrong[0] += .1
    row = probe.compare_arrays(wrong, expected, np)
    assert not row['allclose'] and row['outside_tolerance_elements'] == 1


@pytest.mark.parametrize('actual', [np.array([np.nan], np.float32), np.array([1.], np.float64), np.ones((1, 1), np.float32)])
def test_nonfinite_or_changed_output_rejected(actual):
    with pytest.raises(ValueError):
        probe.compare_arrays(actual, np.ones(1, np.float32), np)


def test_zero_reference_has_explicit_undefined_relative_error():
    row = probe.compare_arrays(np.zeros(2, np.float32), np.zeros(2, np.float32), np)
    assert row['array_equal'] and row['allclose'] and row['relative_l2'] is None
