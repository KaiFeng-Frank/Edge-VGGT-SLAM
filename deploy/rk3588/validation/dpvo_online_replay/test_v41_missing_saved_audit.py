"""Tiny saved-auditor boundary tests; generated arrays only."""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
PATH = Path(os.environ.get('V41_SAVED_AUDITOR', HERE/'audit-v41-missing-export.py'))
if not PATH.is_file():
    PATH = HERE.parent/'deploy/rk3588/validation/dpvo_m16_online'/PATH.name
SPEC = importlib.util.spec_from_file_location('_test_v41_saved_auditor', PATH)
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)


def test_fixed_absolute_boundary_and_zero_norm_json():
    reference = np.zeros(1, np.float32)
    at = np.array([np.float32(1e-5)], np.float32)
    outside = np.nextafter(at, np.float32(np.inf))
    a, b = AUDITOR.compare(at, reference), AUDITOR.compare(outside, reference)
    assert a['gate_passed'] and a['failed_elements'] == 0
    assert not b['gate_passed'] and b['failed_elements'] == 1
    assert a['relative_L2'] is None and b['relative_L2'] is None
    json.dumps((a, b), allow_nan=False)


def test_fixed_relative_boundary():
    reference = np.ones(1, np.float32)
    center = np.array([np.float32(1 + 1e-5 + 1e-4)], np.float32)
    below = np.nextafter(center, np.float32(-np.inf))
    above = np.nextafter(center, np.float32(np.inf))
    assert AUDITOR.compare(below, reference)['gate_passed']
    assert not AUDITOR.compare(above, reference)['gate_passed']


@pytest.mark.parametrize('problem', ['nonfinite', 'dtype', 'shape'])
def test_invalid_saved_array_never_passes(problem):
    reference = np.zeros(2, np.float32)
    actual = reference.copy()
    if problem == 'nonfinite': actual[0] = np.nan
    elif problem == 'dtype': actual = actual.astype(np.float64)
    else: actual = actual[:1]
    with pytest.raises(ValueError): AUDITOR.compare(actual, reference)
