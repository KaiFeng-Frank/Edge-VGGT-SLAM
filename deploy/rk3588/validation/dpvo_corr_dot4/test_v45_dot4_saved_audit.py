"""Tiny generated saved-audit guards; no native/model/real-feature execution."""
import hashlib
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

TEST = Path(__file__).resolve().parent
HELPER = Path(os.environ.get('V45_DOT4_AUDITOR', TEST / 'audit-v45-dot4-ab.py'))
if not HELPER.exists():
    HELPER = TEST.parent / 'deploy/rk3588/validation/dpvo_corr_dot4/audit-v45-dot4-ab.py'
spec = importlib.util.spec_from_file_location('_v45_dot4_saved_test', HELPER)
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


def test_signed_correlation_has_no_probability_domain_or_clipping():
    value = np.array([[[-7.25, 8.5]]], np.float32)
    result = audit.comparison(value, value)
    assert result['gate_passed'] and result['bitwise_equal']


def test_original_element_gate_boundary_is_frozen():
    low = np.float32(1e-5)
    high = np.nextafter(low, np.float32(np.inf))
    result = audit.comparison(np.array([low, high], np.float32), np.zeros(2, np.float32))
    assert result['rtol'] == 1e-4 and result['atol'] == 1e-5
    assert result['failed_elements'] == 1 and not result['gate_passed']


def test_numeric_equality_cannot_masquerade_as_literal_bitwise_equality():
    result = audit.comparison(np.array([-0.0], np.float32), np.array([0.0], np.float32))
    assert result['gate_passed'] and not result['bitwise_equal']


@pytest.mark.parametrize('invalid', (np.nan, np.inf))
def test_nonfinite_cannot_be_accepted_or_serialized_as_metric(invalid):
    with pytest.raises(ValueError, match='Nonfinite'):
        audit.comparison(np.array([invalid], np.float32), np.zeros(1, np.float32))


def test_saved_npz_tampering_rejected_by_file_SHA_before_comparison(tmp_path):
    path = tmp_path / 'generated.npz'; value = np.array([-2.0], np.float32)
    np.savez(path, corr=value)
    item = dict(file=path.name, bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        arrays=dict(corr=audit.metadata(value)))
    assert audit.load_artifact(tmp_path, item, {})['corr'].tobytes() == value.tobytes()
    np.savez(path, corr=np.array([-3.0], np.float32))
    with pytest.raises(ValueError, match='SHA/size'):
        audit.load_artifact(tmp_path, item, {})


@pytest.mark.parametrize('change', ('missing', 'warm_label'))
def test_incomplete_ABBA_or_incorrect_cold_exclusion_rejected(change):
    calls = [dict(method=m, round=r, cold=r == 0) for r in range(4)
        for m in (('A', 'B') if r % 2 == 0 else ('B', 'A'))]
    audit.schedule(calls)
    if change == 'missing': calls.pop()
    else: calls[0]['cold'] = False
    with pytest.raises(ValueError, match='ABBA'):
        audit.schedule(calls)
