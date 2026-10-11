"""Tiny saved-array/metadata tampering checks; no real arrays or native imports."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
HELPER = HERE/'audit-v36-frame-corr-ab.py'
if not HELPER.exists():
    HELPER = HERE.parent/'deploy/rk3588/validation/dpvo_frame_cache/audit-v36-frame-corr-ab.py'
spec = importlib.util.spec_from_file_location('v36_corr_AB_audit_synthetic', HELPER)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def artifact(tmp_path):
    value = np.arange(12, dtype=np.float32).reshape(1, 2, 6)
    path = tmp_path/'tiny.npz'
    np.savez(path, corr=value)
    record = dict(file=path.name, bytes=path.stat().st_size, sha256=audit.sha(path),
                  arrays={'corr': audit.metadata(value)})
    return value, record


@pytest.mark.parametrize('tamper', ['file_hash', 'raw_hash', 'shape', 'unsafe_name'])
def test_saved_artifact_does_not_trust_metadata(tmp_path, tamper):
    value, record = artifact(tmp_path)
    loaded = audit.load_artifact(tmp_path, record, audit.Checks(), {}, namespace='synthetic')
    assert np.array_equal(loaded['corr'], value)
    bad = copy.deepcopy(record)
    if tamper == 'file_hash':
        bad['sha256'] = '0'*64
    elif tamper == 'raw_hash':
        bad['arrays']['corr']['tensor_sha256'] = '0'*64
    elif tamper == 'shape':
        bad['arrays']['corr']['shape'] = [1, 3, 4]
    else:
        bad['file'] = '../tiny.npz'
    with pytest.raises(ValueError):
        audit.load_artifact(tmp_path, bad, audit.Checks(), {}, namespace='synthetic')


def test_fixed_gate_uses_adjacent_fp32_values_not_relaxed_tolerance():
    expected = np.ones(1, np.float32)
    above = np.array([np.float32(1.00011)], np.float32)
    below = np.nextafter(above, np.array([-np.inf], np.float32))
    assert audit.comparison(below, expected)['gate_passed']
    failed = audit.comparison(above, expected)
    assert failed['failed_elements'] == 1 and not failed['gate_passed']
    forged = dict(failed, failed_elements=0, gate_passed=True)
    with pytest.raises(ValueError, match='failed_elements'):
        audit.check_comparison(forged, failed, audit.Checks(), 'synthetic')


def test_nonfinite_is_rejected_and_json_safe():
    result = audit.comparison(np.array([np.nan], np.float32), np.ones(1, np.float32))
    assert result['finite'] is False and result['gate_passed'] is False
    assert result['failed_elements'] is None and result['relative_L2'] is None
    json.dumps(result, allow_nan=False)


def test_norm_summary_roundoff_does_not_change_element_gate():
    ref = np.arange(1, 5, dtype=np.float32)
    measured = audit.comparison(ref + np.float32(1e-6), ref)
    recorded = dict(measured, relative_L2=measured['relative_L2']+1e-16)
    audit.check_comparison(recorded, measured, audit.Checks(), 'norm')
    recorded['failed_elements'] = 1
    with pytest.raises(ValueError):
        audit.check_comparison(recorded, measured, audit.Checks(), 'norm')


def test_slot_changes_keep_edge_identity_coordinates_and_original_order():
    kk = np.repeat(np.arange(3, dtype=np.int64)*audit.M, 3)
    ii = kk//audit.M
    jj = np.tile(np.arange(3, dtype=np.int64), 3)
    coords = np.arange(54, dtype=np.float32).reshape(1, 9, 2, 1, 3)
    graph = dict(ii=ii, jj=jj, kk=kk, coords=coords)
    ref = np.arange(18, dtype=np.float32).reshape(1, 9, 2)
    cases = audit.expected_cases(graph, graph, ref, ref)
    assert len(cases) == 4
    for rewritten, reference, slots in cases[1:]:
        selected = [edge for edge in range(9) if int(ii[edge]) in slots and int(jj[edge]) in slots]
        assert np.array_equal(rewritten['coords'], coords[:, selected])
        assert np.array_equal(reference, ref[:, selected])
        for edge, old in enumerate(selected):
            assert slots[rewritten['ii'][edge]] == ii[old]
            assert slots[rewritten['jj'][edge]] == jj[old]
            assert rewritten['kk'][edge]//audit.M == rewritten['ii'][edge]
    assert cases[-1][0]['ii'].size == 4


def test_repeated_call_hash_is_checked_and_later_array_scope_explicit():
    representative = audit.metadata(np.arange(4, dtype=np.float32))
    call = dict(array=copy.deepcopy(representative), repeat_bitwise_equal=True)
    audit.check_repeat(call, representative, audit.Checks())
    call['array']['tensor_sha256'] = '0'*64
    with pytest.raises(ValueError):
        audit.check_repeat(call, representative, audit.Checks())


def test_changed_source_declaration_is_not_accepted():
    pins = {'/board/frozen.py': 'a'*64}
    record = dict(source_pins=pins, source_hashes_after=dict(pins), sources_and_inputs_unchanged=True)
    audit.unchanged(record, audit.Checks(), 'tiny')
    record['source_hashes_after']['/board/frozen.py'] = 'b'*64
    with pytest.raises(ValueError):
        audit.unchanged(record, audit.Checks(), 'tiny')


def test_read_error_is_failed_receipt_without_acceptance(tmp_path):
    result = audit.audit(tmp_path, tmp_path, tmp_path, tmp_path/'absent-build.json', tmp_path, tmp_path)
    assert result['status'] == 'failed' and result['candidate_accepted'] is False
    assert result['checks']['failed'] == 1
    json.dumps(result, allow_nan=False)
