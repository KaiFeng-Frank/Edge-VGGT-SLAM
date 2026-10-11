"""Tiny numerical/metadata/slot-remap tests, no real saved data or libraries."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v36_ab_synthetic', HERE / 'run-v36-frame-corr-ab-board.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def test_abba_order_each_arm_four_calls():
    schedule = [m for r in range(4) for m in driver.round_order(r)]
    assert schedule == ['old', 'new', 'new', 'old', 'old', 'new', 'new', 'old']
    assert schedule.count('old') == schedule.count('new') == 4


@pytest.mark.parametrize('permutation,remove', [([2, 0, 1], None), ([0, 2], 1)])
def test_slot_remap_and_deletion_preserve_identity_xy_and_edge_order(permutation, remove):
    m = 2
    kk = np.repeat(np.arange(3*m, dtype=np.int64), 3)
    jj = np.tile(np.arange(3, dtype=np.int64), 3*m)
    ii = kk//m
    coords = np.arange(18*2*2*3, dtype=np.float32).reshape(1, 18, 2, 2, 3)
    original = dict(ii=ii, jj=jj, kk=kk, coords=coords)
    result, selected = driver.remap_graph(original, permutation, m, remove)
    assert list(selected) == sorted(selected.tolist())
    assert np.array_equal(result['coords'], coords[:, selected])
    assert all(not a.flags.writeable for a in result.values())
    for index, old in enumerate(selected):
        assert permutation[result['ii'][index]] == ii[old]
        assert permutation[result['jj'][index]] == jj[old]
        assert result['kk'][index]//m == result['ii'][index]
        assert result['kk'][index]%m == kk[old]%m
    if remove is not None:
        assert len(selected) == 8


def valid_build():
    return dict(schema='v36_immutable_frame_correlation_native_build',
                status='compiled_and_synthetic_smoke_passed', script_sha256=driver.BUILD_SCRIPT_SHA,
                cleanup_errors=[], sources_and_inputs_unchanged=True,
                output={'sha256': 'a'*64}, candidate_source_sha256={'dpvo_frame_corr.py': driver.CANDIDATE_HELPER_SHA},
                smoke={'after_close': {'closed': True, 'current_context_bytes': 0}})


@pytest.mark.parametrize('field', ['status', 'helper', 'library', 'cleanup', 'close'])
def test_build_failure_or_lineage_mismatch_rejected(field):
    record = valid_build()
    driver.validate_candidate_build(record, 'a'*64)
    if field == 'status':
        record['status'] = 'failed'
    elif field == 'helper':
        record['candidate_source_sha256']['dpvo_frame_corr.py'] = 'b'*64
    elif field == 'library':
        record['output']['sha256'] = 'b'*64
    elif field == 'cleanup':
        record['cleanup_errors'] = ['synthetic cleanup error']
    else:
        record['smoke']['after_close']['current_context_bytes'] = 128
    with pytest.raises(ValueError):
        driver.validate_candidate_build(record, 'a'*64)


def test_array_metadata_tampering_and_safe_path(tmp_path):
    path = tmp_path/'small.npz'
    value = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    np.savez(path, corr=value)
    item = dict(file=path.name, bytes=path.stat().st_size, sha256=driver.sha(path))
    def pin(file, expected):
        assert driver.sha(file) == expected
        return file
    arrays = {'corr': driver.meta(value)}
    loaded = driver.load_artifact(tmp_path, item, arrays, pin)
    assert loaded['corr'].flags.owndata and not loaded['corr'].flags.writeable
    arrays['corr']['tensor_sha256'] = '0'*64
    with pytest.raises(ValueError, match='metadata'):
        driver.load_artifact(tmp_path, item, arrays, pin)
    with pytest.raises(ValueError, match='basename'):
        driver.load_artifact(tmp_path, dict(item, file='../small.npz'), arrays, pin)


def test_fixed_gate_is_not_relaxed_and_nonfinite_is_json_safe():
    expected = np.array([1], np.float32)
    # Float32(1.00011) rounds just OUTSIDE the unchanged float64 gate.
    # Use its immediately preceding representable FP32 value as inside.
    inside = np.nextafter(np.array([np.float32(1.00011)], np.float32), np.array([-np.inf], np.float32))
    outside = np.nextafter(inside, np.array([np.inf], np.float32))
    assert driver.comparison(inside, expected)['gate_passed']
    assert driver.comparison(outside, expected)['failed_elements'] == 1
    bad = driver.comparison(np.array([np.nan], np.float32), expected)
    assert not bad['finite'] and not bad['gate_passed']
    json.dumps(bad, allow_nan=False)


def test_host_guard_precedes_actual_array_and_library_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(driver.platform, 'machine', lambda: 'x86_64')
    out = tmp_path/'must-not-exist'
    argv = []
    for key in ('capture-dir', 'motion-dir', 'parent-dir', 'old-library', 'old-build', 'old-replay-driver',
                'candidate-dir', 'candidate-library', 'candidate-build'):
        argv.extend(['--'+key, str(tmp_path/('absent-'+key))])
    argv.extend(['--output-dir', str(out)])
    for key in ('old-build-sha256', 'candidate-library-sha256', 'candidate-build-sha256'):
        argv.extend(['--'+key, '0'*64])
    with pytest.raises(ValueError, match='board only'):
        driver.main(argv)
    assert not out.exists()


def test_frozen_parent_callback_and_core_are_actual_source_pins():
    parent = Path('/run/user/1000/edge-vggt-v35/public-snapshot/deploy/rk3588/validation/dpvo_fast_rknn/run-v35-dpvo-fast-rknn-board.py')
    assert driver.sha(parent) == driver.PARENT_REPLAY_SHA
    assert driver.sha(HERE/'dpvo_frame_corr.py') == driver.CANDIDATE_HELPER_SHA
    assert driver.sha(HERE/'build-v36-frame-corr-board.py') == driver.BUILD_SCRIPT_SHA
