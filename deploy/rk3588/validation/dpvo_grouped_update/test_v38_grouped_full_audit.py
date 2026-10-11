"""Tiny generated arrays/text only; no models, hardware or captured features."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
OPTIONS = ([Path(os.environ['V38_AUDIT_HELPER'])] if 'V38_AUDIT_HELPER' in os.environ else [
    HERE/'audit-v38-grouped-full.py',
    HERE.parent/'deploy/rk3588/validation/dpvo_grouped_update/audit-v38-grouped-full.py'])
HELPER = next(path for path in OPTIONS if path.is_file())
SPEC = importlib.util.spec_from_file_location('_v38_full_saved_audit_test', HELPER)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_fixed_gate_boundary_is_not_relaxed():
    ref = np.zeros(2, np.float32)
    boundary = np.float32(1e-5)
    result = AUDIT.numerical(np.array([boundary, np.nextafter(boundary, np.float32(np.inf))]), ref)
    assert result['failed_elements'] == 1 and not result['gate_passed']
    assert result['atol'] == 1e-5 and result['rtol'] == 1e-4


def test_nonfinite_result_is_rejected_and_json_safe():
    row = AUDIT.numerical(np.array([np.nan, np.inf],np.float32), np.zeros(2,np.float32))
    assert not row['finite'] and not row['gate_passed'] and row['nonfinite_elements'] == 2
    assert row['relative_L2'] is None and row['failed_elements'] is None
    json.dumps(row, allow_nan=False)


def test_output_shape_mismatch_is_not_broadcast():
    with pytest.raises(ValueError, match='shapes'):
        AUDIT.numerical(np.zeros((2,1),np.float32), np.zeros((1,2),np.float32))


def graph():
    rows = [(s,t,s*3+p) for s,targets in ((0,[0]),(1,[0,1]),(2,[1,2])) for t in targets for p in range(3)]
    table = np.array(rows,dtype=np.int64)[np.random.default_rng(14).permutation(len(rows))]
    return tuple(table[:,i] for i in range(3))


def test_ragged_shuffled_original_edges_recover_group_CF_bytes():
    edges = graph(); groups = AUDIT.independent_groups(*edges,patches=3)
    assert [g['key'] for g in groups] == ['B1_D1_M3','B2_D2_M3']
    assert groups[1]['sources'] == [1,2] and groups[1]['targets'] == [[0,1],[1,2]]
    full = np.arange(15*2,dtype=np.float32).reshape(1,15,2)
    restored = np.full_like(full,np.nan)
    for group in groups:
        index = group['indices']; cf = full[0,index,:].transpose(0,3,1,2)
        restored[0,index.ravel(),:] = cf.transpose(0,2,3,1).reshape(-1,2)
    assert np.array_equal(full,restored)
    assert len(np.unique(np.concatenate([g['indices'].ravel() for g in groups]))) == 15


@pytest.mark.parametrize('error',['missing','duplicate'])
def test_incomplete_or_duplicate_patch_does_not_count_as_complete(error):
    ii,jj,kk = (v.copy() for v in graph())
    if error == 'missing': ii,jj,kk = ii[:-1],jj[:-1],kk[:-1]
    else:
        source = int(ii[0]); target = int(jj[0])
        other = np.flatnonzero((ii==source)&(jj==target)&(kk!=kk[0]))[0]
        kk[other] = kk[0]
    with pytest.raises(ValueError, match='Incomplete'):
        AUDIT.independent_groups(ii,jj,kk,patches=3)


def profile(text=None):
    text = text or ('ID   OpType DataType Target InputShape OutputShape Cycles(DDR/NPU/Total) Time(us) MacUsage(%) WorkLoad(0/1/2) RW(KB) FullName\n'
        '1 InputOperator FLOAT16 CPU \\ (1,2) 0/0/0 4 0 0.0%/0.0%/0.0% 0 InputOperator:net\n'
        '2 Add FLOAT16 NPU (1,2) (1,2) 0/0/0 10 0 33.3%/33.3%/33.3% 1 Add:/Add\n'
        '3 Exp FLOAT16 CPU (1,2) (1,2) 0/0/0 5 0 0.0%/0.0%/0.0% 0 Exp:/Exp\n'
        'Total 9 0 10 19\n')
    data = text.encode()
    return dict(raw_text=text,bytes=len(data),raw_sha256=hashlib.sha256(data).hexdigest())


def test_actual_targets_are_parsed_not_inferred_from_compilation():
    row = AUDIT.parse_profile(profile())
    assert row['internal_target_counts'] == {'NPU':1,'CPU':1}
    assert row['CPU_IO_rows'] == 1 and row['target_microseconds'] == {'CPU':9,'GPU':0,'NPU':10}
    assert row['internal_CPU'][0]['op'] == 'Exp' and row['total_microseconds'] == 19


@pytest.mark.parametrize('error',['raw_sha','total'])
def test_profile_tampering_cannot_pass(error):
    row = profile()
    if error == 'raw_sha': row['raw_sha256'] = '0'*64
    else: row = profile(row['raw_text'].replace('Total 9 0 10 19','Total 0 0 0 0'))
    with pytest.raises(ValueError, match='SHA|total'):
        AUDIT.parse_profile(row)
