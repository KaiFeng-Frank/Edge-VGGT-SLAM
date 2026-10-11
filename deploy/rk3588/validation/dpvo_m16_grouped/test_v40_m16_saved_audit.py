"""Tiny generated NumPy payloads only; no SDK, Torch, ORT or real model runs."""
import copy
import hashlib
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get('V40_CONTRACT_SOURCE_DIR',
    HERE if (HERE/'audit-v40-m16-grouped.py').is_file() else
    HERE.parent/'deploy'/'rk3588'/'validation'/'dpvo_m16_grouped'))
PARENT = Path(os.environ.get('V40_PARENT_GROUPED_SOURCE',SOURCE.parent/'dpvo_grouped_update'))
spec = importlib.util.spec_from_file_location('_v40_test_saved_audit',SOURCE/'audit-v40-m16-grouped.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
p = a.load_parent(PARENT)


def test_fixed_element_threshold_cannot_be_replaced_by_relative_norm():
    reference = np.zeros(128,dtype=np.float32)
    actual = reference.copy(); actual[17] = np.float32(1.01e-5)
    comparison = p.numerical(actual,reference)
    assert comparison['failed_elements'] == 1 and not comparison['gate_passed']
    recorded = copy.deepcopy(comparison); recorded.update(gate_passed=True,failed_elements=0)
    checks = a.Checks(p); checks.metrics(comparison,recorded,'dishonest_gate')
    assert set(checks.failures) == {'dishonest_gate:gate_passed','dishonest_gate:failed_elements'}


def test_nonfinite_is_rejected_with_json_safe_diagnostic():
    reference = np.zeros(2,dtype=np.float32); actual = np.array([0,np.nan],dtype=np.float32)
    result = p.numerical(actual,reference)
    assert not result['finite'] and not result['gate_passed'] and result['nonfinite_elements'] == 1
    assert result['relative_L2'] is result['max_abs'] is None


def test_independent_M16_groups_preserve_original_permuted_edge_order():
    edges = [(source,target,source*16+patch) for source,targets in ((0,(0,2)),(3,(1,2)),(7,(0,1,3)))
             for target in targets for patch in range(16)]
    values = np.asarray(edges,dtype=np.int64)[np.random.default_rng(17).permutation(len(edges))]
    groups = p.independent_groups(*values.T,patches=16)
    assert [(g['key'],g['sources']) for g in groups] == [('B2_D2_M16',[0,3]),('B1_D3_M16',[7])]
    covered = np.concatenate([g['indices'].ravel() for g in groups])
    assert np.array_equal(np.sort(covered),np.arange(len(edges)))
    for group in groups:
        for index,source in enumerate(group['sources']):
            selected = values[group['indices'][index]]
            assert np.all(selected[...,0] == source)
            assert np.array_equal(selected[...,2]%16,np.broadcast_to(np.arange(16),selected.shape[:2]))


def test_incomplete_or_duplicated_dependency_is_not_a_valid_group():
    ii = np.zeros(32,dtype=np.int64); jj = np.repeat(np.arange(2,dtype=np.int64),16)
    kk = np.tile(np.arange(16,dtype=np.int64),2); kk[-1] = kk[-2]
    with pytest.raises(ValueError,match='Incomplete'):
        p.independent_groups(ii,jj,kk,patches=16)


def saved_payload(tmp_path):
    path = tmp_path/'tiny.npz'; values = dict(net=np.arange(6,dtype=np.float32).reshape(1,2,3))
    np.savez(path,**values)
    return dict(file=path.name,bytes=path.stat().st_size,sha256=a.sha(path),
                arrays={key:p.raw(value) for key,value in values.items()})


def test_artifact_bytes_and_raw_metadata_are_independently_checked(tmp_path):
    row = saved_payload(tmp_path); checks = a.Checks(p)
    actual = checks.read_arrays(tmp_path,row)
    assert not checks.failures and actual['net'][0,1,2] == 5
    row['sha256'] = '0'*64; row['arrays']['net']['tensor_sha256'] = 'f'*64
    changed = a.Checks(p); changed.read_arrays(tmp_path,row)
    assert 'artifact:tiny.npz' in changed.failures and 'raw_bytes:tiny.npz:net' in changed.failures


@pytest.mark.parametrize('filename',['../tiny.npz','/tiny.npz'])
def test_artifact_path_cannot_escape_saved_directory(tmp_path,filename):
    row = saved_payload(tmp_path); row['file'] = filename
    with pytest.raises(ValueError,match='Unsafe artifact'):
        a.Checks(p).read_arrays(tmp_path,row)


def test_actual_NHWC_input_and_padded_CF_output_are_distinct():
    logical = dict(name='delta',shape=[2,2,4,16])
    common = dict(index=0,name='delta',n_elems=256,size=512,type=1,quantization_type=2,
                  scale=1,zero_point=0,fractional_length=0)
    output = dict(common,shape=logical['shape'],format=0,size_with_stride=2048)
    assert a.native_attr_valid(output,logical,0,output=True)
    input_row = dict(common,shape=[2,4,16,2],format=1,size_with_stride=512)
    assert a.native_attr_valid(input_row,logical,0)
    input_row['size_with_stride'] = 2048
    assert not a.native_attr_valid(input_row,logical,0)
    output['size_with_stride'] = 256
    assert not a.native_attr_valid(output,logical,0,output=True)


def test_nonidentity_affine_metadata_is_not_neutral():
    logical = dict(name='delta',shape=[2,2,4,16])
    row = dict(index=0,name='delta',shape=logical['shape'],format=0,type=1,n_elems=256,size=512,
               size_with_stride=512,quantization_type=2,scale=.5,zero_point=0,fractional_length=0)
    assert not a.native_attr_valid(row,logical,0,output=True)


def test_native_error_is_rejected_even_with_consistent_raw_log_SHA():
    data = b'E RKNN normalize error (-1)\n'
    row = dict(raw_text=data.decode(),bytes=len(data),raw_sha256=hashlib.sha256(data).hexdigest(),
               truncated=False,RKNN_error_detected=False)
    checks = a.Checks(p);checks.native_log(row,'SDK_log')
    assert checks.failures == ['SDK_log']


def test_profile_total_and_text_tampering_are_rejected():
    text = ('ID   OpType DType Target Shape DDR Time(us) WorkLoad(0/1/2)\n'
        '1 InputOperator FLOAT16 CPU (1,1,1,1) 0/0/0 5 0 0.0%/0.0%/0.0% 0 InputOperator:input\n'
        '2 Conv FLOAT16 NPU (1,1,1,1) 0/0/0 7 0 100.0%/0.0%/0.0% 0 Conv:generated\n'
        'Total 5 0 7 12\n')
    row = dict(raw_text=text,bytes=len(text.encode()),raw_sha256=hashlib.sha256(text.encode()).hexdigest())
    result = p.parse_profile(row)
    assert result['internal_target_counts'] == {'NPU':1} and result['total_microseconds'] == 12
    changed = dict(row,raw_sha256='f'*64)
    with pytest.raises(ValueError,match='SHA'):
        p.parse_profile(changed)
    text = text.replace('Total 5 0 7 12','Total 5 0 7 13')
    changed = dict(raw_text=text,bytes=len(text.encode()),raw_sha256=hashlib.sha256(text.encode()).hexdigest())
    with pytest.raises(ValueError,match='total'):
        p.parse_profile(changed)


def test_parent_rejected_before_modified_python_can_be_imported(tmp_path):
    (tmp_path/'audit-v38-grouped-full.py').write_text('raise AssertionError("must not import")\n')
    with pytest.raises(ValueError,match='ancestor SHA'):
        a.load_parent(tmp_path)
