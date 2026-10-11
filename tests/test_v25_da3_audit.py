"""Synthetic metadata/NumPy tests; no SDK/model/ORT/board execution."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
SOURCE = HERE/'audit-v25-da3-rknn.py'
if not SOURCE.exists():
    SOURCE = HERE.parent/'deploy/rk3588/validation/audit-v25-da3-rknn.py'
spec = importlib.util.spec_from_file_location('synthetic_v25_auditor', SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
V24 = audit.default_v24_auditor()


@pytest.mark.parametrize('public_exists',[False,True])
def test_portable_parent_name_priority(tmp_path,monkeypatch,public_exists):
    monkeypatch.setattr(audit,'__file__',str(tmp_path/'audit-v25-da3-rknn.py'))
    alias=tmp_path/'audit-da3-npu.py';alias.write_text('placeholder')
    public=tmp_path/'audit-v24-da3-rknn.py'
    if public_exists:public.write_text('placeholder')
    assert audit.default_v24_auditor()==(public if public_exists else alias)


def compiler(variant):
    return dict(script_sha256=audit.COMPILERS[variant],configuration=audit.compiler_config(variant),
        status='compiled_not_runtime_validated',released=True,input={'sha256':'a'*64},
        versions={'rknn-toolkit2':'2.3.2'},
        operations=[dict(name=n,code=0) for n in ('config','load_onnx','build','export_rknn')],
        frozen_v24_r1_reference_sha256=audit.OLDER_COMPILERS[0],
        frozen_v12_reference_sha256=audit.OLDER_COMPILERS[1],
        frozen_v24_r2_reference_sha256=audit.COMPILERS['A'])


@pytest.mark.parametrize('variant',['A','B'])
def test_only_expected_fixed_compiler_configuration(variant):
    value=compiler(variant)
    audit.validate_compiler(value,variant,'a'*64)
    value['configuration']['disable_rules'].append('unapproved_rule')
    with pytest.raises(ValueError,match='configuration/provenance'):
        audit.validate_compiler(value,variant,'a'*64)


@pytest.mark.parametrize('field',['script_sha256','frozen_v24_r1_reference_sha256',
                                'frozen_v12_reference_sha256','frozen_v24_r2_reference_sha256'])
def test_B_parent_or_compiler_pin_change_rejected(field):
    value=compiler('B');value[field]='f'*64
    with pytest.raises(ValueError,match='configuration/provenance'):
        audit.validate_compiler(value,'B','a'*64)


def timing():
    return dict(call_records=[dict(pair_index=i,forward_seconds=t) for i,t in
                zip(audit.NORMAL_SCHEDULE,[99.,1.,2.,3.,77.,88.])],first_pair_warm_mean_seconds=2.)


def test_warm_mean_excludes_cold_and_other_pairs():
    value=audit.warm_timing(timing())
    assert value['warm_seconds']==[1.,2.,3.] and value['warm_mean_seconds']==2.
    assert value['cold_seconds']==99. and value['other_pair_single_seconds']==[77.,88.]


@pytest.mark.parametrize('mutation',['mean','schedule','missing','nonfinite'])
def test_bad_normal_timing_rejected(mutation):
    value=timing()
    if mutation=='mean':value['first_pair_warm_mean_seconds']=2.1
    elif mutation=='schedule':value['call_records'][2]['pair_index']=1
    elif mutation=='missing':value['call_records'].pop()
    else:value['call_records'][1]['forward_seconds']=float('nan')
    with pytest.raises(ValueError):audit.warm_timing(value)


def test_pooled_ABBA_arithmetic_and_profile_excluded():
    groups={k:{'warm_seconds':[v]*3} for k,v in [('A1',1.),('A2',3.),('B1',.5),('B2',1.5)]}
    groups['profile_B1']={'warm_seconds':[900.]}
    result=audit.merged_timing(groups)
    assert result['A']['pooled_warm_mean_seconds']==2.
    assert result['B']['pooled_warm_mean_seconds']==1.
    assert result['A']['warm_sample_count']==6
    assert result['observed_relative_reduction']==.5
    assert result['statistical_significance_or_controlled_speedup_claim'] is False


def test_profile_prose_totals_and_native_camera_shape():
    helper=audit.load_v24(V24)
    raw=('1 InputOperator FLOAT16 CPU \\ (2,3,252,336) 0/0/0 4 0 InputOperator:images\n'
         '2 Transpose FLOAT16 CPU (2,768,1,1) (1,768,1,2) 0/0/0 395 0 Transpose:/cam_dec/Reshape_output_0_tp\n'
         '3 Conv FLOAT16 NPU (2,768,1,1) (2,9,1,1) 1/2/2 5 0 Conv:/cam_dec/fc_t/Gemm#2\n'
         'Total Operator Elapsed Per Frame Time(us): 404\n'
         'Total Memory Read/Write Per Frame Size(KB): 3\n'
         'OpType CallNumber CPUTime(us) GPUTime(us) NPUTime(us) TotalTime(us)\n'
         'Total 399 0 5 404\n')
    detail=dict(raw_text=raw,bytes=len(raw.encode()),raw_sha256=hashlib.sha256(raw.encode()).hexdigest())
    parsed=helper.parse_profile(detail)
    assert parsed['targets']=={'CPU':2,'NPU':1} and parsed['internal_CPU_layers']==1
    rows=audit.sdk_profile_focus(detail)['camera_transpose_rows']
    assert len(rows)==1 and rows[0]['time_us']==395.
    assert '(2,768,1,1)' in rows[0]['raw_row']
    detail['raw_sha256']='b'*64
    with pytest.raises(ValueError,match='bytes/SHA'):helper.parse_profile(detail)


def test_fixed_FP32_difference_is_reported_not_relaxed():
    helper=audit.load_v24(V24)
    expected=np.ones((2,3),np.float32);actual=expected.copy();actual[1,2]+=.01
    result=helper.compare(actual,expected)
    assert result['failed_elements']==1 and not result['elementwise_gate_passed']
    assert result['rtol']==1e-4 and result['atol']==1e-5 and not result['C_order_bytes_equal']


def test_v24_source_tampering_refuses_import(tmp_path):
    path=tmp_path/'bad.py';path.write_text('raise RuntimeError("must not execute")\n')
    with pytest.raises(ValueError,match='SHA mismatch'):audit.load_v24(path)


def test_missing_platform_explicit_unknown(tmp_path):
    result=audit.platform_observation(tmp_path/'missing.json','B',json.loads)
    assert result['status']=='unknown' and not result['temperatures_or_frequencies_controlled']


@pytest.mark.parametrize('bad',['variant','reversed'])
def test_platform_identity_and_order(tmp_path,bad):
    path=tmp_path/'platform_samples.json'
    value=dict(variant='B',scope='observations only',before=dict(monotonic=10.,sysfs={'freq':'408000'}),after=dict(monotonic=11.,sysfs={'freq':'1416000'}))
    reader=lambda p:json.loads(p.read_text())
    path.write_text(json.dumps(value))
    out=audit.platform_observation(path,'B',reader)
    assert out['before']['sysfs']['freq']=='408000' and not out['per_call_temperatures_or_during_run_frequency_known']
    if bad=='variant':value['variant']='A'
    else:value['after']['monotonic']=9.
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):audit.platform_observation(path,'B',reader)


def test_incomplete_bundle_returns_failed_not_empty_pass(tmp_path):
    args=SimpleNamespace(v24_auditor=V24,cpu_dir=tmp_path/'missing',export_dir=tmp_path,
                         ort_dir=tmp_path,compile_a=tmp_path,compile_b=tmp_path)
    result=audit.audit(args)
    assert result['status']=='failed' and result['checks']['failures']
    assert not result['model_OR_ORT_executed_by_auditor']


def test_output_refuses_overwrite_before_loading_sources(tmp_path):
    path=tmp_path/'existing.json';path.write_text('{}')
    with pytest.raises(SystemExit) as error:audit.main(['--output',str(path)])
    assert error.value.code==2 and path.read_text()=='{}'
