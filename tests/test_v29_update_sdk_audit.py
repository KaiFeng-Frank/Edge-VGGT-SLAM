"""Synthetic only: portable profile, mixed dtype and rejection evidence gates."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
PATH=HERE/'audit-v29-update-sdk.py'
if not PATH.is_file():PATH=HERE.parent/'deploy/rk3588/validation/audit-v29-update-sdk.py'
spec=importlib.util.spec_from_file_location('_v29_sdk_audit_test',PATH)
sdk=importlib.util.module_from_spec(spec);spec.loader.exec_module(sdk)


def profile():
    widths=(5,19,9,7,41,23,25,13,20,20,13,12)
    def row(values):return ''.join(str(value).ljust(width) for value,width in zip(values,widths))
    header=row(sdk.COLUMNS)
    rows=[row((1,'InputOperator','FLOAT16','CPU','\\','(1,1)','0/0/0',1,'','0%/0%/0%',0,'InputOperator:net')),
          row((2,'Gather','FLOAT16','CPU','(1,1)','(1,1)','0/0/0',7,'','0%/0%/0%',0,'Gather:one')),
          row((3,'Conv','FLOAT16','NPU','(1,1)','(1,1)','0/0/0',11,'','100%/0%/0%',0,'Conv:two'))]
    text='\n'.join(['Network Layer Information Table',header,'-'*220,*rows,'-'*220,
        'Total Operator Elapsed Per Frame Time(us): 19',
        'OpType CallNumber CPUTime(us) GPUTime(us) NPUTime(us) TotalTime(us) TimeRatio(%)','-'*99,
        'Gather 1 7 0 0 7 36.84%', 'Conv 1 0 0 11 11 57.89%', 'InputOperator 1 1 0 0 1 5.26%',
        '-'*99,'Total 8 0 11 19','-'*99])
    return bind(text)


def bind(text):
    blob=text.encode();return dict(query_command=3,bytes=len(blob),raw_sha256=hashlib.sha256(blob).hexdigest(),raw_text=text)


def test_actual_target_rows_and_summary_are_independently_reconciled():
    result=sdk.parse_profile(profile())
    assert result['row_count']==3 and result['target_counts']=={'CPU':2,'NPU':1}
    assert result['internal_target_counts']=={'CPU':1,'NPU':1}
    assert result['target_time_us']=={'CPU':8,'GPU':0,'NPU':11}
    assert result['operator_summary']['Gather']['CPU_us']==7
    assert result['total_operator_us']==19


@pytest.mark.parametrize('old,new',[
    ('Gather 1 7 0 0 7','Gather 1 6 0 0 6'),
    ('Total 8 0 11 19','Total 9 0 11 20'),
    ('Time(us)','UnknownTime'),
    ('CPU    ','DSP    '),
    ('2    Gather','4    Gather'),
])
def test_profile_rejects_target_format_missing_rows_or_forged_totals(old,new):
    value=profile();assert old in value['raw_text']
    with pytest.raises(ValueError):sdk.parse_profile(bind(value['raw_text'].replace(old,new,1)))


def test_profile_rejects_unbound_raw_bytes():
    value=profile();value['bytes']+=1
    with pytest.raises(ValueError,match='byte/hash'):sdk.parse_profile(value)


@pytest.mark.parametrize('value,accepted',[(0.,True),(1.,True),(-1.1920928955078125e-7,False),(1.00001,False),(np.nan,False),(np.inf,False)])
def test_exact_weight_domain_preserves_micro_negative_and_nonfinite_rejection(value,accepted):
    assert sdk.weight_domain(np.array([value],dtype=np.float32)) is accepted


def input_fixture():
    contracts=[];records=[];attrs=[]
    for i in range(13):
        isint=i in (3,5,7,9,10,12);name='input_'+str(i)
        contracts.append(dict(name=name,shape=[1,2],dtype=7 if isint else 1))
        records.append(dict(name=name,index=i,logical_shape=[1,2],source_dtype='int64' if isint else 'float32',transport_dtype='int64' if isint else 'float32',transport_type=8 if isint else 0,bytes=16 if isint else 8,format=3,pass_through=0,owned=True,exact_integer_cast_verified=isint))
        attrs.append(dict(index=i,name=name,shape=[1,2],n_elems=2,format=3,type=8 if isint else 1,size=16 if isint else 4,quantization_type=2,scale=1.,zero_point=0,fractional_length=0))
    return contracts,records,attrs


def test_INT64_neutral_affine_is_integer_not_a_float_buffer():
    contract,records,attrs=input_fixture()
    assert sdk.input_records_valid(contract,records,attrs)
    assert sum(x['transport_type']==8 for x in records)==6
    records[3]['transport_type']=0
    assert not sdk.input_records_valid(contract,records,attrs)


@pytest.mark.parametrize('field,value',[('scale',.5),('zero_point',1),('size',4),('format',1)])
def test_transport_rejects_nonidentity_quantization_size_or_guessed_layout(field,value):
    contract,records,attrs=input_fixture();attrs[3][field]=value
    assert not sdk.input_records_valid(contract,records,attrs)


def test_saved_nonfinite_arrays_are_counted_without_nan_JSON_or_loosened_gate():
    parent=sdk.load_parent(PATH.with_name('audit-v29-update-ort.py'))
    actual=np.array([np.nan,np.inf,-np.inf,0.],dtype=np.float32)
    counts=parent.finite(actual);comparison=parent.residual(actual,np.ones(4,dtype=np.float32))
    assert (counts['nan'],counts['positive_inf'],counts['negative_inf'])==(1,1,1)
    assert comparison['gate_passed'] is False and comparison['relative_l2'] is None
    assert comparison['nonfinite_pairs']==3
