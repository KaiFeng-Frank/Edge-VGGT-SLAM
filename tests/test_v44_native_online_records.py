"""Generated native descriptors and negative guards; no model/SDK/real arrays."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

TEST=Path(__file__).resolve().parent
HERE=Path(os.environ.get('V44_ONLINE_SOURCE',TEST))
if not (HERE/'audit-v44-native-online-records.py').exists():HERE=TEST.parent/'deploy/rk3588/validation/dpvo_online_replay'
VALIDATION=Path(os.environ.get('V44_VALIDATION',HERE.parent))
if not (VALIDATION/'dpvo_native_input/audit-v43-native-full-ab.py').exists():
    VALIDATION=HERE.parent/'public-snapshot/deploy/rk3588/validation'
spec=importlib.util.spec_from_file_location('_v44_records_auditor',HERE/'audit-v44-native-online-records.py')
auditor=importlib.util.module_from_spec(spec);sys.modules[spec.name]=auditor;spec.loader.exec_module(auditor)
parent=auditor.load_parent(VALIDATION/'dpvo_native_input/audit-v43-native-full-ab.py')


def native_log(raw=''):
    data=raw.encode()
    return dict(raw_text=raw,bytes=len(data),raw_sha256=hashlib.sha256(data).hexdigest(),truncated=False,RKNN_error_detected=False)


def context():
    queries=[];bindings=[]
    for i,(name,c) in enumerate(zip(('net','inp','corr'),(384,384,882))):
        shape=[1,(c+7)//8,3,16,8];elements=1
        for v in shape:elements*=v
        attr=dict(index=i,name=name,n_dims=5,shape=shape,dims=shape+[0]*11,fmt=2,type=1,qnt_type=2,
            fl=0,zp=0,scale=1.,scale_finite=True,n_elems=elements,size=2*elements,size_with_stride=2*elements,
            w_stride=16,h_stride=0,pass_through=0)
        queries.extend(dict(command=q,input_index=i,return_code=0,attribute=copy.deepcopy(attr)) for q in (8,10))
        bindings.append(dict(name=name,native_input_attribute=attr,pass_through=1,transport_format=2,
            transport_dtype='float16',native_query_commands=[8,10],transport_bytes=2*elements))
    return dict(native_input_queries=queries,input_bindings=bindings,native_inputs_validated=True,
        input_transport=parent.NATIVE,diagnostic_allow_native_fp16=True,transport_equivalence_validated=False,
        native_query_ancestor_sha256='89f0cead99df024b25ed3d2638e412a075f91d6736ecdb2b35ace2304eef0642',
        native_input_query_log=native_log(),packing_FPCR_at_initialization=dict(supported=True,FPCR_modified_by_packer=False),
        synthetic=False,collect_perf_detail=False,core_mask=7,SDK_version='2.3.2 (generated)',driver_version='0.7.2')


def group():
    rows=[]
    for i,(name,c) in enumerate(zip(('net','inp','corr'),(384,384,882))):
        shape=[1,(c+7)//8,3,16,8];elements=1
        for v in shape:elements*=v
        rows.append(dict(index=i,name=name,logical_shape=[1,c,3,16],source_dtype='float32',owned=True,
            buffer_shape=shape,logical_attribute_format=1,native_attribute_format=2,native_query_commands=[8,10],
            request_format=2,request_type=1,transport_dtype='float16',buffer_storage_dtype='uint16',pass_through=1,
            bytes=elements*2,input_transport=parent.NATIVE,padded_channels=shape[1]*8-c,
            tail_padding_value='FP16_positive_zero',SDK_input_conversion_requested=False))
    return dict(model_key='B1_D3_M16',candidate_accepted=False,strict_FP32_gate_passed=None,
        strict_component_gate_passed=False,execution_status='completed',mode='normal',outputs_released=True,
        outputs_finite=True,weight_domain_valid=True,all_outputs_zero=False,input_transport=parent.NATIVE,
        diagnostic_allow_native_fp16=True,transport_equivalence_validated=False,native_log=native_log(),
        timing=dict(input_records=rows,synchronous=True,frame_id=0,pack_seconds=.001,input_set_seconds=.001,
            run_api_seconds=.001,outputs_get_seconds=.001,owned_copy_seconds=.001,outputs_release_seconds=.001,
            forward_seconds=.007))


def record():
    row=dict(index=0,route='full',status='completed_diagnostic_not_adopted',candidate_accepted=False,
        SDK_FP32_equivalence_validated=False,historical_SDK_strict_FP32_rejected=True,Torch_fallback=False,
        outputs_committed_by_callback=False,NPU_used=True,fullgraph_returned=True,
        execution_backend='RKNN_complete_source_groups',strict_FP32_gate_passed=None,edges=48,
        model_keys=['B1_D3_M16'],callback_seconds=.011,degree_plan=dict(all_edges_exactly_once=True,
            original_edge_order_restored=True,padding=False,edges=48),
        SDK_execution=dict(candidate_accepted=False,VO_state_committed=False,full_graph_returned=True,
            padding=False,dropped_edges=False,versus_reference=None,strict_FP32_gate_passed=None,
            sequential_groups=1,groups=[group()],full_graph_seconds=.009))
    e=row['edges'];row['inputs']={n:dict(shape=[1,e,c],dtype='float32',bytes=e*c*4)
        for n,c in [('net',384),('inp',384),('corr',882)]}
    row['inputs'].update({n:dict(shape=[e],dtype='int64',bytes=e*8) for n in ('ii','jj','kk')})
    row['outputs']={n:dict(shape=[1,e,c],dtype='float32',bytes=e*c*4)
        for n,c in [('net',384),('delta',2),('weight',2)]}
    return row


def test_generated_protocol_context_group_and_complete_record():
    assert auditor.validate_native_context(parent,context(),'B1_D3_M16')==6
    assert auditor.validate_group(parent,group())==3
    row=record();summary=auditor.validate_records(parent,[row],[copy.deepcopy(row)])
    assert summary['SDK_calls']==1 and summary['native_pack_records']==3 and summary['full_callbacks']==1
    second=copy.deepcopy(row);second['index']=1
    summary=auditor.validate_records(parent,[row,second],[copy.deepcopy(row),copy.deepcopy(second)])
    assert summary['SDK_calls']==2  # Synchronous frame IDs may remain zero, unlike explicit call counters.


@pytest.mark.parametrize('failure',('missing_query','bad_command','scale','fl','pass_through','FPCR','synthetic'))
def test_native_context_rejects_changed_actual_query_contract(failure):
    m=context()
    if failure=='missing_query':m['native_input_queries'].pop()
    if failure=='bad_command':m['native_input_queries'][0]['command']=1
    if failure in ('scale','fl','pass_through'):
        field={'scale':'scale','fl':'fl','pass_through':'pass_through'}[failure]
        for q in m['native_input_queries']:
            if q['input_index']==0:q['attribute'][field]=2 if failure=='scale' else 1
        m['input_bindings'][0]['native_input_attribute'][field]=2 if failure=='scale' else 1
    if failure=='FPCR':m['packing_FPCR_at_initialization']['supported']=False
    if failure=='synthetic':m['synthetic']=True
    with pytest.raises(ValueError):auditor.validate_native_context(parent,m,'B1_D3_M16')


@pytest.mark.parametrize('failure',('FP32_type','pass0','bytes','not_owned','tail','release','accepted','negative_clock'))
def test_per_call_native_pack_and_completion_cannot_be_relabelled(failure):
    row=group();pack=row['timing']['input_records'][2]
    if failure=='FP32_type':pack['request_type']=0
    if failure=='pass0':pack['pass_through']=0
    if failure=='bytes':pack['bytes']-=2
    if failure=='not_owned':pack['owned']=False
    if failure=='tail':pack['tail_padding_value']='uninitialized'
    if failure=='release':row['outputs_released']=False
    if failure=='accepted':row['strict_component_gate_passed']=True
    if failure=='negative_clock':row['timing']['input_set_seconds']=-.001
    with pytest.raises(ValueError):auditor.validate_group(parent,row)


def test_actual_error_text_rejected_even_if_API_error_flag_is_false():
    with pytest.raises(ValueError):auditor.validate_log(native_log('E RKNN unsupported input layout\n'))
    row=native_log();row['raw_sha256']='0'*64
    with pytest.raises(ValueError):auditor.validate_log(row)


@pytest.mark.parametrize('failure',('wrong_frame_id','wrong_scatter_edges','teacher_claim','float_indices'))
def test_complete_record_sequence_no_padding_no_numerical_adoption(failure):
    row=record()
    if failure=='wrong_frame_id':row['SDK_execution']['groups'][0]['timing']['frame_id']=-1
    if failure=='wrong_scatter_edges':row['degree_plan']['original_edge_order_restored']=False
    if failure=='teacher_claim':row['strict_FP32_gate_passed']=True
    if failure=='float_indices':row['inputs']['kk']['dtype']='float32'
    with pytest.raises(ValueError):auditor.validate_records(parent,[row],[copy.deepcopy(row)])


def test_source_pins_and_parent_auditor_bytes_are_strict(tmp_path):
    pins={'source':'1'*64};auditor.validate_source_maps(pins,dict(pins))
    with pytest.raises(ValueError):auditor.validate_source_maps(pins,{'source':'2'*64})
    with pytest.raises(ValueError):auditor.validate_source_maps({}, {})
    context_pins={'/generated/context/'+name:digest for name,digest in auditor.SDK_SOURCES.items()}
    auditor.validate_required_pins(context_pins,auditor.SDK_SOURCES)
    context_pins['/generated/context/librknnrt.so']='0'*64
    with pytest.raises(ValueError):auditor.validate_required_pins(context_pins,auditor.SDK_SOURCES)
    with pytest.raises(ValueError):auditor.validate_required_pins(pins,auditor.SDK_SOURCES)
    model=dict(compile_dir='/generated/B1_D3_M16',model=dict(file='block.rknn',bytes=4,sha256='1'*64))
    model_metrics=dict(source_pins={'/generated/B1_D3_M16/block.rknn':'1'*64},model_host_buffer_bytes=5)
    assert auditor.validate_model_binding(model_metrics,model)==5
    model_metrics['model_host_buffer_bytes']=4
    with pytest.raises(ValueError):auditor.validate_model_binding(model_metrics,model)
    fake=tmp_path/'audit.py';fake.write_text('raise RuntimeError("must not import")')
    with pytest.raises(ValueError,match='SHA differs'):auditor.load_parent(fake)


@pytest.mark.parametrize('constant',('NaN','Infinity','-Infinity'))
def test_nonfinite_JSON_is_not_accepted_or_normalized(tmp_path,constant):
    p=tmp_path/'run.json';p.write_text('{"value":'+constant+'}')
    with pytest.raises(ValueError,match='Nonfinite JSON'):auditor.read_json(p)
