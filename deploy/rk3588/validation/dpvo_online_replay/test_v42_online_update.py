"""Tiny generated topology/fake execution and JSON contracts; no SDK/models."""
import ast
import copy
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

TEST = Path(__file__).resolve().parent
HERE = TEST if (TEST/'v42_online_update.py').is_file() else Path(os.environ['V42_ONLINE_SOURCE'])
PARENT = Path(os.environ.get('V42_PARENT_SDK', HERE.parent/'public-snapshot/deploy/rk3588/validation/dpvo_grouped_update'))


def load(path, label):
    spec = importlib.util.spec_from_file_location(label, path)
    value = importlib.util.module_from_spec(spec); sys.modules[label] = value
    spec.loader.exec_module(value); return value


online = load(HERE/'v42_online_update.py', '_v42_toy_online')
C = load(HERE/'v42_online_contracts.py', '_v42_toy_contracts')
old = C.load_v41_parent(HERE/'v41_online_contracts.py')
C.load_m16_parent(HERE/'v40_m16_grouped_contracts.py')
C.load_parent(HERE/'v37_grouped_contracts.py')
adapter = load(PARENT/'source_frame_batches.py', '_v42_toy_adapter')
adapter.load_pinned_plan(PARENT/'source_frame_plan.py')
runtime = load(PARENT/'v37_grouped_rknn.py', '_v42_toy_runtime')


def io(b, d):
    return ([dict(name=n,shape=[b,c,d,16],dtype='float32') for n,c in zip(C.INPUT_NAMES,(384,384,882))],
            [dict(name=n,shape=[b,c,d,16],dtype='float32') for n,c in zip(C.OUTPUT_NAMES,(384,2,2))])


def graph(degrees):
    ii=[];jj=[];kk=[]
    for i,d in enumerate(degrees):
        for j in range(d):
            for k in range(16):ii.append(i);jj.append(j);kk.append(i*16+k)
    index=np.random.default_rng(5).permutation(len(ii))
    ii,jj,kk=(np.ascontiguousarray(np.array(v,np.int64)[index]) for v in (ii,jj,kk))
    values=[np.broadcast_to(np.arange(len(ii),dtype=np.float32)[None,:,None]+1,(1,len(ii),c)).copy()
            for c in (384,384,882)]
    return (*values,ii,jj,kk)


class FakeCache:
    def __init__(self, fail_after=None):
        self.specs={key:{} for key in online.MODEL_KEYS};self.calls=[]
        self.closed=False;self.fail_after=fail_after
    def execute(self,key,feed,reference=None,*,timeout_ms):
        self.calls.append(key)
        if len(self.calls)==self.fail_after:raise RuntimeError('toy SDK failure')
        b,_,d,m=feed['net'].shape
        return dict(net_out=feed['net']+np.float32(3),delta=feed['inp'][:,:2].copy(),
            weight=np.full((b,2,d,m),.5,np.float32)),dict(execution_status='completed',
            outputs_finite=True,weight_domain_valid=True,all_outputs_zero=False,
            outputs_released=True,strict_FP32_gate_passed=None)
    def metrics(self):
        return dict(closed=self.closed,cleanup_errors=[],resident_contexts=0,current_DMA_bytes=0,
            current_model_host_bytes=0,histories=[],sessions={k:dict(npu_used=False) for k in self.calls})
    def close(self):self.closed=True


def callback(tmp_path, cache=None, probe=None):
    return online.OnlineM16Update(runtime,adapter,cache or FakeCache(),
        motion_probe=probe or (lambda *v:pytest.fail('Full groups cannot call Torch')),
        diagnostic_allow_fp32_rejected=True,failure_dir=tmp_path)


def test_complete_D3_and_old_groups_scatter_exactly_without_input_mutation(tmp_path):
    cache=FakeCache();owner=callback(tmp_path,cache);values=graph([3,4,4])
    before=[v.copy() for v in values];out=owner(*values)
    assert cache.calls==['B1_D3_M16','B2_D4_M16']
    assert np.array_equal(out['net'],values[0]+3) and np.array_equal(out['delta'],values[1][:,:,:2])
    assert all(np.array_equal(a,b) for a,b in zip(values,before))
    assert all(v.flags.owndata for v in out.values())
    state=owner.metrics();assert state['full_completed']==1 and state['candidate_accepted'] is False
    assert state['historical_SDK_strict_FP32_rejected'] is True and state['calls'][0]['NPU_used'] is False
    owner.close();owner.close();assert owner.metrics()['closed']


@pytest.mark.parametrize('degrees',([3,3],[1],[8]))
def test_uncompiled_complete_shape_still_saves_inputs_and_stops(tmp_path,degrees):
    cache=FakeCache();owner=callback(tmp_path,cache);values=graph(degrees)
    with pytest.raises(online.OnlineUpdateError,match='Unknown complete online shape'):owner(*values)
    assert cache.calls==[] and owner.metrics()['full_completed']==0
    with np.load(tmp_path/'update-000000-failed-inputs.npz',allow_pickle=False) as saved:
        assert all(np.array_equal(saved[n],v) for n,v in zip(('net','inp','corr','ii','jj','kk'),values))
    owner.close()


def test_late_group_failure_returns_no_partial_graph(tmp_path):
    cache=FakeCache(fail_after=2);owner=callback(tmp_path,cache);values=graph([3,4]);before=[v.copy() for v in values]
    with pytest.raises(online.OnlineUpdateError,match='toy SDK failure'):owner(*values)
    assert cache.calls==['B1_D3_M16','B1_D4_M16'] and owner.metrics()['full_completed']==0
    assert not owner.metrics()['calls'][0]['fullgraph_returned']
    assert all(np.array_equal(a,b) for a,b in zip(values,before));owner.close()


def test_D1_route_remains_explicit_startup_only(tmp_path):
    calls=[]
    def probe(net,inp,corr,ii,jj,kk):
        calls.append(1);return dict(net=net,delta=np.ones((1,16,2),np.float32),weight=np.full((1,16,2),.5,np.float32))
    owner=callback(tmp_path,probe=probe);values=list(graph([1]));values[4].fill(1)
    owner.route='motion_probe';owner(*values)
    assert calls==[1] and owner.metrics()['motion_probe_completed']==1 and not owner.cache.calls
    assert owner.metrics()['calls'][0]['execution_backend']=='original_Torch_startup_probe';owner.close()


def test_exact_nine_shapes_and_D3_dtype_and_integer_dimensions():
    assert set(C.MODEL_KEYS)==online.MODEL_KEYS==set(old.MODEL_KEYS)|{'B1_D3_M16'}
    for b,d in [(1,3)]+[(b,d) for b in (1,2) for d in range(4,8)]:assert C.model_key(*io(b,d))==f'B{b}_D{d}_M16'
    for b,d in ((2,3),(1,2),(1,1),(3,4)):
        with pytest.raises(ValueError):C.contracts(*io(b,d))
    ins,outs=io(1,3);ins[0]['shape'][0]=1.0
    with pytest.raises(ValueError):C.contracts(ins,outs)
    ins,outs=io(1,3);outs[0]['dtype']='float16'
    with pytest.raises(ValueError):C.contracts(ins,outs)


def catalogue(tmp_path):
    parent=dict(models={},strict_FP32_gate_passed=True)
    added=dict(models={},strict_FP32_gate_passed=True)
    eof=dict(models={},strict_FP32_gate_passed=False)
    entries={}
    for key in old.MODEL_KEYS:
        b,d=int(key[1]),int(key[4]);ins,outs=io(b,d)
        (added if key in old.NEW_KEYS else parent)['models'][key]=dict(inputs=ins,outputs=outs)
        entries[key]=dict(compile_dir=str(tmp_path/key),compile_sha256='a'*64,
            model=dict(file='block.rknn',bytes=100,sha256='b'*64),lineage='v41' if key in old.NEW_KEYS else 'v40')
    oldrow=dict(schema=old.INDEX_SCHEMA,status='staged_eight_online_shapes_diagnostic',
        parent_manifest_sha256=C.EXPORT_MANIFEST_SHA256,additions_manifest_sha256='c'*64,
        candidate_accepted=False,models=entries)
    oldpath=tmp_path/'old-index.json';oldpath.write_text(json.dumps(oldrow));oldsha=C.sha256(oldpath)
    ins,outs=io(1,3);eof['models']['B1_D3_M16']=dict(inputs=ins,outputs=outs)
    row=dict(schema=C.INDEX_SCHEMA,status='staged_nine_online_shapes_diagnostic',
        old_online_compile_index_sha256=oldsha,parent_manifest_sha256=C.EXPORT_MANIFEST_SHA256,
        additions_manifest_sha256='c'*64,eof_manifest_sha256='d'*64,candidate_accepted=False,
        models=copy.deepcopy(entries))
    row['models']['B1_D3_M16']=dict(compile_dir=str(tmp_path/'D3'),compile_sha256='e'*64,
        model=dict(file='block.rknn',bytes=100,sha256='f'*64),lineage='v42')
    kwargs=dict(old_online_compile_index=oldpath,old_online_compile_index_sha256=oldsha,
        additions_manifest_sha256='c'*64,eof_manifest_sha256='d'*64,parent_compiler='unused',
        addition_compiler='unused',addition_compiler_sha256=C.V41_COMPILER_SHA256,
        eof_compiler='unused',eof_compiler_sha256='a'*64,m16_contracts='unused',v41_contracts='unused',
        online_contracts='unused',online_contracts_sha256='b'*64,verify_artifacts=False)
    return row,(parent,added,eof),kwargs


@pytest.mark.parametrize('mutation',(None,'old_entry','missing','extra','lineage','budget'))
def test_index_retains_all_original_entries_and_only_adds_D3(tmp_path,mutation):
    row,manifests,kwargs=catalogue(tmp_path)
    if mutation=='old_entry':row['models']['B1_D5_M16']['compile_sha256']='0'*64
    elif mutation=='missing':row['models'].pop('B1_D5_M16')
    elif mutation=='extra':row['models']['B2_D3_M16']=row['models']['B1_D3_M16']
    elif mutation=='lineage':row['models']['B1_D3_M16']['lineage']='v41'
    elif mutation=='budget':row['models']['B1_D3_M16']['model']['bytes']=256<<20
    path=tmp_path/'index.json';path.write_text(json.dumps(row))
    if mutation:
        with pytest.raises(ValueError):C.validate_compile_index(path,C.sha256(path),*manifests,**kwargs)
    else:
        _,specs,compiled,_=C.validate_compile_index(path,C.sha256(path),*manifests,**kwargs)
        assert set(specs)==online.MODEL_KEYS and compiled=={} and specs['B1_D3_M16']['export_strict_FP32_gate_passed'] is False


def test_callback_preserves_original_classes_methods_byte_semantics():
    legacy=Path(os.environ.get('V42_V41_CALLBACK',HERE.parent.parent/'edge-vggt-v41/online/v41_online_update.py'))
    oldtree=ast.parse(legacy.read_text());newtree=ast.parse((HERE/'v42_online_update.py').read_text())
    classes=lambda tree:{v.name:ast.dump(v,include_attributes=False) for v in tree.body if isinstance(v,ast.ClassDef)}
    before=classes(oldtree);after=classes(newtree)
    before['OnlineM16Update']=before['OnlineM16Update'].replace('Exact eight online B1/B2 D4..7 M16 models required',
        'Exact old eight online models plus B1 D3 M16 required')
    assert before==after


def test_factory_rejects_missing_diagnostic_without_loading_sources(tmp_path,monkeypatch):
    factory=load(HERE/'v42_online_factory.py','_v42_toy_factory')
    monkeypatch.setattr(factory,'load',lambda *a:pytest.fail('Sources cannot load without flag'))
    kwargs={n:tmp_path for n,p in inspect.signature(factory.open_board_online_from_index).parameters.items()
            if p.default is inspect.Parameter.empty}
    kwargs.update(motion_probe=lambda *a:None,diagnostic_allow_fp32_rejected=False)
    with pytest.raises(ValueError,match='Explicit FP32-rejected'):factory.open_board_online_from_index(**kwargs)


def new_export_metadata():
    weights={f'p{i}':{'synthetic':i} for i in range(94)}
    params={f'p{i}':{'synthetic':i} for i in range(50)}
    parent=dict(original_weights_before=weights,original_Update_parameters=params)
    desc=lambda shape,dtype='float32':dict(shape=shape,dtype=dtype,bytes=int(np.prod(shape))*(8 if dtype=='int64' else 4),tensor_sha256='a'*64)
    teacher=dict(file='teacher.npz',bytes=100,sha256='b'*64,
        arrays={n:desc([1,640,c]) for n,c in (('net',384),('delta',2),('weight',2))})
    kind='new_original_Torch_on_saved_NPU_feedback_input'
    binding=dict(parent_run_sha256=C.SOURCE_BINDING['source_run_sha256'],
        parent_failure_sha256=C.SOURCE_BINDING['source_failure_sha256'],
        before_termination_sha256=C.SOURCE_BINDING['source_before_termination_sha256'],
        failed_update_index=203,selected_source_slot=11,selected_target_slots=[9,10,11],
        canonical_edge_indices=desc([1,3,16],'int64'),teacher_reference_kind=kind,original_CPU_trajectory_gold_used=False)
    ins,outs=io(1,3)
    model=dict(inputs=ins,outputs=outs,batch=1,degree=3,patches=16,source_binding=binding,
        numerical_validation_completed=True,parameter_coverage=dict(original_parameter_count=50,
        all_original_parameters_covered=True,mapping=[dict(original_matches=[dict(parameter=k) for k in params])]))
    gate=dict(finite=True,rtol=1e-4,atol=1e-5,gate_passed=True,failed_elements=0)
    case=dict(model_key='B1_D3_M16',edges=48,original_call_index=203,observation_id=191,
        source_binding=binding,original_full_inputs_artifact=dict(sha256=C.SOURCE_BINDING['source_failed_input_sha256']),
        full_original_teacher_artifact=teacher)
    for label in ('Torch_vs_original_group_gold','ORT_vs_original_group_gold','ORT_vs_grouped_Torch'):
        case[label]={n:copy.deepcopy(gate) for n in ('net','delta','weight')}
    row=dict(schema=C.EXPORT_SCHEMA,script_sha256=C.EOF_EXPORTER_SHA256,
        source_M16_manifest_sha256=C.EXPORT_MANIFEST_SHA256,
        parent_exporter_sha256=C.load_m16_parent().EXPORTER_SHA256,
        v37_exporter_sha256=C.load_m16_parent().PARENT_EXPORTER_SHA256,
        **C.SOURCE_BINDING,status='completed_EOF_M16_D3_numerical_accepted_not_adopted',
        strict_FP32_gate_passed=True,candidate_accepted=False,outputs_finite=True,weight_domain_valid=True,
        historical_SDK_strict_FP32_rejected=True,diagnostic_allow_fp32_rejected=True,terminal_sigmoid='stable',
        original_parameters_unchanged=True,original_weights_before=weights,original_weights_after=copy.deepcopy(weights),
        original_Update_parameters=params,source_pins={},source_hashes_after={},sources_and_inputs_unchanged=True,
        cleanup_errors=[],ORT_session_references_released=True,affinity_restored=True,
        strict_FP32_thresholds=dict(rtol=1e-4,atol=1e-5),strict_FP32_thresholds_relaxed=False,
        output_clipping=False,host_model_inference=False,NPU_used=False,BA_executed=False,VO_state_committed=False,
        original_full_Update_reexecuted=True,original_full_Update_forward_calls=1,teacher_reference_kind=kind,
        original_CPU_trajectory_gold_used=False,source_member_padding=False,neighbor_or_edge_removal=False,
        full_teacher_artifact=teacher,models={'B1_D3_M16':model},cases=[case],strict_comparison_count=9)
    return row,parent


@pytest.mark.parametrize('mutation',(None,'teacher_claim','cross_source','gate_count','reject_label'))
def test_D3_teacher_binding_does_not_masquerade_as_CPU_trajectory_gold(tmp_path,mutation):
    row,parent=new_export_metadata()
    if mutation=='teacher_claim':row['original_CPU_trajectory_gold_used']=True
    elif mutation=='cross_source':row['cases'][0]['source_binding']['selected_source_slot']=10
    elif mutation=='gate_count':row['cases'][0]['ORT_vs_original_group_gold']['delta']['failed_elements']=1
    elif mutation=='reject_label':row['historical_SDK_strict_FP32_rejected']=False
    path=tmp_path/'export.json';path.write_text(json.dumps(row))
    kwargs=dict(parent_manifest=parent,exporter_sha256=C.EOF_EXPORTER_SHA256)
    if mutation:
        with pytest.raises(ValueError):C.validate_export_manifest(path,C.sha256(path),**kwargs)
    else:
        verified,_=C.validate_export_manifest(path,C.sha256(path),**kwargs)
        assert verified['teacher_reference_kind']=='new_original_Torch_on_saved_NPU_feedback_input'
