"""Tiny synthetic NumPy/fake SDK tests. No model, ORT or SDK is executed."""
import ctypes as ct
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest


def helper(name):
    here=Path(__file__).resolve().parent
    for path in (here/name,here.parent/'deploy/rk3588/validation'/name):
        if path.is_file():return path
    raise FileNotFoundError(name)


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj
    spec.loader.exec_module(obj);return obj


runner=load(helper('run-v29-update-candidate-board.py'),'_synthetic_v29_candidate')
packing=load(helper('v28_rknn_inputs.py'),'_synthetic_v29_frozen_inputs')


@pytest.fixture(scope='module')
def abi():
    configured=os.environ.get('EDGE_VGGT_ABI_HELPER')
    path=Path(configured) if configured else helper('run-v12-rknn-block-board.py')
    assert runner.sha(path)==runner.ABI_SHA
    return load(path,'_synthetic_v29_abi_only')


def feed():
    return dict(net=np.arange(12,dtype=np.float32).reshape(1,3,4),
        inp=np.full((1,3,4),.25,np.float32),corr=np.ones((1,3,6),np.float32),
        previous_indices=np.array([0,0,0],np.int64),previous_valid=np.zeros((1,3,1),np.float32),
        following_indices=np.array([2,0,0],np.int64),following_valid=np.zeros((1,3,1),np.float32),
        kk_indices=np.array([[0,2],[1,0]],np.int64),kk_valid=np.array([[1,1],[1,0]],np.float32),
        kk_inverse=np.array([0,1,0],np.int64),ij_indices=np.array([[0,1,2]],np.int64),
        ij_valid=np.ones((1,3),np.float32),ij_inverse=np.zeros(3,np.int64))


def contracts():
    return [dict(name=k,shape=list(v.shape),dtype=7 if k in packing.INDEX_NAMES else 1) for k,v in feed().items()]


def values(taps=True):
    result=dict(net_out=np.arange(12,dtype=np.float32).reshape(1,3,4),
                delta=np.full((1,3,2),.125,np.float32),weight=np.full((1,3,2),.5,np.float32))
    if taps:result['/group/Where_output_0']=np.arange(16,dtype=np.float32).reshape(1,2,2,4)
    return result


def output_contracts(taps=True):
    return [dict(name=k,shape=list(v.shape),dtype=1) for k,v in values(taps).items()]


def attrs(rows):
    result=[]
    for i,row in enumerate(rows):
        kind=8 if row['dtype']==7 else 1;size=math.prod(row['shape'])*(8 if kind==8 else 2)
        result.append(dict(index=i,name=row['name'],shape=row['shape'],n_elems=math.prod(row['shape']),
            size=size,format=3,type=kind,quantization_type=2,scale=1.,zero_point=0,fractional_length=0))
    return result


def golden():return {key:values()[name] for name,key in zip(runner.CORE_NAMES,runner.REF_NAMES)}
def state(step=0,case='three_frame_all_pairs_identity'):
    return dict(case=case,step=step,model_key='E864')


class SDK:
    def __init__(self,abi,fault=None):
        self.abi,self.fault=abi,fault;self.runs=0;self.releases=0;self.destroyed=0;self.buffers={}
    def rknn_init(self,context,*unused):
        ct.cast(context,ct.POINTER(self.abi.CONTEXT)).contents.value=23
        return -1 if self.fault=='init' else 0
    def rknn_set_core_mask(self,context,mask):assert mask==7;return 0
    def rknn_inputs_set(self,context,count,requests):
        assert count==13
        for (name,value),request in zip(feed().items(),requests):
            kind=ct.c_int64 if name in packing.INDEX_NAMES else ct.c_float
            assert request.type==(8 if name in packing.INDEX_NAMES else 0)
            assert request.fmt==3 and request.pass_through==0 and request.size==value.nbytes
            got=np.ctypeslib.as_array(ct.cast(request.buf,ct.POINTER(kind)),shape=(value.size,))
            np.testing.assert_array_equal(got.reshape(value.shape),value)
        return -2 if self.fault=='input' else 0
    def rknn_run(self,context,record):
        self.runs+=1;record=ct.cast(record,ct.POINTER(self.abi.RunExtend)).contents
        assert record.non_block==0 and record.timeout_ms==30000;record.frame_id=19
        if self.fault=='native_log':os.write(2,b'E RKNN synthetic internal failure\n')
        return -3 if self.fault=='run' else 0
    def rknn_outputs_get(self,context,count,outputs,extension):
        self.buffers=values()
        if self.fault=='nan':self.buffers['delta'].flat[0]=np.nan
        if self.fault=='negative':self.buffers['weight'].flat[0]=-1.19e-7
        if self.fault=='inf_tap':self.buffers['/group/Where_output_0'].flat[0]=-np.inf
        assert count==len(self.buffers)
        for i,array in enumerate(self.buffers.values()):
            assert outputs[i].want_float==1 and outputs[i].is_prealloc==0 and outputs[i].index==i
            outputs[i].buf=array.ctypes.data;outputs[i].size=array.nbytes
            if self.fault=='partial':return -4
        if self.fault=='bytes':outputs[0].size-=4
        ct.cast(extension,ct.POINTER(self.abi.OutputExtend)).contents.frame_id=20 if self.fault=='frame' else 19
        return 0
    def rknn_outputs_release(self,context,count,outputs):
        self.releases+=1
        for array in self.buffers.values():array.fill(-9999)
        return -5 if self.fault=='release' else 0
    def rknn_destroy(self,context):self.destroyed+=1;return -6 if self.fault=='destroy' else 0


def execute(abi,fault=None):
    sdk=SDK(abi,fault);cleanup=[]
    raw,timing=runner.execute_sdk(sdk,abi.CONTEXT(1),abi,packing,feed(),
        packing.validate_input_attrs(contracts(),attrs(contracts())),
        runner.output_bindings(packing,output_contracts(),attrs(output_contracts())),cleanup)
    return sdk,raw,timing,cleanup


def test_sdk_owned_copy_and_all_four_outputs(abi):
    sdk,raw,timing,cleanup=execute(abi)
    assert sdk.runs==sdk.releases==1 and cleanup==[] and timing['outputs_released']
    assert len(timing['input_records'])==13 and timing['forward_seconds']>=timing['run_api_seconds']>=0
    for key,expected in values().items():
        np.testing.assert_array_equal(raw[key],expected)
        assert not np.shares_memory(raw[key],sdk.buffers[key])


@pytest.mark.parametrize('fault,releases',[('input',0),('run',0),('partial',1),('frame',1),('bytes',1),('release',1)])
def test_sdk_protocol_failure_preserves_release(abi,fault,releases):
    sdk=SDK(abi,fault);cleanup=[]
    with pytest.raises((ValueError,RuntimeError)):
        runner.execute_sdk(sdk,abi.CONTEXT(1),abi,packing,feed(),
            packing.validate_input_attrs(contracts(),attrs(contracts())),
            runner.output_bindings(packing,output_contracts(),attrs(output_contracts())),cleanup)
    assert sdk.releases==releases and bool(cleanup)==(fault=='release')


@pytest.mark.parametrize('fault,finite,domain',[('nan',False,True),('negative',True,False),('inf_tap',False,True)])
def test_invalid_raw_saved_without_nan_json_or_clipping(abi,tmp_path,fault,finite,domain):
    sdk,raw,timing,cleanup=execute(abi,fault)
    row=runner.save_call(tmp_path,0,state(),raw,dict(outputs=output_contracts()),golden(),golden(),timing)
    assert row['all_outputs_finite']==finite and row['weight_domain_passed']==domain
    with np.load(tmp_path/row['outputs']['file'],allow_pickle=False) as saved:
        for key in raw:np.testing.assert_array_equal(saved[key],raw[key])
    json.dumps(row,allow_nan=False)
    assert cleanup==[] and sdk.releases==1


@pytest.mark.parametrize('field,value',[('format',1),('type',8),('scale',.5),('name','other'),('size',1)])
def test_extra_tap_attr_requires_exact_float_layout_name_bytes(field,value):
    raw=attrs(output_contracts());raw[3][field]=value
    with pytest.raises(ValueError):runner.output_bindings(packing,output_contracts(),raw)


def write_npz(folder,name,arrays):
    path=folder/name;np.savez(path,**arrays)
    return dict(file=name,bytes=path.stat().st_size,sha256=runner.sha(path),arrays={k:runner.meta(v) for k,v in arrays.items()})


@pytest.mark.parametrize('mutation',['none','nonfinite','nonbinary','index','prefix','empty'])
def test_ORT_feed_runs_real_frozen_value_guards_before_execution(abi,tmp_path,mutation):
    capture=tmp_path/'capture';export=tmp_path/'export';capture.mkdir();export.mkdir();arrays=feed()
    if mutation=='nonfinite':arrays['corr'].flat[0]=np.nan
    if mutation=='nonbinary':arrays['kk_valid'].flat[0]=.5
    if mutation=='index':arrays['previous_indices'][0]=3
    if mutation=='prefix':arrays['kk_valid'][1]=[0,1]
    if mutation=='empty':arrays['kk_valid'][1]=[0,0]
    row=dict(inputs=write_npz(capture,'input.npz',dict(net=arrays['net'],ctx=arrays['inp'],corr=arrays['corr'])),
        topology=write_npz(export,'topology.npz',{k:v for k,v in arrays.items() if k not in ('net','inp','corr')}),
        original=write_npz(capture,'reference.npz',golden()),wrapper=write_npz(export,'wrapped.npz',golden()))
    pins=runner.Inputs();args=NS(capture_dir=capture,export_dir=export)
    if mutation=='none':
        got,*unused=runner.load_feed(row,args,pins,packing,dict(inputs=contracts()),abi)
        for k,v in arrays.items():np.testing.assert_array_equal(got[k],v)
        assert pins.unchanged()
    else:
        with pytest.raises(ValueError):runner.load_feed(row,args,pins,packing,dict(inputs=contracts()),abi)


def scenes():
    capture=dict(scenarios=[]);export=dict(scenarios=[])
    for key,prefix in [('E96','first_motion_0_to_1'),('E864','three_frame_all_pairs')]:
        for suffix in ('identity','shifted'):
            name=prefix+'_'+suffix;states=[dict(step=x,inputs={},outputs={}) for x in range(4)]
            capture['scenarios'].append(dict(name=name,states=states))
            export['scenarios'].append(dict(name=name,states=states,model_key=key,topology={}))
    return capture,export


@pytest.mark.parametrize('keys,key,first,count',[(('E864',),None,False,8),(('E96','E864'),None,False,16),
                                              (('E96','E864'),'E864',False,8),(('E864',),'E864',True,1)])
def test_state_selection_keeps_all_original_cases_steps(keys,key,first,count):
    capture,export=scenes();rows=runner.records(capture,export,dict(models=dict.fromkeys(keys)),key,first)
    assert len(rows)==count
    assert rows[0]['case'].endswith('_identity') and rows[0]['step']==0
    if not first:assert {x['step'] for x in rows}=={0,1,2,3}


def test_missing_saved_state_and_unknown_model_reject():
    capture,export=scenes();capture['scenarios'][0]['states'].pop()
    with pytest.raises(ValueError):runner.records(capture,export,dict(models={'E96':{}}))
    with pytest.raises(ValueError):runner.records(*scenes(),dict(models={'E96':{}}),'E864')


@pytest.mark.parametrize('line,expected',[('I RKNN started\n',False),('\x1b[31mE RKNN unsupported\x1b[0m\n',True)])
def test_native_error_text_is_separate_from_zero_return(tmp_path,line,expected):
    path=tmp_path/'log';path.write_text(line)
    assert bool(runner.native_errors(path))==expected


@pytest.mark.parametrize('mutation',['none','export_parent','capture_parent','graph_parent',
                                    'core_name','tap_type','tap_shape','graph_bytes','filename'])
def test_manifest_requires_pinned_parent_and_original_outputs(tmp_path,monkeypatch,mutation):
    capture=tmp_path/'capture';export=tmp_path/'export';candidate=tmp_path/'candidate'
    for path in (capture,export,candidate):path.mkdir()
    (export/'parent.onnx').write_bytes(b'parent graph');(candidate/'candidate.onnx').write_bytes(b'candidate graph')
    ancestor=dict(original_weights_unchanged=True,sources_and_inputs_unchanged=True,source_pins={},source_hashes_after={},cleanup_errors=[])
    original=dict(file='parent.onnx',sha256=runner.sha(export/'parent.onnx'),bytes=12,
                  inputs=contracts(),outputs=output_contracts(False))
    (capture/'capture.json').write_text(json.dumps(dict(ancestor,status='captured_original_Update_states_not_VO')))
    monkeypatch.setattr(runner,'CAPTURE_SHA',runner.sha(capture/'capture.json'))
    (export/'export.json').write_text(json.dumps(dict(ancestor,status='exported_Update_board_ORT_pending',
        capture_sha256=runner.CAPTURE_SHA,exact_wrapper_FP32_gate_passed=True,models={'E864':original})))
    monkeypatch.setattr(runner,'EXPORT_SHA',runner.sha(export/'export.json'))
    model=dict(file='candidate.onnx',sha256=runner.sha(candidate/'candidate.onnx'),bytes=15,
        source_onnx_sha256=original['sha256'],inputs=contracts(),outputs=output_contracts(),role='synthetic_debug_taps')
    manifest=dict(schema='v29_DPVO_Update_candidate_manifest',source_export_sha256=runner.EXPORT_SHA,
                  source_capture_sha256=runner.CAPTURE_SHA,models={'E864':model})
    if mutation=='export_parent':manifest['source_export_sha256']='wrong'
    if mutation=='capture_parent':manifest['source_capture_sha256']='wrong'
    if mutation=='graph_parent':model['source_onnx_sha256']='wrong'
    if mutation=='core_name':model['outputs'][0]['name']='renamed'
    if mutation=='tap_type':model['outputs'][3]['dtype']=7
    if mutation=='tap_shape':model['outputs'][3]['shape'][0]=-1
    if mutation=='graph_bytes':model['bytes']+=1
    if mutation=='filename':model['file']='../candidate.onnx'
    path=candidate/'manifest.json';path.write_text(json.dumps(manifest));pins=runner.Inputs()
    args=NS(capture_dir=capture,export_dir=export,manifest=path,manifest_sha256=runner.sha(path))
    if mutation=='none':
        runner.load_chain(args,pins)
        assert pins.unchanged()
        (candidate/'candidate.onnx').write_bytes(b'changed after pin')
        assert not pins.unchanged()
    else:
        with pytest.raises(ValueError):runner.load_chain(args,pins)


@pytest.mark.parametrize('fault',[None,'provider','output_names','type','count'])
def test_CPU_ORT_metadata_and_fixed_saved_state_coverage(tmp_path,monkeypatch,abi,fault):
    class Session:
        def __init__(self,*unused,**kwargs):assert kwargs['providers']==['CPUExecutionProvider']
        def get_providers(self):return ['CUDAExecutionProvider'] if fault=='provider' else ['CPUExecutionProvider']
        def get_inputs(self):return [NS(name=x['name'],shape=x['shape'],type='tensor(int64)' if x['dtype']==7 else 'tensor(float)') for x in contracts()]
        def get_outputs(self):
            rows=[NS(name=x['name'],shape=x['shape'],type='tensor(float)') for x in output_contracts()]
            if fault=='output_names':rows[3].name='wrong'
            if fault=='type':rows[3].type='tensor(int64)'
            return rows
        def run(self,names,prepared):
            assert names==list(values()) and set(prepared)==set(feed())
            assert all(x.flags.c_contiguous and x.flags.owndata for x in prepared.values())
            result=list(values().values());return result[:-1] if fault=='count' else result
    stub=NS(SessionOptions=lambda:NS(),InferenceSession=Session,
            GraphOptimizationLevel=NS(ORT_DISABLE_ALL=0),__version__='synthetic_no_real_ORT')
    monkeypatch.setitem(sys.modules,'onnxruntime',stub)
    bound=packing.validate_input_attrs(contracts(),attrs(contracts()))
    monkeypatch.setattr(runner,'load_feed',lambda *args:(feed(),golden(),golden(),bound))
    model=dict(file='candidate.onnx',inputs=contracts(),outputs=output_contracts())
    selected=[state(i,name) for name in ('three_frame_all_pairs_identity','three_frame_all_pairs_shifted') for i in range(4)]
    args=NS(manifest=tmp_path/'manifest.json');report=dict(calls=[],cleanup_errors=[])
    if fault:
        with pytest.raises(ValueError):runner.run_ort(args,runner.Inputs(),dict(models={'E864':model}),selected,{},packing,abi,tmp_path,report)
        assert report['calls']==[]
    else:
        runner.run_ort(args,runner.Inputs(),dict(models={'E864':model}),selected,{},packing,abi,tmp_path,report)
        assert len(report['calls'])==8 and all(x['core_strict_FP32_gate_passed'] for x in report['calls'])
        assert report['providers']==['CPUExecutionProvider']


def sdk_setup(tmp_path,monkeypatch,abi,fault=None,mode='sdk-normal',first=False):
    model=dict(inputs=contracts(),outputs=output_contracts(),sha256='synthetic-onnx')
    out=tmp_path/'out';out.mkdir();ort=tmp_path/'ort';ort.mkdir();compiled=tmp_path/'compile';compiled.mkdir()
    states=[state(i,name) for name in ('three_frame_all_pairs_identity','three_frame_all_pairs_shifted') for i in range(4)]
    ort_calls=[runner.save_call(ort,i,s,values(),model,golden(),golden(),{}) for i,s in enumerate(states)]
    receipt=dict(schema='v29_Update_candidate_board_diagnostic',mode='ort',manifest_sha256='test-manifest',
        capture_sha256=runner.CAPTURE_SHA,export_sha256=runner.EXPORT_SHA,execution_status='completed',
        sources_and_inputs_unchanged=True,source_pins={},source_hashes_after={},cleanup_errors=[],candidate_accepted=True,calls=ort_calls)
    (ort/'run.json').write_text(json.dumps(receipt,allow_nan=False))
    (compiled/'block.rknn').write_bytes(b'synthetic graph');compiler=tmp_path/'compiler.py';compiler.write_text('# synthetic compiler pin\n')
    config=dict(target_platform='rk3588',float_dtype='float16',optimization_level=3,do_quantization=False)
    (compiled/'compile.json').write_text(json.dumps(dict(status='compiled_not_runtime_validated',script_sha256=runner.sha(compiler),
        input=dict(sha256=model['sha256']),released=True,configuration=config,
        operations=[dict(name=k,code=0) for k in ('config','load_onnx','build','export_rknn')],
        output=dict(path='block.rknn',size=15,sha256=runner.sha(compiled/'block.rknn')))))
    header=tmp_path/'header';header.write_bytes(b'fake ABI header');library=tmp_path/'library';library.write_bytes(b'fake SDK library')
    monkeypatch.setattr(runner,'HEADER_SHA',runner.sha(header));monkeypatch.setattr(runner,'LIB_SHA',runner.sha(library))
    args=NS(model_key='E864',compile_dir=compiled,compile_sha256=runner.sha(compiled/'compile.json'),
        compiler_helper=compiler,compiler_sha256=runner.sha(compiler),ort_dir=ort,ort_sha256=runner.sha(ort/'run.json'),
        manifest_sha256='test-manifest',allow_unvalidated_ort_diagnostic=False,header=header,library=library,mode=mode,first_only=first)
    sdk=SDK(abi,fault);monkeypatch.setattr(runner.ct,'CDLL',lambda path:None);monkeypatch.setattr(abi,'bind_api',lambda _:sdk)
    def query(api,context,command,value):
        if command==5:return NS(api_version=b'2.3.2',drv_version=b'fake')
        if command==0:return NS(n_input=13,n_output=4)
        if command==1:return attrs(contracts())[value.index]
        if command==2:return attrs(output_contracts())[value.index]
        if command==6:return NS(total_dma_allocated_size=1024,total_weight_size=512,total_internal_size=256)
        raise AssertionError(command)
    monkeypatch.setattr(abi,'query',query);monkeypatch.setattr(abi,'attr_record',lambda record:record)
    monkeypatch.setattr(abi,'perf_detail',lambda *args:dict(synthetic=True,text='no actual NPU'))
    monkeypatch.setattr(runner,'load_feed',lambda *args:(feed(),golden(),golden(),packing.validate_input_attrs(contracts(),attrs(contracts()))))
    report=dict(calls=[],cleanup_errors=[]);return args,runner.Inputs(),dict(models={'E864':model}),states[:1] if first else states,packing,abi,out,report,sdk


@pytest.mark.parametrize('mode,first,count',[('sdk-normal',False,11),('sdk-normal',True,1),('sdk-profile',False,1)])
def test_context_schedule_owned_taps_candidateORT_and_cleanup(tmp_path,monkeypatch,abi,mode,first,count):
    args,pins,manifest,states,p,abi,out,report,sdk=sdk_setup(tmp_path,monkeypatch,abi,mode=mode,first=first)
    runner.run_sdk(args,pins,manifest,states,p,abi,out,report)
    assert len(report['calls'])==sdk.runs==sdk.releases==count and sdk.destroyed==1
    assert report['cleanup_errors']==[] and pins.unchanged()
    assert all(x['candidate_ORT_strict_FP32_gate_passed'] and x['repeat_bitwise_equal'] for x in report['calls'])
    assert all(tuple(x['versus_candidate_ORT'])==tuple(values()) for x in report['calls'])
    assert bool(report['calls'][0].get('profile'))==(mode=='sdk-profile')
    assert report['native_log']['errors']==[]


@pytest.mark.parametrize('fault,raises,rows',[('init',True,0),('run',True,0),('native_log',True,1),
                                           ('destroy',False,1),('nan',False,1),('negative',False,1)])
def test_sdk_failure_or_invalid_output_destroys_context(tmp_path,monkeypatch,abi,fault,raises,rows):
    args,pins,manifest,states,p,abi,out,report,sdk=sdk_setup(tmp_path,monkeypatch,abi,fault,first=True)
    if raises:
        with pytest.raises((ValueError,RuntimeError)):runner.run_sdk(args,pins,manifest,states,p,abi,out,report)
    else:runner.run_sdk(args,pins,manifest,states,p,abi,out,report)
    assert sdk.destroyed==1 and len(report['calls'])==rows
    assert bool(report['cleanup_errors'])==(fault=='destroy')
    if fault=='nan':assert not report['calls'][0]['all_outputs_finite']
    if fault=='negative':assert not report['calls'][0]['weight_domain_passed']
    json.dumps(report,allow_nan=False)


def test_host_guard_precedes_output_creation_and_import(tmp_path,monkeypatch):
    monkeypatch.setattr(runner.platform,'machine',lambda:'x86_64')
    out=tmp_path/'never-created'
    args=['--mode','ort','--capture-dir',str(tmp_path),'--export-dir',str(tmp_path),
          '--manifest',str(tmp_path/'none'),'--manifest-sha256','none','--output-dir',str(out)]
    with pytest.raises(ValueError,match='board only'):runner.main(args)
    assert not out.exists()
