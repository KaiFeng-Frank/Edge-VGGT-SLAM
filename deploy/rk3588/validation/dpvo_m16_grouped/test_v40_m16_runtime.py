"""Generated buffers, fake C ABI calls and metadata only; no neural execution."""
import ctypes as ct
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest


TEST_DIRECTORY = Path(__file__).resolve().parent
HERE = (TEST_DIRECTORY if (TEST_DIRECTORY/'run-v40-m16-grouped-board.py').exists() else
        TEST_DIRECTORY.parent/'deploy/rk3588/validation/dpvo_m16_grouped')
PUBLIC = Path(os.environ['V40_PUBLIC_PARENT']) if 'V40_PUBLIC_PARENT' in os.environ else None
PARENT = Path(os.environ.get('V40_PARENT_SDK', str(
    PUBLIC/'deploy/rk3588/validation/dpvo_grouped_update' if PUBLIC else HERE.with_name('dpvo_grouped_update'))))
CONTRACT = Path(os.environ.get('V40_M16_CONTRACTS', str(
    HERE/'v40_m16_grouped_contracts.py' if (HERE/'v40_m16_grouped_contracts.py').exists() else
    HERE.parent/'contracts/v40_m16_grouped_contracts.py')))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


bridge = load(HERE/'v40_m16_grouped_rknn.py', '_v40_test_bridge')
runtime = bridge.load_runtime(PARENT/'v37_grouped_rknn.py', CONTRACT,
    bridge.sha256(CONTRACT), PARENT/'v37_grouped_contracts.py')
C = runtime.C
driver = load(HERE/'run-v40-m16-grouped-board.py', '_v40_test_driver')
adapter = load(PARENT/'source_frame_batches.py', '_v40_test_adapter')


def io(b=2, d=4):
    return ([dict(name=name, dtype='float32', shape=[b,c,d,16])
             for name,c in zip(C.INPUT_NAMES,(384,384,882))],
            [dict(name=name, dtype='float32', shape=[b,c,d,16])
             for name,c in zip(C.OUTPUT_NAMES,(384,2,2))])


def attrs(rows, fmt=0, padding=0):
    result = []
    for index,row in enumerate(rows):
        b,c,d,m = row['shape']; n=b*c*d*m
        result.append(dict(index=index,name=row['name'],shape=[b,d,m,c] if fmt==1 else row['shape'],
            format=fmt,type=1,n_elems=n,size=n*2,size_with_stride=n*2+padding,
            quantization_type=2,scale=1.,zero_point=0,fractional_length=0,pass_through=0,w_stride=0,h_stride=0))
    return result


@pytest.fixture
def abi():
    path = Path(os.environ.get('V40_ABI_HELPER',str(
        PUBLIC/'deploy/rk3588/validation/run-v12-rknn-block-board.py' if PUBLIC else
        HERE.parent/'run-v12-rknn-block-board.py')))
    assert bridge.sha256(path)==C.ABI_SHA256
    return load(path,'_v40_test_ABI_definitions')


class FakeSDK:
    def __init__(self, abi, rows, fmt, failure=None):
        self.abi,self.rows,self.fmt,self.failure=abi,rows,fmt,failure
        self.buffers=[];self.runs=self.releases=self.destroys=0
        self.expected={row['name']:np.full(row['shape'],.5 if row['name']=='weight' else i+1,dtype=np.float32)
                       for i,row in enumerate(rows)}
    def rknn_inputs_set(self, context, count, requests):
        assert count==3 and all(requests[i].fmt==1 and requests[i].type==0 and requests[i].pass_through==0 for i in range(3))
        return 0
    def rknn_run(self, context, extension):
        self.runs+=1;ct.cast(extension,ct.POINTER(self.abi.RunExtend)).contents.frame_id=9
        return 0
    def rknn_outputs_get(self, context, count, outputs, received):
        for index,row in enumerate(self.rows):
            raw=self.expected[row['name']]
            if self.fmt==1:raw=raw.transpose(0,2,3,1)
            owned=np.array(raw,copy=True,order='C');self.buffers.append(owned)
            outputs[index].buf,outputs[index].size=owned.ctypes.data,owned.nbytes
            assert outputs[index].want_float==1
            if self.failure=='partial':return -2
        ct.cast(received,ct.POINTER(self.abi.OutputExtend)).contents.frame_id=8 if self.failure=='frame' else 9
        return 0
    def rknn_outputs_release(self, context, count, outputs):
        self.releases+=1
        for value in self.buffers:value.fill(999)
        return 0
    def rknn_destroy(self, context):self.destroys+=1;return 0


@pytest.mark.parametrize('fmt',(0,1,3))
def test_true_B_M16_native_pack_copy_before_release_and_close(abi,fmt):
    inputs,outputs=io();api=FakeSDK(abi,outputs,fmt)
    session=runtime.GroupedSession(abi,api,abi.CONTEXT(3),inputs,outputs,attrs(inputs,1),attrs(outputs,fmt,128))
    feed={row['name']:np.zeros(row['shape'],dtype=np.float32) for row in inputs}
    values,row=session.execute(feed,api.expected)
    assert row['strict_component_gate_passed'] and row['candidate_accepted'] is False
    assert all(v.flags.owndata and np.array_equal(v,api.expected[k]) for k,v in values.items())
    assert all((v==999).all() for v in api.buffers) and api.releases==1
    assert session.metrics()['npu_used'] is False
    session.close();session.close()
    assert api.destroys==1 and session.metrics()['current_DMA_bytes']==0


@pytest.mark.parametrize('failure',('partial','frame'))
def test_partial_output_and_bad_frame_cleanup(abi,failure):
    inputs,outputs=io();api=FakeSDK(abi,outputs,1,failure)
    session=runtime.GroupedSession(abi,api,abi.CONTEXT(3),inputs,outputs,attrs(inputs,1),attrs(outputs,1))
    _,row=session.execute({r['name']:np.zeros(r['shape'],dtype=np.float32) for r in inputs})
    assert row['execution_status']=='failed' and api.releases==api.destroys==1
    assert session.metrics()['closed'] and session.metrics()['npu_used'] is False


def test_numeric_rejection_does_not_become_acceptance(abi):
    inputs,outputs=io(1,5);api=FakeSDK(abi,outputs,0)
    session=runtime.GroupedSession(abi,api,abi.CONTEXT(3),inputs,outputs,attrs(inputs,1),attrs(outputs))
    references={k:v.copy() for k,v in api.expected.items()};references['delta'].flat[0]+=.25
    values,row=session.execute({r['name']:np.zeros(r['shape'],dtype=np.float32) for r in inputs},references)
    assert values and row['execution_status']=='completed' and row['outputs_finite'] and row['weight_domain_valid']
    assert row['strict_FP32_gate_passed'] is False and row['strict_component_gate_passed'] is False
    assert row['candidate_accepted'] is False;session.close()


def test_private_M16_binding_does_not_change_other_parent_instance():
    other=load(PARENT/'v37_grouped_rknn.py','_v40_unmodified_parent_test')
    assert other.C.INPUT_NAMES==runtime.C.INPUT_NAMES
    with pytest.raises(ValueError,match='M48'):other.C.contracts(*io())
    assert runtime.C.contracts(*io())


@pytest.mark.parametrize('profile,nfull,nsdk',((False,9,36),(True,3,12)))
def test_schedule_has_complete_four_groups_per_state(profile,nfull,nsdk):
    rows=driver.schedule(profile)
    assert len(rows)==nfull and 4*len(rows)==nsdk
    assert set(o for o,_,_ in rows)=={34,127,191}


def test_missing_diagnostic_flag_has_no_source_load_or_output_write(monkeypatch,tmp_path):
    monkeypatch.setattr(driver,'load',lambda *a,**k:pytest.fail('No load without explicit diagnostic flag'))
    args=[]
    for name in ('manifest','compile-index','capture-dir','contracts','compiler','parent-runtime','parent-contracts',
                 'runtime-helper','adapter','abi-helper','header','log-helper','output-dir'):
        args.extend(['--'+name,str(tmp_path/name)])
    for name in ('compile-index','contracts','compiler','runtime-helper'):
        args.extend(['--'+name+'-sha256','a'*64])
    with pytest.raises(ValueError,match='before side effects'):driver.main(args)
    assert not (tmp_path/'output-dir').exists()


@pytest.mark.parametrize('mutant',(None,'missing','budget','ancestry'))
def test_compile_index_actual_catalogue_and_budget(tmp_path,mutant):
    manifest=dict(models={});models={}
    for b,d in C.ACTUAL_SHAPES:
        ins,outs=io(b,d);key=f'B{b}_D{d}_M16'
        manifest['models'][key]=dict(inputs=ins,outputs=outs)
        models[key]=dict(compile_dir=str(tmp_path/key),compile_sha256='a'*64,
                        model=dict(file='block.rknn',bytes=20,sha256='b'*64))
    index=dict(schema='v40_M16_grouped_Update_compile_index',status='staged_all_actual_shapes',
        manifest_sha256=driver.MANIFEST_SHA,compiler_sha256='c'*64,contracts_sha256='d'*64,
        candidate_accepted=False,host_model_inference_executed=False,host_calibration_executed=False,models=models)
    if mutant=='missing':models.pop(C.MODEL_KEYS[0])
    if mutant=='budget':models[C.MODEL_KEYS[0]]['model']['bytes']=256<<20
    if mutant=='ancestry':index['compiler_sha256']='e'*64
    path=tmp_path/'index.json';path.write_text(json.dumps(index))
    kwargs=dict(compiler_path='unused',compiler_sha256='c'*64,contracts_path=CONTRACT,contracts_sha256='d'*64,verify_artifacts=False)
    if mutant:
        with pytest.raises(ValueError):driver.validate_compile_index(C,path,C.sha256(path),manifest,**kwargs)
    else:
        _,specs,compiled,_=driver.validate_compile_index(C,path,C.sha256(path),manifest,**kwargs)
        assert len(specs)==8 and not compiled


def toy_graph():
    rows=np.array([(s,t,s*16+p) for s,targets in ((0,range(4)),(1,range(4)),(2,range(5)))
                   for t in targets for p in range(16)],dtype=np.int64)
    rows=rows[np.random.default_rng(3).permutation(len(rows))]
    plan=adapter.build_degree_plan(*(rows[:,i].copy() for i in range(3)),patches_per_frame=16)
    net=np.arange(len(rows)*3,dtype=np.float32).reshape(1,len(rows),3)
    values=dict(net=net,inp=np.ones_like(net),corr=np.zeros((1,len(rows),7),dtype=np.float32))
    return plan,values,adapter.SnapshotIdentity(0,0,0,1)


@pytest.mark.parametrize('failure',(None,'late','unknown'))
def test_original_edge_order_atomic_restore_no_hashes_in_full_wall(monkeypatch,failure):
    plan,inputs,identity=toy_graph();before={k:v.copy() for k,v in inputs.items()};seen=[]
    class Cache:
        def execute(self,key,feed,reference=None,*,timeout_ms):
            assert reference is None;seen.append(key)
            if failure=='unknown' and len(seen)==2:raise ValueError('No compiled context for this actual shape')
            values=dict(net_out=feed['net']+feed['inp'],delta=feed['net'][:,:2].copy(),weight=np.full_like(feed['net'][:,:2],.5))
            return values,dict(execution_status='failed' if failure=='late' and len(seen)==2 else 'completed',
                outputs_finite=True,weight_domain_valid=True,all_outputs_zero=False,candidate_accepted=False)
    monkeypatch.setattr(driver,'raw_metadata',lambda *a:pytest.fail('No output hashing inside full wall'))
    if failure:
        with pytest.raises((runtime.GroupExecutionError,ValueError)) as caught:
            driver.full_call(runtime,adapter,plan,inputs,Cache(),identity)
        assert seen==['B2_D4_M16','B1_D5_M16'] and caught.value.full_wall_seconds>=0
        assert len(caught.value.completed_group_records)==(2 if failure=='late' else 1)
    else:
        outputs,row,groups=driver.full_call(runtime,adapter,plan,inputs,Cache(),identity)
        assert np.array_equal(outputs['net'],inputs['net']+1)
        assert np.array_equal(outputs['delta'],inputs['net'][:,:,:2]) and (outputs['weight']==.5).all()
        assert row['full_graph_returned'] and not row['VO_state_committed'] and not row['candidate_accepted']
        assert len(groups)==2 and row['full_wall_seconds']>=row['full_graph_seconds']>=0
    assert all(np.array_equal(inputs[k],v) for k,v in before.items())


def test_stale_fixed_capture_identity_does_not_return_partial_graph():
    plan,inputs,identity=toy_graph();seen=[]
    def execute(batch,feeds):
        seen.append(batch)
        return dict(net=feeds['net'].copy(),delta=feeds['net'][:,:2].copy(),weight=np.full_like(feeds['net'][:,:2],.5))
    def current():return identity if not seen else adapter.SnapshotIdentity(0,0,0,2)
    with pytest.raises(RuntimeError,match='Stale'):
        adapter.execute_grouped_update(plan,inputs['net'],inputs['inp'],inputs['corr'],execute,
            snapshot_identity=identity,current_identity=current)
    assert len(seen)==1
