"""Small generated arrays and fake cache only; no model/SDK/board execution."""
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

TEST = Path(__file__).resolve().parent
HERE = (TEST if (TEST/'v41_online_update.py').is_file() else
        TEST.parent/'deploy/rk3588/validation/dpvo_m16_online')
PARENT = Path(os.environ.get('V41_PARENT_SDK', str(HERE.with_name('dpvo_grouped_update'))))


def load(path, label):
    spec = importlib.util.spec_from_file_location(label, path)
    value = importlib.util.module_from_spec(spec); sys.modules[label] = value
    spec.loader.exec_module(value); return value


online = load(HERE/'v41_online_update.py', '_v41_test_online')
C = load(HERE/'v41_online_contracts.py', '_v41_test_contracts')
C.load_m16_parent(HERE/'v40_m16_grouped_contracts.py')
C.load_parent(HERE/'v37_grouped_contracts.py')
adapter = load(PARENT/'source_frame_batches.py', '_v41_test_adapter')
adapter.load_pinned_plan(PARENT/'source_frame_plan.py')
runtime = load(PARENT/'v37_grouped_rknn.py', '_v41_test_original_runtime')


def graph(sources=3, degree=4):
    # A fixed adversarial edge permutation verifies exact fullgraph scatter.
    ii=np.repeat(np.arange(sources,dtype=np.int64),degree*16)
    jj=np.tile(np.repeat(np.arange(degree,dtype=np.int64),16),sources)
    kk=np.concatenate([np.tile(np.arange(i*16,(i+1)*16,dtype=np.int64),degree) for i in range(sources)])
    permutation=np.random.default_rng(2).permutation(ii.size)
    ii,jj,kk=(np.ascontiguousarray(v[permutation]) for v in (ii,jj,kk))
    values=[np.broadcast_to(np.arange(ii.size,dtype=np.float32)[None,:,None]+1,(1,ii.size,c)).copy()
            for c in (384,384,882)]
    return (*values,ii,jj,kk)


class FakeCache:
    def __init__(self, fail_after=None):
        self.specs={key:{} for key in online.MODEL_KEYS}
        self.calls=[];self.closed=False;self.fail_after=fail_after
    def execute(self,key,feed,reference=None,*,timeout_ms):
        self.calls.append(key)
        if len(self.calls)==self.fail_after:
            raise RuntimeError('fake SDK failure')
        shape=feed['net'].shape; b,_,d,m=shape
        output=dict(net_out=feed['net']+np.float32(3),
            delta=np.broadcast_to(feed['inp'][:,:2],(b,2,d,m)).copy(),
            weight=np.full((b,2,d,m),.5,np.float32))
        return output,dict(execution_status='completed',outputs_finite=True,
            weight_domain_valid=True,all_outputs_zero=False,outputs_released=True,
            strict_FP32_gate_passed=None)
    def metrics(self):
        return dict(closed=self.closed,cleanup_errors=[],resident_contexts=0,
            current_DMA_bytes=0,current_model_host_bytes=0,histories=[],
            sessions={k:dict(npu_used=False) for k in set(self.calls)})
    def close(self):self.closed=True


def create(tmp_path,cache=None,probe=None,reader=None,flag=True):
    return online.OnlineM16Update(runtime,adapter,cache or FakeCache(),
        motion_probe=probe or (lambda *a:pytest.fail('Full Update cannot call Torch')),
        diagnostic_allow_fp32_rejected=flag,failure_dir=tmp_path,identity_reader=reader)


def test_full_source_splitting_permutation_owned_outputs_and_close(tmp_path):
    cache=FakeCache();owner=create(tmp_path,cache);inputs=graph()
    output=owner(*inputs)
    assert cache.calls==['B2_D4_M16','B1_D4_M16']
    assert np.array_equal(output['net'],inputs[0]+3)
    assert np.array_equal(output['delta'],inputs[1][:,:,:2])
    assert all(v.flags.owndata for v in output.values())
    assert owner.metrics()['full_completed']==1 and owner.metrics()['shape_histogram']['B1_D4_M16']==1
    assert owner.metrics()['calls'][0]['NPU_used'] is False
    assert owner.metrics()['calls'][0]['SDK_execution']['strict_FP32_gate_passed'] is None
    assert owner.metrics()['candidate_accepted'] is False
    assert 'tensor_sha256' not in owner.metrics()['calls'][0]['inputs']['net']
    state=owner.metrics();state['calls'].clear();assert len(owner.metrics()['calls'])==1
    owner.close();owner.close();assert cache.closed and owner.metrics()['closed']


@pytest.mark.parametrize('degree',(1,3,8))
def test_unknown_full_shape_saves_all_inputs_without_any_execution(tmp_path,degree):
    cache=FakeCache();owner=create(tmp_path,cache)
    inputs=graph(1,degree)
    with pytest.raises(online.OnlineUpdateError,match='Unknown complete online shape'):
        owner(*inputs)
    assert not cache.calls and owner.metrics()['failed_calls']==1
    with np.load(tmp_path/'update-000000-failed-inputs.npz',allow_pickle=False) as archive:
        assert set(archive.files)=={'net','inp','corr','ii','jj','kk'}
        assert np.array_equal(archive['kk'],inputs[-1])
    assert json.loads((tmp_path/'update-000000-failed.json').read_text())['fullgraph_returned'] is False
    with pytest.raises(RuntimeError,match='closed or failed'):owner(*inputs)
    owner.close()


def test_explicit_motion_probe_only_not_E_heuristic(tmp_path):
    calls=[]
    def probe(net,inp,corr,ii,jj,kk):
        calls.append(1);net.fill(9)
        return dict(net=net,delta=np.ones((1,16,2),np.float32),weight=np.full((1,16,2),.5,np.float32))
    owner=create(tmp_path,probe=probe);owner.route='motion_probe'
    inputs=list(graph(1,1));inputs[4].fill(1)
    output=owner(*inputs)
    assert calls==[1] and (output['net']==9).all() and (inputs[0]!=9).any()
    assert owner.metrics()['motion_probe_completed']==1 and owner.metrics()['full_completed']==0
    assert owner.metrics()['calls'][0]['fullgraph_returned'] is False
    owner.close()


def test_late_group_failure_returns_no_fullgraph_and_does_not_mutate(tmp_path):
    cache=FakeCache(fail_after=2);owner=create(tmp_path,cache)
    inputs=graph();saved=[v.copy() for v in inputs]
    with pytest.raises(online.OnlineUpdateError,match='fake SDK failure'):owner(*inputs)
    assert owner.metrics()['full_completed']==0 and len(cache.calls)==2
    assert all(np.array_equal(v,b) for v,b in zip(inputs,saved))
    assert owner.metrics()['calls'][0]['fullgraph_returned'] is False
    owner.close()


def test_stale_revision_after_first_group_does_not_return_arrays(tmp_path):
    cache=FakeCache();token=adapter.SnapshotIdentity(1,2,3,4)
    def reader():
        return token if not cache.calls else adapter.SnapshotIdentity(2,2,3,4)
    owner=create(tmp_path,cache,reader=reader)
    with pytest.raises(online.OnlineUpdateError,match='Stale'):owner(*graph())
    assert len(cache.calls)==1 and owner.metrics()['full_completed']==0
    owner.close()


def test_missing_diagnostic_flag_rejected_before_any_sdk(tmp_path):
    with pytest.raises(ValueError,match='explicit diagnostic'):create(tmp_path,flag=False)
    assert not list(tmp_path.iterdir())


def test_factory_missing_diagnostic_flag_has_no_source_load(tmp_path,monkeypatch):
    factory=load(HERE/'v41_online_factory.py','_v41_test_factory')
    monkeypatch.setattr(factory,'load',lambda *a:pytest.fail('No source import without flag'))
    names=('compile_index','compile_index_sha256','parent_manifest','additions_manifest','additions_manifest_sha256',
        'addition_exporter','addition_exporter_sha256','parent_compiler','addition_compiler','addition_compiler_sha256',
        'contracts','contracts_sha256','m16_contracts','parent_contracts','parent_runtime','runtime_bridge','adapter_path',
        'callback_path','callback_sha256','abi_path','header_path','library_path','log_helper_path','failure_dir')
    with pytest.raises(ValueError,match='Explicit FP32-rejected'):
        factory.open_board_online_from_index(**dict.fromkeys(names,tmp_path),motion_probe=lambda *a:None,
            diagnostic_allow_fp32_rejected=False)


def test_invalid_route_cannot_execute_any_backend(tmp_path):
    cache=FakeCache();owner=create(tmp_path,cache);owner.route='guess'
    with pytest.raises(online.OnlineUpdateError,match='explicitly'):owner(*graph())
    assert not cache.calls and owner.metrics()['full_completed']==0
    owner.close()


def io(b,d):
    return ([dict(name=n,shape=[b,c,d,16],dtype='float32') for n,c in zip(C.INPUT_NAMES,(384,384,882))],
            [dict(name=n,shape=[b,c,d,16],dtype='float32') for n,c in zip(C.OUTPUT_NAMES,(384,2,2))])


def test_exact_online_catalogue_excludes_old_large_batches_and_D1():
    for b in (1,2):
        for d in range(4,8):assert C.model_key(*io(b,d))==f'B{b}_D{d}_M16'
    for b,d in ((3,6),(5,6),(1,1),(1,8)):
        with pytest.raises(ValueError,match='Only complete'):C.contracts(*io(b,d))


@pytest.mark.parametrize('mutant',(None,'missing','lineage','budget'))
def test_compile_index_requires_exact_six_old_two_new(tmp_path,mutant):
    old=dict(models={},strict_FP32_gate_passed=True)
    new=dict(models={},strict_FP32_gate_passed=False);models={}
    for key in C.MODEL_KEYS:
        b,d=int(key[1]),int(key[4]);ins,outs=io(b,d)
        (new if key in C.NEW_KEYS else old)['models'][key]=dict(inputs=ins,outputs=outs)
        models[key]=dict(compile_dir=str(tmp_path/key),compile_sha256='a'*64,
            model=dict(file='block.rknn',bytes=100,sha256='b'*64),lineage='v41' if key in C.NEW_KEYS else 'v40')
    if mutant=='missing':models.pop(C.MODEL_KEYS[0])
    if mutant=='lineage':models[C.NEW_KEYS[0]]['lineage']='v40'
    if mutant=='budget':models[C.NEW_KEYS[0]]['model']['bytes']=256<<20
    index=dict(schema=C.INDEX_SCHEMA,status='staged_eight_online_shapes_diagnostic',
        parent_manifest_sha256=C.EXPORT_MANIFEST_SHA256,additions_manifest_sha256='c'*64,
        candidate_accepted=False,models=models)
    path=tmp_path/'index.json';path.write_text(json.dumps(index))
    kwargs=dict(additions_manifest_sha256='c'*64,parent_compiler='unused',addition_compiler='unused',
        addition_compiler_sha256='d'*64,m16_contracts='unused',online_contracts='unused',
        online_contracts_sha256='e'*64,verify_artifacts=False)
    if mutant:
        with pytest.raises(ValueError):C.validate_compile_index(path,C.sha256(path),old,new,**kwargs)
    else:
        _,specs,compiled,_=C.validate_compile_index(path,C.sha256(path),old,new,**kwargs)
        assert len(specs)==8 and compiled=={} and specs[C.NEW_KEYS[0]]['export_strict_FP32_gate_passed'] is False
