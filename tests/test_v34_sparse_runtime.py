"""Tiny fake SDK buffers/attributes only; no model, board, image or ORT run."""
import ctypes as ct
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def entry(name):
    here=Path(__file__).resolve().parent
    for path in [here/name, here.parent/'deploy/rk3588/validation/dpvo_sparse_context_sdk'/name,
                 here.parent/'deploy/rk3588/validation'/name]:
        if path.exists():return path
    if name=='run-v12-rknn-block-board.py' and os.environ.get('RKNN_ABI_HELPER'):
        return Path(os.environ['RKNN_ABI_HELPER'])
    pytest.fail('Required published helper missing: '+name)


def load(name):
    spec=importlib.util.spec_from_file_location(name.replace('-','_'),entry(name))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


H=load('v34_sparse_rknn.py')
D=load('run-v34-dpvo-sparse-context-board.py')


def index_attr(count=96):
    return dict(index=1,name='patch_indices',shape=[count],type=8,n_elems=count,size=count*8,
                format=3,quantization_type=0,zero_point=0,scale=1.0)


def test_actual_INT64_UNDEFINED_and_FP16_rank3_context_contract():
    assert H.validate_index_attr(index_attr())['format']==3
    row=dict(index=1,name='context',shape=[1,96,384],format=3,type=1,n_elems=36864,size=73728,
             quantization_type=0,zero_point=0,scale=1.0)
    assert H.validate_context_attr(row)==row


@pytest.mark.parametrize('change', [{'type':6,'size':384},{'type':0,'size':384},{'format':2},{'shape':[1,96]}])
def test_indices_no_implicit_cast_or_opaque_layout(change):
    row=index_attr();row.update(change)
    with pytest.raises(ValueError):H.validate_index_attr(row)


def test_normal_AB_BA_AB_and_sparse_only_separate_profiles():
    assert [(i,r) for i,r,_ in D.schedule(False)]==[(0,'dense'),(0,'sparse'),(0,'sparse'),(0,'dense'),
        (0,'dense'),(0,'sparse'),(1,'sparse'),(1,'dense'),(2,'dense'),(2,'sparse')]
    assert D.schedule(True)==[(0,'sparse','profile'),(1,'sparse','profile'),(2,'sparse','profile')]


@pytest.mark.parametrize('padding', [False,True])
def test_typed_packing_copies_logical_outputs_before_release_and_rejects_padding(padding):
    abi=load('run-v12-rknn-block-board.py')
    image=np.arange(180,dtype=np.float32).reshape(1,3,12,5)
    # Different image shape needs no model: packing is a fake and the feature-domain
    # guard uses real /4 integer geometry for these tiny inputs.
    image=np.arange(720,dtype=np.float32).reshape(1,3,12,20)
    indices=np.array([6,8,6],np.int64)
    base=SimpleNamespace(pack_input=lambda x,fmt:np.array(x.transpose(0,2,3,1),copy=True,order='C'))
    events,buffers,activity,cleanup=[],[],{},[]
    specs=(('fmap',(1,2,3,5)),('context',(1,3,4)))
    def set_inputs(ctx,n,requests):
        events.append('set');assert n==2
        assert [r.type for r in requests]==[0,8] and [r.fmt for r in requests]==[1,3]
        assert [r.pass_through for r in requests]==[0,0]
        arr=np.ctypeslib.as_array(ct.cast(requests[1].buf,ct.POINTER(ct.c_int64)),shape=(3,))
        assert arr.tobytes()==indices.tobytes() and requests[1].size==24
        return 0
    def run(ctx,ext):events.append('run');ext._obj.frame_id=11;return 0
    def get(ctx,n,outputs,received):
        events.append('get');received._obj.frame_id=11
        for out,(name,shape) in zip(outputs,specs):
            assert out.want_float==1 and out.is_prealloc==0
            arr=np.full(shape,len(buffers)+1.,np.float32);buffers.append(arr)
            out.buf=arr.ctypes.data;out.size=arr.nbytes+(16 if padding and name=='context' else 0)
        return 0
    def release(*a):
        events.append('release')
        for arr in buffers:arr.fill(-999)
        return 0
    api=SimpleNamespace(rknn_inputs_set=set_inputs,rknn_run=run,rknn_outputs_get=get,rknn_outputs_release=release)
    if padding:
        with pytest.raises(ValueError,match='logical byte'):
            H.execute_sparse(api,abi,base,abi.CONTEXT(7),image,indices,1,index_attr(3),cleanup,activity,outputs=specs)
    else:
        actual,timing=H.execute_sparse(api,abi,base,abi.CONTEXT(7),image,indices,1,index_attr(3),cleanup,activity,outputs=specs)
        assert np.all(actual['fmap']==1) and np.all(actual['context']==2)
        assert timing['output_buffers_released'] and timing['sdk_io_seconds']>=sum(timing[k] for k in
            ['input_packing_seconds','input_set_seconds','run_api_seconds','outputs_get_seconds','output_copy_seconds','outputs_release_seconds'])
    assert events==['set','run','get','release'] and activity['sdk_run_api_calls']==1 and cleanup==[]


def test_context_initialization_failure_destroys_allocated_context_once():
    destroyed=[]
    def check(code,name):
        if code:raise RuntimeError(name)
    def init(ptr,*args):ptr._obj.value=31;return -1
    abi=SimpleNamespace(CONTEXT=ct.c_uint64,check_call=check)
    api=SimpleNamespace(rknn_init=init,rknn_destroy=lambda ctx:destroyed.append(ctx.value) or 0)
    record,cleanup={},[]
    with pytest.raises(RuntimeError,match='init'):
        H.Context(None,abi,api,'fake.rknn',True,False,cleanup,record)
    assert destroyed==[31] and record['context_destroyed'] and cleanup==[]


def test_host_guard_before_any_model_entry(monkeypatch):
    monkeypatch.setattr(D.platform,'machine',lambda:'x86_64')
    with pytest.raises(ValueError,match='board'):
        D.board_guard()
