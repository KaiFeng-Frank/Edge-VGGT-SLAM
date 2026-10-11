"""Tiny fake ctypes SDK only; no native library, board or model is executed."""
import ctypes as ct
import importlib.util
from pathlib import Path

import pytest

HERE=Path(__file__).resolve().parent


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


query=load(HERE/'query-v42-native-inputs-board.py','_v42_fake_native_query')
abi=load(HERE.parent/'run-v12-rknn-block-board.py','_v42_fake_query_ABI')


class FakeAPI:
    def __init__(self,failed_attr=None,destroy_code=0,init_code=0):
        self.calls=[];self.failed_attr=failed_attr;self.destroy_code=destroy_code;self.init_code=init_code

    def rknn_init(self,context,buffer,size,flags,extra):
        self.calls.append(('init',size,flags))
        ct.cast(context,ct.POINTER(abi.CONTEXT))[0]=123
        return self.init_code

    def rknn_set_core_mask(self,context,mask):
        self.calls.append(('core',mask));return 0

    def rknn_query(self,context,command,value,size):
        self.calls.append(('query',command))
        if command==5:
            version=ct.cast(value,ct.POINTER(abi.SDKVersion)).contents
            version.api_version=b'2.3.2 fake';version.drv_version=b'fake'
        elif command==0:
            counts=ct.cast(value,ct.POINTER(abi.InputOutputNum)).contents
            counts.n_input=counts.n_output=3
        elif command in (1,8,10):
            attr=ct.cast(value,ct.POINTER(abi.TensorAttr)).contents
            if self.failed_attr==(command,attr.index):return -6
            attr.n_dims=5 if command==8 else 4
            dims=(1,56,3,16,16) if command==8 else (1,3,16,882)
            for i,v in enumerate(dims):attr.dims[i]=v
            attr.name=b'fake_correlated_input';attr.n_elems=2688;attr.size=5376
            attr.type=1;attr.fmt=2 if command==8 else 1;attr.qnt_type=2
            attr.scale=1;attr.w_stride=16;attr.h_stride=3;attr.size_with_stride=5376;attr.pass_through=1
        elif command==6:
            memory=ct.cast(value,ct.POINTER(abi.MemSize)).contents
            memory.total_dma_allocated_size=4096
        else:pytest.fail('Only declared query commands allowed')
        return 0

    def rknn_destroy(self,context):
        self.calls.append(('destroy',));return self.destroy_code


def test_query_all_commands_full_attrs_and_immediate_destroy():
    api=FakeAPI();row={};held=[]
    query.probe_context(api,abi,b'fake payload',row,held)
    assert api.calls[0]==('init',12,0) and api.calls[1]==('core',7)
    assert [v[1] for v in api.calls if v[0]=='query']==[5,0,1,1,1,8,8,8,10,10,10,6]
    assert api.calls[-1]==('destroy',) and len(row['queries'])==9
    assert row['status']=='queried' and row['context_destroyed'] is True and not held
    assert row['sdk_run_calls']==0 and row['model_inference_executed'] is False and row['NPU_used'] is False
    a=row['queries'][3]['attribute']
    assert a['n_dims']==5 and a['shape']==[1,56,3,16,16] and len(a['dims'])==16
    assert a['fmt']==2 and a['h_stride']==3 and a['w_stride']==16 and a['size_with_stride']==5376 and a['pass_through']==1


def test_unsupported_native_command_retained_no_guessed_logical_native():
    api=FakeAPI(failed_attr=(8,1));row={};held=[]
    query.probe_context(api,abi,b'fake payload',row,held)
    assert row['status']=='queried_with_failed_native_or_logical_commands'
    failed=[r for r in row['queries'] if not r['query_succeeded']]
    assert len(failed)==1 and failed[0]['command']==8 and failed[0]['input_index']==1 and failed[0]['return_code']==-6
    assert row['queries'][0]['kind']=='logical_input' and row['context_destroyed'] is True


def test_partial_initialization_still_destroys_context():
    api=FakeAPI(init_code=-2);row={};held=[]
    query.probe_context(api,abi,b'fake payload',row,held)
    assert api.calls==[('init',12,0),('destroy',)]
    assert row['status']=='failed' and row['context_destroyed'] is True and not held


def test_failed_destroy_retains_host_buffer_and_failure_metadata():
    api=FakeAPI(destroy_code=-2);row={};held=[]
    query.probe_context(api,abi,b'fake payload',row,held)
    assert row['status']=='failed' and row['context_destroyed'] is False
    assert row['cleanup_errors']==['rknn_destroy returned -2']
    assert len(held)==1 and row['model_host_buffer_retained_after_failed_destroy']==13
