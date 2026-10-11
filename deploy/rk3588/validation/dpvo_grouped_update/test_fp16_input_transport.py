"""Generated FP32 arrays only; no weights, compiler, SDK or inference."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

HERE=Path(__file__).parent
R=load(HERE/'v37_grouped_rknn.py','_v38_pack_runtime')
A=load(HERE.parent/'run-v12-rknn-block-board.py','_v38_pack_ABI')
P=load(HERE/'fp16_input_transport.py','_v38_pack_impl')

def fixture():
    feed={};bindings=[]
    for index,(name,channels) in enumerate(zip(R.INPUT_NAMES,(384,384,882))):
        shape=(2,channels,1,48)
        feed[name]=(np.arange(np.prod(shape),dtype=np.float32).reshape(shape)%19-.3)*.2
        bindings.append(dict(index=index,name=name,shape=shape,sdk_type=1,sdk_quantization_type=2,
            sdk_native_bytes=np.prod(shape)*2,sdk_size_with_stride=np.prod(shape)*2,
            transport_bytes=np.prod(shape)*4,input_permutation=(0,2,3,1),sdk_shape=(2,1,48,channels),
            sdk_format=1,transport_format=1))
    return feed,bindings

def test_layout_dtype_size_and_owned_source():
    feed,bindings=fixture();before={name:value.tobytes() for name,value in feed.items()}
    packed=P.make_packer(R)(A,feed,bindings)
    for i,row in enumerate(bindings):
        own=packed.owned_buffers[i]
        assert np.array_equal(own,feed[row['name']].transpose(0,2,3,1).astype(np.float16))
        assert own.flags.c_contiguous and own.flags.owndata and not np.shares_memory(own,feed[row['name']])
        assert packed.requests[i].type==1 and packed.requests[i].fmt==1 and packed.requests[i].pass_through==0
        assert packed.requests[i].size==own.nbytes==row['sdk_native_bytes']
    assert {name:value.tobytes() for name,value in feed.items()}==before

def test_fp16_overflow_rejected_before_submission():
    feed,bindings=fixture();feed['net'].flat[0]=70000
    with pytest.raises(ValueError,match='overflowed FP16'):P.make_packer(R)(A,feed,bindings)

def test_padded_or_non_fp16_input_rejected():
    feed,bindings=fixture();bindings[0]['sdk_size_with_stride']+=64
    with pytest.raises(ValueError,match='Unpadded native FP16'):P.make_packer(R)(A,feed,bindings)
