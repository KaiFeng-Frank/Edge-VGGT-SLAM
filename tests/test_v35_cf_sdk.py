import ast
import ctypes as ct
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]/"deploy/rk3588/validation/dpvo_source_frame_channel_first"
spec=importlib.util.spec_from_file_location('cf_sdk',ROOT/'v35_rknn_source_frame_cf.py')
io=importlib.util.module_from_spec(spec);sys.modules[spec.name]=io;spec.loader.exec_module(io)

def contract(d=2,m=3):
    ins=[dict(name=n,dtype='float32',shape=[1,c,d,m]) for n,c in [('net',384),('inp',384),('corr',882)]]
    outs=[dict(name=n,dtype='float32',shape=[1,c,d,m]) for n,c in [('net_out',384),('delta',2),('weight',2)]]
    return ins,outs

def attrs(rows,fmt=0):
    result=[]
    for i,r in enumerate(rows):
        shape=r['shape'];native=[shape[0],shape[2],shape[3],shape[1]] if fmt==1 else shape
        count=int(np.prod(shape))
        result.append(dict(index=i,name=r['name'],shape=native,format=fmt,type=1,quantization_type=2,
            scale=1.0,zero_point=0,fractional_length=0,n_elems=count,size=count*2,size_with_stride=count*2))
    return result

def test_cf_contract_axes():
    ins,outs=contract(10,96);a,b=io.contracts(ins,outs)
    assert a['net']['shape']==(1,384,10,96)
    assert a['corr']['shape']==(1,882,10,96)
    assert b['weight']['shape']==(1,2,10,96)

def test_old_layout_and_same_size_misbindings_rejected():
    ins,outs=contract()
    for r in ins+outs:r['shape']=[r['shape'][0],r['shape'][2],r['shape'][3],r['shape'][1]]
    with pytest.raises(ValueError):io.contracts(ins,outs)
    ins,outs=contract();ins[2]['shape']=[1,3,2,882]
    with pytest.raises(ValueError):io.contracts(ins,outs)

def test_exact_nhwc_channel_mapping_and_owned_packing():
    class Input(ct.Structure):
        _fields_=[('index',ct.c_uint32),('buf',ct.c_void_p),('size',ct.c_uint32),('pass_through',ct.c_uint8),('type',ct.c_uint32),('fmt',ct.c_uint32)]
    class ABI:pass
    ABI.Input=Input
    ins,outs=contract();bindings=io.validate_input_attrs(ins,outs,attrs(ins,1))
    arrays={r['name']:np.arange(np.prod(r['shape']),dtype=np.float32).reshape(r['shape']) for r in ins}
    packed=io.pack_inputs(ABI,arrays,bindings)
    for row,buffer in zip(ins,packed.owned_buffers):
        for c in (0,row['shape'][1]-1):
            for d in range(2):
                for m in range(3):assert buffer[0,d,m,c]==arrays[row['name']][0,c,d,m]
        assert not np.shares_memory(buffer,arrays[row['name']])
        assert buffer.flags.c_contiguous
    before=packed.owned_buffers[0].copy();arrays['net'].fill(-123)
    np.testing.assert_array_equal(packed.owned_buffers[0],before)

def test_output_padding_does_not_change_returned_logical_bytes():
    ins,outs=contract();rows=attrs(outs)
    for r in rows:r['size_with_stride']+=128
    bound=io.validate_output_attrs(ins,outs,rows)
    assert [r['returned_FP32_bytes'] for r in bound]==[int(np.prod(v['shape']))*4 for v in outs]
    rows[0]['size_with_stride']=rows[0]['size']-2
    with pytest.raises(ValueError):io.validate_output_attrs(ins,outs,rows)

def test_input_padding_and_packed_outputs_rejected():
    ins,outs=contract();rows=attrs(ins);rows[0]['size_with_stride']+=32
    with pytest.raises(ValueError):io.validate_input_attrs(ins,outs,rows)
    rows=attrs(outs);rows[0]['format']=2
    with pytest.raises(ValueError):io.validate_output_attrs(ins,outs,rows)

def test_cf_dimension_domain_rejected():
    for d,m in ((0,3),(26,3),(2,0),(2,97)):
        with pytest.raises(ValueError):io.contracts(*contract(d,m))
    ins,outs=contract();outs[1]['shape'][1]=3
    with pytest.raises(ValueError):io.contracts(ins,outs)

def test_transport_implementation_matches_frozen_ancestor():
    parent=ROOT/'v33_rknn_source_frame_padded.py'
    assert io.sha256(parent)==io.PARENT_SHA256
    def nodes(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    a,b=nodes(parent),nodes(ROOT/'v35_rknn_source_frame_cf.py')
    for name in ('_bind','pack_inputs','compare','_native_log','SourceFrameSession'):
        assert a[name]==b[name],name
