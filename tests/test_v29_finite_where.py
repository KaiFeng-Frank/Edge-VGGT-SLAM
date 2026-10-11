"""Small generated finite branch tests, no captured state/model execution."""
import importlib.util
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import pytest
ROOT=Path(__file__).parent
if not (ROOT/'rewrite-v29-finite-where.py').is_file():ROOT=ROOT.parent/'deploy/rk3588/validation'
s=importlib.util.spec_from_file_location('_v29_select_test',ROOT/'rewrite-v29-finite-where.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def graph():
    h=onnx.helper
    return h.make_model(h.make_graph([h.make_node('Where',['mask','x','y'],['out'],name='select')],'tiny',
        [h.make_tensor_value_info('mask',9,[1,2,3,1]),h.make_tensor_value_info('x',1,[1,2,3,4]),h.make_tensor_value_info('y',1,[1,2,1,4])],
        [h.make_tensor_value_info('out',1,[1,2,3,4])]),opset_imports=[h.make_opsetid('',17)],ir_version=9)
@pytest.mark.parametrize('mode',[0,1,2])
def test_broadcast_binary_select(mode):
    rng=np.random.default_rng(3);x=rng.normal(size=(1,2,3,4)).astype(np.float32);y=rng.normal(size=(1,2,1,4)).astype(np.float32)
    x.ravel()[:4]=[np.finfo(np.float32).max,-np.finfo(np.float32).max,0,-0.]
    mask=np.zeros((1,2,3,1),bool) if mode==0 else np.ones((1,2,3,1),bool)
    if mode==2:mask[0,0,2]=False;mask[0,1,1:]=False
    g=graph();before=g.SerializeToString();candidate,removed=m.rewrite(g)
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
    out=ort.InferenceSession(candidate.SerializeToString(),opts,providers=['CPUExecutionProvider']).run(None,{'mask':mask,'x':x,'y':y})[0]
    np.testing.assert_array_equal(out,np.where(mask,x,y));assert np.isfinite(out).all() and removed==['select'] and before==g.SerializeToString()
def test_refuse_inf_constant():
    g=graph();g.graph.initializer.append(onnx.numpy_helper.from_array(np.array(-np.inf,np.float32),name='bad'))
    with pytest.raises(ValueError,match='nonfinite constant'):m.rewrite(g)
