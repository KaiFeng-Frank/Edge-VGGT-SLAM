"""Tiny synthetic graph only; no checkpoint or captured arrays."""
import importlib.util
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import pytest
ROOT=Path(__file__).parent
if not (ROOT/'rewrite-v29-weight-sigmoid.py').is_file():ROOT=ROOT.parent/'deploy/rk3588/validation'
spec=importlib.util.spec_from_file_location('_v29_sigmoid',ROOT/'rewrite-v29-weight-sigmoid.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def graph():
    h=onnx.helper
    return h.make_model(h.make_graph([h.make_node('Sigmoid',['x'],['weight'],name='final')],'tiny',
        [h.make_tensor_value_info('x',1,[1,2,256])],[h.make_tensor_value_info('weight',1,[1,2,256])]),opset_imports=[h.make_opsetid('',17)],ir_version=9)
@pytest.mark.parametrize('opt',[ort.GraphOptimizationLevel.ORT_DISABLE_ALL,ort.GraphOptimizationLevel.ORT_ENABLE_ALL])
def test_domain_and_math_for_finite_extremes(opt):
    x=np.concatenate([np.linspace(-100,100,500,dtype=np.float32),np.array([-3.4e38,3.4e38,-1000,1000,-0.,0.,-1e-30,1e-30,-16,16,-80,80],np.float32)]).reshape(1,2,256)
    original=graph();before=original.SerializeToString();candidate=mod.rewrite(original)
    o=ort.SessionOptions();o.graph_optimization_level=opt;o.intra_op_num_threads=1;o.inter_op_num_threads=1
    actual=ort.InferenceSession(candidate.SerializeToString(),sess_options=o,providers=['CPUExecutionProvider']).run(None,{'x':x})[0]
    z=x.astype(np.float64);gold=np.exp(np.minimum(z,0))/(1+np.exp(-np.abs(z)))
    assert np.isfinite(actual).all() and ((actual>=0)&(actual<=1)).all()
    np.testing.assert_allclose(actual,gold,rtol=1e-4,atol=1e-5)
    assert original.SerializeToString()==before
    assert len(candidate.graph.node)==9 and not any(n.op_type=='Sigmoid' for n in candidate.graph.node)
def test_reject_rewrite_twice():
    with pytest.raises(ValueError):mod.rewrite(mod.rewrite(graph()))
def test_reject_missing_or_ambiguous_terminal():
    g=graph();g.graph.node[0].output[0]='other'
    with pytest.raises(ValueError):mod.rewrite(g)
