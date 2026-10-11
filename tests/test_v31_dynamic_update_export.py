"""Tiny graph/guard tests only; no checkpoint or captured inputs are loaded."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest


HERE=Path(__file__).resolve().parent
ENTRY=HERE/'run-v31-dpvo-update-export-board.py'
if not ENTRY.exists():ENTRY=HERE.parent/'deploy/rk3588/validation/dpvo_dynamic_update/run-v31-dpvo-update-export-board.py'
SPEC=importlib.util.spec_from_file_location('v31_export_driver',ENTRY)
D=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(D)


def contract_model():
    onnx=pytest.importorskip('onnx')
    def value(name):
        dtype=7 if name.endswith('_indices') or name.endswith('_inverse') else 1
        return onnx.helper.make_tensor_value_info(name,dtype,D.EXPECTED_SHAPES[name])
    graph=onnx.helper.make_graph([], 'contracts', [value(n) for n in D.INPUT_NAMES],
        [value(n) for n in D.OUTPUT_NAMES])
    return onnx.helper.make_model(graph)


def test_all_required_dynamic_axes_and_fixed_geometry():
    inputs,outputs=D.check_contracts(contract_model())
    for item in inputs+outputs:
        symbols={i:d for i,d in enumerate(item['shape']) if isinstance(d,str)}
        assert symbols==D.DYNAMIC_AXES[item['name']]
    assert {s for v in D.DYNAMIC_AXES.values() for s in v.values()}=={'E','Gkk','Lkk','Gij','Lij'}
    assert len(inputs)==13 and len(outputs)==3


@pytest.mark.parametrize('bad', ['fixedE','fixedG','fixedL','wrongSymbol','B','C','dtype','missingInput','outputE'])
def test_wrong_contract_fails(bad):
    m=contract_model()
    if bad=='missingInput':m.graph.input.pop()
    elif bad=='dtype':m.graph.input[3].type.tensor_type.elem_type=1
    else:
        dims=m.graph.output[2].type.tensor_type.shape.dim if bad=='outputE' else m.graph.input[
            7 if bad in ('fixedG','fixedL') else 0].type.tensor_type.shape.dim
        axis=0 if bad in ('fixedG','B') else 2 if bad=='C' else 1
        if bad=='wrongSymbol':dims[axis].dim_param='fixed_from_trace'
        else:dims[axis].dim_value=96 if bad in ('fixedE','outputE') else 3
    with pytest.raises(ValueError):D.check_contracts(m)


def test_guard_before_runtime_or_output(tmp_path,monkeypatch):
    monkeypatch.setattr(D.platform,'machine',lambda:'x86_64')
    out=tmp_path/'must_not_exist'
    names=('v26-source-dir','capture-dir','checkpoint','adapter','sigmoid-rewriter')
    argv=[s for name in names for s in ('--'+name,str(tmp_path/'absent'))]+['--output-dir',str(out)]
    with pytest.raises(ValueError,match='board-only'):D.main(argv)
    assert not out.exists()


def test_signed_zero_is_not_bitwise_equal():
    check=D.compare(np.array([0.],np.float32),np.array([-0.],np.float32))
    assert check['gate_passed'] and not check['bitwise_equal']


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name]=m;spec.loader.exec_module(m)
    return m


def test_tiny_torch_export_uses_dynamic_E_groups_and_changed_topology(tmp_path,monkeypatch):
    """Small random toy Update using frozen adapter; no official model weights."""
    torch=pytest.importorskip('torch');onnx=pytest.importorskip('onnx')
    adapter_path=ENTRY.parent.parent/'dpvo_update_export.py'
    rewrite_path=ENTRY.parent.parent/'rewrite-v29-weight-sigmoid.py'
    # In a public checkout these frozen dependencies live one directory above the new driver.
    if not adapter_path.exists():adapter_path=ENTRY.with_name('dpvo_update_export.py')
    if not rewrite_path.exists():rewrite_path=ENTRY.with_name('rewrite-v29-weight-sigmoid.py')
    assert D.sha(adapter_path)==D.ADAPTER_SHA and D.sha(rewrite_path)==D.REWRITER_SHA
    adapter=load(adapter_path,'_v31_tiny_adapter');rewriter=load(rewrite_path,'_v31_tiny_sigmoid')
    nn=torch.nn
    class Agg(nn.Module):
        def __init__(self):
            super().__init__();self.expand=True
            self.g=nn.Linear(4,4);self.f=nn.Linear(4,4);self.h=nn.Linear(4,4)
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__();self.norm=nn.LayerNorm(4,eps=1e-3);self.corr=nn.Sequential(nn.Linear(6,4),nn.ReLU())
            self.c1=nn.Sequential(nn.Linear(4,4),nn.ReLU(),nn.Linear(4,4));self.c2=copy.deepcopy(self.c1)
            self.agg_kk=Agg();self.agg_ij=Agg();self.gru=nn.Sequential(nn.Linear(4,4),nn.Tanh())
            self.d=nn.Linear(4,2);self.w=nn.Sequential(nn.Linear(4,2),nn.Sigmoid())
    torch.manual_seed(1931)
    wrapped=adapter.ExportableUpdate(Tiny().eval())
    shapes=copy.deepcopy(D.EXPECTED_SHAPES)
    for name in ('net','inp','net_out'):shapes[name][2]=4
    shapes['corr'][2]=6
    monkeypatch.setattr(D,'EXPECTED_SHAPES',shapes)
    def args(ii,jj,kk,caps):
        e=len(ii);topology=adapter.build_topology(np.array(ii,np.int64),np.array(jj,np.int64),
            np.array(kk,np.int64),group_caps=caps)
        values=(torch.randn(1,e,4),torch.randn(1,e,4),torch.randn(1,e,6),*topology.tensors())
        adapter.validate_inputs(wrapped,values)
        return values
    first=args([0,0,1],[0,1,0],[0,0,1],{'kk':(3,3),'ij':(4,2)})
    path=tmp_path/'tiny-dynamic.onnx'
    with torch.inference_mode():
        D.export_graph(torch,wrapped,first,path)
    raw=onnx.load(path);onnx.checker.check_model(raw,full_check=True)
    D.check_contracts(raw)
    final=rewriter.rewrite(raw);D.check_rewrite(raw,final);D.check_contracts(final)
    from onnx.reference import ReferenceEvaluator
    evaluator=ReferenceEvaluator(final)
    # Change E, both G and both L, and the membership/neighbor values; complete keys remain.
    tests=[first,args([0,0,1,1,2],[0,1,0,1,2],[0,0,1,1,2],{'kk':(5,4),'ij':(6,3)}),
        args([2,0,1,0],[1,0,1,2],[2,0,1,0],{'kk':(4,3),'ij':(5,2)})]
    for values in tests:
        with torch.inference_mode():expected=[v.detach().numpy() for v in wrapped(*values)]
        actual=evaluator.run(list(D.OUTPUT_NAMES),{name:value.detach().numpy() for name,value in zip(D.INPUT_NAMES,values)})
        assert all(D.compare(a,b)['gate_passed'] for a,b in zip(actual,expected))
        assert np.all((actual[2]>=0)&(actual[2]<=1))
