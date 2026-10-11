"""Tiny synthetic shape, source-input and policy tests; no original forward."""
import copy
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
import pytest

HERE = Path(__file__).resolve().parent
BASE = Path(os.environ.get("DPVO_HALF_ENCODER_HELPERS", HERE if (HERE/"rewrite-v39-dpvo-half-encoder.py").is_file()
                           else HERE.parent/"deploy/rk3588/validation/dpvo_encoder_halfres"))


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, BASE/filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


editor = load("_v39_editor_test", "rewrite-v39-dpvo-half-encoder.py")
board = load("_v39_board_test", "run-v39-half-encoder-board.py")


def toy(method):
    """A tiny-weight graph with the declared fixed shape contracts, not DPVO."""
    nodes, init = [], []
    def constant(name, value, out):
        nodes.append(h.make_node("Constant", [], [out], name=name,
                                value=nh.from_array(np.asarray(value, np.int64))))
    constant("/original/fnet/Constant", [1,3,464,608], "in_shape")
    nodes.append(h.make_node("Reshape", ["normalized_bgr","in_shape"], ["x"], name="/original/fnet/Reshape"))
    nodes.append(h.make_node("AveragePool", ["x"], ["pooled"], name="pool", kernel_shape=[4,4], strides=[4,4]))
    for branch, channels in (("fnet",128),("inet",384)):
        init.append(nh.from_array(np.ones((channels,3,1,1),np.float32), name=branch+"_weights"))
        nodes.append(h.make_node("Conv", ["pooled",branch+"_weights"], [branch+"_conv"], name=branch+"_conv"))
    constant("/original/fnet/Constant_1", [1,1,128,116,152], "f_shape")
    constant("/original/inet/Constant", [1,1,384,116,152], "i_shape")
    init.append(nh.from_array(np.asarray([1],np.int64),name="squeeze_axes"))
    for branch, shape, reshape_name, output in (("fnet","f_shape","/original/fnet/Reshape_1","fmap"),
                                              ("inet","i_shape","/original/inet/Reshape","imap")):
        nodes.append(h.make_node("Reshape",[branch+"_conv",shape],[branch+"_five"],name=reshape_name))
        nodes.append(h.make_node("Squeeze",[branch+"_five","squeeze_axes"],[output],name=branch+"_squeeze"))
    inputs=[h.make_tensor_value_info("normalized_bgr",T.FLOAT,[1,3,464,608])]
    outputs=[h.make_tensor_value_info("fmap",T.FLOAT,[1,128,116,152])]
    if method=="reciprocal":
        inputs.append(h.make_tensor_value_info("patch_indices",T.INT64,[96]))
        init.append(nh.from_array(np.asarray([1,384,17632],np.int64),name="v33_sparse_context/flat_shape"))
        nodes.extend([h.make_node("Reshape",["imap","v33_sparse_context/flat_shape"],["flat"],name="flatten"),
                      h.make_node("Gather",["flat","patch_indices"],["selected"],name="gather",axis=2),
                      h.make_node("Transpose",["selected"],["context"],name="transpose",perm=[0,2,1])])
        outputs.append(h.make_tensor_value_info("context",T.FLOAT,[1,96,384]))
    else:
        outputs.append(h.make_tensor_value_info("imap",T.FLOAT,[1,384,116,152]))
    model=h.make_model(h.make_graph(nodes,"synthetic",inputs,outputs,init),opset_imports=[h.make_opsetid("",17)])
    model.ir_version=8
    return model


@pytest.mark.parametrize("method",["original","reciprocal"])
def test_editor_only_declared_shape_changes(method):
    parent=toy(method)
    before=parent.SerializeToString()
    new,proof=editor.half_graph(parent,method)
    assert parent.SerializeToString()==before
    assert editor.validate_preservation(parent,new,method)==proof
    assert proof["all_learned_and_other_numeric_values_protobuf_byte_exact"]
    assert editor.contract(new.graph.input)[0]["shape"]==[1,3,232,304]
    assert editor.contract(new.graph.output)[-1]["shape"]==([1,16,384] if method=="reciprocal" else [1,384,58,76])
    assert len(proof["shape_constant_changes"])==(4 if method=="reciprocal" else 3)


@pytest.mark.parametrize("mutation",["weight","constant_output","constant_doc","flat_doc","numeric_node","header","output_name"])
def test_editor_rejects_unlisted_mutations(mutation):
    parent=toy("reciprocal");new,_=editor.half_graph(parent,"reciprocal")
    if mutation=="weight":new.graph.initializer[0].raw_data=b"\0"*len(new.graph.initializer[0].raw_data)
    elif mutation=="constant_output":new.graph.node[0].output[0]="changed"
    elif mutation=="constant_doc":new.graph.node[0].doc_string="changed"
    elif mutation=="flat_doc":new.graph.initializer[-1].doc_string="changed"
    elif mutation=="numeric_node":new.graph.node[2].attribute[0].ints[0]=3
    elif mutation=="header":new.producer_name="changed"
    else:new.graph.output[0].name="changed"
    with pytest.raises(ValueError):editor.validate_preservation(parent,new,"reciprocal")


def test_editor_rejects_wrong_constant_consumer():
    parent=toy("original");parent.graph.node[1].input[1]="wrong"
    with pytest.raises(ValueError,match="Reshape shape"):editor.half_graph(parent,"original")


@pytest.mark.parametrize("machine,system,compatible",[("x86_64","Linux",b"rockchip,rk3588\0"),
    ("aarch64","Darwin",b"rockchip,rk3588\0"),("aarch64","Linux",b"other,board\0"),
    ("aarch64","Linux",b"rockchip,rk3588s-invalid\0")])
def test_board_guard_rejects_nonboard(machine,system,compatible):
    with pytest.raises(ValueError,match="board only"):board.board_guard(machine,system,compatible)


def test_board_guard_accepts_exact_tokens_only():
    board.board_guard("aarch64","Linux",b"test,board\0rockchip,rk3588s\0")


def test_half_resize_fixed_shape_and_interpolation():
    calls=[]
    def resize(value, size, interpolation):
        calls.append((value.shape,size,interpolation))
        return np.zeros((232,304,3),np.uint8)
    cv=SimpleNamespace(INTER_AREA=3,resize=resize)
    board.half_bgr(np.zeros((464,608,3),np.uint8),cv)
    assert calls==[((464,608,3),(304,232),3)]
    with pytest.raises(ValueError):board.half_bgr(np.zeros((464,608,3),np.float32),cv)


def test_calibration_schedule_uses_original_order_and_disjoint_set():
    rows=board.rgb_rows("\n".join(f"{1000+i/30:.9f} rgb/{i}.png" for i in range(798)))
    selected=board.calibration_selection(rows)
    assert [r["source_index"] for r in selected]==[256+16*i for i in range(32)]
    assert not {r["source_index"] for r in selected}.intersection(range(192))
    with pytest.raises(ValueError):board.rgb_rows("1000 rgb/../bad.png\n")


def test_rng_original_x_then_y_then_depth_and_continuity():
    import torch
    def random_centers(width,height,count,generator):
        x=torch.randint(1,width-1,(1,count),generator=generator)
        y=torch.randint(1,height-1,(1,count),generator=generator)
        return torch.stack((x,y),dim=-1)[0].numpy().copy()
    frame=SimpleNamespace(random_centers=random_centers)
    actual=torch.Generator().manual_seed(20261011)
    first=board.sample(frame,actual,torch);second=board.sample(frame,actual,torch)
    reference=torch.Generator().manual_seed(20261011)
    x=torch.randint(1,75,(1,16),generator=reference);y=torch.randint(1,57,(1,16),generator=reference)
    depth=torch.rand((16,),generator=reference,dtype=torch.float32)
    assert np.array_equal(first["centers_xy"],torch.stack((x,y),dim=-1)[0].numpy())
    assert np.array_equal(first["consumed_inverse_depth"],depth.numpy())
    assert np.array_equal(first["state_after_depth"],reference.get_state().numpy())
    assert np.array_equal(first["state_after_depth"],second["state_before"])
    assert np.array_equal(first["patch_indices"],first["centers_xy"][:,1]*76+first["centers_xy"][:,0])


def test_gather_keeps_order_and_duplicate_centers():
    dense=np.arange(384*58*76,dtype=np.float32).reshape(1,384,58,76)
    indices=np.asarray([77,152,77,4407,0,81,500,900]*2,np.int64)
    actual=board.gather_context(dense,indices)
    reference=np.stack([dense[0,:,v//76,v%76] for v in indices])[None]
    assert actual.shape==(1,16,384) and actual.tobytes()==reference.tobytes()
    with pytest.raises(ValueError):board.gather_context(dense,indices.astype(np.float32))


def test_exact_bytes_distinguish_signed_zero_without_relaxing_gate():
    a=np.asarray([0.,1.,2.],np.float32);b=a.copy();b[0]=-0.
    result=board.compare(a,b)
    assert result["gate_passed"] and not result["bitwise_equal"]
    b[2]=2.001
    assert board.compare(a,b)["failed_elements"]==1
    b[1]=np.inf
    with pytest.raises(ValueError):board.compare(a,b)


def test_halfpixel_intrinsics_crop_then_resize():
    K=np.asarray([[517.3,0,318.6],[0,516.5,255.3],[0,0,1]])
    C=np.asarray([[1,0,-16],[0,1,-8],[0,0,1]])
    A=np.asarray([[.5,0,-.25],[0,.5,-.25],[0,0,1]])
    mapped=A@C@K
    assert np.allclose([mapped[0,0],mapped[1,1],mapped[0,2],mapped[1,2]],[258.65,258.25,151.05,123.4],rtol=0,atol=1e-12)
    assert (58//4,76//4)==(14,19)
