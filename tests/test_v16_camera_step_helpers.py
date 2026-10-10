"""Synthetic 18-value recurrence/transport checks, not original model inference."""
import importlib.util
import ctypes as ct
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / 'deploy/rk3588/validation/export-v16-camera-step-board.py'
spec = importlib.util.spec_from_file_location('v16_step_helpers_test', EXPORT)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def pose(value):
    return np.full((1, 2, 9), value, dtype=np.float32)


def test_four_iterations_feed_unactivated_prediction_and_assign_first_delta():
    seed = pose(99)
    deltas = [pose(-2), pose(1), pose(4), pose(-1)]
    prediction = None
    conditions = []
    for delta in deltas:
        conditions.append(seed.copy() if prediction is None else prediction.copy())
        prediction = helper.accumulate_delta(prediction, delta)
    assert [x[0, 0, 7] for x in conditions] == [99, -2, -1, 3]
    assert np.array_equal(prediction, pose(2))
    assert np.array_equal(seed, pose(99))
    assert np.array_equal(deltas[0], pose(-2))


def test_activation_is_only_relu_fov_and_does_not_mutate_prediction():
    array = np.arange(-9, 9, dtype=np.float32).reshape(1, 2, 9)
    before = array.copy()
    activated = helper.activate_numpy_pose(array)
    assert np.array_equal(activated[..., :7], before[..., :7])
    assert np.array_equal(activated[..., 7:], np.maximum(before[..., 7:], 0))
    assert np.array_equal(array, before)
    assert activated.dtype == np.float32


@pytest.mark.parametrize('bad', [np.zeros((1, 2, 9), dtype=np.float64), np.zeros((2, 9), dtype=np.float32),
                               pose(np.nan), pose(np.inf)])
def test_pose_input_rejects_dtype_shape_and_nonfinite(bad):
    with pytest.raises(ValueError):
        helper.accumulate_delta(None, bad)


def test_fp32_addition_order_is_retained():
    a = pose(2 ** 24)
    result = helper.accumulate_delta(None, a)
    result = helper.accumulate_delta(result, pose(1))
    result = helper.accumulate_delta(result, pose(-(2 ** 24)))
    result = helper.accumulate_delta(result, pose(3))
    assert np.array_equal(result, pose(3))


def test_dynamic_input_dependency_check_requires_both_inputs():
    value = lambda name: SimpleNamespace(name=name)
    node = lambda out, ins: SimpleNamespace(output=[out], input=ins)
    model = SimpleNamespace(graph=SimpleNamespace(input=[value('camera_tokens'), value('conditioning_pose')],
        output=[value('pose_delta')], node=[node('embedding', ['conditioning_pose']),
                                         node('pose_delta', ['embedding', 'camera_tokens'])]))
    assert helper.both_inputs_reach_output(model)
    model.graph.node[0].input = ['constant_seed']
    assert not helper.both_inputs_reach_output(model)


def test_import_does_not_load_torch_or_onnx():
    code = ('import importlib.util,sys; s=importlib.util.spec_from_file_location("check",sys.argv[1]); '
            'm=importlib.util.module_from_spec(s);s.loader.exec_module(m);'
            'assert "torch" not in sys.modules and "onnx" not in sys.modules')
    subprocess.run([sys.executable, '-B', '-c', code, str(EXPORT)], check=True)


RUNTIME = ROOT / 'deploy/rk3588/validation/run-v16-camera-step-board.py'
spec = importlib.util.spec_from_file_location('v16_runtime_helpers_test', RUNTIME)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
abi,_ = runtime.load_frozen('run-v12-rknn-block-board.py',runtime.RUNTIME_SHA,'v16_fake_test_ABI')


def attribute(index,name,shape):
    value = abi.TensorAttr(index=index,n_dims=len(shape),name=name.encode(),n_elems=int(np.prod(shape)),
                           size=int(np.prod(shape))*2,fmt=3,type=1,qnt_type=2,scale=1.,zp=0)
    for i,size in enumerate(shape):
        value.dims[i] = size
    return value


@pytest.mark.parametrize('reverse',[False,True])
def test_actual_sdk_names_bind_both_inputs_in_either_index_order(reverse):
    raw_index = int(reverse)
    inputs = [attribute(raw_index,'camera_tokens',(1,2,2048)),
              attribute(1-raw_index,'conditioning_pose',(1,2,9))]
    before = abi.SHAPE
    binding = runtime.validate_bindings(abi,inputs,attribute(0,'pose_delta',(1,2,9)))
    assert binding['camera_tokens']['index']==raw_index
    assert binding['conditioning_pose']['index']==1-raw_index
    assert abi.SHAPE==before


def test_sdk_binding_rejects_nonidentity_quantization_and_missing_name():
    raw = attribute(0,'camera_tokens',(1,2,2048))
    cond = attribute(1,'conditioning_pose',(1,2,9))
    cond.scale=.5
    with pytest.raises(ValueError,match='identity'):
        runtime.validate_bindings(abi,[raw,cond],attribute(0,'pose_delta',(1,2,9)))
    cond.name=b'wrong_name'
    with pytest.raises(ValueError,match='conditioning'):
        runtime.validate_bindings(abi,[raw,cond],attribute(0,'pose_delta',(1,2,9)))


class FakeSDK:
    """18 synthetic output values; no model or SDK loading."""
    def __init__(self,mode='ok'):
        self.buffer=(ct.c_float*18)(*range(18))
        self.releases=0
        self.mode=mode

    def rknn_inputs_set(self,context,count,request):
        assert count==2
        assert request[1].index==1 and request[1].size==4096*4
        assert request[0].index==0 and request[0].size==18*4
        assert request[0].type==request[1].type==0
        assert request[0].pass_through==request[1].pass_through==0
        assert ct.cast(request[1].buf,ct.POINTER(ct.c_float))[0]==17
        assert ct.cast(request[0].buf,ct.POINTER(ct.c_float))[0]==-2
        return 0

    def rknn_run(self,context,extend):
        ct.cast(extend,ct.POINTER(abi.RunExtend)).contents.frame_id=12
        return 0

    def rknn_outputs_get(self,context,count,returned,extend):
        assert count==1 and returned[0].want_float==1
        returned[0].buf=ct.addressof(self.buffer)
        returned[0].size=72
        ct.cast(extend,ct.POINTER(abi.OutputExtend)).contents.frame_id=13 if self.mode=='frame_mismatch' else 12
        return -1 if self.mode=='get_failure' else 0

    def rknn_outputs_release(self,context,count,returned):
        self.releases+=1
        for i in range(18):
            self.buffer[i]=-999
        return 0


def fake_step(api):
    bindings={'camera_tokens':{'index':1,'format':3},'conditioning_pose':{'index':0,'format':3}}
    activity={'npu_run_api_called':False,'npu_run_api_call_count':0}
    result=runtime.execute_step(abi,api,abi.CONTEXT(1),np.full((1,2,2048),17,dtype=np.float32),
                                pose(-2),bindings,30000,[],activity)
    assert activity['npu_run_api_call_count']==1
    return result


def test_two_input_step_copies_output_before_release():
    api=FakeSDK()
    output,row=fake_step(api)
    assert np.array_equal(output,np.arange(18,dtype=np.float32).reshape(1,2,9))
    assert api.releases==1 and row['output_buffer_released']
    assert all(x==-999 for x in api.buffer)


@pytest.mark.parametrize('mode',['frame_mismatch','get_failure'])
def test_sdk_outputs_released_on_failure_after_allocation(mode):
    api=FakeSDK(mode)
    with pytest.raises((ValueError,RuntimeError)):
        fake_step(api)
    assert api.releases==1
