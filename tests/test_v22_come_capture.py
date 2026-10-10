"""Synthetic stage/identity/hook guards; no real images or model inference."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

FILE = Path(__file__).resolve().parents[1]/'deploy/rk3588/validation/capture-v22-come-mlp-board.py'
spec = importlib.util.spec_from_file_location('v22_capture',FILE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def reference_pair():
    ids = [1305031102.175304,1305031102.443271]
    selected = dict(revision=1,frame_ids=ids,images=[dict(frame_id=fid,timestamp=fid,
        samefile_validated=True,schedule_token=dict(frame_id=fid,timestamp=fid)) for fid in ids])
    checkpoint = dict(bytes=m.CHECKPOINT_BYTES,sha256=m.CHECKPOINT_SHA)
    profile = dict(kind='original_VGGT_rectified_stage_and_block_capture',
        status='completed_original_CPU_profile_and_capture',image_size=336,
        original_weights=True,architecture_changed=False,npu_used=False,ORB_used=False,
        checkpoint=checkpoint,selected_pair=selected,input_tensor=dict(shape=[2,3,252,336]))
    reference = dict(status='completed_fixed_input_diagnostic',image_size=336,precision='float32',
        device='cpu',original_weights=True,npu_used=False,ORB_used=False,checkpoint=checkpoint,
        pairs=[dict(revision=1,frame_ids=ids,input_tensor=profile['input_tensor'])],selected_inputs=[selected])
    return deepcopy(profile),deepcopy(reference)


def test_exact_original_pair_contract_is_accepted():
    profile,reference = reference_pair()
    pair,selection = m.validate_reference(profile,reference)
    assert pair['frame_ids']==selection['frame_ids']


@pytest.mark.parametrize('change',[
    lambda p,r:p.update(image_size=112),
    lambda p,r:p.update(architecture_changed=True),
    lambda p,r:p.update(npu_used=True),
    lambda p,r:r.update(precision='float16'),
    lambda p,r:r.update(ORB_used=True),
    lambda p,r:r['checkpoint'].update(sha256='0'*64),
    lambda p,r:r['pairs'][0]['frame_ids'].__setitem__(1,float('inf')),
    lambda p,r:r['selected_inputs'][0]['images'][1]['schedule_token'].update(timestamp=1.),
    lambda p,r:r['pairs'][0]['input_tensor'].update(shape=[2,3,336,336]),
    lambda p,r:p['selected_pair']['images'][1].update(samefile_validated=False),
])
def test_modified_source_geometry_or_model_is_rejected(change):
    profile,reference = reference_pair()
    change(profile,reference)
    with pytest.raises(ValueError):
        m.validate_reference(profile,reference)


def test_dino14_features_drop_only_special_preserve_patch_order_without_norm():
    stage = np.arange(2*11*4,dtype=np.float32).reshape(2,11,4)
    got = m.predictor_features(stage,grid_hw=(2,3))
    assert got.shape==(1,2,4,2,3)
    for view in range(2):
        for y in range(2):
            for x in range(3):
                np.testing.assert_array_equal(got[0,view,:,y,x],stage[view,5+y*3+x])
    assert got.flags.c_contiguous


@pytest.mark.parametrize('stage',[
    np.ones((1,11,4),np.float32),np.ones((2,10,4),np.float32),
    np.ones((2,11,4),np.float64),np.full((2,11,4),np.nan,np.float32),
])
def test_predictor_stage_shape_dtype_and_finite_checks(stage):
    with pytest.raises(ValueError):
        m.predictor_features(stage,grid_hw=(2,3))


def test_gate_is_bitwise_including_signed_zero():
    a = np.zeros((1,2,3),np.float32)
    b = a.copy()
    b[0,0,0] = np.float32(-0.)
    assert np.array_equal(a,b)
    assert not m.bitwise_equal(a,b)


def test_board_guard_rejects_host_and_old_output(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'BASE',tmp_path)
    with pytest.raises(ValueError,match='aarch64'):
        m.board_guard('x86_64',tmp_path/'experiments/new')
    with pytest.raises(ValueError,match='BASE/experiments'):
        m.board_guard('aarch64',tmp_path/'elsewhere')
    (tmp_path/'experiments/old').mkdir(parents=True)
    with pytest.raises(FileExistsError):
        m.board_guard('aarch64',tmp_path/'experiments/old')


class Tensor:
    def __init__(self,value,dtype='torch.float32'):
        self.value = np.asarray(value)
        self.device = SimpleNamespace(type='cpu')
        self.dtype = dtype
    def detach(self): return self
    def clone(self): return Tensor(self.value.copy(),self.dtype)
    def contiguous(self): return self


class Handle:
    def __init__(self): self.removed = False
    def remove(self): self.removed = True


class Module:
    def __init__(self,fail=False): self.hooks,self.fail = [],fail
    def register_forward_hook(self,hook):
        if self.fail: raise RuntimeError('registration failed')
        h = Handle()
        self.hooks.append((hook,h))
        return h
    def register_forward_pre_hook(self,hook,with_kwargs=False):
        return self.register_forward_hook(hook)


def model(fail=False):
    block = Module()
    block.norm2,block.mlp,block.ls2 = Module(),Module(fail),Module()
    dino = Module()
    return SimpleNamespace(aggregator=SimpleNamespace(global_blocks=[block],
        patch_embed=SimpleNamespace(blocks=[Module() for _ in range(14)]+[dino]))),block,dino


def test_hooks_only_copy_values_and_are_removed():
    mdl,block,dino = model()
    hooks = m.Captures(mdl)
    stage = Tensor(np.arange(12,dtype=np.float32).reshape(1,3,4))
    result = dino.hooks[0][0](dino,(),stage)
    assert result is None
    np.testing.assert_array_equal(hooks.values['dino14_output'].value,stage.value)
    assert not np.shares_memory(hooks.values['dino14_output'].value,stage.value)
    with pytest.raises(ValueError,match='invocation'):
        dino.hooks[0][0](dino,(),stage)
    handles = list(hooks.handles)
    hooks.close()
    assert not hooks.handles and all(h.removed for h in handles)


def test_registration_failure_removes_partial_hooks():
    mdl,block,dino = model(fail=True)
    with pytest.raises(RuntimeError,match='registration failed'):
        m.Captures(mdl)
    assert all(h.removed for module in (dino,block.norm2) for _,h in module.hooks)
