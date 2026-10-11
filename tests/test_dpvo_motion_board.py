"""Synthetic graph construction/identity/guard tests; no images or checkpoints."""
import importlib.util
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
IMPL = HERE if (HERE / 'run-v26-dpvo-motion-board.py').is_file() else HERE.parent / 'deploy/rk3588/validation'

def load(name, filename):
    spec=importlib.util.spec_from_file_location(name,IMPL / filename)
    value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

driver=load("motion_synthetic","run-v26-dpvo-motion-board.py")
ops=load("motion_ops_synthetic","dpvo_cpu_ops.py")


def features():
    fmap=torch.arange(3*128*8*12,dtype=torch.float32).reshape(1,3,128,8,12)
    imap=torch.arange(3*384*8*12,dtype=torch.float32).reshape(1,3,384,8,12)*.1
    return fmap,imap


def test_integer_patch_centers_repeat_seed_and_original_feature_samples():
    fmap,imap=features()
    with torch.inference_mode():
        plan=driver.build_patch_inputs(fmap,imap,ops,patches_per_view=2,seed=26)
        repeated=driver.build_patch_inputs(fmap,imap,ops,patches_per_view=2,seed=26)
    assert all(torch.equal(plan[k],repeated[k]) for k in plan if isinstance(plan[k],torch.Tensor))
    assert torch.equal(plan["centers"],plan["centers"].floor())
    assert bool(((plan["centers"][...,0]>=1)&(plan["centers"][...,0]<11)).all())
    assert bool(((plan["centers"][...,1]>=1)&(plan["centers"][...,1]<7)).all())
    for view in range(3):
        for local in range(2):
            x,y=map(int,plan["centers"][view,local])
            k=view*2+local
            assert torch.equal(plan["gmap"][0,k],fmap[0,view,:,y-1:y+2,x-1:x+2])
            assert torch.equal(plan["contexts"][0,k],imap[0,view,:,y,x])
            assert torch.equal(plan["patches"][0,k,0],torch.arange(x-1,x+2)[None].float().expand(3,3))
            assert torch.equal(plan["patches"][0,k,1],torch.arange(y-1,y+2)[:,None].float().expand(3,3))
            assert torch.equal(plan["patches"][0,k,2],torch.ones(3,3))


@pytest.mark.parametrize("name,edges", [("first_motion_0_to_1",2),("three_frame_all_pairs",18)])
def test_identity_graph_order_context_zeros_and_group_neighbors(name,edges):
    with torch.inference_mode():
        plan=driver.build_patch_inputs(*features(),ops,patches_per_view=2)
        graph=driver.make_scenario(plan,name)
    assert graph["ii"].numel()==graph["jj"].numel()==graph["kk"].numel()==edges
    assert graph["coords"].shape==(1,edges,2,3,3)
    assert torch.equal(graph["coords"],plan["patches"][:,graph["kk"],:2])
    assert torch.equal(graph["ctx"],plan["contexts"][:,graph["kk"]])
    assert graph["net0"].shape==(1,edges,384) and not bool(graph["net0"].any())
    assert torch.equal(graph["ii"],graph["kk"]//2)
    previous,following=ops.neighbors(graph["kk"],graph["jj"])
    if name=="first_motion_0_to_1":
        assert torch.equal(graph["kk"],torch.arange(2))
        assert torch.equal(graph["jj"],torch.ones(2,dtype=torch.int64))
        assert torch.equal(previous,torch.full((2,),-1)) and torch.equal(previous,following)
    else:
        assert torch.equal(graph["kk"],torch.arange(6).repeat_interleave(3))
        assert torch.equal(graph["jj"],torch.arange(3).repeat(6))
        for base in range(0,18,3):
            assert previous[base:base+3].tolist()==[-1,base,base+1]
            assert following[base:base+3].tolist()==[base+1,base+2,-1]


def test_default_original_counts_are_96_and_864():
    with torch.inference_mode():
        plan=driver.build_patch_inputs(*features(),ops)
        assert driver.make_scenario(plan,driver.SCENARIOS[0])["kk"].numel()==96
        assert driver.make_scenario(plan,driver.SCENARIOS[1])["kk"].numel()==864


@pytest.mark.parametrize("bad", ["views","channels","dtype","count","scenario"])
def test_invalid_graph_inputs_are_not_silently_adapted(bad):
    fmap,imap=features()
    with pytest.raises(ValueError):
        if bad=="views": driver.build_patch_inputs(fmap[:,:2],imap[:,:2],ops)
        elif bad=="channels": driver.build_patch_inputs(fmap[:,:,:127],imap,ops)
        elif bad=="dtype": driver.build_patch_inputs(fmap.double(),imap,ops)
        elif bad=="count": driver.build_patch_inputs(fmap,imap,ops,patches_per_view=0)
        else: driver.make_scenario({"patches_per_view":2},"tracked_or_mock_pose")


def test_board_guard_rejects_host_before_output_creation_or_model_loading(monkeypatch,tmp_path):
    monkeypatch.setattr(driver.platform,"machine",lambda:"x86_64")
    monkeypatch.setattr(driver,"module",lambda *a:pytest.fail("Model/source loaded before guard"))
    argv=[]
    for name in ("common","cpu-ops","source-dir","source-manifest","checkpoint","input-manifest","capture-dir","output-dir"):
        argv += ["--"+name,str(tmp_path/name)]
    for name in ("common","cpu-ops","source-manifest","checkpoint","input-manifest","capture"):
        argv += ["--"+name+"-sha256","0"*64]
    with pytest.raises(ValueError,match="board-only"):
        driver.main(argv)
    assert not (tmp_path/"output-dir").exists()


def test_incomplete_capture_is_rejected_before_loading_arrays(tmp_path):
    with pytest.raises(ValueError,match="completed frozen"):
        driver.load_features(tmp_path,{"status":"failed"},{"frames":[]},lambda *a:pytest.fail("Pin read before status gate"))
