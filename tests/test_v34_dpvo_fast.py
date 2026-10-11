"""Synthetic FAST wiring only: no real model, SDK, ORT, image or BA execution."""
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "deploy/rk3588/validation/dpvo_fast"
DRIVER = ROOT / "run-v34-dpvo-fast-board.py"
GROUP = ROOT.parent / "dpvo_stream"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


driver = load(DRIVER, "v34_fast_synthetic")
core = load(GROUP / "dpvo_stream.py", "v34_frozen_stream_synthetic")
encoder = load(GROUP / "v30_frame_encoder.py", "v34_frozen_encoder_synthetic")
BASE = """
_C.PATCHES_PER_FRAME = 80
_C.PATCH_LIFETIME = 12
_C.REMOVAL_WINDOW = 20
_C.OPTIMIZATION_WINDOW = 12
_C.KEYFRAME_INDEX = 4
_C.KEYFRAME_THRESH = 12.5
_C.MOTION_MODEL = 'DAMPED_LINEAR'
_C.MOTION_DAMPING = 0.5
_C.CENTROID_SEL_STRAT = 'RANDOM'
_C.BUFFER_SIZE = 4096
_C.MIXED_PRECISION = True
_C.LOOP_CLOSURE = False
_C.CLASSIC_LOOP_CLOSURE = False
"""
FAST = """
PATCHES_PER_FRAME: 48
PATCH_LIFETIME: 11
REMOVAL_WINDOW: 16
OPTIMIZATION_WINDOW: 7
KEYFRAME_THRESH: 15.0
"""


def config():
    fields, _ = driver.parse_fast_config(BASE, FAST)
    return core.StreamConfig(**fields)


def test_FAST_overlays_original_base_and_keeps_research_capacity():
    fields, original = driver.parse_fast_config(BASE, FAST)
    cfg = core.StreamConfig(**fields)
    assert (cfg.patches_per_frame, cfg.patch_lifetime,
            cfg.removal_window, cfg.optimization_window) == (48, 11, 16, 7)
    assert cfg.keyframe_index == 4 and cfg.keyframe_threshold == 15.0
    assert cfg.capacity == 64 and cfg.observation_capacity == 4096
    assert not cfg.loop_closure and not cfg.classic_loop_closure
    assert original["MIXED_PRECISION"] is True


@pytest.mark.parametrize("overlay", [
    FAST.replace("48", "96"),
    FAST.replace("48", "True"),
    FAST + "PATCHES_PER_FRAME: 48\n",
    FAST + "NONEXISTENT: 1\n",
    FAST + "LOOP_CLOSURE: True\n",
])
def test_changed_unsupported_or_duplicate_FAST_rejected(overlay):
    with pytest.raises(ValueError):
        driver.parse_fast_config(BASE, overlay)


def test_load_FAST_rehashes_both_sources_and_does_not_apply_default(tmp_path, monkeypatch):
    (tmp_path / "dpvo").mkdir(); (tmp_path / "config").mkdir()
    (tmp_path / "dpvo/config.py").write_text(BASE)
    (tmp_path / "config/fast.yaml").write_text(FAST)
    # A contradictory default must not be read/applied.
    (tmp_path / "config/default.yaml").write_text("PATCHES_PER_FRAME: 999\n")
    pins = {name: driver.sha(tmp_path / name) for name in driver.FAST_SOURCE_PINS}
    monkeypatch.setattr(driver, "FAST_SOURCE_PINS", pins)
    fields, receipt = driver.load_fast_config(tmp_path)
    assert fields["patches_per_frame"] == 48
    assert receipt["merge_order"] == ["dpvo/config.py", "config/fast.yaml"]
    assert receipt["default_yaml_not_applied"]
    assert "CPU FP32" in receipt["executed_precision"]
    (tmp_path / "config/fast.yaml").write_text(FAST + "# changed bytes\n")
    with pytest.raises(ValueError, match="source SHA"):
        driver.load_fast_config(tmp_path)


def test_FAST_sampling_from_original_dense_reference_uses_current_48_centers():
    # Deliberately fewer dense pixels than archived M96; repetitions permitted.
    f = np.arange(128 * 4 * 5, dtype=np.float32).reshape(1, 1, 128, 4, 5)
    c = np.arange(384 * 4 * 5, dtype=np.float32).reshape(1, 1, 384, 4, 5)
    centers = np.tile(np.array([[1, 1], [3, 2]], np.int64), (24, 1))
    result = encoder.features_to_patches(f, c, centers)
    assert result["gmap"].shape == (48, 128, 3, 3)
    assert result["context"].shape == (48, 384)
    assert result["patches"].shape == (48, 3, 3, 3)
    np.testing.assert_array_equal(result["context"], c[0, 0, :, centers[:, 1], centers[:, 0]])
    np.testing.assert_array_equal(result["patches"][:, :2, 1, 1], centers)


def test_native_cache_local_IDs_and_generation_use_FAST_M48():
    calls, closed = [], []
    class FakePrepared:
        def __init__(self, library, gmap, fmap, **kwargs):
            self.preparation = {"synthetic": True}
            assert gmap.shape == (1, 96, 128, 3, 3)
            assert fmap.shape == (1, 2, 128, 4, 4)
        def correlate(self, coords, kk, jj, **kwargs):
            calls.append((kk.copy(), jj.copy(), kwargs["cache_key"]))
            return np.zeros((1, len(kk), 882), np.float32), {"synthetic": True}
        def close(self): closed.append(True)
        def metrics(self): return {"closed": True, "current_context_bytes": 0}
    def frame(name):
        g=np.zeros((48,128,3,3),np.float32);f=np.zeros((128,4,4),np.float32)
        g.flags.writeable=f.flags.writeable=False
        return {"frame_id":name,"gmap":g,"fmap":f}
    records=[frame("zero"),frame("one"),frame("two")]
    corr=driver.CompactCorrelation(FakePrepared, "synthetic-no-native-library", patch_count=48)
    ii=np.array([0,2,2],np.int64);jj=np.array([2,0,2],np.int64);kk=np.array([47,96,143],np.int64)
    coords=np.zeros((1,3,2,3,3),np.float32)
    corr(coords,ii=ii,jj=jj,kk=kk,frames=records)
    np.testing.assert_array_equal(calls[0][0],[47,48,95])
    np.testing.assert_array_equal(calls[0][1],[1,0,1])
    assert '"patches_per_frame":48' in calls[0][2]
    corr(coords,ii=ii,jj=jj,kk=kk,frames=records)
    assert len(corr.generations)==1
    records[2]=frame("different-observation")
    corr(coords,ii=ii,jj=jj,kk=kk,frames=records)
    assert len(corr.generations)==2 and len(closed)==1
    corr.close();assert len(closed)==2 and not corr.metrics()["current_context_open"]


def test_init3072_and_random_depth_consumption_with_FAKE_BA():
    # All geometric callbacks are synthetic. No BA or trained Update is invoked.
    ba_calls, update_edges, draws = [], [], []
    class Geometry:
        def se3_compose(self, a, b):
            out=a.copy();out[...,:3]+=b[...,:3];return out
        def se3_inverse(self, a):
            out=a.copy();out[...,:3]*=-1;return out
        def se3_exp(self, a):
            out=np.zeros(a.shape[:-1]+(7,),np.float32);out[...,:3]=a[...,:3];out[...,6]=1;return out
        def se3_log(self, a):
            out=np.zeros(a.shape[:-1]+(6,),np.float32);out[...,:3]=a[...,:3];return out
        def reproject(self, poses, patches, K, ii, jj, kk):
            np.testing.assert_array_equal(ii,kk//48)
            return patches[:,kk,:2].copy()
        def flow_mag(self, poses, patches, K, ii, jj, kk, beta):
            return np.full((1,len(ii),3,3),20.,np.float32),np.ones((1,len(ii),3,3),bool)
        def BA(self, poses, patches, K, target, weight, lmbda, ii, jj, kk, t0, t1, M, iterations, eff_impl):
            assert M==48 and iterations==2 and not eff_impl
            ba_calls.append((len(ii),t0,t1));return {"synthetic":True}
    def rng(M):
        draws.append(M);return np.linspace(.1,.9,M,dtype=np.float32)
    def update(net,ctx,corr,ii,jj,kk):
        update_edges.append(len(ii));d=np.zeros((1,len(ii),2),np.float32);d[...,0]=3.
        return {"net":net.copy(),"delta":d,"weight":np.full_like(d,.5)}
    def correlate(coords,**kw):return np.zeros((1,len(kw["kk"]),882),np.float32)
    stream=core.DPVOStream(config(),geometry=Geometry(),update=update,correlate=correlate,rng=rng)
    def frame(i):
        patches=np.ones((48,3,3,3),np.float32)
        patches[:,0]=np.arange(3,dtype=np.float32)[None,None,:]
        patches[:,1]=np.arange(3,dtype=np.float32)[None,:,None]
        return dict(frame_id=str(i),timestamp=float(i),captured_monotonic=100.+i,
                    intrinsics=np.array([20,20,4,4],np.float32),
                    fmap=np.zeros((128,4,4),np.float32),gmap=np.zeros((48,128,3,3),np.float32),
                    context=np.zeros((48,384),np.float32),patches=patches)
    for i in range(8):stream.process_encoded(frame(i))
    snap=stream.snapshot()
    assert snap["initialized"] and len(snap["ii"])==3072
    assert len(ba_calls)==12 and all(x[0]==3072 for x in ba_calls)
    assert draws==[48]*8 and stream.metrics()["patch_count"]==48
    # Even median-initialized frame consumes its original random depth draw.
    stream.process_encoded(frame(8));assert draws==[48]*9
    before=stream.metrics()["fresh_poses"] if "fresh_poses" in stream.metrics() else None
    result=stream.terminate()
    assert len(ba_calls)==25 and result["fresh_pose_valid"] is False
    with pytest.raises(core.StreamFailure, match="terminated"):
        stream.terminate()
    if before is not None:assert stream.metrics()["fresh_poses"]==before


def test_board_guard_precedes_output_creation_or_Torch_import(tmp_path, monkeypatch):
    def denied():raise ValueError("synthetic board guard denial")
    monkeypatch.setattr(driver,"require_board",denied)
    target=tmp_path/"must-not-exist"
    names=("source-dir","checkpoint","geometry","native-helper","library","build-receipt",
           "encoder-model","encoder-export","encoder-ort","encoder-capture-dir","dataset-dir",
           "source-manifest","stream-module","encoder-module","parent-driver")
    args=[item for n in names for item in ("--"+n,str(tmp_path/(n+".missing")))]
    args += ["--output-dir",str(target)]
    torch_was_loaded="torch" in sys.modules
    with pytest.raises(ValueError,match="guard denial"):driver.main(args)
    assert not target.exists() and ("torch" in sys.modules)==torch_was_loaded
