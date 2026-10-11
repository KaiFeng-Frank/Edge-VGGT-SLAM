"""Tiny synthetic inputs only, independent loops; never load real features."""
import ctypes as ct
import importlib.util
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
SOURCE=HERE / "dpvo_corr.py"
if not SOURCE.is_file():
    SOURCE=HERE.parent / "deploy" / "rk3588" / "validation" / "dpvo_native" / SOURCE.name
NATIVE_SOURCE=SOURCE.with_suffix(".cpp")
spec=importlib.util.spec_from_file_location("dpvo_corr_v27_synthetic",SOURCE)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture(scope="session")
def library(tmp_path_factory):
    directory=tmp_path_factory.mktemp("native_corr")
    path=directory / "libdpvo_corr_synthetic.so"
    command=[os.environ.get("CXX","g++"),"-std=c++17","-O3","-fPIC","-shared","-fopenmp",
             "-fno-fast-math","-ffp-contract=off",str(NATIVE_SOURCE),"-o",str(path)]
    subprocess.run(command,check=True,capture_output=True,text=True,cwd=directory)
    return path


def oracle(gmap,fmap,coords,kk,jj):
    """Independent float64 pool/dot/4-neighbour loops; FP32 coordinates first."""
    b,_,c,ph,pw=gmap.shape
    _,v,_,h,w=fmap.shape
    pool=np.empty((b,v,c,h//4,w//4),np.float64)
    for y in range(h//4):
        for x in range(w//4):
            pool[...,y,x]=fmap[...,y*4:y*4+4,x*4:x*4+4].astype(np.float64).mean(axis=(-2,-1))
    result=np.empty((b,len(kk),7,7,ph,pw,2),np.float64)
    for batch in range(b):
        for edge in range(len(kk)):
            for py in range(ph):
                for px in range(pw):
                    source=gmap[batch,kk[edge],:,py,px].astype(np.float64)
                    for level,(scale,target) in enumerate(((1,fmap),(4,pool))):
                        x,y=coords[batch,edge,:,py,px]/np.float32(scale)
                        ix,iy=int(np.floor(x)),int(np.floor(y))
                        dx,dy=float(np.float32(x-np.floor(x))),float(np.float32(y-np.floor(y)))
                        for ox in range(7):
                            for oy in range(7):
                                total=0.0
                                for sx,sy,factor in ((0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),
                                                     (0,1,(1-dx)*dy),(1,1,dx*dy)):
                                    tx,ty=ix+ox-3+sx,iy+oy-3+sy
                                    if 0<=tx<target.shape[-1] and 0<=ty<target.shape[-2]:
                                        total+=factor*sum(float(source[ch])*float(target[batch,jj[edge],ch,ty,tx]) for ch in range(c))
                                result[batch,edge,ox,oy,py,px,level]=total
    return result.reshape(b,len(kk),98*ph*pw)


def fixture(batch=1,channels=8,ph=2,pw=3,edges=5,seed=3):
    rng=np.random.default_rng(seed)
    g=rng.normal(0,.3,(batch,3,channels,ph,pw)).astype(np.float32)
    f=rng.normal(0,.3,(batch,2,channels,9,11)).astype(np.float32)
    coords=rng.uniform(-2,12,(batch,edges,2,ph,pw)).astype(np.float32)
    coords[:,:,0,0,0]=-.125
    coords[:,:,1,0,0]=8.75
    kk=rng.integers(0,3,edges,dtype=np.int64)
    jj=rng.integers(0,2,edges,dtype=np.int64)
    return g,f,coords,kk,jj


@pytest.mark.parametrize("threads",[1,2])
@pytest.mark.parametrize("batch",[1,2])
def test_nonsquare_fractional_border_oracle(library,threads,batch):
    g,f,coords,kk,jj=fixture(batch=batch)
    with module.PreparedCorrelation(library,g,f,threads=threads,cache_key="synthetic-v1") as prepared:
        actual,timing=prepared.correlate(coords,kk,jj,cache_key="synthetic-v1")
        np.testing.assert_allclose(actual,oracle(g,f,coords,kk,jj),rtol=1e-4,atol=1e-5)
        assert timing["native"]["threads"]==threads
        assert timing["total_seconds"] >= timing["native_call_seconds"] >= timing["native"]["kernel_seconds"]
        assert prepared.metrics()["completed_calls"] == 1
    assert prepared.metrics()["closed"] and prepared.metrics()["current_context_bytes"]==0


def test_xy_order_level_inner_and_identity_dot(library):
    g=np.zeros((1,1,8,2,3),np.float32);g[:,:,0]=1;g[:,:,1]=2
    f=np.zeros((1,1,8,8,12),np.float32);f[:,:,0]=3;f[:,:,1]=5
    coords=np.empty((1,1,2,2,3),np.float32);coords[:,:,0]=3.5;coords[:,:,1]=2.25
    with module.PreparedCorrelation(library,g,f,threads=2,cache_key="literal") as prepared:
        result,_=prepared.correlate(coords,np.array([0],np.int64),np.array([0],np.int64),cache_key="literal")
    grid=result.reshape(1,1,7,7,2,3,2)
    assert np.array_equal(grid[0,0,3,3],np.full((2,3,2),13,np.float32))
    assert np.all(grid[0,0,3,0,:,:,0]==3.25)
    assert np.all(grid[0,0,0,3,:,:,0]==13)
    assert result[0,0,((3*7+3)*2*3+1*3+2)*2+1]==13


@pytest.mark.parametrize("edges",[96,864])
def test_c128_patch3_workload_shape_tiny_features(library,edges):
    g,f,coords,kk,jj=fixture(channels=128,ph=3,pw=3,edges=edges)
    with module.PreparedCorrelation(library,g,f,threads=2,cache_key="shape") as prepared:
        result,_=prepared.correlate(coords,kk,jj,cache_key="shape")
        assert result.shape==(1,edges,882)
        select=np.array([0,edges//2,edges-1])
        np.testing.assert_allclose(result[:,select],oracle(g,f,coords[:,select],kk[select],jj[select]),rtol=1e-4,atol=1e-5)
        repeated,_=prepared.correlate(coords,kk,jj,cache_key="shape")
        assert result.tobytes()==repeated.tobytes()


def test_noncontiguous_buffers_and_snapshot_owned(library):
    g,f,coords,kk,jj=fixture()
    g=g[...,::-1];f=f[...,::-1];coords=coords[...,::-1]
    original_g,original_f=g.copy(),f.copy()
    with module.PreparedCorrelation(library,g,f,threads=1,cache_key="owned") as prepared:
        first,_=prepared.correlate(coords,kk,jj,cache_key="owned")
        g.fill(20);f.fill(-20)
        second,_=prepared.correlate(coords,kk,jj,cache_key="owned")
        assert first.tobytes()==second.tobytes()
        np.testing.assert_allclose(first,oracle(original_g,original_f,coords,kk,jj),rtol=1e-4,atol=1e-5)
        with pytest.raises(ValueError,match="generation"):
            prepared.correlate(coords,kk,jj,cache_key="changed")
    assert np.isfinite(first).all()
    prepared.close()
    with pytest.raises(module.CorrelationError,match="closed"):
        prepared.correlate(coords,kk,jj,cache_key="owned")


def test_empty_edges_safe_and_far_oob_zero(library):
    g,f,coords,kk,jj=fixture()
    with module.PreparedCorrelation(library,g,f,threads=2,cache_key="empty") as prepared:
        result,_=prepared.correlate(coords[:,:0],kk[:0],jj[:0],cache_key="empty")
        assert result.shape==(1,0,588)
        coords.fill(-1000000)
        outside,_=prepared.correlate(coords,kk,jj,cache_key="empty")
        assert not outside.any()


@pytest.mark.parametrize("bad",["NaN","Inf","range","negative_patch","bad_view"])
def test_native_request_errors_are_explicit(library,bad):
    g,f,coords,kk,jj=fixture()
    if bad=="NaN":coords.flat[0]=np.nan
    elif bad=="Inf":coords.flat[0]=np.inf
    elif bad=="range":coords.flat[0]=2**30
    elif bad=="negative_patch":kk[0]=-1
    else:jj[0]=2
    with module.PreparedCorrelation(library,g,f,threads=1,cache_key="bad") as prepared:
        with pytest.raises(module.CorrelationError):
            prepared.correlate(coords,kk,jj,cache_key="bad")
        assert prepared.metrics()["failed_calls"]==1
        assert prepared.metrics()["completed_calls"]==0


@pytest.mark.parametrize("bad",["double_coords","int32_index","radius","shape"])
def test_python_dtype_shape_and_radius_gates(library,bad):
    g,f,coords,kk,jj=fixture(); radius=3
    if bad=="double_coords":coords=coords.astype(np.float64)
    elif bad=="int32_index":kk=kk.astype(np.int32)
    elif bad=="radius":radius=2
    else:coords=coords[:,:,:,:,:2]
    with module.PreparedCorrelation(library,g,f,threads=1,cache_key="bad") as prepared:
        with pytest.raises(ValueError):
            prepared.correlate(coords,kk,jj,cache_key="bad",radius=radius)


@pytest.mark.parametrize("bad",["NaN","Inf","FP64","channels","pool_small","thread_count"])
def test_prepare_rejects_invalid_features(library,bad):
    g,f,_,_,_=fixture();threads=1
    if bad=="NaN":g.flat[0]=np.nan
    elif bad=="Inf":f.flat[0]=np.inf
    elif bad=="FP64":g=g.astype(np.float64)
    elif bad=="channels":g=g[:,:,:7];f=f[:,:,:7]
    elif bad=="pool_small":f=f[...,:3,:]
    else:threads=0
    with pytest.raises((ValueError,module.CorrelationError)):
        module.PreparedCorrelation(library,g,f,threads=threads,cache_key="prepare-bad")


def test_finite_features_overflow_is_not_silent_nan(library):
    g,f,coords,kk,jj=fixture()
    f.fill(np.finfo(np.float32).max)
    with pytest.raises(module.CorrelationError,match="pooling"):
        module.PreparedCorrelation(library,g,f,threads=1,cache_key="overflow")
    g.fill(np.finfo(np.float32).max);f.fill(1)
    with module.PreparedCorrelation(library,g,f,threads=1,cache_key="dot-overflow") as prepared:
        coords.fill(2)
        with pytest.raises(module.CorrelationError,match="correlation"):
            prepared.correlate(coords,kk,jj,cache_key="dot-overflow")


def test_direct_abi_lengths_and_dimension_overflow(library):
    api,_=module.load_library(library)
    g,f,_,_,_=fixture()
    handle=ct.c_void_p();stats=module.NativeStats();error=ct.create_string_buffer(128)
    gp=module._pointer(g,ct.c_float);fp=module._pointer(f,ct.c_float)
    result=api.dpvo_corr_v27_prepare(gp,g.size-1,fp,f.size,*[1,3,2,8,2,3,9,11],1,
                                    ct.byref(handle),ct.byref(stats),error,len(error))
    assert result and not handle.value and b"length" in error.value
    result=api.dpvo_corr_v27_prepare(gp,g.size,fp,f.size,*[2**31-1,2**31-1,2,8,2,3,9,11],1,
                                    ct.byref(handle),ct.byref(stats),error,len(error))
    assert result and not handle.value and (b"overflow" in error.value or b"budget" in error.value)
    assert api.dpvo_corr_v27_destroy(ct.byref(handle),error,len(error))==0


def test_oneshot_total_includes_prepare_close_and_hash_pin(library):
    g,f,coords,kk,jj=fixture()
    result,timing=module.correlate_once(library,g,f,coords,kk,jj,threads=2,cache_key="once",library_sha256=module._sha(library))
    np.testing.assert_allclose(result,oracle(g,f,coords,kk,jj),rtol=1e-4,atol=1e-5)
    assert timing["total_seconds"]>=timing["preparation"]["total_seconds"]+timing["correlation"]["total_seconds"]
    assert timing["includes_prepare_and_close"] and timing["closed"]
    with pytest.raises(module.CorrelationError,match="SHA256"):
        module.PreparedCorrelation(library,g,f,threads=1,cache_key="hash",library_sha256="0"*64)


def test_generation_and_nested_metrics_are_independent(library):
    g,f,_,_,_=fixture()
    with module.PreparedCorrelation(library,g,f,threads=1,cache_key="immutable") as prepared:
        with pytest.raises(AttributeError):
            prepared.cache_key="bad-generation"
        receipt=prepared.metrics()
        receipt["preparation"]["native"]["context_bytes"]=0
        assert prepared.metrics()["preparation"]["native"]["context_bytes"]>0
        assert ct.sizeof(module.NativeStats)==prepared._api.dpvo_corr_v27_stats_size()


def test_thread_count_does_not_change_scalar_output(library):
    g,f,coords,kk,jj=fixture(batch=2,channels=16)
    outputs=[]
    for threads in (1,2):
        value,_=module.correlate_once(library,g,f,coords,kk,jj,threads=threads,cache_key="thread-order")
        outputs.append(value)
    assert outputs[0].tobytes()==outputs[1].tobytes()


def test_direct_abi_wrong_output_length_and_alias_are_rejected(library):
    g,f,coords,kk,jj=fixture()
    with module.PreparedCorrelation(library,g,f,threads=1,cache_key="abi") as prepared:
        api=prepared._api;stats=module.NativeStats();error=ct.create_string_buffer(128)
        output=np.empty((1,5,588),np.float32)
        request=[prepared._context,module._pointer(coords,ct.c_float),coords.size,
                 module._pointer(kk,ct.c_int64),module._pointer(jj,ct.c_int64),kk.size,
                 kk.size,3,module._pointer(output,ct.c_float),output.size-1,
                 ct.byref(stats),error,len(error)]
        assert api.dpvo_corr_v27_correlate(*request)!=0 and b"length" in error.value
        request[8]=module._pointer(coords,ct.c_float);request[9]=output.size
        assert api.dpvo_corr_v27_correlate(*request)!=0 and b"overlap" in error.value
