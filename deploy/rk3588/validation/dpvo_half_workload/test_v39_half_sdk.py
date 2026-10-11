"""Generated fake SDK only; no native library, real model or board execution."""
import ctypes as ct
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).parent
PUBLIC = Path(os.environ.get("EDGE_V39_PUBLIC_SOURCE", "/run/user/1000/edge-vggt-v38/public-snapshot"))
VALIDATION = PUBLIC/"deploy/rk3588/validation"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


m = load(ROOT/"run-v39-half-int8-board.py", "half_sdk_test")
abi = load(VALIDATION/"run-v12-rknn-block-board.py", "half_sdk_fake_abi")
guard = load(VALIDATION/"dpvo_encoder_int8/query-v37-spatial-int8-io-board.py", "half_sdk_fake_guard")
transport = load(VALIDATION/"dpvo_sparse_context_sdk/v34_sparse_rknn.py", "half_sdk_fake_transport")


class FakeSDK:
    def __init__(self, *, old_shape=False, output_nhwc=False, bad_scale=False, partial_get=False, failed_init=False):
        self.old_shape, self.output_nhwc, self.bad_scale = old_shape, output_nhwc, bad_scale
        self.partial_get, self.failed_init = partial_get, failed_init
        self.released = self.destroyed = self.runs = 0
        self.buffers, self.submitted = [], None

    def rknn_init(self, context, *_):
        ct.cast(context, ct.POINTER(abi.CONTEXT)).contents.value = 123
        return -1 if self.failed_init else 0

    def rknn_set_core_mask(self, context, mask):
        assert mask == 7
        return 0

    def rknn_query(self, context, command, value, _):
        if command == 5:
            item = ct.cast(value, ct.POINTER(abi.SDKVersion)).contents
            item.api_version = b"2.3.2-fake";item.drv_version = b"fake"
        elif command == 0:
            item = ct.cast(value, ct.POINTER(abi.InputOutputNum)).contents
            item.n_input = item.n_output = 2
        elif command == 6:
            item = ct.cast(value, ct.POINTER(abi.MemSize)).contents
            item.total_dma_allocated_size = 12345
        elif command in (1, 2):
            item = ct.cast(value, ct.POINTER(abi.TensorAttr)).contents
            i = item.index
            h, w = (464, 608) if self.old_shape else (232, 304)
            if command == 1 and i == 0:
                shape, name, fmt, dtype = (1,h,w,3), "normalized_bgr", 1, 2
            elif command == 1:
                shape, name, fmt, dtype = (16,), "patch_indices", 3, 8
            elif i == 0:
                shape, name, fmt, dtype = (1,128,h//4,w//4), "fmap", 0, 2
                if self.output_nhwc:
                    shape, fmt = (1,h//4,w//4,128), 1
            else:
                shape, name, fmt, dtype = (1,16,384), "context", 3, 2
            item.n_dims = len(shape)
            for j, d in enumerate(shape):item.dims[j] = d
            item.name = name.encode();item.fmt = fmt;item.type = dtype
            item.n_elems = int(np.prod(shape));item.size = item.n_elems*(8 if dtype == 8 else 1)
            item.qnt_type = 0 if dtype == 8 else 2
            item.scale = 1 if dtype == 8 else 0 if self.bad_scale else .0125
            item.zp = 0
        else:
            raise AssertionError(command)
        return 0

    def rknn_inputs_set(self, context, count, values):
        assert count == 2
        self.submitted = [(v.type,v.fmt,v.pass_through,v.size) for v in values]
        assert self.submitted == [(0,1,0,1*232*304*3*4),(8,3,0,16*8)]
        raw = np.ctypeslib.as_array(ct.cast(values[0].buf, ct.POINTER(ct.c_float)), shape=(232*304*3,))
        assert np.array_equal(raw.reshape(1,232,304,3), self.expected_image.transpose(0,2,3,1))
        return 0

    def rknn_run(self, context, extension):
        self.runs += 1
        ct.cast(extension, ct.POINTER(abi.RunExtend)).contents.frame_id = self.runs
        return 0

    def rknn_outputs_get(self, context, count, outputs, extension):
        ct.cast(extension, ct.POINTER(abi.OutputExtend)).contents.frame_id = self.runs
        for i, (_, shape) in enumerate(m.OUTPUTS):
            outputs[i].buf = 0
            if self.partial_get and i == 1:
                return -1
            assert outputs[i].want_float == 1 and outputs[i].is_prealloc == 0
            arr = np.full(shape, i+.25, dtype=np.float32)
            self.buffers.append(arr)
            outputs[i].buf = arr.ctypes.data;outputs[i].size = arr.nbytes
        return 0

    def rknn_outputs_release(self, context, count, outputs):
        self.released += 1
        for a in self.buffers:a.fill(np.nan)
        return 0

    def rknn_destroy(self, context):
        self.destroyed += 1
        return 0


def make(sdk):
    record, cleanup = {}, []
    return m.HalfContext(sdk,abi,guard,transport,Path("fake.rknn"),False,record,cleanup),record,cleanup


def inputs():
    image = np.linspace(-.5,1.5,1*3*232*304,dtype=np.float32).reshape(m.INPUT_SHAPE)
    return image,np.full(16,76+1,dtype=np.int64)


def test_typed_conversion_owned_copy_release_and_idempotent_close():
    sdk = FakeSDK();ctx,record,cleanup = make(sdk)
    sdk.expected_image,indices = inputs();activity={}
    out,timing = ctx.execute(sdk.expected_image,indices,activity)
    assert all(np.isfinite(a).all() and a.flags.owndata for a in (v.base if v.base is not None else v for v in out.values()))
    assert np.all(out["fmap"] == .25) and np.all(out["context"] == 1.25)
    assert sdk.released == activity["sdk_run_api_calls"] == ctx.completed_calls == 1
    assert timing["output_buffers_released"] and timing["input_pass_through"] == [0,0]
    ctx.close();ctx.close()
    assert sdk.destroyed == 1 and record["context_destroyed"] and cleanup == []
    with pytest.raises(ValueError,match="Closed"):ctx.execute(sdk.expected_image,indices,{})


@pytest.mark.parametrize("options,error",[(dict(old_shape=True),"shape"),(dict(output_nhwc=True),"Canonical"),(dict(bad_scale=True),"affine")])
def test_attrs_retained_on_startup_rejection_and_destroy(options,error):
    sdk=FakeSDK(**options);record,cleanup={},[]
    with pytest.raises(ValueError,match=error):
        m.HalfContext(sdk,abi,guard,transport,Path("fake.rknn"),False,record,cleanup)
    assert "input_attrs" in record and "output_attrs" in record
    assert sdk.destroyed == 1 and sdk.runs == 0 and cleanup == []


def test_failed_init_nonzero_context_destroyed():
    sdk=FakeSDK(failed_init=True);record,cleanup={},[]
    with pytest.raises(RuntimeError):m.HalfContext(sdk,abi,guard,transport,Path("fake.rknn"),False,record,cleanup)
    assert sdk.destroyed == 1 and record["init_return_code"] == -1


def test_partial_get_releases_preserves_primary_error():
    sdk=FakeSDK(partial_get=True);ctx,record,cleanup=make(sdk)
    sdk.expected_image,indices=inputs();activity={}
    with pytest.raises(RuntimeError,match="outputs_get"):ctx.execute(sdk.expected_image,indices,activity)
    ctx.close()
    assert sdk.released == sdk.destroyed == activity["sdk_run_api_calls"] == 1
    assert ctx.completed_calls == 0 and cleanup == []


@pytest.mark.parametrize("mutation",["dtype","range","image_nan"])
def test_invalid_inputs_fail_before_sdk_run(mutation):
    sdk=FakeSDK();ctx,record,cleanup=make(sdk)
    sdk.expected_image,indices=inputs()
    if mutation == "dtype":indices=indices.astype(np.int32)
    elif mutation == "range":indices[0]=-1
    else:sdk.expected_image[0,0,0,0]=np.nan
    with pytest.raises(ValueError):ctx.execute(sdk.expected_image,indices,{})
    ctx.close()
    assert sdk.runs == 0 and sdk.destroyed == 1


def test_fixed_gate_preserves_numeric_failure_and_profile_schedule():
    b=np.ones((2,),dtype=np.float32);a=b.copy();a[0]+=.00015
    r=m.comparison(a,b)
    assert r["failed_elements"] == 1 and not r["gate_passed"]
    assert r["rtol"] == 1e-4 and r["atol"] == 1e-5
    assert len(m.schedule(False)) == 6 and m.schedule(True) == [(0,"profile"),(1,"profile"),(2,"profile")]


def test_host_main_rejects_before_library_or_output(tmp_path):
    paths=["compile-dir","probe-dir","policy","query-helper","transport-helper","abi-helper","header"]
    argv=[item for name in paths for item in ("--"+name,str(tmp_path/"absent"))]
    out=tmp_path/"out"
    argv += ["--output-dir",str(out),"--compile-sha256","0"*64,"--compiler-sha256",m.COMPILER_SHA,"--diagnostic-allow-fp32-rejected"]
    with pytest.raises(ValueError,match="board only"):m.main(argv)
    assert not out.exists()
