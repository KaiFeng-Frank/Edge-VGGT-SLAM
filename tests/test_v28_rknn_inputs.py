"""Synthetic mixed-input tests: no SDK library, real model or real features."""
import ctypes as ct
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def helper_path(name):
    here = Path(__file__).resolve().parent
    local = here / name
    return local if local.is_file() else here.parent / "deploy/rk3588/validation" / name


packing = load(helper_path("v28_rknn_inputs.py"), "_synthetic_v28_inputs")
ABI_SHA = "37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608"


@pytest.fixture(scope="module")
def abi():
    path = Path(os.environ.get("EDGE_VGGT_ABI_HELPER", helper_path("run-v12-rknn-block-board.py")))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == ABI_SHA
    # Import Python ctypes definitions only; never call its hardware/header guard.
    return load(path, "_synthetic_v28_abi")


def contracts(e=3, c=4, corr=6, kk=(2, 2), ij=(1, 3)):
    shapes = [(1,e,c), (1,e,c), (1,e,corr), (e,), (1,e,1),
              (e,), (1,e,1), kk, kk, (e,), ij, ij, (e,)]
    return [dict(name=name, shape=list(shape), dtype="int64" if name in packing.INDEX_NAMES else "float32")
            for name, shape in zip(packing.INPUT_NAMES, shapes)]


def output_contracts(e=3, c=4):
    return [dict(name=name, shape=[1,e,c if name == "net_out" else 2], dtype="float32")
            for name in packing.OUTPUT_NAMES]


def attrs(rows, int_type=8, quant=2):
    result = []
    for index, row in enumerate(rows):
        kind = int_type if row["dtype"] == "int64" else 1
        count = math.prod(row["shape"])
        size = count * {1:2, 6:4, 8:8}[kind]
        result.append(dict(index=index, name=row["name"], shape=list(row["shape"]),
                           n_elems=count, size=size, format=3, type=kind,
                           quantization_type=quant, scale=1., zero_point=0,
                           fractional_length=0, pass_through=0, w_stride=0,
                           h_stride=0, size_with_stride=size))
    return result


def feed():
    return dict(net=np.arange(12, dtype=np.float32).reshape(1,3,4),
                inp=np.full((1,3,4), .5, np.float32),
                corr=np.arange(18, dtype=np.float32).reshape(1,3,6),
                previous_indices=np.array([0,0,0], np.int64),
                previous_valid=np.array([0,0,1], np.float32).reshape(1,3,1),
                following_indices=np.array([2,0,0], np.int64),
                following_valid=np.array([1,0,0], np.float32).reshape(1,3,1),
                kk_indices=np.array([[0,2],[1,0]], np.int64),
                kk_valid=np.array([[1,1],[1,0]], np.float32),
                kk_inverse=np.array([0,1,0], np.int64),
                ij_indices=np.array([[0,1,2]], np.int64),
                ij_valid=np.ones((1,3), np.float32),
                ij_inverse=np.zeros(3, np.int64))


@pytest.mark.parametrize("int_type,quant", [(8,0),(8,2),(6,0),(6,2)])
def test_real_typed_buffers_and_owned_lifetime(abi, int_type, quant):
    bindings = packing.validate_input_attrs(contracts(), attrs(contracts(), int_type, quant))
    arrays = feed()
    packed = packing.pack_inputs(abi, arrays, bindings)
    original = {key:value.copy() for key,value in arrays.items()}
    for request, row, buffer in zip(packed.requests, bindings, packed.owned_buffers):
        assert request.index == row["index"] and request.pass_through == 0 and request.fmt == 3
        assert request.size == buffer.nbytes == row["transport_bytes"]
        assert buffer.flags.c_contiguous and buffer.flags.owndata
        kind = ct.c_int64 if request.type == 8 else ct.c_int32 if request.type == 6 else ct.c_float
        view = np.ctypeslib.as_array(ct.cast(request.buf,ct.POINTER(kind)), shape=(buffer.size,))
        np.testing.assert_array_equal(view.reshape(buffer.shape), original[row["name"]])
        arrays[row["name"]].fill(0)
        np.testing.assert_array_equal(buffer, original[row["name"]])
        if row["name"] in packing.INDEX_NAMES:
            assert request.type == int_type
            assert request.size == buffer.size * (8 if int_type == 8 else 4)
        else:
            assert request.type == 0 and request.size == row["sdk_bytes"] * 2
    assert len(packed.records) == len(packed.requests) == len(packed.owned_buffers) == 13
    json.dumps(packed.records, allow_nan=False)


def test_observed_E864_neutral_affine_attr_contract_without_inference():
    rows = contracts(864,384,882,(288,3),(9,96))
    bindings = packing.validate_input_attrs(rows, attrs(rows))
    assert sum(row["transport_type"] == 8 for row in bindings) == 6
    assert sum(row["transport_type"] == 0 for row in bindings) == 7
    assert all(row["neutral_affine_metadata"] and row["sdk_format"] == 3 for row in bindings)
    out = packing.validate_output_attrs(output_contracts(864,384), attrs(output_contracts(864,384)))
    assert [row["returned_FP32_bytes"] for row in out] == [1327104,6912,6912]


@pytest.mark.parametrize("field,value", [("scale",.5),("zero_point",1),("fractional_length",1),
                                        ("quantization_type",1),("type",0),("type",7)])
def test_integer_nonidentity_or_wrong_type_is_rejected(field,value):
    raw = attrs(contracts()); raw[3][field] = value
    with pytest.raises(ValueError): packing.validate_input_attrs(contracts(), raw)


@pytest.mark.parametrize("field,value", [("scale",float("nan")),("scale",.999),
                                        ("zero_point",1),("fractional_length",1),
                                        ("quantization_type",1),("type",8)])
def test_float_nonidentity_or_nonfloat_is_rejected(field,value):
    raw = attrs(contracts()); raw[0][field] = value
    with pytest.raises(ValueError): packing.validate_input_attrs(contracts(), raw)


@pytest.mark.parametrize("field,value", [("format",1),("format",2),("index",2),
                                        ("shape",[3,1,4]),("n_elems",11),("size",48),
                                        ("name","other")])
def test_raw_attrs_must_exactly_match_contract(field,value):
    raw = attrs(contracts()); raw[0][field] = value
    with pytest.raises(ValueError): packing.validate_input_attrs(contracts(), raw)


def test_unknown_duplicate_missing_and_uint32_bytes_reject_without_allocation():
    for raw in (attrs(contracts())[:-1], attrs(contracts()) + [attrs(contracts())[0]]):
        with pytest.raises(ValueError): packing.validate_input_attrs(contracts(), raw)
    raw = attrs(contracts()); raw[1]["name"] = "net"
    with pytest.raises(ValueError): packing.validate_input_attrs(contracts(),raw)
    row = dict(name="net", shape=[1,2**30,1], dtype="float32")
    logical = dict(name="net",shape=(1,2**30,1),logical_dtype="float32")
    with pytest.raises(ValueError, match="uint32"):
        packing._attr(attrs([row])[0], logical, 0)


@pytest.mark.parametrize("key,position,value", [("previous_indices",0,-1),
       ("following_indices",0,3),("kk_indices",(1,1),3),("kk_inverse",0,2),
       ("ij_inverse",0,1),("previous_indices",0,2**32+1)])
def test_index_domain_failures_cannot_wrap_or_use_masked_invalid_index(abi,key,position,value):
    arrays = feed(); arrays[key][position] = value
    bound = packing.validate_input_attrs(contracts(),attrs(contracts(),6))
    with pytest.raises(ValueError,match="domain"):
        packing.pack_inputs(abi,arrays,bound)


@pytest.mark.parametrize("key,dtype", [("previous_indices",np.int32),
                                     ("kk_inverse",np.float32),("net",np.float64)])
def test_source_dtype_is_not_silently_reinterpreted(abi,key,dtype):
    arrays = feed(); arrays[key] = arrays[key].astype(dtype)
    with pytest.raises(ValueError,match="dtype/shape"):
        packing.pack_inputs(abi,arrays,packing.validate_input_attrs(contracts(),attrs(contracts())))


@pytest.mark.parametrize("key,value", [("net",float("nan")),("corr",float("inf")),("kk_valid",.5)])
def test_nonfinite_and_nonbinary_mask_reject(abi,key,value):
    arrays=feed(); arrays[key].flat[0]=value
    with pytest.raises(ValueError):
        packing.pack_inputs(abi,arrays,packing.validate_input_attrs(contracts(),attrs(contracts())))


def test_noncontiguous_arrays_are_c_order_owned_and_names_cannot_be_missing(abi):
    arrays=feed(); arrays["net"]=np.asfortranarray(arrays["net"])
    assert not arrays["net"].flags.c_contiguous
    bound=packing.validate_input_attrs(contracts(),attrs(contracts()))
    packed=packing.pack_inputs(abi,arrays,bound)
    assert packed.owned_buffers[0].flags.c_contiguous
    np.testing.assert_array_equal(packed.owned_buffers[0],arrays["net"])
    del arrays["ij_inverse"]
    with pytest.raises(ValueError):packing.pack_inputs(abi,arrays,bound)


def test_outputs_use_float_owned_bytes_and_reject_integer_or_layout():
    rows=output_contracts();raw=attrs(rows)
    bindings=packing.validate_output_attrs(rows,raw)
    assert [row["name"] for row in bindings]==list(packing.OUTPUT_NAMES)
    for row in bindings:
        assert row["want_float"]==1 and row["is_prealloc"]==0
        assert row["returned_FP32_bytes"]==2*row["sdk_bytes"]
        assert row["logical_shape"]==list(row["shape"])
    for field,value in (("type",8),("format",1),("name","net")):
        changed=attrs(rows);changed[0][field]=value
        with pytest.raises(ValueError):packing.validate_output_attrs(rows,changed)


class SyntheticSDK:
    """Fake ctypes buffers are overwritten on release to detect alias bugs."""
    def __init__(self,abi,source,fault=None):
        self.abi,self.source,self.fault=abi,source,fault
        self.inputs_seen=0;self.run_calls=0;self.releases=0
        self.values={"net":np.arange(12,dtype=np.float32).reshape(1,3,4),
                     "delta":np.full((1,3,2),.25,np.float32),
                     "weight":np.full((1,3,2),.5,np.float32)}

    def rknn_inputs_set(self,context,count,requests):
        assert count==13
        for row,request in zip(contracts(),requests):
            dtype=ct.c_int64 if row["dtype"]=="int64" else ct.c_float
            assert request.type==(8 if row["dtype"]=="int64" else 0)
            assert request.pass_through==0 and request.fmt==3
            shape=tuple(row["shape"]); n=math.prod(shape)
            view=np.ctypeslib.as_array(ct.cast(request.buf,ct.POINTER(dtype)),shape=(n,))
            np.testing.assert_array_equal(view.reshape(shape),self.source[row["name"]])
            assert request.size==view.nbytes
        self.inputs_seen+=1
        return -2 if self.fault=="input" else 0

    def rknn_run(self,context,run):
        self.run_calls+=1
        record=ct.cast(run,ct.POINTER(self.abi.RunExtend)).contents
        assert record.non_block==0 and record.timeout_ms>0
        record.frame_id=37
        return -3 if self.fault=="run" else 0

    def rknn_outputs_get(self,context,count,outputs,received):
        assert count==3
        for i,(key,value) in enumerate(self.values.items()):
            assert outputs[i].want_float==1 and outputs[i].is_prealloc==0 and outputs[i].index==i
            if self.fault=="nonfinite" and key=="delta":value.flat[0]=float("nan")
            if self.fault=="micro_negative" and key=="weight":value.flat[0]=-1.19e-7
            outputs[i].buf=value.ctypes.data;outputs[i].size=value.nbytes
            if self.fault=="partial":return -4
        if self.fault=="bytes":outputs[0].size-=4
        ct.cast(received,ct.POINTER(self.abi.OutputExtend)).contents.frame_id=38 if self.fault=="frame" else 37
        return 0

    def rknn_outputs_release(self,context,count,outputs):
        self.releases+=1
        for value in self.values.values():value.fill(-9999)
        return -5 if self.fault=="release" else 0


@pytest.fixture(scope="module")
def runtime():
    path=Path(os.environ.get("EDGE_VGGT_UPDATE_RUNTIME",helper_path("run-v28-dpvo-update-rknn-board.py")))
    return load(path,"_synthetic_v28_runtime")


def test_runtime_execute_copies_before_release_and_retains_mixed_inputs(abi,runtime):
    arrays=feed(); sdk=SyntheticSDK(abi,arrays);cleanup=[]
    outputs,row=runtime.execute(sdk,abi.CONTEXT(1),abi,packing,arrays,
        packing.validate_input_attrs(contracts(),attrs(contracts())),output_contracts(),cleanup)
    assert sdk.inputs_seen==sdk.run_calls==sdk.releases==1 and cleanup==[]
    assert row["frame_id"]==37 and row["output_buffers_released"]
    assert len(row["input_records"])==13 and row["forward_seconds"]>=row["release_seconds"]>=0
    np.testing.assert_array_equal(outputs["net"],np.arange(12,dtype=np.float32).reshape(1,3,4))
    assert (outputs["weight"]==.5).all() and not any(np.shares_memory(v,sdk.values[k]) for k,v in outputs.items())


@pytest.mark.parametrize("fault,release_count", [("input",0),("run",0),("partial",1),
                    ("frame",1),("bytes",1),("nonfinite",1),("micro_negative",1),("release",1)])
def test_runtime_execute_failure_releases_partial_outputs_and_preserves_negative_guard(abi,runtime,fault,release_count):
    arrays=feed();sdk=SyntheticSDK(abi,arrays,fault);cleanup=[]
    with pytest.raises((ValueError,RuntimeError)):
        runtime.execute(sdk,abi.CONTEXT(1),abi,packing,arrays,
            packing.validate_input_attrs(contracts(),attrs(contracts())),output_contracts(),cleanup)
    assert sdk.releases==release_count
    assert bool(cleanup)==(fault=="release")
