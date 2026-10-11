"""Tiny synthetic protocol tests. Never load a model, SDK library or hardware."""
from pathlib import Path
import ctypes as ct
import importlib.util
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest


HERE = Path(__file__).resolve().parents[1] / "deploy" / "rk3588" / "validation" / "dpvo_source_frame"
spec = importlib.util.spec_from_file_location("v33_synthetic_IO", HERE / "v33_rknn_source_frame.py")
io = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = io
spec.loader.exec_module(io)
spec = importlib.util.spec_from_file_location("v33_synthetic_compiler", HERE / "compile-v33-source-frame-update.py")
compiler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compiler)


class Input(ct.Structure):
    _fields_ = [("index",ct.c_uint32),("buf",ct.c_void_p),("size",ct.c_uint32),
               ("pass_through",ct.c_uint8),("type",ct.c_int),("fmt",ct.c_int)]


class Output(ct.Structure):
    _fields_ = [("want_float",ct.c_uint8),("is_prealloc",ct.c_uint8),
               ("index",ct.c_uint32),("buf",ct.c_void_p),("size",ct.c_uint32)]


class RunExtend(ct.Structure):
    _fields_ = [("frame_id",ct.c_uint64),("non_block",ct.c_int32),
               ("timeout_ms",ct.c_int32),("fence_fd",ct.c_int32)]


class OutputExtend(ct.Structure):
    _fields_ = [("frame_id",ct.c_uint64)]


def checked(code, operation):
    if code:
        raise RuntimeError(operation + " returned " + str(code))


ABI = SimpleNamespace(Input=Input, Output=Output, RunExtend=RunExtend,
                      OutputExtend=OutputExtend, check_call=checked)


def contract(degree=2, patches=2):
    shapes={"net":(1,degree,patches,384),"inp":(1,degree,patches,384),
            "corr":(1,degree,patches,882),"net_out":(1,degree,patches,384),
            "delta":(1,degree,patches,2),"weight":(1,degree,patches,2)}
    return ([dict(name=k,dtype="float32",shape=list(shapes[k])) for k in io.INPUT_NAMES],
            [dict(name=k,dtype="float32",shape=list(shapes[k])) for k in io.OUTPUT_NAMES])


def attrs(rows, fmt=3):
    result=[]
    for i,r in enumerate(rows):
        shape=r["shape"]
        if fmt==1:shape=[shape[0],shape[2],shape[3],shape[1]]
        elems=int(np.prod(shape))
        result.append(dict(index=i,name=r["name"],shape=shape,n_elems=elems,size=elems*2,
            format=fmt,type=1,quantization_type=2,scale=1.,zero_point=0,fractional_length=0,
            size_with_stride=elems*2,w_stride=0,h_stride=0,pass_through=0))
    return result


def feed(rows):
    return {r["name"]:np.arange(np.prod(r["shape"]),dtype=np.float32).reshape(r["shape"])/1000
            for r in rows}


class FakeAPI:
    def __init__(self, outputs, fault=None):
        self.values={r["name"]:np.full(r["shape"],.5 if r["name"]=="weight" else .125,
                                     dtype=np.float32) for r in outputs}
        self.expected={k:v.copy() for k,v in self.values.items()}
        self.buffers=[];self.calls=[];self.fault=fault
    def rknn_inputs_set(self, ctx, count, requests):
        self.calls.append("set")
        assert count==3
        self.requests=[dict(fmt=int(r.fmt),type=int(r.type),size=int(r.size),pass_through=int(r.pass_through))
                       for r in requests]
        self.input_copies=[np.ctypeslib.as_array(ct.cast(r.buf,ct.POINTER(ct.c_float)),
                                             shape=(r.size//4,)).copy() for r in requests]
        return -1 if self.fault=="set" else 0
    def rknn_run(self,ctx,extend):
        self.calls.append("run")
        assert extend._obj.non_block==0 and extend._obj.fence_fd==-1
        extend._obj.frame_id=27
        if self.fault=="native_log":os.write(2,b"E RKNN synthetic normalization failure\n")
        return -1 if self.fault=="run" else 0
    def rknn_outputs_get(self,ctx,count,outputs,extend):
        self.calls.append("get")
        if self.fault=="get":return -1
        self.buffers=[v.copy() for v in self.values.values()]
        for i,v in enumerate(self.buffers):
            assert outputs[i].want_float==1 and outputs[i].is_prealloc==0
            outputs[i].buf=v.ctypes.data;outputs[i].size=v.nbytes
        if self.fault=="size":outputs[1].size-=4
        extend._obj.frame_id=28 if self.fault=="frame" else 27
        return 0
    def rknn_outputs_release(self,ctx,count,outputs):
        self.calls.append("release")
        for v in self.buffers:v.fill(-999)
        return -1 if self.fault=="release" else 0
    def rknn_destroy(self,ctx):
        self.calls.append("destroy")
        return -1 if self.fault=="destroy" else 0


def session(fault=None, fmt=3, **kwargs):
    ins,outs=contract();api=FakeAPI(outs,fault)
    return io.SourceFrameSession(ABI,api,ct.c_uint64(1),ins,outs,attrs(ins,fmt),attrs(outs),**kwargs),api,ins,outs


@pytest.mark.parametrize("fmt",[0,1,3])
def test_actual_layout_controls_buffer_and_request(fmt):
    ins,outs=contract();x=feed(ins)
    bindings=io.validate_input_attrs(ins,outs,attrs(ins,fmt))
    packed=io.pack_inputs(ABI,x,bindings)
    for r,b,row in zip(packed.requests,packed.owned_buffers,ins):
        expected=x[row["name"]].transpose(0,2,3,1) if fmt==1 else x[row["name"]]
        assert np.array_equal(b,expected) and b.flags.c_contiguous
        assert r.fmt==fmt and r.type==0 and r.pass_through==0 and r.buf==b.ctypes.data
        assert not np.shares_memory(b,x[row["name"]])


@pytest.mark.parametrize("bad",["shape","name","type","scale","bytes","stride","output_NHWC"])
def test_unsupported_native_metadata_rejected(bad):
    ins,outs=contract();a=attrs(ins)
    if bad=="output_NHWC":
        with pytest.raises(ValueError,match="layout"):io.validate_output_attrs(ins,outs,attrs(outs,1))
        return
    if bad=="shape":a[0]["shape"][2]+=1
    if bad=="name":a[0]["name"]="other"
    if bad=="type":a[0]["type"]=8
    if bad=="scale":a[0]["scale"]=.5
    if bad=="bytes":a[0]["size"]+=2
    if bad=="stride":a[0]["size_with_stride"]+=2
    with pytest.raises(ValueError):io.validate_input_attrs(ins,outs,a)


def test_success_owns_output_after_release_and_close_idempotent():
    s,api,ins,_=session(fmt=1)
    values,r=s.execute(feed(ins),api.expected)
    assert r["execution_status"]=="completed" and r["strict_component_gate_passed"]
    assert not r["candidate_accepted"]
    assert r["strict_FP32_gate_passed"] and r["outputs_released"]
    assert r["timing"]["outputs_release_seconds"]>=0 and r["total_execute_seconds"]>=r["timing"]["forward_seconds"]
    for k,v in values.items():assert np.array_equal(v,api.expected[k]) and not np.any(v==-999)
    s.close();s.close()
    assert api.calls==["set","run","get","release","destroy"]
    m=s.metrics();assert m["synthetic"] and not m["npu_used"] and m["context_destroyed"]
    m["cleanup_errors"].append("external mutation");assert not s.metrics()["cleanup_errors"]


@pytest.mark.parametrize("fault",["set","run","get","frame","size","release"])
def test_protocol_failure_releases_when_acquired_and_destroys(fault):
    s,api,ins,_=session(fault)
    values,r=s.execute(feed(ins),api.expected)
    assert r["execution_status"]=="failed" and not r["candidate_accepted"]
    assert api.calls.count("release")==int(fault in ["frame","size","release"])
    assert api.calls.count("destroy")==1 and s.metrics()["closed"]


@pytest.mark.parametrize("kind",["nan","negative_weight","strict","zero"])
def test_invalid_numeric_raw_values_preserved_without_json_nan(kind):
    s,api,ins,_=session()
    if kind=="nan":api.values["net_out"].flat[0]=np.nan
    if kind=="negative_weight":api.values["weight"].flat[0]=-1e-7;api.expected["weight"].flat[0]=0
    if kind=="strict":api.values["net_out"].flat[0]+=.001
    if kind=="zero":
        for v in api.values.values():v.fill(0)
    values,r=s.execute(feed(ins),api.expected)
    assert r["execution_status"]=="completed" and not r["candidate_accepted"]
    assert values and not s.metrics()["closed"]
    json.dumps(r,allow_nan=False);s.close()


def test_nonfinite_input_never_calls_run():
    s,api,ins,_=session();x=feed(ins);x["corr"].flat[0]=np.inf
    values,r=s.execute(x,api.expected)
    assert not values and r["execution_status"]=="failed" and api.calls==["destroy"]


def test_native_error_logs_override_success_codes_and_restore_fds(tmp_path):
    s,api,ins,_=session("native_log",capture_logs=True,log_directory=tmp_path)
    old=[os.fstat(i) for i in [1,2]]
    values,r=s.execute(feed(ins),api.expected)
    new=[os.fstat(i) for i in [1,2]]
    assert [(x.st_dev,x.st_ino) for x in old]==[(x.st_dev,x.st_ino) for x in new]
    assert values and r["execution_status"]=="failed" and r["native_log"]["RKNN_error_detected"]
    assert s.metrics()["context_destroyed"] and not r["candidate_accepted"]


def manifest_file(tmp_path, *, strict=True):
    ins,outs=contract(10,96);model=tmp_path/"model.onnx";model.write_bytes(b"synthetic-not-a-model")
    source=tmp_path/"source.py";source.write_text("# synthetic source\n")
    pins={str(source):io.sha256(source)}
    m=dict(schema=io.MANIFEST_SCHEMA,status=io.MANIFEST_STATUS,
        original_parameters_unchanged=True,sources_and_inputs_unchanged=True,cleanup_errors=[],
        candidate_accepted=False,strict_FP32_gate_passed=strict,outputs_finite=True,weight_domain_valid=True,
        inputs=ins,outputs=outs,source_pins=pins,source_hashes_after=pins,cases=[{"name":"tiny"}],
        model=dict(file=model.name,bytes=model.stat().st_size,sha256=io.sha256(model)))
    path=tmp_path/"export.json";path.write_text(json.dumps(m));return m,path


def test_manifest_host_declaration_board_rehash_and_explicit_rejection(tmp_path):
    m,path=manifest_file(tmp_path,strict=False)
    with pytest.raises(ValueError,match="diagnostic flag"):io.validate_manifest(path,io.sha256(path))
    parsed,model,pins=io.validate_manifest(path,io.sha256(path),allow_fp32_rejected_diagnostic=True,
                                        verify_source_files=True)
    assert parsed==m and len(pins)==3 and model.name=="model.onnx"
    # Host compiler cannot read board /mnt paths, but records exact declarations.
    source="/mnt/synthetic-board-only/source.py";m["source_pins"]={source:"0"*64};m["source_hashes_after"]=m["source_pins"]
    path.write_text(json.dumps(m))
    io.validate_manifest(path,io.sha256(path),allow_fp32_rejected_diagnostic=True)
    with pytest.raises(FileNotFoundError):io.validate_manifest(path,io.sha256(path),allow_fp32_rejected_diagnostic=True,verify_source_files=True)


@pytest.mark.parametrize("bad",["finite","domain","source","escape","dtype","D","M"])
def test_manifest_rejects_invalid_provenance_or_contract(tmp_path,bad):
    m,p=manifest_file(tmp_path)
    if bad=="finite":m["outputs_finite"]=False
    if bad=="domain":m["weight_domain_valid"]=False
    if bad=="source":m["source_hashes_after"]={}
    if bad=="escape":m["model"]["file"]="../model.onnx"
    if bad=="dtype":m["inputs"][0]["dtype"]="float16"
    if bad=="D":m["inputs"][0]["shape"][1]=26
    if bad=="M":m["inputs"][0]["shape"][2]=97
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError):io.validate_manifest(p,io.sha256(p))


class FakeCompiler:
    def __init__(self,fail=None):self.calls=[];self.fail=fail
    def config(self,**kw):self.calls.append(("config",kw));return -1 if self.fail=="config" else 0
    def load_onnx(self,**kw):self.calls.append(("load_onnx",kw));return -1 if self.fail=="load_onnx" else 0
    def build(self,**kw):self.calls.append(("build",kw));return -1 if self.fail=="build" else 0
    def export_rknn(self,path):self.calls.append(("export_rknn",path));return -1 if self.fail=="export_rknn" else 0


@pytest.mark.parametrize("failure",[None,"config","load_onnx","build","export_rknn"])
def test_compiler_exact_original_config_and_stops_on_failure(tmp_path,failure):
    fake=FakeCompiler(failure);ops=[]
    if failure:
        with pytest.raises(RuntimeError):compiler.compile_graph(fake,tmp_path/"model.onnx",tmp_path/"block.rknn",ops)
    else:compiler.compile_graph(fake,tmp_path/"model.onnx",tmp_path/"block.rknn",ops)
    assert fake.calls[0]==("config",dict(target_platform="rk3588",float_dtype="float16",optimization_level=3))
    assert not any("disable_rules" in str(kw) for _,kw in fake.calls)
    if any(n=="build" for n,_ in fake.calls):assert next(kw for n,kw in fake.calls if n=="build")==dict(do_quantization=False)
    assert [x["name"] for x in ops]==[x[0] for x in fake.calls]


@pytest.mark.parametrize("tamper",[None,"model","config","helper","release","rejection"])
def test_compiled_binary_provenance_must_match_export(tmp_path,tamper):
    m,path=manifest_file(tmp_path,strict=False)
    artifact=tmp_path/"block.rknn";artifact.write_bytes(b"synthetic-not-an-RKNN")
    compiler_path=HERE/"compile-v33-source-frame-update.py"
    cp=io.sha256(compiler_path)
    p=tmp_path/"compile.json"
    sources={str(path):io.sha256(path)}
    report=dict(schema="v33_source_frame_Update_RKNN_compilation",
        status="compiled_source_frame_diagnostic_not_runtime_validated",
        script_sha256=cp,io_helper_sha256=io.sha256(HERE/"v33_rknn_source_frame.py"),
        manifest_sha256=io.sha256(path),input={"sha256":m["model"]["sha256"]},
        configuration=dict(io.CONFIGURATION),released=True,sources_and_inputs_unchanged=True,
        source_pins=sources,source_hashes_after=sources,export_strict_FP32_gate_passed=False,
        candidate_accepted=False,allow_fp32_rejected_diagnostic=True,
        output=dict(file=artifact.name,bytes=artifact.stat().st_size,sha256=io.sha256(artifact)))
    if tamper=="model":report["input"]["sha256"]="0"*64
    if tamper=="config":report["configuration"]["optimization_level"]=2
    if tamper=="helper":report["io_helper_sha256"]="0"*64
    if tamper=="release":report["released"]=False
    if tamper=="rejection":report["allow_fp32_rejected_diagnostic"]=False
    p.write_text(json.dumps(report))
    kwargs=dict(compiler_path=compiler_path,compiler_sha256=cp)
    if tamper:
        with pytest.raises(ValueError):io.validate_compilation(p,io.sha256(p),m,io.sha256(path),**kwargs)
    else:
        got,model,pins=io.validate_compilation(p,io.sha256(p),m,io.sha256(path),**kwargs)
        assert got==report and model==artifact and len(pins)==3

