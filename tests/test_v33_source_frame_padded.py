"""Synthetic output-buffer protocol tests; no SDK/model/hardware is loaded."""
import ctypes as ct
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


HERE = Path(__file__).resolve().parents[1] / "deploy" / "rk3588" / "validation" / "dpvo_source_frame"
spec = importlib.util.spec_from_file_location(
    "v33_padded_synthetic_io", HERE / "v33_rknn_source_frame_padded.py")
io = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = io
spec.loader.exec_module(io)


class Input(ct.Structure):
    _fields_ = [("index", ct.c_uint32), ("buf", ct.c_void_p), ("size", ct.c_uint32),
               ("pass_through", ct.c_uint8), ("type", ct.c_int), ("fmt", ct.c_int)]


class Output(ct.Structure):
    _fields_ = [("want_float", ct.c_uint8), ("is_prealloc", ct.c_uint8),
               ("index", ct.c_uint32), ("buf", ct.c_void_p), ("size", ct.c_uint32)]


class RunExtend(ct.Structure):
    _fields_ = [("frame_id", ct.c_uint64), ("non_block", ct.c_int32),
               ("timeout_ms", ct.c_int32), ("fence_fd", ct.c_int32)]


class OutputExtend(ct.Structure):
    _fields_ = [("frame_id", ct.c_uint64)]


def checked(code, operation):
    if code:
        raise RuntimeError(operation + " returned " + str(code))


ABI = SimpleNamespace(Input=Input, Output=Output, RunExtend=RunExtend,
                      OutputExtend=OutputExtend, check_call=checked)


def contract(degree=2, patches=3):
    inputs = [dict(name=n, dtype="float32", shape=[1, degree, patches, c])
              for n, c in zip(io.INPUT_NAMES, [384, 384, 882])]
    outputs = [dict(name=n, dtype="float32", shape=[1, degree, patches, c])
               for n, c in zip(io.OUTPUT_NAMES, [384, 2, 2])]
    return inputs, outputs


def attrs(rows, *, fmt=0, dtype=1, padded=False):
    result = []
    for index, row in enumerate(rows):
        b, d, m, c = row["shape"]
        shape = [b, m, c, d] if fmt == 1 else [b, d, m, c]
        n = int(np.prod(shape))
        size = n * (4 if dtype == 0 else 2)
        result.append(dict(index=index, name=row["name"], shape=shape, n_elems=n,
                           size=size, format=fmt, type=dtype, quantization_type=2,
                           scale=1.0, zero_point=0, fractional_length=0,
                           size_with_stride=size + 256 if padded else size,
                           w_stride=0, h_stride=0, pass_through=0))
    return result


class FakeAPI:
    def __init__(self, outputs, size_delta=0):
        self.reference = {r["name"]: np.full(r["shape"],
                          .5 if r["name"] == "weight" else .125, np.float32)
                          for r in outputs}
        self.size_delta = size_delta
        self.calls = []
        self.buffers = []

    def rknn_inputs_set(self, context, count, requests):
        self.calls.append("set")
        assert count == 3
        for r in requests:
            assert r.type == 0 and r.fmt == 1 and r.pass_through == 0 and r.buf
        return 0

    def rknn_run(self, context, extend):
        self.calls.append("run")
        assert extend._obj.non_block == 0
        extend._obj.frame_id = 123
        return 0

    def rknn_outputs_get(self, context, count, outputs, extend):
        self.calls.append("get")
        self.buffers = [a.copy() for a in self.reference.values()]
        for request, buffer in zip(outputs, self.buffers):
            assert request.want_float == 1 and request.is_prealloc == 0
            request.buf = buffer.ctypes.data
            request.size = buffer.nbytes
        outputs[0].size += self.size_delta
        extend._obj.frame_id = 123
        return 0

    def rknn_outputs_release(self, context, count, outputs):
        self.calls.append("release")
        for buffer in self.buffers:
            buffer.fill(-999)
        return 0

    def rknn_destroy(self, context):
        self.calls.append("destroy")
        return 0


def make_session(size_delta=0):
    ins, outs = contract()
    api = FakeAPI(outs, size_delta)
    session = io.SourceFrameSession(ABI, api, ct.c_uint64(1), ins, outs,
        attrs(ins, fmt=1), attrs(outs, padded=True))
    feed = {r["name"]: np.full(r["shape"], .25, np.float32) for r in ins}
    return session, api, feed


@pytest.mark.parametrize("fmt", [0, 3])
@pytest.mark.parametrize("dtype", [0, 1])
def test_only_output_internal_padding_allowed(fmt, dtype):
    ins, outs = contract()
    binding = io.validate_output_attrs(ins, outs, attrs(outs, fmt=fmt, dtype=dtype, padded=True))
    for row, original in zip(binding, outs):
        assert row["native_output_padding_bytes"] == 256
        assert row["sdk_size_with_stride"] == row["sdk_native_bytes"] + 256
        assert row["returned_FP32_bytes"] == int(np.prod(original["shape"])) * 4
        assert row["shape"] == tuple(original["shape"]) and row["permutation"] is None


def test_poison_release_does_not_modify_owned_logical_output():
    session, api, feed = make_session()
    values, record = session.execute(feed, api.reference)
    assert record["execution_status"] == "completed"
    assert record["strict_component_gate_passed"] and not record["candidate_accepted"]
    assert record["outputs_released"] and record["timing"]["outputs_release_seconds"] >= 0
    for key, value in values.items():
        assert np.array_equal(value, api.reference[key])
        assert value.flags.owndata or value.base.flags.owndata
        assert not any(np.shares_memory(value, buffer) for buffer in api.buffers)
    assert all(np.all(buffer == -999) for buffer in api.buffers)
    session.close()
    session.close()
    assert api.calls == ["set", "run", "get", "release", "destroy"]
    metrics = session.metrics()
    assert metrics["context_destroyed"] and not metrics["npu_used"]
    assert metrics["implementation_provenance"]["parent_sha256"] == io.PARENT_SHA256
    assert all(r["native_output_padding_bytes"] == 256 for r in metrics["output_bindings"])
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize("stride", [-1, 1, 737279, True, 1.5, 2**32])
def test_undersized_or_invalid_output_stride_rejected(stride):
    ins, outs = contract(10, 96)
    native = attrs(outs)
    native[0]["size_with_stride"] = stride
    with pytest.raises(ValueError, match="stride footprint"):
        io.validate_output_attrs(ins, outs, native)


@pytest.mark.parametrize("fmt", [0, 1, 3])
def test_input_padding_remains_rejected(fmt):
    ins, outs = contract()
    with pytest.raises(ValueError, match="Padded native byte"):
        io.validate_input_attrs(ins, outs, attrs(ins, fmt=fmt, padded=True))


@pytest.mark.parametrize("delta", [-4, 4, 256])
def test_actual_returned_byte_count_still_exact_and_released(delta):
    session, api, feed = make_session(delta)
    values, record = session.execute(feed, api.reference)
    assert record["execution_status"] == "failed" and "bytes/pointer" in record["error"]
    assert not values and record["outputs_released"] and not record["candidate_accepted"]
    assert api.calls == ["set", "run", "get", "release", "destroy"]
    assert session.metrics()["context_destroyed"]


def test_unimplemented_output_nhwc_is_not_enabled():
    ins, outs = contract()
    with pytest.raises(ValueError, match="layout"):
        io.validate_output_attrs(ins, outs, attrs(outs, fmt=1, padded=True))


def test_observed_D10_M96_query_metadata_no_sdk_execution():
    ins, outs = contract(10, 96)
    native_ins = attrs(ins, fmt=1)
    native_outs = attrs(outs)
    for row, stride in zip(native_outs, [1179648, 24576, 24576]):
        row["size_with_stride"] = stride
    input_bindings = io.validate_input_attrs(ins, outs, native_ins)
    output_bindings = io.validate_output_attrs(ins, outs, native_outs)
    assert [r["transport_format"] for r in input_bindings] == [1, 1, 1]
    assert [r["native_output_padding_bytes"] for r in output_bindings] == [442368, 20736, 20736]
    assert [r["returned_FP32_bytes"] for r in output_bindings] == [1474560, 7680, 7680]


@pytest.mark.parametrize("stride", [0, None])
def test_parent_unspecified_stride_semantics_preserved(stride):
    ins, outs = contract()
    native = attrs(outs)
    for row in native:
        if stride is None:
            row.pop("size_with_stride")
        else:
            row["size_with_stride"] = stride
    assert all(r["native_output_padding_bytes"] == 0
               for r in io.validate_output_attrs(ins, outs, native))


def test_frozen_parent_pin_rejects_modified_copy(tmp_path, monkeypatch):
    pins = {}
    path = io.pin_parent_source(pins)
    assert pins[str(path)] == io.PARENT_SHA256
    fake_helper = tmp_path / "padded.py"
    fake_helper.write_text("# synthetic current helper\n")
    fake_parent = tmp_path / io.PARENT_NAME
    fake_parent.write_text("# modified parent\n")
    monkeypatch.setattr(io, "__file__", str(fake_helper))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        io.pin_parent_source({})


def test_compilation_binds_parent_and_new_runtime_separately(tmp_path):
    compiler = tmp_path / "compiler.py"
    compiler.write_text("# synthetic compiler\n")
    artifact = tmp_path / "block.rknn"
    artifact.write_bytes(b"not-a-real-model")
    manifest = dict(model=dict(sha256="1" * 64), strict_FP32_gate_passed=True)
    record = dict(schema="v33_source_frame_Update_RKNN_compilation",
        status="compiled_source_frame_diagnostic_not_runtime_validated",
        script_sha256=io.sha256(compiler), io_helper_sha256=io.PARENT_SHA256,
        manifest_sha256="2" * 64, input=dict(sha256="1" * 64),
        configuration=io.CONFIGURATION, released=True, sources_and_inputs_unchanged=True,
        source_pins={}, source_hashes_after={}, export_strict_FP32_gate_passed=True,
        candidate_accepted=False, output=dict(file=artifact.name,
        bytes=artifact.stat().st_size, sha256=io.sha256(artifact)))
    path = tmp_path / "compile.json"
    path.write_text(json.dumps(record))
    _, model, pins = io.validate_compilation(path, io.sha256(path), manifest, "2" * 64,
        compiler_path=compiler, compiler_sha256=io.sha256(compiler))
    assert model == artifact
    assert pins[str(HERE / io.PARENT_NAME)] == io.PARENT_SHA256
    assert pins[str(HERE / "v33_rknn_source_frame_padded.py")] == io.sha256(io.__file__)
    record["io_helper_sha256"] = io.sha256(io.__file__)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="implementation chain"):
        io.validate_compilation(path, io.sha256(path), manifest, "2" * 64,
            compiler_path=compiler, compiler_sha256=io.sha256(compiler))
