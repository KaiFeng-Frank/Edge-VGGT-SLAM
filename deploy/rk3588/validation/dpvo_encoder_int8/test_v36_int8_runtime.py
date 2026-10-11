"""Synthetic typed transport/lifecycle tests, never a real model or library."""
import ctypes as ct
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

DIRECTORY = Path(__file__).parent
ENTRY = DIRECTORY / "run-v36-dpvo-int8-board.py"
spec = importlib.util.spec_from_file_location("v36_ptq_runtime_test", ENTRY)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


@pytest.fixture
def dependencies():
    abi_path = Path(os.environ.get("RKNN_ABI_HELPER", str(DIRECTORY / "run-v12-rknn-block-board.py")))
    transport_path = Path(os.environ.get("V34_TYPED_TRANSPORT", str(DIRECTORY / "v34_sparse_rknn.py")))
    query_path = Path(os.environ.get("V36_IO_QUERY_HELPER", str(DIRECTORY.parent / "toolchain/query-v36-int8-io-board.py")))
    if not all(p.exists() for p in (abi_path, transport_path, query_path)):
        pytest.skip("requires pinned ABI/typed transport/query sources; set RKNN_ABI_HELPER, V34_TYPED_TRANSPORT, V36_IO_QUERY_HELPER")
    for p, digest in [(abi_path, runtime.ABI_SHA), (transport_path, runtime.TRANSPORT_SHA), (query_path, runtime.QUERY_SOURCE_SHA)]:
        assert runtime.sha(p) == digest
    return (runtime.load(abi_path, "_v36_runtime_fake_abi"), runtime.load(transport_path, "_v36_runtime_fake_transport"),
            runtime.load(query_path, "_v36_runtime_fake_query"))


def test_schedule_has_no_hidden_warmup():
    assert runtime.schedule(False) == [(0, "cold"), (0, "warm"), (0, "warm"), (0, "warm"), (1, "other_frame"), (2, "other_frame")]
    assert runtime.schedule(True) == [(0, "profile"), (1, "profile"), (2, "profile")]


def test_fp32_input_pack_is_nhwc_copy_not_quantization():
    image = np.zeros((1, 3, 464, 608), np.float32)
    image[0, 0], image[0, 1], image[0, 2] = -.5, .25, 1.5
    packed = runtime.pack_input(image, 1)
    assert packed.shape == (1, 464, 608, 3) and packed.dtype == np.float32 and packed.flags.c_contiguous
    np.testing.assert_array_equal(packed[0, 0, 0], [-.5, .25, 1.5])
    assert not np.shares_memory(packed, image)


def test_strict_gate_is_not_relaxed_for_ptq():
    expected = np.array([0, 1], np.float32)
    assert runtime.comparison(expected.copy(), expected)["gate_passed"]
    changed = np.array([.00002, 1], np.float32)
    row = runtime.comparison(changed, expected)
    assert row["failed_elements"] == 1 and not row["gate_passed"]
    positive_zero, negative_zero = np.array([0.], np.float32), np.array([-0.], np.float32)
    row = runtime.comparison(positive_zero, negative_zero)
    assert row["array_equal"] and not row["bitwise_equal"]


def test_host_guard_precedes_model_or_source_read(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(runtime, "sha", lambda p: pytest.fail("read before board guard"))
    argv = []
    for name in ("compile-dir", "query-dir", "query-helper", "capture-dir", "reference-dir", "transport-helper", "abi-helper", "header", "output-dir"):
        argv += ["--" + name, str(tmp_path / name)]
    for name in ("compile", "query", "capture", "reference"):
        argv += ["--" + name + "-sha256", "0"*64]
    with pytest.raises(ValueError, match="board only"):
        runtime.main(argv)
    assert list(tmp_path.iterdir()) == []


def attributes(abi):
    specs = [(0, "normalized_bgr", (1, 464, 608, 3), 1, 2, .0078125, -64),
             (1, "patch_indices", (96,), 3, 8, 1., 0),
             (0, "fmap", (1, 128, 116, 152), 0, 2, .0122, 8),
             (1, "context", (1, 96, 384), 3, 2, .0285, -6)]
    result = []
    for index, name, shape, fmt, dtype, scale, zp in specs:
        attr = abi.TensorAttr(index=index)
        attr.name, attr.n_dims, attr.fmt, attr.type = name.encode(), len(shape), fmt, dtype
        for i, value in enumerate(shape):
            attr.dims[i] = value
        attr.n_elems = int(np.prod(shape))
        attr.size = attr.n_elems * (8 if dtype == 8 else 1)
        attr.size_with_stride = attr.size
        attr.qnt_type, attr.scale, attr.zp = 2, scale, zp
        result.append(attr)
    return result


class NativeAPI:
    def __init__(self, abi, init_code=0):
        self.abi, self.init_code, self.calls = abi, init_code, []
        self.attrs = attributes(abi)

    def rknn_init(self, pointer, filename, size, flags, extension):
        assert size == 0 and flags in (0, 8)
        ct.cast(pointer, ct.POINTER(self.abi.CONTEXT)).contents.value = 9
        self.calls.append("init")
        return self.init_code

    def rknn_set_core_mask(self, context, mask):
        assert mask == 7
        self.calls.append("coremask")
        return 0

    def rknn_query(self, context, command, pointer, size):
        self.calls.append(("query", command))
        if command == 5:
            row = ct.cast(pointer, ct.POINTER(self.abi.SDKVersion)).contents
            row.api_version, row.drv_version = b"2.3.2 synthetic", b"synthetic"
        elif command == 0:
            row = ct.cast(pointer, ct.POINTER(self.abi.InputOutputNum)).contents
            row.n_input = row.n_output = 2
        elif command in (1, 2):
            row = ct.cast(pointer, ct.POINTER(self.abi.TensorAttr)).contents
            attr = self.attrs[row.index + (2 if command == 2 else 0)]
            ct.memmove(pointer, ct.byref(attr), size)
        elif command == 6:
            ct.cast(pointer, ct.POINTER(self.abi.MemSize)).contents.total_dma_allocated_size = 1024
        else:
            raise AssertionError("not a query contract")
        return 0

    def rknn_destroy(self, context):
        self.calls.append("destroy")
        return 0


def proof(api, abi):
    return dict(sdk_versions=dict(api="2.3.2 synthetic", driver="synthetic"),
                input_attrs=[abi.attr_record(v) for v in api.attrs[:2]],
                output_attrs=[abi.attr_record(v) for v in api.attrs[2:]])


def test_actual_int8_context_guard_and_idempotent_close(dependencies):
    abi, _, query = dependencies
    api, report = NativeAPI(abi), dict(context={}, cleanup_errors=[])
    context = runtime.PTQContext(api, abi, query, proof(api, abi), "synthetic.rknn", False, report)
    assert context.index_attr["type"] == 8 and report["context"]["output_attrs"][0]["type"] == 2
    context.close()
    context.close()
    assert api.calls.count("destroy") == 1 and report["context"]["context_destroyed"]


def test_init_failure_destroys_and_preserves_primary_error(dependencies):
    abi, _, query = dependencies
    api, report = NativeAPI(abi, init_code=-17), dict(context={}, cleanup_errors=[])
    with pytest.raises(RuntimeError, match="rknn_init returned -17"):
        runtime.PTQContext(api, abi, query, proof(api, abi), "synthetic.rknn", False, report)
    assert api.calls == ["init", "destroy"]


def test_changed_native_descriptor_cannot_fall_back(dependencies):
    abi, _, query = dependencies
    api, report = NativeAPI(abi), dict(context={}, cleanup_errors=[])
    expected = proof(api, abi)
    api.attrs[1].type = 0
    with pytest.raises(ValueError, match="no dtype/layout fallback"):
        runtime.PTQContext(api, abi, query, expected, "synthetic.rknn", False, report)
    assert api.calls[-1] == "destroy"


class TransferAPI:
    def __init__(self, get_code=0, release_code=0, wrong_bytes=False):
        self.get_code, self.release_code, self.wrong_bytes = get_code, release_code, wrong_bytes
        self.buffers = [(ct.c_float*2)(1.25, 2.5), (ct.c_float*2)(3.75, 4.5)]
        self.calls = []

    def rknn_inputs_set(self, context, count, requests):
        self.calls.append("set")
        assert count == 2 and [v.type for v in requests] == [0, 8]
        assert [v.fmt for v in requests] == [1, 3] and [v.pass_through for v in requests] == [0, 0]
        image = np.ctypeslib.as_array(ct.cast(requests[0].buf, ct.POINTER(ct.c_float)), shape=(requests[0].size//4,))
        np.testing.assert_array_equal(image[:3], [-.5, .25, 1.5])
        indices = np.ctypeslib.as_array(ct.cast(requests[1].buf, ct.POINTER(ct.c_int64)), shape=(96,))
        assert np.array_equal(indices, np.full(96, 153, np.int64))
        return 0

    def rknn_run(self, context, extension):
        self.calls.append("run")
        return 0

    def rknn_outputs_get(self, context, count, outputs, extension):
        self.calls.append("get")
        for i, row in enumerate(outputs):
            assert row.want_float == 1 and row.is_prealloc == 0
            row.buf, row.size = ct.cast(self.buffers[i], ct.c_void_p).value, 16 if self.wrong_bytes else 8
        return self.get_code

    def rknn_outputs_release(self, context, count, outputs):
        self.calls.append("release")
        for buffer in self.buffers:
            buffer[0] = buffer[1] = 0
        return self.release_code


def transfer(dependencies, api, cleanup):
    abi, typed, _ = dependencies
    image = np.zeros((1, 3, 464, 608), np.float32)
    image[0, 0], image[0, 1], image[0, 2] = -.5, .25, 1.5
    indices = np.full(96, 153, np.int64)
    index = abi.attr_record(attributes(abi)[1])
    return typed.execute_sparse(api, abi, SimpleNamespace(pack_input=runtime.pack_input), abi.CONTEXT(9),
        image, indices, 1, index, cleanup, {}, outputs=(("fmap", (1, 1, 1, 2)), ("context", (1, 2, 1))))


def test_owned_fp32_copy_survives_release(dependencies):
    api, cleanup = TransferAPI(), []
    arrays, timing = transfer(dependencies, api, cleanup)
    assert api.calls == ["set", "run", "get", "release"] and cleanup == []
    np.testing.assert_array_equal(arrays["fmap"].ravel(), [1.25, 2.5])
    np.testing.assert_array_equal(arrays["context"].ravel(), [3.75, 4.5])
    assert timing["output_bytes"] == [8, 8] and timing["output_buffers_released"]
    assert timing["sdk_io_seconds"] >= sum(timing[k] for k in ["input_set_seconds", "run_api_seconds", "outputs_get_seconds", "output_copy_seconds", "outputs_release_seconds"])


def test_padded_output_bytes_rejected_but_released(dependencies):
    api, cleanup = TransferAPI(wrong_bytes=True), []
    with pytest.raises(ValueError, match="logical byte count"):
        transfer(dependencies, api, cleanup)
    assert api.calls[-1] == "release"


def test_get_error_and_release_error_keep_original_failure(dependencies):
    api, cleanup = TransferAPI(get_code=-7, release_code=-11), []
    with pytest.raises(RuntimeError, match="rknn_outputs_get returned -7"):
        transfer(dependencies, api, cleanup)
    assert api.calls[-1] == "release" and len(cleanup) == 1
