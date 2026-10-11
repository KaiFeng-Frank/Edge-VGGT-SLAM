"""Synthetic ctypes/metadata tests only: no SDK library or model execution."""
import copy
import ctypes as ct
import importlib.util
import os
from pathlib import Path

import pytest

ENTRY = Path(__file__).parent / "query-v36-int8-io-board.py"
if not ENTRY.exists():
    ENTRY = Path(__file__).parents[1] / "deploy/rk3588/validation/board_ptq/query-v36-int8-io-board.py"
spec = importlib.util.spec_from_file_location("v36_io_query_tests", ENTRY)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
ABI = Path(os.environ.get("RKNN_ABI_HELPER", str(ENTRY.parent.parent / "run-v12-rknn-block-board.py")))
if not ABI.exists():
    ABI = Path(__file__).parents[1] / "deploy/rk3588/validation/run-v12-rknn-block-board.py"


def descriptor(index, name, shape, dtype, fmt=0, qnt=0, scale=0., zp=0):
    elements = 1
    for dim in shape:
        elements *= dim
    return dict(index=index, name=name, shape=list(shape), n_elems=elements,
        size=elements * helper.DTYPES[dtype][1], size_with_stride=elements * helper.DTYPES[dtype][1],
        format=fmt, type=dtype, quantization_type=qnt, scale=scale, zero_point=zp)


@pytest.mark.parametrize("machine,system,compatible", [
    ("x86_64", "Linux", b"rockchip,rk3588\0"),
    ("aarch64", "Darwin", b"rockchip,rk3588\0"),
    ("aarch64", "Linux", b"vendor,other-arm\0")])
def test_board_guard_rejects_non_board(machine, system, compatible):
    with pytest.raises(ValueError):
        helper.board_guard(machine, system, compatible)


def test_board_guard_accepts_both_variants():
    for variant in (b"rockchip,rk3588\0", b"rockchip,rk3588s\0"):
        helper.board_guard("aarch64", "Linux", variant)


def test_native_affine_and_nonspatial_integer_contracts():
    row = descriptor(0, "normalized_bgr", (1, 464, 608, 3), 2, 1, 2, .0125, -37)
    result = helper.validate_attr(row, 0, "normalized_bgr", (1, 3, 464, 608), spatial=True)
    assert result["dtype"] == "int8" and result["native_shape"] == [1, 464, 608, 3]
    index = descriptor(1, "patch_indices", (96,), 8, 3)
    assert helper.validate_attr(index, 1, "patch_indices", (96,), integer=True)["integer_index_cast_performed"] is False


@pytest.mark.parametrize("mutation", ["index_type", "index_scale", "packed", "scale", "zp", "shape"])
def test_bad_native_contracts_fail_closed(mutation):
    if mutation.startswith("index"):
        row = descriptor(1, "patch_indices", (96,), 8, 3)
        if mutation == "index_type":
            row["type"] = 0
        else:
            row.update(quantization_type=2, scale=.5)
        args = (1, "patch_indices", (96,))
        options = dict(integer=True)
    else:
        row = descriptor(0, "normalized_bgr", (1, 464, 608, 3), 2, 1, 2, .0125, -37)
        if mutation == "packed":
            row["format"] = 2
        elif mutation == "scale":
            row["scale"] = 0
        elif mutation == "zp":
            row["zero_point"] = 128
        else:
            row["shape"] = [1, 3, 464, 608]
        args, options = (0, "normalized_bgr", (1, 3, 464, 608)), dict(spatial=True)
    with pytest.raises(ValueError):
        helper.validate_attr(row, *args, **options)


@pytest.fixture
def abi():
    if not ABI.exists():
        pytest.skip("requires pinned RKNN ABI helper; set RKNN_ABI_HELPER for isolated tests")
    assert helper.sha(ABI) == helper.ABI_SHA
    return helper.load_abi(ABI)


class FakeAPI:
    def __init__(self, abi, init_code=0, query_code=0, destroy_code=0):
        self.abi = abi
        self.init_code, self.query_code, self.destroy_code = init_code, query_code, destroy_code
        self.calls = []
        self.inputs = [descriptor(0, "normalized_bgr", (1, 464, 608, 3), 2, 1, 2, .02, -30),
                       descriptor(1, "patch_indices", (96,), 8, 3)]
        self.outputs = [descriptor(0, "fmap", (1, 128, 116, 152), 2, 0, 2, .03, 0),
                        descriptor(1, "context", (1, 96, 384), 1, 3)]

    def rknn_init(self, pointer, filename, size, flags, extension):
        self.calls.append("init")
        assert size == flags == 0 and extension is None
        ct.cast(pointer, ct.POINTER(self.abi.CONTEXT)).contents.value = 17
        return self.init_code

    def rknn_query(self, context, command, pointer, size):
        self.calls.append(("query", command))
        if self.query_code:
            return self.query_code
        if command == 5:
            row = ct.cast(pointer, ct.POINTER(self.abi.SDKVersion)).contents
            row.api_version, row.drv_version = b"2.3.2", b"synthetic"
        elif command == 0:
            row = ct.cast(pointer, ct.POINTER(self.abi.InputOutputNum)).contents
            row.n_input, row.n_output = 2, 2
        elif command in (1, 2):
            row = ct.cast(pointer, ct.POINTER(self.abi.TensorAttr)).contents
            expected = (self.inputs if command == 1 else self.outputs)[row.index]
            row.name, row.n_dims = expected["name"].encode(), len(expected["shape"])
            for i, value in enumerate(expected["shape"]):
                row.dims[i] = value
            for key, field in (("n_elems", "n_elems"), ("size", "size"), ("format", "fmt"),
                               ("type", "type"), ("quantization_type", "qnt_type"),
                               ("scale", "scale"), ("zero_point", "zp"), ("size_with_stride", "size_with_stride")):
                setattr(row, field, expected[key])
        elif command == 6:
            ct.cast(pointer, ct.POINTER(self.abi.MemSize)).contents.total_dma_allocated_size = 1024
        else:
            raise AssertionError("Unexpected query command")
        return 0

    def rknn_destroy(self, context):
        self.calls.append("destroy")
        return self.destroy_code

    def __getattr__(self, name):
        raise AssertionError("Probe must not access inference method: " + name)


def test_probe_queries_only_and_destroys(abi):
    api, report = FakeAPI(abi), {"cleanup_errors": []}
    helper.probe_sdk(api, abi, Path("synthetic-model-never-read.rknn"), report)
    assert report["native_IO_contract_checked"] and report["context_destroyed"]
    assert api.calls == ["init", ("query", 5), ("query", 0), ("query", 1), ("query", 1),
                         ("query", 2), ("query", 2), ("query", 6), "destroy"]


@pytest.mark.parametrize("init_code,query_code", [(-7, 0), (0, -11)])
def test_failure_preserves_error_and_destroys(abi, init_code, query_code):
    api, report = FakeAPI(abi, init_code, query_code, destroy_code=-19), {"cleanup_errors": []}
    with pytest.raises(RuntimeError, match="returned " + str(init_code or query_code)):
        helper.probe_sdk(api, abi, "synthetic.rknn", report)
    assert api.calls[-1] == "destroy"
    assert report["context_destroyed"] is False and len(report["cleanup_errors"]) == 1


def test_destroy_failure_is_not_reported_success(abi):
    api, report = FakeAPI(abi, destroy_code=-19), {"cleanup_errors": []}
    with pytest.raises(RuntimeError, match="rknn_destroy returned -19"):
        helper.probe_sdk(api, abi, "synthetic.rknn", report)
    assert not report["context_destroyed"] and report["cleanup_errors"]


def test_main_host_guard_precedes_all_model_reads(monkeypatch, tmp_path):
    monkeypatch.setattr(helper.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(helper, "sha", lambda path: pytest.fail("Source/model read before board guard"))
    with pytest.raises(ValueError, match="board only"):
        helper.main(["--model", "missing.rknn", "--model-sha256", "0"*64, "--compile-json", "missing.json",
                     "--compile-sha256", "0"*64, "--abi-helper", "missing.py", "--header", "missing.h",
                     "--output-dir", str(tmp_path / "never-created")])
    assert not (tmp_path / "never-created").exists()
