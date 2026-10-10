"""Host-only C-pointer/lifecycle tests, not XNNPACK hardware validation.

The emulated FP16 kernel rounds operands/output but accumulates in FP32.
Actual ARM FP16 accumulation error requires the separate native/model tests.
"""

from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
import hashlib
from pathlib import Path
import shutil
import subprocess
import threading
import time

import numpy as np
import pytest
import torch
from torch import nn

from vggt_slam import xnnpack_linear as xnn


class MockFunction:
    def __init__(self, implementation):
        self.implementation = implementation

    def __call__(self, *args):
        return self.implementation(*args)


class MockLibrary:
    """Emulates fixed packs and checks the exact C ABI buffer contract."""

    def __init__(self):
        self.operators = {}
        self.pool = None
        self.create_attempts = self.runs = self.initializes = 0
        self.fail_create_at = None
        self.fail_create_with_handle = False
        self.null_operator = self.fail_pool = self.nonfinite_output = self.overwrite_guard = False
        self.fail_reshape = self.fail_setup = self.fail_run = False
        self.fail_delete = False
        self.run_delay = 0.0
        self.active_runs = self.max_active_runs = 0
        self.events, self.batches, self.input_tails = [], [], []
        self.guard = threading.Lock()
        implementations = {
            "xnn_initialize": self.initialize,
            "xnn_create_fully_connected_nc_f16": lambda *a: self.create("float16", *a),
            "xnn_create_fully_connected_nc_f32": lambda *a: self.create("float32", *a),
            "xnn_reshape_fully_connected_nc_f16": self.reshape,
            "xnn_reshape_fully_connected_nc_f32": self.reshape,
            "xnn_setup_fully_connected_nc_f16": self.setup,
            "xnn_setup_fully_connected_nc_f32": self.setup,
            "xnn_run_operator": self.run,
            "xnn_delete_operator": self.delete,
            "pthreadpool_create": self.create_pool,
            "pthreadpool_get_threads_count": self.pool_threads,
            "pthreadpool_destroy": self.destroy_pool,
            # Existence alone must not tempt production into global cleanup.
            "xnn_deinitialize": lambda: pytest.fail("Must not deinitialize Torch's XNNPACK"),
        }
        for name, implementation in implementations.items():
            setattr(self, name, MockFunction(implementation))

    @staticmethod
    def key(pointer):
        return pointer.value if isinstance(pointer, ct.c_void_p) else pointer

    @staticmethod
    def view(pointer, count, dtype):
        dtype = np.dtype(dtype)
        address = ct.cast(pointer, ct.c_void_p).value
        raw = (ct.c_ubyte * (count * dtype.itemsize)).from_address(address)
        return np.frombuffer(raw, dtype=dtype)

    def initialize(self, allocator):
        assert allocator is None
        self.initializes += 1
        return 0

    def create_pool(self, threads):
        self.events.append("pool_create")
        if self.fail_pool:
            return None
        self.pool = {"handle": 9876, "threads": threads}
        return self.pool["handle"]

    def pool_threads(self, pool):
        assert self.key(pool) == self.pool["handle"]
        return self.pool["threads"]

    def destroy_pool(self, pool):
        assert not self.operators
        assert self.key(pool) == self.pool["handle"]
        self.events.append("pool_destroy")
        self.pool = None

    def create(self, dtype, k, n, stride_a, stride_c, kernel, bias, lower, upper,
               flags, code_cache, weights_cache, output):
        assert stride_a == k and stride_c == n
        assert bias is None and code_cache is None and weights_cache is None
        assert lower == -float("inf") and upper == float("inf")
        assert flags == (24 if dtype == "float16" else 16)
        # Flag 8's source W is float32 even when the packed GEMM is FP16.
        weight = self.view(kernel, k * n + 4, "float32")
        assert np.all(weight[k * n:] == 0)
        weight = weight[:k * n].reshape(n, k).astype(dtype, copy=True)
        self.create_attempts += 1
        ident = self.create_attempts
        failure = ident == self.fail_create_at
        if failure and not self.fail_create_with_handle:
            return 6
        if self.null_operator:
            return 0
        self.operators[ident] = {"dtype": dtype, "k": k, "n": n, "weight": weight}
        ct.cast(output, ct.POINTER(ct.c_void_p)).contents.value = ident
        self.events.append("operator_create")
        return 6 if failure else 0

    def reshape(self, operator, rows, pool):
        assert self.key(pool) == self.pool["handle"]
        if self.fail_reshape:
            return 4
        self.operators[self.key(operator)]["rows"] = rows
        self.batches.append(rows)
        return 0

    def setup(self, operator, inputs, outputs):
        if self.fail_setup:
            return 3
        state = self.operators[self.key(operator)]
        extra = xnn.XNN_EXTRA_BYTES // np.dtype(state["dtype"]).itemsize
        a = self.view(inputs, state["rows"] * state["k"] + extra, state["dtype"])
        assert np.all(a[-extra:] == 0)
        self.input_tails.append(a[-extra:].copy())
        state["inputs"], state["outputs"] = inputs, outputs
        return 0

    def run(self, operator, pool):
        self.runs += 1
        if self.fail_run:
            return 5
        with self.guard:
            self.active_runs += 1
            self.max_active_runs = max(self.max_active_runs, self.active_runs)
        try:
            if self.run_delay:
                time.sleep(self.run_delay)
            state = self.operators[self.key(operator)]
            rows, k, n, dtype = state["rows"], state["k"], state["n"], state["dtype"]
            a = self.view(state["inputs"], rows * k, dtype).reshape(rows, k).astype(np.float32)
            output = self.view(state["outputs"], rows * n, dtype).reshape(rows, n)
            output[:] = a @ state["weight"].astype(np.float32).T
            if self.nonfinite_output:
                output[0, 0] = np.nan
            if self.overwrite_guard:
                self.view(state["outputs"], rows * n + 1, dtype)[-1] = 0
            return 0
        finally:
            with self.guard:
                self.active_runs -= 1

    def delete(self, operator):
        self.operators.pop(self.key(operator))
        self.events.append("operator_delete")
        return 3 if self.fail_delete else 0


@pytest.fixture(autouse=True, scope="module")
def torch_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def install(model, library, **kwargs):
    return xnn.install_xnnpack_linear(model, library=library, **kwargs)


def reference(layer, inputs, dtype):
    if dtype == "float16":
        # Match the mock's documented FP32 summation implementation. A Torch
        # GEMM can sum in a different order and cross a Half rounding midpoint;
        # that would test BLAS choices rather than this wrapper's C ABI/IO.
        a = inputs.numpy().astype(np.float16).astype(np.float32).reshape(-1, layer.in_features)
        w = layer.weight.detach().numpy().astype(np.float16).astype(np.float32)
        output = torch.from_numpy((a @ w.T).astype(np.float16).astype(np.float32).reshape(
            *inputs.shape[:-1], layer.out_features))
    else:
        output = inputs @ layer.weight.detach().T
    return output + (layer.bias.detach() if layer.bias is not None else 0)


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("bias", [True, False])
def test_fixed_operator_tails_noncontiguous_no_output_alias_and_restore(dtype, bias):
    torch.manual_seed(31)
    model = nn.Sequential(nn.Linear(133, 129, bias=bias)).eval()
    original, original_weight = model[0], model[0].weight
    library = MockLibrary()
    handle = install(model, library, dtype=dtype)
    assert model[0].weight is original_weight
    retained_outputs = []
    with torch.inference_mode():
        for rows in (17, 3, 29, 1):
            inputs = torch.randn(2, 133, rows).transpose(1, 2)
            assert not inputs.is_contiguous() or rows == 1
            expected = reference(original, inputs, dtype)
            actual = model(inputs)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
            assert actual.dtype == torch.float32 and actual.device.type == "cpu"
            retained_outputs.append((actual, actual.clone()))
        for actual, saved in retained_outputs:
            torch.testing.assert_close(actual, saved, rtol=0, atol=0)
        assert len({item[0].data_ptr() for item in retained_outputs}) == 4
    metrics = handle.metrics()
    assert metrics["operator_creates"] == 1 and metrics["xnnpack_calls"] == 4
    assert metrics["precision"] == dtype and metrics["accumulator"] == dtype
    assert not metrics["abi_verified"] and metrics["library_is_mock"]
    assert not metrics["xnnpack_used"] and not metrics["npu_used"]
    assert len(library.input_tails) == 4 and library.batches == [34, 6, 58, 2]
    assert metrics["packed_weight_bytes_estimated"] == (133 * 129 + 129) * np.dtype(dtype).itemsize
    assert metrics["packing_seconds"] > 0 and metrics["scratch_bytes"] > 0
    handle.restore()
    assert model[0] is original and model[0].weight is original_weight
    assert library.operators  # Restore does not prematurely release owned ops.
    handle.close()
    handle.close()
    final = handle.metrics()
    assert final["closed"] and final["restored"]
    assert final["xnnpack_calls"] == 4 and final["operator_creates"] == final["operator_deletes"] == 1
    assert final["packed_weight_bytes_estimated"] == final["scratch_bytes"] == 0
    assert final["peak_packed_weight_bytes_estimated"] > 0 and final["peak_scratch_bytes"] > 0
    assert library.events[-2:] == ["operator_delete", "pool_destroy"]


def test_bias_added_in_float32_not_rounded_to_half():
    model = nn.Sequential(nn.Linear(128, 129)).eval()
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.fill_(0.123456789)
    original_bias = model[0].bias.detach().clone()
    with install(model, MockLibrary(), dtype="float16"), torch.inference_mode():
        actual = model(torch.zeros(1, 128))
        torch.testing.assert_close(actual[0], original_bias, rtol=0, atol=0)
        assert not torch.equal(actual[0], original_bias.half().float())


def test_scope_threshold_alias_and_different_weights_same_shape():
    torch.manual_seed(22)
    model = nn.Module()
    shared = nn.Linear(128, 128)
    model.aggregator = nn.Sequential(shared, shared, nn.Linear(128, 128), nn.Linear(128, 7))
    model.camera = nn.Linear(128, 128)
    model.eval()
    original_agg = list(model.aggregator)
    original_camera = model.camera
    library = MockLibrary()
    handle = install(model, library, scope="aggregator")
    assert model.aggregator[0] is model.aggregator[1]
    assert model.aggregator[3] is original_agg[3] and model.camera is original_camera
    with torch.inference_mode():
        inputs = torch.randn(7, 128)
        for index in (0, 2, 1, 2):
            torch.testing.assert_close(model.aggregator[index](inputs),
                reference(original_agg[index], inputs, "float32"), rtol=2e-5, atol=2e-6)
    metrics = handle.metrics()
    assert metrics["replaced_modules"] == 3 and metrics["operator_creates"] == 2
    assert set(metrics["skipped_modules"]) == {"camera", "aggregator.3"}
    handle.close()
    assert list(model.aggregator) == original_agg


def test_empty_and_vector_input_do_not_run_stale_buffers():
    model = nn.Sequential(nn.Linear(128, 129)).eval()
    original = model[0]
    with install(model, MockLibrary()) as handle, torch.inference_mode():
        inputs = torch.randn(128)
        torch.testing.assert_close(model(inputs), reference(original, inputs, "float32"), rtol=2e-5, atol=2e-6)
        assert model(torch.empty(2, 0, 128)).shape == (2, 0, 129)
        assert handle.metrics()["linear_calls"] == 2 and handle.metrics()["xnnpack_calls"] == 1


@pytest.mark.parametrize("mode", ["version", "original_parameter", "wrapper_parameter", "dtype", "bias"])
def test_parameter_mutations_never_use_static_pack(mode):
    model = nn.Sequential(nn.Linear(128, 128)).eval()
    original = model[0]
    with install(model, MockLibrary()) as handle, torch.inference_mode():
        if mode == "version":
            original.weight.add_(1)
        elif mode == "original_parameter":
            original.weight = nn.Parameter(torch.zeros_like(original.weight))
        elif mode == "wrapper_parameter":
            model[0].weight = nn.Parameter(torch.zeros_like(original.weight))
        elif mode == "dtype":
            model[0].half()
        else:
            original.bias = None
        with pytest.raises(xnn.XNNPACKError, match="changed|float32"):
            model(torch.randn(1, 128))
        assert handle.metrics()["xnnpack_calls"] == 0 and handle.metrics()["errors"] == 1


def test_inference_validation_and_closed_wrapper():
    model = nn.Sequential(nn.Linear(128, 128)).eval()
    handle = install(model, MockLibrary())
    wrapper = model[0]
    with pytest.raises(xnn.XNNPACKError, match="autograd"):
        model(torch.zeros(1, 128))
    with torch.inference_mode():
        for inputs in (torch.zeros(1, 128).half(), torch.zeros(1, 127), torch.tensor(1.0)):
            with pytest.raises(xnn.XNNPACKError):
                model(inputs)
        model.train()
        with pytest.raises(xnn.XNNPACKError, match="eval"):
            model(torch.zeros(1, 128))
        model.eval()
        handle.close()
        with pytest.raises(xnn.XNNPACKError, match="closed"):
            wrapper(torch.zeros(1, 128))
    assert handle.metrics()["errors"] == 6


@pytest.mark.parametrize("failure", ["fail_reshape", "fail_setup", "fail_run", "nonfinite_output", "overwrite_guard"])
def test_c_failures_record_metrics_without_fallback(failure):
    model = nn.Sequential(nn.Linear(128, 128)).eval()
    library = MockLibrary()
    setattr(library, failure, True)
    with install(model, library) as handle, torch.inference_mode():
        with pytest.raises(xnn.XNNPACKError):
            model(torch.ones(17, 128))
        metrics = handle.metrics()
        assert metrics["errors"] == 1 and metrics["last_error"]
        assert metrics["finite_failures"] == int(failure == "nonfinite_output")
        assert metrics["xnnpack_calls"] == int(failure in {"nonfinite_output", "overwrite_guard"})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 70000.0])
def test_invalid_or_overflowing_inputs_fail_before_c_run(value):
    model = nn.Sequential(nn.Linear(128, 128)).eval()
    with install(model, MockLibrary(), dtype="float16") as handle, torch.inference_mode():
        inputs = torch.zeros(1, 128)
        inputs[0, 0] = value
        with pytest.raises(xnn.XNNPACKFiniteError):
            model(inputs)
        assert handle.metrics()["finite_failures"] == 1 and handle.metrics()["xnnpack_run_attempts"] == 0


@pytest.mark.parametrize("which,value", [("weight", float("nan")), ("bias", float("inf")), ("weight", 70000.0)])
def test_invalid_weights_leave_model_unchanged_and_destroy_pool(which, value):
    model = nn.Sequential(nn.Linear(128, 128)).eval()
    original = model[0]
    with torch.no_grad():
        getattr(original, which).view(-1)[0] = value
    library = MockLibrary()
    with pytest.raises(xnn.XNNPACKFiniteError):
        install(model, library, dtype="float16")
    assert model[0] is original and library.pool is None and not library.operators


@pytest.mark.parametrize("nonnull_failed_handle", [False, True])
def test_partial_install_failure_cleans_all_and_is_atomic(nonnull_failed_handle):
    model = nn.Sequential(nn.Linear(128, 128), nn.Linear(128, 128)).eval()
    original = list(model)
    library = MockLibrary()
    library.fail_create_at = 2
    library.fail_create_with_handle = nonnull_failed_handle
    with pytest.raises(xnn.XNNPACKError, match="out_of_memory"):
        install(model, library)
    assert list(model) == original and not library.operators and library.pool is None
    assert library.events[-1] == "pool_destroy"


@pytest.mark.parametrize("failure", ["fail_pool", "null_operator"])
def test_null_native_resources_are_explicit_errors(failure):
    model = nn.Sequential(nn.Linear(128, 128)).eval()
    original = model[0]
    library = MockLibrary()
    setattr(library, failure, True)
    with pytest.raises(xnn.XNNPACKError, match="NULL"):
        install(model, library)
    assert model[0] is original and library.pool is None and not library.operators


def test_affinity_only_during_pool_creation_and_restored_on_failure(monkeypatch):
    state = {"current": {0}, "calls": []}
    monkeypatch.setattr(xnn.os, "sched_getaffinity", lambda pid: state["current"].copy())

    def set_affinity(pid, cpus):
        assert pid == 0
        state["current"] = set(cpus)
        state["calls"].append(set(cpus))

    monkeypatch.setattr(xnn.os, "sched_setaffinity", set_affinity)
    for failure in (False, True):
        library = MockLibrary()
        library.fail_pool = failure
        create_pool = library.create_pool

        def create(threads):
            assert state["current"] == {4, 5, 6, 7}
            return create_pool(threads)

        library.pthreadpool_create.implementation = create
        model = nn.Sequential(nn.Linear(128, 128)).eval()
        if failure:
            with pytest.raises(xnn.XNNPACKUnavailable):
                install(model, library, cpu_affinity=[4, 5, 6, 7])
        else:
            handle = install(model, library, cpu_affinity=[4, 5, 6, 7])
            assert handle.metrics()["cpu_affinity"] == [4, 5, 6, 7]
            handle.close()
        assert state["current"] == {0}
    assert state["calls"] == [{4, 5, 6, 7}, {0}] * 2


def test_serialized_scratch_and_operator_state_across_threads():
    model = nn.Sequential(nn.Linear(128, 129)).eval()
    original = model[0]
    library = MockLibrary()
    library.run_delay = 0.01
    inputs = [torch.randn(rows, 128) for rows in (17, 2, 31, 7)]

    def run(value):
        with torch.inference_mode():
            return model(value)

    with install(model, library) as handle:
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(run, inputs))
        for actual, value in zip(results, inputs):
            torch.testing.assert_close(actual, reference(original, value, "float32"), rtol=2e-5, atol=2e-6)
        assert library.max_active_runs == 1 and handle.metrics()["operator_creates"] == 1


def test_external_module_replacement_is_not_clobbered_but_resources_close():
    model = nn.Sequential(nn.Linear(128, 128)).eval()
    library = MockLibrary()
    handle = install(model, library)
    replacement = nn.Identity()
    model[0] = replacement
    with pytest.raises(xnn.XNNPACKError, match="changed by the caller"):
        handle.close()
    assert model[0] is replacement and library.pool is None and not library.operators
    assert handle.metrics()["closed"] and not handle.metrics()["restored"]


def test_cleanup_error_does_not_prevent_other_owned_resources_release():
    model = nn.Sequential(nn.Linear(128, 128), nn.Linear(128, 128)).eval()
    original = list(model)
    library = MockLibrary()
    library.fail_delete = True
    handle = install(model, library)
    with pytest.raises(xnn.XNNPACKError, match="delete"):
        handle.close()
    assert list(model) == original and library.pool is None and not library.operators
    assert handle.metrics()["errors"] == 2 and handle.metrics()["closed"]
    handle.close()  # No double deletion after a failing cleanup attempt.


@pytest.mark.parametrize("kwargs", [{"threads": 0}, {"threads": True}, {"threads": 1.5},
    {"dtype": "bfloat16"}, {"scope": "camera"}, {"cpu_affinity": []},
    {"cpu_affinity": [True]}, {"cpu_affinity": [1, True]}, {"cpu_affinity": [-1]}, {"cpu_affinity": "4-7"}])
def test_invalid_configuration_has_no_native_side_effects(kwargs):
    model = nn.Sequential(nn.Linear(128, 128)).eval()
    library = MockLibrary()
    with pytest.raises(ValueError):
        install(model, library, **kwargs)
    assert library.initializes == 0


@pytest.mark.parametrize("mode", ["training", "half", "small", "root", "wrong_scope"])
def test_invalid_model_selection_precedes_initialization(mode):
    model = nn.Sequential(nn.Linear(128, 128)).eval()
    kwargs = {}
    if mode == "training":
        model.train()
    elif mode == "half":
        model.half()
    elif mode == "small":
        model = nn.Sequential(nn.Linear(128, 127)).eval()
    elif mode == "root":
        model = nn.Linear(128, 128).eval()
    else:
        kwargs["scope"] = "aggregator"
    library = MockLibrary()
    with pytest.raises(ValueError):
        install(model, library, **kwargs)
    assert library.initializes == 0


def test_real_library_must_match_installed_torch_and_headers(monkeypatch, tmp_path):
    monkeypatch.setattr(xnn.torch, "__version__", "2.3.1")
    monkeypatch.setattr(xnn.torch, "__file__", str(tmp_path / "torch" / "__init__.py"))
    include = tmp_path / "torch" / "include"
    include.mkdir(parents=True)
    for name in xnn.HEADER_SHA256:
        (include / name).write_bytes(b"wrong ABI header")
    monkeypatch.setattr(xnn.ct, "CDLL", pytest.fail)  # Verification must precede loading.
    with pytest.raises(xnn.XNNPACKUnavailable, match="Unverified ABI"):
        xnn._verify_torch_abi()


def test_custom_library_requires_same_file_or_identical_binary(tmp_path):
    root = tmp_path / "torch"
    lib = root / "lib"
    lib.mkdir(parents=True)
    original = lib / "libtorch_cpu.so"
    original.write_bytes(b"verified Torch library binary")
    symlink = tmp_path / "symlink.so"
    symlink.symlink_to(original)
    assert xnn._verify_library_file(symlink, str(root))["library_binary_verified"] == "installed_torch_file"
    copy = tmp_path / "copy.so"
    copy.write_bytes(original.read_bytes())
    assert xnn._verify_library_file(copy, str(root))["library_binary_verified"] == "byte_identical_copy"
    copy.write_bytes(b"wrong Torch library binary!!!")
    assert copy.stat().st_size == original.stat().st_size
    with pytest.raises(xnn.XNNPACKUnavailable, match="byte-identical"):
        xnn._verify_library_file(copy, str(root))
    copy.write_bytes(b"wrong size")
    with pytest.raises(xnn.XNNPACKUnavailable, match="differs"):
        xnn._verify_library_file(copy, str(root))
    with pytest.raises(xnn.XNNPACKUnavailable, match="Cannot verify"):
        xnn._verify_library_file(tmp_path / "absent.so", str(root))


def test_kernel_affinity_intersection_is_not_reported_as_requested(monkeypatch):
    state = {"current": {0}}
    monkeypatch.setattr(xnn.os, "sched_getaffinity", lambda pid: state["current"].copy())
    monkeypatch.setattr(xnn.os, "sched_setaffinity", lambda pid, cpus: state.update(current=set(cpus) & {0, 4}))
    library = MockLibrary()
    with pytest.raises(xnn.XNNPACKUnavailable, match="restricted"):
        install(nn.Sequential(nn.Linear(128, 128)).eval(), library, cpu_affinity=[4, 5])
    assert state["current"] == {0} and not library.events


def test_reference_headers_and_ctypes_prototypes_match(tmp_path):
    root = Path(__file__).resolve().parents[1] / "deploy" / "rk3588" / "reference"
    for name, expected in xnn.HEADER_SHA256.items():
        reference_name = "xnnpack_torch_v2.3.1.h" if name == "xnnpack.h" else "pthreadpool_torch_v2.3.1.h"
        contents = (root / reference_name).read_bytes()
        assert hashlib.sha256(contents).hexdigest() == expected
        (tmp_path / name).write_bytes(contents)
    library = MockLibrary()
    xnn._API(library)
    assert library.xnn_create_fully_connected_nc_f16.argtypes == [
        ct.c_size_t] * 4 + [ct.c_void_p] * 2 + [ct.c_float] * 2 + [ct.c_uint32] + [
        ct.c_void_p] * 2 + [ct.POINTER(ct.c_void_p)]
    assert library.xnn_create_fully_connected_nc_f32.argtypes[4:6] == [ct.POINTER(ct.c_float)] * 2
    assert library.pthreadpool_create.restype is ct.c_void_p
    assert library.pthreadpool_destroy.restype is None
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler unavailable for pinned header prototype verification")
    source = tmp_path / "abi.c"
    source.write_text('''#include "xnnpack.h"
typedef enum xnn_status (*create16)(size_t,size_t,size_t,size_t,const void*,const void*,float,float,
    uint32_t,xnn_code_cache_t,xnn_weights_cache_t,xnn_operator_t*);
typedef enum xnn_status (*create32)(size_t,size_t,size_t,size_t,const float*,const float*,float,float,
    uint32_t,xnn_code_cache_t,xnn_weights_cache_t,xnn_operator_t*);
typedef enum xnn_status (*reshape)(xnn_operator_t,size_t,pthreadpool_t);
typedef enum xnn_status (*setup16)(xnn_operator_t,const void*,void*);
typedef enum xnn_status (*setup32)(xnn_operator_t,const float*,float*);
create16 a=xnn_create_fully_connected_nc_f16;
create32 b=xnn_create_fully_connected_nc_f32;
reshape c=xnn_reshape_fully_connected_nc_f16,d=xnn_reshape_fully_connected_nc_f32;
setup16 e=xnn_setup_fully_connected_nc_f16;
setup32 f=xnn_setup_fully_connected_nc_f32;
enum xnn_status (*g)(xnn_operator_t,pthreadpool_t)=xnn_run_operator;
enum xnn_status (*h)(xnn_operator_t)=xnn_delete_operator;
pthreadpool_t (*i)(size_t)=pthreadpool_create;
size_t (*j)(pthreadpool_t)=pthreadpool_get_threads_count;
void (*k)(pthreadpool_t)=pthreadpool_destroy;
_Static_assert(XNN_EXTRA_BYTES==16,"input tail");
_Static_assert(XNN_FLAG_FP32_STATIC_WEIGHTS==8,"FP32 source weights");
_Static_assert(XNN_FLAG_YIELD_WORKERS==16,"yield workers");
''')
    completed = subprocess.run([compiler, "-std=c11", "-Werror", "-fsyntax-only", "-I", str(tmp_path),
                                str(source)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
