"""Host-only ABI and memory/lifecycle tests; these do not validate an NPU."""

from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time

import numpy as np
import pytest
import torch
from torch import nn

from vggt_slam import rknn_linear as rk


class MockFunction:
    def __init__(self, implementation):
        self.implementation = implementation

    def __call__(self, *args):
        return self.implementation(*args)


class MockLibrary:
    """Emulates the C pointer ABI and FP16 inputs, never a hardware device."""

    def __init__(self):
        self.contexts = {}
        self.allocations = {}
        self.creates = self.destroys = self.runs = self.syncs = 0
        self.fail_run = False
        self.fail_allocate_at = None
        self.allocates = 0
        self.active_runs = self.max_active_runs = 0
        self.guard = threading.Lock()
        self.run_delay = 0
        self.last_shape = None
        for name in ("matmul_create", "matmul_destroy", "matmul_run", "matmul_set_io_mem",
                     "matmul_set_core_mask", "create_mem", "destroy_mem", "mem_sync"):
            setattr(self, "rknn_" + name, MockFunction(getattr(self, name)))

    @staticmethod
    def _context_id(context):
        return context.value if isinstance(context, rk.RKNNContext) else context

    def matmul_create(self, context_ptr, info_ptr, attrs_ptr):
        info = ct.cast(info_ptr, ct.POINTER(rk.RKNNMatmulInfo)).contents
        assert info.type == rk.RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32
        assert info.B_layout in {0, 1} and info.AC_layout == 0
        assert info.K % 32 == info.N % 16 == 0
        self.creates += 1
        context_id = self.creates
        self.last_shape = (info.M, info.K, info.N)
        self.contexts[context_id] = {"shape": self.last_shape, "memories": {}, "bound_inputs": {},
                                     "B_layout": info.B_layout}
        ct.cast(context_ptr, ct.POINTER(rk.RKNNContext)).contents.value = context_id
        attrs = ct.cast(attrs_ptr, ct.POINTER(rk.RKNNMatmulIOAttr)).contents
        b_shape = (info.N // 16, info.K // 32, 16, 32) if info.B_layout == 1 else (info.K, info.N)
        for name, shape, dtype in (("A", (info.M, info.K), rk.RKNN_TENSOR_FLOAT16),
                                    ("B", b_shape, rk.RKNN_TENSOR_FLOAT16),
                                    ("C", (info.M, info.N), rk.RKNN_TENSOR_FLOAT32)):
            attr = getattr(attrs, name)
            attr.name = name.encode()
            attr.n_dims = len(shape)
            for index, dimension in enumerate(shape):
                attr.dims[index] = dimension
            attr.type = dtype
            attr.size = np.prod(shape).item() * (4 if dtype == rk.RKNN_TENSOR_FLOAT32 else 2)
        return 0

    def matmul_destroy(self, context):
        context_id = self._context_id(context)
        assert not any(owner == context_id for owner, _, _ in self.allocations.values())
        del self.contexts[context_id]
        self.destroys += 1
        return 0

    def create_mem(self, context, size):
        self.allocates += 1
        if self.fail_allocate_at == self.allocates:
            return ct.POINTER(rk.RKNNTensorMem)()
        raw = (ct.c_uint8 * size)()
        memory = rk.RKNNTensorMem(virt_addr=ct.addressof(raw), size=size)
        pointer = ct.pointer(memory)
        self.allocations[ct.addressof(memory)] = (self._context_id(context), raw, pointer)
        return pointer

    def destroy_mem(self, context, pointer):
        owner, _, _ = self.allocations.pop(ct.addressof(pointer.contents))
        assert owner == self._context_id(context)
        return 0

    def matmul_set_io_mem(self, context, pointer, attr_ptr):
        attr = ct.cast(attr_ptr, ct.POINTER(rk.RKNNMatmulTensorAttr)).contents
        name = attr.name.decode()
        state = self.contexts[self._context_id(context)]
        state["memories"][name] = pointer
        # The real runtime's normal-layout B packing caches contents at
        # bind time. Cache A too, so updated inputs also require rebinding.
        if name in {"A", "B"}:
            state["bound_inputs"][name] = self._view(pointer, tuple(attr.dims[:attr.n_dims]), np.float16).copy()
        return 0

    def matmul_set_core_mask(self, context, mask):
        assert self._context_id(context) in self.contexts
        return 0

    def mem_sync(self, context, pointer, direction):
        assert direction in {rk.RKNN_MEMORY_SYNC_TO_DEVICE, rk.RKNN_MEMORY_SYNC_FROM_DEVICE}
        assert ct.addressof(pointer.contents) in self.allocations
        self.syncs += 1
        return 0

    @staticmethod
    def _view(pointer, shape, dtype):
        raw = (ct.c_uint8 * pointer.contents.size).from_address(pointer.contents.virt_addr)
        return np.frombuffer(raw, dtype=dtype, count=np.prod(shape).item()).reshape(shape)

    def matmul_run(self, context):
        self.runs += 1
        if self.fail_run:
            return -8
        with self.guard:
            self.active_runs += 1
            self.max_active_runs = max(self.max_active_runs, self.active_runs)
        try:
            if self.run_delay:
                time.sleep(self.run_delay)
            state = self.contexts[self._context_id(context)]
            m, k, n = state["shape"]
            memories = state["memories"]
            a = state["bound_inputs"]["A"].astype(np.float32)
            b = state["bound_inputs"]["B"].astype(np.float32)
            if state["B_layout"] == 1:
                # Independent indexing, rather than reversing the same
                # reshape/transpose expression used by the implementation.
                normal_b = np.empty((k, n), dtype=np.float32)
                for k_index in range(k):
                    for n_index in range(n):
                        normal_b[k_index, n_index] = b[n_index // 16, k_index // 32,
                                                       n_index % 16, k_index % 32]
                b = normal_b
            c = self._view(memories["C"], (m, n), np.float32)
            c[:] = a @ b
            return 0
        finally:
            with self.guard:
                self.active_runs -= 1


def install(model, library, **kwargs):
    return rk.install_rknn_linear(model, _library=library, min_weight_elements=1, **kwargs)


def fp16_reference(layer, inputs):
    return inputs.half().float() @ layer.weight.detach().half().float().T + (
        layer.bias.detach() if layer.bias is not None else 0)


@pytest.mark.parametrize("layout", ["normal", "native"])
def test_fp16_padding_batching_and_restore(layout):
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(17, 9)).eval()
    original = model[0]
    inputs = torch.randn(2, 9, 17)
    expected = fp16_reference(original, inputs)
    library = MockLibrary()
    handle = install(model, library, max_rows=8, weight_layout=layout)
    with torch.inference_mode():
        actual = model(inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert actual.dtype == torch.float32 and actual.device.type == "cpu"
    metrics = handle.metrics()
    assert metrics["npu_calls"] == 3 and metrics["npu_used"]
    assert metrics["replaced_modules"] == 1 and metrics["fp16_weight_bytes"] == 32 * 16 * 2
    assert metrics["weight_layout"] == layout
    assert metrics["npu_seconds"] >= 0 and metrics["context_cache_bytes"] > 0
    assert library.last_shape == (2, 32, 16)
    handle.restore()
    assert model[0] is original
    handle.close()
    handle.close()
    assert not library.allocations and not library.contexts
    assert library.creates == library.destroys
    assert handle.metrics()["context_cache_bytes"] == handle.metrics()["fp16_weight_bytes"] == 0


@pytest.mark.parametrize("layout", ["normal", "native"])
def test_shared_shape_reuses_context_but_uploads_each_layers_weight(layout):
    model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16)).eval()
    originals = list(model)
    inputs = torch.randn(3, 16)
    expected = fp16_reference(originals[1], fp16_reference(originals[0], inputs))
    library = MockLibrary()
    with install(model, library, weight_layout=layout) as handle, torch.inference_mode():
        torch.testing.assert_close(model(inputs), expected, rtol=1e-5, atol=1e-5)
        assert library.creates == 1
        assert handle.metrics()["npu_calls"] == 2
        assert handle.metrics()["weight_upload_bytes"] == 2 * 32 * 16 * 2
    assert list(model) == originals
    assert library.destroys == 1


def test_native_packing_order_across_both_tile_axes_with_padding():
    model = nn.Sequential(nn.Linear(35, 19, bias=False)).eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.arange(35 * 19, dtype=torch.float32).reshape(19, 35) / 100)
    original = model[0]
    library = MockLibrary()
    with install(model, library, weight_layout="native"):
        packed = model[0].packed_weight
        assert packed.shape == (2, 2, 16, 32) and packed.flags.c_contiguous
        for n in range(32):
            for k in range(64):
                expected = np.float16(original.weight[n, k].item()) if n < 19 and k < 35 else 0
                assert packed[n // 16, k // 32, n % 16, k % 32] == expected
        inputs = torch.randn(5, 35)
        with torch.inference_mode():
            torch.testing.assert_close(model(inputs), fp16_reference(original, inputs), rtol=1e-5, atol=1e-5)


def test_native_segmented_k_is_rejected_without_mutating_model():
    model = nn.Sequential(nn.Linear(8193, 1, bias=False)).eval()
    original = model[0]
    library = MockLibrary()
    with pytest.raises(ValueError, match="segmented packing"):
        install(model, library, weight_layout="native")
    assert model[0] is original and library.creates == 0


def test_lru_and_device_byte_limits_release_evicted_contexts():
    model = nn.Sequential(nn.Linear(17, 9), nn.Linear(9, 17), nn.Linear(17, 3)).eval()
    library = MockLibrary()
    with install(model, library, max_contexts=1, max_cache_bytes=4096) as handle, torch.inference_mode():
        model(torch.randn(3, 17))
        metrics = handle.metrics()
        assert metrics["context_creates"] == 3 and metrics["context_evictions"] == 2
        assert metrics["peak_contexts"] == 1 and metrics["peak_device_bytes"] <= 4096
        assert library.destroys == 2
    assert not library.contexts and not library.allocations


@pytest.mark.parametrize("failure", ["run", "allocation", "byte_limit"])
def test_failed_npu_never_runs_original_cpu_linear_and_cleans_up(failure, monkeypatch):
    model = nn.Sequential(nn.Linear(17, 9)).eval()
    original = model[0]
    monkeypatch.setattr(original, "forward", lambda *_: pytest.fail("CPU fallback was executed"))
    library = MockLibrary()
    if failure == "run":
        library.fail_run = True
    if failure == "allocation":
        library.fail_allocate_at = 2
    handle = install(model, library, max_cache_bytes=64 if failure == "byte_limit" else 4096)
    with torch.inference_mode(), pytest.raises(rk.RKNNError):
        model(torch.randn(3, 17))
    assert handle.metrics()["npu_calls"] == 0
    assert handle.metrics()["npu_errors"] == 1
    handle.close()
    assert model[0] is original
    assert not library.allocations and not library.contexts


def test_missing_runtime_or_symbol_leaves_model_unchanged(monkeypatch):
    model = nn.Sequential(nn.Linear(4, 4)).eval()
    original = model[0]
    monkeypatch.setattr(rk.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(rk.platform, "system", lambda: "Linux")
    monkeypatch.setattr(rk.ct, "CDLL", lambda _: (_ for _ in ()).throw(OSError("not installed")))
    with pytest.raises(rk.RKNNUnavailable, match="Cannot load RKNN runtime"):
        rk.install_rknn_linear(model, min_weight_elements=1)
    assert model[0] is original
    library = MockLibrary()
    del library.rknn_mem_sync
    with pytest.raises(rk.RKNNUnavailable, match="rknn_mem_sync"):
        install(model, library)
    assert model[0] is original and library.creates == 0


def test_explicit_selection_and_weight_changes_are_visible():
    model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 3)).eval()
    originals = list(model)
    library = MockLibrary()
    handle = rk.install_rknn_linear(model, _library=library, min_weight_elements=100)
    assert handle.metrics()["replaced_module_names"] == ["0"]
    assert handle.metrics()["skipped_modules"] == ["1"]
    assert model[1] is originals[1]
    with torch.no_grad():
        model[0].weight.add_(0.5)
    with torch.inference_mode(), pytest.raises(rk.RKNNError, match="weights changed"):
        model(torch.randn(2, 16))
    assert library.runs == 0
    handle.close()


def test_inference_only_dtype_and_training_errors():
    model = nn.Sequential(nn.Linear(4, 4)).eval()
    library = MockLibrary()
    with install(model, library):
        with pytest.raises(rk.RKNNError, match="autograd"):
            model(torch.randn(2, 4))
        with torch.inference_mode(), pytest.raises(rk.RKNNError, match="CPU float32"):
            model(torch.randn(2, 4).half())
        model.train()
        with torch.inference_mode(), pytest.raises(rk.RKNNError, match="inference-only"):
            model(torch.randn(2, 4))
    assert library.runs == 0


@pytest.mark.parametrize("mask", [3, 5, 6, 7, 0xffff])
def test_unsupported_multicore_masks_fail_before_mutation(mask):
    model = nn.Sequential(nn.Linear(4, 4)).eval()
    original = model[0]
    library = MockLibrary()
    with pytest.raises(ValueError, match="rejected combined-core masks"):
        install(model, library, core_mask=mask)
    assert model[0] is original and library.creates == 0


def test_concurrent_callers_use_one_serial_execution_lane():
    model = nn.Sequential(nn.Linear(16, 16)).eval()
    original = model[0]
    library = MockLibrary()
    library.run_delay = 0.01
    inputs = [torch.randn(4, 16) for _ in range(8)]
    expected = [fp16_reference(original, item) for item in inputs]
    with install(model, library) as handle:
        def call(item):
            with torch.inference_mode():
                return model(item)
        with ThreadPoolExecutor(max_workers=4) as executor:
            outputs = list(executor.map(call, inputs))
        for actual, reference in zip(outputs, expected):
            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
        assert library.max_active_runs == 1
        assert handle.metrics()["npu_calls"] == 8 and library.creates == 1


def test_ctypes_abi_matches_compiled_official_v232_headers(tmp_path):
    compiler = shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        pytest.skip("C compiler unavailable for independent ABI comparison")
    reference = Path(__file__).resolve().parents[1] / "deploy/rk3588/reference"
    shutil.copyfile(reference / "rknn_api_v2.3.2.h", tmp_path / "rknn_api.h")
    shutil.copyfile(reference / "rknn_matmul_api_v2.3.2.h", tmp_path / "rknn_matmul_api.h")
    structures = {"info": ("rknn_matmul_info", rk.RKNNMatmulInfo),
                  "attr": ("rknn_matmul_tensor_attr", rk.RKNNMatmulTensorAttr),
                  "io": ("rknn_matmul_io_attr", rk.RKNNMatmulIOAttr),
                  "mem": ("rknn_tensor_mem", rk.RKNNTensorMem)}
    entries, expected = [], {}
    for label, (c_name, python_type) in structures.items():
        entries.append((f"{label}_size", f"sizeof({c_name})"))
        expected[f"{label}_size"] = ct.sizeof(python_type)
        for field, _ in python_type._fields_:
            entries.append((f"{label}_{field}", f"offsetof({c_name}, {field})"))
            expected[f"{label}_{field}"] = getattr(python_type, field).offset
    entries.extend((("context_size", "sizeof(rknn_context)"),
                    ("fp16_matmul", "RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32"),
                    ("sync_to", "RKNN_MEMORY_SYNC_TO_DEVICE"),
                    ("sync_from", "RKNN_MEMORY_SYNC_FROM_DEVICE")))
    expected.update(context_size=ct.sizeof(rk.RKNNContext), fp16_matmul=1, sync_to=1, sync_from=2)
    formats = ",".join(f'\\"{key}\\":%zu' for key, _ in entries)
    arguments = ",".join(f"(size_t)({expression})" for _, expression in entries)
    source = tmp_path / "probe.c"
    source.write_text('#include <stdio.h>\n#include <stddef.h>\n#include "rknn_matmul_api.h"\n'
                      f'int main(void) {{ printf("{{{formats}}}", {arguments}); return 0; }}\n')
    binary = tmp_path / "probe"
    subprocess.run([compiler, "-std=c11", "-I", str(tmp_path), str(source), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    actual = json.loads(subprocess.check_output([str(binary)], text=True))
    assert actual == expected
