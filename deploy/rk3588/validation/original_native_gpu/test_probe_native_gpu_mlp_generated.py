#!/usr/bin/env python3
"""Generated small arrays and fake ctypes calls only; no Torch/device/library."""
import ast
import ctypes as ct
import importlib.util
from pathlib import Path
import types
import unittest
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('generated_mlp_runner', HERE / 'probe_native_gpu_mlp_board_r2.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.M = 5  # tiny generated matrices, not original-model inference
PUBLIC = next(parent for parent in HERE.parents if (parent / 'vggt_slam').is_dir())
source = (PUBLIC / 'vggt_slam/rknn_linear.py').read_text()
names = {'RKNNMatmulTensorAttr', 'RKNNMatmulIOAttr', 'RKNNMatmulInfo', 'RKNNTensorMem'}
tree = ast.parse(source)
namespace = {'ct': ct}
exec(compile(ast.Module(body=[node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names],
    type_ignores=[]), '<pure ctypes structures>', 'exec'), namespace)
def check(code, operation):
    if code:
        raise RuntimeError(f'{operation}: {code}')
abi = types.SimpleNamespace(**{name: namespace[name] for name in names}, RKNNContext=ct.c_uint64,
    RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32=1, RKNN_MEMORY_SYNC_TO_DEVICE=1,
    RKNN_MEMORY_SYNC_FROM_DEVICE=2, _check=check)


class API:
    def __init__(self, fail_allocation=None, fail_destroy=False):
        self.memory, self.bound, self.attrs = {}, {}, {}
        self.allocations, self.destroyed = 0, 0
        self.fail_allocation, self.fail_destroy = fail_allocation, fail_destroy

    def rknn_matmul_create(self, context, info, attributes):
        context._obj.value = 42
        self.info = info._obj
        m, k, n = self.info.M, self.info.K, self.info.N
        native = bool(self.info.AC_layout)
        dims = dict(A=(k // 8, m, 8) if native else (m, k), B=(n // 16, k // 32, 16, 32),
                    C=(n // 4, m, 4) if native else (m, n))
        for name, shape in dims.items():
            a = getattr(attributes._obj, name)
            a.name, a.n_dims, a.type = name.encode(), len(shape), 0 if name == 'C' else 1
            for i, length in enumerate(shape):
                a.dims[i] = length
            a.size = int(np.prod(shape)) * (4 if name == 'C' else 2)
            self.attrs[name] = a
        return 0

    def rknn_matmul_set_core_mask(self, context, mask):
        assert mask == 1
        return 0

    def rknn_create_mem(self, context, size):
        self.allocations += 1
        if self.allocations == self.fail_allocation:
            return None
        raw = ct.create_string_buffer(size)
        memory = abi.RKNNTensorMem(virt_addr=ct.addressof(raw), size=size, fd=-1)
        pointer = ct.pointer(memory)
        self.memory[memory.virt_addr] = (raw, memory, pointer)
        return pointer

    def rknn_matmul_set_io_mem(self, context, pointer, attr):
        self.bound[bytes(attr._obj.name).split(b'\0')[0].decode()] = pointer
        return 0

    def view(self, name):
        a = self.attrs[name]
        raw = (ct.c_uint8 * a.size).from_address(self.bound[name].contents.virt_addr)
        return np.frombuffer(raw, np.float32 if name == 'C' else np.float16).reshape(tuple(a.dims[:a.n_dims]))

    def rknn_mem_sync(self, *args):
        return 0

    def rknn_matmul_run(self, context):
        m, k, n = self.info.M, self.info.K, self.info.N
        a, b, c = (self.view(name) for name in 'ABC')
        normal_a = a.transpose(1, 0, 2).reshape(m, k) if self.info.AC_layout else a
        normal_b = b.transpose(1, 3, 0, 2).reshape(k, n)
        value = normal_a.astype(np.float32) @ normal_b.astype(np.float32)
        np.copyto(c, value.reshape(m, n // 4, 4).transpose(1, 0, 2) if self.info.AC_layout else value)
        return 0

    def rknn_destroy_mem(self, context, pointer):
        if self.fail_destroy:
            return -7
        del self.memory[pointer.contents.virt_addr]
        self.destroyed += 1
        return 0

    def rknn_matmul_destroy(self, context):
        return 0


class Neon:
    def f32_to_f16(self, source, destination, count):
        source = np.ctypeslib.as_array(ct.cast(source, ct.POINTER(ct.c_float)), shape=(count,))
        dest = np.ctypeslib.as_array(ct.cast(destination, ct.POINTER(ct.c_uint16)), shape=(count,)).view(np.float16)
        np.copyto(dest, source)


class GeneratedChecks(unittest.TestCase):
    def test_native_A8_C4_and_B_roundtrip_owned_output(self):
        api = API()
        context = runner.NativeContext(abi, runner.Counted(api), np, 32, 16, True, 1)
        x = (np.arange(5 * 32).reshape(5, 32) / 128).astype(np.float32)
        weight = (np.arange(16 * 32).reshape(16, 32) / 512).astype(np.float32)
        try:
            context.open(weight)
            context.submit(x, Neon())
            output = context.owned()
            expected = x.astype(np.float16).astype(np.float32) @ weight.astype(np.float16).astype(np.float32).T
            np.testing.assert_array_equal(output, expected)
            self.assertTrue(output.flags.owndata)
            context.views['C'].fill(-1)
            np.testing.assert_array_equal(output, expected)
        finally:
            state = context.close()
        self.assertEqual(state['live_context'], 0)
        self.assertEqual(state['unresolved_memory_count'], 0)

    def test_normal_and_native_have_equal_generated_matmul(self):
        outputs = []
        for native in (False, True):
            context = runner.NativeContext(abi, API(), np, 32, 16, native, 1)
            try:
                context.open(np.arange(512, dtype=np.float32).reshape(16, 32) / 256)
                context.submit(np.arange(160, dtype=np.float32).reshape(5, 32) / 128, Neon())
                outputs.append(context.owned())
            finally:
                context.close()
        np.testing.assert_array_equal(*outputs)

    def test_partial_allocation_cleanup(self):
        api = API(fail_allocation=2)
        context = runner.NativeContext(abi, api, np, 32, 16, True, 1)
        with self.assertRaisesRegex(RuntimeError, 'Null B allocation'):
            context.open(np.ones((16, 32), np.float32))
        state = context.close()
        self.assertEqual(state['live_context'], 0)
        self.assertEqual(state['unresolved_memory_count'], 0)
        self.assertEqual(api.destroyed, 1)

    def test_failed_release_remains_visible(self):
        api = API(fail_destroy=True)
        context = runner.NativeContext(abi, api, np, 32, 16, True, 1)
        context.open(np.ones((16, 32), np.float32))
        state = context.close()
        self.assertEqual(state['unresolved_memory_count'], 3)
        self.assertEqual(len(state['cleanup_errors']), 3)


if __name__ == '__main__':
    assert 'torch' not in sys.modules
    unittest.main()
