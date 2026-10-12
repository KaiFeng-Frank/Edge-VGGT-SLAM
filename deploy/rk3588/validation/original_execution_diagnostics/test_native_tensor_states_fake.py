"""Fake objects only; optionally exercise a root-built extension, never Torch."""
import argparse
import ast
import hashlib
import importlib.util
from pathlib import Path
import types
import unittest

HERE = Path(__file__).resolve().parent
LIBRARY = None
SOURCE = None


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Stale(RuntimeError):
    pass


def oracle():
    node = next(n for n in ast.parse(SOURCE.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == '_tensor_state')
    namespace = {'torch': types.SimpleNamespace(float32='float32'),
                 'PatchCacheStaleState': Stale}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


class Device:
    def __init__(self, owner, kind='cpu'):
        self.owner, self.kind = owner, kind

    @property
    def type(self):
        self.owner.log.append('device.type')
        return self.kind


class Tensor:
    FIELDS = {'device', 'dtype', '_version', 'shape', 'stride', 'data_ptr',
              'requires_grad', 'is_floating_point'}

    def __init__(self, *, dtype='float32', device='cpu', floating=True, fail=None):
        self.log, self.dtype_value, self.device_value = [], dtype, device
        self.floating, self.fail, self.dtype_reads = floating, fail, 0
        self.change_dtype = False

    def __getattribute__(self, name):
        if name in Tensor.FIELDS:
            log = object.__getattribute__(self, 'log')
            log.append(name)
            fail = object.__getattribute__(self, 'fail')
            if fail is not None and name == fail[0]:
                raise fail[1](fail[2])
            if name == 'device': return Device(self, self.device_value)
            if name == 'dtype':
                self.dtype_reads += 1
                return 'second-read' if self.change_dtype and self.dtype_reads == 2 else self.dtype_value
            if name == '_version': return 7
            if name == 'shape': return [2, 3]
            if name == 'requires_grad': return False
        return object.__getattribute__(self, name)

    def is_floating_point(self):
        self.log.append('is_floating_point()')
        return self.floating

    def stride(self):
        self.log.append('stride()')
        return [3, 1]

    def data_ptr(self):
        self.log.append('data_ptr()')
        return 1234


def outcome(function):
    try:
        return ('returned', function())
    except BaseException as exc:
        cause = None if exc.__cause__ is None else (type(exc.__cause__), str(exc.__cause__))
        context = None if exc.__context__ is None else (type(exc.__context__), str(exc.__context__))
        return ('raised', type(exc), str(exc), cause, context, exc.__suppress_context__)


class StaticTests(unittest.TestCase):
    def test_two_assignments_only_and_original_guards(self):
        shim = load_module('v76_fake_shim', HERE / 'native_tensor_states_shim.py')
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(), shim.CACHE_SHA)
        original = shim.fingerprint_ast(SOURCE.read_text())
        before = [ast.dump(n, include_attributes=False) for n in original.body]
        changed = shim.replace_tensor_loops(original, '_generated_bridge')
        differences = [n for n, (a, b) in enumerate(zip(before, changed.body))
                       if a != ast.dump(b, include_attributes=False)]
        self.assertEqual(len(differences), 2)
        self.assertEqual([changed.body[n].targets[0].id for n in differences], ['parameters', 'buffers'])
        tree = ast.parse(SOURCE.read_text())
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                        and n.name == '_aggregate_forward')
        self.assertEqual(sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                             and n.func.attr == '_guard' for n in ast.walk(function)), 2)

    def test_oracle_attribute_order(self):
        ns = oracle()
        tensor = Tensor()
        ns['_tensor_state']('p', tensor, parameter=True)
        self.assertEqual(tensor.log, ['device', 'device.type', 'dtype', '_version', 'dtype',
            'device', 'shape', 'stride', 'stride()', 'data_ptr', 'data_ptr()', 'requires_grad'])
        tensor = Tensor(dtype='int64', floating=False)
        ns['_tensor_state']('b', tensor, parameter=False)
        self.assertEqual(tensor.log, ['device', 'device.type', 'is_floating_point',
            'is_floating_point()', '_version', 'dtype', 'device', 'shape', 'stride', 'stride()',
            'data_ptr', 'data_ptr()', 'requires_grad'])


class NativeTests(unittest.TestCase):
    def setUp(self):
        if LIBRARY is None:
            self.skipTest('Native extension not built/executed in source-only host review')
        self.native = load_module('_v76_tensor_states', LIBRARY)

    def compare(self, tensor, parameter):
        ns = oracle()
        reference = outcome(lambda: tuple(ns['_tensor_state'](n, t, parameter=parameter)
                                         for n, t in [('test', tensor)]))
        expected_log = tensor.log[:]
        tensor.log.clear(); tensor.dtype_reads = 0
        actual = outcome(lambda: self.native.tensor_states(iter([('test', tensor)]), parameter, ns))
        self.assertEqual(actual, reference)
        self.assertEqual(tensor.log, expected_log)

    def test_parameter_and_both_buffer_policies(self):
        for tensor, flag in [(Tensor(), True), (Tensor(), False),
                             (Tensor(dtype='int64', floating=False), False)]:
            with self.subTest(parameter=flag, dtype=tensor.dtype_value): self.compare(tensor, flag)

    def test_policy_and_version_error_boundaries(self):
        cases = [Tensor(device='cuda'), Tensor(dtype='float16'),
                 Tensor(fail=('_version', RuntimeError, 'no version counter')),
                 Tensor(fail=('_version', ValueError, 'custom version failure')),
                 Tensor(fail=('device', RuntimeError, 'early device failure')),
                 Tensor(fail=('shape', LookupError, 'late shape failure'))]
        for tensor in cases:
            with self.subTest(case=tensor.fail or tensor.device_value): self.compare(tensor, True)

    def test_second_dtype_read_is_not_reused(self):
        tensor = Tensor(); tensor.change_dtype = True
        self.compare(tensor, True)

    def test_generic_unpack_and_iteration_failure(self):
        for item in [(1,), (1, 2, 3), iter([1]), iter([1, 2, 3])]:
            # Re-create consumed generic pairs separately; no tensor read can occur.
            values = list(item)
            ns = oracle()
            def reference():
                return tuple(ns['_tensor_state'](n, t, parameter=True) for n, t in [iter(values)])
            def actual():
                return self.native.tensor_states(iter([iter(values)]), True, ns)
            self.assertEqual(outcome(actual), outcome(reference))
        def broken():
            raise RuntimeError('enumeration failed')
            yield None
        self.assertEqual(outcome(lambda: self.native.tensor_states(broken(), True, oracle())),
            outcome(lambda: tuple(n for n in broken())))

    def test_live_globals_and_empty_input(self):
        ns = oracle()
        tensor = Tensor(dtype='different-fp32-token')
        ns['torch'].float32 = tensor.dtype_value
        expected = tuple(ns['_tensor_state'](n, t, parameter=True) for n, t in [('p', tensor)])
        self.assertEqual(self.native.tensor_states(iter([('p', tensor)]), True, ns), expected)
        self.assertEqual(self.native.tensor_states(iter([]), False, ns), ())


def main():
    global LIBRARY, SOURCE
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-source', type=Path, required=True)
    parser.add_argument('--library', type=Path)
    args = parser.parse_args()
    SOURCE, LIBRARY = args.cache_source, args.library
    suite = unittest.defaultTestLoader.loadTestsFromNames(['StaticTests', 'NativeTests'], module=__import__(__name__))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful(): raise SystemExit(1)


if __name__ == '__main__':
    main()
