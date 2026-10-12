"""Generated NumPy buffers and fake API/NEON only; no SDK/model execution."""
import ctypes as ct
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import unittest

import numpy as np


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent
probe = load('_v64_fake_group_probe', HERE / 'probe_original_pair_group.py')
ROOT = next(parent for parent in HERE.parents if (parent / 'vggt_slam').is_dir())
base = load('_v64_fake_ABI', ROOT / 'deploy/rk3588/validation/run-v12-rknn-block-board.py')


class FakeNeon:
    @staticmethod
    def f32_to_f16(source, destination, count):
        a = np.ctypeslib.as_array(ct.cast(source, ct.POINTER(ct.c_float)), shape=(count,))
        b = np.ctypeslib.as_array(ct.cast(destination, ct.POINTER(ct.c_uint16)), shape=(count,)).view(np.float16)
        np.copyto(b, a, casting='unsafe')

    @staticmethod
    def f16_to_f32(source, destination, count):
        a = np.ctypeslib.as_array(ct.cast(source, ct.POINTER(ct.c_uint16)), shape=(count,)).view(np.float16)
        b = np.ctypeslib.as_array(ct.cast(destination, ct.POINTER(ct.c_float)), shape=(count,))
        np.copyto(b, a)


class FakeAPI:
    def __init__(self, count, failure=None):
        self.count, self.failure, self.release_counts = count, failure, []
        self.shapes = [probe.SHAPE if i % 2 == 0 else (1, 106, 1024) for i in range(count)]
        self.storage = [np.full(shape, i + .25, np.float16) for i, shape in enumerate(self.shapes)]

    def rknn_inputs_set(self, context, count, inputs):
        assert count == 1
        assert (inputs[0].index, inputs[0].type, inputs[0].pass_through, inputs[0].fmt) == (0, 1, 0, 3)
        assert inputs[0].size == np.prod(probe.SHAPE) * 2
        return 0

    def rknn_run(self, context, run):
        run._obj.frame_id = 17
        return 0

    def rknn_outputs_get(self, context, count, outputs, received):
        assert count == self.count
        received._obj.frame_id = 18 if self.failure == 'frame_id' else 17
        for index in range(count):
            assert (outputs[index].index, outputs[index].want_float) == (index, 0)
            if self.failure == 'partial' and index % 2:
                continue
            value = self.storage[index]
            outputs[index].buf, outputs[index].size = value.ctypes.data, value.nbytes
        if self.failure == 'size':
            outputs[count - 1].size = 1
        return -9 if self.failure == 'partial' else 0

    def rknn_outputs_release(self, context, count, outputs):
        self.release_counts.append(count)
        if self.failure == 'release':
            return -8
        for value in self.storage:
            value.fill(np.nan)  # Invalidate every SDK-owned buffer immediately.
        return 0


class ExecuteTests(unittest.TestCase):
    def make_route(self, count, failure=None):
        api = FakeAPI(count, failure)
        bindings = [dict(index=i, shape=list(shape)) for i, shape in enumerate(api.shapes)]
        counters = dict.fromkeys(('sdk_run_attempts', 'sdk_runs_succeeded',
                                 'outputs_acquired', 'outputs_released'), 0)
        errors = []
        array = np.zeros(probe.SHAPE, np.float32)
        def call():
            return probe.execute(base, api, FakeNeon(), np, base.CONTEXT(1),
                                 array, bindings, counters, errors)
        return api, counters, errors, call

    def test_two_and_four_outputs_are_owned_after_release_and_widening(self):
        for count in (2, 4):
            with self.subTest(count=count):
                api, counters, errors, call = self.make_route(count)
                values, timing = call()
                self.assertEqual(len(values), count)
                for i, value in enumerate(values):
                    self.assertEqual(value.shape, api.shapes[i])
                    self.assertEqual(value.dtype, np.float32)
                    self.assertTrue(value.flags.owndata)
                    self.assertTrue(np.isfinite(value).all())
                    self.assertTrue((value == i + .25).all())
                    self.assertTrue(np.isnan(api.storage[i]).all())
                self.assertEqual(api.release_counts, [count])
                self.assertEqual(counters['outputs_acquired'], count)
                self.assertEqual(counters['outputs_released'], count)
                self.assertEqual(errors, [])
                self.assertTrue(timing['outputs_released'])
                self.assertEqual(timing['output_count'], count)

    def test_partial_get_failure_releases_all_requested_descriptors(self):
        api, counters, errors, call = self.make_route(4, 'partial')
        with self.assertRaises(RuntimeError):
            call()
        self.assertEqual(api.release_counts, [4])
        self.assertEqual(counters['outputs_acquired'], 2)
        self.assertEqual(counters['outputs_released'], 2)
        self.assertEqual(counters['sdk_run_attempts'], 1)
        self.assertEqual(errors, [])

    def test_bad_size_or_frame_id_still_releases_outputs(self):
        for failure in ('size', 'frame_id'):
            with self.subTest(failure=failure):
                api, counters, errors, call = self.make_route(4, failure)
                with self.assertRaises(ValueError):
                    call()
                self.assertEqual(api.release_counts, [4])
                self.assertEqual(counters['outputs_acquired'], 4)
                self.assertEqual(counters['outputs_released'], 4)
                self.assertEqual(errors, [])

    def test_release_failure_is_an_error_without_success_count(self):
        api, counters, errors, call = self.make_route(4, 'release')
        with self.assertRaises(RuntimeError):
            call()
        self.assertEqual(api.release_counts, [4])
        self.assertEqual(counters['outputs_acquired'], 4)
        self.assertEqual(counters['outputs_released'], 0)
        self.assertEqual(errors, ['rknn_outputs_release returned -8'])


if __name__ == '__main__':
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ExecuteTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    record = dict(scope='Generated NumPy buffers + fake typed ABI/API/NEON only',
                  actual_model_or_SDK_or_compiler_executed=False,
                  probe_sha256=hashlib.sha256((HERE / 'probe_original_pair_group.py').read_bytes()).hexdigest(),
                  test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  tests_run=result.testsRun, passed=result.wasSuccessful(),
                  errors=len(result.errors), failures=len(result.failures),
                  seconds=time.perf_counter()-started)
    (HERE / 'probe-generated-tests.json').write_text(json.dumps(record, indent=2) + '\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
