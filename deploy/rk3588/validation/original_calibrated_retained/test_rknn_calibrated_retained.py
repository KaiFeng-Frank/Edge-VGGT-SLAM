"""Generated arrays/fake SDK only; imports no Torch or native SDK library."""
import ctypes as ct
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
PUBLIC = next(parent for parent in HERE.parents if (parent / 'vggt_slam').is_dir())
sys.path.insert(0, str(PUBLIC))
spec = importlib.util.spec_from_file_location('candidate', HERE / 'rknn_calibrated_retained.py')
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
spec = importlib.util.spec_from_file_location('typed_abi', PUBLIC / 'deploy/rk3588/validation/run-v12-rknn-block-board.py')
abi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(abi)


class FakeSDK:
    def __init__(self, public, *, get_code=0, release_code=0):
        self.public, self.get_code, self.release_code = public, get_code, release_code
        self.values = [np.full(row['shape'], i + 1.25, dtype=np.float32) for i, row in enumerate(public)]
        self.release_calls = 0

    def rknn_inputs_set(self, ctx, count, inputs):
        assert count == 1 and inputs[0].type == 0 and inputs[0].pass_through == 0 and inputs[0].fmt == 3
        assert inputs[0].size == np.prod(candidate.parent.FRAME) * 4
        return 0

    def rknn_run(self, ctx, pointer):
        ct.cast(pointer, ct.POINTER(abi.RunExtend)).contents.frame_id = 123
        return 0

    def rknn_outputs_get(self, ctx, count, outputs, received):
        assert count == len(self.public)
        for i, value in enumerate(self.values):
            assert outputs[i].want_float == 1 and outputs[i].index == i
            if self.get_code and i > 0:
                break
            outputs[i].buf, outputs[i].size = value.ctypes.data, value.nbytes
        ct.cast(received, ct.POINTER(abi.OutputExtend)).contents.frame_id = 123
        return self.get_code

    def rknn_outputs_release(self, ctx, count, outputs):
        self.release_calls += 1
        for value in self.values:
            value.fill(-100.)  # A returned owned copy must survive release.
        return self.release_code


def runtime(public, **kwargs):
    obj = object.__new__(candidate.RKNNCalibratedRetained)
    obj.base, obj.np, obj.timeout = abi, np, 30000
    obj.api = FakeSDK(public, **kwargs)
    obj.stats = dict(sdk_run_attempts=0, sdk_runs_succeeded=0,
        outputs_acquired=0, outputs_released=0, sdk_io_seconds=0., cleanup_errors=[])
    obj.stats['fp32_io'] = dict(calls_attempted=0, calls_completed=0,
        owned_output_copy_seconds=0., finite_check_seconds=0., route_seconds=0.,
        **{name + '_seconds': 0. for name in ('sdk_api', 'rknn_inputs_set',
            'rknn_run', 'rknn_outputs_get', 'rknn_outputs_release')})
    return obj


class GeneratedChecks(unittest.TestCase):
    def run_owned(self, index):
        public = candidate.parent.bindings(index)
        obj = runtime(public)
        values = obj._run_group(abi.CONTEXT(1), np.zeros(candidate.parent.FRAME, np.float32), public)
        self.assertEqual(len(values), len(public))
        for i, value in enumerate(values):
            self.assertEqual(value.shape, tuple(public[i]['shape']))
            self.assertTrue(value.flags.c_contiguous)
            self.assertTrue(np.all(value == i + 1.25))
            self.assertFalse(np.shares_memory(value, obj.api.values[i]))
        self.assertEqual(obj.stats['outputs_acquired'], len(public))
        self.assertEqual(obj.stats['outputs_released'], len(public))
        self.assertEqual(obj.stats['sdk_runs_succeeded'], 1)

    def test_two_owned_original_outputs(self):
        self.run_owned(0)

    def test_three_owned_outputs_including_global20(self):
        self.run_owned(3)

    def test_partial_failed_get_is_released(self):
        public = candidate.parent.bindings(0)
        obj = runtime(public, get_code=-1)
        with self.assertRaises(Exception):
            obj._run_group(abi.CONTEXT(1), np.zeros(candidate.parent.FRAME, np.float32), public)
        self.assertEqual(obj.api.release_calls, 1)
        self.assertEqual(obj.stats['outputs_acquired'], 1)
        self.assertEqual(obj.stats['outputs_released'], 1)
        self.assertEqual(obj.stats['fp32_io']['calls_completed'], 0)

    def test_release_failure_is_not_reported_released(self):
        public = candidate.parent.bindings(0)
        obj = runtime(public, release_code=-1)
        with self.assertRaises(Exception):
            obj._run_group(abi.CONTEXT(1), np.zeros(candidate.parent.FRAME, np.float32), public)
        self.assertEqual(obj.stats['outputs_acquired'], 2)
        self.assertEqual(obj.stats['outputs_released'], 0)
        self.assertTrue(obj.stats['cleanup_errors'])

    def test_actual_affine_descriptor_is_required(self):
        shape = candidate.parent.FRAME
        row = dict(index=0, name='input', shape=list(shape), n_elems=int(np.prod(shape)),
            type=2, format=3, pass_through=0, scale=.0125, quantization_type=2,
            zero_point=57, size=int(np.prod(shape)))
        base = SimpleNamespace(attr_record=lambda _: row)
        self.assertEqual(candidate.calibrated_io_attr(base, None, shape, 'input'), row)
        for field, bad in (('scale', 0.), ('scale', float('nan')), ('zero_point', 128),
                           ('quantization_type', 0), ('format', 0)):
            old = row[field]
            row[field] = bad
            with self.assertRaises(ValueError):
                candidate.calibrated_io_attr(base, None, shape, 'input')
            row[field] = old

    def test_original_execution_and_cleanup_are_inherited(self):
        base = candidate.parent.RKNNRetainedOriginalAggregator
        derived = candidate.RKNNCalibratedRetained
        for name in ('_forward', '_check_lifecycle', '_check_methods', '_images',
                     '_validate_block', '_check_receipts', 'close', 'metrics'):
            self.assertIs(getattr(base, name), getattr(derived, name))
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main()
