"""Tiny generated ctypes buffers only; never load or execute an SDK/model."""
import ctypes as ct
import importlib.util
from pathlib import Path
import unittest
import numpy as np


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


probe = load('_fake_tap_probe', Path(__file__).with_name('probe_fc1_tap.py'))
base = load('_fake_tap_ABI', Path(__file__).resolve().parents[1] / 'run-v12-rknn-block-board.py')


class FakeAPI:
    def __init__(self, count, failure=None):
        self.count, self.failure, self.releases = count, failure, []
        self.values = [np.arange(8, dtype=np.float32) + i for i in range(count)]

    def rknn_inputs_set(self, context, count, inputs):
        assert count == 1 and inputs[0].type == 0 and inputs[0].pass_through == 0 and inputs[0].fmt == 3
        return 0

    def rknn_run(self, context, run):
        run._obj.frame_id = 17
        return 0

    def rknn_outputs_get(self, context, count, outputs, received):
        assert count == self.count
        received._obj.frame_id = 17
        for i in range(1 if self.failure == 'partial' else count):
            assert outputs[i].want_float == 1 and outputs[i].index == i
            outputs[i].buf, outputs[i].size = self.values[i].ctypes.data, self.values[i].nbytes
        if self.failure == 'size': outputs[count-1].size = 1
        return -9 if self.failure == 'partial' else 0

    def rknn_outputs_release(self, context, count, outputs):
        self.releases.append(count)
        if self.failure == 'release': return -8
        for value in self.values: value.fill(np.nan)  # Invalidate SDK-owned storage.
        return 0


class TapOwnershipTests(unittest.TestCase):
    def run_route(self, count, failure=None):
        api = FakeAPI(count, failure)
        bindings = [dict(shape=[1, 2, 4]) for _ in range(count)]
        counters = dict.fromkeys(('sdk_run_attempts', 'sdk_runs_succeeded', 'output_sets_get_attempts',
            'outputs_acquired', 'output_sets_acquired', 'output_sets_release_attempts',
            'output_sets_released', 'outputs_released'), 0)
        errors = []
        call = lambda: probe.execute(base, api, np, base.CONTEXT(1), np.zeros([1,2,4], np.float32),
                                    3, bindings, counters, errors)
        return api, counters, errors, call

    def test_two_and_four_owned_outputs_survive_release(self):
        for count in (2, 4):
            with self.subTest(count=count):
                api, counters, errors, call = self.run_route(count)
                values, timing = call()
                for i, value in enumerate(values):
                    np.testing.assert_array_equal(value.reshape(-1), np.arange(8, dtype=np.float32)+i)
                    self.assertTrue(value.flags.owndata)
                self.assertEqual(api.releases, [count]); self.assertEqual(errors, [])
                self.assertEqual(counters['outputs_acquired'], count)
                self.assertEqual(counters['outputs_released'], count)
                self.assertEqual(timing['extra_tap_output_bytes'], (count-2)*32)

    def test_partial_get_and_bad_size_release_all_requests(self):
        for failure, allocated in (('partial', 1), ('size', 4)):
            with self.subTest(failure=failure):
                api, counters, errors, call = self.run_route(4, failure)
                with self.assertRaises((RuntimeError, ValueError)): call()
                self.assertEqual(api.releases, [4]); self.assertEqual(errors, [])
                self.assertEqual(counters['outputs_acquired'], allocated)
                self.assertEqual(counters['outputs_released'], allocated)

    def test_release_failure_is_reported_and_not_counted_as_success(self):
        api, counters, errors, call = self.run_route(4, 'release')
        with self.assertRaises(RuntimeError): call()
        self.assertEqual(api.releases, [4]); self.assertEqual(counters['outputs_released'], 0)
        self.assertEqual(counters['output_sets_released'], 0); self.assertTrue(errors)

    def test_summary_preserves_first_four_and_three_denominators(self):
        calls = [dict(role=role, timing=dict(io_total_seconds=float(i+1), tap_output_copy_seconds=0.))
                 for i, role in enumerate(probe.ORDER)]
        summary = probe.summarize(calls)
        self.assertEqual(summary['A']['cold_io_seconds'], 1.)
        self.assertEqual(summary['B']['cold_io_seconds'], 2.)
        self.assertEqual(summary['A']['excluding_first_mean_io_seconds'], 17/3)
        self.assertEqual(summary['B']['excluding_first_mean_io_seconds'], 16/3)
        for row in summary.values():
            self.assertEqual(row['all_count'], 4); self.assertEqual(row['excluding_first_count'], 3)
            self.assertEqual(row['all_mean_io_seconds'], 4.5)


if __name__ == '__main__': unittest.main(verbosity=2)
