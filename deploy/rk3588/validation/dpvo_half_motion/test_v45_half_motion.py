"""Small generated/fake admission tests; no neural/native/BA/image execution."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
VALIDATION = Path(os.environ['DPVO_V45_VALIDATION_ROOT']) if 'DPVO_V45_VALIDATION_ROOT' in os.environ else (
    HERE.parent if (HERE.parent / 'dpvo_half_workload/policy.json').is_file()
    else HERE.parent / 'public-snapshot/deploy/rk3588/validation')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def forbidden(*args, **kwargs):
    raise AssertionError('No SE3 solve or BA needed for this two-observation generated test')


class FakeGeometry:
    se3_compose = se3_inverse = se3_exp = se3_log = flow_mag = BA = staticmethod(forbidden)

    def reproject(self, poses, patches, intrinsics, ii, jj, kk):
        return patches[:, kk, :2].copy()


class DepthDraws:
    def __init__(self):
        self.counts = []

    def __call__(self, count):
        self.counts.append(count)
        return np.linspace(.2, .8, count, dtype=np.float32)


def record(index):
    patches = np.zeros((16, 3, 3, 3), np.float32)
    y, x = np.mgrid[0:3, 0:3]
    patches[:, 0], patches[:, 1], patches[:, 2] = x, y, .5
    return dict(frame_id='generated-' + str(index), timestamp=100. + index,
        captured_monotonic=200. + index, intrinsics=np.array([40, 40, 8, 8], np.float32),
        fmap=np.zeros((128, 4, 4), np.float32), gmap=np.zeros((16, 128, 3, 3), np.float32),
        context=np.zeros((16, 384), np.float32), patches=patches)


def stream(core, config, probe, *, invalid=False):
    draws = DepthDraws()

    def update(net, context, corr, ii, jj, kk):
        delta = np.zeros((1, len(ii), 2), np.float32)
        delta[..., 0] = probe
        if invalid:
            delta[0, 0, 0] = np.nan
        return dict(net=net.copy(), delta=delta, weight=np.full_like(delta, .5))

    def correlate(coords, *, ii, jj, kk, frames):
        assert np.array_equal(ii, kk // 16)
        assert all(not value.flags.writeable for frame in frames for value in frame.values()
                   if isinstance(value, np.ndarray))
        return np.zeros((1, len(ii), 882), np.float32)

    value = core.DPVOStream(core.StreamConfig(**config), geometry=FakeGeometry(),
                            update=update, correlate=correlate, rng=draws)
    value.process_encoded(record(0))
    return value, draws


class MotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = load(HERE / 'run-v45-half-motion-cpu-board.py', '_v45_tiny_driver')
        cls.baseline = load(VALIDATION / 'dpvo_stream/dpvo_stream.py', '_v45_tiny_baseline')
        cls.candidate = load(HERE / 'dpvo_stream_motion1.py', '_v45_tiny_candidate')
        cls.base_policy = json.loads((VALIDATION / 'dpvo_half_workload/policy.json').read_text())
        cls.policy = json.loads((HERE / 'policy-motion1.json').read_text())

    def test_exact_two_line_source_delta_and_original_guards(self):
        proof = self.driver.validate_stream_delta(VALIDATION / 'dpvo_stream/dpvo_stream.py',
                                                  HERE / 'dpvo_stream_motion1.py')
        self.assertTrue(proof['exact_two_line_delta_verified'])
        self.assertEqual(self.baseline.SOURCE_PINS, self.candidate.SOURCE_PINS)
        self.assertEqual(self.baseline.GEOMETRY_FUNCTIONS, self.candidate.GEOMETRY_FUNCTIONS)
        self.assertNotIn('torch', sys.modules)

    def test_config_KF15_and_full_policy_isolation(self):
        self.driver.validate_single_variable_policy(self.policy, self.base_policy)
        self.assertEqual(self.policy['stream_config'], self.base_policy['stream_config'])
        self.assertEqual(self.policy['stream_config']['keyframe_threshold'], 15.0)
        changed = copy.deepcopy(self.policy)
        changed['stream_config']['keyframe_threshold'] = 7.5
        with self.assertRaises(ValueError):
            self.driver.validate_single_variable_policy(changed, self.base_policy)

    def test_strict_feature_pixel_threshold_bounds(self):
        for probe, admitted in ((.999, False), (1., True), (1.5, True), (2., True)):
            with self.subTest(probe=probe):
                candidate, draws = stream(self.candidate, self.policy['stream_config'], probe)
                result = candidate.process_encoded(record(1))
                self.assertEqual(result['admitted'], admitted)
                self.assertEqual(result['motion_probe'], float(np.float32(probe)))
                self.assertFalse(result['fresh_pose_valid'])
                self.assertIsNone(result['pose_w2c'])
                self.assertEqual(draws.counts, [16, 16])
                if not admitted:
                    self.assertEqual(result['reason'], 'initial_motion_below_1')

    def test_admission_diff_and_depth_draws_preserved(self):
        old, old_draws = stream(self.baseline, self.base_policy['stream_config'], 1.5)
        new, new_draws = stream(self.candidate, self.policy['stream_config'], 1.5)
        old_result, new_result = old.process_encoded(record(1)), new.process_encoded(record(1))
        self.assertFalse(old_result['admitted'])
        self.assertTrue(new_result['admitted'])
        self.assertEqual(old_result['motion_probe'], new_result['motion_probe'])
        self.assertEqual(old_draws.counts, new_draws.counts)
        self.assertEqual(old.metrics()['BA_calls'], 0)
        self.assertEqual(new.metrics()['BA_calls'], 0)

    def test_invalid_neural_output_still_fails_before_commit(self):
        candidate, draws = stream(self.candidate, self.policy['stream_config'], 1.5, invalid=True)
        with self.assertRaises(self.candidate.StreamFailure):
            candidate.process_encoded(record(1))
        self.assertEqual(candidate.counter, 1)
        self.assertEqual(candidate.n, 1)
        self.assertEqual(candidate.metrics()['status'], 'failed')
        self.assertEqual(draws.counts, [16, 16])


if __name__ == '__main__':
    unittest.main()
