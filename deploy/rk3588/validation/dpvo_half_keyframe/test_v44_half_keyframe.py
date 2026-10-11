"""Tiny generated checks only: policy isolation and projection units, no models."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
PUBLIC = Path(os.environ["DPVO_V44_VALIDATION_ROOT"]) if "DPVO_V44_VALIDATION_ROOT" in os.environ else (
    HERE.parent if (HERE.parent / "dpvo_half_workload/policy.json").is_file()
    else HERE.parent / "public-snapshot/deploy/rk3588/validation")


def load(path, name, digest=None):
    if digest is not None:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class SingleVariableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.driver = load(HERE / "run-v44-half-keyframe-cpu-board.py", "_v44_tiny_driver")
        cls.base = json.loads((PUBLIC / "dpvo_half_workload/policy.json").read_text())
        cls.policy = json.loads((HERE / "policy-keyframe7p5.json").read_text())

    def test_only_declared_runtime_change(self):
        proof = self.driver.validate_single_variable_policy(self.policy, self.base)
        old, new = self.base["stream_config"], self.policy["stream_config"]
        self.assertEqual([key for key in old if old[key] != new[key]], ["keyframe_threshold"])
        self.assertEqual(proof["motion_probe_threshold_feature_pixels_unchanged"], 2.0)
        self.assertEqual(new["keyframe_threshold"], 7.5)
        self.assertNotIn("torch", sys.modules)

    def test_second_runtime_change_fails_closed(self):
        for section, key, value in (("stream_config", "patches_per_frame", 17),
                                    ("stream_config", "patch_lifetime", 5),
                                    ("stream_config", "optimization_window", 5),
                                    ("rng", "seed", 7),
                                    ("lifecycle", "motion_probe_threshold_feature_pixels", 1.0)):
            with self.subTest(section=section, key=key):
                altered = copy.deepcopy(self.policy)
                altered[section][key] = value
                with self.assertRaises(ValueError):
                    self.driver.validate_single_variable_policy(altered, self.base)

    def test_corresponding_rays_scale_feature_flow(self):
        # This establishes units only. Fractional corresponding patch rays are
        # deliberately used; actual integer half-grid patches have wider support.
        geometry = load(PUBLIC / "dpvo_geometry/dpvo_cpu_geometry_vectorized.py",
                        "_v44_tiny_geometry",
                        "c893f12d188baced438504cfc592a8900e18e40ef8223af0a7e944e4484b0c3b")
        poses = np.zeros((1, 3, 7), np.float64)
        poses[..., 6] = 1
        poses[0, 1] = geometry.se3_exp(np.array([.03, -.01, .015, .01, .04, -.02]))
        poses[0, 2] = geometry.se3_exp(np.array([-.025, .02, -.01, -.03, .015, .04]))
        y, x = np.mgrid[-1:2, -1:2].astype(np.float64)
        patches = np.empty((1, 4, 3, 3, 3), np.float64)
        for i in range(4):
            patches[0, i, 0] = 35 + 12 * i + x
            patches[0, i, 1] = 24 + 9 * i + y
            patches[0, i, 2] = .3 + .12 * i
        full_K = np.broadcast_to(np.array([517.3, 516.5, 302.6, 247.3]) / 4,
                                 (1, 3, 4)).copy()
        half_K = full_K * .5
        half_K[..., 2:] -= .25 / 4  # image halfpixel offset, then original RES4
        half_patches = patches.copy()
        half_patches[:, :, :2] = half_patches[:, :, :2] * .5 - .25 / 4
        ii = np.array([0, 0, 1, 1, 0, 1], np.int64)
        jj = np.array([1, 2, 0, 2, 2, 0], np.int64)
        kk = np.array([0, 1, 2, 3, 1, 2], np.int64)
        full_flow, full_valid = geometry.flow_mag(poses, patches, full_K, ii, jj, kk, beta=.5)
        half_flow, half_valid = geometry.flow_mag(poses, half_patches, half_K, ii, jj, kk, beta=.5)
        np.testing.assert_allclose(half_flow, full_flow * .5, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(half_valid, full_valid)
        full_xy = geometry.projective_transform(poses, patches, full_K, ii, jj, kk)
        half_xy = geometry.projective_transform(poses, half_patches, half_K, ii, jj, kk)
        np.testing.assert_allclose(half_xy, full_xy * .5 - .25 / 4, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
