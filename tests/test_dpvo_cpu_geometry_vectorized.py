"""Tiny generated-array comparisons; no checkpoint/features/real BA replay."""
import hashlib
import importlib.util
import os
from pathlib import Path
import unittest
from unittest import mock

import numpy as np


HERE = Path(__file__).resolve().parent
IMPL = HERE if (HERE / "dpvo_cpu_geometry_vectorized.py").is_file() else HERE.parent / "deploy/rk3588/validation/dpvo_geometry"
REFERENCE = Path(os.environ.get("DPVO_V28_REFERENCE", str(IMPL / "dpvo_cpu_geometry.py")))
REFERENCE_SHA = "13eb1e03eb4a5ed1bd1533549f2b9f4994abfaa61e0b55304b99e6fe77d813df"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if hashlib.sha256(REFERENCE.read_bytes()).hexdigest() != REFERENCE_SHA:
    raise RuntimeError("The frozen v28 comparison reference has changed")
scalar = load("v28_scalar_geometry", REFERENCE)
batch = load("v29_vectorized_geometry", IMPL / "dpvo_cpu_geometry_vectorized.py")


def scene(dtype=np.float64, edges=53, seed=38):
    rng = np.random.default_rng(seed)
    poses = np.zeros((1, 4, 7), dtype)
    poses[..., 6] = 1
    poses[0, 1:] = scalar.se3_exp(rng.uniform(-.015, .015, (3, 6)).astype(dtype))
    K = np.broadcast_to(np.array([90, 92, 75, 58], dtype), (1, 4, 4)).copy()
    patches = np.zeros((1, 12, 3, 3, 3), dtype)
    y, x = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), indexing="ij")
    patches[0, :, 0] = rng.uniform(35, 115, 12)[:, None, None] + x
    patches[0, :, 1] = rng.uniform(25, 85, 12)[:, None, None] + y
    patches[0, :, 2] = rng.uniform(.3, 1.2, 12)[:, None, None]
    ii, jj = rng.integers(0, 4, (2, edges), dtype=np.int64)
    kk = rng.integers(0, 12, edges, dtype=np.int64)
    dummy = np.zeros((1, edges, 2), dtype)
    weight = rng.uniform(.05, .95, dummy.shape).astype(dtype)
    xy = scalar.fastba_linearize(poses, patches, K, dummy, weight, ii, jj, kk)["projection"]
    target = xy + rng.uniform(-.3, .3, dummy.shape).astype(dtype)
    return poses, patches, K, target, weight, ii, jj, kk


class VectorizedGeometryTests(unittest.TestCase):
    def same_system(self, data, t0=1, t1=4, damping=None):
        p, patches, K, target, weight, ii, jj, kk = data
        damping = np.array([1e-4], p.dtype) if damping is None else damping
        actual = batch.fastba_system(p, patches, K, target, weight, damping, ii, jj, kk, t0, t1)
        expected = scalar.fastba_system(p, patches, K, target, weight, damping, ii, jj, kk, t0, t1)
        self.assertEqual(set(actual), set(expected))
        rtol, atol = (1e-4, 1e-5) if p.dtype == np.float32 else (1e-10, 1e-10)
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key], rtol=rtol, atol=atol,
                                       err_msg="system field " + key)
        return actual, expected

    def test_all_system_fields_two_precisions(self):
        for dtype in (np.float64, np.float32):
            for seed in (38, 42, 79):
                with self.subTest(dtype=dtype, seed=seed):
                    self.same_system(scene(dtype, seed=seed))

    def test_two_step_updates_and_all_diagnostics(self):
        for dtype in (np.float64, np.float32):
            data = scene(dtype)
            p, patches, K, target, weight, ii, jj, kk = data
            p1, d1, p2, d2 = p.copy(), patches.copy(), p.copy(), patches.copy()
            args = (K, target, weight, np.array([.1], dtype), ii, jj, kk, 1, 4, 3)
            one = scalar.BA(p1, d1, *args, iterations=2, eff_impl=True)
            two = batch.BA(p2, d2, *args, iterations=2, eff_impl=True)
            rtol, atol = (1e-4, 1e-5) if dtype == np.float32 else (1e-10, 1e-10)
            np.testing.assert_allclose(p2, p1, rtol=rtol, atol=atol)
            np.testing.assert_allclose(d2, d1, rtol=rtol, atol=atol)
            np.testing.assert_array_equal(p2[0, 0], p[0, 0])
            self.assertEqual(set(one), set(two))
            for a, b in zip(one["iterations"], two["iterations"]):
                for key in a:
                    np.testing.assert_allclose(b[key], a[key], rtol=rtol, atol=atol)
            self.assertEqual(one["optimized_pose_interval"], two["optimized_pose_interval"])

    def test_self_frame_coalesced_before_products(self):
        data = scene(np.float32)
        data[0][0, :, 3:] *= np.float32(1.00002)
        data[6][:] = data[5]
        self.same_system(data, t0=0)

    def test_fixed_endpoints_and_structure_only(self):
        for window in ((0, 4), (2, 4), (3, 4), (4, 4)):
            with self.subTest(window=window):
                self.same_system(scene(), *window)

    def test_duplicate_edges_across_chunk_boundary(self):
        for dtype in (np.float64, np.float32):
            data = scene(dtype, edges=517)
            # Explicit repeated IDs span the 256-edge assembly chunks.
            data[5][250:265] = 2
            data[6][250:265] = 3
            data[7][250:265] = 7
            self.same_system(data)

    def test_sorted_unique_patch_damping_and_zero_weights(self):
        data = scene()
        data[4][:, ::2] = 0
        damping = np.linspace(.01, 1, len(np.unique(data[7])))
        self.same_system(data, damping=damping)

    def test_strict_gate_and_inactive_edges(self):
        data = scene()
        # Integer principal-point projection makes 128 exact, avoiding a
        # test-fixture subtraction that can round to 127.99999999999999.
        data[0].fill(0)
        data[0][..., 6] = 1
        data[1][0, :, 0] = 75
        data[1][0, :, 1] = 58
        xy = scalar.fastba_linearize(*data[:5], *data[5:])["projection"]
        data[3][0, 0] = xy[0, 0] + [128, 0]
        data[3][0, 1] = xy[0, 1] + [127.999999, 0]
        data[3][0, 2] = xy[0, 2] + [256, 0]
        before = scalar.fastba_linearize(*data[:5], *data[5:])
        after = batch.fastba_linearize(*data[:5], *data[5:])
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
        self.assertFalse(after["active"][0])
        self.assertTrue(after["active"][1])
        self.assertFalse(after["active"][2])
        self.same_system(data)

    def test_empty_edges_and_zero_pose_window(self):
        for dtype in (np.float64, np.float32):
            for window in ((1, 4), (4, 4)):
                self.same_system(scene(dtype, edges=0), *window)

    def test_depth_reset_and_floor_remain_identical(self):
        data = scene()
        data[1][0, :, 2] = np.array([25, -.1] * 6)[:, None, None]
        data[4].fill(0)
        p, d, K, target, weight, ii, jj, kk = data
        p1, d1, p2, d2 = p.copy(), d.copy(), p.copy(), d.copy()
        args = (K, target, weight, np.array([1e-4]), ii, jj, kk, 4, 4, 3)
        scalar.BA(p1, d1, *args, iterations=1)
        batch.BA(p2, d2, *args, iterations=1)
        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(d1, d2)

    def test_solve_failure_does_not_partially_commit(self):
        for implementation in (scalar, batch):
            p, d, K, target, weight, ii, jj, kk = scene()
            old_pose, old_depth = p.copy(), d.copy()
            with mock.patch.object(np.linalg, "cholesky", side_effect=np.linalg.LinAlgError("injected failure")):
                with self.assertRaises(np.linalg.LinAlgError):
                    implementation.BA(p, d, K, target, weight, np.array([1e-4]), ii, jj, kk, 1, 4, 3)
            np.testing.assert_array_equal(p, old_pose)
            np.testing.assert_array_equal(d, old_depth)

    def test_invalid_inputs_fail_with_same_exception_type(self):
        for mutation in ("weight", "lambda", "K", "window", "readonly"):
            failures = []
            for implementation in (scalar, batch):
                p, d, K, target, weight, ii, jj, kk = scene()
                lm, t0 = np.array([1e-4]), 1
                if mutation == "weight": weight[0, 0, 0] = -1
                if mutation == "lambda": lm[0] = np.nan
                if mutation == "K": K[0, 1, 0] += 1
                if mutation == "window": t0 = -1
                if mutation == "readonly": p.flags.writeable = False
                try:
                    implementation.BA(p, d, K, target, weight, lm, ii, jj, kk, t0, 4, 3)
                except Exception as exc:
                    failures.append(type(exc))
            self.assertEqual(failures, [ValueError, ValueError], mutation)


if __name__ == "__main__":
    unittest.main()
