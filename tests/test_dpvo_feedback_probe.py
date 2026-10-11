"""Tiny injected feedback checks; no images, learned weights or devices."""
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import importlib.util
from pathlib import Path
_here = Path(__file__).resolve().parent
_impl = _here if (_here / "dpvo_cpu_geometry.py").is_file() else _here.parent / "deploy/rk3588/validation/dpvo_geometry"
_spec = importlib.util.spec_from_file_location("_v28_geometry_test", _impl / "dpvo_cpu_geometry.py")
geo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(geo)

ROOT = Path(__file__).parent
if not (ROOT / "dpvo_cpu_geometry.py").is_file():
    ROOT = ROOT.parent / "deploy/rk3588/validation/dpvo_geometry"
spec = importlib.util.spec_from_file_location("feedback_probe", ROOT / "run-v28-dpvo-feedback-board.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def fixture():
    poses = np.zeros((1, 3, 7), np.float32)
    poses[..., 6] = 1
    intrinsics = np.broadcast_to(np.array([70, 71, 55, 40], np.float32), (1, 3, 4)).copy()
    patches = np.zeros((1, 6, 3, 3, 3), np.float32)
    y, x = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), indexing="ij")
    for i, value in enumerate(patches[0]):
        value[0], value[1], value[2] = 30 + i * 6 + x, 24 + (i % 3) * 9 + y, 1
    kk, jj = np.repeat(np.arange(6, dtype=np.int64), 3), np.tile(np.arange(3, dtype=np.int64), 6)
    ii = np.repeat(np.arange(3, dtype=np.int64), 2)[kk]
    graph = dict(ii=ii, jj=jj, kk=kk, ctx=np.zeros((1, 18, 384), np.float32))
    net = np.zeros((1, 18, 384), np.float32)
    truth = poses.copy()
    truth[0, 1] = geo.se3_exp(np.array([-.06, .005, .01, .003, -.006, .002], np.float32))
    truth[0, 2] = geo.se3_exp(np.array([-.10, -.012, .007, -.004, .005, .001], np.float32))
    target = geo.reproject(truth, patches, intrinsics, ii, jj, kk)[:, :, :, 1, 1]
    return poses, patches, intrinsics, graph, net, target


class TinyCorrelation:
    cache_key = "synthetic-only-owned-inputs"

    def __init__(self):
        self.coords = []

    def correlate(self, coords, kk, jj, *, cache_key, radius):
        if cache_key != self.cache_key or radius != 3:
            raise ValueError("incorrect cache generation/radius")
        self.coords.append(coords.copy())
        return np.zeros((1, len(kk), 882), np.float32), {"synthetic": True}


class FeedbackTests(unittest.TestCase):
    def test_four_rounds_carry_net_and_reproject_updated_geometry(self):
        poses, patches, K, graph, net, target = fixture()
        correlation = TinyCorrelation()
        seen = []

        def update(previous, context, corr, ii, jj, kk):
            seen.append(previous.copy())
            return previous + np.float32(1), target - correlation.coords[-1][:, :, :, 1, 1], np.ones_like(target)

        records, snapshots = [], []
        for _ in range(4):
            net, arrays, row = probe.feedback_round(geo, poses, patches, K, graph, net, correlation, update)
            records.append(row)
            snapshots.append(arrays)
        self.assertEqual(len(correlation.coords), 4)
        for i, previous in enumerate(seen):
            np.testing.assert_array_equal(previous, np.full_like(previous, i))
        np.testing.assert_array_equal(net, np.full_like(net, 4))
        self.assertGreater(np.linalg.norm(correlation.coords[1] - correlation.coords[0]), .01)
        self.assertGreater(records[0]["pose_left_increment_norm"], .001)
        self.assertGreater(records[0]["inverse_depth_change_norm"], 0)
        self.assertLess(records[0]["residual_after"]["weighted_squared_residual"], records[0]["residual_before"]["weighted_squared_residual"])
        self.assertTrue(all(len(row["ba"]["iterations"]) == 2 for row in records))
        np.testing.assert_array_equal(poses[0, 0], [0, 0, 0, 0, 0, 0, 1])
        for row in records:
            self.assertAlmostEqual(sum(row["seconds"].values()), row["round_total_seconds"], places=12)
        np.testing.assert_array_equal(graph["ctx"], np.zeros_like(graph["ctx"]))
        before = snapshots[0]["poses_after"].copy()
        poses[0, 1, 0] += 1
        np.testing.assert_array_equal(snapshots[0]["poses_after"], before)

    def test_target_uses_current_center_plus_current_delta(self):
        poses, patches, K, graph, net, _ = fixture()
        correlation = TinyCorrelation()
        delta = np.broadcast_to(np.array([.12, -.04], np.float32), (1, 18, 2)).copy()
        update = lambda *args: (net.copy(), delta.copy(), np.ones_like(delta))
        _, arrays, _ = probe.feedback_round(geo, poses, patches, K, graph, net, correlation, update)
        np.testing.assert_array_equal(arrays["target"], arrays["coords_before"][:, :, :, 1, 1] + delta)

    def test_nonfinite_neural_output_rejected_before_BA(self):
        poses, patches, K, graph, net, target = fixture()
        original_pose, original_patch = poses.copy(), patches.copy()
        update = lambda *args: (net.copy(), np.full_like(target, np.nan), np.ones_like(target))
        with self.assertRaisesRegex(ValueError, "delta"):
            probe.feedback_round(geo, poses, patches, K, graph, net, TinyCorrelation(), update)
        np.testing.assert_array_equal(poses, original_pose)
        np.testing.assert_array_equal(patches, original_patch)

    def test_solve_failure_is_fatal_and_not_replaced_with_old_output(self):
        poses, patches, K, graph, net, target = fixture()
        original_pose = poses.copy()
        def fail(*args, **kwargs):
            raise np.linalg.LinAlgError("synthetic Cholesky failure")
        failing = SimpleNamespace(reproject=geo.reproject, BA=fail)
        update = lambda *args: (net + 1, np.zeros_like(target), np.ones_like(target))
        with self.assertRaisesRegex(np.linalg.LinAlgError, "Cholesky"):
            probe.feedback_round(failing, poses, patches, K, graph, net, TinyCorrelation(), update)
        np.testing.assert_array_equal(poses, original_pose)

    def test_bad_corr_shape_does_not_invoke_update(self):
        poses, patches, K, graph, net, _ = fixture()
        correlation = TinyCorrelation()
        correlation.correlate = lambda *args, **kwargs: (np.zeros((1, 18, 880), np.float32), {})
        calls = []
        with self.assertRaisesRegex(ValueError, "correlation"):
            probe.feedback_round(geo, poses, patches, K, graph, net, correlation, lambda *args: calls.append(True))
        self.assertFalse(calls)

    def test_host_rejected_before_device_tree_or_model_import(self):
        with patch.object(probe.platform, "machine", return_value="x86_64"):
            with self.assertRaisesRegex(ValueError, "board only"):
                probe.require_board()


if __name__ == "__main__":
    unittest.main()
