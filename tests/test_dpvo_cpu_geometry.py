"""Synthetic-only CPU SE3/BA checks; no checkpoint, images or devices."""
import unittest

import numpy as np

import importlib.util
from pathlib import Path
_here = Path(__file__).resolve().parent
_impl = _here if (_here / "dpvo_cpu_geometry.py").is_file() else _here.parent / "deploy/rk3588/validation/dpvo_geometry"
_spec = importlib.util.spec_from_file_location("_v28_geometry_test", _impl / "dpvo_cpu_geometry.py")
geo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(geo)


def scene(dtype=np.float64, count=12):
    rng = np.random.default_rng(38)
    poses = np.zeros((1, 3, 7), dtype)
    poses[..., 6] = 1
    poses[0, 1:] = geo.se3_exp(np.array([[-.18, .025, .01, .015, -.025, .012],
                                        [-.32, -.018, .025, -.012, .02, -.008]], dtype))
    intrinsics = np.broadcast_to(np.array([90, 92, 75, 58], dtype), (1, 3, 4)).copy()
    patches = np.zeros((1, count, 3, 3, 3), dtype)
    grid_y, grid_x = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), indexing="ij")
    for patch in patches[0]:
        patch[0] = rng.uniform(30, 120) + grid_x
        patch[1] = rng.uniform(20, 90) + grid_y
        patch[2] = rng.uniform(.3, 1.4)
    ii = np.zeros(count, np.int64)
    jj = np.ones(count, np.int64)
    kk = np.arange(count, dtype=np.int64)
    target = geo.projective_transform(poses, patches, intrinsics, ii, jj, kk)[:, :, 1, 1].copy()
    weight = np.ones_like(target)
    return poses, patches, intrinsics, target, weight, ii, jj, kk


class GeometryTests(unittest.TestCase):
    def test_SE3_group_xyzw_and_world_to_camera(self):
        xi = np.array([.2, -.1, .3, .2, -.3, .1], np.float64)
        pose = geo.se3_exp(xi)
        np.testing.assert_allclose(geo.se3_log(pose), xi, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(geo.se3_matrix(geo.se3_compose(pose, geo.se3_inverse(pose))), np.eye(4), atol=1e-14)
        point_world = np.array([.4, -.2, 2, 1])
        camera_point = geo.se3_matrix(pose) @ point_world
        np.testing.assert_allclose(camera_point[:3], geo.quaternion_act(pose[3:], point_world[:3]) + pose[:3], atol=1e-14)
        np.testing.assert_allclose(geo.se3_matrix(geo.se3_inverse(pose)) @ camera_point, point_world, atol=1e-14)

    def test_log_quaternion_sign_and_near_pi(self):
        xi = np.array([.1, .2, -.4, np.pi - 1e-4, 0, 0])
        pose = geo.se3_exp(xi)
        neg = pose.copy()
        neg[3:] *= -1
        np.testing.assert_allclose(geo.se3_log(pose), geo.se3_log(neg), atol=1e-14)
        np.testing.assert_allclose(geo.se3_exp(geo.se3_log(pose)), pose, atol=1e-12)
        np.testing.assert_allclose(geo.se3_log(np.array([0., 0, 0, 1, 0, 0, 0]))[3:], [-np.pi, 0, 0], atol=1e-14)

    def test_LieTorch_normalizes_while_fastBA_raw_path_does_not(self):
        pose = geo.se3_exp(np.array([.2, -.1, .3, .2, -.1, .3]))
        pose[3:] *= 1.00002
        identity = np.array([0., 0, 0, 0, 0, 0, 1])
        normalized = geo.se3_compose(pose, identity)
        raw = geo.se3_compose(pose, identity, normalize=False)
        self.assertAlmostEqual(np.linalg.norm(normalized[3:]), 1)
        np.testing.assert_array_equal(raw[3:], pose[3:])
        self.assertGreater(np.max(np.abs(geo.se3_matrix(pose, normalize=False) - geo.se3_matrix(pose))), 1e-6)

    def test_left_retraction_and_CUDA_tiny_translation_rule(self):
        pose = geo.se3_exp(np.array([.2, -.1, .05, .1, .2, -.1]))
        update = np.array([-.1, .2, .03, -.05, .1, .02])
        np.testing.assert_allclose(geo.se3_matrix(geo.se3_retract(pose, update)),
                                   geo.se3_matrix(geo.se3_exp(update)) @ geo.se3_matrix(pose), atol=1e-14)
        tiny = np.array([1., 0, 0, 0, 0, 5e-5])
        np.testing.assert_array_equal(geo.se3_exp(tiny, fastba=True)[:3], tiny[:3])
        self.assertGreater(abs(geo.se3_exp(tiny)[:3][1]), 1e-5)

    def test_full_patch_reprojection_and_layout(self):
        poses, patches, K, _, _, ii, jj, kk = scene()
        coords = geo.reproject(poses, patches, K, ii, jj, kk)
        self.assertEqual(coords.shape, (1, 12, 2, 3, 3))
        relative = geo.se3_matrix(poses[0, 1]) @ geo.se3_matrix(geo.se3_inverse(poses[0, 0]))
        x, y, inverse_depth = patches[0, 0, :, 1, 1]
        camera = np.array([(x - K[0, 0, 2]) / K[0, 0, 0], (y - K[0, 0, 3]) / K[0, 0, 1], 1]) / inverse_depth
        moved = relative @ np.r_[camera, 1]
        expected = K[0, 1, :2] * moved[:2] / moved[2] + K[0, 1, 2:]
        np.testing.assert_allclose(coords[0, 0, :, 1, 1], expected, atol=1e-12)

    def test_world_point_cloud_and_scale_gauge_projection(self):
        poses, patches, K, _, _, ii, jj, kk = scene()
        poses[0, 0] = geo.se3_exp(np.array([.15, -.08, .02, .05, -.07, .025]))
        homogeneous = geo.point_cloud(poses, patches, K, ii)
        world = homogeneous[0, 0, 1, 1, :3] / homogeneous[0, 0, 1, 1, 3]
        camera = geo.se3_matrix(poses[0, 1]) @ np.r_[world, 1]
        expected = K[0, 1, :2] * camera[:2] / camera[2] + K[0, 1, 2:]
        np.testing.assert_allclose(expected, geo.reproject(poses, patches, K, ii, jj, kk)[0, 0, :, 1, 1], atol=1e-12)
        scaled_pose, scaled_patch = poses.copy(), patches.copy()
        scaled_pose[..., :3] *= 3
        scaled_patch[:, :, 2] /= 3
        np.testing.assert_allclose(geo.reproject(scaled_pose, scaled_patch, K, ii, jj, kk),
                                   geo.reproject(poses, patches, K, ii, jj, kk), atol=1e-12)

    def test_original_flow_weights_and_translation_only_rotation(self):
        poses, patches, K, _, _, ii, jj, kk = scene()
        coords = geo.projective_transform(poses, patches, K, ii, jj, kk)
        source = geo.projective_transform(poses, patches, K, ii, ii, kk)
        full, valid = geo.flow_mag(poses, patches, K, ii, jj, kk, beta=1.)
        np.testing.assert_allclose(full, np.linalg.norm(coords - source, axis=-1), atol=1e-12)
        self.assertTrue(valid.all())
        translation = geo.projective_transform(poses, patches, K, ii, jj, kk, tonly=True)
        pure, _ = geo.flow_mag(poses, patches, K, ii, jj, kk, beta=0.)
        np.testing.assert_allclose(pure, np.linalg.norm(translation - source, axis=-1), atol=1e-12)

    def test_analytic_Jacobians_left_pose_and_inverse_depth(self):
        poses, patches, K, target, weight, ii, jj, kk = scene(count=2)
        poses[0, 0] = geo.se3_exp(np.array([.04, -.02, .01, -.08, .03, .01]))
        linear = geo.fastba_linearize(poses, patches, K, target, weight, ii, jj, kk)
        epsilon = 1e-6
        for frame, key in ((0, "Ji"), (1, "Jj")):
            numeric = np.empty((2, 2, 6))
            for component in range(6):
                update = np.zeros(6)
                update[component] = epsilon
                plus, minus = poses.copy(), poses.copy()
                plus[0, frame] = geo.se3_retract(poses[0, frame], update)
                minus[0, frame] = geo.se3_retract(poses[0, frame], -update)
                xy_plus = geo.fastba_linearize(plus, patches, K, target, weight, ii, jj, kk)["projection"][0]
                xy_minus = geo.fastba_linearize(minus, patches, K, target, weight, ii, jj, kk)["projection"][0]
                numeric[:, :, component] = (xy_plus - xy_minus) / (2 * epsilon)
            np.testing.assert_allclose(linear[key], numeric, rtol=1e-7, atol=1e-7)
        for edge in range(2):
            plus, minus = patches.copy(), patches.copy()
            plus[0, edge, 2] += epsilon
            minus[0, edge, 2] -= epsilon
            p = geo.fastba_linearize(poses, plus, K, target, weight, ii, jj, kk)["projection"][0, edge]
            m = geo.fastba_linearize(poses, minus, K, target, weight, ii, jj, kk)["projection"][0, edge]
            np.testing.assert_allclose(linear["Jz"][edge], (p - m) / (2 * epsilon), rtol=1e-7, atol=1e-7)

    def test_Schur_solution_matches_complete_normal_system(self):
        poses, patches, K, target, weight, ii, jj, kk = scene()
        target += np.array([1.2, -.8])
        system = geo.fastba_system(poses, patches, K, target, weight, np.array([1e-4]), ii, jj, kk, 1, 2)
        # Pose damping belongs to reduced S, not to an independently damped C.
        damping = 1 + 1e-4 * np.diag(system["B"] - (system["E"] * system["Q"][None]) @ system["E"].T)
        full = np.block([[system["B"] + np.diag(damping), system["E"]],
                         [system["E"].T, np.diag(1 / system["Q"])]])
        direct = np.linalg.solve(full, np.r_[system["v"], system["u"]])
        np.testing.assert_allclose(system["dx"].reshape(-1), direct[:6], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(system["dz"], direct[6:], rtol=1e-10, atol=1e-10)

    def test_BA_decreases_reprojection_and_preserves_fixed_pose(self):
        truth, patches, K, target, weight, ii, jj, kk = scene()
        poses = truth.copy()
        poses[0, 1] = geo.se3_retract(poses[0, 1], np.array([.04, -.015, .01, .008, -.01, .006]))
        fixed = poses[0, 0].copy()
        before = np.linalg.norm(geo.reproject(poses, patches, K, ii, jj, kk)[:, :, :, 1, 1] - target)
        result = geo.BA(poses, patches, K, target, weight, np.array([1e4]), ii, jj, kk, 1, 2, 12, iterations=6)
        after = np.linalg.norm(geo.reproject(poses, patches, K, ii, jj, kk)[:, :, :, 1, 1] - target)
        self.assertLess(after, before * .03)
        np.testing.assert_array_equal(fixed, poses[0, 0])
        np.testing.assert_array_equal(truth[0, 2], poses[0, 2])
        self.assertEqual(len(result["iterations"]), 6)

    def test_structure_only_updates_shared_patch_depth_not_poses(self):
        poses, patches, K, target, weight, ii, jj, kk = scene(count=3)
        original = poses.copy()
        target_patches = patches.copy()
        target_patches[0, :, 2] = .8
        target = geo.reproject(poses, target_patches, K, ii, jj, kk)[:, :, :, 1, 1]
        patches[0, :, 2] = .4
        geo.BA(poses, patches, K, target, weight, np.array([1e-4]), ii, jj, kk, 2, 2, 3, iterations=4)
        np.testing.assert_array_equal(poses, original)
        np.testing.assert_allclose(patches[0, :, 2], .8, atol=1e-7)

    def test_online_residual_threshold_and_variable_K_rejection(self):
        poses, patches, K, target, weight, ii, jj, kk = scene(count=1)
        poses.fill(0)
        poses[..., 6] = 1
        patches[0, 0, 0] = 75
        patches[0, 0, 1] = 58
        target = geo.fastba_linearize(poses, patches, K, target, weight, ii, jj, kk)["projection"].copy()
        target[0, 0, 0] += 128
        self.assertFalse(geo.fastba_linearize(poses, patches, K, target, weight, ii, jj, kk)["active"][0])
        target[0, 0, 0] -= .0001
        self.assertTrue(geo.fastba_linearize(poses, patches, K, target, weight, ii, jj, kk)["active"][0])
        K[0, 1, 0] += 1
        with self.assertRaisesRegex(ValueError, "variable K"):
            geo.fastba_linearize(poses, patches, K, target, weight, ii, jj, kk)

    def test_online_depth_reset_and_lower_bound_even_when_masked(self):
        poses, patches, K, target, weight, ii, jj, kk = scene(count=2)
        patches[0, 0, 2] = 25
        patches[0, 1, 2] = -.1
        weight.fill(0)
        geo.BA(poses, patches, K, target, weight, np.array([1e-4]), ii, jj, kk, 2, 2, 2, iterations=1)
        np.testing.assert_array_equal(patches[0, 0, 2], np.ones((3, 3)))
        np.testing.assert_array_equal(patches[0, 1, 2], np.full((3, 3), 1e-4))

    def test_duplicate_patch_edges_and_FP32_API(self):
        poses, patches, K, target, weight, ii, jj, kk = scene(np.float32, count=3)
        ii, jj, kk = np.repeat(ii, 2), np.repeat(jj, 2), np.repeat(kk, 2)
        target, weight = np.repeat(target, 2, axis=1), np.repeat(weight, 2, axis=1)
        before_pose, before_patch = poses.copy(), patches.copy()
        result = geo.BA(poses, patches, K, target, weight, np.array([1e-4], np.float32), ii, jj, kk, 1, 2, 3, iterations=2, eff_impl=True)
        self.assertEqual(poses.dtype, np.float32)
        self.assertEqual(patches.dtype, np.float32)
        np.testing.assert_allclose(poses, before_pose, atol=1e-5)
        np.testing.assert_allclose(patches, before_patch, atol=1e-5)
        self.assertTrue(result["dense_reference"])

    def test_nonfinite_bad_quaternion_index_and_window_rejected(self):
        poses, patches, K, target, weight, ii, jj, kk = scene(count=1)
        for bad in (np.array([1.]), np.array([np.nan] * 7), np.array([0., 0, 0, 0, 0, 0, 0])):
            with self.assertRaises(ValueError):
                geo.se3_matrix(bad)
        with self.assertRaises(ValueError):
            geo.reproject(poses, patches, K, ii, jj, np.array([99], np.int64))
        with self.assertRaises(ValueError):
            geo.fastba_system(poses, patches, K, target, weight, np.array([1e-4]), ii, jj, kk, 1, 1)


if __name__ == "__main__":
    unittest.main()
