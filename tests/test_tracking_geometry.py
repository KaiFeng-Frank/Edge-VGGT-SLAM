"""Independent point/projection checks for local tracking in an SL4 gauge."""

import numpy as np
import pytest

from vggt_slam.tracking_geometry import associate_pose


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def _pose(rotation=None, translation=None):
    result = np.eye(4)
    if rotation is not None:
        result[:3, :3] = rotation
    if translation is not None:
        result[:3, 3] = translation
    return result


def _calibration():
    return np.array([[471., 1.5, 319.5], [0, 468., 239.5], [0, 0, 1]])


def _pixels(projected):
    return projected[:2] / projected[2]


@pytest.mark.parametrize("gauge", ["sim3", "sl4", "reflection", "shear"])
def test_global_projection_equals_independent_anchor_to_current_projection(gauge):
    rotation = _rotation([0.3, -0.7, 0.2], 0.17)
    transform = _pose(rotation, [0.23, -0.12, 0.31])
    homography = _pose(_rotation([0.1, 0.2, 0.8], -0.4), [2.1, -1.2, 0.7])
    homography[:3, :3] *= 2.7
    if gauge == "sl4":
        homography[3, :3] = [0.02, -0.03, 0.015]
        homography /= np.linalg.det(homography) ** 0.25
    elif gauge == "reflection":
        homography[:3, 0] *= -1
    elif gauge == "shear":
        homography[0, 1] += 0.6
    calibration = _calibration()
    anchor_points = np.array([[-1., 0.5, 1.7, -0.4, 0.8],
                              [0.3, -0.4, 0.7, 1.2, -0.9],
                              [3.2, 4.8, 6.1, 3.9, 5.2],
                              [1., 1., 1., 1., 1.]])
    current_xyz = rotation @ anchor_points[:3] + transform[:3, 3, None]
    expected_pixels = _pixels(calibration @ current_xyz)
    global_homogeneous = homography @ anchor_points
    # The map may store dehomogenized Cartesian coordinates.  This additional
    # rescaling must still preserve pixels through the global projection.
    global_points = global_homogeneous / global_homogeneous[3]
    result = associate_pose(transform, calibration, homography)
    actual_pixels = _pixels(result["projection_world_to_tracking"] @ global_points)
    np.testing.assert_allclose(actual_pixels, expected_pixels, rtol=1e-12, atol=1e-10)
    current_points = np.vstack((current_xyz, np.ones(current_xyz.shape[1])))
    np.testing.assert_allclose(result["projective_homography_world_from_current"] @ current_points,
                               global_homogeneous, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result["T_anchor_from_current"] @ current_points,
                               anchor_points, rtol=1e-12, atol=1e-12)
    center_h = np.r_[result["projective_camera_center"], 1.]
    np.testing.assert_allclose(result["projection_world_to_tracking"] @ center_h,
                               np.zeros(3), atol=1e-10)


@pytest.mark.parametrize("homogeneous_scale", [1., -3.7, 1e-80, -1e80])
def test_positive_sim3_returns_orientation_center_and_explicit_nonmetric_scale(homogeneous_scale):
    current_rotation = _rotation([0.2, 0.7, -0.1], 0.33)
    current_translation = np.array([0.5, -0.8, 0.2])
    transform = _pose(current_rotation, current_translation)
    world_rotation = _rotation([0.5, -0.3, 0.8], -0.29)
    scale = 3.2
    world_translation = np.array([4.7, -2.1, 8.9])
    homography = _pose(scale * world_rotation, world_translation) * homogeneous_scale
    result = associate_pose(transform, _calibration(), homography)
    expected_center = world_translation - scale * world_rotation @ current_rotation.T @ current_translation
    expected_pose = _pose(world_rotation @ current_rotation.T, expected_center)
    np.testing.assert_allclose(result["rigid_global_pose"], expected_pose, atol=2e-14)
    np.testing.assert_allclose(result["projective_camera_center"], expected_center, atol=2e-14)
    np.testing.assert_allclose(result["rigid_global_pose"][:3, :3].T @ result["rigid_global_pose"][:3, :3],
                               np.eye(3), atol=2e-14)
    assert np.linalg.det(result["rigid_global_pose"][:3, :3]) == pytest.approx(1)
    assert result["sim3_scale"] == pytest.approx(scale)
    assert result["sim3_fit_relative_error"] < 1e-14
    assert result["rigid_global_pose_reason"] is None
    assert result["projective_camera_center_reason"] is None
    assert result["local_units"] == "vggt_model_units"
    assert result["metric_scale_validated"] is False
    assert result["global_gauge"] == "projective"


@pytest.mark.parametrize("homography,reason", [
    (np.diag([2., 3., 2., 1.]), "nonuniform_scale_or_shear"),
    (np.array([[1., .3, 0, 0], [0, 1., 0, 0], [0, 0, 1., 0], [0, 0, 0, 1.]]),
     "nonuniform_scale_or_shear"),
    (np.diag([-2., 2., 2., 1.]), "reflection"),
    (np.array([[1., 0, 0, 0], [0, 1., 0, 0], [0, 0, 1., 0], [.02, 0, 0, 1.]]),
     "not_affine"),
])
def test_valid_non_sim3_gauges_keep_exact_projection_but_no_global_rigid_pose(homography, reason):
    result = associate_pose(np.eye(4), _calibration(), homography)
    assert result["rigid_global_pose"] is None
    assert reason in result["rigid_global_pose_reason"]
    assert result["projective_camera_center"] is not None
    np.testing.assert_allclose(result["projection_world_to_tracking"] @ homography,
                               _calibration() @ np.eye(4)[:3], atol=1e-13)


@pytest.mark.parametrize("homogeneous_scale", [1., -3., 1e-80, 1e80])
def test_camera_center_at_infinity_does_not_invalidate_projective_projection(homogeneous_scale):
    # An invertible determinant-one SL4 gauge with the current origin at infinity.
    homography = np.array([[0., 0, 0, 1.], [0, -1., 0, 0],
                           [0, 0, 1., 0], [1., 0, 0, 0]]) * homogeneous_scale
    result = associate_pose(np.eye(4), _calibration(), homography)
    assert result["projective_camera_center"] is None
    assert "infinity" in result["projective_camera_center_reason"]
    assert result["rigid_global_pose"] is None
    assert "affine_normalization" in result["rigid_global_pose_reason"]
    assert np.all(np.isfinite(result["projection_world_to_tracking"]))


def test_camera_center_denominator_cancellation_and_large_affine_position():
    transform = _pose(translation=[1., 0, 0])
    projective = np.eye(4)
    projective[3, 0] = 1
    result = associate_pose(transform, _calibration(), projective)
    assert result["projective_camera_center"] is None
    assert "infinity" in result["projective_camera_center_reason"]
    affine = _pose(translation=[1e16, -2e16, 3e16])
    finite = associate_pose(transform, _calibration(), affine)
    assert finite["projective_camera_center_reason"] is None
    np.testing.assert_allclose(finite["projective_camera_center"], [1e16 - 1, -2e16, 3e16])
    assert finite["rigid_global_pose"] is not None


def test_sim3_tolerance_is_explicit_and_never_relaxes_local_rigid_validation():
    homography = np.diag([2., 2.00001, 2., 1.])
    accepted = associate_pose(np.eye(4), _calibration(), homography, sim3_tolerance=1e-5)
    rejected = associate_pose(np.eye(4), _calibration(), homography, sim3_tolerance=1e-7)
    assert accepted["rigid_global_pose"] is not None
    assert accepted["sim3_fit_relative_error"] < 1e-5
    assert rejected["rigid_global_pose"] is None
    nonrigid = np.diag([1.01, 1., 1., 1.])
    with pytest.raises(ValueError, match="orthonormal"):
        associate_pose(nonrigid, _calibration(), np.eye(4), sim3_tolerance=0.1)


def test_tiny_perspective_row_in_distant_local_pose_is_not_a_rigid_transform():
    transform = _pose(translation=[1e9, 0, 0])
    transform[3, 0] = 1e-8
    with pytest.raises(ValueError, match="homogeneous row"):
        associate_pose(transform, _calibration(), np.eye(4))


def test_denominator_numerical_degeneracy_is_invariant_to_homography_scale():
    transform = _pose(translation=[1., 0, 0])
    homography = np.eye(4)
    homography[3, 0] = 1 - np.finfo(np.float64).eps
    for scale in (1., -1e-80, 1e80):
        result = associate_pose(transform, _calibration(), homography * scale)
        assert result["projective_camera_center"] is None
        assert "numerically_degenerate" in result["projective_camera_center_reason"]


@pytest.mark.parametrize("argument,replacement,error", [
    ("T", np.eye(3), "shape"),
    ("K", np.eye(4), "shape"),
    ("H", np.eye(3), "shape"),
    ("T", np.full((4, 4), np.nan), "finite"),
    ("K", np.full((3, 3), np.inf), "finite"),
    ("H", np.full((4, 4), -np.inf), "finite"),
    ("K", np.diag([1., 0, 1.]), "invertible"),
    ("H", np.diag([1., 1., 1., 0]), "invertible"),
    ("T", np.diag([-1., 1., 1., 1.]), "proper rotation"),
    ("T", np.diag([2., 2., 2., 1.]), "orthonormal"),
    ("T", np.diag([1., 1., 1., 2.]), "homogeneous row"),
    ("T", np.eye(4, dtype=np.complex128) * (1 + 1j), "real matrix"),
    ("K", [["invalid"] * 3] * 3, "real matrix"),
])
def test_invalid_inputs_fail_explicitly(argument, replacement, error):
    values = {"T": np.eye(4), "K": _calibration(), "H": np.eye(4)}
    values[argument] = replacement
    with pytest.raises(ValueError, match=error):
        associate_pose(values["T"], values["K"], values["H"])


@pytest.mark.parametrize("tolerance", [0, -1, 1, np.nan, np.inf, True, "1e-5", [1e-5], 1j, 10 ** 1000])
def test_invalid_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError, match="sim3_tolerance"):
        associate_pose(np.eye(4), _calibration(), np.eye(4), tolerance)


def test_float32_rounded_rotation_and_output_ownership():
    transform = _pose(_rotation([0.7, 0.2, -0.1], 0.71), [0.2, 0.3, -0.1]).astype(np.float32)
    calibration = _calibration().astype(np.float32)
    homography = _pose(translation=[2., 3., 4.]).astype(np.float32)
    originals = [value.copy() for value in (transform, calibration, homography)]
    first = associate_pose(transform, calibration, homography)
    second = associate_pose(transform, calibration, homography)
    np.testing.assert_allclose(first["T_anchor_from_current"] @ transform, np.eye(4), atol=1e-14)
    np.testing.assert_allclose(first["projection_world_to_tracking"] @ homography,
                               calibration @ transform[:3], atol=1e-12)
    np.testing.assert_allclose(first["rigid_global_pose"][:3, :3].T @ first["rigid_global_pose"][:3, :3],
                               np.eye(3), atol=1e-14)
    for original, actual in zip(originals, (transform, calibration, homography)):
        np.testing.assert_array_equal(actual, original)
    for name in ("T_anchor_from_current", "projective_homography_world_from_current",
                 "projection_world_to_tracking", "projective_camera_center", "rigid_global_pose"):
        assert first[name].dtype == np.float64
        assert not np.shares_memory(first[name], second[name])
        assert not any(np.shares_memory(first[name], value) for value in (transform, calibration, homography))
    first["projection_world_to_tracking"][:] = 0
    assert np.any(second["projection_world_to_tracking"] != 0)
