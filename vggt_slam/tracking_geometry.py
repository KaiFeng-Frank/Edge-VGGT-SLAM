"""Associate an anchor-local tracking pose with VGGT's projective map gauge.

VGGT/SL4 node matrices are homographies, not generally rigid camera poses.  This
module therefore exposes the exact global projection separately from an optional
rigid pose.  Distances remain in model/map units; no metric scale is established.
Only NumPy is required, and the caller must snapshot the anchor homography and
its graph revision together if the graph can change concurrently.
"""

import numpy as np


_RIGID_TOLERANCE = 1e-6
_HOMOGENEOUS_EPSILON = 64 * np.finfo(np.float64).eps


def _matrix(value, shape, name):
    try:
        raw = np.asarray(value)
        if np.iscomplexobj(raw):
            raise ValueError(f"{name} must be real-valued")
        matrix = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a real matrix with shape {shape}") from error
    if matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _inverse(matrix, name):
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            inverse = np.linalg.inv(matrix)
    except (np.linalg.LinAlgError, FloatingPointError) as error:
        raise ValueError(f"{name} must be invertible with a finite inverse") from error
    if not np.all(np.isfinite(inverse)):
        raise ValueError(f"{name} must be invertible with a finite inverse")
    return inverse


def _rigid_inverse(transform):
    rotation = transform[:3, :3]
    # PnP's canonical SE3 row is exact.  A small perspective term in this row
    # can have a large effect for a distant translated camera, so accepting it
    # with a rotation-style tolerance would admit a nonrigid transform.
    if not np.array_equal(transform[3], [0, 0, 0, 1]):
        raise ValueError("T_current_from_anchor must have homogeneous row [0, 0, 0, 1]")
    with np.errstate(over="ignore", invalid="ignore"):
        orthogonal = np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=_RIGID_TOLERANCE)
        determinant = np.linalg.det(rotation)
    if not orthogonal:
        raise ValueError("T_current_from_anchor must have an orthonormal rotation")
    if not np.isclose(determinant, 1, rtol=0, atol=_RIGID_TOLERANCE):
        raise ValueError("T_current_from_anchor must have a proper rotation (determinant +1)")
    # A full inverse preserves the exact projection identity for rotations
    # rounded to float32 but still within the rigid validation tolerance.
    return _inverse(transform, "T_current_from_anchor")


def _camera_center(homography, anchor_from_current, world_from_current):
    # Test the denominator relative to the dot-product terms, not relative to
    # XYZ: an affine camera can legitimately be far from the world origin.
    # Scaling each vector first makes this test independent of H's arbitrary
    # homogeneous scale and avoids overflowing the magnitude bound.
    row = homography[3]
    anchor_center = anchor_from_current[:, 3]
    scaled_row = row / np.max(np.abs(row))
    scaled_center = anchor_center / np.max(np.abs(anchor_center))
    terms = scaled_row * scaled_center
    denominator = np.sum(terms)
    bound = np.sum(np.abs(terms))
    if abs(denominator) <= _HOMOGENEOUS_EPSILON * bound:
        return None, "projective_camera_center_at_infinity_or_numerically_degenerate"
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            center = world_from_current[:3, 3] / world_from_current[3, 3]
    except FloatingPointError:
        return None, "projective_camera_center_is_not_finite"
    if not np.all(np.isfinite(center)):
        return None, "projective_camera_center_is_not_finite"
    return center, None


def _positive_sim3(homography, tolerance):
    """Return a tolerance-level positive Sim3 fit, without repairing other gauges."""
    if homography[3, 3] == 0:
        return None, None, None, "anchor_homography_has_no_affine_normalization"
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            normalized = homography / homography[3, 3]
    except FloatingPointError:
        return None, None, None, "anchor_homography_has_no_finite_affine_normalization"
    if not np.all(np.isfinite(normalized)):
        return None, None, None, "anchor_homography_has_no_finite_affine_normalization"
    if not np.allclose(normalized[3], [0, 0, 0, 1], rtol=0, atol=tolerance):
        return None, None, None, "anchor_homography_is_not_affine"
    try:
        left, singular_values, right = np.linalg.svd(normalized[:3, :3])
    except np.linalg.LinAlgError:
        return None, None, None, "anchor_sim3_validation_did_not_converge"
    if not np.all(np.isfinite(singular_values)):
        return None, None, None, "anchor_sim3_scale_is_not_finite"
    largest = singular_values[0]
    if largest <= 0 or singular_values[-1] <= 0:
        return None, None, None, "anchor_linear_part_is_degenerate"
    rotation = left @ right
    if np.linalg.det(rotation) <= 0:
        return None, None, None, "anchor_linear_part_is_a_reflection"
    # Avoid overflowing a direct mean for large, otherwise valid scales.
    scale = largest * np.mean(singular_values / largest)
    if not np.isfinite(scale) or scale <= 0:
        return None, None, None, "anchor_sim3_scale_is_not_finite_and_positive"
    relative_error = float(np.max(np.abs(singular_values / scale - 1)))
    if relative_error > tolerance:
        return None, None, relative_error, "anchor_linear_part_has_nonuniform_scale_or_shear"
    return rotation, float(scale), relative_error, None


def associate_pose(T_current_from_anchor, K_tracking, H_world_anchor, sim3_tolerance=1e-5):
    """Relate a local PnP pose to an anchor's SL4/projective map transform.

    ``T_current_from_anchor`` is the proper rigid object-to-camera transform
    returned by PnP: ``X_current = T_current_from_anchor @ X_anchor``.  Anchor
    points, translations, and depth use VGGT model units, not calibrated meters.
    Its homogeneous row must be exactly [0, 0, 0, 1]; rotation rounding up to
    1e-6 in orthogonality/determinant checks is accepted independently of the
    Sim3 tolerance.
    ``K_tracking`` is the invertible 3x3 calibration for the actual tracking
    image after its resize/crop/padding.  ``H_world_anchor`` maps homogeneous
    anchor camera points into the map's projective gauge.

    The returned homography maps current camera points into that same gauge.
    The 3x4 projection maps homogeneous world points into tracking-image pixels
    (divide by its third coordinate).  Its scale is arbitrary.  A finite
    projective center is a point in map coordinates, not a metric position.

    ``rigid_global_pose`` is camera-to-world orientation and center only when H,
    divided by its bottom-right entry, passes a *positive* Sim3 check: an affine
    last row, uniform positive singular values, and a proper rotation.  Fits
    within ``sim3_tolerance`` are accepted; the rotation uses their closest SO3
    factor.  Transporting local 3D points still requires ``sim3_scale`` as well
    as this pose.  No global metric scale or rigid gauge is asserted otherwise.

    Invalid/nonfinite/singular inputs raise ValueError.  A valid projective H
    that is not Sim3 still returns its projection and homography, with an empty
    rigid pose and an explicit reason.  None of the inputs is modified.
    """
    if (isinstance(sim3_tolerance, (bool, np.bool_))
            or not isinstance(sim3_tolerance, (int, float, np.integer, np.floating))):
        raise ValueError("sim3_tolerance must be finite and strictly between 0 and 1")
    try:
        tolerance = float(sim3_tolerance)
    except (ValueError, OverflowError) as error:
        raise ValueError("sim3_tolerance must be finite and strictly between 0 and 1") from error
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("sim3_tolerance must be finite and strictly between 0 and 1")
    transform = _matrix(T_current_from_anchor, (4, 4), "T_current_from_anchor")
    calibration = _matrix(K_tracking, (3, 3), "K_tracking")
    homography = _matrix(H_world_anchor, (4, 4), "H_world_anchor")
    anchor_from_current = _rigid_inverse(transform)
    _inverse(calibration, "K_tracking")
    anchor_from_world = _inverse(homography, "H_world_anchor")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            world_from_current = homography @ anchor_from_current
            projection = calibration @ transform[:3] @ anchor_from_world
    except FloatingPointError as error:
        raise ValueError("pose association exceeded finite floating-point range") from error
    if not (np.all(np.isfinite(world_from_current)) and np.all(np.isfinite(projection))):
        raise ValueError("pose association must produce finite matrices")
    center, center_reason = _camera_center(homography, anchor_from_current, world_from_current)
    rotation, scale, fit_error, rigid_reason = _positive_sim3(homography, tolerance)
    rigid_pose = None
    if rigid_reason is None:
        if center is None:
            rigid_reason = center_reason
        else:
            # Project the already validated local rotation onto SO3 solely to
            # remove input rounding, then compose with the validated Sim3 fit.
            left, _, right = np.linalg.svd(anchor_from_current[:3, :3])
            rigid_pose = np.eye(4, dtype=np.float64)
            rigid_pose[:3, :3] = rotation @ left @ right
            rigid_pose[:3, 3] = center
    return {
        "T_anchor_from_current": anchor_from_current,
        "projective_homography_world_from_current": world_from_current,
        "projection_world_to_tracking": projection,
        "projective_camera_center": center,
        "projective_camera_center_reason": center_reason,
        "rigid_global_pose": rigid_pose,
        "rigid_global_pose_reason": rigid_reason,
        "sim3_scale": scale,
        "sim3_fit_relative_error": fit_error,
        "sim3_tolerance": tolerance,
        "local_units": "vggt_model_units",
        "metric_scale_validated": False,
        "global_gauge": "projective",
    }
