"""Vectorized CPU NumPy reference for pinned DPVO online fast-BA geometry.

Derived mathematical contract: princeton-vl/DPVO commit
0ac95b656d1fda91c271d2a106460d19ad966fc7, projective_ops.py,
fastba/ba_cuda.cu and lietorch/include/{se3,so3}.h.
Copyright (c) 2022 Princeton Vision & Learning Lab, MIT license; see LICENSE.

Poses are world-to-camera [tx,ty,tz,qx,qy,qz,qw], tangent order is
translation then rotation, and retraction is LEFT Exp(dx)*pose. Patch
channel 2 is inverse depth. This reference implements inference geometry,
not training autograd, Sim3 loop closure, CUDA arithmetic equivalence,
learned Update, pose interpolation, or a complete VO lifecycle.

The v28 reference SHA256 is
13eb1e03eb4a5ed1bd1533549f2b9f4994abfaa61e0b55304b99e6fe77d813df.
Only normal-system assembly changes: batched small matrix products and ordered
np.add.at replace the Python edge loop. Jacobians, active-edge gates, Schur
solve, damping, SE3 and inverse-depth updates are unchanged. Self-frame edges
still combine Ji+Jj before assembly. Chunking bounds new temporary arrays and
preserves edge accumulation order; batched products may change FP roundoff.
No actual-data equivalence, speedup, CUDA or trajectory claim is made here.
"""
import math

import numpy as np

COMMIT = "0ac95b656d1fda91c271d2a106460d19ad966fc7"


def _finite_float(value, name, last=None):
    x = np.asarray(value)
    if x.dtype not in (np.dtype("float32"), np.dtype("float64")) or not np.isfinite(x).all():
        raise ValueError(name + " must be finite float32/float64")
    if last is not None and x.shape[-1:] != (last,):
        raise ValueError(name + " has incorrect last dimension")
    return x


def _poses(value):
    x = _finite_float(value, "poses", 7)
    if np.any(np.abs(np.linalg.norm(x[..., 3:], axis=-1) - 1) > 1e-3):
        raise ValueError("poses require near-unit xyzw quaternions; invalid poses are not repaired")
    return x


def _hat(x):
    x = np.asarray(x)
    out = np.zeros(x.shape[:-1] + (3, 3), x.dtype)
    out[..., 0, 1], out[..., 0, 2] = -x[..., 2], x[..., 1]
    out[..., 1, 0], out[..., 1, 2] = x[..., 2], -x[..., 0]
    out[..., 2, 0], out[..., 2, 1] = -x[..., 1], x[..., 0]
    return out


def quaternion_multiply(a, b):
    a, b = np.broadcast_arrays(a, b)
    xyz = a[..., 3:] * b[..., :3] + b[..., 3:] * a[..., :3] + np.cross(a[..., :3], b[..., :3])
    real = a[..., 3] * b[..., 3] - np.sum(a[..., :3] * b[..., :3], axis=-1)
    return np.concatenate((xyz, real[..., None]), axis=-1)


def quaternion_act(q, xyz):
    uv = 2 * np.cross(q[..., :3], xyz)
    return xyz + q[..., 3:] * uv + np.cross(q[..., :3], uv)


def _normalized_pose(pose, normalize):
    pose = _poses(pose)
    if normalize:
        pose = pose.copy()
        pose[..., 3:] /= np.linalg.norm(pose[..., 3:], axis=-1, keepdims=True)
    return pose


def se3_compose(a, b, *, normalize=True):
    a, b = np.broadcast_arrays(_normalized_pose(a, normalize), _normalized_pose(b, normalize))
    output = np.concatenate((a[..., :3] + quaternion_act(a[..., 3:], b[..., :3]),
                             quaternion_multiply(a[..., 3:], b[..., 3:])), axis=-1)
    return _normalized_pose(output, normalize)


def se3_inverse(pose, *, normalize=True):
    pose = _normalized_pose(pose, normalize)
    q = pose[..., 3:].copy()
    q[..., :3] *= -1
    return np.concatenate((-quaternion_act(q, pose[..., :3]), q), axis=-1)


def se3_matrix(pose, *, normalize=True):
    pose = _normalized_pose(pose, normalize)
    q = pose[..., 3:]
    cross = _hat(q[..., :3])
    out = np.zeros(pose.shape[:-1] + (4, 4), pose.dtype)
    out[..., :3, :3] = np.eye(3, dtype=pose.dtype) + 2 * q[..., 3:, None] * cross + 2 * (cross @ cross)
    out[..., :3, 3] = pose[..., :3]
    out[..., 3, 3] = 1
    return out


def se3_exp(xi, *, fastba=False):
    """Lie exponential; fastba=True preserves CUDA's tiny-angle t=tau rule."""
    xi = _finite_float(xi, "tangent", 6)
    out = np.empty(xi.shape[:-1] + (7,), xi.dtype)
    for tangent, value in zip(xi.reshape(-1, 6), out.reshape(-1, 7)):
        tau, phi = tangent[:3], tangent[3:]
        theta2 = float(phi @ phi)
        theta = math.sqrt(theta2)
        epsilon = 1e-4 if fastba else 1e-6
        if theta < epsilon:
            imag = .5 - theta2 / 48 + theta2 * theta2 / 3840
            real = 1 - theta2 / 8 + theta2 * theta2 / 384
        else:
            imag, real = math.sin(theta / 2) / theta, math.cos(theta / 2)
        value[3:6], value[6] = imag * phi, real
        if fastba and theta <= 1e-4:
            value[:3] = tau
        else:
            a = .5 - theta2 / 24 if theta < 1e-6 else (1 - math.cos(theta)) / theta2
            b = 1 / 6 - theta2 / 120 if theta < 1e-6 else (theta - math.sin(theta)) / (theta * theta2)
            first = np.cross(phi, tau)
            value[:3] = tau + a * first + b * np.cross(phi, first)
    return _normalized_pose(out, not fastba)


def se3_log(pose):
    """Pinned LieTorch atan-based logarithm, including its near-pi branch."""
    pose = _normalized_pose(pose, True)
    out = np.empty(pose.shape[:-1] + (6,), pose.dtype)
    for source, value in zip(pose.reshape(-1, 7), out.reshape(-1, 6)):
        q = source[3:].copy()
        norm = float(np.linalg.norm(q[:3]))
        w = float(q[3])
        if norm * norm < 1e-12:
            factor = 2 / w - 2 / 3 * norm * norm / (w * w * w)
        elif abs(w) < 1e-6:
            factor = (math.pi if w > 0 else -math.pi) / norm
        else:
            factor = 2 * math.atan(norm / w) / norm
        phi = factor * q[:3]
        theta = float(np.linalg.norm(phi))
        coefficient = 1 / 12 if theta < 1e-6 else (1 - theta * math.cos(theta / 2) / (2 * math.sin(theta / 2))) / (theta * theta)
        cross = _hat(phi)
        value[:3] = (np.eye(3, dtype=pose.dtype) - .5 * cross + coefficient * (cross @ cross)) @ source[:3]
        value[3:] = phi
    return out


def se3_retract(pose, xi, *, fastba=False):
    return se3_compose(se3_exp(xi, fastba=fastba), pose, normalize=not fastba)


def _inputs(poses, patches, intrinsics, ii, jj, kk):
    poses, patches = _poses(poses), _finite_float(patches, "patches")
    intrinsics = _finite_float(intrinsics, "intrinsics", 4)
    if poses.ndim != 3 or poses.shape[0] != 1 or patches.ndim != 5 or patches.shape[:1] != (1,) or patches.shape[2] != 3:
        raise ValueError("CPU reference requires B=1 poses[1,N,7], patches[1,K,3,P,P]")
    if intrinsics.shape != (1, poses.shape[1], 4):
        raise ValueError("intrinsics must be [1,N,4]")
    index = []
    for value, limit in ((ii, poses.shape[1]), (jj, poses.shape[1]), (kk, patches.shape[1])):
        x = np.asarray(value)
        if x.dtype != np.int64 or x.ndim != 1 or np.any(x < 0) or np.any(x >= limit):
            raise ValueError("edge IDs must be in-range int64 vectors")
        index.append(x)
    ii, jj, kk = index
    if not (ii.size == jj.size == kk.size):
        raise ValueError("edge ID lengths differ")
    if ii.size and (np.any(intrinsics[0, np.concatenate((ii, jj)), :2] <= 0)):
        raise ValueError("referenced intrinsics require positive focal lengths")
    return poses, patches, intrinsics, ii, jj, kk


def _transformed(poses, patches, intrinsics, ii, jj, kk, *, common_K=False, tonly=False):
    source_K = intrinsics[0, ii] if not common_K else np.broadcast_to(intrinsics[0, 0], (len(ii), 4))
    xy = patches[0, kk, :2].transpose(0, 2, 3, 1)
    source_xyz = np.concatenate(((xy - source_K[:, None, None, 2:]) / source_K[:, None, None, :2],
                                 np.ones(xy.shape[:-1] + (1,), xy.dtype)), axis=-1)
    relative = se3_compose(poses[0, jj], se3_inverse(poses[0, ii], normalize=not common_K), normalize=not common_K)
    if tonly:
        relative[..., 3:] = np.array([0, 0, 0, 1], relative.dtype)
    xyz = quaternion_act(relative[:, None, None, 3:], source_xyz) + relative[:, None, None, :3] * patches[0, kk, 2, ..., None]
    return xyz, relative


def projective_transform(poses, patches, intrinsics, ii, jj, kk, *, depth=False, valid=False, tonly=False):
    """Original projective_ops.transform projection/validity (no autograd)."""
    poses, patches, intrinsics, ii, jj, kk = _inputs(poses, patches, intrinsics, ii, jj, kk)
    xyz, _ = _transformed(poses, patches, intrinsics, ii, jj, kk, tonly=tonly)
    inverse = 1 / np.maximum(xyz[..., 2], .1)
    target_K = intrinsics[0, jj]
    xy = xyz[..., :2] * inverse[..., None] * target_K[:, None, None, :2] + target_K[:, None, None, 2:]
    if depth:
        xy = np.concatenate((xy, inverse[..., None]), axis=-1)
    if valid:
        return xy[None], (xyz[..., 2] > .2)[None].astype(xy.dtype)
    return xy[None]


def reproject(poses, patches, intrinsics, ii, jj, kk):
    """Online input to correlation: original projective_ops layout [1,E,2,P,P]."""
    return projective_transform(poses, patches, intrinsics, ii, jj, kk).transpose(0, 1, 4, 2, 3).copy()


def point_cloud(poses, patches, intrinsics, source_frame_ids):
    """Original homogeneous world points [1,K,P,P,4], before division by W."""
    count = patches.shape[1]
    ids = np.asarray(source_frame_ids)
    kk = np.arange(count, dtype=np.int64)
    poses, patches, intrinsics, ids, _, kk = _inputs(poses, patches, intrinsics, ids, ids, kk)
    K = intrinsics[0, ids]
    xy = patches[0, :, :2].transpose(0, 2, 3, 1)
    camera = np.concatenate(((xy - K[:, None, None, 2:]) / K[:, None, None, :2],
                             np.ones(xy.shape[:-1] + (1,), xy.dtype)), axis=-1)
    inverse_pose = se3_inverse(poses[0, ids])
    depth = patches[0, :, 2, ..., None]
    world = quaternion_act(inverse_pose[:, None, None, 3:], camera) + inverse_pose[:, None, None, :3] * depth
    return np.concatenate((world, depth), axis=-1)[None]


def flow_mag(poses, patches, intrinsics, ii, jj, kk, beta=.3):
    """Original full-motion/translation-only flow used for keyframe selection."""
    if not isinstance(beta, (int, float)) or not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("beta must be finite in [0,1]")
    source = projective_transform(poses, patches, intrinsics, ii, ii, kk)
    full, valid = projective_transform(poses, patches, intrinsics, ii, jj, kk, valid=True)
    translation = projective_transform(poses, patches, intrinsics, ii, jj, kk, tonly=True)
    return beta * np.linalg.norm(full - source, axis=-1) + (1 - beta) * np.linalg.norm(translation - source, axis=-1), valid > .5


def _adjoint(relative):
    matrix = se3_matrix(relative, normalize=False)
    adjoint = np.zeros(relative.shape[:-1] + (6, 6), relative.dtype)
    adjoint[..., :3, :3] = adjoint[..., 3:, 3:] = matrix[..., :3, :3]
    adjoint[..., :3, 3:] = _hat(relative[..., :3]) @ matrix[..., :3, :3]
    return adjoint


def fastba_linearize(poses, patches, intrinsics, target, weight, ii, jj, kk):
    """Center projection and online fastba Jacobians, gate and residual."""
    poses, patches, intrinsics, ii, jj, kk = _inputs(poses, patches, intrinsics, ii, jj, kk)
    if patches.shape[-2:] != (3, 3):
        raise ValueError("pinned online fastba requires 3x3 patches with center[1,1]")
    selected = intrinsics[0, np.concatenate((ii, jj))]
    if np.any(selected != intrinsics[0, 0]):
        raise ValueError("online fastba kernel uses intrinsics[0]; variable K is unsupported")
    target, weight = _finite_float(target, "target"), _finite_float(weight, "weight")
    if target.shape != (1, len(ii), 2) or weight.shape != target.shape or np.any(weight < 0):
        raise ValueError("target/weight must be [1,E,2] with nonnegative weights")
    xyz, relative = _transformed(poses, patches, intrinsics, ii, jj, kk, common_K=True)
    xyz = xyz[:, 1, 1]
    X, Y, Z = xyz.T
    fx, fy, cx, cy = intrinsics[0, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = xyz[:, :2] / Z[:, None] * np.array([fx, fy], xyz.dtype) + np.array([cx, cy], xyz.dtype)
    residual = target[0] - xy
    active = (np.linalg.norm(residual, axis=-1) < 128) & (Z > .2) & (xy[:, 0] > -64) & (xy[:, 1] > -64) & (xy[:, 0] < 2 * cx + 64) & (xy[:, 1] < 2 * cy + 64)
    invz = np.zeros_like(Z)
    invz[Z >= .2] = 1 / Z[Z >= .2]
    Jp = np.zeros((len(ii), 2, 3), xyz.dtype)
    Jp[:, 0, 0], Jp[:, 0, 2] = fx * invz, -fx * X * invz * invz
    Jp[:, 1, 1], Jp[:, 1, 2] = fy * invz, -fy * Y * invz * invz
    action = np.concatenate((patches[0, kk, 2, 1, 1, None, None] * np.broadcast_to(np.eye(3, dtype=xyz.dtype), (len(ii), 3, 3)), -_hat(xyz)), axis=-1)
    Jj = Jp @ action
    Ji = -Jj @ _adjoint(relative)
    Jz = (Jp @ relative[..., :3, None])[..., 0]
    return {"projection": xy[None], "residual": residual[None], "active": active,
            "weighted_precision": weight[0] * active[:, None], "Ji": Ji, "Jj": Jj, "Jz": Jz}


def fastba_system(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1):
    """Dense online system; ordered batched assembly, unchanged Schur solve."""
    if type(t0) is not int or type(t1) is not int or not 0 <= t0 <= t1 <= poses.shape[1] or np.any(ii >= t1) or np.any(jj >= t1):
        raise ValueError("invalid optimized pose interval or edges beyond t1")
    linear = fastba_linearize(poses, patches, intrinsics, target, weight, ii, jj, kk)
    unique, inverse = np.unique(kk, return_inverse=True)
    dtype = poses.dtype
    n, m = 6 * (t1 - t0), len(unique)
    if n * n + n * m > 16 * 1024 * 1024:
        raise ValueError("dense reference system exceeds explicit 16M-element budget")
    B, E = np.zeros((n, n), dtype), np.zeros((n, m), dtype)
    C, v, u = np.zeros(m, dtype), np.zeros(n, dtype), np.zeros(m, dtype)
    active_edges = np.flatnonzero(linear["active"])
    # Each chunk owns O(256 * 2 * 2 * 6 * 6) temporary values/indices,
    # independent of the dense pose/depth system size. add.at is required:
    # ordinary advanced-index += loses contributions at repeated indices.
    for begin in range(0, len(active_edges), 256):
        edges = active_edges[begin:begin + 256]
        frames = np.stack((ii[edges], jj[edges]), axis=1)
        patch = inverse[edges]
        valid_slots = (frames >= t0) & (frames < t1)
        jacobian = np.stack((linear["Ji"][edges], linear["Jj"][edges]), axis=1)
        # The scalar dictionary has a single block when source==destination.
        # Combine first, rather than separately accumulating four products;
        # this preserves the scalar Jacobian-sum-before-product order.
        same_frame = frames[:, 0] == frames[:, 1]
        jacobian[same_frame, 0] += jacobian[same_frame, 1]
        valid_slots[same_frame, 1] = False
        rows = 6 * (frames - t0)[..., None] + np.arange(6, dtype=np.int64)
        w = linear["weighted_precision"][edges]
        r = linear["residual"][0, edges]
        Jz = linear["Jz"][edges]
        wr, wz = w * r, w * Jz
        transposed = jacobian.swapaxes(-1, -2)
        rhs = (transposed @ wr[:, None, :, None])[..., 0]
        coupling = (transposed @ wz[:, None, :, None])[..., 0]
        pair_values = transposed[:, :, None] @ (w[:, None, :, None] * jacobian)[:, None]
        pairs = valid_slots[:, :, None] & valid_slots[:, None, :]
        pair_indices = rows[:, :, None, :, None] * n + rows[:, None, :, None, :]
        coupling_indices = rows * m + patch[:, None, None]
        # Flat traversal is edge -> source/destination slot -> block entries,
        # so repeated frame/patch contributions arrive in scalar edge order.
        np.add.at(v, rows[valid_slots].reshape(-1), rhs[valid_slots].reshape(-1))
        np.add.at(E.reshape(-1), coupling_indices[valid_slots].reshape(-1),
                  coupling[valid_slots].reshape(-1))
        np.add.at(B.reshape(-1), pair_indices[pairs].reshape(-1),
                  pair_values[pairs].reshape(-1))
        np.add.at(C, patch, (Jz[:, None, :] @ wz[..., None])[:, 0, 0])
        np.add.at(u, patch, (Jz[:, None, :] @ wr[..., None])[:, 0, 0])
    damping = np.asarray(lmbda, dtype=dtype).reshape(-1)
    if damping.size not in (1, m) or not np.isfinite(damping).all() or np.any(damping <= 0):
        raise ValueError("lambda must be positive finite scalar or sorted-unique-patch vector")
    Q = 1 / (C + damping)
    S = B - (E * Q[None]) @ E.T
    y = v - E @ (Q * u)
    if n:
        S = S.copy()
        diagonal = np.diag_indices(n)
        S[diagonal] += 1 + 1e-4 * S[diagonal]
        chol = np.linalg.cholesky(S)  # Fail explicitly; no silent zero/fallback.
        dx = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
    else:
        dx = np.empty(0, dtype)
    dz = Q * (u - E.T @ dx)
    if not np.isfinite(dx).all() or not np.isfinite(dz).all():
        raise FloatingPointError("nonfinite BA update")
    return {"B": B, "E": E, "C": C, "v": v, "u": u, "Q": Q, "S": S, "y": y,
            "dx": dx.reshape(-1, 6), "dz": dz, "unique_patch_ids": unique,
            "active_edge_count": int(linear["active"].sum()),
            "weighted_squared_residual": float(np.sum(linear["weighted_precision"][linear["active"]] * linear["residual"][0, linear["active"]] ** 2))}


def BA(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, M, iterations=2, eff_impl=False):
    """Original online in-place API; also returns per-iteration diagnostics.

Copy-on-success prevents a failed solve from partially updating the caller.
An eff_impl=True call uses the same dense mathematics, not CUDA's optimized
block-E storage; it is explicitly a CPU reference, not a speed claim.
"""
    poses, patches, intrinsics, ii, jj, kk = _inputs(poses, patches, intrinsics, ii, jj, kk)
    if not poses.flags.writeable or not patches.flags.writeable:
        raise ValueError("BA in-place destinations must be writable")
    if type(iterations) is not int or iterations < 0 or iterations > 100 or type(M) is not int or M < 1 or type(eff_impl) is not bool:
        raise ValueError("invalid BA iteration/patch-count options")
    if type(t0) is not int or type(t1) is not int or not 0 <= t0 <= t1 <= poses.shape[1] or np.any(ii >= t1) or np.any(jj >= t1):
        raise ValueError("invalid optimized pose interval or edges beyond t1")
    work_pose, work_patch = poses.copy(), patches.copy()
    records = []
    for iteration in range(iterations):
        system = fastba_system(work_pose, work_patch, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1)
        work_pose[0, t0:t1] = se3_retract(work_pose[0, t0:t1], system["dx"], fastba=True)
        ids = system["unique_patch_ids"]
        updated = work_patch[0, ids, 2, 0, 0] + system["dz"]
        updated = np.maximum(np.where(updated > 20, 1, updated), 1e-4)
        work_patch[0, ids, 2] = updated[:, None, None]
        records.append({"iteration": iteration, "active_edges": system["active_edge_count"],
                        "weighted_squared_residual_before": system["weighted_squared_residual"],
                        "pose_update_norm": float(np.linalg.norm(system["dx"])),
                        "inverse_depth_update_norm": float(np.linalg.norm(system["dz"]))})
    np.copyto(poses, work_pose)
    np.copyto(patches, work_patch)
    return {"iterations": records, "optimized_pose_interval": [t0, t1],
            "dense_reference": True, "upstream_eff_impl_requested": eff_impl,
            "CUDA_bitwise_equivalence_validated": False,
            "VO_or_trajectory_accuracy_validated": False}
