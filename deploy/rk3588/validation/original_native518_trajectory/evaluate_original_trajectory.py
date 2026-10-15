#!/usr/bin/env python3
"""NumPy-only saved TUM trajectory metrics; no model or accelerator execution.

TUM rows: timestamp tx ty tz qx qy qz qw, camera-to-world poses.
https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
RPE E = inv(inv(G_i) G_j) (inv(P_i) P_j), as defined by TUM:
https://cvg.cit.tum.de/_media/spezial/bib/sturm12lecturenotes.pdf

ATE uses a fitted, positive-scale Sim3. RPE uses that SAME fitted scale and
adjacent matched poses by default, not a fixed one-second interval. Report the
matched timestamps and intervals when comparing the original CPU/NPU branches.
This helper neither accepts a precision change nor validates metric scale.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def validated_rows(rows):
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 8 or len(rows) < 3 or not np.isfinite(rows).all():
        raise ValueError('Need at least three finite TUM rows with eight columns')
    if not np.all(np.diff(rows[:, 0]) > 0):
        raise ValueError('TUM timestamps must be strictly increasing and unique')
    if np.any(np.linalg.norm(rows[:, 4:8], axis=1) < 1e-12):
        raise ValueError('TUM quaternion norm must be nonzero')
    return rows


def read_tum(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            values = line.split()
            if len(values) != 8:
                raise ValueError('TUM trajectory lines must have exactly eight columns')
            rows.append([float(value) for value in values])
    return validated_rows(rows)


def poses_from_tum(rows):
    rows = validated_rows(rows)
    quaternion = rows[:, 4:8] / np.linalg.norm(rows[:, 4:8], axis=1)[:, None]
    x, y, z, w = quaternion.T
    rotation = np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                         2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                         2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=1).reshape(-1, 3, 3)
    poses = np.broadcast_to(np.eye(4), (len(rows), 4, 4)).copy()
    poses[:, :3, :3] = rotation
    poses[:, :3, 3] = rows[:, 1:4]
    return poses


def associate_timestamps(estimated, ground_truth, *, max_difference=0.02, offset=0.):
    """One-to-one closest candidates, then chronological estimated order.

    Offset is ADDED to estimated timestamps. No interpolation, extrapolation or
    retimestamping is performed; unmatched estimates remain visible in metrics.
    """
    estimated, ground_truth = validated_rows(estimated), validated_rows(ground_truth)
    if not math.isfinite(max_difference) or max_difference <= 0 or not math.isfinite(offset):
        raise ValueError('Association tolerance must be positive and finite; offset must be finite')
    gt_times = ground_truth[:, 0]
    candidates = []
    for i, stamp in enumerate(estimated[:, 0]):
        shifted = stamp + offset
        lo = np.searchsorted(gt_times, shifted-max_difference, side='left')
        hi = np.searchsorted(gt_times, shifted+max_difference, side='right')
        for j in range(int(lo), int(hi)):
            difference = abs(float(gt_times[j]-shifted))
            if difference <= max_difference:
                candidates.append((difference, i, j))
    selected, used_est, used_gt = [], set(), set()
    for _, i, j in sorted(candidates):
        if i not in used_est and j not in used_gt:
            selected.append((i, j))
            used_est.add(i)
            used_gt.add(j)
    selected.sort()
    if len(selected) < 3:
        raise ValueError('Fewer than three one-to-one timestamp matches')
    indices = np.array(selected, dtype=np.int64)
    if not np.all(np.diff(indices[:, 1]) > 0):
        raise ValueError('Closest one-to-one association is not chronological in ground truth')
    return indices[:, 0], indices[:, 1]


def fit_sim3(estimated_positions, ground_truth_positions):
    """Fit y = scale * rotation @ x + translation, determinant(rotation)=+1."""
    x = np.asarray(estimated_positions, dtype=np.float64)
    y = np.asarray(ground_truth_positions, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or len(x) < 3:
        raise ValueError('Sim3 requires matched Nx3 positions, N>=3')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Sim3 positions must be finite')
    x_mean, y_mean = np.mean(x, axis=0), np.mean(y, axis=0)
    xc, yc = x-x_mean, y-y_mean
    covariance = yc.T @ xc / len(x)
    if np.linalg.matrix_rank(covariance) < 2:
        raise ValueError('Sim3 rotation/scale is underdetermined for this trajectory')
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1] = -1.
    rotation = (u * correction) @ vt
    variance = np.mean(np.sum(xc*xc, axis=1))
    scale = float(np.sum(singular*correction)/variance)
    translation = y_mean-scale*(rotation @ x_mean)
    if scale <= 0 or not math.isfinite(scale) or not np.isfinite(translation).all():
        raise ValueError('Sim3 fit is not a finite positive-scale alignment')
    return scale, rotation, translation


def align_poses(poses, scale, rotation, translation):
    aligned = np.asarray(poses, dtype=np.float64).copy()
    aligned[:, :3, 3] = scale*(aligned[:, :3, 3] @ rotation.T) + translation
    aligned[:, :3, :3] = rotation @ aligned[:, :3, :3]
    return aligned


def relative_pose_errors(estimated, ground_truth, *, delta_frames=1):
    """Translation norm in metres and SO3 angle in radians, E=DeltaG^-1 DeltaP."""
    estimated = np.asarray(estimated, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    if (estimated.shape != ground_truth.shape or estimated.ndim != 3 or estimated.shape[1:] != (4, 4)
            or not np.isfinite(estimated).all() or not np.isfinite(ground_truth).all()
            or type(delta_frames) is not int or not 0 < delta_frames < len(estimated)):
        raise ValueError('RPE requires finite matched Nx4x4 poses and 0<delta_frames<N')
    def relative(poses):
        first, second = poses[:-delta_frames], poses[delta_frames:]
        inverse_rotation = first[:, :3, :3].transpose(0, 2, 1)
        return (inverse_rotation @ second[:, :3, :3],
                (inverse_rotation @ (second[:, :3, 3]-first[:, :3, 3])[:, :, None])[:, :, 0])
    rp, tp = relative(estimated)
    rg, tg = relative(ground_truth)
    inverse_rg = rg.transpose(0, 2, 1)
    error_rotation = inverse_rg @ rp
    error_translation = (inverse_rg @ (tp-tg)[:, :, None])[:, :, 0]
    cosine = np.clip((np.trace(error_rotation, axis1=1, axis2=2)-1.)/2., -1., 1.)
    skew = np.stack((error_rotation[:, 2, 1]-error_rotation[:, 1, 2],
                     error_rotation[:, 0, 2]-error_rotation[:, 2, 0],
                     error_rotation[:, 1, 0]-error_rotation[:, 0, 1]), axis=1)
    sine = np.linalg.norm(skew, axis=1)/2.
    return np.linalg.norm(error_translation, axis=1), np.arctan2(sine, cosine)


def error_statistics(values, units):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Error statistics require a nonempty finite vector')
    return dict(units=units, count=len(values), rmse=float(np.sqrt(np.mean(values*values))),
                mean=float(np.mean(values)), median=float(np.median(values)), std=float(np.std(values)),
                minimum=float(np.min(values)), maximum=float(np.max(values)))


def evaluate_tum(estimated, ground_truth, *, max_difference=0.02, offset=0., delta_frames=1):
    estimated, ground_truth = validated_rows(estimated), validated_rows(ground_truth)
    est_indices, gt_indices = associate_timestamps(estimated, ground_truth, max_difference=max_difference, offset=offset)
    matched_est, matched_gt = estimated[est_indices], ground_truth[gt_indices]
    est_poses, gt_poses = poses_from_tum(matched_est), poses_from_tum(matched_gt)
    scale, rotation, translation = fit_sim3(matched_est[:, 1:4], matched_gt[:, 1:4])
    aligned = align_poses(est_poses, scale, rotation, translation)
    ate = np.linalg.norm(aligned[:, :3, 3]-gt_poses[:, :3, 3], axis=1)
    translation_rpe, rotation_rpe = relative_pose_errors(aligned, gt_poses, delta_frames=delta_frames)
    est_intervals = matched_est[delta_frames:, 0]-matched_est[:-delta_frames, 0]
    gt_intervals = matched_gt[delta_frames:, 0]-matched_gt[:-delta_frames, 0]
    differences = np.abs(matched_gt[:, 0]-(matched_est[:, 0]+offset))
    return dict(schema='original_VGGT_saved_TUM_Sim3_ATE_RPE', scope='offline saved trajectory only',
                quality_accepted=False, realtime_validated=False, metric_scale_validated=False,
                association=dict(policy='closest one-to-one candidates; no interpolation; original timestamps',
                    tolerance_seconds=max_difference, offset_added_to_estimated_seconds=offset,
                    estimated_count=len(estimated), ground_truth_count=len(ground_truth), matched_count=len(est_indices),
                    estimated_matched_fraction=len(est_indices)/len(estimated),
                    unmatched_estimated_count=len(estimated)-len(est_indices),
                    maximum_abs_timestamp_difference_seconds=float(np.max(differences)),
                    matches=[dict(estimated_index=int(i), ground_truth_index=int(j),
                                  estimated_timestamp=float(estimated[i, 0]), ground_truth_timestamp=float(ground_truth[j, 0]))
                             for i, j in zip(est_indices, gt_indices)]),
                alignment=dict(method='positive-scale Sim3 from matched positions; same scale used for RPE',
                               scale=scale, rotation=rotation.tolist(), translation=translation.tolist()),
                ate_translation=error_statistics(ate, 'metres'),
                rpe_translation=error_statistics(translation_rpe, 'metres'),
                rpe_rotation=error_statistics(np.degrees(rotation_rpe), 'degrees'),
                rpe_pair_policy=dict(delta_matched_frames=delta_frames, pair_count=len(translation_rpe),
                    estimated_interval_seconds=dict(minimum=float(np.min(est_intervals)), mean=float(np.mean(est_intervals)), maximum=float(np.max(est_intervals))),
                    ground_truth_interval_seconds=dict(minimum=float(np.min(gt_intervals)), mean=float(np.mean(gt_intervals)), maximum=float(np.max(gt_intervals)))))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ground_truth', type=Path)
    parser.add_argument('estimated', type=Path)
    parser.add_argument('--max-difference', type=float, default=0.02)
    parser.add_argument('--offset', type=float, default=0.)
    parser.add_argument('--delta-frames', type=int, default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = evaluate_tum(read_tum(args.estimated), read_tum(args.ground_truth),
                          max_difference=args.max_difference, offset=args.offset, delta_frames=args.delta_frames)
    report['source_sha256'] = file_sha256(__file__)
    report['files'] = dict(ground_truth_sha256=file_sha256(args.ground_truth), estimated_sha256=file_sha256(args.estimated))
    encoded = json.dumps(report, indent=2, allow_nan=False)+'\n'
    if args.output is not None:
        args.output.write_text(encoded)
    print(encoded, end='')


if __name__ == '__main__':
    main()
