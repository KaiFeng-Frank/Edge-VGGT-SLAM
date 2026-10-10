#!/usr/bin/env python3
"""Analyze three saved CPU FP32 pairs; never decode images or run a model.

GT is used only for pair-relative post-experiment rotation/direction diagnostics.
Depth scales are arbitrary: one positive least-squares scale per two-frame
batch is descriptive, fitted and scored on the same pixels, without depth GT.
No complete-dataset ATE/RPE, calibration, metric-scale or realtime gate is run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from scripts import evaluate_edge_trajectory as geometry

CHECKPOINT_SHA = 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
CAPTURE_SCRIPT_SHA = '64adbcbe7f6e013aa738b8bed57d21ef8fde10c4b63bd7c647e9fa2137a79d7c'
LOAD_FN_SHA = 'b0fe762f65497dca9a22069a3e4a37fd3508398c55383a8c1cb85c403fb3e9dd'
GT_TOLERANCE_SECONDS = .02
MIN_GT_TRANSLATION_METRES = .01


def pixel_affine(raw_hw, input_shape, image_size):
    """Verify the recorded crop shape, then restore original pixel centres."""
    if (len(raw_hw) != 2 or any(type(n) is not int or n <= 0 for n in raw_hw)
            or len(input_shape) != 4 or input_shape[:2] != [2, 3]):
        raise ValueError('Invalid recorded raw/input dimensions')
    height, width = raw_hw
    new_width = image_size
    new_height = max(14, round(height * (new_width / width) / 14) * 14)
    crop_y = (new_height - image_size) // 2 if new_height > image_size else 0
    expected_hw = [min(new_height, image_size), new_width]
    if input_shape[2:] != expected_hw:
        raise ValueError('Recorded input shape differs from the pinned original crop preprocessing')
    sx, sy = new_width / width, new_height / height
    A = np.array([[sx, 0., (sx - 1.) / 2.],
                  [0., sy, (sy - 1.) / 2. - crop_y], [0., 0., 1.]])
    return A, dict(raw_hw=raw_hw, actual_input_hw=expected_hw,
                   resized_hw_before_crop=[new_height, new_width], crop_y=crop_y,
                   pixel_centres='u_model=(u_raw+.5)*sx-.5; v_model=(v_raw+.5)*sy-.5-crop_y')


def tensor_sha(value):
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast('B')).hexdigest()


def load_capture(path, image_size):
    path = Path(path)
    receipt_file = path / 'capture.json'
    receipt = geometry.strict_json(receipt_file.read_text())
    if (receipt['status'] != 'completed_fixed_input_diagnostic' or receipt['image_size'] != image_size
            or receipt['max_pairs'] != 3 or receipt['script_sha256'] != CAPTURE_SCRIPT_SHA
            or receipt['device'] != 'cpu' or receipt['precision'] != 'float32'
            or receipt['ORB_used'] is not False or receipt['npu_used'] is not False
            or receipt['checkpoint']['sha256'] != CHECKPOINT_SHA or receipt['checkpoint']['bytes'] != 5026874952
            or receipt['source_hashes'].get('third_party/vggt/vggt/utils/load_fn.py') != LOAD_FN_SHA
            or len(receipt['pairs']) != 3 or len(receipt['selected_inputs']) != 3):
        raise ValueError('Capture is not the pinned three-pair original CPU FP32 diagnostic')
    if receipt['rectification']['R'] != np.eye(3).tolist() or receipt['rectification']['crop'] is not None:
        raise ValueError('GT comparison requires unchanged optical axes and no rectification crop')
    result, hashes = [], {'capture.json': geometry.sha(receipt_file)}
    for i, (record, selected) in enumerate(zip(receipt['pairs'], receipt['selected_inputs']), 1):
        if (record['revision'] != i or selected['revision'] != i
                or record['frame_ids'] != selected['frame_ids'] or len(record['frame_ids']) != 2
                or record['file'] != f'predictions_{i:04d}.npz'):
            raise ValueError('Capture pair revision/identity is inconsistent')
        file = path / record['file']
        digest = geometry.sha(file)
        if digest != record['sha256']:
            raise ValueError('Saved prediction file SHA differs from its capture receipt')
        with np.load(file, allow_pickle=False) as saved:
            if set(saved.files) != {'depth', 'depth_conf', 'intrinsic', 'extrinsic', 'pose_enc', 'frame_ids'}:
                raise ValueError('Unexpected saved original prediction keys')
            if not np.array_equal(saved['frame_ids'], record['frame_ids']):
                raise ValueError('NPZ frame IDs differ from the exact recorded pair')
            arrays = {name: saved[name].copy() for name in record['arrays']}
        for name, value in arrays.items():
            info = record['arrays'][name]
            if (value.dtype != np.float32 or list(value.shape) != info['shape']
                    or value.nbytes != info['bytes'] or not np.isfinite(value).all()
                    or tensor_sha(value) != info['tensor_sha256']):
                raise ValueError('Saved tensor values/header differ from their capture receipt')
        A, mapping = pixel_affine(receipt['rectification']['image_hw'], record['input_tensor']['shape'], image_size)
        h, w = mapping['actual_input_hw']
        expected = {'depth': (1, 2, h, w, 1), 'depth_conf': (1, 2, h, w),
                    'intrinsic': (1, 2, 3, 3), 'extrinsic': (1, 2, 3, 4), 'pose_enc': (1, 2, 9)}
        if set(arrays) != set(expected) or any(arrays[name].shape != shape for name, shape in expected.items()):
            raise ValueError('Unexpected original prediction tensor shapes')
        extrinsics = []
        for matrix in arrays['extrinsic'][0]:
            T = np.eye(4); T[:3] = matrix
            extrinsics.append(geometry.rigid(T, 'Original VGGT extrinsic'))
        result.append(dict(record=record, selected=selected, arrays=arrays, A=A, mapping=mapping,
                           relative=extrinsics[1] @ geometry.inverse(extrinsics[0])))
        hashes[record['file']] = digest
    return receipt, result, hashes


def verify_identity(low, high):
    for key in ('source_hashes', 'rectification', 'model', 'script_sha256'):
        if low[key] != high[key]:
            raise ValueError(f'112 and 518 {key} differ')
    for key in ('rgb_index', 'calibration', 'checkpoint'):
        if low[key]['sha256'] != high[key]['sha256']:
            raise ValueError(f'112 and 518 {key} SHA differ')
    if low['schedule'] != high['schedule']:
        raise ValueError('112 and 518 schedules differ')
    if low['selected_inputs'] != high['selected_inputs']:
        raise ValueError('112 and 518 exact source identities/SHA/schedule tokens differ')
    for pair in low['selected_inputs']:
        if len(pair['images']) != 2:
            raise ValueError('Recorded original source pair has the wrong count')
        for frame_id, image in zip(pair['frame_ids'], pair['images']):
            if (image['frame_id'] != frame_id or image['timestamp'] != frame_id
                    or image['schedule_token']['frame_id'] != frame_id
                    or image['schedule_token']['timestamp'] != frame_id
                    or image['samefile_validated'] is not True):
                raise ValueError('Original source identity was not exactly validated')


def positive_stats(values):
    values = np.asarray(values, np.float64)
    keep = np.isfinite(values) & (values > 0)
    positive = values[keep]
    return dict(pixels=int(values.size), positive_finite_pixels=int(keep.sum()),
                excluded_pixels=int(values.size - keep.sum()),
                min=float(positive.min()) if len(positive) else None,
                p05=float(np.percentile(positive, 5)) if len(positive) else None,
                median=float(np.median(positive)) if len(positive) else None,
                p95=float(np.percentile(positive, 95)) if len(positive) else None,
                max=float(positive.max()) if len(positive) else None)


def bilinear_sample(values, uv):
    """Float64 numerical sampling; every four-neighbour value must be valid."""
    x, y = uv[:, 0], uv[:, 1]
    h, w = values.shape
    valid = np.isfinite(uv).all(axis=1) & (x >= 0) & (y >= 0) & (x < w - 1) & (y < h - 1)
    rows = np.flatnonzero(valid)
    result = np.full(len(uv), np.nan)
    ix, iy = np.floor(x[rows]).astype(int), np.floor(y[rows]).astype(int)
    corners = np.column_stack((values[iy, ix], values[iy, ix + 1], values[iy + 1, ix], values[iy + 1, ix + 1]))
    good = np.isfinite(corners).all(axis=1) & (corners > 0).all(axis=1)
    dx, dy = x[rows] - ix, y[rows] - iy
    weights = np.column_stack(((1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy))
    result[rows[good]] = np.sum(corners[good] * weights[good], axis=1)
    return result


def residual_stats(low, high, scale=1.):
    delta = scale * low - high
    return dict(samples=len(low), mae=float(np.mean(np.abs(delta))), rmse=float(np.sqrt(np.mean(delta ** 2))),
                relative_l2=float(np.linalg.norm(delta) / np.linalg.norm(high)),
                mean_absolute_relative=float(np.mean(np.abs(delta) / high)),
                median_ratio_518_over_112=float(np.median(high / low)))


def compare_depth(low, high):
    dl = low['arrays']['depth'][0, ..., 0].astype(np.float64)
    dh = high['arrays']['depth'][0, ..., 0].astype(np.float64)
    y, x = np.indices(dl.shape[1:])
    points = np.column_stack((x.ravel(), y.ravel(), np.ones(x.size)))
    mapped = points @ (high['A'] @ np.linalg.inv(low['A'])).T
    values = []
    frame_results = []
    for i in range(2):
        lo, hi = dl[i].ravel(), bilinear_sample(dh[i], mapped[:, :2])
        valid = np.isfinite(lo) & (lo > 0) & np.isfinite(hi) & (hi > 0)
        values.append((lo[valid], hi[valid]))
        frame_results.append(dict(frame_id=low['record']['frame_ids'][i], raw_112=positive_stats(dl[i]),
                                  raw_518=positive_stats(dh[i]), matched_pixels=int(valid.sum()),
                                  excluded_from_comparison=int(len(lo) - valid.sum())))
    a, b = np.concatenate([v[0] for v in values]), np.concatenate([v[1] for v in values])
    if not len(a):
        return dict(status='no_common_positive_depth', frames=frame_results)
    scale = float(np.dot(a, b) / np.dot(a, a))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Cannot obtain a finite positive descriptive depth scale')
    return dict(status='descriptive_only', units='arbitrary_VGGT_batch_Z_units', frames=frame_results,
                sampling='518 float64 bilinear at the same rectified original pixel centres as the 112 grid',
                rectification_border_and_confidence_not_filtered=True,
                unaligned=residual_stats(a, b), positive_scale_112_to_518=scale,
                scale_scope='one scalar for both frames of this independent prediction batch',
                scale_rule='positive least squares through origin on all matched positive pixels',
                scale_fit_and_score_use_same_pixels=True, depth_GT_used=False,
                scale_aligned=residual_stats(a, b, scale))


def camera_stats(K_model, A, K_calibrated):
    K = np.linalg.inv(A) @ K_model
    if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError('Invalid predicted intrinsics in original pixel coordinates')
    return dict(K_original_rectified_pixels=K.tolist(),
                focal_relative_error_to_supplied_K=[float(K[0, 0] / K_calibrated[0, 0] - 1),
                                                    float(K[1, 1] / K_calibrated[1, 1] - 1)],
                principal_point_delta_to_supplied_K_pixels=(K[:2, 2] - K_calibrated[:2, 2]).tolist())


def analyze(run112, run518, groundtruth):
    low, lp, lh = load_capture(run112, 112)
    high, hp, hh = load_capture(run518, 518)
    verify_identity(low, high)
    gt_sha = geometry.sha(groundtruth)
    times, poses, gt_info = geometry.read_gt(groundtruth)
    K_calibrated = np.asarray(low['rectification']['output_K'], np.float64)
    result = dict(kind='three_fixed_pair_resolution_numerical_diagnostic', status='completed',
                  analyzer_sha256=geometry.sha(__file__), geometry_dependency_sha256=geometry.sha(geometry.__file__),
                  model_inference_executed=False, image_decode_executed=False, tracking_executed=False,
                  board_operations_executed=False, input_identity_verified=True, raw_inputs_modified=False,
                  full_dataset_ATE_RPE_validated=False, accuracy_gate_validated=False,
                  metric_scale_validated=False, real_time_validated=False,
                  provenance=dict(capture_112=lh, capture_518=hh, producer_sha256=CAPTURE_SCRIPT_SHA,
                    checkpoint_sha256=CHECKPOINT_SHA, rgb_index_sha256=low['rgb_index']['sha256'],
                    calibration_sha256=low['calibration']['sha256'], source_hashes=low['source_hashes'],
                    schedule_metrics_sha256=low['schedule']['metrics_sha256'],
                    schedule_events_sha256=low['schedule']['events_sha256']),
                  groundtruth=dict(sha256=gt_sha, **gt_info, used_after_capture_only=True,
                    pose_convention='T_world_from_camera; qx qy qz qw; R=I rectification preserves axes',
                    association='same nearest original GT sample for both resolutions',
                    maximum_time_difference_seconds=GT_TOLERANCE_SECONDS,
                    direction_frame='destination camera, relative T_second_from_first',
                    minimum_translation_for_direction_m=MIN_GT_TRANSLATION_METRES),
                  sample_selection=low['schedule']['selection'], pairs=[])
    for l, h in zip(lp, hp):
        ids = l['record']['frame_ids']
        indices = [geometry.nearest(times, t, GT_TOLERANCE_SECONDS) for t in ids]
        gt_relative = None if any(i is None for i in indices) else geometry.inverse(poses[indices[1]]) @ poses[indices[0]]
        matches = [dict(frame_id=t, GT_timestamp=None if i is None else float(times[i]),
                        difference_seconds=None if i is None else float(times[i] - t)) for t, i in zip(ids, indices)]
        pair = dict(revision=l['record']['revision'], frame_ids=ids, GT_matches=matches,
                    source_images=[dict(frame_id=image['frame_id'], basename=Path(image['path']).name,
                                        sha256=image['sha256']) for image in l['selected']['images']],
                    mapping_112=l['mapping'], mapping_518=h['mapping'],
                    relative_pose_convention='T_second_from_first=Tcw[1]@inverse(Tcw[0])',
                    resolution_rotation_difference_degrees=geometry.rotation_degrees(l['relative'][:3, :3] @ h['relative'][:3, :3].T),
                    resolution_translation_direction_difference_degrees=geometry.errors(l['relative'], h['relative'], 1e-12)['translation_direction_error_degrees'],
                    depth=compare_depth(l, h), predictions={})
        for label, row in (('112', l), ('518', h)):
            pair['predictions'][label] = dict(relative_Tcw=row['relative'].tolist(),
                relative_translation_norm_arbitrary_units=float(np.linalg.norm(row['relative'][:3, 3])),
                GT_pair_diagnostic=None if gt_relative is None else geometry.errors(row['relative'], gt_relative, MIN_GT_TRANSLATION_METRES),
                GT_unavailable_reason='no_original_GT_within_20ms' if gt_relative is None else None,
                model_inference_seconds=row['record']['model_inference_seconds'],
                K_original_pixels=[camera_stats(K, row['A'], K_calibrated) for K in row['arrays']['intrinsic'][0]])
        pair['K_112_minus_518_original_pixels'] = [(np.linalg.inv(l['A']) @ a - np.linalg.inv(h['A']) @ b).tolist()
            for a, b in zip(l['arrays']['intrinsic'][0], h['arrays']['intrinsic'][0])]
        result['pairs'].append(pair)
    for run, hashes in ((run112, lh), (run518, hh)):
        if any(geometry.sha(Path(run) / name) != digest for name, digest in hashes.items()):
            raise ValueError('Saved capture input changed during analysis')
    if geometry.sha(groundtruth) != gt_sha:
        raise ValueError('Ground truth changed during analysis')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-112', type=Path, required=True)
    parser.add_argument('--capture-518', type=Path, required=True)
    parser.add_argument('--groundtruth', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(argv)
    if a.output.exists():
        parser.error('Output must be a new JSON file')
    result = analyze(a.capture_112, a.capture_518, a.groundtruth)
    with a.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], pairs=len(result['pairs']), output=a.output.name,
                          sha256=geometry.sha(a.output))))
    return 0


if __name__ == '__main__':
    sys.exit(main())
