#!/usr/bin/env python3
"""Read saved edge trajectories and TUM GT; never run images/models/tracking.

GT is used only after the experiment. Default diagnostics are local relative
rotation and translation direction, plus fixed-one-second RPE within a gauge.
Optional fitted translation scale is evaluation-only, not metric calibration.
TUM convention: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
TUM_FORMAT_URL = 'https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats'
RPE_SECONDS = 1.0


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number')
    return float(value)


def strict_json(text):
    def invalid(value):
        raise ValueError(f'Nonfinite JSON constant: {value}')
    return json.loads(text, parse_constant=invalid)


def rigid(value, label):
    T = np.asarray(value, dtype=np.float64)
    if (T.shape != (4, 4) or not np.isfinite(T).all()
            or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5, rtol=0)
            or not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-5, rtol=0)):
        raise ValueError(f'{label} must be a finite proper SE3 transform')
    return T


def inverse(T):
    result = np.eye(4)
    result[:3, :3] = T[:3, :3].T
    result[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return result


def quaternion_xyzw(q):
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError('Finite nonzero TUM quaternion required')
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def read_gt(path):
    times, poses, norms = [], [], []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        fields = [float(x) for x in line.split()]
        if len(fields) != 8 or not np.isfinite(fields).all():
            raise ValueError('TUM GT rows need timestamp tx ty tz qx qy qz qw')
        if times and fields[0] <= times[-1]:
            raise ValueError('GT timestamps must increase strictly')
        T = np.eye(4)
        T[:3, :3] = quaternion_xyzw(fields[4:])
        T[:3, 3] = fields[1:4]
        times.append(fields[0]); poses.append(T); norms.append(np.linalg.norm(fields[4:]))
    if len(times) < 2:
        raise ValueError('At least two GT poses are required')
    return np.array(times), np.array(poses), {'samples': len(times),
        'quaternions_normalized_on_read': True,
        'maximum_input_quaternion_norm_deviation': float(np.max(np.abs(np.array(norms)-1)))}


def nearest(times, timestamp, tolerance):
    position = int(np.searchsorted(times, timestamp))
    candidates = [i for i in (position-1, position) if 0 <= i < len(times)]
    if not candidates:
        return None
    index = min(candidates, key=lambda i: (abs(float(times[i])-timestamp), i))
    return index if abs(float(times[index])-timestamp) <= tolerance else None


def rotation_degrees(R):
    sine = np.linalg.norm([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]]) / 2
    cosine = np.clip((np.trace(R)-1)/2, -1, 1)
    return float(np.degrees(np.arctan2(sine, cosine)))


def errors(estimate, truth, minimum_translation):
    e, g = estimate[:3, 3], truth[:3, 3]
    en, gn = float(np.linalg.norm(e)), float(np.linalg.norm(g))
    reason = ('GT_translation_below_diagnostic_minimum' if gn < minimum_translation
              else 'estimated_translation_zero' if en < 1e-12 else None)
    direction = None if reason else float(np.degrees(np.arccos(np.clip(np.dot(e, g)/(en*gn), -1, 1))))
    return {'rotation_error_degrees': rotation_degrees(estimate[:3, :3] @ truth[:3, :3].T),
            'translation_direction_error_degrees': direction, 'translation_direction_unavailable_reason': reason,
            'GT_translation_norm_m': gn, 'estimated_translation_norm_arbitrary_units': en}


def distribution(values):
    a = np.array([x for x in values if x is not None], dtype=np.float64)
    if not len(a):
        return {'count': 0, 'mean': None, 'rms': None, 'p50': None, 'p95': None, 'max': None}
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite metric')
    return {'count': len(a), 'mean': float(a.mean()), 'rms': float(np.sqrt(np.mean(a*a))),
            'p50': float(np.percentile(a, 50)), 'p95': float(np.percentile(a, 95)), 'max': float(a.max())}


def summaries(rows):
    return {'samples': len(rows),
            'rotation_error_degrees': distribution([x['rotation_error_degrees'] for x in rows]),
            'translation_direction_error_degrees': distribution([x['translation_direction_error_degrees'] for x in rows]),
            'translation_direction_unavailable': dict(Counter(x['translation_direction_unavailable_reason'] for x in rows
                                                          if x['translation_direction_unavailable_reason']))}


def read_tracking(path):
    rows, last_time, last_sequence, anchors = [], -math.inf, -1, {}
    for text in Path(path).read_text().splitlines():
        row = strict_json(text)
        timestamp = finite(row['timestamp'], 'timestamp')
        if timestamp != finite(row['frame_id'], 'frame_id') or timestamp <= last_time:
            raise ValueError('Exact increasing dataset frame timestamps required')
        sequence = row['sequence']
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= last_sequence:
            raise ValueError('Tracking sequence must increase strictly')
        if row['historical_replay'] is not False or type(row['pose_valid']) is not bool:
            raise ValueError('Only fresh, explicit pose-valid tracking observations may be evaluated')
        if row['pose_valid']:
            gauge = row['gauge_id']
            if isinstance(gauge, bool) or not isinstance(gauge, int) or gauge < 0:
                raise ValueError('Valid local pose needs a nonnegative gauge ID')
            anchor = finite(row['anchor_frame_id'], 'anchor_frame_id')
            if anchor > timestamp or anchors.setdefault(gauge, anchor) != anchor:
                raise ValueError('Gauge cannot change anchor or use a future anchor')
            row['_T'] = rigid(row['T_current_from_anchor'], 'T_current_from_anchor')
        elif row['T_current_from_anchor'] is not None:
            raise ValueError('Invalid pose must be null')
        rows.append(row); last_time, last_sequence = timestamp, sequence
    if not rows:
        raise ValueError('Empty tracking log')
    return rows


def fixed_rpe_pairs(rows, tolerance):
    groups = defaultdict(list)
    for row in rows:
        if row['pose_valid']:
            groups[(row['gauge_id'], row['anchor_frame_id'])].append(row)
    pairs, counts = [], Counter()
    for group in groups.values():
        times = np.array([x['timestamp'] for x in group])
        for i, first in enumerate(group):
            target = first['timestamp'] + RPE_SECONDS
            if target > times[-1] + tolerance:
                counts['start_has_no_remaining_one_second_in_gauge'] += 1
                continue
            counts['eligible_start_rows'] += 1
            j = nearest(times, target, tolerance)
            if j is None or j <= i:
                counts['no_valid_target_within_time_tolerance'] += 1
                continue
            pairs.append((first, group[j]))
    counts['same_gauge_pairs'] = len(pairs)
    return pairs, dict(counts)


def evaluate_run(directory, gt_times, gt_poses, args):
    metrics_path, tracking_path, events_path = [directory / x for x in ('metrics.json', 'tracking.jsonl', 'events.jsonl')]
    pins = {str(p): sha(p) for p in (metrics_path, tracking_path, events_path)}
    metrics = strict_json(metrics_path.read_text())
    if metrics.get('status') != 'completed_experiment' or metrics.get('ORB_used') is not False:
        raise ValueError('Completed NO-ORB edge experiment required')
    rows = read_tracking(tracking_path)
    source = metrics.get('source', {})
    if source.get('delivered') != len(rows) or metrics.get('pose_timing', {}).get('fresh_valid_poses') != sum(x['pose_valid'] for x in rows):
        raise ValueError('Tracking coverage disagrees with experiment metrics')
    matches = {x['timestamp']: nearest(gt_times, x['timestamp'], args.max_time_difference) for x in rows}
    local, sample_data, groups = [], {}, defaultdict(list)
    unmatched = Counter()
    for row in rows:
        if not row['pose_valid']:
            continue
        current_idx = matches[row['timestamp']]
        anchor_idx = nearest(gt_times, row['anchor_frame_id'], args.max_time_difference)
        if current_idx is None or anchor_idx is None:
            unmatched['current_GT_missing' if current_idx is None else 'anchor_GT_missing'] += 1
            continue
        truth = inverse(gt_poses[current_idx]) @ gt_poses[anchor_idx]
        item = {'timestamp': row['timestamp'], 'anchor_timestamp': row['anchor_frame_id'],
                'gauge_id': row['gauge_id'], 'sequence': row['sequence'],
                'GT_current_timestamp': float(gt_times[current_idx]), 'GT_anchor_timestamp': float(gt_times[anchor_idx]),
                'GT_current_time_error_seconds': float(gt_times[current_idx]-row['timestamp']),
                'GT_anchor_time_error_seconds': float(gt_times[anchor_idx]-row['anchor_frame_id']),
                **errors(row['_T'], truth, args.minimum_translation)}
        local.append(item)
        sample_data[row['timestamp']] = (row, truth, item)
        groups[(row['gauge_id'], row['anchor_frame_id'])].append((row, truth, item))
    scale_by_group, group_reports = {}, []
    for key, values in groups.items():
        report = {'gauge_id': key[0], 'anchor_timestamp': key[1], **summaries([v[2] for v in values])}
        if args.fit_gauge_scale:
            e = np.array([v[0]['_T'][:3, 3] for v in values])
            g = np.array([v[1][:3, 3] for v in values])
            denominator = float(np.sum(e*e))
            scale = float(np.sum(e*g)/denominator) if denominator > 1e-20 else None
            if scale is not None and scale > 0:
                scale_by_group[key] = scale
                residual = np.linalg.norm(scale*e-g, axis=1)
                report['GT_fitted_scale_evaluation_only'] = {'positive_scale': scale, 'fit_samples': len(values),
                    'fit_and_score_use_same_samples': True, 'anchor_translation_residual_m': distribution(residual)}
            else:
                report['GT_fitted_scale_evaluation_only'] = {'positive_scale': None, 'unavailable_reason': 'nonpositive_or_unobservable_scale'}
        group_reports.append(report)
    pairs, pair_counts = fixed_rpe_pairs(rows, args.rpe_time_tolerance)
    rpe, rpe_data = [], {}
    for first, second in pairs:
        i, j = matches[first['timestamp']], matches[second['timestamp']]
        if i is None or j is None:
            pair_counts['pairs_missing_GT'] = pair_counts.get('pairs_missing_GT', 0) + 1
            continue
        estimate = second['_T'] @ inverse(first['_T'])
        truth = inverse(gt_poses[j]) @ gt_poses[i]
        item = {'first_timestamp': first['timestamp'], 'second_timestamp': second['timestamp'],
                'actual_delta_seconds': second['timestamp']-first['timestamp'], 'gauge_id': first['gauge_id'],
                'anchor_timestamp': first['anchor_frame_id'], 'first_sequence': first['sequence'], 'second_sequence': second['sequence'],
                **errors(estimate, truth, args.minimum_translation)}
        scale = scale_by_group.get((first['gauge_id'], first['anchor_frame_id']))
        if scale is not None:
            item['translation_residual_m_after_evaluation_only_GT_scale'] = float(np.linalg.norm(scale*estimate[:3, 3]-truth[:3, 3]))
        rpe.append(item); rpe_data[(first['timestamp'], second['timestamp'])] = item
    events = [strict_json(x) for x in events_path.read_text().splitlines()]
    batches = [tuple(finite(x['frame_id'], 'batch frame ID') for x in e['batch']) for e in events if e.get('kind') == 'map_submitted']
    predictions = []
    for path in sorted(directory.glob('predictions_*.npz')):
        pins[str(path)] = sha(path)
        with np.load(path, allow_pickle=False) as saved:
            ids, extrinsic, intrinsic = saved['frame_ids'], saved['extrinsic'], saved['intrinsic']
        if (ids.shape != (2,) or not np.isfinite(ids).all() or ids[0] >= ids[1]
                or extrinsic.shape != (2, 3, 4) or intrinsic.shape != (2, 3, 3) or not np.isfinite(intrinsic).all()
                or list(batches).count(tuple(ids.tolist())) != 1):
            raise ValueError('Saved two-frame prediction must match one exact actual submitted batch')
        transforms = []
        for E in extrinsic:
            T = np.eye(4); T[:3] = E; transforms.append(rigid(T, 'saved model world-to-camera extrinsic'))
        item = {'path': str(path), 'sha256': pins[str(path)], 'frame_ids': ids.tolist(),
                'model_K_fx_fy': [[float(K[0, 0]), float(K[1, 1])] for K in intrinsic],
                'GT_used_in_model_inference': False}
        i, j = [nearest(gt_times, float(t), args.max_time_difference) for t in ids]
        if i is None or j is None:
            item.update(status='unmatched_GT', diagnostic=None)
        else:
            item.update(status='evaluated', diagnostic=errors(transforms[1] @ inverse(transforms[0]),
                        inverse(gt_poses[j]) @ gt_poses[i], args.minimum_translation))
        predictions.append(item)
    valid = [x for x in rows if x['pose_valid']]
    original = source.get('original_frames')
    if isinstance(original, bool) or not isinstance(original, int) or original < len(rows):
        raise ValueError('Explicit original source frame denominator required')
    report = {'run_dir': str(directory), 'backend': metrics['mapper_config']['backend'],
        'input_sha256': pins, 'coverage': {'source_frames_from_metrics': original,
            'tracking_observations': len(rows), 'fresh_valid_local_poses': len(valid),
            'valid_fraction_of_source': len(valid)/original, 'valid_fraction_of_observed': len(valid)/len(rows),
            'observed_timestamps_matched_GT': sum(x is not None for x in matches.values()),
            'unique_GT_samples_for_observations': len({x for x in matches.values() if x is not None}),
            'local_pose_GT_evaluation_rows': len(local), 'unmatched_local_pose_reasons': dict(unmatched),
            'tracking_status_counts': dict(Counter(x['status'] for x in rows)),
            'first_valid_after_source_start_seconds_from_metrics': metrics['pose_timing']['first_valid_seconds_after_start']},
        'local_anchor_relative': {**summaries(local), 'per_gauge': group_reports, 'rows': local},
        'fixed_one_second_RPE': {**summaries(rpe), 'selection_counts': pair_counts,
            'translation_residual_m_after_GT_fitted_scale': distribution([x.get('translation_residual_m_after_evaluation_only_GT_scale') for x in rpe]),
            'rows': rpe}, 'saved_model_two_frame_relative': predictions,
        'source_and_inputs_unchanged': all(sha(path) == digest for path, digest in pins.items())}
    if not report['source_and_inputs_unchanged']:
        raise ValueError('Source logs or saved predictions changed during evaluation')
    return report, sample_data, rpe_data, {x['timestamp'] for x in rows}


def compare_runs(evaluations):
    output = []
    for i in range(len(evaluations)):
        for j in range(i+1, len(evaluations)):
            a, b = evaluations[i], evaluations[j]
            samples = sorted(set(a[1]) & set(b[1]))
            anchored = [t for t in samples if a[1][t][0]['anchor_frame_id'] == b[1][t][0]['anchor_frame_id']]
            pairs = sorted(set(a[2]) & set(b[2]))
            output.append({'run_a': a[0]['run_dir'], 'run_b': b[0]['run_dir'],
                'common_observed_exact_timestamps': len(a[3] & b[3]), 'common_valid_GT_evaluated_exact_timestamps': len(samples),
                'common_exact_current_and_anchor_timestamps': len(anchored),
                'common_exact_one_second_RPE_endpoint_pairs': len(pairs),
                'common_RPE_run_a': summaries([a[2][p] for p in pairs]),
                'common_RPE_run_b': summaries([b[2][p] for p in pairs]),
                'fair_full_run_accuracy_regression_validated': False,
                'limitations': ['Fresh coverage, anchor choice and local geometry differ between runs',
                    'Common RPE endpoints control evaluation timestamps, not identical tracking landmarks or model batches',
                    'No 518 baseline, repeated trials or full-run 10-20% accuracy-loss acceptance']})
    return output


def evaluate(args):
    if not 0 < args.max_time_difference <= .020 or not 0 < args.rpe_time_tolerance <= .020:
        raise ValueError('GT/RPE nearest time tolerances must be positive and <=20ms')
    if not math.isfinite(args.minimum_translation) or args.minimum_translation <= 0:
        raise ValueError('Positive finite diagnostic minimum translation required')
    gt_path = args.groundtruth.resolve(strict=True)
    gt_sha = sha(gt_path)
    times, poses, info = read_gt(gt_path)
    directories = [p.resolve(strict=True) for p in args.run_dir]
    if len(set(directories)) != len(directories):
        raise ValueError('Duplicate experiment directory')
    evaluations = [evaluate_run(p, times, poses, args) for p in directories]
    if sha(gt_path) != gt_sha:
        raise ValueError('GT changed during evaluation')
    return {'status': 'completed_diagnostic', 'script_sha256': sha(__file__), 'evaluation_only': True,
        'GT_used_in_model_or_tracking': False, 'model_inference_executed': False, 'tracking_executed': False,
        'images_read': False, 'metric_pose_validated': False, 'accuracy_acceptance_validated': False,
        'real_time_validated': False, 'groundtruth': {'path': str(gt_path), 'sha256': gt_sha, **info},
        'conventions': {'GT': 'T_world_from_camera optical center, quaternion qx qy qz qw',
            'local_estimate': 'T_current_from_anchor, rigid local PnP',
            'GT_relative': 'inverse(T_world_current_GT) @ T_world_anchor_GT',
            'RPE_estimate': 'T_second_from_anchor @ inverse(T_first_from_anchor)',
            'translation_direction': 'Compared in destination camera coordinates; positive scale invariant',
            'format_primary_source': TUM_FORMAT_URL},
        'parameters': {'nearest_GT_seconds': args.max_time_difference, 'timestamp_offset_seconds': 0,
            'GT_interpolation': False, 'RPE_delta_seconds': RPE_SECONDS, 'RPE_target_tolerance_seconds': args.rpe_time_tolerance,
            'minimum_GT_translation_for_direction_m': args.minimum_translation,
            'fit_positive_per_gauge_scale_evaluation_only': args.fit_gauge_scale},
        'limitations': ['Missing/lost observations remain in source coverage, never filled with old poses',
            'RPE pairs never cross local gauge/anchor boundaries',
            'Optional GT scale is fitted and scored on this evaluation data, never a frontend prior',
            'Global SL4/projective centers are not evaluated as metric camera poses',
            'These saved CPU/XNN experiments do not establish NPU coverage, 518 accuracy retention, endurance or realtime acceptance'],
        'runs': [x[0] for x in evaluations], 'cross_run_common_samples': compare_runs(evaluations),
        'groundtruth_unchanged': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, action='append', required=True)
    parser.add_argument('--groundtruth', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New JSON diagnostic file')
    parser.add_argument('--max-time-difference', type=float, default=.020)
    parser.add_argument('--rpe-time-tolerance', type=float, default=.020)
    parser.add_argument('--minimum-translation', type=float, default=.01)
    parser.add_argument('--fit-gauge-scale', action='store_true')
    args = parser.parse_args()
    try:
        report = evaluate(args)
    except Exception as exc:
        report = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}', 'evaluation_only': True,
                  'model_inference_executed': False, 'tracking_executed': False}
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'error': report.get('error'),
                      'output': str(args.output), 'sha256': sha(args.output)}))
    return 0 if report['status'] == 'completed_diagnostic' else 1


if __name__ == '__main__':
    sys.exit(main())
