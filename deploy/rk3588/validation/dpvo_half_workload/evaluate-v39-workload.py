#!/usr/bin/env python3
"""Saved-only v39 CPU/full-resolution FAST trajectory comparison.

Uses byte-pinned v34 evaluation math derived from Princeton-VL DPVO (MIT,
see adjacent LICENSE). No image, neural network, ORT, native code or BA runs.
GT is read only after the caller supplies completed/frozen run receipts.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

MATH_SHA = '09b7a090a8dffa2401693d61baff9fbf73ddf368deceadf2f3631137b81d55d8'
POLICY_SHA = 'cc6a31f89bdcff0b43c0d51266ad62c48d0e297346845a98d42bad5c8874ef4a'
CPU_SCHEMA = 'v39_DPVO_half_original_Torch_CPU_research_replay'
BASELINE_SCHEMA = 'v34_DPVO_original_FAST_fresh_frame_CPU_replay'


def require(ok, why):
    if not bool(ok):
        raise ValueError(why)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_math(path):
    require(sha(path) == MATH_SHA, 'Frozen v34 mathematical helper SHA differs')
    spec = importlib.util.spec_from_file_location('_v39_frozen_trajectory_math', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def fresh_rows(run, source, math, *, half):
    rows = run['frames']
    require(len(rows) == run['frames_completed'] <= len(source), 'Processed row count differs')
    poses, process_times, publication_times = {}, [], []
    last_capture = last_reply = -np.inf
    for i, row in enumerate(rows):
        result, original = row['result'], source[i]
        require(row['index'] == result['observation_id'] == i and
                row['frame_id'] == result['frame_id'] == original['frame_id'] and
                row['timestamp'] == result['timestamp'] == original['timestamp'],
                'Original source identity differs')
        capture, reply = row['arrival_monotonic'], row['pose_available_monotonic']
        processed = row['processed_monotonic'] if half else reply
        require(np.isfinite([capture, processed, reply]).all() and
                capture > last_capture and reply > last_reply and
                capture <= processed <= reply and capture == result['captured_monotonic'],
                'Monotonic source/publication binding differs')
        require(abs(processed - capture - row['processing_seconds']) < 1e-8,
                'Processing clock identity differs')
        if half:
            require(abs(reply - capture - row['admission_to_publication_seconds']) < 1e-8
                    and abs(reply - processed - row['publication_seconds']) < 1e-8,
                    'Publication including serialization/flush clock identity differs')
        last_capture, last_reply = capture, reply
        require(type(result['fresh_pose_valid']) is bool, 'Fresh flag must be bool')
        if result['fresh_pose_valid']:
            require(result['admitted'] is True and result['initialized'] is True and
                    result['updates'] >= 1 and result['reason'] == 'updated_current_observation',
                    'Initialization/reconstructed/stale row is labeled fresh')
            poses[i] = math.matrices([result['pose_w2c']])[0]
            process_times.append(processed - capture)
            publication_times.append(reply - capture)
        else:
            require(result['pose_w2c'] is None, 'A non-fresh row published a pose')
    metrics = run.get('metrics_before_termination', run.get('stream_metrics_at_exit'))
    require(isinstance(metrics, dict) and metrics['observations'] == len(rows)
            and metrics['fresh_poses'] == len(poses), 'Fresh/lifecycle accounting differs')
    return poses, process_times, publication_times


def load_run(directory, digest, math, *, half, policy):
    root = Path(directory)
    require(sha(root / 'run.json') == digest, 'Run SHA differs')
    run = json.loads((root / 'run.json').read_text())
    require(run['schema'] == (CPU_SCHEMA if half else BASELINE_SCHEMA), 'Wrong run schema')
    require(sha(root / 'input-manifest.json') == run['input_manifest_sha256'],
            'Source manifest SHA differs')
    schedule = json.loads((root / 'input-manifest.json').read_text())
    source = schedule['frames']
    stamps = np.array([row['timestamp'] for row in source], np.float64)
    require(len(source) == run['frames_requested'] == 192 and np.isfinite(stamps).all()
            and np.all(np.diff(stamps) > 0) and schedule['GT_read'] is False,
            'Complete 192-source predeclared plan differs')
    require(run['GT_or_sensor_depth_input'] is False and run['ORB_used'] is False
            and run['added_LK_PnP_frontend'] is False, 'Unexpected inference prior/frontend')
    cfg = run['config']
    if half:
        require(run['policy_sha256'] == POLICY_SHA and run['policy'] == policy
                and cfg == policy['stream_config'], 'Frozen v39 policy/config binding differs')
    else:
        require(tuple(cfg[key] for key in ('patches_per_frame', 'patch_lifetime',
                'removal_window', 'optimization_window', 'keyframe_index',
                'keyframe_threshold')) == (48, 11, 16, 7, 4, 15.0),
                'Original full-resolution FAST baseline configuration differs')
    require(cfg['capacity'] == 64 and cfg['loop_closure'] is False
            and cfg['classic_loop_closure'] is False, 'Capacity/loop scope differs')
    poses, processing, publication = fresh_rows(run, source, math, half=half)
    return {'run': run, 'schedule': schedule, 'source': source, 'stamps': stamps,
            'fresh': poses, 'processing': processing, 'publication': publication}


def evaluate_eof(data, math, association, gt_c2w, plan):
    terminated = data['run'].get('termination')
    if terminated is None:
        return {'state': 'not_present', 'counted_as_fresh': False}
    require(terminated['fresh_pose_valid'] is False and terminated['online_publication'] is False
            and terminated['reconstructed_poses_counted_as_fresh'] is False,
            'EOF reconstruction is labeled fresh')
    observations = terminated['observations']
    require(len(observations) == data['run']['frames_completed'], 'EOF source count differs')
    for i, row in enumerate(observations):
        source = data['source'][i]
        require(row['observation_id'] == i and row['frame_id'] == source['frame_id']
                and row['timestamp'] == source['timestamp'], 'EOF source identity differs')
    out = {'state': terminated['trajectory_state'], 'counted_as_fresh': False,
           'offline_updates': terminated['offline_updates'],
           'relative_reconstruction_count': len(terminated['reconstructed_observation_ids'])}
    if terminated['initialized']:
        w2c, c2w = math.matrices(terminated['pose_w2c']), math.matrices(terminated['pose_c2w'])
        require(len(w2c) == len(observations) and
                np.allclose(math.inverse(w2c), c2w, rtol=1e-5, atol=1e-5),
                'EOF inverse pose binding differs')
        out['evaluation'] = math.evaluate_population(dict(enumerate(w2c)), association, gt_c2w, plan)
    return out


def ratios(reference, candidate):
    """Positive errors: lower is better. No division by zero or mismatched sets."""
    result = {}
    for metric in ('ATE_sim3_aligned_meters', 'RPE_rotation_degrees',
                   'RPE_translation_GT_scale_meters',
                   'RPE_translation_direction_degrees_nonzero_vectors'):
        a, b = reference[metric], candidate[metric]
        same_count = a['count'] == b['count'] and a['count'] > 0
        values = {}
        for statistic in ('rmse', 'mean', 'median', 'p95'):
            x, y = a[statistic], b[statistic]
            usable = same_count and x is not None and y is not None and x > 0
            values[statistic] = {'candidate_over_baseline': y/x if usable else None,
                                'error_change_percent': 100*(y/x-1) if usable else None,
                                'unavailable_reason': None if usable else
                                'missing_zero_baseline_or_different_nonzero_population'}
        result[metric] = {'baseline_count': a['count'], 'candidate_count': b['count'],
                          'statistics': values}
    return result


def compare(cpu_dir, baseline_dir, groundtruth, cpu_sha, baseline_sha, gt_sha,
            policy_path, math_path):
    require(sha(policy_path) == POLICY_SHA and sha(groundtruth) == gt_sha,
            'Policy/GT SHA differs')
    math = load_math(math_path)
    policy = json.loads(Path(policy_path).read_text())
    baseline = load_run(baseline_dir, baseline_sha, math, half=False, policy=policy)
    cpu = load_run(cpu_dir, cpu_sha, math, half=True, policy=policy)
    require(baseline['source'] == cpu['source'] and
            baseline['schedule']['rgb_index_sha256'] == cpu['schedule']['rgb_index_sha256']
            and baseline['schedule']['dataset'] == cpu['schedule']['dataset'],
            'Original images, ordered sensor timestamps or RGB index differ')
    # Construct endpoints on all source timestamps, before success/GT filtering.
    plan = math.endpoint_plan(cpu['stamps'])
    gt = np.loadtxt(groundtruth, comments='#')
    require(gt.ndim == 2 and gt.shape[1] == 8 and np.isfinite(gt).all(), 'Finite TUM GT required')
    association = math.nearest_association(cpu['stamps'], gt[:, 0])
    gt_c2w = math.matrices(gt[:, 1:])
    common = math.common_fresh_analysis(baseline['fresh'], cpu['fresh'], association,
             gt_c2w, plan, [row['frame_id'] for row in cpu['source']])
    # Neutral names; the original mathematical helper calls its arguments
    # default/FAST, which are not the experiment identities used here.
    common['full_resolution_FAST_CPU'] = common.pop('default')
    common['half_resolution_CPU'] = common.pop('FAST')
    populations = {}
    for label, data in [('full_resolution_FAST_CPU', baseline), ('half_resolution_CPU', cpu)]:
        run = data['run']
        duration = run.get('replay_elapsed_seconds')
        populations[label] = {'run_schema': run['schema'], 'run_status': run['status'],
            'delivered_observations': len(run['frames']), 'planned_observations': 192,
            'fresh_pose_source_coverage': len(data['fresh'])/192,
            'fresh_online': math.evaluate_population(data['fresh'], association, gt_c2w, plan),
            'EOF_reoptimized_and_reconstructed': evaluate_eof(data, math, association, gt_c2w, plan),
            'fresh_processing_seconds': math.summary(data['processing']),
            'fresh_software_admission_to_publication_seconds': math.summary(data['publication']),
            'publication_timing_boundary': 'Includes event JSON serialization and flush' if data is cpu
                 else 'Frozen v34 processing completion; publication I/O excluded',
            'replay_elapsed_seconds': duration,
            'fresh_over_unpaced_input_active_elapsed_Hz': len(data['fresh'])/duration
                 if duration is not None and np.isfinite(duration) and duration > 0 else None,
            'cleanup_errors': run['cleanup_errors'],
            'source_stability_reported': run.get('sources_and_inputs_unchanged'),
            'weight_stability_reported': run.get('original_weights_unchanged')}
    pins = {'cpu_run': cpu_sha, 'baseline_run': baseline_sha, 'GT': gt_sha,
            'policy': POLICY_SHA, 'math_helper': MATH_SHA,
            'cpu_manifest': cpu['run']['input_manifest_sha256'],
            'baseline_manifest': baseline['run']['input_manifest_sha256']}
    paths = {'cpu_run': Path(cpu_dir)/'run.json', 'baseline_run': Path(baseline_dir)/'run.json',
             'GT': Path(groundtruth), 'policy': Path(policy_path), 'math_helper': Path(math_path),
             'cpu_manifest': Path(cpu_dir)/'input-manifest.json',
             'baseline_manifest': Path(baseline_dir)/'input-manifest.json'}
    require(all(sha(paths[key]) == digest for key, digest in pins.items()),
            'Saved source changed during evaluation')
    return {'schema': 'v39_DPVO_predeclared_half_vs_full_FAST_saved_trajectory_comparison',
        'status': 'saved_numeric_quality_diagnostic_not_adoption_or_realtime_acceptance',
        'evaluator_sha256': sha(__file__), 'input_sha256': pins,
        'same_original_192_source_plan': True, 'fixed_all_source_1second_endpoint_plan': plan,
        'complete_populations': populations, 'common_fresh': common,
        'common_fresh_error_ratios': ratios(common['full_resolution_FAST_CPU'],
                                           common['half_resolution_CPU']),
        'limits': ['GT is only post-run evaluation; no runtime prior or GT-based configuration selection.',
            'One positive Sim3 per population for monocular gauge; no per-window fit.',
            'Coverage uses all192 planned sensor observations and endpoints; common fresh precedes GT/error filtering.',
            'EOF/reconstructed/initialization/skipped rows are never fresh observations.',
            'Short unpaced file replay is not physical capture, queue-age,20Hz,map,endurance or original-VGGT quality acceptance.',
            'The v34 and v39 software publication clocks have different I/O boundaries and separate thermal histories; no causal speedup ratio.',
            'Ratios can be unavailable for zero baseline error, degeneracy or different nonzero direction populations.'],
        'model_forward_calls': 0, 'ORT_calls': 0, 'SDK_calls': 0,
        'native_calls': 0, 'BA_calls': 0, 'images_read': 0, 'board_jobs': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cpu-run-dir', 'baseline-run-dir', 'groundtruth', 'policy', 'math-helper', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('cpu-run-sha256', 'baseline-run-sha256', 'groundtruth-sha256'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    result = compare(args.cpu_run_dir, args.baseline_run_dir, args.groundtruth,
            args.cpu_run_sha256, args.baseline_run_sha256, args.groundtruth_sha256,
            args.policy, args.math_helper)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(load_math(args.math_helper).plain(result), stream,
                  ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'],
                     'common_fresh_count': result['common_fresh']['common_fresh_count']}))


if __name__ == '__main__':
    main()
