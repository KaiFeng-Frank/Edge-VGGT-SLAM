#!/usr/bin/env python3
"""JSON/GT-only v41 online Update trajectory diagnostic.

Uses frozen v39 population checks and v34 trajectory math without re-labeling
run schemas. No image, model, ORT, SDK, native library, BA or board execution.
Derived DPVO evaluation conventions: Princeton-VL, MIT; see adjacent LICENSE.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

V39_SHA = '56905a1c1906c7730b841e58f22b0460140875fc2bc616464452e907360baf4f'
MATH_SHA = '09b7a090a8dffa2401693d61baff9fbf73ddf368deceadf2f3631137b81d55d8'
POLICY_SHA = 'cc6a31f89bdcff0b43c0d51266ad62c48d0e297346845a98d42bad5c8874ef4a'
SCHEMAS = {
    'online_RKNN': 'v41_cached_undistort_online_RKNN_research_replay',
    'cached_half_CPU': 'v41_cached_undistort_original_Torch_CPU_research_replay',
    'full_resolution_FAST_CPU': 'v34_DPVO_original_FAST_fresh_frame_CPU_replay',
}


def require(condition, reason):
    if not bool(condition):
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen(v39_path, math_path):
    require(sha(v39_path) == V39_SHA and sha(math_path) == MATH_SHA,
            'Frozen evaluation source SHA differs')
    spec = importlib.util.spec_from_file_location('_v41_frozen_v39_evaluation', v39_path)
    v39 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v39)
    return v39, v39.load_math(math_path)


def verify_failed_eof_only(run):
    """Narrow opt-in: the known D3 failure AFTER all192 source publications.

    This is JSON consistency evidence, not a successful EOF or SDK audit.
    Counts intentionally bind this complete169-Update experiment, rather than
    accepting arbitrary failed/partial runs under a general relaxed loader.
    """
    require(run['status'] == 'failed' and run['active_stage'] == 'termination_offline'
            and run['frames_requested'] == run['frames_completed'] == len(run['frames']) == 192
            and run['cleanup_errors'] == [] and run.get('termination') is None,
            'Failed-EOF opt-in requires complete192 publications and only offline failure')
    before, after = run['metrics_before_termination'], run['stream_metrics_at_exit']
    require(before['status'] == 'running' and before['failure'] is None
            and before['observations'] == 192 and before['update_calls'] == before['BA_calls'] == 169
            and before['BA_iterations'] == 338 and before['offline_updates'] == 0
            and sum(row['result']['updates'] for row in run['frames']) == 169,
            'Complete online Update/BA counters differ before EOF')
    require(after['status'] == 'failed' and after['failure']['phase'] == 'offline_termination'
            and after['failure']['type'] == 'OnlineUpdateError'
            and after['update_calls'] == 170 and after['BA_calls'] == 169
            and after['BA_iterations'] == 338 and after['offline_updates'] == 0
            and all(after[key] == before[key] for key in
                    ('observations', 'fresh_poses', 'admitted', 'motion_skipped', 'motion_probe_calls')),
            'Failure is not isolated to the first offline Update')
    final, preclose = run['online_metrics'], run['online_metrics_before_close']
    require(final['requests'] == 204 and final['full_attempts'] == 170
            and final['full_completed'] == 169 and final['failed_calls'] == 1
            and final['motion_probe_attempts'] == final['motion_probe_completed'] == 34
            and final['failed'] is True and final['closed'] is True,
            'Online callback completion/failure counters differ')
    require(all(final[key] == preclose[key] for key in
                ('requests', 'full_attempts', 'full_completed', 'failed_calls',
                 'motion_probe_attempts', 'motion_probe_completed', 'calls')),
            'Callback history changed during close')
    calls = final['calls']
    require(len(calls) == 204 and [row['index'] for row in calls] == list(range(204))
            and all(row['status'] == 'completed_diagnostic_not_adopted' for row in calls[:-1])
            and sum(row['route'] == 'full' for row in calls[:-1]) == 169
            and sum(row['route'] == 'motion_probe' for row in calls[:-1]) == 34,
            'A prior online callback failed or is missing')
    last = calls[-1]
    expected_error = "ValueError('Unknown complete online shape(s): B1_D3_M16')"
    require(last['index'] == 203 and last['route'] == 'full'
            and last['status'] == 'failed_no_partial_fullgraph'
            and last['error'] == after['failure']['reason'] == expected_error
            and last['fullgraph_returned'] is False
            and last['outputs_committed_by_callback'] is False,
            'Last callback is not the known unknown-shape fail-closed EOF event')
    keys = [f"B{row['sources']}_D{row['degree']}_M{row['patches']}"
            for row in last['degree_plan']['batches']]
    require(keys == last['model_keys'] and sorted(set(keys) - set(final['model_keys']))
            == ['B1_D3_M16'], 'Reported last topology does not identify the sole D3 catalogue gap')
    cache = final['cache']
    require(cache['closed'] is True and cache['cleanup_errors'] == []
            and all(cache[key] == 0 for key in
                    ('resident_contexts', 'current_model_host_bytes', 'current_DMA_bytes')),
            'Failed-EOF run did not close its contexts/buffers')
    require(run['candidate_accepted'] is False and run['historical_SDK_strict_FP32_rejected'] is True
            and run['SDK_FP32_equivalence_validated'] is False
            and run['FP32_thresholds_relaxed'] is False
            and run['complete_Update_Torch_fallback'] is False,
            'Failure/rejection must not be converted to acceptance or fallback')
    return {'state': 'failed_not_completed', 'counted_as_fresh': False,
        'overall_run_status': 'failed', 'overall_run_success': False,
        'failure_phase': 'termination_offline', 'unknown_shape': 'B1_D3_M16',
        'last_callback_index': 203, 'full_online_updates_completed': 169,
        'offline_updates_completed': 0, 'EOF_trajectory_evaluation_available': False,
        'partial_fullgraph_returned': False, 'callback_error': last['error'],
        'failure_artifact_reported': last.get('failure_artifact'),
        'failure_artifact_bytes_independently_read': False,
        'cleanup_consistent_in_JSON': True}


def load_run(directory, digest, label, policy, v39, math, *, allow_failed_eof_only=False):
    """Read each actual schema directly; never use the CPU-only v39 loader."""
    directory = Path(directory)
    require(sha(directory / 'run.json') == digest, label + ': run SHA differs')
    run = json.loads((directory / 'run.json').read_text())
    require(run['schema'] == SCHEMAS[label], label + ': actual run schema differs')
    eof_failure = None
    if not run['status'].startswith('completed_'):
        require(allow_failed_eof_only and label == 'online_RKNN',
                label + ': completed clean run required; failed EOF requires explicit opt-in')
        eof_failure = verify_failed_eof_only(run)
    require(run['cleanup_errors'] == [], label + ': clean run required')
    require(sha(directory / 'input-manifest.json') == run['input_manifest_sha256'],
            label + ': input manifest SHA differs')
    schedule = json.loads((directory / 'input-manifest.json').read_text())
    source = schedule['frames']
    stamps = np.asarray([row['timestamp'] for row in source], np.float64)
    require(len(source) == run['frames_requested'] == run['frames_completed'] == 192
            and len(run['frames']) == 192 and np.isfinite(stamps).all()
            and np.all(np.diff(stamps) > 0) and schedule['GT_read'] is False,
            label + ': complete predeclared 192-source replay required')
    require(run['GT_or_sensor_depth_input'] is False and run['ORB_used'] is False
            and run['added_LK_PnP_frontend'] is False
            and run['original_VGGT_default_unchanged'] is True,
            label + ': unexpected inference prior/frontend/default change')
    require(run['sources_and_inputs_unchanged'] is True
            and run['original_weights_unchanged'] is True,
            label + ': source or weight stability report differs')
    cfg = run['config']
    half = label != 'full_resolution_FAST_CPU'
    if half:
        require(run['policy_sha256'] == POLICY_SHA and run['policy'] == policy
                and cfg == policy['stream_config'], label + ': frozen half policy differs')
    else:
        require(tuple(cfg[key] for key in ('patches_per_frame', 'patch_lifetime',
                'removal_window', 'optimization_window', 'keyframe_index',
                'keyframe_threshold')) == (48, 11, 16, 7, 4, 15.0),
                'Full-resolution original FAST configuration differs')
    require(cfg['capacity'] == 64 and cfg['loop_closure'] is False
            and cfg['classic_loop_closure'] is False, label + ': capacity/loop scope differs')
    require(type(run['NPU_used']) is bool and (half or run['NPU_used'] is False),
            label + ': backend flag differs')
    if label == 'cached_half_CPU':
        require(run['NPU_used'] is False, 'Cached CPU baseline claims NPU execution')
    fresh, processing, publication = v39.fresh_rows(run, source, math, half=half)
    if label == 'online_RKNN' and fresh:
        require(run['NPU_used'] is True, 'Fresh online RKNN run reports no NPU execution')
    duration, rss = run['replay_elapsed_seconds'], run['peak_rss_mib']
    require(np.isfinite([duration, rss]).all() and duration > 0 and rss >= 0,
            label + ': elapsed/RSS metadata differs')
    return {'run': run, 'source': source, 'schedule': schedule, 'stamps': stamps,
            'eof_failure': eof_failure,
            'fresh': fresh, 'processing': processing, 'publication': publication}


def population(data, label, v39, math, association, gt_c2w, plan):
    run = data['run']
    half = label != 'full_resolution_FAST_CPU'
    first = min(data['fresh']) if data['fresh'] else None
    all_stages = {}
    for key in ('decode_undistort_crop_resize_seconds', 'decode_undistort_crop_seconds',
                'encode_seconds', 'stream_seconds', 'processing_seconds',
                'admission_to_publication_seconds', 'publication_seconds'):
        if any(key in row for row in run['frames']):
            require(all(key in row and np.isfinite(row[key]) and row[key] >= 0
                        for row in run['frames']), label + ': stage clock differs')
            all_stages[key] = math.summary([row[key] for row in run['frames']])
    return {
        'run_schema': run['schema'], 'run_status': run['status'],
        'planned_observations': 192, 'delivered_observations': len(run['frames']),
        'fresh_pose_source_coverage': len(data['fresh']) / 192,
        'fresh_online': math.evaluate_population(data['fresh'], association, gt_c2w, plan),
        'EOF_reoptimized_and_reconstructed': data['eof_failure']
            if data['eof_failure'] is not None else v39.evaluate_eof(data, math, association, gt_c2w, plan),
        'overall_run_success': data['eof_failure'] is None,
        'fresh_processing_seconds': math.summary(data['processing']),
        'fresh_software_admission_to_publication_seconds': math.summary(data['publication']),
        'publication_timing_boundary': 'Includes event JSON serialization and flush' if half
            else 'Frozen v34 processing completion; publication I/O excluded',
        'first_fresh_source_index': first,
        'first_fresh_sensor_timestamp': data['source'][first]['timestamp'] if first is not None else None,
        'first_fresh_after_replay_begin_seconds': run['frames'][first]['pose_available_monotonic']
            - run['replay_begin_monotonic'] if first is not None else None,
        'all_observation_stage_seconds': all_stages,
        'replay_elapsed_seconds': run['replay_elapsed_seconds'],
        'fresh_over_unpaced_input_active_elapsed_Hz': len(data['fresh']) / run['replay_elapsed_seconds'],
        'peak_process_rss_mib_reported': run['peak_rss_mib'],
        'timing_scope_reported': run.get('timing_scope'),
        'backend_NPU_used_reported': run['NPU_used'],
        'candidate_accepted_reported': run.get('candidate_accepted'),
        'realtime_validated_reported': run.get('realtime_validated'),
        'lifecycle_before_EOF_reported': run.get('metrics_before_termination', run.get('stream_metrics_at_exit')),
        'cleanup_errors': run['cleanup_errors'],
        'source_stability_reported': run['sources_and_inputs_unchanged'],
        'weight_stability_reported': run['original_weights_unchanged'],
    }


def verify_same_source_plan(data):
    candidate = data['online_RKNN']
    require(all(value['source'] == candidate['source']
                and value['schedule']['dataset'] == candidate['schedule']['dataset']
                and value['schedule']['rgb_index_sha256'] == candidate['schedule']['rgb_index_sha256']
                for value in data.values()),
            'Original image pins, complete sensor plan or RGB index differs')


def compare(npu_dir, cached_cpu_dir, baseline_dir, groundtruth, *, npu_sha256,
            cached_cpu_sha256, baseline_sha256, groundtruth_sha256,
            policy_path, v39_path, math_path, allow_failed_eof_only=False):
    v39, math = load_frozen(v39_path, math_path)
    require(sha(policy_path) == POLICY_SHA, 'Frozen workload policy SHA differs')
    policy = json.loads(Path(policy_path).read_text())
    directories = dict(online_RKNN=Path(npu_dir), cached_half_CPU=Path(cached_cpu_dir),
                       full_resolution_FAST_CPU=Path(baseline_dir))
    digests = dict(online_RKNN=npu_sha256, cached_half_CPU=cached_cpu_sha256,
                   full_resolution_FAST_CPU=baseline_sha256)
    data = {label: load_run(directory, digests[label], label, policy, v39, math,
                           allow_failed_eof_only=allow_failed_eof_only)
            for label, directory in directories.items()}
    candidate = data['online_RKNN']
    # Frozen v39 compares content/identity, not the v30/v39 manifest schema or
    # descriptive selection string. Both actual run schemas remain untouched.
    verify_same_source_plan(data)
    require(candidate['run']['seed'] == data['cached_half_CPU']['run']['seed']
            and candidate['run']['random_generator_before']
                == data['cached_half_CPU']['run']['random_generator_before'],
            'Half CPU/RKNN initial RNG binding differs')
    require(all(value['run']['original_weights_before']
                == candidate['run']['original_weights_before'] for value in data.values()),
            'Reported original DPVO weight manifests differ')
    # The plan and intersections precede GT loading or looking at errors.
    plan = math.endpoint_plan(candidate['stamps'])
    common_ids = {label: sorted(set(value['fresh']) & set(candidate['fresh']))
                  for label, value in data.items() if label != 'online_RKNN'}
    require(sha(groundtruth) == groundtruth_sha256, 'GT SHA differs')
    gt = np.loadtxt(groundtruth, comments='#')
    require(gt.ndim == 2 and gt.shape[1] == 8 and np.isfinite(gt).all(),
            'Finite TUM timestamp+xyz+xyzw GT required')
    association = math.nearest_association(candidate['stamps'], gt[:, 0])
    gt_c2w = math.matrices(gt[:, 1:])
    populations = {label: population(value, label, v39, math, association, gt_c2w, plan)
                   for label, value in data.items()}
    common, error_ratios = {}, {}
    for reference in common_ids:
        row = math.common_fresh_analysis(data[reference]['fresh'], candidate['fresh'],
                association, gt_c2w, plan, [r['frame_id'] for r in candidate['source']])
        require(row['common_fresh_source_indices'] == common_ids[reference],
                'Pre-GT common population differs')
        row[reference], row['online_RKNN'] = row.pop('default'), row.pop('FAST')
        key = 'online_RKNN_vs_' + reference
        common[key] = row
        error_ratios[key] = v39.ratios(row[reference], row['online_RKNN'])
    paths = {'v39_evaluator': Path(v39_path), 'v34_math': Path(math_path),
             'policy': Path(policy_path), 'GT': Path(groundtruth)}
    pins = {'v39_evaluator': V39_SHA, 'v34_math': MATH_SHA, 'policy': POLICY_SHA,
            'GT': groundtruth_sha256}
    for label, directory in directories.items():
        paths[label + '_run'] = directory / 'run.json'
        paths[label + '_manifest'] = directory / 'input-manifest.json'
        pins[label + '_run'] = digests[label]
        pins[label + '_manifest'] = data[label]['run']['input_manifest_sha256']
    require(all(sha(paths[key]) == digest for key, digest in pins.items()),
            'Saved inputs/source changed during evaluation')
    return {'schema': 'v41_DPVO_online_RKNN_saved_trajectory_comparison',
        'status': 'saved_fresh_trajectory_diagnostic_from_failed_EOF_run_not_run_success'
            if candidate['eof_failure'] is not None else
            'saved_numeric_quality_diagnostic_not_adoption_or_realtime_acceptance',
        'allow_failed_EOF_only_requested': bool(allow_failed_eof_only),
        'failed_EOF_only_exception_used': candidate['eof_failure'] is not None,
        'original_NPU_run_status': candidate['run']['status'],
        'overall_NPU_run_success': candidate['eof_failure'] is None,
        'evaluator_sha256': sha(__file__), 'input_sha256': pins,
        'same_original_192_source_plan': True,
        'fixed_all_source_1second_endpoint_plan': plan,
        'GT_association': {'policy': 'nearest timestamp <=20ms, one-to-one in original source order',
                           'associated_source_count': len(association)},
        'complete_populations': populations, 'common_fresh': common,
        'common_fresh_error_ratios': error_ratios,
        'limits': [
            'Default requires completed runs. Explicit failed-EOF-only mode admits only the complete192 online record with the checked first-offline D3 failure; overall failure remains failure.',
            'Only frozen JSON and GT numbers are read. SDK, image pixels and weight bytes are not independently audited here.',
            'GT is post-run evaluation only, never an inference prior or workload/model selection input.',
            'One positive Sim3 per run per declared full/common/EOF population; reused for ATE and RPE translation scale, never per window.',
            'Both common-fresh populations are selected before GT/error filtering; their denominators can differ between comparisons.',
            'The 1-second endpoint plan uses all192 sensor timestamps. Missing/failed/initializing/motion-skipped poses reduce coverage.',
            'EOF reoptimization/reconstruction uses a separate population and cannot count as fresh output.',
            'Translation-direction ratios retain the frozen evaluator policy: nonzero subsets may differ even on common endpoints.',
            'Unpaced file processing is not physical capture, queued-frame age,20Hz,30-minute stability or map-quality validation.',
            'Separate process/thermal histories and differing v34 publication-I/O boundaries do not establish a causal speedup.',
            'These DPVO research paths do not establish original VGGT518 quality, default-model replacement or acceptance.',
            'Historical FP32 SDK and half-workload trajectory rejection labels are not changed by this evaluation.'],
        'model_forward_calls': 0, 'ORT_calls': 0, 'SDK_calls': 0,
        'native_calls': 0, 'BA_calls': 0, 'images_read': 0, 'board_jobs': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('npu-run-dir', 'cached-cpu-run-dir', 'baseline-run-dir', 'groundtruth',
                 'policy', 'v39-evaluator', 'math-helper', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('npu-run-sha256', 'cached-cpu-run-sha256', 'baseline-run-sha256', 'groundtruth-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--allow-failed-EOF-only', action='store_true',
                        help='Evaluate completed192 fresh rows only for the narrowly checked failed first EOF Update; never mark the run successful')
    args = parser.parse_args()
    result = compare(args.npu_run_dir, args.cached_cpu_run_dir, args.baseline_run_dir,
            args.groundtruth, npu_sha256=args.npu_run_sha256,
            cached_cpu_sha256=args.cached_cpu_run_sha256, baseline_sha256=args.baseline_run_sha256,
            groundtruth_sha256=args.groundtruth_sha256, policy_path=args.policy,
            v39_path=args.v39_evaluator, math_path=args.math_helper,
            allow_failed_eof_only=args.allow_failed_EOF_only)
    _, math = load_frozen(args.v39_evaluator, args.math_helper)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(math.plain(result), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'common_fresh_counts': {
        label: row['common_fresh_count'] for label, row in result['common_fresh'].items()}}))


if __name__ == '__main__':
    main()
