#!/usr/bin/env python3
"""JSON/GT-only completed v45 native-input closed-flag online Update trajectory diagnostic.

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
PARENT_EVALUATOR_SHA = '75092ee7bd976a2ea5e8ec94ea6b669c5ad1e7c33cb5417b128aac49c71e66e9'
POLICY_SHA = 'cc6a31f89bdcff0b43c0d51266ad62c48d0e297346845a98d42bad5c8874ef4a'
SCHEMAS = {
    'online_RKNN': 'v45_cached_undistort_native_FP16_closed_flag_online_RKNN_research_replay',
    'previous_online_RKNN': 'v44_cached_undistort_native_FP16_online_RKNN_research_replay',
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


def verify_completed_EOF(run):
    """Check completed lifecycle metadata; EOF outputs remain a separate cohort."""
    require(run['schema'] in (SCHEMAS['online_RKNN'], SCHEMAS['previous_online_RKNN'])
        and run['status'] == 'completed_unpaced_half_online_RKNN_research_not_realtime'
        and run['frames_requested'] == run['frames_completed'] == len(run['frames']) == 192
        and run['cleanup_errors'] == [], 'Completed initialized online replay required')
    before, after, term = run['metrics_before_termination'], run['stream_metrics'], run['termination']
    require(before['status'] == 'running' and before['failure'] is None and before['offline_updates'] == 0
        and after['status'] == 'terminated' and after['failure'] is None and after['offline_updates'] == 12
        and after['update_calls'] == before['update_calls'] + 12
        and after['BA_calls'] == before['BA_calls'] + 12
        and after['BA_iterations'] == before['BA_iterations'] + 24
        and all(before[key] == after[key] for key in ('observations', 'admitted', 'motion_skipped', 'fresh_poses', 'motion_probe_calls')),
        'EOF must complete twelve offline updates without new observations or fresh poses')
    require(term['offline_updates'] == 12 and term['fresh_pose_valid'] is False
        and term['online_publication'] is False and term['reconstructed_poses_counted_as_fresh'] is False
        and term['initialized'] is True and term['trajectory_state'] == 'initialized_offline_reconstruction',
        'EOF reconstruction must not count as fresh publication')
    state, cache = run['online_metrics'], run['online_metrics']['cache']
    require(state['closed'] is True and state['failed'] is False and state['failed_calls'] == 0
        and state['full_attempts'] == state['full_completed'] == after['update_calls']
        and state['motion_probe_attempts'] == state['motion_probe_completed'] == after['motion_probe_calls']
        and state['requests'] == state['full_completed'] + state['motion_probe_completed']
        and len(state['calls']) == state['requests']
        and [row['index'] for row in state['calls']] == list(range(state['requests']))
        and all(row['status'] == 'completed_diagnostic_not_adopted' for row in state['calls'])
        and cache['closed'] is True and cache['cleanup_errors'] == []
        and all(cache[key] == 0 for key in ('resident_contexts', 'current_DMA_bytes', 'current_model_host_bytes')),
        'Completed callback counts/context cleanup differ')
    require(run['candidate_accepted'] is False and run['historical_SDK_strict_FP32_rejected'] is True
        and run['SDK_FP32_equivalence_validated'] is False and run['FP32_thresholds_relaxed'] is False
        and run['complete_Update_Torch_fallback'] is False,
        'Completion cannot change numerical rejection or adoption')
    return dict(metadata_consistency_verified=True, offline_updates_completed=12,
        offline_BA_steps=24, EOF_counted_as_fresh=False, full_Update_calls=state['full_completed'],
        explicit_Torch_motion_probe_calls=state['motion_probe_completed'], callback_failures=0,
        cache_closed=True, resident_contexts=0, current_DMA_bytes=0)


def load_run(directory, digest, label, policy, v39, math):
    """Read each actual schema directly; never use the CPU-only v39 loader."""
    directory = Path(directory)
    require(sha(directory / 'run.json') == digest, label + ': run SHA differs')
    run = json.loads((directory / 'run.json').read_text())
    require(run['schema'] == SCHEMAS[label], label + ': actual run schema differs')
    require(run['status'].startswith('completed_'), label + ': completed clean run required; no failed-run exception')
    if label in ('online_RKNN', 'previous_online_RKNN'):
        verify_completed_EOF(run)
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
    if label in ('online_RKNN', 'previous_online_RKNN') and fresh:
        require(run['NPU_used'] is True, 'Fresh online RKNN run reports no NPU execution')
    duration, rss = run['replay_elapsed_seconds'], run['peak_rss_mib']
    require(np.isfinite([duration, rss]).all() and duration > 0 and rss >= 0,
            label + ': elapsed/RSS metadata differs')
    return {'run': run, 'source': source, 'schedule': schedule, 'stamps': stamps,
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
        'EOF_reoptimized_and_reconstructed': v39.evaluate_eof(data, math, association, gt_c2w, plan),
        'overall_run_success': True,
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
            policy_path, v39_path, math_path, previous_online_dir, previous_online_sha256, v44_evaluator_path):
    require(sha(v44_evaluator_path) == PARENT_EVALUATOR_SHA, 'Frozen v44 evaluator source differs')
    v39, math = load_frozen(v39_path, math_path)
    require(sha(policy_path) == POLICY_SHA, 'Frozen workload policy SHA differs')
    policy = json.loads(Path(policy_path).read_text())
    directories = dict(online_RKNN=Path(npu_dir), cached_half_CPU=Path(cached_cpu_dir),
                       full_resolution_FAST_CPU=Path(baseline_dir), previous_online_RKNN=Path(previous_online_dir))
    digests = dict(online_RKNN=npu_sha256, cached_half_CPU=cached_cpu_sha256,
                   full_resolution_FAST_CPU=baseline_sha256, previous_online_RKNN=previous_online_sha256)
    data = {label: load_run(directory, digests[label], label, policy, v39, math)
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
    paths = {'v44_evaluator': Path(v44_evaluator_path), 'v39_evaluator': Path(v39_path), 'v34_math': Path(math_path),
             'policy': Path(policy_path), 'GT': Path(groundtruth)}
    pins = {'v44_evaluator': PARENT_EVALUATOR_SHA, 'v39_evaluator': V39_SHA, 'v34_math': MATH_SHA, 'policy': POLICY_SHA,
            'GT': groundtruth_sha256}
    for label, directory in directories.items():
        paths[label + '_run'] = directory / 'run.json'
        paths[label + '_manifest'] = directory / 'input-manifest.json'
        pins[label + '_run'] = digests[label]
        pins[label + '_manifest'] = data[label]['run']['input_manifest_sha256']
    require(all(sha(paths[key]) == digest for key, digest in pins.items()),
            'Saved inputs/source changed during evaluation')
    return {'schema': 'v45_DPVO_closed_flag_online_RKNN_completed_saved_trajectory_comparison',
        'status': 'saved_numeric_quality_diagnostic_not_adoption_or_realtime_acceptance',
        'parent_v44_evaluator_sha256': PARENT_EVALUATOR_SHA,
        'original_NPU_run_status': candidate['run']['status'],
        'overall_NPU_run_success': True,
        'completed_EOF_lifecycle_JSON_checks': verify_completed_EOF(candidate['run']),
        'evaluator_sha256': sha(__file__), 'input_sha256': pins,
        'same_original_192_source_plan': True,
        'fixed_all_source_1second_endpoint_plan': plan,
        'GT_association': {'policy': 'nearest timestamp <=20ms, one-to-one in original source order',
                           'associated_source_count': len(association)},
        'complete_populations': populations, 'common_fresh': common,
        'common_fresh_error_ratios': error_ratios,
        'limits': [
            'All four runs must be completed and clean. No failed-run exception or schema relabeling is provided.',
            'Only frozen JSON and GT numbers are read. SDK, image pixels and weight bytes are not independently audited here.',
            'GT is post-run evaluation only, never an inference prior or workload/model selection input.',
            'One positive Sim3 per run per declared full/common/EOF population; reused for ATE and RPE translation scale, never per window.',
            'All three common-fresh populations are selected before GT/error filtering; their denominators can differ between comparisons.',
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
                 'policy', 'v39-evaluator', 'math-helper', 'v44-evaluator', 'previous-online-run-dir', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('npu-run-sha256', 'cached-cpu-run-sha256', 'baseline-run-sha256', 'previous-online-run-sha256', 'groundtruth-sha256'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    result = compare(args.npu_run_dir, args.cached_cpu_run_dir, args.baseline_run_dir,
            args.groundtruth, npu_sha256=args.npu_run_sha256,
            cached_cpu_sha256=args.cached_cpu_run_sha256, baseline_sha256=args.baseline_run_sha256,
            groundtruth_sha256=args.groundtruth_sha256, policy_path=args.policy,
            v39_path=args.v39_evaluator, math_path=args.math_helper,
            previous_online_dir=args.previous_online_run_dir, previous_online_sha256=args.previous_online_run_sha256,
            v44_evaluator_path=args.v44_evaluator)
    _, math = load_frozen(args.v39_evaluator, args.math_helper)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(math.plain(result), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'common_fresh_counts': {
        label: row['common_fresh_count'] for label, row in result['common_fresh'].items()}}))


if __name__ == '__main__':
    main()
