#!/usr/bin/env python3
"""Saved-only v45 single-variable half initial-motion trial versus both CPU baselines.

Reads each actual run schema without rewriting it. Reuses byte-pinned v39
fresh/EOF/ratio helpers and v34 trajectory math (Princeton-VL DPVO, MIT).
No image, model, native library, Torch, ORT, SDK or BA execution.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

V39_EVALUATOR_SHA = '56905a1c1906c7730b841e58f22b0460140875fc2bc616464452e907360baf4f'
MATH_SHA = '09b7a090a8dffa2401693d61baff9fbf73ddf368deceadf2f3631137b81d55d8'
POLICY_SHA = 'ca0f1632f73765c913cd707c03df54d45b03d136f45e33b6cc034ac85cb2e7b4'
BASE_POLICY_SHA = 'cc6a31f89bdcff0b43c0d51266ad62c48d0e297346845a98d42bad5c8874ef4a'
DRIVER_SHA = '3b0a5b809cab8277e1275bb0b3d4bacd99c14622e9b4091d980286276457543a'
BASE_STREAM_SHA = '295233513c1b670c953db25bbf4bfad8bdb72b2001f40ac3e1bf5d17083f9bde'
STREAM_SHA = 'bcc7cea80c758b49c2abef1a4bd961ff130a0144fe2d0f7e5db76009754ca8ce'
BASE_DRIVER_SHA = '18ad8f7bc3cc48790d2329b35372fd77473942c698cd3e7e291298194e498c2d'
FULL_DRIVER_SHA = '0f62c1df8ea68f88ec13b71b54f1f7b0b070736b7058ebd85fac013c2694b526'
HALF_RUN_SHA = 'c1f0a4636601fb1cd702ca57018af5fd937ecdbb9e722b0293428810bf66d05d'
FULL_RUN_SHA = 'a0f18920b156f9f819630b591fa71af25f5e869a9a160ccd5c7d4b471218162e'
CONTRACTS = {
    'candidate': ('v45_half_motion1_original_Torch_CPU_research_replay',
                  'completed_unpaced_half_motion1_Torch_CPU_research_not_realtime', DRIVER_SHA),
    'half_CPU_baseline': ('v41_cached_undistort_original_Torch_CPU_research_replay',
                         'completed_unpaced_half_Torch_CPU_research_not_realtime', BASE_DRIVER_SHA),
    'full_FAST_CPU_baseline': ('v34_DPVO_original_FAST_fresh_frame_CPU_replay',
                             'completed_unpaced_original_FAST_CPU_VO_research_not_realtime', FULL_DRIVER_SHA),
}
PRIMARY_METRICS = ('ATE_sim3_aligned_meters', 'RPE_rotation_degrees',
                   'RPE_translation_GT_scale_meters')
DIRECTION_METRIC = 'RPE_translation_direction_degrees_nonzero_vectors'


def require(ok, why):
    if not bool(ok):
        raise ValueError(why)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path, digest, name):
    require(sha(path) == digest, 'Frozen helper SHA differs: ' + Path(path).name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_completed_run(run, label):
    schema, status, driver = CONTRACTS[label]
    require(run['schema'] == schema and run['status'] == status and
            run['script_sha256'] == driver, 'Actual schema/status/driver binding differs: ' + label)
    require(run['frames_requested'] == run['frames_completed'] == len(run['frames']) == 192,
            'Completed192 observations required: ' + label)
    require(run['cleanup_errors'] == [] and run['affinity_restored'] is True,
            'Cleanup/affinity restoration failed: ' + label)
    require(run['GT_or_sensor_depth_input'] is False and run['ORB_used'] is False and
            run['added_LK_PnP_frontend'] is False and run['NPU_used'] is False and
            run['original_VGGT_default_unchanged'] is True,
            'Unexpected runtime prior/frontend/backend/default: ' + label)
    require(run['original_weights_unchanged'] is True and len(run['original_weights_before']) == 94 and
            run['original_weights_before'] == run['original_weights_after'],
            'Original94 weights not stable in saved metadata: ' + label)
    require(run['sources_and_inputs_unchanged'] is True and bool(run['source_pins']) and
            run['source_pins'] == run['source_hashes_after'],
            'Sources/inputs not stable in saved metadata: ' + label)
    require(run['encoder_metrics']['closed'] is True and
            run['encoder_metrics']['completed_frames'] == 192 and
            run['random_depth_draw_calls'] == 192, 'Encoder/RNG accounting differs: ' + label)
    if label != 'full_FAST_CPU_baseline':
        corr, maps = run['correlation'], run['undistort_cache']
        require(corr['closed'] is True and corr['current_context_bytes'] == 0 and
                corr['current_cached_frames'] == 0 and maps['closed'] is True and
                maps['current_map_bytes'] == 0, 'Cached CPU context/map close differs: ' + label)
    else:
        require(run['correlation']['current_context_open'] is False,
                'Full FAST correlation context remains open')
    terminated = run.get('termination')
    require(isinstance(terminated, dict) and terminated['initialized'] is True and
            terminated['offline_updates'] == 12, 'Complete offline EOF required: ' + label)


def load_run(directory, digest, label, policy, base_policy, frozen, math):
    root = Path(directory)
    require(sha(root / 'run.json') == digest, 'Run SHA differs: ' + label)
    run = json.loads((root / 'run.json').read_text())
    check_completed_run(run, label)
    require(sha(root / 'input-manifest.json') == run['input_manifest_sha256'],
            'Input manifest SHA differs: ' + label)
    schedule = json.loads((root / 'input-manifest.json').read_text())
    source = schedule['frames']
    stamps = np.array([row['timestamp'] for row in source], np.float64)
    require(len(source) == 192 and np.isfinite(stamps).all() and
            np.all(np.diff(stamps) > 0) and schedule['GT_read'] is False,
            'Complete ordered192-source plan differs: ' + label)
    config = run['config']
    if label == 'candidate':
        require(run['policy_sha256'] == POLICY_SHA and run['policy'] == policy and
                config == policy['stream_config'] and
                run['research_trial'] == policy['research_trial'] and
                run['baseline_policy_sha256'] == BASE_POLICY_SHA and
                run['cached_CPU_parent_source_sha256'] == BASE_DRIVER_SHA and
                config == base_policy['stream_config'] and config['keyframe_threshold'] == 15.0 and
                run['candidate_stream_sha256'] == STREAM_SHA and
                run['motion_probe_threshold_feature_pixels'] == 1.0 and
                run['candidate_stream_source_audit'] == dict(
                    baseline_stream_sha256=BASE_STREAM_SHA, approved_stream_sha256=STREAM_SHA,
                    exact_two_line_delta_verified=True, original_source_verification_guards_unchanged=True),
                'Single-variable candidate policy binding differs')
    elif label == 'half_CPU_baseline':
        require(digest == HALF_RUN_SHA and run['policy_sha256'] == BASE_POLICY_SHA and
                run['policy'] == base_policy and config == base_policy['stream_config'],
                'Frozen original half CPU baseline binding differs')
    else:
        expected = dict(base_policy['stream_config'], patches_per_frame=48,
                        patch_lifetime=11, removal_window=16, optimization_window=7,
                        keyframe_index=4, keyframe_threshold=15.0)
        require(digest == FULL_RUN_SHA and config == expected,
                'Frozen original full FAST baseline config differs')
    fresh, processing, publication = frozen.fresh_rows(run, source, math,
                                                      half=label != 'full_FAST_CPU_baseline')
    return dict(run=run, schedule=schedule, source=source, stamps=stamps,
                fresh=fresh, processing=processing, publication=publication)


def without_clocks(value):
    if isinstance(value, dict):
        return {key: without_clocks(item) for key, item in value.items()
                if not key.endswith(('_seconds', '_monotonic'))}
    if isinstance(value, list):
        return [without_clocks(item) for item in value]
    return value


def check_first_divergence(candidate, baseline):
    left, right = candidate['frames'], baseline['frames']
    require(len(left) == len(right) and len(left) >= 2, 'Same source prefix required')
    differing = [i for i, (a, b) in enumerate(zip(left, right))
                 if without_clocks(a) != without_clocks(b)]
    admission_differing = [i for i, (a, b) in enumerate(zip(left, right))
                          if a['result']['admitted'] != b['result']['admitted']]
    first = admission_differing[0] if admission_differing else None
    old, new = right[1]['result'], left[1]['result']
    same_probe = new['motion_probe'] == old['motion_probe'] == 1.0345186591148376
    expected = (same_probe and old['admitted'] is False and new['admitted'] is True and
                old['reason'] == 'initial_motion_below_2' and new['reason'] == 'initialization_wait' and
                old['fresh_pose_valid'] is False and new['fresh_pose_valid'] is False and
                old['pose_w2c'] is None and new['pose_w2c'] is None)
    prefix_same = without_clocks(left[0]) == without_clocks(right[0])
    return dict(expected_prefix_source_indices=[0],
                source0_nonclock_exact_match=prefix_same,
                first_nonclock_frame_difference=differing[0] if differing else None,
                first_admission_decision_difference=first,
                observation1_same_original_motion_probe=same_probe,
                observation1_expected_threshold_admission_change_verified=expected,
                observation1_baseline_motion_probe=old['motion_probe'],
                observation1_candidate_motion_probe=new['motion_probe'],
                observation1_baseline_admitted=old['admitted'],
                observation1_candidate_admitted=new['admitted'],
                trial_isolation_verified=prefix_same and first == 1 and expected,
                ignored_fields='Only dictionary keys ending _seconds or _monotonic; sensor timestamps and numerical diagnostics retained',
                old_gold_or_capture_comparison_after_admission_divergence=False,
                source_state_scope='Saved JSON only; subsequent admissions/graphs/poses come from the actual new feedback path, never replayed old probes or gold')


def quality_gates(ratios):
    result = {}
    for percent, maximum in ((10, 1.10), (20, 1.20)):
        values = {key: ratios[key]['statistics']['rmse']['candidate_over_baseline']
                  for key in PRIMARY_METRICS}
        passed = {key: value is not None and np.isfinite(value) and value <= maximum
                  for key, value in values.items()}
        result[str(percent) + '_percent'] = dict(primary_metric_RMSE_maximum_ratio=maximum,
            primary_metric_RMSE_ratios=values, primary_metric_pass=passed,
            primary_ATE_rotation_RPE_translation_RPE_pass=all(passed.values()))
    result['additional_direction_diagnostic_not_acceptance_standard'] = ratios[DIRECTION_METRIC]
    result['unavailable_or_different_nonzero_population_is_not_a_pass'] = True
    return result


def compare(args):
    paths = {'candidate_run': args.candidate_run_dir / 'run.json',
             'half_CPU_run': args.half_baseline_run_dir / 'run.json',
             'full_FAST_run': args.full_baseline_run_dir / 'run.json',
             'policy': args.policy, 'baseline_policy': args.baseline_policy,
             'trial_driver': args.trial_driver, 'frozen_evaluator': args.frozen_evaluator,
             'math_helper': args.math_helper, 'groundtruth': args.groundtruth,
             'candidate_stream': args.candidate_stream_module, 'baseline_stream': args.baseline_stream_module}
    pins = {'candidate_run': args.candidate_run_sha256, 'half_CPU_run': args.half_baseline_run_sha256,
            'full_FAST_run': args.full_baseline_run_sha256, 'policy': POLICY_SHA,
            'baseline_policy': BASE_POLICY_SHA, 'trial_driver': DRIVER_SHA,
            'frozen_evaluator': V39_EVALUATOR_SHA, 'math_helper': MATH_SHA,
            'groundtruth': args.groundtruth_sha256, 'candidate_stream': STREAM_SHA,
            'baseline_stream': BASE_STREAM_SHA}
    require(all(sha(paths[key]) == digest for key, digest in pins.items()),
            'An evaluation input/source SHA differs')
    frozen = load_module(args.frozen_evaluator, V39_EVALUATOR_SHA, '_v45_frozen_v39')
    math = load_module(args.math_helper, MATH_SHA, '_v45_frozen_math')
    driver = load_module(args.trial_driver, DRIVER_SHA, '_v45_declared_trial')
    policy, base_policy = json.loads(args.policy.read_text()), json.loads(args.baseline_policy.read_text())
    driver.validate_single_variable_policy(policy, base_policy)
    stream_source_proof = driver.validate_stream_delta(args.baseline_stream_module, args.candidate_stream_module)
    data = {label: load_run(directory, digest, label, policy, base_policy, frozen, math)
            for label, directory, digest in (
                ('candidate', args.candidate_run_dir, args.candidate_run_sha256),
                ('half_CPU_baseline', args.half_baseline_run_dir, args.half_baseline_run_sha256),
                ('full_FAST_CPU_baseline', args.full_baseline_run_dir, args.full_baseline_run_sha256))}
    trial, half = data['candidate'], data['half_CPU_baseline']
    for label, value in data.items():
        require(value['source'] == trial['source'] and
                value['schedule']['rgb_index_sha256'] == trial['schedule']['rgb_index_sha256'] and
                value['schedule']['dataset'] == trial['schedule']['dataset'],
                'Ordered original192 source images/timestamps differ: ' + label)
        require(value['run']['seed'] == trial['run']['seed'] == 20261011 and
                value['run']['random_generator_before'] == trial['run']['random_generator_before'] and
                value['run']['original_weights_before'] == trial['run']['original_weights_before'],
                'Original weights/initial shared RNG differs: ' + label)
        key = label + '_manifest'
        paths[key] = {'candidate': args.candidate_run_dir, 'half_CPU_baseline': args.half_baseline_run_dir,
                      'full_FAST_CPU_baseline': args.full_baseline_run_dir}[label] / 'input-manifest.json'
        pins[key] = value['run']['input_manifest_sha256']
    require(trial['run']['random_generator_after'] == half['run']['random_generator_after'],
            'Shared half sampling/depth RNG terminal metadata differs')
    isolation = check_first_divergence(trial['run'], half['run'])
    # Plan and fresh intersections are fixed before GT/error reads.
    plan = math.endpoint_plan(trial['stamps'])
    common_indices = {label: sorted(set(value['fresh']) & set(trial['fresh']))
                      for label, value in data.items() if label != 'candidate'}
    gt = np.loadtxt(args.groundtruth, comments='#')
    require(gt.ndim == 2 and gt.shape[1] == 8 and np.isfinite(gt).all() and
            np.all(np.diff(gt[:, 0]) > 0), 'Finite ordered TUM GT required')
    association = math.nearest_association(trial['stamps'], gt[:, 0])
    gt_c2w = math.matrices(gt[:, 1:])
    populations = {}
    for label, value in data.items():
        run, fresh = value['run'], value['fresh']
        duration = run['replay_elapsed_seconds']
        require(np.isfinite(duration) and duration > 0, 'Finite input-active duration required')
        evaluation = math.evaluate_population(fresh, association, gt_c2w, plan)
        populations[label] = dict(run_schema=run['schema'], run_status=run['status'],
            run_sha256=pins[{'candidate': 'candidate_run', 'half_CPU_baseline': 'half_CPU_run',
                            'full_FAST_CPU_baseline': 'full_FAST_run'}[label]],
            encoder_backend_reported=run['encoder_metrics'].get('backend',
                run['encoder_metrics'].get('encoder_backend')),
            delivered_observations=192, planned_observations=192, fresh_count=len(fresh),
            fresh_pose_source_coverage=len(fresh) / 192,
            first_fresh_source_index=min(fresh) if fresh else None,
            fresh_online=evaluation,
            EOF_reoptimized_and_reconstructed=frozen.evaluate_eof(value, math, association, gt_c2w, plan),
            fresh_processing_seconds=math.summary(value['processing']),
            fresh_software_admission_to_publication_seconds=math.summary(value['publication']),
            replay_elapsed_seconds=duration, fresh_over_unpaced_input_active_elapsed_Hz=len(fresh) / duration,
            termination_offline_seconds=run['termination_seconds'], peak_rss_mib=run['peak_rss_mib'],
            cleanup_errors=run['cleanup_errors'], source_stability_reported=True,
            weight_stability_reported=True, close_checks_are_saved_metadata_not_rerun=True)
    comparisons = {}
    for label, reference in data.items():
        if label == 'candidate':
            continue
        common = math.common_fresh_analysis(reference['fresh'], trial['fresh'], association,
                 gt_c2w, plan, [row['frame_id'] for row in trial['source']])
        require(common['common_fresh_source_indices'] == common_indices[label],
                'Common fresh selection changed after GT read')
        common['reference'], common['candidate'] = common.pop('default'), common.pop('FAST')
        ratios = frozen.ratios(common['reference'], common['candidate'])
        gates = quality_gates(ratios)
        source_nonregressing = len(trial['fresh']) >= len(reference['fresh'])
        endpoints_nonregressing = len(populations['candidate']['fresh_online']['used_endpoint_pairs']) >= \
            len(populations[label]['fresh_online']['used_endpoint_pairs'])
        for percentage in ('10_percent', '20_percent'):
            gates[percentage]['primary_numeric_and_nonregressing_source_endpoint_coverage'] = bool(
                gates[percentage]['primary_ATE_rotation_RPE_translation_RPE_pass'] and
                source_nonregressing and endpoints_nonregressing)
        comparisons[label] = dict(common_fresh=common, common_fresh_error_ratios=ratios,
            quality_gates=gates, candidate_source_coverage_nonregressing=source_nonregressing,
            candidate_fixed_endpoint_coverage_nonregressing=endpoints_nonregressing,
            own_source_and_endpoint_coverage_in_complete_populations=True)
    require(all(sha(paths[key]) == digest for key, digest in pins.items()),
            'Saved input/source changed during evaluation')
    return math.plain(dict(schema='v45_single_variable_half_motion_saved_trajectory_comparison',
        status='saved_numeric_quality_diagnostic_not_adoption_or_realtime_acceptance',
        evaluator_sha256=sha(__file__), input_sha256=pins, same_original_192_source_rows=True,
        fixed_all_source_1second_endpoint_plan=plan, complete_populations=populations,
        comparisons=comparisons, source_result_first_divergence=isolation,
        candidate_stream_source_proof=stream_source_proof,
        original_weights_initial_RNG_and_half_terminal_RNG_stable_in_saved_metadata=True,
        candidate_accepted=False, realtime_validated=False, original_VGGT_default_unchanged=True,
        limits=['Only initial motion rejection2->1 is a runtime variable; keyframe15 and all original half StreamConfig fields stay unchanged.',
            'Common fresh uses actual source identities before GT/error filtering; all192-source and endpoint coverage remain visible.',
            'GT nearest<=20ms one-to-one, fixed1second endpoints and one positive Sim3 per population use the frozen math unchanged.',
            'Headline10/20percent limits use ATE/rotationRPE/translationRPE RMSE; direction is additional diagnostic, not a new acceptance requirement.',
            'EOF12 offline Updates and reconstructed/initialization/rejected rows never count as fresh.',
            'All close/source/weight/RNG checks use saved metadata. No model, feature, native, SDK or board rerun.',
            'Full FAST has its saved original CPU ORT encoder; half uses original Torch CPU FP32. Separate thermal/timing histories do not prove a causal speedup.',
            'This short unpaced DPVO trial is not20Hz physical-camera, map/endurance, original-VGGT quality or adoption validation.'],
        model_forward_calls=0, ORT_calls=0, SDK_calls=0, native_calls=0, BA_calls=0,
        images_read=0, board_jobs=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate-run-dir', 'half-baseline-run-dir', 'full-baseline-run-dir',
                 'groundtruth', 'policy', 'baseline-policy', 'trial-driver', 'frozen-evaluator',
                 'math-helper', 'candidate-stream-module', 'baseline-stream-module', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('candidate-run-sha256', 'half-baseline-run-sha256',
                 'full-baseline-run-sha256', 'groundtruth-sha256'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    result = compare(args)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'],
        'trial_isolation_verified': result['source_result_first_divergence']['trial_isolation_verified'],
        'primary_RMSE_ratios': {label: row['quality_gates']['20_percent']['primary_metric_RMSE_ratios']
                              for label, row in result['comparisons'].items()}}, allow_nan=False))


if __name__ == '__main__':
    main()
