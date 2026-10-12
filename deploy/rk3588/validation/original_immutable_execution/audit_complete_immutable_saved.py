#!/usr/bin/env python3
"""Compare saved v77 standard and exclusive immutable sessions; no model or SDK execution."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np

BASELINE_RESULT_SHA = '5114f279884368b6d832500c9eda30a9f56b8eea91da23763551dc58abab3bec'
CANDIDATE_RESULT_SHA = '4aea0ebd69c703c4877da4faedb754f1e1327516ae7a77e9c2081b59220c8676'
COMPARATOR_SHA = '75b81c6b8dd2c9fcb96f76bdb5fecb5d15e5c7794a161deee8aa4f323d020e7c'
HELPER_SHA = '1948fa3f49814d8fc1025516025df79adc7b1bd0e0a54a86f842b1590f445777'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(baseline, candidate, helper_sha256):
    assert sha(baseline / 'result.json') == BASELINE_RESULT_SHA
    assert sha(candidate / 'result.json') == CANDIDATE_RESULT_SHA
    assert helper_sha256 == HELPER_SHA
    reports = [json.loads((root / 'result.json').read_text()) for root in (baseline, candidate)]
    a, b = reports
    assert a['status'] == b['status'] == 'completed_not_quality_acceptance'
    assert a['cleanup_errors'] == b['cleanup_errors'] == []
    for field in ('input_indices', 'input_timestamps', 'image_shape', 'manifest_sha256',
                  'dino_manifest_sha256', 'depth_manifest_sha256',
                  'cpu_affinity', 'model_threads', 'camera_split4', 'reused_cpu_reference',
                  'strict_rtol', 'strict_atol', 'paired_aggregator', 'grouped_dino',
                  'retained_aggregator', 'save_aggregator_outputs', 'half_io_library_sha256',
                  'half_io_source_sha256', 'half_io_dino', 'internal_precision_changed'):
        assert a[field] == b[field], field
    assert a['script_sha256'] == b['script_sha256'] == COMPARATOR_SHA
    assert b['runtime_source_sha256'] == a['runtime_source_sha256']
    assert b['runtime_source_sha256']['checkpoint_residency.py'] == helper_sha256
    assert a['strict_rtol'] == 1e-4 and a['strict_atol'] == 1e-5
    assert a['checkpoint_pageout'] is b['checkpoint_pageout'] is True
    assert len(a['input_indices']) == 17 and len(a['branches']) == len(b['branches']) == 1
    assert a['branches'].keys() == b['branches'].keys()
    name = next(iter(a['branches']))
    for root, report in zip((baseline, candidate), reports):
        stages = report['branches'][name]['checkpoint_pageout_stages']
        assert [v['stage'] for v in stages] == ['aggregator_installed', 'camera_installed', 'DINO_installed', 'depth_installed']
        for info in stages:
            relative = Path(info['file'])
            assert not relative.is_absolute() and '..' not in relative.parts
            receipt_path = root / relative
            assert sha(receipt_path) == info['sha256']
            receipt = json.loads(receipt_path.read_text())
            assert receipt['script_sha256'] == helper_sha256 and receipt['stage'] == info['stage']
            assert receipt['status'] == info['status'] == 'advice_completed_not_memory_or_quality_acceptance'
            assert receipt['errors'] == [] and receipt['tensor_storage_VA_version_unchanged'] and receipt['checkpoint_VMA_layout_unchanged']
            assert len(receipt['advice_calls']) == 1 and receipt['advice_calls'][0]['returncode'] == 0
    private_root = candidate.parent
    policy_path = private_root / 'immutable-policy-r2.json'
    assert sha(policy_path) == '15a1322b1475df9a0156ba9457f4d8aed404016ce052da8f86091b33d0384ca2'
    policy = json.loads(policy_path.read_text())
    assert policy['status'] == 'completed_original_entrypoint_pending_saved_output_review'
    assert policy['mode'] == 'immutable' and policy['exclusive_immutable_model_contract']
    assert policy['full_mutable_model_guard_semantics_preserved'] is False
    assert policy['cleanup_errors'] == [] and policy['model_graph_or_weights_changed'] is False
    assert all(policy[k] for k in ('installer_restored', 'entrypoint_unchanged',
                                  'policy_source_unchanged', 'cache_source_unchanged'))
    assert sha(private_root / 'run_immutable_comparison_board.py') == policy['script_sha256']
    assert sha(private_root / 'immutable_execution_session.py') == policy['policy_source_sha256']
    assert policy['entrypoint_sha256'] == COMPARATOR_SHA
    assert len(policy['sessions']) == 1
    session = policy['sessions'][0]
    assert session['contract'] == 'exclusive_immutable_model'
    assert session['full_seal_validated'] and session['final_full_validated']
    assert session['closed'] and not session['failed'] and session['cleanup_errors'] == []
    assert session['fixed_tensors'] == 1343 and session['version_checks'] == 32
    assert session['method_checks'] == 64 and session['restored_overrides'] == 3
    assert session['invalidations'] == 0 and not session['original_mutable_model_guard_semantics_equivalent']
    controls = []
    for directory in ('baseline-control-r1', 'immutable-control-r2'):
        path = private_root / directory / 'control.json'
        control = json.loads(path.read_text())
        assert control['status'] == 'worker_completed_pending_result_review'
        assert control['child_returncode'] == 0 and control['restore_errors'] == []
        assert control['requested_frequency_hz'] == 1000000000
        assert control['frequency_governor_limits_restored'] and control['restored'] == control['baseline']
        assert int(control['restored']['cur_freq']) == 300000000
        observations = control['observations']
        assert observations and all(x['actual_frequency_hz'] == 1000000000 for x in observations)
        assert all(math.isfinite(x['seconds']) for x in observations)
        assert all(x['seconds'] < y['seconds'] for x, y in zip(observations, observations[1:]))
        temperatures = [v for x in observations for v in x['temperatures_c'].values()]
        assert temperatures and all(math.isfinite(v) for v in temperatures)
        assert max(temperatures) < control['temperature_limit_c'] == 75.0
        controls.append(dict(file=directory + '/control.json', sha256=sha(path),
            samples=len(observations), all_sampled_frequencies_hz=1000000000,
            max_temperature_C=max(temperatures), child_returncode=0,
            restored_frequency_hz=300000000, governor_and_limits_restored=True,
            seconds=control['seconds']))
    rows = []
    for prefix, expected_keys in [('pair', {'pose_enc', 'extrinsic', 'intrinsic', 'depth', 'depth_conf'}),
                                  ('aggregator', {'layer_04', 'layer_11', 'layer_17', 'layer_23', 'target_tokens'})]:
        for i in range(1, 17):
            relative = Path(name) / f'{prefix}-{i:02d}.npz'
            left, right = baseline / relative, candidate / relative
            with np.load(left, allow_pickle=False) as x, np.load(right, allow_pickle=False) as y:
                assert set(x.files) == set(y.files) == expected_keys
                for key in sorted(expected_keys):
                    xx, yy = x[key], y[key]
                    assert xx.dtype == yy.dtype and xx.shape == yy.shape
                    assert np.isfinite(xx).all() and np.isfinite(yy).all()
                    assert xx.tobytes() == yy.tobytes(), (str(relative), key)
            rows.append(dict(file=str(relative), baseline_sha256=sha(left), candidate_sha256=sha(right),
                             whole_file_byte_equal=left.read_bytes() == right.read_bytes(), arrays=len(expected_keys)))
    pose_a, pose_b = [root / name / 'poses.tum' for root in (baseline, candidate)]
    pose_byte_equal = pose_a.read_bytes() == pose_b.read_bytes()
    assert pose_byte_equal
    assert len(rows) == 32 and sum(row['arrays'] for row in rows) == 160
    for left, right in zip(a['branches'][name]['updates'], b['branches'][name]['updates']):
        for field in ('pair', 'timestamp', 'original_fp32_comparison'):
            assert left[field] == right[field], field
    timing = []
    for report in reports:
        branch = report['branches'][name]
        assert branch['status'] == 'completed' and len(branch['updates']) == 16 and branch['cache']['closed']
        baseline_cache = a['branches'][name]['cache']
        cache = dict(branch['cache'])
        if report is b:
            session = cache.pop('immutable_execution_session')
            assert session == policy['sessions'][0]
        assert cache.keys() == baseline_cache.keys()
        assert all(value == baseline_cache[key] for key, value in cache.items()
                   if not key.endswith('_seconds'))
        contexts = 0
        for key in ('rknn', 'camera_rknn', 'dino_rknn', 'depth_rknn'):
            info = branch[key]
            assert info['closed'] and info['live_contexts'] == 0 and info['cleanup_errors'] == []
            assert info['contexts_created'] == info['contexts_destroyed']
            contexts += info['contexts_created']
        assert contexts == 12
        assert branch['rknn']['sdk_run_attempts'] == branch['rknn']['sdk_runs_succeeded'] == 64
        assert branch['rknn']['outputs_acquired'] == branch['rknn']['outputs_released'] == 144
        assert branch['dino_rknn']['sdk_run_attempts'] == branch['dino_rknn']['sdk_runs_succeeded'] == 102
        assert branch['camera_rknn']['npu_run_api_call_count'] == 64
        assert branch['depth_rknn']['sdk_run_attempts'] == branch['depth_rknn']['sdk_runs_succeeded'] == 32
        assert [row['pair'] for row in branch['updates']] == list(range(1, 17))
        assert [row['timestamp'] for row in branch['updates']] == report['input_timestamps'][1:]
        for i, update in enumerate(branch['updates'], 1):
            meta = update['aggregator_artifact']
            path = baseline if report is a else candidate
            path = path / name / meta['file']
            assert meta['file'] == f'aggregator-{i:02d}.npz'
            assert path.stat().st_size == meta['bytes'] and sha(path) == meta['sha256']
        values = [row['complete_service_seconds'] for row in branch['updates']]
        assert all(math.isfinite(value) and value > 0 for value in values)
        strict_checks = [x for row in branch['updates'] for x in row['original_fp32_comparison'].values()]
        assert len(strict_checks) == 80 and all(x['finite'] and x['strict_pass'] is False for x in strict_checks)
        timing.append(dict(mean_seconds=statistics.mean(values), first_seconds=values[0],
                           warm15_mean_seconds=statistics.mean(values[1:]),
                           service_count=len(values), P95_seconds=float(np.quantile(values, 0.95)),
                           service_seconds=values, SDK_calls=262,
                           peak_RSS_MiB=report['peak_RSS_MiB'],
                           post_cleanup_RSS_MiB=report['post_cleanup_RSS_MiB'],
                           active_perpair_RSS_MiB=[u['checkpoint_pageout_post_inference']['status_kB']['VmRSS']/1024 for u in branch['updates']],
                           cache_guard_seconds=branch['cache']['guard_seconds'],
                           retained_guard_seconds=branch['rknn']['guard_seconds'],
                           closed_contexts=contexts,
                           original_FP32_strict_checks=80, original_FP32_strict_passes=0))
    return dict(schema='v77_original_complete_exclusive_immutable_saved_audit', checkpoint_residency_helper_sha256=helper_sha256, four_stages_verified_both_runs=True,
                status='saved_arrays_verified', input_indices=a['input_indices'],
                model_OR_ORT_OR_SDK_executed=False, original_model_only=True, candidate_accepted=False,
                reported_original_runtime_source_hashes_identical=True, executed_original_comparator_sha256=COMPARATOR_SHA,
                source_identity_scope='Hashes reported by the two original comparator receipts; wrapper/policy identity is bound independently. No claim that current public worktree entrypoints remain unchanged.',
                immutable_wrapper_sha256=policy['script_sha256'], immutable_policy_sha256=policy['policy_source_sha256'],
                baseline_requested_frequency_hz=1000000000, candidate_requested_frequency_hz=1000000000,
                frequency_samples_and_restoration_verified_here=True, controls=controls,
                policy_receipt_sha256=sha(policy_path), immutable_session=session,
                full_mutable_model_guard_semantics_equivalent=False,
                exclusive_immutable_model_contract=True,
                baseline_result_sha256=sha(baseline / 'result.json'), candidate_result_sha256=sha(candidate / 'result.json'),
                array_count=sum(row['arrays'] for row in rows), all_160_arrays_byte_equal=True,
                whole_npz_files_byte_equal=sum(row['whole_file_byte_equal'] for row in rows), artifacts=rows,
                poses_tum_byte_equal=pose_byte_equal, baseline_poses_sha256=sha(pose_a), candidate_poses_sha256=sha(pose_b),
                baseline=timing[0], candidate=timing[1], timing_comparison_interleaved=False,
                mean_service_change_percent=100*(timing[1]['mean_seconds']/timing[0]['mean_seconds']-1),
                warm15_service_change_percent=100*(timing[1]['warm15_mean_seconds']/timing[0]['warm15_mean_seconds']-1),
                original_FP32_comparison_receipts_unchanged=True, original_FP32_comparisons_recomputed_here=False,
                timing_scope='Sequential offline full original VGGT + SL4 on the same 16 pairs at sampled 1GHz: standard mutable-model guards versus explicit exclusive immutable-session guards. Initialization/seal/final validation are outside per-pair service. This is not timestamp replay or interleaved causal speed proof, and arbitrary live mutation detection is intentionally different.',
                real_time_validated=False, trajectory_quality_accepted=False,
                auditor_sha256=sha(__file__))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--helper-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    result = audit(args.baseline, args.candidate, args.helper_sha256)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'array_count', 'poses_tum_byte_equal')}))
