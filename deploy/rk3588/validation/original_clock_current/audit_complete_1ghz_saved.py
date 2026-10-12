#!/usr/bin/env python3
"""Compare saved v74 300 MHz / v75 1 GHz outputs; no model or SDK execution."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np

BASELINE_RESULT_SHA = '94190f294f13a07f0390edc60c1488f73f3257893413b6fd0812c1bcaff9997b'
CANDIDATE_RESULT_SHA = 'fd97453561cd258f69684953bb8993f709cb7f79c95f7e463ed98acfe0586358'
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
        assert branch['cache'].keys() == baseline_cache.keys()
        assert all(value == baseline_cache[key] for key, value in branch['cache'].items()
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
        values = [row['complete_service_seconds'] for row in branch['updates']]
        assert all(math.isfinite(value) and value > 0 for value in values)
        strict_checks = [x for row in branch['updates'] for x in row['original_fp32_comparison'].values()]
        assert len(strict_checks) == 80 and all(x['finite'] and x['strict_pass'] is False for x in strict_checks)
        timing.append(dict(mean_seconds=statistics.mean(values), first_seconds=values[0],
                           warm15_mean_seconds=statistics.mean(values[1:]),
                           service_count=len(values), P95_seconds=float(np.quantile(values, 0.95)),
                           service_seconds=values, SDK_calls=262,
                           peak_RSS_MiB=report['peak_RSS_MiB'], closed_contexts=contexts,
                           original_FP32_strict_checks=80, original_FP32_strict_passes=0))
    return dict(schema='v75_original_complete_1GHz_saved_audit', helper_sha256=helper_sha256, four_stages_verified_both_runs=True,
                status='saved_arrays_verified', input_indices=a['input_indices'],
                model_OR_ORT_OR_SDK_executed=False, original_model_only=True, candidate_accepted=False,
                production_sources_identical=True, comparator_sha256=COMPARATOR_SHA,
                baseline_frequency_hz=300000000, candidate_requested_frequency_hz=1000000000,
                frequency_samples_or_restoration_verified_here=False,
                baseline_result_sha256=sha(baseline / 'result.json'), candidate_result_sha256=sha(candidate / 'result.json'),
                array_count=sum(row['arrays'] for row in rows), all_160_arrays_byte_equal=True,
                whole_npz_files_byte_equal=sum(row['whole_file_byte_equal'] for row in rows), artifacts=rows,
                poses_tum_byte_equal=pose_byte_equal, baseline_poses_sha256=sha(pose_a), candidate_poses_sha256=sha(pose_b),
                baseline=timing[0], candidate=timing[1], timing_comparison_interleaved=False,
                original_FP32_comparison_receipts_unchanged=True, original_FP32_comparisons_recomputed_here=False,
                timing_scope='Sequential offline full original VGGT + SL4 on the same 16 pairs: saved v74 300 MHz versus v75 requested 1 GHz. Frequency samples/restoration are audited separately; this is not timestamp replay or isolated causal speed proof.',
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
