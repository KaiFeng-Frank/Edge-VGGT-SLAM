#!/usr/bin/env python3
"""Compare saved full original-model runs; no model or SDK execution."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(baseline, candidate, helper_sha256):
    reports = [json.loads((root / 'result.json').read_text()) for root in (baseline, candidate)]
    a, b = reports
    assert a['status'] == b['status'] == 'completed_not_quality_acceptance'
    assert a['cleanup_errors'] == b['cleanup_errors'] == []
    for field in ('input_indices', 'input_timestamps', 'image_shape', 'manifest_sha256',
                  'dino_manifest_sha256', 'depth_manifest_sha256',
                  'cpu_affinity', 'model_threads', 'camera_split4', 'reused_cpu_reference'):
        assert a[field] == b[field], field
    expected_sources = dict(a['runtime_source_sha256'], **{'checkpoint_residency.py': helper_sha256})
    assert b['runtime_source_sha256'] == expected_sources
    assert b['checkpoint_pageout'] is True
    assert len(a['input_indices']) == 17 and len(a['branches']) == len(b['branches']) == 1
    assert a['branches'].keys() == b['branches'].keys()
    name = next(iter(a['branches']))
    stages = b['branches'][name]['checkpoint_pageout_stages']
    assert [v['stage'] for v in stages] == ['aggregator_installed', 'camera_installed', 'DINO_installed', 'depth_installed']
    for info in stages:
        relative = Path(info['file'])
        assert not relative.is_absolute() and '..' not in relative.parts
        receipt_path = candidate / relative
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
    timing = []
    for report in reports:
        branch = report['branches'][name]
        assert branch['status'] == 'completed' and len(branch['updates']) == 16 and branch['cache']['closed']
        contexts = 0
        for key in ('rknn', 'camera_rknn', 'dino_rknn', 'depth_rknn'):
            info = branch[key]
            assert info['closed'] and info['live_contexts'] == 0 and info['cleanup_errors'] == []
            assert info['contexts_created'] == info['contexts_destroyed']
            contexts += info['contexts_created']
        assert contexts == 12
        values = [row['complete_service_seconds'] for row in branch['updates']]
        timing.append(dict(mean_seconds=statistics.mean(values), first_seconds=values[0],
                           warm15_mean_seconds=statistics.mean(values[1:]),
                           peak_RSS_MiB=report['peak_RSS_MiB'], closed_contexts=contexts,
                           original_FP32_strict_passes=sum(x['strict_pass'] for row in branch['updates']
                                                          for x in row['original_fp32_comparison'].values())))
    return dict(schema='v74_original_checkpoint_staged_pageout_complete_saved_audit', helper_sha256=helper_sha256, four_stages_verified=True,
                status='saved_arrays_verified', input_indices=a['input_indices'],
                baseline_result_sha256=sha(baseline / 'result.json'), candidate_result_sha256=sha(candidate / 'result.json'),
                array_count=sum(row['arrays'] for row in rows), all_160_arrays_byte_equal=True,
                whole_npz_files_byte_equal=sum(row['whole_file_byte_equal'] for row in rows), artifacts=rows,
                poses_tum_byte_equal=pose_byte_equal, baseline_poses_sha256=sha(pose_a), candidate_poses_sha256=sha(pose_b),
                baseline=timing[0], candidate=timing[1], timing_comparison_interleaved=False,
                timing_scope='Sequential offline full model + SL4, not timestamp replay or causal speed proof',
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
