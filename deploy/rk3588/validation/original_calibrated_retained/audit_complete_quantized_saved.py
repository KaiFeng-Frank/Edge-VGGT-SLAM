#!/usr/bin/env python3
"""Saved NumPy/JSON/TUM only: original CPU, current FP16 and INT8 groups.

No new inference, SDK, Torch, calibration or compiler. Fixed numerical gates
and original timestamp/Sim3/adjacent-RPE methods remain separate from quality
acceptance. Aggregator CPU arrays are unavailable in the original v53 run.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

import numpy as np

PUBLIC = Path('/dev/shm/edge-vggt-v50/public-snapshot')
PRED_KEYS = ('pose_enc', 'extrinsic', 'intrinsic', 'depth', 'depth_conf')
AGG_KEYS = ('layer_04', 'layer_11', 'layer_17', 'layer_23', 'target_tokens')
SHAPES = dict(pose_enc=(2, 9), extrinsic=(2, 3, 4), intrinsic=(2, 3, 3),
              depth=(2, 84, 112, 1), depth_conf=(2, 84, 112))
INDICES = list(range(32, 798, 48)) + [797]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def load_run(root, cpu=False):
    report = json.loads((root / 'result.json').read_text())
    cleanup = report.get('cleanup_errors', []) if cpu else report['cleanup_errors']
    if (report['status'] != 'completed_not_quality_acceptance' or cleanup
            or report['input_indices'] != INDICES or report['image_shape'] != [2, 3, 84, 112]
            or report['strict_rtol'] != 1e-4 or report['strict_atol'] != 1e-5):
        raise ValueError('Need a completed matched original run with fixed tolerance')
    if cpu:
        name = 'cpu_fp32'
    else:
        if len(report['branches']) != 1:
            raise ValueError('Need exactly one complete accelerator branch')
        name = next(iter(report['branches']))
    branch = report['branches'][name]
    if (branch['status'] != 'completed' or len(branch['updates']) != 16
            or [r['pair'] for r in branch['updates']] != list(range(1, 17))
            or [r['timestamp'] for r in branch['updates']] != report['input_timestamps'][1:]):
        raise ValueError('Need the same 16 chronological overlapping pairs')
    return report, branch, root / name


def owned_arrays(path, keys, shapes, *, report):
    report['array_files'].append(dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path)))
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(keys):
            raise ValueError('Original saved output set differs: ' + path.name)
        arrays = {key: np.array(saved[key], copy=True) for key in keys}
    for key, value in arrays.items():
        if value.dtype != np.float32 or value.shape != shapes[key]:
            raise ValueError('Original output shape/dtype differs: ' + path.name + ':' + key)
    return arrays


def accumulate(output, key, actual, reference, compare):
    row = output.setdefault(key, dict(arrays=0, strict_passed_arrays=0, byte_equal_arrays=0,
        elements=0, nonfinite_actual_elements=0, nonfinite_reference_elements=0,
        outside_tolerance=0, max_abs=0., squared_error=0., reference_squared_norm=0.,
        absolute_error_sum=0.))
    row['arrays'] += 1
    row['elements'] += actual.size
    row['nonfinite_actual_elements'] += int(np.count_nonzero(~np.isfinite(actual)))
    row['nonfinite_reference_elements'] += int(np.count_nonzero(~np.isfinite(reference)))
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        row['outside_tolerance'] += int(actual.size)
        return
    result = compare(np, actual, reference)
    row['strict_passed_arrays'] += int(result['strict_pass'])
    row['byte_equal_arrays'] += int(actual.tobytes() == reference.tobytes())
    row['outside_tolerance'] += result['outside_tolerance']
    row['max_abs'] = max(row['max_abs'], result['max_abs'])
    expected = reference.astype(np.float64)
    error = actual.astype(np.float64) - expected
    row['squared_error'] += float(np.sum(error * error))
    row['reference_squared_norm'] += float(np.sum(expected * expected))
    row['absolute_error_sum'] += float(np.sum(np.abs(error)))


def finalize(rows):
    for row in rows.values():
        invalid = row['nonfinite_actual_elements'] or row['nonfinite_reference_elements']
        row['relative_l2'] = None if invalid else float(np.sqrt(row['squared_error']) / max(np.sqrt(row['reference_squared_norm']), 1e-30))
        row['rmse'] = None if invalid else float(np.sqrt(row['squared_error'] / row['elements']))
        row['mean_abs'] = None if invalid else row['absolute_error_sum'] / row['elements']
        if invalid:
            row['max_abs'] = None
        row['strict_failed_arrays'] = row['arrays'] - row['strict_passed_arrays']
        for key in ('squared_error', 'reference_squared_norm', 'absolute_error_sum'):
            del row[key]


def audit(args):
    trajectory_path = args.public / 'deploy/rk3588/validation/original_full_aggregator/evaluate_original_trajectory.py'
    compare_path = args.public / 'deploy/rk3588/validation/original_pair_group/check_compile_pair_group.py'
    if sha(compare_path) != '4094e393d0504954a67993c24e82c5534862393930f60db3ec64c2c3e4098b76':
        raise ValueError('Frozen FP32 comparison helper differs')
    if sha(trajectory_path) != 'f7841599c6f0e1d58503444c4b90b369acd581fc697469f8fdbf01ecab55ff65':
        raise ValueError('Original trajectory association/Sim3/RPE helper differs')
    trajectory = module(trajectory_path, 'trajectory_saved_only')
    compare = module(compare_path, 'compare_saved_only').compare
    loaded = [load_run(args.cpu, True), load_run(args.fp16), load_run(args.int8)]
    cpu, fp16, int8 = loaded
    if (int8[0].get('calibrated_retained_aggregator') is not True
            or int8[0].get('internal_precision_changed') is not True
            or int8[0].get('requested_aggregator_quantized_dtype') != 'w8a8'):
        raise ValueError('INT8 run must explicitly identify the changed aggregator precision')
    for original, _, _ in loaded[1:]:
        if original['input_timestamps'] != cpu[0]['input_timestamps']:
            raise ValueError('Original CPU and NPU timestamps differ')
    identity_fields = ('image_shape', 'input_indices', 'input_timestamps', 'model_threads', 'cpu_affinity',
        'script_sha256', 'manifest_sha256', 'dino_manifest_sha256', 'depth_manifest_sha256',
        'camera_split4', 'paired_aggregator', 'retained_aggregator', 'grouped_dino',
        'checkpoint_pageout', 'half_io_library_sha256', 'reused_cpu_reference')
    if any(fp16[0][key] != int8[0][key] for key in identity_fields):
        raise ValueError('FP16/INT8 complete run identity/configuration differs')
    report = dict(schema='v80_original_VGGT_saved_complete_INT8_quality_diagnostic',
        status='saved_data_audited_not_quality_accepted', original_model_only=True,
        candidate_accepted=False, quality_accepted=False, realtime_validated=False,
        source_sha256=sha(__file__), strict_rtol=1e-4, strict_atol=1e-5,
        input_indices=INDICES, input_timestamps=cpu[0]['input_timestamps'],
        result_sha256={key: sha(root / 'result.json') for key, root in
            (('cpu', args.cpu), ('fp16', args.fp16), ('int8', args.int8))},
        helper_sha256=dict(trajectory=sha(trajectory_path), compare=sha(compare_path)),
        original_run_identity_fields_equal=True, array_files=[],
        original_CPU_top_cleanup_field_recorded='cleanup_errors' in cpu[0],
        predictions={k: {} for k in ('fp16_vs_cpu', 'int8_vs_cpu', 'int8_vs_fp16')},
        aggregator={'int8_vs_fp16': {}}, trajectories={}, timing={}, resources={},
        aggregator_vs_original_CPU_available=False)
    shapes = {key: ((2, 53, 1024) if key == 'target_tokens' else (1, 2, 53, 2048)) for key in AGG_KEYS}
    for index in range(1, 17):
        arrays = [owned_arrays(folder / f'pair-{index:02d}.npz', PRED_KEYS, SHAPES, report=report)
                  for _, _, folder in loaded]
        for key in PRED_KEYS:
            for name, actual, expected in (('fp16_vs_cpu', arrays[1][key], arrays[0][key]),
                                           ('int8_vs_cpu', arrays[2][key], arrays[0][key]),
                                           ('int8_vs_fp16', arrays[2][key], arrays[1][key])):
                accumulate(report['predictions'][name], key, actual, expected, compare)
        left, right = [owned_arrays(row[2] / f'aggregator-{index:02d}.npz', AGG_KEYS, shapes, report=report)
                       for row in (fp16, int8)]
        for key in AGG_KEYS:
            accumulate(report['aggregator']['int8_vs_fp16'], key, right[key], left[key], compare)
    for value in report['predictions'].values():
        finalize(value)
    finalize(report['aggregator']['int8_vs_fp16'])
    gt, poses = trajectory.read_tum(args.groundtruth), []
    for key, (run, branch, folder) in zip(('cpu', 'fp16', 'int8'), loaded):
        rows = trajectory.read_tum(folder / 'poses.tum')
        if len(rows) != 17 or not np.array_equal(rows[:, 0], np.asarray(run['input_timestamps'])):
            raise ValueError('Need 17 original pose timestamps, including the anchor')
        poses.append(rows)
        report['trajectories'][key + '_vs_GT'] = trajectory.evaluate_tum(rows, gt)
        durations = [item['complete_service_seconds'] for item in branch['updates']]
        report['timing'][key] = dict(count=16, mean_seconds=statistics.mean(durations),
            warm15_mean_seconds=statistics.mean(durations[1:]), first_seconds=durations[0],
            p95_seconds=float(np.percentile(durations, 95)),
            poses_sha256=sha(folder / 'poses.tum'), non_interleaved_offline_scope=True)
        if key != 'cpu':
            resources = {}
            for name in ('rknn', 'dino_rknn', 'camera_rknn', 'depth_rknn'):
                info = branch[name]
                resources[name] = {field: info[field] for field in
                    ('closed', 'live_contexts', 'contexts_created', 'contexts_destroyed', 'cleanup_errors')}
                if not info['closed'] or info['live_contexts'] or info['cleanup_errors'] or info['contexts_created'] != info['contexts_destroyed']:
                    raise ValueError('Complete original backend cleanup failed')
            report['resources'][key] = resources
    report['trajectories']['fp16_vs_cpu'] = trajectory.evaluate_tum(poses[1], poses[0])
    report['trajectories']['int8_vs_cpu'] = trajectory.evaluate_tum(poses[2], poses[0])
    report['trajectories']['int8_vs_fp16'] = trajectory.evaluate_tum(poses[2], poses[1])
    report['groundtruth_sha256'] = sha(args.groundtruth)
    report['metric_scale_validated'] = False
    report['limits'] = ['Sim3 scale is fitted and reused for adjacent matched-pose RPE; no metric-scale acceptance.',
        'This measures saved offline 16-pair outputs; it does not establish fresh-camera rate or 100-ms latency.',
        'Original FP32 strict failures exclude a numerical-equivalence claim; they do not independently replace the user ATE/RPE 10-20% quality criterion. Broad quality acceptance remains unverified.',
        'Original v53 CPU aggregator arrays were not saved; internal tensors are compared to the same FP16 run.']
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fp16', type=Path, required=True)
    p.add_argument('--int8', type=Path, required=True)
    p.add_argument('--cpu', type=Path, default=Path('/dev/shm/edge-vggt-v53/complete-ab-r1'))
    p.add_argument('--groundtruth', type=Path, default=Path('/dev/shm/edge-vggt-v53/groundtruth.txt'))
    p.add_argument('--public', type=Path, default=PUBLIC)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    try:
        report = audit(args)
    except Exception as error:
        report = dict(schema='v80_original_VGGT_saved_complete_INT8_quality_diagnostic',
            status='failed_saved_audit', candidate_accepted=False,
            error=type(error).__name__ + ': ' + str(error), source_sha256=sha(__file__))
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        raise
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], output_sha256=sha(args.output)), allow_nan=False))


if __name__ == '__main__':
    main()
