#!/usr/bin/env python3
"""Compare saved complete original-VGGT pair outputs and original SL4 poses.

NumPy and the existing saved-TUM metrics helper only: no model, Torch, ORT or
SDK execution. All timing comparisons are separate saved runs, not interleaved
or paced latency measurements. Fixed FP32 gates never imply quality acceptance.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path('/dev/shm')
FIELDS = {'pose_enc': (2,9), 'extrinsic': (2,3,4), 'intrinsic': (2,3,3),
          'depth': (2,84,112,1), 'depth_conf': (2,84,112)}
OLD_BRANCH = 'npu_fp16_DINO_aggregator_camera_depth_logits_cpu_fp32_boundaries'
NEW_BRANCH = OLD_BRANCH + '_calibrated_aggregator_calibrated_dino'
RTOL, ATOL = 1e-4, 1e-5


def compare_arrays(actual_directory, reference_directory, sha):
    names = {f'pair-{i:02d}.npz' for i in range(1,17)}
    for directory in (actual_directory, reference_directory):
        if {p.name for p in directory.glob('pair-*.npz')} != names:
            raise ValueError('Exactly 16 saved complete-pair NPZ files required: ' + str(directory))
    result = dict(rtol=RTOL, atol=ATOL, finite_same_shape_dtype=True, pair_count=16,
        pair_pass_count=0, array_count=80, array_pass_count=0, byte_equal_array_count=0,
        fields={}, artifacts=[], pairs=[])
    for key in FIELDS:
        result['fields'][key] = dict(array_count=16, array_pass_count=0, byte_equal_array_count=0,
            element_count=0, failed_element_count=0, relative_l2_values=[], max_abs_values=[])
    for index in range(1,17):
        filename = f'pair-{index:02d}.npz'
        actual_path, reference_path = actual_directory/filename, reference_directory/filename
        pair = dict(pair=index, fields={})
        with np.load(actual_path, allow_pickle=False) as actual, np.load(reference_path, allow_pickle=False) as reference:
            if set(actual.files) != set(FIELDS) or set(reference.files) != set(FIELDS):
                raise ValueError('Exactly five original canonical arrays required')
            for key, shape in FIELDS.items():
                a, b = actual[key], reference[key]
                if (a.shape != shape or b.shape != shape or a.dtype != np.float32 or b.dtype != np.float32
                        or not np.isfinite(a).all() or not np.isfinite(b).all()):
                    raise ValueError(f'Finite original FP32 shape/dtype mismatch: {filename} {key}')
                af, bf = a.astype(np.float64), b.astype(np.float64)
                delta = af-bf
                denominator = float(np.linalg.norm(bf))
                numerator = float(np.linalg.norm(delta))
                if denominator == 0 and numerator != 0:
                    raise ValueError('Relative L2 undefined for a zero reference with a nonzero error')
                outside = int(np.count_nonzero(np.abs(delta) > ATOL+RTOL*np.abs(bf)))
                row = dict(strict_pass=outside==0, failed_element_count=outside,
                    byte_equal=a.tobytes(order='C')==b.tobytes(order='C'),
                    relative_l2=numerator/denominator if denominator else 0., max_abs=float(np.abs(delta).max()))
                pair['fields'][key] = row
                aggregate = result['fields'][key]
                aggregate['array_pass_count'] += int(row['strict_pass'])
                aggregate['byte_equal_array_count'] += int(row['byte_equal'])
                aggregate['element_count'] += a.size
                aggregate['failed_element_count'] += outside
                aggregate['relative_l2_values'].append(row['relative_l2'])
                aggregate['max_abs_values'].append(row['max_abs'])
        result['pair_pass_count'] += int(all(v['strict_pass'] for v in pair['fields'].values()))
        result['pairs'].append(pair)
        result['artifacts'].append(dict(file=filename, actual_sha256=sha(actual_path), reference_sha256=sha(reference_path)))
    for row in result['fields'].values():
        row.update(relative_l2_mean=float(np.mean(row['relative_l2_values'])),
            relative_l2_maximum=max(row['relative_l2_values']), max_abs_maximum=max(row['max_abs_values']))
        result['array_pass_count'] += row['array_pass_count']
        result['byte_equal_array_count'] += row['byte_equal_array_count']
    return result


def seconds(values):
    a = np.asarray(values, dtype=np.float64)
    if len(a)!=16 or not np.isfinite(a).all() or np.any(a<=0):
        raise ValueError('Need 16 finite positive saved complete timings')
    return dict(count=16, first_seconds=float(a[0]), mean_seconds=float(a.mean()),
        p95_seconds=float(np.percentile(a,95)), subsequent_count=15,
        subsequent_mean_seconds=float(a[1:].mean()), values_seconds=a.tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('result', type=Path, nargs='?', default=Path(__file__).parent/'complete-ab-r1/result.json')
    p.add_argument('--candidate-run', type=Path, help='Candidate directory containing result.json; overrides positional result')
    p.add_argument('--cpu-run', type=Path, default=ROOT/'edge-vggt-v53/complete-ab-r1')
    p.add_argument('--previous-run', type=Path, default=ROOT/'edge-vggt-v54/depth-ab-r1')
    p.add_argument('--ground-truth', type=Path, default=ROOT/'edge-vggt-v53/groundtruth.txt')
    p.add_argument('--trajectory-helper', type=Path, default=ROOT/'edge-vggt-v53/evaluate_original_trajectory.py')
    p.add_argument('--output', type=Path, default=Path(__file__).parent/'ptq-trajectory-metrics.json')
    args = p.parse_args()
    if args.candidate_run is not None:
        args.result = args.candidate_run/'result.json'
    spec = importlib.util.spec_from_file_location('_existing_original_saved_trajectory', args.trajectory_helper)
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    sha = T.file_sha256
    paths = {'candidate': args.result, 'original_cpu': args.cpu_run/'result.json',
             'previous_depth': args.previous_run/'result.json'}
    reports = {name: json.loads(path.read_text()) for name,path in paths.items()}
    candidate = reports['candidate']
    if candidate.get('paired_aggregator') is not True or set(candidate['branches']) != {NEW_BRANCH}:
        raise ValueError('Expected the completed original 24-pair candidate branch only')
    branches = {'candidate':NEW_BRANCH, 'previous_depth':OLD_BRANCH, 'original_cpu':'cpu_fp32'}
    for name, report in reports.items():
        if (report['status'] != 'completed_not_quality_acceptance' or report['original_model_only'] is not True
                or report['input_indices'] != candidate['input_indices']
                or report['input_timestamps'] != candidate['input_timestamps']
                or len(report['input_timestamps']) != 17 or report['image_shape'] != [2,3,84,112]
                or report['strict_rtol'] != RTOL or report['strict_atol'] != ATOL
                or report['branches'][branches[name]]['status'] != 'completed'
                or report.get('cleanup_errors',[])):
            raise ValueError('Completed original runs on identical 17 observations required: ' + name)
        updates = report['branches'][branches[name]]['updates']
        if ([row['pair'] for row in updates] != list(range(1,17))
                or [row['timestamp'] for row in updates] != candidate['input_timestamps'][1:]):
            raise ValueError('Saved update order/timestamps differ: ' + name)
    folders = {'candidate':args.result.parent/NEW_BRANCH, 'previous_depth':args.previous_run/OLD_BRANCH,
               'original_cpu':args.cpu_run/'cpu_fp32'}
    for name in ('candidate','previous_depth'):
        reference = reports[name]['reused_cpu_reference']
        if reference['result_sha256'] != sha(paths['original_cpu']):
            raise ValueError('Candidate was not evaluated against this same original CPU run')
        if set(reference['array_sha256']) != {f'pair-{i:02d}.npz' for i in range(1,17)}:
            raise ValueError('Incomplete original FP32 reference pin set')
        for filename,expected in reference['array_sha256'].items():
            if sha(folders['original_cpu']/filename) != expected:
                raise ValueError('Reused original FP32 reference bytes changed')
    strict = compare_arrays(folders['candidate'], folders['original_cpu'], sha)
    previous = compare_arrays(folders['candidate'], folders['previous_depth'], sha)
    trajectory, files = {}, {f'{name}_result_sha256':sha(path) for name,path in paths.items()}
    ground_truth = T.read_tum(args.ground_truth)
    for name,folder in folders.items():
        pose_path = folder/'poses.tum'
        rows = T.read_tum(pose_path)
        if rows.shape != (17,8) or not np.array_equal(rows[:,0],np.array(candidate['input_timestamps'],np.float64)):
            raise ValueError('Original saved SL4 pose timestamps differ: ' + name)
        trajectory[name] = T.evaluate_tum(rows,ground_truth,max_difference=.02,offset=0.,delta_frames=1)
        if trajectory[name]['association']['matched_count'] != 17:
            raise ValueError('Exactly 17 GT matches are required')
        files[f'{name}_poses_sha256'] = sha(pose_path)
    if any(trajectory[n]['association']['matches'] != trajectory['candidate']['association']['matches']
           for n in ('previous_depth','original_cpu')):
        raise ValueError('The three trajectories have different GT associations')
    trajectory['delta'] = {name:{key:trajectory['candidate'][key]['rmse']-trajectory[name][key]['rmse']
        for key in ('ate_translation','rpe_translation','rpe_rotation')} for name in ('previous_depth','original_cpu')}
    timing = dict(scope='offline complete original model + SL4; excludes NPZ/poses publication; not paced latency',
        comparison_interleaved=False, complete_service={}, candidate_phase={})
    for name,report in reports.items():
        timing['complete_service'][name] = seconds([r['complete_service_seconds'] for r in report['branches'][branches[name]]['updates']])
    updates = candidate['branches'][NEW_BRANCH]['updates']
    for name in updates[0]['model_phase_seconds']:
        timing['candidate_phase'][name] = seconds([r['model_phase_seconds'][name] for r in updates])
    new_mean = timing['complete_service']['candidate']['mean_seconds']
    for name in ('previous_depth','original_cpu'):
        timing[f'saved_mean_reduction_vs_{name}_fraction'] = 1.-new_mean/timing['complete_service'][name]['mean_seconds']
    branch = candidate['branches'][NEW_BRANCH]
    cleanup = {name:{key:value[key] for key in ('contexts_created','contexts_destroyed','live_contexts','closed','failed','cleanup_errors')}
               for name,value in branch.items() if name.endswith('rknn')}
    aggregate = branch['rknn']
    if (aggregate['sdk_run_attempts']!=384 or aggregate['sdk_runs_succeeded']!=384
            or aggregate['completed_pairs']!=384 or aggregate['completed_blocks']!=768
            or aggregate['contexts_created']!=24 or aggregate['contexts_destroyed']!=24
            or aggregate['pending_pairs']!=0 or aggregate['pair_calls']!={f'pair{i}':16 for i in range(24)}
            or aggregate['block_calls']!={f'{r}{i}':16 for r in ('frame','global') for i in range(24)}
            or any(v['live_contexts']!=0 or v['cleanup_errors'] or v['failed'] or not v['closed']
                   or v['contexts_created']!=v['contexts_destroyed'] for v in cleanup.values())):
        raise ValueError('Saved pair/backend completion or cleanup counts differ')
    files['ground_truth_sha256'] = sha(args.ground_truth)
    report = dict(schema='original_VGGT_saved_complete_calibrated_DINO_pair_comparison',
        status='numerical_rejected_not_quality_acceptance' if strict['array_pass_count']<80 else 'fixed_gate_passed_not_quality_acceptance',
        scope='saved arrays and original SL4 trajectories only', candidate_accepted=False,
        quality_accepted=False,realtime_validated=False, source_sha256=sha(__file__),
        trajectory_helper_sha256=sha(args.trajectory_helper), input_indices=candidate['input_indices'],
        candidate_model_threads=candidate.get('model_threads'),
        input_timestamps=candidate['input_timestamps'],files=files,trajectory=trajectory,
        strict_comparison=strict,comparison_to_previous_depth=previous,timing=timing,cleanup=cleanup,
        aggregator=dict(sdk_runs=384,original_block_calls=768,completed_pairs=384,
            sdk_io_seconds=aggregate['sdk_io_seconds'],bridge_check_seconds=aggregate['bridge_check_seconds']))
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output),sha256=sha(args.output),status=report['status'],
        strict_arrays_passed=strict['array_pass_count'],previous_depth_byte_equal_arrays=previous['byte_equal_array_count'],
        candidate_ATE_rmse=trajectory['candidate']['ate_translation']['rmse'],
        candidate_RPE_translation_rmse=trajectory['candidate']['rpe_translation']['rmse'],
        candidate_RPE_rotation_rmse=trajectory['candidate']['rpe_rotation']['rmse'],
        candidate_service_mean_seconds=new_mean),allow_nan=False))


if __name__=='__main__':
    main()
