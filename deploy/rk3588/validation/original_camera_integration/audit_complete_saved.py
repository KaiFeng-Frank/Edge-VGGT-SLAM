#!/usr/bin/env python3
"""Audit saved original VGGT complete-run arrays; no inference or SDK imports."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

PREDICTIONS = {'pose_enc': (2, 9), 'extrinsic': (2, 3, 4), 'intrinsic': (2, 3, 3),
               'depth': (2, 84, 112, 1), 'depth_conf': (2, 84, 112)}
AGGREGATOR = {**{f'layer_{i:02d}': (1, 2, 53, 2048) for i in (4, 11, 17, 23)},
              'target_tokens': (2, 53, 1024)}
EVALUATOR_SHA = 'f7841599c6f0e1d58503444c4b90b369acd581fc697469f8fdbf01ecab55ff65'
GT_SHA = 'aac0319a6ef4e1cdf61e779d2152b95aa7e9f7b1749d6d18717b43ddabffede2'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4 << 20), b''):
            h.update(part)
    return h.hexdigest()


def load_report(directory, expected):
    p = directory / 'result.json'
    require(sha(p) == expected, 'Saved result SHA differs')
    return json.loads(p.read_text())


def arrays(path, shapes):
    with np.load(path, allow_pickle=False) as saved:
        require(len(saved.files) == len(shapes) and set(saved.files) == set(shapes), 'NPZ keys differ')
        result = {key: saved[key] for key in shapes}
    for key, value in result.items():
        require(value.dtype == np.float32 and value.shape == shapes[key]
                and value.flags.c_contiguous and np.isfinite(value).all(), f'Array shape/type/finite differs: {key}')
    return result


def raw(value):
    return hashlib.sha256(memoryview(value).cast('B')).hexdigest()


def metric(actual, reference):
    expected = reference.astype(np.float64)
    delta = actual.astype(np.float64) - expected
    outside = int(np.count_nonzero(np.abs(delta) > 1e-5 + 1e-4 * np.abs(expected)))
    denominator = float(np.linalg.norm(expected))
    require(denominator > 0, 'Zero reference L2 not supported by this recorded metric')
    return dict(max_abs=float(np.abs(delta).max()), relative_l2=float(np.linalg.norm(delta) / denominator),
                outside_tolerance_float64=outside,
                strict_pass=bool(np.allclose(actual, reference, rtol=1e-4, atol=1e-5)))


def mean(values):
    # Match the board's sequential CPython 3.11 sum, not Python 3.13 sum compensation.
    total = 0.
    for value in values:
        total += float(value)
    return total / len(values)


def timing(values):
    require(values and all(math.isfinite(x) and x >= 0 for x in values), 'Invalid timing')
    return dict(count=len(values), mean_seconds=mean(values),
                p95_seconds=float(np.percentile(values, 95, method='linear')), first_seconds=float(values[0]),
                excluding_first_mean_seconds=mean(values[1:]) if len(values) > 1 else None)


def inspect(directory, expected_sha, cpu_directory, runner_source=None):
    directory, cpu_directory = directory.resolve(strict=True), cpu_directory.resolve(strict=True)
    report = load_report(directory, expected_sha)
    require(report['status'] == 'completed_not_quality_acceptance' and report['original_model_only'] is True
            and report['quality_accepted'] is False and report['realtime_validated'] is False
            and report['strict_rtol'] == 1e-4 and report['strict_atol'] == 1e-5
            and report['image_shape'] == [2, 3, 84, 112] and report['save_aggregator_outputs'] is True
            and len(report['branches']) == 1 and report['cleanup_errors'] == [], 'Completed original diagnostic required')
    if runner_source is not None:
        require(sha(runner_source) == report['script_sha256'], 'Frozen runner source differs')
    tag, branch = next(iter(report['branches'].items()))
    folder = (directory / tag).resolve(strict=True)
    require(folder.is_relative_to(directory) and branch['status'] == 'completed', 'Branch path/status differs')
    updates = branch['updates']
    require(len(updates) == 16 and [v['pair'] for v in updates] == list(range(1, 17))
            and len(report['input_indices']) == len(report['input_timestamps']) == 17
            and [v['timestamp'] for v in updates] == report['input_timestamps'][1:], 'Input pair identity differs')
    reference = report['reused_cpu_reference']
    cpu = load_report(cpu_directory, reference['result_sha256'])
    require(cpu['original_model_only'] is True and cpu['branches']['cpu_fp32']['status'] == 'completed'
            and all(cpu[key] == report[key] for key in ('input_indices', 'input_timestamps', 'image_shape')),
            'Original CPU reference inputs differ')
    expected_files = {f'pair-{i:02d}.npz' for i in range(1, 17)}
    require({p.name for p in folder.glob('pair-*.npz')} == expected_files
            and set(reference['array_sha256']) == expected_files
            and {p.name for p in folder.glob('aggregator-*.npz')} == {f'aggregator-{i:02d}.npz' for i in range(1, 17)},
            'Complete saved array set required')
    artifact_rows, cpu_metrics = [], {key: [] for key in PREDICTIONS}
    for update in updates:
        pair = update['pair']
        pfile, afile = folder / f'pair-{pair:02d}.npz', folder / f'aggregator-{pair:02d}.npz'
        cfile = cpu_directory / 'cpu_fp32' / pfile.name
        require(sha(cfile) == reference['array_sha256'][pfile.name], 'Original CPU NPZ SHA differs')
        provenance = update['aggregator_artifact']
        require(provenance['file'] == afile.name and provenance['bytes'] == afile.stat().st_size
                and provenance['sha256'] == sha(afile) and provenance['captured_after_timed_service'] is True,
                'Saved intermediate artifact identity/timing differs')
        predicted, aggregate, original = arrays(pfile, PREDICTIONS), arrays(afile, AGGREGATOR), arrays(cfile, PREDICTIONS)
        artifact_rows.append(dict(pair=pair, prediction_file_sha256=sha(pfile), aggregator_file_sha256=sha(afile),
                                 prediction_raw_sha256={k: raw(v) for k, v in predicted.items()},
                                 aggregator_raw_sha256={k: raw(v) for k, v in aggregate.items()}))
        for key in PREDICTIONS:
            calculated = metric(predicted[key], original[key])
            recorded = update['original_fp32_comparison'][key]
            require(recorded['finite'] is True and calculated['strict_pass'] == recorded['strict_pass']
                    and calculated['max_abs'] == recorded['max_abs']
                    and math.isclose(calculated['relative_l2'], recorded['relative_l2'], rel_tol=1e-12, abs_tol=1e-15),
                    'Saved original CPU metric differs: ' + key)
            cpu_metrics[key].append(calculated)
    lifecycle, total_contexts, total_runs = {}, 0, 0
    for key, contexts, runs in (('rknn', 4 if report['retained_aggregator'] else 24,
                                 64 if report['retained_aggregator'] else 384),
                                ('dino_rknn', 6, 102), ('camera_rknn', 1, 64), ('depth_rknn', 1, 32)):
        stats = branch[key]
        successful = stats.get('sdk_runs_succeeded', stats.get('npu_run_api_call_count'))
        require(stats['contexts_created'] == stats['contexts_destroyed'] == contexts and stats['live_contexts'] == 0
                and stats['closed'] is True and stats['failed'] is False and stats['cleanup_errors'] == []
                and successful == runs, 'SDK lifecycle/count differs: ' + key)
        if 'sdk_run_attempts' in stats:
            require(stats['sdk_run_attempts'] == runs, 'SDK attempts differ')
        lifecycle[key] = dict(created=contexts, destroyed=contexts, live=0, successful_runs=runs)
        if key == 'rknn':
            buffers = 144 if report['retained_aggregator'] else 768
            require(stats['outputs_acquired'] == stats['outputs_released'] == buffers,
                    'Aggregator output-release counts differ')
            lifecycle[key]['outputs_acquired_released'] = buffers
        if key == 'dino_rknn':
            half = stats['half_io']
            require(half['calls_attempted'] == half['calls_completed'] == half['input_conversion_calls']
                    == half['output_conversion_calls'] == 102, 'DINO half-IO counts differ')
            lifecycle[key]['owned_outputs_converted'] = 102
        if key == 'camera_rknn':
            require(stats['completed_head_calls'] == 16 and stats['completed_refinements'] == 64
                    and len(stats['last_steps']) == 4
                    and all(s['output_buffer_released'] is True for s in stats['last_steps']),
                    'Camera refinement/last-output release differs')
            lifecycle[key]['last_call_released_outputs'] = 4
        if key == 'depth_rknn':
            require(stats['completed_chunk_calls'] == stats['completed_frames'] == 32,
                    'Original depth chunk counts differ')
        total_contexts += contexts
        total_runs += runs
    cache = branch['cache']
    require(cache['closed'] is True and cache['restored'] is True and cache['cache_bytes'] == 0
            and cache['requests'] == 16 and cache['hit_frames'] == 15 and cache['computed_frames'] == 17
            and cache['errors'] == cache['invalidations'] == 0, 'Patch cache lifecycle/count differs')
    phase = {key: mean([v['model_phase_seconds'][key] for v in updates]) for key in updates[0]['model_phase_seconds']}
    summary = dict(result_sha256=expected_sha, runner_sha256=report['script_sha256'], branch=tag,
                   retained_aggregator=report['retained_aggregator'], input_indices=report['input_indices'],
                   camera_split4=report.get('camera_split4', False),
                   camera_model_sha256=branch['camera_rknn']['model_sha256'],
                   runtime_source_sha256=report['runtime_source_sha256'],
                   prediction_arrays_checked=80, aggregator_arrays_checked=80, original_CPU_arrays_checked=80,
                   artifact_rows=artifact_rows, original_CPU_strict_pass=sum(v['strict_pass'] for values in cpu_metrics.values() for v in values),
                   original_CPU_strict_count=80, original_CPU_error_by_field={k: dict(
                       mean_relative_l2=mean([v['relative_l2'] for v in values]), max_relative_l2=max(v['relative_l2'] for v in values),
                       max_abs=max(v['max_abs'] for v in values), strict_pass_count=sum(v['strict_pass'] for v in values)) for k, values in cpu_metrics.items()},
                   timing=dict(prediction=timing([v['prediction_seconds'] for v in updates]),
                               complete_service=timing([v['complete_service_seconds'] for v in updates]), phase_mean_seconds=phase,
                               aggregator_excluding_DINO_mean_seconds=phase['aggregator_including_DINO']-phase['DINO'],
                               guard_seconds_per_pair=cache['guard_seconds']/16),
                   lifecycle=lifecycle, total_contexts_created_destroyed=total_contexts, total_successful_SDK_runs=total_runs,
                   patch_cache_closed_restored=True, cleanup_errors=[], poses_sha256=sha(folder/'poses.tum'),
                   model_threads=report['model_threads'], cpu_affinity=report['cpu_affinity'],
                   peak_RSS_MiB=report['peak_RSS_MiB'])
    return report, folder, summary


def trajectory(report, folder, cpu_directory, ground_truth, evaluator):
    estimated = evaluator.read_tum(folder / 'poses.tum')
    cpu_path = cpu_directory / 'cpu_fp32/poses.tum'
    cpu = evaluator.read_tum(cpu_path)
    require(len(estimated) == len(cpu) == 17
            and np.array_equal(estimated[:, 0], np.asarray(report['input_timestamps']))
            and np.array_equal(estimated[:, 0], cpu[:, 0]), 'Saved poses/timestamps differ')
    quality = evaluator.evaluate_tum(estimated, ground_truth, max_difference=.02, offset=0., delta_frames=1)
    reference = evaluator.evaluate_tum(cpu, ground_truth, max_difference=.02, offset=0., delta_frames=1)
    require(quality['association']['matches'] == reference['association']['matches']
            and quality['association']['matched_count'] == 17
            and quality['rpe_pair_policy']['pair_count'] == 16, 'Same-GT pose association differs')
    fields = ('ate_translation', 'rpe_translation', 'rpe_rotation')
    return dict(estimated=quality, original_CPU=reference, CPU_poses_sha256=sha(cpu_path),
                RMSE_change_vs_CPU_percent={k: 100*(quality[k]['rmse']/reference[k]['rmse']-1) for k in fields},
                within_20_percent_RMSE_vs_CPU={k: quality[k]['rmse'] <= 1.2*reference[k]['rmse'] for k in fields})


def difference(report, folder, baseline, baseline_folder):
    require(all(report[k] == baseline[k] for k in ('input_indices', 'input_timestamps', 'image_shape',
                'model_threads', 'cpu_affinity', 'strict_rtol', 'strict_atol', 'reused_cpu_reference')),
            'Baseline and candidate inputs/reference differ')
    compared = {}
    for prefix, shapes in (('pair', PREDICTIONS), ('aggregator', AGGREGATOR)):
        results = {key: [] for key in shapes}
        file_equal = 0
        for i in range(1,17):
            p, q = folder/f'{prefix}-{i:02d}.npz', baseline_folder/f'{prefix}-{i:02d}.npz'
            current, previous = arrays(p, shapes), arrays(q, shapes)
            file_equal += sha(p) == sha(q)
            for key in shapes:
                m = metric(current[key], previous[key])
                m['byte_equal'] = current[key].tobytes() == previous[key].tobytes()
                results[key].append(m)
        compared[prefix] = dict(arrays_compared=80, arrays_byte_equal=sum(v['byte_equal'] for a in results.values() for v in a),
                               NPZ_files_byte_equal=file_equal, field_summary={k: dict(byte_equal_count=sum(v['byte_equal'] for v in values),
                               strict_pass_count=sum(v['strict_pass'] for v in values), max_abs=max(v['max_abs'] for v in values),
                               mean_relative_l2=mean([v['relative_l2'] for v in values])) for k, values in results.items()})
    compared['poses_file_byte_equal'] = sha(folder/'poses.tum') == sha(baseline_folder/'poses.tum')
    return compared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('run-directory', 'cpu-reference-directory', 'output'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--result-sha256', required=True)
    parser.add_argument('--runner-source', type=Path)
    parser.add_argument('--baseline-directory', type=Path)
    parser.add_argument('--baseline-sha256')
    parser.add_argument('--trajectory-evaluator', type=Path, required=True)
    parser.add_argument('--groundtruth', type=Path, required=True)
    parser.add_argument('--control', nargs=2, action='append', default=[], metavar=('FILE', 'SHA256'))
    args = parser.parse_args()
    if (args.baseline_directory is None) != (args.baseline_sha256 is None):
        parser.error('--baseline-directory and --baseline-sha256 must be paired')
    report, folder, summary = inspect(args.run_directory, args.result_sha256, args.cpu_reference_directory, args.runner_source)
    require(sha(args.trajectory_evaluator) == EVALUATOR_SHA and sha(args.groundtruth) == GT_SHA,
            'Frozen NumPy evaluator/ground truth differs')
    spec = importlib.util.spec_from_file_location('_v71_saved_TUM_evaluator', args.trajectory_evaluator)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    ground_truth = evaluator.read_tum(args.groundtruth)
    summary['trajectory'] = trajectory(report, folder, args.cpu_reference_directory, ground_truth, evaluator)
    output = dict(schema='v71_original_VGGT_saved_complete_array_audit', status='passed_saved_audit',
                  auditor_source_sha256=sha(__file__), numpy_version=np.__version__, run=summary,
                  evaluator_source_sha256=EVALUATOR_SHA, groundtruth_sha256=GT_SHA,
                  scope='Saved NPZ/JSON/TUM/source only; no model, ONNX, ORT, SDK or board execution.',
                  limits=['Prediction NPZ SHA values are frozen by this audit; the run receipt directly binds the intermediate NPZ SHA and CPU reference SHA.',
                          'Strict CPU comparisons use the recorded NumPy allclose rule; scalar relative-L2 reduction tolerance does not relax its elementwise gate.',
                          'Intermediate copies and NPZ writes occur after the timed service; timings are offline service, not fresh-input latency.',
                          'Aggregator phase includes DINO and guard work; do not sum nested phases twice.',
                          'Peak RSS is historical process accounting, not independent physical RAM or an 8GB validation.',
                          'Camera release records cover only the final four refinements; DINO records owned conversion counts, not a global SDK output-release counter.',
                          'ATE uses separately fitted positive-scale Sim3; RPE uses the same fitted scale and adjacent matched poses, not a one-second interval.',
                          'These fixed offline 17-pose comparisons do not establish live-camera quality or realtime acceptance.'],
                  candidate_accepted=False, quality_accepted=False, realtime_validated=False)
    if args.baseline_directory is not None:
        previous, old_folder, old_summary = inspect(args.baseline_directory, args.baseline_sha256, args.cpu_reference_directory)
        old_summary['trajectory'] = trajectory(previous, old_folder, args.cpu_reference_directory, ground_truth, evaluator)
        output.update(baseline=old_summary, difference=difference(report, folder, previous, old_folder))
        output['difference']['timing_reduction_percent'] = {key: 100*(1-summary['timing']['complete_service'][key]/old_summary['timing']['complete_service'][key])
            for key in ('mean_seconds', 'excluding_first_mean_seconds')}
    output['controls'] = []
    for path, digest in args.control:
        path = Path(path)
        require(sha(path) == digest, 'Control SHA differs')
        control = json.loads(path.read_text())
        require(control['status'] == 'completed' and control['returncode'] == 0
                and control['cleanup_errors'] == [] and control['frequency_written'] is False
                and control['samples'] and all(s['npu_hz'] == 300000000 for s in control['samples']),
                'Control completion/frequency differs')
        temperatures = [v for s in control['samples'] for v in s['temperatures_c'].values()]
        require(all(math.isfinite(v) for v in temperatures), 'Nonfinite temperature sample')
        output['controls'].append(dict(file=path.name, sha256=digest, samples=len(control['samples']),
            sampled_NPU_hz=300000000, no_clock_write=True, maximum_sampled_temperature_c=max(temperatures)))
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=output['status'], result_sha256=args.result_sha256,
                         prediction_arrays_checked=80, aggregator_arrays_checked=80,
                         original_CPU_strict_pass=summary['original_CPU_strict_pass'],
                         contexts=summary['total_contexts_created_destroyed'], output_sha256=sha(args.output)), allow_nan=False))


if __name__ == '__main__':
    main()
