#!/usr/bin/env python3
"""Saved NumPy/JSON/TUM only: original CPU, baseline INT8 aggregator/FP16 camera and INT8 camera candidate.

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
    loaded = [load_run(args.cpu, True), load_run(args.baseline), load_run(args.candidate)]
    cpu, baseline, candidate = loaded
    if (candidate[0].get('calibrated_retained_aggregator') is not True
            or candidate[0].get('internal_precision_changed') is not True
            or candidate[0].get('requested_aggregator_quantized_dtype') != 'w8a8'):
        raise ValueError('Candidate must identify the existing INT8 aggregator precision')
    if (baseline[0].get('calibrated_retained_aggregator') is not True
            or candidate[0].get('calibrated_camera_head') is not True
            or candidate[0].get('requested_camera_quantized_dtype') != 'w8a8'
            or candidate[0].get('original_camera_ONNX_unchanged') is not True
            or candidate[0].get('camera_compile_receipt_sha256') != '4d4094c76570dd497af3553b6dfcbc768ebd65147717db7aca8b20bc4c0efc89'
            or baseline[0].get('calibrated_camera_head', False) is not False):
        raise ValueError('Expected baseline FP16 camera and explicitly compiled INT8 camera candidate')
    for original, _, _ in loaded[1:]:
        if original['input_timestamps'] != cpu[0]['input_timestamps']:
            raise ValueError('Original CPU and NPU timestamps differ')
    identity_fields = ('image_shape', 'input_indices', 'input_timestamps', 'model_threads', 'cpu_affinity',
        'script_sha256', 'manifest_sha256', 'dino_manifest_sha256', 'depth_manifest_sha256',
        'camera_split4', 'paired_aggregator', 'retained_aggregator', 'grouped_dino',
        'checkpoint_pageout', 'half_io_library_sha256', 'reused_cpu_reference',
        'quantized_manifest_sha256')
    if any(baseline[0][key] != candidate[0][key] for key in identity_fields):
        raise ValueError('Baseline/candidate complete run identity/configuration differs')
    old_sources, new_sources = baseline[0]['runtime_source_sha256'], candidate[0]['runtime_source_sha256']
    if (any(new_sources.get(k) != v for k, v in old_sources.items())
            or set(new_sources) - set(old_sources) != {'private/rknn_calibrated_camera.py'}
            or new_sources['private/rknn_calibrated_camera.py'] != '0a3db35965ffc7d17ff0855a5040ecb0f22f8be23a88605ec1a1e7793ad236ec'):
        raise ValueError('Only the explicitly frozen camera adapter may be added to runtime source identities')
    report = dict(schema='v82_original_VGGT_saved_complete_camera_INT8_quality_diagnostic',
        status='saved_data_audited_not_quality_accepted', original_model_only=True,
        candidate_accepted=False, quality_accepted=False, realtime_validated=False,
        source_sha256=sha(__file__), strict_rtol=1e-4, strict_atol=1e-5,
        input_indices=INDICES, input_timestamps=cpu[0]['input_timestamps'],
        result_sha256={key: sha(root / 'result.json') for key, root in
            (('cpu', args.cpu), ('baseline', args.baseline), ('candidate', args.candidate))},
        helper_sha256=dict(trajectory=sha(trajectory_path), compare=sha(compare_path)),
        original_run_identity_fields_equal=True, array_files=[],
        aggregator_quantized_manifest_sha256=baseline[0]['quantized_manifest_sha256'],
        original_camera_ONNX_unchanged=True,
        camera_compile_receipt_sha256=candidate[0]['camera_compile_receipt_sha256'],
        camera_adapter_source_sha256=new_sources['private/rknn_calibrated_camera.py'],
        original_CPU_top_cleanup_field_recorded='cleanup_errors' in cpu[0],
        predictions={k: {} for k in ('baseline_vs_cpu', 'candidate_vs_cpu', 'candidate_vs_baseline')},
        aggregator={'candidate_vs_baseline': {}}, trajectories={}, timing={}, resources={},
        aggregator_vs_original_CPU_available=False)
    shapes = {key: ((2, 53, 1024) if key == 'target_tokens' else (1, 2, 53, 2048)) for key in AGG_KEYS}
    for index in range(1, 17):
        arrays = [owned_arrays(folder / f'pair-{index:02d}.npz', PRED_KEYS, SHAPES, report=report)
                  for _, _, folder in loaded]
        for key in PRED_KEYS:
            for name, actual, expected in (('baseline_vs_cpu', arrays[1][key], arrays[0][key]),
                                           ('candidate_vs_cpu', arrays[2][key], arrays[0][key]),
                                           ('candidate_vs_baseline', arrays[2][key], arrays[1][key])):
                accumulate(report['predictions'][name], key, actual, expected, compare)
        left, right = [owned_arrays(row[2] / f'aggregator-{index:02d}.npz', AGG_KEYS, shapes, report=report)
                       for row in (baseline, candidate)]
        for key in AGG_KEYS:
            accumulate(report['aggregator']['candidate_vs_baseline'], key, right[key], left[key], compare)
    for value in report['predictions'].values():
        finalize(value)
    finalize(report['aggregator']['candidate_vs_baseline'])
    report['unchanged_compute_outputs'] = dict(aggregator_byte_equal_arrays=sum(
        r['byte_equal_arrays'] for r in report['aggregator']['candidate_vs_baseline'].values()),
        depth_and_confidence_byte_equal_arrays=sum(report['predictions']['candidate_vs_baseline'][k]['byte_equal_arrays']
            for k in ('depth', 'depth_conf')))
    if report['unchanged_compute_outputs'] != dict(aggregator_byte_equal_arrays=80, depth_and_confidence_byte_equal_arrays=32):
        raise ValueError('Camera-only candidate changed an aggregator/depth/confidence output')
    if any(r['nonfinite_actual_elements'] or r['nonfinite_reference_elements']
            for comparison in report['predictions'].values() for r in comparison.values()):
        raise ValueError('Nonfinite prediction array in matched saved runs')
    gt, poses = trajectory.read_tum(args.groundtruth), []
    for key, (run, branch, folder) in zip(('cpu', 'baseline', 'candidate'), loaded):
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
            report['timing'][key]['recorded_peak_RSS_MiB'] = run['peak_RSS_MiB']
            report['timing'][key]['model_phase_mean_seconds'] = {phase: statistics.mean(
                item['model_phase_seconds'][phase] for item in branch['updates'])
                for phase in ('DINO', 'aggregator_including_DINO', 'camera_head', 'depth_head')}
            report['timing'][key]['camera_adapter_mean_seconds'] = branch['camera_rknn']['head_forward_seconds'] / 16
            report['timing'][key]['camera_SDK_IO_mean_seconds'] = branch['camera_rknn']['sdk_io_seconds'] / 16
        if key != 'cpu':
            resources = {}
            for name in ('rknn', 'dino_rknn', 'camera_rknn', 'depth_rknn'):
                info = branch[name]
                resources[name] = {field: info[field] for field in
                    ('closed', 'live_contexts', 'contexts_created', 'contexts_destroyed', 'cleanup_errors')}
                if not info['closed'] or info['live_contexts'] or info['cleanup_errors'] or info['contexts_created'] != info['contexts_destroyed']:
                    raise ValueError('Complete original backend cleanup failed')
            report['resources'][key] = resources
            expected = dict(rknn=4, dino_rknn=6, camera_rknn=2 if key == 'candidate' else 1, depth_rknn=1)
            if any(resources[name]['contexts_created'] != count for name, count in expected.items()):
                raise ValueError('Expected 13 candidate initializations including the temporary FP16 camera validation, versus 12 baseline')
            camera = branch['camera_rknn']
            if [camera[k] for k in ('completed_head_calls', 'completed_refinements', 'npu_run_api_call_count')] != [16, 64, 64]:
                raise ValueError('All 16 original four-step camera forwards must complete')
            report['resources'][key]['total_contexts_created_and_destroyed'] = sum(expected.values())
    report['trajectories']['baseline_vs_cpu'] = trajectory.evaluate_tum(poses[1], poses[0])
    report['trajectories']['candidate_vs_cpu'] = trajectory.evaluate_tum(poses[2], poses[0])
    report['trajectories']['candidate_vs_baseline'] = trajectory.evaluate_tum(poses[2], poses[1])
    report['trajectory_metric_change_percent'] = {reference: {metric: {
        field: 100 * (report['trajectories']['candidate_vs_GT'][metric][field]
            / report['trajectories'][reference + '_vs_GT'][metric][field] - 1)
        for field in ('rmse', 'maximum')} for metric in ('ate_translation', 'rpe_translation', 'rpe_rotation')}
        for reference in ('cpu', 'baseline')}
    report['groundtruth_sha256'] = sha(args.groundtruth)
    if args.wrapper:
        wrapper = json.loads(args.wrapper.read_text())
        if (wrapper['status'] != 'completed_original_entrypoint_pending_saved_output_review'
                or wrapper['cleanup_errors'] or wrapper['camera_installation_failures']
                or any(wrapper[k] is not True for k in ('camera_resources_closed', 'installers_restored', 'sources_unchanged'))
                or wrapper['camera_compile_receipt_sha256'] != report['camera_compile_receipt_sha256']
                or wrapper['camera_adapter_source_sha256'] != report['camera_adapter_source_sha256']
                or wrapper['quantized_manifest_sha256'] != report['aggregator_quantized_manifest_sha256']):
            raise ValueError('Scoped camera wrapper did not restore its installers/resources')
        report['wrapper'] = dict(receipt_sha256=sha(args.wrapper), script_sha256=wrapper['script_sha256'],
            installer_restore=True, camera_resources_closed=True, cleanup_errors=[], immutable_execution=wrapper['immutable_execution'])
    if args.control:
        control = json.loads(args.control.read_text())
        observations = control['observations']
        if (control['child_returncode'] != 0 or control['restore_errors']
                or control['frequency_governor_limits_restored'] is not True
                or control['requested_frequency_hz'] != 1000000000 or not observations
                or any(o['actual_frequency_hz'] != 1000000000 for o in observations)
                or control['restored'] != control['baseline']
                or control['restored']['cur_freq'] != '300000000'):
            raise ValueError('Bounded 1GHz complete run frequency/restoration failed')
        maximum = max(v for o in observations for v in o['temperatures_c'].values())
        if maximum >= control['temperature_limit_c']:
            raise ValueError('Saved temperature exceeded the controller stop limit')
        report['controller'] = dict(receipt_sha256=sha(args.control), child_returncode=0,
            observations=len(observations), sampled_frequency_hz=1000000000,
            sampled_max_temperature_c=maximum, restored_frequency_hz=300000000,
            frequency_governor_limits_and_raw_request_restored=True, restore_errors=[])
    report['metric_scale_validated'] = False
    report['limits'] = ['Sim3 scale is fitted and reused for adjacent matched-pose RPE; no metric-scale acceptance.',
        'This measures saved offline 16-pair outputs; it does not establish fresh-camera rate or 100-ms latency.',
        'Original FP32 strict failures exclude a numerical-equivalence claim; they do not independently replace the user ATE/RPE 10-20% quality criterion. Broad quality acceptance remains unverified.',
        'Original v53 CPU aggregator arrays were not saved; internal tensors are compared to the same baseline run.',
        'Baseline uses INT8 aggregator and FP16 camera; candidate changes only the camera to requested W8A8. DINO/depth stay FP16.',
        'Camera has two contexts created/destroyed because the original FP16 constructor validation precedes its replacement; only one camera context serves inference.',
        'Phase DINO is included in aggregator_including_DINO; these phase means must not be added twice.',
        'Camera adapter and outer model observer timing scopes are reported separately.',
        'Worst adjacent RPE is reported separately from aggregate RMSE; a small RMSE change does not establish all-pair quality.']
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--cpu', type=Path, default=Path('/dev/shm/edge-vggt-v53/complete-ab-r1'))
    p.add_argument('--groundtruth', type=Path, default=Path('/dev/shm/edge-vggt-v53/groundtruth.txt'))
    p.add_argument('--public', type=Path, default=PUBLIC)
    p.add_argument('--wrapper', type=Path)
    p.add_argument('--control', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    try:
        report = audit(args)
    except Exception as error:
        report = dict(schema='v82_original_VGGT_saved_complete_camera_INT8_quality_diagnostic',
            status='failed_saved_audit', candidate_accepted=False,
            error=type(error).__name__ + ': ' + str(error), source_sha256=sha(__file__))
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        raise
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], output_sha256=sha(args.output)), allow_nan=False))


if __name__ == '__main__':
    main()
