#!/usr/bin/env python3
"""Audit only saved CameraHead probe NPZ/JSON; no model, Torch, ORT or SDK."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE_SHA = 'a88c2d07a26d2ac6abb1f26a9e707e199a0d2572ef82b1d05ee62162ee6248a5'
EXPORT_SHA = '8ccf7ff593346962766b4eae6da7eb95ce43eeaa3183a8fc0e82bf134c172503'
SCHEDULE = 'AB' + 'ABBA' * 3


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def saved(root, item):
    root, relative = Path(root).resolve(strict=True), Path(item['file'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe saved artifact')
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root) and path.stat().st_size == item['bytes']
        and sha(path) == item['sha256'], 'Saved artifact identity differs: ' + str(path))
    return path


def comparison(value, reference):
    require(value.shape == reference.shape and value.dtype == reference.dtype == np.float32,
        'Invalid saved numeric comparison')
    ref = reference.astype(np.float64)
    difference = np.abs(value.astype(np.float64) - ref)
    outside = int((difference > 1e-5 + 1e-4 * np.abs(ref)).sum())
    return dict(bitwise_equal=value.tobytes() == reference.tobytes(), strict_pass=outside == 0,
        outside=outside, elements=value.size, max_abs=float(difference.max()),
        relative_l2=float(np.linalg.norm(difference) / max(np.linalg.norm(ref), np.finfo(np.float64).tiny)))


def check_comparison(actual, recorded):
    for key, value in actual.items():
        if isinstance(value, float):
            require(math.isfinite(recorded[key]) and math.isclose(value, recorded[key], rel_tol=1e-10, abs_tol=1e-12),
                'Saved numeric metric differs: ' + key)
        else:
            require(value == recorded[key], 'Saved numeric metric differs: ' + key)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result-dir', type=Path, required=True)
    p.add_argument('--result-sha256', required=True)
    p.add_argument('--control-json', type=Path, required=True)
    p.add_argument('--probe-source', type=Path, default=ROOT / 'probe_camera_int8_board.py')
    p.add_argument('--reference-npz', type=Path, help='Optional frozen original v16 step-iterations.npz, for independent CPU metric recomputation')
    p.add_argument('--reference-export-json', type=Path,
        default=ROOT.parent / 'edge-vggt-v50/public-snapshot/deploy/rk3588/validation/vggt-v16-camera-step-export112-r1.json')
    p.add_argument('--output', type=Path, default=ROOT / 'camera-int8-saved-audit.json')
    a = p.parse_args(argv)
    require(not a.output.exists(), 'Use a new audit output file')
    result_path = a.result_dir / 'result.json'
    require(sha(result_path) == a.result_sha256 and sha(a.probe_source) == SOURCE_SHA, 'Frozen result/source pin differs')
    r = json.loads(result_path.read_text())
    c = json.loads(a.control_json.read_text())
    require(r['schema'] == 'v82_original_CameraHead_split4_FP16_INT8_probe'
        and r['status'] == 'completed_original_full_CameraHead_component_AB_not_quality_accepted'
        and r['script_sha256'] == SOURCE_SHA and r['original_model_only']
        and not r['original_Torch_model_constructed'] and not r['candidate_accepted']
        and not r['quality_accepted'] and not r['full_model_or_trajectory_or_realtime_validated']
        and r['init_flags'] == 0 and r['core_mask'] == 7 and r['normal_benchmark_timing']
        and r['schedule'] == SCHEDULE and r['hidden_SDK_warmup_calls'] == 0
        and r['first_per_route'] == 1 and r['warm_per_route'] == 6
        and r['required_outer_iterations'] == r['trunk_blocks_per_step'] == 4
        and r['expected_npu_hz'] == 1000000000 and not r['clock_writes']
        and r['strict_rtol'] == 1e-4 and r['strict_atol'] == 1e-5,
        'Original component/normal benchmark scope differs')
    require(r['contexts_created'] == r['contexts_destroyed'] == 2 and r['live_contexts'] == 0
        and r['outputs_acquired'] == r['outputs_released'] == 56
        and r['sdk_run_attempts'] == r['sdk_runs_succeeded'] == r['npu_run_api_call_count'] == 56
        and r['npu_run_api_called'] and r['lifecycle_verified'] and not r['cleanup_errors']
        and r['source_and_inputs_unchanged'] and r['affinity_restored'] and r['environment_restored'],
        'SDK lifecycle/closure/restoration differs')
    require(r['original_camera_parameters_verified_by_frozen_lineage'] == 69
        and r['original_reference_activation_contract_checks'] == 4
        and r['activation'] == dict(source_sha256='428bdb087f457b809a66e22d7d01147c1cb4dab43bb0d98bc61c7f05d8383a1e',
            trans_act='linear', quat_act='linear', fl_act='relu'), 'Original parameter/activation contract differs')
    require(len(r['calls']) == 14 and ''.join(x['route'] for x in r['calls']) == SCHEDULE,
        'AB + ABBA*3 saved schedule differs')
    require(len(r['graphs']) == 2 and [g['route'] for g in r['graphs']] == ['A', 'B'], 'Two original routes required')
    for graph in r['graphs']:
        require(len(graph['inputs']) == 2 and graph['output']['name'] == 'pose_delta'
            and graph['output']['shape'] == [1, 2, 9] and graph['output']['index'] == 0,
            'Original CameraHead output ABI differs')
        require([(x['index'], x['name'], x['shape']) for x in graph['inputs']]
            == [(0, 'camera_tokens', [1, 2, 2048]), (1, 'conditioning_pose', [1, 2, 9])],
            'Original CameraHead input ABI differs')
        for d in graph['inputs'] + [graph['output']]:
            n = math.prod(d['shape']); integer = d['type'] in (2, 3)
            identity = d['type'] == 1 and d['quantization_type'] == 2 and d['zero_point'] == 0 and d['scale'] == 1.
            require(d['format'] == 3 and d['pass_through'] == 0 and d['n_elems'] == n
                and d['type'] in (0, 1, 2, 3) and d['size'] == n * {0:4, 1:2, 2:1, 3:1}[d['type']]
                and math.isfinite(d['scale'])
                and (not integer or (d['quantization_type'] == 2 and d['scale'] > 0
                    and ((-128 <= d['zero_point'] <= 127) if d['type'] == 2 else (0 <= d['zero_point'] <= 255))))
                and (integer or d['quantization_type'] == 0 or identity), 'Finite floating/affine logical ABI differs')
    references = None
    reference_sha = None
    seed_sha = None
    if a.reference_export_json.is_file():
        require(sha(a.reference_export_json) == EXPORT_SHA, 'Original export receipt pin differs')
        export = json.loads(a.reference_export_json.read_text())
        seed_sha = export['artifacts']['conditioning-seed.npy']['tensor_sha256']
        if a.reference_npz is not None:
            item = export['artifacts']['step-iterations.npz']
            require(a.reference_npz.stat().st_size == item['bytes'] and sha(a.reference_npz) == item['sha256'],
                'Original CPU reference NPZ pin differs')
            reference_sha = sha(a.reference_npz)
            with np.load(a.reference_npz, allow_pickle=False) as archive:
                references = {name: archive[name] for name in archive.files}
            require(set(references) == {f'{name}_{i}' for name in ('conditioning', 'delta', 'preactivation', 'activation') for i in range(4)},
                'All original CPU refinement references required')
            for name, value in references.items():
                meta = item['tensors'][name]
                require(value.shape == (1, 2, 9) and value.dtype == np.float32 and value.flags.c_contiguous
                    and np.isfinite(value).all() and hashlib.sha256(value.tobytes()).hexdigest() == meta['sha256'],
                    'Original CPU reference raw hash/shape/dtype/finite differs')
            references['final_pose'] = references['activation_3']
    else:
        require(a.reference_npz is None, 'Original export receipt required for CPU reference verification')
    first, previous, data, ledger, platforms = {}, {}, [], [], []
    expected_keys = {f'{name}_{i}' for name in ('conditioning', 'delta', 'preactivation', 'activation') for i in range(4)} | {'final_pose'}
    count = 0
    for index, call in enumerate(r['calls']):
        route = call['route']
        require(call['index'] == index and len(call['steps']) == 4 and call['first_for_route'] == (route not in first),
            'Per-route first/iteration metadata differs')
        path = saved(a.result_dir, call['artifact'])
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == expected_keys and set(call['artifact']['tensors']) == expected_keys,
                'Exactly four refinements plus final pose required')
            values = {name: archive[name] for name in archive.files}
        for name, value in values.items():
            meta = call['artifact']['tensors'][name]
            require(value.shape == (1, 2, 9) and value.dtype == np.float32 and value.flags.c_contiguous
                and np.isfinite(value).all() and meta['shape'] == [1, 2, 9] and meta['dtype'] == 'float32'
                and meta['finite'] and meta['bytes'] == value.nbytes
                and meta['sha256'] == hashlib.sha256(value.tobytes()).hexdigest(),
                'Saved tensor raw hash/shape/dtype/finite differs')
            count += 1
        if seed_sha is not None:
            require(hashlib.sha256(values['conditioning_0'].tobytes()).hexdigest() == seed_sha,
                'First conditioning differs from original learned empty-pose seed')
        for i in range(4):
            conditioning, delta, pred, act = [values[f'{name}_{i}'] for name in ('conditioning', 'delta', 'preactivation', 'activation')]
            if i:
                require(conditioning.tobytes() == values[f'preactivation_{i-1}'].tobytes(),
                    'SDK recursively consumed an activated/different prediction')
                expected = np.add(values[f'preactivation_{i-1}'], delta, dtype=np.float32)
            else:
                expected = delta
            require(pred.tobytes() == expected.tobytes(), 'Original FP32 accumulation differs')
            expected_act = pred.copy(); expected_act[..., 7:][expected_act[..., 7:] < 0] = np.float32(0.)
            require(act.tobytes() == expected_act.tobytes(), 'Original linear/linear/ReLU saved activation differs')
            step = call['steps'][i]
            require(step['iteration'] == i and step['synchronous'] and step['input_type'] == 'FP32'
                and step['input_pass_through'] == 0 and step['want_float'] == 1
                and step['output_bytes'] == 72 and step['output_buffer_released'], 'Owned synchronous SDK step receipt differs')
            require(all(math.isfinite(v) and v >= 0 for k, v in step.items() if k.endswith('_seconds')),
                'Invalid per-step SDK timing')
        require(values['final_pose'].tobytes() == values['activation_3'].tobytes(), 'Final pose is not the fourth original activation')
        require(math.isclose(call['sdk_io_seconds'], sum(s['io_total_seconds'] for s in call['steps']), rel_tol=1e-12, abs_tol=1e-12)
            and math.isfinite(call['complete_head_seconds']) and call['complete_head_seconds'] >= call['sdk_io_seconds'],
            'Whole-head/SDK timing accounting differs')
        if route not in first:
            first[route] = values
        for name, value in values.items():
            check_comparison(comparison(value, first[route][name]), call['vs_first_same_route'][name])
            if references is not None:
                check_comparison(comparison(value, references[name]), call['vs_original_CPU'][name])
            if route == 'B':
                check_comparison(comparison(value, first['A'][name]), call['vs_first_A'][name])
        if route in previous:
            equal = all(values[name].tobytes() == previous[route][name].tobytes() for name in expected_keys)
            require(equal == call['bitwise_equal_previous_same_route'], 'Same-route stability flag differs')
        previous[route] = values
        data.append(values)
        platforms.extend((call['platform_before'], call['platform_after']))
        ledger.append(dict(call=index, route=route, NPZ_sha256=call['artifact']['sha256'], tensors=17,
            final_raw_sha256=call['artifact']['tensors']['final_pose']['sha256']))
    require(count == 238, '14 NPZ / 238 saved tensors required')
    cross = {name: comparison(value, first['A'][name]) for name, value in first['B'].items()}
    for name, actual in cross.items(): check_comparison(actual, r['first_B_vs_first_A'][name])
    summaries = {}
    for route in 'AB':
        calls = [x for x in r['calls'] if x['route'] == route]
        warm = [x for x in calls if not x['first_for_route']]
        require(len(calls) == 7 and len(warm) == 6 and r['summary_seconds'][route]['calls'] == 7
            and r['summary_seconds'][route]['warm_count'] == 6, 'Seven calls / six warm per route required')
        mean = float(np.mean([x['complete_head_seconds'] for x in warm])); sdk_mean = float(np.mean([x['sdk_io_seconds'] for x in warm]))
        require(math.isclose(mean, r['summary_seconds'][route]['warm_mean'], rel_tol=1e-12, abs_tol=1e-12)
            and math.isclose(sdk_mean, r['summary_seconds'][route]['warm_sdk_io_mean'], rel_tol=1e-12, abs_tol=1e-12)
            and calls[0]['complete_head_seconds'] == r['summary_seconds'][route]['first'], 'Warm timing recomputation differs')
        stable = all(all(data[x['index']][name].tobytes() == first[route][name].tobytes() for name in expected_keys) for x in calls)
        summaries[route] = dict(calls=7, warm_count=6, first_seconds=calls[0]['complete_head_seconds'],
            warm_head_mean_seconds=mean, warm_SDK_mean_seconds=sdk_mean, warm_head_min_seconds=min(x['complete_head_seconds'] for x in warm),
            warm_head_max_seconds=max(x['complete_head_seconds'] for x in warm), all_17_outputs_bitwise_stable=stable)
    observations = c['observations']
    temperatures = [v for item in observations + platforms for v in item['temperatures_c'].values()]
    require(c['status'] == 'worker_completed_pending_result_review' and c['child_returncode'] == 0
        and c['requested_frequency_hz'] == 1000000000 and c['temperature_limit_c'] == 75.
        and c['timeout_seconds'] <= 180 and not c['restore_errors'] and c['frequency_governor_limits_restored']
        and c['baseline'] == c['restored'] and c['restored']['cur_freq'] == c['restored']['userspace/set_freq'] == '300000000'
        and not c['userspace_request_was_undefined'] and observations
        and all(x['actual_frequency_hz'] == 1000000000 for x in observations)
        and all(x['npu_hz'] == 1000000000 and x['governor'] == 'userspace' for x in platforms)
        and all(math.isfinite(t) and t < 75 for t in temperatures), 'Bounded frequency/temperature/restoration differs')
    report = dict(schema='v82_original_CameraHead_FP16_INT8_saved_audit', status='passed_saved_artifact_math_lifecycle_and_controller_audit',
        original_model_only=True, actual_model_Torch_ORT_SDK_or_board_execution_by_auditor=False,
        auditor_sha256=sha(__file__), probe_source_sha256=SOURCE_SHA, probe_result_sha256=a.result_sha256,
        controller_sha256=sha(a.control_json), original_reference_NPZ_sha256=reference_sha,
        original_CPU_comparison_recomputed=references is not None, original_empty_seed_hash_independently_checked=seed_sha is not None,
        NPZ_files_checked=14, tensor_arrays_checked=238, file_and_raw_hash_shape_dtype_finite_passed=True,
        original_unactivated_FP32_recurrence_and_linear_linear_ReLU_passed=True,
        contexts_created=2, contexts_destroyed=2, live_contexts=0, SDK_steps=56, outputs_acquired=56, outputs_released=56,
        source_and_inputs_unchanged=True, affinity_and_process_environment_restored=True, cleanup_errors=[],
        controls=dict(observations=len(observations), call_before_after_samples=28, all_observed_NPU_hz=1000000000,
            max_temperature_c=max(temperatures), worker_elapsed_seconds=c['seconds'], baseline_equals_restored=True,
            restored=c['restored'], restore_errors=[]), summaries=summaries,
        B_over_A_warm_head_ratio=summaries['B']['warm_head_mean_seconds'] / summaries['A']['warm_head_mean_seconds'],
        first_B_vs_first_A_final_pose=cross['final_pose'],
        first_routes_vs_original_CPU_final_pose={route: next(x for x in r['calls'] if x['route'] == route)['vs_original_CPU']['final_pose'] for route in 'AB'},
        artifact_ledger=ledger,
        limits=['Only the frozen v15 real-token CameraHead component is compared; this does not establish full-model/trajectory/map/realtime quality.',
            'Each route completes four refinements, and later deltas use that route\'s own accumulated unactivated output; original fixed-conditioning deltas do not independently isolate single-step quantization error.',
            'Original CPU metric recomputation is conditional on the separately pinned original reference NPZ; otherwise those CPU metrics are recorded from the frozen board receipt, not independently recalculated here.',
            'SDK DataType/quantization labels do not prove every internal operator integer math; flags0 timings include complete four-step API ownership and CPU head_act, excluding file hashes/saves/comparisons.',
            'Short seven-call/six-warm measurements do not establish 8GB, 30-minute stability, 20Hz fresh pose, P95 input-to-pose, camera streaming or quality acceptance.'])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], output=str(a.output), sha256=sha(a.output),
        original_CPU_comparison_recomputed=references is not None, summaries=summaries)))


if __name__ == '__main__':
    main()
