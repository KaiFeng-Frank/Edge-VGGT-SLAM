"""Read saved NumPy outputs only; no Torch, ORT, SDK or device execution."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

RESULT_SHA = '4b252fbbb7715d41c3f5f8f4e697ec037797e33870960b49d68ffb7d58f34d24'
CAPTURE_SHA = '3b4eb0b24d320699dd7d198fd3158454a3043ed7244a0290b267cbfc450e8fee'
SHAPES = {'frame': (2, 53, 1024), 'global_output': (1, 106, 1024)}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_sha(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def artifact(root, row):
    path = (root / row['file']).resolve(strict=True)
    require(path.is_relative_to(root.resolve()), 'Artifact escapes saved directory')
    require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
            'Artifact size/SHA mismatch: ' + row['file'])
    return path


def compare(value, reference):
    require(value.shape == reference.shape and value.dtype == reference.dtype,
            'Comparison shape/dtype mismatch')
    delta = value.astype(np.float64) - reference.astype(np.float64)
    failed = np.abs(delta) > 1e-5 + 1e-4 * np.abs(reference.astype(np.float64))
    denominator = float(np.linalg.norm(reference.astype(np.float64)))
    error = float(np.linalg.norm(delta))
    return dict(bitwise_equal=value.tobytes() == reference.tobytes(),
                max_abs=float(np.abs(delta).max()),
                relative_l2=error / denominator if denominator else (0. if not error else None),
                strict_rtol=1e-4, strict_atol=1e-5, failed_elements=int(failed.sum()),
                strict_pass=not bool(failed.any()))


def verify_metrics(actual, saved):
    require(actual.keys() == saved.keys(), 'Metric keys mismatch')
    for key, value in actual.items():
        if type(value) is float:
            require(math.isclose(value, saved[key], rel_tol=1e-12, abs_tol=1e-14),
                    'Float summary mismatch: ' + key)
        else:
            require(value == saved[key], 'Exact metric mismatch: ' + key)


def audit(directory, capture_directory):
    result_path, capture_path = directory / 'result.json', capture_directory / 'capture.json'
    require(sha(result_path) == RESULT_SHA and sha(capture_path) == CAPTURE_SHA, 'Receipt pin mismatch')
    report, capture = json.loads(result_path.read_text()), json.loads(capture_path.read_text())
    require(report['schema'] == 'v83_original_pair_custom_GPU_attention_AB'
            and report['status'] == 'completed_component_not_quality_accepted'
            and report['candidate_accepted'] is False, 'Actual diagnostic identity differs')
    require(report['schedule'] == list('ABBAABBA') and len(report['calls']) == 16, 'Call order/count mismatch')
    references, reference_artifacts = {}, []
    for case in capture['cases']:
        refs = []
        for role, name in (('frame', 'frame'), ('global', 'global_output')):
            row = case['roles'][role]['artifacts']['block_output']
            value = np.load(artifact(capture_directory, row), allow_pickle=False)
            require(value.shape == SHAPES[name] and value.dtype == np.float32 and np.isfinite(value).all()
                    and raw_sha(value) == row['raw_sha256'] and value.nbytes == row['tensor_bytes'],
                    'Original saved CPU reference differs')
            refs.append(value)
            reference_artifacts.append(dict(case=case['tag'], role=role, **row))
        references[case['tag']] = refs
    require(list(references) == ['v53', 'v56-calib00'], 'Original input order differs')
    first, rows, cells, thermal, frequencies = {}, [], {}, [], []
    for index, call in enumerate(report['calls']):
        case, sequence, route = call['case'], call['sequence'], call['route']
        require(case == list(references)[index // 8] and sequence == index % 8
                and route == report['schedule'][sequence], 'Actual call order differs')
        path = artifact(directory, call['saved_output'])
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == set(SHAPES), 'Saved NPZ keys differ')
            values = [archive[name].copy() for name in SHAPES]
        for name, value in zip(SHAPES, values):
            require(value.shape == SHAPES[name] and value.dtype == np.float32
                    and value.flags.c_contiguous and np.isfinite(value).all(), 'Saved output is malformed')
        key = (case, route)
        is_first = key not in first
        require(call['first'] is is_first, 'First-call exclusion differs')
        first.setdefault(key, values)
        cpu = [compare(x, y) for x, y in zip(values, references[case])]
        baseline = [compare(x, y) for x, y in zip(values, first[(case, 'A')])]
        repeat = [compare(x, y) for x, y in zip(values, first[key])]
        for actual, saved in ((cpu, call['original_CPU']), (baseline, call['same_input_first_A']),
                              (repeat, call['repeat_first_route'])):
            require(len(saved) == 2, 'Output comparison count differs')
            for a, s in zip(actual, saved):
                verify_metrics(a, s)
        for name, value, error in zip(SHAPES, values, cpu):
            cells.setdefault((case, route, name), []).append(dict(raw_sha256=raw_sha(value), **error))
        for when in ('before', 'after'):
            frequencies.append(call[when]['actual_frequency_hz'])
            thermal.extend(call[when]['temperatures_c'].values())
        require(call['timing']['outputs_released'] is True and call['timing']['output_count'] == 2
                and call['timing']['input_api_type'] == 1 and call['timing']['output_want_float'] == 0,
                'Owned half I/O route differs')
        rows.append(dict(case=case, sequence=sequence, route=route, first=is_first,
                         artifact=call['saved_output'], raw_sha256=[raw_sha(x) for x in values],
                         original_CPU=cpu, repeat_first_route=repeat))
    summaries = {}
    for route in 'AB':
        times = [c['timing']['route_seconds'] for c in report['calls'] if c['route'] == route and not c['first']]
        require(len(times) == 6, 'Warm denominator must be six for each route')
        summaries[route] = dict(warm_calls=6, mean_seconds=float(np.mean(times)),
                                p95_seconds=float(np.percentile(times, 95)))
        verify_metrics(summaries[route], report['summary'][route])
    require(report['contexts_created'] == report['contexts_destroyed'] == 2 and report['live_contexts'] == 0
            and report['cleanup_errors'] == [] and report['affinity_restored'] is True
            and report['source_and_inputs_unchanged'] is True, 'SDK/identity cleanup differs')
    require(report['counters'] == dict(sdk_run_attempts=16, sdk_runs_succeeded=16,
            outputs_acquired=32, outputs_released=32), 'SDK run/output counters differ')
    custom = report['custom_after_context_destroy']
    require(custom['compute_attempted'] == custom['compute_completed'] == 16
            and custom['init_attempted'] == custom['init_completed'] == 2
            and custom['destroy_attempted'] == custom['destroy_completed'] == 2
            and custom['live_GPU_bridges'] == custom['states_waiting_for_SDK_destroy'] == custom['active_compute'] == 0
            and custom['closed'] is True and not custom['errors'] and not custom['cleanup_errors'], 'Custom callback cleanup differs')
    require(custom['target'] == 'SDK_CPU_callback_private_GPU_staging'
            and custom['all_attention_tail_compute_on_board_GPU'] is True
            and custom['SDK_owned_GPU_handles_used'] is False, 'Actual CPU callback route differs')
    bridges = {}
    for role, bridge in custom['GPU_bridges'].items():
        require(bridge['calls_attempted'] == bridge['calls_completed'] == 8
                and bridge['events_created'] == bridge['events_released'] == 40
                and bridge['fully_released'] is True and bridge['live_OpenCL_objects'] == bridge['live_owned_DMA_fds'] == 0
                and bridge['cleanup_errors'] == [] and bridge['pending_owned_host_uploads'] == 0
                and bridge['GPU_staging_copies'] is True and bridge['DMA_imported'] is False,
                'Private GPU staging resource counters differ')
        bridges[role] = {key: bridge[key] for key in ('calls_attempted', 'calls_completed', 'events_created',
                         'events_released', 'live_OpenCL_objects', 'live_owned_DMA_fds', 'fully_released')}
    cell_summary = []
    for (case, route, name), values in cells.items():
        cell_summary.append(dict(case=case, route=route, output=name, calls=len(values),
            unique_raw_sha256=len({v['raw_sha256'] for v in values}),
            original_CPU_relative_l2_range=[min(v['relative_l2'] for v in values), max(v['relative_l2'] for v in values)],
            original_CPU_max_abs_range=[min(v['max_abs'] for v in values), max(v['max_abs'] for v in values)],
            original_CPU_failed_elements_range=[min(v['failed_elements'] for v in values), max(v['failed_elements'] for v in values)],
            original_CPU_strict_passed_outputs=sum(v['strict_pass'] for v in values)))
    require(all(v['unique_raw_sha256'] == 1 for v in cell_summary), 'A/B output repeatability differs')
    require(set(frequencies) == {1000000000} and max(thermal) < 75, 'Saved clock/thermal bound differs')
    return dict(schema='v83_original_attention_CPU_callback_GPU_staging_saved_audit', status='passed_saved_verification_candidate_rejected',
        auditor_source_sha256=sha(Path(__file__)), result_sha256=RESULT_SHA, capture_sha256=CAPTURE_SHA,
        actual_runner_source_sha256=report['script_sha256'], checked_NPZ_files=16, checked_output_arrays=32,
        checked_original_CPU_reference_arrays=4, all_artifact_hashes_shapes_dtypes_finite_verified=True,
        all_saved_comparison_metrics_independently_recomputed=True,
        summary=summaries, B_to_A_warm_mean_ratio=summaries['B']['mean_seconds']/summaries['A']['mean_seconds'],
        cells=cell_summary, calls=rows, original_CPU_reference_artifacts=reference_artifacts,
        resources=dict(contexts_created=2, contexts_destroyed=2, live_contexts=0,
                       SDK_counters=report['counters'], GPU_bridges=bridges, cleanup_errors=[]),
        saved_NPU_frequency_hz=sorted(set(frequencies)), call_snapshot_max_temperature_c=max(thermal),
        saved_GPU_observed_hz=sorted({c['GPU_observed_hz'] for c in report['calls']}),
        candidate_accepted=False, full_model_or_realtime_validated=False,
        limits=['Saved arrays and receipt assertions only; source/SDK binaries and device state were not re-executed.',
                'Both routes are stable across all four repetitions for each input/output.',
                'Both routes fail the unchanged original FP32 tensor gate. B is stable but slower; not adopted.',
                'No causal attribution to DMA coherence, compiler, memory ordering or internal precision from these outputs alone.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--capture-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    saved = audit(args.directory, args.capture_directory)
    args.output.write_text(json.dumps(saved, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: saved[key] for key in ('status', 'checked_output_arrays', 'summary', 'B_to_A_warm_mean_ratio')}))
