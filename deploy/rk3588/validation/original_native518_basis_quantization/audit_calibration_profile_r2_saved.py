#!/usr/bin/env python3
"""Audit saved calibration-only RKNN profiles and raw outputs; no inference."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

RESULT_SHA = '4013651f463a39475560b7c2e0544b3ce0fb813ece5a73e343cae9299e84d91b'
CONTROL_SHA = '579642cf38242f3e289fcaaa4435f283b1f3c2b9be572457b3a29f47c5441ab6'
CAPTURE_SHA = 'cc2dd7fbddef8ee12de00b7a5cc88a16ba9d531783ab09ae8e66abe50df43408'
SOURCE_PINS = {
    'profile_calibration_pair_board_r2.py': 'e5671ade47ca65244c529b3ddb325c6aac1d6234a215e43a4652ae18ce135a79',
    'profile_owned_context_r2.py': '5aa75a7630ecb9793773553da1f7351a97ca5bac728d4f01f9b74d83d3ca2426',
    'execute_calibration_profile_host_r2.py': 'e41003f87bb13fc98e49266a5ef7d6e1904dcb8d759c9735134c8bcdd0ef5ec8',
    'run_hybrid_pair_board.py': 'd3157c5d1cecc4de541e9ff9a0d2adbf02e4a0269018892f24cddecbe47fce30',
    'compile_hybrid_pair_step2_board.py': 'f1c94014ad6f7555c32ac0f83d489ade46295cef96362ed7f7c9878f01cf9780',
    'npu_frequency_test.py': 'cb991cf03abc872e3590dd92f26fd0f5ac548d620c5ca791f9354537a9f793b9',
}
COMPILE_PINS = {
    'originalFP16_A': ('../v127-basis-fp16-work/compile-A-r1', '71b8fc05994e15cf3a658d01f57b89f909172617ff513dea97971dc13d18bb5c'),
    'mixedA': ('step2-A-r2', '1569bb244cda902926a776fc28e267751640ebe5756f84fb2967799028f1fb6e'),
    'mixedF': ('step2-F-r2', '1f32f8b99c8c6848bd122befb86831ee2d57aff85fb7ae07b9a60ebc0832b81c'),
    'mixedB': ('step2-B-r2', 'c85684e5ed7170ddd0dc04a4934d3132175863c335f0cb3f3fa90e96651baba3'),
    'joint_MLP': ('joint_mlp_candidate/compile-r1', '01f9034d151e0b0388c98b1fbf1ccf81163dace668c4afa13ccce9ecf433388f'),
}
FRAME, GLOBAL = [2, 1041, 1024], [1, 2082, 1024]
NAMES = ['frame_output', 'global_output']


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def pin(path, digest, size=None):
    path = Path(path)
    require(sha(path) == digest and (size is None or path.stat().st_size == size), 'File identity differs: ' + str(path))
    return path


def relative(folder, row):
    name = Path(row['file'])
    require(not name.is_absolute() and '..' not in name.parts, 'Relative artifact required')
    return pin(folder / name, row['sha256'], row['bytes'])


def read_array(folder, row, shape):
    path = relative(folder, row)
    a = np.load(path, allow_pickle=False)
    require(a.dtype == np.float32 and list(a.shape) == shape == row['shape'] and row['dtype'] == 'float32' and a.flags.c_contiguous, 'Saved dtype/shape/layout differs')
    raw = a.tobytes()
    require(a.nbytes == row['tensor_bytes'] and hashlib.sha256(raw).hexdigest() == row['raw_sha256'], 'Raw payload identity differs')
    finite = np.isfinite(a)
    stats = dict(finite=bool(finite.all()), finite_count=int(finite.sum()), nan_count=int(np.isnan(a).sum()),
        positive_inf_count=int(np.isposinf(a).sum()), negative_inf_count=int(np.isneginf(a).sum()))
    for key, value in stats.items():
        if key in row:
            require(row[key] == value, 'Reported nonfinite count differs: ' + key)
    require(sum(stats[k] for k in ('finite_count', 'nan_count', 'positive_inf_count', 'negative_inf_count')) == a.size, 'Incomplete element accounting')
    finite_values = a[finite]
    stats['finite_value_range'] = [float(finite_values.min()), float(finite_values.max())] if finite_values.size else None
    return raw, dict(file=path.name, bytes=path.stat().st_size, sha256=row['sha256'], raw_sha256=row['raw_sha256'],
        shape=shape, dtype='float32', tensor_bytes=a.nbytes, **stats)


def profile_rows(path):
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        match = re.match(r'^\s*(\d+)\s+(\S+)\s+(FLOAT16|FLOAT32|INT8|INT16|UINT8|INT32)\s+(CPU|NPU)\s+', line)
        if match:
            rows.append(dict(id=int(match[1]), op_type=match[2], dtype=match[3], target=match[4], full_name=line.split()[-1], source_line=line_no))
    require(rows and [r['id'] for r in rows] == list(range(1, len(rows) + 1)), 'Incomplete profile table parse')
    return rows


def descriptors(call):
    ins, outs = call['input_descriptors'], call['output_descriptors']
    require(len(ins) == 1 and len(outs) == 2, 'Complete pair descriptor arity differs')
    for d, name, shape in zip(ins + outs, ['input'] + NAMES, [FRAME, FRAME, GLOBAL]):
        require(d['name'] == name and d['shape'] == shape and d['n_elems'] == math.prod(shape) and d['pass_through'] == 0, 'Logical IO descriptor differs')
        require(d['type'] in (1, 2) and d['size'] == d['n_elems'] * (2 if d['type'] == 1 else 1), 'Unexpected native descriptor dtype/bytes')
        require(d['quantization_type'] == 2 and math.isfinite(d['scale']) and d['scale'] > 0, 'Invalid descriptor quantization metadata')
        if d['type'] == 2:
            require(-128 <= d['zero_point'] <= 127, 'Signed INT8 zero-point differs')
    return [{k: d[k] for k in ('name', 'shape', 'type', 'quantization_type', 'zero_point', 'scale')} for d in ins + outs]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--output', type=Path)
    args = p.parse_args(); work = args.work_dir.resolve(); folder = work / 'calibration-profile-r2'
    result = json.loads(pin(folder / 'result.json', RESULT_SHA).read_text())
    control = json.loads(pin(work / 'calibration-profile-control-r2/control.json', CONTROL_SHA).read_text())
    require(result['schema'] == 'v128_original_native518_calibration_pair_flags8_placement_diagnostic' and result['status'] == 'completed_calibration_profile_diagnostic_not_performance_or_quality_accepted', 'Wrong diagnostic receipt')
    require(result['input_id'] == 'pair0' and result['input_split'] == 'calibration' and result['source_pair_indices'] == [0, 8] and not result['heldout_arrays_accessed'], 'Calibration-only scope differs')
    require(result['all_calls_completed'] and result['all_resources_closed'] and result['source_and_inputs_unchanged'] and result['affinity_restored'] and result['environment_restored'] and not result['cleanup_errors'], 'Incomplete or unclean worker')
    require(not result['candidate_accepted'] and not result['numerical_acceptance'] and not result['nonfinite_values_rewritten'] and not result['production_finite_guard_modified'], 'Numeric acceptance/guard scope differs')
    source_rows = []
    for name, digest in SOURCE_PINS.items():
        path = pin(work / 'executed-calibration-profile-source-r2' / name, digest)
        source_rows.append(dict(file=name, bytes=path.stat().st_size, sha256=digest))
    require(result['script_sha256'] == SOURCE_PINS['profile_calibration_pair_board_r2.py'] and result['owned_IO_source_sha256'] == SOURCE_PINS['profile_owned_context_r2.py'], 'Executed source identity differs')
    external_pins = {
        '/usr/lib/librknnrt.so': 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8',
        '/mnt/vggt-data/vggt-slam2/src/deploy/rk3588/validation/run-v12-rknn-block-board.py': '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608',
        '/mnt/vggt-data/vggt-slam2/experiments/vggt-v99-source-r1/run_native_block_r1.py': '56ce45635a5d010e862bf59db2f9fcc6e8ec642b2b023c8c40cad0bd5808050f',
    }
    require(all(result['pinned_files'][k] == v for k, v in external_pins.items()) and result['original_owned_IO_source_sha256'] == '0bfcb295ad5490e8196e8fcc1244e11a99b07bfb3ba7d5c350cb8c81d3ff3be4', 'Recorded original board SDK/ABI/helper pins differ')
    cap = json.loads(pin(work / 'capture-batch0-r1/result.json', CAPTURE_SHA).read_text())
    export_sha = 'e23c8f7cdf6faa895a8bde34a41806bf2022e630f5361529d0271374742d740e'
    exported = json.loads(pin(work.parent / 'v126-basis-recovery-work/export-r2/result.json', export_sha).read_text())
    require(result['export_receipt']['sha256'] == export_sha and result['original_parameters'] == exported['original_parameters'] and all(len(exported['original_parameters'][k]) == 18 for k in ('frame', 'global')), 'Original 36-value parent export differs')
    record = next(r for r in cap['records'] if r['id'] == 'pair0')
    require(result['capture_receipt']['sha256'] == CAPTURE_SHA and record['input'] == result['input'] and record['split'] == 'calibration' and record['source_pair_indices'] == [0, 8], 'Actual input lineage differs')
    _, input_stats = read_array(work / 'capture-batch0-r1', record['input'], FRAME)
    require(input_stats['finite'], 'Original input is not finite')
    routes = list(COMPILE_PINS)
    require(result['routes'] == routes and [(c['route'], c['init_flags']) for c in result['calls']] == [(r, f) for r in routes for f in (0, 8)], 'Fresh flags0/8 schedule differs')
    totals = dict(contexts_created=0, contexts_destroyed=0, runs_completed=0, output_buffers_acquired=0, output_buffers_released=0)
    audited_routes, filenames, invalid_routes = [], set(), []
    for route in routes:
        compile_folder, digest = COMPILE_PINS[route]; compile_folder = work / compile_folder
        compiled = json.loads(pin(compile_folder / 'result.json', digest).read_text())
        model = compiled['rknn']; relative(compile_folder, model)
        require(result['compile_receipts'][route]['sha256'] == digest and result['models'][route]['sha256'] == model['sha256'] and result['models'][route]['bytes'] == model['bytes'], 'Actual model/compile differs')
        require(compiled['checkpoint'] == result['checkpoint'], 'Checkpoint lineage differs')
        if route == 'originalFP16_A':
            require(compiled['original_parameter_value_count'] == 36 and compiled['six_original_FP32_calls_passed'] and compiled['export_receipt']['sha256'] == export_sha, 'Original A 36-value parent lineage differs')
        else:
            require(compiled['original_parameters'] == result['original_parameters'], 'Original 36-value lineage differs')
        calls, raw_by_flags = [], {}
        for call in (c for c in result['calls'] if c['route'] == route):
            flags = call['init_flags']; require(call['index'] == len(filenames) // 2, 'Call order differs')
            require([r['name'] for r in call['outputs']] == NAMES, 'Complete two-output boundary required')
            payloads, outputs = [], []
            for row, shape in zip(call['outputs'], (FRAME, GLOBAL)):
                require(row['file'] not in filenames, 'Duplicate saved artifact')
                filenames.add(row['file']); raw, audited = read_array(folder, row, shape)
                payloads.append(raw); outputs.append(audited)
            raw_by_flags[flags] = payloads
            invalid = not all(r['finite'] for r in outputs)
            require(call['invalid_numeric'] == invalid and call['all_outputs_finite'] == (not invalid), 'Finite guard outcome differs')
            require(call['status'] == ('completed_invalid_numeric_diagnostic' if invalid else 'completed'), 'Numeric status differs')
            if invalid:
                require(call['original_output_guard_raised'] and call['raw_failed_outputs_preserved'] and call['numerical_guard_error'] == 'Nonfinite owned SDK output', 'Original guard evidence missing')
            owner = call['owner_metrics']
            expected = dict(contexts_created=1, contexts_destroyed=1, runs_completed=1, output_buffers_acquired=2, output_buffers_released=2)
            for key, count in expected.items():
                require(owner[key] == count, 'Per-call resource count differs: ' + key); totals[key] += count
            require(owner['closed'] and owner['resource_counts_balanced'] and owner['live_contexts'] == owner['live_output_buffers'] == 0 and not owner['cleanup_errors'] and owner['failed'] == invalid, 'Owner cleanup/error state differs')
            require(owner['core_mask'] == 1 and owner['init_flags'] == flags and owner['model_sha256'] == model['sha256'], 'Actual owner execution mode differs')
            require(owner['run_attempts'] == owner['output_get_successes'] == owner['output_release_attempts'] == 1 and owner['input_set_successes'] == 1, 'IO operation ledger differs')
            require(owner['input_descriptors'] == call['input_descriptors'] and owner['output_descriptors'] == call['output_descriptors'], 'Descriptor snapshots differ')
            item = dict(index=call['index'], init_flags=flags, invalid_numeric=invalid, descriptors=descriptors(call), outputs=outputs,
                SDK_model_memory_bytes=call['SDK_model_memory_bytes'], closed_failed_owner=owner['failed'], counts=expected)
            if flags == 8:
                rows = profile_rows(relative(folder, call['profile']))
                item['profile'] = dict(call['profile'], operator_rows=len(rows),
                    dtype_target_counts=dict(Counter(r['dtype'] + '/' + r['target'] for r in rows)),
                    boundary_rows=[r for r in rows if r['op_type'] in ('InputOperator', 'OutputOperator') or '__global_input' in r['full_name']],
                    attention_rows=[r for r in rows if 'SDPA' in r['op_type']],
                    residual_Add_rows=[r for r in rows if r['op_type'] == 'Add'],
                    SDK_reported_run_duration_us=call['SDK_reported_run_duration_us'])
                require(call['pointer_copied_before_next_query'], 'Profile pointer ownership flag differs')
            calls.append(item)
        parity = all(a == b for a, b in zip(raw_by_flags[0], raw_by_flags[8]))
        require(parity and next(c for c in result['calls'] if c['route'] == route and c['init_flags'] == 8)['all_outputs_byte_equal_to_flags0'], 'Flags0/8 final raw-byte attribution gate failed')
        require(calls[0]['descriptors'] == calls[1]['descriptors'], 'Flags0/8 logical descriptors changed')
        if calls[0]['invalid_numeric']:
            invalid_routes.append(route)
        audited_routes.append(dict(route=route, model_sha256=model['sha256'], compile_sha256=digest, flags0_8_raw_byte_equal=True, calls=calls))
    require(len(filenames) == result['saved_array_count'] == 20 and {p.name for p in folder.glob('*.npy')} == filenames, 'Saved output inventory differs')
    expected = dict(contexts_created=10, contexts_destroyed=10, runs_completed=10, output_buffers_acquired=20, output_buffers_released=20)
    require(totals == result['resource_totals'] == result['expected_resource_counts'] == expected, 'Aggregated resource ledger differs')
    require(invalid_routes == result['invalid_numeric_routes'] and not result['all_outputs_finite'] and result['attribution_byte_parity_passed'], 'Global diagnostic summary differs')
    require(control['status'] == 'worker_completed_pending_result_review' and control['child_returncode'] == 0 and not control['restore_errors'] and control['frequency_governor_limits_restored'] and control['baseline'] == control['restored'], 'Controller cleanup differs')
    observations = control['observations']; temperatures = [v for o in observations for v in o['temperatures_c'].values()]
    require(observations and temperatures and all(o['actual_frequency_hz'] == 1000000000 for o in observations) and max(temperatures) < control['temperature_limit_c'] == 75, 'Recorded controller clock/thermal bounds differ')
    report = dict(schema='v128_independent_saved_calibration_profile_r2_audit', status='saved_evidence_verified_invalid_numeric_routes_not_accepted',
        auditor_sha256=sha(__file__), result_sha256=RESULT_SHA, control_sha256=CONTROL_SHA, executed_sources=source_rows,
        board_dependency_pins_from_frozen_receipt_only={Path(k).name: v for k, v in external_pins.items()},
        input_id='pair0', source_pair_indices=[0, 8], split='calibration', input=input_stats,
        output_count=20, all_saved_output_file_raw_shape_dtype_and_count_checks_passed=True,
        flags0_8_all_five_routes_raw_byte_equal=True, invalid_numeric_routes=invalid_routes, routes=audited_routes,
        resource_totals=totals, all_owners_closed_no_live_outputs=True,
        controller=dict(child_returncode=0, seconds=control['seconds'], maximum_recorded_temperature_c=max(temperatures),
            observation_count=len(observations), baseline=control['baseline'], restored=control['restored'], frequency_governor_limits_restored=True),
        numerical_acceptance=False, candidate_accepted=False, host_inference_or_SDK_or_board_calls=False,
        limitations=[
            'Only calibration pair0 was audited; no heldout error or trajectory quality was evaluated.',
            'Flags0/8 final byte parity verifies endpoint-faithful profiling, not equality of every hidden intermediate.',
            'Profile DataType/Target and logical endpoint descriptors do not prove accumulator precision or nonfinite causality.',
            'A/F frame endpoints are signed INT8; B endpoints are floating. Different compiled boundaries prevent a clean basis-conditioning attribution.',
            'SDK memory fields are per-context reports, not physical unique-byte totals. Profile durations are not normal performance measurements.',
            'Immutable prepared-input bytes were checked against saved capture; preprocessing and whole-network accuracy were not rerun.'
        ])
    output = args.output or work / 'calibration-profile-r2-saved-audit.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output=output.name, sha256=sha(output), output_count=20, invalid_numeric_routes=invalid_routes, resources=totals)))


if __name__ == '__main__':
    main()
