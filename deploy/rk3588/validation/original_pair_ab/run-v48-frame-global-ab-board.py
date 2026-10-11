#!/usr/bin/env python3
"""Board-only synchronous original VGGT separate/fused frame0→global0 A/B.

Exactly four alternating AB/BA rounds, twelve actual SDK calls, no warmup.
Separate global0 consumes that call's actual NPU frame0 output as an unchanged
view. All three contexts live in one process. Saved original CPU arrays are
comparison references only. This bounded component diagnostic cannot establish
fresh-observation pose rate, whole-model quality or real-time operation.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

sys.dont_write_bytecode = True
SCHEDULE = (('separate', 'fused'), ('fused', 'separate'),
            ('separate', 'fused'), ('fused', 'separate'))
CPUS = {6, 7}
ENVIRONMENT = {'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1',
               'MKL_NUM_THREADS': '2', 'NUMEXPR_NUM_THREADS': '2'}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def load_source(path, expected):
    if sha(path) != expected:
        raise ValueError('A/B contracts SHA mismatch')
    spec = importlib.util.spec_from_file_location('_v48_original_ab_contracts', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute_sdk_call(R, api, context, inputs, input_format, output_shape,
                     timeout_ms, cleanup_errors, ledger, role):
    """Owned copy before release; no error comparison, query or file I/O here."""
    import numpy as np
    if inputs.dtype != np.float32 or not inputs.flags.c_contiguous:
        raise ValueError('SDK input must be a contiguous original FP32 array/view')
    expected_bytes = math.prod(output_shape) * 4
    request = (R.Input * 1)()
    request[0] = R.Input(0, inputs.ctypes.data, inputs.nbytes, 0, 0, input_format)
    outputs = (R.Output * 1)()
    outputs[0].want_float = 1
    outputs[0].is_prealloc = 0
    outputs[0].index = 0
    run, received = R.RunExtend(0, 0, timeout_ms, -1), R.OutputExtend()
    row = {'sdk_call_index': len(ledger), 'role': role, 'status': 'failed',
           'synchronous': True, 'input_api_type': 'FP32', 'input_pass_through': 0,
           'input_format': input_format, 'input_shape': list(inputs.shape),
           'input_bytes': int(inputs.nbytes), 'input_pointer': int(inputs.ctypes.data),
           'output_shape': list(output_shape), 'want_float': 1, 'is_prealloc': 0,
           'run_timeout_ms': timeout_ms, 'output_buffer_released': None}
    ledger.append(row)
    owned, got, begin = None, False, time.monotonic_ns()
    try:
        R.check_call(api.rknn_inputs_set(context, 1, request), 'rknn_inputs_set')
        set_end = time.monotonic_ns()
        row['npu_run_api_called'] = True
        R.check_call(api.rknn_run(context, ct.byref(run)), 'rknn_run')
        run_end = time.monotonic_ns()
        R.check_call(api.rknn_outputs_get(context, 1, outputs, ct.byref(received)), 'rknn_outputs_get')
        got = True
        get_end = time.monotonic_ns()
        if run.frame_id != received.frame_id:
            raise ValueError('Synchronous SDK run/output frame identity mismatch')
        if not outputs[0].buf or outputs[0].size != expected_bytes:
            raise ValueError('SDK returned an invalid logical FP32 output pointer/byte count')
        view = np.ctypeslib.as_array(ct.cast(outputs[0].buf, ct.POINTER(ct.c_float)),
                                   shape=(math.prod(output_shape),))
        owned = view.copy().reshape(output_shape)
        copy_end = time.monotonic_ns()
        row.update(runtime_frame_id=int(run.frame_id), output_bytes=int(outputs[0].size))
    except Exception as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if got or outputs[0].buf:
            release_begin = time.monotonic_ns()
            code = api.rknn_outputs_release(context, 1, outputs)
            release_end = time.monotonic_ns()
            row.update(outputs_release_seconds=(release_end - release_begin) / 1e9,
                       output_buffer_released=code == 0,
                       io_begin_monotonic_ns=begin, io_end_monotonic_ns=release_end,
                       io_total_seconds=(release_end - begin) / 1e9)
            if code != 0:
                cleanup_errors.append(f'{role} SDK output release returned {code}')
            R.check_call(code, 'rknn_outputs_release')
    row.update(input_set_seconds=(set_end - begin) / 1e9,
               run_api_seconds=(run_end - set_end) / 1e9,
               outputs_get_seconds=(get_end - run_end) / 1e9,
               output_copy_seconds=(copy_end - get_end) / 1e9, status='completed')
    return owned, row


def bridge_frame_output(frame, frame_shape, global_shape):
    """Only view original ordered NPU data; reference arrays never enter here."""
    import numpy as np
    if (frame.dtype != np.float32 or frame.shape != frame_shape
            or not frame.flags.c_contiguous or not np.isfinite(frame).all()
            or math.prod(frame_shape) != math.prod(global_shape)):
        raise ValueError('Actual finite ordered FP32 frame0 output required')
    global_input = frame.reshape(global_shape)
    if (not np.shares_memory(global_input, frame)
            or global_input.ctypes.data != frame.ctypes.data
            or not global_input.flags.c_contiguous):
        raise ValueError('Original frame0→global0 bridge must be an unchanged same-pointer view')
    return global_input


def run_route(route, inputs, contexts, invoke, C):
    """Measured route includes checked bridge/hash work, separately itemized."""
    if route not in ('separate', 'fused'):
        raise ValueError('Unknown original execution route')
    begin = time.monotonic_ns()
    if route == 'fused':
        final, row = invoke('fused', contexts['fused'], inputs, C.GLOBAL_SHAPE)
        end = time.monotonic_ns()
        return final, None, {'route_begin_monotonic_ns': begin,
                            'route_end_monotonic_ns': end,
                            'route_total_seconds': (end - begin) / 1e9,
                            'sdk_io_sum_seconds': row['io_total_seconds'],
                            'sdk_call_indices': [row['sdk_call_index']], 'bridge': None}
    frame, frame_row = invoke('frame0', contexts['frame0'], inputs, C.FRAME_SHAPE)
    bridge_begin = time.monotonic_ns()
    global_input = bridge_frame_output(frame, C.FRAME_SHAPE, C.GLOBAL_SHAPE)
    # This diagnostic hash is inside the measured complete route and outside
    # both SDK IO intervals. Preserve its overhead rather than subtracting it.
    before = C.raw_sha(global_input)
    bridge_end = time.monotonic_ns()
    final, global_row = invoke('global0', contexts['global0'], global_input, C.GLOBAL_SHAPE)
    verify_begin = time.monotonic_ns()
    after = C.raw_sha(global_input)
    if before != after:
        raise ValueError('SDK changed the actual frame0→global0 input data')
    if global_row['input_pointer'] != frame.ctypes.data:
        raise ValueError('Global0 SDK input did not use the current frame0 output pointer')
    end = time.monotonic_ns()
    return final, frame, {'route_begin_monotonic_ns': begin,
                          'route_end_monotonic_ns': end,
                          'route_total_seconds': (end - begin) / 1e9,
                          'sdk_io_sum_seconds': frame_row['io_total_seconds'] + global_row['io_total_seconds'],
                          'sdk_call_indices': [frame_row['sdk_call_index'], global_row['sdk_call_index']],
                          'bridge': {'kind': 'current_NPU_frame0_owned_FP32_same_pointer_view',
                                     'input_raw_sha256_before': before, 'input_raw_sha256_after': after,
                                     'input_unchanged': True, 'source_frame_sdk_call_index': frame_row['sdk_call_index'],
                                     'target_global_sdk_call_index': global_row['sdk_call_index'],
                                     'source_shape': list(frame.shape), 'target_shape': list(global_input.shape),
                                     'same_pointer': True, 'shares_memory': True,
                                     'pre_global_view_finite_hash_seconds': (bridge_end - bridge_begin) / 1e9,
                                     'post_global_hash_verify_seconds': (end - verify_begin) / 1e9}}


def snapshot(R):
    result = {'monotonic_ns': time.monotonic_ns(), 'process_memory': R.process_memory(),
              'thermal_zones': [], 'system_memory_kib': {}, 'cpu_frequency_khz': {}}
    for zone in sorted(Path('/sys/class/thermal').glob('thermal_zone*')):
        row = {'path': str(zone)}
        try:
            row.update(type=(zone / 'type').read_text().strip(),
                       temperature_millicelsius=int((zone / 'temp').read_text().strip()))
        except (OSError, ValueError) as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
        result['thermal_zones'].append(row)
    for line in Path('/proc/meminfo').read_text().splitlines():
        words = line.split()
        if words[0] in ('MemTotal:', 'MemAvailable:', 'SwapTotal:', 'SwapFree:', 'CmaTotal:', 'CmaFree:'):
            result['system_memory_kib'][words[0].rstrip(':')] = int(words[1])
    for cpu in sorted(CPUS):
        try:
            result['cpu_frequency_khz'][str(cpu)] = int(Path(
                f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq').read_text().strip())
        except (OSError, ValueError):
            result['cpu_frequency_khz'][str(cpu)] = None
    return result


def prepare_context(R, C, api, role, model, contexts, records):
    context = R.CONTEXT()
    contexts[role] = context  # Register before init, including partially allocated failure.
    row = {'role': role, 'model': str(model), 'context_destroyed': None,
           'init_flags': 0, 'core_mask': 7, 'tensor_attributes_validated': False}
    records[role] = row
    begin = time.monotonic_ns()
    filename = ct.create_string_buffer(os.fsencode(model))
    R.check_call(api.rknn_init(ct.byref(context), ct.cast(filename, ct.c_void_p), 0, 0, None), 'rknn_init')
    row['init_seconds'] = (time.monotonic_ns() - begin) / 1e9
    R.check_call(api.rknn_set_core_mask(context, 7), 'rknn_set_core_mask')
    version = R.query(api, context, 5, R.SDKVersion())
    row.update(sdk_version=bytes(version.api_version).decode(errors='replace'),
               driver_version=bytes(version.drv_version).decode(errors='replace'))
    if not row['sdk_version'].startswith('2.3.2') or row['driver_version'] != '0.7.2':
        raise ValueError('A/B requires pinned runtime 2.3.2 / driver 0.7.2')
    count = R.query(api, context, 0, R.InputOutputNum())
    if count.n_input != 1 or count.n_output != 1:
        raise ValueError('Original block must have exactly one input and output')
    for direction, command in [('input', 1), ('output', 2)]:
        record = R.attr_record(R.query(api, context, command, R.TensorAttr(index=0)))
        row[direction + '_attr'] = record
        C.validate_sdk_attr(record, role, direction)
    row['tensor_attributes_validated'] = True
    memory = R.MemSize()
    code = api.rknn_query(context, 6, ct.byref(memory), ct.sizeof(memory))
    row['model_memory_query_code'] = int(code)
    row['model_memory_bytes'] = ({name: int(getattr(memory, name))
                                for name, _ in memory._fields_ if name != 'reserved'} if code == 0 else None)


def save_array(output_dir, name, array, C):
    import numpy as np
    path = output_dir / name
    # Exclusive open makes accidental reuse a failure rather than replacement.
    with path.open('xb') as stream:
        np.save(stream, array, allow_pickle=False)
    return {'file': name, 'bytes': path.stat().st_size, 'sha256': C.sha(path),
            'shape': list(array.shape), 'dtype': str(array.dtype), 'raw_sha256': C.raw_sha(array)}


def summary(calls):
    import numpy as np
    result = {}
    for route in ('separate', 'fused'):
        rows = [row for row in calls if row['route'] == route]
        result[route] = {'requested_call_count': 4, 'completed_call_count': len(rows)}
        for key in ('route_total_seconds', 'sdk_io_sum_seconds'):
            first = rows[0][key] if rows else None
            warm = [row[key] for row in rows[1:]]
            result[route][key] = {'first': first, 'subsequent_values': warm,
                                 'subsequent_mean': float(np.mean(warm)) if warm else None}
        result[route]['strict_original_FP32_gate_passed'] = (
            len(rows) == 4 and all(row['versus_original_global_reference']['strict_FP32_gate_passed'] for row in rows))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('ab-contracts', 'pair-contracts', 'pair-dir', 'runtime-helper', 'sdk-header',
                 'pair-compile-receipt', 'pair-ort-receipt', 'pair-model', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('ab-contracts-sha256', 'pair-compile-sha256', 'pair-ort-sha256', 'pair-model-sha256'):
        parser.add_argument('--' + name, required=True)
    for role in ('frame0', 'global0'):
        for name in ('export-receipt', 'ort-receipt', 'compile-receipt', 'model'):
            parser.add_argument('--' + role + '-' + name, type=Path, required=True)
    parser.add_argument('--diagnostic-allow-FP32-rejected', action='store_true', required=True)
    parser.add_argument('--run-timeout-ms', type=int, default=30000)
    args = parser.parse_args(argv)
    if not 1 <= args.run_timeout_ms <= 60000:
        parser.error('run-timeout-ms must be 1..60000')
    # Reject the host before any captured array or actual SDK library loading.
    if sys.platform != 'linux' or platform.machine().lower() not in ('aarch64', 'arm64'):
        parser.error('Only the connected Linux RK3588S may run actual SDK A/B')
    C = load_source(args.ab_contracts, args.ab_contracts_sha256)
    if (Path.cwd().resolve() != C.BASE or args.output_dir.resolve() == C.BASE / 'experiments'
            or not args.output_dir.resolve().is_relative_to(C.BASE / 'experiments')):
        parser.error('Board project cwd and a new experiments output directory required')
    pins = {str(Path(__file__).resolve()): sha(__file__),
            str(args.ab_contracts.resolve()): args.ab_contracts_sha256}
    R = C.load_source(args.runtime_helper, C.PARENT_RUNTIME_SHA, pins)
    compatible = R.require_board()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'v48_original_VGGT_frame0_global0_controlled_SDK_AB', 'status': 'failed',
              'script_sha256': sha(__file__), 'ab_contracts_sha256': args.ab_contracts_sha256,
              'pair_export_sha256': C.PAIR_EXPORT_SHA, 'pair_compile_sha256': args.pair_compile_sha256,
              'pair_ort_sha256': args.pair_ort_sha256, 'pair_model_sha256': args.pair_model_sha256,
              'device_tree_compatible': compatible, 'inference_on_board_only': True,
              'full_model_run': False, 'candidate_accepted': False, 'real_time_validated': False,
              'whole_model_quality_validated': False, 'historical_RKNN_FP32_rejections_retained': True,
              'diagnostic_allow_FP32_rejected': True, 'all_internal_operators_on_NPU_verified': False,
              'schedule': [list(row) for row in SCHEDULE], 'normal_benchmark': True,
              'contexts_live_together': True, 'requested_route_call_count': 8,
              'requested_sdk_call_count': 12, 'unmeasured_warmup_calls': 0,
              'abi': R.abi_record(), 'cpu_affinity_requested': sorted(CPUS),
              'sdk_io_interval': 'inputs_set + synchronous run + outputs_get + owned FP32 copy + release',
              'route_interval': 'complete route including checked view, finite checks and bridge hashes',
              'timing_excludes': ['context init', 'SDK queries', 'comparison', 'NPY/JSON saves',
                                 'source/artifact hashes', 'system/thermal snapshots'],
              'limitations': ['Fixed same captured input; no fresh observation or camera timing',
                              'Original frame0/global0 only; other VGGT layers and heads are outside scope',
                              'Four calls per route do not prove 20 Hz, P95, stability or trajectory quality',
                              'Alternating order reduces ordering bias; clocks/cooling remain uncontrolled',
                              'Three co-resident contexts may consume more memory than a future executor',
                              'Checkpoint provenance is inherited from pinned original export; no new checkpoint rehash'],
              'calls': [], 'sdk_calls': [], 'context_records': {}, 'paired_comparisons': [],
              'cleanup_errors': [], 'memory_rss_scope': 'process high-water including NumPy diagnostic arrays'}
    original_affinity = os.sched_getaffinity(0)
    original_environment = {key: os.environ.get(key) for key in ENVIRONMENT}
    contexts, api, inputs, input_before = {}, None, None, None
    started = time.monotonic_ns()
    try:
        if not CPUS <= original_affinity:
            raise ValueError('Requested CPU 6/7 affinity unavailable')
        os.sched_setaffinity(0, CPUS)
        os.environ.update(ENVIRONMENT)
        report.update(cpu_affinity=sorted(os.sched_getaffinity(0)), environment=ENVIRONMENT,
                      python=sys.version)
        P = C.load_source(args.pair_contracts, C.PAIR_CONTRACTS_SHA, pins)
        pair, arrays, _, pair_pins = P.load_pair(args.pair_dir, C.PAIR_EXPORT_SHA)
        pins.update(pair_pins)
        models = {}
        for role in ('frame0', 'global0'):
            row = C.PARENTS[role]
            export = C.load_json(getattr(args, role + '_export_receipt'), row['export_sha256'], pins)
            ort = C.load_json(getattr(args, role + '_ort_receipt'), row['ort_sha256'], pins)
            compiled = C.load_json(getattr(args, role + '_compile_receipt'), row['compile_sha256'], pins)
            C.check_parent(role, export, ort, compiled, pair)
            model = C.pin(getattr(args, role + '_model'), row['model_sha256'], pins)
            if model.stat().st_size != row['model_bytes']:
                raise ValueError('Original separate compiled model byte count mismatch')
            models[role] = model
        compiled = C.load_json(args.pair_compile_receipt, args.pair_compile_sha256, pins)
        P.check_compile(compiled, C.PAIR_EXPORT_SHA, args.pair_model_sha256)
        if compiled.get('script_sha256') != C.PAIR_COMPILER_SHA:
            raise ValueError('Fused compiler source must be the frozen original pair compiler')
        ort = C.load_json(args.pair_ort_receipt, args.pair_ort_sha256, pins)
        C.check_fused_ort(ort, args.pair_ort_sha256, compiled)
        models['fused'] = C.pin(args.pair_model, args.pair_model_sha256, pins)
        if models['fused'].stat().st_size != compiled['output']['size']:
            raise ValueError('Fused compiled model byte count mismatch')
        C.pin(args.sdk_header, C.SDK_HEADER_SHA, pins)
        library = C.pin('/usr/lib/librknnrt.so', C.SDK_LIBRARY_SHA, pins)
        inputs = arrays['input.npy']
        inputs.setflags(write=False)
        input_before = C.raw_sha(inputs)
        report.update(original_checkpoint=pair['checkpoint'], selected_pair=pair['selected_pair'],
                      original_parameter_tensors=36, original_checkpoint_rehashed_by_this_runner=False,
                      input_raw_sha256_before=input_before, memory_before_contexts=snapshot(R),
                      original_parent_artifacts=C.PARENTS, runtime_helper_sha256=C.PARENT_RUNTIME_SHA)
        api = R.bind_api(ct.CDLL(str(library)))
        for role in ('frame0', 'global0', 'fused'):
            prepare_context(R, C, api, role, models[role], contexts, report['context_records'])
        report['memory_after_all_contexts_init'] = snapshot(R)

        def invoke(role, context, actual_input, output_shape):
            record = report['context_records'][role]['input_attr']
            if tuple(actual_input.shape) != tuple(record['shape']):
                raise ValueError('Actual SDK input shape differs from queried original role')
            return execute_sdk_call(R, api, context, actual_input, record['format'], output_shape,
                                    args.run_timeout_ms, report['cleanup_errors'], report['sdk_calls'], role)

        route_counts, first_outputs = {'separate': 0, 'fused': 0}, {}
        for round_index, order in enumerate(SCHEDULE):
            round_outputs = {}
            for route in order:
                before = snapshot(R)
                actual, frame, timing = run_route(route, inputs, contexts, invoke, C)
                after = snapshot(R)
                # All comparison, serialization and informational SDK queries
                # follow the measured route and each individual SDK IO interval.
                call_index = route_counts[route]
                entry = {'route': route, 'route_call_index': call_index,
                         'round_index': round_index, 'order_in_round': list(order).index(route),
                         'phase': 'first' if call_index == 0 else 'subsequent', **timing,
                         'memory_and_temperature_before': before, 'memory_and_temperature_after': after,
                         'versus_original_global_reference': P.compare(actual, arrays['output.npy']),
                         'actual': save_array(args.output_dir, f'round_{round_index:02d}_{route}_final.npy', actual, C)}
                if frame is not None:
                    entry['frame_actual'] = save_array(args.output_dir, f'round_{round_index:02d}_frame0.npy', frame, C)
                    entry['frame_versus_original_reference'] = P.compare(frame, arrays['frame-reference.npy'])
                    if entry['frame_actual']['raw_sha256'] != timing['bridge']['input_raw_sha256_before']:
                        raise ValueError('Saved frame0 output differs from actual global0 input')
                if route not in first_outputs:
                    first_outputs[route] = actual
                entry['versus_first_same_route'] = P.compare(actual, first_outputs[route])
                report['calls'].append(entry)
                route_counts[route] += 1
                round_outputs[route] = actual
                for index in timing['sdk_call_indices']:
                    row = report['sdk_calls'][index]
                    perf = R.PerfRun()
                    code = api.rknn_query(contexts[row['role']], 4, ct.byref(perf), ct.sizeof(perf))
                    row.update(perf_run_query_code=int(code),
                               sdk_run_duration_us=int(perf.run_duration) if code == 0 else None)
            report['paired_comparisons'].append({'round_index': round_index, 'order': list(order),
                                                 'fused_versus_actual_separate': P.compare(
                                                     round_outputs['fused'], round_outputs['separate'])})
        report['summaries'] = summary(report['calls'])
        report['status'] = 'completed_original_FP32_rejected_diagnostic_not_adopted'
        if all(row['strict_original_FP32_gate_passed'] for row in report['summaries'].values()):
            report['status'] = 'completed_original_FP32_gate_diagnostic_not_adopted'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if api is not None:
            for role, context in reversed(list(contexts.items())):
                if not context.value:
                    continue
                try:
                    code = api.rknn_destroy(context)
                    report['context_records'][role]['context_destroyed'] = code == 0
                    R.check_call(code, role + ' rknn_destroy')
                except Exception as exc:
                    report['cleanup_errors'].append(f'{role} context destroy: {type(exc).__name__}: {exc}')
        try:
            report['memory_after_contexts_destroy'] = snapshot(R)
            if inputs is not None:
                report['input_raw_sha256_after'] = C.raw_sha(inputs)
                report['original_input_tensor_unchanged'] = report['input_raw_sha256_after'] == input_before
                if not report['original_input_tensor_unchanged']:
                    report['cleanup_errors'].append('Original in-memory input changed')
            after, unchanged = C.recheck(pins)
            report.update(source_and_input_hashes=pins, source_and_input_hashes_after=after,
                          source_and_inputs_unchanged=unchanged)
            if not unchanged:
                report['cleanup_errors'].append('Frozen original source/input/runtime changed')
        except Exception as exc:
            report['cleanup_errors'].append(f'Final hashes/memory: {type(exc).__name__}: {exc}')
        try:
            os.sched_setaffinity(0, original_affinity)
            for key, value in original_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            report['affinity_restored'] = os.sched_getaffinity(0) == original_affinity
            report['environment_restored'] = {key: os.environ.get(key) for key in ENVIRONMENT} == original_environment
            if not report['affinity_restored'] or not report['environment_restored']:
                report['cleanup_errors'].append('Process affinity/environment restoration mismatch')
        except Exception as exc:
            report['cleanup_errors'].append(f'Process restoration: {type(exc).__name__}: {exc}')
        report['successful_synchronous_call_count'] = sum(row['status'] == 'completed' for row in report['sdk_calls'])
        report['npu_run_api_call_count'] = sum(row.get('npu_run_api_called') is True for row in report['sdk_calls'])
        report['completed_route_call_count'] = len(report['calls'])
        report['total_seconds_including_all_diagnostics'] = (time.monotonic_ns() - started) / 1e9
        if report['status'].startswith('completed_') and (
                report['successful_synchronous_call_count'] != 12 or report['completed_route_call_count'] != 8
                or not all(row['context_destroyed'] is True for row in report['context_records'].values())):
            report['cleanup_errors'].append('Expected calls or all context cleanup not proven')
        if report['cleanup_errors']:
            report['status'] = 'failed'
        with (args.output_dir / 'run.json').open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
    print(json.dumps({'status': report['status'], 'error': report.get('error'),
                      'run_sha256': sha(args.output_dir / 'run.json')}, allow_nan=False), flush=True)
    return 0 if report['status'].startswith('completed_') else 2


if __name__ == '__main__':
    raise SystemExit(main())
