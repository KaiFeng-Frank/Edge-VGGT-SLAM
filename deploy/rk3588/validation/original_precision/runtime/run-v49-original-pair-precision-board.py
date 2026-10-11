#!/usr/bin/env python3
"""Board-only original VGGT FP16/PTQ pair comparison on two heldout inputs.

Two co-resident contexts execute four AB/BA rounds per original heldout pair:
sixteen synchronous SDK calls, no hidden warmup. Both are implementations of
the same complete original frame0/global0 graph. Each actual output is compared
only to that heldout pair's original Torch FP32 output. No candidate is adopted.
Profiling uses another process/output directory and fresh profiling contexts.
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
ROLES = ('nonquant_fp16', 'ptq_w8a8')
SCHEDULE = (ROLES, ROLES[::-1], ROLES, ROLES[::-1])
CPUS = {6, 7}
ENVIRONMENT = {'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1',
               'MKL_NUM_THREADS': '2', 'NUMEXPR_NUM_THREADS': '2'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path, expected):
    if sha(path) != expected:
        raise ValueError('Runtime contracts SHA mismatch')
    spec = importlib.util.spec_from_file_location('_v49_original_precision_contracts', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def schedule_for_sample(index):
    if type(index) is not int or index not in (0, 1):
        raise ValueError('Exactly two ordered heldout samples are supported')
    return SCHEDULE if index == 0 else tuple(row[::-1] for row in SCHEDULE)


def execute_sdk_call(R, api, context, inputs, input_format, output_shape,
                     timeout_ms, cleanup_errors, ledger, role):
    """FP32 API input and owned FP32 output; only typed API/copy inside IO time.

    R.require_board is a second host guard. Tiny tests inject a fake API and
    fake guard; they never load a shared SDK library or an actual model.
    """
    R.require_board()
    import numpy as np
    if (inputs.dtype != np.float32 or not inputs.flags.c_contiguous
            or input_format != 3 or not np.isfinite(inputs).all()
            or type(timeout_ms) is not int or not 1 <= timeout_ms <= 60000
            or any(type(v) is not int or v <= 0 for v in output_shape)):
        raise ValueError('Finite contiguous FP32 input and logical rank3 API contract required')
    output_elements = math.prod(output_shape)
    expected_bytes = output_elements * 4
    if not 0 < expected_bytes <= 256 * 1024 * 1024:
        raise ValueError('Logical FP32 output bytes are outside the probe bound')
    request = (R.Input * 1)()
    request[0] = R.Input(0, inputs.ctypes.data, inputs.nbytes, 0, 0, input_format)
    outputs = (R.Output * 1)()
    outputs[0].want_float, outputs[0].is_prealloc, outputs[0].index = 1, 0, 0
    run, received = R.RunExtend(0, 0, timeout_ms, -1), R.OutputExtend()
    row = {'sdk_call_index': len(ledger), 'role': role, 'status': 'failed',
           'synchronous': True, 'input_api_type': 'FP32', 'input_api_type_enum': 0,
           'input_pass_through': 0, 'input_format': input_format,
           'input_shape': list(inputs.shape), 'input_bytes': int(inputs.nbytes),
           'input_pointer': int(inputs.ctypes.data), 'output_shape': list(output_shape),
           'expected_FP32_output_bytes': expected_bytes,
           'want_float': 1, 'is_prealloc': 0, 'manual_quantization_or_dequantization': False,
           'run_timeout_ms': timeout_ms, 'output_buffer_released': None}
    ledger.append(row)
    got, owned = False, None
    begin = time.monotonic_ns()
    try:
        code = api.rknn_inputs_set(context, 1, request)
        R.check_call(code, 'rknn_inputs_set')
        set_end = time.monotonic_ns()
        row['inputs_set_code'] = int(code)
        row['npu_run_api_called'] = True
        code = api.rknn_run(context, ct.byref(run))
        R.check_call(code, 'rknn_run')
        run_end = time.monotonic_ns()
        row['run_code'] = int(code)
        code = api.rknn_outputs_get(context, 1, outputs, ct.byref(received))
        R.check_call(code, 'rknn_outputs_get')
        got = True
        get_end = time.monotonic_ns()
        row['outputs_get_code'] = int(code)
        row.update(runtime_frame_id=int(run.frame_id), output_frame_id=int(received.frame_id),
                   SDK_output_pointer=int(outputs[0].buf or 0), output_bytes=int(outputs[0].size))
        if run.frame_id != received.frame_id:
            raise ValueError('Synchronous SDK run/output frame identity mismatch')
        if not outputs[0].buf or outputs[0].size != expected_bytes:
            raise ValueError('want_float SDK output must have the exact logical FP32 pointer/byte count')
        view = np.ctypeslib.as_array(ct.cast(outputs[0].buf, ct.POINTER(ct.c_float)),
                                   shape=(output_elements,))
        owned = view.copy().reshape(output_shape)
        if np.shares_memory(owned, view) or owned.ctypes.data == outputs[0].buf:
            raise ValueError('SDK output copy still aliases SDK-owned storage')
        copy_end = time.monotonic_ns()
        row.update(owned_output_pointer=int(owned.ctypes.data), output_owned_copy_before_release=True)
    except BaseException as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if got or outputs[0].buf:
            release_begin = time.monotonic_ns()
            try:
                code = api.rknn_outputs_release(context, 1, outputs)
            except BaseException as exc:
                row['output_buffer_released'] = False
                cleanup_errors.append(f'{role} SDK output release: {type(exc).__name__}: {exc}')
                raise
            release_end = time.monotonic_ns()
            row.update(outputs_release_code=int(code), output_buffer_released=code == 0,
                       outputs_release_seconds=(release_end - release_begin) / 1e9,
                       io_begin_monotonic_ns=begin, io_end_monotonic_ns=release_end,
                       io_total_seconds=(release_end - begin) / 1e9)
            if code != 0:
                cleanup_errors.append(f'{role} SDK output release returned {code}')
            R.check_call(code, 'rknn_outputs_release')
    row.update(input_set_seconds=(set_end - begin) / 1e9,
               run_api_seconds=(run_end - set_end) / 1e9,
               outputs_get_seconds=(get_end - run_end) / 1e9,
               output_validation_and_copy_seconds=(copy_end - get_end) / 1e9,
               pre_release_bookkeeping_seconds=(release_begin - copy_end) / 1e9,
               status='completed')
    return owned, row


def prepare_context(R, C, api, role, model, contexts, records, profile):
    R.require_board()
    context = R.CONTEXT()
    contexts[role] = context  # Include nonzero partial init in final destroy.
    row = {'role': role, 'model': str(model), 'context_destroyed': None,
           'init_flags': 8 if profile else 0, 'core_mask': 7,
           'tensor_attributes_validated': False}
    records[role] = row
    filename = ct.create_string_buffer(os.fsencode(model))
    begin = time.monotonic_ns()
    code = api.rknn_init(ct.byref(context), ct.cast(filename, ct.c_void_p), 0, row['init_flags'], None)
    row['init_code'] = int(code)
    R.check_call(code, role + ' rknn_init')
    row['init_seconds'] = (time.monotonic_ns() - begin) / 1e9
    if not context.value:
        raise ValueError('SDK initialization returned no context')
    code = api.rknn_set_core_mask(context, 7)
    row['core_mask_code'] = int(code)
    R.check_call(code, 'rknn_set_core_mask')
    version = R.query(api, context, 5, R.SDKVersion())
    row.update(sdk_version=bytes(version.api_version).decode(errors='replace'),
               driver_version=bytes(version.drv_version).decode(errors='replace'))
    if not row['sdk_version'].startswith('2.3.2') or row['driver_version'] != '0.7.2':
        raise ValueError('Precision probe requires runtime2.3.2 / driver0.7.2')
    count = R.query(api, context, 0, R.InputOutputNum())
    if count.n_input != 1 or count.n_output != 1:
        raise ValueError('Original paired graph must have exactly one input/output')
    for direction, command in [('input', 1), ('output', 2)]:
        record = R.attr_record(R.query(api, context, command, R.TensorAttr(index=0)))
        row[direction + '_attr'] = record
        row[direction + '_API_conversion_policy'] = C.validate_sdk_attr(record, direction)
        if role == 'nonquant_fp16' and record['type'] not in (0, 1):
            raise ValueError('Frozen nonquant comparator unexpectedly has integer logical IO')
    row['tensor_attributes_validated'] = True
    row['integer_logical_IO_observed'] = any(row[d + '_attr']['type'] in (2, 3) for d in ('input', 'output'))
    row['all_internal_operators_INT8_verified'] = False
    row['all_internal_operators_on_NPU_verified'] = False
    memory = R.MemSize()
    code = api.rknn_query(context, 6, ct.byref(memory), ct.sizeof(memory))
    row['model_memory_query_code'] = int(code)
    row['model_memory_bytes'] = ({name: int(getattr(memory, name))
                                for name, _ in memory._fields_ if name != 'reserved'} if code == 0 else None)


def destroy_contexts(R, api, contexts, records, cleanup_errors):
    """Destroy every nonzero context, including failed partial initializations."""
    if api is None:
        return
    for role, context in reversed(list(contexts.items())):
        if not context.value:
            continue
        try:
            code = api.rknn_destroy(context)
            records[role]['destroy_code'] = int(code)
            records[role]['context_destroyed'] = code == 0
            R.check_call(code, role + ' rknn_destroy')
        except BaseException as exc:
            cleanup_errors.append(f'{role} context destroy: {type(exc).__name__}: {exc}')


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


def save_array(output_dir, name, value, C):
    import numpy as np
    path = output_dir / name
    with path.open('xb') as stream:
        np.save(stream, value, allow_pickle=False)
    return {'file': name, 'bytes': path.stat().st_size, 'sha256': C.sha(path),
            'shape': list(value.shape), 'dtype': str(value.dtype),
            'tensor_bytes': int(value.nbytes), 'raw_sha256': C.raw_sha(value)}


def implementation_difference(ptq, nonquant):
    """Only a diagnostic; neither implementation replaces the Torch reference."""
    import numpy as np
    delta = ptq.astype(np.float64) - nonquant.astype(np.float64)
    return {'reference_for_original_FP32_gate': False,
            'array_equal': bool(np.array_equal(ptq, nonquant)),
            'max_abs_difference': float(np.max(np.abs(delta))),
            'difference_l2': float(np.linalg.norm(delta)),
            'rmse_difference': float(np.sqrt(np.mean(delta * delta)))}


def summarize(calls, evaluation_ids, normal):
    """Warm denominators contain three measured repeats for each heldout input."""
    result = {'normal_benchmark': normal, 'samples': {}, 'implementations': {}}
    for sample_id in evaluation_ids:
        result['samples'][sample_id] = {}
        for role in ROLES:
            rows = [r for r in calls if r['sample_id'] == sample_id and r['role'] == role]
            if len(rows) != 4 or [r['sample_call_index'] for r in rows] != list(range(4)):
                raise ValueError('Exactly first plus three measured calls per implementation/input required')
            warm = rows[1:]
            mean = sum(r['io_total_seconds'] for r in warm) / 3
            result['samples'][sample_id][role] = {
                'first_for_sample_io_seconds': rows[0]['io_total_seconds'],
                'first_for_sample_is_first_in_context': rows[0]['first_in_context'],
                'subsequent_call_count': 3,
                'subsequent_io_seconds': [r['io_total_seconds'] for r in warm],
                'subsequent_io_mean_seconds': mean,
                'all_outputs_byte_equal_to_first_for_sample': all(r['byte_equal_to_first_for_sample'] for r in rows),
                'strict_original_FP32_gate_passed': all(r['original_FP32_comparison']['strict_FP32_gate_passed'] for r in rows),
                'original_FP32_first_comparison': rows[0]['original_FP32_comparison']}
    for role in ROLES:
        rows = [r for r in calls if r['role'] == role and r['sample_call_index'] > 0]
        if len(rows) != 6:
            raise ValueError('Warm implementation mean must contain exactly six measured repeats')
        result['implementations'][role] = {
            'mean_denominator': 6, 'included_sdk_call_indices': [r['sdk_call_index'] for r in rows],
            'two_input_subsequent_io_mean_seconds': sum(r['io_total_seconds'] for r in rows) / 6,
            'strict_original_FP32_gate_passed': all(result['samples'][i][role]['strict_original_FP32_gate_passed'] for i in evaluation_ids)}
    if normal:
        baseline = result['implementations'][ROLES[0]]['two_input_subsequent_io_mean_seconds']
        ptq = result['implementations'][ROLES[1]]['two_input_subsequent_io_mean_seconds']
        result['PTQ_IO_difference'] = {'denominator_implementation': ROLES[0],
                                      'nonquant_minus_PTQ_seconds': baseline - ptq,
                                      'nonquant_minus_PTQ_percent_of_nonquant': 100 * (baseline - ptq) / baseline if baseline else None,
                                      'component_only': True, 'fresh_observation_pose_rate': False}
    else:
        result['profile_IO_is_not_a_normal_speed_comparison'] = True
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('contracts', 'capture-contracts', 'capture-dir', 'ptq-compiler', 'ptq-contracts',
                 'ptq-compile-receipt', 'ptq-model', 'nonquant-compile-receipt', 'nonquant-model',
                 'runtime-helper', 'sdk-header', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('contracts-sha256', 'capture-contracts-sha256', 'capture-sha256', 'ptq-compiler-sha256',
                 'ptq-contracts-sha256', 'ptq-compile-sha256', 'ptq-model-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--run-timeout-ms', type=int, default=30000)
    parser.add_argument('--collect-perf-detail', action='store_true')
    parser.add_argument('--diagnostic-allow-FP32-rejected', action='store_true',
                        help='Required: retain fixed original FP32 failures, never adopt a candidate')
    args = parser.parse_args(argv)
    # Reject a host before reading captured tensors or loading SDK source/library.
    if sys.platform != 'linux' or platform.machine().lower() not in ('aarch64', 'arm64'):
        parser.error('Actual original model SDK calls run only on the connected RK3588S')
    if not args.diagnostic_allow_FP32_rejected or not 1 <= args.run_timeout_ms <= 60000:
        parser.error('Explicit diagnostic gate and timeout1..60000 required')
    C = load_contract(args.contracts, args.contracts_sha256)
    pins = {str(Path(__file__).resolve()): sha(__file__)}
    C.pin(args.contracts, args.contracts_sha256, pins)
    R = C.source(args.runtime_helper, C.PARENT_RUNTIME_SHA, pins)
    compatible = R.require_board()
    output_dir = args.output_dir.resolve()
    if (Path.cwd().resolve() != C.BASE or output_dir == C.BASE / 'experiments'
            or not output_dir.is_relative_to(C.BASE / 'experiments')
            or output_dir.is_relative_to(args.capture_dir.resolve())):
        parser.error('Board project cwd and new output directory beneath experiments required')
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'v49_original_VGGT_pair_precision_SDK_AB', 'status': 'failed',
              'script_sha256': sha(__file__), 'contracts_sha256': args.contracts_sha256,
              'capture_contracts_sha256': args.capture_contracts_sha256,
              'capture_manifest_sha256': args.capture_sha256,
              'pair_export_sha256': C.PAIR_EXPORT_SHA, 'pair_graph_sha256': C.PAIR_GRAPH_SHA,
              'passed_board_ORT_receipt_sha256': C.PAIR_ORT_SHA,
              'ptq_compile_sha256': args.ptq_compile_sha256, 'ptq_model_sha256': args.ptq_model_sha256,
              'ptq_compiler_sha256': args.ptq_compiler_sha256,
              'nonquant_compile_sha256': C.NONQUANT_COMPILE_SHA, 'nonquant_model_sha256': C.NONQUANT_MODEL_SHA,
              'device_tree_compatible': compatible, 'machine': platform.machine(), 'python': sys.version,
              'inference_on_board_only': True, 'full_model_run': False, 'candidate_accepted': False,
              'real_time_validated': False, 'whole_model_quality_validated': False,
              'historical_RKNN_FP32_rejections_retained': True, 'rtol': C.RTOL, 'atol': C.ATOL,
              'original_FP32_reference': 'same heldout input original Torch frame0/global0 output',
              'normal_benchmark': not args.collect_perf_detail,
              'perf_detail_collection_enabled': args.collect_perf_detail,
              'profile_contexts_are_fresh_and_separate_from_normal': args.collect_perf_detail,
              'contexts_live_together': True, 'requested_sdk_call_count': 16,
              'unmeasured_warmup_calls': 0, 'abi': R.abi_record(), 'cpu_affinity_requested': sorted(CPUS),
              'sdk_io_interval': 'FP32 inputs_set + synchronous run + want_float outputs_get + owned FP32 copy + release',
              'timing_excludes': ['init', 'queries', 'input/reference checks and hashes', 'error metrics',
                                 'repeat comparison', 'NPY/JSON saves', 'source hashes', 'thermal/memory snapshots'],
              'limitations': ['Original frame0/global0 component only; remaining original layers/heads are outside scope',
                              'Same saved input repeated; no fresh observation, physical camera or full SLAM timing',
                              'Two heldout pairs from one sequence do not establish whole-model quality or generalization',
                              'Sixteen calls do not establish20Hz, P95, stability or8GB deployment',
                              'AB/BA reduces order bias; clocks, cooling and frequency are observed, not controlled',
                              'w8a8 build and integer boundary IO do not prove every internal operator is INT8/NPU',
                              'SDK performs boundary conversion; its quantization error is included in original FP32 comparison',
                              'Full checkpoint provenance is inherited from SHA-pinned original capture; not reloaded by this runner'],
              'calls': [], 'sdk_calls': [], 'context_records': {}, 'paired_implementation_differences': [],
              'input_records': {}, 'cleanup_errors': [], 'memory_rss_scope': 'two SDK contexts plus NumPy diagnostics'}
    original_affinity = os.sched_getaffinity(0)
    original_environment = {key: os.environ.get(key) for key in ENVIRONMENT}
    contexts, api, live_inputs = {}, None, {}
    started = time.monotonic_ns()
    try:
        if not CPUS <= original_affinity:
            raise ValueError('Requested CPU6/7 affinity unavailable')
        os.sched_setaffinity(0, CPUS)
        os.environ.update(ENVIRONMENT)
        report.update(cpu_affinity=sorted(os.sched_getaffinity(0)), environment=ENVIRONMENT)
        CC = C.source(args.capture_contracts, args.capture_contracts_sha256, pins)
        captured, samples, bound = CC.load_capture(args.capture_dir, args.capture_sha256)
        pins.update(bound)
        C.pin(args.ptq_compiler, args.ptq_compiler_sha256, pins)
        C.pin(args.ptq_contracts, args.ptq_contracts_sha256, pins)
        C.pin(args.ptq_compile_receipt, args.ptq_compile_sha256, pins)
        ptq = json.loads(args.ptq_compile_receipt.read_text())
        C.check_quantized_compile(ptq, args.ptq_compiler_sha256, args.capture_sha256,
                                  args.ptq_model_sha256, args.ptq_contracts_sha256, args.capture_contracts_sha256)
        evaluations = C.bind_compilation_to_capture(ptq, captured, samples)
        dataset = args.ptq_compile_receipt.resolve().parent / ptq['dataset']['file']
        C.pin(dataset, ptq['dataset']['sha256'], pins)
        expected_dataset = ''.join(row['path'] + '\n' for row in ptq['dataset']['rows']).encode('utf-8')
        if dataset.read_bytes() != expected_dataset or dataset.stat().st_size != ptq['dataset']['bytes']:
            raise ValueError('Actual calibration dataset file does not match its exact eight captured inputs')
        C.pin(args.nonquant_compile_receipt, C.NONQUANT_COMPILE_SHA, pins)
        nonquant = json.loads(args.nonquant_compile_receipt.read_text())
        C.check_nonquantized_compile(nonquant)
        models = {ROLES[0]: C.pin(args.nonquant_model, C.NONQUANT_MODEL_SHA, pins),
                  ROLES[1]: C.pin(args.ptq_model, args.ptq_model_sha256, pins)}
        if (models[ROLES[0]].stat().st_size != nonquant['output']['size']
                or models[ROLES[1]].stat().st_size != ptq['output']['size']):
            raise ValueError('Compiled original pair artifact byte count mismatch')
        C.pin(args.sdk_header, C.SDK_HEADER_SHA, pins)
        library = C.pin('/usr/lib/librknnrt.so', C.SDK_LIBRARY_SHA, pins)
        report.update(original_checkpoint=captured['checkpoint'], capture_script_sha256=captured['script_sha256'],
                      original_parameter_tensors=36, checkpoint_rehashed_by_this_runner=False,
                      evaluation_sample_ids=[s['sample_id'] for s in evaluations],
                      calibration_sample_ids=[r['sample_id'] for r in ptq['dataset']['rows']],
                      runtime_helper_sha256=C.PARENT_RUNTIME_SHA, ptq_configuration=ptq['configuration'],
                      ptq_input_transform=ptq['input_transform'], actual_calibration_receipt_bound=True,
                      memory_before_contexts=snapshot(R),
                      schedule=[{'sample_id': s['sample_id'], 'round_index': r, 'order': list(order)}
                                for i, s in enumerate(evaluations) for r, order in enumerate(schedule_for_sample(i))])
        api = R.bind_api(ct.CDLL(str(library)))
        for role in ROLES:
            prepare_context(R, C, api, role, models[role], contexts, report['context_records'], args.collect_perf_detail)
        report['memory_after_contexts_init'] = snapshot(R)
        import numpy as np
        report['numpy_version'] = np.__version__
        context_counts = {role: 0 for role in ROLES}
        for sample_index, sample in enumerate(evaluations):
            sample_id = sample['sample_id']
            inputs = CC.load_array(args.capture_dir, sample['artifacts']['input.npy'], C.INPUT_SHAPE, 'float32')
            reference = CC.load_array(args.capture_dir, sample['artifacts']['output.npy'], C.OUTPUT_SHAPE, 'float32')
            C.validate_inputs(inputs, reference)
            inputs.setflags(write=False)
            live_inputs[sample_id] = inputs
            before_raw = C.raw_sha(inputs)
            report['input_records'][sample_id] = {'input_raw_sha256_before': before_raw,
                                                 'original_reference_raw_sha256': C.raw_sha(reference),
                                                 'input_artifact': sample['artifacts']['input.npy'],
                                                 'original_reference_artifact': sample['artifacts']['output.npy'],
                                                 'source_images': sample['images']}
            first_outputs = {}
            for round_index, order in enumerate(schedule_for_sample(sample_index)):
                actuals = {}
                for role in order:
                    before = snapshot(R)
                    if C.raw_sha(inputs) != before_raw:
                        raise ValueError('Original heldout input changed before SDK submission')
                    actual, sdk = execute_sdk_call(R, api, contexts[role], inputs,
                                                  report['context_records'][role]['input_attr']['format'],
                                                  C.OUTPUT_SHAPE, args.run_timeout_ms, report['cleanup_errors'],
                                                  report['sdk_calls'], role)
                    sdk.update(sample_id=sample_id, round_index=round_index,
                               context_call_index=context_counts[role], sample_call_index=round_index,
                               first_in_context=context_counts[role] == 0,
                               input_raw_sha256_before=before_raw, input_raw_sha256_after=C.raw_sha(inputs))
                    if sdk['input_raw_sha256_after'] != before_raw:
                        raise ValueError('SDK mutated the original FP32 heldout input')
                    comparison = C.compare(actual, reference)
                    if role not in first_outputs:
                        first_outputs[role] = actual
                    entry = {'sample_id': sample_id, 'role': role, 'round_index': round_index,
                             'sample_call_index': round_index, 'context_call_index': context_counts[role],
                             'first_in_context': context_counts[role] == 0,
                             'first_for_sample': round_index == 0, 'sdk_call_index': sdk['sdk_call_index'],
                             'io_total_seconds': sdk['io_total_seconds'],
                             'original_FP32_comparison': comparison,
                             'byte_equal_to_first_for_sample': C.raw_sha(actual) == C.raw_sha(first_outputs[role]),
                             'input_unchanged': True, 'system_before': before, 'system_after': snapshot(R),
                             'output': save_array(output_dir, f'{sample_id}-{role}-call{round_index}.npy', actual, C)}
                    perf = R.PerfRun()
                    code = api.rknn_query(contexts[role], 4, ct.byref(perf), ct.sizeof(perf))
                    sdk.update(perf_run_query_code=int(code), sdk_run_duration_us=int(perf.run_duration) if code == 0 else None)
                    if args.collect_perf_detail:
                        detail = R.perf_detail(api, contexts[role])
                        detail_name = f'{sample_id}-{role}-call{round_index}-operators.txt'
                        raw = detail.pop('raw_text')
                        with (output_dir / detail_name).open('x', encoding='utf-8') as stream:
                            stream.write(raw)
                        detail.update(file=detail_name, saved_text_sha256=C.sha(output_dir / detail_name))
                        entry['separate_profile_detail'] = detail
                    report['calls'].append(entry)
                    context_counts[role] += 1
                    actuals[role] = actual
                report['paired_implementation_differences'].append({
                    'sample_id': sample_id, 'round_index': round_index, 'order': list(order),
                    'PTQ_minus_nonquant_implementation_diagnostic': implementation_difference(actuals[ROLES[1]], actuals[ROLES[0]])})
            report['input_records'][sample_id].update(input_raw_sha256_after=C.raw_sha(inputs), input_unchanged=C.raw_sha(inputs) == before_raw)
        report['summaries'] = summarize(report['calls'], report['evaluation_sample_ids'], not args.collect_perf_detail)
        report['status'] = 'completed_original_FP32_rejected_diagnostic_not_adopted'
        if all(r['strict_original_FP32_gate_passed'] for r in report['summaries']['implementations'].values()):
            report['status'] = 'completed_original_FP32_gate_diagnostic_not_adopted'
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        destroy_contexts(R, api, contexts, report['context_records'], report['cleanup_errors'])
        try:
            report['memory_after_contexts_destroy'] = snapshot(R)
            for sample_id, inputs in live_inputs.items():
                row = report['input_records'][sample_id]
                row.update(input_raw_sha256_after=C.raw_sha(inputs), input_unchanged=C.raw_sha(inputs) == row['input_raw_sha256_before'])
                if not row['input_unchanged']:
                    report['cleanup_errors'].append(f'{sample_id} original in-memory input changed')
            after, unchanged = C.recheck(pins)
            report.update(source_and_input_hashes=pins, source_and_input_hashes_after=after, source_and_inputs_unchanged=unchanged)
            if not unchanged:
                report['cleanup_errors'].append('Original source/capture/model/SDK bytes changed')
        except BaseException as exc:
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
        except BaseException as exc:
            report['cleanup_errors'].append(f'Process restoration: {type(exc).__name__}: {exc}')
        report['successful_synchronous_call_count'] = sum(r['status'] == 'completed' for r in report['sdk_calls'])
        report['npu_run_api_call_count'] = sum(r.get('npu_run_api_called') is True for r in report['sdk_calls'])
        report['completed_comparison_call_count'] = len(report['calls'])
        report['total_seconds_including_diagnostics'] = (time.monotonic_ns() - started) / 1e9
        if report['status'].startswith('completed_') and (
                report['successful_synchronous_call_count'] != 16 or len(report['calls']) != 16
                or len(report['paired_implementation_differences']) != 8
                or not all(r['context_destroyed'] is True for r in report['context_records'].values())):
            report['cleanup_errors'].append('Expected16 calls/8 pairs/all context cleanup not proven')
        if report['cleanup_errors']:
            report['status'] = 'failed'
        with (output_dir / 'run.json').open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
    print(json.dumps({'status': report['status'], 'error': report.get('error'),
                      'run_sha256': sha(output_dir / 'run.json')}, allow_nan=False), flush=True)
    return 0 if report['status'].startswith('completed_') else 2


if __name__ == '__main__':
    raise SystemExit(main())
