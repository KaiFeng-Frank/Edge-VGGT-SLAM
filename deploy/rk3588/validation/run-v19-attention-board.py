#!/usr/bin/env python3
"""Board-only synchronous full-head versus three-context attention probe.

The original complete-key attention is unchanged. Baseline uses core mask 7;
parallel uses three independent contexts, masks 1/2/4, and three Python workers.
Each measured round includes FP32 Q/K/V packing, submission, joining, SDK I/O,
owned output copies/releases and ordered head concatenation. Profiling must be
run in a separate process/output directory and is not a normal speed benchmark.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = Path('/mnt/vggt-data/vggt-slam2')
RUNTIME_SHA = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
EXPORTER_SHA = '7e5523916fbf36b78d6834c40cd6856155edc4ce3052f6a4dde8181a1806131b'
COMPILER_SHA = 'a40bf09e53ef9b378db4781be18f9e53f824afcc809c9009e0a65f4f0f91b4e5'
LIBRARY_SHA = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
CPUS = {6, 7}
INPUT_NAMES = ('query', 'key', 'value')
MODEL_NAMES = ('baseline', 'shard-00', 'shard-01', 'shard-02')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            digest.update(part)
    return digest.hexdigest()


def require_sha(path, expected):
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in '0123456789abcdef' for c in expected) or sha(path) != expected):
        raise ValueError(f'Required SHA256 mismatch: {path}')


def load_helper(name, expected):
    path = HERE / name
    require_sha(path, expected)
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_compile(receipt, onnx_sha, model_sha):
    if (receipt.get('status') != 'compiled_not_runtime_validated'
            or receipt.get('script_sha256') != COMPILER_SHA
            or receipt.get('model_inference_executed') is not False
            or receipt.get('input', {}).get('sha256') != onnx_sha
            or receipt.get('output', {}).get('sha256') != model_sha
            or receipt.get('released') is not True or receipt.get('release_error')
            or receipt.get('configuration') != dict(target_platform='rk3588', float_dtype='float16',
                                                     optimization_level=3, do_quantization=False)):
        raise ValueError('Compile receipt does not bind the unchanged floating attention graph/model')
    operations = receipt.get('operations', [])
    if ([row.get('name') for row in operations] != ['config', 'load_onnx', 'build', 'export_rknn']
            or any(row.get('code') not in (0, None) for row in operations)):
        raise ValueError('Compiler did not complete all required operations')


def check_export(receipt, capture_sha, capture, arrays, exporter):
    shape = tuple(arrays['query'].shape)
    plan = exporter.head_plan(shape)
    expected = dict(kind='original_VGGT_independent_head_attention_ONNX_export',
                    status='exported_and_structurally_checked', script_sha256=EXPORTER_SHA,
                    capture_sha256=capture_sha, original_shape=list(shape),
                    flattened_shape=[1, shape[0] * shape[1], shape[2], 64],
                    head_shard_plan=plan, host_model_inference_executed=False, host_ORT_executed=False,
                    production_changed=False, npu_used=False, learned_parameter_count=0,
                    graph_inference_executed=False, source_and_inputs_unchanged=True,
                    saved_reference_partition_inverse_bitwise_equal=True)
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError('Export is not the frozen original complete-key attention and head partition')
    pairs = [('block', 'block'), ('checkpoint', 'checkpoint'), ('selected_pair', 'selected_pair'),
             ('profile_sha256', 'profile_sha256'), ('source_pins', 'source_pins'),
             ('original_parameter_metadata', 'original_block_parameter_metadata'),
             ('original_array_metadata', 'original_block_array_metadata'),
             ('attention_semantics', 'attention_semantics'), ('flattening_order', 'flattening_order')]
    if any(receipt.get(a) != capture.get(b) for a, b in pairs):
        raise ValueError('Export and operand capture identity/semantics differ')
    rows = receipt.get('models', [])
    if [row.get('name') for row in rows] != list(MODEL_NAMES):
        raise ValueError('Exact baseline and three ordered shard models required')
    ranges = [(0, shape[0] * shape[1]), *[(row['begin'], row['end']) for row in plan]]
    for row, (begin, end) in zip(rows, ranges):
        want = [1, end - begin, shape[2], 64]
        if (row.get('begin') != begin or row.get('end') != end
                or row.get('independent_heads') != end - begin
                or row.get('input_shape') != want or row.get('output_shape') != want
                or row.get('input_names') != list(INPUT_NAMES) or row.get('output_names') != ['output']
                or row.get('complete_key_count') != shape[2] or row.get('token_or_key_partitioned') is not False
                or row.get('external_data_used') is not False
                or set(row.get('artifacts', {})) != {name + '.npy' for name in (*INPUT_NAMES, 'output')}):
            raise ValueError('Export model shape/name/complete-key/head ordering mismatch')
    return rows


def safe_artifact(directory, entry):
    name = entry.get('file')
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError('Relative export artifact path required')
    path = (directory / name).resolve(strict=True)
    if not path.is_relative_to(directory) or path.stat().st_size != entry.get('bytes'):
        raise ValueError('Export artifact escaped directory or changed size')
    require_sha(path, entry.get('sha256'))
    return path


def load_inputs(args, exporter, runtime):
    import numpy as np
    capture, arrays, pins = exporter.load_operands(args.capture_dir, args.capture_sha256)
    export_dir = args.export_dir.resolve(strict=True)
    export_path = export_dir / 'export.json'
    require_sha(export_path, args.export_sha256)
    exported = json.loads(export_path.read_text())
    rows = check_export(exported, args.capture_sha256, capture, arrays, exporter)
    selected = rows[:1] if args.mode == 'baseline' else rows[1:]
    count = len(selected)
    for values in (args.models, args.model_sha256, args.compile_receipts, args.compile_sha256):
        if len(values) != count:
            raise ValueError('Baseline needs one model/receipt/SHA; parallel needs three ordered entries')
    pins.update({str(export_path): args.export_sha256,
                 str(HERE / 'export-v19-attention.py'): EXPORTER_SHA,
                 str(HERE / 'run-v12-rknn-block-board.py'): RUNTIME_SHA,
                 str(HERE / 'compile-v12-rknn-block.py'): COMPILER_SHA,
                 str(Path(__file__).resolve()): sha(__file__)})
    require_sha(HERE / 'compile-v12-rknn-block.py', COMPILER_SHA)
    header = HERE.parent / 'reference/rknn_api_v2.3.2.h'
    require_sha(header, runtime.HEADER_SHA256)
    pins[str(header)] = runtime.HEADER_SHA256
    flat = {key: exporter.flatten_heads(value) for key, value in arrays.items()}
    specs = []
    for index, row in enumerate(selected):
        graph = safe_artifact(export_dir, row['onnx'])
        pins[str(graph)] = row['onnx']['sha256']
        for name in (*INPUT_NAMES, 'output'):
            entry = row['artifacts'][name + '.npy']
            path = safe_artifact(export_dir, entry)
            value = np.load(path, allow_pickle=False)
            wanted = flat[name][:, row['begin']:row['end']]
            if exporter.array_record(value) != entry.get('array') or not np.array_equal(value, wanted):
                raise ValueError(f'Exported {row["name"]}/{name} differs from exact captured head slice')
            pins[str(path)] = entry['sha256']
        compiled_path = args.compile_receipts[index].resolve(strict=True)
        model = args.models[index].resolve(strict=True)
        require_sha(compiled_path, args.compile_sha256[index])
        require_sha(model, args.model_sha256[index])
        compiled = json.loads(compiled_path.read_text())
        check_compile(compiled, row['onnx']['sha256'], args.model_sha256[index])
        if (not 0 < model.stat().st_size <= runtime.MAX_MODEL_BYTES
                or model.stat().st_size != compiled.get('output', {}).get('size')):
            raise ValueError('Compiled attention model size mismatch')
        pins[str(compiled_path)] = args.compile_sha256[index]
        pins[str(model)] = args.model_sha256[index]
        specs.append(dict(name=row['name'], begin=row['begin'], end=row['end'],
                          shape=tuple(row['input_shape']), model=model,
                          model_sha256=args.model_sha256[index], compile_sha256=args.compile_sha256[index],
                          onnx_sha256=row['onnx']['sha256'], core_mask=7 if count == 1 else 1 << index))
    return capture, exported, flat, specs, pins


def validate_attr(runtime, attr, shape, name, index):
    row = runtime.attr_record(attr)
    identity = (row['type'] == 1 and row['quantization_type'] == 2
                and row['zero_point'] == 0 and row['scale'] == 1.0)
    if (row['index'] != index or row['name'] != name or tuple(row['shape']) != tuple(shape)
            or row['n_elems'] != math.prod(shape)):
        raise ValueError(f'RKNN logical tensor identity/shape mismatch: {row}')
    if row['type'] not in (0, 1) or (row['quantization_type'] != 0 and not identity):
        raise ValueError(f'Only unquantized FP32/FP16 or exact FP16 identity affine is allowed: {row}')
    if (row['size'] != row['n_elems'] * (4 if row['type'] == 0 else 2)
            or row['format'] not in (0, 1, 3) or not math.isfinite(row['scale'])):
        raise ValueError(f'RKNN tensor bytes/layout/scale mismatch: {row}')
    return row


def comparison(runtime, actual, expected):
    import numpy as np
    row = runtime.comparison(actual, expected)
    errors = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    within = errors <= 1e-5 + 1e-4 * np.abs(expected.astype(np.float64))
    row.update(rtol=1e-4, atol=1e-5, elementwise_gate_passed=bool(within.all()),
               outside_tolerance=int((~within).sum()), elements=int(within.size),
               interpretation='Strict original FP32 element gate, independent of successful SDK execution')
    return row


def create_session(api, runtime, spec, flags, sessions, report):
    context = runtime.CONTEXT()
    item = dict(context=context, spec=spec, attrs=None, report=report)
    sessions.append(item)  # Even an init error can leave a nonzero owned context.
    filename = ct.create_string_buffer(os.fsencode(spec['model']))
    tick = time.perf_counter()
    runtime.check_call(api.rknn_init(ct.byref(context), ct.cast(filename, ct.c_void_p), 0, flags, None), 'rknn_init')
    report['init_seconds'] = time.perf_counter() - tick
    runtime.check_call(api.rknn_set_core_mask(context, spec['core_mask']), 'rknn_set_core_mask')
    report['core_mask_set_successfully'] = True
    version = runtime.query(api, context, 5, runtime.SDKVersion())
    report['sdk_version'] = bytes(version.api_version).decode(errors='replace')
    report['driver_version'] = bytes(version.drv_version).decode(errors='replace')
    if not report['sdk_version'].startswith('2.3.2'):
        raise ValueError('Attention runtime ABI requires RKNN 2.3.2')
    counts = runtime.query(api, context, 0, runtime.InputOutputNum())
    report['input_output_counts'] = dict(inputs=int(counts.n_input), outputs=int(counts.n_output))
    if counts.n_input != 3 or counts.n_output != 1:
        raise ValueError('Attention graph requires exactly query/key/value and one output')
    inputs = [runtime.query(api, context, 1, runtime.TensorAttr(index=i)) for i in range(3)]
    output = runtime.query(api, context, 2, runtime.TensorAttr(index=0))
    report['input_attrs'] = [runtime.attr_record(attr) for attr in inputs]
    report['output_attr'] = runtime.attr_record(output)
    report['tensor_attributes_validated'] = False
    for i, (name, attr) in enumerate(zip(INPUT_NAMES, inputs)):
        validate_attr(runtime, attr, spec['shape'], name, i)
    validate_attr(runtime, output, spec['shape'], 'output', 0)
    report['tensor_attributes_validated'] = True
    item['attrs'] = inputs
    memory = runtime.MemSize()
    code = api.rknn_query(context, 6, ct.byref(memory), ct.sizeof(memory))
    report['model_memory_query_code'] = int(code)
    report['model_memory_bytes'] = {name: int(getattr(memory, name)) for name, _ in memory._fields_
                                    if name != 'reserved'} if code == 0 else None
    return item


def execute_call(api, runtime, session, inputs, timeout_ms, row, barrier=None):
    """One owned context; no global runtime.SHAPE mutation and no async SDK API."""
    import numpy as np
    spec, context = session['spec'], session['context']
    request = (runtime.Input * 3)()
    for i, name in enumerate(INPUT_NAMES):
        value = inputs[name]
        if (value.dtype != np.float32 or tuple(value.shape) != spec['shape']
                or not value.flags.c_contiguous):
            raise ValueError('Packed C-contiguous FP32 Q/K/V shape mismatch')
        request[i] = runtime.Input(i, value.ctypes.data, value.nbytes, 0, 0, int(session['attrs'][i].fmt))
    outputs = (runtime.Output * 1)()
    outputs[0].want_float, outputs[0].is_prealloc, outputs[0].index = 1, 0, 0
    run = runtime.RunExtend(0, 0, timeout_ms, -1)
    received = runtime.OutputExtend()
    row.update(input_type='FP32', input_pass_through=0, want_float=1, is_prealloc=0,
               synchronous=True, run_timeout_ms=timeout_ms, worker_thread_id=threading.get_native_id(),
               worker_cpu_affinity=sorted(os.sched_getaffinity(0)), worker_enter_monotonic=time.monotonic())
    if barrier is not None:
        barrier.wait(timeout=10.0)
    begin = time.perf_counter()
    row['io_begin_monotonic'] = time.monotonic()
    got = False
    try:
        runtime.check_call(api.rknn_inputs_set(context, 3, request), 'rknn_inputs_set')
        set_end = time.perf_counter()
        row.update(run_api_called=True, run_begin_monotonic=time.monotonic())
        code = api.rknn_run(context, ct.byref(run))
        row['run_end_monotonic'] = time.monotonic()
        runtime.check_call(code, 'rknn_run')
        row['run_api_succeeded'] = True
        run_end = time.perf_counter()
        runtime.check_call(api.rknn_outputs_get(context, 1, outputs, ct.byref(received)), 'rknn_outputs_get')
        got = True
        get_end = time.perf_counter()
        if run.frame_id != received.frame_id:
            raise ValueError('Synchronous run/output frame identity mismatch')
        if not outputs[0].buf or outputs[0].size != math.prod(spec['shape']) * 4:
            raise ValueError('Invalid FP32 SDK output pointer/byte size')
        view = np.ctypeslib.as_array(ct.cast(outputs[0].buf, ct.POINTER(ct.c_float)), shape=(math.prod(spec['shape']),))
        actual = view.copy().reshape(spec['shape'])
        copy_end = time.perf_counter()
        row.update(input_set_seconds=set_end-begin, run_api_seconds=run_end-set_end,
                   outputs_get_seconds=get_end-run_end, output_copy_seconds=copy_end-get_end,
                   runtime_frame_id=int(run.frame_id), received_frame_id=int(received.frame_id),
                   output_bytes=int(outputs[0].size), output_copied_before_release=True)
    except Exception as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if got or outputs[0].buf:
            tick = time.perf_counter()
            code = api.rknn_outputs_release(context, 1, outputs)
            row.update(outputs_release_seconds=time.perf_counter()-tick,
                       output_buffer_released=code == 0, outputs_release_code=int(code))
            runtime.check_call(code, 'rknn_outputs_release')
        row.update(io_end_monotonic=time.monotonic(), io_total_seconds=time.perf_counter()-begin)
    return actual


def interval_overlap(rows, begin_key, end_key):
    intervals = [(row[begin_key], row[end_key]) for row in rows if begin_key in row and end_key in row]
    if len(intervals) != len(rows):
        return dict(complete=False, max_simultaneous=None, all_three_overlap_seconds=None)
    if any(not math.isfinite(a) or not math.isfinite(b) or b < a for a, b in intervals):
        raise ValueError('Invalid monotonic call interval')
    events = sorted([(a, 1) for a, _ in intervals] + [(b, -1) for _, b in intervals], key=lambda x: (x[0], x[1]))
    current = maximum = 0
    for _, change in events:
        current += change
        maximum = max(maximum, current)
    return dict(complete=True, max_simultaneous=maximum,
                all_three_overlap_seconds=max(0., min(b for _, b in intervals)-max(a for a, _ in intervals))
                if len(rows) == 3 else None,
                interpretation='Host blocking API intervals; not proof of concurrent NPU kernel scheduling')


def execute_round(api, runtime, sessions, flat, original_shape, timeout_ms, group, executor=None):
    import numpy as np
    begin = time.perf_counter()
    group['begin_monotonic'] = time.monotonic()
    packed = [{name: np.array(flat[name][:, item['spec']['begin']:item['spec']['end']],
                             dtype=np.float32, order='C', copy=True) for name in INPUT_NAMES} for item in sessions]
    pack_end = time.perf_counter()
    group['packing_seconds'] = pack_end - begin
    group['packed_input_bytes'] = sum(value.nbytes for inputs in packed for value in inputs.values())
    group['calls'] = [dict(context_index=i, name=item['spec']['name'], core_mask=item['spec']['core_mask'])
                      for i, item in enumerate(sessions)]
    if len(sessions) == 1:
        values = [execute_call(api, runtime, sessions[0], packed[0], timeout_ms, group['calls'][0])]
        joined = time.perf_counter()
        group['submit_seconds'] = 0.0
    else:
        if executor is None or len(sessions) != 3:
            raise ValueError('Three owned contexts and persistent executor required')
        barrier = threading.Barrier(3)
        futures = [executor.submit(execute_call, api, runtime, item, inputs, timeout_ms, row, barrier)
                   for item, inputs, row in zip(sessions, packed, group['calls'])]
        submitted = time.perf_counter()
        # Always join every submitted call before any context can be destroyed.
        values, errors = [], []
        for future in futures:
            try:
                values.append(future.result())
            except Exception as exc:
                errors.append(exc)
        joined = time.perf_counter()
        group.update(submit_seconds=submitted-pack_end, join_seconds=joined-submitted)
        if errors:
            group['worker_errors'] = [f'{type(exc).__name__}: {exc}' for exc in errors]
            raise errors[0]
    if len(sessions) == 1:
        actual = values[0].reshape(original_shape)
    else:
        actual = np.concatenate(values, axis=1).reshape(original_shape)
    end = time.perf_counter()
    group.update(concat_seconds=end-joined, submit_and_join_seconds=joined-pack_end,
                 end_to_end_seconds=end-begin, end_monotonic=time.monotonic(),
                 completed_synchronous_calls=len(values),
                 call_io_seconds_sum=sum(row['io_total_seconds'] for row in group['calls']),
                 io_overlap=interval_overlap(group['calls'], 'io_begin_monotonic', 'io_end_monotonic'),
                 run_api_overlap=interval_overlap(group['calls'], 'run_begin_monotonic', 'run_end_monotonic'))
    return actual


def destroy_sessions(api, runtime, sessions, cleanup_errors):
    for item in reversed(sessions):
        row, context = item['report'], item['context']
        row['context_destroyed'] = None
        if context.value:
            try:
                code = api.rknn_destroy(context)
                row['destroy_return_code'] = int(code)
                runtime.check_call(code, 'rknn_destroy')
                row['context_destroyed'] = True
            except Exception as exc:
                row['context_destroyed'] = False
                cleanup_errors.append(f'{item["spec"]["name"]}: {type(exc).__name__}: {exc}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('baseline', 'parallel'), required=True)
    for name in ('capture-dir', 'export-dir', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('capture-sha256', 'export-sha256'):
        parser.add_argument('--' + name, required=True)
    for name in ('models', 'compile-receipts'):
        parser.add_argument('--' + name, type=Path, nargs='+', required=True)
    for name in ('model-sha256', 'compile-sha256'):
        parser.add_argument('--' + name, nargs='+', required=True)
    parser.add_argument('--library', type=Path, default=Path('/usr/lib/librknnrt.so'))
    parser.add_argument('--run-timeout-ms', type=int, default=30000)
    parser.add_argument('--collect-perf-detail', action='store_true')
    args = parser.parse_args(argv)
    runtime = load_helper('run-v12-rknn-block-board.py', RUNTIME_SHA)
    compatible = runtime.require_board()  # No output creation or SDK before the hardware guard.
    if not 1 <= args.run_timeout_ms <= 60000:
        parser.error('run-timeout-ms must be within 1..60000')
    output = args.output.resolve()
    if os.path.lexists(args.output):
        parser.error('Refusing an existing output or dangling symlink')
    if output == BASE / 'experiments' or not output.is_relative_to(BASE / 'experiments'):
        parser.error('New output must be strictly below board experiments')
    protected_dirs = [args.capture_dir.resolve(), args.export_dir.resolve()]
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in protected_dirs):
        parser.error('Output must not overlap capture or export directory')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(kind='original_VGGT_complete_key_attention_RKNN_board_probe', status='failed',
                  script_sha256=sha(__file__), runtime_helper_sha256=RUNTIME_SHA,
                  exporter_sha256=EXPORTER_SHA, compiler_sha256=COMPILER_SHA,
                  mode=args.mode, core_masks=[7] if args.mode == 'baseline' else [1, 2, 4],
                  first_round_count=1, subsequent_measured_round_count=3, unmeasured_warmup_calls=0,
                  async_enabled=False, init_flags=8 if args.collect_perf_detail else 0,
                  perf_detail_collection_enabled=args.collect_perf_detail,
                  normal_benchmark_timing=not args.collect_perf_detail, abi=runtime.abi_record(),
                  inference_on_board_only=True, production_changed=False, full_model_run=False, ORB_used=False,
                  npu_run_api_called=False, npu_run_api_call_count=0, all_operators_on_npu_verified=False,
                  realtime_validated=False, whole_model_quality_validated=False,
                  endpoint='packing + worker submit/join + synchronous SDK I/O + ordered head concat',
                  endpoint_excludes=['context initialization', 'error metrics', 'hashing', 'output file writes',
                                     'SDK perf queries/detail collection'],
                  timing_limits=['Four rounds are not a statistical performance or real-time validation',
                                 'Parallel host API intervals do not establish concurrent NPU kernel execution',
                                 'Profiling flags add overhead; compare normal receipts only'],
                  device_tree_compatible=compatible, contexts=[], rounds=[], output_arrays=[], cleanup_errors=[])
    api, executor, sessions, pins = None, None, [], {}
    affinity = os.sched_getaffinity(0)
    env_keys = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
    old_env = {key: os.environ.get(key) for key in env_keys}
    try:
        if not CPUS <= affinity:
            raise RuntimeError(f'CPUs 6,7 not allowed: {sorted(affinity)}')
        os.sched_setaffinity(0, CPUS)
        for key in env_keys:
            os.environ[key] = '2'
        exporter = load_helper('export-v19-attention.py', EXPORTER_SHA)
        capture, exported, flat, specs, pins = load_inputs(args, exporter, runtime)
        library = args.library.resolve(strict=True)
        require_sha(library, LIBRARY_SHA)
        pins[str(library)] = LIBRARY_SHA
        report.update(capture_sha256=args.capture_sha256, export_sha256=args.export_sha256,
                      block=capture['block'], original_shape=exported['original_shape'],
                      flattened_shape=exported['flattened_shape'], head_shard_plan=exported['head_shard_plan'],
                      attention_semantics=exported['attention_semantics'], source_and_input_hashes=dict(pins),
                      cpu_affinity=sorted(os.sched_getaffinity(0)), threads=2,
                      memory_before_sdk=runtime.process_memory(), native_library_sha256=LIBRARY_SHA)
        api = runtime.bind_api(ct.CDLL(str(library)))
        for spec in specs:
            row = {key: value for key, value in spec.items() if key != 'model'}
            row.update(model_file=str(spec['model']), context_destroyed=None)
            report['contexts'].append(row)
            create_session(api, runtime, spec, report['init_flags'], sessions, row)
        if len(sessions) == 3:
            executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix='v19-rknn-attention')
        import numpy as np
        expected = flat['output'].reshape(exported['original_shape'])
        first = None
        for i in range(4):
            group = dict(round_index=i, phase='first' if i == 0 else 'subsequent')
            report['rounds'].append(group)
            actual = execute_round(api, runtime, sessions, flat, tuple(exported['original_shape']),
                                   args.run_timeout_ms, group, executor)
            group['versus_cpu_reference'] = comparison(runtime, actual, expected)
            if first is None:
                first = actual.copy()
            group['versus_first_round'] = comparison(runtime, actual, first)
            for item, call in zip(sessions, group['calls']):
                perf = runtime.PerfRun()
                code = api.rknn_query(item['context'], 4, ct.byref(perf), ct.sizeof(perf))
                call['perf_run_query_code'] = int(code)
                call['sdk_run_duration_us'] = int(perf.run_duration) if code == 0 else None
                if args.collect_perf_detail:
                    call['perf_detail'] = runtime.perf_detail(api, item['context'])
            path = output / f'actual_{i:02d}.npy'
            with path.open('xb') as stream:
                np.save(stream, actual, allow_pickle=False)
            report['output_arrays'].append(dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path),
                                                array=exporter.array_record(actual)))
        warm = [row['end_to_end_seconds'] for row in report['rounds'][1:]]
        report['subsequent_end_to_end_seconds'] = dict(values=warm, mean=float(np.mean(warm)),
                                                       median=float(np.median(warm)), min=min(warm), max=max(warm))
        report['strict_FP32_reference_gate_all_rounds_passed'] = all(
            group['versus_cpu_reference']['elementwise_gate_passed'] for group in report['rounds'])
        report['execution_completed_does_not_mean_quality_passed'] = True
        report['status'] = 'completed'
    except Exception as exc:
        report.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        if executor is not None:
            try:
                executor.shutdown(wait=True, cancel_futures=True)
                report['workers_joined'] = True
            except Exception as exc:
                report['cleanup_errors'].append(f'Worker shutdown: {type(exc).__name__}: {exc}')
        else:
            report['workers_joined'] = True
        if api is not None:
            destroy_sessions(api, runtime, sessions, report['cleanup_errors'])
        if pins:
            try:
                after = {path: sha(path) for path in pins}
                report.update(source_and_input_hashes_after=after, source_and_inputs_unchanged=after == pins)
                if after != pins:
                    report['cleanup_errors'].append('Source/input/runtime hash changed')
            except Exception as exc:
                report['cleanup_errors'].append(f'Source recheck: {type(exc).__name__}: {exc}')
        try:
            os.sched_setaffinity(0, affinity)
            report['cpu_affinity_restored'] = os.sched_getaffinity(0) == affinity
        except Exception as exc:
            report['cleanup_errors'].append(f'Affinity restore: {type(exc).__name__}: {exc}')
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        report['memory_final'] = runtime.process_memory()
        calls = [call for row in report['rounds'] for call in row.get('calls', [])]
        report['npu_run_api_call_count'] = sum(call.get('run_api_called', False) for call in calls)
        report['npu_run_api_called'] = report['npu_run_api_call_count'] > 0
        report['successful_synchronous_call_count'] = sum(
            call.get('output_copied_before_release', False) and call.get('output_buffer_released', False) for call in calls)
        report['output_release_attempt_count'] = sum('outputs_release_code' in call for call in calls)
        report['total_seconds'] = time.monotonic() - started
        if report['cleanup_errors'] or any('error' in call for call in calls):
            report['status'] = 'failed'
        path = output / 'run.json'
        with path.open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
    print(json.dumps(dict(status=report['status'], error=report.get('error'), output=str(path), sha256=sha(path))), flush=True)
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
