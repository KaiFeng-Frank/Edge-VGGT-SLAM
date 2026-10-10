#!/usr/bin/env python3
"""Board-only CPU ORT check of original full and three head-shard graphs.

Uses the exact captured post-QK-norm/RoPE operands and original CPU SDPA output.
All graphs retain the complete key domain. One sequential numerical call per
graph is saved; this is not a CPU/NPU speed, concurrency or full-model test.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

sys.dont_write_bytecode = True
BASE = Path('/mnt/vggt-data/vggt-slam2')
SOURCE = BASE / 'src'
HERE = Path(__file__).resolve().parent
EXPORTER_NAME = 'export-v19-attention.py'
EXPORTER_SHA = '7e5523916fbf36b78d6834c40cd6856155edc4ce3052f6a4dde8181a1806131b'
RTOL, ATOL, CALLS = 1e-4, 1e-5, 1
INPUT_NAMES = ('query', 'key', 'value')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def require_sha(path, expected):
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in '0123456789abcdef' for c in expected) or sha(path) != expected):
        raise ValueError(f'Required SHA256 mismatch: {path}')


def load_exporter():
    path = HERE / EXPORTER_NAME
    require_sha(path, EXPORTER_SHA)
    spec = importlib.util.spec_from_file_location('_v19_attention_checked_exporter', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_inputs(args, np):
    exporter = load_exporter()
    capture, arrays, pins = exporter.load_operands(args.capture_dir, args.capture_sha256)
    export_path = args.export_dir / 'export.json'
    require_sha(export_path, args.export_sha256)
    receipt = json.loads(export_path.read_text())
    shape = tuple(arrays['query'].shape)
    flat = {name: exporter.flatten_heads(value) for name, value in arrays.items()}
    plan = exporter.head_plan(shape)
    checks = dict(kind='original_VGGT_independent_head_attention_ONNX_export',
                  status='exported_and_structurally_checked', script_sha256=EXPORTER_SHA,
                  capture_sha256=args.capture_sha256, block=capture['block'],
                  original_shape=list(shape), flattened_shape=list(flat['query'].shape),
                  flattening_order=capture['flattening_order'], attention_semantics=capture['attention_semantics'],
                  head_shard_plan=plan, checkpoint=capture['checkpoint'], selected_pair=capture['selected_pair'],
                  profile_sha256=capture['profile_sha256'], source_pins=capture['source_pins'],
                  original_parameter_metadata=capture['original_block_parameter_metadata'],
                  original_array_metadata=capture['original_block_array_metadata'],
                  frozen_helpers={exporter.CAPTURE_HELPER_NAME: exporter.CAPTURE_HELPER_SHA,
                                  exporter.LOADER_NAME: exporter.LOADER_SHA},
                  learned_parameter_count=0, host_model_inference_executed=False, host_ORT_executed=False,
                  graph_inference_executed=False, saved_reference_partition_inverse_bitwise_equal=True,
                  source_and_inputs_unchanged=True)
    if any(receipt.get(key) != value for key, value in checks.items()):
        raise ValueError('Export provenance or original complete-key layout mismatch')
    if receipt.get('block_reference_checks') != {key: capture[key] for key in (
            'unhooked_vs_frozen_reference', 'captured_vs_unhooked', 'captured_vs_frozen_reference')}:
        raise ValueError('Export original block-reference proof changed')
    models = receipt.get('models', [])
    expected_specs = [dict(name='baseline', kind='full_independent_head_baseline', begin=0,
                           end=shape[0]*shape[1], independent_heads=shape[0]*shape[1], index=None),
                      *[dict(name=f'shard-{row["index"]:02d}', kind='head_shard', **row) for row in plan]]
    if len(models) != 4:
        raise ValueError('Exactly baseline and three ordered head-shard graphs required')
    groups = []
    for row, spec in zip(models, expected_specs):
        expected_shape = [1, spec['independent_heads'], shape[2], 64]
        contract = dict(**{key: value for key, value in spec.items() if key != 'input_shape'},
                        input_names=list(INPUT_NAMES), output_names=['output'],
                        input_shape=expected_shape, output_shape=expected_shape,
                        complete_key_count=shape[2], token_or_key_partitioned=False,
                        node_count=5, op_counts={'Transpose': 1, 'MatMul': 2, 'Mul': 1, 'Softmax': 1},
                        external_data_used=False)
        if any(row.get(key) != value for key, value in contract.items()):
            raise ValueError('Export graph/head-range/complete-key contract mismatch')
        model_entry = row.get('onnx', {})
        expected_file = spec['name'] + '/attention.onnx'
        if model_entry.get('file') != expected_file:
            raise ValueError('Unexpected ONNX artifact path')
        model_path = args.export_dir / expected_file
        require_sha(model_path, model_entry.get('sha256'))
        if model_path.stat().st_size != model_entry.get('bytes'):
            raise ValueError('ONNX byte count mismatch')
        pins[str(model_path)] = model_entry['sha256']
        artifacts = row.get('artifacts', {})
        if set(artifacts) != set(name + '.npy' for name in exporter.ARRAY_KEYS):
            raise ValueError('Exact exported Q/K/V/reference files required')
        values = {}
        for name in exporter.ARRAY_KEYS:
            entry = artifacts[name + '.npy']
            expected_file = spec['name'] + '/' + name + '.npy'
            if entry.get('file') != expected_file:
                raise ValueError('Unexpected exported operand path')
            path = args.export_dir / expected_file
            require_sha(path, entry.get('sha256'))
            if path.stat().st_size != entry.get('bytes'):
                raise ValueError('Exported operand byte count mismatch')
            value = np.load(path, allow_pickle=False)
            if exporter.array_record(value) != entry.get('array'):
                raise ValueError('Exported operand raw shape/type/SHA mismatch')
            original_slice = flat[name][:, spec['begin']:spec['end']]
            if not np.array_equal(value, original_slice):
                raise ValueError('Exported operand values differ from exact original head slice')
            values[name] = value
            pins[str(path)] = entry['sha256']
        groups.append(dict(name=spec['name'], shape=tuple(expected_shape),
                           inputs={name: values[name] for name in INPUT_NAMES}, reference=values['output'],
                           model_bytes=model_path.read_bytes(), model_sha256=model_entry['sha256']))
    pins[str(export_path)] = args.export_sha256
    pins[str(HERE / EXPORTER_NAME)] = EXPORTER_SHA
    pins[str(Path(__file__).resolve())] = sha(__file__)
    return exporter, capture, arrays, groups, pins


def check_session(session, shape):
    if session.get_providers() != ['CPUExecutionProvider']:
        raise ValueError('Only CPUExecutionProvider is permitted')
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if [value.name for value in inputs] != list(INPUT_NAMES) or [value.name for value in outputs] != ['output']:
        raise ValueError('Exactly query/key/value and one output, in pinned order, are required')
    if any(value.type != 'tensor(float)' or value.shape != list(shape) for value in [*inputs, *outputs]):
        raise ValueError('Fixed original FP32 head/token shapes are required')
    session.disable_fallback()


def compare(actual, expected, np):
    if (actual.dtype != np.float32 or expected.dtype != np.float32 or actual.shape != expected.shape
            or not actual.size or not np.isfinite(actual).all() or not np.isfinite(expected).all()):
        raise ValueError('Finite same-shape CPU FP32 output required')
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    reference = expected.astype(np.float64)
    absolute = np.abs(difference)
    within = absolute <= ATOL + RTOL * np.abs(reference)
    return dict(finite=True, array_equal=bool(np.array_equal(actual, expected)),
                max_abs=float(absolute.max()), rmse=float(np.sqrt(np.mean(difference*difference))),
                relative_l2=float(np.linalg.norm(difference) / max(float(np.linalg.norm(reference)), 1e-30)),
                passed=bool(within.all()), outside_tolerance=int((~within).sum()), rtol=RTOL, atol=ATOL)


def save_actual(value, filename, output, np):
    path = output / filename
    owned = np.array(value, copy=True, order='C')
    with path.open('xb') as stream:
        np.save(stream, owned, allow_pickle=False)
    return owned, dict(file=filename, bytes=path.stat().st_size, sha256=sha(path),
                       shape=list(owned.shape), dtype=str(owned.dtype), finite=bool(np.isfinite(owned).all()),
                       raw_sha256=hashlib.sha256(memoryview(owned).cast('B')).hexdigest())


def execute_checks(sessions, groups, exporter, arrays, output, report, np):
    all_checks_passed = True
    if len(sessions) != 4 or len(groups) != 4:
        raise ValueError('Exactly four full/head-shard sessions are required')
    report['successful_session_run_count'] = 0
    report['runs'] = []
    for index in range(CALLS):
        results, row = [], dict(index=index, phase='single_numerical_check', graphs=[])
        report['runs'].append(row)
        for session, group in zip(sessions, groups):
            timer = time.perf_counter()
            returned = session.run(['output'], group['inputs'])
            seconds = time.perf_counter() - timer
            if len(returned) != 1:
                raise ValueError('One actual output is required')
            report['successful_session_run_count'] += 1
            value, artifact = save_actual(returned[0], f'{group["name"]}-actual-{index:02d}.npy', output, np)
            graph_row = dict(name=group['name'], session_run_seconds=seconds, actual=artifact)
            row['graphs'].append(graph_row)
            check = compare(value, group['reference'], np)
            graph_row['versus_original_CPU_SDPA'] = check
            all_checks_passed &= check['passed']
            results.append(value)
        full, shards = results[0], results[1:]
        restored = exporter.restore_heads(shards, tuple(arrays['output'].shape))
        combined, artifact = save_actual(restored, f'combined-original-layout-actual-{index:02d}.npy', output, np)
        row['combined'] = dict(actual=artifact,
                               versus_original_CPU_SDPA=compare(combined, arrays['output'], np),
                               versus_full_ORT=compare(combined, full.reshape(combined.shape), np))
        all_checks_passed &= all(value['passed'] for name, value in row['combined'].items() if name != 'actual')
    report['all_full_shard_and_combined_checks_passed'] = bool(all_checks_passed)
    report['repeatability_validated'] = False
    if not all_checks_passed:
        raise ValueError('Predeclared original-SDPA gate failed; actual outputs saved')


def require_board(args):
    if platform.machine() != 'aarch64' or Path.cwd().resolve() != BASE:
        raise ValueError('Run only on the board from /mnt/vggt-data/vggt-slam2')
    if Path(__file__).resolve().parents[3] != SOURCE:
        raise ValueError('Use the deployed helper below board src/deploy/rk3588/validation')
    for value in (args.capture_dir, args.export_dir):
        if not value.resolve().is_relative_to(BASE / 'experiments'):
            raise ValueError('Capture and export inputs must be on the board experiments volume')
    if os.path.lexists(args.output):
        raise ValueError('Refusing an existing output path or symlink')
    output = args.output.resolve()
    if output == BASE / 'experiments' or not output.is_relative_to(BASE / 'experiments'):
        raise ValueError('Output must be strictly below the board experiments directory')
    return output


def run_check(args):
    report = dict(kind='original_VGGT_attention_full_and_shards_board_ORT_check', status='started',
                  script_sha256=sha(__file__), exporter_sha256=EXPORTER_SHA,
                  capture_sha256=args.capture_sha256, export_sha256=args.export_sha256,
                  inference_on_board_only=True, cpu_ORT_used=True, npu_used=False, ORB_used=False,
                  production_changed=False, original_scale_and_complete_key_domain=True,
                  whole_model_quality_validated=False, realtime_validated=False,
                  planned_calls_per_graph=CALLS, planned_total_session_runs=4*CALLS,
                  gate=dict(rtol=RTOL, atol=ATOL), cleanup_errors=[],
                  timing_scope='CPU session.run only, excludes init/operand load/copy/compare/hash/save; sequential shards, not an NPU speed benchmark')
    started, sessions, pins = time.monotonic(), [], None
    allowed = os.sched_getaffinity(0)
    try:
        if not {6, 7}.issubset(allowed):
            raise ValueError('CPU6,7 must be in the process allowed CPU set')
        os.sched_setaffinity(0, {6, 7})
        enforced = dict(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
        report['environment_before'] = {key: os.environ.get(key) for key in enforced}
        os.environ.update(enforced)
        import numpy as np
        import onnxruntime as ort
        exporter, capture, arrays, groups, pins = prepare_inputs(args, np)
        report.update(block=capture['block'], original_shape=capture['attention_shape'],
                      source_and_input_hashes=dict(pins), head_shard_plan=exporter.head_plan(tuple(arrays['query'].shape)),
                      checkpoint=capture['checkpoint'], selected_pair=capture['selected_pair'],
                      original_block_reference_checks={key: capture[key] for key in (
                          'unhooked_vs_frozen_reference', 'captured_vs_unhooked', 'captured_vs_frozen_reference')})
        report['runtime'] = dict(python=sys.version, numpy=np.__version__, onnxruntime=ort.__version__,
                                 cpu_affinity=sorted(os.sched_getaffinity(0)), intra_op_threads=2, inter_op_threads=1,
                                 provider='CPUExecutionProvider', environment={key: os.environ.get(key) for key in enforced})
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        timer = time.perf_counter()
        for group in groups:
            # Byte-only models; the pinned exporter contains no external data.
            sessions.append(ort.InferenceSession(group['model_bytes'], sess_options=options,
                                                providers=['CPUExecutionProvider']))
            check_session(sessions[-1], group['shape'])
        report['four_session_construction_seconds'] = time.perf_counter() - timer
        execute_checks(sessions, groups, exporter, arrays, args.output, report, np)
        report['status'] = 'passed_original_SDPA_full_and_three_shards_FP32_gate'
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        sessions.clear()
        gc.collect()
        report['ORT_session_references_released'] = True
        if pins is not None:
            try:
                after = {path: sha(path) for path in pins}
                report['source_and_input_hashes_after'] = after
                report['source_and_inputs_unchanged'] = after == pins
                if not report['source_and_inputs_unchanged']:
                    raise ValueError('A frozen input/source changed during the check')
            except BaseException as exc:
                report['cleanup_errors'].append(str(exc))
        try:
            os.sched_setaffinity(0, allowed)
            report['cpu_affinity_restored'] = os.sched_getaffinity(0) == allowed
        except BaseException as exc:
            report['cleanup_errors'].append(f'CPU affinity restoration: {exc}')
        if report['cleanup_errors']:
            report['status'] = 'failed'
        report['total_seconds_including_validation_sessions_checks_save_cleanup'] = time.monotonic()-started
        report['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.
        report['rss_scope'] = 'process high-water mark; may include inherited pre-exec history'
        with (args.output/'check.json').open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--capture-sha256', required=True)
    parser.add_argument('--export-dir', type=Path, required=True)
    parser.add_argument('--export-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        args.output = require_board(args)
    except ValueError as exc:
        parser.error(str(exc))
    args.capture_dir, args.export_dir = args.capture_dir.resolve(), args.export_dir.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = run_check(args)
    print(json.dumps(dict(status=report['status'], receipt=str(args.output/'check.json'),
                          sha256=sha(args.output/'check.json'), error=report.get('error'))))
    return 0 if report['status'] == 'passed_original_SDPA_full_and_three_shards_FP32_gate' else 1


if __name__ == '__main__':
    sys.exit(main())
