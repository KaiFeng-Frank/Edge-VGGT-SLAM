#!/usr/bin/env python3
"""Two heldout original VGGT pair inputs, one CPU ORT run each on RK3588S.

This separate pre-PTQ check preserves the compiled-source freeze. It does not
execute the Torch model, RKNN SDK, calibration or the complete VGGT pipeline.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

sys.dont_write_bytecode = True
QUANT_CONTRACTS_SHA = '47237b1a1e1614822c75180efd15be3e5c708b610f587f698fe9b37fcf1bc649'
CAPTURE_CONTRACTS_SHA = '6ce1f87f1683e8082a4d567718acb853101e5ee6bbaeef1ced02993a8e3133c8'
ENVIRONMENT = {'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '2'}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def raw_sha(value):
    return hashlib.sha256(memoryview(value).cast('B')).hexdigest()


def load_contract(path, expected):
    if sha(path) != expected:
        raise ValueError('Frozen heldout-check dependency SHA mismatch')
    spec = importlib.util.spec_from_file_location('_v49_heldout_' + Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_session(session, C):
    session.disable_fallback()
    if session.get_providers() != ['CPUExecutionProvider']:
        raise ValueError('Heldout checks require only the CPUExecutionProvider')
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError('Exactly one original pair input/output required')
    for node, name, shape in ((inputs[0], 'input', C.INPUT_SHAPE),
                              (outputs[0], 'output', C.OUTPUT_SHAPE)):
        if node.name != name or node.type != 'tensor(float)' or node.shape != list(shape):
            raise ValueError(f'Original pair ORT {name} metadata mismatch')


def save_array(directory, name, value):
    import numpy as np
    path = Path(directory) / name
    with path.open('xb') as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return {'file': name, 'bytes': path.stat().st_size, 'sha256': sha(path),
            'shape': list(value.shape), 'dtype': str(value.dtype),
            'tensor_bytes': value.nbytes, 'raw_sha256': raw_sha(value)}


def run_heldout(session, samples, capture_dir, K, P, C, output_dir, report):
    """Board gate also covers direct calls; tiny tests supply a fake session."""
    C.require_board()
    import numpy as np
    evals = [sample for sample in samples if sample.get('split') == 'evaluation']
    if (len(samples) != 10 or len(evals) != 2
            or sum(s.get('split') == 'calibration' for s in samples) != 8
            or len({s['sample_id'] for s in samples}) != 10
            or P.RTOL != 1e-4 or P.ATOL != 1e-5):
        raise ValueError('Unchanged 8 calibration / 2 evaluation split and strict FP32 gate required')
    check_session(session, C)
    for index, sample in enumerate(evals):
        row = {'sample_id': sample['sample_id'], 'split': 'evaluation',
               'phase': 'first' if index == 0 else 'subsequent_same_session',
               'images': sample['images'], 'timings_seconds': {}}
        report['calls'].append(row)
        timer = time.monotonic_ns()
        tokens = K.load_array(capture_dir, sample['artifacts']['input.npy'], C.INPUT_SHAPE, 'float32')
        reference = K.load_array(capture_dir, sample['artifacts']['output.npy'], C.OUTPUT_SHAPE, 'float32')
        before, reference_before = raw_sha(tokens), raw_sha(reference)
        row.update(original_input=sample['artifacts']['input.npy'],
                   original_Torch_reference=sample['artifacts']['output.npy'])
        row['timings_seconds']['load_saved_arrays_and_initial_hashes'] = (time.monotonic_ns() - timer) / 1e9
        timer = time.monotonic_ns()
        row['used_input'] = save_array(output_dir, sample['sample_id'] + '-used-input.npy', tokens)
        if row['used_input']['raw_sha256'] != before:
            raise ValueError('Saved actual ORT input differs from original captured input')
        row['timings_seconds']['save_used_input_with_hashes'] = (time.monotonic_ns() - timer) / 1e9
        report['attempted_session_run_count'] += 1
        timer = time.monotonic_ns()
        returned = session.run(['output'], {'input': tokens})
        row['timings_seconds']['session_run'] = (time.monotonic_ns() - timer) / 1e9
        report['successful_session_run_count'] += 1
        timer = time.monotonic_ns()
        if (not isinstance(returned, list) or len(returned) != 1
                or not isinstance(returned[0], np.ndarray) or returned[0].dtype != np.float32
                or returned[0].shape != C.OUTPUT_SHAPE or not np.isfinite(returned[0]).all()):
            raise ValueError('Actual ORT output must be finite FP32 with the original complete shape')
        actual = np.array(returned[0], copy=True, order='C')
        del returned
        row['timings_seconds']['validate_and_copy_owned_output'] = (time.monotonic_ns() - timer) / 1e9
        timer = time.monotonic_ns()
        row['output'] = save_array(output_dir, sample['sample_id'] + '-actual-output.npy', actual)
        row['timings_seconds']['save_complete_output_with_hashes'] = (time.monotonic_ns() - timer) / 1e9
        timer = time.monotonic_ns()
        row['versus_same_sample_original_Torch_FP32'] = P.compare(actual, reference)
        row['timings_seconds']['strict_FP32_comparison'] = (time.monotonic_ns() - timer) / 1e9
        timer = time.monotonic_ns()
        row['input_tensor_unchanged'] = raw_sha(tokens) == before
        row['reference_tensor_unchanged'] = raw_sha(reference) == reference_before
        row['timings_seconds']['post_call_input_reference_hashes'] = (time.monotonic_ns() - timer) / 1e9
        if not row['input_tensor_unchanged'] or not row['reference_tensor_unchanged']:
            raise ValueError('ORT check mutated an original input/reference tensor')
    report['strict_FP32_gate_passed'] = all(
        row['versus_same_sample_original_Torch_FP32']['strict_FP32_gate_passed'] for row in report['calls'])
    report['input_tensor_unchanged'] = all(row['input_tensor_unchanged'] for row in report['calls'])
    report['reference_tensor_unchanged'] = all(row['reference_tensor_unchanged'] for row in report['calls'])
    return report['strict_FP32_gate_passed']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('contracts', 'pair-contracts', 'pair-dir', 'capture-contracts', 'capture-dir',
                 'toolchain-prefix', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('contracts-sha256', 'capture-contracts-sha256', 'capture-sha256'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    if sys.platform != 'linux' or platform.machine().lower() not in ('aarch64', 'arm64'):
        parser.error('Actual heldout ORT inference is permitted only on Linux RK3588S ARM64')
    if args.contracts_sha256 != QUANT_CONTRACTS_SHA or args.capture_contracts_sha256 != CAPTURE_CONTRACTS_SHA:
        parser.error('Use the frozen v49 quant/capture contracts without modifying their source')
    C = load_contract(args.contracts, QUANT_CONTRACTS_SHA)
    compatible = C.require_board()
    output = args.output_dir.resolve()
    if (Path.cwd().resolve() != C.BASE or output == C.BASE / 'experiments'
            or not output.is_relative_to(C.BASE / 'experiments')):
        parser.error('Board project cwd and new experiments output directory required')
    if Path(sys.prefix).resolve() != args.toolchain_prefix.resolve() or Path(sys.prefix).resolve() == C.BASE / 'venv':
        parser.error('The selected isolated v49 toolchain prefix must own this interpreter')
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'v49_original_VGGT_pair_heldout_board_CPU_ORT_check', 'status': 'failed',
              'script_sha256': sha(__file__), 'contracts_sha256': QUANT_CONTRACTS_SHA,
              'capture_contracts_sha256': CAPTURE_CONTRACTS_SHA, 'capture_manifest_sha256': args.capture_sha256,
              'pair_export_sha256': C.PAIR_EXPORT_SHA, 'pair_graph_sha256': C.PAIR_GRAPH_SHA,
              'pair_contracts_sha256': C.PAIR_CONTRACTS_SHA,
              'board': {'machine': platform.machine(), 'device_tree_compatible': compatible, 'python': sys.version},
              'toolchain_prefix': str(args.toolchain_prefix.resolve()),
              'inference_on_board_only': True, 'model_inference_on_host': False,
              'Torch_model_executed': False, 'RKNN_SDK_executed': False, 'NPU_used': False,
              'calibration_executed': False, 'full_VGGT_executed': False, 'candidate_accepted': False,
              'real_time_validated': False, 'whole_model_quality_validated': False,
              'rtol': 1e-4, 'atol': 1e-5, 'attempted_session_run_count': 0,
              'successful_session_run_count': 0, 'warmup_session_run_count': 0,
              'strict_FP32_gate_passed': False, 'calls': [], 'cleanup_errors': [],
              'session_created': False, 'session_released': False,
              'timing_scope': 'Each session.run only; loading, save/copy/hash, comparison, initialization and cleanup reported separately',
              'evaluation_route': 'original frame0 input -> unchanged complete original frame0/global0 frozen graph',
              'standalone_global_reference_input_reused': False}
    pins = {str(Path(__file__).resolve()): report['script_sha256'],
            str(args.contracts.resolve()): QUANT_CONTRACTS_SHA}
    affinity = os.sched_getaffinity(0)
    before_environment = {key: os.environ.get(key) for key in ENVIRONMENT}
    before_cwd = Path.cwd()
    session = None
    started = time.monotonic_ns()
    try:
        if not {6, 7}.issubset(affinity):
            raise ValueError('CPUs6,7 must be permitted for the fixed original CPU ORT comparison')
        os.sched_setaffinity(0, {6, 7})
        os.environ.update(ENVIRONMENT)
        report['versions'] = {name: importlib.metadata.version(name) for name in C.EXPECTED_VERSIONS}
        C.check_versions(report['versions'], sys.version_info, sys.prefix, args.toolchain_prefix)
        P = C.load_source(args.pair_contracts, C.PAIR_CONTRACTS_SHA, pins)
        pair, arrays, graph, bound = P.load_pair(args.pair_dir, C.PAIR_EXPORT_SHA)
        pins.update(bound)
        del arrays
        K = C.load_source(args.capture_contracts, CAPTURE_CONTRACTS_SHA, pins)
        captured, samples, bound = K.load_capture(args.capture_dir, args.capture_sha256)
        pins.update(bound)
        if samples != captured['samples'] or captured['frozen_pair']['onnx_sha256'] != C.PAIR_GRAPH_SHA:
            raise ValueError('Heldout capture belongs to another original pair graph')
        evals = [sample for sample in samples if sample['split'] == 'evaluation']
        report.update(capture_script_sha256=captured['script_sha256'], checkpoint=pair['checkpoint'],
                      evaluation_samples=evals, evaluated_sample_ids=[s['sample_id'] for s in evals],
                      calibration_sample_ids_excluded=[s['sample_id'] for s in samples if s['split'] == 'calibration'])
        import numpy as np
        import onnxruntime as ort
        report['module_source_paths'] = {}
        for name, module in (('numpy', np), ('onnxruntime', ort)):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(args.toolchain_prefix.resolve()) or module.__version__ != C.EXPECTED_VERSIONS[name]:
                raise ValueError('NumPy/ORT must belong to the explicitly selected frozen isolated toolchain')
            report['module_source_paths'][name] = str(path)
            pins[str(path)] = sha(path)
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        report['runtime'] = {'cpu_affinity': [6, 7], 'intra_op_threads': 2, 'inter_op_threads': 1,
                             'providers_requested': ['CPUExecutionProvider'], 'fallback_disabled': True,
                             'execution_mode': 'ORT_SEQUENTIAL', 'graph_optimization_level': 'ORT_ENABLE_ALL'}
        timer = time.monotonic_ns()
        graph_bytes = graph.read_bytes()
        report['read_frozen_graph_seconds'] = (time.monotonic_ns() - timer) / 1e9
        if hashlib.sha256(graph_bytes).hexdigest() != C.PAIR_GRAPH_SHA:
            raise ValueError('Actual session graph bytes differ from the frozen original pair')
        timer = time.monotonic_ns()
        session = ort.InferenceSession(graph_bytes, sess_options=options, providers=['CPUExecutionProvider'])
        report['session_creation_seconds'] = (time.monotonic_ns() - timer) / 1e9
        report['session_created'] = True
        del graph_bytes
        passed = run_heldout(session, samples, args.capture_dir, K, P, C, output, report)
        report['actual_providers'] = session.get_providers()
        report['status'] = 'passed' if passed else 'numerical_rejected'
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        timer = time.monotonic_ns()
        try:
            session = None
            gc.collect()
            report['session_released'] = True
            report['session_cleanup_scope'] = 'Last Python InferenceSession reference cleared and garbage collected; no explicit native close API used'
        except Exception as exc:
            report['cleanup_errors'].append(f'ORT session cleanup: {type(exc).__name__}: {exc}')
        report['session_cleanup_seconds'] = (time.monotonic_ns() - timer) / 1e9
        timer = time.monotonic_ns()
        try:
            os.sched_setaffinity(0, affinity)
            for key, value in before_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            report['CPU_affinity_restored'] = os.sched_getaffinity(0) == affinity
            report['environment_restored'] = {key: os.environ.get(key) for key in ENVIRONMENT} == before_environment
            report['cwd_restored'] = Path.cwd() == before_cwd
            after, unchanged = C.recheck(pins)
            report.update(source_and_input_hashes=pins, source_and_input_hashes_after=after,
                          source_and_inputs_unchanged=unchanged)
            if not unchanged or not all(report[k] for k in ('CPU_affinity_restored', 'environment_restored', 'cwd_restored')):
                report['cleanup_errors'].append('Original sources/inputs or process affinity/environment/cwd restoration mismatch')
            produced = {str(output / row[key]['file']): row[key]['sha256']
                        for row in report['calls'] for key in ('used_input', 'output') if key in row}
            saved_after, saved_unchanged = C.recheck(produced)
            report.update(saved_array_hashes=produced, saved_array_hashes_after=saved_after,
                          saved_arrays_unchanged=saved_unchanged)
            if not saved_unchanged:
                report['cleanup_errors'].append('Used input/full output files changed after saving')
        except Exception as exc:
            report['cleanup_errors'].append(f'Integrity/process cleanup: {type(exc).__name__}: {exc}')
        report['integrity_and_process_cleanup_seconds'] = (time.monotonic_ns() - timer) / 1e9
        if report['cleanup_errors']:
            report['status'] = 'failed'
        report['total_seconds'] = (time.monotonic_ns() - started) / 1e9
        report['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        report['peak_rss_scope'] = 'Heldout pair ORT process high-water; not whole SLAM memory or 8 GB validation'
        with (output / 'check.json').open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
    print(json.dumps({'status': report['status'], 'check_sha256': sha(output / 'check.json')}), flush=True)
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
