#!/usr/bin/env python3
"""Run original frame0 then global0 as one ORT graph on RK3588S only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode = True


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_contract(path, expected):
    if sha(path) != expected:
        raise ValueError('Pair contracts SHA mismatch')
    spec = importlib.util.spec_from_file_location('_v47_ort_contracts', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_session(session, C):
    session.disable_fallback()
    if session.get_providers() != ['CPUExecutionProvider']:
        raise ValueError('ORT CPU provider only required')
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError('One pair input/output required')
    for node, name, shape in ((session.get_inputs()[0], 'input', C.INPUT_SHAPE),
                              (session.get_outputs()[0], 'output', C.OUTPUT_SHAPE)):
        if node.name != name or node.type != 'tensor(float)' or node.shape != list(shape):
            raise ValueError(f'Original pair ORT {name} contract mismatch')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('pair-dir', 'contracts', 'output-dir'):
        p.add_argument('--' + name, required=True, type=Path)
    for name in ('pair-sha256', 'contracts-sha256'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args(argv)
    C = load_contract(a.contracts, a.contracts_sha256)
    C.require_board()
    if Path.cwd().resolve() != C.BASE or not a.output_dir.resolve().is_relative_to(C.BASE / 'experiments'):
        p.error('Board project cwd and new experiments output required')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'original_VGGT_pair_CPU_ORT_check_v47', 'status': 'failed',
              'script_sha256': sha(__file__), 'contracts_sha256': a.contracts_sha256,
              'pair_export_sha256': a.pair_sha256, 'model_inference_on_host': False,
              'inference_on_board_only': True, 'Torch_model_executed': False, 'NPU_used': False,
              'full_VGGT_executed': False, 'candidate_accepted': False,
              'real_time_validated': False, 'whole_model_quality_validated': False,
              'rtol': C.RTOL, 'atol': C.ATOL, 'successful_session_run_count': 0,
              'calls': [], 'cleanup_errors': [],
              'timing_scope': 'session.run only; excludes load, init, comparisons, hashes and NPY saves',
              'input_route': 'original frame0 input -> complete frame0 -> C-order view -> complete global0',
              'standalone_global_input_reused': False}
    pins = {str(Path(__file__).resolve()): report['script_sha256'],
            str(a.contracts.resolve()): a.contracts_sha256}
    affinity = os.sched_getaffinity(0)
    env_keys = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')
    env = {k: os.environ.get(k) for k in env_keys}
    session = None
    started = time.perf_counter()
    try:
        if not {6, 7}.issubset(affinity):
            raise ValueError('CPUs6,7 must be allowed')
        os.sched_setaffinity(0, {6, 7})
        os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
        import numpy as np
        import onnxruntime as ort
        export, arrays, graph, bound = C.load_pair(a.pair_dir, a.pair_sha256)
        pins.update(bound)
        tokens, reference = arrays['input.npy'], arrays['output.npy']
        before_input = hashlib.sha256(memoryview(tokens).cast('B')).hexdigest()
        report.update(input_shape=list(tokens.shape), output_shape=list(reference.shape),
                      original_input_and_reference_exact=True, checkpoint=export['checkpoint'],
                      selected_pair=export['selected_pair'], architecture=export['architecture'],
                      onnx=export['onnx'], artifacts=export['artifacts'],
                      runtime={'onnxruntime': ort.__version__, 'numpy': np.__version__,
                               'cpu_affinity': [6, 7], 'intra_op_threads': 2, 'inter_op_threads': 1})
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        timer = time.perf_counter()
        session = ort.InferenceSession(graph.read_bytes(), sess_options=options,
                                       providers=['CPUExecutionProvider'])
        report['session_creation_seconds'] = time.perf_counter() - timer
        check_session(session, C)
        first = None
        for index in range(4):
            timer = time.perf_counter()
            actual = session.run(['output'], {'input': tokens})[0]
            seconds = time.perf_counter() - timer
            report['successful_session_run_count'] += 1
            path = a.output_dir / f'actual_{index:02d}.npy'
            np.save(path, actual, allow_pickle=False)
            row = {'index': index, 'phase': 'first' if index == 0 else 'warm', 'seconds': seconds,
                   'artifact': {'file': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path),
                                'shape': list(actual.shape), 'dtype': str(actual.dtype)}}
            report['calls'].append(row)
            row['versus_original_CPU_pair'] = C.compare(actual, reference)
            if first is None:
                first = actual.copy()
            row['bitwise_equal_to_first'] = actual.tobytes() == first.tobytes()
        report['strict_FP32_gate_passed'] = all(r['versus_original_CPU_pair']['strict_FP32_gate_passed']
                                                for r in report['calls'])
        report['repeated_outputs_bitwise_equal'] = all(r['bitwise_equal_to_first'] for r in report['calls'])
        if not report['repeated_outputs_bitwise_equal']:
            raise ValueError('Repeated fixed original CPU pair outputs changed')
        report['input_tensor_unchanged'] = hashlib.sha256(memoryview(tokens).cast('B')).hexdigest() == before_input
        if not report['input_tensor_unchanged']:
            raise ValueError('ORT mutated the original pair input')
        report['cold_seconds'] = report['calls'][0]['seconds']
        report['warm_seconds'] = [r['seconds'] for r in report['calls'][1:]]
        report['warm_mean_seconds'] = sum(report['warm_seconds']) / 3
        report['status'] = 'passed' if report['strict_FP32_gate_passed'] else 'numerical_rejected'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        session = None
        report['session_released'] = True
        try:
            os.sched_setaffinity(0, affinity)
            for k, v in env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            report['affinity_restored'] = os.sched_getaffinity(0) == affinity
            after, unchanged = C.recheck(pins)
            report.update(source_hashes=pins, source_hashes_after=after, source_and_inputs_unchanged=unchanged)
            if not unchanged or not report['affinity_restored']:
                report['cleanup_errors'].append('Source/input or CPU affinity restoration mismatch')
        except Exception as exc:
            report['cleanup_errors'].append(str(exc))
        if report['cleanup_errors']:
            report['status'] = 'failed'
        report['total_seconds'] = time.perf_counter() - started
        report['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.
        (a.output_dir / 'check.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'check_sha256': sha(a.output_dir / 'check.json')}))
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    sys.exit(main())
