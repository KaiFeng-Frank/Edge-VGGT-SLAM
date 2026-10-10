#!/usr/bin/env python3
"""ARM-only CPU ONNX graph check against captured official DINO block 0.

Run from /mnt/vggt-data/vggt-slam2 after activating the isolated ARM ORT
environment. This does not load Torch, execute VGGT, or access the NPU.
The fixed allclose gate validates this exported graph/input only, not SLAM
quality or full-model accuracy. No existing receipt is overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import sys
import time
import traceback

sys.dont_write_bytecode = True
BASE = Path('/mnt/vggt-data/vggt-slam2')
SHAPE = (1, 782, 1024)
RTOL = 1e-4
ATOL = 1e-5
CHECKPOINT_SHA = 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
IMAGE_SHA = 'c23e57b9612522a10e61b1723749acddfe5b96b3a3642ec8c9db06e604d7f511'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha_argument(value):
    if not re.fullmatch('[0-9a-f]{64}', value):
        raise argparse.ArgumentTypeError('Supply a complete lowercase SHA256')
    return value


def check_array(array, label, np):
    if array.shape != SHAPE or array.dtype != np.float32:
        raise ValueError(f'{label} must be float32 {SHAPE}; got {array.dtype} {array.shape}')
    if not array.flags.c_contiguous or not np.isfinite(array).all():
        raise ValueError(f'{label} must be contiguous and finite')


def compare(actual, reference, np):
    check_array(actual, 'ORT output', np)
    actual64 = actual.astype(np.float64)
    reference64 = reference.astype(np.float64)
    difference = actual64 - reference64
    absolute = np.abs(difference)
    tolerance = ATOL + RTOL * np.abs(reference64)
    actual_norm = float(np.linalg.norm(actual64))
    reference_norm = float(np.linalg.norm(reference64))
    cosine = (float(np.sum(actual64 * reference64) / (actual_norm * reference_norm))
              if actual_norm > 0 and reference_norm > 0 else None)
    return {'finite': True, 'shape': list(actual.shape), 'dtype': 'float32',
            'max_abs': float(absolute.max()), 'mean_abs': float(absolute.mean()),
            'rmse': float(np.sqrt(np.mean(difference * difference))),
            'relative_l2': float(np.linalg.norm(difference) / max(reference_norm, 1e-30)),
            'cosine': cosine, 'allclose': bool(np.all(absolute <= tolerance)),
            'outside_tolerance_elements': int(np.count_nonzero(absolute > tolerance)),
            'max_tolerance_ratio': float(np.max(absolute / tolerance))}


def run(args):
    receipt = {'kind': 'CPU_ORT_official_DINO_block0_graph_check', 'status': 'started',
               'script_sha256': sha(__file__), 'model': {'path': str(args.model),
                                                       'expected_sha256': args.model_sha},
               'capture': {'path': str(args.capture_dir / 'capture.json'),
                           'expected_sha256': args.capture_sha},
               'no_npu': True, 'npu_used': False, 'Torch_model_executed': False,
               'full_VGGT_executed': False, 'accuracy_validated': False,
               'real_time_validated': False, 'graph_numerical_check_only': True,
               'tolerance': {'rtol': RTOL, 'atol': ATOL, 'predefined': True,
                             'formula': 'abs(ORT - CPU_reference) <= atol + rtol * abs(CPU_reference)'},
               'timing_scope': 'session.run only; excludes session creation, artifact load, comparisons and output I/O',
               'runs': []}
    started = time.monotonic()
    try:
        capture_path = args.capture_dir / 'capture.json'
        capture_sha = sha(capture_path)
        model_sha = sha(args.model)
        receipt['capture']['sha256'] = capture_sha
        receipt['model'].update(sha256=model_sha, bytes=args.model.stat().st_size)
        if capture_sha != args.capture_sha or model_sha != args.model_sha:
            raise ValueError('Explicit model/capture SHA mismatch')
        capture = json.loads(capture_path.read_text())
        if (capture.get('status') != 'completed' or
                capture.get('kind') != 'official_dino_block0_CPU_capture' or
                capture.get('checkpoint', {}).get('sha256') != CHECKPOINT_SHA or
                capture.get('checkpoint', {}).get('bytes') != 5026874952 or
                capture.get('image', {}).get('sha256') != IMAGE_SHA):
            raise ValueError('Capture does not identify the official checkpoint/original first Office image')
        cpu = capture.get('cpu_reference', {})
        if (cpu.get('device') != 'cpu' or cpu.get('dtype') != 'float32' or
                cpu.get('all_elementwise_equal') is not True or cpu.get('finite') is not True):
            raise ValueError('Capture lacks a finite/repeated exact CPU FP32 reference')
        artifact_hashes = {}
        for filename in ('input.npy', 'output.npy'):
            metadata = capture['artifacts'][filename]
            path = args.capture_dir / filename
            artifact_hashes[filename] = sha(path)
            if (artifact_hashes[filename] != metadata['sha256'] or
                    path.stat().st_size != metadata['bytes'] or
                    metadata['shape'] != list(SHAPE) or metadata['dtype'] != 'float32' or
                    metadata['finite'] is not True):
                raise ValueError(f'Capture artifact metadata/hash mismatch: {filename}')
        receipt['artifact_sha256'] = artifact_hashes
        allowed = set(os.sched_getaffinity(0))
        if not {6, 7}.issubset(allowed):
            raise ValueError(f'CPU 6,7 must be allowed, actual: {sorted(allowed)}')
        os.sched_setaffinity(0, {6, 7})
        enforced_env = {'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '2'}
        receipt['environment_before'] = {key: os.environ.get(key) for key in enforced_env}
        os.environ.update(enforced_env)
        import numpy as np
        import onnxruntime as ort
        receipt['runtime'] = {'python': sys.version, 'platform': platform.platform(),
                              'onnxruntime': ort.__version__, 'numpy': np.__version__,
                              'cpu_affinity': sorted(os.sched_getaffinity(0)),
                              'intra_op_threads': 2, 'inter_op_threads': 1,
                              'environment': {key: os.environ.get(key) for key in
                                              (*enforced_env, 'OMP_WAIT_POLICY', 'GOMP_SPINCOUNT')},
                              'available_execution_providers': ort.get_available_providers()}
        tokens = np.load(args.capture_dir / 'input.npy', allow_pickle=False)
        reference = np.load(args.capture_dir / 'output.npy', allow_pickle=False)
        check_array(tokens, 'Captured block input', np)
        check_array(reference, 'Captured CPU reference', np)
        # Byte-based construction also refuses unbound external-data files;
        # the complete graph/initializer content must live in the SHA'd model.
        model_bytes = args.model.read_bytes()
        if hashlib.sha256(model_bytes).hexdigest() != args.model_sha:
            raise ValueError('Model changed before session construction')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        timer = time.perf_counter()
        session = ort.InferenceSession(model_bytes, sess_options=options,
                                       providers=['CPUExecutionProvider'])
        receipt['session_creation_seconds'] = time.perf_counter() - timer
        session.disable_fallback()
        if session.get_providers() != ['CPUExecutionProvider']:
            raise ValueError(f'Unexpected execution providers: {session.get_providers()}')
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError('Expected one graph input and one output')
        for label, node in (('input', inputs[0]), ('output', outputs[0])):
            receipt['model'][label] = {'name': node.name, 'type': node.type, 'shape': node.shape}
            if node.type != 'tensor(float)' or node.shape != list(SHAPE):
                raise ValueError(f'Graph {label} must have fixed float32 shape {SHAPE}')
        previous = None
        for index in range(4):
            timer = time.perf_counter()
            result = session.run([outputs[0].name], {inputs[0].name: tokens})
            seconds = time.perf_counter() - timer
            if len(result) != 1:
                raise ValueError('ORT returned an unexpected output count')
            actual = result[0]
            comparison = compare(actual, reference, np)
            receipt['runs'].append({'phase': 'cold_first_run' if index == 0 else 'warm',
                                     'run_index': index, 'seconds': seconds,
                                     'elementwise_equal_to_previous_ORT':
                                         None if previous is None else bool(np.array_equal(actual, previous)),
                                     'comparison': comparison})
            previous = actual
        receipt['cold_seconds'] = receipt['runs'][0]['seconds']
        receipt['warm_seconds'] = [row['seconds'] for row in receipt['runs'][1:]]
        receipt['warm_mean_seconds'] = sum(receipt['warm_seconds']) / 3
        receipt['all_runs_within_tolerance'] = all(row['comparison']['allclose'] for row in receipt['runs'])
        receipt['repeated_ORT_elementwise_equal'] = all(
            row['elementwise_equal_to_previous_ORT'] is True for row in receipt['runs'][1:])
        receipt['input_files_unchanged'] = (
            sha(args.model) == model_sha and sha(capture_path) == capture_sha and
            all(sha(args.capture_dir / name) == value for name, value in artifact_hashes.items()))
        if not receipt['input_files_unchanged']:
            raise ValueError('Model/capture/artifacts changed during validation')
        if not receipt['all_runs_within_tolerance']:
            raise ValueError('ORT graph differs from CPU reference beyond the predefined FP32 allclose gate')
        receipt['status'] = 'passed'
    except Exception as exc:
        receipt['status'] = 'failed'
        receipt['error'] = f'{type(exc).__name__}: {exc}'
        receipt['traceback'] = traceback.format_exc()
    finally:
        receipt['total_seconds_including_load_comparison'] = time.monotonic() - started
        receipt['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.
        receipt['rss_scope'] = 'this process high-water mark, including inherited pre-exec history if any'
        with args.output.open('x') as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write('\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--capture-sha', type=sha_argument, required=True,
                        help='SHA256 of capture-dir/capture.json')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-sha', type=sha_argument, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New JSON receipt under board experiments')
    args = parser.parse_args()
    if platform.machine() != 'aarch64' or Path.cwd().resolve() != BASE:
        parser.error('Run only on the RK3588S board from /mnt/vggt-data/vggt-slam2')
    if Path(__file__).resolve().parents[3] != BASE / 'src':
        parser.error('Run the deployed validation script under the board src directory')
    if os.path.lexists(args.output):
        parser.error('Refusing an existing output receipt or symlink')
    args.output = args.output.resolve()
    if not args.output.is_relative_to(BASE / 'experiments') or args.output.suffix != '.json':
        parser.error('--output must be a new JSON receipt below the board experiments directory')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt = run(args)
    print(json.dumps({'status': receipt['status'], 'output': str(args.output),
                      'sha256': sha(args.output), 'error': receipt.get('error')}))
    return 0 if receipt['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
