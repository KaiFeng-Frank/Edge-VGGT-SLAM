#!/usr/bin/env python3
"""Board-only two-input CPU/ORT check of the complete original CameraHead.

The second raw camera-token input uses a fixed nonuniform perturbation. Both
outputs are compared to the original four-iteration Torch head. This checks
export numerics and sampled input dependence, not NPU or whole-model quality.
The original Torch head is released before constructing the large ORT graph.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

sys.dont_write_bytecode = True
BASE = Path('/mnt/vggt-data/vggt-slam2')
SOURCE = BASE / 'src'
CAPTURE_SHA = 'd50c217354b2f48f50626ff1ea2c98671d5611743ae0e97bed1c04b2991341a0'
EXPORT_SHA = '3f97e96e6e9f7beaf5a5de56569774aa79fa17eb7e30664e54cdbbfec224a033'
RTOL, ATOL = 1e-4, 1e-5


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('Explicit lowercase SHA256 required')
    if sha(path) != expected:
        raise ValueError(f'SHA256 mismatch: {path}')
    return expected


def load_capture_helper():
    path = Path(__file__).with_name('capture-v15-camera-board.py')
    require_sha(path, CAPTURE_SHA)
    spec = importlib.util.spec_from_file_location('_v15_ORT_camera_capture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def compare(actual, reference, np):
    if (actual.dtype != np.float32 or actual.shape != (1, 2, 9)
            or reference.dtype != np.float32 or reference.shape != actual.shape
            or not np.isfinite(actual).all() or not np.isfinite(reference).all()):
        raise ValueError('Finite FP32 [1,2,9] outputs required')
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(delta)
    tolerance = ATOL + RTOL * np.abs(reference.astype(np.float64))
    return {'allclose': bool(np.all(absolute <= tolerance)), 'max_abs': float(absolute.max()),
            'rmse': float(np.sqrt(np.mean(delta * delta))),
            'relative_l2': float(np.linalg.norm(delta) / max(float(np.linalg.norm(reference.astype(np.float64))), 1e-30)),
            'outside_tolerance_elements': int(np.count_nonzero(absolute > tolerance)),
            'max_tolerance_ratio': float(np.max(absolute / tolerance))}


def save(output, name, array, helper):
    import numpy as np
    info = helper.array_info(array)
    path = output / name
    with path.open('xb') as stream:
        np.save(stream, array, allow_pickle=False)
    return {'file': name, 'sha256': sha(path), 'bytes': path.stat().st_size,
            'tensor_sha256': info['sha256'], 'tensor_bytes': info['bytes'],
            'shape': info['shape'], 'dtype': info['dtype'], 'finite': info['finite']}


def run(args):
    report = {'kind': 'complete_original_CameraHead_two_input_CPU_ORT_check', 'status': 'failed',
              'script_sha256': sha(__file__), 'CPU_camera_forward_call_count': 0, 'CPU_trunk_call_count': 0,
              'ORT_call_count': 0, 'host_model_inference_executed': False, 'npu_used': False, 'ORB_used': False,
              'production_modified': False, 'real_time_validated': False, 'full_model_quality_validated': False,
              'tolerance': {'rtol': RTOL, 'atol': ATOL, 'predefined': True}, 'artifacts': {},
              'input_dependence_scope': 'Two specified inputs only; no universal equivalence claim',
              'cleanup_errors': [], 'runs': []}
    begin, allowed, pins, hooks = time.monotonic(), None, {}, []
    head = arrays = session = None
    try:
        capture, capture_path = load_capture_helper()
        report['device_tree_compatible'] = capture.require_board()
        helper, helper_path = capture.base_capture()
        exported_path = Path(__file__).with_name('export-v15-camera-board.py')
        pins = {str(Path(__file__).resolve()): sha(__file__), str(capture_path): CAPTURE_SHA,
                str(helper_path): capture.BASE_CAPTURE_SHA, str(exported_path): require_sha(exported_path, EXPORT_SHA),
                str(args.model): require_sha(args.model, args.model_sha256),
                str(args.export_receipt): require_sha(args.export_receipt, args.export_sha256),
                str(args.capture_dir / 'capture.json'): require_sha(args.capture_dir / 'capture.json', args.capture_sha256)}
        metadata = json.loads((args.capture_dir / 'capture.json').read_text())
        exported = json.loads(args.export_receipt.read_text())
        if (metadata.get('status') != 'completed' or metadata.get('script_sha256') != CAPTURE_SHA
                or metadata.get('kind') != 'original_complete_CameraHead_CPU_capture'
                or metadata.get('CPU_camera_forward_call_count') != 3 or metadata.get('CPU_trunk_call_count') != 48
                or metadata.get('camera_iterations') != 4 or metadata.get('npu_used') is not False
                or metadata.get('source_and_inputs_unchanged') is not True or metadata.get('cleanup_errors')
                or exported.get('status') != 'exported_and_structurally_checked'
                or exported.get('script_sha256') != EXPORT_SHA or exported.get('cleanup_errors')
                or exported.get('capture', {}).get('sha256') != args.capture_sha256
                or exported.get('architecture') != metadata.get('architecture')
                or exported.get('onnx', {}).get('sha256') != args.model_sha256
                or exported.get('weight_sharing', {}).get('original_parameters_verified') != 69):
            raise ValueError('Exact original complete-head capture/export provenance required')
        originals = {**helper.SOURCE_PINS, **capture.EXTRA_PINS}
        if metadata.get('source_hashes') != originals:
            raise ValueError('Original source provenance mismatch')
        for name, digest in originals.items():
            path = SOURCE / name
            pins[str(path)] = require_sha(path, digest)
        for name, item in metadata['artifacts'].items():
            path = args.capture_dir / name
            pins[str(path)] = require_sha(path, item['sha256'])
            if path.stat().st_size != item['bytes']:
                raise ValueError(f'Artifact byte mismatch: {name}')
        allowed = set(os.sched_getaffinity(0))
        if not capture.CPUS <= allowed:
            raise ValueError('CPU 4,5,6,7 must be allowed')
        os.sched_setaffinity(0, capture.CPUS)
        os.environ.update({name: '4' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
        sys.path[:0] = [str(SOURCE), str(SOURCE / 'third_party/vggt')]
        import numpy as np
        import torch
        from vggt.heads.camera_head import CameraHead
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        if torch.__version__ != metadata['runtime']['torch']:
            raise ValueError('Exact captured board Torch version required')
        tokens = np.load(args.capture_dir / 'input.npy', allow_pickle=False)
        original_output = np.load(args.capture_dir / 'output.npy', allow_pickle=False)
        if (tokens.shape != capture.INPUT_SHAPE or helper.array_info(tokens)['sha256'] != metadata['artifacts']['input.npy']['tensor_sha256']
                or original_output.shape != capture.OUTPUT_SHAPE
                or helper.array_info(original_output)['sha256'] != metadata['artifacts']['output.npy']['tensor_sha256']):
            raise ValueError('Captured token/reference raw hash or shape mismatch')
        std = float(np.std(tokens.astype(np.float64)))
        if not 0 < std < float('inf'):
            raise ValueError('Nonzero finite token standard deviation required')
        perturbation = (0.05 * std * np.sin(np.arange(tokens.size, dtype=np.float64) * 0.6180339887498949)).astype(np.float32).reshape(tokens.shape)
        second = np.ascontiguousarray(tokens + perturbation)
        if np.array_equal(tokens, second) or not np.isfinite(second).all():
            raise ValueError('Predetermined second input is not different and finite')
        report['second_input_definition'] = {'formula': 'x + float32(0.05 * std64(x) * sin(arange(x.size) * 0.6180339887498949))',
                                             'std64': std, 'parameters_chosen_before_results': True}
        report['artifacts']['input-01.npy'] = save(args.output_dir, 'input-01.npy', second, helper)
        with np.load(args.capture_dir / 'camera-state.npz', allow_pickle=False) as saved:
            arrays = {key: np.ascontiguousarray(saved[key]) for key in saved.files}
        tensors = metadata['artifacts']['camera-state.npz']['tensors']
        if len(arrays) != 69 or set(arrays) != set(tensors) or any(helper.array_info(value) != tensors[key] for key, value in arrays.items()):
            raise ValueError('Exact 69 original CameraHead parameter values required')
        with torch.device('meta'):
            head = CameraHead()
        head.load_state_dict({key: torch.from_numpy(value) for key, value in arrays.items()}, strict=True, assign=True)
        head.eval().requires_grad_(False)
        if capture.camera_contract(head) != metadata['architecture']:
            raise ValueError('Original four-iteration CameraHead architecture mismatch')
        def seen(module, inputs, result):
            report['CPU_trunk_call_count'] += 1
        hooks = [block.register_forward_hook(seen) for block in head.trunk]
        references = []
        with torch.inference_mode():
            for value in (tokens, second):
                tick = time.perf_counter()
                poses = head([torch.from_numpy(value).unsqueeze(2)], num_iterations=4)
                report['CPU_camera_forward_call_count'] += 1
                report.setdefault('CPU_reference_seconds', []).append(time.perf_counter() - tick)
                if len(poses) != 4:
                    raise ValueError('Complete four-iteration output required')
                references.append(np.ascontiguousarray(poses[-1].detach().numpy().copy()))
        if not np.array_equal(references[0], original_output) or report['CPU_trunk_call_count'] != 32:
            raise ValueError('Original CPU reference/trunk-call mismatch')
        report['artifacts']['cpu-output-01.npy'] = save(args.output_dir, 'cpu-output-01.npy', references[1], helper)
        report['CPU_second_output_changed'] = not np.array_equal(references[0], references[1])
        for hook in hooks:
            hook.remove()
        hooks.clear()
        del head, arrays, poses
        head = arrays = None
        gc.collect()
        sys.path.insert(0, str(args.ort_site))
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        tick = time.perf_counter()
        session = ort.InferenceSession(str(args.model), sess_options=options, providers=['CPUExecutionProvider'])
        session.disable_fallback()
        report['ORT_session_creation_seconds'] = time.perf_counter() - tick
        if session.get_providers() != ['CPUExecutionProvider']:
            raise ValueError('CPUExecutionProvider only is required')
        ins, outs = session.get_inputs(), session.get_outputs()
        if (len(ins) != 1 or len(outs) != 1 or ins[0].name != 'camera_tokens' or outs[0].name != 'pose_enc'
                or ins[0].type != 'tensor(float)' or outs[0].type != 'tensor(float)'
                or ins[0].shape != list(capture.INPUT_SHAPE) or outs[0].shape != list(capture.OUTPUT_SHAPE)):
            raise ValueError('Exact raw-token/final-pose ONNX IO required')
        report['runtime'] = {'torch': torch.__version__, 'onnxruntime': ort.__version__, 'onnxruntime_path': ort.__file__,
                             'cpu_affinity': sorted(os.sched_getaffinity(0)), 'threads': 4, 'providers': session.get_providers()}
        ort_outputs = []
        for index, (value, reference) in enumerate(zip((tokens, second), references)):
            tick = time.perf_counter()
            actual = session.run(['pose_enc'], {'camera_tokens': value})[0]
            seconds = time.perf_counter() - tick
            report['ORT_call_count'] += 1
            evidence = compare(actual, reference, np)
            report['runs'].append({'input_index': index, 'seconds': seconds, 'comparison': evidence})
            name = f'ort-output-{index:02d}.npy'
            report['artifacts'][name] = save(args.output_dir, name, actual, helper)
            ort_outputs.append(actual)
        report['ORT_second_output_changed'] = not np.array_equal(*ort_outputs)
        report['source_and_input_hashes'] = pins.copy()
        report['source_and_input_hashes_after'] = {path: sha(path) for path in pins}
        report['source_and_inputs_unchanged'] = report['source_and_input_hashes_after'] == pins
        if not report['source_and_inputs_unchanged']:
            raise ValueError('Source/capture/model changed during check')
        if not report['CPU_second_output_changed'] or not report['ORT_second_output_changed']:
            raise ValueError('Output is insensitive to the specified different raw-token input')
        if not all(row['comparison']['allclose'] for row in report['runs']):
            raise ValueError('Complete ONNX head exceeds the predefined FP32 numerical gate')
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        for hook in hooks:
            try:
                hook.remove()
            except BaseException as exc:
                report['cleanup_errors'].append(f'Hook cleanup: {exc}')
        head = arrays = session = None
        gc.collect()
        if allowed is not None:
            try:
                os.sched_setaffinity(0, allowed)
                report['affinity_restored'] = set(os.sched_getaffinity(0)) == allowed
                if not report['affinity_restored']:
                    raise RuntimeError('Affinity restoration mismatch')
            except BaseException as exc:
                report['cleanup_errors'].append(f'Affinity cleanup: {exc}')
        if report['cleanup_errors']:
            report['status'] = 'failed'
        report['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.
        report['total_seconds_including_load_compare_IO'] = time.monotonic() - begin
        with (args.output_dir / 'check.json').open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--capture-sha256', required=True)
    parser.add_argument('--export-receipt', type=Path, required=True)
    parser.add_argument('--export-sha256', required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-sha256', required=True)
    parser.add_argument('--ort-site', type=Path, default=BASE / 'experiments/orb-v12-dino-block/ort-site')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    capture, _ = load_capture_helper()
    capture.require_board()
    args.output_dir = args.output_dir.resolve()
    if (Path.cwd().resolve() != BASE or Path(__file__).resolve().parents[3] != SOURCE
            or not args.output_dir.is_relative_to(BASE / 'experiments/vggt-v15-camera')
            or os.path.lexists(args.output_dir)):
        parser.error('Run deployed source from board base; output must be a new camera experiment directory')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = run(args)
    path = args.output_dir / 'check.json'
    print(json.dumps({'status': report['status'], 'error': report.get('error'), 'output': str(path), 'sha256': sha(path)}))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
