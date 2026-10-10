#!/usr/bin/env python3
"""Export the entire original four-iteration CameraHead, on RK3588S only.

Uses one raw [1,2,2048] camera-token input, not normalized tokens or image
features from another run. Export tracing executes only on the board; the host
may compile this artifact but must not run a Torch/ORT model. All original
weights are shared across iterations. This is not a whole VGGT speed claim.
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
CAPTURE_SCRIPT_SHA = 'd50c217354b2f48f50626ff1ea2c98671d5611743ae0e97bed1c04b2991341a0'
MAX_ONNX_BYTES = 1536 * 1024 * 1024


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('An explicit lowercase SHA256 is required')
    if sha(path) != expected:
        raise ValueError(f'SHA256 mismatch: {path}')
    return expected


def load_capture_helper():
    path = Path(__file__).with_name('capture-v15-camera-board.py')
    require_sha(path, CAPTURE_SCRIPT_SHA)
    spec = importlib.util.spec_from_file_location('_v15_camera_capture_contract', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def weight_bindings(model, expected, np, numpy_helper):
    """Resolve exporter Identity aliases; verify every original parameter value."""
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    aliases = {node.output[0]: node.input[0] for node in model.graph.node
               if node.op_type == 'Identity' and len(node.input) == len(node.output) == 1}
    evidence = []
    for key, metadata in expected.items():
        name, seen = 'camera_head.' + key, set()
        while name not in initializers and name in aliases:
            if name in seen:
                raise ValueError('Cyclic ONNX parameter alias')
            seen.add(name)
            name = aliases[name]
        if name not in initializers:
            raise ValueError(f'ONNX is missing the original shared parameter: {key}')
        array = numpy_helper.to_array(initializers[name])
        if array.dtype != np.float32 or list(array.shape) != metadata['shape']:
            raise ValueError(f'ONNX original parameter shape/dtype mismatch: {key}')
        raw_sha = hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()
        if raw_sha != metadata['sha256']:
            raise ValueError(f'ONNX changed an original CameraHead weight: {key}')
        evidence.append({'parameter': key, 'initializer': name, 'raw_sha256': raw_sha,
                         'shape': metadata['shape'], 'identity_aliases': sorted(seen)})
    total = sum(int(np.prod(tensor.dims)) * np.dtype(numpy_helper.to_array(tensor).dtype).itemsize
                for tensor in model.graph.initializer)
    expected_total = sum(item['bytes'] for item in expected.values())
    if total > expected_total + 1024 * 1024:
        raise ValueError('ONNX initializer bytes exceed unique original weights: iteration duplication is forbidden')
    return {'original_parameters_verified': len(evidence), 'initializer_count': len(initializers),
            'initializer_bytes': total, 'original_unique_parameter_bytes': expected_total,
            'weights_not_duplicated_per_iteration': True, 'parameters': evidence}


def export(output, capture_dir, capture_sha, onnx_site):
    report = {'kind': 'original_complete_CameraHead_board_ONNX_export', 'status': 'failed',
              'script_sha256': sha(__file__), 'camera_iterations': 4,
              'host_model_inference_executed': False, 'board_export_trace_forward_calls': 0,
              'board_export_trace_trunk_calls': 0, 'CPU_performance_forward_calls': 0,
              'npu_used': False, 'ORB_used': False, 'production_modified': False,
              'real_time_validated': False, 'full_model_quality_validated': False,
              'all_operators_on_npu_verified': False, 'trace_vs_capture': [], 'cleanup_errors': [],
              'output_dir': str(output)}
    started, allowed, hooks, pins = time.monotonic(), None, [], {}
    try:
        captured, helper_path = load_capture_helper()
        report['device_tree_compatible'] = captured.require_board()
        source_helper, original_helper_path = captured.base_capture()
        receipt_path = capture_dir / 'capture.json'
        pins = {str(receipt_path): require_sha(receipt_path, capture_sha),
                str(helper_path): CAPTURE_SCRIPT_SHA, str(original_helper_path): captured.BASE_CAPTURE_SHA,
                str(Path(__file__).resolve()): sha(__file__)}
        metadata = json.loads(receipt_path.read_text())
        if (metadata.get('status') != 'completed' or metadata.get('kind') != 'original_complete_CameraHead_CPU_capture'
                or metadata.get('script_sha256') != CAPTURE_SCRIPT_SHA or metadata.get('npu_used') is not False
                or metadata.get('ORB_used') is not False or metadata.get('camera_iterations') != 4
                or metadata.get('CPU_camera_forward_call_count') != 3 or metadata.get('CPU_trunk_call_count') != 48
                or metadata.get('export_trace_call_count') != 0 or metadata.get('cleanup_errors')
                or metadata.get('affinity_restored') is not True
                or metadata.get('source_and_inputs_unchanged') is not True
                or metadata.get('checkpoint_stat_unchanged') is not True
                or metadata.get('checkpoint', {}).get('sha256') != captured.CHECKPOINT_SHA):
            raise ValueError('Completed exact original CameraHead CPU capture is required')
        originals = {**source_helper.SOURCE_PINS, **captured.EXTRA_PINS}
        if metadata.get('source_hashes') != originals:
            raise ValueError('Capture does not use the frozen original model/source')
        for name, digest in originals.items():
            path = SOURCE / name
            pins[str(path)] = require_sha(path, digest)
        for item in metadata['input_images']:
            pins[item['path']] = require_sha(item['path'], item['sha256'])
        reference = metadata.get('cpu_reference', {})
        if any(reference.get(key) != value for key, value in {'device': 'cpu', 'dtype': 'float32', 'no_grad': True,
                'eval': True, 'finite': True, 'all_elementwise_equal': True, 'repeats': 3}.items()):
            raise ValueError('Original complete three-call CPU reference is required')
        allowed = set(os.sched_getaffinity(0))
        if not captured.CPUS <= allowed:
            raise ValueError('CPU 4,5,6,7 must be allowed for reproducible export tracing')
        os.sched_setaffinity(0, captured.CPUS)
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[key] = '4'
        sys.path[:0] = [str(onnx_site), str(SOURCE), str(SOURCE / 'third_party/vggt')]
        import numpy as np
        import torch
        import onnx
        from onnx import numpy_helper
        from vggt.heads.camera_head import CameraHead
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        if onnx.__version__ != '1.16.2' or torch.__version__ != metadata['runtime']['torch']:
            raise ValueError('Use isolated ONNX 1.16.2 and the exact captured board Torch version')
        report['runtime'] = {'python': sys.version, 'torch': torch.__version__, 'onnx': onnx.__version__,
                             'onnx_module_path': onnx.__file__, 'numpy': np.__version__, 'threads': 4,
                             'cpu_affinity': sorted(os.sched_getaffinity(0))}
        for filename in ('input.npy', 'output.npy', 'camera-state.npz', 'pose-iterations.npz'):
            path = capture_dir / filename
            item = metadata['artifacts'][filename]
            pins[str(path)] = require_sha(path, item['sha256'])
            if path.stat().st_size != item['bytes']:
                raise ValueError(f'Capture artifact byte mismatch: {filename}')
        input_array = np.load(capture_dir / 'input.npy', allow_pickle=False)
        output_array = np.load(capture_dir / 'output.npy', allow_pickle=False)
        for filename, array, shape in (('input.npy', input_array, captured.INPUT_SHAPE),
                                        ('output.npy', output_array, captured.OUTPUT_SHAPE)):
            info = source_helper.array_info(array)
            item = metadata['artifacts'][filename]
            if tuple(array.shape) != shape or item['shape'] != list(shape) or item['tensor_sha256'] != info['sha256']:
                raise ValueError('Input/reference raw shape/hash mismatch')
        with np.load(capture_dir / 'pose-iterations.npz', allow_pickle=False) as saved:
            poses = {key: saved[key].copy() for key in saved.files}
        if set(poses) != {f'iteration_{i}' for i in range(4)}:
            raise ValueError('All four original CPU iteration references are required')
        for key, array in poses.items():
            if (array.shape != captured.OUTPUT_SHAPE or source_helper.array_info(array)
                    != metadata['artifacts']['pose-iterations.npz']['tensors'][key]):
                raise ValueError('Original CPU pose iteration reference mismatch')
        if not np.array_equal(output_array, poses['iteration_3']):
            raise ValueError('Final pose differs from the original fourth iteration')
        with np.load(capture_dir / 'camera-state.npz', allow_pickle=False) as saved:
            arrays = {key: np.ascontiguousarray(saved[key]) for key in saved.files}
        tensors = metadata['artifacts']['camera-state.npz']['tensors']
        if len(arrays) != 69 or set(arrays) != set(tensors):
            raise ValueError('Exact 69 original CameraHead parameter tensors required')
        for key, array in arrays.items():
            if source_helper.array_info(array) != tensors[key]:
                raise ValueError(f'Original CameraHead parameter SHA/dtype/shape mismatch: {key}')
        with torch.device('meta'):
            head = CameraHead()
        head.load_state_dict({key: torch.from_numpy(value) for key, value in arrays.items()}, strict=True, assign=True)
        head.eval().requires_grad_(False)
        if any(value.is_meta for value in (*head.parameters(), *head.buffers())):
            raise ValueError('Unmaterialized original CameraHead tensor')
        architecture = captured.camera_contract(head)
        if architecture != metadata['architecture']:
            raise ValueError('Original four-iteration architecture/epsilon metadata mismatch')
        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.camera_head = head
            def forward(self, camera_tokens):
                return self.camera_head([camera_tokens.unsqueeze(2)], num_iterations=4)[-1]
        wrapper = Wrapper().eval()
        def trunk_seen(module, args, result):
            report['board_export_trace_trunk_calls'] += 1
        def head_seen(module, args, result):
            report['board_export_trace_forward_calls'] += 1
            row = {'iterations': len(result), 'all_iteration_array_equal': True}
            if len(result) != 4:
                raise ValueError('Export changed the four-iteration CameraHead')
            for i, value in enumerate(result):
                actual = value.detach().numpy()
                if not np.isfinite(actual).all() or not np.array_equal(actual, poses[f'iteration_{i}']):
                    row['all_iteration_array_equal'] = False
            report['trace_vs_capture'].append(row)
            if not row['all_iteration_array_equal']:
                raise ValueError('Original CPU export trace differs from the captured four-iteration reference')
        hooks = [block.register_forward_hook(trunk_seen) for block in head.trunk]
        hooks.append(head.register_forward_hook(head_seen))
        model_path = output / 'camera-head.onnx'
        tick = time.perf_counter()
        with torch.no_grad():
            torch.onnx.export(wrapper, (torch.from_numpy(input_array),), str(model_path),
                              input_names=['camera_tokens'], output_names=['pose_enc'],
                              export_params=True, opset_version=17, do_constant_folding=False,
                              keep_initializers_as_inputs=False, training=torch.onnx.TrainingMode.EVAL)
        report['export_seconds_including_board_trace_and_IO'] = time.perf_counter() - tick
        for hook in hooks:
            hook.remove()
        hooks.clear()
        if (report['board_export_trace_forward_calls'] < 1
                or report['board_export_trace_trunk_calls'] != 16 * report['board_export_trace_forward_calls']):
            raise ValueError('Incomplete original four-iteration trace')
        if not 0 < model_path.stat().st_size <= MAX_ONNX_BYTES:
            raise ValueError('ONNX size exceeds the bounded 1.5GiB single-head budget')
        del wrapper, head, arrays
        gc.collect()
        model = onnx.load(str(model_path), load_external_data=False)
        if len(model.graph.input) != 1 or len(model.graph.output) != 1:
            raise ValueError('Export must have one raw token input and one final pose output')
        def shape(value):
            return tuple(dim.dim_value for dim in value.type.tensor_type.shape.dim)
        if (model.graph.input[0].name != 'camera_tokens' or shape(model.graph.input[0]) != captured.INPUT_SHAPE
                or model.graph.output[0].name != 'pose_enc' or shape(model.graph.output[0]) != captured.OUTPUT_SHAPE):
            raise ValueError('Export input/output name or fixed shape mismatch')
        report['weight_sharing'] = weight_bindings(model, tensors, np, numpy_helper)
        onnx.checker.check_model(str(model_path), full_check=True)
        report['onnx'] = {'path': str(model_path), 'bytes': model_path.stat().st_size, 'sha256': sha(model_path),
                          'input_name': 'camera_tokens', 'input_shape': list(captured.INPUT_SHAPE),
                          'output_name': 'pose_enc', 'output_shape': list(captured.OUTPUT_SHAPE),
                          'node_count': len(model.graph.node), 'initializer_count': len(model.graph.initializer),
                          'opset': 17, 'constant_folding': False,
                          'all_original_parameters_verified': True}
        report['capture'] = {'path': str(receipt_path), 'sha256': capture_sha,
                             'input': metadata['artifacts']['input.npy'], 'output': metadata['artifacts']['output.npy'],
                             'unique_state_bytes': metadata['unique_state_bytes'], 'cpu_reference': reference}
        report['architecture'] = architecture
        report['source_and_input_hashes'] = pins.copy()
        report['source_and_input_hashes_after'] = {path: sha(path) for path in pins}
        report['source_and_inputs_unchanged'] = report['source_and_input_hashes_after'] == pins
        if not report['source_and_inputs_unchanged']:
            raise ValueError('Original source/captured input/state changed during export')
        report['status'] = 'exported_and_structurally_checked'
    except BaseException as exc:
        report.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        for hook in hooks:
            try:
                hook.remove()
            except BaseException as exc:
                report['cleanup_errors'].append(f'Hook cleanup: {exc}')
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
        report['total_seconds_including_hash_trace_IO'] = time.monotonic() - started
        with (output / 'export.json').open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--capture-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True, help='New export directory')
    parser.add_argument('--onnx-site', type=Path, default=BASE / 'experiments/vggt-v15-camera/onnx-site')
    args = parser.parse_args()
    captured, _ = load_capture_helper()
    captured.require_board()
    if Path.cwd().resolve() != BASE or Path(__file__).resolve().parents[3] != SOURCE:
        parser.error('Run the deployed source from board /mnt/vggt-data/vggt-slam2')
    output = args.output.resolve()
    if (not output.is_relative_to(BASE / 'experiments/vggt-v15-camera') or os.path.lexists(args.output)
            or output == args.capture_dir.resolve()):
        parser.error('Use a new output directory below board experiments/vggt-v15-camera')
    output.mkdir(parents=True, exist_ok=False)
    report = export(output, args.capture_dir.resolve(strict=True), args.capture_sha256, args.onnx_site.resolve(strict=True))
    path = output / 'export.json'
    print(json.dumps({'status': report['status'], 'error': report.get('error'),
                      'output': str(path), 'sha256': sha(path)}))
    return 0 if report['status'] == 'exported_and_structurally_checked' else 1


if __name__ == '__main__':
    sys.exit(main())
