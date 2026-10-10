#!/usr/bin/env python3
"""Construct the original 37-token DINO block graph, without model inference.

Uses the frozen V12 builder and exact Erf GELU fusion rewrite. The rewrite is
not a tanh approximation, but FP32 operation order changes: board ORT and RKNN
numerics and actual operator targets still require separate validation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
SHAPE = (1, 37, 1024)
CAPTURE_SCRIPT_SHA = '3799e1297bb95166a9039ef657245d88d3fc078ebd63a8f8524e17796fe30272'
BUILDER_SHA = '2759fce1551b2a64da140a7be0b16c9d233241e0bb427b4d52c58d2243d6bf07'
GELU_SHA = '59f5ea1be93812571947f7afb71ea9d908057cf277c3cbc2cfb9bbaaf10e5a78'
CHECKPOINT_SHA = 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
IMAGE_SHA = 'c23e57b9612522a10e61b1723749acddfe5b96b3a3642ec8c9db06e604d7f511'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError(f'Invalid required SHA256 for {path}')
    if sha(path) != expected:
        raise ValueError(f'SHA256 mismatch for {path}')
    return expected


def load_frozen(name, expected, module_name):
    path = Path(__file__).with_name(name)
    require_sha(path, expected)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def validate_capture_metadata(metadata):
    if (metadata.get('kind') != 'official_dino_block0_CPU_capture'
            or metadata.get('experiment') != 'v14_original_DINO_crop112'
            or metadata.get('status') != 'completed'
            or metadata.get('script_sha256') != CAPTURE_SCRIPT_SHA):
        raise ValueError('An exact completed V14 original crop112 CPU capture is required')
    required = {'source_unchanged': True, 'image_unchanged': True,
                'checkpoint_stat_unchanged': True, 'frozen_helper_unchanged': True,
                'full_VGGT_constructed': False, 'no_npu': True, 'npu_used': False, 'orb_used': False,
                'original_DINO_depth': 24, 'constructed_DINO_depth': 1, 'executed_blocks': [0]}
    if any(metadata.get(key) != value for key, value in required.items()):
        raise ValueError('Capture provenance/source or original CPU execution contract mismatch')
    if (metadata.get('checkpoint', {}).get('sha256') != CHECKPOINT_SHA
            or metadata.get('checkpoint', {}).get('bytes') != 5026874952
            or metadata.get('image', {}).get('sha256') != IMAGE_SHA
            or metadata.get('source_hashes') != metadata.get('source_hashes_after')):
        raise ValueError('Original checkpoint/image/source provenance mismatch')
    architecture = metadata.get('architecture', {})
    exact = {'embed_dim': 1024, 'num_heads': 16, 'head_dim': 64, 'patch_size': 14,
             'mlp_hidden_dim': 4096, 'norm1_eps': 1e-6, 'norm2_eps': 1e-6,
             'gelu_approximate': 'none', 'qk_norm': False, 'rope': False,
             'dropout': 0., 'drop_path': 0., 'eval': True, 'num_register_tokens': 4,
             'cls_tokens': 1, 'patch_tokens': 32, 'sequence_length': 37,
             'position_included_in_input': True, 'interpolate_antialias': True,
             'interpolate_offset': 0., 'constructor_image_size': 518,
             'original_position_weights_shape': [1, 1370, 1024]}
    if any(architecture.get(key) != value for key, value in exact.items()):
        raise ValueError('Original 518 position weights / 37-token block architecture mismatch')
    if any(metadata.get('preprocessing', {}).get(key) != value for key, value in
           {'mode': 'crop', 'image_size': 112, 'normalized_shape': [1, 3, 56, 112]}.items()):
        raise ValueError('Actual original crop112 preprocessing is required')
    cpu = metadata.get('cpu_reference', {})
    if any(cpu.get(key) != value for key, value in
           {'device': 'cpu', 'dtype': 'float32', 'eval': True, 'no_grad': True,
            'finite': True, 'all_elementwise_equal': True, 'repeats': 3,
            'capture_call_excluded': False}.items()):
        raise ValueError('Original three-call CPU reference is required')
    times = cpu.get('seconds', [])
    if len(times) != 3 or any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in times):
        raise ValueError('Invalid three-call CPU timing evidence')
    if metadata.get('cleanup_errors') or metadata.get('affinity_restored') is not True:
        raise ValueError('Capture cleanup did not complete')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--capture-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True, help='New ONNX file')
    parser.add_argument('--report', type=Path, required=True, help='New export JSON')
    args = parser.parse_args(argv)
    directory = args.capture_dir.resolve(strict=True)
    receipt_path = directory / 'capture.json'
    protected = {receipt_path, *(directory / name for name in ('input.npy', 'output.npy', 'block-state.npz')),
                 Path(__file__).resolve()}
    output, report_path = args.output.resolve(), args.report.resolve()
    if (output == report_path or output in protected or report_path in protected
            or os.path.lexists(args.output) or os.path.lexists(args.report)):
        parser.error('Use distinct new output/report paths, separate from inputs and source')
    pins = {str(receipt_path): require_sha(receipt_path, args.capture_sha256),
            str(Path(__file__).resolve()): sha(__file__)}
    metadata = json.loads(receipt_path.read_text())
    validate_capture_metadata(metadata)
    capture_module, capture_source = load_frozen('capture-v14-small-dino-board.py', CAPTURE_SCRIPT_SHA,
                                               '_v14_small_capture_source')
    original, original_source = capture_module.load_helper()
    if metadata['source_hashes'] != original.SOURCE_PINS:
        raise ValueError('Capture original source hashes do not match the frozen implementation')
    pins[str(capture_source)] = CAPTURE_SCRIPT_SHA
    pins[str(original_source)] = capture_module.HELPER_SHA
    builder, builder_path = load_frozen('export-v12-dino-block.py', BUILDER_SHA, '_v14_small_builder')
    rewriter, rewriter_path = load_frozen('export-v12-dino-block-gelu.py', GELU_SHA, '_v14_small_exact_gelu')
    pins[str(builder_path)], pins[str(rewriter_path)] = BUILDER_SHA, GELU_SHA
    import numpy as np
    import onnx
    from onnx import helper
    for filename in ('input.npy', 'output.npy', 'block-state.npz'):
        path = directory / filename
        item = metadata['artifacts'][filename]
        pins[str(path)] = require_sha(path, item['sha256'])
        if path.stat().st_size != item['bytes']:
            raise ValueError(f'Artifact file size mismatch: {filename}')
        if filename != 'block-state.npz':
            array = np.load(path, allow_pickle=False, mmap_mode='r')
            info = original.array_info(array)
            if (array.shape != SHAPE or item['shape'] != list(SHAPE)
                    or item['dtype'] != 'float32' or item['tensor_sha256'] != info['sha256']):
                raise ValueError(f'Artifact tensor contract mismatch: {filename}')
    with np.load(directory / 'block-state.npz', allow_pickle=False) as saved:
        state = {key: np.ascontiguousarray(saved[key]) for key in saved.files}
    if set(state) != set(builder.STATE_KEYS) or set(metadata['artifacts']['block-state.npz']['tensors']) != set(state):
        raise ValueError('Exactly 14 original block weight keys are required')
    for key, value in state.items():
        if original.array_info(value) != metadata['artifacts']['block-state.npz']['tensors'][key]:
            raise ValueError(f'Original block tensor metadata/hash mismatch: {key}')
    model = rewriter.rewrite_gelu(builder.build_block_onnx(state, SHAPE, heads=16, layer_norm='standard'))
    helper.set_model_props(model, {
        'experiment': 'v14_original_DINO_crop112', 'capture_sha256': args.capture_sha256,
        'checkpoint_sha256': CHECKPOINT_SHA, 'image_sha256': IMAGE_SHA,
        'input_sha256': metadata['artifacts']['input.npy']['sha256'],
        'block_state_sha256': metadata['artifacts']['block-state.npz']['sha256'],
        'builder_sha256': BUILDER_SHA, 'gelu_rewriter_sha256': GELU_SHA,
        'exporter_sha256': sha(__file__), 'input_shape': json.dumps(list(SHAPE)),
        'gelu_pattern': 'Div -> Erf -> Add(1) -> Mul(hidden) -> Mul(0.5)',
        'position_preparation': 'original 518 weights interpolated by actual crop112 CPU capture',
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(model.SerializeToString())
    onnx.checker.check_model(str(output), full_check=True)
    after = {path: sha(path) for path in pins}
    if pins != after:
        raise RuntimeError('A captured input, weight, or frozen source changed during export')
    report = {
        'status': 'exported_and_structurally_checked', 'kind': 'v14_original_small_DINO_block_export',
        'output': {'path': str(output), 'sha256': sha(output), 'bytes': output.stat().st_size},
        'input_shape': list(SHAPE), 'output_shape': list(SHAPE), 'heads': 16,
        'cpu_reference': metadata['cpu_reference'], 'architecture': metadata['architecture'],
        'source_and_input_hashes': pins, 'source_and_input_hashes_after': after,
        'source_and_inputs_unchanged': True, 'node_count': len(model.graph.node),
        'op_counts': dict(Counter(node.op_type for node in model.graph.node)),
        'onnx_version': onnx.__version__, 'opset': 17, 'ir_version': model.ir_version,
        'mathematical_gelu': 'exact Erf GELU; no tanh approximation',
        'fp32_operation_order_changed': True, 'model_inference_executed': False,
        'numerical_equivalence_validated': False, 'all_operators_on_npu_verified': False,
        'full_model_quality_validated': False, 'real_time_validated': False, 'production_modified': False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'output': str(output), 'sha256': sha(output),
                      'report': str(report_path), 'report_sha256': sha(report_path)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
