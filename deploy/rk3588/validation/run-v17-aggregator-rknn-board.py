#!/usr/bin/env python3
"""Board-only single original aggregator block RKNN probe.

The frozen v12 driver supplies the checked C ABI, synchronous I/O and cleanup.
Its raw receipt retains its historical label; run.json identifies this actual
aggregator experiment. No graph operator coverage or full-model claim is inferred
from compilation or successful SDK calls.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
BASE = Path('/mnt/vggt-data/vggt-slam2')
HERE = Path(__file__).resolve().parent
RUNTIME_SHA = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
COMPILER_SHA = 'a40bf09e53ef9b378db4781be18f9e53f824afcc809c9009e0a65f4f0f91b4e5'
LIBRARY_SHA = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4 << 20), b''):
            digest.update(part)
    return digest.hexdigest()


def load_helper(name, expected):
    path = HERE / name
    if sha(path) != expected:
        raise ValueError(f'Helper identity mismatch: {name}')
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
        raise ValueError('Compile receipt does not bind this original FP16 graph/model')
    operations = receipt.get('operations', [])
    if [r['name'] for r in operations] != ['config', 'load_onnx', 'build', 'export_rknn'] or any(r['code'] not in (0, None) for r in operations):
        raise ValueError('Compiler did not complete all four required operations')


def check_export(export, exporter_sha, profile_sha, block, row, shape):
    required = dict(status='exported_and_structurally_checked', script_sha256=exporter_sha,
                    profile_sha256=profile_sha, block=block, input_shape=list(shape),
                    input_name='input', output_name='output', original_weights=True,
                    parameter_count=18, position_and_tables_fixed=True,
                    host_trigonometry_recomputed=False, host_model_inference_executed=False,
                    host_ORT_executed=False, external_data=False, external_data_used=False)
    if any(export.get(key) != value for key, value in required.items()):
        raise ValueError('Export receipt does not bind this original capture/block')
    if (export.get('capture_artifacts') != row['artifacts']
            or export.get('captured_array_metadata') != row['arrays']
            or export.get('captured_parameter_metadata') != row['weights']):
        raise ValueError('Export weights, positions or original reference differ from capture')
    for key in ('input', 'output'):
        entry = export.get('artifacts', {}).get(key + '.npy', {})
        if (entry.get('shape') != list(shape) or entry.get('dtype') != 'float32'
                or entry.get('raw_sha256') != row['arrays'][key]['sha256']):
            raise ValueError('Export graph input/output metadata differs from capture')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('capture-dir', 'export-receipt', 'compile-receipt', 'model', 'output'):
        parser.add_argument('--' + key, required=True, type=Path)
    for key in ('profile-sha256', 'export-sha256', 'exporter-sha256', 'compile-sha256', 'model-sha256'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--exporter-name', choices=('export-v17-aggregator-block.py',
                         'export-v17-aggregator-block-r2.py', 'export-v17-aggregator-block-r3.py'),
                         default='export-v17-aggregator-block.py')
    parser.add_argument('--block', choices=('frame0', 'global0'), required=True)
    parser.add_argument('--collect-perf-detail', action='store_true')
    args = parser.parse_args(argv)
    runtime = load_helper('run-v12-rknn-block-board.py', RUNTIME_SHA)
    runtime.require_board()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(BASE / 'experiments') or args.output == BASE / 'experiments':
        parser.error('New output directory must be below board experiments')
    exporter = load_helper(args.exporter_name, args.exporter_sha256)
    profile, row, arrays, weights, pins = exporter.load_capture(args.capture_dir, args.profile_sha256, args.block)
    pins[str(HERE / args.exporter_name)] = args.exporter_sha256
    for path, expected in ((args.export_receipt, args.export_sha256), (args.compile_receipt, args.compile_sha256),
                           (args.model, args.model_sha256)):
        if sha(path) != expected:
            raise ValueError(f'Explicit SHA mismatch: {path.name}')
        pins[str(path.resolve())] = expected
    export = json.loads(args.export_receipt.read_text())
    compiled = json.loads(args.compile_receipt.read_text())
    check_export(export, args.exporter_sha256, args.profile_sha256, args.block, row, arrays['input'].shape)
    check_compile(compiled, export['onnx']['sha256'], args.model_sha256)
    if args.model.stat().st_size != compiled['output']['size']:
        raise ValueError('Compiled model size mismatch')
    library = Path('/usr/lib/librknnrt.so')
    if sha(library) != LIBRARY_SHA:
        raise ValueError('RKNN runtime binary identity mismatch')
    pins[str(Path(__file__).resolve())] = sha(__file__)
    runtime.SHAPE = tuple(arrays['input'].shape)
    runtime.CORE_MASK = 7
    capture_metadata = dict(actual_kind='original_aggregator_block_capture', block=args.block,
                            profile_sha256=args.profile_sha256, export_sha256=args.export_sha256,
                            original_FP32_reference=True, fixed_original_pos_sha256=row['arrays']['pos']['sha256'])

    def load_actual_capture(directory, expected_sha):
        if Path(directory).resolve() != args.capture_dir.resolve() or expected_sha != args.profile_sha256:
            raise ValueError('Driver requested a different capture')
        return arrays['input'], arrays['output'], capture_metadata, dict(pins)

    runtime.load_capture = load_actual_capture
    original_execute = runtime.execute_call
    output_arrays = []

    def execute_and_save(*a, **kw):
        actual, timing = original_execute(*a, **kw)
        # This write is outside the already measured SDK I/O interval.
        import numpy as np
        path = args.output / f'actual_{len(output_arrays):02d}.npy'
        np.save(path, actual, allow_pickle=False)
        output_arrays.append(dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path)))
        return actual, timing

    runtime.execute_call = execute_and_save
    args.output.mkdir(parents=True, exist_ok=False)
    raw_path = args.output / 'driver.json'
    driver_args = ['--model', str(args.model), '--model-sha256', args.model_sha256,
                   '--capture-dir', str(args.capture_dir), '--capture-sha256', args.profile_sha256,
                   '--output', str(raw_path), '--core-mask', '7']
    if args.collect_perf_detail:
        driver_args.append('--collect-perf-detail')
    code = runtime.main(driver_args)
    raw = json.loads(raw_path.read_text())
    receipt = dict(kind='original_VGGT_aggregator_block_RKNN_probe', block=args.block, status=raw['status'],
                   script_sha256=sha(__file__), runtime_helper_sha256=RUNTIME_SHA,
                   exporter_name=args.exporter_name, exporter_sha256=args.exporter_sha256,
                   profile_sha256=args.profile_sha256,
                   export_sha256=args.export_sha256, compile_sha256=args.compile_sha256,
                   model_sha256=args.model_sha256, input_shape=list(runtime.SHAPE),
                   ORB_used=False, inference_on_board_only=True, full_model_run=False,
                   real_time_validated=False, accuracy_validated=False,
                   operator_coverage_requires_review=True, normal_benchmark=not args.collect_perf_detail,
                   output_arrays=output_arrays, driver_receipt=dict(file=raw_path.name, sha256=sha(raw_path)),
                   raw_driver_kind_is_legacy_label=True,
                   npu_run_api_call_count=raw['npu_run_api_call_count'],
                   successful_synchronous_call_count=raw['successful_synchronous_call_count'],
                   cleanup_errors=raw['cleanup_errors'], context_destroyed=raw.get('context_destroyed'),
                   error=raw.get('error'))
    (args.output / 'run.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt, indent=2))
    return code


if __name__ == '__main__':
    sys.exit(main())
