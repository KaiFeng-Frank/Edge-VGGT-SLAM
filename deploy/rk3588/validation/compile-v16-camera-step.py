#!/usr/bin/env python3
"""Compile the board-exported shared CameraHead step; no host inference."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
COMPILER_SHA = 'a40bf09e53ef9b378db4781be18f9e53f824afcc809c9009e0a65f4f0f91b4e5'
EXPORTER_SHA = 'b26efe9c6cd62a2b7977e6e9c096cf0bd9bbf63076a063b9a332ef59565af258'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx', type=Path, required=True)
    parser.add_argument('--onnx-sha256', required=True)
    parser.add_argument('--export-receipt', type=Path, required=True)
    parser.add_argument('--export-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if (not args.output_dir.resolve().is_relative_to(Path('/dev/shm'))
            or not Path.cwd().resolve().is_relative_to(Path('/dev/shm'))
            or not Path(os.environ.get('TMPDIR', '')).resolve().is_relative_to(Path('/dev/shm'))):
        parser.error('Host cwd, TMPDIR and output must be below /dev/shm')
    receipt = args.export_receipt.resolve(strict=True)
    if sha(receipt) != args.export_sha256:
        parser.error('Export receipt SHA mismatch')
    meta = json.loads(receipt.read_text())
    if (meta.get('status') != 'exported_and_structurally_checked'
            or meta.get('script_sha256') != EXPORTER_SHA or meta.get('cleanup_errors')
            or meta.get('source_and_inputs_unchanged') is not True
            or meta.get('outer_iteration_contract', {}).get('all_four_original_iterations_array_equal') is not True
            or meta.get('weight_sharing', {}).get('original_parameters_verified') != 68
            or meta.get('onnx', {}).get('sha256') != args.onnx_sha256
            or sha(args.onnx) != args.onnx_sha256):
        parser.error('Exact shared-step export provenance is required')
    helper_path = Path(__file__).with_name('compile-v12-rknn-block.py')
    if sha(helper_path) != COMPILER_SHA:
        parser.error('Frozen RKNN compiler SHA mismatch')
    spec = importlib.util.spec_from_file_location('_v16_original_RKNN_compiler', helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_argv, before = sys.argv, sha(__file__)
    try:
        sys.argv = [str(helper_path), '--onnx', str(args.onnx), '--onnx-sha256', args.onnx_sha256,
                    '--output-dir', str(args.output_dir)]
        code = module.main()
    finally:
        sys.argv = original_argv
    path = args.output_dir / 'compile.json'
    if path.is_file():
        result = json.loads(path.read_text())
        result.update(kind='original_shared_CameraHead_step_RKNN_host_compilation', wrapper_sha256=before,
                      frozen_compiler_sha256=COMPILER_SHA, export_receipt_sha256=args.export_sha256,
                      exporter_script_sha256=EXPORTER_SHA, trunk_blocks_per_step=4,
                      required_outer_iterations=4, original_parameter_tensors_per_graph=68,
                      host_model_inference_executed=False,
                      packed_weight_sharing_runtime_verified=False,
                      wrapper_and_frozen_compiler_unchanged=(sha(__file__) == before and sha(helper_path) == COMPILER_SHA
                                                           and sha(receipt) == args.export_sha256))
        if not result['wrapper_and_frozen_compiler_unchanged']:
            result.update(status='failed', error='Compiler/export provenance changed')
            code = 1
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return code


if __name__ == '__main__':
    sys.exit(main())
