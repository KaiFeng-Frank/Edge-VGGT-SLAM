#!/usr/bin/env python3
"""Compile the board-exported complete CameraHead; no host model inference.

Reuse the frozen RKNN 2.3.2 FP16/non-quantized compiler configuration. Host
cwd/TMPDIR/output must be in /dev/shm. Runtime targets and quality still need
actual board measurements. Do not use the former 256MiB block model budget.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
COMPILER_SHA = 'a40bf09e53ef9b378db4781be18f9e53f824afcc809c9009e0a65f4f0f91b4e5'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx', type=Path, required=True)
    parser.add_argument('--onnx-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not args.output_dir.resolve().is_relative_to(Path('/dev/shm')):
        parser.error('Host compiled output must be in /dev/shm')
    helper_path = Path(__file__).with_name('compile-v12-rknn-block.py')
    if sha(helper_path) != COMPILER_SHA:
        parser.error('Frozen compiler helper SHA mismatch')
    before = sha(__file__)
    spec = importlib.util.spec_from_file_location('_v15_camera_frozen_compiler', helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    code = module.main()
    path = args.output_dir / 'compile.json'
    if path.is_file():
        report = json.loads(path.read_text())
        report.update(kind='complete_original_CameraHead_RKNN_host_compilation',
                      wrapper_sha256=before, frozen_compiler_sha256=COMPILER_SHA,
                      complete_camera_iterations=4, constant_weights_expected_shared=True,
                      host_model_inference_executed=False,
                      wrapper_and_frozen_compiler_unchanged=(sha(__file__) == before and sha(helper_path) == COMPILER_SHA))
        if not report['wrapper_and_frozen_compiler_unchanged']:
            report.update(status='failed', error='Compiler wrapper/source changed during compilation')
            code = 1
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return code


if __name__ == '__main__':
    sys.exit(main())
