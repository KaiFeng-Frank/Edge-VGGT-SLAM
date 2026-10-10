#!/usr/bin/env python3
"""Probe the fixed [1,37,1024] original DINO graph on RK3588S, without ORB.

The frozen V12 ABI, floating tensor gates, synchronous calls and cleanup are
reused in a private module namespace with only SHAPE and the default core mask
specialized. Normal timing and SDK performance collection are separate runs.
This tests one block; it does not establish whole VGGT pose speed or quality.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode = True
BASE = Path('/mnt/vggt-data/vggt-slam2')
SHAPE = (1, 37, 1024)
RUNTIME_SHA = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
EXPORTER_SHA = 'e6389143bc24be4096cb69c2d7092c6437a780bf4a59df70e76b45c696a335fa'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen(name, expected, module_name):
    path = Path(__file__).with_name(name)
    if sha(path) != expected:
        raise ValueError(f'Frozen helper SHA mismatch: {name}')
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def specialized_runtime():
    runtime, runtime_path = load_frozen('run-v12-rknn-block-board.py', RUNTIME_SHA,
                                      '_v14_small_rknn_runtime')
    exporter, exporter_path = load_frozen('export-v14-small-dino.py', EXPORTER_SHA,
                                        '_v14_small_capture_contract')
    original_loader = runtime.load_capture
    # Globals belong solely to this fresh module instance; no frozen file or
    # production module is changed, and no model tensor is truncated/resized.
    runtime.SHAPE = SHAPE
    runtime.CORE_MASK = 7
    runtime.__doc__ = __doc__

    def load_small_capture(directory, expected_sha):
        inputs, reference, metadata, pins = original_loader(directory, expected_sha)
        exporter.validate_capture_metadata(metadata)
        capture, capture_path = exporter.load_frozen(
            'capture-v14-small-dino-board.py', exporter.CAPTURE_SCRIPT_SHA, '_v14_small_capture_provenance')
        original, original_path = capture.load_helper()
        if metadata['source_hashes'] != original.SOURCE_PINS:
            raise ValueError('Captured original source hashes do not match frozen DINO/preprocessing')
        pins.update({str(runtime_path): RUNTIME_SHA, str(exporter_path): EXPORTER_SHA,
                     str(capture_path): exporter.CAPTURE_SCRIPT_SHA,
                     str(original_path): capture.HELPER_SHA,
                     str(Path(__file__).resolve()): sha(__file__)})
        metadata = dict(metadata)
        metadata['runtime_shape_specialization'] = {
            'shape': list(SHAPE), 'default_core_mask': 7,
            'frozen_runtime_path': str(runtime_path), 'frozen_runtime_sha256': RUNTIME_SHA,
            'wrapper_path': str(Path(__file__).resolve()), 'wrapper_sha256': sha(__file__),
            'frozen_file_modified': False, 'tensor_truncation_or_padding': False,
            'operator_coverage_requires_separate_SDK_perf_detail': True,
        }
        return inputs, reference, metadata, pins

    runtime.load_capture = load_small_capture
    return runtime


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Narrow output placement without duplicating the frozen SDK driver parser.
    preflight = argparse.ArgumentParser(add_help=False)
    preflight.add_argument('--output', type=Path)
    known, _ = preflight.parse_known_args(argv)
    if known.output is not None:
        output = known.output.resolve()
        root = BASE / 'experiments/vggt-realtime-1hz'
        if not output.is_relative_to(root) or output == root:
            preflight.error('--output must be a new JSON below board experiments/vggt-realtime-1hz')
    return specialized_runtime().main(argv)


if __name__ == '__main__':
    sys.exit(main())
