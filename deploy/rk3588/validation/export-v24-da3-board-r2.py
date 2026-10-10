#!/usr/bin/env python3
"""Fixed-S2 export-only position-grid adapter for the frozen DA3 exporter.

The original PositionGetter's integer grid is copied and verified once. Only
the wrapper/ONNX trace uses that constant grid; original Net references still
run through the unmodified official getter. No learned weight/source changes.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
PARENT_PATH = Path(__file__).with_name('export-v24-da3-board.py')
PARENT_SHA256 = '4e503b58f4072d5bc819abbe35e24b06535f6d4fab842b34f418684288d13b13'
if hashlib.sha256(PARENT_PATH.read_bytes()).hexdigest() != PARENT_SHA256:
    raise ValueError('Frozen R1 export implementation SHA mismatch')
spec = importlib.util.spec_from_file_location('v24_frozen_export_r1', PARENT_PATH)
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)
ORIGINAL_EXPORT = parent.export_model
ORIGINAL_ORT = parent.check_ort
ORIGINAL_MAKE_WRAPPER = parent.make_wrapper
BASE, SHAPE, OUTPUTS = parent.BASE, parent.SHAPE, parent.OUTPUTS
sha, compare, record, official_codec = parent.sha, parent.compare, parent.record, parent.official_codec


def grid_metadata(grid):
    array = grid.detach().cpu().contiguous().numpy()
    if array.dtype != np.int64 or array.ndim != 3 or array.shape[-1] != 2:
        raise ValueError('Original PositionGetter must return int64 [batch,patches,2]')
    return dict(shape=list(array.shape), dtype='int64', bytes=array.nbytes,
                sha256=hashlib.sha256(memoryview(array).cast('B')).hexdigest())


class FixedPositionGetter:
    def __init__(self, original, batch, height, width):
        import torch
        if min(batch, height, width) < 1:
            raise ValueError('Positive fixed grid dimensions required')
        self.batch, self.height, self.width = batch, height, width
        original_grid = original(batch, height, width, device=torch.device('cpu'))
        actual = original_grid.detach().cpu().contiguous().numpy()
        yy, xx = np.meshgrid(np.arange(height, dtype=np.int64), np.arange(width, dtype=np.int64), indexing='ij')
        expected = np.broadcast_to(np.stack((yy, xx), axis=-1).reshape(1, height*width, 2), (batch, height*width, 2))
        if (original_grid.device.type != 'cpu' or original_grid.dtype != torch.int64
                or not np.array_equal(actual, expected)):
            raise ValueError('Official PositionGetter grid/order differs from fixed row-major y,x grid')
        self.grid = original_grid.detach().contiguous().clone()
        self.metadata = grid_metadata(self.grid)

    def __call__(self, batch_size, height, width, device):
        import torch
        # Trace receives symbolic scalar dimensions. The fixed export validates
        # them once rather than allowing a new dynamic Cartesian-product branch.
        if (int(batch_size), int(height), int(width)) != (self.batch, self.height, self.width):
            raise ValueError('The export-only constant position grid has a different fixed shape')
        if torch.device(device).type != 'cpu':
            raise ValueError('Fixed export position grid is CPU only')
        return self.grid.clone()


def bind_parent(report, pins):
    parent.require_sha(PARENT_PATH, PARENT_SHA256)
    pins[str(PARENT_PATH.resolve())] = PARENT_SHA256
    report.update(parent_implementation=dict(file=PARENT_PATH.name, sha256=PARENT_SHA256),
                  export_entrypoint=Path(__file__).name,
                  export_only_change='verified constant original PositionGetter grid; fixed B1/S2/252x336')


def export_model(args, report, pins):
    bind_parent(report, pins)
    restorations = []
    previous = parent.make_wrapper
    def make_fixed_wrapper(model):
        encoder = model.backbone.pretrained
        original = encoder.position_getter
        fixed = FixedPositionGetter(original, 2, 18, 24)
        report['fixed_position_grid'] = dict(**fixed.metadata, batch=2, patch_height=18,
                                           patch_width=24, y_x_row_major_verified=True,
                                           exactly_equal_original_getter=True,
                                           source='original official PositionGetter(2,18,24,cpu)')
        restorations.append((encoder, original))
        encoder.position_getter = fixed
        return ORIGINAL_MAKE_WRAPPER(model)
    parent.make_wrapper = make_fixed_wrapper
    try:
        return ORIGINAL_EXPORT(args, report, pins)
    finally:
        parent.make_wrapper = previous
        for encoder, original in restorations:
            encoder.position_getter = original
        report['position_getter_restored'] = all(encoder.position_getter is original for encoder,original in restorations)
        if not report['position_getter_restored']:
            report.setdefault('cleanup_errors', []).append('Original PositionGetter was not restored')


def check_ort(args, report, pins):
    bind_parent(report, pins)
    return ORIGINAL_ORT(args, report, pins)


def main(argv=None):
    # Parent main uses this entrypoint path for defaults and script provenance.
    # The separate frozen implementation is also explicitly pinned before/after.
    parent.__file__ = str(Path(__file__).resolve())
    parent.export_model, parent.check_ort = export_model, check_ort
    return parent.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
