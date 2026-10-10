#!/usr/bin/env python3
"""Keep original DA3 DPT positional arithmetic as fixed export constants.

Only the export wrapper uses cached FP32 positional embeddings from the pinned
official functions. Original Net references and all learned weights stay intact.
The frozen R1/R2 implementations remain separate and explicitly hash-bound.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import types

sys.dont_write_bytecode = True
R2_PATH = Path(__file__).with_name('export-v24-da3-board-r2.py')
R2_SHA256 = 'a4cc8bf99dfea17e054c9838c55daeaa6b8cf6c686c245bc78af5166dfe79c60'
if hashlib.sha256(R2_PATH.read_bytes()).hexdigest() != R2_SHA256:
    raise ValueError('Frozen R2 position adapter SHA mismatch')
spec = importlib.util.spec_from_file_location('v24_frozen_export_r2', R2_PATH)
r2 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = r2
spec.loader.exec_module(r2)
parent = r2.parent
BASE, SHAPE, OUTPUTS = r2.BASE, r2.SHAPE, r2.OUTPUTS
sha, compare, record, official_codec = r2.sha, r2.compare, r2.record, r2.official_codec


class FixedHeadPositionEmbedding:
    def __init__(self, create_grid, grid_to_embed):
        self.create_grid, self.grid_to_embed = create_grid, grid_to_embed
        self.cache = {}
        self.calls = 0

    def __call__(self, x, W, H, ratio=0.1):
        import torch
        if x.ndim != 4 or x.dtype != torch.float32 or x.device.type != 'cpu':
            raise ValueError('Fixed DPT position embeddings require CPU FP32 NCHW features')
        batch, channels, ph, pw = map(int, x.shape)
        W, H, ratio = int(W), int(H), float(ratio)
        if min(batch, channels, ph, pw, W, H) < 1 or channels % 4 or ratio != 0.1:
            raise ValueError('Invalid original DPT position-grid dimensions/ratio')
        if (W, H) != (336, 252):
            raise ValueError('Fixed original 252x336 image geometry required')
        key = (batch, channels, ph, pw, W, H, ratio, str(x.dtype), str(x.device))
        if key not in self.cache:
            if torch.jit.is_tracing():
                raise ValueError('Unseen DPT positional-embedding key during fixed ONNX trace')
            grid = self.create_grid(pw, ph, aspect_ratio=W/H, dtype=x.dtype, device=x.device)
            pe = self.grid_to_embed(grid, channels)*ratio
            # The original expands the batch dimension. Broadcasting this exact
            # singleton tensor preserves the same per-pixel FP32 addition.
            pe = pe.permute(2, 0, 1)[None].detach().contiguous()
            if pe.dtype != torch.float32 or tuple(pe.shape) != (1, channels, ph, pw) or not torch.isfinite(pe).all():
                raise ValueError('Invalid FP32 positional embedding from original functions')
            self.cache[key] = pe
        self.calls += 1
        return x+self.cache[key]

    def records(self):
        # Hashing is outside the wrapper/trace forward timing regions.
        result = []
        for key, pe in self.cache.items():
            batch, channels, ph, pw, W, H, ratio, dtype, device = key
            result.append(dict(batch=batch, channels=channels, patch_height=ph, patch_width=pw,
                               original_width=W, original_height=H, ratio=ratio, dtype=dtype,
                               device=device, constant=record(pe.numpy()),
                               original_expanded_shape=[batch, channels, ph, pw]))
        return result


def bind_parent(report, pins):
    parent.require_sha(R2_PATH, R2_SHA256)
    pins[str(R2_PATH.resolve())] = R2_SHA256
    report.update(parent_position_implementation=dict(file=R2_PATH.name, sha256=R2_SHA256),
                  export_entrypoint=Path(__file__).name,
                  DPT_PE_export_only_change='official FP32 create_uv_grid + position_grid_to_embed * ratio; x+samePE',
                  wrapper_check_includes_position_cache_construction=True)


def export_model(args, report, pins):
    bind_parent(report, pins)
    previous = r2.ORIGINAL_MAKE_WRAPPER
    restorations, caches = [], []
    def make_fixed_head_wrapper(model):
        from depth_anything_3.model.utils.head_utils import create_uv_grid, position_grid_to_embed
        head = model.head
        old = head._add_pos_embed
        own = '_add_pos_embed' in head.__dict__
        own_value = head.__dict__.get('_add_pos_embed')
        cache = FixedHeadPositionEmbedding(create_uv_grid, position_grid_to_embed)
        restorations.append((head, old, own, own_value))
        caches.append(cache)
        def fixed(_head, x, W, H, ratio=0.1):
            return cache(x, W, H, ratio)
        head._add_pos_embed = types.MethodType(fixed, head)
        return previous(model)
    r2.ORIGINAL_MAKE_WRAPPER = make_fixed_head_wrapper
    try:
        return r2.export_model(args, report, pins)
    finally:
        r2.ORIGINAL_MAKE_WRAPPER = previous
        for head, _old, own, own_value in restorations:
            if own:
                head._add_pos_embed = own_value
            else:
                delattr(head, '_add_pos_embed')
        report['DPT_position_embedding_restored'] = all(
            head._add_pos_embed == old and ('_add_pos_embed' in head.__dict__) == own
            for head,old,own,_value in restorations)
        report['fixed_DPT_position_embeddings'] = [row for cache in caches for row in cache.records()]
        report['fixed_DPT_position_calls'] = sum(cache.calls for cache in caches)
        report['export_entrypoint'] = Path(__file__).name
        if not report['DPT_position_embedding_restored']:
            report.setdefault('cleanup_errors', []).append('Original DPT position method was not restored')


def check_ort(args, report, pins):
    bind_parent(report, pins)
    try:
        return r2.check_ort(args, report, pins)
    finally:
        report['export_entrypoint'] = Path(__file__).name


def main(argv=None):
    parent.__file__ = str(Path(__file__).resolve())
    parent.export_model, parent.check_ort = export_model, check_ort
    return parent.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
