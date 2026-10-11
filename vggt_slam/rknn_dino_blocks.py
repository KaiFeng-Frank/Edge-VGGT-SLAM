"""RKNN execution of all 24 blocks inside VGGT's original DINO module.

Token preparation, the original block loop, and final normalization stay in
Torch FP32. Each unchanged block graph takes one [1,53,1024] frame; a cold
two-frame batch runs that context twice in order. Install before the tail patch
cache; close that cache before this handle. No model or numerical acceptance is
implied by selecting this execution backend.
"""
from __future__ import annotations

import ctypes as ct
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import threading

from .rknn_aggregator_blocks import (
    ABI_SHA, BASE, CHECKPOINT_SHA, HEADER_SHA, LIBRARY_SHA,
    RKNNOriginalAggregatorBlocks, artifact, floating_attr, pin,
)

SHAPE = (1, 53, 1024)
STATE_KEYS = frozenset((
    'norm1.weight', 'norm1.bias', 'attn.qkv.weight', 'attn.qkv.bias',
    'attn.proj.weight', 'attn.proj.bias', 'ls1.gamma', 'norm2.weight',
    'norm2.bias', 'mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight',
    'mlp.fc2.bias', 'ls2.gamma',
))


class RKNNOriginalDinoBlocks(RKNNOriginalAggregatorBlocks):
    """Own 24 contexts/forward overrides; reuse the tested owned-copy SDK IO."""

    def __init__(self, model, manifest_path, manifest_sha256, *, source_root=BASE / 'src',
                 library_path='/usr/lib/librknnrt.so', core_mask=7, timeout_ms=30000):
        self.lock = threading.RLock()
        self.contexts, self.patches, self.guard = [], [], None
        self.closed = self.failed = False
        self.api = self.base = self.dino = None
        self.stats = dict(context_init_attempts=0, contexts_created=0, contexts_destroyed=0,
            sdk_run_attempts=0, sdk_runs_succeeded=0, sdk_io_seconds=0., completed_blocks=0,
            completed_frame_blocks=0, block_calls={}, block_frame_calls={}, batch1_blocks=0,
            batch2_blocks=0, cleanup_errors=[], model_memory=[], candidate_accepted=False,
            all_operators_on_npu_verified=False)
        try:
            if type(core_mask) is not int or core_mask not in (1, 7) or type(timeout_ms) is not int or not 0 < timeout_ms <= 600000:
                raise ValueError('core_mask must be 1/7; timeout_ms must be 1..600000')
            source_root = Path(source_root).resolve(strict=True)
            abi_path = pin(source_root / 'deploy/rk3588/validation/run-v12-rknn-block-board.py', ABI_SHA)
            spec = importlib.util.spec_from_file_location('_v54_original_dino_ABI', abi_path)
            self.base = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.base)
            self.stats['board'] = self.base.require_board()
            pin(source_root / 'deploy/rk3588/reference/rknn_api_v2.3.2.h', HEADER_SHA)
            import numpy as np
            import torch
            self.np, self.torch, self.timeout = np, torch, timeout_ms
            manifest_path = pin(manifest_path, manifest_sha256)
            manifest = json.loads(manifest_path.read_text())
            directory = manifest_path.parent
            if (manifest['schema'] != 'v54_original_VGGT_24_DINO_block_RKNN_manifest'
                    or manifest['image_shape'] != [1, 3, 84, 112] or manifest['depth'] != 24
                    or manifest['checkpoint'] != {'sha256': CHECKPOINT_SHA, 'bytes': 5026874952}
                    or len(manifest['blocks']) != 24):
                raise ValueError('Only the complete original 24-block DINO 84x112 graph set is supported')
            for relative, expected in manifest['source_hashes'].items():
                source = (source_root / relative).resolve(strict=True)
                if not source.is_relative_to(source_root):
                    raise ValueError('Source path escapes original source root')
                pin(source, expected)
            dino = model.aggregator.patch_embed
            self.dino = dino
            if (any(m.training for m in model.modules())
                    or getattr(dino, '_vggt_rknn_dino_owner', None) is not None
                    or getattr(dino, '_vggt_patch_cache_owner', None) is not None
                    or getattr(model.aggregator, '_vggt_patch_cache_owner', None) is not None
                    or dino.chunked_blocks is not False or len(dino.blocks) != 24 or dino.n_blocks != 24
                    or dino.embed_dim != 1024 or dino.num_heads != 16 or dino.patch_size != 14
                    or dino.num_register_tokens != 4 or model.inference_image_size != 112):
                raise ValueError('Install once on original eval CPU DINO before its tail cache')
            prepared, indices = [], set()
            for row in manifest['blocks']:
                index = row['index']
                if row['role'] != 'dino' or type(index) is not int or not 0 <= index < 24 or index in indices:
                    raise ValueError('Exactly original DINO indices 0..23 are required')
                indices.add(index)
                if (row['input_shape'] != list(SHAPE) or row['output_shape'] != list(SHAPE)
                        or row['input_name'] != 'input' or row['output_name'] != 'output' or row['heads'] != 16):
                    raise ValueError('Original DINO block shapes and public IO names differ')
                artifact(directory, row)
                rknn_path = artifact(directory, row['rknn'])
                block = dino.blocks[index]
                state = dict(block.named_parameters())
                if set(state) != STATE_KEYS or set(row['parameter_raw_sha256']) != STATE_KEYS:
                    raise ValueError('All 14 original DINO block parameters are required')
                for name, value in state.items():
                    if value.device.type != 'cpu' or value.dtype != torch.float32 or not value.is_contiguous():
                        raise ValueError('Original DINO parameters must stay contiguous CPU FP32')
                    array = value.detach().numpy()
                    if not np.isfinite(array).all() or hashlib.sha256(memoryview(array).cast('B')).hexdigest() != row['parameter_raw_sha256'][name]:
                        raise ValueError(f'Original dino{index} parameter mismatch: {name}')
                if (row['epsilons'] != {'norm1': float(block.norm1.eps), 'norm2': float(block.norm2.eps)}
                        or block.attn.num_heads != 16 or block.attn.rope is not None
                        or not isinstance(block.attn.q_norm, torch.nn.Identity)
                        or not isinstance(block.attn.k_norm, torch.nn.Identity)
                        or row['gelu_approximate'] != 'none' or getattr(block.mlp.act, 'approximate', None) != 'none'
                        or row['all_14_caller_parameter_values_preserved_in_graph'] is not True):
                    raise ValueError('Original DINO normalization/attention/exact GELU contract differs')
                prepared.append((block, index, rknn_path))
            self.api = self.base.bind_api(ct.CDLL(str(pin(library_path, LIBRARY_SHA))))
            self.stats.update(manifest_sha256=manifest_sha256, core_mask=core_mask, timeout_ms=timeout_ms,
                verified_original_parameters=336, preserved_dino_blocks=24,
                token_preparation_preserved=True, final_norm_preserved=True)
            for block, index, path in prepared:
                context = self.base.CONTEXT()
                self.contexts.append(context)  # A failed init may still return a context needing destruction.
                filename = ct.create_string_buffer(str(path).encode())
                self.stats['context_init_attempts'] += 1
                code = self.api.rknn_init(ct.byref(context), ct.cast(filename, ct.c_void_p), 0, 0, None)
                self.stats['contexts_created'] += int(bool(context.value))
                self.base.check_call(code, 'rknn_init')
                if not context.value:
                    raise RuntimeError('RKNN returned an empty DINO context')
                self.base.check_call(self.api.rknn_set_core_mask(context, core_mask), 'rknn_set_core_mask')
                version = self.base.query(self.api, context, 5, self.base.SDKVersion())
                if not re.match(r'^2\.3\.2(?:\D|$)', bytes(version.api_version).decode()):
                    raise ValueError('Frozen typed ABI requires RKNN runtime 2.3.2')
                counts = self.base.query(self.api, context, 0, self.base.InputOutputNum())
                if (counts.n_input, counts.n_output) != (1, 1):
                    raise ValueError('Each original DINO graph requires one input and one output')
                attrs = [floating_attr(self.base, self.base.query(self.api, context, cmd, self.base.TensorAttr(index=0)), SHAPE, name)
                         for cmd, name in ((1, 'input'), (2, 'output'))]
                memory = self.base.query(self.api, context, 6, self.base.MemSize())
                key = f'dino{index}'
                self.stats['model_memory'].append(dict(block=key, sdk_dma_bytes=int(memory.total_dma_allocated_size),
                                                       input=attrs[0], output=attrs[1]))
                forward = self._forward(block, key, context, attrs[0]['format'])
                self.patches.append((block, 'forward' in block.__dict__, block.__dict__.get('forward'), block.forward, forward))
            # Original module methods stay intact until all 24 contexts passed their typed queries.
            self.guard = dino.register_forward_pre_hook(self._guard, with_kwargs=True)
            for block, _, _, _, forward in self.patches:
                block.forward = forward
            dino._vggt_rknn_dino_owner = self
        except BaseException:
            self.failed = True
            self.close()
            raise

    def _guard(self, module, args, kwargs):
        images = args[0] if args else kwargs.get('x')
        masks = args[1] if len(args) > 1 else kwargs.get('masks')
        if (self.closed or self.failed or module.training or self.torch.is_grad_enabled()
                or len(args) > 2 or masks is not None or kwargs.get('is_training', True) is not True
                or not isinstance(images, self.torch.Tensor) or images.device.type != 'cpu'
                or images.dtype != self.torch.float32 or images.requires_grad
                or tuple(images.shape) not in ((1, 3, 84, 112), (2, 3, 84, 112))):
            raise RuntimeError('DINO RKNN requires the original CPU FP32 one/two-frame image input; no fallback')

    def _forward(self, block, key, context, fmt):
        def forward(x):
            with self.lock:
                if (self.closed or self.failed or block.training or self.torch.is_grad_enabled()
                        or not isinstance(x, self.torch.Tensor) or x.device.type != 'cpu'
                        or x.dtype != self.torch.float32 or x.requires_grad or x.layout != self.torch.strided
                        or tuple(x.shape) not in (SHAPE, (2, *SHAPE[1:]))):
                    raise RuntimeError(f'{key}: expected original one/two-frame DINO tokens; no fallback')
                array = x.detach().contiguous().numpy()
                if not self.np.isfinite(array).all():
                    raise ValueError(f'{key}: nonfinite original DINO input')
                outputs = []
                try:
                    for frame in range(len(array)):
                        output = self._run(context, array[frame:frame+1], fmt, SHAPE)
                        if not self.np.isfinite(output).all():
                            raise ValueError(f'{key}: nonfinite RKNN DINO output')
                        outputs.append(output)
                        self.stats['completed_frame_blocks'] += 1
                        self.stats['block_frame_calls'][key] = self.stats['block_frame_calls'].get(key, 0) + 1
                    merged = outputs[0] if len(outputs) == 1 else self.np.concatenate(outputs, axis=0)
                    result = self.torch.from_numpy(merged)
                    self.stats['completed_blocks'] += 1
                    self.stats['block_calls'][key] = self.stats['block_calls'].get(key, 0) + 1
                    self.stats[f'batch{len(outputs)}_blocks'] += 1
                    return result
                except BaseException:
                    self.failed = True
                    raise
        return forward

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            if self.guard is not None:
                try:
                    self.guard.remove()
                except BaseException as exc:
                    self.stats['cleanup_errors'].append(f'Hook removal: {type(exc).__name__}: {exc}')
            for block, had, value, original, forward in self.patches:
                if block.forward is forward:
                    if had:
                        block.forward = value
                    else:
                        del block.forward
                elif block.forward != original:
                    self.stats['cleanup_errors'].append('DINO block forward changed after installation; not overwritten')
            if self.dino is not None and getattr(self.dino, '_vggt_rknn_dino_owner', None) is self:
                del self.dino._vggt_rknn_dino_owner
            for context in self.contexts:
                if context.value and self.api is not None:
                    try:
                        self.base.check_call(self.api.rknn_destroy(context), 'rknn_destroy')
                        self.stats['contexts_destroyed'] += 1
                    except BaseException as exc:
                        self.stats['cleanup_errors'].append(f'{type(exc).__name__}: {exc}')
                    finally:
                        context.value = 0
            if self.stats['cleanup_errors']:
                raise RuntimeError('RKNN DINO cleanup failed: ' + '; '.join(self.stats['cleanup_errors']))


def install_rknn_dino_blocks(model, manifest_path, manifest_sha256, **options):
    return RKNNOriginalDinoBlocks(model, manifest_path, manifest_sha256, **options)
