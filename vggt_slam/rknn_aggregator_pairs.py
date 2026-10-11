"""24 complete original frame/global RKNN pairs, with the Aggregator unchanged.

Frame forward executes both original blocks and owns both outputs. Global
forward accepts only the unchanged contiguous view of that returned frame and
consumes its pending global result. All layers, intermediate concatenations,
special tokens and target tokens remain in the original Aggregator loop.
Install before the DINO tail cache and close that cache first. No CPU fallback
or numerical/trajectory acceptance is implied by this execution backend.
"""
from __future__ import annotations

import ctypes as ct
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import threading
import time

from .rknn_aggregator_blocks import (
    ABI_SHA, BASE, CHECKPOINT_SHA, EPS, HEADER_SHA, LIBRARY_SHA, SHAPES,
    STATE_KEYS, RKNNOriginalAggregatorBlocks, artifact, floating_attr, pin,
)

COMPOSER_SHA = '3ff8b0d0515ff44ea3625bd2c6529d63a0bed3487d409128ee4b443922a14bfd'
OUTPUT_BINDINGS = [dict(index=0, name='frame_output', shape=list(SHAPES['frame']), dtype='float32'),
                   dict(index=1, name='global_output', shape=list(SHAPES['global']), dtype='float32')]


def pair_output_attr(base, attr, binding):
    row = base.attr_record(attr)
    identity = row['type'] == 1 and row['quantization_type'] == 2 and row['zero_point'] == 0 and row['scale'] == 1.
    count = math.prod(binding['shape'])
    if (row['index'] != binding['index'] or row['name'] != binding['name']
            or row['shape'] != binding['shape'] or row['n_elems'] != count
            or row['type'] not in (0, 1) or (row['quantization_type'] != 0 and not identity)
            or row['format'] != 3 or row['pass_through'] != 0 or not math.isfinite(row['scale'])
            or row['size'] != count * (4 if row['type'] == 0 else 2)):
        raise ValueError(f'Expected fixed original floating pair output: {row}')
    return row


class RKNNOriginalAggregatorPairs(RKNNOriginalAggregatorBlocks):
    def __init__(self, model, manifest_path, manifest_sha256, *, source_root=BASE / 'src',
                 library_path='/usr/lib/librknnrt.so', core_mask=7, timeout_ms=30000):
        self.lock = threading.RLock()
        self.contexts, self.patches, self.guard = [], [], None
        self.pending, self.next_index, self.aggregator_globals = {}, 0, {}
        self.closed = self.failed = False
        self.api = self.base = self.aggregator = None
        self.stats = dict(context_init_attempts=0, contexts_created=0, contexts_destroyed=0,
            sdk_run_attempts=0, sdk_runs_succeeded=0, sdk_io_seconds=0., completed_blocks=0,
            completed_pairs=0, block_calls={}, pair_calls={}, bridge_check_seconds=0.,
            cleanup_errors=[], model_memory=[], all_operators_on_npu_verified=False,
            candidate_accepted=False)
        try:
            if type(core_mask) is not int or core_mask not in (1, 7) or type(timeout_ms) is not int or not 0 < timeout_ms <= 600000:
                raise ValueError('core_mask must be 1/7; timeout_ms must be 1..600000')
            source_root = Path(source_root).resolve(strict=True)
            abi_path = pin(source_root / 'deploy/rk3588/validation/run-v12-rknn-block-board.py', ABI_SHA)
            spec = importlib.util.spec_from_file_location('_v55_original_pair_ABI', abi_path)
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
            if (manifest['schema'] != 'v55_original_VGGT_24_frame_global_pair_RKNN_manifest'
                    or manifest['image_shape'] != [2, 3, 84, 112]
                    or manifest['checkpoint'] != {'sha256': CHECKPOINT_SHA, 'bytes': 5026874952}
                    or manifest['depth'] != 24 or manifest['aa_order'] != ['frame', 'global']
                    or manifest['aa_block_size'] != 1 or len(manifest['blocks']) != 24
                    or manifest['pair_composer_source_sha256'] != COMPOSER_SHA
                    or manifest['all_original_48_blocks_retained'] is not True
                    or manifest['original_aggregator_concat_must_be_retained'] is not True):
                raise ValueError('Only 24 complete original frame/global two-output pairs are supported')
            for relative, expected in manifest['source_hashes'].items():
                source = (source_root / relative).resolve(strict=True)
                if not source.is_relative_to(source_root):
                    raise ValueError('Source path escapes original source root')
                pin(source, expected)
            agg = model.aggregator
            self.aggregator = agg
            original_forward = getattr(agg, 'forward', None)
            self.aggregator_globals = getattr(getattr(original_forward, '__func__', original_forward), '__globals__', {})
            if (any(m.training for m in model.modules())
                    or getattr(agg, '_vggt_rknn_blocks_owner', None) is not None
                    or getattr(agg, '_vggt_patch_cache_owner', None) is not None
                    or len(agg.frame_blocks) != 24 or len(agg.global_blocks) != 24
                    or agg.aa_order != ['frame', 'global'] or agg.aa_block_size != 1 or agg.aa_block_num != 24
                    or agg.patch_start_idx != 5 or model.inference_image_size != 112
                    or (agg.retained_layer_indices is not None and not {4, 11, 17, 23} <= set(agg.retained_layer_indices))):
                raise ValueError('Install once on the unchanged original eval aggregator before its patch cache')
            prepared, indices = [], set()
            for row in manifest['blocks']:
                index = row['index']
                if row['role'] != 'pair' or type(index) is not int or not 0 <= index < 24 or index in indices:
                    raise ValueError('Exactly original pair indices 0..23 are required')
                indices.add(index)
                if (row['input_name'] != 'input' or row['input_shape'] != list(SHAPES['frame'])
                        or row['output_bindings'] != OUTPUT_BINDINGS or row['heads'] != 16
                        or set(row['parents']) != set(SHAPES) or set(row['positions']) != set(SHAPES)
                        or set(row['parameter_raw_sha256']) != set(SHAPES)
                        or row['all_36_original_parameter_values_preserved_by_parent_composition'] is not True
                        or row['parent_nodes_plus_one_reshape'] is not True
                        or row['original_aggregator_concat_must_be_retained'] is not True):
                    raise ValueError('Complete original pair composition and public bindings differ')
                artifact(directory, row)
                path = artifact(directory, row['rknn'])
                blocks, positions = {}, {}
                for role, shape in SHAPES.items():
                    parent = row['parents'][role]
                    if (parent['role'] != role or parent['index'] != index
                            or parent['input_shape'] != list(shape) or parent['output_shape'] != list(shape)
                            or parent['input_name'] != 'input' or parent['output_name'] != 'output'
                            or parent['heads'] != 16 or parent['positions'] != row['positions'][role]
                            or parent['parameter_raw_sha256'] != row['parameter_raw_sha256'][role]):
                        raise ValueError('Original frame/global parent identities differ')
                    pos_path = artifact(directory, row['positions'][role])
                    pos = np.load(pos_path, allow_pickle=False)
                    if (pos.dtype != np.int64 or pos.shape != (*shape[:2], 2) or not pos.flags.c_contiguous
                            or parent['positions']['shape'] != list(pos.shape) or parent['positions']['dtype'] != 'int64'
                            or parent['positions']['tensor_bytes'] != pos.nbytes
                            or hashlib.sha256(memoryview(pos).cast('B')).hexdigest() != parent['positions']['raw_sha256']):
                        raise ValueError('Fixed original RoPE positions differ')
                    block = getattr(agg, role + '_blocks')[index]
                    state = dict(block.named_parameters())
                    if set(state) != STATE_KEYS or set(parent['parameter_raw_sha256']) != STATE_KEYS:
                        raise ValueError('All 18 original parameters in each complete parent are required')
                    for name, value in state.items():
                        if value.device.type != 'cpu' or value.dtype != torch.float32 or not value.is_contiguous():
                            raise ValueError('Original parameters must stay dense CPU FP32')
                        array = value.detach().numpy()
                        if not np.isfinite(array).all() or hashlib.sha256(memoryview(array).cast('B')).hexdigest() != parent['parameter_raw_sha256'][name]:
                            raise ValueError(f'Original {role}{index} parameter differs: {name}')
                    if (set(parent['epsilons']) != set(EPS) or block.attn.num_heads != 16
                            or parent['gelu_approximate'] != 'none' or getattr(block.mlp.act, 'approximate', None) != 'none'
                            or parent['rope_frequency'] != 100 or parent['rope_scaling_factor'] != 1.
                            or block.attn.rope is None or block.attn.rope.base_frequency != 100
                            or block.attn.rope.scaling_factor != 1.
                            or parent['all_18_caller_parameter_values_preserved_in_graph'] is not True):
                        raise ValueError('Original normalization/attention/exact GELU differs')
                    for name in EPS:
                        norm = block
                        for field in name.split('.'):
                            norm = getattr(norm, field)
                        if parent['epsilons'][name] != float(norm.eps):
                            raise ValueError('Original LayerNorm epsilon changed')
                    blocks[role], positions[role] = block, pos.copy()
                if not np.array_equal(positions['frame'].reshape(positions['global'].shape), positions['global']):
                    raise ValueError('Global positions must be the original frame position view')
                prepared.append((index, blocks, positions, path))
            self.api = self.base.bind_api(ct.CDLL(str(pin(library_path, LIBRARY_SHA))))
            self.stats.update(manifest_sha256=manifest_sha256, core_mask=core_mask, timeout_ms=timeout_ms,
                verified_original_parameters=864, preserved_intermediate_layers=[4, 11, 17, 23],
                original_aggregator_loop_preserved=True, two_owned_outputs_per_sdk_run=True)
            for index, blocks, positions, path in prepared:
                context = self.base.CONTEXT()
                self.contexts.append(context)  # Retain partial failed init for destruction.
                filename = ct.create_string_buffer(str(path).encode())
                self.stats['context_init_attempts'] += 1
                code = self.api.rknn_init(ct.byref(context), ct.cast(filename, ct.c_void_p), 0, 0, None)
                self.stats['contexts_created'] += int(bool(context.value))
                self.base.check_call(code, 'rknn_init')
                if not context.value:
                    raise RuntimeError('RKNN returned an empty pair context')
                self.base.check_call(self.api.rknn_set_core_mask(context, core_mask), 'rknn_set_core_mask')
                version = self.base.query(self.api, context, 5, self.base.SDKVersion())
                if not re.match(r'^2\.3\.2(?:\D|$)', bytes(version.api_version).decode()):
                    raise ValueError('Frozen typed ABI requires RKNN runtime 2.3.2')
                counts = self.base.query(self.api, context, 0, self.base.InputOutputNum())
                if (counts.n_input, counts.n_output) != (1, 2):
                    raise ValueError('Each complete pair requires one input and both original outputs')
                input_attr = floating_attr(self.base, self.base.query(self.api, context, 1,
                    self.base.TensorAttr(index=0)), SHAPES['frame'], 'input')
                output_attrs = [pair_output_attr(self.base, self.base.query(self.api, context, 2,
                    self.base.TensorAttr(index=binding['index'])), binding) for binding in OUTPUT_BINDINGS]
                memory = self.base.query(self.api, context, 6, self.base.MemSize())
                self.stats['model_memory'].append(dict(block=f'pair{index}',
                    sdk_dma_bytes=int(memory.total_dma_allocated_size), input=input_attr, outputs=output_attrs))
                for role, block in blocks.items():
                    forward = self._forward(block, role, index, context, positions[role], input_attr['format'])
                    self.patches.append((block, 'forward' in block.__dict__, block.__dict__.get('forward'), block.forward, forward))
            # Change no forward until all 24 contexts and all 864 original parameters passed.
            self.guard = agg.register_forward_pre_hook(self._guard, with_kwargs=True)
            for block, _, _, _, forward in self.patches:
                block.forward = forward
            agg._vggt_rknn_blocks_owner = self  # Excludes simultaneous 48-block installation.
        except BaseException:
            self.failed = True
            self.close()
            raise

    def _invalidate(self):
        self.pending.clear()
        self.next_index = 0
        self.failed = True

    def _guard(self, module, args, kwargs):
        with self.lock:
            try:
                super()._guard(module, args, kwargs)
                if self.pending or self.next_index != 0 or self.aggregator_globals.get('viz_attention', False) is not False:
                    raise RuntimeError('Original pair loop incomplete or attention visualization requested; no fallback')
            except BaseException:
                self._invalidate()
                raise

    def _forward(self, block, role, index, context, expected_pos, fmt):
        shape, key = SHAPES[role], f'{role}{index}'
        def forward(x, pos=None, viz_attention=False):
            with self.lock:
                try:
                    if (self.closed or self.failed or block.training or self.torch.is_grad_enabled()
                            or viz_attention is not False or not isinstance(x, self.torch.Tensor)
                            or x.device.type != 'cpu' or x.dtype != self.torch.float32 or tuple(x.shape) != shape
                            or x.requires_grad or x.layout != self.torch.strided
                            or not isinstance(pos, self.torch.Tensor) or pos.device.type != 'cpu' or pos.dtype != self.torch.int64
                            or tuple(pos.shape) != expected_pos.shape or not self.np.array_equal(pos.detach().numpy(), expected_pos)
                            or index != self.next_index):
                        raise RuntimeError(f'{key}: unsupported original pair input/position/order/mode; no fallback')
                    if role == 'frame':
                        if self.pending:
                            raise RuntimeError('Previous frame/global pair has not been consumed')
                        array = x.detach().contiguous().numpy()
                        if not self.np.isfinite(array).all():
                            raise ValueError(f'{key}: nonfinite original input')
                        frame, global_ = self._run_pair(context, array, fmt)
                        if not self.np.isfinite(frame).all() or not self.np.isfinite(global_).all():
                            raise ValueError(f'{key}: nonfinite RKNN pair output')
                        result = self.torch.from_numpy(frame)
                        # A snapshot also detects caller writes through a NumPy alias;
                        # inference-mode Torch tensors need not have version counters.
                        self.pending[index] = (threading.get_ident(), frame, frame.copy(), global_)
                    else:
                        begin = time.perf_counter()
                        try:
                            pending = self.pending.get(index)
                            if pending is None or pending[0] != threading.get_ident() or not x.is_contiguous():
                                raise RuntimeError('Global must consume its same-thread pending frame view')
                            array = x.detach().numpy()
                            _, frame, snapshot, global_ = pending
                            if (array.ctypes.data != frame.ctypes.data or not self.np.shares_memory(array, frame)
                                    or not self.np.array_equal(array.reshape(frame.shape), snapshot)):
                                raise RuntimeError('Global input is not the unchanged original frame output view')
                            result = self.torch.from_numpy(global_)
                            del self.pending[index]
                            self.next_index = (index + 1) % 24
                            self.stats['completed_pairs'] += 1
                            pair_key = f'pair{index}'
                            self.stats['pair_calls'][pair_key] = self.stats['pair_calls'].get(pair_key, 0) + 1
                        finally:
                            self.stats['bridge_check_seconds'] += time.perf_counter() - begin
                    self.stats['completed_blocks'] += 1
                    self.stats['block_calls'][key] = self.stats['block_calls'].get(key, 0) + 1
                    return result
                except BaseException:
                    self._invalidate()
                    raise
        return forward

    def _run_pair(self, context, array, fmt):
        base, api = self.base, self.api
        inputs = (base.Input * 1)(base.Input(0, array.ctypes.data, array.nbytes, 0, 0, fmt))
        outputs = (base.Output * 2)()
        for index in range(2):
            outputs[index].index, outputs[index].want_float = index, 1
        run, received = base.RunExtend(0, 0, self.timeout, -1), base.OutputExtend()
        got, begin = False, time.perf_counter()
        try:
            base.check_call(api.rknn_inputs_set(context, 1, inputs), 'rknn_inputs_set')
            self.stats['sdk_run_attempts'] += 1
            code = api.rknn_run(context, ct.byref(run))
            self.stats['sdk_runs_succeeded'] += int(code == 0)
            base.check_call(code, 'rknn_run')
            base.check_call(api.rknn_outputs_get(context, 2, outputs, ct.byref(received)), 'rknn_outputs_get')
            got = True
            if run.frame_id != received.frame_id:
                raise RuntimeError('Synchronous pair output frame ID mismatch')
            owned = []
            for index, role in enumerate(('frame', 'global')):
                shape = SHAPES[role]
                if not outputs[index].buf or outputs[index].size != math.prod(shape) * 4:
                    raise RuntimeError('Pair output pointer/FP32 byte-size mismatch')
                view = self.np.ctypeslib.as_array(ct.cast(outputs[index].buf, ct.POINTER(ct.c_float)), shape=(math.prod(shape),))
                owned.append(view.copy().reshape(shape))
            return tuple(owned)  # Both own their memory before either SDK buffer is released.
        finally:
            try:
                if got or any(output.buf for output in outputs):
                    code = api.rknn_outputs_release(context, 2, outputs)
                    if code:
                        self.stats['cleanup_errors'].append(f'rknn_outputs_release returned {code}')
                    base.check_call(code, 'rknn_outputs_release')
            finally:
                self.stats['sdk_io_seconds'] += time.perf_counter() - begin

    def close(self):
        with self.lock:
            self.pending.clear()
            self.next_index = 0
            if not self.closed and self.guard is not None:
                try:
                    self.guard.remove()
                except BaseException as exc:
                    self.stats['cleanup_errors'].append(f'Hook removal: {type(exc).__name__}: {exc}')
                finally:
                    self.guard = None
            super().close()

    def metrics(self):
        with self.lock:
            return dict(super().metrics(), pending_pairs=len(self.pending))


def install_rknn_aggregator_pairs(model, manifest_path, manifest_sha256, **options):
    return RKNNOriginalAggregatorPairs(model, manifest_path, manifest_sha256, **options)
