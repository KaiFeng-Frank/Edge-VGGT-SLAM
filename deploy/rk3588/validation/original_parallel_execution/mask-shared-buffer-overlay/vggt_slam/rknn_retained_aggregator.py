"""Four complete original VGGT aggregator groups with retained consumers.

Keep original image/DINO/special-token/position preparation, four Torch
concatenations and global20 clone/norm. All 24 frame/global pairs execute;
unconsumed concatenation allocations are omitted, never simulated with dummy
block outputs. Install before patch_cache; close that cache first. This
diagnostic backend does not imply FP32, trajectory or realtime acceptance.
"""
from __future__ import annotations

import copy
from vggt_slam.shared_context_registry import initialize as initialize_shared_context, destroy as destroy_shared_context

import ctypes as ct
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import threading
import time

from vggt_slam import rknn_aggregator_blocks as original

BASE = original.BASE
PARENT_SHA = '1872f2ed730f4b6d139562d831661126a088b1ac758249d6a1688cd7e9bfae42'
HELPER_SHA = '4fea665334e9d12a3d55839c3150680e269bab9794074d698bed2ad0a6e41c93'
NEON_LIBRARY_SHA = 'd08147c438461669ad8ebd3c6483e0a7908053e95cc6f83d0fe7eb7a5cd668c9'
NEON_SOURCE_SHA = '77707b735a6291c417a997ed64acbe4e1cd60cc5f9dd150d83fe1836868bd5e3'
RANGES = ((0, 4), (5, 11), (12, 17), (18, 23))
RETAINED = (4, 11, 17, 23)
FRAME, GLOBAL = original.SHAPES['frame'], original.SHAPES['global']
HOOK_FIELDS = ('_forward_pre_hooks', '_forward_hooks', '_backward_pre_hooks', '_backward_hooks')
CONFIG = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
    mean_values=None, std_values=None, quant_img_RGB2BGR=False, model_pruning=False,
    remove_reshape=False, quantize_weight=False, enable_flash_attention=False)
BUILD = dict(do_quantization=False, dataset=None, auto_hybrid=False)


def method_identity(value):
    return id(getattr(value, '__self__', None)), id(getattr(value, '__func__', value))


def artifact(directory, row):
    """Groups may exceed the old per-block 256 MiB limit; cap each at 1 GiB."""
    directory = Path(directory).resolve(strict=True)
    relative = Path(row['file'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Relative group artifact required')
    path = (directory / relative).resolve(strict=True)
    if (not path.is_relative_to(directory) or not path.is_file()
            or type(row['bytes']) is not int or not 0 < row['bytes'] <= 1 << 30
            or path.stat().st_size != row['bytes']):
        raise ValueError('Group artifact size/path differs or exceeds 1 GiB')
    return original.pin(path, row['sha256'])


def bindings(ordinal):
    rows = [dict(index=0, name='frame_output', shape=list(FRAME), dtype='float32'),
            dict(index=1, name='global_output', shape=list(GLOBAL), dtype='float32')]
    if ordinal == 3:
        rows.append(dict(index=2, name='target_global_output', shape=list(GLOBAL), dtype='float32'))
    return rows


def half_descriptor(base, attr, shape, name, index):
    row = base.attr_record(attr)
    identity = row['quantization_type'] == 2 and row['zero_point'] == 0 and row['scale'] == 1.
    if (row['index'] != index or row['name'] != name or row['shape'] != list(shape)
            or row['type'] != 1 or row['format'] != 3 or row['pass_through'] != 0
            or row['n_elems'] != math.prod(shape) or row['size'] != 2 * math.prod(shape)
            or not math.isfinite(row['scale']) or (row['quantization_type'] != 0 and not identity)):
        raise ValueError('Dense original FP16 NONE/identity-affine descriptor required: ' + str(row))
    return row


class RKNNRetainedOriginalAggregator:
    def __init__(self, model, manifest_path, manifest_sha256, *, half_io_library,
                 half_io_library_sha256, source_root=BASE / 'src',
                 library_path='/usr/lib/librknnrt.so', core_mask=7, timeout_ms=30000,
                 parent_manifest=BASE / 'experiments/vggt-v55-pair-graphs-r1/compiled.json'):
        self.lock = threading.RLock()
        self.closed = self.failed = self.active = self.installed = False
        self.contexts, self.groups, self.guard = [], [], None
        self.aggregator = self.base = self.api = self.neon = self.forward_callable = None
        self.stats = dict(context_init_attempts=0, contexts_created=0, contexts_destroyed=0,
            sdk_run_attempts=0, sdk_runs_succeeded=0, outputs_acquired=0, outputs_released=0,
            sdk_io_seconds=0., completed_groups=0, completed_pairs=0, completed_blocks=0,
            completed_aggregator_calls=0, group_calls={}, aggregator_forward_seconds=0.,
            guard_seconds=0., cleanup_errors=[], model_memory=[], candidate_accepted=False,
            numerical_or_trajectory_quality_accepted=False, all_operators_on_npu_verified=False)
        self.stats['half_io'] = dict(calls_attempted=0, calls_completed=0,
            input_conversion_calls=0, output_conversion_calls=0,
            **{name + '_seconds': 0. for name in ('input_conversion', 'output_conversion',
                'owned_half_copy', 'sdk_api', 'route', 'rknn_inputs_set', 'rknn_run',
                'rknn_outputs_get', 'rknn_outputs_release')})
        try:
            if (type(core_mask) is not int or core_mask not in (1, 2, 4, 7)
                    or type(timeout_ms) is not int or not 0 < timeout_ms <= 600000
                    or half_io_library_sha256 != NEON_LIBRARY_SHA):
                raise ValueError('Frozen NEON library, mask1/7 and timeout1..600000 required')
            self.model, self.timeout, self.core_mask = model, timeout_ms, core_mask
            source_root = Path(source_root).resolve(strict=True)
            helper = original.pin(source_root / 'vggt_slam/rknn_aggregator_blocks.py', HELPER_SHA)
            if Path(original.__file__).resolve() != helper:
                raise ValueError('Typed helper import escaped selected source root')
            abi = original.pin(source_root / 'deploy/rk3588/validation/run-v12-rknn-block-board.py', original.ABI_SHA)
            original.pin(source_root / 'deploy/rk3588/reference/rknn_api_v2.3.2.h', original.HEADER_SHA)
            spec = importlib.util.spec_from_file_location('_v67_retained_group_ABI', abi)
            self.base = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.base)
            self.stats['board'] = self.base.require_board()
            import numpy as np
            import torch
            self.np, self.torch = np, torch
            parent_path = original.pin(parent_manifest, PARENT_SHA)
            parent = json.loads(parent_path.read_text())
            manifest_path = original.pin(manifest_path, manifest_sha256)
            manifest = json.loads(manifest_path.read_text())
            directory = manifest_path.parent
            checkpoint = dict(sha256=original.CHECKPOINT_SHA, bytes=5026874952)
            if (parent['schema'] != 'v55_original_VGGT_24_frame_global_pair_RKNN_manifest'
                    or parent['checkpoint'] != checkpoint or len(parent['blocks']) != 24
                    or [r['index'] for r in parent['blocks']] != list(range(24))
                    or manifest['schema'] != 'v67_original_VGGT_retained_groups_RKNN_manifest'
                    or manifest['checkpoint'] != checkpoint or manifest['original_parent_manifest_sha256'] != PARENT_SHA
                    or manifest['source_hashes'] != parent['source_hashes']
                    or manifest['capture_sha256'] != parent['capture_sha256']
                    or manifest['image_shape'] != [2, 3, 84, 112] or manifest['depth'] != 24
                    or manifest['aa_order'] != ['frame', 'global'] or manifest['aa_block_size'] != 1
                    or len(manifest['blocks']) != 4 or manifest['retained_consumer_indices'] != list(RETAINED)
                    or manifest['all_original_24_pairs_retained'] is not True
                    or manifest['original_parameter_value_count'] != 864
                    or manifest['original_retained_concats_required'] is not True
                    or manifest['original_target_token_clone_and_norm_must_be_retained'] is not True
                    or manifest['original_target_tokens_global_index'] != 20
                    or manifest['status'] != 'compiled_not_quality_accepted'
                    or manifest['cleanup_errors'] or manifest['candidate_accepted'] is not False):
                raise ValueError('Complete original four-group retained/target contract differs')
            for relative, digest in manifest['source_hashes'].items():
                source = (source_root / relative).resolve(strict=True)
                if not source.is_relative_to(source_root):
                    raise ValueError('Original source escaped selected source root')
                original.pin(source, digest)
            agg = self.aggregator = model.aggregator
            if (any(m.training for m in model.modules())
                    or getattr(agg, '_vggt_rknn_blocks_owner', None) is not None
                    or getattr(agg, '_vggt_patch_cache_owner', None) is not None
                    or len(agg.frame_blocks) != 24 or len(agg.global_blocks) != 24
                    or agg.depth != 24 or agg.aa_order != ['frame', 'global'] or agg.aa_block_size != 1
                    or agg.aa_block_num != 24 or agg.patch_start_idx != 5 or agg.patch_size != 14
                    or agg.target_layer != 21 or model.inference_image_size != 112
                    or agg.retained_layer_indices != set(RETAINED)
                    or getattr(model, 'point_head', None) is not None or getattr(model, 'track_head', None) is not None):
                raise ValueError('Install once on original fixed camera/depth eval path before patch cache')
            if any(v.device.type != 'cpu' or v.dtype != torch.float32 or v.requires_grad for v in model.parameters()):
                raise ValueError('Original model parameters must remain frozen CPU FP32')
            self.original_forward = agg.forward
            if getattr(self.original_forward, '__func__', None) is not type(agg).forward:
                raise ValueError('Original Aggregator forward must be unmodified before installation')
            self.aggregator_globals = self.original_forward.__func__.__globals__
            self.slice_expand = self.aggregator_globals['slice_expand_and_flatten']
            if self.aggregator_globals.get('viz_attention', False) is not False:
                raise ValueError('Attention visualization cannot use retained group execution')
            self.original_had_forward = 'forward' in agg.__dict__
            self.original_forward_value = agg.__dict__.get('forward')
            self.original_patch_embed = agg.patch_embed
            self.original_position_getter = agg.position_getter
            self._reject_bypassed_hooks()
            prepared, positions, identities = [], {}, set()
            for ordinal, (row, (start, end)) in enumerate(zip(manifest['blocks'], RANGES)):
                indices, public = list(range(start, end + 1)), bindings(ordinal)
                if (row['role'] != 'retained_pair_group' or row['index'] != ordinal or row['indices'] != indices
                        or row['parents'] != parent['blocks'][start:end + 1]
                        or row['input_name'] != 'input' or row['input_shape'] != list(FRAME)
                        or row['output_bindings'] != public or row['heads'] != 16
                        or row['positions'] != {str(i): parent['blocks'][i]['positions'] for i in indices}
                        or row['parameter_raw_sha256'] != {str(i): parent['blocks'][i]['parameter_raw_sha256'] for i in indices}
                        or row['all_original_parent_nodes_attributes_and_initializer_payloads_retained'] is not True
                        or row['all_original_frame_and_global_computation_tensors_retained'] is not True
                        or row['original_FP32_parent_graph_math_preserved'] is not True
                        or row['only_retained_consumer_outputs_public'] is not True
                        or row['original_retained_concats_required'] is not True
                        or row['retained_consumer_indices'] != list(RETAINED)
                        or row['target_global_output_source_pair_index'] != (20 if ordinal == 3 else None)
                        or row['target_global_output_is_unnormalized'] != (True if ordinal == 3 else None)):
                    raise ValueError('Original contiguous group lineage/public output scope differs')
                expected_sources = [dict(index=i, name=b['name'], pair_index=end, role=role)
                    for i, (b, role) in enumerate(zip(public[:2], ('frame', 'global')))]
                if ordinal == 3:
                    expected_sources.append(dict(index=2, name='target_global_output', pair_index=20, role='global'))
                if row['public_output_sources'] != expected_sources:
                    raise ValueError('Original target/retained producer identities differ')
                artifact(directory, row)
                rknn = artifact(directory, row['rknn'])
                self._check_receipts(directory, row, ordinal, manifest['export_manifest_sha256'], rknn)
                for index in indices:
                    pair = parent['blocks'][index]
                    if (pair['role'] != 'pair' or pair['index'] != index or pair['input_shape'] != list(FRAME)
                            or pair['output_bindings'] != bindings(0) or set(pair['parents']) != set(original.SHAPES)
                            or pair['all_36_original_parameter_values_preserved_by_parent_composition'] is not True):
                        raise ValueError('Exact complete original parent pair required')
                    for role, shape in original.SHAPES.items():
                        identities.add((role, index))
                        p = pair['parents'][role]
                        if (p['role'] != role or p['index'] != index or p['input_shape'] != list(shape)
                                or p['output_shape'] != list(shape) or p['input_name'] != 'input' or p['output_name'] != 'output'
                                or p['heads'] != 16 or p['positions'] != pair['positions'][role]
                                or p['parameter_raw_sha256'] != pair['parameter_raw_sha256'][role]):
                            raise ValueError('Original frame/global parent identity differs')
                        pos = np.load(artifact(directory, p['positions']), allow_pickle=False)
                        if (pos.dtype != np.int64 or pos.shape != (*shape[:2], 2) or not pos.flags.c_contiguous
                                or p['positions']['shape'] != list(pos.shape) or p['positions']['dtype'] != 'int64'
                                or p['positions']['tensor_bytes'] != pos.nbytes
                                or hashlib.sha256(memoryview(pos).cast('B')).hexdigest() != p['positions']['raw_sha256']):
                            raise ValueError('Original fixed RoPE position identity differs')
                        if role in positions and not np.array_equal(positions[role], pos):
                            raise ValueError('Original per-layer positions differ')
                        positions[role] = pos.copy()
                        self._validate_block(getattr(agg, role + '_blocks')[index], p, f'{role}{index}')
                prepared.append((ordinal, indices, public, rknn))
            if len(identities) != 48 or not np.array_equal(positions['frame'].reshape(positions['global'].shape), positions['global']):
                raise ValueError('All48 original block identities and exact frame/global position views required')
            for value in positions.values():
                value.setflags(write=False)
            self.positions = positions
            neon_path = original.pin(half_io_library, half_io_library_sha256)
            original.pin(source_root / 'vggt_slam/native/fp16_neon.c', NEON_SOURCE_SHA)
            self.api = self.base.bind_api(ct.CDLL(str(original.pin(library_path, original.LIBRARY_SHA))))
            self.neon = ct.CDLL(str(neon_path))
            for name in ('f32_to_f16', 'f16_to_f32'):
                fn = getattr(self.neon, name)
                fn.argtypes, fn.restype = [ct.c_void_p, ct.c_void_p, ct.c_size_t], None
            for ordinal, indices, public, path in prepared:
                self._open_group(ordinal, indices, public, path)
            self.module_state, self.tensor_state = self._module_state(), self._tensor_state()
            self.config_state, self.backend_state = self._config_state(), self._backend_state()
            self.forward_callable = self._forward
            agg.forward = self.forward_callable
            agg._vggt_rknn_blocks_owner = self
            self.installed = True
            self.guard = agg.register_forward_pre_hook(self._guard, with_kwargs=True)
            self.stats.update(manifest_sha256=manifest_sha256, verified_original_parameters=864,
                preserved_intermediate_layers=list(RETAINED), target_tokens_source_global_index=20,
                all_original_48_blocks_retained=True, retained_Torch_concats=4, output_list_length=24,
                unconsumed_concat_allocations_omitted=20, original_prefix_preserved=True,
                source_hashes=dict(manifest['source_hashes']), core_mask=core_mask, timeout_ms=timeout_ms,
                init_flags=0, input_api_type=1, input_pass_through=0, output_want_float=0,
                half_io_library_sha256=NEON_LIBRARY_SHA, half_io_source_sha256=NEON_SOURCE_SHA,
                requested_float_dtype='float16', compiled_boundary_rounding_preservation_proven=False)
        except BaseException:
            self.failed = True
            self.close()
            raise

    def _open_group(self, ordinal, indices, public, path):
        context = self.base.CONTEXT()
        self.contexts.append(context)
        filename = ct.create_string_buffer(str(path).encode())
        self.stats['context_init_attempts'] += 1
        code = initialize_shared_context(self.api, ct.byref(context), ct.cast(filename, ct.c_void_p), 0, 0, None)
        self.stats['contexts_created'] += int(bool(context.value))
        self.base.check_call(code, 'rknn_init')
        if not context.value:
            raise ValueError('Empty original group context')
        self.base.check_call(self.api.rknn_set_core_mask(context, self.core_mask), 'rknn_set_core_mask')
        version = self.base.query(self.api, context, 5, self.base.SDKVersion())
        if not re.match(r'^2\.3\.2(?:\D|$)', bytes(version.api_version).decode()):
            raise ValueError('Frozen typed runtime SDK2.3.2 required')
        count = self.base.query(self.api, context, 0, self.base.InputOutputNum())
        if (count.n_input, count.n_output) != (1, len(public)):
            raise ValueError('Original two/three output group SDK ABI differs')
        inp = half_descriptor(self.base, self.base.query(self.api, context, 1, self.base.TensorAttr(index=0)), FRAME, 'input', 0)
        outs = [half_descriptor(self.base, self.base.query(self.api, context, 2, self.base.TensorAttr(index=b['index'])),
                b['shape'], b['name'], b['index']) for b in public]
        memory = self.base.query(self.api, context, 6, self.base.MemSize())
        self.stats['model_memory'].append(dict(group=ordinal, sdk_dma_bytes=int(memory.total_dma_allocated_size), input=inp, outputs=outs))
        self.groups.append((ordinal, indices, context, public))


    def _check_receipts(self, directory, row, ordinal, export_sha, rknn):
        compile_path = artifact(directory, row['compile_receipt'])
        checked_path = artifact(directory, row['check_receipt'])
        c, k = json.loads(compile_path.read_text()), json.loads(checked_path.read_text())
        if (c['schema'] != 'v67_original_VGGT_retained_group_check_compile' or k['schema'] != c['schema']
                or c['mode'] != 'compile' or k['mode'] != 'check'
                or c['group_index'] != ordinal or k['group_index'] != ordinal
                or c['indices'] != row['indices'] or k['indices'] != row['indices']
                or c['manifest_sha256'] != export_sha or k['manifest_sha256'] != export_sha
                or c['graph_sha256'] != row['sha256'] or k['graph_sha256'] != row['sha256']
                or c['check_receipt_sha256'] != row['check_receipt']['sha256']
                or c['original_parent_manifest_sha256'] != PARENT_SHA or k['original_parent_manifest_sha256'] != PARENT_SHA
                or c['status'] != 'compiled_not_quality_accepted'
                or k['status'] != 'checked_separate_ONNX_equivalence_not_quality_accepted'
                or c['configuration'] != CONFIG or c['build_options'] != BUILD
                or c['released'] is not True or c['release_code'] not in (None, 0)
                or c['cleanup_errors'] or k['cleanup_errors']
                or c['source_and_inputs_unchanged'] is not True or k['source_and_inputs_unchanged'] is not True
                or c['candidate_accepted'] is not False or k['candidate_accepted'] is not False
                or k['all_separate_ONNX_checks_byte_equal'] is not True
                or c['output_bindings'] != row['output_bindings'] or k['output_bindings'] != row['output_bindings']
                or c['script_sha256'] != k['script_sha256']
                or [v['name'] for v in c['operations']] != ['config', 'load_onnx', 'build', 'export_rknn']
                or any(v['status'] != 'returned' or v['code'] not in (None, 0) for v in c['operations'])
                or artifact(compile_path.parent, c['rknn']) != rknn):
            raise ValueError('Pinned complete per-group compile/check receipts differ')

    def _validate_block(self, block, parent, name):
        state = dict(block.named_parameters())
        if set(state) != original.STATE_KEYS or set(parent['parameter_raw_sha256']) != original.STATE_KEYS:
            raise ValueError('All18 original parameters per block required')
        for key, value in state.items():
            array = value.detach().numpy()
            if (value.device.type != 'cpu' or value.dtype != self.torch.float32 or not value.is_contiguous()
                    or value.requires_grad or not self.np.isfinite(array).all()
                    or hashlib.sha256(memoryview(array).cast('B')).hexdigest() != parent['parameter_raw_sha256'][key]):
                raise ValueError('Original parameter identity differs: ' + name + '.' + key)
        if (set(parent['epsilons']) != set(original.EPS) or block.attn.num_heads != 16
                or parent['gelu_approximate'] != 'none' or getattr(block.mlp.act, 'approximate', None) != 'none'
                or parent['rope_frequency'] != 100 or parent['rope_scaling_factor'] != 1.
                or block.attn.rope is None or block.attn.rope.base_frequency != 100
                or block.attn.rope.scaling_factor != 1.
                or parent['all_18_caller_parameter_values_preserved_in_graph'] is not True):
            raise ValueError('Original exact attention/GELU/RoPE contract differs')
        for key in original.EPS:
            norm = block
            for field in key.split('.'):
                norm = getattr(norm, field)
            if parent['epsilons'][key] != float(norm.eps):
                raise ValueError('Original LayerNorm epsilon differs')

    def _block_modules(self):
        for role in ('frame', 'global'):
            for index, block in enumerate(getattr(self.aggregator, role + '_blocks')):
                for name, module in block.named_modules():
                    yield f'{role}{index}.{name}', module

    def _reject_bypassed_hooks(self):
        global_module = self.torch.nn.modules.module
        if any(getattr(global_module, '_global' + name, {}) for name in HOOK_FIELDS):
            raise RuntimeError('Global module hooks cannot be bypassed by grouped execution')
        if any(getattr(module, field, {}) for _, module in self._block_modules() for field in HOOK_FIELDS):
            raise RuntimeError('Original frame/global block/submodule hooks cannot be silently bypassed')

    def _module_state(self):
        return tuple((name, id(m), type(m), m.training, id(type(m).forward), method_identity(m.forward))
                     for name, m in self._block_modules())

    def _tensor_state(self):
        values = []
        for role in ('frame', 'global'):
            blocks = getattr(self.aggregator, role + '_blocks')
            values.extend((role + '.' + n, v) for n, v in blocks.named_parameters())
            values.extend((role + '.' + n, v) for n, v in blocks.named_buffers())
        values.extend((n, getattr(self.aggregator, n)) for n in ('camera_token', 'register_token', '_resnet_mean', '_resnet_std'))
        return tuple((n, id(v), v._version, v.dtype, v.device, tuple(v.shape), tuple(v.stride()), v.data_ptr(), v.requires_grad)
                     for n, v in values)

    def _config_state(self):
        agg = self.aggregator
        return (id(type(agg).forward), tuple(agg.aa_order), agg.depth, agg.aa_block_size, agg.aa_block_num,
            agg.patch_size, agg.patch_start_idx, agg.target_layer, agg.use_reentrant,
            frozenset(agg.retained_layer_indices), id(agg.rope), agg.rope.base_frequency, agg.rope.scaling_factor,
            method_identity(agg._process_frame_attention), method_identity(agg._process_global_attention),
            id(agg.position_getter), method_identity(agg.position_getter.__call__), id(agg.patch_embed),
            tuple((role, index, block.attn.num_heads, getattr(block.attn, 'scale', None),
                block.attn.rope.base_frequency, block.attn.rope.scaling_factor,
                block.mlp.act.approximate, tuple(self._norm_eps(block, name) for name in original.EPS))
                for role in ('frame', 'global') for index, block in enumerate(getattr(agg, role + '_blocks'))))

    @staticmethod
    def _norm_eps(block, name):
        for part in name.split('.'):
            block = getattr(block, part)
        return float(block.eps)

    def _backend_state(self):
        return (id(self.base), id(self.api), id(self.neon), self.timeout, self.core_mask,
            tuple(context.value for context in self.contexts),
            tuple(id(getattr(self.api, name)) for name in ('rknn_inputs_set', 'rknn_run', 'rknn_outputs_get', 'rknn_outputs_release', 'rknn_destroy')),
            id(self.neon.f32_to_f16), id(self.neon.f16_to_f32),
            tuple((ordinal, tuple(indices), id(ctx), ctx.value,
                tuple((b['index'], b['name'], tuple(b['shape']), b['dtype']) for b in public))
                for ordinal, indices, ctx, public in self.groups),
            tuple((role, value.shape, value.dtype, value.flags.writeable,
                hashlib.sha256(value.tobytes()).hexdigest()) for role, value in self.positions.items()))

    def _check_methods(self):
        agg = self.aggregator
        if (self.closed or self.failed or self.torch.is_grad_enabled() or self.model.training or agg.training
                or getattr(agg, '_vggt_rknn_blocks_owner', None) is not self
                or agg.patch_embed is not self.original_patch_embed or agg.position_getter is not self.original_position_getter
                or self.aggregator_globals.get('viz_attention', False) is not False
                or self.aggregator_globals.get('slice_expand_and_flatten') is not self.slice_expand):
            raise RuntimeError('Original retained aggregator method/owner/mode changed; no fallback')
        cache = getattr(agg, '_vggt_patch_cache_owner', None)
        if cache is None:
            if agg.forward is not self.forward_callable:
                raise RuntimeError('Retained aggregator forward changed')
        elif (getattr(cache, '_original_aggregator', None) is not self.forward_callable
                or agg.forward is not getattr(cache, '_aggregate_callable', None)):
            raise RuntimeError('Patch-cache ownership of retained aggregator changed')

    def _check_lifecycle(self):
        tick = time.perf_counter()
        try:
            self._check_methods()
            self._reject_bypassed_hooks()
            if (self._module_state() != self.module_state or self._tensor_state() != self.tensor_state
                    or self._config_state() != self.config_state or self._backend_state() != self.backend_state):
                raise RuntimeError('Original modules/parameters/buffers/config/backend changed; reinstall required')
        finally:
            self.stats['guard_seconds'] += time.perf_counter() - tick

    def _images(self, images, similarity):
        if (similarity is not False or not isinstance(images, self.torch.Tensor)
                or images.device.type != 'cpu' or images.dtype != self.torch.float32 or images.requires_grad
                or images.layout != self.torch.strided or tuple(images.shape) != (1, 2, 3, 84, 112)
                or not self.torch.isfinite(images).all().item()):
            raise RuntimeError('Only normal original CPU FP32 fixed two-frame inference is supported')

    def _guard(self, module, args, kwargs):
        with self.lock:
            try:
                self._check_methods()
                self._images(args[0] if args else kwargs.get('images'),
                    kwargs.get('compute_similarity', args[1] if len(args) > 1 else False))
            except BaseException:
                self.failed = True
                raise

    def _forward(self, images, compute_similarity=False):
        with self.lock:
            tick = time.perf_counter()
            owns = False
            try:
                self._check_lifecycle()
                self._images(images, compute_similarity)
                if self.active:
                    raise RuntimeError('Recursive retained aggregator execution is unsupported')
                self.active, owns = True, True
                agg, torch, np = self.aggregator, self.torch, self.np
                B, S, C_in, H, W = images.shape
                # Exact original prefix, including normalized inputs observed by patch_cache.
                images = (images - agg._resnet_mean) / agg._resnet_std
                images = images.view(B * S, C_in, H, W)
                patch_tokens = agg.patch_embed(images)
                if isinstance(patch_tokens, dict):
                    patch_tokens = patch_tokens['x_norm_patchtokens']
                camera_token = self.slice_expand(agg.camera_token, B, S)
                register_token = self.slice_expand(agg.register_token, B, S)
                tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
                if (tokens.device.type != 'cpu' or tokens.dtype != torch.float32 or tokens.requires_grad
                        or tuple(tokens.shape) != FRAME or not torch.isfinite(tokens).all().item()):
                    raise ValueError('Original prepared special/patch tokens differ')
                pos = agg.position_getter(B * S, H // agg.patch_size, W // agg.patch_size, device=images.device)
                pos = pos + 1
                pos_special = torch.zeros(B * S, agg.patch_start_idx, 2).to(images.device).to(pos.dtype)
                pos = torch.cat([pos_special, pos], dim=1)
                if (pos.device.type != 'cpu' or pos.dtype != torch.int64 or tuple(pos.shape) != (2, 53, 2)
                        or not np.array_equal(pos.detach().numpy(), self.positions['frame'])
                        or not np.array_equal(pos.view(1, 106, 2).detach().numpy(), self.positions['global'])):
                    raise ValueError('Original actual position getter output differs from all frozen graph positions')
                array, output_list, target_tokens = tokens.detach().contiguous().numpy(), [None] * 24, None
                for ordinal, indices, context, public in self.groups:
                    values = self._run_group(context, array, public)
                    frame_tokens, global_tokens = torch.from_numpy(values[0]), torch.from_numpy(values[1])
                    output_list[indices[-1]] = torch.cat([frame_tokens.view(1, 2, 53, 1024),
                        global_tokens.view(1, 2, 53, 1024)], dim=-1)
                    if ordinal == 3:
                        # _process_global_attention returns B*S,P,C before original capture.
                        print(agg.target_layer)
                        target_tokens = torch.from_numpy(values[2]).view(2, 53, 1024).clone()
                        target_tokens = target_tokens / (target_tokens.norm(dim=-1, keepdim=True) + 1e-8)
                    array = values[1].reshape(FRAME)
                    if not array.flags.c_contiguous or array.ctypes.data != values[1].ctypes.data:
                        raise ValueError('Actual original global-to-next-frame view differs')
                    self.stats['completed_groups'] += 1
                    self.stats['completed_pairs'] += len(indices)
                    self.stats['completed_blocks'] += 2 * len(indices)
                    key = f'group{ordinal}'
                    self.stats['group_calls'][key] = self.stats['group_calls'].get(key, 0) + 1
                if target_tokens is None or not torch.isfinite(target_tokens).all().item():
                    raise ValueError('Original finite target_tokens return required')
                self._check_lifecycle()
                self.stats['completed_aggregator_calls'] += 1
                return output_list, agg.patch_start_idx, target_tokens, None
            except BaseException:
                self.failed = True
                raise
            finally:
                if owns:
                    self.active = False
                self.stats['aggregator_forward_seconds'] += time.perf_counter() - tick

    def _run_group(self, context, array, public):
        base, api, np, timing = self.base, self.api, self.np, self.stats['half_io']
        timing['calls_attempted'] += 1
        started = time.perf_counter()
        def call(name, fn, *args):
            tick = time.perf_counter()
            try:
                return fn(*args)
            finally:
                elapsed = time.perf_counter() - tick
                timing[name + '_seconds'] += elapsed
                timing['sdk_api_seconds'] += elapsed
        try:
            if array.shape != FRAME or array.dtype != np.float32 or not array.flags.c_contiguous or not np.isfinite(array).all():
                raise ValueError('Dense finite original frame input required')
            feed = np.empty(FRAME, dtype=np.float16)
            tick = time.perf_counter()
            self.neon.f32_to_f16(array.ctypes.data, feed.ctypes.data, array.size)
            timing['input_conversion_calls'] += 1
            timing['input_conversion_seconds'] += time.perf_counter() - tick
            if not np.isfinite(feed).all():
                raise ValueError('Original FP16 conversion overflow')
            inputs = (base.Input * 1)(base.Input(0, feed.ctypes.data, feed.nbytes, 0, 1, 3))
            count, got, halves = len(public), False, []
            outputs = (base.Output * count)()
            for i in range(count):
                outputs[i].index, outputs[i].want_float = i, 0
            run, received = base.RunExtend(0, 0, self.timeout, -1), base.OutputExtend()
            begin = time.perf_counter()
            try:
                base.check_call(call('rknn_inputs_set', api.rknn_inputs_set, context, 1, inputs), 'rknn_inputs_set')
                self.stats['sdk_run_attempts'] += 1
                code = call('rknn_run', api.rknn_run, context, ct.byref(run))
                self.stats['sdk_runs_succeeded'] += int(code == 0)
                base.check_call(code, 'rknn_run')
                code = call('rknn_outputs_get', api.rknn_outputs_get, context, count, outputs, ct.byref(received))
                self.stats['outputs_acquired'] += sum(bool(o.buf) for o in outputs)
                base.check_call(code, 'rknn_outputs_get')
                got = True
                if run.frame_id != received.frame_id:
                    raise ValueError('Synchronous original group frame ID differs')
                tick = time.perf_counter()
                try:
                    for output, binding in zip(outputs, public):
                        shape, size = tuple(binding['shape']), math.prod(binding['shape'])
                        if not output.buf or output.size != size * 2:
                            raise ValueError('Original half output pointer/size differs')
                        view = np.ctypeslib.as_array(ct.cast(output.buf, ct.POINTER(ct.c_uint16)), shape=(size,)).view(np.float16)
                        value = np.empty(shape, dtype=np.float16)
                        np.copyto(value.reshape(-1), view)
                        halves.append(value)
                finally:
                    timing['owned_half_copy_seconds'] += time.perf_counter() - tick
            finally:
                try:
                    if got or any(o.buf for o in outputs):
                        acquired = sum(bool(o.buf) for o in outputs)
                        code = call('rknn_outputs_release', api.rknn_outputs_release, context, count, outputs)
                        if code == 0:
                            self.stats['outputs_released'] += acquired
                        else:
                            self.stats['cleanup_errors'].append('rknn_outputs_release returned ' + str(code))
                        base.check_call(code, 'rknn_outputs_release')
                finally:
                    self.stats['sdk_io_seconds'] += time.perf_counter() - begin
            owned = []
            for half in halves:
                value = np.empty(half.shape, dtype=np.float32)
                tick = time.perf_counter()
                self.neon.f16_to_f32(half.ctypes.data, value.ctypes.data, half.size)
                timing['output_conversion_calls'] += 1
                timing['output_conversion_seconds'] += time.perf_counter() - tick
                if not value.flags.owndata or not np.isfinite(value).all():
                    raise ValueError('Finite owned original group FP32 output required')
                owned.append(value)
            timing['calls_completed'] += 1
            return owned
        finally:
            timing['route_seconds'] += time.perf_counter() - started

    def close(self):
        with self.lock:
            if self.active:
                raise RuntimeError('Cannot close retained aggregator during its active forward')
            if self.installed and getattr(self.aggregator, '_vggt_patch_cache_owner', None) is not None:
                raise RuntimeError('Close original DINO patch_cache before retained aggregator')
            if self.closed:
                if self.stats['cleanup_errors']:
                    raise RuntimeError('Retained aggregator cleanup failed: ' + '; '.join(self.stats['cleanup_errors']))
                return
            self.closed = True
            if self.guard is not None:
                try:
                    self.guard.remove()
                except BaseException as exc:
                    self.stats['cleanup_errors'].append('Guard removal: ' + str(exc))
            if self.installed:
                if self.aggregator.forward is self.forward_callable:
                    if self.original_had_forward:
                        self.aggregator.forward = self.original_forward_value
                    else:
                        del self.aggregator.forward
                else:
                    self.stats['cleanup_errors'].append('Aggregator forward changed after installation; not overwritten')
                if getattr(self.aggregator, '_vggt_rknn_blocks_owner', None) is self:
                    del self.aggregator._vggt_rknn_blocks_owner
            for context in reversed(self.contexts):
                if context.value and self.api is not None:
                    try:
                        self.base.check_call(destroy_shared_context(self.api, context), 'rknn_destroy')
                        context.value = 0
                        self.stats['contexts_destroyed'] += 1
                    except BaseException as exc:
                        self.stats['cleanup_errors'].append('rknn_destroy: ' + str(exc))
            if self.stats['cleanup_errors'] or any(c.value for c in self.contexts):
                raise RuntimeError('Retained aggregator cleanup failed: ' + '; '.join(self.stats['cleanup_errors']))

    def metrics(self):
        with self.lock:
            return copy.deepcopy(dict(self.stats, closed=self.closed, failed=self.failed,
                live_contexts=sum(bool(c.value) for c in self.contexts), original_forward_restored=
                    bool(self.closed and self.aggregator is not None and
                         method_identity(self.aggregator.forward) == method_identity(self.original_forward))))


def install_retained_aggregator(model, manifest_path, manifest_sha256, **options):
    return RKNNRetainedOriginalAggregator(model, manifest_path, manifest_sha256, **options)
