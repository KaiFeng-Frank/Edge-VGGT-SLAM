"""Six complete original four-layer DINO graphs; FP16 diagnostic not accepted.

Replace forward_features' 24-layer loop, never simulate intermediate block
outputs. Original token preparation, final norm, outer DINO hooks and returned
dictionary remain active. Blocks/submodule/global hooks would be bypassed and
are rejected. Install before the tail cache; close that cache before this handle.
"""
from __future__ import annotations
import ctypes as ct
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import threading

from vggt_slam.rknn_half_io_dino import RKNNHalfIOOriginalDino
from vggt_slam.rknn_dino_blocks import STATE_KEYS
from vggt_slam.rknn_aggregator_blocks import BASE, CHECKPOINT_SHA, ABI_SHA, HEADER_SHA, LIBRARY_SHA, pin, artifact
from vggt_slam.rknn_half_io_pairs import NEON_LIBRARY_SHA, NEON_SOURCE_SHA, require_half_descriptor

PARENT_SHA = 'd68d05f984688e2c793f2e9d8320531202a4f9871790d7c2d96c74b5f41c9325'
CAPTURE_SHA = 'ca981ff8d95d2e2c37c94fde2a6f7d1a30c2f5d8568b127eb77f2bff18a6dbe1'
HALF_IO_RUNTIME_SHA = 'e39469432aa5663f0a15756be0067ec92a0fb7127e6702a794298c3e902f3cd7'
HELPER_SOURCE_PINS = {
    'rknn_half_io_dino.py': HALF_IO_RUNTIME_SHA,
    'rknn_dino_blocks.py': '6cf230ea3ae6c65b9a3c7ec0b31e9abb346edb2fc0413b6b77a3a3b510b495de',
    'rknn_half_io_pairs.py': '53b573f494f2598183659ba4e8dcb7503c770d9e2ac7b95cae9016279805d9a4',
    'rknn_aggregator_pairs.py': '08270da911213d0d22c21c9a7f001aaa7e16d7f7b32220f8a2b0c1ddc4b09b88',
    'rknn_aggregator_blocks.py': '4fea665334e9d12a3d55839c3150680e269bab9794074d698bed2ad0a6e41c93',
}
SHAPE = (1, 53, 1024)
HOOK_FIELDS = ('_forward_pre_hooks', '_forward_hooks', '_backward_pre_hooks', '_backward_hooks')
CONFIG_FIELDS = ('chunked_blocks', 'n_blocks', 'embed_dim', 'num_heads', 'patch_size',
                 'num_register_tokens', 'interpolate_antialias', 'interpolate_offset', 'use_reentrant')


def method_identity(value):
    return id(getattr(value, '__func__', value)), id(getattr(value, '__self__', None))


def half_io_stats():
    result = {name + '_seconds': 0. for name in ('input_conversion', 'output_conversion',
        'owned_half_copy', 'sdk_api', 'route', 'rknn_inputs_set', 'rknn_run',
        'rknn_outputs_get', 'rknn_outputs_release')}
    result.update(calls_attempted=0, calls_completed=0, input_conversion_calls=0, output_conversion_calls=0)
    return result


class RKNNGroupedOriginalDino(RKNNHalfIOOriginalDino):
    def __init__(self, model, manifest_path, manifest_sha256, *, half_io_library,
                 half_io_library_sha256, source_root=BASE / 'src', library_path='/usr/lib/librknnrt.so',
                 core_mask=7, timeout_ms=30000,
                 parent_manifest=BASE / 'experiments/vggt-v54-dino-graphs-r1/compiled.json'):
        self.lock = threading.RLock()
        self.contexts, self.groups = [], []
        self.closed = self.failed = False
        self.api = self.base = self.dino = self.aggregator = None
        self.guard = self.aggregate_guard = self.feature_forward = None
        self.original_feature = self.original_feature_value = None
        self.original_feature_had = False
        self.stats = dict(context_init_attempts=0, contexts_created=0, contexts_destroyed=0,
            sdk_run_attempts=0, sdk_runs_succeeded=0, sdk_io_seconds=0., cleanup_errors=[],
            model_memory=[], completed_feature_calls=0, completed_groups=0, completed_frame_groups=0,
            executed_original_layer_frames=0, group_calls={}, group_frame_calls={},
            batch1_feature_calls=0, batch2_feature_calls=0, original_torch_block_forward_calls=0,
            candidate_accepted=False, numerical_or_trajectory_quality_accepted=False,
            full_model_quality_validated=False, real_time_validated=False,
            all_operators_on_npu_verified=False, half_io=half_io_stats())
        try:
            if (half_io_library_sha256 != NEON_LIBRARY_SHA or type(core_mask) is not int or core_mask not in (1, 7)
                    or type(timeout_ms) is not int or not 0 < timeout_ms <= 600000):
                raise ValueError('Frozen NEON library, core_mask 1/7 and timeout 1..600000 required')
            source_root = Path(source_root).resolve(strict=True)
            abi_path = pin(source_root / 'deploy/rk3588/validation/run-v12-rknn-block-board.py', ABI_SHA)
            spec = importlib.util.spec_from_file_location('_v58_grouped_DINO_ABI', abi_path)
            self.base = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.base)
            self.stats['board'] = self.base.require_board()
            pin(source_root / 'deploy/rk3588/reference/rknn_api_v2.3.2.h', HEADER_SHA)
            for name, expected in HELPER_SOURCE_PINS.items():
                module = importlib.import_module('vggt_slam.' + name[:-3])
                path = pin(source_root / 'vggt_slam' / name, expected)
                if Path(module.__file__).resolve() != path:
                    raise ValueError('Frozen original helper import escaped selected source root')
            import numpy as np
            import torch
            self.np, self.torch, self.timeout = np, torch, timeout_ms
            parent_path = pin(parent_manifest, PARENT_SHA); parent = json.loads(parent_path.read_text())
            manifest_path = pin(manifest_path, manifest_sha256); manifest = json.loads(manifest_path.read_text())
            directory = manifest_path.parent
            checkpoint = dict(sha256=CHECKPOINT_SHA, bytes=5026874952)
            if (parent['schema'] != 'v54_original_VGGT_24_DINO_block_RKNN_manifest' or parent['checkpoint'] != checkpoint
                    or len(parent['blocks']) != 24 or [row['index'] for row in parent['blocks']] != list(range(24))
                    or parent['capture_sha256'] != CAPTURE_SHA
                    or manifest['schema'] != 'v58_original_VGGT_6_DINO_group_RKNN_manifest'
                    or manifest['original_parent_manifest_sha256'] != PARENT_SHA or manifest['checkpoint'] != checkpoint
                    or manifest['capture_sha256'] != CAPTURE_SHA or manifest['source_hashes'] != parent['source_hashes']
                    or manifest['original_non_block_parameter_raw_sha256'] != parent['original_non_block_parameter_raw_sha256']
                    or manifest['image_shape'] != [1, 3, 84, 112] or manifest['depth'] != 24
                    or manifest['embed_dim'] != 1024 or manifest['heads'] != 16 or len(manifest['groups']) != 6
                    or manifest['status'] != 'compiled_not_quality_accepted' or manifest['cleanup_errors']
                    or manifest['all_separate_ONNX_checks_byte_equal'] is not True):
                raise ValueError('Complete original 24-layer six-group parent/manifest contract differs')
            for relative, expected in manifest['source_hashes'].items():
                path = (source_root / relative).resolve(strict=True)
                if not path.is_relative_to(source_root): raise ValueError('Original source path escapes source root')
                pin(path, expected)
            dino = self.dino = model.aggregator.patch_embed
            self.aggregator = model.aggregator
            if (any(module.training for module in model.modules())
                    or getattr(dino, '_vggt_rknn_dino_owner', None) is not None
                    or getattr(dino, '_vggt_patch_cache_owner', None) is not None
                    or getattr(self.aggregator, '_vggt_patch_cache_owner', None) is not None
                    or dino.chunked_blocks is not False or len(dino.blocks) != 24 or dino.n_blocks != 24
                    or dino.embed_dim != 1024 or dino.num_heads != 16 or dino.patch_size != 14
                    or dino.num_register_tokens != 4 or model.inference_image_size != 112):
                raise ValueError('Install once on complete original eval CPU DINO before its tail cache')
            self._reject_bypassed_hooks()
            prepared = []
            for ordinal, row in enumerate(manifest['groups']):
                indices = list(range(4 * ordinal, 4 * ordinal + 4))
                if (row['role'] != 'dino_group' or row['index'] != ordinal or row['start_index'] != 4 * ordinal
                        or row['indices'] != indices or row['parents'] != parent['blocks'][4 * ordinal:4 * ordinal + 4]
                        or row['input_shape'] != list(SHAPE) or row['output_shape'] != list(SHAPE)
                        or row['input_name'] != 'input' or row['output_name'] != 'output' or row['heads'] != 16
                        or row['boundary_mode'] != 'fp32'
                        or row['cleanup_errors']
                        or row['all_original_parent_nodes_attributes_and_initializer_payloads_retained'] is not True
                        or row['original_FP32_parent_graph_math_preserved'] is not True
                        or row['parameter_raw_sha256'] != {str(i): parent['blocks'][i]['parameter_raw_sha256'] for i in indices}):
                    raise ValueError('Complete original contiguous four-block graph lineage differs')
                artifact(directory, row)
                rknn = artifact(directory, row['rknn'])
                for index in indices:
                    original = parent['blocks'][index]; block = dino.blocks[index]
                    state = dict(block.named_parameters())
                    if set(state) != STATE_KEYS or set(original['parameter_raw_sha256']) != STATE_KEYS:
                        raise ValueError('All 14 original parameters per DINO block required')
                    for name, value in state.items(): self._parameter(value, original['parameter_raw_sha256'][name], f'dino{index}.{name}')
                    if (original['epsilons'] != {'norm1': float(block.norm1.eps), 'norm2': float(block.norm2.eps)}
                            or block.attn.num_heads != 16 or block.attn.rope is not None
                            or not isinstance(block.attn.q_norm, torch.nn.Identity) or not isinstance(block.attn.k_norm, torch.nn.Identity)
                            or original['gelu_approximate'] != 'none' or getattr(block.mlp.act, 'approximate', None) != 'none'
                            or original['all_14_caller_parameter_values_preserved_in_graph'] is not True):
                        raise ValueError('Original DINO attention/norm/exact GELU contract differs')
                prepared.append((ordinal, indices, rknn))
            non_block = {name: value for name, value in dino.named_parameters() if not name.startswith('blocks.')}
            if set(non_block) != set(parent['original_non_block_parameter_raw_sha256']):
                raise ValueError('Original DINO token/patch/final-norm parameters differ')
            for name, value in non_block.items(): self._parameter(value, parent['original_non_block_parameter_raw_sha256'][name], name)
            self.original_feature_had = 'forward_features' in dino.__dict__
            self.original_feature_value = dino.__dict__.get('forward_features')
            self.original_feature = dino.forward_features
            self.original_prepare = dino.prepare_tokens_with_masks
            self.original_norm = dino.norm
            self.module_state = self._module_state()
            self.tensor_state = self._tensor_state()
            self.config_state = tuple((name, getattr(dino, name, None)) for name in CONFIG_FIELDS)
            self.api = self.base.bind_api(ct.CDLL(str(pin(library_path, LIBRARY_SHA))))
            neon_path = pin(half_io_library, NEON_LIBRARY_SHA)
            pin(source_root / 'vggt_slam/native/fp16_neon.c', NEON_SOURCE_SHA)
            self.neon = ct.CDLL(str(neon_path))
            for name in ('f32_to_f16', 'f16_to_f32'):
                function = getattr(self.neon, name); function.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t]; function.restype = None
            for ordinal, indices, path in prepared:
                context = self.base.CONTEXT(); self.contexts.append(context)
                self.stats['context_init_attempts'] += 1
                filename = ct.create_string_buffer(str(path).encode())
                code = self.api.rknn_init(ct.byref(context), ct.cast(filename, ct.c_void_p), 0, 0, None)
                self.stats['contexts_created'] += int(bool(context.value)); self.base.check_call(code, 'rknn_init')
                if not context.value: raise RuntimeError('Empty original DINO group context')
                self.base.check_call(self.api.rknn_set_core_mask(context, core_mask), 'rknn_set_core_mask')
                version = self.base.query(self.api, context, 5, self.base.SDKVersion())
                if not re.match(r'^2\.3\.2(?:\D|$)', bytes(version.api_version).decode()): raise ValueError('Frozen runtime 2.3.2 required')
                counts = self.base.query(self.api, context, 0, self.base.InputOutputNum())
                if (counts.n_input, counts.n_output) != (1, 1): raise ValueError('Exactly one group input/output required')
                attrs = [self.base.attr_record(self.base.query(self.api, context, command, self.base.TensorAttr(index=0))) for command in (1, 2)]
                for attr, name in zip(attrs, ('input', 'output')): require_half_descriptor(attr, SHAPE, name, 0)
                memory = self.base.query(self.api, context, 6, self.base.MemSize())
                self.stats['model_memory'].append(dict(group=ordinal, indices=indices,
                    sdk_dma_bytes=int(memory.total_dma_allocated_size), input=attrs[0], output=attrs[1]))
                self.groups.append((ordinal, context))
            self.stats.update(manifest_sha256=manifest_sha256, original_parent_manifest_sha256=PARENT_SHA,
                runtime_source_sha256=HALF_IO_RUNTIME_SHA, verified_original_parameters=336,
                verified_original_non_block_parameters=len(non_block), preserved_dino_blocks=24, group_size=4,
                group_count=6, token_preparation_preserved=True, final_norm_preserved=True,
                original_block_module_hooks_bypassed=False, internal_precision_changed=False,
                compiled_boundary_rounding_equivalence_validated=False,
                precision='original_FP16_grouped_DINO_models_explicit_FP16_IO_CPU_FP32_group_boundaries',
                core_mask=core_mask, timeout_ms=timeout_ms, init_flags=0, half_io_experiment=True,
                half_io_library_sha256=NEON_LIBRARY_SHA, half_io_source_sha256=NEON_SOURCE_SHA,
                half_io_library=str(neon_path), input_api_type=1, input_pass_through=0, output_want_float=0)
            self.feature_forward = self._features
            dino.forward_features = self.feature_forward
            dino._vggt_rknn_dino_owner = self
            self.guard = dino.register_forward_pre_hook(self._dino_guard, with_kwargs=True)
            # This still runs when the tail cache serves DINO without invoking it.
            self.aggregate_guard = self.aggregator.register_forward_pre_hook(self._aggregate_guard, with_kwargs=True)
        except BaseException:
            self.failed = True
            self.close()
            raise

    def _parameter(self, value, expected, name):
        if value.device.type != 'cpu' or value.dtype != self.torch.float32 or not value.is_contiguous():
            raise ValueError(f'Original contiguous CPU FP32 parameter required: {name}')
        array = value.detach().numpy()
        if not self.np.isfinite(array).all() or hashlib.sha256(memoryview(array).cast('B')).hexdigest() != expected:
            raise ValueError(f'Original parameter raw identity differs: {name}')

    def _reject_bypassed_hooks(self):
        self._reject_global_hooks()
        if any(getattr(module, name, {}) for block in self.dino.blocks for module in block.modules() for name in HOOK_FIELDS):
            raise RuntimeError('Original block/submodule hooks cannot be bypassed by grouped execution')

    def _reject_global_hooks(self):
        global_module = self.torch.nn.modules.module
        if any(getattr(global_module, '_global' + name, {}) for name in HOOK_FIELDS):
            raise RuntimeError('Global module hooks would observe skipped block-module calls; unsupported')

    def _module_state(self):
        return tuple((name, id(module), type(module), module.training, method_identity(module.forward))
                     for name, module in self.dino.named_modules() if name)

    def _tensor_state(self):
        return tuple((name, id(value), value._version, value.dtype, value.device, tuple(value.shape),
                      tuple(value.stride()), value.data_ptr(), value.requires_grad)
                     for name, value in list(self.dino.named_parameters()) + list(self.dino.named_buffers()))

    def _check_methods(self):
        if (self.closed or self.failed or self.torch.is_grad_enabled() or self.dino.training
                or self.dino.forward_features is not self.feature_forward
                or getattr(self.dino, '_vggt_rknn_dino_owner', None) is not self
                or method_identity(self.dino.prepare_tokens_with_masks) != method_identity(self.original_prepare)
                or self.dino.norm is not self.original_norm or self.aggregator.patch_embed is not self.dino):
            raise RuntimeError('Original grouped DINO method/owner changed; no fallback')
        self._reject_global_hooks()

    def _check_lifecycle(self):
        self._check_methods()
        if (self._module_state() != self.module_state or self._tensor_state() != self.tensor_state
                or tuple((name, getattr(self.dino, name, None)) for name in CONFIG_FIELDS) != self.config_state):
            raise RuntimeError('Original grouped DINO methods/structure/parameters/config changed; no fallback')
        self._reject_bypassed_hooks()

    def _aggregate_guard(self, module, args, kwargs):
        with self.lock:
            try: self._check_methods()
            except BaseException: self.failed = True; raise

    def _dino_guard(self, module, args, kwargs):
        with self.lock:
            try:
                super()._guard(module, args, kwargs)
            except BaseException: self.failed = True; raise

    def _features(self, x, masks=None):
        with self.lock:
            try:
                self._check_lifecycle()
                if (masks is not None or not isinstance(x, self.torch.Tensor) or x.device.type != 'cpu'
                        or x.dtype != self.torch.float32 or x.requires_grad or x.layout != self.torch.strided
                        or tuple(x.shape) not in ((1, 3, 84, 112), (2, 3, 84, 112))):
                    raise RuntimeError('Only original one/two-frame FP32 DINO images without masks are supported')
                tokens = self.original_prepare(x, masks)
                if tokens.device.type != 'cpu' or tokens.dtype != self.torch.float32 or tuple(tokens.shape) != (len(x), *SHAPE[1:]):
                    raise RuntimeError('Original token preparation output contract differs')
                array = tokens.detach().contiguous().numpy()
                for ordinal, context in self.groups:
                    outputs = []
                    for frame in range(len(array)):
                        value = self._run(context, array[frame:frame + 1], 3, SHAPE)
                        if value.dtype != self.np.float32 or value.shape != SHAPE or not value.flags.owndata or not self.np.isfinite(value).all():
                            raise ValueError('Group output must be owned finite original FP32 tokens')
                        outputs.append(value)
                        self.stats['completed_frame_groups'] += 1
                        self.stats['executed_original_layer_frames'] += 4
                        key = f'group{ordinal}'
                        self.stats['group_frame_calls'][key] = self.stats['group_frame_calls'].get(key, 0) + 1
                    array = outputs[0] if len(outputs) == 1 else self.np.concatenate(outputs, axis=0)
                    self.stats['completed_groups'] += 1
                    self.stats['group_calls'][key] = self.stats['group_calls'].get(key, 0) + 1
                tokens = self.torch.from_numpy(array)
                normalized = self.original_norm(tokens)
                result = dict(x_norm_clstoken=normalized[:, 0],
                    x_norm_regtokens=normalized[:, 1:self.dino.num_register_tokens + 1],
                    x_norm_patchtokens=normalized[:, self.dino.num_register_tokens + 1:], x_prenorm=tokens, masks=masks)
                self._check_lifecycle()
                self.stats['completed_feature_calls'] += 1
                self.stats[f'batch{len(array)}_feature_calls'] += 1
                return result
            except BaseException:
                self.failed = True
                raise

    def close(self):
        with self.lock:
            if self.closed: return
            if (self.feature_forward is not None and self.dino is not None
                    and getattr(self.dino, '_vggt_patch_cache_owner', None) is not None):
                raise RuntimeError('Close the original DINO tail cache before its grouped backend')
            self.closed = True
            for hook in (self.guard, self.aggregate_guard):
                if hook is not None:
                    try: hook.remove()
                    except BaseException as exc: self.stats['cleanup_errors'].append(f'Hook removal: {exc}')
            if self.dino is not None and self.feature_forward is not None:
                if self.dino.forward_features is self.feature_forward:
                    if self.original_feature_had: self.dino.forward_features = self.original_feature_value
                    else: del self.dino.forward_features
                else: self.stats['cleanup_errors'].append('DINO forward_features changed after installation; not overwritten')
            if self.dino is not None and getattr(self.dino, '_vggt_rknn_dino_owner', None) is self:
                del self.dino._vggt_rknn_dino_owner
            for context in self.contexts:
                if context.value and self.api is not None:
                    try:
                        self.base.check_call(self.api.rknn_destroy(context), 'rknn_destroy')
                        self.stats['contexts_destroyed'] += 1
                    except BaseException as exc: self.stats['cleanup_errors'].append(f'{type(exc).__name__}: {exc}')
                    finally: context.value = 0
            if self.stats['cleanup_errors']:
                raise RuntimeError('Grouped original DINO cleanup failed: ' + '; '.join(self.stats['cleanup_errors']))


def install_grouped_dino(model, manifest_path, manifest_sha256, **options):
    return RKNNGroupedOriginalDino(model, manifest_path, manifest_sha256, **options)
