"""48 original VGGT frame/global RKNN blocks; original Aggregator stays intact.

Install on the verified original eval CPU FP32 model before its DINO tail cache.
Keep weights/configuration fixed until close; startup verifies files and all 864
parameters once. Close the tail cache first, then this handle to restore CPU
forwards. Single-output graphs support normal max_loops=0 inference only:
attention visualization/similarity requests fail explicitly without fallback.
"""
from __future__ import annotations

import copy
import ctypes as ct
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import threading
import time

BASE = Path('/mnt/vggt-data/vggt-slam2')
CHECKPOINT_SHA = 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
ABI_SHA = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
LIBRARY_SHA = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
HEADER_SHA = 'c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a'
SHAPES = {'frame': (2, 53, 1024), 'global': (1, 106, 1024)}
EPS = ('norm1', 'norm2', 'attn.q_norm', 'attn.k_norm')
STATE_KEYS = set(('norm1.weight', 'norm1.bias', 'attn.qkv.weight', 'attn.qkv.bias',
                  'attn.proj.weight', 'attn.proj.bias', 'ls1.gamma', 'norm2.weight',
                  'norm2.bias', 'mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight',
                  'mlp.fc2.bias', 'ls2.gamma', 'attn.q_norm.weight', 'attn.q_norm.bias',
                  'attn.k_norm.weight', 'attn.k_norm.bias'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4 << 20), b''):
            h.update(part)
    return h.hexdigest()


def pin(path, expected):
    path = Path(path).resolve(strict=True)
    if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected) or sha(path) != expected:
        raise ValueError(f'Required original artifact SHA256 mismatch: {path}')
    return path


def artifact(directory, row):
    path = (directory / row['file']).resolve(strict=True)
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError('Artifact path escapes the manifest directory')
    pin(path, row['sha256'])
    if path.stat().st_size != row['bytes'] or not 0 < row['bytes'] <= 256 << 20:
        raise ValueError('Artifact byte size mismatch or exceeds 256 MiB')
    return path


def floating_attr(base, attr, shape, name):
    row = base.attr_record(attr)
    identity = row['type'] == 1 and row['quantization_type'] == 2 and row['zero_point'] == 0 and row['scale'] == 1.
    if (row['index'] != 0 or row['name'] != name or row['shape'] != list(shape)
            or row['n_elems'] != math.prod(shape) or row['type'] not in (0, 1)
            or (row['quantization_type'] != 0 and not identity) or row['format'] != 3
            or row['pass_through'] != 0 or not math.isfinite(row['scale'])
            or row['size'] != math.prod(shape) * (4 if row['type'] == 0 else 2)):
        raise ValueError(f'Expected fixed logical FP32/FP16 input/output: {row}')
    return row


class RKNNOriginalAggregatorBlocks:
    def __init__(self, model, manifest_path, manifest_sha256, *, source_root=BASE / 'src',
                 library_path='/usr/lib/librknnrt.so', core_mask=7, timeout_ms=30000):
        self.lock = threading.RLock()
        self.contexts, self.patches, self.guard = [], [], None
        self.closed = self.failed = False
        self.api = self.base = self.aggregator = None
        self.stats = dict(context_init_attempts=0, contexts_created=0, contexts_destroyed=0, sdk_run_attempts=0,
                          sdk_runs_succeeded=0, completed_blocks=0, sdk_io_seconds=0.,
                          block_calls={}, cleanup_errors=[], model_memory=[], all_operators_on_npu_verified=False,
                          candidate_accepted=False)
        try:
            if type(core_mask) is not int or core_mask not in (1, 7) or type(timeout_ms) is not int or not 0 < timeout_ms <= 600000:
                raise ValueError('core_mask must be 1/7 and timeout_ms positive and at most 600000')
            source_root = Path(source_root).resolve(strict=True)
            abi_path = pin(source_root / 'deploy/rk3588/validation/run-v12-rknn-block-board.py', ABI_SHA)
            spec = importlib.util.spec_from_file_location('_v53_original_rknn_ABI', abi_path)
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
            if (manifest['schema'] != 'v53_original_VGGT_48_block_RKNN_manifest'
                    or manifest['image_shape'] != [2, 3, 84, 112]
                    or manifest['checkpoint'] != {'sha256': CHECKPOINT_SHA, 'bytes': 5026874952}
                    or manifest['depth'] != 24 or manifest['aa_order'] != ['frame', 'global']
                    or manifest['aa_block_size'] != 1 or len(manifest['blocks']) != 48):
                raise ValueError('Only the complete original 48-block 84x112 pair is supported')
            for relative, expected in manifest['source_hashes'].items():
                source = (source_root / relative).resolve(strict=True)
                if not source.is_relative_to(source_root):
                    raise ValueError('Source path escapes original source root')
                pin(source, expected)
            agg = model.aggregator
            self.aggregator = agg
            if (any(m.training for m in model.modules()) or getattr(agg, '_vggt_rknn_blocks_owner', None) is not None
                    or getattr(agg, '_vggt_patch_cache_owner', None) is not None
                    or len(agg.frame_blocks) != 24 or len(agg.global_blocks) != 24
                    or agg.aa_order != ['frame', 'global'] or agg.aa_block_size != 1 or agg.aa_block_num != 24
                    or agg.patch_start_idx != 5 or model.inference_image_size != 112
                    or (agg.retained_layer_indices is not None and not {4, 11, 17, 23} <= set(agg.retained_layer_indices))):
                raise ValueError('Install once on the unchanged original eval aggregator, before its patch cache')
            prepared, identities = [], set()
            for row in manifest['blocks']:
                role, index = row['role'], row['index']
                if role not in SHAPES or type(index) is not int or not 0 <= index < 24 or (role, index) in identities:
                    raise ValueError('Exactly frame/global indices 0..23 are required')
                identities.add((role, index))
                shape = SHAPES[role]
                if (row['input_shape'] != list(shape) or row['output_shape'] != list(shape)
                        or row['input_name'] != 'input' or row['output_name'] != 'output' or row['heads'] != 16):
                    raise ValueError('Original block shapes and public IO names differ')
                artifact(directory, row)  # Original ONNX remains separately bound.
                rknn_path = artifact(directory, row['rknn'])
                pos_path = artifact(directory, row['positions'])
                pos = np.load(pos_path, allow_pickle=False)
                if (pos.dtype != np.int64 or pos.shape != (*shape[:2], 2) or not pos.flags.c_contiguous
                        or row['positions']['shape'] != list(pos.shape) or row['positions']['dtype'] != 'int64'
                        or row['positions']['tensor_bytes'] != pos.nbytes
                        or hashlib.sha256(memoryview(pos).cast('B')).hexdigest() != row['positions']['raw_sha256']):
                    raise ValueError('Fixed original RoPE position identity differs')
                block = getattr(agg, role + '_blocks')[index]
                state = dict(block.named_parameters())
                if set(state) != STATE_KEYS or set(row['parameter_raw_sha256']) != STATE_KEYS:
                    raise ValueError('All 18 original block parameters are required')
                for name, value in state.items():
                    if value.device.type != 'cpu' or value.dtype != torch.float32 or not value.is_contiguous():
                        raise ValueError('Original parameters must remain dense CPU FP32')
                    array = value.detach().numpy()
                    if not np.isfinite(array).all() or hashlib.sha256(memoryview(array).cast('B')).hexdigest() != row['parameter_raw_sha256'][name]:
                        raise ValueError(f'Original {role}{index} parameter mismatch: {name}')
                if (set(row['epsilons']) != set(EPS) or block.attn.num_heads != 16
                        or row['gelu_approximate'] != 'none' or getattr(block.mlp.act, 'approximate', None) != 'none'
                        or row['rope_frequency'] != 100 or row['rope_scaling_factor'] != 1.
                        or block.attn.rope is None or block.attn.rope.base_frequency != 100
                        or block.attn.rope.scaling_factor != 1.
                        or row['all_18_caller_parameter_values_preserved_in_graph'] is not True):
                    raise ValueError('Original normalization/attention/exact GELU contract differs')
                for module in EPS:
                    value = block
                    for field in module.split('.'):
                        value = getattr(value, field)
                    if row['epsilons'][module] != float(value.eps):
                        raise ValueError('Original LayerNorm epsilon changed')
                prepared.append((block, role, index, shape, pos.copy(), rknn_path))
            if len(identities) != 48:
                raise ValueError('All 48 distinct original block identities are required')
            self.api = self.base.bind_api(ct.CDLL(str(pin(library_path, LIBRARY_SHA))))
            self.stats.update(manifest_sha256=manifest_sha256, core_mask=core_mask, timeout_ms=timeout_ms,
                              verified_original_parameters=864, preserved_intermediate_layers=[4, 11, 17, 23])
            for block, role, index, shape, pos, path in prepared:
                context = self.base.CONTEXT()
                self.contexts.append(context)  # Retain even partial failed init for cleanup.
                filename = ct.create_string_buffer(str(path).encode())
                self.stats['context_init_attempts'] += 1
                code = self.api.rknn_init(ct.byref(context), ct.cast(filename, ct.c_void_p), 0, 0, None)
                self.stats['contexts_created'] += int(bool(context.value))
                self.base.check_call(code, 'rknn_init')
                if not context.value:
                    raise RuntimeError('RKNN returned an empty context')
                self.base.check_call(self.api.rknn_set_core_mask(context, core_mask), 'rknn_set_core_mask')
                version = self.base.query(self.api, context, 5, self.base.SDKVersion())
                if not re.match(r'^2\.3\.2(?:\D|$)', bytes(version.api_version).decode()):
                    raise ValueError('Frozen ABI requires RKNN runtime 2.3.2')
                counts = self.base.query(self.api, context, 0, self.base.InputOutputNum())
                if (counts.n_input, counts.n_output) != (1, 1):
                    raise ValueError('Each unchanged original graph requires one input and one output')
                attrs = [floating_attr(self.base, self.base.query(self.api, context, cmd, self.base.TensorAttr(index=0)), shape, name)
                         for cmd, name in ((1, 'input'), (2, 'output'))]
                memory = self.base.query(self.api, context, 6, self.base.MemSize())
                key = f'{role}{index}'
                self.stats['model_memory'].append(dict(block=key, sdk_dma_bytes=int(memory.total_dma_allocated_size),
                                                       input=attrs[0], output=attrs[1]))
                forward = self._forward(block, key, context, shape, pos, attrs[0]['format'])
                self.patches.append((block, 'forward' in block.__dict__, block.__dict__.get('forward'), block.forward, forward))
            # No model forward changes until every file/parameter/context/query passed.
            self.guard = agg.register_forward_pre_hook(self._guard, with_kwargs=True)
            for block, _, _, _, forward in self.patches:
                block.forward = forward
            agg._vggt_rknn_blocks_owner = self
        except BaseException:
            self.failed = True
            self.close()
            raise

    def _guard(self, module, args, kwargs):
        images = args[0] if args else kwargs.get('images')
        similarity = kwargs.get('compute_similarity', args[1] if len(args) > 1 else False)
        if (self.closed or self.failed or module.training or self.torch.is_grad_enabled()
                or similarity is not False or not isinstance(images, self.torch.Tensor)
                or tuple(images.shape) != (1, 2, 3, 84, 112) or images.device.type != 'cpu'
                or images.dtype != self.torch.float32 or images.requires_grad):
            raise RuntimeError('RKNN blocks require normal original CPU FP32 fixed-pair inference; no fallback')

    def _forward(self, block, key, context, shape, expected_pos, fmt):
        def forward(x, pos=None, viz_attention=False):
            with self.lock:
                if (self.closed or self.failed or block.training or self.torch.is_grad_enabled() or viz_attention is not False
                        or not isinstance(x, self.torch.Tensor) or x.device.type != 'cpu' or x.dtype != self.torch.float32
                        or tuple(x.shape) != shape or x.requires_grad or x.layout != self.torch.strided
                        or not isinstance(pos, self.torch.Tensor) or pos.device.type != 'cpu' or pos.dtype != self.torch.int64
                        or tuple(pos.shape) != expected_pos.shape or not self.np.array_equal(pos.detach().numpy(), expected_pos)):
                    raise RuntimeError(f'{key}: unsupported original block input/position/mode; no fallback')
                array = x.detach().contiguous().numpy()
                if not self.np.isfinite(array).all():
                    raise ValueError(f'{key}: nonfinite original input')
                try:
                    output = self._run(context, array, fmt, shape)
                    if not self.np.isfinite(output).all():
                        raise ValueError(f'{key}: nonfinite RKNN output')
                    self.stats['completed_blocks'] += 1
                    self.stats['block_calls'][key] = self.stats['block_calls'].get(key, 0) + 1
                    return self.torch.from_numpy(output)
                except BaseException:
                    self.failed = True
                    raise
        return forward

    def _run(self, context, array, fmt, shape):
        base, api = self.base, self.api
        requests = (base.Input * 1)(base.Input(0, array.ctypes.data, array.nbytes, 0, 0, fmt))
        outputs = (base.Output * 1)()
        outputs[0].want_float = 1
        run, received = base.RunExtend(0, 0, self.timeout, -1), base.OutputExtend()
        got = False
        begin = time.perf_counter()
        try:
            base.check_call(api.rknn_inputs_set(context, 1, requests), 'rknn_inputs_set')
            self.stats['sdk_run_attempts'] += 1
            code = api.rknn_run(context, ct.byref(run))
            self.stats['sdk_runs_succeeded'] += int(code == 0)
            base.check_call(code, 'rknn_run')
            base.check_call(api.rknn_outputs_get(context, 1, outputs, ct.byref(received)), 'rknn_outputs_get')
            got = True
            if run.frame_id != received.frame_id or not outputs[0].buf or outputs[0].size != array.nbytes:
                raise RuntimeError('Synchronous float output frame/pointer/byte-size mismatch')
            view = self.np.ctypeslib.as_array(ct.cast(outputs[0].buf, ct.POINTER(ct.c_float)), shape=(math.prod(shape),))
            return view.copy().reshape(shape)  # Owned before SDK release, retained by Torch.
        finally:
            if got or outputs[0].buf:
                code = api.rknn_outputs_release(context, 1, outputs)
                if code:
                    self.stats['cleanup_errors'].append(f'rknn_outputs_release returned {code}')
                base.check_call(code, 'rknn_outputs_release')
            self.stats['sdk_io_seconds'] += time.perf_counter() - begin

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            if self.guard is not None:
                self.guard.remove()
            for block, had, value, original, forward in self.patches:
                if block.forward is forward:
                    if had:
                        block.forward = value
                    else:
                        del block.forward
                elif block.forward != original:
                    self.stats['cleanup_errors'].append('Block forward changed after installation; not overwritten')
            if self.aggregator is not None and getattr(self.aggregator, '_vggt_rknn_blocks_owner', None) is self:
                del self.aggregator._vggt_rknn_blocks_owner
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
                raise RuntimeError('RKNN block cleanup failed: ' + '; '.join(self.stats['cleanup_errors']))

    def metrics(self):
        with self.lock:
            return copy.deepcopy(dict(self.stats, closed=self.closed, failed=self.failed,
                                      live_contexts=sum(bool(x.value) for x in self.contexts)))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def install_rknn_aggregator_blocks(model, manifest_path, manifest_sha256, **options):
    return RKNNOriginalAggregatorBlocks(model, manifest_path, manifest_sha256, **options)
