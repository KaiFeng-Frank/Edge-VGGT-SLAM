"""Private calibrated RKNN diagnostic of VGGT's complete original DINO module.

Token preparation, the original block loop, and final normalization stay in
Torch FP32. Each original block graph takes one [1,53,1024] frame; a cold
two-frame batch runs that context twice in order. Install before the tail patch
cache; close that cache before this handle. No model or numerical acceptance is
implied by selecting this execution backend.
All original guard/forward/cache/cleanup boundaries are retained. Only manifest
provenance, artifact roots and local floating/affine I/O validation differ.
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

from vggt_slam.rknn_aggregator_blocks import (
    ABI_SHA, BASE, CHECKPOINT_SHA, HEADER_SHA, LIBRARY_SHA,
    RKNNOriginalAggregatorBlocks, artifact, pin,
)

SHAPE = (1, 53, 1024)
SCHEMA = 'v57_original_VGGT_calibrated_DINO_RKNN_manifest'
PARENT_SCHEMA = 'v54_original_VGGT_24_DINO_block_RKNN_manifest'
PARENT_SHA = 'd68d05f984688e2c793f2e9d8320531202a4f9871790d7c2d96c74b5f41c9325'
CAPTURE_SHA = 'a1e6f1b2a89e1a96987befdea4f0a048aaf06066efa9fff72a370ac77d45507c'
CAPTURE_SCRIPT_SHA = '3fc8701a55e693c934bec60fa15e2afb34cfcbeb749e82c4a601dc87bfb5b901'
ORIGINAL_RUNTIME_SHA = '6cf230ea3ae6c65b9a3c7ec0b31e9abb346edb2fc0413b6b77a3a3b510b495de'
PAIRS = [(start, start + 48) for start in range(0, 673, 96)]
STATE_KEYS = frozenset((
    'norm1.weight', 'norm1.bias', 'attn.qkv.weight', 'attn.qkv.bias',
    'attn.proj.weight', 'attn.proj.bias', 'ls1.gamma', 'norm2.weight',
    'norm2.bias', 'mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight',
    'mlp.fc2.bias', 'ls2.gamma',
))


def calibrated_attr(base, attr, shape, name):
    row = base.attr_record(attr)
    identity = row['type'] == 1 and row['quantization_type'] == 2 and row['zero_point'] == 0 and row['scale'] == 1.
    integer = row['type'] in (2, 3)
    count = math.prod(shape)
    if (row['index'] != 0 or row['name'] != name or row['shape'] != list(shape)
            or row['n_elems'] != count or row['type'] not in (0, 1, 2, 3)
            or row['format'] != 3 or row['pass_through'] != 0 or not math.isfinite(row['scale'])
            or row['size'] != count * {0: 4, 1: 2, 2: 1, 3: 1}.get(row['type'], 0)
            or (integer and (row['quantization_type'] != 2 or row['scale'] <= 0.
                or not ((-128 <= row['zero_point'] <= 127) if row['type'] == 2
                        else (0 <= row['zero_point'] <= 255))))
            or (not integer and row['quantization_type'] != 0 and not identity)):
        raise ValueError(f'Expected fixed floating or finite affine INT8/UINT8 DINO I/O: {row}')
    return row


def calibration_identity(manifest, source_root):
    if (manifest.get('schema') != SCHEMA or type(manifest.get('compiled_count')) is not int
            or manifest['compiled_count'] != 24 or len(manifest.get('blocks', [])) != 24
            or manifest.get('parent_manifest_sha256') != PARENT_SHA
            or manifest.get('capture_sha256') != CAPTURE_SHA
            or manifest.get('status') != 'compiled_not_quality_accepted'
            or manifest.get('candidate_accepted') is not False):
        raise ValueError('Exactly 24 calibrated original DINO blocks and fixed parent/capture identities are required')
    parent_directory = Path(manifest['parent_manifest_directory']).resolve(strict=True)
    calibration_directory = Path(manifest['calibration_directory']).resolve(strict=True)
    parent = json.loads(pin(parent_directory / 'compiled.json', PARENT_SHA).read_text())
    capture = json.loads(pin(calibration_directory / 'capture.json', CAPTURE_SHA).read_text())
    replaced = {'schema', 'capture_sha256', 'blocks'}
    if (parent['schema'] != PARENT_SCHEMA or not set(parent) <= set(manifest)
            or any(manifest[key] != value for key, value in parent.items() if key not in replaced)
            or [row['index'] for row in manifest['blocks']] != list(range(24))):
        raise ValueError('Original DINO parent metadata must remain unchanged')
    for row, original in zip(manifest['blocks'], parent['blocks']):
        if ({key: value for key, value in row.items() if key != 'rknn'}
                != {key: value for key, value in original.items() if key != 'rknn'}):
            raise ValueError('Original DINO graph/parameter/source metadata differs')
    true_fields = ('original_model_only', 'actual_inference_on_board_only',
        'checkpoint_verified_before_deserialization', 'model_state_unchanged', 'hooks_removed',
        'source_and_images_unchanged', 'checkpoint_stat_unchanged', 'affinity_restored', 'environment_restored')
    if (capture['schema'] != 'v56_original_VGGT_calibration_capture'
            or capture['status'] != 'completed_original_FP32_calibration_capture'
            or capture['script_sha256'] != CAPTURE_SCRIPT_SHA or capture['checkpoint'] != manifest['checkpoint']
            or capture['image_shape'] != [2, 3, 84, 112] or capture['cleanup_errors']
            or capture['full_forward_attempts'] != 8 or capture['full_forward_completed'] != 8
            or any(capture[field] is not True for field in true_fields)
            or capture['NPU_or_cache_installed'] is not False
            or capture['ORT_or_SDK_or_compiler_or_Solver_executed'] is not False
            or capture['candidate_accepted'] is not False
            or capture['full_model_quality_or_realtime_validated'] is not False):
        raise ValueError('Calibration must come from eight completed original CPU FP32 forwards')
    for relative, expected in capture['source_hashes'].items():
        source = (source_root / relative).resolve(strict=True)
        if not source.is_relative_to(source_root):
            raise ValueError('Calibration source path escapes original source root')
        pin(source, expected)
    if any(capture['source_hashes'].get(key) != value for key, value in parent['source_hashes'].items()):
        raise ValueError('Calibration and DINO graph sources differ')
    samples, groups = capture['samples'], capture['groups']['dino']
    if (len(samples) != 8 or [row['indices'] for row in samples] != [list(pair) for pair in PAIRS]
            or any(row['calls'] != {'aggregator': list(range(24)), 'dino': list(range(24))}
                   or row['split'] != 'calibration' or row['input_unchanged'] is not True for row in samples)
            or set(index for pair in PAIRS for index in pair) & set(capture['evaluation_indices_excluded'])
            or len(groups) != 24 or [group['index'] for group in groups] != list(range(24))):
        raise ValueError('Eight disjoint original pairs and 24 ordered DINO capture groups are required')
    expected = [(f'calib{i:02d}', list(pair), frame) for i, pair in enumerate(PAIRS) for frame in (0, 1)]
    for group in groups:
        rows, dataset = group['samples'], group['dataset']
        if (group['input_shape'] != list(SHAPE) or len(rows) != 16
                or [(row['sample_id'], row['source_pair_indices'], row['origin_batch_row']) for row in rows] != expected
                or dataset['line_count'] != 16 or dataset['input_count'] != 1
                or any(row['origin_batch_shape'] != [2, 53, 1024] or row['artifact']['shape'] != list(SHAPE)
                    or row['artifact']['dtype'] != 'float32' or row['artifact']['finite'] is not True for row in rows)):
            raise ValueError('Each DINO layer requires 16 ordered real FP32 rows from original B2 inputs')
        dataset_path = artifact(calibration_directory, dataset)
        paths = [(calibration_directory / row['artifact']['file']).resolve(strict=True) for row in rows]
        if (any(not path.is_relative_to(calibration_directory) or not path.is_file() for path in paths)
                or dataset_path.read_text().splitlines() != [str(path) for path in paths]):
            raise ValueError('DINO calibration dataset differs from fixed capture metadata')
    return parent_directory, capture


class RKNNCalibratedOriginalDinoBlocks(RKNNOriginalAggregatorBlocks):
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
            pin(source_root / 'vggt_slam/rknn_dino_blocks.py', ORIGINAL_RUNTIME_SHA)
            abi_path = pin(source_root / 'deploy/rk3588/validation/run-v12-rknn-block-board.py', ABI_SHA)
            spec = importlib.util.spec_from_file_location('_v57_calibrated_original_dino_ABI', abi_path)
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
            parent_directory, capture = calibration_identity(manifest, source_root)
            if (manifest['schema'] != SCHEMA
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
                artifact(parent_directory, row)
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
                token_preparation_preserved=True, final_norm_preserved=True,
                precision='NPU_W8A8_calibrated_original_DINO_FP32_API_boundaries',
                parent_manifest_sha256=PARENT_SHA, capture_sha256=CAPTURE_SHA,
                capture_script_sha256=CAPTURE_SCRIPT_SHA, calibration_pairs=8,
                calibration_inputs_per_dino_layer=16, calibration_source_hashes=dict(capture['source_hashes']),
                activation_arrays_rehashed_at_runtime=False, numerical_or_trajectory_quality_accepted=False)
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
                attrs = [calibrated_attr(self.base, self.base.query(self.api, context, cmd, self.base.TensorAttr(index=0)), SHAPE, name)
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


def install_calibrated_dino(model, manifest_path, manifest_sha256, **options):
    return RKNNCalibratedOriginalDinoBlocks(model, manifest_path, manifest_sha256, **options)
