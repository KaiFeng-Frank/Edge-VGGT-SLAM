"""Temporary MLP-only Co-Me research wrapper; no production integration.

Adapted from co-me-tokens/CoMe direct.py at commit
5c7cfbc14da37f5dbb991e3a6f9e2c4a5efb8833, by the Co-Me authors.
This file is CC BY-NC 4.0, not the repository's BSD code license.
https://github.com/co-me-tokens/CoMe
https://creativecommons.org/licenses/by-nc/4.0/

Original weights, attention, norm2, LayerScale and full residuals remain in
their original modules. Only MLP inputs are averaged and outputs repeated.
That approximation changes predictions and is not the complete Co-Me method.
All calls are CPU-only; no CUDA/NVTX imports or silent precision fallback.
Use one serialized model owner and an eval model under torch.no_grad().
Parameter hashes are computed at context entry/exit, outside model timing.
last_scene retains only the newest scene's saved CPU tensors; its hashes are
computed explicitly by scene_record(), after the timed model call finishes.
Timed forwards include BF16 feature conversion, predictor execution, plan
checks/construction, and the first-use CPU gather-index cache. They contain
no per-layer tensor hashes, output statistics or device synchronizations.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
import types

import numpy as np

MERGE_HELPER_SHA256 = '7acb792a94a5e4cb55bed25674e1e30ecf8692296e6e7fa4d7e98bb9da26bf5d'


def _load_merge():
    path = Path(__file__).with_name('v22_token_merge.py')
    if hashlib.sha256(path.read_bytes()).hexdigest() != MERGE_HELPER_SHA256:
        raise ValueError('Frozen CPU merge helper SHA256 changed')
    name = '_v22_fullmodel_merge_' + MERGE_HELPER_SHA256[:12]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


def _tensor_record(tensor):
    """Metadata/bytes only. Call outside inference; BF16 remains raw BF16."""
    if tensor.device.type != 'cpu':
        raise ValueError('Only CPU tensor metadata is permitted')
    value = tensor.detach().contiguous()
    raw = value.reshape(-1).view(__import__('torch').uint8).numpy()
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'bytes': raw.nbytes, 'sha256': hashlib.sha256(memoryview(raw)).hexdigest()}


def _weight_manifest(model):
    rows = {name: _tensor_record(param) for name, param in model.named_parameters()}
    encoded = json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()
    return {'parameters': rows, 'parameter_count': len(rows),
            'parameter_bytes': sum(v['bytes'] for v in rows.values()),
            'manifest_sha256': hashlib.sha256(encoded).hexdigest()}


class ComeMLPResearch:
    """Context manager around original VGGT DINO15..23/frame0..23/global0..23.

    predictor is the released BF16 CPU predictor, or a clearly labelled tiny
    synthetic callback with the same signature. A synthetic plan_callback
    can replace confidence->mask for lifecycle tests; that use is reported.
    expected_grid_hw/views/channels defaults bind the first S2 336 experiment.
    The caller owns all image/model loading, timing, output comparison and
    original checkpoint/source identity. No inference is initiated here.
    """

    def __init__(self, model, predictor, *, threshold=.5, expected_grid_hw=(18, 24),
                 expected_views=2, channels=1024, plan_callback=None):
        import torch
        if (not isinstance(model, torch.nn.Module) or model.training or not callable(predictor)
                or not np.isfinite(threshold) or not 0 <= threshold <= 1
                or type(expected_views) is not int or expected_views < 1
                or type(channels) is not int or channels < 1
                or len(expected_grid_hw) != 2
                or any(type(n) is not int or n < 3 or n % 3 for n in expected_grid_hw)):
            raise ValueError('CPU eval model, predictor and a divisible explicit scene shape are required')
        if isinstance(predictor, torch.nn.Module) and predictor.training:
            raise ValueError('The confidence predictor must already be in eval mode')
        self.model, self.predictor = model, predictor
        self.threshold = float(threshold)
        self.grid = tuple(expected_grid_hw)
        self.views, self.channels = expected_views, channels
        self.plan_callback = plan_callback
        self.merge = _load_merge()
        aggregator = getattr(model, 'aggregator', None)
        if (aggregator is None or getattr(aggregator, 'patch_start_idx', None) != 5
                or len(getattr(getattr(aggregator, 'patch_embed', None), 'blocks', [])) != 24
                or len(getattr(aggregator, 'frame_blocks', [])) != 24
                or len(getattr(aggregator, 'global_blocks', [])) != 24):
            raise ValueError('Original 24 DINO/frame/global blocks and five special tokens are required')
        self.aggregator = aggregator
        self._targets = [('dino', i, aggregator.patch_embed.blocks[i].mlp) for i in range(15, 24)]
        self._targets += [('frame', i, block.mlp) for i, block in enumerate(aggregator.frame_blocks)]
        self._targets += [('global', i, block.mlp) for i, block in enumerate(aggregator.global_blocks)]
        if len({id(module) for _, _, module in self._targets}) != 57:
            raise ValueError('Research wrapper requires 57 distinct original MLP modules')
        self._lock = threading.RLock()
        self._hooks, self._saved_forwards = [], []
        self._invocations = {}
        self._active = None
        self._last_scene = None
        self._entered = False
        self._closed = False
        self._stats = {'scenes_started': 0, 'scenes_completed': 0, 'scenes_failed': 0,
                       'predictor_calls': 0, 'plan_builds': 0,
                       'mlp_calls': {'dino': 0, 'frame': 0, 'global': 0},
                       'predictor_seconds': 0., 'plan_seconds': 0., 'merge_seconds': 0.,
                       'original_mlp_seconds': 0., 'restore_seconds': 0.,
                       'full_mlp_rows': 0, 'compact_mlp_rows': 0, 'cleanup_errors': []}
        self.weights_before = self.weights_after = None

    def __enter__(self):
        if self._entered or self._closed:
            raise RuntimeError('Research context cannot be entered twice')
        self.weights_before = _weight_manifest(self.model)
        try:
            self._hooks.append(self.aggregator.register_forward_pre_hook(self._begin_scene))
            self._hooks.append(self.aggregator.patch_embed.blocks[14].register_forward_hook(self._build_scene_plan))
            self._hooks.append(self.aggregator.register_forward_hook(self._end_scene, always_call=True))
            for branch, index, module in self._targets:
                instance_forward = module.__dict__.get('forward', None)
                had_instance_forward = 'forward' in module.__dict__
                original = module.forward
                self._saved_forwards.append((module, had_instance_forward, instance_forward))
                module.forward = types.MethodType(self._wrap_mlp(original, branch, index), module)
            self._entered = True
            return self
        except BaseException:
            self._restore()
            raise

    def _begin_scene(self, module, args):
        import torch
        owner = threading.get_ident()
        with self._lock:
            self._invocations.setdefault(owner, []).append(False)
            if self._active is not None:
                raise RuntimeError('Concurrent or recursive scene inference is unsupported')
            if not args or not isinstance(args[0], torch.Tensor):
                raise ValueError('Aggregator must receive original prepared CPU RGB')
            images = args[0]
            height, width = self.grid
            if (images.device.type != 'cpu' or images.dtype != torch.float32
                    or tuple(images.shape) != (1, self.views, 3, height * 14, width * 14)
                    or torch.is_grad_enabled()):
                raise ValueError('Expected fixed B1 scene, original CPU FP32 pixels, and no_grad execution')
            self._stats['scenes_started'] += 1
            self._active = {'scene_id': self._stats['scenes_started'], 'owner': owner,
                            'input_shape': list(images.shape), 'plan': None,
                            'mlp_calls': {'dino': 0, 'frame': 0, 'global': 0},
                            'called_layers': [], 'predictor_seconds': 0., 'plan_seconds': 0.,
                            'merge_seconds': 0., 'original_mlp_seconds': 0., 'restore_seconds': 0.}
            self._last_scene = None
            self._invocations[owner][-1] = True

    def _build_scene_plan(self, module, args, output):
        import torch
        with self._lock:
            scene = self._active
            if scene is None or scene['owner'] != threading.get_ident() or scene['plan'] is not None:
                raise RuntimeError('DINO14 hook is outside its one fresh scene')
        height, width = self.grid
        if (not isinstance(output, torch.Tensor) or output.device.type != 'cpu'
                or output.dtype != torch.float32
                or tuple(output.shape) != (self.views, 5 + height * width, self.channels)):
            raise ValueError('DINO14 output does not match original captured patch layout')
        features = output[:, 5:].detach().reshape(1, self.views, height, width, self.channels)
        start = time.perf_counter()
        predictor_features = features.to(dtype=torch.bfloat16)
        confidence = self.predictor(predictor_features, BPHWC=(1, self.views, height, width, self.channels))
        predictor_seconds = time.perf_counter() - start
        if (not isinstance(confidence, torch.Tensor) or confidence.device.type != 'cpu'
                or confidence.dtype != torch.bfloat16
                or tuple(confidence.shape) != (self.views, height * width, 1)):
            raise ValueError('Predictor must return original CPU BF16 confidence without precision fallback')
        start = time.perf_counter()
        if self.plan_callback is None:
            mask = self.merge.merge_mask_from_confidence(confidence, grid_hw=self.grid, threshold=self.threshold)
        else:
            mask = self.plan_callback(confidence, grid_hw=self.grid, threshold=self.threshold)
        plan = self.merge.build_plan(mask, grid_hw=self.grid)
        blocks = confidence.reshape(self.views, height // 3, 3, width // 3, 3)
        group_means = blocks.permute(0, 1, 3, 2, 4).reshape(self.views, -1, 9).mean(dim=-1)
        percentile = torch.quantile(group_means.float().reshape(1, -1), self.threshold, dim=-1, keepdim=True)
        plan_seconds = time.perf_counter() - start
        with self._lock:
            scene.update(plan=plan, features=features, predictor_features=predictor_features,
                         confidence=confidence.detach(), group_means=group_means.detach(),
                         confidence_quantile=percentile.detach(), predictor_seconds=predictor_seconds,
                         plan_seconds=plan_seconds, feature_source='original_DINO_blocks_14_output_before_final_norm')
            self._stats['predictor_calls'] += 1
            self._stats['plan_builds'] += 1
            self._stats['predictor_seconds'] += predictor_seconds
            self._stats['plan_seconds'] += plan_seconds

    def _wrap_mlp(self, original, branch, index):
        def forward(module, x, *args, **kwargs):
            with self._lock:
                scene = self._active
                if scene is None or scene['owner'] != threading.get_ident() or scene['plan'] is None:
                    raise RuntimeError('MLP needs the current scene DINO14 plan; no previous-plan fallback')
                plan = scene['plan']
            if x.shape[-1] != self.channels:
                raise ValueError('Original MLP channel count changed')
            start = time.perf_counter()
            compact = self.merge.merge_tokens(x, plan)
            merged_at = time.perf_counter()
            transformed = original(compact, *args, **kwargs)
            mlp_at = time.perf_counter()
            restored = self.merge.restore_tokens(transformed, plan).reshape(x.shape)
            restored_at = time.perf_counter()
            with self._lock:
                if scene is not self._active:
                    raise RuntimeError('Scene changed during original MLP execution')
                scene['mlp_calls'][branch] += 1
                scene['called_layers'].append([branch, index])
                self._stats['mlp_calls'][branch] += 1
                self._stats['full_mlp_rows'] += self.views * plan.tokens_per_view
                self._stats['compact_mlp_rows'] += plan.compact_rows
                for key, duration in (('merge_seconds', merged_at - start),
                                      ('original_mlp_seconds', mlp_at - merged_at),
                                      ('restore_seconds', restored_at - mlp_at)):
                    scene[key] += duration
                    self._stats[key] += duration
            return restored
        return forward

    def _end_scene(self, module, args, output):
        owner = threading.get_ident()
        with self._lock:
            invocations = self._invocations.get(owner, [])
            accepted = invocations.pop() if invocations else False
            if not invocations:
                self._invocations.pop(owner, None)
            if not accepted:
                return
            scene = self._active
            self._active = None
            if output is None:
                scene['status'] = 'failed'
                self._stats['scenes_failed'] += 1
            else:
                expected = {'dino': 9, 'frame': 24, 'global': 24}
                if scene['plan'] is None or scene['mlp_calls'] != expected or len({tuple(v) for v in scene['called_layers']}) != 57:
                    scene['status'] = 'failed'
                    self._last_scene = scene
                    self._stats['scenes_failed'] += 1
                    raise RuntimeError('Expected exactly 57 original MLP branches for a completed scene')
                scene['status'] = 'completed'
                self._stats['scenes_completed'] += 1
            self._last_scene = scene

    @property
    def last_scene(self):
        """Newest scene tensors for saving after inference; no implicit hashes."""
        with self._lock:
            if self._active is not None:
                raise RuntimeError('Scene artifacts may only be read after its forward finishes')
            return None if self._last_scene is None else dict(self._last_scene)

    def scene_record(self):
        scene = self.last_scene
        if scene is None:
            return None
        result = {k: v for k, v in scene.items() if k not in ('owner', 'plan', 'features', 'predictor_features',
                  'confidence', 'group_means', 'confidence_quantile')}
        result['plan'] = None if scene['plan'] is None else scene['plan'].summary()
        result['threshold_fraction'] = self.threshold
        result['arrays'] = {key: _tensor_record(scene[key]) for key in
            ('features', 'predictor_features', 'confidence', 'group_means', 'confidence_quantile') if key in scene}
        result['feature_hashes_computed_after_forward'] = True
        result['synthetic_plan_callback_used'] = self.plan_callback is not None
        result['synthetic_predictor_callback_used'] = not isinstance(self.predictor, __import__('torch').nn.Module)
        return result

    def metrics(self):
        with self._lock:
            result = json.loads(json.dumps(self._stats))
            result.update(closed=self._closed, active_scene=self._active is not None,
                          merge_helper_sha256=MERGE_HELPER_SHA256, research_only=True,
                          attention_merged=False, full_residual_merged=False, npu_used=False,
                          original_MLP_modules_retained=True, expected_MLP_calls_per_scene=57,
                          synthetic_plan_callback_used=self.plan_callback is not None,
                          synthetic_predictor_callback_used=not isinstance(self.predictor, __import__('torch').nn.Module),
                          original_weights_before=None if self.weights_before is None else {
                              k: v for k, v in self.weights_before.items() if k != 'parameters'},
                          original_weights_after=None if self.weights_after is None else {
                              k: v for k, v in self.weights_after.items() if k != 'parameters'},
                          original_weight_hashes_unchanged=(None if self.weights_after is None
                              else self.weights_after == self.weights_before))
            return result

    def _restore(self):
        for hook in reversed(self._hooks):
            try: hook.remove()
            except BaseException as exc: self._stats['cleanup_errors'].append(f'hook: {type(exc).__name__}: {exc}')
        self._hooks.clear()
        for module, had_instance_forward, instance_forward in reversed(self._saved_forwards):
            try:
                if had_instance_forward: module.forward = instance_forward
                else: del module.forward
            except BaseException as exc: self._stats['cleanup_errors'].append(f'forward: {type(exc).__name__}: {exc}')
        self._saved_forwards.clear()
        self._active = None
        self._invocations.clear()

    def close(self):
        if self._closed:
            return
        with self._lock:
            if self._active is not None:
                raise RuntimeError('Cannot restore a model while scene inference is active')
            self._restore()
            self._closed = True
        if self.weights_before is not None:
            self.weights_after = _weight_manifest(self.model)
            if self.weights_after != self.weights_before:
                raise RuntimeError('Original model weights changed during the research context')
        if self._stats['cleanup_errors']:
            raise RuntimeError('Research hooks/forwards could not be completely restored')

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except BaseException as cleanup_exc:
            if exc is None:
                raise
            self._stats['cleanup_errors'].append(f'context exit: {type(cleanup_exc).__name__}: {cleanup_exc}')
        return False
