"""Opt-in, inference-only reuse of the previous main batch's DINO tail.

Only normalized CPU FP32 image content identifies a hit. Similarity/loop calls
run original DINO and leave the main cache alone. The module hierarchy is never
replaced, so an installed RKNN Linear handle still owns its original parents.

One board diagnostic established bit equality for one input group. This module
does not establish bit equality for arbitrary batches, accuracy, or realtime.
Install after choosing the backend; close before restoring that backend.
"""
from __future__ import annotations

import copy
import hashlib
import threading
import time

import torch
from torch import nn


class PatchCacheError(RuntimeError):
    """Unsupported inference or a changed fixed-model lifecycle; no fallback."""


class PatchCacheStaleState(PatchCacheError):
    """Parameters, buffers, modules, forwards, or backend changed after install."""


def _forward_identity(function):
    return (id(getattr(function, "__self__", None)),
            id(getattr(function, "__func__", function)))


def _tensor_state(name, tensor, *, parameter):
    if tensor.device.type != "cpu" or (parameter or tensor.is_floating_point()) and tensor.dtype != torch.float32:
        raise PatchCacheStaleState(f"{name}: fixed model must retain CPU FP32 parameters/floating buffers")
    try:
        version = tensor._version
    except RuntimeError as exc:
        raise PatchCacheStaleState(f"{name}: cannot verify tensor version") from exc
    return (name, id(tensor), version, tensor.dtype, tensor.device,
            tuple(tensor.shape), tuple(tensor.stride()), tensor.data_ptr(), tensor.requires_grad)


class PatchCacheHandle:
    """Own two forward overrides and one independent cloned patch-token frame."""

    def __init__(self, model: nn.Module, max_bytes: int):
        if not isinstance(model, nn.Module):
            raise TypeError("Patch cache requires a torch.nn.Module model")
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        aggregator = getattr(model, "aggregator", None)
        patch_embed = getattr(aggregator, "patch_embed", None)
        if not isinstance(aggregator, nn.Module) or not isinstance(patch_embed, nn.Module):
            raise PatchCacheError("Patch cache requires model.aggregator.patch_embed")
        if aggregator is patch_embed:
            raise PatchCacheError("Aggregator and DINO must be distinct modules")
        for module in (aggregator, patch_embed):
            if getattr(module, "_vggt_patch_cache_owner", None) is not None:
                raise PatchCacheError("A patch cache is already installed")
        self.model, self.aggregator, self.patch_embed = model, aggregator, patch_embed
        self.max_bytes = max_bytes
        self.lock = threading.RLock()
        self.closed = self.restored = False
        self._installed = False
        self._context = None
        self._tail_key = self._tail_tokens = None
        self._original_aggregator = aggregator.forward
        self._original_patch = patch_embed.forward
        self._original_overrides = [(module, "forward" in module.__dict__, module.__dict__.get("forward"))
                                    for module in (aggregator, patch_embed)]
        self._aggregate_callable = self._aggregate_forward
        self._patch_callable = self._patch_forward
        self._stats = dict(requests=0, input_frames=0, hit_frames=0, miss_frames=0,
                           computed_frames=0, loop_bypass_calls=0, loop_bypass_frames=0,
                           loop_computed_frames=0, original_dino_calls=0,
                           completed_original_dino_calls=0, errors=0, invalidations=0,
                           peak_cache_bytes=0, lock_wait_seconds=0., guard_seconds=0.,
                           hash_seconds=0., lookup_seconds=0., original_dino_seconds=0.,
                           loop_original_dino_seconds=0., assembly_seconds=0.,
                           tail_clone_seconds=0., cache_forward_seconds=0.,
                           aggregator_seconds=0.)
        self._last_error = None
        self._restore_errors = []
        self._state = self._fingerprint()
        try:
            aggregator.forward = self._aggregate_callable
            patch_embed.forward = self._patch_callable
            aggregator._vggt_patch_cache_owner = self
            patch_embed._vggt_patch_cache_owner = self
            self._installed = True
        except BaseException:
            for module, had, value in self._original_overrides:
                if had:
                    module.forward = value
                elif "forward" in module.__dict__:
                    del module.forward
                if getattr(module, "_vggt_patch_cache_owner", None) is self:
                    del module._vggt_patch_cache_owner
            self._clear()
            raise

    def _fingerprint(self):
        modules = []
        backends = []
        for name, module in self.model.named_modules():
            if module.training:
                raise PatchCacheStaleState("Patch cache requires all model modules to remain in eval mode")
            forward = (self._original_aggregator if module is self.aggregator else
                       self._original_patch if module is self.patch_embed else module.forward)
            hooks = tuple((attribute, tuple((key, id(value)) for key, value in getattr(module, attribute).items()))
                          for attribute in ("_forward_pre_hooks", "_forward_hooks"))
            scalar_fields = tuple((key, getattr(module, key)) for key in
                ("eps", "patch_size", "embed_dim", "num_register_tokens", "num_heads",
                 "interpolate_antialias", "interpolate_offset", "chunked_blocks", "use_reentrant")
                if hasattr(module, key))
            modules.append((name, id(module), type(module), id(type(module).forward),
                            _forward_identity(forward), hooks, scalar_fields))
            # RKNN replacement objects remain intact. Mutable context LRU/stats
            # are deliberately excluded; backend/library/config identities are not.
            if hasattr(module, "packed_weight") and hasattr(module, "pool"):
                pool, packed = module.pool, module.packed_weight
                api = getattr(pool, "api", None)
                backends.append((name, id(pool), id(api), id(getattr(api, "library", None)),
                    id(packed), str(getattr(packed, "dtype", None)), tuple(getattr(packed, "shape", ())),
                    id(getattr(module, "weight_token", None)), getattr(module, "versions", None),
                    tuple((key, getattr(pool, key, None)) for key in
                          ("closed", "max_rows", "max_contexts", "max_cache_bytes", "core_mask", "weight_layout")),
                    tuple(id(getattr(api, key, None)) for key in
                          ("rknn_matmul_create", "rknn_matmul_run", "rknn_matmul_set_io_mem"))))
        parameters = tuple(_tensor_state(name, tensor, parameter=True) for name, tensor in self.model.named_parameters())
        buffers = tuple(_tensor_state(name, tensor, parameter=False) for name, tensor in self.model.named_buffers())
        return tuple(modules), parameters, buffers, tuple(backends)

    def _guard(self):
        started = time.perf_counter()
        try:
            if self.closed or self.restored or not self._installed:
                raise PatchCacheError("Patch cache is closed or restored")
            if torch.is_grad_enabled():
                raise PatchCacheError("Patch cache requires no_grad() or inference_mode()")
            if (self.aggregator.forward is not self._aggregate_callable
                    or self.patch_embed.forward is not self._patch_callable
                    or getattr(self.aggregator, "_vggt_patch_cache_owner", None) is not self
                    or getattr(self.patch_embed, "_vggt_patch_cache_owner", None) is not self):
                raise PatchCacheStaleState("A patch-cache forward or owner changed after installation")
            if self._fingerprint() != self._state:
                raise PatchCacheStaleState("Fixed model modules/parameters/buffers/backend changed; restore and reinstall")
        finally:
            self._stats["guard_seconds"] += time.perf_counter() - started

    @staticmethod
    def _input(images):
        if (not isinstance(images, torch.Tensor) or images.device.type != "cpu"
                or images.dtype != torch.float32 or images.requires_grad or images.ndim != 4
                or images.shape[0] < 1 or images.shape[1] != 3 or min(images.shape[2:]) < 1
                or not torch.isfinite(images).all().item()):
            raise PatchCacheError("DINO cache input must be finite normalized CPU FP32 NCHW without gradients")

    def _clear(self):
        self._tail_key = self._tail_tokens = None

    def _failure(self, exc, *, loop):
        self._stats["errors"] += 1
        self._last_error = f"{type(exc).__name__}: {exc}"
        # Ordinary loop failures never touch main-tail state. A changed model
        # invalidates it even if that change was first noticed in a loop call.
        if not loop or isinstance(exc, PatchCacheStaleState):
            self._clear()
            self._stats["invalidations"] += 1

    def _aggregate_forward(self, images, compute_similarity=False):
        waiting = time.perf_counter()
        with self.lock:
            self._stats["lock_wait_seconds"] += time.perf_counter() - waiting
            started = time.perf_counter()
            loop = compute_similarity is True
            owns_context = False
            try:
                if type(compute_similarity) is not bool:
                    raise PatchCacheError("compute_similarity must be an explicit bool")
                self._guard()
                if self._context is not None:
                    raise PatchCacheError("Recursive aggregator calls are unsupported")
                self._context = dict(thread=threading.get_ident(), loop=loop, used=False)
                owns_context = True
                result = self._original_aggregator(images, compute_similarity)
                if not self._context["used"]:
                    raise PatchCacheError("Original aggregator did not call DINO exactly once")
                # Detect a concurrent tracked mutation before accepting the result.
                self._guard()
                return result
            except BaseException as exc:
                self._failure(exc, loop=loop)
                raise
            finally:
                if owns_context:
                    self._context = None
                self._stats["aggregator_seconds"] += time.perf_counter() - started

    def _original(self, images, *, loop):
        self._stats["original_dino_calls"] += 1
        started = time.perf_counter()
        try:
            result = self._original_patch(images)
        finally:
            seconds = time.perf_counter() - started
            self._stats["original_dino_seconds"] += seconds
            if loop:
                self._stats["loop_original_dino_seconds"] += seconds
        tokens = result.get("x_norm_patchtokens") if isinstance(result, dict) else None
        if (not isinstance(tokens, torch.Tensor) or tokens.device.type != "cpu"
                or tokens.dtype != torch.float32 or tokens.requires_grad or tokens.ndim != 3
                or tokens.shape[0] != images.shape[0] or min(tokens.shape[1:]) < 1
                or not torch.isfinite(tokens).all().item()):
            raise PatchCacheError("Original DINO must return finite CPU FP32 x_norm_patchtokens")
        self._stats["completed_original_dino_calls"] += 1
        self._stats["loop_computed_frames" if loop else "computed_frames"] += len(images)
        return result, tokens

    def _patch_forward(self, images, *args, **kwargs):
        waiting = time.perf_counter()
        with self.lock:
            self._stats["lock_wait_seconds"] += time.perf_counter() - waiting
            context = self._context
            if context is None or context["thread"] != threading.get_ident():
                exc = PatchCacheError("Cached DINO must be called through the original aggregator")
                self._failure(exc, loop=True)
                raise exc
            started = time.perf_counter()
            try:
                if context["used"] or args or kwargs:
                    raise PatchCacheError("Cached aggregator permits exactly one plain DINO image-tensor call")
                context["used"] = True
                self._input(images)
                if context["loop"]:
                    self._stats["loop_bypass_calls"] += 1
                    self._stats["loop_bypass_frames"] += len(images)
                    result, _ = self._original(images, loop=True)
                    return result
                self._stats["requests"] += 1
                self._stats["input_frames"] += len(images)
                hashed = time.perf_counter()
                keys = [(tuple(frame.shape), str(frame.dtype),
                         hashlib.sha256(frame.detach().contiguous().numpy()).hexdigest()) for frame in images]
                self._stats["hash_seconds"] += time.perf_counter() - hashed
                looked_up = time.perf_counter()
                hits = [key == self._tail_key and self._tail_tokens is not None for key in keys]
                missing = [index for index, hit in enumerate(hits) if not hit]
                self._stats["hit_frames"] += sum(hits)
                self._stats["miss_frames"] += len(missing)
                self._stats["lookup_seconds"] += time.perf_counter() - looked_up
                if missing:
                    # Preserve original miss order. A cold batch uses the original
                    # input directly; no synthetic warmup or per-frame DINO calls.
                    miss_images = images if len(missing) == len(images) else images[missing].contiguous()
                    _, computed = self._original(miss_images, loop=False)
                assembled = time.perf_counter()
                if len(missing) == len(images):
                    tokens = computed
                else:
                    parts, miss_row = [], 0
                    for hit in hits:
                        if hit:
                            parts.append(self._tail_tokens)
                        else:
                            parts.append(computed[miss_row])
                            miss_row += 1
                    tokens = torch.stack(parts, dim=0)
                    del parts
                self._stats["assembly_seconds"] += time.perf_counter() - assembled
                size = tokens[-1].numel() * tokens.element_size()
                if size > self.max_bytes:
                    raise PatchCacheError(f"Tail patch tokens need {size} bytes, exceeding max_bytes={self.max_bytes}")
                cloned = time.perf_counter()
                # The persistent cache never retains a view of a whole batch.
                self._tail_tokens = tokens[-1].detach().clone().contiguous()
                self._tail_key = keys[-1]
                self._stats["peak_cache_bytes"] = max(self._stats["peak_cache_bytes"], size)
                self._stats["tail_clone_seconds"] += time.perf_counter() - cloned
                return {"x_norm_patchtokens": tokens}
            finally:
                self._stats["cache_forward_seconds"] += time.perf_counter() - started

    def restore(self):
        with self.lock:
            if self.restored:
                return
            if self._context is not None:
                raise PatchCacheError("Cannot restore a patch cache from inside its active forward")
            errors = []
            for (module, had, value), installed, original in zip(self._original_overrides,
                    (self._aggregate_callable, self._patch_callable),
                    (self._original_aggregator, self._original_patch)):
                if module.forward is installed:
                    if had:
                        module.forward = value
                    else:
                        del module.forward
                elif _forward_identity(module.forward) != _forward_identity(original):
                    errors.append("Cannot restore forward changed by caller")
                if getattr(module, "_vggt_patch_cache_owner", None) is self:
                    del module._vggt_patch_cache_owner
            self._clear()
            self._installed = False
            self.restored = not errors
            self._restore_errors = errors
            if errors:
                raise PatchCacheError("; ".join(errors))

    def close(self):
        with self.lock:
            if self.closed:
                return
            # An in-forward close cannot invalidate the ongoing computation.
            if self._context is not None:
                raise PatchCacheError("Cannot close a patch cache from inside its active forward")
            try:
                self.restore()
            finally:
                self._clear()
                self.closed = True

    def metrics(self):
        with self.lock:
            size = 0 if self._tail_tokens is None else self._tail_tokens.numel() * self._tail_tokens.element_size()
            key = None if self._tail_key is None else dict(shape=list(self._tail_key[0]),
                    dtype=self._tail_key[1], normalized_content_sha256=self._tail_key[2])
            return copy.deepcopy(dict(self._stats, enabled=self._installed and not self.closed and not self.restored,
                    cache_entries=int(self._tail_tokens is not None), cache_bytes=size, max_bytes=self.max_bytes,
                    tail_content_key=key, closed=self.closed, restored=self.restored,
                    last_error=self._last_error, restore_errors=list(self._restore_errors),
                    policy="previous_main_batch_last_frame_only", loop_policy="original_DINO_no_main_cache_read_or_write",
                    cached_payload="cloned_x_norm_patchtokens_CPU_FP32_only",
                    model_lifecycle="fixed_backend_modules_parameters_buffers_versions_storage",
                    arbitrary_batch_bit_equality_validated=False, accuracy_validated=False, real_time_validated=False))


def install_patch_cache(model, max_bytes=8 * 1024 * 1024):
    """Install after the CPU/RKNN backend; return an explicitly owned handle."""
    return PatchCacheHandle(model, max_bytes)
