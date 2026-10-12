"""Private, opt-in exclusive immutable execution contract; no VGGT math changes.

After all backend/cache/observer installs, the worker exclusively owns the model.
No module/registry/hook/class/config changes, tensor replacement, .data writes or
external model access are permitted until close. Normal tensor version changes
are still rejected per call. Arbitrary monkeypatch detection is deliberately
deferred to final full validation; this is NOT the old mutable-model contract.
Close this session before cache/backends, in a finally that still closes them if
validation fails. Imports no Torch and performs no model or SDK execution.
"""
from __future__ import annotations
import threading
import time
import types


class ImmutableExecutionSession:
    def __init__(self, model, cache, retained, dino, *, contract):
        if contract != 'exclusive_immutable_model':
            raise ValueError('Explicit exclusive_immutable_model contract required')
        if cache.model is not model or retained.model is not model or dino.aggregator is not model.aggregator:
            raise ValueError('All handles must own the same original model')
        self.model, self.cache, self.retained, self.dino = model, cache, retained, dino
        self.thread = threading.get_ident()
        self.closed = self.failed = False
        self.overrides = []
        self.torch = retained.torch
        self.stale = cache._guard.__func__.__globals__['PatchCacheStaleState']
        self.full_validation = False
        self.original_cache_guard = cache._guard
        self.original_fingerprint = cache._fingerprint
        self.original_retained = retained._check_lifecycle
        self.original_dino = dino._check_lifecycle
        self.stats = dict(contract=contract, full_seal_validated=False, final_full_validated=False,
            seal_seconds=0., final_full_validation_seconds=0., version_checks=0,
            version_check_seconds=0., method_checks=0, method_check_seconds=0.,
            fixed_tensors=0, invalidations=0, restored_overrides=0, cleanup_errors=[],
            arbitrary_live_module_scalar_registry_mutation_detection=False,
            original_mutable_model_guard_semantics_equivalent=False)
        begin = time.perf_counter()
        self.snapshot = self._full_validate()
        self.fixed = tuple((name, value, self._state(value))
            for name, value in (*model.named_parameters(), *model.named_buffers()))
        self.stats.update(full_seal_validated=True, fixed_tensors=len(self.fixed),
                          seal_seconds=time.perf_counter() - begin)
        try:
            self._override(cache, '_fingerprint', self._fingerprint)
            self._override(retained, '_check_lifecycle', self._retained_check)
            self._override(dino, '_check_lifecycle', self._dino_check)
        except BaseException:
            self._restore()
            raise

    @staticmethod
    def _state(value):
        return value._version, value.dtype, value.device, value.data_ptr(), value.requires_grad

    def _full_validate(self):
        # Initialization/close may occur outside inference_mode; read-only checks
        # use the already loaded backend's no_grad context, never import Torch.
        with self.torch.no_grad():
            self.full_validation = True
            try:
                self.original_cache_guard()
                self.original_retained()
                self.original_dino()
                return self.cache._state
            finally:
                self.full_validation = False

    def _override(self, owner, name, function):
        had, old = name in owner.__dict__, owner.__dict__.get(name)
        bound = types.MethodType(lambda _owner: function(), owner)
        self.overrides.append((owner, name, had, old, bound))
        setattr(owner, name, bound)

    def _fail(self, message):
        self.failed = True
        self.stats['invalidations'] += 1
        self.cache._clear()
        raise self.stale(message)

    def _entry(self, *, nested=False):
        if self.closed or self.failed or threading.get_ident() != self.thread:
            self._fail('Immutable session closed, failed or called by another thread')
        if nested:
            context = self.cache._context
            if context is None or context['thread'] != self.thread:
                self._fail('Immutable backend must run inside its owned patch-cache call')

    def _fingerprint(self):
        if self.full_validation:
            return self.original_fingerprint()
        self._entry()
        begin = time.perf_counter()
        try:
            for name, value, expected in self.fixed:
                try:
                    actual = self._state(value)
                except RuntimeError as exc:
                    self.failed = True
                    self.cache._clear()
                    self.stats['invalidations'] += 1
                    raise self.stale(name + ': cannot verify fixed tensor metadata') from exc
                if actual != expected:
                    self._fail(name + ': fixed tensor version/metadata changed')
            self.stats['version_checks'] += 1
            return self.snapshot
        finally:
            self.stats['version_check_seconds'] += time.perf_counter() - begin

    def _retained_check(self):
        begin = time.perf_counter()
        try:
            self._entry(nested=True)
            self.retained._check_methods()
            global_module = self.torch.nn.modules.module
            fields = ('_forward_pre_hooks', '_forward_hooks', '_backward_pre_hooks', '_backward_hooks')
            if any(getattr(global_module, '_global' + name, {}) for name in fields):
                self._fail('Global hooks cannot observe bypassed original aggregator blocks')
            if self.retained._backend_state() != self.retained.backend_state:
                self._fail('Retained backend identity/configuration changed')
            self.stats['method_checks'] += 1
        finally:
            elapsed = time.perf_counter() - begin
            self.retained.stats['guard_seconds'] += elapsed
            self.stats['method_check_seconds'] += elapsed

    def _dino_check(self):
        begin = time.perf_counter()
        try:
            self._entry(nested=True)
            # Includes the original global-hook rejection; no tree enumeration.
            self.dino._check_methods()
            self.stats['method_checks'] += 1
        finally:
            self.stats['method_check_seconds'] += time.perf_counter() - begin

    def _restore(self):
        for owner, name, had, old, bound in reversed(self.overrides):
            try:
                if owner.__dict__.get(name) is not bound:
                    raise RuntimeError('Session-owned override changed: ' + name)
                if had:
                    setattr(owner, name, old)
                else:
                    delattr(owner, name)
                self.stats['restored_overrides'] += 1
            except BaseException as exc:
                self.stats['cleanup_errors'].append(f'{type(exc).__name__}: {exc}')

    def close(self):
        if self.closed:
            return
        if threading.get_ident() != self.thread or self.cache._context is not None:
            raise RuntimeError('Session close requires its idle owner thread')
        error = None
        begin = time.perf_counter()
        try:
            self._full_validate()
            self.stats['final_full_validated'] = True
        except BaseException as exc:
            self.failed = True
            self.cache._clear()
            error = exc
            self.stats['cleanup_errors'].append(f'Final validation: {type(exc).__name__}: {exc}')
        finally:
            self.stats['final_full_validation_seconds'] += time.perf_counter() - begin
            self._restore()
            self.closed = True
        if error is not None:
            raise error
        if self.stats['cleanup_errors']:
            raise RuntimeError('Immutable session restoration failed: ' + '; '.join(self.stats['cleanup_errors']))

    def metrics(self):
        return dict(self.stats, closed=self.closed, failed=self.failed,
                    cleanup_errors=list(self.stats['cleanup_errors']))


def install_immutable_session(model, cache, retained, dino, *, contract='exclusive_immutable_model'):
    return ImmutableExecutionSession(model, cache, retained, dino, contract=contract)
