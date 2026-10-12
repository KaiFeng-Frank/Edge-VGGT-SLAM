"""Tiny generated metadata checks; no Torch/model/SDK imports or execution."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('immutable_candidate', ROOT/'vggt_slam/immutable_execution_session.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PatchCacheStaleState(RuntimeError):
    pass


class FakeTorch:
    def __init__(self):
        self.grad = True
        self.global_hooks = SimpleNamespace(_global_forward_hooks={})
        self.nn = SimpleNamespace(modules=SimpleNamespace(module=self.global_hooks))

    @contextmanager
    def no_grad(self):
        previous = self.grad
        self.grad = False
        try:
            yield
        finally:
            self.grad = previous

    def is_grad_enabled(self):
        return self.grad


class Tensor:
    def __init__(self, pointer, dtype='float32'):
        self._version, self.dtype, self.device = 0, dtype, 'cpu'
        self.pointer, self.requires_grad = pointer, False

    def data_ptr(self):
        return self.pointer


class Model:
    def __init__(self):
        self.aggregator = SimpleNamespace(forward=object(), patch_embed=SimpleNamespace(forward=object()))
        self.parameters = {'weight': Tensor(123), 'bias': Tensor(456)}
        self.buffers = {'positions': Tensor(789, 'int64')}
        self.scalar = 4

    def named_parameters(self):
        return iter(self.parameters.items())

    def named_buffers(self):
        return iter(self.buffers.items())


class Cache:
    def __init__(self, model, torch):
        self.model, self.torch = model, torch
        self.forward, self.patch = model.aggregator.forward, model.aggregator.patch_embed.forward
        self._context, self.tail = None, 'cached'
        self.full_calls = self.guard_calls = 0
        self._state = self._fingerprint()

    def _fingerprint(self):
        self.full_calls += 1
        return (self.model.scalar, tuple((name, id(value), module.ImmutableExecutionSession._state(value))
                for name, value in (*self.model.named_parameters(), *self.model.named_buffers())))

    def _guard(self):
        self.guard_calls += 1
        if (self.torch.is_grad_enabled() or self.model.aggregator.forward is not self.forward
                or self.model.aggregator.patch_embed.forward is not self.patch):
            raise PatchCacheStaleState('guard method/mode differs')
        if self._fingerprint() != self._state:
            raise PatchCacheStaleState('full model differs')

    def _clear(self):
        self.tail = None


class Retained:
    def __init__(self, model, torch):
        self.model, self.torch = model, torch
        self.stats, self.backend_state = {'guard_seconds': 0.}, 'pinned'
        self.backend = 'pinned'
        self.full_calls = self.method_calls = 0

    def _check_methods(self):
        self.method_calls += 1
        if self.torch.is_grad_enabled():
            raise RuntimeError('no_grad required')

    def _backend_state(self):
        return self.backend

    def _check_lifecycle(self):
        self.full_calls += 1
        self._check_methods()
        if self.backend != self.backend_state or self.model.scalar != 4:
            raise RuntimeError('retained original full state differs')


class Dino:
    def __init__(self, model, torch):
        self.aggregator, self.model, self.torch = model.aggregator, model, torch
        self.full_calls = self.method_calls = 0

    def _check_methods(self):
        self.method_calls += 1
        if self.torch.is_grad_enabled() or self.torch.global_hooks._global_forward_hooks:
            raise RuntimeError('DINO mode/global hooks differ')

    def _check_lifecycle(self):
        self.full_calls += 1
        self._check_methods()
        if self.model.scalar != 4:
            raise RuntimeError('DINO original full state differs')


def fixture():
    model, torch = Model(), FakeTorch()
    cache, retained, dino = Cache(model, torch), Retained(model, torch), Dino(model, torch)
    session = module.install_immutable_session(model, cache, retained, dino)
    return model, torch, cache, retained, dino, session


class Tests(unittest.TestCase):
    def test_two_original_cache_guards_and_three_overrides_only(self):
        model, torch, cache, retained, dino, session = fixture()
        sealed_calls = cache.full_calls
        with torch.no_grad():
            cache._guard()
            cache._context = {'thread': threading.get_ident()}
            retained._check_lifecycle()
            dino._check_lifecycle()
            cache._guard()
            cache._context = None
        self.assertEqual(cache.full_calls, sealed_calls)
        self.assertEqual(cache.guard_calls, 3)  # One full seal + two unchanged entry/exit guards.
        self.assertEqual((retained.full_calls, dino.full_calls), (1, 1))
        self.assertEqual(session.metrics()['version_checks'], 2)
        self.assertGreater(retained.stats['guard_seconds'], 0.)
        session.close()
        self.assertTrue(session.metrics()['final_full_validated'])
        self.assertEqual(session.metrics()['restored_overrides'], 3)
        self.assertTrue(torch.grad)
        for obj, name in ((cache, '_fingerprint'), (retained, '_check_lifecycle'), (dino, '_check_lifecycle')):
            self.assertNotIn(name, obj.__dict__)

    def test_each_fixed_tensor_metadata_change_invalidates_and_restores(self):
        for field, value in [('_version', 1), ('dtype', 'float16'), ('device', 'other'), ('pointer', 999), ('requires_grad', True)]:
            with self.subTest(field=field):
                model, torch, cache, retained, dino, session = fixture()
                setattr(model.parameters['weight'], field, value)
                with torch.no_grad(), self.assertRaises(PatchCacheStaleState):
                    cache._guard()
                self.assertIsNone(cache.tail)
                with self.assertRaises(PatchCacheStaleState):
                    session.close()
                self.assertTrue(session.closed)
                self.assertEqual(session.metrics()['restored_overrides'], 3)

    def test_contract_defers_scalar_mutation_to_full_close(self):
        model, torch, cache, retained, dino, session = fixture()
        model.scalar = 5  # Forbidden by the opt-in contract; no live scalar-reflection promise.
        with torch.no_grad():
            cache._guard()
        with self.assertRaises(PatchCacheStaleState):
            session.close()
        self.assertEqual(session.metrics()['restored_overrides'], 3)

    def test_backend_and_global_hooks_still_rejected(self):
        for change in ('backend', 'hook'):
            with self.subTest(change=change):
                model, torch, cache, retained, dino, session = fixture()
                cache._context = {'thread': threading.get_ident()}
                if change == 'backend':
                    retained.backend = 'changed'
                else:
                    torch.global_hooks._global_forward_hooks[1] = object()
                with torch.no_grad(), self.assertRaises(PatchCacheStaleState):
                    retained._check_lifecycle()
                cache._context = None
                retained.backend = retained.backend_state
                torch.global_hooks._global_forward_hooks.clear()
                session.close()

    def test_active_or_foreign_thread_close_cannot_remove_overrides(self):
        model, torch, cache, retained, dino, session = fixture()
        owned = cache.__dict__['_fingerprint']
        cache._context = {'thread': threading.get_ident()}
        with self.assertRaises(RuntimeError):
            session.close()
        self.assertFalse(session.closed)
        self.assertIs(cache.__dict__['_fingerprint'], owned)
        cache._context = None
        errors = []
        def foreign_close():
            try:
                session.close()
            except RuntimeError as exc:
                errors.append(str(exc))
        thread = threading.Thread(target=foreign_close)
        thread.start(); thread.join()
        self.assertEqual(len(errors), 1)
        self.assertFalse(session.closed)
        self.assertIs(cache.__dict__['_fingerprint'], owned)
        session.close()

    def test_foreign_override_not_overwritten(self):
        model, torch, cache, retained, dino, session = fixture()
        foreign = lambda: cache._state
        cache._fingerprint = foreign
        with self.assertRaises(RuntimeError):
            session.close()
        self.assertIs(cache._fingerprint, foreign)
        self.assertTrue(session.closed)
        self.assertNotIn('_check_lifecycle', retained.__dict__)
        self.assertNotIn('_check_lifecycle', dino.__dict__)


if __name__ == '__main__':
    unittest.main()
