"""Tiny fake cache contracts only; no SDK/native/real model execution."""
import copy
import importlib.util
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
PARENT = Path(os.environ.get('V44_PARENT_RUNTIME',
    HERE.parent/'dpvo_grouped_update/v37_grouped_rknn.py'))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value); return value


parent = load(PARENT, '_v44_test_frozen_cache_parent')
candidate = load(HERE/'v44_closed_flag_cache.py', '_v44_test_closed_flag_candidate')
new_class = candidate.make_cache_class(parent.BoundedContextCache, parent_runtime_path=PARENT)


def specs():
    result = {}
    for degree in (4, 5):
        ins = [dict(name=n, shape=[1, c, degree, 48], dtype='float32')
            for n, c in zip(('net', 'inp', 'corr'), (384, 384, 882))]
        outs = [dict(name=n, shape=[1, c, degree, 48], dtype='float32')
            for n, c in zip(('net_out', 'delta', 'weight'), (384, 2, 2))]
        result[f'B1_D{degree}_M48'] = dict(inputs=ins, outputs=outs, model_bytes=10, degree=degree)
    return result


class Session:
    def __init__(self, spec, fault=None, events=None):
        self._lock = threading.RLock(); self.metrics_calls = 0; self.fault = fault
        self.seen = []; self.events = events
        self._metrics = dict(closed=fault == 'returned_closed', context_destroyed=False,
            cleanup_errors=[], current_model_host_bytes=11 if fault != 'wrong_host' else 12,
            current_DMA_bytes=8 if fault != 'excess_dma' else 100,
            nested=dict(source=['generated']*4, attrs=[dict(shape=[1, 2, 3])]),
            synthetic=True, npu_used=False, calls=0)

    def metrics(self):
        with self._lock:
            self.metrics_calls += 1
            return copy.deepcopy(self._metrics)

    def execute(self, feed, reference=None, *, timeout_ms):
        with self._lock:
            self.seen.append((feed, reference, timeout_ms)); self._metrics['calls'] += 1
            if self.fault == 'execute': raise RuntimeError('Generated execute failure')
            return dict(owned=np.array(feed['tiny'], copy=True, order='C')), dict(execution_status='completed')

    def close(self):
        with self._lock:
            self._metrics['closed'] = True
            if self.fault == 'destroy':
                self._metrics['cleanup_errors'] = ['Generated destroy failure']
            else:
                self._metrics.update(context_destroyed=True,
                    current_model_host_bytes=0, current_DMA_bytes=0)


def cache(cls, *, fault=None, **limits):
    sessions = []
    def factory(spec):
        value = Session(spec, fault); sessions.append(value); return value
    instance = cls(specs(), factory, max_contexts=limits.get('count', 2),
        max_model_bytes=limits.get('host', 22), max_dma_bytes=limits.get('dma', 16))
    return instance, sessions


def execute(instance, key='B1_D4_M48'):
    return instance.execute(key, {'tiny': np.arange(6, dtype=np.float32).reshape(2, 3)}, timeout_ms=123)


def test_exact_AST_delta_inherited_methods_and_runtime_identity():
    proof = candidate.source_change_proof(PARENT)
    assert proof['changed_expression_count'] == 1 and proof['all_other_get_AST_preserved']
    for name in proof['inherited_methods']:
        assert getattr(new_class, name) is getattr(parent.BoundedContextCache, name)
    original = SimpleNamespace(BoundedContextCache=parent.BoundedContextCache,
        GroupedSession=parent.GroupedSession, execute_full_graph=parent.execute_full_graph,
        pack_inputs=parent.pack_inputs, source_pins={'generated': 'a'*64})
    bound = candidate.bind_runtime(original, parent_runtime_path=PARENT, diagnostic_allow_closed_flag=True)
    assert original.BoundedContextCache is parent.BoundedContextCache
    assert bound.GroupedSession is original.GroupedSession
    assert bound.pack_inputs is original.pack_inputs and bound.execute_full_graph is original.execute_full_graph
    assert original.source_pins == {'generated': 'a'*64}
    assert bound.source_pins is not original.source_pins and bound.closed_flag_only


def test_only_warm_closed_query_avoids_metrics_copy_owned_output_unchanged():
    values = []
    for cls in (parent.BoundedContextCache, new_class):
        instance, sessions = cache(cls); execute(instance)
        session = sessions[0]; start = session.metrics_calls
        feed = {'tiny': np.arange(6, dtype=np.float32).reshape(2, 3)}; ref = object()
        output, record = instance.execute('B1_D4_M48', feed, ref, timeout_ms=456)
        assert session.metrics_calls-start == (1 if cls is parent.BoundedContextCache else 0)
        assert session.seen[-1][0] is feed and session.seen[-1][1] is ref and session.seen[-1][2] == 456
        assert output['owned'].flags.owndata and not np.shares_memory(output['owned'], feed['tiny'])
        assert record['model_key'] == 'B1_D4_M48' and record['cache_call_seconds'] >= 0
        before = session.metrics_calls; instance.metrics()
        assert session.metrics_calls > before  # Whole reporting still uses original full metrics.
        instance.close(); state = instance.metrics(); instance.close()
        assert state['closed'] and state['current_DMA_bytes'] == state['current_model_host_bytes'] == 0
        values.append((output['owned'], state))
    assert np.array_equal(values[0][0], values[1][0]) and values[0][1] == values[1][1]


class TraceLock:
    def __init__(self, label, events): self.label, self.events, self.inner = label, events, threading.RLock()
    def __enter__(self): self.inner.acquire(); self.events.append(('enter', self.label)); return self
    def __exit__(self, *exc): self.events.append(('exit', self.label)); self.inner.release()


def test_lock_order_cache_then_session_matches_original():
    traces = []
    for cls in (parent.BoundedContextCache, new_class):
        instance, sessions = cache(cls); execute(instance); events = []
        instance._lock = TraceLock('cache', events); sessions[0]._lock = TraceLock('session', events)
        execute(instance); traces.append(events.copy()); instance.close()
    assert traces[0] == traces[1] == [('enter', 'cache'), ('enter', 'session'), ('exit', 'session'),
        ('enter', 'session'), ('exit', 'session'), ('exit', 'cache')]


@pytest.mark.parametrize('limits', (dict(count=1), dict(host=11), dict(dma=8)))
def test_count_host_DMA_eviction_order_history_and_budgets_identical(limits):
    states = []
    for cls in (parent.BoundedContextCache, new_class):
        instance, _ = cache(cls, **limits)
        for key in ('B1_D4_M48', 'B1_D5_M48', 'B1_D4_M48'): execute(instance, key)
        instance.close(); states.append(instance.metrics())
    assert states[0] == states[1] and states[0]['created'] == 3 and states[0]['evicted'] == 2


def test_closed_live_unknown_shape_and_closed_cache_fail_identically():
    for failure in ('live_closed', 'unknown', 'cache_closed'):
        states = []
        for cls in (parent.BoundedContextCache, new_class):
            instance, sessions = cache(cls); execute(instance)
            if failure == 'live_closed': sessions[0].close()
            elif failure == 'cache_closed': instance.close()
            key = 'B9_D9_M16' if failure == 'unknown' else 'B1_D4_M48'
            with pytest.raises(ValueError) as error: execute(instance, key)
            instance.close(); states.append((str(error.value), instance.metrics()))
        assert states[0] == states[1]


def test_cold_allocation_and_destroy_failure_paths_keep_full_metrics():
    for failure in ('returned_closed', 'wrong_host', 'excess_dma', 'destroy'):
        states = []
        for cls in (parent.BoundedContextCache, new_class):
            instance, sessions = cache(cls, fault=failure)
            if failure == 'destroy':
                execute(instance)
                with pytest.raises(RuntimeError): instance.clear()
            else:
                with pytest.raises(ValueError): execute(instance)
            assert sessions[0].metrics_calls > 0
            instance.close(); states.append(instance.metrics())
        assert states[0] == states[1]


def test_generated_execute_exception_retains_identical_cleanup_and_no_output():
    states = []
    for cls in (parent.BoundedContextCache, new_class):
        instance, _ = cache(cls, fault='execute')
        with pytest.raises(RuntimeError, match='Generated execute failure'): execute(instance)
        instance.close(); states.append(instance.metrics())
    assert states[0] == states[1]


def test_diagnostic_flag_and_original_session_metrics_contract_required():
    runtime = SimpleNamespace(BoundedContextCache=parent.BoundedContextCache, GroupedSession=Session,
        source_pins={})
    with pytest.raises(ValueError, match='Explicit diagnostic'): candidate.bind_runtime(runtime, parent_runtime_path=PARENT)
    with pytest.raises(ValueError, match='Original GroupedSession.metrics'):
        candidate.bind_runtime(runtime, parent_runtime_path=PARENT, diagnostic_allow_closed_flag=True)
