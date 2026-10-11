"""Generated arrays/fake session verification only; no model or SDK execution."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import os
import threading

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
CALLBACK_PARENT = Path(os.environ.get('V42_FROZEN_CALLBACK', str(HERE.with_name('dpvo_online_replay')/'v41_online_update.py')))
PARENT = Path(os.environ.get('V42_PARENT_SDK', str(HERE.with_name('dpvo_grouped_update'))))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


old = load(CALLBACK_PARENT, '_v42_perf_parent_callback')
new = load(HERE/'v42_metadata_only_update.py', '_v42_perf_candidate_callback')
adapter = load(PARENT/'source_frame_batches.py', '_v42_perf_frozen_adapter')
runtime = load(PARENT/'v37_grouped_rknn.py', '_v42_perf_frozen_runtime')


class Session:
    def __init__(self):
        self._lock = threading.RLock()
        self._metrics = dict(npu_used=False, closed=False)
        self.metrics_calls = 0

    def metrics(self):
        with self._lock:
            self.metrics_calls += 1
            return copy.deepcopy(self._metrics)


class FakeCache:
    def __init__(self, fail_after=None):
        self.specs = {key: {} for key in old.MODEL_KEYS}
        self._lock = threading.RLock()
        self._history, self._live = [], {}
        self.calls, self.metrics_calls = [], 0
        self.closed, self.fail_after, self.bad_cleanup = False, fail_after, False

    def execute(self, key, feed, reference=None, *, timeout_ms):
        with self._lock:
            self.calls.append(key)
            self._live.setdefault(key, Session())
            if len(self.calls) == self.fail_after:
                raise RuntimeError('fake SDK failure')
            b, _, d, m = feed['net'].shape
            values = dict(net_out=feed['net'] + np.float32(3),
                          delta=feed['inp'][:, :2].copy(),
                          weight=np.full((b, 2, d, m), .5, np.float32))
            return values, dict(execution_status='completed', outputs_finite=True,
                weight_domain_valid=True, all_outputs_zero=False,
                outputs_released=True, strict_FP32_gate_passed=None)

    def metrics(self):
        with self._lock:
            self.metrics_calls += 1
            # Mirror the exact frozen cache's duplicated session metrics copies.
            _ = [s.metrics() for s in self._live.values()]
            return copy.deepcopy(dict(closed=self.closed,
                cleanup_errors=['fake cleanup failure'] if self.bad_cleanup else [],
                resident_contexts=len(self._live), current_DMA_bytes=0,
                current_model_host_bytes=0, histories=self._history,
                sessions={k: s.metrics() for k, s in self._live.items()}))

    def close(self):
        with self._lock:
            self.closed = True
            for key, session in self._live.items():
                session._metrics['closed'] = True
                self._history.append(dict(model_key=key, metrics=session.metrics()))
            self._live.clear()


def graph(sources=3, degree=4):
    ii = np.repeat(np.arange(sources, dtype=np.int64), degree*16)
    jj = np.tile(np.repeat(np.arange(degree, dtype=np.int64), 16), sources)
    kk = np.concatenate([np.tile(np.arange(i*16, (i+1)*16, dtype=np.int64), degree)
                         for i in range(sources)])
    perm = np.random.default_rng(2).permutation(ii.size)
    values = [np.broadcast_to(np.arange(ii.size, dtype=np.float32)[None, :, None]+1,
                             (1, ii.size, c)).copy() for c in (384, 384, 882)]
    return (*values, *(np.ascontiguousarray(v[perm]) for v in (ii, jj, kk)))


def owner(module, path, cache, reader=None):
    path.mkdir()
    return module.OnlineM16Update(runtime, adapter, cache,
        motion_probe=lambda *a: pytest.fail('Full graph cannot use Torch'),
        diagnostic_allow_fp32_rejected=True, failure_dir=path,
        identity_reader=reader)


def test_only_exact_metadata_block_changed():
    raw = (CALLBACK_PARENT).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == new.PARENT_CALLBACK_SHA256
    block = """                    state = self.cache.metrics()
                    used = {v['model_key']: v['metrics'].get('npu_used') is True
                            for v in state.get('histories', [])}
                    used.update({key: v.get('npu_used') is True for key, v in state.get('sessions', {}).items()})"""
    helper = (HERE/'v42_cache_activity.py').read_text()
    source = raw.decode()
    i = source.index('MODEL_KEYS =')
    expected = source[:i]+helper[helper.index('PARENT_CALLBACK_SHA256'):]+'\n\n'+source[i:]
    expected = expected.replace(block, '                    used = npu_used_by_model(self.cache)')
    assert expected == (HERE/'v42_metadata_only_update.py').read_text()


def test_activity_matches_history_live_precedence_and_does_not_copy_metadata():
    cache = FakeCache()
    cache._history = [dict(model_key='evicted', metrics=dict(npu_used=True)),
                      dict(model_key='repeat', metrics=dict(npu_used=True)),
                      dict(model_key='repeat', metrics=dict(npu_used=False)),
                      dict(model_key='live', metrics=dict(npu_used=True))]
    cache._live = {'live': Session(), 'true': Session(), 'integer': Session()}
    cache._live['true']._metrics['npu_used'] = True  # Simulated recorded flag only.
    cache._live['integer']._metrics['npu_used'] = 1
    state = cache.metrics()
    expected = {v['model_key']: v['metrics'].get('npu_used') is True for v in state['histories']}
    expected.update({k: v.get('npu_used') is True for k, v in state['sessions'].items()})
    before = [s.metrics_calls for s in cache._live.values()]
    actual = new.npu_used_by_model(cache)
    assert actual == expected
    assert before == [s.metrics_calls for s in cache._live.values()]
    actual['true'] = False
    assert new.npu_used_by_model(cache)['true'] is True


def test_full_output_owned_equal_hot_metrics_removed_close_checks_retained(tmp_path):
    arrays = graph()
    saved = [v.copy() for v in arrays]
    results = []
    for label, module in [('old', old), ('new', new)]:
        cache = FakeCache(); callback = owner(module, tmp_path/label, cache)
        output = callback(*arrays)
        assert cache.calls == ['B2_D4_M16', 'B1_D4_M16']
        assert cache.metrics_calls == (1 if label == 'old' else 0)
        assert all(v.flags.owndata for v in output.values())
        assert all(not np.shares_memory(v, a) for v in output.values() for a in arrays)
        results.append(output)
        m = callback.metrics()
        assert m['full_completed'] == 1 and m['calls'][0]['NPU_used'] is False
        assert m['calls'][0]['strict_FP32_gate_passed'] is None
        assert m['candidate_accepted'] is False
        before = cache.metrics_calls
        callback.close(); callback.close()
        assert cache.metrics_calls == before+1 and cache.closed
    assert all(np.array_equal(results[0][key], results[1][key]) for key in results[0])
    assert all(np.array_equal(v, b) for v, b in zip(arrays, saved))


@pytest.mark.parametrize('mode', ('late_failure', 'unknown_degree', 'stale'))
def test_complete_graph_failure_boundaries_retained(tmp_path, mode):
    arrays = graph(degree=3 if mode == 'unknown_degree' else 4)
    saved = [v.copy() for v in arrays]
    for label, module in [('old', old), ('new', new)]:
        cache = FakeCache(fail_after=2 if mode == 'late_failure' else None)
        token = adapter.SnapshotIdentity(1, 2, 3, 4)
        reader = (lambda: token if not cache.calls else adapter.SnapshotIdentity(2, 2, 3, 4))
        callback = owner(module, tmp_path/label, cache, reader if mode == 'stale' else None)
        with pytest.raises(module.OnlineUpdateError): callback(*arrays)
        assert callback.metrics()['full_completed'] == 0
        assert callback.metrics()['calls'][0]['fullgraph_returned'] is False
        assert len(cache.calls) == dict(late_failure=2, unknown_degree=0, stale=1)[mode]
        with np.load(tmp_path/label/'update-000000-failed-inputs.npz', allow_pickle=False) as z:
            assert set(z.files) == {'net', 'inp', 'corr', 'ii', 'jj', 'kk'}
            assert all(np.array_equal(z[k], v) for k, v in zip(('net', 'inp', 'corr', 'ii', 'jj', 'kk'), arrays))
        with pytest.raises(RuntimeError, match='closed or failed'): callback(*arrays)
        callback.close()
    assert all(np.array_equal(v, b) for v, b in zip(arrays, saved))


def test_cleanup_failure_remains_visible(tmp_path):
    cache = FakeCache(); callback = owner(new, tmp_path/'close', cache)
    cache.bad_cleanup = True
    with pytest.raises(RuntimeError, match='cleanup failed'): callback.close()
    assert cache.metrics_calls == 1
    assert callback.metrics()['closed'] and callback.metrics()['failed']
