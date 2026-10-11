"""Tiny generated arrays, fake sessions and AST/CLI guards; no real inference."""
import ast
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

TEST = Path(__file__).resolve().parent
HERE = TEST if (TEST/'v43_online_update.py').is_file() else Path(os.environ.get(
    'V43_ONLINE_SOURCE', TEST.parent/'deploy/rk3588/validation/dpvo_online_replay'))
HELPER = Path(os.environ.get('V43_METADATA_HELPER', HERE/'v42_cache_activity.py'
    if (HERE/'v42_cache_activity.py').is_file() else HERE.parent/'dpvo_update_performance/v42_cache_activity.py'))
PARENT = Path(os.environ.get('V43_PARENT_SDK', HERE.parent/'dpvo_grouped_update'))
if not PARENT.is_dir():
    PARENT = HERE.parent/'public-snapshot/deploy/rk3588/validation/dpvo_grouped_update'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    spec.loader.exec_module(value); return value


old = load(HERE/'v42_online_update.py', '_v43_toy_original_callback')
new = load(HERE/'v43_online_update.py', '_v43_toy_metadata_callback')
factory = load(HERE/'v43_online_factory.py', '_v43_toy_metadata_factory')
driver = load(HERE/'run-v43-online-npu-board.py', '_v43_toy_driver')
adapter = load(PARENT/'source_frame_batches.py', '_v43_toy_original_adapter')
adapter.load_pinned_plan(PARENT/'source_frame_plan.py')
runtime = load(PARENT/'v37_grouped_rknn.py', '_v43_toy_original_runtime')


class Session:
    def __init__(self, used=False):
        self._lock = threading.RLock()
        self._metrics = dict(npu_used=used)
        self.metrics_calls = 0

    def metrics(self):
        self.metrics_calls += 1
        return copy.deepcopy(self._metrics)


class Cache:
    def __init__(self, fail_after=None):
        self.specs = {key: {} for key in old.MODEL_KEYS}
        self._lock = threading.RLock(); self._history = []; self._live = {}
        self.calls = []; self.metrics_calls = 0
        self.fail_after = fail_after; self.closed = False; self.bad_cleanup = False

    def execute(self, key, feed, reference=None, *, timeout_ms):
        self.calls.append(key); self._live.setdefault(key, Session(True))
        if len(self.calls) == self.fail_after:
            raise RuntimeError('Generated SDK failure')
        b, _, d, m = feed['net'].shape
        return dict(net_out=feed['net']+np.float32(3), delta=feed['inp'][:, :2].copy(),
            weight=np.full((b, 2, d, m), .5, np.float32)), dict(execution_status='completed',
            outputs_finite=True, weight_domain_valid=True, all_outputs_zero=False,
            outputs_released=True, strict_FP32_gate_passed=None)

    def metrics(self):
        self.metrics_calls += 1
        return copy.deepcopy(dict(closed=self.closed,
            cleanup_errors=['Generated cleanup failure'] if self.bad_cleanup else [],
            resident_contexts=len(self._live), current_DMA_bytes=0, current_model_host_bytes=0,
            histories=self._history, sessions={k: s.metrics() for k, s in self._live.items()}))

    def close(self):
        self.closed = True
        self._history.extend(dict(model_key=k, metrics=s.metrics()) for k, s in self._live.items())
        self._live.clear()


def graph(degrees):
    ii = []; jj = []; kk = []
    for i, d in enumerate(degrees):
        for j in range(d):
            for k in range(16): ii.append(i); jj.append(j); kk.append(i*16+k)
    perm = np.random.default_rng(4).permutation(len(ii))
    topology = tuple(np.ascontiguousarray(np.asarray(v, np.int64)[perm]) for v in (ii, jj, kk))
    values = tuple(np.broadcast_to(np.arange(len(ii), dtype=np.float32)[None, :, None]+1,
        (1, len(ii), c)).copy() for c in (384, 384, 882))
    return values+topology


def owner(module, path, cache, probe=None, reader=None):
    path.mkdir()
    return module.OnlineM16Update(runtime, adapter, cache,
        motion_probe=probe or (lambda *a: pytest.fail('Full Update cannot call Torch')),
        diagnostic_allow_fp32_rejected=True, failure_dir=path, identity_reader=reader)


def test_only_reporting_block_changes_original_classes_and_all_guards():
    parent = (HERE/'v42_online_update.py').read_text()
    assert hashlib.sha256(parent.encode()).hexdigest() == new.PARENT_CALLBACK_SHA256
    block = """                    state = self.cache.metrics()
                    used = {v['model_key']: v['metrics'].get('npu_used') is True
                            for v in state.get('histories', [])}
                    used.update({key: v.get('npu_used') is True for key, v in state.get('sessions', {}).items()})"""
    assert parent.count(block) == 1
    expected = ast.parse(parent.replace(block, '                    used = npu_used_by_model(self.cache)'))
    actual = ast.parse((HERE/'v43_online_update.py').read_text())
    classes = lambda t: [ast.dump(n, include_attributes=False) for n in t.body if isinstance(n, ast.ClassDef)]
    assert classes(expected) == classes(actual)
    assert old.MODEL_KEYS == new.MODEL_KEYS and 'B1_D3_M16' in new.MODEL_KEYS
    helper = HELPER
    assert hashlib.sha256(helper.read_bytes()).hexdigest() == new.METADATA_HELPER_SHA256
    function = lambda t: next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name=='npu_used_by_model')
    assert ast.dump(function(ast.parse(helper.read_text())), include_attributes=False) == ast.dump(function(actual), include_attributes=False)


def test_history_live_precedence_true_identity_and_return_ownership():
    cache = Cache()
    cache._history = [dict(model_key='evicted', metrics=dict(npu_used=True)),
        dict(model_key='repeat', metrics=dict(npu_used=True)),
        dict(model_key='repeat', metrics=dict(npu_used=False)),
        dict(model_key='live', metrics=dict(npu_used=True))]
    cache._live = dict(live=Session(False), true=Session(True), integer=Session(1))
    state = cache.metrics()
    expected = {v['model_key']: v['metrics'].get('npu_used') is True for v in state['histories']}
    expected.update({k: v.get('npu_used') is True for k, v in state['sessions'].items()})
    counts = [v.metrics_calls for v in cache._live.values()]
    output = new.npu_used_by_model(cache)
    assert output == expected and counts == [v.metrics_calls for v in cache._live.values()]
    output['true'] = False
    assert new.npu_used_by_model(cache)['true'] is True


def test_D3_and_original_groups_match_own_outputs_without_hot_metrics_copy(tmp_path):
    arrays = graph([3, 4, 4]); saved = [a.copy() for a in arrays]; outputs = []
    for label, module in (('original', old), ('metadata', new)):
        cache = Cache(); callback = owner(module, tmp_path/label, cache)
        output = callback(*arrays)
        assert cache.calls == ['B1_D3_M16', 'B2_D4_M16']
        assert cache.metrics_calls == (1 if module is old else 0)
        assert all(v.flags.owndata for v in output.values())
        assert all(not np.shares_memory(v, a) for v in output.values() for a in arrays)
        state = callback.metrics()
        assert state['calls'][0]['NPU_used'] is True  # Simulated flag; no SDK execution.
        assert state['full_completed'] == 1 and state['candidate_accepted'] is False
        assert state['calls'][0]['strict_FP32_gate_passed'] is None
        before = cache.metrics_calls; callback.close(); callback.close()
        assert cache.metrics_calls == before+1 and cache.closed
        outputs.append(output)
    assert all(np.array_equal(outputs[0][k], outputs[1][k]) for k in outputs[0])
    assert all(np.array_equal(a, b) for a, b in zip(arrays, saved))


@pytest.mark.parametrize('degrees', ([3, 3], [1], [8]))
def test_unknown_shape_still_saves_complete_inputs_and_stops(tmp_path, degrees):
    cache = Cache(); callback = owner(new, tmp_path/'failure', cache); arrays = graph(degrees)
    with pytest.raises(new.OnlineUpdateError, match='Unknown complete online shape'): callback(*arrays)
    assert not cache.calls and callback.metrics()['full_completed'] == 0
    with np.load(tmp_path/'failure/update-000000-failed-inputs.npz', allow_pickle=False) as z:
        assert all(np.array_equal(z[k], a) for k, a in zip(('net', 'inp', 'corr', 'ii', 'jj', 'kk'), arrays))
    with pytest.raises(RuntimeError, match='closed or failed'): callback(*arrays)
    callback.close()


@pytest.mark.parametrize('failure', ('late_group', 'stale_owner'))
def test_failed_group_or_identity_returns_no_partial_fullgraph(tmp_path, failure):
    cache = Cache(fail_after=2 if failure=='late_group' else None)
    token = adapter.SnapshotIdentity(1, 2, 3, 4)
    reader = lambda: token if not cache.calls else adapter.SnapshotIdentity(2, 2, 3, 4)
    callback = owner(new, tmp_path/'failure', cache, reader=reader if failure=='stale_owner' else None)
    with pytest.raises(new.OnlineUpdateError): callback(*graph([3, 4]))
    state = callback.metrics()
    assert state['full_completed'] == 0 and not state['calls'][0]['fullgraph_returned']
    assert len(cache.calls) == (2 if failure=='late_group' else 1)
    callback.close()


def test_explicit_D1_startup_route_and_cleanup_checks_unchanged(tmp_path):
    calls = []
    def probe(net, inp, corr, ii, jj, kk):
        calls.append(1)
        return dict(net=net, delta=np.ones((1, 16, 2), np.float32), weight=np.full((1, 16, 2), .5, np.float32))
    cache = Cache(); callback = owner(new, tmp_path/'probe', cache, probe=probe)
    arrays = list(graph([1])); arrays[4].fill(1); callback.route = 'motion_probe'
    callback(*arrays)
    assert calls == [1] and not cache.calls and callback.metrics()['motion_probe_completed'] == 1
    cache.bad_cleanup = True
    with pytest.raises(RuntimeError, match='cleanup failed'): callback.close()
    assert callback.metrics()['closed'] and callback.metrics()['failed']


def factory_arguments():
    return dict(v42_factory=HERE/'v42_online_factory.py', v42_factory_sha256=factory.PARENT_FACTORY_SHA256,
        v42_callback=HERE/'v42_online_update.py', v42_callback_sha256=factory.PARENT_CALLBACK_SHA256,
        metadata_helper=HELPER, metadata_helper_sha256=factory.METADATA_HELPER_SHA256,
        callback_path=HERE/'v43_online_update.py', callback_sha256=factory.CALLBACK_SHA256,
        contracts=HERE/'v42_online_contracts.py', contracts_sha256=factory.CONTRACTS_SHA256,
        diagnostic_allow_fp32_rejected=True)


def test_factory_delegates_unmodified_options_and_adds_only_source_provenance(monkeypatch):
    seen = []; callback = SimpleNamespace(source_pins={}, catalogue_provenance={'candidate_accepted': False})
    monkeypatch.setattr(factory, '_load_parent', lambda path: SimpleNamespace(
        open_board_online_from_index=lambda **kwargs: (seen.append(kwargs), callback)[1]))
    arguments = factory_arguments(); sentinel = object(); arguments['compile_index'] = sentinel
    result = factory.open_board_online_from_index(**arguments)
    assert result is callback and seen[0]['compile_index'] is sentinel
    assert len(seen[0]) == len(arguments)-6
    assert result.catalogue_provenance['metadata_only_reporting'] is True
    assert result.catalogue_provenance['numerical_execution_changed'] is False
    assert result.catalogue_provenance['native_input_transport_added'] is False
    assert str((HERE/'v42_online_factory.py').resolve()) in result.source_pins


def test_factory_requires_diagnostic_before_any_source_loading(monkeypatch):
    monkeypatch.setattr(factory, '_load_parent', lambda *a: pytest.fail('Cannot load without diagnostic'))
    arguments = factory_arguments(); arguments['diagnostic_allow_fp32_rejected'] = False
    with pytest.raises(ValueError, match='Explicit FP32-rejected'): factory.open_board_online_from_index(**arguments)


def test_factory_mutated_source_after_creation_closes_callback(monkeypatch):
    checks = []; closed = []
    def recheck(pins):
        checks.append(1)
        if len(checks) == 2: raise ValueError('Simulated changed source')
    callback = SimpleNamespace(close=lambda: closed.append(1))
    monkeypatch.setattr(factory, '_recheck', recheck)
    monkeypatch.setattr(factory, '_load_parent', lambda path: SimpleNamespace(open_board_online_from_index=lambda **kw: callback))
    with pytest.raises(ValueError, match='Simulated changed source'): factory.open_board_online_from_index(**factory_arguments())
    assert closed == [1]


def test_driver_trajectory_helpers_full_frame_loop_EOF_and_cleanup_AST_unchanged():
    parent = ast.parse((HERE/'run-v42-online-npu-board.py').read_text())
    child = ast.parse((HERE/'run-v43-online-npu-board.py').read_text())
    helpers = lambda t: {n.name: ast.dump(n, include_attributes=False) for n in t.body
        if isinstance(n, ast.FunctionDef) and n.name not in ('main', 'parse_args')}
    assert helpers(parent) == helpers(child)
    main = lambda t: next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name=='main')
    loop = lambda t: next(n for n in ast.walk(t) if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple)
        and [v.id for v in n.target.elts] == ['index', 'item'])
    assert ast.dump(loop(main(parent)), include_attributes=False) == ast.dump(loop(main(child)), include_attributes=False)
    cleanup = lambda t: [ast.dump(n, include_attributes=False) for n in next(n for n in t.body if isinstance(n, ast.Try)).finalbody]
    assert cleanup(main(parent)) == cleanup(main(child))
    EOF = lambda t: [ast.dump(n, include_attributes=False) for n in ast.walk(t) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute) and n.func.attr=='terminate']
    assert EOF(parent) == EOF(child) and len(EOF(child)) == 1


def test_driver_cli_and_source_pins_are_exact():
    names = ('parent-driver', 'policy', 'torch-encoder', 'source-dir', 'checkpoint', 'geometry',
        'stream-module', 'encoder-module', 'frame-cache-helper', 'library', 'build-receipt',
        'encoder-capture-dir', 'dataset-dir', 'output-dir', 'cached-cpu-driver', 'online-factory',
        'online-config', 'v41-online-driver', 'undistort-helper')
    argv = [v for key in names for v in ('--'+key, 'fake')]+[
        '--online-factory-sha256', 'x', '--online-config-sha256', 'y']
    with pytest.raises(SystemExit): driver.parse_args(argv)
    row = driver.parse_args(argv+['--v42-online-driver', 'old', '--diagnostic-allow-fp32-rejected'])
    assert row.v42_online_driver == Path('old') and row.npu_encoder is None
    sha = lambda name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    assert driver.ONLINE_FACTORY == sha('v43_online_factory.py')
    assert driver.ONLINE_CALLBACK == sha('v43_online_update.py')
    assert driver.ONLINE_CONTRACTS == sha('v42_online_contracts.py')
    assert driver.V42_ONLINE_DRIVER == sha('run-v42-online-npu-board.py')
