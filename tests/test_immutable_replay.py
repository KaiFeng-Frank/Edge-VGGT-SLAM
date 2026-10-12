"""Replay checks with existing service doubles; no Torch or SDK run."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT/'scripts/replay_original_vggt.py'
SESSION_SOURCE = ROOT/'vggt_slam/immutable_execution_session.py'
spec = importlib.util.spec_from_file_location(
    '_existing_pageout_service_doubles', ROOT/'tests/test_checkpoint_residency_replay.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def load_replay():
    spec = importlib.util.spec_from_file_location('_immutable_replay_candidate', CANDIDATE)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def arguments(root, *, enabled=True, omit=(), no_cache=False):
    args = fixture_arguments(root, enabled=False, omit=omit, no_cache=no_cache)
    if enabled:
        args.append('--immutable-execution')
    return args


fixture_arguments = fixture.arguments


def generated_run(root, *, enabled=True, failure=None):
    events, handles = [], []
    session_module = ModuleType('vggt_slam.immutable_execution_session')

    class Session:
        closed = False
        def close(self):
            # Final validation/restoration precedes removal of these bindings.
            assert all(not backend.closed for backend in handles)
            events.append('fullvalidate_and_restore_session')
            self.closed = True
            if failure == 'close':
                raise RuntimeError('generated final validation failure')
        def metrics(self):
            if failure == 'metrics':
                raise RuntimeError('generated session metrics failure')
            return dict(contract='exclusive_immutable_model', closed=self.closed,
                        final_validation_passed=failure != 'close', generated_service_double=True)

    session = Session()
    def install(model, cache, retained, dino, *, contract):
        assert contract == 'exclusive_immutable_model'
        assert (cache.name, retained.name, dino.name) == ('cache', 'aggregator', 'DINO')
        assert hasattr(model, 'before') and hasattr(model, 'after')
        assert all(not backend.closed for backend in (cache, retained, dino))
        handles[:] = (cache, retained, dino)
        events.append('install_session_after_cache')
        if failure == 'install':
            raise RuntimeError('generated session installation failure')
        if failure == 'predict':
            def rejected_prediction(*_args):
                raise RuntimeError('generated prediction failure')
            model.before = rejected_prediction
        return session
    session_module.install_immutable_session = install
    original_read = Path.read_bytes
    expected_source = ROOT/'vggt_slam/immutable_execution_session.py'
    def read_bytes(path):
        return original_read(SESSION_SOURCE) if path == expected_source else original_read(path)
    with patch.object(fixture, 'load_replay', load_replay), \
            patch.object(fixture, 'arguments', lambda path, **_kwargs: arguments(path, enabled=enabled)), \
            patch.object(fixture, 'SOURCES', fixture.SOURCES+('immutable_execution_session.py',)), \
            patch.object(Path, 'read_bytes', read_bytes), \
            patch.dict(sys.modules, {'vggt_slam.immutable_execution_session': session_module}):
        code, report, service_events, live = fixture.generated_run(root, enabled=False)
    return code, report, service_events, live, events, session


@pytest.mark.parametrize('omit,no_cache', [
    (('--rknn-retained-aggregator',), False), (('--rknn-paired-aggregator',), False),
    (('--rknn-manifest', '--rknn-manifest-sha256'), False),
    (('--half-io-library', '--half-io-library-sha256'), False),
    (('--rknn-grouped-dino',), False),
    (('--rknn-dino-manifest', '--rknn-dino-manifest-sha256'), False),
    (('--rknn-camera-head',), False), (('--rknn-camera-split4',), False),
    (('--rknn-depth-manifest', '--rknn-depth-manifest-sha256'), False), ((), True),
])
def test_incomplete_contract_rejects_before_any_board_or_loading(tmp_path, omit, no_cache):
    replay = load_replay()
    with patch.object(sys, 'argv', arguments(tmp_path, omit=omit, no_cache=no_cache)), \
            patch.object(Path, 'is_file', side_effect=AssertionError('must reject before board read')), \
            patch.object(replay.os, 'sched_setaffinity', side_effect=AssertionError('must reject before setup')), \
            contextlib.redirect_stderr(io.StringIO()) as stderr:
        with pytest.raises(SystemExit) as exc:
            replay.main()
    assert exc.value.code == 2 and '--immutable-execution requires' in stderr.getvalue()
    assert not (tmp_path/'output').exists()


def test_owned_session_installation_reporting_and_restore_before_cache(tmp_path):
    code, report, events, live, session_events, session = generated_run(tmp_path)
    assert code == 0 and not live and session.closed
    assert session_events == ['install_session_after_cache', 'fullvalidate_and_restore_session']
    assert [x for x in events if x.startswith('install_')] == [
        'install_aggregator', 'install_camera', 'install_DINO', 'install_depth', 'install_cache']
    assert [x for x in events if x.startswith('close_')] == [
        'close_replay', 'close_cache', 'close_depth', 'close_DINO', 'close_camera', 'close_aggregator']
    assert report['immutable_execution'] and report['immutable_execution_contract'] == 'exclusive_immutable_model'
    assert report['arbitrary_model_mutation_detection_preserved'] is False
    assert report['immutable_session']['closed'] and report['immutable_session']['final_validation_passed']
    assert report['immutable_execution_source_sha256'] == hashlib.sha256(SESSION_SOURCE.read_bytes()).hexdigest()
    assert report['fresh_pose_count'] == 2 and events.count('predict') == 2
    assert not report['candidate_accepted'] and not report['real_time_validated']


@pytest.mark.parametrize('failure', ['install', 'predict', 'close', 'metrics'])
def test_session_errors_still_close_every_backend_and_restore_affinity(tmp_path, failure):
    code, report, events, live, session_events, session = generated_run(tmp_path, failure=failure)
    assert code == 1 and report['status'] == 'failed' and not live
    expected = ['close_cache', 'close_depth', 'close_DINO', 'close_camera', 'close_aggregator']
    if failure != 'install':
        expected.insert(0, 'close_replay')
        assert session.closed and session_events[-1] == 'fullvalidate_and_restore_session'
    else:
        assert 'start_replay' not in events and 'predict' not in events
    assert [x for x in events if x.startswith('close_')] == expected
    assert events[-1] == 'affinity_0,1,2,3,4,5,6,7'
    if failure in ('close', 'metrics'):
        assert report['cleanup_errors']


def test_disabled_default_preserves_service_flow_and_never_installs_session(tmp_path):
    code, report, events, live, session_events, session = generated_run(tmp_path, enabled=False)
    assert code == 0 and not live and not session_events and not session.closed
    assert report['immutable_execution'] is False
    assert 'immutable_session' not in report and 'immutable_execution_contract' not in report
    assert 'immutable_execution_session.py' not in report['runtime_source_sha256']
    assert report['fresh_pose_count'] == 2 and events.count('predict') == 2
