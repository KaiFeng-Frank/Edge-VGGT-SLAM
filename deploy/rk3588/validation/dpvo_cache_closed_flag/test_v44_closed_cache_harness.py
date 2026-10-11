"""Generated API/source checks only; no SDK/model/real-array forward."""
import ast
import importlib.util
import os
from pathlib import Path
import types
import uuid

import pytest


def load(path):
    spec = importlib.util.spec_from_file_location('_v44_harness_test_' + uuid.uuid4().hex, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


HERE = Path(__file__).resolve().parent
HELPER = load(HERE / 'run-v44-native-closed-cache-ab-board.py')


def required_args(root):
    args = []
    for name in ('native-harness', 'parent-harness', 'closed-flag-helper', 'factory',
                 'config', 'transport', 'packer', 'packing-library', 'query',
                 'build-receipt', 'conversion-receipt', 'single-ab-receipt'):
        args.extend(('--' + name, str(root / name), '--' + name + '-sha256', '0' * 64))
    return args + ['--inputs-dir', str(root / 'inputs'), '--output-dir', str(root / 'new-output')]


@pytest.mark.parametrize('missing', ('fp32-rejected', 'native-fp16', 'closed-flag'))
def test_all_flags_checked_before_import_or_write(tmp_path, monkeypatch, missing):
    def forbidden(*_args):
        raise AssertionError('Parent import must not happen before diagnostic admission')
    monkeypatch.setattr(HELPER, 'load_native_harness', forbidden)
    args = required_args(tmp_path)
    args.extend('--diagnostic-allow-' + v for v in ('fp32-rejected', 'native-fp16', 'closed-flag') if v != missing)
    with pytest.raises(ValueError, match='All three'):
        HELPER.main(args)
    assert not (tmp_path / 'new-output').exists()


def test_board_guard_before_output_and_sdk(tmp_path, monkeypatch):
    def reject():
        raise ValueError('generated host rejection')
    H = types.SimpleNamespace(board_guard=reject)
    G = types.SimpleNamespace(load_parent=lambda *_args: H)
    monkeypatch.setattr(HELPER, 'load_native_harness', lambda *_args: G)
    args = required_args(tmp_path) + ['--diagnostic-allow-' + v for v in ('fp32-rejected', 'native-fp16', 'closed-flag')]
    with pytest.raises(ValueError, match='generated host rejection'):
        HELPER.main(args)
    assert not (tmp_path / 'new-output').exists()


def fake_pair(root, *, wrong_session=False):
    parent = root / 'parent.py'
    parent.write_text('# generated parent marker\n')
    helper = root / 'closed.py'
    helper.write_text('# generated cache helper marker\n')
    owner_file = root / 'dependency'
    owner_file.write_text('generated payload\n')
    pins = {str(helper): HELPER.sha(helper)}
    pin_calls, replacements = [], []

    def pin(path, digest, all_pins):
        path = str(Path(path).resolve())
        if path == str(parent):
            assert digest == HELPER.PARENT_RUNTIME_SHA256
        elif HELPER.sha(path) != digest:
            raise ValueError('generated pin mismatch')
        if path in all_pins and all_pins[path] != digest:
            raise ValueError('generated pin conflict')
        pin_calls.append(path)
        all_pins[path] = digest
        return Path(path)

    class OriginalCache:
        pass
    class NewCache(OriginalCache):
        pass
    class NativeSession:
        pass
    native = types.SimpleNamespace(GroupedSession=NativeSession, BoundedContextCache=OriginalCache,
        execute_full_graph=object(), pack_inputs=object(), source_pins={str(parent): HELPER.PARENT_RUNTIME_SHA256})
    proof = dict(changed_expression_count=1, numerical_execution_changed=False, candidate_accepted=False)

    def bind(runtime, *, parent_runtime_path, diagnostic_allow_closed_flag):
        assert runtime is native and parent_runtime_path == parent and diagnostic_allow_closed_flag is True
        result = types.SimpleNamespace(**vars(runtime))
        result.BoundedContextCache = NewCache
        if wrong_session:
            result.GroupedSession = object()
        result.source_pins = dict(runtime.source_pins, **{str(helper): HELPER.sha(helper)})
        result.closed_flag_source_proof = proof
        return result

    def replace(owner, runtime, config, all_pins, directory):
        assert owner.cache == 'empty' and config['parent_runtime'] == str(parent)
        assert all_pins[str(helper)] == HELPER.sha(helper)
        assert all_pins[str(owner_file)] == HELPER.sha(owner_file)
        assert all_pins[str(parent)] == HELPER.PARENT_RUNTIME_SHA256
        sources = dict(owner.source_pins)
        sources.update(runtime.source_pins)
        sources.update(all_pins)
        replacements.append((runtime, dict(sources), directory))
        owner.runtime, owner.cache, owner.source_pins = runtime, runtime.BoundedContextCache(), sources

    owners = {method: types.SimpleNamespace(cache='empty', source_pins={str(owner_file): HELPER.sha(owner_file)},
        catalogue_provenance=dict(candidate_accepted=False)) for method in ('A', 'B')}
    # Give independent empty markers: bind_pair explicitly rejects aliasing.
    owners['A'].cache, owners['B'].cache = ['empty'], ['empty']
    def replace_markers(*args):
        args[0].cache = 'empty'
        return replace(*args)
    return owners, native, types.SimpleNamespace(bind_runtime=bind), types.SimpleNamespace(replace_empty_cache=replace_markers), \
        types.SimpleNamespace(pin=pin), dict(parent_runtime=str(parent)), pins, \
        dict(A=root / 'A', B=root / 'B'), replacements, pin_calls, proof


def test_only_cache_changed_and_pin_payload_identical(tmp_path):
    owners, native, closed, G, H, config, pins, dirs, replacements, calls, proof = fake_pair(tmp_path)
    actual = HELPER.bind_pair(owners, native, closed, G, H, config, pins, dirs)
    assert len(replacements) == 2
    assert replacements[0][0] is native
    assert replacements[1][0].GroupedSession is native.GroupedSession
    assert replacements[1][0].BoundedContextCache is not native.BoundedContextCache
    assert replacements[0][1] == replacements[1][1] == owners['A'].source_pins == owners['B'].source_pins
    assert owners['A'].catalogue_provenance['cache_closed_flag_only'] is False
    assert owners['B'].catalogue_provenance['cache_closed_flag_only'] is True
    assert actual == proof and actual is not proof
    actual['changed_expression_count'] = 8
    assert proof['changed_expression_count'] == 1


def test_changed_session_rejected_before_cache_replacement(tmp_path):
    owners, native, closed, G, H, config, pins, dirs, replacements, *_rest = fake_pair(tmp_path, wrong_session=True)
    with pytest.raises(ValueError, match='Only the cache class'):
        HELPER.bind_pair(owners, native, closed, G, H, config, pins, dirs)
    assert replacements == []


def test_identical_owner_rejected(tmp_path):
    owners, native, closed, G, H, config, pins, dirs, replacements, *_rest = fake_pair(tmp_path)
    owners['B'] = owners['A']
    with pytest.raises(ValueError, match='independent'):
        HELPER.bind_pair(owners, native, closed, G, H, config, pins, dirs)
    assert replacements == []


def test_ancestral_schedule_and_helpers_pinned():
    adjacent = HERE.parent / 'dpvo_native_input/run-v43-native-full-ab-board.py'
    repo = HERE.parent / 'deploy/rk3588/validation/dpvo_native_input/run-v43-native-full-ab-board.py'
    default = adjacent if adjacent.exists() else repo
    path = Path(os.environ.get('V44_NATIVE_HARNESS', default))
    G = HELPER.load_native_harness(path, HELPER.NATIVE_HARNESS_SHA256)
    parent_default = path.parent.parent / 'dpvo_update_performance/run-v42-callback-ab-board.py'
    H = G.load_parent(Path(os.environ.get('V44_PARENT_HARNESS', parent_default)), G.PARENT_HARNESS_SHA256)
    schedule = H.schedule()
    assert len(schedule) == 24
    for i, observation in enumerate((34, 127, 191)):
        rows = schedule[i * 8:(i + 1) * 8]
        assert [r['method'] for r in rows] == list(('A', 'B', 'B', 'A') * 2)
        assert {r['observation_id'] for r in rows} == {observation}
        assert sum(r['excluded_first_state_method_call'] for r in rows) == 2
    tree = ast.parse((HERE / 'run-v44-native-closed-cache-ab-board.py').read_text())
    calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert {'H.run_calls', 'H.closed_lifecycle', 'G.replace_empty_cache', 'transport.load_runtime'} <= calls
    assert not {'torch.load', 'onnxruntime.InferenceSession', 'ctypes.CDLL', 'rknn_run'} & calls
    assert 'run-v44-native-closed-cache-ab-board.py' not in str(path)
