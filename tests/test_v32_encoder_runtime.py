"""Tiny fake-SDK tests; no source images, model weights or actual inference."""
import ctypes as ct
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def source(name, environment=None):
    if environment and os.environ.get(environment):
        path = Path(os.environ[environment])
        if not path.is_file():
            pytest.fail(environment + ' does not name an existing file')
        return path
    here = Path(__file__).resolve().parent
    for parent in (here, *here.parents):
        for path in (parent / name, parent / 'deploy/rk3588/validation' / name,
                     parent / 'deploy/rk3588/validation/dpvo_encoder_opt' / name):
            if path.is_file():
                return path
    pytest.fail('Missing public helper: ' + name)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runner():
    return load(source('run-v32-dpvo-encoder-board.py'), '_v32_runtime_test')


@pytest.fixture
def parent(monkeypatch):
    module = load(source('run-v26-dpvo-rknn-board.py', 'DPVO_ENCODER_PARENT_SOURCE'),
                  '_v32_parent_test')
    monkeypatch.setattr(module, 'INPUT_SHAPE', (1, 3, 2, 4))
    monkeypatch.setattr(module, 'OUTPUTS', (('fmap', (1, 2, 1, 2)),
                                         ('imap', (1, 3, 1, 2))))
    return module


def test_prescribed_normal_and_profile_schedules(runner):
    normal = runner.schedule(False)
    assert [(frame, method) for frame, method, _ in normal] == [
        (0, 'original'), (0, 'spatial'), (0, 'spatial'), (0, 'original'),
        (0, 'original'), (0, 'spatial'), (1, 'spatial'), (1, 'original'),
        (2, 'original'), (2, 'spatial')]
    assert runner.schedule(True) == [(0, 'spatial', 'profile'),
                                    (1, 'spatial', 'profile'),
                                    (2, 'spatial', 'profile')]
    calls = [dict(method=method, timing_role=role, forward_seconds=i + 1.)
             for i, (_, method, role) in enumerate(normal)]
    summary = runner.timing_summary(calls)
    assert summary['original']['calls'] == summary['spatial']['calls'] == 5
    assert summary['original']['warm_seconds'] == [4., 5.]
    assert summary['spatial']['warm_seconds'] == [3., 6.]
    assert summary['original']['warm_mean_seconds'] == 4.5
    assert summary['spatial']['first_seconds'] == 2.


@pytest.mark.parametrize('failure', ['init', 'core_mask', 'version', 'attributes', 'memory'])
def test_context_initialization_failures_destroy_once(runner, failure):
    destroyed = []
    queries = []

    def check(code, name):
        if code:
            raise RuntimeError(name)

    def init(pointer, *args):
        pointer._obj.value = 71
        return -1 if failure == 'init' else 0

    def query(api, context, command, output):
        queries.append(command)
        if command == 5:
            return SimpleNamespace(api_version=b'2.2.0' if failure == 'version' else b'2.3.2',
                                   drv_version=b'fake')
        if command == 0:
            return SimpleNamespace(n_input=1, n_output=2)
        if command in (1, 2):
            return output
        memory = SimpleNamespace(total_dma_allocated_size=2**31 if failure == 'memory' else 32)
        memory._fields_ = [('total_dma_allocated_size', ct.c_uint64)]
        return memory

    abi = SimpleNamespace(CONTEXT=ct.c_uint64, check_call=check, query=query,
                          SDKVersion=lambda: None, InputOutputNum=lambda: None,
                          TensorAttr=lambda index: SimpleNamespace(index=index, fmt=0),
                          MemSize=lambda: None)
    api = SimpleNamespace(rknn_init=init,
                          rknn_set_core_mask=lambda *args: -1 if failure == 'core_mask' else 0,
                          rknn_destroy=lambda context: destroyed.append(context.value) or 0)

    def attribute(*args, **kwargs):
        if failure == 'attributes':
            raise ValueError('bad output/input attributes')
        return {'synthetic': True}

    base = SimpleNamespace(INPUT_SHAPE=(1, 3, 2, 4),
                           OUTPUTS=(('fmap', (1, 2, 1, 2)), ('imap', (1, 3, 1, 2))),
                           validate_attribute=attribute)
    record, cleanup = {}, []
    with pytest.raises((ValueError, RuntimeError)):
        runner.Context(base, abi, api, 'synthetic.rknn', 0, cleanup, record)
    assert destroyed == [71]
    assert record['context_destroyed'] is True
    assert cleanup == []


def test_frozen_delegate_copies_before_release_and_accounts_IO(parent):
    abi = load(source('run-v12-rknn-block-board.py', 'RKNN_ABI_HELPER'), '_v32_ABI_test')
    events, buffers, activity, cleanup = [], [], {}, []
    images = np.arange(24, dtype=np.float32).reshape(1, 3, 2, 4)

    def set_inputs(context, count, inputs):
        events.append('set')
        assert inputs[0].fmt == 1 and inputs[0].pass_through == 0
        packed = np.ctypeslib.as_array(ct.cast(inputs[0].buf, ct.POINTER(ct.c_float)),
                                     shape=(24,)).copy().reshape(1, 2, 4, 3)
        np.testing.assert_array_equal(packed, images.transpose(0, 2, 3, 1))
        return 0

    def run(context, extension):
        events.append('run')
        extension._obj.frame_id = 9
        return 0

    def get(context, count, outputs, received):
        events.append('get')
        received._obj.frame_id = 9
        for i, (_, shape) in enumerate(parent.OUTPUTS):
            array = np.full(shape, i + 1., dtype=np.float32)
            buffers.append(array)
            outputs[i].buf = array.ctypes.data
            outputs[i].size = array.nbytes
        return 0

    def release(*args):
        events.append('release')
        for array in buffers:
            array.fill(-999.)
        return 0

    api = SimpleNamespace(rknn_inputs_set=set_inputs, rknn_run=run,
                          rknn_outputs_get=get, rknn_outputs_release=release)
    actual, timing = parent.execute_call(api, abi, abi.CONTEXT(17), images, 1,
                                         60000, cleanup, activity)
    assert events == ['set', 'run', 'get', 'release']
    assert activity['sdk_run_api_calls'] == 1 and cleanup == []
    assert timing['output_buffers_released'] and timing['input_api_format'] == 1
    assert timing['sdk_io_seconds'] >= sum(timing[key] for key in (
        'input_packing_seconds', 'input_set_seconds', 'run_api_seconds',
        'outputs_get_seconds', 'output_copy_seconds', 'outputs_release_seconds'))
    for i, (name, _) in enumerate(parent.OUTPUTS):
        assert (actual[name] == i + 1).all()
        assert not np.shares_memory(actual[name], buffers[i])
    assert parent.canonical_outputs(actual)['fmap'].shape == (1, 1, 2, 1, 2)


def fake_main(monkeypatch, tmp_path, runner, parent, bad_output=None):
    """Replace only external services; run the real entrypoint/finally/save code."""
    root = tmp_path / 'experiments'
    root.mkdir()
    monkeypatch.setattr(runner, 'BASE', root)
    monkeypatch.setattr(runner, 'board_guard', lambda: None)
    monkeypatch.setattr(runner.os, 'sched_getaffinity', lambda _: {4, 5, 6, 7})
    monkeypatch.setattr(runner.os, 'sched_setaffinity', lambda *args: None)
    files = {}
    for name in ('parent-source', 'compiler-source', 'abi-helper', 'header', 'library'):
        files[name] = root / (name + '.py')
        files[name].write_text('tiny ' + name)
    directories = {}
    for name in ('capture', 'export', 'ort', 'compile', 'candidate-compile', 'rewrite', 'failed-ort'):
        directories[name] = root / name
        directories[name].mkdir()
    original = directories['compile'] / 'original.rknn'
    candidate = directories['candidate-compile'] / 'candidate.rknn'
    original.write_bytes(b'original')
    candidate.write_bytes(b'spatial')
    graph = directories['rewrite'] / 'encoders.onnx'
    graph.write_bytes(b'tiny graph')
    ancestor = {'historical_FP32_gate_passed': False}
    documents = {
        'rewrite': {'historical': True}, 'failed-ort': {'historical': True},
        'candidate-compile': {'input': {'sha256': runner.sha(graph)},
                              'configuration': {'target_platform': 'rk3588',
                                                'float_dtype': 'float16',
                                                'optimization_level': 3,
                                                'do_quantization': False},
                              'output': {'path': 'candidate.rknn', 'sha256': runner.sha(candidate),
                                         'size': candidate.stat().st_size}}}
    for name, value in documents.items():
        (directories[name] / ('rewrite.json' if name == 'rewrite' else
                             'ort.json' if name == 'failed-ort' else 'compile.json')).write_text(json.dumps(value))
    monkeypatch.setattr(runner, 'PARENT_SHA', runner.sha(files['parent-source']))
    monkeypatch.setattr(runner, 'REWRITE_SHA', runner.sha(directories['rewrite'] / 'rewrite.json'))
    monkeypatch.setattr(runner, 'FAILED_ORT_SHA', runner.sha(directories['failed-ort'] / 'ort.json'))
    monkeypatch.setattr(runner, 'CANDIDATE_ONNX_SHA', runner.sha(graph))
    monkeypatch.setattr(parent, 'HEADER_SHA', runner.sha(files['header']))
    monkeypatch.setattr(parent, 'LIBRARY_SHA', runner.sha(files['library']))
    rows = []
    for frame in range(3):
        image = np.full(parent.INPUT_SHAPE, frame + 1., dtype=np.float32)
        reference = {name: np.full((shape[0], 1) + shape[1:], frame + 1., dtype=np.float32)
                     for name, shape in parent.OUTPUTS}
        rows.append(dict(images=image, seed=dict(frame_id=str(100. + frame), array=parent.metadata(image)),
                         reference=reference, ort_reference=reference))
    captured = dict(preprocessing={'synthetic': True}, state_before={}, loader={})
    monkeypatch.setattr(parent, 'load_inputs', lambda *args: (original, captured, {}, {}, rows))
    abi = SimpleNamespace(require_board=lambda: None, process_memory=lambda: {}, bind_api=lambda _: None)
    monkeypatch.setattr(parent, 'load_abi', lambda *args: abi)

    def lineage(rewrite, failed):
        assert rewrite['historical'] and failed['historical']
        return ancestor

    compiler = SimpleNamespace(validate_lineage=lineage,
                               validate_compilation=lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, 'module', lambda path, name: parent if 'original' in name else compiler)
    monkeypatch.setattr(runner.ct, 'CDLL', lambda _: None)
    executed, closed = [], []

    class FakeContext:
        def __init__(self, base, abi, api, model, flags, cleanup, record):
            self.method = 'original' if Path(model) == original else 'spatial'
            self.record = record

        def execute(self, image, activity):
            assert any(row['images'] is image for row in rows)  # No hidden preprocessing.
            row = next(row for row in rows if row['images'] is image)
            executed.append((self.method, row['seed']['frame_id'], parent.metadata(image)))
            activity['sdk_run_api_calls'] += 1
            value = float(image.flat[0]) + .1  # Finite numerical rejection is diagnostic, not fatal.
            outputs = {name: np.full(shape, value, dtype=np.float32) for name, shape in parent.OUTPUTS}
            if bad_output == 'nonfinite':
                outputs['fmap'].flat[0] = np.nan
            elif bad_output == 'zero':
                outputs['imap'].fill(0.)
            return outputs, {'sdk_io_seconds': .01, 'output_buffers_released': True}

        def close(self):
            closed.append(self.method)
            self.record['context_destroyed'] = True

    monkeypatch.setattr(runner, 'Context', FakeContext)
    args = []
    for name, path in files.items():
        args.extend(['--' + name, str(path)])
    for name, directory in directories.items():
        args.extend(['--' + name + '-dir', str(directory)])
    for name in ('capture', 'export', 'ort', 'compile', 'candidate-compile'):
        args.extend(['--' + name + '-sha256', runner.sha(directories['candidate-compile'] / 'compile.json')
                     if name == 'candidate-compile' else 'a' * 64])
    args.extend(['--compiler-source-sha256', runner.sha(files['compiler-source']),
                 '--output-dir', str(root / 'out')])
    result = runner.main(args)
    return result, root / 'out', executed, closed, rows


def test_main_uses_same_original_input_and_retains_numerical_rejection(monkeypatch, tmp_path, runner, parent):
    code, out, executed, closed, rows = fake_main(monkeypatch, tmp_path, runner, parent)
    saved = json.loads((out / 'run.json').read_text())
    assert code == 0 and saved['candidate_accepted'] is False
    assert saved['historical_FP32_gate_passed'] is False
    assert saved['actual_strict_FP32_gate_passed'] is False
    assert saved['NPU_used'] is True and saved['sdk_run_api_calls'] == 10
    assert [(method, identity) for method, identity, _ in executed] == [
        (method, rows[frame]['seed']['frame_id']) for frame, method, _ in runner.schedule(False)]
    assert all(info == next(row['seed']['array'] for row in rows if row['seed']['frame_id'] == identity)
               for _, identity, info in executed)
    assert closed == ['original', 'spatial']
    assert saved['sources_and_inputs_unchanged'] and saved['affinity_restored']
    assert saved['cleanup_errors'] == []


@pytest.mark.parametrize('bad_output', ['nonfinite', 'zero'])
def test_nonfinite_and_zero_are_saved_then_fail_closed(monkeypatch, tmp_path, runner, parent, bad_output):
    code, out, executed, closed, _ = fake_main(monkeypatch, tmp_path, runner, parent, bad_output)
    saved = json.loads((out / 'run.json').read_text())
    assert code == 1 and saved['status'] == 'failed'
    assert len(executed) == len(saved['calls']) == 1
    assert saved['calls'][0]['finite_nonzero'] is False
    assert saved['calls'][0]['artifact']['sha256'] == runner.sha(out / saved['calls'][0]['artifact']['file'])
    with np.load(out / saved['calls'][0]['artifact']['file'], allow_pickle=False) as actual:
        if bad_output == 'nonfinite':
            assert not np.isfinite(actual['fmap']).all()
        else:
            assert not np.any(actual['imap'])
    assert saved['candidate_accepted'] is False and closed == ['original', 'spatial']
    assert saved['cleanup_errors'] == [] and saved['sources_and_inputs_unchanged']


def test_host_guard_precedes_any_model_or_output_access(monkeypatch, tmp_path, runner):
    monkeypatch.setattr(runner.platform, 'machine', lambda: 'x86_64')
    with pytest.raises(ValueError, match='board-only'):
        runner.board_guard()
    assert list(tmp_path.iterdir()) == []
