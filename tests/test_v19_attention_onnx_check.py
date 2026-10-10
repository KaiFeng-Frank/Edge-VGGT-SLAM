"""Synthetic operands and fake CPU ORT only; no real captures/models/ORT."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name, label):
    path = ROOT/'deploy/rk3588/validation'/name
    spec = importlib.util.spec_from_file_location(label, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def checker():
    return load('check-v19-attention-onnx-board.py', 'v19_attention_check_fake')


@pytest.fixture
def saved(checker, monkeypatch, tmp_path):
    pytest.importorskip('onnx')
    exporter = load('export-v19-attention.py', 'v19_attention_export_fake')
    shape = (2, 5, 7, 64)
    rng = np.random.default_rng(93)
    arrays = {name: rng.standard_normal(shape).astype(np.float32) for name in exporter.ARRAY_KEYS}
    capture = dict(block='synthetic', flattening_order='synthetic C-order', attention_semantics={'scale': 0.125},
                   checkpoint={}, selected_pair={}, profile_sha256='a'*64,
                   original_block_parameter_metadata={}, original_block_array_metadata={}, source_pins={})
    for key in ('unhooked_vs_frozen_reference', 'captured_vs_unhooked', 'captured_vs_frozen_reference'):
        capture[key] = dict(array_equal=True)
    monkeypatch.setattr(exporter, 'load_operands', lambda *_: (capture, arrays, {}))
    export_dir = tmp_path/'export'
    export_dir.mkdir()
    exported = exporter.export(tmp_path/'fake-capture', 'b'*64, export_dir)
    assert exported['status'] == 'exported_and_structurally_checked'
    args = SimpleNamespace(capture_dir=tmp_path/'fake-capture', capture_sha256='b'*64,
                           export_dir=export_dir, export_sha256=checker.sha(export_dir/'export.json'))
    monkeypatch.setattr(checker, 'load_exporter', lambda: exporter)
    return args, exporter, capture, arrays, exported


def test_exact_original_slices_are_recovered_without_inference(checker, saved):
    args, _, _, arrays, _ = saved
    _, _, _, groups, pins = checker.prepare_inputs(args, np)
    assert [row['name'] for row in groups] == ['baseline', 'shard-00', 'shard-01', 'shard-02']
    assert [row['shape'] for row in groups] == [(1, 10, 7, 64), (1, 4, 7, 64), (1, 3, 7, 64), (1, 3, 7, 64)]
    full = arrays['query'].reshape(1, 10, 7, 64)
    np.testing.assert_array_equal(np.concatenate([row['inputs']['query'] for row in groups[1:]], axis=1), full)
    assert all(checker.sha(path) == expected for path, expected in pins.items())


@pytest.mark.parametrize('mutation', ['head_order', 'key_count'])
def test_export_cannot_change_head_order_or_truncate_key_domain(checker, saved, mutation):
    args, _, _, _, receipt = saved
    if mutation == 'head_order':
        receipt['models'][2], receipt['models'][3] = receipt['models'][3], receipt['models'][2]
    else:
        receipt['models'][1]['complete_key_count'] -= 1
    path = args.export_dir/'export.json'
    path.write_text(json.dumps(receipt))
    args.export_sha256 = checker.sha(path)
    with pytest.raises(ValueError, match='contract'):
        checker.prepare_inputs(args, np)


def test_changed_values_rejected_even_with_updated_artifact_file_and_raw_hash(checker, saved):
    args, exporter, _, _, receipt = saved
    entry = receipt['models'][1]['artifacts']['query.npy']
    path = args.export_dir/entry['file']
    changed = np.load(path, allow_pickle=False)
    changed[0, 0, 0, 0] += .25
    np.save(path, changed, allow_pickle=False)
    entry.update(sha256=checker.sha(path), bytes=path.stat().st_size, array=exporter.array_record(changed))
    report = args.export_dir/'export.json'
    report.write_text(json.dumps(receipt))
    args.export_sha256 = checker.sha(report)
    with pytest.raises(ValueError, match='values differ'):
        checker.prepare_inputs(args, np)


class FakeSession:
    def __init__(self, group, shift=0.):
        self.group, self.shift, self.calls, self.fallback_disabled = group, shift, 0, False
        self.inputs = [SimpleNamespace(name=name, type='tensor(float)', shape=list(group['shape'])) for name in checker_names()]
        self.outputs = [SimpleNamespace(name='output', type='tensor(float)', shape=list(group['shape']))]
        self.providers = ['CPUExecutionProvider']

    def get_inputs(self): return self.inputs
    def get_outputs(self): return self.outputs
    def get_providers(self): return self.providers
    def disable_fallback(self): self.fallback_disabled = True

    def run(self, names, feeds):
        assert names == ['output'] and tuple(feeds) == checker_names()
        assert all(feeds[name] is self.group['inputs'][name] for name in checker_names())
        self.calls += 1
        return [self.group['reference'].copy() + np.float32(self.shift)]


def checker_names():
    return ('query', 'key', 'value')


@pytest.mark.parametrize('mutation', ['provider', 'input_order', 'shape', 'dtype'])
def test_fake_session_contract_is_cpu_only_ordered_and_fixed_fp32(checker, mutation):
    shape = (1, 4, 7, 64)
    session = FakeSession(dict(shape=shape))
    if mutation == 'provider': session.providers.append('CUDAExecutionProvider')
    if mutation == 'input_order': session.inputs[0], session.inputs[1] = session.inputs[1], session.inputs[0]
    if mutation == 'shape': session.outputs[0].shape[2] -= 1
    if mutation == 'dtype': session.inputs[0].type = 'tensor(float16)'
    with pytest.raises(ValueError):
        checker.check_session(session, shape)


def test_all_four_single_calls_save_actual_arrays_and_original_layout(checker, saved, tmp_path):
    args, exporter, _, arrays, _ = saved
    _, _, _, groups, _ = checker.prepare_inputs(args, np)
    sessions = [FakeSession(group) for group in groups]
    output = tmp_path/'actual'
    output.mkdir()
    report = {}
    for session, group in zip(sessions, groups):
        checker.check_session(session, group['shape'])
    checker.execute_checks(sessions, groups, exporter, arrays, output, report, np)
    assert report['successful_session_run_count'] == 4 and report['all_full_shard_and_combined_checks_passed']
    assert not report['repeatability_validated'] and all(session.calls == 1 and session.fallback_disabled for session in sessions)
    row = report['runs'][0]
    assert row['phase'] == 'single_numerical_check'
    assert row['combined']['versus_original_CPU_SDPA']['array_equal']
    assert row['combined']['versus_full_ORT']['array_equal']
    np.testing.assert_array_equal(np.load(output/row['combined']['actual']['file']), arrays['output'])
    assert len(list(output.glob('*.npy'))) == 5


def test_bad_shard_numeric_output_is_saved_and_fails_predeclared_gate(checker, saved, tmp_path):
    args, exporter, _, arrays, _ = saved
    _, _, _, groups, _ = checker.prepare_inputs(args, np)
    sessions = [FakeSession(group, shift=.01 if index == 2 else 0.) for index, group in enumerate(groups)]
    output = tmp_path/'bad-output'
    output.mkdir()
    report = {}
    with pytest.raises(ValueError, match='gate failed'):
        checker.execute_checks(sessions, groups, exporter, arrays, output, report, np)
    assert report['successful_session_run_count'] == 4 and not report['all_full_shard_and_combined_checks_passed']
    assert report['runs'][0]['combined']['versus_original_CPU_SDPA']['outside_tolerance'] > 0
    assert report['runs'][0]['combined']['versus_full_ORT']['outside_tolerance'] > 0
    assert len(list(output.glob('*.npy'))) == 5


def test_wrong_actual_shape_saved_but_never_accepted(checker, saved, tmp_path):
    args, exporter, _, arrays, _ = saved
    _, _, _, groups, _ = checker.prepare_inputs(args, np)
    sessions = [FakeSession(group) for group in groups]
    sessions[0].run = lambda *_: [np.zeros((1, 10, 6, 64), np.float32)]
    output = tmp_path/'shape-error'
    output.mkdir()
    report = {}
    with pytest.raises(ValueError, match='same-shape'):
        checker.execute_checks(sessions, groups, exporter, arrays, output, report, np)
    assert report['successful_session_run_count'] == 1
    assert (output/'baseline-actual-00.npy').is_file()
