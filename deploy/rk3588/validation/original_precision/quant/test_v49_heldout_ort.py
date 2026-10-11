"""Tiny generated arrays and fake ORT metadata/session; no ORT/SDK import."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem.replace('-', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = load(HERE / 'check-v49-heldout-pair-onnx-board.py')


@pytest.fixture
def tiny():
    C = NS(INPUT_SHAPE=(2, 3, 4), OUTPUT_SHAPE=(1, 6, 4), require_board=lambda: 'fake-only')
    sources = {}
    samples = []
    for index in range(10):
        split = 'calibration' if index < 8 else 'evaluation'
        value = (np.arange(24, dtype=np.float32) + np.float32(index)).reshape(C.INPUT_SHAPE)
        sources[index] = value
        samples.append({'sample_id': split + '_' + str(index), 'split': split,
                        'images': [{'synthetic': index * 2}, {'synthetic': index * 2 + 1}],
                        'artifacts': {'input.npy': {'kind': 'input', 'index': index},
                                      'output.npy': {'kind': 'output', 'index': index}}})
    def saved(directory, item, shape, dtype):
        value = sources[item['index']]
        return (value.copy() if item['kind'] == 'input' else (value + np.float32(2)).reshape(shape).copy())
    def compare(actual, reference):
        return {'rtol': 1e-4, 'atol': 1e-5,
                'strict_FP32_gate_passed': bool(np.all(np.abs(actual.astype(np.float64) - reference)
                                                     <= 1e-5 + 1e-4 * np.abs(reference)))}
    K = NS(load_array=saved)
    P = NS(RTOL=1e-4, ATOL=1e-5, compare=compare)
    class FakeSession:
        def __init__(self):
            self.calls = []
            self.providers = ['CPUExecutionProvider']
            self.output_shape = list(C.OUTPUT_SHAPE)
            self.mutate = False
            self.error = np.float32(0)
        def disable_fallback(self):
            self.fallback_disabled = True
        def get_providers(self):
            return self.providers
        def get_inputs(self):
            return [NS(name='input', type='tensor(float)', shape=list(C.INPUT_SHAPE))]
        def get_outputs(self):
            return [NS(name='output', type='tensor(float)', shape=self.output_shape)]
        def run(self, names, inputs):
            assert names == ['output'] and set(inputs) == {'input'}
            value = inputs['input']
            self.calls.append(value.copy())
            output = (value + np.float32(2) + self.error).reshape(C.OUTPUT_SHAPE).copy()
            if self.mutate:
                value.flat[0] += np.float32(1)
            return [output]
    report = {'calls': [], 'attempted_session_run_count': 0, 'successful_session_run_count': 0}
    return C, K, P, samples, sources, FakeSession(), report


def invoke(tiny, tmp_path):
    C, K, P, samples, sources, session, report = tiny
    return H.run_heldout(session, samples, tmp_path, K, P, C, tmp_path, report)


def test_only_two_heldout_calls_with_saved_actual_input_and_full_output(tiny, tmp_path):
    C, K, P, samples, sources, session, report = tiny
    assert invoke(tiny, tmp_path)
    assert report['attempted_session_run_count'] == report['successful_session_run_count'] == 2
    assert len(session.calls) == 2 and session.fallback_disabled
    assert [row['sample_id'] for row in report['calls']] == ['evaluation_8', 'evaluation_9']
    for index, row in enumerate(report['calls'], 8):
        saved_input = np.load(tmp_path / row['used_input']['file'], allow_pickle=False)
        output = np.load(tmp_path / row['output']['file'], allow_pickle=False)
        assert np.array_equal(saved_input, sources[index])
        assert np.array_equal(output, (sources[index] + np.float32(2)).reshape(C.OUTPUT_SHAPE))
        assert row['input_tensor_unchanged'] and row['reference_tensor_unchanged']
        assert set(row['timings_seconds']) == {'load_saved_arrays_and_initial_hashes',
            'save_used_input_with_hashes', 'session_run', 'validate_and_copy_owned_output',
            'save_complete_output_with_hashes', 'strict_FP32_comparison', 'post_call_input_reference_hashes'}
    with pytest.raises(FileExistsError):
        invoke(tiny, tmp_path)
    assert len(session.calls) == 2


def test_fp32_numerical_rejection_still_saves_both_complete_outputs(tiny, tmp_path):
    tiny[5].error = np.float32(0.1)
    assert not invoke(tiny, tmp_path)
    assert tiny[6]['successful_session_run_count'] == 2
    assert tiny[6]['strict_FP32_gate_passed'] is False
    assert all((tmp_path / row['output']['file']).is_file() for row in tiny[6]['calls'])


def test_mutated_original_input_rejects_before_second_call(tiny, tmp_path):
    tiny[5].mutate = True
    with pytest.raises(ValueError, match='mutated'):
        invoke(tiny, tmp_path)
    assert tiny[6]['successful_session_run_count'] == 1
    assert tiny[6]['calls'][0]['input_tensor_unchanged'] is False


@pytest.mark.parametrize('failure', ['provider', 'output_shape'])
def test_ort_metadata_rejects_before_any_call(tiny, tmp_path, failure):
    if failure == 'provider':
        tiny[5].providers = ['CPUExecutionProvider', 'unexpected-provider']
    else:
        tiny[5].output_shape = [1, 24]
    with pytest.raises(ValueError):
        invoke(tiny, tmp_path)
    assert not tiny[5].calls


def test_host_main_rejects_before_loading_dependency_source(tmp_path, monkeypatch):
    monkeypatch.setattr(H.platform, 'machine', lambda: 'x86_64')
    args = []
    for name in ('contracts', 'pair-contracts', 'pair-dir', 'capture-contracts', 'capture-dir',
                 'toolchain-prefix', 'output-dir'):
        args.extend(['--' + name, str(tmp_path / name)])
    for name in ('contracts-sha256', 'capture-contracts-sha256', 'capture-sha256'):
        args.extend(['--' + name, '0' * 64])
    with pytest.raises(SystemExit) as exc:
        H.main(args)
    assert exc.value.code == 2 and not (tmp_path / 'output-dir').exists()
