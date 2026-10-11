"""Tiny generated arrays/fake sessions only; no SDK, ORT or real model calls."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

HERE = Path(os.environ.get('VGGT_PAIR_SOURCE_DIR', Path(__file__).parent))


def module(filename):
    spec = importlib.util.spec_from_file_location(filename.replace('-', '_'), HERE / filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


C = module('v47_pair_contracts.py')
SDK = module('run-v47-pair-rknn-board.py')
ORT = module('check-v47-pair-onnx-board.py')
COMPILER = module('compile-v47-pair.py')


def descriptor(name):
    shape = C.INPUT_SHAPE if name == 'input' else C.OUTPUT_SHAPE
    return dict(name=name, shape=list(shape), n_elems=int(np.prod(shape)), type=1,
                quantization_type=2, scale=1.0, zero_point=0, format=3, size=int(np.prod(shape)) * 2)


def test_distinct_sdk_shapes_even_when_elements_equal():
    assert C.validate_sdk_attr(descriptor('input'))['shape'] != C.validate_sdk_attr(descriptor('output'))['shape']
    wrong = descriptor('output')
    wrong['shape'] = list(C.INPUT_SHAPE)
    with pytest.raises(ValueError):
        C.validate_sdk_attr(wrong)


@pytest.mark.parametrize('key,value', [('format', 1), ('type', 2), ('scale', .9),
                                      ('zero_point', 1), ('size', 1), ('name', 'unknown')])
def test_unknown_sdk_descriptor_fails_before_run(key, value):
    attr = descriptor('input')
    attr[key] = value
    with pytest.raises(ValueError):
        C.validate_sdk_attr(attr)


def test_fixed_FP32_gate_keeps_rejection():
    expected = np.zeros((1, 2, 3), np.float32)
    actual = expected.copy()
    actual[0, 0, 0] = np.float32(2e-5)
    stats = C.compare(actual, expected)
    assert stats['strict_FP32_gate_passed'] is False
    assert stats['outside_tolerance_elements'] == 1
    assert stats['max_tolerance_ratio'] > 1
    actual[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        C.compare(actual, expected)


def test_ORT_shape_provider_roles():
    session = NS(disable_fallback=lambda: None, get_providers=lambda: ['CPUExecutionProvider'],
                 get_inputs=lambda: [NS(name='input', type='tensor(float)', shape=list(C.INPUT_SHAPE))],
                 get_outputs=lambda: [NS(name='output', type='tensor(float)', shape=list(C.OUTPUT_SHAPE))])
    ORT.check_session(session, C)
    session.get_outputs = lambda: [NS(name='output', type='tensor(float)', shape=list(C.INPUT_SHAPE))]
    with pytest.raises(ValueError):
        ORT.check_session(session, C)


def test_compile_ORT_rejection_explicit_only():
    row = dict(schema='original_VGGT_pair_CPU_ORT_check_v47', pair_export_sha256='p',
               successful_session_run_count=4, session_released=True, source_and_inputs_unchanged=True,
               input_tensor_unchanged=True, cleanup_errors=[], NPU_used=False, inference_on_board_only=True,
               rtol=1e-4, atol=1e-5, strict_FP32_gate_passed=False, status='numerical_rejected')
    with pytest.raises(ValueError):
        COMPILER.check_ORT(row, 'p')
    COMPILER.check_ORT(row, 'p', allow_rejected=True)
    assert row['strict_FP32_gate_passed'] is False
    row['successful_session_run_count'] = 3
    with pytest.raises(ValueError):
        COMPILER.check_ORT(row, 'p', allow_rejected=True)


def test_private_runtime_binding_only_generated_arrays(tmp_path):
    small = NS(INPUT_SHAPE=(2, 3, 4), OUTPUT_SHAPE=(1, 6, 4),
               validate_sdk_attr=lambda x: x, compare=C.compare)
    generated = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    expected = generated.reshape(1, 6, 4).copy()
    events = []

    def inherited(*a, **kw):
        events.extend(['run', 'copy_before_release', 'release'])
        return expected.copy(), {'io_total_seconds': .0123, 'output_buffer_released': True}

    runtime = NS(execute_call=inherited, attr_record=lambda a: a)
    saved = []
    SDK.bind_pair_runtime(runtime, small, generated, expected,
                          {'pair_dir': str(tmp_path), 'pair_export_sha256': 'pin'},
                          {}, tmp_path, saved)
    assert runtime.SHAPE == small.OUTPUT_SHAPE
    assert runtime.load_capture(tmp_path, 'pin')[0] is generated
    with pytest.raises(ValueError):
        runtime.load_capture(tmp_path, 'wrong')
    actual, timing = runtime.execute_call(None)
    assert events == ['run', 'copy_before_release', 'release']
    assert timing['io_total_seconds'] == .0123
    assert np.array_equal(actual, expected)
    assert saved[0]['sha256'] == hashlib.sha256((tmp_path / saved[0]['file']).read_bytes()).hexdigest()
    assert np.array_equal(np.load(tmp_path / saved[0]['file']), expected)


def test_host_board_guard_rejects_before_executor(monkeypatch):
    monkeypatch.setattr(C.platform, 'machine', lambda: 'x86_64')
    with pytest.raises(ValueError, match='ARM64'):
        C.require_board()
