"""Synthetic-only graph tests; never run original weights or images on host."""
import builtins
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope='module')
def exporter():
    path = Path(__file__).resolve().parents[1] / 'deploy/rk3588/validation/export-v23-mlp.py'
    spec = importlib.util.spec_from_file_location('v23_mlp_export_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def state():
    rng = np.random.default_rng(123)
    return {'mlp.fc1.weight': rng.normal(0, .3, (16, 4)).astype(np.float32),
            'mlp.fc1.bias': rng.normal(0, .2, 16).astype(np.float32),
            'mlp.fc2.weight': rng.normal(0, .3, (4, 16)).astype(np.float32),
            'mlp.fc2.bias': np.linspace(.2, .5, 4, dtype=np.float32)}


def run_synthetic(model, value):
    # Only tiny random weights from this file are evaluated.
    from onnx.reference import ReferenceEvaluator
    return ReferenceEvaluator(model).run(None, {'input': value})[0]


def reference(value, state):
    x = value @ state['mlp.fc1.weight'].T + state['mlp.fc1.bias']
    erf = np.vectorize(math.erf, otypes=[np.float64])(x.astype(np.float64) / math.sqrt(2))
    gelu = (.5 * x.astype(np.float64) * (1 + erf)).astype(np.float32)
    return gelu @ state['mlp.fc2.weight'].T + state['mlp.fc2.bias']


@pytest.mark.parametrize('bucket', [1, 7, 13])
def test_complete_graph_exact_gelu_equation(exporter, state, bucket):
    x = np.random.default_rng(4).normal(size=(1, bucket, 4)).astype(np.float32)
    graph, bindings = exporter.build_mlp_onnx(state, bucket)
    np.testing.assert_allclose(run_synthetic(graph, x), reference(x, state), rtol=1e-4, atol=1e-5)
    ops = [node.op_type for node in graph.graph.node]
    assert ops.count('Gemm') == 2 and ops.count('Erf') == 1
    assert not {'LayerNormalization', 'Softmax', 'Attention'} & set(ops)
    assert len(bindings) == 4


def test_disjoint_partial_hidden_sum_adds_bias_once(exporter, state):
    x = np.random.default_rng(3).normal(size=(1, 7, 4)).astype(np.float32)
    baseline, _ = exporter.build_mlp_onnx(state, 7)
    partials = []
    for bounds in [(0, 5), (5, 10), (10, 16)]:
        graph, bindings = exporter.build_mlp_onnx(state, 7, hidden_slice=bounds)
        assert len(bindings) == 3
        assert 'mlp.fc2.bias' not in {item.name for item in graph.graph.initializer}
        assert len(next(node for node in graph.graph.node if node.name == 'fc2').input) == 2
        partials.append(run_synthetic(graph, x))
    summed = ((partials[0] + partials[1]) + partials[2]) + state['mlp.fc2.bias']
    np.testing.assert_allclose(summed, run_synthetic(baseline, x), rtol=1e-4, atol=1e-5)
    assert not np.allclose(summed + 2 * state['mlp.fc2.bias'], run_synthetic(baseline, x))


def test_explicit_full_hidden_range_still_omits_bias(exporter, state):
    x = np.ones((1, 3, 4), dtype=np.float32)
    partial, bindings = exporter.build_mlp_onnx(state, 3, hidden_slice=(0, 16))
    complete, _ = exporter.build_mlp_onnx(state, 3)
    assert len(bindings) == 3
    np.testing.assert_allclose(run_synthetic(partial, x) + state['mlp.fc2.bias'],
                               run_synthetic(complete, x), rtol=1e-4, atol=1e-5)


def test_zero_padding_and_crop_preserve_original_rows(exporter, state):
    x = np.random.default_rng(6).normal(size=(1, 5, 4)).astype(np.float32)
    padded = np.zeros((1, 8, 4), dtype=np.float32)
    padded[:, :5] = x
    graph5, _ = exporter.build_mlp_onnx(state, 5)
    graph8, _ = exporter.build_mlp_onnx(state, 8)
    actual = run_synthetic(graph8, padded)
    np.testing.assert_allclose(actual[:, :5], run_synthetic(graph5, x), rtol=1e-4, atol=1e-5)
    assert np.any(actual[:, 5:] != 0)  # Original biases: padding output must be cropped.


def test_original_orientation_slice_bindings_and_no_mutation(exporter, state):
    from onnx import numpy_helper
    before = {key: value.copy() for key, value in state.items()}
    graph, bindings = exporter.build_mlp_onnx(state, 3, hidden_slice=(5, 10))
    initializers = {item.name: numpy_helper.to_array(item) for item in graph.graph.initializer}
    np.testing.assert_array_equal(initializers['mlp.fc1.weight'], state['mlp.fc1.weight'][5:10])
    np.testing.assert_array_equal(initializers['mlp.fc1.bias'], state['mlp.fc1.bias'][5:10])
    np.testing.assert_array_equal(initializers['mlp.fc2.weight'], state['mlp.fc2.weight'][:, 5:10])
    for binding in bindings:
        assert binding['original'] == exporter.array_record(state[binding['parameter']])
        assert binding['embedded'] == exporter.array_record(initializers[binding['initializer']])
    for key, original in before.items():
        np.testing.assert_array_equal(state[key], original)
    assert all(node.attribute[-1].i == 1 for node in graph.graph.node if node.op_type == 'Gemm')


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'float16', 'nonfinite', 'wrong_shape', 'strided'])
def test_invalid_state_rejected(exporter, state, mutation):
    if mutation == 'missing':
        state.pop('mlp.fc2.bias')
    elif mutation == 'extra':
        state['ls2.gamma'] = np.ones(4, dtype=np.float32)
    elif mutation == 'float16':
        state['mlp.fc1.bias'] = state['mlp.fc1.bias'].astype(np.float16)
    elif mutation == 'nonfinite':
        state['mlp.fc1.bias'][0] = np.nan
    elif mutation == 'wrong_shape':
        state['mlp.fc2.weight'] = state['mlp.fc2.weight'][:, :-1].copy()
    elif mutation == 'strided':
        state['mlp.fc2.weight'] = state['mlp.fc2.weight'][:, ::-1]
    with pytest.raises(ValueError):
        exporter.build_mlp_onnx(state, 7)


@pytest.mark.parametrize('bounds', [(-1, 64), (64, 64), (64, 192), (0., 64), [0], (0, 63), (63, 128)])
def test_invalid_hidden_slice_rejected(exporter, bounds):
    with pytest.raises(ValueError):
        exporter.validate_slice(bounds, 128, alignment=64)


def test_actual_three_slice_partition_and_default(exporter):
    cuts = [(0, 1344), (1344, 2688), (2688, 4096)]
    assert exporter.validate_slice(None, 4096, alignment=64) == ((0, 4096), True)
    for bounds in cuts:
        assert exporter.validate_slice(bounds, 4096, alignment=64) == (bounds, False)
    assert [i for begin, end in cuts for i in range(begin, end)] == list(range(4096))


@pytest.fixture
def tiny_capture(exporter, state, tmp_path, monkeypatch):
    # Fixture identity is patched only in the test. Production has no override.
    monkeypatch.setattr(exporter, 'CHANNELS', 4)
    monkeypatch.setattr(exporter, 'HIDDEN', 16)
    monkeypatch.setattr(exporter, 'ORIGINAL_SHAPE', (1, 7, 4))
    saved_state = dict(state, **{'ls2.gamma': np.ones(4, dtype=np.float32)})
    np.savez(tmp_path / 'mlp-state.npz', **saved_state)
    values = {'norm2': np.arange(28, dtype=np.float32).reshape(1, 7, 4),
              'mlp_output': np.ones((1, 7, 4), dtype=np.float32)}
    np.savez(tmp_path / 'activations.npz', **values)
    capture = dict(kind='original_VGGT_CoMe_MLP_stage_capture',
        status='captured_original_FP32_stages_bitwise_bound', script_sha256=exporter.CAPTURE_SCRIPT_SHA,
        original_weights=True, architecture_changed=False, npu_used=False, ORB_used=False,
        GT_used=False, host_inference_executed=False, inference_on_board_only=True,
        full_model_forward_count=1, cleanup_errors=[], hooks_removed=True, sources_and_inputs_unchanged=True,
        checkpoint=dict(sha256=exporter.CHECKPOINT_SHA, bytes=exporter.CHECKPOINT_BYTES),
        architecture=dict(mlp_class='Mlp', gelu_approximate='none', dropout=0., channels=4,
                          views=2, special_tokens=5, grid_hw=[18, 24]),
        mlp_state_keys=list(exporter.ARCHIVE_KEYS), global0_vs_profile=dict(input=True, output=True, pos=True),
        full_predictions_vs_original=dict(depth=True, depth_conf=True, extrinsic=True, intrinsic=True, pose_enc=True),
        global0_state={key: exporter.array_record(value) for key, value in saved_state.items()},
        arrays={key: exporter.array_record(value) for key, value in values.items()},
        artifacts={name: dict(bytes=(tmp_path / name).stat().st_size, sha256=exporter.sha(tmp_path / name))
                   for name in ['mlp-state.npz', 'activations.npz']})
    receipt = tmp_path / 'capture.json'
    receipt.write_text(json.dumps(capture))
    monkeypatch.setattr(exporter, 'CAPTURE_SHA', exporter.sha(receipt))
    return tmp_path, capture, state, monkeypatch


def test_loader_verifies_np_values_without_torch_or_onnx(exporter, tiny_capture, monkeypatch):
    folder, capture, state, _ = tiny_capture
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split('.')[0] in {'torch', 'onnx', 'onnxruntime'}:
            raise AssertionError('Capture loader must not need any inference framework')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    loaded, arrays, weights, pins = exporter.load_capture(folder, exporter.CAPTURE_SHA)
    assert loaded == capture and set(arrays) == {'input', 'output'} and len(pins) == 3
    assert set(weights) == set(state) and 'ls2.gamma' not in weights
    for key, value in weights.items():
        np.testing.assert_array_equal(value, state[key])


@pytest.mark.parametrize('mutation', ['capture_pin', 'state_file_pin', 'input_raw_pin', 'excluded_ls_raw_pin',
                                     'wrong_gelu', 'missing_archive_key', 'false_reference_gate'])
def test_loader_rejects_provenance_or_architecture_changes(exporter, tiny_capture, mutation):
    folder, capture, state, monkeypatch = tiny_capture
    if mutation == 'capture_pin':
        with pytest.raises(ValueError, match='frozen'):
            exporter.load_capture(folder, '0' * 64)
        return
    if mutation == 'state_file_pin':
        capture['artifacts']['mlp-state.npz']['sha256'] = '0' * 64
    elif mutation == 'input_raw_pin':
        capture['arrays']['norm2']['sha256'] = '0' * 64
    elif mutation == 'excluded_ls_raw_pin':
        capture['global0_state']['ls2.gamma']['sha256'] = '0' * 64
    elif mutation == 'wrong_gelu':
        capture['architecture']['gelu_approximate'] = 'tanh'
    elif mutation == 'missing_archive_key':
        capture['mlp_state_keys'].pop()
    elif mutation == 'false_reference_gate':
        capture['full_predictions_vs_original']['intrinsic'] = False
    (folder / 'capture.json').write_text(json.dumps(capture))
    monkeypatch.setattr(exporter, 'CAPTURE_SHA', exporter.sha(folder / 'capture.json'))
    with pytest.raises(ValueError):
        exporter.load_capture(folder, exporter.CAPTURE_SHA)


def test_existing_output_rejected_before_loading_capture(exporter, tmp_path):
    args = SimpleNamespace(output=tmp_path, bucket=874, capture_dir=tmp_path / 'absent',
                           capture_sha256=exporter.CAPTURE_SHA, hidden_slice=None)
    with pytest.raises(FileExistsError):
        exporter.export(args)


def test_cli_names_and_explicit_supported_buckets(exporter):
    assert exporter.BUCKETS == (874, 1024, 512)
    assert exporter.parse_slice('1344:2688') == (1344, 2688)
    with pytest.raises(Exception):
        exporter.parse_slice('1344,2688')
