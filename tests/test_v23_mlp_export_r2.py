"""Tiny random graph/provenance tests; no original-weight inference on host."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope='module')
def exporter():
    path = Path(__file__).resolve().parents[1] / 'deploy/rk3588/validation/export-v23-mlp-r2.py'
    spec = importlib.util.spec_from_file_location('v23_mlp_export_r2_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def state():
    rng = np.random.default_rng(23)
    return {'mlp.fc1.weight': rng.normal(0, .3, (16, 4)).astype(np.float32),
            'mlp.fc1.bias': rng.normal(0, .2, 16).astype(np.float32),
            'mlp.fc2.weight': rng.normal(0, .3, (4, 16)).astype(np.float32),
            'mlp.fc2.bias': np.linspace(.2, .5, 4, dtype=np.float32)}


def evaluate_tiny(model, inputs):
    from onnx.reference import ReferenceEvaluator
    return ReferenceEvaluator(model).run(None, {'input': inputs})[0]


@pytest.mark.parametrize('rows', [1, 7, 13])
def test_rank3_graph_matches_r1_equation(exporter, state, rows):
    parent = exporter.load_parent()
    old, _ = parent.build_mlp_onnx(state, rows)
    new, bindings = exporter.build_mlp_onnx(state, rows)
    x = np.random.default_rng(8).normal(size=(1, rows, 4)).astype(np.float32)
    np.testing.assert_allclose(evaluate_tiny(new, x), evaluate_tiny(old, x), rtol=1e-4, atol=1e-5)
    ops = [node.op_type for node in new.graph.node]
    assert ops.count('MatMul') == 2 and ops.count('Erf') == 1
    assert not {'Gemm', 'Reshape', 'Transpose', 'LayerNormalization', 'Softmax'} & set(ops)
    assert len(bindings) == 4
    assert list(new.graph.input[0].type.tensor_type.shape.dim)[0].dim_value == 1


def test_exact_frozen_div_erf_add_mul_mul_pattern(exporter, state):
    from onnx import numpy_helper
    graph, _ = exporter.build_mlp_onnx(state, 3)
    region = [(node.name, node.op_type, list(node.input), list(node.output))
              for node in graph.graph.node if node.name.startswith('gelu')]
    assert region == [
        ('gelu_scaled_input', 'Div', ['mlp.fc1', 'gelu_sqrt_two'], ['gelu_scaled_input']),
        ('gelu_erf', 'Erf', ['gelu_scaled_input'], ['gelu_erf']),
        ('gelu_erf_plus_one', 'Add', ['gelu_erf', 'gelu_one'], ['gelu_erf_plus_one']),
        ('gelu_before_half', 'Mul', ['mlp.fc1', 'gelu_erf_plus_one'], ['gelu_before_half']),
        ('gelu', 'Mul', ['gelu_before_half', 'gelu_half'], ['gelu'])]
    constants = {item.name: numpy_helper.to_array(item) for item in graph.graph.initializer}
    assert constants['gelu_sqrt_two'].dtype == np.float32
    assert constants['gelu_sqrt_two'].shape == ()
    assert constants['gelu_sqrt_two'].item() == np.float32(np.sqrt(2)).item()


def test_three_partials_fixed_order_sum_bias_once(exporter, state):
    x = np.random.default_rng(18).normal(size=(1, 7, 4)).astype(np.float32)
    full, _ = exporter.build_mlp_onnx(state, 7)
    partials = []
    for bounds in [(0, 5), (5, 10), (10, 16)]:
        graph, bindings = exporter.build_mlp_onnx(state, 7, hidden_slice=bounds)
        assert len(bindings) == 3
        assert 'mlp.fc2.bias' not in {item.name for item in graph.graph.initializer}
        assert graph.graph.node[-1].op_type == 'MatMul'
        partials.append(evaluate_tiny(graph, x))
    actual = ((partials[0] + partials[1]) + partials[2]) + state['mlp.fc2.bias']
    np.testing.assert_allclose(actual, evaluate_tiny(full, x), rtol=1e-4, atol=1e-5)


def test_transposed_initializers_bind_original_weights_without_mutation(exporter, state):
    from onnx import numpy_helper
    before = {key: value.copy() for key, value in state.items()}
    graph, bindings = exporter.build_mlp_onnx(state, 3, hidden_slice=(5, 10))
    tensors = {item.name: numpy_helper.to_array(item) for item in graph.graph.initializer}
    np.testing.assert_array_equal(tensors['mlp.fc1.weight'], state['mlp.fc1.weight'][5:10].T)
    np.testing.assert_array_equal(tensors['mlp.fc2.weight'], state['mlp.fc2.weight'][:, 5:10].T)
    parent = exporter.load_parent()
    for row in bindings:
        assert row['original'] == parent.array_record(state[row['parameter']])
        assert row['embedded'] == parent.array_record(tensors[row['initializer']])
        assert row['weight_transposed_in_initializer'] == row['parameter'].endswith('weight')
    for key, value in before.items():
        np.testing.assert_array_equal(state[key], value)


def test_explicit_full_slice_has_no_fc2_bias(exporter, state):
    graph, bindings = exporter.build_mlp_onnx(state, 3, hidden_slice=(0, 16))
    assert len(bindings) == 3 and graph.graph.node[-1].op_type == 'MatMul'


def test_padding_crop_is_row_independent(exporter, state):
    x = np.random.default_rng(5).normal(size=(1, 5, 4)).astype(np.float32)
    padded = np.zeros((1, 8, 4), np.float32)
    padded[:, :5] = x
    small, _ = exporter.build_mlp_onnx(state, 5)
    large, _ = exporter.build_mlp_onnx(state, 8)
    np.testing.assert_allclose(evaluate_tiny(large, padded)[:, :5], evaluate_tiny(small, x), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('bucket', [0, -1, 1., True])
def test_bad_bucket_rejected(exporter, state, bucket):
    with pytest.raises(ValueError):
        exporter.build_mlp_onnx(state, bucket)


@pytest.mark.parametrize('bounds', [(-1, 5), (8, 8), (8, 17), (0., 8)])
def test_bad_slice_rejected(exporter, state, bounds):
    with pytest.raises(ValueError):
        exporter.build_mlp_onnx(state, 3, hidden_slice=bounds)


def test_r2_receipt_identifies_entrypoint_and_frozen_parent(exporter, monkeypatch, tmp_path):
    parent = exporter.load_parent()
    receipt = tmp_path / 'export.json'

    def synthetic_export(args):
        report = dict(script_sha256=exporter.PARENT_SHA, input_pins={}, includes_fc2_bias=False,
                      onnx=dict(file='mlp.onnx', sha256='0' * 64, bytes=0))
        receipt.write_text(json.dumps(report))
        return report, receipt

    parent.export = synthetic_export
    monkeypatch.setattr(exporter, 'load_parent', lambda: parent)
    report, saved = exporter.export(SimpleNamespace())
    assert report == json.loads(saved.read_text())
    assert report['script_sha256'] == exporter.sha(exporter.__file__)
    assert report['parent_exporter']['sha256'] == exporter.PARENT_SHA
    assert report['gelu_pattern_source']['sha256'] == exporter.GELU_PATTERN_SHA
    assert len(report['input_pins']) == 3
    assert report['fp32_numerical_equivalence_validated'] is False
    assert report['fp32_operation_order_changed_from_r1'] is True


def test_parent_source_pin_enforced(exporter, monkeypatch):
    monkeypatch.setattr(exporter, 'PARENT_SHA', '0' * 64)
    with pytest.raises(ValueError, match='R1'):
        exporter.load_parent()


def test_gelu_source_pin_enforced_before_export(exporter, monkeypatch):
    monkeypatch.setattr(exporter, 'GELU_PATTERN_SHA', '0' * 64)
    with pytest.raises(ValueError, match='GELU'):
        exporter.export(SimpleNamespace())
