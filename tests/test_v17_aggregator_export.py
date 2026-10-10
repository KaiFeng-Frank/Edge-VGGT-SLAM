"""Small synthetic graph checks only; no real image/model or host ORT execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / 'deploy/rk3588/validation/export-v17-aggregator-block.py'
spec = importlib.util.spec_from_file_location('v17_aggregator_export_test', EXPORT)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def fixture():
    rng = np.random.default_rng(18)
    dim, heads, shape = 8, 2, (2, 7, 8)
    shapes = {key: (dim,) for key in exporter.BASE_KEYS if key.startswith('norm') or key.endswith('gamma')}
    shapes.update({'attn.qkv.weight': (24, 8), 'attn.qkv.bias': (24,),
                   'attn.proj.weight': (8, 8), 'attn.proj.bias': (8,),
                   'mlp.fc1.weight': (32, 8), 'mlp.fc1.bias': (32,),
                   'mlp.fc2.weight': (8, 32), 'mlp.fc2.bias': (8,)})
    shapes.update({key: (4,) for key in exporter.QK_KEYS})
    weights = {key: np.ascontiguousarray(rng.normal(0, .2, size).astype(np.float32)) for key, size in shapes.items()}
    for key in weights:
        if key.endswith('.weight') and ('norm' in key):
            weights[key] += 1
    positions = np.zeros((2, 7, 2), dtype=np.int64)
    positions[:, 5:] = [[1, 2], [2, 1]]
    # Synthetic coefficients deliberately differ between y/x token coordinates.
    cosine = np.array([[1, 1], [.8, .8], [.6, .6]], np.float32)
    sine = np.array([[0, 0], [.6, .6], [.8, .8]], np.float32)
    inputs = rng.normal(0, .7, shape).astype(np.float32)
    return weights, shape, positions, cosine, sine, heads, inputs


def graph_fixture():
    pytest.importorskip('onnx')
    weights, shape, pos, cos, sin, heads, inputs = fixture()
    graph = exporter.build_aggregator_onnx(weights, shape, pos, cos, sin, heads=heads)
    return graph, weights, shape, pos, cos, sin, heads, inputs


def test_loader_import_does_not_import_torch_onnx_or_ort():
    program = ('import sys,importlib.util; s=importlib.util.spec_from_file_location("v17",sys.argv[1]); '
               'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
               'assert all(k not in sys.modules for k in ("torch","onnx","onnxruntime"))')
    subprocess.run([sys.executable, '-B', '-c', program, str(EXPORT)], check=True)


def test_exact_four_layer_norms_and_static_shape():
    graph, weights, shape, *rest = graph_fixture()
    import onnx
    assert [dimension.dim_value for dimension in graph.graph.input[0].type.tensor_type.shape.dim] == list(shape)
    assert [dimension.dim_value for dimension in graph.graph.output[0].type.tensor_type.shape.dim] == list(shape)
    norms = {n.name: n for n in graph.graph.node if n.op_type == 'LayerNormalization'}
    assert set(norms) == set(exporter.EPS_FIELDS)
    for node in norms.values():
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        assert attrs['epsilon'] == np.float32(1e-5)
        assert attrs['axis'] == -1 and attrs['stash_type'] == onnx.TensorProto.FLOAT


def test_all_eighteen_parameters_are_original_values():
    graph, weights, *_ = graph_fixture()
    from onnx import numpy_helper
    initializers = {value.name: numpy_helper.to_array(value) for value in graph.graph.initializer}
    for key, value in weights.items():
        name = key + '_transposed' if key in ('attn.qkv.weight', 'attn.proj.weight', 'mlp.fc1.weight', 'mlp.fc2.weight') else key
        expected = value.T if name.endswith('_transposed') else value
        assert np.array_equal(initializers[name], expected), key
    assert len(weights) == 18
    assert not any(value.data_location for value in graph.graph.initializer)


def test_rope_uses_original_lookup_order_and_score_inputs():
    graph, _, _, pos, cos, sin, *_ = graph_fixture()
    from onnx import numpy_helper
    arrays = {a.name: numpy_helper.to_array(a) for a in graph.graph.initializer}
    for axis, label in enumerate(('y', 'x')):
        for table, kind in ((cos, 'cos'), (sin, 'sin')):
            assert np.array_equal(arrays[f'rope_{label}_{kind}'], table[pos[..., axis]][:, None])
    nodes = {n.name: n for n in graph.graph.node}
    assert list(nodes['key_transpose'].input) == ['key_rope']
    assert list(nodes['attention_scores'].input) == ['query_rope', 'key_transpose']
    assert list(nodes['attn.q_norm'].input)[0] == 'query'
    assert list(nodes['attn.k_norm'].input)[0] == 'key'
    assert nodes['gelu_scaled_input'].op_type == 'Div'
    assert nodes['gelu_erf'].op_type == 'Erf'
    assert not any(n.op_type in ('Cos', 'Sin') for n in graph.graph.node)


def independent_numpy_block(weights, inputs, pos, cosine, sine, heads):
    from scipy.special import erf
    def linear(name, x):
        return x @ weights[name + '.weight'].T + weights[name + '.bias']
    def norm(name, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        centered = x - mean
        variance = np.mean(centered * centered, axis=-1, keepdims=True)
        return centered / np.sqrt(variance + np.float32(1e-5)) * weights[name + '.weight'] + weights[name + '.bias']
    def rope(x):
        pieces = []
        for axis, component in enumerate(np.split(x, 2, axis=-1)):
            first, second = np.split(component, 2, axis=-1)
            rotated = np.concatenate((-second, first), axis=-1)
            pieces.append(component * cosine[pos[..., axis]][:, None] + rotated * sine[pos[..., axis]][:, None])
        return np.concatenate(pieces, axis=-1)
    batch, tokens, dim = inputs.shape
    qkv = linear('attn.qkv', norm('norm1', inputs)).reshape(batch, tokens, 3, heads, dim // heads)
    q, k, v = qkv.transpose(2, 0, 3, 1, 4)
    q, k = rope(norm('attn.q_norm', q)), rope(norm('attn.k_norm', k))
    scores = (q @ k.transpose(0, 1, 3, 2)) * np.float32((dim // heads) ** -.5)
    scores -= np.max(scores, axis=-1, keepdims=True)
    p = np.exp(scores)
    p /= np.sum(p, axis=-1, keepdims=True)
    attended = (p @ v).transpose(0, 2, 1, 3).reshape(inputs.shape)
    residual = inputs + linear('attn.proj', attended) * weights['ls1.gamma']
    hidden = linear('mlp.fc1', norm('norm2', residual))
    activated = ((hidden * (erf(hidden / np.float32(np.sqrt(2))) + np.float32(1))) * np.float32(.5))
    return residual + linear('mlp.fc2', activated) * weights['ls2.gamma']


def test_tiny_synthetic_graph_matches_independent_numpy_formula():
    # ReferenceEvaluator runs this eight-channel synthetic graph only, not ORT,
    # a captured VGGT model, an image, or any real weight array.
    graph, weights, _, pos, cos, sin, heads, inputs = graph_fixture()
    from onnx.reference import ReferenceEvaluator
    actual = ReferenceEvaluator(graph).run(None, {'input': inputs})[0]
    expected = independent_numpy_block(weights, inputs, pos, cos, sin, heads)
    assert actual.dtype == np.float32 and np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize('change', ['missing_qnorm', 'extra_weight', 'weight_dtype', 'weight_shape', 'weight_nan'])
def test_parameter_contract_rejects_mismatch(change):
    weights, shape, *_ = fixture()
    if change == 'missing_qnorm':
        del weights['attn.q_norm.bias']
    elif change == 'extra_weight':
        weights['extra.weight'] = np.zeros(1, np.float32)
    elif change == 'weight_dtype':
        weights['attn.k_norm.bias'] = weights['attn.k_norm.bias'].astype(np.float64)
    elif change == 'weight_shape':
        weights['attn.k_norm.bias'] = np.zeros(5, np.float32)
    else:
        weights['attn.k_norm.bias'][0] = np.nan
    with pytest.raises(ValueError):
        exporter.validate_state(weights, shape[-1], 2)


@pytest.mark.parametrize('change', ['negative_pos', 'pos_dtype', 'table_shape', 'table_dtype', 'zero_pos_nonidentity', 'nonfinite'])
def test_position_table_contract_rejects_mismatch(change):
    _, shape, pos, cos, sin, heads, _ = fixture()
    if change == 'negative_pos':
        pos[0, 0, 0] = -1
    elif change == 'pos_dtype':
        pos = pos.astype(np.float32)
    elif change == 'table_shape':
        sin = sin[:-1]
    elif change == 'table_dtype':
        cos = cos.astype(np.float64)
    elif change == 'zero_pos_nonidentity':
        sin[0, 0] = .1
    else:
        sin[1, 0] = np.nan
    with pytest.raises(ValueError):
        exporter.validate_positions(pos, cos, sin, shape, heads)


def test_loader_requires_explicit_correct_receipt_sha(tmp_path):
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps({'status': 'failed'}))
    with pytest.raises(ValueError, match='SHA256'):
        exporter.load_capture(tmp_path, '0' * 64, 'frame0')
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='Completed frozen'):
        exporter.load_capture(tmp_path, expected, 'frame0')
