"""Tiny synthetic ONNX only; no real-image/model, host ORT or NPU execution."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / 'deploy/rk3588/validation/export-v19-aggregator-mlp2d.py'
spec = importlib.util.spec_from_file_location('v19_mlp2d_test', EXPORT)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)
spec = importlib.util.spec_from_file_location('v19_tiny_fixture', Path(__file__).with_name('test_v17_aggregator_export.py'))
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def graphs(global_shape=False):
    pytest.importorskip('onnx')
    weights, shape, positions, cosine, sine, heads, inputs = fixture.fixture()
    if global_shape:
        shape = (1, shape[0] * shape[1], shape[2])
        positions = positions.reshape(*shape[:2], 2)
        inputs = inputs.reshape(shape)
    old = exporter.r3_module().build_aggregator_onnx(weights, shape, positions, cosine, sine, heads=heads)
    old_bytes = old.SerializeToString()
    new = exporter.rewrite_mlp2d(old)
    assert old.SerializeToString() == old_bytes
    return old, new, weights, shape, inputs


@pytest.mark.parametrize('global_shape', [False, True])
def test_tiny_reference_equivalence_and_full_shape(global_shape):
    from onnx.reference import ReferenceEvaluator
    old, new, _, shape, inputs = graphs(global_shape)
    before = ReferenceEvaluator(old).run(None, {'input': inputs})[0]
    after = ReferenceEvaluator(new).run(None, {'input': inputs})[0]
    assert after.shape == before.shape == shape
    assert np.isfinite(after).all()
    np.testing.assert_allclose(after, before, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('global_shape', [False, True])
def test_all_original_initializers_unrelated_nodes_and_exact_gelu_stay_byte_identical(global_shape):
    from onnx import helper, numpy_helper
    old, new, weights, shape, _ = graphs(global_shape)
    assert [value.SerializeToString() for value in old.graph.initializer] == [
        value.SerializeToString() for value in new.graph.initializer[:-2]]
    changed = {'mlp.fc1.matmul', 'mlp.fc1', 'mlp.fc2.matmul', 'mlp.fc2'}
    introduced = {'mlp2d_input', 'mlp.fc1', 'mlp.fc2', 'mlp2d_restore'}
    assert {n.name: n.SerializeToString() for n in old.graph.node if n.name not in changed} == {
        n.name: n.SerializeToString() for n in new.graph.node if n.name not in introduced}
    arrays = {tensor.name: numpy_helper.to_array(tensor) for tensor in new.graph.initializer}
    for name, value in weights.items():
        key = name + '_transposed' if name in ('attn.qkv.weight', 'attn.proj.weight', 'mlp.fc1.weight', 'mlp.fc2.weight') else name
        np.testing.assert_array_equal(arrays[key], value.T if key.endswith('_transposed') else value)
    assert len(weights) == 18
    assert len(new.graph.initializer) == len(old.graph.initializer) + 2
    np.testing.assert_array_equal(arrays['mlp2d_flat_shape'], [shape[0] * shape[1], shape[2]])
    np.testing.assert_array_equal(arrays['mlp2d_restore_shape'], shape)
    counts = {kind: sum(n.op_type == kind for n in new.graph.node) for kind in ('MatMul', 'Gemm', 'Erf', 'Reshape')}
    assert counts == {'MatMul': 4, 'Gemm': 2, 'Erf': 1, 'Reshape': 4}
    for node in new.graph.node:
        if node.op_type == 'Gemm':
            assert {a.name: helper.get_attribute_value(a) for a in node.attribute} == {
                'alpha': 1., 'beta': 1., 'transA': 0, 'transB': 0}
    shapes = {value.name: tuple(d.dim_value for d in value.type.tensor_type.shape.dim)
              for value in list(new.graph.value_info) + list(new.graph.output)}
    assert shapes['mlp.fc1'] == shapes['gelu'] == (shape[0] * shape[1], shape[2] * 4)
    assert shapes['mlp.fc2'] == shapes['output'] == shape


@pytest.mark.parametrize('mutation,match', [
    ('pattern', 'MLP pattern'), ('consumer', 'unexpected consumer'),
    ('reserved', 'already exist'), ('dtype', 'FP32 MLP'), ('dynamic', 'rank-three'),
])
def test_unknown_pattern_and_unsafe_shapes_are_rejected(mutation, match):
    from onnx import helper, numpy_helper
    old, _, _, _, _ = graphs()
    if mutation == 'pattern':
        next(n for n in old.graph.node if n.name == 'mlp.fc1').input.reverse()
    elif mutation == 'consumer':
        old.graph.node.append(helper.make_node('Identity', ['mlp.fc1.matmul'], ['extra'], name='extra'))
    elif mutation == 'reserved':
        old.graph.node.append(helper.make_node('Identity', ['output'], ['mlp2d_input'], name='extra'))
    elif mutation == 'dtype':
        for tensor in old.graph.initializer:
            if tensor.name == 'mlp.fc1.bias':
                tensor.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(tensor).astype(np.float16), tensor.name))
    elif mutation == 'dynamic':
        dim = old.graph.input[0].type.tensor_type.shape.dim[0]
        dim.ClearField('dim_value')
        dim.dim_param = 'batch'
    with pytest.raises(ValueError, match=match):
        exporter.rewrite_mlp2d(old)


def test_double_rewrite_is_rejected():
    _, new, *_ = graphs()
    with pytest.raises(ValueError, match='MLP pattern'):
        exporter.rewrite_mlp2d(new)


def test_loader_import_does_not_import_torch_onnx_or_ort():
    program = ('import sys,importlib.util; s=importlib.util.spec_from_file_location("v19",sys.argv[1]); '
               'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
               'assert all(k not in sys.modules for k in ("torch","onnx","onnxruntime")); '
               'assert m.PROFILE_SCRIPT_SHA and m.SOURCE_PINS and m.GRAPH_HELPERS')
    subprocess.run([sys.executable, '-B', '-c', program, str(EXPORT)], check=True)
