"""Generated dimension-small graphs only; never execute captured VGGT tensors."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as H, numpy_helper as N, TensorProto as T
from onnx.reference import ReferenceEvaluator
import pytest

SCRIPT = Path(os.environ.get('VGGT_PAIR_COMPOSER',
              Path(__file__).with_name('export-v47-frame-global-pair.py')))
spec = importlib.util.spec_from_file_location('v47_pair', SCRIPT)
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def make_frame():
    shape = [2, 3, 4]
    weights = np.arange(16, dtype=np.float32).reshape(4, 4) / np.float32(31)
    bias = np.arange(4, dtype=np.float32) / np.float32(5)
    nodes = [H.make_node('MatMul', ['input', 'weight'], ['linear'], name='linear'),
             H.make_node('Add', ['linear', 'bias'], ['output'], name='output')]
    graph = H.make_graph(nodes, 'frame', [H.make_tensor_value_info('input', T.FLOAT, shape)],
                         [H.make_tensor_value_info('output', T.FLOAT, shape)],
                         [N.from_array(weights, 'weight'), N.from_array(bias, 'bias')],
                         value_info=[H.make_tensor_value_info('linear', T.FLOAT, shape)])
    return H.make_model(graph, opset_imports=[H.make_opsetid('', 17)], ir_version=8)


def make_global():
    shape = [1, 6, 4]
    nodes = [H.make_node('Mul', ['input', 'weight'], ['query'], name='query'),
             H.make_node('Transpose', ['query'], ['key'], name='key', perm=[0, 2, 1]),
             H.make_node('MatMul', ['query', 'key'], ['score'], name='score'),
             H.make_node('Softmax', ['score'], ['probability'], name='probability', axis=-1),
             H.make_node('MatMul', ['probability', 'input'], ['attention'], name='attention'),
             H.make_node('Add', ['input', 'attention'], ['output'], name='output')]
    graph = H.make_graph(nodes, 'global', [H.make_tensor_value_info('input', T.FLOAT, shape)],
                         [H.make_tensor_value_info('output', T.FLOAT, shape)],
                         [N.from_array(np.float32(.125), 'weight')])
    return H.make_model(graph, opset_imports=[H.make_opsetid('', 17)], ir_version=8)


def test_tiny_sequential_math_and_order():
    f, g = make_frame(), make_global()
    pair, proof = C.compose_pair(f, g)
    rgb_features = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / np.float32(29)
    frame = ReferenceEvaluator(f).run(None, {'input': rgb_features})[0]
    expected = ReferenceEvaluator(g).run(None, {'input': frame.reshape(1, 6, 4)})[0]
    actual = ReferenceEvaluator(pair).run(None, {'input': rgb_features})[0]
    assert np.array_equal(actual, expected)
    assert proof['parent_initializer_protobuf_preserved_except_names']
    assert [v.name for v in pair.graph.input] == ['input']
    assert [v.name for v in pair.graph.output] == ['output']
    assert {v.name for v in pair.graph.initializer} == {
        'frame0/weight', 'frame0/bias', 'global0/weight', 'frame_global_shape'}
    # Swapping token and frame axes is a wrong bridge despite the same shape.
    wrong = ReferenceEvaluator(g).run(None, {'input': frame.transpose(1, 0, 2).reshape(1, 6, 4)})[0]
    assert not np.array_equal(actual, wrong)


def test_preservation_does_not_mutate_parents():
    f, g = make_frame(), make_global()
    before = f.SerializeToString(), g.SerializeToString()
    pair, _ = C.compose_pair(f, g)
    assert (f.SerializeToString(), g.SerializeToString()) == before
    assert sum(n.op_type == 'Reshape' for n in pair.graph.node) == 1
    assert len(pair.graph.node) == len(f.graph.node) + len(g.graph.node) + 1


@pytest.mark.parametrize('change', ['node', 'initializer', 'bridge', 'output'])
def test_changed_pair_rejected(change):
    f, g = make_frame(), make_global()
    pair, _ = C.compose_pair(f, g)
    if change == 'node':
        pair.graph.node[0].op_type = 'Add'
    elif change == 'initializer':
        pair.graph.initializer[0].raw_data = b'\0' * len(pair.graph.initializer[0].raw_data)
    elif change == 'bridge':
        pair.graph.node[len(f.graph.node)].input[0] = 'input'
    else:
        pair.graph.output[0].name = 'frame0/output'
    with pytest.raises(ValueError):
        C.validate_preservation(f, g, pair)


@pytest.mark.parametrize('change', ['shape', 'opset', 'external', 'symbolic'])
def test_unsupported_parent_rejected(change):
    f, g = make_frame(), make_global()
    if change == 'shape':
        for value in [g.graph.input[0], g.graph.output[0]]:
            value.type.tensor_type.shape.dim[1].dim_value = 5
    elif change == 'opset':
        g.opset_import[0].version = 18
    elif change == 'external':
        f.graph.initializer[0].data_location = T.EXTERNAL
    else:
        f.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'batch'
    with pytest.raises((ValueError, onnx.checker.ValidationError, onnx.shape_inference.InferenceError)):
        C.compose_pair(f, g)


def test_zero_finite_meta_and_nonfinite_guard():
    value = np.zeros((1, 3, 4), np.float32)
    assert C.array_metadata(value)['bytes'] == 48
    value[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        C.array_metadata(value)
