"""Tiny random ONNX/ORT only; no real weights, frames or full-model forward."""
import copy
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T, helper, numpy_helper
import onnxruntime as ort
import pytest


HERE = Path(__file__).resolve().parent
IMPL = HERE if (HERE / "rewrite-v29-dpvo-group-max.py").exists() else HERE.parent / "deploy/rk3588/validation"
spec = importlib.util.spec_from_file_location("_v29_tiny_graph_editor", IMPL / "rewrite-v29-dpvo-group-max.py")
EDITOR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(EDITOR)


def toy(batch=1, groups=4, slots=5, channels=3):
    shape = [batch, groups, slots, channels]
    mask_shape = [batch, groups, slots, 1]
    nodes, inputs, outputs, initializers = [], [], [], []
    for number, (constant, where) in enumerate(EDITOR.TARGETS):
        prefix = "toy" + str(number)
        logits, values, valid = (prefix + x for x in ("_logits", "_values", "_valid"))
        inputs += [helper.make_tensor_value_info(logits, T.FLOAT, shape),
                   helper.make_tensor_value_info(values, T.FLOAT, shape),
                   helper.make_tensor_value_info(valid, T.FLOAT, mask_shape)]
        def init(name, array):
            initializers.append(numpy_helper.from_array(array, prefix + name))
            return prefix + name
        scalar_zero = init("_zero", np.array(0, dtype=np.float32))
        axes = init("_axis2", np.array([2], dtype=np.int64))
        group_zero = init("_group_zero", np.zeros([batch, groups, 1, channels], dtype=np.float32))
        full_zero = init("_full_zero", np.zeros(shape, dtype=np.float32))
        filler = constant + "_output_0"
        nodes.append(helper.make_node("Constant", [], [filler], name=constant,
                     value=numpy_helper.from_array(np.full(shape, -np.inf, dtype=np.float32))))
        def node(op, names, suffix, **attrs):
            result = prefix + suffix
            nodes.append(helper.make_node(op, names, [result], name=result, **attrs))
            return result
        present = node("Greater", [valid, scalar_zero], "_present")
        count = node("ReduceSum", [valid, axes], "_count", keepdims=1)
        nonempty = node("Greater", [count, scalar_zero], "_nonempty")
        chosen = prefix + "_chosen"
        nodes.append(helper.make_node("Where", [present, logits, filler], [chosen], name=where))
        maximum = node("ReduceMax", [chosen], "_maximum", axes=[2], keepdims=1)
        maximum = node("Where", [nonempty, maximum, group_zero], "_safe_maximum")
        diff = node("Sub", [logits, maximum], "_centered_before_mask")
        centered = node("Where", [present, diff, full_zero], "_centered")
        exponential = node("Exp", [centered], "_exp")
        exponential = node("Mul", [exponential, valid], "_masked_exp")
        denominator = node("ReduceSum", [exponential, axes], "_sum", keepdims=1)
        empty = node("Not", [nonempty], "_empty")
        empty = node("Cast", [empty], "_empty_float", to=T.FLOAT)
        denominator = node("Add", [denominator, empty], "_denominator")
        weights = node("Div", [exponential, denominator], "_weights")
        multiplied = node("Mul", [values, weights], "_weighted_values")
        output = node("ReduceSum", [multiplied, axes], "_aggregate", keepdims=0)
        outputs.append(helper.make_tensor_value_info(output, T.FLOAT, [batch, groups, channels]))
    graph = helper.make_graph(nodes, "tiny_only_group_max", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    return model


def session(model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def data(model, counts, scale=1.0, seed=5):
    rng = np.random.default_rng(seed)
    feed = {}
    for row in model.graph.input:
        shape = tuple(x.dim_value for x in row.type.tensor_type.shape.dim)
        if row.name.endswith("_valid"):
            mask = np.zeros(shape, dtype=np.float32)
            for group, count in enumerate(counts):
                mask[:, group, :count] = 1
            feed[row.name] = mask
        else:
            feed[row.name] = (rng.standard_normal(shape).astype(np.float32)
                              * np.float32(scale if row.name.endswith("_logits") else 1.0))
    return feed


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("counts", [(5, 5, 5, 5), (1, 3, 5, 0), (0, 0, 0, 0)])
@pytest.mark.parametrize("scale", [1.0, 1000.0, 1e30])
def test_random_padding_empty_extreme_finite_logits(batch, counts, scale):
    original = toy(batch=batch)
    rewritten, report = EDITOR.rewrite_model(original)
    feed = data(original, counts, scale)
    for value in feed.values():
        assert np.isfinite(value).all()
    expected = session(original).run(None, feed)
    actual = session(rewritten).run(None, feed)
    for a, b in zip(actual, expected):
        assert np.isfinite(a).all() and np.isfinite(b).all()
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)
        # Same exact CPU graph apart from the two max-padding operands.
        assert a.tobytes() == b.tobytes()
    assert report["remaining_nonfinite_constants"] == 0


def test_same_graph_runtime_mask_changes_and_channelwise_maxima():
    original = toy()
    rewritten, _ = EDITOR.rewrite_model(original)
    baseline, candidate = session(original), session(rewritten)
    for counts in [(1, 3, 5, 0), (5, 1, 0, 2), (1, 3, 5, 0)]:
        feed = data(original, counts)
        for name, value in feed.items():
            if name.endswith("_logits"):
                value[..., 0] = np.arange(5, dtype=np.float32)
                value[..., 1] = -np.arange(5, dtype=np.float32)
                value[..., 2] = np.array([1e30, -1e30, 4, -5, 6], dtype=np.float32)
        for a, b in zip(candidate.run(None, feed), baseline.run(None, feed)):
            assert np.isfinite(a).all() and a.tobytes() == b.tobytes()


def test_max_finite_float32_values_with_synthetic_ORT():
    original = toy()
    candidate, _ = EDITOR.rewrite_model(original)
    feed = data(original, (2, 1, 0, 5))
    limit = np.finfo(np.float32).max
    for name, value in feed.items():
        if name.endswith("_logits"):
            value[..., 0] = -limit
            value[:, :, 0, 1] = limit
            value[:, :, 1:, 1] = -limit
    for a, b in zip(session(candidate).run(None, feed), session(original).run(None, feed)):
        assert np.isfinite(a).all() and a.tobytes() == b.tobytes()


def test_replacement_reads_same_group_first_member_not_global_edge_zero():
    original = toy(groups=4, slots=5, channels=3)
    candidate, report = EDITOR.rewrite_model(original)
    expose = copy.deepcopy(candidate)
    for binding in report["bindings"]:
        expose.graph.output.append(helper.make_tensor_value_info(
            binding["output_name_preserved"], T.FLOAT, binding["original_shape"]))
    feed = data(original, (1, 2, 0, 5))
    for name, value in feed.items():
        if name.endswith("_logits"):
            value[:, :, 0, :] = np.arange(4, dtype=np.float32)[None, :, None] + 20
    outputs = session(expose).run(None, feed)
    for binding, actual in zip(report["bindings"], outputs[-2:]):
        expected = np.broadcast_to(feed[binding["logits"]][:, :, :1, :], actual.shape)
        assert actual.tobytes() == expected.tobytes()


def test_other_nodes_initializers_header_and_io_bytes_preserved():
    original = toy()
    before = original.SerializeToString()
    candidate, report = EDITOR.rewrite_model(original)
    assert original.SerializeToString() == before
    assert report["original_non_target_nodes_and_order_unchanged"]
    assert report["all_original_initializers_bytes_and_order_unchanged"]
    assert report["model_header_inputs_outputs_value_info_unchanged"]
    assert len(candidate.graph.node) == len(original.graph.node) + 2
    assert len(candidate.graph.initializer) == len(original.graph.initializer) + 3
    for old, new in zip(original.graph.initializer, candidate.graph.initializer):
        assert hashlib.sha256(old.SerializeToString()).digest() == hashlib.sha256(new.SerializeToString()).digest()


@pytest.mark.parametrize("mask", [np.array([[0, 1, 0]]), np.array([[1, 0, 1]]),
                                  np.array([[1, .5, 0]]), np.array([])])
def test_reject_topologies_outside_prefix_contract(mask):
    with pytest.raises(ValueError, match="prefix|precede|binary|nonempty"):
        EDITOR.validate_prefix_mask(mask)


@pytest.mark.parametrize("mutation", ["finite_sentinel", "nan", "extra_use", "collision", "shape"])
def test_editor_rejects_unexpected_source_bindings(mutation):
    model = toy()
    constant = next(x for x in model.graph.node if x.name == "/Constant_4")
    if mutation in ("finite_sentinel", "nan"):
        value = -1e9 if mutation == "finite_sentinel" else np.nan
        constant.attribute[0].t.CopyFrom(numpy_helper.from_array(np.full([1, 4, 5, 3], value, dtype=np.float32)))
    elif mutation == "extra_use":
        model.graph.node.append(helper.make_node("Identity", [constant.output[0]], ["unused"], name="extra_use"))
    elif mutation == "collision":
        model.graph.initializer.append(numpy_helper.from_array(np.array([0], dtype=np.int64), "v29_group_max_first_slot_index"))
    else:
        constant.attribute[0].t.CopyFrom(numpy_helper.from_array(np.full([1, 1, 5, 3], -np.inf, dtype=np.float32)))
    with pytest.raises((ValueError, onnx.checker.ValidationError, onnx.shape_inference.InferenceError)):
        EDITOR.rewrite_model(model)
