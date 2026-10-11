"""Only tiny synthetic tensors; never execute the captured DPVO encoder."""
import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
ENTRY = HERE / "rewrite-v27-dpvo-instance-norm.py"
if not ENTRY.exists():
    ENTRY = HERE.parent / "deploy/rk3588/validation/rewrite-v27-dpvo-instance-norm.py"
SPEC = importlib.util.spec_from_file_location("v27_norm_editor", ENTRY)
editor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = editor
SPEC.loader.exec_module(editor)


def tiny_model(shape=(2, 3, 5, 7), epsilon=1e-5, gamma=None, beta=None):
    channels = shape[1]
    gamma = np.linspace(.7, 1.3, channels, dtype=np.float32) if gamma is None else gamma
    beta = np.linspace(-.3, .2, channels, dtype=np.float32) if beta is None else beta
    nodes = [helper.make_node("Identity", ["x"], ["identity_x"], name="before"),
             helper.make_node("InstanceNormalization", ["identity_x", "gamma", "beta"], ["norm"], name="instance", epsilon=epsilon),
             helper.make_node("Identity", ["norm"], ["out"], name="after")]
    graph = helper.make_graph(nodes, "tiny_synthetic_only", [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
                              [helper.make_tensor_value_info("out", TensorProto.FLOAT, shape)],
                              [numpy_helper.from_array(gamma, name="gamma"), numpy_helper.from_array(beta, name="beta")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def run_tiny(model, x):
    ort = pytest.importorskip("onnxruntime", reason="requires ONNXRuntime for tiny synthetic normalization only")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    assert session.get_providers() == ["CPUExecutionProvider"]
    return session.run(["out"], {"x": x})[0]


@pytest.mark.parametrize("case", ["random", "constant_per_instance_channel", "small_variance", "affine_zero_scale", "hw_singleton"])
def test_tiny_original_rewrite_and_population_oracle(case):
    shape = (2, 3, 1, 1) if case == "hw_singleton" else (2, 3, 5, 7)
    rng = np.random.default_rng(27)
    x = rng.normal(size=shape).astype(np.float32)
    if case == "constant_per_instance_channel":
        x = np.broadcast_to(np.arange(6, dtype=np.float32).reshape(2, 3, 1, 1), shape).copy()
    if case == "small_variance":
        x = np.float32(.01) + x * np.float32(1e-6)
    gamma = np.asarray([0., 2., -1.], np.float32) if case == "affine_zero_scale" else None
    model = tiny_model(shape, gamma=gamma)
    rewritten, rows = editor.rewrite_instance_norm(model, expected_count=1)
    original, result = run_tiny(model, x), run_tiny(rewritten, x)
    constants = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    x64 = x.astype(np.float64)
    centered = x64 - x64.mean(axis=(2, 3), keepdims=True)
    eps = float(np.float32(1e-5))
    oracle = (centered / np.sqrt((centered * centered).mean(axis=(2, 3), keepdims=True) + eps) *
              constants["gamma"][None, :, None, None] + constants["beta"][None, :, None, None])
    assert rows[0]["new_ops"] == editor.RECIPE
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, oracle, rtol=editor.RTOL, atol=editor.ATOL)
    if case == "constant_per_instance_channel":
        # The original ORT kernel can evaluate x*scale+offset and cancel large
        # terms.  That is not bitwise-equivalent to centered subtraction.  The
        # static candidate must still equal the population-variance oracle;
        # the unchanged real-encoder gate remains pending on the board.
        expected_beta = np.broadcast_to(constants["beta"][None, :, None, None], shape)
        np.testing.assert_array_equal(result, expected_beta)
        assert np.isfinite(original).all()
    else:
        np.testing.assert_allclose(result, original, rtol=editor.RTOL, atol=editor.ATOL)


def test_byte_preservation_and_no_hidden_axis_transpose():
    model = tiny_model()
    before = model.SerializeToString()
    rewritten, rows = editor.rewrite_instance_norm(model, expected_count=1)
    assert model.SerializeToString() == before
    proof = editor.validate_preservation(model, rewritten, rows)
    assert all(proof[k] for k in ("original_initializers_byte_exact", "non_instance_norm_nodes_byte_exact",
                                  "input_output_protobuf_byte_exact", "model_header_byte_exact"))
    assert proof["added_nonlearned_constants"] == 2
    assert not any(n.op_type in ("Transpose", "ReduceMean", "InstanceNormalization") for n in rewritten.graph.node)
    assert sum(n.op_type == "GlobalAveragePool" for n in rewritten.graph.node) == 2
    assert rows[0]["epsilon_fp32"]["bytes"] == 4


def test_original_constant_node_affine_is_preserved():
    model = tiny_model()
    affine = [helper.make_node("Constant", [], [t.name], name="constant_" + t.name, value=t)
              for t in model.graph.initializer]
    del model.graph.initializer[:]
    saved = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(affine + saved)
    rewritten, rows = editor.rewrite_instance_norm(model, expected_count=1)
    assert [n.SerializeToString() for n in rewritten.graph.node[:2]] == [n.SerializeToString() for n in affine]
    assert rows[0]["scale"]["shape"] == [3]


@pytest.mark.parametrize("change,match", [
    ("count", "count"), ("collision", "collides"), ("epsilon", "epsilon"),
    ("dynamic", "fixed"), ("rank", "NCHW"), ("external", "External"),
    ("opset", "opset"), ("affine_nan", "finite"), ("preservation", "preservation"),
])
def test_reject_recipe_mutations(change, match):
    model = tiny_model()
    if change == "count":
        with pytest.raises(ValueError, match=match):
            editor.rewrite_instance_norm(model, expected_count=10)
        return
    if change == "collision":
        model.graph.node[0].name = "v27_in_000/mean"
    elif change == "epsilon":
        model.graph.node[1].attribute[0].f = float("nan")
    elif change == "dynamic":
        model.graph.input[0].type.tensor_type.shape.dim[0].ClearField("dim_value")
        model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch"
        model.graph.output[0].type.tensor_type.shape.dim[0].ClearField("dim_value")
        model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = "batch"
    elif change == "rank":
        model = tiny_model(shape=(2, 3, 5))
    elif change == "external":
        model.graph.initializer[0].data_location = TensorProto.EXTERNAL
        kv = model.graph.initializer[0].external_data.add()
        kv.key, kv.value = "location", "untrusted.bin"
    elif change == "opset":
        model.opset_import[0].version = 18
    elif change == "affine_nan":
        model.graph.initializer[0].CopyFrom(numpy_helper.from_array(np.asarray([1., np.nan, 1.], np.float32), name="gamma"))
    elif change == "preservation":
        rewritten, rows = editor.rewrite_instance_norm(model, expected_count=1)
        broken = copy.deepcopy(rewritten)
        broken.graph.node[-1].name = "changed_after"
        with pytest.raises(ValueError, match=match):
            editor.validate_preservation(model, broken, rows)
        return
    with pytest.raises(ValueError, match=match):
        editor.rewrite_instance_norm(model, expected_count=1)


def test_cli_rejects_wrong_pinned_parent_before_loading(tmp_path):
    graph, extractor = tmp_path / "toy.onnx", tmp_path / "extractor.py"
    onnx.save(tiny_model(), graph)
    extractor.write_text("not original\n")
    output = tmp_path / "new"
    with pytest.raises(ValueError, match="frozen v26"):
        editor.main(["--onnx", str(graph), "--onnx-sha256", editor.sha256_file(graph),
                     "--extractor", str(extractor), "--output-dir", str(output)])
    assert not output.exists()
