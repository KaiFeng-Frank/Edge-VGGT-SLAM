"""Tiny generated ONNX graphs and NumPy expressions only; no model execution."""
import importlib.util
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("reciprocal_rewriter", HERE / "rewrite-v38-spatial-reciprocal.py")
editor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(editor)


def toy_graph():
    nodes, initializers, inputs, outputs = [], [], [], []
    for i in range(10):
        p = f"v27_in_{i:03d}"
        c, h, w = (32, 232, 304) if i < 5 else (64, 116, 152)
        inputs.append(helper.make_tensor_value_info(f"x{i}", TensorProto.FLOAT, [1, c, h, w]))
        initializers.append(numpy_helper.from_array(np.float32(1e-5), f"{p}/epsilon"))
        nodes += [
            helper.make_node("GlobalAveragePool", [f"x{i}"], [f"{p}/mean"], name=f"{p}/node_00_GlobalAveragePool"),
            helper.make_node("Sub", [f"x{i}", f"{p}/mean"], [f"{p}/centered"], name=f"{p}/node_01_Sub"),
            helper.make_node("Mul", [f"{p}/centered"] * 2, [f"{p}/square"], name=f"{p}/node_02_Mul"),
            helper.make_node("GlobalAveragePool", [f"{p}/square"], [f"{p}/variance"], name=f"{p}/node_03_GlobalAveragePool"),
            helper.make_node("Add", [f"{p}/variance", f"{p}/epsilon"], [f"{p}/variance_eps"], name=f"{p}/node_04_Add"),
            helper.make_node("Sqrt", [f"{p}/variance_eps"], [f"{p}/denominator"], name=f"{p}/node_05_Sqrt"),
            helper.make_node("Div", [f"{p}/centered", f"{p}/denominator"], [f"{p}/normalized"], name=f"{p}/node_06_Div"),
        ]
        outputs.append(helper.make_tensor_value_info(f"{p}/normalized", TensorProto.FLOAT, [1, c, h, w]))
    for i, name in enumerate(editor.OTHER_DIV_NAMES):
        nodes.append(helper.make_node("Div", [f"x{i}", "unrelated_scalar"], [f"aux{i}"], name=name))
        c, h, w = (32, 232, 304)
        outputs.append(helper.make_tensor_value_info(f"aux{i}", TensorProto.FLOAT, [1, c, h, w]))
    initializers.append(numpy_helper.from_array(np.float32(4.0), "unrelated_scalar"))
    for i in range(54):
        initializers.append(numpy_helper.from_array(np.array([i], dtype=np.float32), f"unused_{i}"))
    g = helper.make_graph(nodes, "tiny_static_shapes_no_real_weights", inputs, outputs, initializers)
    g.doc_string = "Preserve graph metadata byte-for-byte"
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], producer_name="tiny-test")
    m.ir_version = 8
    m.metadata_props.add(key="test", value="unchanged")
    return m


def test_exact_structural_rewrite_and_serialized_revalidation():
    parent = toy_graph()
    before = parent.SerializeToString()
    candidate, proof = editor.rewrite(parent)
    assert parent.SerializeToString() == before
    roundtrip = onnx.load_from_string(candidate.SerializeToString())
    assert editor.validate_preservation(parent, roundtrip) == proof
    assert all(value is True for key, value in proof.items() if not key.endswith("count"))
    assert len(candidate.graph.initializer) == 66
    assert sum(n.op_type == "Div" for n in candidate.graph.node) == 12
    assert [n.SerializeToString() for n in candidate.graph.node if n.name in editor.OTHER_DIV_NAMES] == [
        n.SerializeToString() for n in parent.graph.node if n.name in editor.OTHER_DIV_NAMES]
    for row in editor.selected_replacements(parent):
        assert row["denominator_shape"][2:] == [1, 1]


@pytest.mark.parametrize("mutation", ["initializer", "epsilon", "unrelated_node", "header", "output", "one"])
def test_candidate_mutations_are_rejected(mutation):
    parent = toy_graph()
    candidate, _ = editor.rewrite(parent)
    if mutation == "initializer":
        candidate.graph.initializer[20].raw_data = np.array([123], np.float32).tobytes()
    elif mutation == "epsilon":
        candidate.graph.initializer[0].raw_data = np.float32(2e-5).tobytes()
    elif mutation == "unrelated_node":
        next(n for n in candidate.graph.node if n.name == "/original/Div").name = "edited"
    elif mutation == "header":
        candidate.producer_name = "changed"
    elif mutation == "output":
        candidate.graph.output[0].doc_string = "changed"
    elif mutation == "one":
        candidate.graph.initializer[-1].raw_data = np.float32(0.5).tobytes()
    with pytest.raises(ValueError):
        editor.validate_preservation(parent, candidate)


@pytest.mark.parametrize("mutation", ["epsilon", "shape", "missing", "collision", "extra_div"])
def test_parent_contract_mutations_are_rejected(mutation):
    parent = toy_graph()
    if mutation == "epsilon":
        parent.graph.initializer[0].raw_data = np.float32(2e-5).tobytes()
    elif mutation == "shape":
        parent.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 16
        parent.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 16
        parent.graph.output[-2].type.tensor_type.shape.dim[1].dim_value = 16
    elif mutation == "missing":
        next(n for n in parent.graph.node if n.name == editor.SELECTED_NAMES[0]).name = "different"
    elif mutation == "collision":
        parent.graph.initializer[-1].name = editor.ONE_NAME
    elif mutation == "extra_div":
        parent.graph.node.append(helper.make_node("Div", ["x0", "unrelated_scalar"], ["third_div"], name="third_div"))
    with pytest.raises(ValueError):
        editor.rewrite(parent)


@pytest.mark.parametrize("kind", ["constant", "low_variance", "nonzero", "zero"])
def test_tiny_reciprocal_expression_retains_epsilon_but_not_bitwise_claim(kind):
    rng = np.random.default_rng(11)
    x = rng.normal(size=(1, 3, 4, 5)).astype(np.float32)
    if kind == "constant":
        x.fill(5.0)
    elif kind == "low_variance":
        x = np.float32(0.1) + x * np.float32(1e-6)
    elif kind == "zero":
        x.fill(0.0)
    centered = x - x.mean(axis=(2, 3), keepdims=True, dtype=np.float32)
    variance = (centered * centered).mean(axis=(2, 3), keepdims=True, dtype=np.float32)
    epsilon = np.float32(1e-5)
    denominator = np.sqrt(variance + epsilon)
    direct = centered / denominator
    reciprocal = centered * (np.float32(1.0) / denominator)
    assert denominator.shape == (1, 3, 1, 1)
    assert direct.shape == reciprocal.shape == x.shape
    assert np.isfinite(reciprocal).all()
    assert epsilon.tobytes() == editor.EPSILON_BYTES
    np.testing.assert_allclose(reciprocal, direct, rtol=1e-4, atol=1e-5)


def test_rounding_difference_is_not_silently_claimed_bitwise():
    x = np.array([0.1, 1.3, -0.7, 17.0], dtype=np.float32)
    d = np.array([0.3, 0.13, 1.7, 3.1], dtype=np.float32)
    assert np.any((x / d).view(np.uint32) != (x * (np.float32(1) / d)).view(np.uint32))


def test_ancestor_precision_rejection_cannot_be_erased():
    recipe = {"schema": "v33_DPVO_encoder_sparse_context_rewrite", "status": "statically_checked_not_runtime_validated",
              "candidate_accepted": False, "output": {"sha256": editor.PARENT_SHA256},
              "sources_and_parent_unchanged": True, "source_pins": {"source": "sha"}, "source_hashes_after": {"source": "sha"}}
    reference = {"schema": "v33_DPVO_sparse_context_board_CPU_ORT",
                 "status": "completed_suffix_bitwise_gate_parent_rejection_retained_not_VO",
                 "receipt_sha256": {"spatial_sparse": editor.PARENT_RECIPE_SHA256},
                 "original_CPU_strict_gate_each_method": {"spatial": False},
                 "historical_spatial_rejection": {"exact_FP32_gate_passed": False, "retained": True}}
    editor.validate_ancestors(recipe, reference)
    reference["original_CPU_strict_gate_each_method"]["spatial"] = True
    with pytest.raises(ValueError, match="rejection"):
        editor.validate_ancestors(recipe, reference)


def test_cli_rejects_unpinned_toy_and_retains_failed_receipt(tmp_path):
    parent = tmp_path / "toy.onnx"
    onnx.save(toy_graph(), parent)
    recipe = tmp_path / "recipe.json"
    reference = tmp_path / "reference.json"
    recipe.write_text("{}")
    reference.write_text("{}")
    out = tmp_path / "output"
    assert editor.main(["--parent", str(parent), "--parent-recipe", str(recipe),
                        "--parent-reference", str(reference), "--output-dir", str(out)]) == 1
    import json
    result = json.loads((out / "rewrite.json").read_text())
    assert result["candidate_accepted"] is False
    assert result["historical_parent_FP32_gate_passed"] is False
    assert not (out / "encoders-reciprocal.onnx").exists()
    assert result["sources_and_parent_unchanged"] is True
