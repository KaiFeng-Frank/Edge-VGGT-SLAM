"""Pure NumPy/static ONNX audit guards; no model or ONNX runtime."""
import importlib.util
from pathlib import Path

import numpy as np
import onnx
import pytest


REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "deploy/rk3588/validation/dpvo_source_frame_channel_first"
spec = importlib.util.spec_from_file_location("v35_cf_saved_audit", HERE / "audit-v35-channel-first-export.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
helpers = audit.load_helpers(REPO / "deploy/rk3588/validation/dpvo_source_frame")


def test_cf_to_canonical_exact_bytes_guard():
    canonical = {k: np.arange(10 * 96 * c, dtype=np.float32).reshape(1, 10, 96, c)
                 for k, c in zip(audit.INPUTS, (384, 384, 882))}
    cf = {k: np.ascontiguousarray(v.transpose(0, 3, 1, 2)) for k, v in canonical.items()}
    audit.check_layout_pair(cf, canonical, inputs=True, checks=helpers.Checks())
    cf["net"][0, 0, 1, 0] += 1
    with pytest.raises(ValueError, match="inverse layout"):
        audit.check_layout_pair(cf, canonical, inputs=True, checks=helpers.Checks())


def test_original_initializer_exact_reshape_and_tamper_guard():
    kernel = np.arange(12, dtype=np.float32).reshape(3, 4)
    affine = np.array([41, 42, 43, 44], np.float32)
    originals = {"linear.weight": helpers.metadata(kernel), "norm.weight": helpers.metadata(affine)}
    arrays = [kernel.reshape(3, 4, 1, 1), affine.reshape(1, 4, 1, 1)]
    mapping = [dict(initializer="kernel", array=helpers.metadata(arrays[0]),
                    original_matches=[dict(parameter="linear.weight", transform="conv1x1_kernel_reshape")]),
               dict(initializer="gamma", array=helpers.metadata(arrays[1]),
                    original_matches=[dict(parameter="norm.weight", transform="channel_affine_broadcast_reshape")])]
    coverage = dict(permitted_transforms=audit.PERMITTED, parameter_values_unchanged=True,
                    original_tensor_count=2, initializer_count=2, mapping=mapping,
                    all_original_parameters_covered=True)
    graph = onnx.helper.make_graph([], "tiny", [], [],
                                  initializer=[onnx.numpy_helper.from_array(v, name)
                                               for v, name in zip(arrays, ("kernel", "gamma"))])
    model = onnx.helper.make_model(graph)
    result = audit.verify_initializers(model, originals, coverage, helpers.Checks(), helpers, onnx)
    assert result["parameters"] == 2 and result["all_covered"]
    mapping[0]["original_matches"][0]["transform"] = "identity"
    with pytest.raises(ValueError, match="reshape mapping"):
        audit.verify_initializers(model, originals, coverage, helpers.Checks(), helpers, onnx)


def test_rejection_is_legal_evidence_but_changed_verdict_is_not():
    reference = np.array([0, 1], np.float32)
    actual = np.array([0.001, 1], np.float32)
    computed = helpers.residuals(actual, reference)
    assert not computed["gate_passed"] and computed["failed_elements"] == 1
    recorded = {k: v for k, v in computed.items() if k != "max_normalized_tolerance"}
    audit.metric_check(computed, recorded, helpers.Checks())
    recorded["gate_passed"] = True
    with pytest.raises(ValueError, match="gate_passed"):
        audit.metric_check(computed, recorded, helpers.Checks())


def norm_graph(eps=1e-3):
    nodes = []
    for i in range(4):
        prefix = "norm" + str(i)
        nodes += [onnx.helper.make_node("ReduceMean", ["x"], [prefix + "mean"], axes=[1], keepdims=1),
                  onnx.helper.make_node("Sub", ["x", prefix + "mean"], [prefix + "centered"]),
                  onnx.helper.make_node("Mul", [prefix + "centered"] * 2, [prefix + "square"]),
                  onnx.helper.make_node("ReduceMean", [prefix + "square"], [prefix + "var"], axes=[1], keepdims=1),
                  onnx.helper.make_node("Constant", [], [prefix + "eps"],
                                        value=onnx.numpy_helper.from_array(np.array(eps, np.float32))),
                  onnx.helper.make_node("Add", [prefix + "var", prefix + "eps"], [prefix + "denom2"]),
                  onnx.helper.make_node("Sqrt", [prefix + "denom2"], [prefix + "denom"])]
    return onnx.helper.make_model(onnx.helper.make_graph(nodes, "tiny-norm", [], []))


def test_channel_norm_centered_population_variance_original_epsilon_guard():
    result = audit.verify_channel_norm_graph(norm_graph(), helpers.Checks(), onnx)
    assert result["norms"] == 4 and result["centered_population_variance"]
    with pytest.raises(ValueError, match="epsilon"):
        audit.verify_channel_norm_graph(norm_graph(0.02), helpers.Checks(), onnx)
