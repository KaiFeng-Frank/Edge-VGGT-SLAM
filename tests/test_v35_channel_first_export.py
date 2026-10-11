"""Small CPU toy graph export/static checks; no actual DPVO model or ORT."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "deploy/rk3588/validation/dpvo_source_frame_channel_first"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


driver = load(HERE / "export-v35-source-frame-channel-first-board.py", "v35_export_toy")


def test_layout_roundtrip_is_exact_owned_contiguous():
    original = {"net": np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4),
                "delta": np.arange(12, dtype=np.float32).reshape(1, 2, 3, 2)}
    packed = driver.pack_channel_first(original)
    restored = driver.restore_canonical(packed)
    for name, value in original.items():
        assert packed[name].shape == (1, value.shape[3], 2, 3)
        assert packed[name].flags.c_contiguous and restored[name].flags.c_contiguous
        assert np.array_equal(restored[name], value)
        assert not np.shares_memory(packed[name], value)


def test_host_refusal_precedes_inputs_or_output_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(driver.platform, "system", lambda: "Linux")
    monkeypatch.setattr(driver.platform, "machine", lambda: "x86_64")
    args = []
    for flag in ("v26-source-dir", "checkpoint", "reference-dir", "parent-source",
                 "utility-source", "wrapper-source", "output-dir"):
        args.extend(("--" + flag, str(tmp_path / flag)))
    args.extend(("--wrapper-sha256", "0" * 64))
    with pytest.raises(ValueError, match="RK3588-only"):
        driver.main(args)
    assert not (tmp_path / "output-dir").exists()


def test_explicit_output_contract_rejects_canonical_layout():
    class Utility:
        @staticmethod
        def require_outputs(values, degree, patches):
            assert values["net_out"].shape == (1, degree, patches, 384)
            assert values["delta"].shape == (1, degree, patches, 2)
            assert np.isfinite(values["net_out"]).all()
            assert ((values["weight"] >= 0) & (values["weight"] <= 1)).all()
            return {"finite": True, "domain": True}

    values = {name: np.zeros((1, channels, 2, 3), np.float32)
              for name, channels in zip(driver.OUTPUTS, (384, 2, 2))}
    assert driver.require_cf_outputs(values, Utility, 2, 3)["finite"]
    with pytest.raises(ValueError, match="shape"):
        driver.require_cf_outputs(driver.restore_canonical(values), Utility, 2, 3)


def test_tiny_torch_export_keeps_cf_graph_and_original_initializer_bytes(tmp_path):
    import hashlib
    import inspect
    from collections import Counter
    import onnx
    import torch

    toy = load(Path(__file__).resolve().parent / "test_v35_source_frame_channel_first.py", "v35_toy_export_fixture")
    original = toy.ToyUpdate()
    wrapped = toy.cf.SourceFrameUpdateChannelFirst(original)
    _, inputs = toy.inputs(1, 3, 2)
    path = tmp_path / "tiny.onnx"
    kwargs = dict(input_names=list(driver.INPUTS), output_names=list(driver.OUTPUTS),
                  opset_version=17, do_constant_folding=True)
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    with torch.no_grad():
        torch.onnx.export(wrapped, inputs, str(path), **kwargs)
    graph = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(graph, full_check=True)
    counts = Counter(n.op_type for n in graph.graph.node)
    assert counts["Conv"] == 21 and counts["ReduceMean"] == 8 and counts["Softmax"] == 2
    forbidden = {"Transpose", "MatMul", "Gemm", "Gather", "GatherElements", "GatherND", "Where", "Pad"}
    assert not forbidden & set(counts)
    for node in graph.graph.node:
        attributes = {v.name: onnx.helper.get_attribute_value(v) for v in node.attribute}
        if node.op_type == "ReduceMean":
            assert attributes["axes"] == [1]
    softmax_axes = sorted(onnx.helper.get_attribute_value(a)
                          for node in graph.graph.node if node.op_type == "Softmax"
                          for a in node.attribute if a.name == "axis")
    assert softmax_axes == [2, 3]
    assert [[d.dim_value for d in v.type.tensor_type.shape.dim] for v in graph.graph.input] == [
        [1, 4, 3, 2], [1, 4, 3, 2], [1, 7, 3, 2]]
    assert [[d.dim_value for d in v.type.tensor_type.shape.dim] for v in graph.graph.output] == [
        [1, 4, 3, 2], [1, 2, 3, 2], [1, 2, 3, 2]]

    class Utility:
        @staticmethod
        def meta(value):
            return {"shape": list(value.shape), "dtype": str(value.dtype),
                    "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}

    coverage = driver.initializer_coverage(graph, original.state_dict(), onnx, Utility)
    assert coverage["original_tensor_count"] == 50 and coverage["all_original_parameters_covered"]
    assert all(v["transform"] in coverage["permitted_transforms"]
               for item in coverage["mapping"] for v in item["original_matches"])
    graph.graph.initializer.append(onnx.numpy_helper.from_array(np.array([97.123], np.float32), "bad_weight"))
    with pytest.raises(ValueError, match="Unbound"):
        driver.initializer_coverage(graph, original.state_dict(), onnx, Utility)
