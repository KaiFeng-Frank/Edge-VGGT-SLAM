"""Tiny synthetic tensors only; never evaluate the real DPVO encoder."""
import copy
import importlib.util
import os
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def _load():
    entry = Path(__file__).with_name("rewrite-v33-dpvo-sparse-context.py")
    if not entry.exists():
        entry = Path(__file__).resolve().parents[1] / "deploy/rk3588/validation/dpvo_sparse_context/rewrite-v33-dpvo-sparse-context.py"
    spec = importlib.util.spec_from_file_location("v33_sparse_context", entry)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load()
IMAGE = [1, 3, 3, 5]
F = [1, 2, 3, 5]
I = [1, 4, 3, 5]
M = 3


def toy_model():
    # Constants are tiny and synthetic; original image input is unrelated to them.
    fmap = np.arange(np.prod(F), dtype=np.float32).reshape(F) / np.float32(4)
    imap = (np.arange(np.prod(I), dtype=np.float32).reshape(I) - np.float32(10)) / np.float32(4)
    graph = helper.make_graph([
        helper.make_node("Identity", ["f"], ["fmap"], name="original_f"),
        helper.make_node("Identity", ["i"], ["imap"], name="original_i"),
    ], "toy_only", [helper.make_tensor_value_info("normalized_bgr", TensorProto.FLOAT, IMAGE)],
        [helper.make_tensor_value_info("fmap", TensorProto.FLOAT, F),
         helper.make_tensor_value_info("imap", TensorProto.FLOAT, I)],
        [numpy_helper.from_array(fmap, name="f"), numpy_helper.from_array(imap, name="i")])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)


def candidate(original):
    return MODULE.append_sparse_context(original, image_shape=IMAGE, fmap_shape=F, imap_shape=I, patch_count=M)


@pytest.mark.parametrize("xy", [np.array([[1, 1], [2, 1], [3, 1]], dtype=np.int64),
                                np.array([[3, 1], [1, 1], [3, 1]], dtype=np.int64)])
def test_runtime_indices_order_and_duplicates_select_original_context(xy):
    original = toy_model()
    untouched = original.SerializeToString()
    index = MODULE.centers_to_indices(xy, feature_hw=(3, 5), patch_count=M)
    assert np.array_equal(index, xy[:, 1] * 5 + xy[:, 0])
    image = np.zeros(IMAGE, np.float32)
    fmap, imap = ReferenceEvaluator(original).run(None, {"normalized_bgr": image})
    new_fmap, context = ReferenceEvaluator(candidate(original)).run(None, {
        "normalized_bgr": image, "patch_indices": index})
    expected = imap[0, :, xy[:, 1], xy[:, 0]][None]
    assert context.shape == (1, M, 4)
    assert context.tobytes() == expected.tobytes()
    assert new_fmap.tobytes() == fmap.tobytes()
    assert original.SerializeToString() == untouched
    assert MODULE.preservation(original, candidate(original))["all_original_nodes_ordered_protobuf_byte_exact"]


def test_different_runtime_indices_are_not_baked_constants():
    session = ReferenceEvaluator(candidate(toy_model()))
    outputs = []
    for values in [[6, 7, 8], [8, 6, 8]]:
        outputs.append(session.run(None, {"normalized_bgr": np.zeros(IMAGE, np.float32),
            "patch_indices": np.asarray(values, np.int64)})[1])
    assert outputs[0].tobytes() != outputs[1].tobytes()
    assert np.array_equal(outputs[1][:, 0], outputs[0][:, 2])
    assert np.array_equal(outputs[1][:, 1], outputs[0][:, 0])
    assert np.array_equal(outputs[1][:, 2], outputs[0][:, 2])


def test_original_nchw_sampling_supports_negative_zero_and_noncontiguous_centers():
    xy = np.asarray([[3, 1], [1, 1], [3, 1]], dtype=np.int64)[::-1]
    values = MODULE.centers_to_indices(xy, feature_hw=(3, 5), patch_count=M)
    assert values.flags.c_contiguous and values.dtype == np.int64
    guarded = MODULE.validate_patch_indices(values, feature_hw=(3, 5), patch_count=M)
    guarded[0] = 6
    assert values[0] == 8


@pytest.mark.parametrize("bad", [
    np.ones((M, 2), np.float32), np.ones((M, 2), np.int32),
    np.ones((M, 3), np.int64), np.ones((M + 1, 2), np.int64),
    np.array([[0, 1], [1, 1], [2, 1]], np.int64),
    np.array([[4, 1], [1, 1], [2, 1]], np.int64),
    np.array([[1, 0], [1, 1], [2, 1]], np.int64),
    np.array([[1, 2], [1, 1], [2, 1]], np.int64),
])
def test_centers_reject_dtype_shape_and_original_border(bad):
    with pytest.raises(ValueError):
        MODULE.centers_to_indices(bad, feature_hw=(3, 5), patch_count=M)


@pytest.mark.parametrize("bad", [np.array([6, 7, 8], np.int32), np.array([6, 7], np.int64),
                                  np.array([-1, 7, 8], np.int64), np.array([15, 7, 8], np.int64),
                                  np.array([5, 7, 8], np.int64), np.array([9, 7, 8], np.int64)])
def test_index_guard_rejects_borders_and_wrong_layout(bad):
    with pytest.raises(ValueError):
        MODULE.validate_patch_indices(bad, feature_hw=(3, 5), patch_count=M)


def test_cannot_rewrite_unpinned_real_or_toy_graph(tmp_path):
    path = tmp_path / "toy.onnx"
    onnx.save_model(toy_model(), path)
    with pytest.raises(ValueError, match="two exact frozen"):
        MODULE.rewrite_pinned_graph(path, MODULE.sha256_file(path))


@pytest.mark.parametrize("collision", ["context", "patch_indices", MODULE.PREFIX + "Gather"])
def test_generated_names_collision_fails_before_modification(collision):
    original = toy_model()
    original.graph.node[0].name = collision
    before = original.SerializeToString()
    with pytest.raises(ValueError, match="collide"):
        candidate(original)
    assert original.SerializeToString() == before


@pytest.mark.parametrize("alter", ["opset", "dtype", "symbolic", "extra_output", "external"])
def test_graph_contract_rejects_malformed_parent(alter):
    original = toy_model()
    if alter == "opset":
        original.opset_import[0].version = 18
    elif alter == "dtype":
        original.graph.input[0].type.tensor_type.elem_type = TensorProto.DOUBLE
    elif alter == "symbolic":
        original.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "B"
    elif alter == "extra_output":
        original.graph.output.append(copy.deepcopy(original.graph.output[0]))
    elif alter == "external":
        original.graph.initializer[0].data_location = TensorProto.EXTERNAL
    with pytest.raises((ValueError, onnx.checker.ValidationError)):
        candidate(original)


def test_preservation_rejects_changed_recipe_or_original_tensor():
    original = toy_model()
    bad = candidate(original)
    bad.graph.node[-2].attribute[0].i = 1
    with pytest.raises(ValueError, match="preservation"):
        MODULE.preservation(original, bad)
    bad = candidate(original)
    bad.graph.initializer[0].raw_data = bytes(len(bad.graph.initializer[0].raw_data))
    with pytest.raises(ValueError, match="preservation"):
        MODULE.preservation(original, bad)


def test_cli_newdir_guard_and_source_pin_no_model_execution(tmp_path):
    with pytest.raises(FileExistsError):
        MODULE.main(["--onnx", str(tmp_path / "unused.onnx"), "--onnx-sha256", next(iter(MODULE.PARENTS)),
                     "--official-source-dir", str(tmp_path), "--output-dir", str(tmp_path)])
