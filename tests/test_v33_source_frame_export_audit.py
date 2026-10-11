"""Pure NumPy/static-ONNX toy checks. No model or runtime execution."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


_spec = importlib.util.spec_from_file_location(
    "source_frame_export_audit", (Path(__file__).resolve().parents[1] / "deploy" / "rk3588" / "validation" / "dpvo_source_frame") / "audit-v33-source-frame-export.py")
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def graph(patches=3):
    rows = [(source, target, source * patches + patch)
            for source, targets in ((2, (0, 5)), (0, (1, 3)))
            for patch in range(patches) for target in targets]
    permutation = np.random.default_rng(83).permutation(len(rows))
    ii, jj, kk = np.asarray(rows, dtype=np.int64)[permutation].T
    return dict(ii=ii, jj=jj, kk=kk)


def test_independent_minimum_source_and_permutation():
    values = graph()
    (plan, canonical), plans = audit.reconstruct_topology(values, audit.Checks(), patches=3)
    assert len(plans) == 2 and plan["source_slot"] == 0 and plan["degree"] == 2
    assert np.array_equal(values["jj"][canonical], np.repeat([1, 3], 3))
    assert np.array_equal(values["kk"][canonical], np.tile([0, 1, 2], 2))
    assert np.array_equal(np.sort(canonical), np.flatnonzero(values["ii"] == 0))


@pytest.mark.parametrize("mutation", ["duplicate_patch", "wrong_owner", "missing_edge", "fp64_indices"])
def test_topology_rejects_invalid_complete_source(mutation):
    values = graph()
    if mutation == "duplicate_patch":
        selected = np.flatnonzero((values["ii"] == 0) & (values["jj"] == 1))
        values["kk"][selected[0]] = values["kk"][selected[1]]
    elif mutation == "wrong_owner":
        values["kk"][0] += 30
    elif mutation == "missing_edge":
        values = {name: value[:-1].copy() for name, value in values.items()}
    else:
        values["ii"] = values["ii"].astype(np.float64)
    with pytest.raises(ValueError):
        audit.reconstruct_topology(values, audit.Checks(), patches=3)


def test_fixed_element_gate_is_independent_of_finite_domain():
    reference = np.asarray([0.0, 1.0], dtype=np.float32)
    candidate = np.asarray([0.000011, 1.0], dtype=np.float32)
    values = audit.residuals(candidate, reference)
    assert values["failed_elements"] == 1 and values["gate_passed"] is False
    assert values["max_normalized_tolerance"] > 1
    assert np.isfinite(candidate).all()


def test_raw_hash_tamper_rejected_even_with_valid_file_hash(tmp_path):
    value = np.arange(4, dtype=np.float32)
    path = tmp_path / "toy.npz"
    np.savez(path, net=value)
    item = dict(file=path.name, bytes=path.stat().st_size, sha256=audit.file_sha(path), arrays={"net": audit.metadata(value)})
    assert np.array_equal(audit.load_artifact(tmp_path, item, audit.Checks())["net"], value)
    item["arrays"]["net"]["tensor_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Raw array"):
        audit.load_artifact(tmp_path, item, audit.Checks())


def test_artifact_path_escape_rejected(tmp_path):
    with pytest.raises(ValueError, match="basename"):
        audit.load_artifact(tmp_path, {"file": "../outside.npz"}, audit.Checks())


def test_initializer_transpose_binding_and_wrong_mapping_rejection():
    import onnx
    original = np.arange(6, dtype=np.float32).reshape(2, 3)
    initializer = onnx.numpy_helper.from_array(original.T.copy(), name="matmul_weight")
    model = onnx.helper.make_model(onnx.helper.make_graph([], "toy", [], [], [initializer]))
    parameters = {"linear.weight": audit.metadata(original)}
    row = dict(initializer="matmul_weight", array=audit.metadata(original.T),
               original_matches=[dict(parameter="linear.weight", transform="transpose")])
    coverage = dict(original_tensor_count=1, initializer_count=1, all_original_parameters_covered=True, mapping=[row])
    assert audit.verify_initializers(model, parameters, coverage, audit.Checks(), onnx)["all_covered"]
    row["original_matches"][0]["transform"] = "identity"
    with pytest.raises(ValueError, match="mapping"):
        audit.verify_initializers(model, parameters, coverage, audit.Checks(), onnx)


def test_recorded_gate_cannot_hide_a_failed_element():
    actual = audit.residuals(np.asarray([1.0], dtype=np.float32), np.asarray([0.0], dtype=np.float32))
    recorded = {key: value for key, value in actual.items() if key != "max_normalized_tolerance"}
    recorded.update(failed_elements=0, gate_passed=True)
    with pytest.raises(ValueError):
        audit.check_recorded_residual(actual, recorded, audit.Checks())


def test_nonfinite_array_is_not_scored_as_pass():
    with pytest.raises(ValueError, match="Finite"):
        audit.residuals(np.asarray([np.nan], dtype=np.float32), np.asarray([0.0], dtype=np.float32))
