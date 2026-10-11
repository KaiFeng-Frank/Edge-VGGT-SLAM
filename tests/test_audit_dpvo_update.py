"""Small NumPy tamper tests. No Torch/ONNX/ORT/checkpoint imports."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
IMPL = HERE if (HERE / "audit-v28-dpvo-update.py").is_file() else HERE.parent / "deploy/rk3588/validation"
spec = importlib.util.spec_from_file_location("_v28_update_audit_test", IMPL / "audit-v28-dpvo-update.py")
MOD = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = MOD
spec.loader.exec_module(MOD)


def test_true_numeric_failure_is_consistent_not_an_audit_failure():
    original = np.ones(4, dtype=np.float32)
    candidate = original + .5
    declared = MOD.compare(candidate, original)
    audit = MOD.Audit()
    result = audit.reported_compare(candidate, original, declared, "failure")
    assert result["gate_passed"] is False and result["failed_elements"] == 4
    assert audit.failures == []


@pytest.mark.parametrize("field,value", [("gate_passed", True), ("failed_elements", 0),
                                        ("rtol", 1.), ("atol", 1.), ("bitwise_equal", True)])
def test_forged_gate_or_relaxed_threshold_detected(field, value):
    original = np.ones(4, dtype=np.float32)
    candidate = original + .5
    declared = MOD.compare(candidate, original)
    declared[field] = value
    audit = MOD.Audit()
    result = audit.reported_compare(candidate, original, declared, "mutant")
    assert not result["gate_passed"]
    assert audit.failures == ["mutant/" + field]


def test_signed_zero_array_equal_does_not_prove_byte_equality():
    positive, negative = np.array([0.], np.float32), np.array([-0.], np.float32)
    assert np.array_equal(positive, negative)
    assert MOD.compare(positive, negative)["gate_passed"] is True
    assert MOD.compare(positive, negative)["bitwise_equal"] is False


def test_original_codomain_and_ORT_elementwise_gate_are_separate():
    values = {"net": np.zeros((1, 2, 384), np.float32), "delta": np.zeros((1, 2, 2), np.float32),
              "weight": np.full((1, 2, 2), -1.1920928955078125e-7, np.float32)}
    audit = MOD.Audit()
    bounds = MOD._outputs(audit, values, 2, "ORT", exact_sigmoid_range=False)
    assert bounds["negative_elements"] == 4 and bounds["exact_mathematical_range"] is False
    assert MOD.compare(values["weight"], np.zeros_like(values["weight"]))["gate_passed"] is True
    assert audit.failures == []
    with pytest.raises(ValueError, match="original sigmoid weight range"):
        MOD._outputs(MOD.Audit(), values, 2, "original")
    assert (values["weight"] < 0).all()  # no mutation/clipping


def test_nonfinite_or_wrong_precision_is_never_valid():
    for invalid in (np.array([np.nan], np.float32), np.array([1.], np.float64)):
        with pytest.raises(ValueError, match="finite|FP32"):
            MOD.compare(invalid, np.array([1.], np.float32))


def test_topology_oracle_preserves_ties_hash_collision_padding_and_inverse():
    arrays, groups = MOD.expected_topology(
        np.array([0, 1, 0, 0], np.int64), np.array([12345, 0, 7, 7], np.int64),
        np.array([1, 2, 1, 1], np.int64), {"kk": (3, 4), "ij": (3, 3)})
    assert arrays["previous_indices"].tolist() == [3, 0, 0, 2]
    assert arrays["previous_valid"].flatten().tolist() == [1, 0, 0, 1]
    assert arrays["ij_inverse"].tolist() == [1, 1, 0, 0]
    assert arrays["kk_indices"][0].tolist() == [0, 2, 3, 0]
    assert groups["kk"]["padding_slots"] == 8
    for group in ("kk", "ij"):
        i, mask, inv = (arrays[group + "_" + key] for key in ("indices", "valid", "inverse"))
        assert np.array_equal(np.sort(i[mask == 1]), np.arange(4))
        rows = np.broadcast_to(np.arange(i.shape[0])[:, None], i.shape)
        assert np.array_equal(inv[i[mask == 1]], rows[mask == 1])
    with pytest.raises(ValueError, match="overflow"):
        MOD.expected_topology(np.zeros(4, np.int64), np.zeros(4, np.int64), np.zeros(4, np.int64),
                              {"kk": (1, 3), "ij": (1, 3)})


def test_artifact_hash_then_raw_metadata_tampering_fails(tmp_path):
    path = tmp_path / "toy.npz"
    values = {"net": np.arange(4, dtype=np.float32)}
    np.savez(path, **values)
    item = {"file": path.name, "bytes": path.stat().st_size, "sha256": MOD.sha(path),
            "arrays": {k: MOD.meta(v) for k, v in values.items()}}
    assert MOD.Audit().artifact(tmp_path, item)["net"].tobytes() == values["net"].tobytes()
    bad = copy.deepcopy(item)
    bad["arrays"]["net"]["tensor_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="raw tensor SHAs"):
        MOD.Audit().artifact(tmp_path, bad)
    np.savez(path, net=values["net"] + 1)
    with pytest.raises(ValueError, match="file size/SHA"):
        MOD.Audit().artifact(tmp_path, item)


def test_profile_call_is_not_part_of_three_warm_mean():
    calls = [{"seconds": value} for value in (7., .01, .02, .03, 9., 9., 9.)]
    audit = MOD.Audit()
    assert MOD._call_times(audit, calls, "seconds", .02, "mean") == .02
    assert not audit.failures
    audit = MOD.Audit()
    MOD._call_times(audit, calls, "seconds", 4., "mutant")
    assert audit.failures == ["mutant three warm-call mean"]
