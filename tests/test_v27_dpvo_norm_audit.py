"""Tiny gate tests only; no real images, model, or ONNX Runtime."""
import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ENTRY = HERE / "audit-v27-dpvo-norm.py"
if not ENTRY.exists():
    ENTRY = HERE.parent / "deploy/rk3588/validation/audit-v27-dpvo-norm.py"
SPEC = importlib.util.spec_from_file_location("v27_norm_audit", ENTRY)
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def gate_receipt():
    gate = {"rtol": 1e-4, "atol": 1e-5, "gate_passed": True, "failed_elements": 0}
    call = {key: {name: dict(gate) for name in ("fmap", "imap")}
            for key in ("versus_original_CPU", "versus_original_ORT")}
    return {"status": "completed_board_ORT_rewrite_gate_not_VO", "exact_FP32_gate_passed": True,
            "calls": [copy.deepcopy(call) for _ in range(3)]}


def test_original_unchanged_gate_accepts_and_detects_one_failed_element():
    reference = np.asarray([0., 1., -1.], np.float32)
    assert audit.compare(reference.copy(), reference)["gate_passed"]
    candidate = reference.copy()
    candidate[0] = np.float32(1.01e-5)
    checked = audit.compare(candidate, reference)
    assert checked["failed_elements"] == 1 and not checked["gate_passed"]
    assert checked["maximum_tolerance_ratio"] > 1
    assert checked["failed_reference_abs_range"] == [0., 0.]


@pytest.mark.parametrize("change", ["failed_status", "false_top_gate", "CPU_failure", "ORT_failure", "relaxed_atol", "missing_output", "wrong_count"])
def test_failed_or_incomplete_receipt_cannot_authorize_validated_acceleration(change):
    receipt = gate_receipt()
    audit.require_gate_before_acceleration(receipt)
    if change == "failed_status":
        receipt["status"] = "failed"
        receipt["error"] = "ValueError('Original FP32 gate failed; do not compile as validated')"
    elif change == "false_top_gate":
        receipt["exact_FP32_gate_passed"] = False
    elif change in ("CPU_failure", "ORT_failure"):
        field = "versus_original_CPU" if change == "CPU_failure" else "versus_original_ORT"
        receipt["calls"][0][field]["fmap"].update(gate_passed=False, failed_elements=1)
    elif change == "relaxed_atol":
        receipt["calls"][0]["versus_original_CPU"]["fmap"]["atol"] = 2e-5
    elif change == "missing_output":
        del receipt["calls"][0]["versus_original_ORT"]["imap"]
    else:
        receipt["calls"].pop()
    with pytest.raises(ValueError):
        audit.require_gate_before_acceleration(receipt)


@pytest.mark.parametrize("invalid", ["dtype", "shape", "nan", "empty"])
def test_invalid_arrays_are_rejected(invalid):
    reference = np.ones((2, 3), np.float32)
    candidate = (reference.astype(np.float64) if invalid == "dtype" else
                 reference[:, :1] if invalid == "shape" else
                 np.full_like(reference, np.nan) if invalid == "nan" else
                 np.empty((0, 3), np.float32))
    if invalid == "empty":
        reference = candidate.copy()
    with pytest.raises(ValueError):
        audit.compare(candidate, reference)


def test_reported_false_success_rejected_and_signed_zero_keeps_byte_gate():
    reference = np.asarray([0., 1.], np.float32)
    candidate = np.asarray([-0., 1.], np.float32)
    checked = audit.compare(candidate, reference)
    assert checked["gate_passed"] and not checked["bitwise_equal"]
    saved = dict(checked)
    saved["bitwise_equal"] = True
    with pytest.raises(ValueError, match="bitwise_equal"):
        audit.verify_reported_gate(saved, checked)
