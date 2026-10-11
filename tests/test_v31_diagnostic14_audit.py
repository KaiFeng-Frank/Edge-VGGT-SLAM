"""Tiny saved-numeric gate tests; no models, sessions, datasets or BA."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load_auditor():
    local = Path(__file__).with_name("audit-v31-diagnostic14.py")
    path = local if local.is_file() else Path(__file__).resolve().parents[1] / "deploy/rk3588/validation/dpvo_dynamic_update/audit-v31-diagnostic14.py"
    spec = importlib.util.spec_from_file_location("v31_saved_diagnostic14", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = load_auditor()


def test_zero_reference_boundary_is_strict_and_has_null_relative_l2():
    reference = np.zeros(1, dtype=np.float32)
    lower = np.asarray([np.float32(AUDIT.ATOL)], dtype=np.float32)
    upper = np.nextafter(lower, np.float32(np.inf))
    assert AUDIT.residual_statistics(reference, lower)["gate_passed"] is True
    result = AUDIT.residual_statistics(reference, upper)
    assert result["failed_elements"] == 1
    assert result["relative_l2"] is None
    assert result["failed_coordinates"][0]["index"] == [0]


def test_reference_denominator_cannot_be_replaced_by_candidate():
    reference = np.asarray([0.004], dtype=np.float32)
    value = np.float32(reference[0] + AUDIT.ATOL + AUDIT.RTOL * float(reference[0]))
    for _ in range(5):
        if float(value) - float(reference[0]) > AUDIT.ATOL + AUDIT.RTOL * abs(float(reference[0])):
            break
        value = np.nextafter(value, np.float32(np.inf))
    candidate = np.asarray([value], dtype=np.float32)
    error = float(value) - float(reference[0])
    assert error <= AUDIT.ATOL + AUDIT.RTOL * abs(float(value))
    assert AUDIT.residual_statistics(reference, candidate)["failed_elements"] == 1


def test_small_global_relative_l2_does_not_accept_one_bad_value():
    reference = np.ones(100, dtype=np.float32)
    candidate = reference.copy()
    reference[0] = 0
    candidate[0] = 2e-5
    result = AUDIT.residual_statistics(reference, candidate)
    assert result["relative_l2"] < 3e-6
    assert result["gate_passed"] is False
    assert result["failed_elements"] == 1


@pytest.mark.parametrize("candidate", [
    np.ones(2, dtype=np.float64),
    np.ones(3, dtype=np.float32),
    np.asarray([np.nan, 1], dtype=np.float32),
    np.asarray([np.inf, 1], dtype=np.float32),
])
def test_non_fp32_shape_and_nonfinite_are_rejected(candidate):
    with pytest.raises(ValueError):
        AUDIT.residual_statistics(np.ones(2, dtype=np.float32), candidate)


def test_no_private_dataset_paths_or_execution_imports():
    source = Path(AUDIT.__file__).read_text()
    assert "/home/" not in source and "/mnt/" not in source
    assert "import torch" not in source and "import onnxruntime" not in source
