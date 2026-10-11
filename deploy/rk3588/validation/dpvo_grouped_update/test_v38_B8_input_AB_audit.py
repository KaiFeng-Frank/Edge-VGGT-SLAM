"""Tiny saved-array contract tests; no neural/native/SDK execution."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("saved_audit", Path(__file__).with_name("audit-v38-B8-input-AB.py"))
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def test_independent_cartesian_permutation_preserves_original_edge_order():
    canonical = [(i, j, i * 2 + k) for i in range(2) for j in range(2) for k in range(2)]
    shuffled = np.asarray(canonical, dtype=np.int64)[[3, 0, 7, 1, 6, 4, 2, 5]]
    order = a.permutation(*shuffled.T.copy(), sources=2, degree=2, patches=2)
    assert np.array_equal(shuffled[order], np.asarray(canonical))
    values = np.arange(24, dtype=np.float32).reshape(1, 8, 3)
    packed = values[0, order].reshape(2, 2, 2, 3).transpose(0, 3, 1, 2)
    restored = np.empty_like(values)
    restored[0, order] = packed.transpose(0, 2, 3, 1).reshape(8, 3)
    assert restored.tobytes() == values.tobytes()


def test_duplicate_edge_and_wrong_source_ownership_rejected():
    rows = np.asarray([(i, j, i * 2 + k) for i in range(2) for j in range(2) for k in range(2)], dtype=np.int64)
    rows[0] = rows[1]
    with pytest.raises(ValueError, match="duplicate"):
        a.permutation(*rows.T.copy(), sources=2, degree=2, patches=2)
    rows[0, 2] = 2
    with pytest.raises(ValueError, match="ownership"):
        a.permutation(*rows.T.copy(), sources=2, degree=2, patches=2)


def test_array_declaration_tamper_is_detected(tmp_path):
    value = np.arange(6, dtype=np.float32).reshape(2, 3)
    path = tmp_path / "tiny.npz"
    np.savez(path, value=value)
    declaration = dict(file=path.name, bytes=path.stat().st_size, sha256=a.sha(path), arrays={"value": a.metadata(value)})
    assert a.artifact(tmp_path, declaration, a.Checks())["value"].tobytes() == value.tobytes()
    declaration["arrays"]["value"]["shape"] = [3, 2]
    with pytest.raises(ValueError, match="shape"):
        a.artifact(tmp_path, declaration, a.Checks())


def test_original_fixed_gate_and_warm_selection_remain_separate():
    reference = np.array([1.0, 0.0], dtype=np.float32)
    actual = np.array([1.0002, 0.0], dtype=np.float32)
    assert a.compare(actual, reference)["gate_passed"] is False
    calls = [{"input_transport": method, "first_for_transport": i < 2, "timing": {"forward_seconds": float(i + 1)}}
             for i, method in enumerate(a.ORDER)]
    summary = a.warm_summary(calls)
    assert summary["fp32"]["warm_forward_seconds"] == [4.0, 5.0]
    assert summary["fp16"]["warm_forward_seconds"] == [3.0, 6.0]
