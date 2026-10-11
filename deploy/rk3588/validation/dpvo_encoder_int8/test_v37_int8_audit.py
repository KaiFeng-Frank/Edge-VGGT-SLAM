"""Tiny saved-data/parser guards; no real model or SDK execution."""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ENTRY = HERE/"audit-v37-dpvo-int8.py"
if not ENTRY.is_file():
    ENTRY = HERE.parent/"deploy/rk3588/validation/dpvo_int8_inference/audit-v37-dpvo-int8.py"
SPEC = importlib.util.spec_from_file_location("v37_int8_saved_audit", ENTRY)
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def test_unchanged_strict_gate_does_not_turn_small_relative_error_into_pass():
    reference = np.array([0., 10000.], np.float32)
    actual = np.array([2e-5, 10000.], np.float32)
    result = A.score(actual, reference)
    assert result["failed_elements"] == 1 and result["gate_passed"] is False
    assert result["relative_l2"] < 1e-8 and result["atol"] == 1e-5 and result["rtol"] == 1e-4


def test_bitwise_gate_distinguishes_signed_zero():
    result = A.score(np.array([0.], np.float32), np.array([-0.], np.float32))
    assert result["gate_passed"] and result["array_equal"] and not result["bitwise_equal"]


def test_real_saved_raw_sha_tamper_rejected(tmp_path):
    path = tmp_path/"tiny.npz"
    array = np.array([1, 2], np.float32)
    np.savez(path, fmap=array)
    item = dict(file=path.name, sha256=A.sha(path), bytes=path.stat().st_size)
    assert np.array_equal(A.read_npz(tmp_path, item, {"fmap": A.metadata(array)}, A.Checks())["fmap"], array)
    changed = A.metadata(array)
    changed["tensor_sha256"] = "0"*64
    with pytest.raises(ValueError, match="tensor_sha256"):
        A.read_npz(tmp_path, item, {"fmap": changed}, A.Checks())


def detail():
    text = ("1 InputOperator INT64 CPU \\ (96) 0/0/0 1 0.0%/0.0%/0.0% 0 InputOperator:indices\n"
            "2 Conv INT8 NPU (1,3,2,2) (1,2,2,2) 1/2/3 9 33.0%/33.0%/34.0% 1 Conv:conv\n"
            "3 Transpose INT8 CPU (1,2,2,2) (2,2,1,2) 0/0/0 30 0.0%/0.0%/0.0% 1 Transpose:norm_tp\n"
            "Total Operator Elapsed Per Frame Time(us): 40\n"
            "Operator Time Consuming Ranking Table\n"
            "Transpose 1 30 0 0 30 75.00%\nConv 1 0 0 9 9 22.50%\nInputOperator 1 1 0 0 1 2.50%\n")
    return dict(query_command=3, bytes=len(text.encode()), raw_sha256=hashlib.sha256(text.encode()).hexdigest(), raw_text=text)


def test_profile_counts_dtype_placement_and_nested_time():
    result = A.parse_profile(detail(), A.Checks())
    assert result["target_counts"] == {"CPU": 2, "NPU": 1}
    assert result["target_dtype_counts"]["NPU"] == {"INT8": 1}
    assert result["CPU_transpose_us"] == 30 and result["CPU_transpose_fraction_of_operator_elapsed"] == .75


@pytest.mark.parametrize("what", ["hash", "total", "ranking"])
def test_profile_hash_or_elapsed_inconsistency_rejected(what):
    value = detail()
    if what == "hash":
        value["raw_sha256"] = "0"*64
    else:
        value["raw_text"] = value["raw_text"].replace("Time(us): 40", "Time(us): 41") if what == "total" else value["raw_text"].replace("Transpose 1 30", "Transpose 1 29")
        value["bytes"] = len(value["raw_text"].encode())
        value["raw_sha256"] = hashlib.sha256(value["raw_text"].encode()).hexdigest()
    with pytest.raises(ValueError):
        A.parse_profile(value, A.Checks())


def test_dense_gather_preserves_duplicate_indices_and_order():
    imap = np.zeros((1, 1, 384, 116, 152), np.float32)
    imap[0, 0, :, 2, 3] = 7
    imap[0, 0, :, 2, 4] = 9
    indices = np.full(96, 2*152+3, np.int64)
    indices[1] = 2*152+4
    result = A.gather_context(imap, indices)
    assert result.shape == (1, 96, 384) and result[0, 0, 0] == result[0, 2, 0] == 7 and result[0, 1, 0] == 9
    with pytest.raises(ValueError, match="index shapes"):
        A.gather_context(imap, indices.astype(np.float32))
