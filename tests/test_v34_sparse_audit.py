"""Small saved-array tamper/scope guards; no real image, model, ORT or SDK."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def entry():
    here = Path(__file__).resolve().parent
    for path in (here / "audit-v34-dpvo-sparse-context.py",
                 here.parent / "deploy/rk3588/validation/dpvo_sparse_context_sdk/audit-v34-dpvo-sparse-context.py"):
        if path.exists():
            return path
    pytest.fail("Published v34 saved-array auditor missing")


spec = importlib.util.spec_from_file_location("v34_saved_array_audit", entry())
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def test_strict_gate_failure_is_not_hidden_by_finite_or_small_relative_error():
    reference = np.array([0.0, 1000.0], np.float32)
    candidate = np.array([2e-5, 1000.0], np.float32)
    result = A.score(candidate, reference)
    assert result["finite"] and result["relative_l2"] < 1e-6
    assert not result["gate_passed"] and result["failed_elements"] == 1


def test_bitwise_gate_distinguishes_signed_zero():
    result = A.score(np.array([0.0], np.float32), np.array([-0.0], np.float32))
    assert result["gate_passed"] and result["array_equal"] and not result["bitwise_equal"]


def payload_row(arrays, selected, full):
    return dict(role="dense", artifact={"includes_full_dense_imap": full},
                arrays={k: A.metadata(v) for k, v in arrays.items()},
                selected_arrays={k: A.metadata(v) for k, v in selected.items()})


def test_canonical_only_payload_never_claims_full_dense_gather(monkeypatch):
    monkeypatch.setattr(A, "SHAPES", {"fmap": (1, 2, 2, 3), "context": (1, 3, 4)})
    selected = {"fmap": np.ones((1, 2, 2, 3), np.float32), "context": np.ones((1, 3, 4), np.float32)}
    row = payload_row(selected, selected, False)
    values, full = A.select_payload(selected, np.array([1, 4, 1], np.int64), row, A.Checks())
    assert values is selected and full is False
    row["artifact"]["includes_full_dense_imap"] = True
    with pytest.raises(ValueError, match="availability"):
        A.select_payload(selected, np.array([1, 4, 1], np.int64), row, A.Checks())


def test_available_dense_gather_preserves_order_and_repeats(monkeypatch):
    monkeypatch.setattr(A, "SHAPES", {"fmap": (1, 2, 2, 3), "context": (1, 3, 4)})
    arrays = {"fmap": np.ones((1, 2, 2, 3), np.float32), "imap": np.arange(24, dtype=np.float32).reshape(1, 4, 2, 3)}
    indices = np.array([4, 1, 4], np.int64)
    selected = {"fmap": arrays["fmap"], "context": arrays["imap"].reshape(4, 6)[:, indices].T[None].copy()}
    result, full = A.select_payload(arrays, indices, payload_row(arrays, selected, True), A.Checks())
    assert full and result["context"].tobytes() == selected["context"].tobytes()


def test_saved_array_raw_sha_tampering_is_rejected(tmp_path):
    path = tmp_path / "tiny.npz"
    values = {"x": np.array([1, 2, 1], np.int64)}
    np.savez(path, **values)
    item = dict(file=path.name, bytes=path.stat().st_size, sha256=A.sha(path))
    declared = {"x": A.metadata(values["x"])}
    assert A.read_npz(tmp_path, item, declared, A.Checks())["x"].dtype == np.int64
    declared["x"]["tensor_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="rehashed array"):
        A.read_npz(tmp_path, item, declared, A.Checks())


def test_float_index_attr_is_not_silently_accepted():
    row = dict(index=1, name="patch_indices", shape=[96], type=8, format=3,
               n_elems=96, size=768, size_with_stride=768, quantization_type=2, zero_point=0, scale=1.0)
    A.check_attr(row, 1, "patch_indices", (96,), 8, (3,), A.Checks())
    row["type"] = 0
    with pytest.raises(ValueError, match="attribute"):
        A.check_attr(row, 1, "patch_indices", (96,), 8, (3,), A.Checks())


def timing_row():
    timing = {k: 0.01 for k in ("input_packing_seconds", "input_set_seconds", "run_api_seconds",
                               "outputs_get_seconds", "output_copy_seconds", "outputs_release_seconds")}
    timing.update(sdk_io_seconds=0.07, output_buffers_released=True, synchronous=True,
                  output_bytes=[9027584, 147456], input_api_types=[0, 8], input_api_formats=[1, 3],
                  input_pass_through=[0, 0], input_logical_dtypes=["float32", "int64"],
                  input_buffer_shapes=[[1, 464, 608, 3], [96]], input_buffer_bytes=[3385344, 768],
                  output_want_float=1, output_is_prealloc=0)
    return dict(role="sparse", timing=timing, forward_seconds=0.07, dense_context_gather_seconds=0.0,
                feature_payload_ready_seconds=0.07)


def test_endpoint_cannot_omit_owned_copy_or_release():
    row = timing_row()
    A.check_timing(row, A.Checks())
    bad = copy.deepcopy(row)
    bad["timing"]["sdk_io_seconds"] = bad["forward_seconds"] = bad["feature_payload_ready_seconds"] = 0.04
    with pytest.raises(ValueError, match="includes packing"):
        A.check_timing(bad, A.Checks())


def test_metadata_only_profile_must_be_explicit_and_bound_to_profile():
    with pytest.raises(SystemExit) as raised:
        A.main(["--normal-dir", ".", "--reference-dir", ".", "--normal-sha256", "0" * 64,
                "--output", "unused.json", "--profile-metadata-only"])
    assert raised.value.code == 2
