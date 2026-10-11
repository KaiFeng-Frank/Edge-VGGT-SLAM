"""Tiny saved-array mutations only; no source image/model/session execution."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ENTRY = Path(__file__).with_name("audit-v33-dpvo-sparse-context.py")
if not ENTRY.exists():
    ENTRY = Path(__file__).resolve().parents[1] / "deploy/rk3588/validation/dpvo_sparse_context/audit-v33-dpvo-sparse-context.py"
SPEC = importlib.util.spec_from_file_location("v33_sparse_saved_audit", ENTRY)
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def fixture():
    life = {"session_initializations": [{"role": "parent"}, {"role": "sparse"}],
        "ORT_session_references_dropped": True, "gc_collect_performed": True,
        "native_explicit_destroy_API_available": False}
    report = {"schema": "v33_DPVO_sparse_context_board_CPU_ORT", "script_sha256": A.DRIVER_SHA,
        "providers": ["CPUExecutionProvider"], "graph_optimization": "ORT_DISABLE_ALL", "cpu_affinity": [4, 5, 6, 7],
        "intra_threads": 4, "inter_threads": 1, "actual_board_execution": True, "NPU_used": False,
        "normalization_applications_in_this_driver": 0,
        "sampling_rule": {"M": 96, "original_RNG_observations": False, "used_source_RGB_for_selection": False},
        "historical_spatial_rejection": {"receipt_sha256": A.FROZEN["spatial_ort"], "status": "failed",
            "exact_FP32_gate_passed": False, "retained": True},
        "source_pins": {"driver": A.DRIVER_SHA, "editor": A.EDITOR_SHA},
        "source_hashes_after": {"driver": A.DRIVER_SHA, "editor": A.EDITOR_SHA},
        "sources_and_inputs_unchanged": True, "cleanup_errors": [], "affinity_restored": True,
        "session_lifecycle": {"original": copy.deepcopy(life), "spatial": copy.deepcopy(life)},
        "receipt_sha256": A.FROZEN.copy()}
    for key in ("complete_VO", "original_VGGT_used", "realtime_validated", "production_changed",
                "candidate_accepted_as_original_model", "model_checkpoint_loaded", "images_decoded"):
        report[key] = False
    return report, {"status": "failed", "exact_FP32_gate_passed": False}


@pytest.mark.parametrize("field", ["history_promoted", "candidate_promoted", "stale_source", "wrong_receipt"])
def test_preserved_parent_failure_provenance_and_scope_cannot_be_promoted(field):
    report, history = fixture()
    A.check_global(report, history, A.Checks())
    if field == "history_promoted":
        report["historical_spatial_rejection"]["exact_FP32_gate_passed"] = True
    elif field == "candidate_promoted":
        report["candidate_accepted_as_original_model"] = True
    elif field == "stale_source":
        report["source_hashes_after"]["editor"] = "0" * 64
    else:
        report["receipt_sha256"]["capture"] = "0" * 64
    with pytest.raises(ValueError):
        A.check_global(report, history, A.Checks())


def test_strict_element_gate_and_byte_identity_remain_separate():
    x = np.array([0, 1], np.float32)
    y = np.array([-0.0, 1], np.float32)
    result = A.score(x, y)
    assert result["gate_passed"] and result["array_equal"] and not result["bitwise_equal"]
    x[0] = np.float32(1.0001e-5)
    assert A.score(x, np.array([0, 1], np.float32))["failed_elements"] == 1


def test_explicit_xy_sampling_keeps_index_order_repeats_and_channels():
    dense = np.arange(24, dtype=np.float32).reshape(1, 4, 2, 3)
    value = A.gather(dense, np.array([4, 1, 4], np.int64))
    assert np.array_equal(value[0, 0], dense[0, :, 1, 1])
    assert np.array_equal(value[0, 1], dense[0, :, 0, 1])
    assert value[0, 0].tobytes() == value[0, 2].tobytes()
    centers, indices = A.diagnostic_indices(2)
    assert indices.tobytes() == (centers[:, 1] * 152 + centers[:, 0]).tobytes()
    assert len(np.unique(indices)) < 96


def test_saved_npz_raw_sha_rejected_even_after_container_hash_is_updated(tmp_path):
    path = tmp_path / "tiny.npz"
    value = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    declared = {"context": A.array_info(value)}
    np.savez(path, context=value)
    item = {"file": path.name, "bytes": path.stat().st_size, "sha256": A.sha(path)}
    assert np.array_equal(A.read_npz(tmp_path, item, declared, A.Checks())["context"], value)
    value[0, 0, 0] += 1
    np.savez(path, context=value)
    item.update(bytes=path.stat().st_size, sha256=A.sha(path))
    with pytest.raises(ValueError, match="raw metadata"):
        A.read_npz(tmp_path, item, declared, A.Checks())


@pytest.mark.parametrize("dtype", [np.float64, np.int64])
def test_fp32_gate_does_not_accept_reinterpreted_output_type(dtype):
    with pytest.raises(ValueError, match="dtype"):
        A.score(np.ones(3, dtype), np.ones(3, np.float32))


def test_artifact_parent_path_cannot_escape_declared_directory(tmp_path):
    with pytest.raises(ValueError, match="simple artifact"):
        A.artifact(tmp_path, {"file": "../unrelated.npz", "bytes": 1, "sha256": "0" * 64}, A.Checks())
