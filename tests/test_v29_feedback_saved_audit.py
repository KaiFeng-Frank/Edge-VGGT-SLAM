"""Small saved-array tamper tests; no model, ORT, geometry or BA execution."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ENTRY = Path(__file__).resolve().parent / "audit-v29-dpvo-ort-feedback.py"
if not ENTRY.exists():
    ENTRY = Path(__file__).resolve().parents[1] / "deploy/rk3588/validation/dpvo_geometry/audit-v29-dpvo-ort-feedback.py"
SPEC = importlib.util.spec_from_file_location("v29_saved_feedback_audit_test", ENTRY)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def saved_item(tmp_path):
    value = np.array([1., -0., 2.], np.float32)
    path = tmp_path / "toy.npz"
    np.savez(path, value=value)
    item = dict(file=path.name, bytes=path.stat().st_size, sha256=audit.sha(path), arrays=dict(value=audit.meta(value)))
    return path, item


def test_saved_artifact_raw_and_file_hashes(tmp_path):
    _, item = saved_item(tmp_path)
    value = audit.artifact(tmp_path, item)["value"]
    assert audit.same_bytes(value, np.array([1., -0., 2.], np.float32))
    item["arrays"]["value"]["tensor_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="raw metadata"):
        audit.artifact(tmp_path, item)


def test_file_tamper_and_path_traversal_rejected(tmp_path):
    path, item = saved_item(tmp_path)
    with path.open("ab") as file:
        file.write(b"tamper")
    with pytest.raises(ValueError, match="file SHA/bytes"):
        audit.artifact(tmp_path, item)
    item["file"] = "../toy.npz"
    with pytest.raises(ValueError, match="Unsafe"):
        audit.artifact(tmp_path, item)


def reference_fixture():
    graph = dict(ctx=np.ones((1, 2, 3), np.float32), ii=np.zeros(2, np.int64),
                 jj=np.ones(2, np.int64), kk=np.arange(2, dtype=np.int64))
    arrays = dict(net_before=np.full((1, 2, 3), 9, np.float32), corr=np.full((1, 2, 4), 2, np.float32),
                  net_after=np.full((1, 2, 3), 10, np.float32), delta=np.zeros((1, 2, 2), np.float32),
                  weight=np.full((1, 2, 2), .5, np.float32))
    reference = dict(net_out=arrays["net_after"].copy(), delta=arrays["delta"].copy(), weight=arrays["weight"].copy())
    checks = {name: audit.strict_compare(arrays[key], reference[name]) for name, key in
              zip(audit.OUTPUTS, ("net_after", "delta", "weight"))}
    inputs = {name: audit.meta(value) for name, value in (("net", arrays["net_before"]), ("inp", graph["ctx"]),
               ("corr", arrays["corr"]), ("ii", graph["ii"]), ("jj", graph["jj"]), ("kk", graph["kk"]))}
    row = dict(inputs=inputs, comparisons=checks, same_current_net_before_and_correlation=True, strict_FP32_gate_passed=True)
    return row, arrays, reference, graph


def test_current_saved_net_and_corr_reference_binding():
    row, arrays, reference, graph = reference_fixture()
    assert all(x["gate_passed"] for x in audit.reference_checks(row, arrays, reference, graph).values())
    row["inputs"]["net"] = audit.meta(np.zeros_like(arrays["net_before"]))
    with pytest.raises(ValueError, match="current ORT net/corr"):
        audit.reference_checks(row, arrays, reference, graph)


def test_wrong_correlation_or_comparison_gate_rejected():
    row, arrays, reference, graph = reference_fixture()
    arrays["corr"] += 1
    with pytest.raises(ValueError, match="current ORT net/corr"):
        audit.reference_checks(row, arrays, reference, graph)
    row, arrays, reference, graph = reference_fixture()
    reference["net_out"] += np.float32(.1)
    with pytest.raises(ValueError, match="strict gate"):
        audit.reference_checks(row, arrays, reference, graph)


def test_reporting_gate_boolean_does_not_override_array_failure():
    row, arrays, reference, graph = reference_fixture()
    arrays["delta"] += 1
    row["comparisons"]["delta"] = audit.strict_compare(arrays["delta"], reference["delta"])
    row["comparisons"]["delta"]["gate_passed"] = True
    with pytest.raises(ValueError, match="strict gate"):
        audit.reference_checks(row, arrays, reference, graph)


def test_scalar_norm_portability_does_not_relax_elementwise_thresholds():
    row, arrays, reference, graph = reference_fixture()
    row["comparisons"]["net_out"]["relative_l2"] = 1e-20
    audit.reference_checks(row, arrays, reference, graph)
    row["comparisons"]["net_out"]["rtol"] *= 2
    with pytest.raises(ValueError):
        audit.reference_checks(row, arrays, reference, graph)


def test_original_weight_domain_still_exact_after_strict_tolerance_pass():
    row, arrays, reference, graph = reference_fixture()
    arrays["weight"][0, 0, 0] = 0
    reference["weight"][0, 0, 0] = np.float32(-1e-7)
    row["comparisons"]["weight"] = audit.strict_compare(arrays["weight"], reference["weight"])
    assert row["comparisons"]["weight"]["gate_passed"]
    with pytest.raises(ValueError, match="weight outside"):
        audit.reference_checks(row, arrays, reference, graph)


def test_topology_ties_keep_original_edge_order_and_no_members_omitted():
    kk = np.array([1, 0, 1, 0, 1, 0], np.int64)
    ii = kk.copy()
    jj = np.array([2, 1, 1, 1, 1, 0], np.int64)
    t = audit.expected_topology(ii, jj, kk)
    assert np.array_equal(t["previous_indices"], [4, 5, 0, 1, 2, 0])
    assert np.array_equal(t["following_indices"], [0, 3, 4, 0, 0, 1])
    assert np.array_equal(t["previous_valid"].ravel(), [1, 1, 0, 1, 1, 0])
    assert np.array_equal(t["following_valid"].ravel(), [0, 1, 1, 0, 1, 1])
    for key in ("kk", "ij"):
        real = t[key+"_indices"][t[key+"_valid"] == 1]
        assert np.array_equal(np.sort(real), np.arange(6))
        inverse = t[key+"_inverse"]
        for group, (indices, valid) in enumerate(zip(t[key+"_indices"], t[key+"_valid"])):
            assert np.all(inverse[indices[valid == 1]] == group)


def timer_fixture():
    seconds = dict(zip(audit.STAGES, (.01, .02, .03, .04, .01, .02, .01, .01)))
    begin, end = 100., 100.15
    ort = dict(begin_monotonic=100.06, end_monotonic=100.10, total_seconds=.04,
               input_check_and_pack_seconds=.005, run_seconds=.028, owned_copy_seconds=.005, output_check_seconds=.002)
    ref = dict(begin_monotonic=100.17, end_monotonic=100.22, reference_forward_seconds=.05)
    return dict(seconds=seconds, begin_monotonic=begin, end_monotonic=end, round_total_seconds=.15,
                ORT_update_timing=ort, original_Torch_reference=ref)


def test_reference_is_outside_feedback_and_ORT_is_inside_Update():
    row = timer_fixture()
    assert audit.timing_checks(row, 99.) == 100.22


@pytest.mark.parametrize("change", [
    lambda r: r["original_Torch_reference"].update(begin_monotonic=100.14, reference_forward_seconds=.08),
    lambda r: r["ORT_update_timing"].update(end_monotonic=100.12),
    lambda r: r["seconds"].update(update=-.04),
    lambda r: r["ORT_update_timing"].update(owned_copy_seconds=-.005),
])
def test_timing_overlap_or_hidden_component_rejected(change):
    row = timer_fixture()
    change(row)
    with pytest.raises(ValueError):
        audit.timing_checks(row)


def test_next_round_waits_for_previous_reference_end():
    with pytest.raises(ValueError, match="prior golden"):
        audit.timing_checks(timer_fixture(), 100.01)


def test_saved_state_continuity_checks_c_order_bytes_and_signed_zero():
    a, b = np.array([0.], np.float32), np.array([-0.], np.float32)
    assert np.array_equal(a, b) and not audit.same_bytes(a, b)
    assert not audit.same_bytes(a, a.astype(np.float64))
