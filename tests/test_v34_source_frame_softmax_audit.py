"""Small audit metadata tests; no model, SDK or ORT execution."""
import importlib.util
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
path = HERE / "audit-v34-source-frame-softmax.py"
if not path.is_file():
    path = HERE.parent / "deploy/rk3588/validation/dpvo_source_frame_softmax" / path.name
spec = importlib.util.spec_from_file_location("v34_saved_audit", path)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class Checks:
    def check(self, condition, label):
        if not condition:
            raise ValueError(label)


def _report():
    r = dict(schema="v33_DPVO_source_frame_Update_board_export",
             status="completed_source_frame_export_diagnostic_not_adopted", experiment_version=34,
             expression_variant="v34_direct_torch_softmax", ORB_used=False, added_LK_PnP_frontend=False,
             cleanup_errors=[], terminal_sigmoid="stable", source_pins={"script": "a"}, source_hashes_after={"script": "a"})
    for name in ("candidate_accepted", "host_model_inference", "NPU_used", "BA_executed", "valid_pose_output",
                 "realtime_validated", "trajectory_accuracy_validated", "FP32_thresholds_relaxed", "output_clipping",
                 "full_E23520_reference_reexecuted", "source_plan_recomputed"):
        r[name] = False
    for name in ("actual_board_execution", "actual_inference_on_RK3588S", "original_VGGT_default_unchanged",
                 "different_model_from_VGGT", "outputs_finite", "weight_domain_valid", "original_parameters_unchanged",
                 "sources_and_inputs_unchanged", "affinity_restored", "ORT_session_reference_released"):
        r[name] = True
    return r


def test_rejection_is_a_legal_evidence_record_not_adoption():
    r = _report()
    r["strict_FP32_gate_passed"] = False
    a.check_scope(r, Checks())
    computed = dict(rtol=1e-4, atol=1e-5, failed_elements=1, gate_passed=False, max_abs=2e-5,
                    relative_l2=2e-8, bitwise_equal=False, max_normalized_tolerance=2.)
    recorded = {k: v for k, v in computed.items() if k != "max_normalized_tolerance"}
    a.check_recorded_metric(computed, recorded, Checks())
    recorded["gate_passed"] = True
    with pytest.raises(ValueError, match="gate_passed"):
        a.check_recorded_metric(computed, recorded, Checks())


@pytest.mark.parametrize("claim", ["candidate_accepted", "full_E23520_reference_reexecuted", "NPU_used"])
def test_unsupported_fresh_gold_or_success_claim_rejected(claim):
    r = _report()
    r[claim] = True
    with pytest.raises(ValueError, match=claim):
        a.check_scope(r, Checks())


def test_changed_source_or_expression_version_rejected():
    r = _report()
    r["source_hashes_after"]["script"] = "b"
    with pytest.raises(ValueError, match="Source"):
        a.check_scope(r, Checks())
    r = _report()
    r["experiment_version"] = 33
    with pytest.raises(ValueError, match="v34"):
        a.check_scope(r, Checks())
