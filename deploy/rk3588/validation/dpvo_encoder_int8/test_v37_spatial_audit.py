"""Tiny adapter lineage/dependency guards; no real arrays or runtime execution."""
import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ENTRY = HERE/"audit-v37-spatial-int8.py"
if not ENTRY.is_file():
    ENTRY = HERE.parent/"deploy/rk3588/validation/dpvo_encoder_int8/audit-v37-spatial-int8.py"
SPEC = importlib.util.spec_from_file_location("spatial_saved_audit_tests", ENTRY)
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


class Checks:
    def check(self, condition, message):
        if not condition:
            raise ValueError(message)


def fixture():
    return dict(schema="v37_DPVO_spatial_sparse_encoder_PTQ_inference_diagnostic",
        status="completed_spatial_encoder_PTQ_parent_rejection_retained_not_VO", script_sha256=A.ENTRY_SHA,
        diagnostic_allow_fp32_rejected=True, historical_parent_FP32_gate_passed=False, historical_parent_rejection_retained=True,
        source_spatial_graph_sha256=A.GRAPH_SHA, reference_method="spatial", reference_spatial_suffix_CPU_bitwise_gate_passed=True,
        original_FP32_source_graph=False, spatial_FP32_graph_rewrite_used=True, original_DPVO_weights_preserved=True,
        candidate_accepted=False, historical_FP16_receipt_sha256=None, calls=[{}])


def test_spatial_saved_only_lineage_fixture_passes_without_promotion():
    A.spatial_declaration(fixture(), Checks())


@pytest.mark.parametrize("key,bad", [("historical_parent_FP32_gate_passed", True),
    ("candidate_accepted", True), ("source_spatial_graph_sha256", "0"*64), ("historical_FP16_receipt_sha256", "0"*64)])
def test_adapter_rejects_promoted_or_invented_reference_fields(key, bad):
    value = fixture()
    value[key] = bad
    with pytest.raises(ValueError):
        A.spatial_declaration(value, Checks())


def test_unpinned_dependency_rejected_before_import(tmp_path):
    path = tmp_path/"wrong.py"
    path.write_text("raise AssertionError('must not import')\n")
    with pytest.raises(ValueError, match="math SHA differs"):
        A.load_math(path)
