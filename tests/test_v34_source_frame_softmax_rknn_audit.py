"""Tiny NumPy/profile metadata tests; no model, SDK or ORT execution."""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
path = HERE / "audit-v34-source-frame-softmax-rknn.py"
if not path.is_file():
    HERE = HERE.parent / "deploy/rk3588/validation/dpvo_source_frame_softmax"
    path = HERE / path.name
spec = importlib.util.spec_from_file_location("v34_saved_rknn_audit", path)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
helper = HERE.parent / "vggt-v33-dpvo-update/audit-v33-source-frame-rknn.py"
if not helper.is_file():
    helper = HERE.parent / "dpvo_source_frame/audit-v33-source-frame-rknn.py"
h = a.load_pinned(helper, a.HELPER_SHA, "v34_pinned_numeric_helpers")


def profile_record(change=None):
    operations = [("InputOperator", "CPU")] * 3 + [("OutputOperator", "CPU")] * 3
    operations += [("Exp", "CPU"), ("Min", "CPU"), ("Exp", "CPU")]
    operations += [("exSoftmax13", "NPU")] * 2 + [("Transpose", "NPU")] * 10
    operations += [("Conv", "NPU")] * 43
    if change:
        operations[9] = change
    raw = "\n".join(f"{i} {op} FLOAT16 {target} (1) (1) 0/0/0 1 tail"
                    for i, (op, target) in enumerate(operations, 1)) + "\n"
    return dict(query_command=3, raw_text=raw, bytes=len(raw.encode()),
                raw_sha256=hashlib.sha256(raw.encode()).hexdigest())


def test_actual_profile_classification_and_direct_softmax_requirement():
    result = a.verify_v34_placement(profile_record(), h.Checks(), h)
    assert (result["internal_NPU"], result["internal_CPU"], result["CPU_IO"]) == (55, 3, 6)
    assert result["NPU_Softmax_count"] == 2 and result["ReduceMax_count"] == 0
    assert result["entire_computation_on_NPU"] is False
    # Total NPU/CPU counts alone would miss this wrong operator identity.
    with pytest.raises(ValueError, match="Softmax/Transpose"):
        a.verify_v34_placement(profile_record(("Conv", "NPU")), h.Checks(), h)


def test_cpu_softmax_or_forged_raw_profile_not_accepted():
    with pytest.raises(ValueError, match="placement"):
        a.verify_v34_placement(profile_record(("exSoftmax13", "CPU")), h.Checks(), h)
    record = profile_record()
    record["raw_text"] += "edited\n"
    with pytest.raises(ValueError, match="profile SHA"):
        a.verify_v34_placement(record, h.Checks(), h)


def test_unpinned_python_helper_is_never_executed(tmp_path):
    marker = tmp_path / "must-not-exist"
    helper = tmp_path / "forged.py"
    helper.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    with pytest.raises(ValueError, match="helper SHA"):
        a.load_pinned(helper, "0" * 64, "untrusted_helper")
    assert not marker.exists()


def test_small_relative_l2_cannot_hide_one_fixed_gate_failure():
    reference = np.asarray([1000, 0], dtype=np.float32)
    candidate = np.asarray([1000, 2e-5], dtype=np.float32)
    computed = h.compare(candidate, reference)
    assert computed["relative_l2"] < 1e-7 and computed["failed_elements"] == 1
    assert computed["gate_passed"] is False
    recorded = {k: v for k, v in computed.items() if k != "max_normalized_tolerance"}
    recorded["gate_passed"] = True
    with pytest.raises(ValueError, match="gate_passed"):
        h.check_comparison(computed, recorded, h.Checks())
