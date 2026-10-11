"""Metadata/guard checks only; never calibrate or build a neural model."""
import copy
import importlib.util
import os
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ENTRY = Path(os.environ.get("RECIPROCAL_PTQ_DRIVER",HERE/"compile-v38-reciprocal-int8-board.py"))
if not ENTRY.exists():
    ENTRY = HERE.parent/"deploy/rk3588/validation/dpvo_encoder_reciprocal/compile-v38-reciprocal-int8-board.py"
SPEC = importlib.util.spec_from_file_location("v38_test_PTQ",ENTRY)
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


def evidence():
    return dict(schema="v38_DPVO_spatial_reciprocal_board_CPU_ORT",
        status="completed_CPU_ORT_reciprocal_diagnostic_parent_rejection_retained_not_VO",
        script_sha256="a"*64,rewrite_sha256="b"*64,rewriter_sha256="c"*64,parent_graph_sha256="d"*64,candidate_graph={"sha256":"e"*64},
        candidate_vs_current_parent_strict_gate_passed=True,parent_reproduced_saved_v33_strict_gate_passed=True,
        original_CPU_strict_gate_each_role={"parent":False,"candidate":False},historical_parent_FP32_gate_passed=False,
        historical_parent_rejection_retained=True,candidate_accepted=False,gate={"rtol":1e-4,"atol":1e-5},
        graph_optimization="ORT_DISABLE_ALL",providers=["CPUExecutionProvider"],cpu_affinity=[4,5,6,7],affinity_restored=True,
        session_lifecycle={"ORT_session_references_dropped":True},sources_and_inputs_unchanged=True,source_pins={},source_hashes_after={},cleanup_errors=[],
        calls=[dict(frame_index=i,role=role,**({"versus_current_parent_ORT":{"fmap":{"gate_passed":True},"context":{"gate_passed":True}}} if role=="candidate" else {}))
               for i in range(3) for role in ("parent","candidate")])


def gate(value):
    D.cpu_gate(value,"a"*64,"b"*64,"c"*64,"d"*64,"e"*64)


def test_CPU_gate_accepts_only_diagnostic_preserving_parent_rejection():
    gate(evidence())


@pytest.mark.parametrize("bad",["strict","promote","graph","threshold","count","individual","cleanup"])
def test_PTQ_refuses_failed_wrong_or_relaxed_CPU_evidence(bad):
    value=evidence()
    if bad=="strict": value["candidate_vs_current_parent_strict_gate_passed"]=False
    elif bad=="promote": value["historical_parent_FP32_gate_passed"]=True
    elif bad=="graph": value["candidate_graph"]["sha256"]="f"*64
    elif bad=="threshold": value["gate"]["atol"]=1e-3
    elif bad=="count": value["calls"].pop()
    elif bad=="individual": value["calls"][1]["versus_current_parent_ORT"]["fmap"]["gate_passed"]=False
    else: value["cleanup_errors"]=["session did not release"]
    with pytest.raises(ValueError):gate(value)


def test_32_calibration_rows_and_all64_ordered_file_pins_must_match():
    rows=[dict(index=i,input_files=[dict(file=f"image-{i}.npy",sha256="a"*64,bytes=100),dict(file=f"indices-{i}.npy",sha256="b"*64,bytes=96)]) for i in range(32)]
    parent={"calibration":{"samples":32,"rows":copy.deepcopy(rows)}}
    assert D.same_calibration(rows,parent)["input_files"]==64
    changed=copy.deepcopy(rows)
    changed[5]["input_files"][1]["sha256"]="c"*64
    with pytest.raises(ValueError):D.same_calibration(changed,parent)
    with pytest.raises(ValueError):D.same_calibration(rows[:-1],parent)
    with pytest.raises(ValueError):D.same_calibration(rows[::-1],parent)


@pytest.mark.parametrize("machine,system,compatible",[("x86_64","Linux",b"rockchip,rk3588"),("aarch64","Linux",b"other,ARM"),("aarch64","Darwin",b"rockchip,rk3588")])
def test_calibration_guard_refuses_host_otherARM_and_wrongOS(machine,system,compatible):
    with pytest.raises(ValueError):D.board_guard(machine,system,compatible)


def test_main_hostguard_precedes_output_or_model_helper_import(tmp_path,monkeypatch):
    argv=[]
    for name in ("dependency-plan","preflight-script","frozen-ptq-helper","ort-driver","parent-ptq-json","parent-onnx","rewriter","rewrite-dir","ort-dir","dataset","output-dir"):
        argv.extend(["--"+name,str(tmp_path/name)])
    for name in ("dependency-plan","preflight-script","ort-driver","rewriter","rewrite","ort"):
        argv.extend(["--"+name+"-sha256","a"*64])
    argv.append("--diagnostic-allow-fp32-rejected")
    monkeypatch.setattr(D.platform,"machine",lambda:"x86_64")
    monkeypatch.setattr(D,"load_pinned",lambda *a,**k:pytest.fail("PTQ dependencies reached on host"))
    with pytest.raises(ValueError,match="Host PTQ prohibited"):D.main(argv)
    assert not (tmp_path/"output-dir").exists()
