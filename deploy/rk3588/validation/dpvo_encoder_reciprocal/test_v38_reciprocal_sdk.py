"""Small lineage/host guards only; frozen SDK arithmetic is not reimplemented."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE=Path(__file__).resolve().parent
def entry(name):
    path=Path(os.environ.get("V38_RECIPROCAL_SOURCE",HERE))/name
    if not path.exists():path=HERE.parent/"deploy/rk3588/validation/dpvo_encoder_reciprocal"/name
    spec=importlib.util.spec_from_file_location(name.replace("-","_"),path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
Q=entry("query-v38-reciprocal-int8-io-board.py")
D=entry("run-v38-reciprocal-int8-board.py")


def compile_fixture():
    rows=[dict(index=i,input_files=[dict(sha256="a"*64),dict(sha256="b"*64)]) for i in range(32)]
    parent=dict(ancestor={"method":"spatial"},configuration=dict(do_quantization=True,auto_hybrid=False),
        calibration=dict(samples=32,rows=rows,dataset_sha256="c"*64))
    recipe=dict(output={"sha256":"d"*64},preservation={"static_weights_exact":True})
    evidence=dict(original_CPU_strict_gate_each_role={"parent":False,"candidate":False})
    args=SimpleNamespace(rewrite_sha256="e"*64,ort_sha256="f"*64)
    value=dict(schema="v38_board_only_DPVO_reciprocal_spatial_encoder_PTQ",
        status="compiled_board_reciprocal_PTQ_parent_rejection_retained_not_runtime_validated",script_sha256=Q.PTQ_SHA,
        parent_PTQ_source_sha256="0fc013d5b6ed45cb97d592e8cd115304d971a000e2ab5353314413f56d41a3fa",parent_PTQ_receipt_sha256=Q.PARENT_PTQ_SHA,
        ancestor=parent["ancestor"],configuration=parent["configuration"],released=True,cleanup_errors=[],sources_and_inputs_unchanged=True,
        source_pins={},source_hashes_after={},diagnostic_only=True,candidate_accepted=False,production_changed=False,
        diagnostic_allow_fp32_rejected=True,historical_parent_FP32_gate_passed=False,historical_parent_rejection_retained=True,
        host_calibration_executed=False,calibration_location="RK3588 aarch64 board CPU",PTQ_requested=True,
        INT8_operator_coverage_validated=False,deployment_inference_executed=False,
        input=dict(sha256=recipe["output"]["sha256"],original_graph_weights_preserved=True),graph_preservation=recipe["preservation"],
        rewrite_sha256=args.rewrite_sha256,rewriter_sha256=Q.REWRITER_SHA,ORT_receipt_sha256=args.ort_sha256,ORT_driver_sha256=Q.CPU_DRIVER_SHA,
        candidate_vs_current_parent_CPU_strict_gate_passed=True,candidate_vs_original_CPU_strict_gate_passed=False,
        calibration={**parent["calibration"],"source_dataset_sha256":"5c4c4712a2241fd0dd3cf83de2f454386f22bcdff4d1695077a8cd545b018372"},
        operations=[dict(name=k,code=0) for k in ["config","load_onnx","board_CPU_calibration_and_build","export_rknn"]],
        output=dict(sha256="1"*64,bytes=100))
    return value,recipe,evidence,parent,args


@pytest.mark.parametrize("bad",["graph","cpu_gate","parent_acceptance","calibration","release"])
def test_new_compile_lineage_rejects_changed_graph_gates_dataset_and_cleanup(bad):
    value,recipe,evidence,parent,args=compile_fixture()
    Q.validate_compile(value,recipe,evidence,parent,args)
    if bad=="graph":value["input"]["sha256"]="2"*64
    elif bad=="cpu_gate":value["candidate_vs_current_parent_CPU_strict_gate_passed"]=False
    elif bad=="parent_acceptance":value["historical_parent_FP32_gate_passed"]=True
    elif bad=="calibration":value["calibration"]["source_dataset_sha256"]="3"*64
    else:value["released"]=False
    with pytest.raises(ValueError):Q.validate_compile(value,recipe,evidence,parent,args)


def query_fixture():
    return dict(schema="v38_DPVO_reciprocal_PTQ_native_IO_query",
        status="queried_reciprocal_native_IO_parent_rejection_retained_not_inference_validated",script_sha256=D.QUERY_SHA,
        compile_receipt_sha256="a"*64,model_sha256="b"*64,rewrite_sha256="c"*64,CPU_ORT_receipt_sha256="d"*64,
        source_reciprocal_graph_sha256="e"*64,source_parent_spatial_graph_sha256="6c3b4de4d151813dc172d711b6a760ad8bdc17120f2f3faa76dbf32bd0c451ab",
        sdk_run_calls=0,model_inference_executed=False,native_IO_contract_checked=True,initialized=True,context_destroyed=True,
        cleanup_errors=[],sources_and_inputs_unchanged=True,source_pins={},source_hashes_after={},diagnostic_allow_fp32_rejected=True,
        historical_parent_FP32_gate_passed=False,historical_parent_rejection_retained=True,candidate_accepted=False,
        parent_query_implementation_sha256=D.PARENT_QUERY_SHA)


@pytest.mark.parametrize("bad",["model","run","destroy","promote"])
def test_runtime_query_guard_uses_actual_new_model_and_zero_run_ownership(bad):
    value=query_fixture()
    validate=lambda:D.validate_query(value,*[c*64 for c in "abcde"])
    validate()
    if bad=="model":value["model_sha256"]="f"*64
    elif bad=="run":value["sdk_run_calls"]=1
    elif bad=="destroy":value["context_destroyed"]=False
    else:value["candidate_accepted"]=True
    with pytest.raises(ValueError):validate()


@pytest.mark.parametrize("driver",[Q,D])
def test_host_guard_refuses_before_SDK_or_model_input_reads(driver):
    driver.board_guard("aarch64","Linux",b"rockchip,rk3588s\0")
    with pytest.raises(ValueError):driver.board_guard("x86_64","Linux",b"rockchip,rk3588\0")
    with pytest.raises(ValueError):driver.board_guard("aarch64","Linux",b"other,ARM\0")


def test_pinned_module_refuses_tampered_source_before_import(tmp_path):
    source=tmp_path/"bad.py";source.write_text("raise RuntimeError('must not execute')")
    def pin(path,digest):
        assert D.sha(path)!=digest
        raise ValueError("SHA guard")
    with pytest.raises(ValueError,match="SHA guard"):D.module(source,"a"*64,"fake",pin)
