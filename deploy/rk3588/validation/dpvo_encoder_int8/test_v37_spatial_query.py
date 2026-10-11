"""Tiny spatial-lineage guards, with no native library or model execution."""
import ast
import copy
import importlib.util
import os
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ENTRY = HERE/"query-v37-spatial-int8-io-board.py"
if not ENTRY.is_file():
    ENTRY = HERE.parent/"deploy/rk3588/validation/dpvo_int8_inference/query-v37-spatial-int8-io-board.py"
SPEC = importlib.util.spec_from_file_location("spatial_query_tests", ENTRY)
Q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(Q)


def fixture():
    return dict(schema="v37_board_only_DPVO_spatial_encoder_PTQ",
        status="compiled_board_spatial_PTQ_parent_rejection_retained_not_runtime_validated",
        script_sha256=Q.COMPILER_SHA, configuration=Q.CONFIGURATION.copy(), released=True,
        cleanup_errors=[], sources_and_inputs_unchanged=True, source_pins={"source": "1"*64}, source_hashes_after={"source": "1"*64},
        diagnostic_only=True, candidate_accepted=False, production_changed=False, host_calibration_executed=False,
        calibration_location="aarch64 board CPU", PTQ_requested=True, INT8_operator_coverage_validated=False,
        deployment_inference_executed=False, original_FP32_gate=dict(rtol=1e-4, atol=1e-5, measured=False),
        input=dict(sha256=Q.GRAPH_SHA, original_graph_weights_preserved=True),
        ancestor=dict(method="spatial", sparse_onnx_sha256=Q.GRAPH_SHA, recipe_sha256=Q.RECIPE_SHA,
                      parent_onnx_sha256=Q.PARENT_GRAPH_SHA, parent_CPU_strict_gate_passed=False,
                      suffix_CPU_bitwise_gate_passed=True, spatial_parent_rejection_retained=True),
        diagnostic_allow_fp32_rejected=True, historical_parent_FP32_gate_passed=False,
        historical_parent_rejection_retained=True, parent_PTQ_source_sha256=Q.PARENT_PTQ_SHA,
        operations=[dict(name=name, code=0) for name in ["config", "load_onnx", "board_CPU_calibration_and_build", "export_rknn"]],
        calibration=dict(samples=32, rows=[{}]*32), output=dict(sha256="2"*64, bytes=1024))


def test_actual_contract_fixture_passes_without_promoting_parent():
    value = fixture()
    Q.validate_compile(value, "2"*64, 1024)
    assert value["ancestor"]["parent_CPU_strict_gate_passed"] is False and value["candidate_accepted"] is False


@pytest.mark.parametrize("path,bad", [
    (("diagnostic_allow_fp32_rejected",), False), (("historical_parent_FP32_gate_passed",), True),
    (("historical_parent_rejection_retained",), False), (("ancestor", "parent_CPU_strict_gate_passed"), True),
    (("ancestor", "recipe_sha256"), "0"*64), (("parent_PTQ_source_sha256",), "0"*64),
    (("input", "sha256"), "0"*64), (("host_calibration_executed",), True), (("released",), False),
])
def test_lineage_and_historical_rejection_cannot_be_silently_changed(path, bad):
    value = fixture()
    owner = value
    for key in path[:-1]:
        owner = owner[key]
    owner[path[-1]] = bad
    with pytest.raises(ValueError):
        Q.validate_compile(value, "2"*64, 1024)


def test_explicit_diagnostic_flag_checked_before_any_output_or_model_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(Q, "board_guard", lambda: None)
    output = tmp_path/"must-not-exist"
    argv = []
    for name in ("model", "compile-json", "abi-helper", "header", "frozen-query-helper"):
        argv.extend(["--" + name, str(tmp_path/"does-not-exist")])
    argv.extend(["--output-dir", str(output), "--model-sha256", "2"*64, "--compile-sha256", "3"*64])
    with pytest.raises(ValueError, match="Explicit.*required before"):
        Q.main(argv)
    assert not output.exists()


def test_zero_run_SDK_body_and_native_validation_keep_frozen_parent_AST():
    parent = Path(os.environ.get("V36_QUERY_IMPLEMENTATION", str(HERE.parents[1]/"edge-vggt-v36/toolchain/query-v36-int8-io-board.py")))
    if not parent.is_file():
        pytest.skip("requires pinned original query implementation; set V36_QUERY_IMPLEMENTATION")
    assert Q.sha(parent) == Q.PARENT_QUERY_SHA
    def functions(path):
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}
    before, after = functions(parent), functions(ENTRY)
    for name in ("require", "sha", "require_sha", "board_guard", "validate_attr", "bind_query_api", "probe_sdk", "load_abi"):
        assert before[name] == after[name], name
