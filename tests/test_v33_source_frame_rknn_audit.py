"""Tiny NumPy/metadata tests only; no SDK, real model or ORT execution."""
import copy
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _auditor_path():
    local = Path(__file__).with_name("audit-v33-source-frame-rknn.py")
    if local.is_file():
        return local
    return Path(__file__).resolve().parents[1] / "deploy/rk3588/validation/dpvo_source_frame/audit-v33-source-frame-rknn.py"


_spec = importlib.util.spec_from_file_location("v33_saved_rknn_audit", _auditor_path())
a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a)


def test_fixed_gate_reject_not_promoted_by_small_relative_l2():
    ref = np.asarray([1000.0, 0.0], dtype=np.float32)
    actual = ref.copy()
    actual[1] = 2e-5
    result = a.compare(actual, ref)
    assert result["relative_l2"] < 1e-7 and result["failed_elements"] == 1
    assert result["gate_passed"] is False
    recorded = {k: v for k, v in result.items() if k != "max_normalized_tolerance"}
    a.check_comparison(result, recorded, a.Checks())
    recorded["gate_passed"] = True
    with pytest.raises(ValueError, match="gate_passed"):
        a.check_comparison(result, recorded, a.Checks())


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_is_not_a_numerical_pass(bad):
    with pytest.raises(ValueError, match="Finite"):
        a.compare(np.asarray([bad], np.float32), np.asarray([1], np.float32))


def test_wrong_dtype_or_shape_rejected():
    with pytest.raises(ValueError, match="Matching"):
        a.compare(np.ones((2,), np.float64), np.ones((2,), np.float32))
    with pytest.raises(ValueError, match="Matching"):
        a.compare(np.ones((1, 2), np.float32), np.ones((2,), np.float32))


def _profile(lines):
    raw = "\n".join(lines) + "\n"
    return dict(query_command=3, bytes=len(raw.encode()), raw_text=raw,
                raw_sha256=hashlib.sha256(raw.encode()).hexdigest())


def test_profile_placement_is_actual_rows_not_summary_substrings():
    record = _profile(["1 InputOperator FLOAT16 CPU \\ (1,2) 0/0/0 2 0.0% Input",
                       "2 Conv FLOAT16 NPU (1,2) (1,2) 0/21/21 7 100.0% conv",
                       "3 Exp FLOAT16 CPU (1,2) (1,2) 0/0/0 4 0.0% exp",
                       "4 OutputOperator FLOAT16 CPU (1,2) \\ 0/0/0 1 0.0% Output",
                       "NPU Exp Conv 1000000 1000000 ranking summary"])
    result = a.parse_profile(record, a.Checks())
    assert (result["operator_rows"], result["internal_NPU"], result["internal_CPU"], result["CPU_IO"]) == (4, 1, 1, 2)
    assert result["internal_CPU_operators"] == {"Exp": 1}
    assert result["internal_NPU_time_us"] == 7 and result["entire_computation_on_NPU"] is False


@pytest.mark.parametrize("rows", [["1 Conv FLOAT16 NPU (1) (1) 0/1/1 1 tail",
                                    "1 Exp FLOAT16 CPU (1) (1) 0/0/0 1 tail"],
                                   ["2 Conv FLOAT16 NPU (1) (1) 0/1/1 1 tail"]])
def test_profile_duplicate_or_missing_ids_rejected(rows):
    with pytest.raises(ValueError, match="operator IDs"):
        a.parse_profile(_profile(rows), a.Checks())


def test_profile_raw_sha_is_checked():
    record = _profile(["1 Conv FLOAT16 NPU (1) (1) 0/1/1 1 tail"])
    record["raw_text"] += "changed"
    with pytest.raises(ValueError, match="profile SHA"):
        a.parse_profile(record, a.Checks())


def test_artifact_raw_sha_and_path_checked(tmp_path):
    values = {"delta": np.asarray([1, 2], dtype=np.float32)}
    path = tmp_path / "output.npz"
    np.savez(path, **values)
    item = dict(file=path.name, bytes=path.stat().st_size, sha256=a.file_sha(path),
                arrays={k: a.metadata(v) for k, v in values.items()})
    assert np.array_equal(a.load_artifact(tmp_path, item, a.Checks())["delta"], values["delta"])
    item["arrays"]["delta"]["tensor_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Raw array"):
        a.load_artifact(tmp_path, item, a.Checks())
    item["file"] = "../output.npz"
    with pytest.raises(ValueError, match="basename"):
        a.load_artifact(tmp_path, item, a.Checks())


def _call():
    contracts = [dict(name=k, shape=[1, 2, 3, 4]) for k in a.INPUTS]
    records = [dict(index=i, name=k, logical_shape=[1, 2, 3, 4], buffer_shape=[1, 3, 4, 2],
                    source_dtype="float32", transport_dtype="float32", native_attribute_format=1,
                    request_format=1, pass_through=0, bytes=96, owned=True) for i, k in enumerate(a.INPUTS)]
    timing = {k: 0.001 for k in ("pack_seconds", "input_set_seconds", "run_api_seconds", "outputs_get_seconds",
                                 "owned_copy_seconds", "outputs_release_seconds")}
    timing.update(forward_seconds=0.007, synchronous=True, input_records=records)
    return dict(timing=timing, total_execute_seconds=0.008, adapter_execute_wall_seconds=0.009), contracts


def test_timing_contains_copy_and_release_and_no_negative_values():
    call, contracts = _call()
    a.verify_call_timing(call, contracts, a.Checks())
    invalid = copy.deepcopy(call)
    invalid["timing"]["forward_seconds"] = 0.002
    with pytest.raises(ValueError, match="omits"):
        a.verify_call_timing(invalid, contracts, a.Checks())
    call["timing"]["outputs_release_seconds"] = -1
    with pytest.raises(ValueError, match="Invalid call time"):
        a.verify_call_timing(call, contracts, a.Checks())


def test_input_layout_and_ownership_not_inferred_from_elements():
    call, contracts = _call()
    call["timing"]["input_records"][0]["buffer_shape"] = [1, 2, 3, 4]
    with pytest.raises(ValueError, match="packing"):
        a.verify_call_timing(call, contracts, a.Checks())
    call, contracts = _call()
    call["timing"]["input_records"][0]["owned"] = False
    with pytest.raises(ValueError, match="packing"):
        a.verify_call_timing(call, contracts, a.Checks())


def test_source_before_after_and_unique_basename():
    record = dict(source_pins={"source/a.py": "1" * 64}, source_hashes_after={"source/a.py": "2" * 64},
                  sources_and_inputs_unchanged=True)
    with pytest.raises(ValueError, match="pins differ"):
        a.check_pins(record, a.Checks())
    with pytest.raises(ValueError, match="basename"):
        a.digest_by_basename({"one/a.py": "1" * 64, "two/a.py": "1" * 64}, "a.py", "1" * 64, a.Checks())


def test_native_padding_does_not_authorize_wrong_returned_bytes():
    contracts_in = [dict(name=k, shape=[1, 2, 3, 4]) for k in a.INPUTS]
    contracts_out = [dict(name=k, shape=[1, 2, 3, 4]) for k in a.OUTPUTS]
    def attr(index, contract, output):
        return dict(index=index, name=contract["name"], shape=[1, 2, 3, 4] if output else [1, 3, 4, 2],
                    n_elems=24, size=48, type=1, format=0 if output else 1, quantization_type=2,
                    scale=1.0, zero_point=0, fractional_length=0, pass_through=0, h_stride=0,
                    size_with_stride=64 if output else 48)
    bindings = [dict(name=k, shape=[1, 2, 3, 4], logical_shape=[1, 2, 3, 4], logical_dtype="float32",
                     sdk_shape=[1, 2, 3, 4], index=i, sdk_native_bytes=48, sdk_size_with_stride=64,
                     native_output_padding_bytes=16, returned_FP32_bytes=96, transport_bytes=96,
                     want_float=1, permutation=None, transport_type=0, transport_format=0, pass_through=0,
                     sdk_format=0, sdk_type=1, sdk_quantization_type=2, neutral_affine_metadata=True,
                     sdk_fractional_length=0, sdk_zero_point=0, sdk_scale=1.0) for i, k in enumerate(a.OUTPUTS)]
    session = dict(input_attributes=[attr(i, c, False) for i, c in enumerate(contracts_in)],
                   output_attributes=[attr(i, c, True) for i, c in enumerate(contracts_out)], output_bindings=bindings)
    a.verify_transport(session, contracts_in, contracts_out, a.Checks())
    session["output_bindings"][0]["returned_FP32_bytes"] = 128
    with pytest.raises(ValueError, match="logical FP32"):
        a.verify_transport(session, contracts_in, contracts_out, a.Checks())
