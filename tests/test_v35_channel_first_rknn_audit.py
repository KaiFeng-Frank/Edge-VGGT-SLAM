"""Small synthetic SDK-record audit tests; no SDK or inference runtime."""
import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "deploy/rk3588/validation/dpvo_source_frame_channel_first"
spec = importlib.util.spec_from_file_location("v35_cf_sdk_saved_audit", HERE / "audit-v35-channel-first-rknn.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
helper = audit.load_pinned(REPO / "deploy/rk3588/validation/dpvo_source_frame/audit-v33-source-frame-rknn.py",
                           audit.HELPER_SHA, "v35_tiny_record_helpers")


def scope_record():
    value = dict(schema="v35_DPVO_source_frame_channel_first_board_export", experiment_version=35,
                 status="completed_channel_first_export_diagnostic_not_adopted",
                 expression_variant="v35_channel_first_original_parameter_conv",
                 inputs=[dict(name=n, dtype="float32", shape=[1,c,10,96])
                         for n,c in zip(("net","inp","corr"),(384,384,882))],
                 outputs=[dict(name=n, dtype="float32", shape=[1,c,10,96])
                          for n,c in zip(audit.OUTPUTS,(384,2,2))],
                 input_layout="B,C,D,M", output_layout="B,C,D,M", canonical_layout="B,D,M,C",
                 outputs_finite=True, weight_domain_valid=True, original_parameters_unchanged=True,
                 cleanup_errors=[], ONNX_operator_counts={"Conv":21}, Softmax_axes=[2,3])
    for name in ("candidate_accepted","host_model_inference","NPU_used","BA_executed","valid_pose_output",
                 "realtime_validated","trajectory_accuracy_validated","FP32_thresholds_relaxed","output_clipping",
                 "full_E23520_reference_reexecuted","source_plan_recomputed","SDK_or_NPU_placement_validated","controlled_speedup_validated"):
        value[name] = False
    return value


def test_CF_contract_cannot_relabel_old_BDMC():
    record = scope_record()
    audit.check_export_scope(record, helper.Checks())
    record["inputs"][0]["shape"] = [1,10,96,384]
    with pytest.raises(ValueError, match="CF IO contract"):
        audit.check_export_scope(record, helper.Checks())


def profile_record():
    items = [("InputOperator","CPU")] * 3 + [("Conv","NPU")] * 42
    items += [("Transpose","NPU")] * 6 + [("exSoftmax13","NPU")] * 2
    items += [("Exp","CPU"),("Min","CPU"),("Exp","CPU")] + [("OutputOperator","CPU")] * 3
    text = "\n".join(f"{i} {op} FLOAT16 {target} input output 0/0/0 5 0" for i,(op,target) in enumerate(items,1)) + "\n"
    return dict(raw_text=text, bytes=len(text.encode()), raw_sha256=hashlib.sha256(text.encode()).hexdigest(),query_command=3)


@pytest.mark.parametrize("change", ["raw_hash", "transpose_target"])
def test_profile_actual_targets_and_raw_integrity(change):
    record = profile_record()
    result = audit.verify_cf_placement(record, helper.Checks(), helper)
    assert result["NPU_Transpose_count"] == 6 and result["internal_CPU"] == 3
    assert not result["entire_computation_on_NPU"]
    if change == "raw_hash":
        record["raw_sha256"] = "0" * 64
    else:
        text = record["raw_text"].replace("Transpose FLOAT16 NPU", "Transpose FLOAT16 CPU", 1)
        record.update(raw_text=text,bytes=len(text.encode()),raw_sha256=hashlib.sha256(text.encode()).hexdigest())
    with pytest.raises(ValueError):
        audit.verify_cf_placement(record, helper.Checks(), helper)


def transport_record():
    inputs = [dict(name=n,shape=[1,c,2,3],dtype="float32") for n,c in zip(("net","inp","corr"),(384,384,882))]
    outputs = [dict(name=n,shape=[1,c,2,3],dtype="float32") for n,c in zip(audit.OUTPUTS,(384,2,2))]
    session = dict(input_attributes=[],output_attributes=[],output_bindings=[])
    for contracts, output in ((inputs,False),(outputs,True)):
        for i,c in enumerate(contracts):
            shape=c["shape"];size=shape[0]*shape[1]*shape[2]*shape[3]
            native=shape if output else [shape[0],shape[2],shape[3],shape[1]]
            row=dict(index=i,name=c["name"],shape=native,n_elems=size,size=size*2,type=1,format=0 if output else 1,
                     quantization_type=2,scale=1.,zero_point=0,fractional_length=0,pass_through=0,h_stride=0,size_with_stride=size*2)
            session["output_attributes" if output else "input_attributes"].append(row)
            if output:
                session["output_bindings"].append(dict(name=c["name"],shape=shape,logical_shape=shape,logical_dtype="float32",
                    sdk_shape=native,index=i,sdk_native_bytes=size*2,sdk_size_with_stride=size*2,native_output_padding_bytes=0,
                    returned_FP32_bytes=size*4,transport_bytes=size*4,want_float=1,permutation=None,
                    transport_type=0,transport_format=0,pass_through=0,sdk_format=0,sdk_type=1,sdk_quantization_type=2,
                    neutral_affine_metadata=True,sdk_fractional_length=0,sdk_zero_point=0,sdk_scale=1.))
    return inputs,outputs,session


@pytest.mark.parametrize("change", ["returned_size", "input_fmt", "neutral_scale"])
def test_actual_transport_format_size_and_neutral_FP16_guards(change):
    inputs,outputs,session = transport_record()
    helper.verify_transport(session,inputs,outputs,helper.Checks())
    bad=copy.deepcopy(session)
    if change == "returned_size":
        bad["output_bindings"][0]["returned_FP32_bytes"] += 4
    elif change == "input_fmt":
        bad["input_attributes"][0]["format"] = 0
    else:
        bad["input_attributes"][0]["scale"] = .5
    with pytest.raises(ValueError):
        helper.verify_transport(bad,inputs,outputs,helper.Checks())
