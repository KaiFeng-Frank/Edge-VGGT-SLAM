"""Tiny saved-value arithmetic/tampering checks; no encoder or SDK execution."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest


def find_helper(name):
    here = Path(__file__).resolve().parent
    paths = (here / name, here.parent / "deploy/rk3588/validation/dpvo_encoder_opt" / name)
    return next(path for path in paths if path.is_file())


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def helper():
    return load(find_helper("audit-v32-dpvo-encoder.py"), "_v32_audit_tiny")


@pytest.fixture
def parser(helper):
    here = Path(__file__).resolve().parent
    candidates = (here / "audit-v26-dpvo-rknn.py",
                  here.parent / "deploy/rk3588/validation/audit-v26-dpvo-rknn.py")
    if os.environ.get("V32_V26_AUDITOR"):
        candidates = (Path(os.environ["V32_V26_AUDITOR"]), *candidates)
    return helper.load_module(next(path for path in candidates if path.is_file()), helper.PARSER_SHA, "_v32_tiny_parser")


def test_strict_adjacent_fp32_threshold_is_not_relaxed(helper, parser):
    reference = np.array([1.0], np.float32)
    boundary = np.float32(1.00011)
    below = np.nextafter(boundary, np.float32(0))
    above = np.nextafter(boundary, np.float32(2))
    assert helper.compare(np.array([below]), reference, parser)["failed_elements"] == 0
    assert helper.compare(np.array([above]), reference, parser)["failed_elements"] == 1


def test_nonfinite_raw_output_remains_rejected_and_json_safe(helper, parser):
    actual = np.array([np.nan, np.inf, -np.inf, 0], np.float32)
    row = helper.compare(actual, np.ones(4, np.float32), parser)
    assert row["gate_passed"] is False and row["relative_l2"] is None
    assert row["finite_counts"] == dict(elements=4, finite=1, nan=1, positive_inf=1, negative_inf=1, all_zero=False)
    json.dumps(row, allow_nan=False)


def test_two_warm_means_do_not_include_first_or_other_frames(helper):
    calls = [dict(frame_index=frame, method=method, timing_role=role, forward_seconds=float(i + 1))
             for i, (frame, method, role) in enumerate(helper.schedule())]
    result = helper.summary(calls)
    assert len(calls) == 10
    assert result["original"]["warm_seconds"] == [4., 5.]
    assert result["spatial"]["warm_seconds"] == [3., 6.]
    assert result["original"]["warm_mean_seconds"] == result["spatial"]["warm_mean_seconds"] == 4.5
    assert helper.schedule(True) == [(i, "spatial", "profile") for i in range(3)]


def profile_detail():
    text = "\n".join([
        "1 InputOperator FLOAT16 CPU x x 0/0/0 2 0 0.0%/0.0%/0.0% 0 InputOperator:input",
        "2 Transpose FLOAT16 CPU (1,2) (2,1) 0/0/0 3 0 0.0%/0.0%/0.0% 0 Transpose:layout",
        "3 Conv FLOAT16 NPU (1,2) (1,2) 0/0/0 5 0 100.0%/0.0%/0.0% 0 Conv:original",
        "4 OutputOperator FLOAT16 CPU x x 0/0/0 1 0 0.0%/0.0%/0.0% 0 OutputOperator:output",
        "Total Operator Elapsed Per Frame Time(us): 11\n"])
    raw = text.encode()
    return dict(raw_text=text, bytes=len(raw), raw_sha256=hashlib.sha256(raw).hexdigest())


def test_actual_profile_targets_and_elapsed_sum(helper, parser):
    value = helper.summarize_profile(profile_detail(), parser)
    assert value["internal_target_counts"] == dict(CPU=1, NPU=1)
    assert value["internal_CPU_transpose_elapsed_us"] == 3
    assert value["CPU_IO_elapsed_us"] == 3
    assert value["target_elapsed_us"] == dict(CPU=6, NPU=5)
    assert value["operator_elapsed_us"] == 11 and value["all_internal_ops_NPU"] is False


@pytest.mark.parametrize("failure", ["hash", "sum", "target"])
def test_profile_tampering_cannot_create_placement_claim(helper, parser, failure):
    detail = profile_detail()
    if failure == "hash":
        detail["raw_sha256"] = "0" * 64
    else:
        detail["raw_text"] = detail["raw_text"].replace("(us): 11", "(us): 12") if failure == "sum" else detail["raw_text"].replace("FLOAT16 NPU", "FLOAT16 FAKE")
        raw = detail["raw_text"].encode()
        detail.update(bytes=len(raw), raw_sha256=hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError):
        helper.summarize_profile(detail, parser)


def test_native_nhwc_input_and_nchw_output_remain_distinct(helper):
    shape = (1, 3, 4, 5)
    descriptor = dict(index=0, name="input", shape=[1, 4, 5, 3], format=1, type=1,
                      n_elems=60, size=120, quantization_type=2, zero_point=0, scale=1.)
    helper.validate_descriptor(descriptor, 0, "input", shape, is_input=True)
    with pytest.raises(ValueError, match="layout"):
        helper.validate_descriptor(descriptor, 0, "input", shape, is_input=False)
    wrong = copy.deepcopy(descriptor)
    wrong["scale"] = .5
    with pytest.raises(ValueError):
        helper.validate_descriptor(wrong, 0, "input", shape, is_input=True)
