"""Small saved-value checks; no model, native operator or dataset execution."""
import copy
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest


def load_auditor():
    local = Path(__file__).with_name("audit-v36-frame-cache-replay.py")
    published = Path(__file__).resolve().parents[1] / "deploy/rk3588/validation/dpvo_frame_cache/audit-v36-frame-cache-replay.py"
    path = Path(os.environ["EDGE_V36_CACHE_AUDITOR"]) if "EDGE_V36_CACHE_AUDITOR" in os.environ else (local if local.is_file() else published)
    spec = importlib.util.spec_from_file_location("v36_cache_saved_auditor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = load_auditor()


def observation():
    source = {"frame_id": 12.25, "timestamp": 12.25}
    result = dict(frame_id=12.25, timestamp=12.25, observation_id=2,
                  captured_monotonic=20.0, initialized=True, admitted=True,
                  fresh_pose_valid=True, pose_w2c=[0., 0., 0., 0., 0., 0., 1.])
    row = dict(index=2, frame_id=12.25, timestamp=12.25, arrival_monotonic=20.,
               pose_available_monotonic=20.6, decode_undistort_crop_seconds=.1,
               encode_seconds=.2, stream_seconds=.3, processing_seconds=.6,
               result=result)
    return row, source


def test_valid_fresh_and_nonfresh_have_distinct_pose_contracts():
    row, source = observation()
    AUDIT.validate_observation(row, source, 2)
    row["result"].update(initialized=False, admitted=False, fresh_pose_valid=False, pose_w2c=None)
    AUDIT.validate_observation(row, source, 2)


@pytest.mark.parametrize("mutation", ("sensor_timestamp", "admission_identity", "arrival_clock", "timing_sum", "fresh_not_admitted", "reuse_nonfresh_pose"))
def test_invalid_saved_observation_rejected(mutation):
    row, source = observation()
    if mutation == "sensor_timestamp":
        source["timestamp"] += .01
    elif mutation == "admission_identity":
        row["result"]["observation_id"] += 1
    elif mutation == "arrival_clock":
        row["result"]["captured_monotonic"] += .01
    elif mutation == "timing_sum":
        row["stream_seconds"] += .01
    elif mutation == "fresh_not_admitted":
        row["result"]["admitted"] = False
    elif mutation == "reuse_nonfresh_pose":
        row["result"]["fresh_pose_valid"] = False
    with pytest.raises(ValueError):
        AUDIT.validate_observation(row, source, 2)


def test_only_declared_software_clocks_removed_from_semantic_compare():
    first = {"timestamp": 100., "diagnostic": {"captured_monotonic": 20., "process_seconds": .5, "residual": 3.}}
    second = copy.deepcopy(first)
    second["diagnostic"].update(captured_monotonic=40., process_seconds=1.)
    assert AUDIT.semantic(first) == AUDIT.semantic(second)
    second["timestamp"] += .01
    assert AUDIT.semantic(first) != AUDIT.semantic(second)
    second = copy.deepcopy(first)
    second["diagnostic"]["residual"] += .01
    assert AUDIT.semantic(first) != AUDIT.semantic(second)


def test_pose_compare_preserves_signed_zero_rejects_nonfinite_and_shape_change():
    positive = [0., 0., 0., 0., 0., 0., 1.]
    negative_zero = copy.deepcopy(positive)
    negative_zero[0] = -0.
    assert AUDIT.exact_pose(positive, positive)
    assert not AUDIT.exact_pose(positive, negative_zero)
    assert not AUDIT.exact_pose([np.nan] * 7, [np.nan] * 7)
    assert not AUDIT.exact_pose(positive, [positive])
    assert not AUDIT.exact_pose(positive, None)
    assert AUDIT.exact_pose(None, None)
