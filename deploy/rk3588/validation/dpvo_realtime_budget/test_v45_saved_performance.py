"""Generated timing records only; no models, datasets or native computation."""
import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("perf", Path(__file__).with_name("analyze-v45-steady-performance.py"))
perf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(perf)


def toy():
    frames = []
    for i in range(2):
        frames.append(dict(index=i, frame_id=str(i), timestamp=float(i),
            arrival_monotonic=float(i), pose_available_monotonic=i + .51,
            decode_undistort_crop_resize_seconds=.1, encode_seconds=.1, stream_seconds=.3,
            publication_seconds=.01, admission_to_publication_seconds=.51,
            result=dict(fresh_pose_valid=True, observation_id=i, frame_id=str(i), timestamp=float(i), updates=1)))
    group = dict(model_key="B1_D4_M16", outputs_released=True, outputs_finite=True,
        weight_domain_valid=True, timing=dict(forward_seconds=.03, pack_seconds=.001,
            run_api_seconds=.02, input_set_seconds=.001, outputs_get_seconds=.001,
            owned_copy_seconds=.001, outputs_release_seconds=.001))
    sdk = dict(full_graph_seconds=.04, groups=[group], sequential_groups=1,
        full_graph_returned=True, padding=False, dropped_edges=False)
    callback = dict(index=0, edges=64, frame_index=1, stage="stream", route="full",
        callback_seconds=.05, SDK_execution=sdk)
    corr = dict(index=0, edges=64, total_seconds=.1, preparation_seconds=.01,
        native_query=dict(native=dict(kernel_seconds=.08)))
    return dict(frames=frames, correlation=dict(call_history_dropped=0, calls=[corr]),
        online_update_records=[callback], update_calls=[dict(index=0, edges=64, frame_index=1,
            stage="stream", route="full", SDK_record_index=0, seconds=.06, capture_and_metadata_seconds=.01)],
        BA_calls=[dict(index=0, frame_index=1, stage="stream", seconds=.02)],
        encoder_metrics=dict(calls=[dict(index=i, SDK_timing=dict(sdk_io_seconds=.06, run_api_seconds=.05)) for i in range(2)]))


def test_disjoint_stream_remainder_and_nested_clocks():
    row, = perf.steady_rows(toy(), 1, 1)
    assert row["stream_remainder"] == pytest.approx(.11)
    assert row["SDK_run_api"] == .02
    assert row["callback"] == .05  # Not added to Update or stream.


@pytest.mark.parametrize("mutation", ["duplicate_update", "clock_overlap", "not_fresh", "identity", "missing_corr", "nan"])
def test_refuse_bad_saved_timing_or_join(mutation):
    run = toy()
    if mutation == "duplicate_update":
        run["update_calls"].append(copy.deepcopy(run["update_calls"][0]))
    elif mutation == "clock_overlap":
        run["update_calls"][0]["seconds"] = .5
    elif mutation == "not_fresh":
        run["frames"][1]["result"]["fresh_pose_valid"] = False
    elif mutation == "identity":
        run["online_update_records"][0]["edges"] = 63
    elif mutation == "missing_corr":
        run["correlation"]["call_history_dropped"] = 1
    else:
        run["BA_calls"][0]["seconds"] = float("nan")
    with pytest.raises(ValueError):
        perf.steady_rows(run, 1, 1)


def test_linear_quantiles_and_units():
    s = perf.statistics([.001, .003])
    assert s["mean"] == 2 and s["p50"] == 2
    assert s["p95"] == pytest.approx(2.9)
