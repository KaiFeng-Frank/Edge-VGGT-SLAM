"""Saved-array toy fixtures only, not captured real DPVO data or inference."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result

audit = load("v36_saved_capture_audit_toy", HERE / "audit-v36-fast-update-capture.py")
producer = load("v36_saved_capture_producer_toy", HERE / "capture-v36-fast-update-board.py")


def fixture(folder):
    capdir, parentdir = folder / "capture", folder / "parent"
    capdir.mkdir(); parentdir.mkdir()
    frames = [dict(observation_id=i, frame_id=str(i), timestamp=float(i),
                   captured_monotonic=float(i + 100), context=np.full((48, 384), i, np.float32))
              for i in (12, 13)]
    rows = np.array([(s, t, s * 48 + p) for s in range(2) for t in range(2) for p in range(48)], np.int64)
    ii, jj, kk = (rows[:, i].copy() for i in range(3))
    state = SimpleNamespace(frames=frames, ii=ii, jj=jj, kk=kk, updates=12,
        poses=np.zeros((1, 2, 7), np.float32), patches=np.ones((1, 96, 3, 3, 3), np.float32),
        intrinsics=np.ones((1, 2, 4), np.float32))
    context = np.stack([f["context"] for f in frames]).reshape(-1, 384)[kk][None].copy()
    before = (np.zeros((1, len(ii), 384), np.float32), context,
              np.zeros((1, len(ii), 882), np.float32), ii, jj, kk)
    recorder = producer.CaptureRecorder(capdir, [13])
    def original(*v):
        return dict(net=v[0] + np.float32(1), delta=np.zeros((1, len(ii), 2), np.float32),
                    weight=np.full((1, len(ii), 2), .5, np.float32))
    recorder.invoke(original, state, False, "online", before)
    manifest = dict(frames=[dict(frame_id=str(i), timestamp=float(i)) for i in range(14)])
    (parentdir / "input-manifest.json").write_text(json.dumps(manifest))
    weights = {"fake_%d" % i: producer.meta(np.zeros(1, np.float32)) for i in range(94)}
    parent = dict(schema="v35_DPVO_original_FAST_dense_spatial_RKNN_encoder_diagnostic_replay",
        status="completed_toy_fixture", script_sha256=producer.PARENT_SHA256,
        cleanup_errors=[], sources_and_inputs_unchanged=True, source_pins={"fake_source": "0" * 64},
        source_hashes_after={"fake_source": "0" * 64}, original_weights_unchanged=True,
        original_weights_before=weights, original_weights_after=weights,
        update_calls=[dict(index=0, edges=len(ii), seconds=0.)],
        config=dict(patches_per_frame=48), loop_closure=False,
        input_manifest_sha256=producer.sha(parentdir / "input-manifest.json"),
        frames_completed=14, frames=[dict(frame_id=str(i), timestamp=float(i), arrival_monotonic=float(i + 100)) for i in range(14)],
        historical_encoder_FP32_gate_passed=False)
    (parentdir / "run.json").write_text(json.dumps(parent))
    cap = dict(schema="v36_full_M48_original_Update_capture", status="completed_original_M48_capture_not_performance_or_candidate",
        script_sha256=audit.CAPTURE_SHA256, parent_runner_sha256=audit.PARENT_SHA256,
        parent_stream_sha256=audit.STREAM_SHA256, cleanup_errors=[],
        original_loader_and_stream_class_restored=True, parent_runner_source_unchanged=True,
        original_Update_replaced=False, original_weights_or_RNG_changed=False, upstream_revision_counters_exposed=False,
        timing_is_ordinary_performance=False, full_grouped_Update_or_NPU_candidate_executed=False,
        original_parent_receipt=producer.verify_parent_receipt(parentdir, recorder),
        original_call_attempts=1, original_call_successes=1, captured=1, capture_observations=[13],
        records=recorder.records, saved_artifact_bytes=recorder.saved_bytes)
    (capdir / "capture.json").write_text(json.dumps(cap))
    return capdir, parentdir, cap, parent


def rewrite_array(capdir, item, key, edit):
    path = capdir / item["file"]
    with np.load(path, allow_pickle=False) as z:
        values = {k: z[k].copy() for k in z.files}
    edit(values[key])
    np.savez(path, **values)
    item.update(bytes=path.stat().st_size, sha256=producer.sha(path),
                arrays={k: producer.meta(v) for k, v in values.items()})


def test_saved_capture_identity_and_all_payload_packscatter_budget(tmp_path):
    capdir, parentdir, _, _ = fixture(tmp_path)
    report = audit.audit_capture(capdir, parentdir)
    assert report["status"].startswith("passed_"), report.get("error")
    case = report["cases"][0]
    assert case["edges"] == 192 and case["topology"]["neural_calls"] == 1
    assert case["static_original_Linear_budget"]["graph_shapes"][0]["net"] == [2, 384, 2, 48]
    assert not report["learned_gold_independently_recomputed"]
    assert not report["actual_neural_grouped_numerics_or_timing_measured"]
    assert report["parent_encoder_FP32_gate_passed"] is False


@pytest.mark.parametrize("mutation", ["ctx", "gold_domain", "EOF", "parent_call", "rawfile"])
def test_tampered_saved_semantics_or_file_is_rejected_without_execution(tmp_path, mutation):
    capdir, parentdir, cap, parent = fixture(tmp_path)
    record = cap["records"][0]
    if mutation == "ctx":
        rewrite_array(capdir, record["inputs_artifact"], "inp", lambda x: x.__setitem__((0, 0, 0), 123.))
    elif mutation == "gold_domain":
        rewrite_array(capdir, record["original_gold_artifact"], "weight", lambda x: x.__setitem__((0, 0, 0), 1.01))
    elif mutation == "EOF": record["phase"] = "EOF"
    elif mutation == "parent_call":
        parent["update_calls"][0]["edges"] -= 1
        path = parentdir / "run.json"
        path.write_text(json.dumps(parent))
        cap["original_parent_receipt"].update(run_sha256=producer.sha(path), run_bytes=path.stat().st_size)
    else:
        with (capdir / record["inputs_artifact"]["file"]).open("ab") as f: f.write(b"tampered")
    cap["saved_artifact_bytes"] = sum(v["bytes"] for v in (record["inputs_artifact"], record["original_gold_artifact"]))
    (capdir / "capture.json").write_text(json.dumps(cap))
    result = audit.audit_capture(capdir, parentdir)
    assert result["status"] == "failed" and result["error"]
    expected = dict(ctx="ctx[kk]", gold_domain="weight domain", EOF="no EOF", parent_call="edge count", rawfile="artifact file")
    assert expected[mutation] in result["error"]
