"""Small fake-callback tests; no images, weights, neural model or SDK calls."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v36_capture_toy", HERE / "capture-v36-fast-update-board.py")
capture = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = capture
spec.loader.exec_module(capture)


def state():
    frames = [dict(observation_id=i + 6, frame_id=str(i + 6), timestamp=float(i + 6),
                   captured_monotonic=float(i + 26),
                   context=np.arange(8, dtype=np.float32).reshape(2, 4) + i * 10)
              for i in range(2)]
    ii = np.repeat(np.arange(2, dtype=np.int64), 4)
    jj = np.tile(np.repeat(np.arange(2, dtype=np.int64), 2), 2)
    kk = np.array([0, 1, 0, 1, 2, 3, 2, 3], dtype=np.int64)
    return SimpleNamespace(frames=frames, ii=ii, jj=jj, kk=kk, updates=12,
        poses=np.zeros((2, 7), np.float32), patches=np.ones((1, 4, 3, 3, 3), np.float32),
        intrinsics=np.ones((2, 4), np.float32))


def inputs(s):
    ctx = np.stack([f["context"] for f in s.frames]).reshape(-1, 4)[s.kk][None].copy()
    return (np.zeros((1, len(s.ii), 4), np.float32), ctx,
            np.zeros((1, len(s.ii), 7), np.float32), s.ii.copy(), s.jj.copy(), s.kk.copy())


def gold(*args):
    return dict(net=args[0] + np.float32(1), delta=np.zeros((1, len(args[3]), 2), np.float32),
                weight=np.full((1, len(args[3]), 2), .5, np.float32))


def recorder(tmp_path):
    return capture.CaptureRecorder(tmp_path, [7], patches=2, channels=4, corr_channels=7)


def test_exact_one_original_callback_return_identity_and_real_precall_values(tmp_path):
    s = state()
    actual_inputs = inputs(s)
    r = recorder(tmp_path)
    calls = []
    def original(*v):
        calls.append(v)
        result = gold(*v)
        calls.append(result)
        return result
    result = r.invoke(original, s, False, "online", actual_inputs)
    assert len(calls) == 2 and result is calls[1]
    assert r.original_calls == r.original_successes == 1
    row = r.records[0]
    assert row["actual_original_call_index"] == 0 and row["before_BA"]
    for artifact, expected in [(row["inputs_artifact"], dict(zip(("net", "inp", "corr", "ii", "jj", "kk"), actual_inputs))),
                               (row["original_gold_artifact"], result)]:
        f = tmp_path / artifact["file"]
        assert capture.sha(f) == artifact["sha256"] and f.stat().st_size == artifact["bytes"]
        with np.load(f, allow_pickle=False) as values:
            for k, value in expected.items():
                assert np.array_equal(values[k], value) and capture.meta(values[k]) == artifact["arrays"][k]
    assert row["snapshot_identity"]["recurrent_revision"] == 0
    assert row["selected_capture_overhead_seconds"] >= 0


def test_probe_repeated_initialization_and_EOF_never_become_selected_capture(tmp_path):
    r, s = recorder(tmp_path), state()
    r.invoke(gold, s, True, "online", inputs(s))
    assert not r.records
    r.invoke(gold, s, False, "online", inputs(s))
    r.invoke(gold, s, False, "online", inputs(s))
    r.invoke(gold, s, False, "EOF", inputs(s))
    assert len(r.records) == 1 and r.records[0]["actual_original_call_index"] == 1
    assert r.records[0]["step_within_observation"] == 0
    assert r.original_calls == r.original_successes == 4


@pytest.mark.parametrize("bad", ["context", "ownership", "nonfinite"])
def test_bad_actual_binding_rejected_before_original_callback(tmp_path, bad):
    r, s = recorder(tmp_path), state()
    v = list(inputs(s))
    if bad == "context": v[1][0, 0, 0] += 1
    elif bad == "ownership": v[5][0] = 2
    else: v[0][0, 0, 0] = np.nan
    def unused(*v): raise AssertionError("No original call on invalid capture")
    with pytest.raises(ValueError): r.invoke(unused, s, False, "online", tuple(v))
    assert not r.records and not list(tmp_path.iterdir())


def test_no_hidden_rerun_and_original_input_mutation_fails_capture(tmp_path):
    r, s = recorder(tmp_path), state()
    count = [0]
    def mutate(*v):
        count[0] += 1
        v[0][0, 0, 0] += 1
        return gold(*v)
    with pytest.raises(ValueError, match="mutated"):
        r.invoke(mutate, s, False, "online", inputs(s))
    assert count == [1] and not r.records


def test_decorated_neural_uses_actual_working_fork_not_old_owner_state(tmp_path):
    r = recorder(tmp_path)
    working, old = state(), state()
    old.frames[-1]["observation_id"] = 999
    class Base:
        def __init__(self, config, **kwargs):
            self.config, self.update, self._state = config, kwargs["update"], old
        def _neural(self, s, ii, jj, kk, net, *, probe):
            return self.update(*inputs(s))
        def process_encoded(self, s):
            return self._neural(s, s.ii, s.jj, s.kk, None, probe=False)
        def terminate(self):
            return self._neural(working, working.ii, working.jj, working.kk, None, probe=False)
    owner = r.decorate_stream(Base)(SimpleNamespace(patches_per_frame=2), update=gold)
    owner.process_encoded(working)
    owner.terminate()
    assert r.records[0]["observation_id"] == 7 and len(r.records) == 1
    assert owner._state is old and owner._capture_phase == "outside" and owner._capture_active is None


def parent_receipt(folder, r):
    frames = [dict(frame_id=str(i), timestamp=float(i)) for i in range(8)]
    manifest = folder / "input-manifest.json"
    manifest.write_text(json.dumps(dict(frames=frames)))
    report = dict(schema="v35_DPVO_original_FAST_dense_spatial_RKNN_encoder_diagnostic_replay",
        script_sha256=capture.PARENT_SHA256, status="completed_diagnostic", cleanup_errors=[],
        sources_and_inputs_unchanged=True, original_weights_unchanged=True,
        update_calls=[dict(index=0, edges=8)], input_manifest_sha256=capture.sha(manifest),
        historical_encoder_FP32_gate_passed=False)
    path = folder / "run.json"
    path.write_text(json.dumps(report))
    return path, report


def test_parent_accounting_source_timestamp_and_existing_FP32_rejection_binding(tmp_path):
    r, s = recorder(tmp_path), state()
    r.invoke(gold, s, False, "online", inputs(s))
    path, report = parent_receipt(tmp_path, r)
    value = capture.verify_parent_receipt(tmp_path, r)
    assert value["parent_encoder_FP32_rejection_retained"] and value["original_update_calls"] == 1
    report["update_calls"][0]["edges"] = 7
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="identity"): capture.verify_parent_receipt(tmp_path, r)


def test_board_guard_precedes_source_import_and_output_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(capture.platform, "machine", lambda: "x86_64")
    out = tmp_path / "never_created"
    with pytest.raises(ValueError, match="board-only"):
        capture.main(["--parent-runner", str(tmp_path / "absent.py"), "--parent-runner-sha256", capture.PARENT_SHA256,
                      "--capture-output-dir", str(out), "--", "--output-dir", str(tmp_path / "parent")])
    assert not out.exists()


def test_exact_explicit_parent_output_argument():
    assert capture.parent_output_dir(["--output-dir=/tmp/parent"]) == Path("/tmp/parent")
    with pytest.raises(ValueError): capture.parent_output_dir(["--output-dir", "a", "--output-dir=b"])
