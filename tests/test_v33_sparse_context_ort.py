"""Fake ORT/guard tests only: no images, weights, actual sessions or board calls."""
import importlib.util
import json
import os
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ENTRY = Path(os.environ.get("SPARSE_ORT_DRIVER", Path(__file__).with_name("run-v33-dpvo-sparse-context-ort-board.py")))
if not ENTRY.exists():
    ENTRY = Path(__file__).resolve().parents[1] / "deploy/rk3588/validation/dpvo_sparse_context/run-v33-dpvo-sparse-context-ort-board.py"
SPEC = importlib.util.spec_from_file_location("v33_sparse_ort_driver", ENTRY)
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)

IMAGE = {"name": "normalized_bgr", "shape": [1, 1, 2, 3], "dtype": "float32"}
INDEX = {"name": "patch_indices", "shape": [3], "dtype": "int64"}
DENSE = [{"name": "fmap", "shape": [1, 2, 2, 3], "dtype": "float32"},
         {"name": "imap", "shape": [1, 4, 2, 3], "dtype": "float32"}]
SPARSE = [DENSE[0], {"name": "context", "shape": [1, 3, 4], "dtype": "float32"}]


def attributes(rows):
    return [SimpleNamespace(name=r["name"], shape=r["shape"].copy(),
            type="tensor(float)" if r["dtype"] == "float32" else "tensor(int64)") for r in rows]


class FakeSession:
    def __init__(self, role, events):
        self.role, self.events = role, events
        self.inputs = attributes([IMAGE] if role == "parent" else [IMAGE, INDEX])
        self.outputs = attributes(DENSE if role == "parent" else SPARSE)
        self.values = [np.arange(np.prod(r["shape"]), dtype=np.float32).reshape(r["shape"])
                       for r in (DENSE if role == "parent" else SPARSE)]
        self.feeds = None

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def run(self, names, feeds):
        assert names == [r.name for r in self.outputs]
        self.feeds = {k: v.copy() for k, v in feeds.items()}
        self.events.append(("run", self.role))
        return self.values

    def __del__(self):
        self.events.append(("dropped", self.role))


class FakeORT:
    GraphOptimizationLevel = SimpleNamespace(ORT_DISABLE_ALL="disabled")

    def __init__(self, fail=None):
        self.events, self.refs, self.options, self.fail = [], [], [], fail

    def SessionOptions(self):
        return SimpleNamespace()

    def InferenceSession(self, graph, *, sess_options, providers):
        assert providers == ["CPUExecutionProvider"]
        self.options.append(vars(sess_options).copy())
        self.events.append(("init", graph))
        if graph == self.fail:
            raise RuntimeError("fake session initialization failed")
        session = FakeSession(graph, self.events)
        self.refs.append(weakref.ref(session))
        return session


def pair(ort, lifecycle):
    return D.session_pair(ort, "parent", "sparse", lifecycle, image=IMAGE,
                          dense_outputs=DENSE, index=INDEX, sparse_outputs=SPARSE)


@pytest.mark.parametrize("frame", [0, 1, 2])
def test_predeclared_indices_are_interior_repeated_ordered_and_image_independent(frame):
    xy = D.deterministic_centers(frame)
    assert xy.dtype == np.int64 and xy.shape == (96, 2) and xy.flags.c_contiguous
    assert xy[:, 0].min() >= 1 and xy[:, 0].max() <= 150
    assert xy[:, 1].min() >= 1 and xy[:, 1].max() <= 114
    assert len(np.unique(xy, axis=0)) < 96
    assert xy.tobytes() == D.deterministic_centers(frame).tobytes()
    for corner in [[1, 1], [150, 114], [1, 114], [150, 1]]:
        assert np.any(np.all(xy == corner, axis=1))


@pytest.mark.parametrize("frame", [True, -1, 3, 1.0])
def test_sampling_rule_has_no_extra_unplanned_frames(frame):
    with pytest.raises(ValueError):
        D.deterministic_centers(frame)


def test_reference_gather_is_rowmajor_keeps_duplicates_and_channel_order():
    imap = np.arange(24, dtype=np.float32).reshape(1, 4, 2, 3)
    result = D.gather_context(imap, np.array([4, 1, 4], np.int64))
    assert result.shape == (1, 3, 4)
    assert result.flags.c_contiguous
    assert np.array_equal(result[0, 0], imap[0, :, 1, 1])
    assert np.array_equal(result[0, 1], imap[0, :, 0, 1])
    assert np.array_equal(result[0, 0], result[0, 2])
    imap.fill(-100)
    assert result[0, 0, 0] == 4


@pytest.mark.parametrize("indices", [np.array([0, 1], np.int32), np.array([-1], np.int64),
                                    np.array([6], np.int64), np.array([[1]], np.int64)])
def test_reference_gather_refuses_bad_index_type_shape_or_range(indices):
    with pytest.raises(ValueError):
        D.gather_context(np.ones((1, 4, 2, 3), np.float32), indices)


def test_sessions_disable_all_use4threads_and_drop_references_after_success():
    ort, lifecycle = FakeORT(), {}
    with pair(ort, lifecycle) as sessions:
        assert list(sessions) == ["parent", "sparse"]
        assert ort.options == [{"intra_op_num_threads": 4, "inter_op_num_threads": 1,
                                "graph_optimization_level": "disabled"}] * 2
    assert sessions == {}
    assert all(ref() is None for ref in ort.refs)
    assert lifecycle["ORT_session_references_dropped"] is True
    assert lifecycle["native_explicit_destroy_API_available"] is False


def test_second_session_init_failure_releases_first_and_preserves_error():
    ort, lifecycle = FakeORT(fail="sparse"), {}
    with pytest.raises(RuntimeError, match="initialization failed"):
        with pair(ort, lifecycle):
            pytest.fail("unreachable")
    assert all(ref() is None for ref in ort.refs)
    assert ort.events.count(("dropped", "parent")) == 1
    assert lifecycle["ORT_session_references_dropped"]


def test_run_snapshots_outputs_and_uses_frozen_feed_without_renormalizing():
    session = FakeSession("sparse", [])
    x = np.linspace(-.5, 1.5, 6, dtype=np.float32).reshape(IMAGE["shape"])
    indices = np.array([4, 1, 4], np.int64)
    actual, seconds = D.run_checked(session, {"normalized_bgr": x, "patch_indices": indices},
                                    [IMAGE, INDEX], SPARSE)
    assert seconds >= 0
    assert session.feeds["normalized_bgr"].tobytes() == x.tobytes()
    assert session.feeds["patch_indices"].tobytes() == indices.tobytes()
    original = actual["context"].copy()
    session.values[1].fill(-77)
    assert actual["context"].tobytes() == original.tobytes()


@pytest.mark.parametrize("field", ["provider", "index_dtype", "index_shape", "output_order"])
def test_native_session_contract_rejects_wrong_provider_ports_and_dtypes(field):
    session = FakeSession("sparse", [])
    if field == "provider":
        session.get_providers = lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif field == "index_dtype":
        session.inputs[1].type = "tensor(int32)"
    elif field == "index_shape":
        session.inputs[1].shape = [3, 1]
    elif field == "output_order":
        session.outputs.reverse()
    with pytest.raises(ValueError):
        D.check_session(session, [IMAGE, INDEX], SPARSE)


@pytest.mark.parametrize("alter", ["feed_order", "feed_dtype", "output_shape", "output_dtype", "nonfinite"])
def test_run_fails_closed_without_shape_reinterpretation_or_finite_fallback(alter):
    session = FakeSession("sparse", [])
    feeds = {"normalized_bgr": np.zeros(IMAGE["shape"], np.float32), "patch_indices": np.array([4, 1, 4], np.int64)}
    if alter == "feed_order":
        feeds = dict(reversed(list(feeds.items())))
    elif alter == "feed_dtype":
        feeds["patch_indices"] = feeds["patch_indices"].astype(np.int32)
    elif alter == "output_shape":
        session.values[1] = session.values[1].reshape(1, 4, 3)
    elif alter == "output_dtype":
        session.values[1] = session.values[1].astype(np.float16)
    elif alter == "nonfinite":
        session.values[1][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        D.run_checked(session, feeds, [IMAGE, INDEX], SPARSE)


def test_strict_gate_and_bitwise_target_are_distinct_and_never_relaxed():
    reference = np.asarray([0, 1], np.float32)
    actual = np.asarray([-0.0, 1], np.float32)
    score = D.compare(actual, reference)
    assert score["gate_passed"] and score["array_equal"] and not score["bitwise_equal"]
    actual[1] += np.float32(.001)
    assert not D.compare(actual, reference)["gate_passed"]


def test_source_artifact_guards_detect_tamper_conflict_and_path_traversal(tmp_path):
    p = tmp_path / "input.npy"
    p.write_bytes(b"toy input")
    pins = D.Pins()
    pins.pin(p, D.sha256_file(p))
    with pytest.raises(ValueError):
        pins.artifact(tmp_path, {"file": "../input.npy", "sha256": D.sha256_file(p), "bytes": p.stat().st_size})
    p.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        pins.finish()


def test_historical_rejection_cannot_be_changed_to_parent_acceptance():
    history = {"status": "failed", "sources_and_inputs_unchanged": True,
               "source_pins": {}, "source_hashes_after": {}, "cleanup_errors": []}
    D._historical_state(history, "failed")
    history["status"] = "completed_CPU_ORT_gate"
    with pytest.raises(ValueError):
        D._historical_state(history, "failed")


def arguments(output):
    values = []
    for name in ("capture", "export", "old-ort", "spatial-rewrite", "spatial-ort", "original-sparse", "spatial-sparse"):
        values.extend(["--" + name + "-dir", "unused", "--" + name + "-sha256", "0" * 64])
    return values + ["--editor", "unused", "--editor-sha256", D.EDITOR_SHA,
                     "--official-source-dir", "unused", "--output-dir", str(output)]


def test_host_guard_is_before_models_source_loading_and_output_writes(monkeypatch, tmp_path):
    output = tmp_path / "output"
    monkeypatch.setattr(D.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(D, "load_inputs", lambda *_: pytest.fail("source/session path reached on host"))
    monkeypatch.setattr(D, "import_ort", lambda: pytest.fail("ORT imported on host"))
    with pytest.raises(ValueError, match="aarch64"):
        D.main(arguments(output))
    assert not output.exists()


def test_preflight_failure_retains_original_exception_restores_affinity_and_receipt(monkeypatch, tmp_path):
    output = tmp_path / "output"
    monkeypatch.setattr(D, "board_guard", lambda *_: None)
    monkeypatch.setattr(D, "load_inputs", lambda *_: (_ for _ in ()).throw(ValueError("toy identity mismatch")))
    monkeypatch.setattr(D, "import_ort", lambda: pytest.fail("ORT must not be initialized"))
    restored = []
    monkeypatch.setattr(D.os, "sched_getaffinity", lambda _: {4, 6})
    monkeypatch.setattr(D.os, "sched_setaffinity", lambda _, values: restored.append(values))
    assert D.main(arguments(output)) == 1
    receipt = json.loads((output / "ort.json").read_text())
    assert "toy identity mismatch" in receipt["error"]
    assert receipt["status"] == "failed" and receipt["calls"] == []
    assert receipt["affinity_restored"] and restored == [{4, 6}]
    assert receipt["cleanup_errors"] == []
    assert receipt["candidate_accepted_as_original_model"] is False
