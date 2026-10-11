"""Tiny fake-session tests only; no real image, graph or ORT execution."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ENTRY = Path(os.environ.get("RECIPROCAL_ORT_DRIVER", HERE/"run-v38-spatial-reciprocal-ort-board.py"))
if not ENTRY.exists():
    ENTRY = HERE.parent/"deploy/rk3588/validation/dpvo_encoder_reciprocal/run-v38-spatial-reciprocal-ort-board.py"
SPEC = importlib.util.spec_from_file_location("v38_test_ort_driver", ENTRY)
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)
HELPER = Path(os.environ.get("FROZEN_ORT_HELPER", "/dev/shm/vggt-v33-dpvo-encoder/run-v33-dpvo-sparse-context-ort-board.py"))
if not HELPER.exists():
    HELPER = ENTRY.parent.parent/"dpvo_sparse_context/run-v33-dpvo-sparse-context-ort-board.py"
B = D.load_pinned_module(HELPER, D.ORT_HELPER_SHA, "v38_test_frozen_math")
INPUTS = [dict(name="normalized_bgr", shape=[1, 1, 2, 3], dtype="float32"),
          dict(name="patch_indices", shape=[3], dtype="int64")]
OUTPUTS = [dict(name="fmap", shape=[1, 2, 2, 3], dtype="float32"),
           dict(name="context", shape=[1, 3, 2], dtype="float32")]


class Session:
    def __init__(self, role, events):
        self.role, self.events = role, events
        self.values = [np.arange(np.prod(r["shape"]), dtype=np.float32).reshape(r["shape"]) for r in OUTPUTS]
        self.feeds = None

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def get_inputs(self):
        return [SimpleNamespace(name=r["name"], shape=r["shape"], type="tensor(float)" if r["dtype"] == "float32" else "tensor(int64)") for r in INPUTS]

    def get_outputs(self):
        return [SimpleNamespace(name=r["name"], shape=r["shape"], type="tensor(float)") for r in OUTPUTS]

    def run(self, names, feeds):
        assert names == [r["name"] for r in OUTPUTS]
        self.feeds = {k: v.copy() for k, v in feeds.items()}
        self.events.append(("run", self.role))
        return self.values

    def __del__(self):
        self.events.append(("dropped", self.role))


class FakeORT:
    GraphOptimizationLevel = SimpleNamespace(ORT_DISABLE_ALL="disabled")

    def __init__(self, fail=False):
        self.fail, self.events, self.refs, self.options = fail, [], [], []

    def SessionOptions(self):
        return SimpleNamespace()

    def InferenceSession(self, graph, *, sess_options, providers):
        assert providers == ["CPUExecutionProvider"]
        self.options.append(vars(sess_options).copy())
        self.events.append(("init", graph))
        if graph == "candidate" and self.fail:
            raise RuntimeError("fake candidate init failure")
        session = Session(graph, self.events)
        self.refs.append(weakref.ref(session))
        return session


@pytest.mark.parametrize("machine,system,compatible", [
    ("x86_64", "Linux", b"rockchip,rk3588\0"),
    ("aarch64", "Darwin", b"rockchip,rk3588\0"),
    ("aarch64", "Linux", b"other,arm\0")])
def test_guard_rejects_host_or_other_arm(machine, system, compatible):
    with pytest.raises(ValueError):
        D.board_guard(machine, system, compatible)


def test_guard_accepts_explicit_rk3588s():
    D.board_guard("aarch64", "Linux", b"board,test\0rockchip,rk3588s\0")


@pytest.mark.parametrize("fail", [False, True])
def test_both_sparse_sessions_are_disabled_cpu4_and_cleanup_preserves_error(fail):
    ort, lifecycle = FakeORT(fail), {}
    if fail:
        with pytest.raises(RuntimeError, match="fake candidate init"):
            with D.paired_sessions(ort, B, "parent", "candidate", lifecycle, inputs=INPUTS, outputs=OUTPUTS):
                pytest.fail("unreachable")
    else:
        with D.paired_sessions(ort, B, "parent", "candidate", lifecycle, inputs=INPUTS, outputs=OUTPUTS) as sessions:
            assert list(sessions) == ["parent", "candidate"]
        assert sessions == {}
    assert all(ref() is None for ref in ort.refs)
    assert all(o == dict(intra_op_num_threads=4, inter_op_num_threads=1, graph_optimization_level="disabled") for o in ort.options)
    assert lifecycle["ORT_session_references_dropped"]
    assert lifecycle["native_explicit_destroy_API_available"] is False


def test_pair_uses_one_saved_input_indices_and_saves_owned_arrays(tmp_path):
    events = []
    sessions = {r: Session(r, events) for r in ("parent", "candidate")}
    inputs = {"normalized_bgr": np.linspace(-.5, 1.5, 6, dtype=np.float32).reshape(INPUTS[0]["shape"]),
              "patch_indices": np.array([4, 1, 4], np.int64)}
    gold = {r["name"]: v.copy() for r, v in zip(OUTPUTS, sessions["parent"].values)}
    report = {"calls": []}
    actual = D.run_frame_pair(sessions, B, inputs, gold, gold, tmp_path, 0, report, contracts=(INPUTS, OUTPUTS))
    assert events == [("run", "parent"), ("run", "candidate")]
    assert [r["role"] for r in report["calls"]] == ["parent", "candidate"]
    for role in sessions:
        assert sessions[role].feeds["normalized_bgr"].tobytes() == inputs["normalized_bgr"].tobytes()
        assert sessions[role].feeds["patch_indices"].tobytes() == inputs["patch_indices"].tobytes()
        sessions[role].values[1].fill(-100)
        assert actual[role]["context"].tobytes() == gold["context"].tobytes()
        with np.load(tmp_path/(role+"-frame-0.npz"), allow_pickle=False) as saved:
            assert saved["context"].tobytes() == gold["context"].tobytes()
    assert all(v["gate_passed"] for v in report["calls"][1]["versus_current_parent_ORT"].values())


def test_actual_strict_rejection_is_recorded_without_promoting_or_clipping(tmp_path):
    sessions = {r: Session(r, []) for r in ("parent", "candidate")}
    sessions["candidate"].values[1][0, 0, 0] = np.float32(.1)
    gold = {r["name"]: v.copy() for r, v in zip(OUTPUTS, sessions["parent"].values)}
    inputs = {"normalized_bgr": np.zeros(INPUTS[0]["shape"], np.float32), "patch_indices": np.array([4, 1, 4], np.int64)}
    report = {"calls": []}
    D.run_frame_pair(sessions, B, inputs, gold, gold, tmp_path, 0, report, contracts=(INPUTS, OUTPUTS))
    score = report["calls"][1]["versus_current_parent_ORT"]["context"]
    assert score["gate_passed"] is False and score["rtol"] == 1e-4 and score["atol"] == 1e-5
    with np.load(tmp_path/"candidate-frame-0.npz", allow_pickle=False) as saved:
        assert saved["context"][0, 0, 0] == np.float32(.1)


def test_nonfinite_output_is_fatal_not_zero_filled(tmp_path):
    sessions = {r: Session(r, []) for r in ("parent", "candidate")}
    sessions["candidate"].values[0][0, 0, 0, 0] = np.nan
    gold = {r["name"]: v.copy() for r, v in zip(OUTPUTS, sessions["parent"].values)}
    report = {"calls": []}
    inputs = {"normalized_bgr": np.zeros(INPUTS[0]["shape"], np.float32), "patch_indices": np.array([4, 1, 4], np.int64)}
    with pytest.raises(ValueError, match="nonfinite"):
        D.run_frame_pair(sessions, B, inputs, gold, gold, tmp_path, 0, report, contracts=(INPUTS, OUTPUTS))
    assert len(report["calls"]) == 1 and not (tmp_path/"candidate-frame-0.npz").exists()


def recipe():
    return dict(schema="v38_DPVO_spatial_encoder_small_reciprocal_rewrite",
        status="statically_checked_parent_FP32_rejection_retained_not_runtime_validated", script_sha256="a"*64,
        parent={"sha256": D.PARENT_SHA}, parent_recipe_sha256=D.PARENT_RECIPE_SHA, parent_reference_sha256=D.REFERENCE_SHA,
        historical_parent_FP32_gate_passed=False, historical_parent_rejection_retained=True, candidate_accepted=False,
        source_pins={}, source_hashes_after={}, sources_and_parent_unchanged=True,
        preservation={**{k: True for k in ["all_original_initializers_ordered_protobuf_byte_exact", "all_nonselected_nodes_ordered_protobuf_byte_exact",
        "original_IO_value_info_graph_metadata_protobuf_byte_exact", "model_header_protobuf_byte_exact", "exact_10_selected_Div_replacement",
        "original_2_nonselected_Div_protobuf_byte_exact", "one_added_nonlearned_FP32_scalar_one"]}, "original_initializer_count":65, "added_initializer_count":1})


@pytest.mark.parametrize("bad", ["parent", "accepted", "initializer", "source"])
def test_recipe_guards_parent_rejection_weights_source_and_bound_graph(bad):
    value = recipe()
    D.recipe_guard(value, "a"*64)
    if bad == "parent": value["parent"]["sha256"] = "b"*64
    elif bad == "accepted": value["historical_parent_FP32_gate_passed"] = True
    elif bad == "initializer": value["preservation"]["original_initializer_count"] = 64
    else: value["source_hashes_after"] = {"changed": "c"*64}
    with pytest.raises(ValueError): D.recipe_guard(value, "a"*64)


def test_main_guard_precedes_module_import_and_output_creation(tmp_path, monkeypatch):
    argv = []
    for key in ("capture-dir", "reference-dir", "parent-onnx", "parent-recipe", "rewrite-dir", "rewriter", "frozen-ort-helper", "output-dir"):
        argv.extend(["--"+key, str(tmp_path/key)])
    argv.extend(["--rewrite-sha256", "a"*64, "--rewriter-sha256", "b"*64, "--diagnostic-allow-fp32-rejected"])
    monkeypatch.setattr(D.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(D, "load_pinned_module", lambda *a, **k: pytest.fail("model dependencies reached"))
    with pytest.raises(ValueError, match="host neural"):
        D.main(argv)
    assert not (tmp_path/"output-dir").exists()
