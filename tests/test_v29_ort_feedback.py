"""Tiny synthetic interface/guard tests; no checkpoint or captured data use."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ENTRY = Path(__file__).resolve().parent / "run-v29-dpvo-ort-feedback-board.py"
if not ENTRY.exists():
    ENTRY = Path(__file__).resolve().parents[1] / "deploy/rk3588/validation/dpvo_geometry/run-v29-dpvo-ort-feedback-board.py"
SPEC = importlib.util.spec_from_file_location("v29_feedback_test_entry", ENTRY)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def contracts(edges=3):
    shapes = [[1, edges, 384], [1, edges, 384], [1, edges, 882],
              [edges], [1, edges, 1], [edges], [1, edges, 1],
              [edges, 1], [edges, 1], [edges], [1, edges], [1, edges], [edges]]
    integer = {"previous_indices", "following_indices", "kk_indices", "kk_inverse", "ij_indices", "ij_inverse"}
    return dict(inputs=[dict(name=name, shape=shape, dtype=7 if name in integer else 1)
                        for name, shape in zip(driver.INPUT_NAMES, shapes)],
                outputs=[dict(name=name, shape=[1, edges, 384 if index == 0 else 2], dtype=1)
                         for index, name in enumerate(driver.OUTPUT_NAMES)])


def acceptance_fixture():
    ancestor = dict(original_weights_unchanged=True, sources_and_inputs_unchanged=True,
                    source_pins={"synthetic.py": "a" * 64}, source_hashes_after={"synthetic.py": "a" * 64}, cleanup_errors=[])
    captured = dict(copy.deepcopy(ancestor), status="captured_original_Update_states_not_VO", scenarios=[])
    exported = dict(copy.deepcopy(ancestor), status="exported_Update_board_ORT_pending",
                    capture_sha256=driver.UPDATE_CAPTURE_SHA, adapter_sha256=driver.ADAPTER_SHA,
                    exact_wrapper_FP32_gate_passed=True, scenarios=[], models={})
    manifest = dict(schema="v29_DPVO_Update_candidate_manifest", source_export_sha256=driver.UPDATE_EXPORT_SHA,
                    source_capture_sha256=driver.UPDATE_CAPTURE_SHA, models={})
    checks = {name: dict(gate_passed=True, rtol=driver.RTOL, atol=driver.ATOL)
              for name in ("net", "delta", "weight")}
    receipt = dict(copy.deepcopy(ancestor), schema="v29_Update_candidate_board_diagnostic",
                   status="completed_numerical_accepted", execution_status="completed", candidate_accepted=True,
                   mode="ort", manifest_sha256=driver.MANIFEST_SHA, capture_sha256=driver.UPDATE_CAPTURE_SHA,
                   export_sha256=driver.UPDATE_EXPORT_SHA, first_only=False, model_keys=["E864", "E96"],
                   selected_saved_states=16, providers=["CPUExecutionProvider"], ORT_optimization="disabled", calls=[])
    for key, edges in (("E96", 96), ("E864", 864)):
        model = dict(contracts(edges), sha256=key + "parent")
        exported["models"][key] = model
        manifest["models"][key] = dict(copy.deepcopy(model), source_onnx_sha256=model["sha256"],
                                        role="stable_weight_sigmoid", original_initializers_byte_unchanged=True,
                                        other_nodes_byte_unchanged=True)
        for case_index in range(2):
            name = key + "case" + str(case_index)
            states = [dict(step=step) for step in range(4)]
            captured["scenarios"].append(dict(name=name, states=copy.deepcopy(states)))
            exported["scenarios"].append(dict(name=name, model_key=key, states=copy.deepcopy(states)))
            for step in range(4):
                receipt["calls"].append(dict(index=len(receipt["calls"]), model_key=key, case=name, step=step,
                                              all_outputs_finite=True, weight_domain_passed=True,
                                              core_strict_FP32_gate_passed=True,
                                              versus_original=copy.deepcopy(checks), versus_wrapper=copy.deepcopy(checks)))
    return receipt, manifest, captured, exported


def test_complete_sixteen_state_acceptance():
    values = acceptance_fixture()
    plan = driver.validate_acceptance(*values, driver.MANIFEST_SHA)
    assert len(plan) == 16 and sum(row[0] == "E864" for row in plan) == 8


@pytest.mark.parametrize("change", [
    lambda r: r.update(candidate_accepted=False),
    lambda r: r.update(status="execution_failed"),
    lambda r: r.update(mode="sdk-normal"),
    lambda r: r.update(first_only=True),
    lambda r: r.update(selected_saved_states=8),
    lambda r: r.update(providers=["CUDAExecutionProvider"]),
    lambda r: r.update(cleanup_errors=["close failure"]),
    lambda r: r["source_hashes_after"].update({"synthetic.py": "b" * 64}),
    lambda r: r["calls"][15].update(step=2),
    lambda r: r["calls"][8].update(weight_domain_passed=False),
    lambda r: r["calls"][0]["versus_original"]["net"].update(atol=1e-3),
])
def test_acceptance_does_not_allow_partial_or_unvalidated_evidence(change):
    values = acceptance_fixture()
    change(values[0])
    with pytest.raises(ValueError):
        driver.validate_acceptance(*values, driver.MANIFEST_SHA)


def test_other_graph_rewrite_is_rejected():
    values = acceptance_fixture()
    values[1]["models"]["E864"]["role"] = "finite_padding"
    with pytest.raises(ValueError, match="Sigmoid"):
        driver.validate_acceptance(*values, driver.MANIFEST_SHA)


def geometry_fixture():
    pins = {"vector.py": driver.GEOMETRY_SHA}
    return dict(status="completed_geometry_only_benchmark", candidate_accepted=True, strict_FP32_gate_passed=True,
                vectorized_geometry_sha256=driver.GEOMETRY_SHA,
                scalar_geometry_sha256="13eb1e03eb4a5ed1bd1533549f2b9f4994abfaa61e0b55304b99e6fe77d813df",
                sources_and_artifacts_unchanged=True, immutable_arrays_unchanged=True,
                source_pins=pins, source_hashes_after=pins.copy(), cleanup_errors=[],
                FP32_gate=dict(rtol=driver.RTOL, atol=driver.ATOL, masks_and_integer_IDs_exact=True),
                ordinary_BA_calls_attempted=32, ordinary_BA_calls_completed=32,
                instrumented_BA_calls_attempted=8, instrumented_BA_calls_completed=8,
                cases=[dict(strict_FP32_gate_passed=True) for _ in range(4)])


def test_geometry_requires_accepted_unchanged_full_gate():
    value = geometry_fixture()
    driver.validate_geometry_evidence(value)
    value["cases"][3]["strict_FP32_gate_passed"] = False
    with pytest.raises(ValueError):
        driver.validate_geometry_evidence(value)


class FakeSession:
    def __init__(self, model):
        self.model, self.feeds, self.raw_outputs = model, [], []
        self.bad = None

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def io(self, rows):
        return [SimpleNamespace(name=x["name"], shape=x["shape"], type="tensor(int64)" if x["dtype"] == 7 else "tensor(float)")
                for x in rows]

    def get_inputs(self):
        return self.io(self.model["inputs"])

    def get_outputs(self):
        return self.io(self.model["outputs"])

    def run(self, names, feed):
        assert names == list(driver.OUTPUT_NAMES)
        self.feeds.append(feed)
        edges = feed["net"].shape[1]
        values = [feed["net"] + np.float32(1), np.zeros((1, edges, 2), np.float32),
                  np.full((1, edges, 2), .5, np.float32)]
        if self.bad == "nonfinite":
            values[0][0, 0, 0] = np.inf
        if self.bad == "weight":
            values[2][0, 0, 0] = np.float32(-1e-7)
        self.raw_outputs = values
        return values


def fake_updater():
    model = contracts()
    topology = {r["name"]: np.zeros(r["shape"], np.int64 if r["dtype"] == 7 else np.float32)
                for r in model["inputs"][3:]}
    graph = dict(ii=np.zeros(3, np.int64), jj=np.ones(3, np.int64), kk=np.arange(3, dtype=np.int64),
                 ctx=np.ones((1, 3, 384), np.float32))
    session = FakeSession(model)
    update = driver.ORTUpdate(session, model, topology, graph)
    return update, session, graph


def test_ORT_own_net_and_owned_IO_in_four_small_calls():
    update, session, graph = fake_updater()
    net = np.zeros((1, 3, 384), np.float32)
    corr = np.zeros((1, 3, 882), np.float32)
    for iteration in range(4):
        before = net.copy()
        net, delta, weight = update(net, graph["ctx"], corr, graph["ii"], graph["jj"], graph["kk"])
        assert np.all(net == iteration + 1)
        assert np.array_equal(session.feeds[-1]["net"], before)
        assert not np.shares_memory(net, session.raw_outputs[0])
        assert not np.shares_memory(session.feeds[-1]["inp"], graph["ctx"])
    assert update.attempted_calls == len(update.calls) == 4
    update.close()
    assert update.session is None
    with pytest.raises(ValueError, match="closed"):
        update(net, graph["ctx"], corr, graph["ii"], graph["jj"], graph["kk"])


@pytest.mark.parametrize("bad", ["nonfinite", "weight"])
def test_no_fallback_or_clipping_of_bad_ORT_output(bad):
    update, session, graph = fake_updater()
    session.bad = bad
    with pytest.raises(ValueError):
        update(np.zeros((1, 3, 384), np.float32), graph["ctx"], np.zeros((1, 3, 882), np.float32),
               graph["ii"], graph["jj"], graph["kk"])
    assert update.attempted_calls == 1 and update.calls == []
    assert session.raw_outputs[0][0, 0, 0] == np.inf if bad == "nonfinite" else session.raw_outputs[2][0, 0, 0] < 0


def test_changed_graph_or_wrong_session_contract_is_rejected():
    update, session, graph = fake_updater()
    with pytest.raises(ValueError, match="graph IDs"):
        update(np.zeros((1, 3, 384), np.float32), graph["ctx"], np.zeros((1, 3, 882), np.float32),
               graph["ii"], graph["jj"] + 1, graph["kk"])
    session.model["outputs"][0]["shape"] = [1, 3, 2]
    session.get_outputs = lambda: [SimpleNamespace(name="wrong", shape=[1], type="tensor(float)")]
    with pytest.raises(ValueError, match="names/order"):
        driver.check_session(session, contracts())


def test_reference_consumes_current_saved_ORT_net_not_old_golden_state():
    graph = dict(ctx=np.ones((1, 2, 384), np.float32), ii=np.zeros(2, np.int64),
                 jj=np.ones(2, np.int64), kk=np.arange(2, dtype=np.int64))
    arrays = dict(net_before=np.full((1, 2, 384), 7, np.float32),
                  corr=np.full((1, 2, 882), 3, np.float32), net_after=np.full((1, 2, 384), 8, np.float32),
                  delta=np.zeros((1, 2, 2), np.float32), weight=np.full((1, 2, 2), .5, np.float32))
    calls = []

    def golden(net, context, corr, ii, jj, kk):
        assert net is arrays["net_before"] and corr is arrays["corr"] and context is graph["ctx"]
        calls.append(net.copy())
        return net + np.float32(1), arrays["delta"].copy(), arrays["weight"].copy()

    expected, check = driver.reference_round(golden, arrays, graph)
    assert len(calls) == 1 and check["strict_FP32_gate_passed"]
    assert check["inputs"]["net"] == driver.meta(arrays["net_before"])
    arrays["net_after"] += 1
    _, failed = driver.reference_round(golden, arrays, graph)
    assert not failed["strict_FP32_gate_passed"] and not failed["comparisons"]["net_out"]["gate_passed"]


def test_bitwise_does_not_hide_signed_zero():
    actual = np.array([0.], np.float32)
    expected = np.array([-0.], np.float32)
    row = driver.strict_compare(actual, expected)
    assert row["gate_passed"] and row["array_equal"] and not row["bitwise_equal"]


def preflight_fixture():
    # Exact v26 motion schema: there is deliberately no graph['corr'].
    graph = dict(ii=np.zeros(3, np.int64), jj=np.ones(3, np.int64), kk=np.arange(3, dtype=np.int64),
                 coords=np.zeros((1, 3, 2, 3, 3), np.float32),
                 ctx=np.ones((1, 3, 384), np.float32), net0=np.zeros((1, 3, 384), np.float32))
    saved = {key: graph[key].copy() for key in ("ctx", "ii", "jj", "kk")}
    saved.update(net=graph["net0"].copy(), corr=np.full((1, 3, 882), .25, np.float32))
    return graph, saved


def test_preflight_reads_original_Update_correlation_not_motion_graph():
    graph, saved = preflight_fixture()
    assert set(graph) == {"ii", "jj", "kk", "coords", "ctx", "net0"}
    assert driver.preflight_correlation(saved, graph) is saved["corr"]


@pytest.mark.parametrize("field", ["net", "ctx", "ii", "jj", "kk", "corr"])
def test_preflight_original_identity_source_mismatch_rejected(field):
    graph, saved = preflight_fixture()
    saved[field].flat[0] = np.inf if field == "corr" else saved[field].flat[0] + 1
    with pytest.raises(ValueError):
        driver.preflight_correlation(saved, graph)


def test_host_guard_precedes_output_or_dependency_loading(monkeypatch, tmp_path):
    monkeypatch.setattr(driver.platform, "machine", lambda: "x86_64")
    names = ("v26-source-dir", "capture-dir", "motion-dir", "checkpoint", "native-source-dir", "library",
             "build-receipt", "output-dir", "frozen-feedback", "adapter", "candidate-manifest",
             "candidate-evidence-dir", "update-capture-dir", "update-export-dir", "geometry-evidence-dir")
    args = [value for name in names for value in ("--" + name, str(tmp_path / name))]
    args += ["--candidate-manifest-sha256", driver.MANIFEST_SHA,
             "--candidate-evidence-sha256", "b" * 64, "--geometry-evidence-sha256", "c" * 64]
    with pytest.raises(ValueError, match="board only"):
        driver.main(args)
    assert not (tmp_path / "output-dir").exists()
