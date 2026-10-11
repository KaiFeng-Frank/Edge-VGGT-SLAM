"""Independent tiny fake-ORT runtime checks; no model/real-session forward.

Use DPVO_DYNAMIC_UPDATE_DIR for an external helper directory when this test is
copied into repository tests/. The default adjacent development layout and
public dpvo_stream layout are supported. Synthetic ONNX bytes are never parsed.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest


def _load(name, filename):
    roots = [Path(os.environ["DPVO_DYNAMIC_UPDATE_DIR"])] if os.environ.get("DPVO_DYNAMIC_UPDATE_DIR") else []
    roots += [Path(__file__).parent,
              Path(__file__).parents[1] / "deploy/rk3588/validation/dpvo_dynamic_update"]
    path = next((root / filename for root in roots if (root / filename).is_file()), None)
    if path is None:
        raise FileNotFoundError("Set DPVO_DYNAMIC_UPDATE_DIR to the verified v31 helper directory")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runtime = _load("v31_dynamic_independent_runtime", "v31_dynamic_update.py")
topology = _load("v31_dynamic_independent_topology", "v31_update_topology.py")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nodes(names):
    return [types.SimpleNamespace(name=name, shape=copy.deepcopy(runtime.SHAPES[name]),
            type="tensor(int64)" if name in runtime.INT_NAMES else "tensor(float)")
            for name in names]


class FakeSession:
    def __init__(self):
        self.inputs = _nodes(runtime.INPUT_NAMES)
        self.outputs = _nodes(runtime.OUTPUT_NAMES)
        self.providers = ["CPUExecutionProvider"]
        self.calls = []
        self.returned = None
        self.output_transform = None
        self.error = None

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def get_providers(self):
        return self.providers

    def run(self, names, feeds):
        assert names == list(runtime.OUTPUT_NAMES)
        assert tuple(feeds) == runtime.INPUT_NAMES
        self.calls.append(feeds)
        if self.error is not None:
            raise self.error
        E = feeds["net"].shape[1]
        values = [feeds["net"] + np.float32(1),
                  np.full((1, E, 2), np.float32(.125)),
                  np.full((1, E, 2), np.float32(.5))]
        if self.output_transform is not None:
            values = self.output_transform(values)
        self.returned = values
        return values


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Pin tiny fake artifacts; replace the entire ORT factory before any load."""
    model = tmp_path / "fake.onnx"
    model.write_bytes(b"synthetic-not-ONNX-not-a-checkpoint")
    row = lambda name: {"name": name, "shape": copy.deepcopy(runtime.SHAPES[name]),
                        "dtype": 7 if name in runtime.INT_NAMES else 1}
    weights = {"synthetic_param_%d" % i: {"sha256": "1" * 64} for i in range(94)}
    subset = {k: weights[k] for k in list(weights)[:50]}
    manifest = dict(schema="v31_DPVO_dynamic_Update_export",
        status="exported_dynamic_ONNX_numerical_validation_pending",
        original_weights_unchanged=True, sources_and_inputs_unchanged=True,
        cleanup_errors=[], source_pins={"fake.py": "2" * 64},
        source_hashes_after={"fake.py": "2" * 64},
        original_weights_before=copy.deepcopy(weights), original_weights_after=copy.deepcopy(weights),
        original_Update_weights_before=copy.deepcopy(subset), original_Update_weights_after=copy.deepcopy(subset),
        rewrite={k: True for k in ("terminal_weight_only", "original_initializers_byte_unchanged",
                                  "other_nodes_byte_unchanged", "IO_byte_unchanged")},
        models={"raw": {"sha256": "3" * 64},
                "final": {"file": model.name, "sha256": _sha(model), "bytes": model.stat().st_size,
                    "role": "dynamic_Update_terminal_stable_sigmoid", "source_onnx_sha256": "3" * 64,
                    "inputs": [row(n) for n in runtime.INPUT_NAMES],
                    "outputs": [row(n) for n in runtime.OUTPUT_NAMES]}})
    manifest_path = tmp_path / "export.json"
    fake = FakeSession()
    caches, factories = [], []

    def cache_factory(**kwargs):
        cache = topology.TopologyCache(**kwargs)
        caches.append(cache)
        return cache

    topo_module = types.SimpleNamespace(INPUT_NAMES=topology.INPUT_NAMES, TopologyCache=cache_factory)

    def session_factory(path, sess_options, providers):
        factories.append(dict(path=path, options=sess_options, providers=providers))
        assert providers == ["CPUExecutionProvider"]
        assert sess_options.intra_op_num_threads == 4
        assert sess_options.inter_op_num_threads == 1
        assert sess_options.graph_optimization_level == "FAKE_DISABLED"
        return fake

    fake_ort = types.SimpleNamespace(__version__="synthetic-no-ORT",
        SessionOptions=lambda: types.SimpleNamespace(),
        GraphOptimizationLevel=types.SimpleNamespace(ORT_DISABLE_ALL="FAKE_DISABLED"),
        InferenceSession=session_factory)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(runtime, "board_guard", lambda: None)

    def create(*, max_edges=98304):
        manifest_path.write_text(json.dumps(manifest))
        return runtime.DynamicORTUpdate(manifest_path, _sha(manifest_path), topo_module, max_edges=max_edges)

    return types.SimpleNamespace(fake=fake, manifest=manifest, manifest_path=manifest_path,
        model=model, caches=caches, factories=factories, create=create, topo_module=topo_module)


def _input(E=3):
    ii = np.arange(E, dtype=np.int64) // 2
    jj = np.arange(E, dtype=np.int64) % 2
    kk = np.arange(E, dtype=np.int64)
    return [np.zeros((1, E, 384), np.float32), np.full((1, E, 384), .1, np.float32),
            np.full((1, E, 882), .2, np.float32), ii, jj, kk]


def test_real_topology_tuple_API_and_changing_E(env):
    handle = env.create()
    try:
        for E in (3, 5, 3):
            result = handle(*_input(E))
            assert result["net"].shape == (1, E, 384)
            assert result["delta"].shape == result["weight"].shape == (1, E, 2)
        metrics = handle.metrics()
        assert (metrics["attempted_calls"], metrics["completed_calls"], metrics["failed_calls"]) == (3, 3, 0)
        assert metrics["topology_cache"]["hits"] == 1
        assert not metrics["NPU_used"] and not metrics["numerical_acceptance_implied"]
        assert metrics["ORT_version"] == "synthetic-no-ORT"
        for row in metrics["calls"]:
            components = [row[k] for k in ("input_validation_seconds", "topology_seconds",
                          "feed_copy_seconds", "ORT_seconds", "output_copy_check_seconds")]
            assert all(t >= 0 for t in components)
            assert sum(components) == pytest.approx(row["total_seconds"], abs=1e-10)
    finally:
        handle.close()
    assert handle.metrics()["topology_cache"]["current_payload_bytes"] == 0


@pytest.mark.parametrize("case", ["axes", "dtype", "names", "94_weights", "50_weights",
                                      "rewrite", "ancestry", "source_after", "unsafe_basename"])
def test_export_contract_rejects_before_session(env, case):
    m = env.manifest
    if case == "axes": m["models"]["final"]["inputs"][1]["shape"] = [1, "DifferentE", 384]
    elif case == "dtype": m["models"]["final"]["inputs"][4]["dtype"] = 7
    elif case == "names": m["models"]["final"]["outputs"].reverse()
    elif case == "94_weights": m["original_weights_after"]["synthetic_param_0"]["sha256"] = "4" * 64
    elif case == "50_weights": m["original_Update_weights_before"].pop("synthetic_param_0")
    elif case == "rewrite": m["rewrite"]["IO_byte_unchanged"] = False
    elif case == "ancestry": m["models"]["final"]["source_onnx_sha256"] = "5" * 64
    elif case == "source_after": m["source_hashes_after"]["fake.py"] = "5" * 64
    elif case == "unsafe_basename": m["models"]["final"]["file"] = "../fake.onnx"
    with pytest.raises(ValueError): env.create()
    assert env.factories == []


@pytest.mark.parametrize("case", ["provider", "input_axis", "input_type", "output_shape", "output_order"])
def test_session_contract_mismatch_closes_cache(env, case):
    if case == "provider": env.fake.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif case == "input_axis": env.fake.inputs[2].shape = [1, "OtherE", 882]
    elif case == "input_type": env.fake.inputs[3].type = "tensor(int32)"
    elif case == "output_shape": env.fake.outputs[1].shape = [1, "E", 3]
    elif case == "output_order": env.fake.outputs.reverse()
    with pytest.raises(ValueError): env.create()
    assert len(env.factories) == 1 and env.fake.calls == []
    assert env.caches[-1].metrics()["closed"]


@pytest.mark.parametrize("case", ["neural_dtype", "graph_dtype", "corr_nan", "inp_inf", "wrong_E", "empty", "over_capacity"])
def test_bad_request_never_calls_session(env, case):
    handle = env.create(max_edges=4)
    args = _input()
    if case == "neural_dtype": args[0] = args[0].astype(np.float64)
    elif case == "graph_dtype": args[3] = args[3].astype(np.int32)
    elif case == "corr_nan": args[2][0, 0, 0] = np.nan
    elif case == "inp_inf": args[1][0, 0, 0] = np.inf
    elif case == "wrong_E": args[1] = np.zeros((1, 2, 384), np.float32)
    elif case == "empty": args = _input(0)
    elif case == "over_capacity": args = _input(5)
    try:
        with pytest.raises(ValueError): handle(*args)
        assert env.fake.calls == []
        m = handle.metrics()
        assert (m["attempted_calls"], m["completed_calls"], m["failed_calls"]) == (1, 0, 1)
    finally:
        handle.close()


@pytest.mark.parametrize("case", ["missing", "extra", "dtype", "shape", "net_nan",
                                      "delta_inf", "weight_nan", "weight_negative", "weight_above_one"])
def test_malformed_output_is_rejected_without_fallback(env, case):
    def transform(values):
        if case == "missing": return values[:2]
        if case == "extra": return values + [np.zeros((1,), np.float32)]
        if case == "dtype": values[1] = values[1].astype(np.float64)
        elif case == "shape": values[0] = values[0][:, :, :383]
        elif case == "net_nan": values[0][0, 0, 0] = np.nan
        elif case == "delta_inf": values[1][0, 0, 0] = np.inf
        elif case == "weight_nan": values[2][0, 0, 0] = np.nan
        elif case == "weight_negative": values[2][0, 0, 0] = np.float32(-1e-7)
        elif case == "weight_above_one": values[2][0, 0, 0] = np.float32(1.000001)
        return values
    env.fake.output_transform = transform
    handle = env.create()
    try:
        with pytest.raises(ValueError): handle(*_input())
        assert len(env.fake.calls) == 1
        assert handle.metrics()["completed_calls"] == 0
        assert handle.metrics()["failed_calls"] == 1
    finally:
        handle.close()


def test_input_output_ownership_and_readonly_noncontiguous_views(env):
    handle = env.create()
    args = _input()
    for i in range(3):
        backing = np.repeat(args[i], 2, axis=-1)
        args[i] = backing[..., ::2]
        args[i].flags.writeable = False
    try:
        result = handle(*args)
        feed = env.fake.calls[-1]
        for k, src in zip(("net", "inp", "corr"), args[:3]):
            assert not np.shares_memory(feed[k], src)
            assert feed[k].flags.c_contiguous and feed[k].flags.owndata
        saved = {k: value.copy() for k, value in result.items()}
        for raw in env.fake.returned: raw.fill(np.float32(7))
        for k in result:
            assert np.array_equal(result[k], saved[k])
            assert result[k].flags.c_contiguous and result[k].flags.owndata
    finally:
        handle.close()


def test_metrics_are_deep_copied_and_close_idempotent(env):
    handle = env.create()
    handle(*_input())
    m = handle.metrics()
    m["calls"][0]["group_shapes"]["kk"][0] = 999
    m["calls"][0]["weight_min"] = -9
    m["topology_cache"]["last_call"]["status"] = "changed"
    again = handle.metrics()
    assert again["calls"][0]["group_shapes"]["kk"][0] != 999
    assert again["calls"][0]["weight_min"] == .5
    assert again["topology_cache"]["last_call"]["status"] == "completed"
    handle.close(); handle.close()
    assert handle.closed and handle.session is None
    assert handle.metrics()["topology_cache"]["closed"]
    with pytest.raises(ValueError): handle(*_input())
    assert handle.metrics()["attempted_calls"] == 1


@pytest.mark.parametrize("name", ["manifest", "model"])
def test_artifact_mutation_is_detected_and_close_still_releases(env, name):
    handle = env.create()
    handle(*_input())
    path = env.manifest_path if name == "manifest" else env.model
    with path.open("ab") as f: f.write(b"changed")
    try:
        with pytest.raises(ValueError, match="artifact changed"): handle.check_sources()
    finally:
        handle.close()
    assert handle.metrics()["topology_cache"]["current_payload_bytes"] == 0
    assert handle.metrics()["completed_calls"] == 1


def test_session_failure_counts_and_releases(env):
    env.fake.error = RuntimeError("synthetic run failure")
    handle = env.create()
    try:
        with pytest.raises(RuntimeError, match="synthetic run failure"): handle(*_input())
        assert len(env.fake.calls) == 1 and handle.metrics()["failed_calls"] == 1
        assert handle.metrics()["completed_calls"] == 0
    finally:
        handle.close()
    assert handle.metrics()["topology_cache"]["current_payload_bytes"] == 0


def test_manifest_and_model_hashes_gate_before_session(env):
    env.manifest_path.write_text(json.dumps(env.manifest))
    with pytest.raises(ValueError, match="Manifest SHA"):
        runtime.DynamicORTUpdate(env.manifest_path, "0" * 64, env.topo_module)
    env.model.write_bytes(b"mutated before construction")
    with pytest.raises(ValueError, match="artifact differs"): env.create()
    assert not env.factories


def test_board_guard_executes_before_loading(env, monkeypatch):
    def refuse(): raise ValueError("synthetic off-board guard")
    monkeypatch.setattr(runtime, "board_guard", refuse)
    with pytest.raises(ValueError, match="off-board guard"): env.create()
    assert not env.factories and not env.caches
