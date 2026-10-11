"""Synthetic topology/contract tests; no neural, ONNX, ORT or board execution."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


@pytest.fixture(scope="module")
def exporter():
    path = Path(os.environ.get("V41_EXPORTER", HERE / "export-v41-m16-missing-source-board.py"))
    if not path.is_file():
        path = HERE.parent / "deploy/rk3588/validation/dpvo_m16_online" / path.name
    return load(path, "_test_v41_missing_exporter")


@pytest.fixture(scope="module")
def adapter():
    directory = Path(os.environ.get("V41_GROUPED_SOURCE", HERE.parent / "dpvo_grouped_update"))
    return load(directory / "source_frame_batches.py", "_test_v41_frozen_source_batches")


@pytest.fixture
def graph(exporter, adapter):
    # The original eight-source initialization band, M16; deterministic shuffled
    # edge order proves the selection uses canonical indices instead of a slice.
    triples = [(source, target, source * 16 + patch)
               for source in range(8) for target in range(8)
               if abs(source - target) <= 3 for patch in range(16)]
    permutation = np.random.default_rng(71).permutation(len(triples))
    ii, jj, kk = np.asarray(triples, dtype=np.int64)[permutation].T.copy()
    plan = adapter.build_degree_plan(ii, jj, kk, patches_per_frame=16)
    case = dict(observation_id=34, original_call_index=34, edges=704,
                degree_plan=plan.metadata(), groups=[])
    for batch in plan.batches:
        case["groups"].append(dict(model_key=f"B{len(batch.source_slots)}_D{batch.degree}_M16",
                                   topology=batch.metadata(),
                                   canonical_edge_indices=exporter.meta(batch.canonical_edge_indices)))
    return plan, case


def test_two_complete_sources_fixed_and_owned(exporter, adapter, graph):
    plan, case = graph
    selected = exporter.complete_sources(adapter, plan, case)
    assert [batch.degree for batch, _, _ in selected] == [4, 6]
    assert [batch.source_slots for batch, _, _ in selected] == [(0,), (2,)]
    for batch, group_index, _ in selected:
        parent = plan.batches[group_index]
        assert batch.target_slots == parent.target_slots[:1]
        assert batch.canonical_edge_indices.shape == (1, batch.degree, 16)
        assert np.array_equal(batch.canonical_edge_indices, parent.canonical_edge_indices[:1])
        assert not batch.canonical_edge_indices.flags.writeable
        assert not np.shares_memory(batch.canonical_edge_indices, parent.canonical_edge_indices)
        assert np.unique(batch.canonical_edge_indices).size == batch.degree * 16


def test_feed_and_samecall_gold_preserve_all_source_members(exporter, adapter, graph):
    plan, case = graph
    original = np.arange(704 * 3, dtype=np.float32).reshape(1, 704, 3)
    gold = -original - np.float32(.5)
    for batch, index, _ in exporter.complete_sources(adapter, plan, case):
        for value in (original, gold):
            actual = adapter.pack_batch(value, batch)
            expected = adapter.pack_batch(value, plan.batches[index])[:1]
            assert np.array_equal(actual, expected)
            assert actual.flags.c_contiguous
            assert not np.shares_memory(actual, value)


@pytest.mark.parametrize("change", ["observation", "call", "edges", "degree_plan", "indices", "targets", "missing"])
def test_altered_frozen_source_identity_rejected(exporter, adapter, graph, change):
    plan, original = graph
    case = deepcopy(original)
    if change == "observation":
        case["observation_id"] = 35
    elif change == "call":
        case["original_call_index"] = 35
    elif change == "edges":
        case["edges"] -= 16
    elif change == "degree_plan":
        case["degree_plan"]["padding"] = True
    else:
        group = next(row for row in case["groups"] if row["model_key"] == "B2_D4_M16")
        if change == "indices":
            group["canonical_edge_indices"]["tensor_sha256"] = "0" * 64
        elif change == "targets":
            group["topology"]["target_slots"][0].pop()
        elif change == "missing":
            case["groups"].remove(group)
    with pytest.raises((ValueError, StopIteration)):
        exporter.complete_sources(adapter, plan, case)


@pytest.mark.parametrize("problem", [None, "nonfinite", "negative", "above_one", "dtype", "shape"])
def test_output_finite_and_weight_domain_separate(exporter, adapter, graph, problem):
    plan, case = graph
    batch = exporter.complete_sources(adapter, plan, case)[0][0]
    values = {name: np.zeros(batch.shape(channels), dtype=np.float32)
              for name, channels in zip(("net", "delta", "weight"), (384, 2, 2))}
    values["weight"].fill(.5)
    if problem == "nonfinite":
        values["net"].flat[0] = np.nan
    elif problem == "negative":
        values["weight"].flat[0] = -np.float32(1e-7)
    elif problem == "above_one":
        values["weight"].flat[0] = np.float32(1.000001)
    elif problem == "dtype":
        values["net"] = values["net"].astype(np.float64)
    elif problem == "shape":
        values["delta"] = values["delta"][:, :, :-1]
    result = exporter.validate_outputs(values, batch)
    assert result["finite"] == (problem not in ("nonfinite", "dtype", "shape"))
    assert result["weight_domain_valid"] == (problem not in ("negative", "above_one"))


def test_host_guard_precedes_output_and_parent_loading(exporter, monkeypatch, tmp_path):
    def refuse():
        raise ValueError("synthetic board guard")
    monkeypatch.setattr(exporter, "board_guard", refuse)
    monkeypatch.setattr(exporter, "load_parent", lambda *_: pytest.fail("Parent import before guard"))
    output = tmp_path / "not-created"
    arguments = []
    for name in ("parent-exporter", "run-dir", "source-export-dir", "update-source-dir", "v26-source-dir", "checkpoint"):
        arguments += ["--" + name, str(tmp_path / name)]
    arguments += ["--output-dir", str(output)]
    with pytest.raises(ValueError, match="synthetic board guard"):
        exporter.main(arguments)
    assert not output.exists()


def test_tampered_parent_rejected_before_import(exporter, tmp_path):
    path = tmp_path / "parent.py"
    path.write_text("raise RuntimeError('must never import')\n")
    with pytest.raises(ValueError, match="SHA differs"):
        exporter.load_parent(path)
