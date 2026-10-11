"""Generated topology/metadata only; no checkpoint, model, ORT, SDK or board."""
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
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exporter():
    path = HERE / "export-v42-m16-eof-source-board.py"
    if not path.exists():
        path = HERE.parent / "deploy/rk3588/validation/dpvo_online_replay" / path.name
    return load(path, "_test_v42_eof_exporter")


@pytest.fixture(scope="module")
def adapter():
    path = Path(os.environ.get("V42_GROUPED_SOURCE", HERE.parent / "dpvo_grouped_update"))
    return load(path / "source_frame_batches.py", "_test_v42_source_batches")


@pytest.fixture
def graph(adapter):
    targets = {4: range(1, 7), 5: range(2, 8), 6: range(3, 9),
               7: range(5, 10), 8: range(6, 11), 9: range(7, 12),
               10: range(8, 12), 11: range(9, 12)}
    triples = np.asarray([(source, target, source * 16 + patch)
                          for source, rows in targets.items()
                          for target in rows for patch in range(16)], dtype=np.int64)
    triples = triples[np.random.default_rng(42).permutation(640)]
    ii, jj, kk = [np.array(triples[:, i], copy=True) for i in range(3)]
    plan = adapter.build_degree_plan(ii, jj, kk, patches_per_frame=16,
                                    max_sources_per_batch=2, max_target_degree=7)
    return plan, dict(degree_plan=plan.metadata()), (ii, jj, kk)


def test_full_source_selection_does_not_cut_targets_or_patch_groups(exporter, graph):
    plan, failure, (ii, jj, kk) = graph
    batch = exporter.complete_source(plan, failure)
    assert batch.source_slots == (11,) and batch.target_slots == ((9, 10, 11),)
    assert batch.canonical_edge_indices.shape == (1, 3, 16)
    assert not batch.canonical_edge_indices.flags.writeable
    for i, target in enumerate((9, 10, 11)):
        indices = batch.canonical_edge_indices[0, i]
        assert np.array_equal(ii[indices], np.full(16, 11, dtype=np.int64))
        assert np.array_equal(jj[indices], np.full(16, target, dtype=np.int64))
        assert np.array_equal(kk[indices], np.arange(176, 192, dtype=np.int64))
    assert np.array_equal(np.sort(batch.canonical_edge_indices.ravel()), np.flatnonzero(ii == 11))


def test_teacher_gather_uses_same_full_input_indices(exporter, adapter, graph):
    plan, failure, _ = graph
    batch = exporter.complete_source(plan, failure)
    original = np.arange(640 * 3, dtype=np.float32).reshape(1, 640, 3)
    teacher = original * np.float32(-2) - np.float32(.5)
    for values in (original, teacher):
        got = adapter.pack_batch(values, batch)
        expected = values[0, batch.canonical_edge_indices].transpose(0, 3, 1, 2)
        assert np.array_equal(got, expected)
        assert got.flags.c_contiguous and not np.shares_memory(got, values)


@pytest.mark.parametrize("change", ["plan", "source", "targets", "patch", "extra_D3"])
def test_no_shape_padding_or_topology_relabel(exporter, adapter, graph, change):
    plan, failure, _ = graph
    if change == "plan":
        failure["degree_plan"]["padding"] = True
    else:
        original = next(b for b in plan.batches if b.degree == 3)
        source = (10,) if change == "source" else original.source_slots
        targets = ((8, 9, 10),) if change == "targets" else original.target_slots
        patches = 15 if change == "patch" else original.patches
        replacement = adapter.DegreeBatch(3, patches, source, targets, original.canonical_edge_indices)
        batches = tuple(replacement if b.degree == 3 else b for b in plan.batches)
        if change == "extra_D3":
            batches += (replacement,)
        plan = adapter.DegreePlan(plan.edges, plan.patches, batches, plan.topology_sha256, plan.frame_capacity)
        failure["degree_plan"] = plan.metadata()
    with pytest.raises(ValueError):
        exporter.complete_source(plan, failure)


@pytest.fixture
def records():
    failure = dict(index=203, route="full", status="failed_no_partial_fullgraph", edges=640,
                   fullgraph_returned=False, outputs_committed_by_callback=False,
                   error="Unknown B1_D3_M16")
    weights = {str(i): {} for i in range(94)}
    run = dict(schema="v41_cached_undistort_online_RKNN_research_replay", status="failed",
               frames_completed=192, error="EOF B1_D3_M16", active_stage="termination_offline",
               online_metrics={"calls": [deepcopy(failure)]}, cleanup_errors=[], affinity_restored=True,
               sources_and_inputs_unchanged=True, source_pins={"source.py": "a" * 64},
               source_hashes_after={"source.py": "a" * 64}, original_weights_unchanged=True,
               original_weights_before=weights, original_weights_after=deepcopy(weights),
               candidate_accepted=False, historical_SDK_strict_FP32_rejected=True,
               SDK_activity={"Update_NPU_used": True, "encoder_NPU_used": True},
               config={"patches_per_frame": 16, "capacity": 64},
               snapshot_before_termination={"metadata": {"initialized": True}})
    return run, failure


def test_failed_input_provenance_is_not_relabelled_cpu_gold(exporter, records):
    run, failure = records
    exporter.validate_run_record(run, failure)
    assert run["status"] == "failed" and run["historical_SDK_strict_FP32_rejected"]


@pytest.mark.parametrize("change", ["stage", "partial", "weights", "source", "CPU", "reject"])
def test_changed_ancestor_or_partial_submission_rejected(exporter, records, change):
    run, failure = records
    if change == "stage":
        run["active_stage"] = "stream"
    elif change == "partial":
        failure["fullgraph_returned"] = True
        run["online_metrics"]["calls"][-1] = deepcopy(failure)
    elif change == "weights":
        run["original_weights_after"]["0"] = {"changed": True}
    elif change == "source":
        run["source_hashes_after"]["source.py"] = "b" * 64
    elif change == "CPU":
        run["SDK_activity"]["encoder_NPU_used"] = False
    else:
        run["historical_SDK_strict_FP32_rejected"] = False
    with pytest.raises(ValueError):
        exporter.validate_run_record(run, failure)


def test_host_guard_precedes_output_and_parent_import(exporter, monkeypatch, tmp_path):
    monkeypatch.setattr(exporter, "board_guard", lambda: (_ for _ in ()).throw(ValueError("synthetic guard")))
    monkeypatch.setattr(exporter, "load_parent", lambda *_: pytest.fail("parent before guard"))
    args = []
    for name in ("parent-exporter", "run-dir", "update-source-dir", "v26-source-dir", "checkpoint"):
        args += ["--" + name, str(tmp_path / name)]
    output = tmp_path / "new"
    args += ["--output-dir", str(output), "--diagnostic-allow-fp32-rejected"]
    with pytest.raises(ValueError, match="synthetic guard"):
        exporter.main(args)
    assert not output.exists()
