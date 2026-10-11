"""Tiny integer/layout tests only; no real features or neural execution."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / "deploy/rk3588/validation/dpvo_dynamic_update/source_frame_plan.py"
spec = importlib.util.spec_from_file_location("v32_source_frame_plan", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def graph():
    rows = [(i, j, i * 3 + p) for i, targets in ((0, (0, 2)), (2, (0, 1, 2)))
            for j in targets for p in range(3)]
    order = np.random.default_rng(14).permutation(len(rows))
    return tuple(np.asarray(rows, dtype=np.int64)[order, n].copy() for n in range(3))


def test_complete_sources_and_inverse_restore_with_arbitrary_global_order():
    ii, jj, kk = graph()
    plans = module.build_source_frame_plan(ii, jj, kk, patches_per_frame=3, frame_capacity=4)
    x = np.arange(ii.size * 2, dtype=np.float32).reshape(1, ii.size, 2)
    restored = np.empty_like(x)
    assert len(plans) == 2
    for plan in plans:
        y = module.to_canonical(x, plan)
        assert y.shape[1:3] == plan.tensor_shape_without_batch_channels
        restored[:, plan.original_edge_indices] = module.to_source_original_order(y, plan)
        assert np.all(ii[plan.canonical_edge_indices] == plan.source_slot)
        assert not np.shares_memory(y, x)
    np.testing.assert_array_equal(restored, x)


def test_both_groups_and_neighbor_links_are_closed_within_each_source():
    ii, jj, kk = graph()
    for plan in module.build_source_frame_plan(ii, jj, kk, patches_per_frame=3, frame_capacity=4):
        ids = plan.canonical_edge_indices
        targets = jj[ids].reshape(-1, 3)
        patches = kk[ids].reshape(-1, 3)
        assert np.all(patches == patches[0])
        assert np.all(targets == targets[:, :1])
        assert np.all(np.diff(targets[:, 0]) > 0)
        assert all(set(np.flatnonzero(kk == k)).issubset(set(ids)) for k in np.unique(kk[ids]))
        labels = ii * 12345 + jj
        assert all(set(np.flatnonzero(labels == k)).issubset(set(ids)) for k in np.unique(labels[ids]))


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "wrong_source", "negative", "int32", "excess_degree"))
def test_incomplete_duplicate_invalid_or_capacity_overflow_fails(mutation):
    ii, jj, kk = graph()
    kwargs = dict(patches_per_frame=3, frame_capacity=4)
    if mutation == "missing":
        ii, jj, kk = ii[:-1], jj[:-1], kk[:-1]
    elif mutation == "duplicate":
        ids = np.flatnonzero(ii == 0)
        jj[ids[1]], kk[ids[1]] = jj[ids[0]], kk[ids[0]]
    elif mutation == "wrong_source": kk[0] = 3
    elif mutation == "negative": jj[0] = -1
    elif mutation == "int32": kk = kk.astype(np.int32)
    else: kwargs["max_target_degree"] = 2
    with pytest.raises(ValueError):
        module.build_source_frame_plan(ii, jj, kk, **kwargs)


def test_label_collision_domain_is_rejected():
    ii, jj, kk = graph()
    with pytest.raises(ValueError, match="collisions"):
        module.build_source_frame_plan(ii, jj, kk, patches_per_frame=3, frame_capacity=12346)


def test_plans_are_owned_values_and_source_mutation_does_not_change_them():
    ii, jj, kk = graph()
    plans = module.build_source_frame_plan(ii, jj, kk, patches_per_frame=3, frame_capacity=4)
    before = [p.canonical_edge_indices.copy() for p in plans]
    ii[:] = jj[:] = kk[:] = 0
    for plan, old in zip(plans, before):
        np.testing.assert_array_equal(plan.canonical_edge_indices, old)
        assert plan.canonical_edge_indices.flags.writeable is False


def test_target_pair_chunk_is_not_a_closed_kk_group():
    ii, jj, kk = graph()
    single_pair = set(np.flatnonzero((ii == 0) & (jj == 0)))
    k = int(kk[next(iter(single_pair))])
    assert not set(np.flatnonzero(kk == k)).issubset(single_pair)
