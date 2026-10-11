"""Tiny synthetic full-edge oracle only: no DPVO weights or captured data."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn


HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


adapter = load("v36_batches_toy", HERE / "source_frame_batches.py")
cf = load("v36_cf_frozen_toy", HERE / "source_frame_update_channel_first.py")


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class SoftAgg(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.f, self.g, self.h = nn.Linear(c, c), nn.Linear(c, c), nn.Linear(c, c)
        self.expand = True


class GatedResidual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(c, c), nn.Sigmoid())
        self.res = nn.Sequential(nn.Linear(c, c), nn.ReLU(), nn.Linear(c, c))

    def forward(self, x):
        return x + self.gate(x) * self.res(x)


class ToyUpdate(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(361)
        c = 4
        self.c1 = nn.Sequential(nn.Linear(c, c), nn.ReLU(inplace=True), nn.Linear(c, c))
        self.c2 = nn.Sequential(nn.Linear(c, c), nn.ReLU(inplace=True), nn.Linear(c, c))
        self.norm = nn.LayerNorm(c, eps=1e-3)
        self.agg_kk, self.agg_ij = SoftAgg(c), SoftAgg(c)
        self.gru = nn.Sequential(nn.LayerNorm(c, eps=1e-3), GatedResidual(c),
                                 nn.LayerNorm(c, eps=1e-3), GatedResidual(c))
        self.corr = nn.Sequential(nn.Linear(7, c), nn.ReLU(inplace=True), nn.Linear(c, c),
                                  nn.LayerNorm(c, eps=1e-3), nn.ReLU(inplace=True), nn.Linear(c, c))
        self.d = nn.Sequential(nn.ReLU(), nn.Linear(c, 2), nn.Identity())
        self.w = nn.Sequential(nn.ReLU(), nn.Linear(c, 2), nn.Identity(), nn.Sigmoid())
        self.eval()


def graph(seed=362):
    # Two same-degree sources have different target sets; another degree differs.
    rows = [(s, t, s * 2 + m) for s, targets in [(0, [0, 2, 4]), (2, [1, 3, 4]), (3, [0])]
            for t in targets for m in range(2)]
    perm = np.random.default_rng(seed).permutation(len(rows))
    return tuple(np.ascontiguousarray(np.array(rows, dtype=np.int64)[perm, k]) for k in range(3))


def values(edges):
    rng = np.random.default_rng(363)
    return tuple(rng.normal(size=(1, edges, c)).astype(np.float32) for c in (4, 4, 7))


def original_full_oracle(original, net, inp, corr, ii, jj, kk):
    """Independent raw-edge neighbor/group loops; never uses degree plans."""
    out = original.norm(net + inp + original.corr(corr))
    previous = np.full(kk.size, -1, np.int64)
    following = np.full(kk.size, -1, np.int64)
    for patch in np.unique(kk):
        edges = np.flatnonzero(kk == patch)
        ordered = edges[np.argsort(jj[edges])]
        previous[ordered[1:]] = ordered[:-1]
        following[ordered[:-1]] = ordered[1:]
    # Zero the inputs before both Linear biases, not their outputs.
    x = out[:, np.maximum(previous, 0)] * torch.from_numpy((previous >= 0).astype(np.float32))[None, :, None]
    out = out + original.c1(x)
    x = out[:, np.maximum(following, 0)] * torch.from_numpy((following >= 0).astype(np.float32))[None, :, None]
    out = out + original.c2(x)
    for mod, labels in [(original.agg_kk, kk), (original.agg_ij, ii * 12345 + jj)]:
        contribution = torch.empty_like(out)
        for label in np.unique(labels):
            edges = np.flatnonzero(labels == label)
            x = out[:, edges]
            grouped = (mod.f(x) * torch.softmax(mod.g(x), dim=1)).sum(dim=1, keepdim=True)
            contribution[:, edges] = mod.h(grouped)
        out = out + contribution
    out = original.gru(out)
    return dict(net=out, delta=original.d(out), weight=original.w(out))


IDENTITY = adapter.SnapshotIdentity(1, 2, 3, 4)


@pytest.mark.parametrize("max_sources", [None, 1, 2])
@pytest.mark.parametrize("edge_seed", [362, 367])
def test_fullgroup_batched_cf_against_independent_rawedge_oracle(max_sources, edge_seed):
    ii, jj, kk = graph(edge_seed)
    plan = adapter.build_degree_plan(ii, jj, kk, patches_per_frame=2,
                                    frame_capacity=8, max_sources_per_batch=max_sources)
    original = ToyUpdate()
    wrapper = cf.SourceFrameUpdateChannelFirst(original, terminal_sigmoid="stable")
    v = values(ii.size)
    before = tuple(x.copy() for x in v)
    calls = []
    def execute(batch, feeds):
        calls.append(batch.metadata())
        tensors = [torch.from_numpy(feeds[k]) for k in adapter.INPUT_NAMES]
        wrapper.validate_inputs(*tensors)
        with torch.no_grad():
            output = wrapper(*tensors)
        return {k: x.numpy().copy() for k, x in zip(adapter.OUTPUT_NAMES, output)}
    actual = adapter.execute_grouped_update(plan, *v, execute,
        snapshot_identity=IDENTITY, current_identity=lambda: IDENTITY)
    with torch.no_grad():
        reference = original_full_oracle(original, *(torch.from_numpy(x) for x in v), ii, jj, kk)
    for name in adapter.OUTPUT_NAMES:
        np.testing.assert_allclose(actual[name], reference[name].numpy(), rtol=1e-4, atol=1e-5)
    assert all(np.array_equal(a, b) for a, b in zip(v, before))
    assert len(calls) == (3 if max_sources == 1 else 2)
    assert all(not x["padding"] and not x["dependency_groups_split"] for x in calls)
    assert list(wrapper.state_dict()) == list(original.state_dict())
    assert len(wrapper.state_dict()) == 50


def test_batch_roundtrip_and_true_source_scatter_for_adversarial_edge_order():
    ii, jj, kk = graph()
    plan = adapter.build_degree_plan(ii, jj, kk, patches_per_frame=2, frame_capacity=8)
    net = np.arange(ii.size * 4, dtype=np.float32).reshape(1, ii.size, 4)
    def identity(batch, feeds):
        return dict(net=feeds["net"], delta=feeds["net"][:, :2],
                    weight=np.full(batch.shape(2), .5, np.float32))
    actual = adapter.execute_grouped_update(plan, net, net, np.zeros((1, ii.size, 7), np.float32),
        identity, snapshot_identity=IDENTITY, current_identity=lambda: IDENTITY)
    assert np.array_equal(actual["net"], net)
    assert np.array_equal(actual["delta"], net[:, :, :2])
    for batch in plan.batches:
        feed = adapter.pack_batch(net, batch)
        for q, s in enumerate(batch.source_slots):
            e = batch.canonical_edge_indices[q]
            assert np.all(ii[e] == s)
            assert np.array_equal(jj[e], np.broadcast_to(batch.target_slots[q], e.T.shape).T)
            assert np.array_equal(kk[e] % 2, np.broadcast_to(np.arange(2), e.shape))
            assert np.array_equal(feed[q].transpose(1, 2, 0), net[0, e])


def test_degree_padding_and_cross_source_softmax_are_wrong_counterexamples():
    # One actual member gives one. A zero padded member changes normalization.
    real = torch.tensor([[[1.]]])
    padded = torch.tensor([[[1.], [0.]]])
    actual = (real * torch.softmax(torch.zeros_like(real), dim=1)).sum(dim=1)
    wrong = (padded * torch.softmax(torch.zeros_like(padded), dim=1)).sum(dim=1)
    assert actual.item() == 1 and wrong.item() == .5
    # Two independent source batches must not share a reduction axis.
    x = torch.tensor([[[1.]], [[7.]]])
    isolated = (x * torch.softmax(torch.zeros_like(x), dim=1)).sum(dim=1)
    mixed = (x * torch.softmax(torch.zeros_like(x), dim=0)).sum(dim=0)
    assert isolated[:, 0].tolist() == [1, 7] and mixed.item() == 4


@pytest.mark.parametrize("failure", ["last_domain", "last_nonfinite", "revision"])
def test_allgroups_before_atomic_full_restore_no_partial_input_commit(failure):
    ii, jj, kk = graph()
    plan = adapter.build_degree_plan(ii, jj, kk, patches_per_frame=2, frame_capacity=8)
    v = values(ii.size)
    raw_before = [x.tobytes() for x in v]
    current = [IDENTITY]
    count = [0]
    def execute(batch, feeds):
        count[0] += 1
        out = dict(net=feeds["net"], delta=np.zeros(batch.shape(2), np.float32),
                   weight=np.full(batch.shape(2), .5, np.float32))
        if count[0] == len(plan.batches):
            if failure == "last_domain": out["weight"][0, 0, 0, 0] = 1.0001
            elif failure == "last_nonfinite": out["net"][0, 0, 0, 0] = np.nan
            else: current[0] = adapter.SnapshotIdentity(1, 2, 3, 5)
        return out
    with pytest.raises((ValueError, RuntimeError)):
        adapter.execute_grouped_update(plan, *v, execute,
            snapshot_identity=IDENTITY, current_identity=lambda: current[0])
    assert [x.tobytes() for x in v] == raw_before


@pytest.mark.parametrize("bad", ["duplicate", "missing", "owner"])
def test_reject_nonclosed_real_groups_without_padding_or_edge_deletion(bad):
    ii, jj, kk = graph()
    if bad == "duplicate": ii, jj, kk = tuple(np.concatenate([x, x[:1]]) for x in (ii, jj, kk))
    elif bad == "missing": ii, jj, kk = tuple(x[:-1] for x in (ii, jj, kk))
    else: kk = kk.copy(); kk[0] += 2
    with pytest.raises(ValueError):
        adapter.build_degree_plan(ii, jj, kk, patches_per_frame=2, frame_capacity=8)


def test_explicit_payload_budget_and_input_shape_dtype_fail_before_callback():
    ii, jj, kk = graph()
    plan = adapter.build_degree_plan(ii, jj, kk, patches_per_frame=2, frame_capacity=8)
    v = values(ii.size)
    def unused(*args): raise AssertionError("Must fail before callback")
    with pytest.raises(ValueError, match="budget"):
        adapter.execute_grouped_update(plan, *v, unused,
            snapshot_identity=IDENTITY, current_identity=lambda: IDENTITY, max_payload_bytes=1)
    v = (v[0].astype(np.float16), v[1], v[2])
    with pytest.raises(ValueError, match="FP32"):
        adapter.execute_grouped_update(plan, *v, unused,
            snapshot_identity=IDENTITY, current_identity=lambda: IDENTITY)


def test_parent_pins_and_static_wholegraph_MAC_not_saved_singleblock_time():
    import hashlib
    assert hashlib.sha256((HERE / "source_frame_update_channel_first.py").read_bytes()).hexdigest() == adapter.CF_SOURCE_SHA256
    rows = [(s, t, s * 48 + m) for s in range(8) for t in range(8) for m in range(48)]
    ii, jj, kk = (np.array(rows, np.int64)[:, i] for i in range(3))
    plan = adapter.build_degree_plan(ii, jj, kk)
    budget = adapter.static_budget(plan)
    assert len(plan.batches) == 1 and plan.batches[0].shape(384) == (8, 384, 8, 48)
    assert plan.edges == 3072
    assert budget["Linear_MAC"] == 2699520 * 3072 + 147456 * (384 + 64)
    assert not budget["MAC_reduced"] and not budget["compilation_for_N_greater_than_one_verified"]
