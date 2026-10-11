"""Tiny random toy modules only; no official checkpoint, image or inference."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn


_spec = importlib.util.spec_from_file_location(
    "source_frame_update", (Path(__file__).resolve().parents[1] / "deploy" / "rk3588" / "validation" / "dpvo_source_frame") / "source_frame_update.py")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
SourceFrameUpdate = _module.SourceFrameUpdate


@pytest.fixture(scope="module", autouse=True)
def one_thread_tiny_tensors():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class ToyGatedResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(channels, channels), nn.Sigmoid())
        self.res = nn.Sequential(nn.Linear(channels, channels),
                                 nn.ReLU(inplace=True), nn.Linear(channels, channels))

    def forward(self, x):
        return x + self.gate(x) * self.res(x)


class ToySoftAgg(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.expand = True
        self.f = nn.Linear(channels, channels)
        self.g = nn.Linear(channels, channels)
        self.h = nn.Linear(channels, channels)


class ToyUpdate(nn.Module):
    """Match the official child registration order and its fifty state keys."""
    def __init__(self, channels=4, correlation_channels=7, seed=18):
        super().__init__()
        torch.manual_seed(seed)
        self.c1 = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(inplace=True),
                                nn.Linear(channels, channels))
        self.c2 = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(inplace=True),
                                nn.Linear(channels, channels))
        self.norm = nn.LayerNorm(channels, eps=1e-3)
        self.agg_kk = ToySoftAgg(channels)
        self.agg_ij = ToySoftAgg(channels)
        self.gru = nn.Sequential(nn.LayerNorm(channels, eps=1e-3),
                                 ToyGatedResidual(channels),
                                 nn.LayerNorm(channels, eps=1e-3),
                                 ToyGatedResidual(channels))
        self.corr = nn.Sequential(nn.Linear(correlation_channels, channels),
                                 nn.ReLU(inplace=True), nn.Linear(channels, channels),
                                 nn.LayerNorm(channels, eps=1e-3),
                                 nn.ReLU(inplace=True), nn.Linear(channels, channels))
        # GradientClip's inference forward is identity; no backward is tested.
        self.d = nn.Sequential(nn.ReLU(inplace=False), nn.Linear(channels, 2), nn.Identity())
        self.w = nn.Sequential(nn.ReLU(inplace=False), nn.Linear(channels, 2),
                               nn.Identity(), nn.Sigmoid())
        self.eval()


def loop_aggregate(original, net, labels):
    """Independent row-group oracle, retaining each group's original edge order."""
    out = torch.empty_like(net)
    for group in np.unique(labels):
        indices = np.flatnonzero(labels == group)
        rows = net[:, indices]
        weights = torch.softmax(original.g(rows), dim=1)
        reduction = torch.sum(original.f(rows) * weights, dim=1)
        expanded = original.h(reduction).unsqueeze(1)
        out[:, indices] = expanded
    return out


def original_semantic_oracle(original, net, inp, corr, ii, jj, kk):
    """Loop neighbors and groups; no target/patch axis reduction reuse."""
    net = original.norm(net + inp + original.corr(corr))
    predecessor = torch.zeros_like(net)
    successor_indices = np.full(len(kk), -1, dtype=np.int64)
    for patch in np.unique(kk):
        members = np.flatnonzero(kk == patch)
        members = members[np.argsort(jj[members], kind="stable")]
        assert len(np.unique(jj[members])) == len(members)
        for position, edge in enumerate(members):
            if position:
                predecessor[:, edge] = net[:, members[position - 1]]
            if position + 1 < len(members):
                successor_indices[edge] = members[position + 1]
    net = net + original.c1(predecessor)
    successor = torch.zeros_like(net)
    for edge, neighbor in enumerate(successor_indices):
        if neighbor >= 0:
            successor[:, edge] = net[:, neighbor]
    net = net + original.c2(successor)
    net = net + loop_aggregate(original.agg_kk, net, kk)
    net = net + loop_aggregate(original.agg_ij, net, ii * 12345 + jj)
    net = original.gru(net)
    return net, original.d(net), original.w(net)


def random_tensors(batch, degree, patches, channels=4, correlation_channels=7, seed=93):
    generator = torch.Generator().manual_seed(seed)
    shape = (batch, degree, patches, channels)
    return (torch.randn(shape, generator=generator), torch.randn(shape, generator=generator),
            torch.randn((batch, degree, patches, correlation_channels), generator=generator))


def assert_gate(candidate, reference):
    assert candidate.shape == reference.shape
    assert candidate.dtype == torch.float32
    assert bool(torch.isfinite(candidate).all())
    # The frozen engineering gate, with no relaxation for layout/reduction changes.
    torch.testing.assert_close(candidate, reference, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("batch,degree,patches", [(1, 1, 1), (1, 2, 3), (2, 3, 2), (1, 10, 3)])
@pytest.mark.parametrize("terminal", ["original", "stable"])
def test_complete_sources_adversarial_edge_permutation(batch, degree, patches, terminal):
    original = ToyUpdate()
    wrapper = SourceFrameUpdate(original, terminal_sigmoid=terminal)
    # Two source frames share targets, but both group labels are source-closed.
    rows = []
    targets = np.arange(degree, dtype=np.int64) * 2 + 1
    for source in (3, 7):
        for patch in range(patches):
            for target in targets[::-1]:
                rows.append((source, int(target), source * patches + patch))
    permutation = np.random.default_rng(173).permutation(len(rows))
    ii, jj, kk = np.asarray(rows, dtype=np.int64)[permutation].T
    generator = torch.Generator().manual_seed(39)
    net = torch.randn((batch, len(ii), 4), generator=generator)
    inp = torch.randn(net.shape, generator=generator)
    corr = torch.randn((batch, len(ii), 7), generator=generator)
    inputs_before = tuple(value.clone() for value in (net, inp, corr))
    restored = (torch.empty_like(net), torch.empty((batch, len(ii), 2)),
                torch.empty((batch, len(ii), 2)))
    with torch.no_grad():
        expected = original_semantic_oracle(original, net, inp, corr, ii, jj, kk)
        for source in (3, 7):
            original_edges = np.flatnonzero(ii == source)
            order = np.lexsort((kk[original_edges] % patches, jj[original_edges]))
            canonical_edges = original_edges[order]
            canonical = tuple(value[:, canonical_edges].reshape(batch, degree, patches, -1).contiguous()
                              for value in (net, inp, corr))
            assert wrapper.validate_inputs(*canonical)["target_degree"] == degree
            outputs = wrapper(*canonical)
            for destination, output in zip(restored, outputs):
                destination[:, canonical_edges] = output.reshape(batch, degree * patches, -1)
    for candidate, reference in zip(restored, expected):
        assert_gate(candidate, reference)
    for value, before in zip((net, inp, corr), inputs_before):
        assert torch.equal(value, before)
    assert bool(((restored[2] >= 0) & (restored[2] <= 1)).all())


def test_boundary_bias_and_c2_reads_c1_updated_state():
    original = ToyUpdate()
    with torch.no_grad():
        original.c1[-1].bias.copy_(torch.tensor([1.2, -0.8, 0.5, 0.9]))
    wrapper = SourceFrameUpdate(original)
    net, inp, corr = random_tensors(1, 3, 2)
    seen = []
    hook = original.c2.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
    with torch.no_grad():
        n0 = original.norm(net + inp + original.corr(corr))
        previous = torch.cat((torch.zeros_like(n0[:, :1]), n0[:, :-1]), dim=1)
        boundary_add = original.c1(previous)
        assert bool((boundary_add[:, 0].abs() > 0.01).any())
        n1 = n0 + boundary_add
        correct = torch.cat((n1[:, 1:], torch.zeros_like(n1[:, -1:])), dim=1)
        incorrect_parallel = torch.cat((n0[:, 1:], torch.zeros_like(n0[:, -1:])), dim=1)
        wrapper(net, inp, corr)
    hook.remove()
    assert len(seen) == 1
    assert torch.equal(seen[0], correct)
    assert not torch.allclose(seen[0], incorrect_parallel)
    assert torch.equal(seen[0][:, -1], torch.zeros_like(seen[0][:, -1]))


def test_exact_state_names_order_parameter_identity_and_internal_sigmoids():
    original = ToyUpdate()
    assert len(original.state_dict()) == 50
    original_children = tuple(original._modules)
    wrapper = SourceFrameUpdate(original)
    assert tuple(wrapper._modules) == original_children
    assert list(wrapper.state_dict()) == list(original.state_dict())
    assert all(dict(wrapper.named_parameters())[key] is value
               for key, value in original.named_parameters())
    assert wrapper.gru is original.gru
    assert wrapper.gru[1].gate[-1] is original.gru[1].gate[-1]
    assert wrapper.gru[3].gate[-1] is original.gru[3].gate[-1]
    binding = wrapper.verify_original_binding(original)
    assert binding["state_tensors"] == 50
    assert binding["original_terminal_sigmoid_module_retained"] is True
    assert binding["raw_parameter_SHA_required_before_and_after_by_owner"] is True


@pytest.mark.parametrize("bias", [(-1000.0, 1000.0), (-20.0, 20.0), (0.0, 0.0)])
def test_stable_sigmoid_only_replaces_weight_tail_and_keeps_domain(bias):
    original = ToyUpdate()
    with torch.no_grad():
        original.w[1].weight.zero_()
        original.w[1].bias.copy_(torch.tensor(bias))
    wrapper = SourceFrameUpdate(original, terminal_sigmoid="stable")
    original_wrapper = SourceFrameUpdate(original, terminal_sigmoid="original")
    calls = {"gate1": 0, "gate2": 0, "tail": 0}
    hooks = []
    for name, module in (("gate1", original.gru[1].gate[-1]),
                         ("gate2", original.gru[3].gate[-1]), ("tail", original.w[-1])):
        def increment(module, args, output, name=name):
            calls[name] += 1
        hooks.append(module.register_forward_hook(increment))
    inputs = random_tensors(1, 2, 2)
    with torch.no_grad():
        candidate = wrapper(*inputs)
        assert calls == {"gate1": 1, "gate2": 1, "tail": 0}
        reference = original_wrapper(*inputs)
    for hook in hooks:
        hook.remove()
    assert calls == {"gate1": 2, "gate2": 2, "tail": 1}
    for actual, expected in zip(candidate, reference):
        assert_gate(actual, expected)
    assert bool(((candidate[-1] >= 0) & (candidate[-1] <= 1)).all())


@pytest.mark.parametrize("mutation", ["extra_module", "training", "child_training", "nonexpanded", "bad_tail", "fp64", "unknown_method"])
def test_constructor_rejects_unsupported_bindings(mutation):
    original = ToyUpdate()
    if mutation == "extra_module":
        original.extra = nn.Identity()
    elif mutation == "training":
        original.train()
    elif mutation == "child_training":
        original.c1.train()
    elif mutation == "nonexpanded":
        original.agg_kk.expand = False
    elif mutation == "bad_tail":
        original.w[-1] = nn.Identity()
    elif mutation == "fp64":
        original.double()
    options = {"terminal_sigmoid": "guessed"} if mutation == "unknown_method" else {}
    with pytest.raises(ValueError):
        SourceFrameUpdate(original, **options)


@pytest.mark.parametrize("mutation", ["inplace_parameter", "replace_parameter", "replace_submodule", "training"])
def test_binding_rejects_mutation(mutation):
    original = ToyUpdate()
    wrapper = SourceFrameUpdate(original)
    if mutation == "inplace_parameter":
        with torch.no_grad():
            original.c1[0].weight.add_(0.1)
    elif mutation == "replace_parameter":
        original.c1[0].weight = nn.Parameter(original.c1[0].weight.clone())
    elif mutation == "replace_submodule":
        original.c1[1] = nn.Identity()
    elif mutation == "training":
        original.gru.train()
    with pytest.raises(ValueError):
        wrapper.verify_original_binding(original)


@pytest.mark.parametrize("mutation", ["nan", "inf", "fp64", "requires_grad", "rank", "shape", "corr_width", "noncontiguous", "zero_degree"])
def test_input_contract_rejects(mutation):
    original = ToyUpdate()
    wrapper = SourceFrameUpdate(original)
    values = list(random_tensors(1, 3, 2))
    if mutation == "nan":
        values[0][0, 0, 0, 0] = float("nan")
    elif mutation == "inf":
        values[2][0, 0, 0, 0] = float("inf")
    elif mutation == "fp64":
        values[1] = values[1].double()
    elif mutation == "requires_grad":
        values[0].requires_grad_()
    elif mutation == "rank":
        values[0] = values[0].reshape(1, -1, 4)
    elif mutation == "shape":
        values[1] = values[1][:, :2].contiguous()
    elif mutation == "corr_width":
        values[2] = values[2][..., :-1].contiguous()
    elif mutation == "noncontiguous":
        values[0] = values[0].transpose(1, 2)
    elif mutation == "zero_degree":
        values = [value[:, :0] for value in values]
    with pytest.raises(ValueError):
        wrapper.validate_inputs(*values)


def test_forward_requires_no_grad_and_eval():
    original = ToyUpdate()
    wrapper = SourceFrameUpdate(original)
    values = random_tensors(1, 2, 2)
    with pytest.raises(RuntimeError):
        wrapper(*values)
    wrapper.train()
    with torch.no_grad(), pytest.raises(RuntimeError):
        wrapper(*values)


def test_extreme_group_logits_stable_per_channel():
    original = ToyUpdate()
    with torch.no_grad():
        for aggregate in (original.agg_kk, original.agg_ij):
            aggregate.g.weight.copy_(torch.eye(4) * 400)
            aggregate.g.bias.copy_(torch.tensor([800.0, -800.0, 400.0, -400.0]))
    wrapper = SourceFrameUpdate(original)
    values = random_tensors(1, 4, 3)
    degree, patches = 4, 3
    ii = np.repeat(2, degree * patches).astype(np.int64)
    jj = np.repeat(np.array([1, 3, 4, 9]), patches)
    kk = np.tile(np.arange(patches) + 2 * patches, degree)
    with torch.no_grad():
        outputs = wrapper(*values)
        flattened = tuple(value.reshape(1, degree * patches, -1) for value in values)
        reference = original_semantic_oracle(original, *flattened, ii, jj, kk)
    for output, expected in zip(outputs, reference):
        assert_gate(output.reshape(expected.shape), expected)
