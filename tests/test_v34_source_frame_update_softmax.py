"""Tiny random modules and static source checks; never load real DPVO weights."""
import ast
import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE / "source_frame_update_softmax.py"
if not CANDIDATE.is_file():
    HERE = Path(__file__).resolve().parents[1] / "deploy/rk3588/validation/dpvo_source_frame_softmax"
    CANDIDATE = HERE / CANDIDATE.name
PARENT = HERE / "source_frame_update.py"
if not PARENT.is_file():
    PARENT = HERE.parent / "dpvo_source_frame/source_frame_update.py"
variant = load(CANDIDATE, "v34_direct_softmax")
parent = load(PARENT, "v34_parent_manual")


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class Aggregate(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.expand = True
        self.f, self.g, self.h = nn.Linear(c, c), nn.Linear(c, c), nn.Linear(c, c)


class Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(c, c), nn.Sigmoid())
        self.res = nn.Sequential(nn.Linear(c, c), nn.ReLU(), nn.Linear(c, c))

    def forward(self, x):
        return x + self.gate(x) * self.res(x)


class ToyUpdate(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(39)
        c, correlation = 4, 7
        self.c1 = nn.Sequential(nn.Linear(c, c), nn.ReLU(), nn.Linear(c, c))
        self.c2 = nn.Sequential(nn.Linear(c, c), nn.ReLU(), nn.Linear(c, c))
        self.norm = nn.LayerNorm(c, eps=1e-3)
        self.agg_kk, self.agg_ij = Aggregate(c), Aggregate(c)
        self.gru = nn.Sequential(nn.LayerNorm(c, eps=1e-3), Residual(c), nn.LayerNorm(c, eps=1e-3), Residual(c))
        self.corr = nn.Sequential(nn.Linear(correlation, c), nn.ReLU(), nn.Linear(c, c),
                                  nn.LayerNorm(c, eps=1e-3), nn.ReLU(), nn.Linear(c, c))
        self.d = nn.Sequential(nn.ReLU(), nn.Linear(c, 2), nn.Identity())
        self.w = nn.Sequential(nn.ReLU(), nn.Linear(c, 2), nn.Identity(), nn.Sigmoid())
        self.eval()


def inputs(batch, degree, patches):
    generator = torch.Generator().manual_seed(29)
    return (torch.randn((batch, degree, patches, 4), generator=generator),
            torch.randn((batch, degree, patches, 4), generator=generator),
            torch.randn((batch, degree, patches, 7), generator=generator))


def loop_group(module, net, axis):
    """Independent scalar grouping over actual patch/target members."""
    result = torch.empty_like(net)
    b, degree, patches, _ = net.shape
    for batch in range(b):
        for other in range(patches if axis == 1 else degree):
            rows = net[batch, :, other] if axis == 1 else net[batch, other, :]
            logits = module.g(rows)
            weights = torch.softmax(logits, dim=0)
            value = module.h((module.f(rows) * weights).sum(dim=0))
            if axis == 1:
                result[batch, :, other] = value
            else:
                result[batch, other, :] = value
    return result


def oracle(original, net, inp, corr):
    net = original.norm(net + inp + original.corr(corr))
    previous = torch.zeros_like(net)
    for target in range(1, net.shape[1]):
        previous[:, target] = net[:, target - 1]
    net = net + original.c1(previous)
    following = torch.zeros_like(net)
    for target in range(net.shape[1] - 1):
        following[:, target] = net[:, target + 1]
    net = net + original.c2(following)
    net = net + loop_group(original.agg_kk, net, 1)
    net = net + loop_group(original.agg_ij, net, 2)
    net = original.gru(net)
    return net, original.d(net), original.w(net)


def test_frozen_parent_and_entire_class_ast_unchanged():
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == variant.PARENT_SOURCE_SHA256
    def klass(path):
        return next(v for v in ast.parse(path.read_text()).body if isinstance(v, ast.ClassDef))
    assert ast.dump(klass(CANDIDATE), include_attributes=False) == ast.dump(klass(PARENT), include_attributes=False)
    body = next(v for v in ast.parse(CANDIDATE.read_text()).body
                if isinstance(v, ast.FunctionDef) and v.name == "_aggregate")
    calls = [v for v in ast.walk(body) if isinstance(v, ast.Call)]
    assert sum(isinstance(v.func, ast.Attribute) and v.func.attr == "softmax" for v in calls) == 1
    assert not any(isinstance(v.func, ast.Attribute) and v.func.attr in ("amax", "exp") for v in calls)


@pytest.mark.parametrize("batch,degree,patches", [(1, 1, 1), (1, 2, 3), (2, 3, 2), (1, 10, 3)])
@pytest.mark.parametrize("terminal", ["original", "stable"])
def test_direct_vs_manual_and_independent_loop_groups(batch, degree, patches, terminal):
    original = ToyUpdate()
    direct = variant.SourceFrameUpdate(original, terminal_sigmoid=terminal)
    manual = parent.SourceFrameUpdate(original, terminal_sigmoid=terminal)
    args = inputs(batch, degree, patches)
    before = [v.clone() for v in args]
    direct.validate_inputs(*args)
    with torch.no_grad():
        actual, old, independent = direct(*args), manual(*args), oracle(original, *args)
    for values in (old, independent):
        for value, reference in zip(actual, values):
            torch.testing.assert_close(value, reference, rtol=1e-4, atol=1e-5)
    assert all(torch.equal(x, y) for x, y in zip(args, before))
    assert bool(torch.isfinite(actual[2]).all()) and bool(((actual[2] >= 0) & (actual[2] <= 1)).all())


def test_original_fifty_keys_parameter_aliases_and_internal_sigmoid_identity():
    original = ToyUpdate()
    wrapper = variant.SourceFrameUpdate(original)
    assert len(wrapper.state_dict()) == 50
    assert list(wrapper.state_dict()) == list(original.state_dict())
    assert tuple(wrapper._modules) == tuple(original._modules)
    assert all(dict(wrapper.named_parameters())[key] is value for key, value in original.named_parameters())
    assert wrapper.gru[1].gate[-1] is original.gru[1].gate[-1]
    assert wrapper.gru[3].gate[-1] is original.gru[3].gate[-1]
    assert wrapper.w[-1] is original.w[-1]
    wrapper.verify_original_binding(original)


def test_direct_axes_and_c2_consumes_c1_updated_values(monkeypatch):
    original = ToyUpdate()
    wrapper = variant.SourceFrameUpdate(original)
    args = inputs(1, 3, 2)
    calls, seen = [], []
    softmax = torch.softmax
    monkeypatch.setattr(torch, "softmax", lambda values, dim: (calls.append(dim), softmax(values, dim))[1])
    hook = original.c2.register_forward_pre_hook(lambda module, values: seen.append(values[0].clone()))
    with torch.no_grad():
        n0 = original.norm(args[0] + args[1] + original.corr(args[2]))
        n1 = n0 + original.c1(torch.cat((torch.zeros_like(n0[:, :1]), n0[:, :-1]), 1))
        expected = torch.cat((n1[:, 1:], torch.zeros_like(n1[:, -1:])), 1)
        wrapper(*args)
    hook.remove()
    assert calls == [1, 2] and len(seen) == 1 and torch.equal(seen[0], expected)


def test_stable_terminal_retains_extreme_weight_domain():
    original = ToyUpdate()
    with torch.no_grad():
        original.w[1].weight.zero_()
        original.w[1].bias.copy_(torch.tensor([-1000., 1000.]))
    wrapper = variant.SourceFrameUpdate(original, terminal_sigmoid="stable")
    with torch.no_grad():
        values = wrapper(*inputs(1, 2, 3))
    assert torch.equal(values[2][..., 0], torch.zeros_like(values[2][..., 0]))
    assert torch.equal(values[2][..., 1], torch.ones_like(values[2][..., 1]))


def test_original_binding_mutation_and_grad_guards():
    original = ToyUpdate()
    wrapper = variant.SourceFrameUpdate(original)
    with pytest.raises(RuntimeError, match="no_grad"):
        wrapper(*inputs(1, 2, 3))
    with torch.no_grad():
        original.c1[0].weight.add_(1)
    with pytest.raises(ValueError, match="binding"):
        wrapper.verify_original_binding(original)


def test_driver_host_rejected_before_reading_input_or_making_output(tmp_path, monkeypatch):
    driver = load(HERE / "export-v34-source-frame-softmax-board.py", "v34_driver")
    monkeypatch.setattr(driver.platform, "machine", lambda: "x86_64")
    arguments = []
    for name in ("v26-source-dir", "checkpoint", "reference-dir", "parent-source", "utility-source", "wrapper-source"):
        arguments += ["--" + name, str(tmp_path / "missing")]
    out = tmp_path / "new-output"
    arguments += ["--wrapper-sha256", "0" * 64, "--output-dir", str(out)]
    with pytest.raises(ValueError, match="RK3588-only"):
        driver.main(arguments)
    assert not out.exists()
