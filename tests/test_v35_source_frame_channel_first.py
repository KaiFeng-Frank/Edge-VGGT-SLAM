"""Tiny synthetic original-semantic modules only; no actual DPVO weights."""
import ast
import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn


REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "deploy/rk3588/validation/dpvo_source_frame_channel_first"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cf = load(HERE / "source_frame_update_channel_first.py", "v35_cf_toy")
PARENT = REPO / "deploy/rk3588/validation/dpvo_source_frame_softmax/source_frame_update_softmax.py"
parent = load(PARENT, "v35_cl_toy")


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


class Aggregate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.expand = True
        self.f = nn.Linear(channels, channels)
        self.g = nn.Linear(channels, channels)
        self.h = nn.Linear(channels, channels)


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(channels, channels), nn.Sigmoid())
        self.res = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, channels))

    def forward(self, value):
        return value + self.gate(value) * self.res(value)


class ToyUpdate(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(351)
        c = 4
        self.c1 = nn.Sequential(nn.Linear(c, c), nn.ReLU(inplace=True), nn.Linear(c, c))
        self.c2 = nn.Sequential(nn.Linear(c, c), nn.ReLU(inplace=True), nn.Linear(c, c))
        self.norm = nn.LayerNorm(c, eps=1e-3)
        self.agg_kk, self.agg_ij = Aggregate(c), Aggregate(c)
        self.gru = nn.Sequential(nn.LayerNorm(c, eps=1e-3), Residual(c),
                                 nn.LayerNorm(c, eps=1e-3), Residual(c))
        self.corr = nn.Sequential(nn.Linear(7, c), nn.ReLU(inplace=True), nn.Linear(c, c),
                                  nn.LayerNorm(c, eps=1e-3), nn.ReLU(inplace=True), nn.Linear(c, c))
        self.d = nn.Sequential(nn.ReLU(), nn.Linear(c, 2), nn.Identity())
        self.w = nn.Sequential(nn.ReLU(), nn.Linear(c, 2), nn.Identity(), nn.Sigmoid())
        with torch.no_grad():
            for value in self.modules():
                if isinstance(value, nn.LayerNorm):
                    value.weight.copy_(torch.tensor([0.8, 1.2, -0.7, 2.1]))
                    value.bias.copy_(torch.tensor([0.1, -0.4, 0.8, -0.2]))
        self.eval()


def inputs(batch, degree, patches):
    generator = torch.Generator().manual_seed(352)
    cl = (torch.randn((batch, degree, patches, 4), generator=generator),
          torch.randn((batch, degree, patches, 4), generator=generator),
          torch.randn((batch, degree, patches, 7), generator=generator))
    return cl, tuple(x.permute(0, 3, 1, 2).contiguous() for x in cl)


@pytest.mark.parametrize("batch,degree,patches", [(1, 1, 1), (1, 3, 2), (2, 2, 3), (1, 10, 3)])
@pytest.mark.parametrize("terminal", ["original", "stable"])
def test_complete_forward_cf_matches_frozen_parent(batch, degree, patches, terminal):
    original = ToyUpdate()
    a = parent.SourceFrameUpdate(original, terminal_sigmoid=terminal)
    b = cf.SourceFrameUpdateChannelFirst(original, terminal_sigmoid=terminal)
    cl, packed = inputs(batch, degree, patches)
    b.validate_inputs(*packed)
    with torch.no_grad():
        expected, actual = a(*cl), b(*packed)
    for reference, value in zip(expected, actual):
        restored = value.permute(0, 2, 3, 1).contiguous()
        torch.testing.assert_close(restored, reference, rtol=1e-4, atol=1e-5)
    assert ((actual[2] >= 0) & (actual[2] <= 1)).all()


def test_channel_norm_has_original_epsilon_affine_and_only_channel_axis():
    norm = nn.LayerNorm(4, eps=0.013).eval()
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([0.7, -1.2, 1.9, 0.2]))
        norm.bias.copy_(torch.tensor([1.1, -0.3, 0.4, 0.8]))
        value = torch.arange(24, dtype=torch.float32).reshape(1, 4, 2, 3) / 7
        actual = cf._channel_norm(norm, value)
        expected = norm(value.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        constant = torch.full((1, 4, 2, 3), 31.0)
        assert torch.equal(cf._channel_norm(norm, constant), norm.bias.reshape(1, 4, 1, 1).expand_as(constant))


def test_c2_uses_c1_updated_values_and_boundary_input_is_zero_with_bias_kept(monkeypatch):
    original = ToyUpdate()
    a, b = parent.SourceFrameUpdate(original), cf.SourceFrameUpdateChannelFirst(original)
    cl, packed = inputs(1, 3, 2)
    seen_cl, seen_cf = {}, {}
    hooks = []
    for name in ("c1", "c2"):
        hooks.append(getattr(original, name).register_forward_pre_hook(
            lambda module, args, name=name: seen_cl.__setitem__(name, args[0].detach().clone())))
    with torch.no_grad():
        a(*cl)
    for hook in hooks:
        hook.remove()
    apply = cf._apply
    def record(module, value):
        for name in ("c1", "c2"):
            if module is getattr(original, name):
                seen_cf[name] = value.detach().clone()
        return apply(module, value)
    monkeypatch.setattr(cf, "_apply", record)
    with torch.no_grad():
        b(*packed)
        assert original.c1(torch.zeros((1, 4))).abs().sum() > 0
    for name in ("c1", "c2"):
        torch.testing.assert_close(seen_cf[name].permute(0, 2, 3, 1), seen_cl[name], rtol=1e-4, atol=1e-5)
    assert torch.count_nonzero(seen_cf["c1"][:, :, 0]) == 0
    assert torch.count_nonzero(seen_cf["c2"][:, :, -1]) == 0


def test_original_50_parameters_keys_aliases_and_parent_pin():
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest() == cf.PARENT_SOURCE_SHA256
    original = ToyUpdate()
    b = cf.SourceFrameUpdateChannelFirst(original)
    assert list(b.state_dict()) == list(original.state_dict())
    assert len(b.state_dict()) == 50
    assert all(v is dict(original.named_parameters())[name] for name, v in b.named_parameters())
    result = b.verify_original_binding(original)
    assert len(result["LayerNorm_configs"]) == 4
    assert all(x["eps"] == 1e-3 for x in result["LayerNorm_configs"].values())
    with torch.no_grad():
        original.c1[0].weight.add_(1)
    with pytest.raises(ValueError, match="binding"):
        b.verify_original_binding(original)


def test_no_in_graph_axis_permutation_or_selection_source():
    tree = ast.parse((HERE / "source_frame_update_channel_first.py").read_text())
    calls = [x for x in ast.walk(tree) if isinstance(x, ast.Call)]
    assert not any(isinstance(x.func, ast.Attribute) and x.func.attr in ("permute", "transpose", "gather", "where", "pad") for x in calls)
    assert any(isinstance(x.func, ast.Attribute) and x.func.attr == "conv2d" for x in calls)


def test_input_layout_and_inference_guards():
    original = ToyUpdate()
    b = cf.SourceFrameUpdateChannelFirst(original)
    cl, packed = inputs(1, 3, 2)
    with pytest.raises(ValueError, match="shape"):
        b.validate_inputs(*cl)
    with pytest.raises(RuntimeError, match="no_grad"):
        b(*packed)
    bad = packed[0].clone()
    bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        b.validate_inputs(bad, packed[1], packed[2])
    original.train()
    with pytest.raises(ValueError, match="eval"):
        b.verify_original_binding(original)


def test_stable_tail_preserves_extreme_domain_without_clipping():
    original = ToyUpdate()
    with torch.no_grad():
        original.w[1].weight.zero_()
        original.w[1].bias.copy_(torch.tensor([-1000., 1000.]))
    b = cf.SourceFrameUpdateChannelFirst(original)
    _, packed = inputs(1, 2, 3)
    with torch.no_grad():
        result = b(*packed)[2]
    assert torch.equal(result[:, 0], torch.zeros_like(result[:, 0]))
    assert torch.equal(result[:, 1], torch.ones_like(result[:, 1]))
