"""Tiny synthetic topology/Update/ONNX tests, never released weights/images."""
import ast
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
IMPL = HERE if (HERE / "dpvo_update_export.py").is_file() else HERE.parent / "deploy/rk3588/validation"
spec = importlib.util.spec_from_file_location("_v28_update_synthetic", IMPL / "dpvo_update_export.py")
ADAPTER = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ADAPTER
spec.loader.exec_module(ADAPTER)
SOURCE = Path(os.environ.get("DPVO_SOURCE_DIR", str(IMPL / "official")))
OPS = Path(os.environ.get("DPVO_CPU_OPS", str(IMPL / "dpvo_cpu_ops.py")))


@pytest.fixture(scope="module", autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(scope="module")
def original_class():
    if not (SOURCE / "dpvo/net.py").is_file() or not OPS.is_file():
        pytest.skip("requires pinned DPVO source checkout and CPU ops; set DPVO_SOURCE_DIR/DPVO_CPU_OPS; "
                    "only random tiny synthetic tensors, never released checkpoint")
    for name, expected in ADAPTER.SOURCE_PINS.items():
        path = OPS if name == "dpvo_cpu_ops.py" else SOURCE / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    spec = importlib.util.spec_from_file_location("_v28_cpu_ops_synthetic", OPS)
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
    namespace = {"torch": torch, "nn": nn, "torch_scatter": ops,
                 "fastba": ops, "DIM": 8}
    for name, selected in (("dpvo/blocks.py", {"GatedResidual", "SoftAgg", "GradClip", "GradientClip"}),
                           ("dpvo/net.py", {"Update"})):
        tree = ast.parse((SOURCE / name).read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in selected]
        assert len(nodes) == len(selected)
        # Original class ASTs are compiled unchanged; only the global DIM is
        # reduced for tiny random synthetic tests, not a real-model claim.
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE / name), "exec"), namespace)
    return namespace["Update"], ops


def graph(ids=(0, 0, 0, 1, 1, 1), jj=(2, 1, 1, 0, 2, 1), kk=(4, 4, 4, 8, 8, 8), **kw):
    return ADAPTER.build_topology(*(torch.tensor(v, dtype=torch.int64) for v in (ids, jj, kk)), **kw)


def test_neighbors_stable_ties_and_not_edge_order():
    t = graph()
    assert t.previous_indices.tolist() == [2, 0, 1, 0, 5, 3]
    assert t.previous_valid.flatten().tolist() == [1, 0, 1, 0, 1, 1]
    assert t.following_indices.tolist() == [0, 2, 0, 5, 0, 4]
    assert t.following_valid.flatten().tolist() == [0, 1, 1, 1, 0, 1]


def test_group_order_signed_ids_and_original_hash_collision():
    t = graph(ids=(0, 1, -1, -1), jj=(12345, 0, 7, 7), kk=(-3, 2, -3, 2))
    assert t.ij_inverse.tolist() == [1, 1, 0, 0]
    assert t.ij_labels.tolist() == [-12338, 12345]
    assert t.kk_indices.tolist() == [[0, 2], [1, 3]]
    assert t.ij_indices.tolist() == [[2, 3], [0, 1]]


def test_key_arithmetic_preserves_int64_wraparound():
    ii = torch.tensor([2**63 - 1, -1], dtype=torch.int64)
    jj = torch.tensor([0, 0], dtype=torch.int64)
    t = ADAPTER.build_topology(ii, jj, torch.tensor([0, 1]))
    labels = ii * 12345 + jj
    assert torch.equal(t.ij_labels, torch.unique(labels))


def test_padded_capacity_copies_owned_inputs_and_receipt():
    ii, jj, kk = (np.array(v, dtype=np.int64) for v in ((0, 0, 1), (0, 1, 2), (5, 5, 6)))
    t = ADAPTER.build_topology(ii, jj, kk, group_caps={"kk": (4, 5), "ij": (5, 3)})
    ii[:] = 99
    assert t.ii.tolist() == [0, 0, 1]
    arrays = t.numpy_dict()
    arrays["ii"][:] = -3
    assert t.ii.tolist() == [0, 0, 1]
    assert len(t.tensors()) == 10
    summary = t.summary(batch=2, channels=8)
    assert summary["groups"]["kk"]["actual_groups"] == 2
    assert summary["groups"]["kk"]["padding_slots"] == 17
    assert summary["groups"]["kk"]["single_gathered_FP32_tensor_bytes"] == 2 * 4 * 5 * 8 * 4


@pytest.mark.parametrize("caps", [{"kk": (1, 3)}, {"kk": (2, 2)}, {"ij": (2, 1)},
                                  {"kk": (0, 4)}, {"kk": (True, 4)}, {"bad": (2, 3)}])
def test_capacity_invalid_or_overflow_never_truncates(caps):
    with pytest.raises(ValueError, match="capacity|group_caps"):
        graph(group_caps=caps)


@pytest.mark.parametrize("values", [np.array([0], dtype=np.int32), torch.tensor([0.]),
                                    torch.tensor([[0]]), torch.tensor([], dtype=torch.int64)])
def test_topology_dtype_shape_or_empty_rejected(values):
    with pytest.raises(ValueError, match="int64|shape"):
        ADAPTER.build_topology(values, values, values)


class TinyAgg(nn.Module):
    """Test-owned simple parameters; no mirrored scatter implementation."""
    def __init__(self, dim=3):
        super().__init__()
        self.expand = True
        self.f = nn.Linear(dim, dim)
        self.g = nn.Linear(dim, dim)
        self.h = nn.Linear(dim, dim)


def test_softmax_is_per_channel_and_padding_never_affects_denominator():
    agg = TinyAgg(2)
    with torch.no_grad():
        for layer in (agg.f, agg.g, agg.h):
            layer.weight.copy_(torch.eye(2))
            layer.bias.zero_()
    x = torch.tensor([[[0., 10.], [10., 0.], [-10., 4.]]])
    t = graph(ids=(0, 0, 1), jj=(0, 0, 1), kk=(0, 0, 1),
              group_caps={"kk": (4, 7), "ij": (4, 7)})
    out = ADAPTER._soft_aggregate(agg, x, t.kk_indices, t.kk_valid, t.kk_inverse)
    assert torch.isfinite(out).all()
    # The two channels prefer different members; scalar softmax cannot do this.
    torch.testing.assert_close(out[0, 0], torch.tensor([9.999546, 9.999546]), rtol=1e-5, atol=1e-5)
    assert torch.equal(out[0, 0], out[0, 1])
    assert torch.equal(out[0, 2], x[0, 2])


def test_large_channel_logits_and_empty_groups_remain_finite():
    agg = TinyAgg(2)
    with torch.no_grad():
        for layer in (agg.f, agg.g, agg.h):
            layer.weight.copy_(torch.eye(2))
            layer.bias.zero_()
    x = torch.tensor([[[1000., -1000.], [999., -999.]]])
    t = graph(ids=(0, 0), jj=(0, 0), kk=(0, 0), group_caps={"kk": (3, 5)})
    out = ADAPTER._soft_aggregate(agg, x, t.kk_indices, t.kk_valid, t.kk_inverse)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[0, 0], torch.tensor([999.73106, -999.26894]))


class Record(nn.Module):
    def __init__(self, response):
        super().__init__()
        self.response = response
        self.seen = None

    def forward(self, x):
        self.seen = x.detach().clone()
        return x * 0 + self.response


def test_missing_neighbor_is_zero_before_biased_c1_and_c2_reads_updated_net():
    model = SimpleNamespace(corr=nn.Identity(), norm=nn.Identity(), c1=Record(3.), c2=Record(7.),
                            agg_kk=TinyAgg(2), agg_ij=TinyAgg(2), gru=nn.Identity(),
                            d=nn.Identity(), w=nn.Identity())
    # Invoke the real adapter forward with a test-only namespace, not an alternate implementation.
    adapter = SimpleNamespace(original_update=model)
    t = graph(ids=(0, 0), jj=(0, 1), kk=(0, 0))
    net = torch.tensor([[[1., 2.], [4., 5.]]])
    ADAPTER.ExportableUpdate.forward(adapter, net, net * 0, net * 0, *t.tensors())
    assert torch.equal(model.c1.seen[0, 0], torch.zeros(2))
    assert torch.equal(model.c1.seen[0, 1], net[0, 0])
    assert torch.equal(model.c2.seen[0, 0], net[0, 1] + 3.)
    assert torch.equal(model.c2.seen[0, 1], torch.zeros(2))


@pytest.fixture
def original(original_class):
    cls, ops = original_class
    torch.manual_seed(817)
    return cls(1).eval(), ops


@pytest.mark.parametrize("padding", [False, True])
@pytest.mark.parametrize("batch", [1, 2])
def test_original_AST_nonzero_three_sequential_states_match(original, padding, batch):
    model, ops = original
    adapter = ADAPTER.ExportableUpdate(model)
    t = graph(group_caps={"kk": (5, 8), "ij": (8, 4)} if padding else None)
    previous, following = ops.neighbors(t.kk, t.jj)
    assert torch.equal(t.previous_indices, previous.clamp_min(0))
    assert torch.equal(t.following_indices, following.clamp_min(0))
    before = {k: v.detach().numpy().tobytes() for k, v in model.state_dict().items()}
    net = torch.randn(batch, 6, 8)
    actual_net = net.clone()
    inp, corr = torch.randn_like(net), torch.randn(batch, 6, 98)
    with torch.no_grad():
        for _ in range(3):
            expected_net, (expected_d, expected_w, unused) = model(net, inp, corr, None, t.ii, t.jj, t.kk)
            actual_net, actual_d, actual_w = adapter.run(actual_net, inp, corr, t)
            assert unused is None
            for got, want in ((actual_net, expected_net), (actual_d, expected_d), (actual_w, expected_w)):
                torch.testing.assert_close(got, want, rtol=ADAPTER.RTOL, atol=ADAPTER.ATOL)
            net = expected_net
    assert before == {k: v.detach().numpy().tobytes() for k, v in model.state_dict().items()}
    assert adapter.original_update is model
    assert len(adapter.state_dict()) == len(model.state_dict()) == 50
    assert {id(p) for p in adapter.parameters()} == {id(p) for p in model.parameters()}


@pytest.mark.parametrize("mutation", ["duplicate_member", "missing_member", "wrong_inverse", "empty_inverse",
                                      "unordered", "nonbinary", "negative", "padding_index",
                                      "bad_neighbor", "masked_neighbor", "corr_shape", "nan", "dtype"])
def test_preflight_rejects_corrupt_external_topology(original, mutation):
    model, _ = original
    adapter = ADAPTER.ExportableUpdate(model)
    t = graph(group_caps={"kk": (3, 4)})
    args = [torch.zeros(1, 6, 8), torch.zeros(1, 6, 8), torch.zeros(1, 6, 98),
            *(v.clone() for v in t.tensors())]
    if mutation == "duplicate_member":
        args[7][0, 1] = args[7][0, 0]
    elif mutation == "missing_member":
        args[8][0, 1] = 0
    elif mutation == "wrong_inverse":
        args[9][0] = 1
    elif mutation == "empty_inverse":
        args[9][0] = 2
    elif mutation == "unordered":
        args[7][0, :2] = args[7][0, :2].flip(0)
    elif mutation == "nonbinary":
        args[8][0, 0] = .5
    elif mutation == "negative":
        args[7][0, 0] = -1
    elif mutation == "padding_index":
        args[7][0, -1] = 1
    elif mutation == "bad_neighbor":
        args[3][0] = 6
    elif mutation == "masked_neighbor":
        args[3][1] = 1
    elif mutation == "corr_shape":
        args[2] = torch.zeros(1, 6, 99)
    elif mutation == "nan":
        args[0][0, 0, 0] = float("nan")
    elif mutation == "dtype":
        args[0] = args[0].to(torch.float64)
    with pytest.raises(ValueError):
        ADAPTER.validate_inputs(adapter, args)


def test_exported_ONNX_uses_changed_topology_with_same_shapes(original, tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    model, _ = original
    adapter = ADAPTER.ExportableUpdate(model)
    caps = {"kk": (2, 3), "ij": (3, 2)}
    a = graph(ids=(0, 0, 0, 0, 0, 0), jj=(0, 1, 2, 0, 1, 2), kk=(0, 0, 0, 1, 1, 1),
              group_caps=caps)
    b = graph(ids=(0, 0, 0, 0, 0, 0), jj=(2, 0, 2, 1, 0, 1), kk=(1, 0, 0, 1, 0, 1),
              group_caps=caps)
    assert [tuple(v.shape) for v in a.tensors()] == [tuple(v.shape) for v in b.tensors()]
    assert not torch.equal(a.kk_inverse, b.kk_inverse)
    assert not torch.equal(a.previous_indices, b.previous_indices)
    torch.manual_seed(97)
    net, inp, corr = torch.randn(1, 6, 8), torch.randn(1, 6, 8), torch.randn(1, 6, 98)
    args = (net, inp, corr, *a.tensors())
    ADAPTER.validate_inputs(adapter, args)
    path = tmp_path / "tiny-update.onnx"
    with torch.no_grad():
        torch.onnx.export(adapter, args, path, input_names=list(ADAPTER.INPUT_NAMES),
                          output_names=list(ADAPTER.OUTPUT_NAMES), opset_version=17)
    graph_onnx = onnx.load(path)
    onnx.checker.check_model(graph_onnx)
    assert [x.name for x in graph_onnx.graph.input] == list(ADAPTER.INPUT_NAMES)
    assert not set(ADAPTER.INPUT_NAMES) & {x.name for x in graph_onnx.graph.initializer}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    results = []
    for t in (a, b, a):
        args = (net, inp, corr, *t.tensors())
        ADAPTER.validate_inputs(adapter, args)
        with torch.no_grad():
            expected = adapter(*args)
            reference_net, (reference_d, reference_w, _) = model(net, inp, corr, None, t.ii, t.jj, t.kk)
        got = session.run(None, {k: v.numpy() for k, v in zip(ADAPTER.INPUT_NAMES, args)})
        for value, want, reference in zip(got, expected, (reference_net, reference_d, reference_w)):
            np.testing.assert_allclose(value, want.numpy(), rtol=ADAPTER.RTOL, atol=ADAPTER.ATOL)
            np.testing.assert_allclose(value, reference.numpy(), rtol=ADAPTER.RTOL, atol=ADAPTER.ATOL)
        results.append(got)
    assert not np.array_equal(results[0][0], results[1][0])
    assert all(np.array_equal(x, y) for x, y in zip(results[0], results[2]))


def test_constructor_never_accepts_training_mode(original):
    model, _ = original
    model.train()
    with pytest.raises(ValueError, match="eval-mode"):
        ADAPTER.ExportableUpdate(model)


def test_mismatched_index_lengths_fail():
    with pytest.raises(ValueError, match="shape"):
        ADAPTER.build_topology(torch.tensor([0, 1]), torch.tensor([0]), torch.tensor([0]))
