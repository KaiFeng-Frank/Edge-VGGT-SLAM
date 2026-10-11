"""Inference-only external-topology adapter for the original DPVO Update.

No checkpoint is loaded here. The caller supplies the unchanged, already loaded
original Update. Its parameters and all dense submodules are reused, not copied.
Only neighbors and grouped scatter are represented by CPU-built external inputs.
Fixed export capacities do not imply fixed adjacency: every request must rebuild
and validate its topology. Full key domains and per-channel softmax are retained.

Grouped ReduceSum and a changed h GEMM row count can change FP32 roundoff. Actual
original-versus-adapter execution must pass rtol=1e-4, atol=1e-5; this module does
not establish that gate, RKNN support, a pose, or a complete VO implementation.

Derived from DPVO (MIT), Copyright (c) 2022 Princeton Vision & Learning Lab.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping

import numpy as np
import torch
from torch import nn


DPVO_COMMIT = "0ac95b656d1fda91c271d2a106460d19ad966fc7"
SOURCE_PINS = {
    "dpvo/net.py": "1aee33d54f0b1cc1f3f842b60790ff1299ef9c1e20bca6467ce7114dca802acc",
    "dpvo/blocks.py": "9c9e2848da524829374e56bb1b14a891e9e2c19bc683af8e958e8982ba33d661",
    "dpvo_cpu_ops.py": "15191a0d481176e5be80bb6c41c3fb3f4d057d95b18ab18f6e0efcd73793dd32",
}
INPUT_NAMES = (
    "net", "inp", "corr", "previous_indices", "previous_valid",
    "following_indices", "following_valid", "kk_indices", "kk_valid",
    "kk_inverse", "ij_indices", "ij_valid", "ij_inverse",
)
OUTPUT_NAMES = ("net_out", "delta", "weight")
RTOL, ATOL = 1e-4, 1e-5


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _indices(value, name):
    if isinstance(value, np.ndarray):
        _require(value.dtype == np.int64, name + " must be int64")
        value = torch.from_numpy(value.copy(order="C"))
    _require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
             and value.dtype == torch.int64 and value.ndim == 1,
             name + " must be a one-dimensional CPU int64 tensor/array")
    return value.detach().clone().contiguous()


def _info(value):
    raw = value.detach().contiguous().numpy()
    return {"shape": list(raw.shape), "dtype": str(raw.dtype),
            "bytes": raw.nbytes, "sha256": hashlib.sha256(raw.tobytes(order="C")).hexdigest()}


@dataclass(frozen=True)
class Topology:
    previous_indices: torch.Tensor
    previous_valid: torch.Tensor
    following_indices: torch.Tensor
    following_valid: torch.Tensor
    kk_indices: torch.Tensor
    kk_valid: torch.Tensor
    kk_inverse: torch.Tensor
    ij_indices: torch.Tensor
    ij_valid: torch.Tensor
    ij_inverse: torch.Tensor
    ii: torch.Tensor
    jj: torch.Tensor
    kk: torch.Tensor
    kk_labels: torch.Tensor
    ij_labels: torch.Tensor

    def tensors(self):
        """The ten CPU tensors, in INPUT_NAMES[3:] order; no graph constants."""
        return tuple(getattr(self, name) for name in INPUT_NAMES[3:])

    def numpy_dict(self):
        """Owned contiguous arrays for artifacts; mutable copies, no alias."""
        return {name: getattr(self, name).detach().numpy().copy(order="C")
                for name in (*INPUT_NAMES[3:], "ii", "jj", "kk", "kk_labels", "ij_labels")}

    def summary(self, *, batch=1, channels=384):
        _require(type(batch) is int and batch > 0 and type(channels) is int and channels > 0,
                 "batch/channels must be positive integers")
        groups = {}
        for name in ("kk", "ij"):
            index, valid = getattr(self, name + "_indices"), getattr(self, name + "_valid")
            count = valid.sum(dim=1).to(torch.int64)
            groups[name] = {
                "capacity": list(index.shape), "actual_groups": int((count > 0).sum()),
                "group_sizes": count.tolist(), "padding_slots": int(valid.numel() - valid.sum()),
                "single_gathered_FP32_tensor_bytes": batch * index.numel() * channels * 4,
                "labels": getattr(self, name + "_labels").tolist(),
            }
        return {"kind": "external_dynamic_DPVO_Update_topology", "edge_count": self.ii.numel(),
                "neighbor_rule": "stable jj sort inside kk; tied jj retain input edge order",
                "ij_group_rule": "original int64 ii*12345+jj (including collisions/wraparound)",
                "softmax": "per-channel, only real members; empty padded groups never expanded",
                "groups": groups, "arrays": {k: _info(v) for k, v in
                    ((name, getattr(self, name)) for name in
                     (*INPUT_NAMES[3:], "ii", "jj", "kk", "kk_labels", "ij_labels"))},
                "workspace_bytes_are_not_RSS_or_peak_memory": True}


def _group_table(labels, capacity, name):
    unique, inverse = torch.unique(labels, sorted=True, return_inverse=True)
    counts = torch.bincount(inverse, minlength=unique.numel())
    actual_g, actual_l = unique.numel(), int(counts.max())
    if capacity is None:
        g, l = actual_g, actual_l
    else:
        _require(isinstance(capacity, (tuple, list)) and len(capacity) == 2
                 and all(type(x) is int and x > 0 for x in capacity),
                 name + " group capacity must be (positive groups, positive slots)")
        g, l = capacity
        _require(g >= actual_g and l >= actual_l, name + " group capacity overflow; no truncation allowed")
    indices = torch.zeros((g, l), dtype=torch.int64)
    valid = torch.zeros((g, l), dtype=torch.float32)
    for group in range(actual_g):
        members = torch.nonzero(inverse == group, as_tuple=False).flatten()
        indices[group, :members.numel()] = members
        valid[group, :members.numel()] = 1
    return indices, valid, inverse.contiguous(), unique


def build_topology(ii, jj, kk, *, group_caps=None):
    """Build from this request's actual int64 graph, never from a cached graph.

    group_caps is optionally {'kk': (groups, slots), 'ij': (groups, slots)}.
    Same-capacity changed topologies work with one artifact. Overflow fails;
    supporting larger capacities requires another export, not dropped edges.
    """
    ii, jj, kk = (_indices(v, n) for v, n in ((ii, "ii"), (jj, "jj"), (kk, "kk")))
    _require(ii.shape == jj.shape == kk.shape and ii.numel() > 0,
             "ii/jj/kk must have the same nonempty edge shape")
    if group_caps is None:
        group_caps = {}
    _require(isinstance(group_caps, Mapping) and set(group_caps).issubset({"kk", "ij"}),
             "group_caps may contain only kk and ij")
    previous, following = torch.full_like(kk, -1), torch.full_like(kk, -1)
    order = torch.argsort(jj, stable=True)
    order = order[torch.argsort(kk[order], stable=True)]
    same = kk[order[1:]] == kk[order[:-1]]
    left, right = order[:-1][same], order[1:][same]
    previous[right], following[left] = left, right
    kk_table = _group_table(kk, group_caps.get("kk"), "kk")
    # Match the original int64 arithmetic, including possible label collisions.
    ij_table = _group_table(ii * 12345 + jj, group_caps.get("ij"), "ij")
    return Topology(previous.clamp_min(0), (previous >= 0).to(torch.float32).reshape(1, -1, 1),
                    following.clamp_min(0), (following >= 0).to(torch.float32).reshape(1, -1, 1),
                    *kk_table[:3], *ij_table[:3], ii, jj, kk, kk_table[3], ij_table[3])


def _cpu_tensor(value, dtype, name):
    _require(isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.dtype == dtype,
             name + " must be a CPU " + str(dtype) + " tensor")
    _require(value.is_contiguous(), name + " must be contiguous")
    if dtype == torch.float32:
        _require(bool(torch.isfinite(value).all()), name + " must be finite")


def _binary(mask, name):
    _require(bool(((mask == 0) | (mask == 1)).all()), name + " must be exactly 0/1")


def _validate_groups(index, valid, inverse, e, name):
    _require(index.ndim == 2 and min(index.shape) > 0 and valid.shape == index.shape,
             name + " group index/mask shape mismatch")
    _require(inverse.shape == (e,), name + " inverse must contain one group per edge")
    _binary(valid, name + "_valid")
    _require(bool(((index >= 0) & (index < e)).all()), name + " indices out of bounds")
    _require(bool(((inverse >= 0) & (inverse < index.shape[0])).all()), name + " inverse out of bounds")
    _require(bool((index[valid == 0] == 0).all()), name + " padding must use safe index 0")
    members = index[valid == 1]
    _require(members.numel() == e and torch.equal(torch.sort(members).values, torch.arange(e)),
             name + " groups must partition every real edge exactly once")
    rows = torch.arange(index.shape[0]).reshape(-1, 1).expand_as(index)[valid == 1]
    _require(torch.equal(inverse[members], rows), name + " inverse disagrees with membership")
    for row, mask in zip(index, valid):
        real = row[mask == 1]
        _require(real.numel() < 2 or bool((real[1:] > real[:-1]).all()),
                 name + " members must retain original edge order")
        _require(bool((mask[:-1] >= mask[1:]).all()), name + " real members must precede padding")


def validate_inputs(adapter, args):
    """CPU preflight, outside exported/timed forward. Reject malformed tables.

    This validates tensor membership, not the origin of a supplied topology.
    Use build_topology on the exact request ii/jj/kk and retain its summary.
    """
    _require(isinstance(adapter, ExportableUpdate) and isinstance(args, (tuple, list))
             and len(args) == len(INPUT_NAMES), "expected adapter and all 13 named inputs")
    for name, value in zip(INPUT_NAMES, args):
        dtype = torch.int64 if name.endswith("_indices") or name.endswith("_inverse") else torch.float32
        _cpu_tensor(value, dtype, name)
    net, inp, corr, pi, pm, ni, nm, ki, km, kr, ji, jm, jr = args
    _require(net.ndim == 3 and min(net.shape) > 0 and inp.shape == net.shape,
             "net/inp must have identical nonempty [B,E,C] shape")
    b, e, c = net.shape
    original = adapter.original_update
    _require(c == original.c1[0].in_features and corr.shape == (b, e, original.corr[0].in_features),
             "net channels/corr shape disagree with original Update")
    for index, mask, label in ((pi, pm, "previous"), (ni, nm, "following")):
        _require(index.shape == (e,) and mask.shape == (1, e, 1), label + " neighbor shape mismatch")
        _binary(mask, label + "_valid")
        _require(bool(((index >= 0) & (index < e)).all()), label + " neighbor index out of bounds")
        _require(bool((index[mask.flatten() == 0] == 0).all()), label + " missing neighbor must use safe index 0")
    _validate_groups(ki, km, kr, e, "kk")
    _validate_groups(ji, jm, jr, e, "ij")
    return True


def _soft_aggregate(module, x, indices, valid, inverse):
    # Dense g/f are evaluated at the original E shape before any padding.
    logits = module.g(x)[:, indices, :]
    values = module.f(x)[:, indices, :]
    mask = valid.unsqueeze(0).unsqueeze(-1)
    present = mask > 0
    nonempty = mask.sum(dim=2, keepdim=True) > 0
    maximum = torch.where(present, logits, torch.full_like(logits, -torch.inf)).amax(dim=2, keepdim=True)
    maximum = torch.where(nonempty, maximum, torch.zeros_like(maximum))
    # Never compute exp of padded logits relative to an actual group maximum.
    centered = torch.where(present, logits - maximum, torch.zeros_like(logits))
    exponential = centered.exp() * mask
    denominator = exponential.sum(dim=2, keepdim=True) + (~nonempty).to(x.dtype)
    weights = exponential / denominator
    group_values = (values * weights).sum(dim=2)
    return module.h(group_values)[:, inverse, :]


class ExportableUpdate(nn.Module):
    """Original inference math with supplied dynamic topology (13 inputs).

    Call validate_inputs before forward/export. Inference only: backward
    semantics of scatter and original GradientClip are not a contract here.
    """
    def __init__(self, original_update):
        super().__init__()
        _require(isinstance(original_update, nn.Module) and not original_update.training,
                 "original_update must be an eval-mode original torch module")
        for name in ("c1", "c2", "norm", "corr", "agg_kk", "agg_ij", "gru", "d", "w"):
            _require(isinstance(getattr(original_update, name, None), nn.Module),
                     "original Update missing submodule " + name)
        for name in ("agg_kk", "agg_ij"):
            _require(getattr(original_update, name).expand is True,
                     "original Update requires expanded aggregation")
        for value in original_update.parameters():
            _cpu_tensor(value, torch.float32, "original parameter")
        # One registration only. No aliases, parameter copies, mode changes,
        # or monkey patching the original golden forward.
        self.original_update = original_update

    def forward(self, net, inp, corr, previous_indices, previous_valid,
                following_indices, following_valid, kk_indices, kk_valid,
                kk_inverse, ij_indices, ij_valid, ij_inverse):
        original = self.original_update
        net = net + inp + original.corr(corr)
        net = original.norm(net)
        net = net + original.c1(previous_valid * net[:, previous_indices, :])
        # This reads the updated net, exactly as the original sequential c2.
        net = net + original.c2(following_valid * net[:, following_indices, :])
        net = net + _soft_aggregate(original.agg_kk, net, kk_indices, kk_valid, kk_inverse)
        net = net + _soft_aggregate(original.agg_ij, net, ij_indices, ij_valid, ij_inverse)
        net = original.gru(net)
        return net, original.d(net), original.w(net)

    def run(self, net, inp, corr, topology):
        args = (net, inp, corr, *topology.tensors())
        validate_inputs(self, args)
        return self(*args)
