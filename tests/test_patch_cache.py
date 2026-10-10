"""Synthetic, frame-independent DINO tests; no VGGT weights or NPU execution."""
import json
import threading
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
from torch import nn

from vggt_slam.patch_cache import (PatchCacheError, PatchCacheStaleState,
                                  install_patch_cache)


class Dino(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(3, 3, bias=False)])
        self.blocks[0].weight.data.copy_(torch.eye(3))
        self.register_buffer("gain", torch.tensor(2., dtype=torch.float32))
        self.register_buffer("indices", torch.arange(3))
        self.calls = []
        self.mode = "normal"
        self.entered = self.proceed = None

    def forward(self, images):
        self.calls.append(images.detach().clone())
        if self.entered is not None:
            self.entered.set()
            assert self.proceed.wait(5)
        if self.mode == "raise":
            raise RuntimeError("synthetic DINO failure")
        pixels = images.flatten(2).transpose(1, 2)
        tokens = self.blocks[0](pixels) * self.gain
        # Tokens are a view whose parent contains two extra token frames.
        parent = torch.cat([tokens[:, :1], tokens, tokens[:, :1]], dim=1)
        patches = parent[:, 1:-1]
        if self.mode == "nan":
            patches = patches.clone()
            patches[0, 0, 0] = float("nan")
        elif self.mode == "double":
            patches = patches.double()
        elif self.mode == "missing":
            return {"another_key": patches}
        elif self.mode == "mutate":
            self.gain.add_(1)
        return {"x_norm_patchtokens": patches, "other_tokens": parent[:, :1]}


class Aggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = Dino()
        self.register_buffer("_resnet_mean", torch.tensor([.2, .3, .4]).reshape(1, 1, 3, 1, 1))
        self.register_buffer("_resnet_std", torch.tensor([.5, .6, .7]).reshape(1, 1, 3, 1, 1))
        self.camera_token = nn.Parameter(torch.ones(3))
        self.forward_calls = 0
        self.mode = "normal"

    def forward(self, images, compute_similarity=False):
        self.forward_calls += 1
        b, s, c, h, w = images.shape
        normalized = ((images-self._resnet_mean)/self._resnet_std).reshape(b*s, c, h, w)
        if self.mode == "recursive":
            return self.forward(images, compute_similarity)
        if self.mode == "kwargs":
            result = self.patch_embed(normalized, masks=None)
        elif self.mode == "no_dino":
            return torch.zeros(1)
        else:
            result = self.patch_embed(normalized)
            if self.mode == "twice":
                self.patch_embed(normalized)
        # Non-DINO model work executes on every call, including a full cache hit.
        return result if compute_similarity else result["x_norm_patchtokens"] + self.camera_token


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.aggregator = Aggregator()

    def forward(self, images, compute_similarity=False):
        return self.aggregator(images.unsqueeze(0), compute_similarity=compute_similarity)


def frames(*ids, height=2, width=3):
    return torch.stack([torch.arange(3*height*width, dtype=torch.float32).reshape(3, height, width)/100 + i
                        for i in ids])


def model():
    return Model().eval()


def run(m, images, loop=False):
    with torch.no_grad():
        return m(images, compute_similarity=loop)


def assert_accounting(handle):
    metrics = handle.metrics()
    assert metrics["hit_frames"] + metrics["miss_frames"] == metrics["input_frames"]
    assert metrics["computed_frames"] == metrics["miss_frames"]
    assert metrics["loop_computed_frames"] == metrics["loop_bypass_frames"]
    json.dumps(metrics, allow_nan=False)


def test_tail_only_overlap_order_independent_storage_and_restore():
    m, reference = model(), model()
    original_modules = {name: id(mod) for name, mod in m.named_modules()}
    handle = install_patch_cache(m)
    with torch.no_grad():
        first = m(frames(1, 5))
        expected_first = reference(frames(1, 5))
        second = m(frames(5, 78, 80))
        expected_second = reference(frames(5, 78, 80))
    assert torch.equal(first, expected_first) and torch.equal(second, expected_second)
    assert [len(call) for call in m.aggregator.patch_embed.calls] == [2, 2]
    assert m.aggregator.forward_calls == 2
    assert original_modules == {name: id(mod) for name, mod in m.named_modules()}
    tail = handle._tail_tokens
    assert tail._base is None
    assert tail.untyped_storage().nbytes() == tail.numel()*tail.element_size()
    assert tail.data_ptr() != second[-1].data_ptr()
    second[-1].fill_(999)
    assert not torch.any(tail == 999)
    # Only 80 remains, rather than keeping the formerly overlapped 5.
    assert torch.equal(run(m, frames(5, 81)), run(reference, frames(5, 81)))
    stats = handle.metrics()
    assert (stats["requests"], stats["input_frames"], stats["hit_frames"], stats["computed_frames"]) == (3, 7, 1, 6)
    assert stats["cache_entries"] == 1
    assert_accounting(handle)
    handle.close()
    handle.restore()
    handle.close()
    closed = handle.metrics()
    assert closed["closed"] and closed["restored"] and not closed["enabled"]
    assert closed["cache_entries"] == closed["cache_bytes"] == 0
    assert closed["tail_content_key"] is None and closed["computed_frames"] == 6
    assert "forward" not in m.aggregator.__dict__ and "forward" not in m.aggregator.patch_embed.__dict__
    assert torch.equal(run(m, frames(90)), run(reference, frames(90)))
    closed["requests"] = -1
    assert handle.metrics()["requests"] == 3


def test_similarity_original_dict_bypasses_without_reading_or_changing_main_tail():
    m, reference = model(), model()
    h = install_patch_cache(m)
    run(m, frames(1, 5))
    before = h.metrics()
    saved_tail = h._tail_tokens
    loop = run(m, frames(468, 18), loop=True)
    expected = run(reference, frames(468, 18), loop=True)
    assert loop.keys() == expected.keys()
    assert all(torch.equal(loop[k], expected[k]) for k in loop)
    assert h._tail_tokens is saved_tail
    assert h.metrics()["tail_content_key"] == before["tail_content_key"]
    assert torch.equal(run(m, frames(5, 78, 80)), run(reference, frames(5, 78, 80)))
    assert [len(call) for call in m.aggregator.patch_embed.calls] == [2, 2, 2]
    after = h.metrics()
    assert (after["requests"], after["hit_frames"], after["loop_bypass_calls"], after["loop_bypass_frames"]) == (2, 1, 1, 2)
    assert_accounting(h)
    h.close()


def test_four_five_five_batches_and_all_hits_recompute_no_other_model_work():
    m, reference = model(), model()
    h = install_patch_cache(m)
    for batch in (frames(17,18,19,20), frames(20,242,243,244,245), frames(245,467,468,469,470)):
        assert torch.equal(run(m, batch), run(reference, batch))
    assert [len(call) for call in m.aggregator.patch_embed.calls] == [4, 4, 4]
    assert torch.equal(run(m, frames(470,470)), run(reference, frames(470,470)))
    assert len(m.aggregator.patch_embed.calls) == 3
    assert m.aggregator.forward_calls == 4
    assert (h.metrics()["hit_frames"], h.metrics()["input_frames"]) == (4, 16)
    assert_accounting(h)
    h.close()


def test_actual_normalized_content_key_and_noncontiguous_input():
    m = model()
    h = install_patch_cache(m)
    run(m, frames(5))
    x = frames(5).transpose(-2, -1).contiguous().transpose(-2, -1)
    assert not x.is_contiguous()
    run(m, x)
    assert h.metrics()["hit_frames"] == 1
    x = frames(5)
    x[0,0,0,0] += .001
    run(m, x)
    assert h.metrics()["hit_frames"] == 1
    run(m, frames(5, height=3, width=2))
    assert h.metrics()["hit_frames"] == 1
    assert_accounting(h)
    h.close()


@pytest.mark.parametrize("change", ["weight_version", "weight_identity", "buffer_version", "buffer_identity",
                                    "dtype", "module", "forward", "hook", "training"])
def test_stale_state_cannot_use_tail_or_fallback(change):
    m = model()
    h = install_patch_cache(m)
    run(m, frames(5))
    dino = m.aggregator.patch_embed
    if change == "weight_version":
        with torch.no_grad(): dino.blocks[0].weight.add_(1)
    elif change == "weight_identity":
        dino.blocks[0].weight = nn.Parameter(dino.blocks[0].weight.detach().clone())
    elif change == "buffer_version": dino.gain.add_(1)
    elif change == "buffer_identity": dino.gain = dino.gain.clone()
    elif change == "dtype": dino.double()
    elif change == "module": dino.blocks[0] = nn.Linear(3,3,bias=False).eval()
    elif change == "forward": dino.blocks[0].forward = lambda x: x
    elif change == "hook": dino.blocks[0].register_forward_hook(lambda m,a,r: r)
    elif change == "training": dino.train()
    with pytest.raises(PatchCacheStaleState): run(m, frames(5))
    assert len(dino.calls) == 1
    assert h.metrics()["cache_entries"] == 0
    assert h.metrics()["errors"] == 1
    h.close()


@pytest.mark.parametrize("change", ["pool", "library", "packed", "closed", "rows"])
def test_fixed_backend_lifecycle_preserves_parent_identity_and_checks_changes(change):
    m = model()
    linear = m.aggregator.patch_embed.blocks[0]
    linear.packed_weight = np.zeros((3,3), dtype=np.float16)
    linear.weight_token = object()
    linear.versions = (0, 0)
    linear.pool = types.SimpleNamespace(api=types.SimpleNamespace(library=object()), closed=False,
                                       max_rows=1024,max_contexts=8,max_cache_bytes=1000,core_mask=1,weight_layout="native")
    h = install_patch_cache(m)
    run(m, frames(5))
    assert m.aggregator.patch_embed.blocks[0] is linear
    if change == "pool": linear.pool = types.SimpleNamespace()
    elif change == "library": linear.pool.api.library = object()
    elif change == "packed": linear.packed_weight = linear.packed_weight.copy()
    elif change == "closed": linear.pool.closed = True
    elif change == "rows": linear.pool.max_rows = 2048
    with pytest.raises(PatchCacheStaleState): run(m, frames(5))
    assert len(m.aggregator.patch_embed.calls) == 1
    h.close()
    assert m.aggregator.patch_embed.blocks[0] is linear


@pytest.mark.parametrize("mode", ["raise", "nan", "double", "missing", "mutate"])
def test_original_failure_or_mutation_is_visible_and_clears_tail(mode):
    m = model()
    h = install_patch_cache(m)
    run(m, frames(5))
    m.aggregator.patch_embed.mode = mode
    with pytest.raises((PatchCacheError, RuntimeError)): run(m, frames(6))
    assert h.metrics()["cache_entries"] == 0 and h.metrics()["errors"] == 1
    h.close()


def test_loop_failure_preserves_main_tail_but_stale_loop_invalidates_it():
    m = model()
    h = install_patch_cache(m)
    run(m, frames(5))
    saved = h._tail_tokens
    m.aggregator.patch_embed.mode = "raise"
    with pytest.raises(RuntimeError): run(m, frames(6), loop=True)
    assert h._tail_tokens is saved
    m.aggregator.patch_embed.gain.add_(1)
    with pytest.raises(PatchCacheStaleState): run(m, frames(5), loop=True)
    assert h.metrics()["cache_entries"] == 0
    h.close()


@pytest.mark.parametrize("mode", ["kwargs", "no_dino", "twice", "recursive"])
def test_unsupported_aggregator_usage_does_not_silently_bypass(mode):
    m = model()
    h = install_patch_cache(m)
    m.aggregator.mode = mode
    with pytest.raises(PatchCacheError): run(m, frames(5))
    assert h._context is None and h.metrics()["cache_entries"] == 0
    m.aggregator.mode = "normal"
    run(m, frames(5))
    h.close()


@pytest.mark.parametrize("invalid", ["nan", "double", "channels", "empty", "requires_grad", "grad_enabled"])
def test_invalid_input_and_grad_execution_are_explicit_errors(invalid):
    m = model()
    h = install_patch_cache(m)
    batch = frames(5)
    if invalid == "nan": batch[0,0,0,0] = float("nan")
    elif invalid == "double": batch = batch.double()
    elif invalid == "channels": batch = batch[:,:2]
    elif invalid == "empty": batch = batch[:0]
    elif invalid == "requires_grad": batch.requires_grad_()
    if invalid == "requires_grad":
        # Normalization under no_grad removes this flag, so check normalized DINO directly.
        with torch.no_grad(), pytest.raises(PatchCacheError):
            h._input(batch)
    elif invalid == "grad_enabled":
        with pytest.raises(PatchCacheError): m(batch)
    else:
        with pytest.raises((PatchCacheError, RuntimeError)): run(m, batch)
    assert not m.aggregator.patch_embed.calls
    h.close()


def test_budget_install_contract_existing_override_and_direct_call_rejection():
    for max_bytes in (True, 0, -1, 1.5):
        with pytest.raises(ValueError): install_patch_cache(model(), max_bytes=max_bytes)
    with pytest.raises(PatchCacheError): install_patch_cache(nn.Linear(3,3).eval())
    with pytest.raises(PatchCacheStaleState): install_patch_cache(Model())
    m = model()
    original = m.aggregator.patch_embed.forward
    override = lambda images: original(images)
    m.aggregator.patch_embed.forward = override
    h = install_patch_cache(m, max_bytes=1)
    with pytest.raises(PatchCacheError): install_patch_cache(m)
    with pytest.raises(PatchCacheError, match="through the original aggregator"):
        with torch.no_grad(): m.aggregator.patch_embed(frames(5))
    with pytest.raises(PatchCacheError, match="exceeding max_bytes"):
        run(m, frames(5))
    assert h.metrics()["cache_bytes"] == 0
    h.close()
    assert m.aggregator.patch_embed.forward is override


def test_inference_mode_and_serialized_close_release_after_active_forward():
    m = model()
    h = install_patch_cache(m)
    with torch.inference_mode(): run(m, frames(5))
    dino = m.aggregator.patch_embed
    dino.entered, dino.proceed = threading.Event(), threading.Event()
    with ThreadPoolExecutor(max_workers=2) as pool:
        forward = pool.submit(run, m, frames(6))
        assert dino.entered.wait(5)
        closed = pool.submit(h.close)
        assert not closed.done()
        dino.proceed.set()
        assert forward.result(timeout=5).shape == (1, 6, 3)
        closed.result(timeout=5)
    assert h.metrics()["closed"] and h.metrics()["cache_bytes"] == 0


def test_changed_owned_forward_restore_never_clobbers_caller_override():
    m = model()
    h = install_patch_cache(m)
    run(m, frames(5))
    new_forward = lambda images: {"x_norm_patchtokens": torch.zeros(len(images),6,3)}
    m.aggregator.patch_embed.forward = new_forward
    with pytest.raises(PatchCacheStaleState): run(m, frames(5))
    with pytest.raises(PatchCacheError, match="changed by caller"): h.close()
    assert m.aggregator.patch_embed.forward is new_forward
    assert "forward" not in m.aggregator.__dict__
    assert h.metrics()["closed"] and not h.metrics()["restored"] and h.metrics()["cache_bytes"] == 0
    h.close()
