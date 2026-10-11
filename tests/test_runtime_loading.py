"""Exercise the real CPU checkpoint loader with tiny models, never VGGT weights."""

import gc
import sys
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from vggt_slam import runtime


@pytest.fixture
def tiny_vggt(monkeypatch):
    # Only these generated fixtures bypass the 5 GB original-file identity.
    # Production load_model has no verification bypass option.
    monkeypatch.setattr(runtime, "verify_original_checkpoint", lambda path: None)
    constructions = []

    class TinyAggregator(nn.Module):
        def __init__(self):
            super().__init__()
            self.camera_token = nn.Parameter(torch.zeros(1, 2, 1, 3))
            self.projection = nn.Linear(3, 4)
            self.register_buffer("scale", torch.arange(1, 5, dtype=torch.float32))
            # VGGT uses FloatTensor explicitly for its two nonpersistent buffers.
            self.register_buffer("_resnet_mean", torch.FloatTensor([0.485, 0.456, 0.406])
                                 .view(1, 1, 3, 1, 1), persistent=False)
            self.register_buffer("_resnet_std", torch.FloatTensor([0.229, 0.224, 0.225])
                                 .view(1, 1, 3, 1, 1), persistent=False)
            self.retained_layer_indices = None

    class TinyVGGT(nn.Module):
        def __init__(self, enable_point=True, enable_track=True):
            super().__init__()
            self.aggregator = TinyAggregator()
            self.camera_head = nn.Linear(4, 2)
            self.depth_head = nn.Linear(4, 1)
            self.depth_head.intermediate_layer_idx = [4, 11, 17, 23]
            self.point_head = nn.Linear(4, 3) if enable_point else None
            self.track_head = nn.Linear(4, 2) if enable_track else None
            self.depth_frames_chunk_size = 8
            constructions.append({"parameters": [parameter.device.type for parameter in self.parameters()],
                                  "enable_point": enable_point, "enable_track": enable_track})

        def forward(self, inputs):
            features = self.aggregator.projection(inputs) * self.aggregator.scale
            features = features + self.aggregator.camera_token.mean()
            return self.depth_head(features) + self.camera_head(features).sum(-1, keepdim=True)

    module = ModuleType("vggt.models.vggt")
    module.VGGT = TinyVGGT
    monkeypatch.setitem(sys.modules, "vggt.models.vggt", module)
    monkeypatch.setattr(torch.hub, "download_url_to_file",
                        lambda *args, **kwargs: pytest.fail("checkpoint loading accessed the network"))
    return TinyVGGT, constructions


@pytest.fixture
def reference(tiny_vggt):
    model_type, _ = tiny_vggt
    model = model_type().eval()
    assert sum(parameter.numel() for parameter in model.parameters()) < 256
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters(), start=1):
            values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
            parameter.copy_((values + index) / 17)
    return model


@pytest.mark.parametrize("wrapper", [None, "state_dict", "model"])
@pytest.mark.parametrize("prune", [False, True])
def test_mmap_assignment_preserves_exact_cpu_storage_and_buffers(
        monkeypatch, tmp_path, tiny_vggt, reference, wrapper, prune):
    checkpoint = tmp_path / "tiny.pt"
    source = reference.state_dict()
    torch.save(source if wrapper is None else {wrapper: source}, checkpoint)
    states = []
    original_load = runtime._load_checkpoint

    def capture_mmap(path):
        payload = original_load(path)
        states.append(payload if wrapper is None else payload[wrapper])
        return payload

    monkeypatch.setattr(runtime, "_load_checkpoint", capture_mmap)
    loaded = runtime.load_model(checkpoint, device="cpu", image_size=112,
                                prune_unused_heads=prune, offline=True)
    construction = tiny_vggt[1][-1]
    assert set(construction["parameters"]) == {"meta"}
    assert construction["enable_point"] is not prune
    assert construction["enable_track"] is not prune
    assert not loaded.training
    assert loaded.inference_image_size == 112
    assert loaded.depth_frames_chunk_size == 1
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    assert all(value.device.type == "cpu" and value.dtype == torch.float32
               for value in list(loaded.parameters()) + list(loaded.buffers()))
    for name, value in loaded.state_dict().items():
        torch.testing.assert_close(value, source[name], rtol=0, atol=0)
        # An equal value after a copy would not satisfy this storage check.
        assert value.data_ptr() == states[0][name].data_ptr()
    for name in ("_resnet_mean", "_resnet_std"):
        value = getattr(loaded.aggregator, name)
        assert f"aggregator.{name}" not in loaded.state_dict()
        torch.testing.assert_close(value, getattr(reference.aggregator, name), rtol=0, atol=0)
    if prune:
        assert loaded.point_head is loaded.track_head is None
        assert loaded.aggregator.retained_layer_indices == {4, 11, 17, 23}
    else:
        assert loaded.point_head is not None and loaded.track_head is not None
        assert loaded.aggregator.retained_layer_indices is None
    # Removing the deserialized state dictionary must not invalidate its mmap.
    states.clear()
    gc.collect()
    inputs = torch.tensor([[1., 2., 3.], [-2., 0., 1.]])
    with torch.inference_mode():
        torch.testing.assert_close(loaded(inputs), reference(inputs), rtol=0, atol=0)


@pytest.mark.parametrize("cached", [False, True])
def test_missing_checkpoint_fails_before_constructing_model(monkeypatch, tmp_path, tiny_vggt, cached):
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path / "empty-cache"))
    checkpoint = None if cached else tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError, match="VGGT checkpoint not found"):
        runtime.load_model(checkpoint, device="cpu", offline=True)
    assert tiny_vggt[1] == []


@pytest.mark.parametrize("payload", [None, [], {}, {"unrelated": torch.tensor([1.])}])
def test_non_vggt_checkpoints_are_rejected(tmp_path, tiny_vggt, payload):
    checkpoint = tmp_path / "invalid.pt"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="does not contain a VGGT state dictionary"):
        runtime.load_model(checkpoint, device="cpu", offline=True)
    assert tiny_vggt[1] == []


@pytest.mark.parametrize("case, message", [
    ("missing", "Missing key"), ("unexpected", "Unexpected key"), ("shape", "size mismatch"),
])
def test_strict_assignment_rejects_incomplete_or_wrong_architecture(tmp_path, reference, case, message):
    checkpoint = tmp_path / "mismatched.pt"
    state = reference.state_dict()
    if case == "missing":
        del state["depth_head.weight"]
    elif case == "unexpected":
        state["unrecognized.weight"] = torch.ones(1)
    else:
        state["aggregator.camera_token"] = torch.ones(7)
    torch.save(state, checkpoint)
    with pytest.raises(RuntimeError, match=message):
        runtime.load_model(checkpoint, device="cpu", prune_unused_heads=True, offline=True)


def test_unmaterialized_nonpersistent_buffer_is_an_explicit_failure(monkeypatch, tmp_path, tiny_vggt, reference):
    model_type, _ = tiny_vggt

    class ModelWithMetaBuffer(model_type):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.register_buffer("unloaded_cache", torch.zeros(1), persistent=False)

    monkeypatch.setattr(sys.modules["vggt.models.vggt"], "VGGT", ModelWithMetaBuffer)
    checkpoint = tmp_path / "valid-parameters.pt"
    torch.save(reference.state_dict(), checkpoint)
    with pytest.raises(RuntimeError, match="Checkpoint left unmaterialized tensors.*unloaded_cache"):
        runtime.load_model(checkpoint, device="cpu", offline=True)


def test_legacy_nonzip_checkpoint_fallback_keeps_exact_values(tmp_path, reference):
    checkpoint = tmp_path / "legacy.pt"
    torch.save(reference.state_dict(), checkpoint, _use_new_zipfile_serialization=False)
    loaded = runtime.load_model(checkpoint, device="cpu", offline=True)
    for name, value in loaded.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)
        assert value.device.type == "cpu" and not value.is_meta


def test_corrupt_checkpoint_does_not_trigger_legacy_fallback(monkeypatch, tmp_path, tiny_vggt):
    checkpoint = tmp_path / "truncated.pt"
    checkpoint.write_bytes(b"PK\x03\x04truncated-zip")
    original_load = torch.load
    calls = []

    def count_load(*args, **kwargs):
        calls.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", count_load)
    with pytest.raises(RuntimeError):
        runtime.load_model(checkpoint, device="cpu", offline=True)
    assert len(calls) == 1
    assert calls[0]["weights_only"] is True and calls[0]["mmap"] is True
    assert tiny_vggt[1] == []
