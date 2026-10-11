"""Generated file and fake-service checks; no original weights or inference."""

import builtins
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from vggt_slam import checkpoint_identity as identity


ROOT = Path(__file__).resolve().parents[1]


def test_original_identity_matches_pinned_provenance():
    model = json.loads((ROOT / "UPSTREAMS.json").read_text())["model"]
    assert identity.ORIGINAL_CHECKPOINT_BYTES == model["size"] == 5026874952
    assert identity.ORIGINAL_CHECKPOINT_SHA256 == model["sha256"] == (
        "d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0"
    )
    assert identity.HASH_CHUNK_BYTES == 4 * 1024 * 1024
    assert tuple(inspect.signature(identity.verify_original_checkpoint).parameters) == ("path",)


def test_wrong_original_size_rejects_before_reading(monkeypatch, tmp_path):
    path = tmp_path / "generated.bin"
    path.write_bytes(b"not original weights")

    def forbidden_open(*args, **kwargs):
        pytest.fail("Wrong-size checkpoint must be rejected before file reads")

    monkeypatch.setattr(Path, "open", forbidden_open)
    with pytest.raises(ValueError, match="checkpoint size mismatch"):
        identity.verify_original_checkpoint(path)


def test_directory_is_not_a_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="regular file"):
        identity.verify_original_checkpoint(tmp_path)


def test_correct_size_with_wrong_hash_rejects(monkeypatch, tmp_path):
    data = b"same length, different content"
    path = tmp_path / "generated.bin"
    path.write_bytes(data)
    # Generated file only: exercise the hash rejection without a 5 GB fixture.
    monkeypatch.setattr(identity, "ORIGINAL_CHECKPOINT_BYTES", len(data))
    monkeypatch.setattr(identity, "ORIGINAL_CHECKPOINT_SHA256", hashlib.sha256(b"x" * len(data)).hexdigest())
    with pytest.raises(ValueError, match="checkpoint SHA256 mismatch"):
        identity.verify_original_checkpoint(path)


def observe_reads(monkeypatch, path, after_first_read=None):
    original_open, sizes = Path.open, []

    class ObservedStream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def fileno(self):
            return self.stream.fileno()

        def read(self, size):
            assert size == identity.HASH_CHUNK_BYTES
            sizes.append(size)
            block = self.stream.read(size)
            if len(sizes) == 1 and after_first_read is not None:
                after_first_read(original_open)
            return block

    def open_observed(value, *args, **kwargs):
        stream = original_open(value, *args, **kwargs)
        return ObservedStream(stream) if value == path and args == ("rb",) else stream

    monkeypatch.setattr(Path, "open", open_observed)
    return sizes


def test_generated_good_file_hashes_in_bounded_chunks(monkeypatch, tmp_path):
    data = bytes(range(43))
    path = tmp_path / "generated.bin"
    path.write_bytes(data)
    # Test-only identity values and chunk size; no production override exists.
    digest = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(identity, "ORIGINAL_CHECKPOINT_BYTES", len(data))
    monkeypatch.setattr(identity, "ORIGINAL_CHECKPOINT_SHA256", digest)
    monkeypatch.setattr(identity, "HASH_CHUNK_BYTES", 7)
    sizes = observe_reads(monkeypatch, path)
    assert identity.verify_original_checkpoint(path) == {"bytes": len(data), "sha256": digest}
    assert sizes == [7] * 8  # Seven data reads and the final EOF read.


def test_generated_file_changed_during_hash_is_rejected(monkeypatch, tmp_path):
    data = b"generated immutable reference"
    path = tmp_path / "generated.bin"
    path.write_bytes(data)
    monkeypatch.setattr(identity, "ORIGINAL_CHECKPOINT_BYTES", len(data))
    monkeypatch.setattr(identity, "ORIGINAL_CHECKPOINT_SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(identity, "HASH_CHUNK_BYTES", 7)

    def append_byte(original_open):
        with original_open(path, "ab") as stream:
            stream.write(b"!")

    observe_reads(monkeypatch, path, append_byte)
    with pytest.raises(ValueError, match="checkpoint changed during verification"):
        identity.verify_original_checkpoint(path)


@pytest.fixture
def fake_runtime(monkeypatch, tmp_path):
    """Import only the loader's code with a fake torch service, never a model."""
    calls = {"torch_load": 0, "VGGT_import": 0, "downloads": []}
    fake_torch = ModuleType("torch")
    fake_torch.device = lambda value: SimpleNamespace(type=str(value))

    def forbidden_load(*args, **kwargs):
        calls["torch_load"] += 1
        pytest.fail("An invalid file reached torch.load")

    def fake_download(url, destination):
        calls["downloads"].append(url)
        Path(destination).write_bytes(b"generated downloaded invalid file")

    fake_torch.load = forbidden_load
    fake_torch.hub = SimpleNamespace(get_dir=lambda: str(tmp_path / "cache"),
                                     download_url_to_file=fake_download)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "vggt.models.vggt":
            calls["VGGT_import"] += 1
            pytest.fail("An invalid file reached the original-model import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    spec = importlib.util.spec_from_file_location("_generated_identity_runtime", ROOT / "vggt_slam/runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, calls


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("wrong", ["size", "hash"])
def test_invalid_identity_stops_before_torch_load_and_model_import(
        monkeypatch, tmp_path, fake_runtime, cached, wrong):
    runtime, calls = fake_runtime
    data = b"generated invalid local file"
    path = (tmp_path / "cache/checkpoints/model.pt") if cached else tmp_path / "local.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if wrong == "hash":
        monkeypatch.setattr(identity, "ORIGINAL_CHECKPOINT_BYTES", len(data))
    with pytest.raises(ValueError, match="size mismatch" if wrong == "size" else "SHA256 mismatch"):
        runtime.load_model(None if cached else path, device="cpu", offline=True)
    assert calls == {"torch_load": 0, "VGGT_import": 0, "downloads": []}


def test_automatic_download_uses_pinned_revision_then_checks_identity(fake_runtime):
    runtime, calls = fake_runtime
    model = json.loads((ROOT / "UPSTREAMS.json").read_text())["model"]
    assert runtime.VGGT_CHECKPOINT_URL == (
        f"https://huggingface.co/facebook/VGGT-1B/resolve/{model['revision']}/model.pt"
    )
    assert tuple(inspect.signature(runtime.load_model).parameters) == (
        "checkpoint", "device", "image_size", "prune_unused_heads", "offline"
    )
    with pytest.raises(ValueError, match="checkpoint size mismatch"):
        runtime.load_model(device="cpu", offline=False)
    assert calls == {"torch_load": 0, "VGGT_import": 0,
                     "downloads": [runtime.VGGT_CHECKPOINT_URL]}
