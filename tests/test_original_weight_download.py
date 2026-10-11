"""Check original-weight download identity without network or model execution."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("original_weight_download", ROOT / "deploy/rk3588/download_weights.py")
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


def test_download_defaults_match_published_original_identity():
    model = json.loads((ROOT / "UPSTREAMS.json").read_text())["model"]
    assert downloader.MODEL_SIZE == model["size"]
    assert downloader.MODEL_SHA256 == model["sha256"]
    assert downloader.MODEL_URL == f'{model["url"]}/resolve/{model["revision"]}/{model["filename"]}'


@pytest.mark.parametrize("field,value", [("size", 1), ("sha256", "0" * 64)])
def test_changed_download_identity_rejected_before_destination_access(tmp_path, field, value):
    destination = tmp_path / "uncreated" / "model.pt"
    args = SimpleNamespace(size=downloader.MODEL_SIZE, sha256=downloader.MODEL_SHA256, output=str(destination))
    setattr(args, field, value)
    with pytest.raises(ValueError, match="Only the pinned original VGGT"):
        downloader.download(args)
    assert not destination.parent.exists()
