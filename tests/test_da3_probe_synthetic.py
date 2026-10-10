"""Synthetic tensor/identity checks only; no real images or model forwards."""
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.dont_write_bytecode = True
TEST_DIR = Path(__file__).resolve().parent
HELPER = TEST_DIR.parent / "deploy/rk3588/validation/da3_probe_common.py"
if not HELPER.is_file():
    HELPER = TEST_DIR / "da3_probe_common.py"  # Isolated source-package layout.
if not HELPER.is_file():
    raise RuntimeError("Missing da3_probe_common.py; use the repository or isolated source-package layout")
source_package = os.environ.get("EDGE_DA3_SOURCE_PACKAGE")
if not source_package:
    pytest.skip("EDGE_DA3_SOURCE_PACKAGE must point to the pinned package containing official/ and manifest.json; no download is performed",
                allow_module_level=True)
ROOT = Path(source_package).resolve()
if not (ROOT / "official").is_dir() or not (ROOT / "manifest.json").is_file():
    raise RuntimeError("EDGE_DA3_SOURCE_PACKAGE does not contain the required official/ and manifest.json")
spec = importlib.util.spec_from_file_location("v24_da3_probe_test_common", HELPER)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)

MANIFEST_SHA = "729c5be1786aee50ab06d24474cf16476c6ca2e50ca6ce18f1885eaf99981e4d"
common.verify_official_source(ROOT / "official", ROOT / "manifest.json", MANIFEST_SHA)
common.install_official_import_path(ROOT / "official")


def test_normalization_preserves_pixels_and_adds_one_batch_dimension():
    raw = np.linspace(0, 1, 2 * 3 * 14 * 14, dtype=np.float32).reshape(2, 3, 14, 14)
    original = raw.copy()
    result = common.normalize_rgb_input(raw)
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1)
    expected = (torch.from_numpy(raw)[None] - mean) / std
    assert result.shape == (1, 2, 3, 14, 14)
    assert result.dtype == torch.float32
    assert result.device.type == "cpu"
    assert torch.equal(result, expected)
    assert np.array_equal(raw, original)
    assert torch.equal(common.normalize_rgb_input(torch.from_numpy(raw)[None]), expected)


@pytest.mark.parametrize("raw", [
    np.zeros((2, 3, 14, 14), dtype=np.float64),
    np.zeros((2, 3, 13, 14), dtype=np.float32),
    np.full((2, 3, 14, 14), np.nan, dtype=np.float32),
    np.full((2, 3, 14, 14), 1.1, dtype=np.float32),
])
def test_invalid_raw_pixels_rejected(raw):
    with pytest.raises(ValueError):
        common.normalize_rgb_input(raw)


def test_canonical_output_only_adds_depth_channel_and_owns_storage():
    output = {"depth": torch.ones(1, 2, 14, 14),
              "depth_conf": torch.ones(1, 2, 14, 14),
              "extrinsics": torch.zeros(1, 2, 3, 4),
              "intrinsics": torch.eye(3).expand(1, 2, 3, 3).clone()}
    raw_pose = torch.zeros(1, 2, 9)
    arrays = common.canonical_arrays(output, raw_pose, (14, 14))
    assert arrays["depth"].shape == (1, 2, 14, 14, 1)
    assert np.array_equal(arrays["extrinsic"], output["extrinsics"].numpy())
    output["depth"].zero_()
    raw_pose.fill_(9)
    assert arrays["depth"].min() == 1
    assert arrays["pose_enc"].max() == 0


def test_source_manifest_digest_cannot_be_replaced():
    with pytest.raises(ValueError, match="SHA256"):
        common.verify_official_source(ROOT / "official", ROOT / "manifest.json", "0" * 64)


def test_host_model_loading_guard_precedes_checkpoint_and_construction():
    if sys.platform != "linux" or __import__("platform").machine() in ("aarch64", "arm64"):
        pytest.skip("This check applies to the x86 host")
    with pytest.raises(RuntimeError, match="ARM board"):
        common.load_official_model(ROOT / "official", ROOT / "manifest.json", MANIFEST_SHA,
                                   ROOT / "nonexistent-model.safetensors")


def test_only_exact_checkpoint_declared_shared_storage_alias_is_expanded():
    original = torch.nn.Parameter(torch.ones(3, 2))
    expected = {"head.a": original.detach(), "head.b": original.detach()}
    stored = {"head.a": torch.full((3, 2), 4.0)}
    expanded, evidence = common.expand_shared_aliases(stored, expected,
                                                       {"model.head.b": "model.head.a"})
    assert set(expanded) == set(expected)
    assert expanded["head.b"] is stored["head.a"]
    assert len(stored) == 1
    assert evidence[0]["official_shared_storage_verified"]


@pytest.mark.parametrize("failure", ["missing_metadata", "distinct_storage", "wrong_target", "unexpected_key"])
def test_alias_repair_cannot_hide_missing_distinct_weight_or_unexpected_key(failure):
    value = torch.ones(3, 2)
    expected = {"head.a": value, "head.b": value}
    stored = {"head.a": torch.full((3, 2), 4.0)}
    metadata = {"model.head.b": "model.head.a"}
    if failure == "missing_metadata":
        metadata = {}
    elif failure == "distinct_storage":
        expected["head.b"] = value.clone()
    elif failure == "wrong_target":
        metadata["model.head.b"] = "model.absent"
    else:
        stored["extra"] = torch.zeros(1)
    with pytest.raises(ValueError):
        common.expand_shared_aliases(stored, expected, metadata)
