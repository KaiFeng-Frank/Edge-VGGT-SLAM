"""Pinned DA3-SMALL research loader; no import of the high-level AMP/GS API.

This helper does not implement a substitute network. The byte-exact official
source, YAML and complete released state are verified before construction.
Calling load_official_model is reserved for the ARM board research probes.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys

sys.dont_write_bytecode = True

OFFICIAL_COMMIT = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
MODEL_REVISION = "e08cab65ca0ec38e7826075418411ab90cab4da3"
CHECKPOINT_SHA256 = "364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf"
CHECKPOINT_BYTES = 137248940
PARAMETER_COUNT = 34299463
STATE_KEY_COUNT = 437
MODEL_CONFIG_SHA256 = "a486e29e82b7ab4a7d4cefc1ea4526cfe2ae438a572c8ca98917cfbcde7447d2"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_metadata(array):
    import numpy as np
    value = np.ascontiguousarray(array)
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "bytes": int(value.nbytes),
            "tensor_sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}


def require_board():
    if platform.machine().lower() not in ("aarch64", "arm64"):
        raise RuntimeError("This probe only executes model forwards on the ARM board")
    compatible = Path("/proc/device-tree/compatible")
    if not compatible.is_file() or b"rockchip,rk3588" not in compatible.read_bytes():
        raise RuntimeError("RK3588/RK3588S board identity is required")


def verify_official_source(source_dir, manifest_path, manifest_sha256):
    root = Path(source_dir).resolve()
    manifest_path = Path(manifest_path)
    actual_manifest = sha256_file(manifest_path)
    if actual_manifest != manifest_sha256:
        raise ValueError("Official source manifest SHA256 mismatch")
    data = json.loads(manifest_path.read_text())
    if (data.get("repo") != "ByteDance-Seed/Depth-Anything-3" or
            data.get("commit") != OFFICIAL_COMMIT or len(data.get("files", [])) != 34):
        raise ValueError("Expected the complete pinned 34-file import closure")
    verified = {}
    for record in data["files"]:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts or str(relative) in verified:
            raise ValueError("Invalid or repeated source manifest path")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Missing official source file")
        payload = path.read_bytes()
        sha = hashlib.sha256(payload).hexdigest()
        blob = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        if (len(payload) != record["bytes"] or sha != record["sha256"] or
                blob != record["git_blob_sha1"]):
            raise ValueError("Official source byte/blob mismatch: " + str(relative))
        verified[str(relative)] = sha
    return {"official_commit": OFFICIAL_COMMIT, "source_manifest_sha256": actual_manifest,
            "source_hashes": verified, "official_files_modified": False,
            "source_file_count": len(verified), "source_dir": str(root)}


def install_official_import_path(source_dir):
    root = (Path(source_dir).resolve() / "src").resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    return root


def weights_metadata(model):
    """Full state value metadata for export before/after checks, no forward."""
    import torch
    result = {}
    for key, tensor in model.state_dict().items():
        if tensor.dtype != torch.float32 or tensor.device.type != "cpu":
            raise ValueError("Expected the complete CPU FP32 official state")
        result[key] = raw_metadata(tensor.detach().contiguous().numpy())
    return result


verify_source_manifest = verify_official_source


def expand_shared_aliases(state, expected, checkpoint_metadata):
    """Restore only checkpoint-declared aliases of exact official shared storage.

    safetensors save_model removes duplicate state names for shared parameters.
    This is not a relaxed load: every missing key must be an explicit released
    alias and have identical official storage, offset, shape, stride and dtype.
    """
    missing = set(expected) - set(state)
    unexpected = set(state) - set(expected)
    shape_mismatches = {key: {"expected": list(expected[key].shape), "released": list(state[key].shape)}
                        for key in set(expected) & set(state) if expected[key].shape != state[key].shape}
    if unexpected or shape_mismatches:
        raise ValueError("Released state mismatch: " + json.dumps({"missing": sorted(missing),
                         "unexpected": sorted(unexpected), "shape_mismatches": shape_mismatches}))
    aliases = {}
    for key, target in (checkpoint_metadata or {}).items():
        if not key.startswith("model.") or not isinstance(target, str) or not target.startswith("model."):
            raise ValueError("Unexpected metadata outside the pinned shared-state alias contract")
        aliases[key[len("model."):]] = target[len("model."):]
    if set(aliases) != missing:
        raise ValueError("Missing state names are not exactly checkpoint-declared aliases: " +
                         json.dumps({"missing": {key: list(expected[key].shape) for key in sorted(missing)},
                                     "alias_keys": sorted(aliases), "unexpected": sorted(unexpected)}))
    result = dict(state)
    evidence = []
    for alias, target in sorted(aliases.items()):
        if target not in state or target not in expected:
            raise ValueError("Shared alias refers to an absent stored parameter: " + alias)
        left, right = expected[alias], expected[target]
        same_storage = (left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
                        and left.storage_offset() == right.storage_offset()
                        and left.shape == right.shape and left.stride() == right.stride()
                        and left.dtype == right.dtype and left.device == right.device)
        if not same_storage or state[target].shape != left.shape or state[target].dtype != left.dtype:
            raise ValueError("Missing state name is not an exact official shared-storage alias: " + alias)
        result[alias] = state[target]
        evidence.append({"alias": alias, "stored_target": target,
                         "official_storage_offset": left.storage_offset(),
                         "shape": list(left.shape), "stride": list(left.stride()),
                         "official_shared_storage_verified": True,
                         "checkpoint_metadata_declared": True})
    return result, evidence


def check_import_origins(source_dir, source_hashes):
    root = Path(source_dir).resolve()
    origins = {}
    for name, module in list(sys.modules.items()):
        if name.startswith("depth_anything_3.") and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(root):
                raise RuntimeError("A different installed DA3 package shadowed pinned source: " + name)
            relative = str(path.relative_to(root))
            if relative not in source_hashes or sha256_file(path) != source_hashes[relative]:
                raise RuntimeError("Unverified DA3 import origin: " + name)
            origins[name] = relative
    return origins


def load_official_model(source_dir, manifest_path, manifest_sha256, checkpoint,
                        checkpoint_sha256=CHECKPOINT_SHA256):
    """Return (official Net, provenance), strict CPU FP32/eval full-state load."""
    require_board()
    provenance = verify_official_source(source_dir, manifest_path, manifest_sha256)
    checkpoint = Path(checkpoint)
    if checkpoint_sha256 != CHECKPOINT_SHA256 or checkpoint.stat().st_size != CHECKPOINT_BYTES:
        raise ValueError("Expected the pinned complete DA3-SMALL checkpoint")
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("DA3-SMALL checkpoint SHA256 mismatch")
    install_official_import_path(source_dir)
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from depth_anything_3.cfg import create_object, load_config
    from omegaconf import OmegaConf

    config = load_config("depth_anything_3.configs.da3-small")
    metadata_config = Path(source_dir).parent / "model-metadata" / "config.json"
    if not metadata_config.is_file() or sha256_file(metadata_config) != MODEL_CONFIG_SHA256:
        raise ValueError("Pinned released model config is missing or changed")
    config_data = json.loads(metadata_config.read_text())
    if config_data.get("model_name") != "da3-small" or config_data["config"] != OmegaConf.to_container(config, resolve=True):
        raise ValueError("Official YAML and released model config differ")
    model_config_record = {"sha256": MODEL_CONFIG_SHA256,
                           "bytes": metadata_config.stat().st_size,
                           "model_revision": MODEL_REVISION}
    model = create_object(config).cpu().float().eval()
    state = load_file(str(checkpoint), device="cpu")
    if len(state) != STATE_KEY_COUNT or not all(key.startswith("model.") for key in state):
        raise ValueError("Expected all 437 uniformly model.-prefixed released keys")
    state = {key[len("model."):]: value for key, value in state.items()}
    expected = model.state_dict()
    if sum(value.numel() for value in state.values()) != PARAMETER_COUNT:
        raise ValueError("DA3-SMALL full-state parameter count mismatch")
    with safe_open(str(checkpoint), framework="pt", device="cpu") as opened:
        metadata = opened.metadata()
    state, alias_evidence = expand_shared_aliases(state, expected, metadata)
    if len(state) != 443 or len(alias_evidence) != 6 or len({x["stored_target"] for x in alias_evidence}) != 2:
        raise ValueError("Expected exactly six declared aliases for the two shared auxiliary LayerNorm parameters")
    if sum(value.numel() for value in state.values()) != PARAMETER_COUNT + 192:
        raise ValueError("Expanded state numel does not match the official shared parameter aliases")
    if sum(value.numel() for value in model.parameters()) != PARAMETER_COUNT:
        raise ValueError("The official model's distinct parameter count differs from the released checkpoint")
    for alias in alias_evidence:
        if model.get_parameter(alias["alias"]) is not model.get_parameter(alias["stored_target"]):
            raise ValueError("Shared state alias is not the same official Parameter object")
        alias["official_same_parameter_object_verified"] = True
    parameters = {}
    for key, value in state.items():
        if value.dtype != torch.float32 or value.shape != expected[key].shape or not torch.isfinite(value).all():
            raise ValueError("Released state shape/dtype/finite mismatch: " + key)
        parameters[key] = raw_metadata(value.numpy())
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Strict state load did not preserve all weights")
    if any(value.dtype != torch.float32 or value.device.type != "cpu" for value in model.parameters()):
        raise RuntimeError("Official model is not entirely CPU FP32")
    provenance.update({"checkpoint": {"model_id": "depth-anything/DA3-SMALL",
                       "revision": MODEL_REVISION, "sha256": CHECKPOINT_SHA256,
                       "bytes": CHECKPOINT_BYTES},
                       "config_sha256": sha256_file(Path(source_dir) / "src/depth_anything_3/configs/da3-small.yaml"),
                       "released_model_config": model_config_record,
                       "state_key_count": STATE_KEY_COUNT,
                       "expanded_state_key_count": len(state),
                       "released_tensor_numel": PARAMETER_COUNT,
                       "expanded_state_numel": PARAMETER_COUNT + 192,
                       "parameter_numel": PARAMETER_COUNT,
                       "checkpoint_declared_shared_aliases": alias_evidence,
                       "state_load_strict": True, "state_prefix_rule": "strip one model. from all 437 stored keys; strictly restore six declared shared aliases",
                       "parameters": parameters, "torch_version": torch.__version__,
                       "device": "cpu", "precision": "float32", "eval": True,
                       "camera_encoder_weights_loaded": True,
                       "gs_modules_constructed": False,
                       "import_origins": check_import_origins(source_dir, provenance["source_hashes"])})
    return model, provenance


def normalize_rgb_input(rgb_float32):
    """Apply the official ImageNet NORMALIZE once; preserve existing FP32 pixels.

    Accept RGB [S,3,H,W] or [1,S,3,H,W]; return CPU FP32 [1,S,3,H,W].
    No decode, uint8 conversion, resize, crop, or second normalization occurs.
    The pinned source path must already be installed and verified by the caller.
    """
    import numpy as np
    import torch
    from depth_anything_3.utils.io.input_processor import InputProcessor
    if isinstance(rgb_float32, torch.Tensor):
        value = rgb_float32
        if value.device.type != "cpu" or value.dtype != torch.float32 or value.requires_grad:
            raise ValueError("Raw RGB input must be a CPU FP32 non-grad tensor")
    else:
        value_np = np.asarray(rgb_float32)
        if value_np.dtype != np.float32:
            raise ValueError("Raw RGB input must be FP32; no implicit pixel conversion")
        value = torch.from_numpy(np.array(value_np, copy=True, order="C"))
    if value.ndim == 4:
        value = value.unsqueeze(0)
    if (value.ndim != 5 or value.shape[0] != 1 or value.shape[1] < 1 or
            value.shape[2] != 3 or min(value.shape[-2:]) < 14 or
            any(int(dimension) % 14 for dimension in value.shape[-2:])):
        raise ValueError("Expected B1 RGB patch14 input")
    if not torch.isfinite(value).all() or value.min() < 0 or value.max() > 1:
        raise ValueError("Raw RGB pixels must be finite and in [0,1]")
    output = InputProcessor.NORMALIZE(value).contiguous()
    if output.dtype != torch.float32 or not torch.isfinite(output).all():
        raise RuntimeError("Official normalization produced invalid output")
    return output


def canonical_arrays(output, raw_pose_encoding, expected_hw):
    """Copy the official core outputs; only add the canonical depth channel."""
    import numpy as np
    import torch
    h, w = expected_hw
    values = {"depth": output["depth"], "depth_conf": output["depth_conf"],
              "extrinsic": output["extrinsics"], "intrinsic": output["intrinsics"],
              "pose_enc": raw_pose_encoding}
    arrays = {}
    for name, tensor in values.items():
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise ValueError("Official output must remain CPU FP32: " + name)
        value = tensor.detach().contiguous().numpy().copy()
        if name == "depth" and value.shape == (1, 2, h, w):
            value = value[..., None]
        expected = {"depth": (1, 2, h, w, 1), "depth_conf": (1, 2, h, w),
                    "extrinsic": (1, 2, 3, 4), "intrinsic": (1, 2, 3, 3),
                    "pose_enc": (1, 2, 9)}[name]
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError("Official output shape/finite mismatch: " + name)
        arrays[name] = np.ascontiguousarray(value)
    return arrays
