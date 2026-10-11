"""Device selection and offline VGGT checkpoint loading for ARM and CUDA."""

from pathlib import Path
from collections.abc import Mapping

import torch

from vggt_slam.checkpoint_identity import verify_original_checkpoint


VGGT_CHECKPOINT_URL = (
    "https://huggingface.co/facebook/VGGT-1B/resolve/"
    "860abec7937da0a4c03c41d3c269c366e82abdf9/model.pt"
)


def resolve_device(device="auto"):
    if device is None or str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("VGGT supports device='cpu', 'cuda', or 'auto'.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable on this machine.")
    return device


def inference_dtype(device):
    device = resolve_device(device)
    if device.type == "cpu":
        # Cortex-A76/A55 do not have native bfloat16 matrix instructions.
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


def validate_image_size(image_size):
    image_size = int(image_size)
    if image_size < 14 or image_size % 14:
        raise ValueError("image_size must be a positive multiple of VGGT's patch size (14).")
    return image_size


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError as exc:
        # Old torch.save files cannot be memory mapped. Keep them supported,
        # while reporting other checkpoint errors unchanged.
        if "mmap can only be used" not in str(exc):
            raise
        return torch.load(path, map_location="cpu", weights_only=True)


def load_model(checkpoint=None, device="auto", image_size=518,
               prune_unused_heads=False, offline=False):
    """Load only the verified original VGGT checkpoint.

    An explicit checkpoint never accesses the network. Every existing or
    downloaded file must match the fixed original size and SHA before loading.

    ``checkpoint=None`` retains upstream's automatic model download, unless
    ``offline=True`` restricts loading to the existing torch.hub checkpoint.
    ``prune_unused_heads`` removes the point and track heads that VGGT-SPARK's
    SLAM forward pass does not call. It also retains only the intermediate
    layers consumed by the camera and depth heads.
    """
    device = resolve_device(device)
    image_size = validate_image_size(image_size)
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint).expanduser()
    else:
        checkpoint_path = Path(torch.hub.get_dir()) / "checkpoints" / "model.pt"
        if not checkpoint_path.is_file() and not offline:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(VGGT_CHECKPOINT_URL, str(checkpoint_path))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"VGGT checkpoint not found: {checkpoint_path}. "
            "Supply --checkpoint with the local pretrained weights."
        )

    verify_original_checkpoint(checkpoint_path)
    from vggt.models.vggt import VGGT

    state_dict = _load_checkpoint(checkpoint_path)
    if isinstance(state_dict, Mapping):
        for key in ("state_dict", "model"):
            if key in state_dict and isinstance(state_dict[key], Mapping):
                state_dict = state_dict[key]
                break
    if not isinstance(state_dict, Mapping) or "aggregator.camera_token" not in state_dict:
        raise ValueError(f"{checkpoint_path} does not contain a VGGT state dictionary.")

    # Keep the trained positional embeddings at 518; the ViT interpolates them
    # for other input resolutions. Changing img_size here breaks the checkpoint.
    # Avoid initializing and then copying over more than 4 GB of random weights.
    # The strict state assignment materializes parameters directly from the mmap.
    with torch.device("meta"):
        model = VGGT(enable_point=not prune_unused_heads, enable_track=not prune_unused_heads)
    if prune_unused_heads:
        state_dict = {
            key: value for key, value in state_dict.items()
            if not key.startswith(("point_head.", "track_head."))
        }
        model.aggregator.retained_layer_indices = set(model.depth_head.intermediate_layer_idx) | {23}
    model.load_state_dict(state_dict, strict=True, assign=True)
    del state_dict
    remaining_meta = [name for name, value in list(model.named_parameters()) + list(model.named_buffers())
                      if value.is_meta]
    if remaining_meta:
        raise RuntimeError(f"Checkpoint left unmaterialized tensors: {remaining_meta}")
    model.eval().requires_grad_(False)
    model = model.to(device=device, dtype=inference_dtype(device))
    model.inference_image_size = image_size
    if device.type == "cpu":
        # DPT feature maps dominate activation memory on the small board.
        model.depth_frames_chunk_size = 1
    return model
