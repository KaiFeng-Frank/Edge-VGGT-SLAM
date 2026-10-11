"""Synchronous model execution boundary for Edge-VGGT-SLAM.

Requests are read-only to the executor: it must not mutate caller pixels.
They are already-preprocessed, C-contiguous CPU float32 RGB arrays in
NCHW order [S, 3, H, W], in [0, 1]. The executor must preserve frame and pixel
order: it must not decode, resize, rectify or attach timestamps to this array.
Source IDs and capture times remain caller metadata, without float32 casts.

Results use the same pixel grid: extrinsic [S, 3, 4] is OpenCV world-to-camera
(x right, y down, z forward), intrinsic [S, 3, 3] is in pixels, depth is
[S, H, W, 1], and depth_conf is [S, H, W]. These are finite CPU float32 arrays.
The local world gauge is that of the model call; SLAM's subsequent SL4 world
is projective, not a claim of metric SE3 or a shared gauge between batches.

Executors own their output storage, which must remain stable until the caller
finishes using it. An SDK adapter must copy outputs before releasing SDK
buffers. Validation does not copy all large arrays. Optional target_tokens
are opaque original-model diagnostics. This module provides
the original Torch adapter; it does not claim a complete RKNN model backend.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import inspect
from typing import Protocol, runtime_checkable

import numpy as np


class UnsupportedExecutionError(RuntimeError):
    """The requested model capability is not implemented by this executor."""


@dataclass(frozen=True)
class ExecutionCapabilities:
    name: str
    supports_compute_similarity: bool

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("An executor must have a nonempty name")
        if type(self.supports_compute_similarity) is not bool:
            raise TypeError("supports_compute_similarity must be a bool")


@runtime_checkable
class ModelExecutor(Protocol):
    capabilities: ExecutionCapabilities

    def predict(self, images: np.ndarray, *, compute_similarity: bool = False) -> Mapping:
        """Return owned canonical arrays, or raise; never silently fall back."""
        ...


def validate_images(images):
    """Validate the prepared CPU request without changing its storage."""
    if (not isinstance(images, np.ndarray) or images.dtype != np.float32
            or images.ndim != 4 or images.shape[0] < 1 or images.shape[1] != 3
            or min(images.shape[-2:]) < 14 or any(n % 14 for n in images.shape[-2:])
            or not images.flags.c_contiguous):
        raise ValueError("Images must be contiguous CPU float32 NCHW RGB on the patch grid")
    if not np.isfinite(images).all() or np.any(images < 0) or np.any(images > 1):
        raise ValueError("Images must be finite RGB values in [0,1]")
    return images


def _prepared_array(value):
    if isinstance(value, np.ndarray):
        array = value
    else:
        import torch
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or value.dtype != torch.float32):
            raise ValueError("Prepared images must be CPU float32 NumPy arrays or tensors")
        array = value.detach().numpy()
    # Preserve the original prepared storage when it is already contiguous.
    return np.ascontiguousarray(array)


def prepare_rgb_images(image_names, image_size, *, prepared_images=None):
    """Use the original VGGT preprocessing, or the supplied prepared pixels.

    CPU contiguous prepared tensors/arrays share their pixel storage with the
    returned array. Noncontiguous prepared inputs require a contiguous copy.
    No file is read when prepared_images is supplied.
    """
    if (type(image_size) is not int or image_size < 14 or image_size % 14
            or len(image_names) < 1):
        raise ValueError("A nonempty image list and positive patch-grid image_size are required")
    if prepared_images is None:
        from vggt.utils.load_fn import load_and_preprocess_images
        prepared_images = load_and_preprocess_images(image_names, image_size=image_size)
    images = validate_images(_prepared_array(prepared_images))
    if (images.shape[0] != len(image_names) or images.shape[-1] != image_size
            or images.shape[-2] > image_size):
        raise ValueError("Prepared images must match the image names and requested pixel grid")
    return images


def _capabilities(executor):
    caps = getattr(executor, "capabilities", None)
    if not isinstance(caps, ExecutionCapabilities) or not callable(getattr(executor, "predict", None)):
        raise TypeError("A model executor must declare ExecutionCapabilities and predict")
    return caps


def require_compute_similarity(executor):
    if not _capabilities(executor).supports_compute_similarity:
        raise UnsupportedExecutionError(
            f"Executor {_capabilities(executor).name!r} does not support compute_similarity; this original-model diagnostic is unavailable"
        )


def _ratio(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Loop image_match_ratio must be a finite numeric scalar")
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in "fiu":
        raise ValueError("Loop image_match_ratio must be a finite numeric scalar")
    result = float(array.reshape(-1)[0])
    # This is the original two-means similarity ratio, not a probability.
    # The original two-means diagnostic can exceed one. The current Solver
    # does not implement loop retrieval or accept it as a loop constraint.
    if not np.isfinite(result):
        raise ValueError("Loop image_match_ratio must be a finite numeric scalar")
    return result


def validate_predictions(predictions, images, *, compute_similarity=False):
    """Validate the complete result before SLAM can mutate loop/graph state."""
    if not isinstance(predictions, Mapping):
        raise ValueError("Model predictions must be a mapping of canonical arrays")
    result = dict(predictions)
    count, _, height, width = images.shape
    shapes = {"extrinsic": (count, 3, 4), "intrinsic": (count, 3, 3),
              "depth": (count, height, width, 1), "depth_conf": (count, height, width)}
    for key, shape in shapes.items():
        value = result.get(key)
        if (not isinstance(value, np.ndarray) or value.dtype != np.float32
                or value.shape != shape or not np.isfinite(value).all()):
            raise ValueError(f"{key} must be a finite float32 CPU array with shape {shape}")
    if "pose_enc" in result:
        value = result["pose_enc"]
        if (not isinstance(value, np.ndarray) or value.dtype != np.float32
                or value.shape != (count, 9) or not np.isfinite(value).all()):
            raise ValueError("Optional pose_enc must be a finite float32 [S,9] array")
    if "images" in result and result["images"] is not images:
        supplied = result["images"]
        if (not isinstance(supplied, np.ndarray) or supplied.dtype != np.float32
                or supplied.shape != images.shape or not np.array_equal(supplied, images)):
            raise ValueError("Executor changed the prepared images or frame order")
    result["images"] = images
    if compute_similarity:
        result["image_match_ratio"] = _ratio(result.get("image_match_ratio"))
    else:
        if result.get("image_match_ratio") is not None:
            raise ValueError("Normal predictions must not return a loop image_match_ratio")
        result["image_match_ratio"] = None
    return result


def as_executor(model_or_executor):
    """Accept an explicit executor or adapt the legacy original Torch model."""
    if hasattr(model_or_executor, "predict") or hasattr(model_or_executor, "capabilities"):
        _capabilities(model_or_executor)
        return model_or_executor
    return TorchVGGTExecutor(model_or_executor)


def get_execution_device(executor):
    """Original model input device; no auxiliary model is constructed."""
    _capabilities(executor)
    return getattr(executor, "device", "cpu")


def execute(executor, images, *, compute_similarity=False):
    """Synchronously execute and validate without a silent backend fallback."""
    _capabilities(executor)
    if type(compute_similarity) is not bool:
        raise TypeError("compute_similarity must be a bool")
    if compute_similarity:
        require_compute_similarity(executor)
    images = validate_images(images)
    if compute_similarity and images.shape[0] != 2:
        raise ValueError("Loop similarity requires a query/retrieved two-frame pair")
    predictions = executor.predict(images, compute_similarity=compute_similarity)
    return validate_predictions(predictions, images, compute_similarity=compute_similarity)


class TorchVGGTExecutor:
    """Original model call and pose codec, isolated from SLAM's graph logic.

    This adapter neither changes weights nor selects a replacement model.
    A model with an explicitly declared compute_similarity argument retains
    that capability. A generic **kwargs signature alone is not evidence of it.
    pose_decoder injection is for small synthetic tests, not another codec.
    """

    def __init__(self, model, *, pose_decoder=None):
        if not callable(model) or not callable(getattr(model, "parameters", None)):
            raise TypeError("Legacy models must be callable and expose parameters, or implement ModelExecutor")
        self.model = model
        self._pose_decoder = pose_decoder
        try:
            device = next(iter(model.parameters())).device
        except StopIteration:
            device = "cpu"
        self.device = device
        call = getattr(model, "forward", None) or model.__call__
        try:
            supports_similarity = "compute_similarity" in inspect.signature(call).parameters
        except (TypeError, ValueError):
            supports_similarity = False
        self.capabilities = ExecutionCapabilities("torch-vggt", supports_similarity)

    @staticmethod
    def _batch_array(value, key, count):
        import torch
        if not isinstance(value, torch.Tensor) or value.ndim < 2 or value.shape[:2] != (1, count):
            raise ValueError(f"Original Torch {key} must have batch/sequence dimensions [1,S,...]")
        return value.detach().float().cpu().numpy()[0]

    def predict(self, images, *, compute_similarity=False):
        import torch
        images = validate_images(images)
        if compute_similarity:
            require_compute_similarity(self)
            if len(images) != 2:
                raise ValueError("Loop similarity requires exactly two frames")
        if self._pose_decoder is None:
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
            decoder = pose_encoding_to_extri_intri
        else:
            decoder = self._pose_decoder
        count = len(images)
        tensor = torch.from_numpy(images).to(self.device)
        with torch.no_grad():
            raw = (self.model(tensor, compute_similarity=True) if compute_similarity
                   else self.model(tensor))
            if not isinstance(raw, Mapping):
                raise ValueError("Original model predictions must be a mapping")
            pose = raw.get("pose_enc")
            if (not isinstance(pose, torch.Tensor) or pose.shape != (1, count, 9)
                    or not torch.isfinite(pose).all()):
                raise ValueError("Original pose_enc must be a finite [1,S,9] tensor")
            extrinsic, intrinsic = decoder(pose, images.shape[-2:])
        result = dict(raw)
        result["extrinsic"] = self._batch_array(extrinsic, "extrinsic", count)
        result["intrinsic"] = self._batch_array(intrinsic, "intrinsic", count)
        for key in ("pose_enc", "depth", "depth_conf"):
            result[key] = self._batch_array(raw.get(key), key, count)
        # Model.eval() can return its internally dtype-cast images. Canonical
        # images are the actual prepared request, not a second decoded input.
        result["images"] = images
        ratio = raw.get("image_match_ratio")
        if isinstance(ratio, torch.Tensor):
            ratio = ratio.detach().cpu().numpy()
        result["image_match_ratio"] = ratio
        return validate_predictions(result, images, compute_similarity=compute_similarity)
