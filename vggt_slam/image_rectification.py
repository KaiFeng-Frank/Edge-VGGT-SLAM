"""Shared BGR decoding and fixed-camera rectification for both edge processes.

The raw files remain unchanged. Rectification keeps the original image size,
camera matrix and optical axes, using R=I without cropping. This module owns
only two fixed float32 remap tables; it retains no decoded image or history.
"""

from collections.abc import Mapping
import copy
import hashlib
import json
import os

import cv2
import numpy as np


def _real_array(value, name):
    try:
        objects = np.asarray(value, dtype=object)
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real numeric array") from exc
    if (array.dtype.kind not in "iuf"
            or any(isinstance(item, (bool, np.bool_)) for item in objects.flat)):
        raise ValueError(f"{name} must be a finite real numeric array")
    array = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite real numeric array")
    return array


class CalibrationRectifier:
    """Use one explicit calibration to decode and rectify uint8 BGR images.

    ``calibration`` must contain exactly ``K``, ``distortion``, ``image_hw``
    and a nonempty ``source`` string. A zero-skew pinhole K and OpenCV's
    4/5/8/12/14-element distortion vectors are supported. All configuration
    values are copied. No thread settings or other process globals are changed.
    """

    def __init__(self, calibration):
        if (not isinstance(calibration, Mapping)
                or set(calibration) != {"K", "distortion", "image_hw", "source"}):
            raise ValueError("Calibration must contain exactly K, distortion, image_hw and source")
        source = calibration["source"]
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Calibration source must be a nonempty string")

        K = _real_array(calibration["K"], "Calibration K")
        if (K.shape != (3, 3) or K[0, 0] <= 0 or K[1, 1] <= 0
                or not np.allclose(K[2], [0., 0., 1.], rtol=0, atol=1e-10)
                or abs(K[0, 1]) > 1e-10 or abs(K[1, 0]) > 1e-10):
            raise ValueError("Calibration K must be a finite positive-focal, zero-skew pinhole matrix")
        distortion = _real_array(calibration["distortion"], "Calibration distortion")
        if distortion.ndim != 1 or distortion.size not in (4, 5, 8, 12, 14):
            raise ValueError("Calibration distortion must be a vector of length 4, 5, 8, 12 or 14")
        image_hw = calibration["image_hw"]
        try:
            values = list(image_hw)
        except TypeError as exc:
            raise ValueError("Calibration image_hw must contain two positive integers") from exc
        if (len(values) != 2
                or any(isinstance(v, (bool, np.bool_))
                       or not isinstance(v, (int, np.integer)) or v <= 0 for v in values)):
            raise ValueError("Calibration image_hw must contain two positive integers")
        self._image_hw = tuple(int(v) for v in values)
        self._K = K
        self._distortion = distortion
        self._R = np.eye(3, dtype=np.float64)
        height, width = self._image_hw
        try:
            self._map_x, self._map_y = cv2.initUndistortRectifyMap(
                K, distortion, self._R, K, (width, height), cv2.CV_32FC1)
        except cv2.error as exc:
            raise ValueError("OpenCV could not build the calibration rectification maps") from exc
        if any(table.shape != self._image_hw or table.dtype != np.float32
               or not np.isfinite(table).all() for table in (self._map_x, self._map_y)):
            raise ValueError("Calibration produced invalid rectification maps")
        for array in (self._K, self._distortion, self._R, self._map_x, self._map_y):
            array.setflags(write=False)

        normalized = dict(K=K.tolist(), distortion=distortion.tolist(),
                          image_hw=list(self._image_hw), source=source)
        canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._metadata = dict(
            **normalized, output_K=K.tolist(), output_distortion=np.zeros_like(distortion).tolist(),
            R=self._R.tolist(), crop=None,
            method="opencv_initUndistortRectifyMap_remap",
            map_type="CV_32FC1", interpolation="INTER_LINEAR",
            border_mode="BORDER_CONSTANT", border_value=[0, 0, 0],
            map_bytes=int(self._map_x.nbytes + self._map_y.nbytes),
            calibration_sha256=hashlib.sha256(canonical).hexdigest(),
            calibration_hash_basis="normalized_json_utf8_sort_keys_compact")

    def rectify_bgr(self, frame):
        """Return a separate rectified BGR image with the same size and dtype."""
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.shape != (*self._image_hw, 3)):
            raise ValueError("Rectification requires a uint8 HxWx3 BGR frame matching calibration image_hw")
        return cv2.remap(frame, self._map_x, self._map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    def decode_bgr(self, path):
        """Decode with OpenCV IMREAD_COLOR, then apply the shared rectification.

        The caller must include both decoding and rectification in the measured
        acquisition-to-publication interval. A decode failure raises OSError.
        """
        try:
            filename = os.fsdecode(os.fspath(path))
        except TypeError as exc:
            raise ValueError("Image path must be a filesystem path") from exc
        try:
            frame = cv2.imread(filename, cv2.IMREAD_COLOR)
        except cv2.error as exc:
            raise OSError(f"OpenCV could not decode BGR image: {filename}") from exc
        if frame is None:
            raise OSError(f"OpenCV could not decode BGR image: {filename}")
        return self.rectify_bgr(frame)

    def metadata(self):
        """Return independent JSON values describing the fixed pixel geometry."""
        return copy.deepcopy(self._metadata)
