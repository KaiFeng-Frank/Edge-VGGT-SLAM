"""Cache the fixed OpenCV 4.10/4.12 undistort stripe maps, using public APIs.

No crop, resize, normalization, neural work or camera-model change is performed.
The stripe schedule follows the official undistort implementation:
https://github.com/opencv/opencv/blob/4.10.0/modules/calib3d/src/undistort.dispatch.cpp#L264-L305
https://github.com/opencv/opencv/blob/4.12.0/modules/calib3d/src/undistort.dispatch.cpp#L263-L304
Actual RK3588 OpenCV 4.10 pixels/timing have not been validated by host tests.
"""
import hashlib

import cv2
import numpy as np


class FixedStripeUndistorter:
    """Fixed K/D/geometry; return a new owned uint8 BGR image on every call.

Map construction occurs once. All original per-stripe remap calls are retained
to isolate map reuse from changing remap scheduling or fixed-point rounding.
The caller must keep its original crop/resize/normalization/RNG sequence.
"""
    def __init__(self, camera_matrix, distortion, image_hw=(480, 640), *,
                 expected_opencv_version, max_map_bytes=2 * 1024 * 1024):
        if expected_opencv_version not in ('4.10.0', '4.12.0'):
            raise ValueError('OpenCV stripe source version has not been reviewed')
        if cv2.__version__ != expected_opencv_version:
            raise ValueError('Actual OpenCV version differs from declared version')
        if (not isinstance(image_hw, (tuple, list)) or len(image_hw) != 2
                or any(type(v) is not int or v <= 0 for v in image_hw)):
            raise ValueError('image_hw must contain two positive integers')
        if type(max_map_bytes) is not int or max_map_bytes <= 0:
            raise ValueError('max_map_bytes must be a positive integer')
        height, width = image_hw
        if height * width * 6 > max_map_bytes:
            raise ValueError('Fixed maps exceed declared memory budget')
        K, D = np.asarray(camera_matrix), np.asarray(distortion)
        if (K.dtype != np.float64 or K.shape != (3, 3) or not np.isfinite(K).all()
                or K[0, 0] <= 0 or K[1, 1] <= 0
                or not np.array_equal(K[2], [0, 0, 1])):
            raise ValueError('Finite original float64 camera matrix required')
        if (D.dtype != np.float64 or D.shape not in ((4,), (5,), (8,), (12,), (14,))
                or not np.isfinite(D).all()):
            raise ValueError('Finite original float64 distortion vector required')
        K, D = K.copy(), D.copy()
        identity = np.eye(3, dtype=np.float64)
        stripe_rows = min(max(1, 4096 // width), height)
        self._stripes = []
        for y in range(0, height, stripe_rows):
            rows = min(stripe_rows, height - y)
            stripe_K = K.copy()
            stripe_K[1, 2] = K[1, 2] - y
            map1, map2 = cv2.initUndistortRectifyMap(
                K, D, identity, stripe_K, (width, rows), cv2.CV_16SC2)
            if (map1.shape != (rows, width, 2) or map1.dtype != np.int16
                    or map2.shape != (rows, width) or map2.dtype != np.uint16):
                raise ValueError('Unexpected OpenCV fixed-point map contract')
            map1.flags.writeable = map2.flags.writeable = False
            self._stripes.append((y, rows, map1, map2))
        self._height, self._width = height, width
        self._version = expected_opencv_version
        self._K_hash = hashlib.sha256(K.tobytes(order='C')).hexdigest()
        self._D_hash = hashlib.sha256(D.tobytes(order='C')).hexdigest()
        self._map_bytes = sum(m1.nbytes + m2.nbytes for _, _, m1, m2 in self._stripes)
        self._stripe_rows = stripe_rows
        self._stripe_count = len(self._stripes)
        self._closed, self._calls = False, 0

    def undistort_bgr(self, image):
        if self._closed:
            raise RuntimeError('Fixed undistortion cache is closed')
        if cv2.__version__ != self._version:
            raise ValueError('OpenCV version changed after map creation')
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.shape != (self._height, self._width, 3)
                or not image.flags.c_contiguous):
            raise ValueError('Original geometry C-order uint8 BGR input required')
        output = np.empty_like(image)
        for y, rows, map1, map2 in self._stripes:
            destination = output[y:y + rows]
            result = cv2.remap(image, map1, map2, cv2.INTER_LINEAR,
                               dst=destination, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)
            if result.shape != destination.shape or not np.shares_memory(result, destination):
                raise RuntimeError('OpenCV remap did not fill the owned destination stripe')
        self._calls += 1
        return output

    def metadata(self):
        return dict(method='fixed_original_undistort_stripe_map_cache',
                    OpenCV_version=self._version, image_hw=[self._height, self._width],
                    camera_matrix_raw_sha256=self._K_hash, distortion_raw_sha256=self._D_hash,
                    stripe_rows=self._stripe_rows, stripe_count=self._stripe_count,
                    map_bytes=self._map_bytes, map_type='CV_16SC2 + CV_16UC1',
                    interpolation='INTER_LINEAR', border='BORDER_CONSTANT 0',
                    R='identity', new_camera_matrix='original K with per-stripe cy-y',
                    crop_resize_normalization_done=False, completed_calls=self._calls,
                    closed=self._closed, actual_board_pixels_validated=False,
                    performance_validated=False)

    def close(self):
        self._stripes.clear()
        self._closed = True
