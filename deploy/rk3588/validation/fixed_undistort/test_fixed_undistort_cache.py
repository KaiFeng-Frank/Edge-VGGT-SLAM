"""Synthetic pixels only; never decode a real frame or execute a neural model."""
import hashlib
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

ENTRY = Path(__file__).with_name('fixed_undistort_cache.py')
spec = importlib.util.spec_from_file_location('v40_fixed_undistort_cache', ENTRY)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

K = np.array([[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1]], np.float64)
D = np.array([.2624, -.9531, -.0054, .0026, 1.1633], np.float64)


def cache(K=K, D=D, image_hw=(480, 640), **kwargs):
    return module.FixedStripeUndistorter(K, D, image_hw,
        expected_opencv_version=cv2.__version__, **kwargs)


def exact(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert a.tobytes(order='C') == b.tobytes(order='C')


@pytest.mark.parametrize('pattern', ['random', 'gradient', 'checkerboard', 'zero'])
def test_original_TUM_geometry_synthetic_pixels_exact(pattern):
    if pattern == 'random':
        image = np.random.default_rng(40).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    else:
        y, x = np.indices((480, 640))
        if pattern == 'gradient':
            image = np.stack((x % 256, y % 256, (x + 3 * y) % 256), -1).astype(np.uint8)
        elif pattern == 'checkerboard':
            image = np.repeat((((x // 7 + y // 5) % 2) * 255)[..., None], 3, -1).astype(np.uint8)
        else:
            image = np.zeros((480, 640, 3), np.uint8)
    before = image.tobytes()
    handle = cache()
    result = handle.undistort_bgr(image)
    expected = cv2.undistort(image, K, D)
    exact(result, expected)
    # Original crop -> exact half INTER_AREA -> unchanged FP32 normalization.
    a = cv2.resize(result[8:-8, 16:-16], (304, 232), interpolation=cv2.INTER_AREA)
    b = cv2.resize(expected[8:-8, 16:-16], (304, 232), interpolation=cv2.INTER_AREA)
    exact(a, b)
    exact(2 * (a.transpose(2, 0, 1).astype(np.float32) / 255) - .5,
          2 * (b.transpose(2, 0, 1).astype(np.float32) / 255) - .5)
    assert image.tobytes() == before and not np.shares_memory(result, image)
    assert handle.metadata()['stripe_rows'] == 6
    assert handle.metadata()['stripe_count'] == 80
    assert handle.metadata()['map_bytes'] == 1843200


def test_partial_last_stripe_zero_distortion_and_calibration_copy():
    local_K = np.array([[260., 0, 154.2], [0, 260., 13.5], [0, 0, 1]], np.float64)
    zero_D = np.zeros(5, np.float64)
    expected_K, expected_D = local_K.copy(), zero_D.copy()
    handle = cache(local_K, zero_D, (31, 321))
    # Constructor owns calibration values; external mutation cannot retarget maps.
    local_K[0, 0] = 17; zero_D[0] = .5
    assert handle.metadata()['camera_matrix_raw_sha256'] == hashlib.sha256(expected_K.tobytes()).hexdigest()
    for seed in (2, 3):
        image = np.random.default_rng(seed).integers(0, 256, (31, 321, 3), dtype=np.uint8)
        exact(handle.undistort_bgr(image), cv2.undistort(image, expected_K, expected_D))
    assert handle.metadata()['stripe_rows'] == 12 and handle.metadata()['stripe_count'] == 3
    assert handle.metadata()['completed_calls'] == 2
    handle.close(); handle.close()
    assert handle.metadata()['closed']
    with pytest.raises(RuntimeError, match='closed'):
        handle.undistort_bgr(image)


def test_maps_built_once_repeated_calls_own_results(monkeypatch):
    build = cv2.initUndistortRectifyMap
    calls = []
    def counted(*args, **kwargs):
        calls.append(args[3].copy())
        return build(*args, **kwargs)
    monkeypatch.setattr(cv2, 'initUndistortRectifyMap', counted)
    local_K = np.array([[20., 0, 7.1], [0, 21., 3.4], [0, 0, 1]], np.float64)
    handle = cache(local_K, np.array([.1, -.01, .001, -.002, 0], np.float64), (6, 2048))
    assert [k[1, 2] for k in calls] == [local_K[1, 2] - y for y in (0, 2, 4)]
    image = np.full((6, 2048, 3), 81, np.uint8)
    first, second = handle.undistort_bgr(image), handle.undistort_bgr(image)
    assert len(calls) == 3 and not np.shares_memory(first, second)
    assert all(not m.flags.writeable for _, _, m1, m2 in handle._stripes for m in (m1, m2))


@pytest.mark.parametrize('bad', [np.zeros((480, 640), np.uint8),
    np.zeros((480, 640, 3), np.float32), np.zeros((232, 304, 3), np.uint8)])
def test_wrong_input_rejected(bad):
    with pytest.raises(ValueError, match='input'):
        cache().undistort_bgr(bad)


@pytest.mark.parametrize('change', ['K_nan', 'D_nan', 'dtype', 'hw', 'budget', 'version'])
def test_constructor_guard(change):
    local_K, local_D, hw, kwargs = K.copy(), D.copy(), (480, 640), {}
    if change == 'K_nan': local_K[0, 0] = np.nan
    elif change == 'D_nan': local_D[0] = np.nan
    elif change == 'dtype': local_K = local_K.astype(np.float32)
    elif change == 'hw': hw = (480, False)
    elif change == 'budget': kwargs['max_map_bytes'] = 1843199
    elif change == 'version':
        with pytest.raises(ValueError, match='version'):
            module.FixedStripeUndistorter(local_K, local_D, hw, expected_opencv_version='4.9.0')
        return
    with pytest.raises(ValueError): cache(local_K, local_D, hw, **kwargs)
