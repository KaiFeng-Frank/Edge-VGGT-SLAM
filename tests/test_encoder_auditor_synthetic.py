"""Tiny NumPy-only auditor checks; no images, weights or model execution."""
import importlib.util
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
IMPL = HERE if (HERE / 'audit-v26-dpvo-encoders.py').is_file() else HERE.parent / 'deploy/rk3588/validation'
spec = importlib.util.spec_from_file_location('encoder_saved_array_auditor', IMPL / 'audit-v26-dpvo-encoders.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def normalized_pixels():
    pixels = np.arange(256, dtype=np.uint8)
    normalized = np.float32(2) * (pixels.astype(np.float32) / np.float32(255)) - np.float32(.5)
    return pixels, normalized


def test_all_256_uint8_normalization_inverses():
    pixels, x = normalized_pixels()
    reconstructed = np.rint((x + np.float32(.5)) * np.float32(127.5)).astype(np.uint8)
    assert np.array_equal(pixels, reconstructed)


def test_all_256_normalization_raw_hashes_exact():
    _, x = normalized_pixels()
    reconstructed = np.rint((x + np.float32(.5)) * np.float32(127.5)).astype(np.uint8)
    normalized = np.float32(2) * (reconstructed.astype(np.float32) / np.float32(255)) - np.float32(.5)
    assert module.meta(normalized) == module.meta(x)


def test_identical_array_zero_error():
    y = np.array([0, 1, -1], dtype=np.float32)
    assert module.errors(y, y) == {'failed_elements': 0, 'max_abs': 0.0, 'relative_l2': 0.0,
                                   'bitwise_equal': True}


def test_nearest_float32_at_zero_absolute_gate_is_inside():
    boundary = np.array([np.float32(1e-5)], dtype=np.float32)
    zero = np.zeros(1, dtype=np.float32)
    assert module.errors(boundary, zero)['failed_elements'] == 0


def test_next_float32_above_absolute_gate_is_rejected():
    boundary = np.array([np.float32(1e-5)], dtype=np.float32)
    zero = np.zeros(1, dtype=np.float32)
    above = np.nextafter(boundary, np.float32(np.inf))
    assert module.errors(above, zero)['failed_elements'] == 1


def test_numerical_equality_is_distinct_from_bitwise_equality():
    zero = np.zeros(1, dtype=np.float32)
    negative_zero = -zero
    assert module.errors(negative_zero, zero)['failed_elements'] == 0
    assert not module.errors(negative_zero, zero)['bitwise_equal']
    assert module.meta(negative_zero) != module.meta(zero)


def test_raw_hash_uses_C_order_for_noncontiguous_array():
    sliced = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
    assert module.meta(sliced) == module.meta(np.ascontiguousarray(sliced))


def test_independent_known_relative_norm_and_count():
    truth = np.array([3, 4], dtype=np.float32)
    perturbed = np.array([4, 4], dtype=np.float32)
    score = module.errors(perturbed, truth)
    assert score['max_abs'] == 1
    assert abs(score['relative_l2'] - .2) < 1e-15
    assert score['failed_elements'] == 1
