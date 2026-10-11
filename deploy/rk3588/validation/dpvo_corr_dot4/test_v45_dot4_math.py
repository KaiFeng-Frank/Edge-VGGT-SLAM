"""Tiny generated FP32 expression tests; the C++ candidate is not executed."""
import ast
import hashlib
from pathlib import Path

import numpy as np
import pytest


def lane_dot(x, y):
    acc = np.zeros(4, np.float32)
    stop = len(x) // 4 * 4
    for c in range(0, stop, 4):
        acc = np.add(acc, np.multiply(x[c:c+4], y[c:c+4]))
    value = ((acc[0] + acc[1]) + acc[2]) + acc[3]
    for c in range(stop, len(x)):
        value = value + x[c] * y[c]
    return value


def four_independent(x, targets):
    acc = np.zeros((4, 4), np.float32)
    stop = len(x) // 4 * 4
    for c in range(0, stop, 4):
        acc = np.add(acc, np.multiply(x[None, c:c+4], targets[:, c:c+4]))
    values = np.empty(4, np.float32)
    for q in range(4):
        value = ((acc[q, 0] + acc[q, 1]) + acc[q, 2]) + acc[q, 3]
        for c in range(stop, len(x)):
            value = value + x[c] * targets[q, c]
        values[q] = value
    return values


@pytest.mark.parametrize("channels", [1, 3, 4, 5, 8, 127, 128, 129])
def test_same_per_output_FP32_order(channels):
    rng = np.random.default_rng(455)
    x = rng.normal(size=channels).astype(np.float32)
    targets = rng.normal(size=(4, channels)).astype(np.float32)
    expected = np.array([lane_dot(x, y) for y in targets], np.float32)
    assert four_independent(x, targets).tobytes() == expected.tobytes()


@pytest.mark.parametrize("origin", [(-5, -3), (-2, 0), (0, 0), (3, 2), (7, 4)])
def test_raw_grid_partial_boundaries_keep_original_zero_and_dot(origin):
    rng = np.random.default_rng(17)
    x = rng.normal(size=9).astype(np.float32)
    target = rng.normal(size=(4, 9, 9)).astype(np.float32)
    old, candidate = np.zeros((8, 8), np.float32), np.zeros((8, 8), np.float32)
    ix, iy = origin
    for oy in range(8):
        ty = iy + oy - 3
        for ox in range(8):
            tx = ix + ox - 3
            if 0 <= tx < 9 and 0 <= ty < 4:
                old[oy, ox] = lane_dot(x, target[ty, tx])
        for ox in range(0, 8, 4):
            tx = ix + ox - 3
            if 0 <= tx and tx + 3 < 9 and 0 <= ty < 4:
                candidate[oy, ox:ox+4] = four_independent(x, target[ty, tx:tx+4])
            else:
                for q in range(4):
                    if 0 <= tx + q < 9 and 0 <= ty < 4:
                        candidate[oy, ox+q] = lane_dot(x, target[ty, tx+q])
    assert candidate.tobytes() == old.tobytes()


def test_no_FMA_or_cross_target_reassociation_and_frozen_include():
    root = Path(__file__).parent
    text = (root / "dpvo_frame_corr_dot4.cpp").read_text()
    assert hashlib.sha256((root / "dpvo_corr.cpp").read_bytes()).hexdigest() == "3cf0c9cf79754fff996efc03c2ca2b162733f10b2b9aa580dee263e5df5b9990"
    assert text.count("=vaddq_f32(a") == 4
    assert "float total=((parts[0]+parts[1])+parts[2])+parts[3];" in text
    assert "tx+3<w && ty>=0 && ty<h" in text
    assert "value=dot(patch,target+(ty*w+xq)*ctx.c,ctx.c);" in text
    assert "vfma" not in text
    ast.parse((root / "build-v45-frame-corr-dot4-board.py").read_text())
