"""Independent NumPy arithmetic oracle for pinned DPVO forward correlation.

Only tiny synthetic workloads or a declared subset of saved-output elements
are intended. No Torch, native helper, model, image or device is executed.
FP32 coordinates and independent non-overlapping FP32 mean pooling are used;
channel products and their sum are evaluated in float64. This is a numerical
reference, not a claim of CUDA or NEON bitwise equivalence.

Source contract: princeton-vl/DPVO at
0ac95b656d1fda91c271d2a106460d19ad966fc7,
dpvo/altcorr/correlation_kernel.cu and dpvo/{net,utils}.py. No upstream code
is imported. The fixed comparison gate remains rtol=1e-4, atol=1e-5.
"""
import hashlib
import math

import numpy as np

RTOL, ATOL = 1e-4, 1e-5
COMMIT = "0ac95b656d1fda91c271d2a106460d19ad966fc7"
SOURCE_SHA256 = {
    "dpvo/altcorr/correlation_kernel.cu": "85667383d879c9093cee5b88ae231f562d98e751b4a549917ebda105a1a291e0",
    "dpvo/utils.py": "5bc5820014045b6979d9979ae80a2b519c64f9491ae83e909a77731152751a4a",
    "dpvo/net.py": "1aee33d54f0b1cc1f3f842b60790ff1299ef9c1e20bca6467ce7114dca802acc",
}


def _array(value, dtype, ndim, name):
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype) or value.ndim != ndim:
        raise ValueError(f"{name} must be an exact {dtype} {ndim}D NumPy array")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def validate(gmap, fmap, coords, kk, jj, radius=3, levels=(1, 4)):
    _array(gmap, np.float32, 5, "gmap")
    _array(fmap, np.float32, 5, "fmap")
    _array(coords, np.float32, 5, "coords")
    _array(kk, np.int64, 1, "kk")
    _array(jj, np.int64, 1, "jj")
    if type(radius) is not int or not 0 <= radius <= 32 or tuple(levels) != (1, 4):
        raise ValueError("This oracle requires radius 0..32 and ordered levels (1,4)")
    b, patches, channels, ph, pw = gmap.shape
    if min(gmap.shape + fmap.shape) < 1 or channels % 8:
        raise ValueError("Positive dimensions and a channel count divisible by eight are required")
    if fmap.shape[0] != b or fmap.shape[2] != channels or min(fmap.shape[-2:]) < 4:
        raise ValueError("Feature batch/channels must match and both spatial dimensions must be >=4")
    if jj.shape != kk.shape or coords.shape != (b, kk.size, 2, ph, pw):
        raise ValueError("Coordinates and edge indices must have matching dimensions")
    if np.any(kk < 0) or np.any(kk >= patches) or np.any(jj < 0) or np.any(jj >= fmap.shape[1]):
        raise ValueError("Edge index outside the corresponding source patch/target view")
    if np.any(np.abs(coords) >= np.float32(2**30)):
        raise ValueError("Coordinate floor/offset exceeds the defined native integer domain")


def _pooled_vector(fmap, batch, view, y, x, level):
    """One target location only; floor output size drops incomplete windows."""
    height, width = fmap.shape[-2:]
    if not (0 <= y < height // level and 0 <= x < width // level):
        return None
    if level == 1:
        return fmap[batch, view, :, y, x]
    total = np.zeros(fmap.shape[2], np.float32)
    for dy in range(level):
        for dx in range(level):
            total = np.add(total, fmap[batch, view, :, y*level+dy, x*level+dx], dtype=np.float32)
    pooled = np.divide(total, np.float32(level*level), dtype=np.float32)
    if not np.isfinite(pooled).all():
        raise ValueError("Nonfinite FP32 pooling arithmetic")
    return pooled


def _element(gmap, fmap, coords, kk, jj, index, radius):
    b, edge, ox, oy, py, px, level_index = index
    level = (1, 4)[level_index]
    # Division precedes floor in the actual CorrBlock, including level 4.
    xy = np.divide(coords[b, edge, :, py, px], np.float32(level), dtype=np.float32)
    floor_xy = np.floor(xy)
    base_x, base_y = map(int, floor_xy)
    dx, dy = map(float, np.subtract(xy, floor_xy, dtype=np.float32))
    source = gmap[b, int(kk[edge]), :, py, px].astype(np.float64)
    # Dot each integer target vector first; bilinear combination follows.
    terms = []
    for sy, sx, weight in ((0, 0, (1-dx)*(1-dy)), (0, 1, dx*(1-dy)),
                          (1, 0, (1-dx)*dy), (1, 1, dx*dy)):
        target = _pooled_vector(fmap, b, int(jj[edge]), base_y+oy-radius+sy,
                                base_x+ox-radius+sx, level)
        if target is not None and weight:
            dot = math.fsum(float(a)*float(c) for a, c in zip(source, target))
            terms.append(weight*dot)
    value = math.fsum(terms)
    if not math.isfinite(value):
        raise ValueError("Nonfinite oracle arithmetic")
    return value


def selected_elements(gmap, fmap, coords, kk, jj, indices, radius=3):
    """Indices are (batch,edge,offsetX,offsetY,patchY,patchX,level)."""
    validate(gmap, fmap, coords, kk, jj, radius)
    shape = (gmap.shape[0], kk.size, 2*radius+1, 2*radius+1, *gmap.shape[-2:], 2)
    result = []
    for index in indices:
        if len(index) != 7 or any(type(i) is not int for i in index):
            raise ValueError("Oracle indices must contain exactly seven Python integers")
        if any(i < 0 or i >= size for i, size in zip(index, shape)):
            raise ValueError("Oracle element index outside the output shape")
        result.append(_element(gmap, fmap, coords, kk, jj, index, radius))
    return np.asarray(result, np.float64)


def tiny_full_oracle(gmap, fmap, coords, kk, jj, radius=3):
    """Full tiny synthetic result [B,E,offsetX*offsetY*patchY*patchX*level]."""
    validate(gmap, fmap, coords, kk, jj, radius)
    shape = (gmap.shape[0], kk.size, 2*radius+1, 2*radius+1, *gmap.shape[-2:], 2)
    if math.prod(shape) > 65536:
        raise ValueError("Full oracle is limited to 65,536 synthetic output elements; use a declared sample")
    values = [_element(gmap, fmap, coords, kk, jj, index, radius) for index in np.ndindex(shape)]
    return np.asarray(values, np.float64).reshape(shape).reshape(shape[0], shape[1], math.prod(shape[2:]))


def raw_metadata(array):
    value = np.ascontiguousarray(array)
    return {"shape": list(value.shape), "dtype": str(value.dtype), "bytes": value.nbytes,
            "tensor_sha256": hashlib.sha256(value.tobytes()).hexdigest()}


def error_stats(actual, expected):
    a, e = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    if a.shape != e.shape or not np.isfinite(a).all() or not np.isfinite(e).all():
        raise ValueError("Comparison requires same-shaped finite arrays")
    delta = a-e
    norm = float(np.linalg.norm(e.ravel()))
    return {"elements": int(a.size), "max_abs": float(np.abs(delta).max(initial=0)),
            "RMSE": float(np.sqrt(np.mean(delta*delta))) if a.size else 0.0,
            "relative_L2": float(np.linalg.norm(delta.ravel())/norm) if norm else None,
            "outside_fixed_FP32_gate": int(np.count_nonzero(np.abs(delta) > ATOL+RTOL*np.abs(e))),
            "rtol": RTOL, "atol": ATOL,
            "byte_equal": a.shape == e.shape and raw_metadata(np.asarray(actual)) == raw_metadata(np.asarray(expected))}
