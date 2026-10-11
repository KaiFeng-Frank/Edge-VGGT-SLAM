"""Pure NumPy IO validation/packing for the 13-input DPVO Update experiment.

No SDK, model, Torch or hardware is loaded. The caller supplies the pinned ABI
module and actual query records. INT64 in the header is an ABI datatype, not a
promise of support or NPU placement for an uncompiled graph. Buffers remain
owned by PackedInputs until the caller completes its synchronous SDK call.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


INPUT_NAMES = (
    "net", "inp", "corr", "previous_indices", "previous_valid",
    "following_indices", "following_valid", "kk_indices", "kk_valid",
    "kk_inverse", "ij_indices", "ij_valid", "ij_inverse",
)
OUTPUT_NAMES = ("net_out", "delta", "weight")
INDEX_NAMES = frozenset(name for name in INPUT_NAMES
                        if name.endswith(("_indices", "_inverse")))
MASK_NAMES = frozenset(name for name in INPUT_NAMES if name.endswith("_valid"))
UINT32_MAX = 2**32 - 1
INT32_MAX = 2**31 - 1


def _require(ok, reason):
    if not ok:
        raise ValueError(reason)


def _contract(rows, names):
    _require(isinstance(rows, (list, tuple)) and len(rows) == len(names),
             "Wrong logical contract count")
    result = {}
    for row in rows:
        name = row["name"]
        _require(name in names and name not in result, "Unknown/duplicate logical name")
        shape = tuple(row["shape"])
        _require(1 <= len(shape) <= 3 and all(type(x) is int and 0 < x <= UINT32_MAX for x in shape),
                 "Only explicit positive rank1..3 logical shapes are supported")
        _require(type(row["dtype"]) in (int, str), "Invalid logical dtype")
        dtype = {1: "float32", 7: "int64", "float32": "float32", "int64": "int64"}.get(row["dtype"])
        expected = "int64" if name in INDEX_NAMES else "float32"
        _require(dtype == expected, "Logical dtype differs: " + name)
        result[name] = dict(name=name, shape=shape, logical_shape=list(shape),
                            logical_dtype=dtype, source_dtype=dtype)
    _require(set(result) == set(names), "Missing logical name")
    return result


def _attr(row, logical, index):
    name, shape = logical["name"], logical["shape"]
    _require(row["index"] == index and row["name"] == name,
             "SDK name/index binding mismatch")
    _require(tuple(row["shape"]) == shape and row["n_elems"] == math.prod(shape),
             "SDK shape/element count differs: " + name)
    # No image-layout permutation is defined for these rank1..3 operands.
    # Do not relabel bytes as NHWC, nor infer a mapping from element count.
    _require(row["format"] in (0, 3), "Unimplemented SDK layout: " + name)
    kind, quant = row["type"], row["quantization_type"]
    scale = float(row["scale"])
    _require(math.isfinite(scale), "Nonfinite SDK scale")
    identity_affine = (quant == 2 and row["zero_point"] == 0 and scale == 1.0
                       and row["fractional_length"] == 0)
    if logical["logical_dtype"] == "int64":
        # Actual 2.3.2 compiled Update attrs label even INT64 as affine with
        # exactly neutral metadata. This does not change the index datatype.
        _require(kind in (6, 8) and (quant == 0 or identity_affine),
                 "Index requires INT32/INT64 with NONE or exact identity-affine")
        native_bytes = 4 if kind == 6 else 8
        transport_type, transport_dtype = kind, "int32" if kind == 6 else "int64"
    else:
        identity_half = kind == 1 and identity_affine
        _require(kind in (0, 1) and (quant == 0 or identity_half),
                 "Float requires FP32/FP16 with NONE or exact FP16 identity-affine")
        native_bytes = 4 if kind == 0 else 2
        transport_type, transport_dtype = 0, "float32"
    _require(row["size"] == math.prod(shape) * native_bytes,
             "SDK logical bytes disagree with shape/dtype")
    transport_bytes = math.prod(shape) * np.dtype(transport_dtype).itemsize
    _require(transport_bytes <= UINT32_MAX, "Input bytes exceed uint32 ABI size")
    return dict(logical, index=index, sdk_type=kind, sdk_format=row["format"],
                sdk_bytes=row["size"], sdk_quantization_type=quant,
                sdk_zero_point=row["zero_point"], sdk_scale=scale,
                sdk_fractional_length=row["fractional_length"],
                neutral_affine_metadata=identity_affine, transport_type=transport_type,
                transport_dtype=transport_dtype, transport_format=row["format"],
                transport_bytes=transport_bytes, pass_through=0)


def validate_input_attrs(contracts, attrs):
    """Bind raw attr_record dictionaries by SDK index and exact logical name.

    Input attrs come from RKNN_QUERY_INPUT_ATTR, not packed native zero-copy
    attrs. Shape/layout remapping and nonidentity integer quantization fail.
    """
    logical = _contract(contracts, INPUT_NAMES)
    _require(len(attrs) == 13, "SDK must expose all13 inputs")
    _require({row["name"] for row in attrs} == set(INPUT_NAMES), "Unknown/duplicate/missing SDK input")
    bound = [_attr(row, logical[row["name"]], index) for index, row in enumerate(attrs)]
    shapes = {name: row["shape"] for name, row in logical.items()}
    net = shapes["net"]
    _require(len(net) == 3 and net[0] == 1 and shapes["inp"] == net,
             "net/inp must match fixed B1 [1,E,C]")
    e = net[1]
    _require(len(shapes["corr"]) == 3 and shapes["corr"][:2] == net[:2],
             "Correlation batch/edge shape differs")
    for name in ("previous", "following"):
        _require(shapes[name + "_indices"] == (e,) and shapes[name + "_valid"] == (1, e, 1),
                 "Neighbor contract shape differs")
    for name in ("kk", "ij"):
        shape = shapes[name + "_indices"]
        _require(len(shape) == 2 and shapes[name + "_valid"] == shape
                 and shapes[name + "_inverse"] == (e,), "Group contract shape differs")
    for row in bound:
        if row["name"] in INDEX_NAMES:
            row["index_limit_exclusive"] = (
                shapes[row["name"].split("_")[0] + "_indices"][0]
                if row["name"].endswith("_inverse") else e)
    return bound


@dataclass(frozen=True)
class PackedInputs:
    requests: object
    owned_buffers: tuple
    records: tuple


def pack_inputs(abi, arrays, bindings):
    """Validate original dtypes and make actual typed C-order owned buffers.

    SDK INT32 is permitted only for the exact bounded index domain and an
    INT64 round trip. No float index, saturating cast, pass-through or SDK
    implicit integer conversion is used. Caller must retain this object.
    """
    _require(len(bindings) == 13 and set(arrays) == set(INPUT_NAMES),
             "Expected exactly13 source arrays/bindings")
    _require({r["name"] for r in bindings} == set(INPUT_NAMES)
             and [r["index"] for r in bindings] == list(range(13)), "Invalid binding list")
    owned, records = [], []
    requests = (abi.Input * 13)()
    for row in bindings:
        name, index = row["name"], row["index"]
        value = arrays[name]
        _require(isinstance(value, np.ndarray) and value.dtype == np.dtype(row["logical_dtype"])
                 and value.shape == row["shape"], "Source dtype/shape differs: " + name)
        if name in INDEX_NAMES:
            _require(bool(((value >= 0) & (value < row["index_limit_exclusive"])).all()),
                     "Index outside original edge/group domain: " + name)
            if row["transport_dtype"] == "int32":
                _require(value.size == 0 or int(value.max()) <= INT32_MAX, "INT32 overflow")
        else:
            _require(bool(np.isfinite(value).all()), "Nonfinite source: " + name)
            if name in MASK_NAMES:
                _require(bool(((value == 0) | (value == 1)).all()), "Mask must be exact0/1")
        buffer = np.array(value, dtype=row["transport_dtype"], order="C", copy=True)
        if name in INDEX_NAMES and row["transport_dtype"] == "int32":
            _require(np.array_equal(buffer.astype(np.int64), value), "INT32 cast was not exact")
        _require(buffer.nbytes == row["transport_bytes"], "Packed bytes differ")
        owned.append(buffer)
        requests[index] = abi.Input(index, buffer.ctypes.data, buffer.nbytes, 0,
                                    row["transport_type"], row["transport_format"])
        records.append(dict(name=name, index=index, source_dtype=str(value.dtype),
                            transport_dtype=str(buffer.dtype), transport_type=row["transport_type"],
                            format=row["transport_format"], bytes=buffer.nbytes,
                            logical_shape=list(row["shape"]), owned=True,
                            exact_integer_cast_verified=name in INDEX_NAMES,
                            pass_through=0))
    return PackedInputs(requests, tuple(owned), tuple(records))


def validate_output_attrs(contracts, attrs):
    """Return exact bindings for the three float outputs with want_float=1.

    These byte sizes are returned FP32 payloads, not FP16 attr.size. The caller
    must copy each output while alive and release all outputs in finally.
    """
    logical = _contract(contracts, OUTPUT_NAMES)
    _require(len(attrs) == 3, "SDK must expose all3 outputs")
    _require({row["name"] for row in attrs} == set(OUTPUT_NAMES), "Unknown/duplicate/missing SDK output")
    result = [_attr(row, logical[row["name"]], index) for index, row in enumerate(attrs)]
    for row in result:
        row.update(want_float=1, is_prealloc=0, returned_FP32_bytes=math.prod(row["shape"]) * 4)
    return result
