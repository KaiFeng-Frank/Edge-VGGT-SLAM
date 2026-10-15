#!/usr/bin/env python3
"""Generated scalar/layout checks only; no model, OpenCL, RKNN or board calls."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

R, D, PAD = 1041, 1024, 1056
SENTINEL, OFFSET = -2147483647, 7


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def a8(row, column):
    return ((column // 8) * R + row) * 8 + column % 8


def c4(row, column):
    return ((column // 4) * R + row) * 4 + column % 4


def b16(k, n, size):
    return (((n // 16) * (size // 32) + k // 32) * 16 + n % 16) * 32 + k % 32


def reference_b(matrix):
    k, n = matrix.shape
    return matrix.reshape(k // 32, 32, n // 16, 16).transpose(2, 0, 3, 1).copy().ravel()


def guarded_array(size):
    return np.full(size + 2 * OFFSET, SENTINEL, np.int32)


def payload(array):
    assert np.all(array[:OFFSET] == SENTINEL) and np.all(array[-OFFSET:] == SENTINEL)
    return array[OFFSET:-OFFSET]


def two_pass_norm(x, gamma, beta):
    values = x.reshape(4, 256).T.copy()
    lane = np.zeros(256, np.float32)
    for item in range(4):
        lane = np.add(lane, values[:, item], dtype=np.float32)
    for step in (128, 64, 32, 16, 8, 4, 2, 1):
        lane[:step] = np.add(lane[:step], lane[step:2 * step], dtype=np.float32)
    mean = np.float32(lane[0] / np.float32(1024))
    lane = np.zeros(256, np.float32)
    for item in range(4):
        delta = np.subtract(values[:, item], mean, dtype=np.float32)
        lane = np.add(lane, np.multiply(delta, delta, dtype=np.float32), dtype=np.float32)
    for step in (128, 64, 32, 16, 8, 4, 2, 1):
        lane[:step] = np.add(lane[:step], lane[step:2 * step], dtype=np.float32)
    variance = np.float32(lane[0] / np.float32(1024))
    denominator = np.sqrt(np.add(variance, np.float32(1e-6), dtype=np.float32), dtype=np.float32)
    y = np.divide(np.subtract(x, mean, dtype=np.float32), denominator, dtype=np.float32)
    return np.add(np.multiply(y, gamma, dtype=np.float32), beta, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel', type=Path, default=Path(__file__).with_name('dino0_native_bridge.cl'))
    parser.add_argument('--kernel-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert sha(args.kernel) == args.kernel_sha256
    source = args.kernel.read_text()
    kernels = re.findall(r'__kernel\s+void\s+(\w+)\s*\(', source)
    assert kernels == ['norm1_pack', 'qkv_head_pack', 'pv_head_merge', 'proj_residual_norm', 'gelu_pack', 'final_residual']
    assert '#pragma OPENCL FP_CONTRACT OFF' in source and '1.0e-6f' in source
    assert 'erf(scaled)' in source and 'hidden / 0x1.6a09e6p+0f' in source
    assert not re.search(r'\b(?:native_sqrt|native_rsqrt|native_exp|native_erf|fma|mad)\s*\(', source)

    logical = np.arange(R * 3072, dtype=np.int32).reshape(R, 3072)
    bias = np.arange(3072, dtype=np.int32) * 7
    native = logical.reshape(R, 3072 // 4, 4).transpose(1, 0, 2).copy().ravel()
    i = np.arange(PAD * 64, dtype=np.int64)
    row, channel = (i // 4) % PAD, (i // 4 // PAD) * 4 + i % 4
    valid = row < R
    ia, ik, iv = a8(row[valid], channel[valid]), b16(channel, row, 64), b16(row, channel, PAD)
    assert np.array_equal(np.sort(ia), np.arange(R * 64))
    assert np.array_equal(np.sort(ik), np.arange(PAD * 64))
    assert np.array_equal(np.sort(iv), np.arange(PAD * 64))
    for head in range(16):
        columns = head * 64 + channel[valid]
        qa, kb, vb = guarded_array(R * 64), guarded_array(PAD * 64), guarded_array(PAD * 64)
        qa[OFFSET + ia] = native[c4(row[valid], columns)] + bias[columns]
        keys, values = np.zeros(PAD * 64, np.int32), np.zeros(PAD * 64, np.int32)
        keys[valid] = native[c4(row[valid], D + columns)] + bias[D + columns]
        values[valid] = native[c4(row[valid], 2 * D + columns)] + bias[2 * D + columns]
        kb[OFFSET + ik], vb[OFFSET + iv] = keys, values
        cs = slice(head * 64, (head + 1) * 64)
        expected_q = logical[:, cs] + bias[cs]
        expected_k = np.zeros((64, PAD), np.int32)
        expected_k[:, :R] = (logical[:, D + head * 64:D + (head + 1) * 64] + bias[D + head * 64:D + (head + 1) * 64]).T
        expected_v = np.zeros((PAD, 64), np.int32)
        expected_v[:R] = logical[:, 2 * D + head * 64:2 * D + (head + 1) * 64] + bias[2 * D + head * 64:2 * D + (head + 1) * 64]
        assert np.array_equal(payload(qa), expected_q.reshape(R, 8, 8).transpose(1, 0, 2).copy().ravel())
        assert np.array_equal(payload(kb), reference_b(expected_k))
        assert np.array_equal(payload(vb), reference_b(expected_v))
    del logical, native

    proj = guarded_array(R * D)
    semantic = np.empty((R, D), np.int32)
    j = np.arange(R * 64, dtype=np.int64)
    merge_rows, merge_channels = (j // 4) % R, (j // 4 // R) * 4 + j % 4
    for head in range(16):
        expected = head * 100000 + np.arange(R * 64, dtype=np.int32).reshape(R, 64)
        pv = expected.reshape(R, 16, 4).transpose(1, 0, 2).copy().ravel()
        proj[OFFSET + a8(merge_rows, head * 64 + merge_channels)] = pv
        semantic[:, head * 64:(head + 1) * 64] = expected
    assert np.array_equal(payload(proj).reshape(D // 8, R, 8).transpose(1, 0, 2).reshape(R, D), semantic)

    for channels in (1024, 4096):
        j = np.arange(R * channels, dtype=np.int64)
        rows, columns = (j // 8) % R, (j // 8 // R) * 8 + j % 8
        assert np.array_equal(np.sort(c4(rows, columns)), np.arange(R * channels))
        assert np.array_equal(a8(rows, columns), j)
    coords = np.arange(D, dtype=np.float32)
    gamma, beta = np.float32(.75) + coords / np.float32(8192), coords / np.float32(16384) - np.float32(.125)
    norm_errors = []
    for x in (np.full(D, .375, np.float32), (coords - np.float32(511.5)) / np.float32(256),
              np.float32(1) + (coords - np.float32(511.5)) / np.float32(1048576)):
        y = two_pass_norm(x, gamma, beta)
        high = x.astype(np.float64)
        mean = math.fsum(high) / D
        variance = math.fsum((high - mean) ** 2) / D
        expected = (high - mean) / math.sqrt(variance + float(np.float32(1e-6))) * gamma + beta
        error = float(np.max(np.abs(y.astype(np.float64) - expected)))
        assert np.allclose(y, expected, rtol=1e-4, atol=1e-5)
        assert np.isfinite(y.astype(np.float16)).all()
        norm_errors.append(error)
    hidden = np.array([-8, -3, -.1, 0, .1, 3, 8], np.float32)
    scaled = hidden / np.float32(float.fromhex('0x1.6a09e6p+0'))
    erf = np.array([math.erf(float(value)) for value in scaled], np.float32)
    gelu = np.float32(.5) * np.multiply(hidden, np.add(np.float32(1), erf, dtype=np.float32), dtype=np.float32)
    expected = np.array([.5 * float(value) * (1 + math.erf(float(value) / math.sqrt(2))) for value in hidden])
    assert np.allclose(gelu, expected, rtol=1e-4, atol=1e-5)
    x, c, b, ls = np.float32(.75), np.float32(-.5), np.float32(.125), np.float32(.25)
    r1 = np.float32(x + np.float32(np.float32(c + b) * ls))
    assert r1 == np.float32(.65625)
    final = np.float32(r1 + np.float32(np.float32(c + b) * np.float32(-.5)))
    assert final == np.float32(.84375)
    report = dict(schema='v121_original_DINO0_bridge_generated_source_layout_scalar_check', status='passed_generated_checks_only',
                  source_sha256=sha(__file__), kernel_sha256=sha(args.kernel),
                  all_1041_rows_and_16_heads_layout_checked=True, padded_15_keys_and_offsets_checked=True,
                  original_bias_layerscale_residual_order_checked=True, generated_LayerNorm_max_abs_errors=norm_errors,
                  generated_Erf_max_abs_error=float(np.max(np.abs(gelu - expected))),
                  OpenCL_or_RKNN_or_model_executed=False, OpenCL_source_compiled=False,
                  actual_model_precision_or_performance_accepted=False)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
