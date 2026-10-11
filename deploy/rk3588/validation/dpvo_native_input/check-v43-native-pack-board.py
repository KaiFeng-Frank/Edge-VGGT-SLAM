#!/usr/bin/env python3
"""Conversion-only generated/saved-feed check. No SDK, model or compiler calls."""
import argparse
import ctypes as ct
import errno
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

PACKER_SHA256 = 'b3d2503d1bb27ec0352cfe54d759efdcee070764ad266841847005fa9b13f8ce'
QUERY_SHA256 = '89f0cead99df024b25ed3d2638e412a075f91d6736ecdb2b35ace2304eef0642'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned(path, wanted):
    actual = digest(path)
    if actual != wanted:
        raise ValueError('Source/input SHA256 differs: ' + Path(path).name)
    return actual


def load_packer(path, wanted):
    pinned(path, wanted)
    spec = importlib.util.spec_from_file_location('_v43_conversion_only_pack', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def native_catalog(query):
    if (query.get('schema') != 'v42_nine_shape_logical_and_native_input_query'
            or query.get('status') != 'queried_all_nine_shapes_inputs_not_inference_or_transport_validated'
            or query.get('sdk_run_calls') != 0 or query.get('model_inference_executed') is not False):
        raise ValueError('Frozen actual query metadata differs')
    catalog = {}
    for row in query['models']:
        attrs = {a['input_index']: a['attribute'] for a in row['queries']
                 if a['command'] == 8 and a['query_succeeded'] and a['return_code'] == 0}
        other = {a['input_index']: a['attribute'] for a in row['queries']
                 if a['command'] == 10 and a['query_succeeded'] and a['return_code'] == 0}
        if set(attrs) != {0, 1, 2} or attrs != other:
            raise ValueError('Actual cmd8/cmd10 contract differs')
        catalog[row['model_key']] = dict(
            shapes={a['name']: tuple(a['shape']) for a in row['logical_inputs']},
            attrs={a['name']: a for a in attrs.values()}, model_sha256=row['model']['sha256'],
            compile_sha256=row['compile_sha256'])
    if len(catalog) != 9:
        raise ValueError('Nine queried shapes required')
    return catalog


def reference_bits(value):
    """Untimed NumPy reference only; production pack never uses astype."""
    B, C, D, M = value.shape
    blocks = (C + 7) // 8
    with np.errstate(over='ignore', invalid='ignore'):
        half = value.astype(np.float16)
    if not np.isfinite(half).all():
        raise ValueError('Reference half overflow')
    padded = np.zeros((B, blocks * 8, D, M), dtype=np.uint16)
    padded[:, :C] = half.view(np.uint16)
    return padded.reshape(B, blocks, 8, D, M).transpose(0, 1, 3, 4, 2).copy()


def finite_boundary_values():
    half_tiny = np.float32(2.0 ** -25)
    return np.array([
        0.0, -0.0, 2.0 ** -24, -(2.0 ** -24), half_tiny, -half_tiny,
        np.nextafter(half_tiny, np.float32(np.inf)),
        np.nextafter(half_tiny, np.float32(0)), 3.0 * (2.0 ** -25),
        2.0 ** -14, 2.0 ** -14 - 2.0 ** -24,
        1.0 + 2.0 ** -11, 1.0 + 3.0 * (2.0 ** -11),
        -(1.0 + 2.0 ** -11), 65504.0, -65504.0, 65519.0,
        np.nextafter(np.float32(65520), np.float32(0)),
    ], dtype=np.float32)


def check_boundary_bits(packer, library, contract):
    values = finite_boundary_values()
    names = ['positive_zero', 'negative_zero', 'min_half_subnormal', 'negative_min_half_subnormal',
             'half_min_subnormal_tie_to_even_zero', 'negative_half_min_subnormal_tie_to_even_zero',
             'above_half_min_subnormal', 'below_half_min_subnormal', 'subnormal_tie_to_even_two',
             'min_half_normal', 'max_half_subnormal', 'normal_tie_to_even_down', 'normal_tie_to_even_up',
             'negative_normal_tie_to_even', 'max_half_finite', 'negative_max_half_finite',
             'above_max_half_finite_rounds_finite', 'below_half_overflow_tie_rounds_finite']
    expected = np.array([0x0000, 0x8000, 0x0001, 0x8001, 0x0000, 0x8000,
                         0x0001, 0x0000, 0x0002, 0x0400, 0x03ff,
                         0x3c00, 0x3c02, 0xbc00, 0x7bff, 0xfbff, 0x7bff, 0x7bff], dtype=np.uint16)
    if not np.array_equal(values.astype(np.float16).view(np.uint16), expected):
        raise AssertionError('NumPy reference does not match explicit IEEE RN boundary bits')
    value = np.zeros(contract['shapes']['net'], dtype=np.float32)
    value.reshape(-1)[:len(values)] = values
    owned = packer.pack_blocked_fp16(value, contract['attrs']['net'], library)
    actual = owned[0, 0, :, :, 0].reshape(-1)[:len(values)]
    if not np.array_equal(actual, expected):
        raise AssertionError('Native conversion IEEE RN boundary bits differ')
    return [dict(case=name, source_fp32_bits=f'0x{int(bits):08x}',
                 expected_fp16_bits=f'0x{int(want):04x}', actual_fp16_bits=f'0x{int(got):04x}', passed=True)
            for name, bits, want, got in zip(names, values.view(np.uint32), expected, actual)]


def check_feed(packer, library, feed, contract, repeat):
    records, converted = [], {}
    if set(feed) != {'net', 'inp', 'corr'}:
        raise ValueError('Saved feed must contain exactly net/inp/corr CF arrays')
    for name in ('net', 'inp', 'corr'):
        value = feed[name]
        if value.shape != contract['shapes'][name]:
            raise ValueError('Saved CF shape differs: ' + name)
        before = hashlib.sha256(value.tobytes()).hexdigest()
        expected = reference_bits(value)  # Never included in pack timing.
        samples = []
        for _ in range(repeat):
            start = time.perf_counter()
            owned = packer.pack_blocked_fp16(value, contract['attrs'][name], library)
            samples.append(time.perf_counter() - start)
        if (owned.dtype != np.uint16 or not owned.flags.owndata
                or not owned.flags.c_contiguous or np.shares_memory(owned, value)):
            raise AssertionError('Output ownership/dtype/contiguity differs')
        different = int(np.count_nonzero(owned != expected))
        if different or before != hashlib.sha256(value.tobytes()).hexdigest():
            raise AssertionError('Conversion bits differ or input changed')
        tail_zero = bool(np.all(owned[:, -1, :, :, 2:] == 0)) if name == 'corr' else None
        if name == 'corr' and not tail_zero:
            raise AssertionError('Six corr padding channels must be +0 bits')
        records.append(dict(name=name, CF_shape=list(value.shape), native_shape=list(owned.shape),
            source_dtype='float32', output_dtype='uint16_IEEE_binary16_bits',
            native_bytes=owned.nbytes, different_reference_bits=different,
            corr_six_tail_channels_positive_zero=tail_zero, owns_buffer=True,
            input_unchanged=True, input_alias=False, source_array_sha256=before,
            output_array_sha256=hashlib.sha256(owned.tobytes()).hexdigest(),
            complete_pack_seconds=samples,
            timing_includes='attribute validation, allocation, FPCR check, finite/half-overflow validation, tile transpose and conversion',
            timing_excludes='input generation/loading, NumPy reference, output hashing/comparison'))
        converted[name] = owned
    return records, converted


def check_rejections(packer, library, contract):
    attrs = contract['attrs']['net']
    shape = contract['shapes']['net']
    results = []
    for label, bad, wanted in [('positive_half_overflow_tie', 65520.0, errno.ERANGE),
                               ('negative_half_overflow_tie', -65520.0, errno.ERANGE),
                               ('huge_finite', np.finfo(np.float32).max, errno.ERANGE),
                               ('NaN', np.nan, errno.EDOM), ('positive_Inf', np.inf, errno.EDOM),
                               ('negative_Inf', -np.inf, errno.EDOM)]:
        value = np.zeros(shape, dtype=np.float32)
        value.reshape(-1)[-1] = bad  # Exercise a late invalid tile.
        before = value.tobytes()
        try:
            packer.pack_blocked_fp16(value, attrs, library)
        except OSError as error:
            if error.errno != wanted:
                raise AssertionError('Wrong numeric rejection errno') from error
        else:
            raise AssertionError('Invalid numeric input accepted: ' + label)
        if value.tobytes() != before:
            raise AssertionError('Numeric rejection changed input')
        results.append(dict(case=label, errno=wanted, rejected=True, input_unchanged=True))
    return results


def check_c_capacity_and_alias(packer, library, contract):
    value = np.zeros(contract['shapes']['net'], dtype=np.float32)
    shape = packer.validate_native_attribute(value, contract['attrs']['net'])
    count = int(np.prod(shape))
    guarded = np.full(count + 64, 0xa55a, dtype=np.uint16)
    output = guarded[32:-32]
    source = value.ctypes.data_as(ct.POINTER(ct.c_float))
    destination = output.ctypes.data_as(ct.POINTER(ct.c_uint16))
    def call(src_elements=value.size, capacity=output.nbytes, dst=destination, dims=value.shape):
        return int(library.pack(source, src_elements, dst, capacity, *dims))
    checks = [('capacity_minus_two_bytes', call(capacity=output.nbytes - 2), errno.ENOSPC),
              ('source_elements_minus_one', call(src_elements=value.size - 1), errno.EINVAL),
              ('unsupported_B', call(dims=(3,) + value.shape[1:]), errno.EINVAL),
              ('input_output_alias', call(dst=value.ctypes.data_as(ct.POINTER(ct.c_uint16))), errno.EINVAL)]
    if not np.all(guarded == 0xa55a) or not np.all(value == 0):
        raise AssertionError('Rejected bounds/alias call wrote memory')
    for label, code, wanted in checks:
        if code != wanted:
            raise AssertionError('C ABI rejection differs: ' + label)
    if call() or not np.all(output == 0) or not np.all(guarded[:32] == 0xa55a) or not np.all(guarded[-32:] == 0xa55a):
        raise AssertionError('C ABI wrote outside exact output bounds')
    return [dict(case=label, errno=code, rejected=True) for label, code, _ in checks] + [dict(case='success_exact_destination_guard', passed=True)]


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--library', required=True)
    result.add_argument('--library-sha256', required=True)
    result.add_argument('--packer', default=str(Path(__file__).with_name('v43_native_fp16_pack.py')))
    result.add_argument('--packer-sha256', default=PACKER_SHA256)
    result.add_argument('--query-json', required=True)
    result.add_argument('--query-sha256', default=QUERY_SHA256)
    result.add_argument('--model-key', default='B2_D6_M16')
    result.add_argument('--all-queried-shapes', action='store_true')
    result.add_argument('--repeat', type=int, default=3)
    result.add_argument('--feed', help='Saved grouped CF FP32 NPZ: exactly net/inp/corr')
    result.add_argument('--feed-sha256')
    result.add_argument('--output-dir', required=True, help='New directory; never overwrites an existing result')
    return result


def main():
    args = parser().parse_args()
    if args.repeat < 1 or args.repeat > 100:
        raise ValueError('Repeat must be within 1..100')
    if bool(args.feed) != bool(args.feed_sha256):
        raise ValueError('Saved feed path and SHA256 must be supplied together')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    record = dict(schema='v43_native_pack_conversion_only', status='started', candidate_accepted=False,
        script_sha256=digest(__file__), packer_sha256=args.packer_sha256,
        library_sha256=args.library_sha256, query_sha256=args.query_sha256,
        source_query_was_actual_board=True, model_inference_executed=False,
        sdk_loaded=False, sdk_run_calls=0, NPU_used=False,
        SDK_FP32_equivalence_validated=False, SDK_native_transport_equivalence_validated=False,
        FPCR_modified=False, generated=[], saved_feed=None, cleanup_errors=[])
    try:
        packer = load_packer(args.packer, args.packer_sha256)
        pinned(args.query_json, args.query_sha256)
        catalog = native_catalog(json.loads(Path(args.query_json).read_text()))
        if args.model_key not in catalog:
            raise ValueError('Model key was not queried')
        library = packer.load_native_library(args.library, args.library_sha256)
        record['abi_version'] = library.version()
        record['FPCR_before'] = packer.fpcr_record(library)
        if not record['FPCR_before']['supported']:
            raise OSError(errno.ENOTSUP, 'Unsupported FPCR: conversion not run')
        rng = np.random.default_rng(4301)
        keys = sorted(catalog) if args.all_queried_shapes else [args.model_key]
        for key in keys:
            contract = catalog[key]
            feed = {name: rng.standard_normal(shape, dtype=np.float32)
                    for name, shape in contract['shapes'].items()}
            feed['net'].reshape(-1)[:len(finite_boundary_values())] = finite_boundary_values()
            rows, _ = check_feed(packer, library, feed, contract, args.repeat)
            record['generated'].append(dict(model_key=key, seed=4301, model_sha256=contract['model_sha256'], inputs=rows))
        contract = catalog[args.model_key]
        record['finite_boundary_cases'] = check_boundary_bits(packer, library, contract)
        record['numeric_rejections'] = check_rejections(packer, library, contract)
        record['C_ABI_capacity_and_alias'] = check_c_capacity_and_alias(packer, library, contract)
        if args.feed:
            pinned(args.feed, args.feed_sha256)
            with np.load(args.feed, allow_pickle=False) as archive:
                feed = {name: np.array(archive[name], copy=True, order='C') for name in archive.files}
            rows, converted = check_feed(packer, library, feed, contract, args.repeat)
            target = output / 'saved-feed-native-blocked-fp16.npz'
            np.savez_compressed(target, **converted)
            record['saved_feed'] = dict(model_key=args.model_key, source_npz_sha256=args.feed_sha256,
                inputs=rows, native_npz=dict(file=target.name, bytes=target.stat().st_size, sha256=digest(target)),
                relation_to_original_Torch='No teacher or network output used; conversion reference only')
        record['FPCR_after'] = packer.fpcr_record(library)
        if record['FPCR_after']['raw'] != record['FPCR_before']['raw']:
            raise AssertionError('Caller FPCR changed')
        record['status'] = 'passed_conversion_reference_and_ownership_only'
    except Exception as error:
        record['status'] = 'failed_conversion_only'
        record['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        (output / 'conversion.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=record['status'], generated_shapes=len(record['generated']), SDK_run_calls=0)))


if __name__ == '__main__':
    main()
