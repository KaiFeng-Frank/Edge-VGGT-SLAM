#!/usr/bin/env python3
"""Read saved conversion JSON/arrays only. Never load a native library or SDK."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

PINS = dict(
    conversion='94c4cbd4adc458e34109d9cfa003112890069509616790fe5d7b8a0909883ee8',
    query='89f0cead99df024b25ed3d2638e412a075f91d6736ecdb2b35ace2304eef0642',
    driver='4ecb01401367c9d3e13431d39e94fc6f86b226387ab2894114b9fe0c5f92f4c3',
    packer='b3d2503d1bb27ec0352cfe54d759efdcee070764ad266841847005fa9b13f8ce',
    library='5de153a44839915a76d68e7cb3141e31c46ae6a3dc83e8b32273bc9ecd3d0c6a',
    feed='cc0c6d17e7eba6215491ccf8dca639bb5b7a78d536b5ae409e58e89b26d5941b',
    feed_receipt='ae0cbf764d982d4a4321d07a83109fd2113fdbe29d6ab99c7c9cb89c46551a0f')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def read_pinned(path, expected):
    require(sha(path) == expected, 'Saved SHA256 differs: ' + Path(path).name)
    return json.loads(Path(path).read_text())


def audit(args):
    result = read_pinned(args.conversion, PINS['conversion'])
    query = read_pinned(args.query_json, PINS['query'])
    original = read_pinned(args.feed_receipt, PINS['feed_receipt'])
    build = json.loads(Path(args.build).read_text())
    require(result['status'] == 'passed_conversion_reference_and_ownership_only', 'Conversion status differs')
    require(result['script_sha256'] == PINS['driver'] and result['packer_sha256'] == PINS['packer']
            and result['library_sha256'] == PINS['library'] and result['query_sha256'] == PINS['query'], 'Source pins differ')
    require(build['output']['sha256'] == PINS['library'] and build['exit_code'] == 0 and build['ABI'] == 1
            and build['sources_unchanged'] is True and build['source_pins'] == build['source_hashes_after'], 'Saved build lineage differs')
    require(build['source_sha256']['v43_native_fp16_pack.py'] == PINS['packer'], 'Build wrapper pin differs')
    require(result['abi_version'] == 1 and result['FPCR_before'] == result['FPCR_after']
            and result['FPCR_before']['raw'] == 0 and result['FPCR_before']['supported'] is True
            and result['FPCR_modified'] is False, 'Reported FPCR state differs')
    require(result['sdk_loaded'] is False and result['sdk_run_calls'] == 0 and result['NPU_used'] is False
            and result['model_inference_executed'] is False and result['candidate_accepted'] is False
            and result['SDK_FP32_equivalence_validated'] is False
            and result['SDK_native_transport_equivalence_validated'] is False and result['cleanup_errors'] == [], 'Conversion-only scope differs')
    models = {row['model_key']: row for row in query['models']}
    require(len(models) == len(result['generated']) == 9
            and {row['model_key'] for row in result['generated']} == set(models), 'Queried generated model set differs')
    for group in result['generated']:
        model = models[group['model_key']]
        require(group['model_sha256'] == model['model']['sha256'], 'Generated model SHA differs')
        require({row['name'] for row in group['inputs']} == {'net', 'inp', 'corr'}, 'Generated names differ')
        shapes = {row['name']: row['shape'] for row in model['logical_inputs']}
        attrs = {row['attribute']['name']: row['attribute'] for row in model['queries'] if row['command'] == 8}
        for row in group['inputs']:
            require(row['CF_shape'] == shapes[row['name']] and row['native_shape'] == attrs[row['name']]['shape']
                    and row['native_bytes'] == attrs[row['name']]['size_with_stride'], 'Generated shape/bytes differs')
            require(row['different_reference_bits'] == 0 and row['owns_buffer'] is True and row['input_unchanged'] is True
                    and row['input_alias'] is False, 'Generated conversion reported failure')
            require(all(math.isfinite(v) and v > 0 for v in row['complete_pack_seconds']), 'Pack clock is invalid')
    expected_bits = ['0x0000', '0x8000', '0x0001', '0x8001', '0x0000', '0x8000',
                     '0x0001', '0x0000', '0x0002', '0x0400', '0x03ff', '0x3c00',
                     '0x3c02', '0xbc00', '0x7bff', '0xfbff', '0x7bff', '0x7bff']
    require([row['expected_fp16_bits'] for row in result['finite_boundary_cases']] == expected_bits
            and all(row['actual_fp16_bits'] == row['expected_fp16_bits'] and row['passed']
                    for row in result['finite_boundary_cases']), 'Reported IEEE RN boundary bits differ')
    require([row['errno'] for row in result['numeric_rejections']] == [34, 34, 34, 33, 33, 33]
            and all(row['rejected'] and row['input_unchanged'] for row in result['numeric_rejections']), 'Reported numeric rejections differ')
    require([row.get('errno') for row in result['C_ABI_capacity_and_alias']] == [28, 22, 22, 22, None]
            and result['C_ABI_capacity_and_alias'][-1]['passed'] is True, 'Reported C guards differ')
    saved = result['saved_feed']
    require(sha(args.feed) == PINS['feed'] == saved['source_npz_sha256'] == original['output']['sha256'], 'Original CF feed pin differs')
    descriptor = saved['native_npz']
    native_path = Path(args.conversion).parent / descriptor['file']
    require(sha(native_path) == descriptor['sha256'] and native_path.stat().st_size == descriptor['bytes'], 'Native saved NPZ differs')
    verified = []
    with np.load(args.feed, allow_pickle=False) as sources, np.load(native_path, allow_pickle=False) as natives:
        require(set(sources.files) == set(natives.files) == {'net', 'inp', 'corr'}, 'Saved input/output names differ')
        for row in saved['inputs']:
            name = row['name']
            source, native = sources[name], natives[name]
            require(source.dtype == np.float32 and native.dtype == np.uint16, 'Saved dtype differs')
            require(list(source.shape) == row['CF_shape'] and list(native.shape) == row['native_shape']
                    and native.nbytes == row['native_bytes'], 'Saved shape/bytes differs')
            require(tensor_sha(source) == row['source_array_sha256'] == original['output']['arrays'][name]['tensor_sha256']
                    and tensor_sha(native) == row['output_array_sha256'], 'Saved array SHA differs')
            require(np.isfinite(source).all() and np.isfinite(native.view(np.float16)).all(), 'Saved source/half has nonfinite elements')
            B, C, D, M = source.shape
            dense = native.transpose(0, 1, 4, 2, 3).reshape(B, ((C + 7) // 8) * 8, D, M)
            expected = source.astype(np.float16).view(np.uint16)
            different = int(np.count_nonzero(dense[:, :C] != expected))
            tail = dense[:, C:]
            require(different == 0 and np.all(tail == 0), 'Actual saved native bits/padding differ')
            verified.append(dict(name=name, CF_shape=list(source.shape), native_shape=list(native.shape),
                native_bytes=native.nbytes, real_channel_elements=source.size,
                different_NumPy_reference_bits=different, padding_elements=tail.size,
                all_padding_positive_zero=True, source_and_native_finite=True,
                source_array_sha256=tensor_sha(source), native_array_sha256=tensor_sha(native),
                actual_saved_arrays_independently_checked=True))
    samples = [sum(row['complete_pack_seconds'][i] for row in saved['inputs']) * 1000
               for i in range(len(saved['inputs'][0]['complete_pack_seconds']))]
    return dict(schema='v43_saved_actual_native_conversion_independent_review',
        status='passed_saved_native_array_bits_and_conversion_metadata', candidate_accepted=False,
        audit_native_or_SDK_or_model_execution=False, original_result_status_preserved=result['status'],
        source_conversion_sha256=PINS['conversion'], source_build_sha256=sha(args.build),
        source_query_sha256=PINS['query'], source_feed_sha256=PINS['feed'],
        source_feed_receipt_sha256=PINS['feed_receipt'], native_NPZ_sha256=descriptor['sha256'],
        driver_sha256=PINS['driver'], packer_sha256=PINS['packer'], library_sha256=PINS['library'],
        compiler_sha256_reported=build['compiler_sha256'], build_and_library_audit_scope='Saved build metadata only; no compiler or native library rerun',
        actual_saved_CF_and_native_arrays=verified,
        generated_shapes_reported_passed=9, generated_input_records_checked=27,
        generated_arrays_independently_replayed=False,
        finite_boundary_bits_reported_checked=18, numeric_rejections_reported_checked=6,
        C_capacity_alias_guard_records_reported_checked=5,
        FPCR_before_after_reported=0, FPCR_modified_reported=False,
        ownership_and_input_preservation_scope='Actual board report plus frozen-source/fake protocol; NPZ alone cannot prove pointer lifetime',
        saved_B2D6_sum_of_three_full_pack_times_ms=samples,
        saved_B2D6_sum_of_three_full_pack_mean_ms=sum(samples) / len(samples),
        timing_scope='Sum of separately measured three complete pack calls; validation/allocation/conversion included; not SDK or full Update timing',
        original_feedback_source=dict(run_sha256=original['source_run']['sha256'], frame_index=34,
            original_status=original['source_run']['original_status'], source_slots=original['extraction']['selected_group']['source_slots']),
        SDK_native_submission_or_network_equivalence_validated=False,
        original_Torch_equivalence_validated=False, model_or_GT_used=False, adoption_or_20Hz_claim=False,
        limitations='Generated arrays and rejection/ownership/FPCR facts are reported board metadata. Only the three actual saved native arrays were independently compared to the saved FP32 feed here.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('conversion', 'build', 'feed', 'feed-receipt', 'query-json', 'output'):
        parser.add_argument('--' + flag, required=True)
    args = parser.parse_args()
    target = Path(args.output)
    require(not target.exists(), 'Audit output already exists')
    result = audit(args)
    result['audit_script_sha256'] = sha(__file__)
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], independently_checked_arrays=3,
                         summed_pack_mean_ms=result['saved_B2D6_sum_of_three_full_pack_mean_ms'])))


if __name__ == '__main__':
    main()
