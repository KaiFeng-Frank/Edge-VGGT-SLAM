#!/usr/bin/env python3
"""Saved-only single-group SDK A/B audit; no executor, SDK or native imports."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

RUN_SHA = '28185202278d0446b62525721338345ce279a88773969d45e9efb8f3f9fbcba2'
FEED_SHA = 'cc0c6d17e7eba6215491ccf8dca639bb5b7a78d536b5ae409e58e89b26d5941b'
QUERY_SHA = '89f0cead99df024b25ed3d2638e412a075f91d6736ecdb2b35ace2304eef0642'
CONVERSION_SHA = '94c4cbd4adc458e34109d9cfa003112890069509616790fe5d7b8a0909883ee8'
TIMING_FIELDS = ('pack_seconds', 'input_set_seconds', 'run_api_seconds', 'outputs_get_seconds',
                 'owned_copy_seconds', 'outputs_release_seconds', 'forward_seconds')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_meta(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes,
                tensor_sha256=hashlib.sha256(value.tobytes()).hexdigest())


def audit(args):
    require(sha(args.run) == RUN_SHA and sha(args.feed) == FEED_SHA
            and sha(args.query_json) == QUERY_SHA, 'Fixed saved input pin differs')
    run = json.loads(Path(args.run).read_text())
    query = json.loads(Path(args.query_json).read_text())
    require(run['schema'] == 'v43_native_single_B2D6_saved_state_ABBA'
            and run['status'] == 'completed_single_B2D6_AB_bitwise_equal_diagnostic_not_adopted'
            and run['model_key'] == 'B2_D6_M16', 'Saved single-group schema/status differs')
    require(run['candidate_accepted'] is False and run['original_strict_FP32_rejected'] is True
            and run['strict_FP32_thresholds_relaxed'] is False and run['recurrent_state_committed'] is False
            and run['Torch_ORT_BA_executed'] is False and run['zero_copy_claimed'] is False, 'Research boundary differs')
    require(run['source_pins'] == run['source_hashes_after'] and run['sources_unchanged'] is True
            and run['cleanup_errors'] == [] and run['affinity_restored'] is True, 'Source/cleanup metadata differs')
    require(run['feed_sha256'] == FEED_SHA and run['conversion_receipt_sha256'] == CONVERSION_SHA, 'Feed/conversion lineage differs')
    model = next(row for row in query['models'] if row['model_key'] == run['model_key'])
    spec = run['validated_model_spec']
    require(spec['model_sha256'] == model['model']['sha256'] and spec['compile_sha256'] == model['compile_sha256'], 'Model/compile lineage differs')
    with np.load(args.feed, allow_pickle=False) as feed:
        require(set(feed.files) == set(run['input_arrays']) == {'net', 'inp', 'corr'}, 'Feed array set differs')
        for name in feed.files:
            require(feed[name].dtype == np.float32 and np.isfinite(feed[name]).all()
                    and tensor_meta(feed[name]) == run['input_arrays'][name], 'Feed array metadata differs')
    initial, closed = run['initial_contexts'], run['closed_contexts']
    require(set(initial) == set(closed) == {'A', 'B'}, 'Exactly two contexts required')
    for method in ('A', 'B'):
        before, after = initial[method], closed[method]
        require(before['closed'] is False and before['sdk_run_api_calls'] == 0
                and after['closed'] is True and after['context_destroyed'] is True
                and after['sdk_run_api_calls'] == after['completed_calls'] == 4
                and after['SDK_outputs_release_attempts'] == after['SDK_outputs_releases'] == 4
                and after['cleanup_errors'] == [] and after['current_DMA_bytes'] == 0
                and after['current_model_host_bytes'] == 0 and after['npu_used'] is True, 'Context lifecycle metadata differs')
        require(after['source_pins'] == after['source_hashes_after'] and after['sources_and_inputs_unchanged'] is True
                and before['core_mask'] == after['core_mask'] == 7 and before['init_flags'] == after['init_flags'] == 0
                and before['collect_perf_detail'] is after['collect_perf_detail'] is False, 'Context source/core/profile metadata differs')
    require(initial['A']['SDK_version'] == initial['B']['SDK_version'], 'SDK versions differ')
    native = initial['B']['native_input_queries']
    require(len(native) == 6 and {(r['command'], r['input_index']) for r in native}
            == {(command, index) for command in (8, 10) for index in range(3)}, 'Six native queries required')
    expected_native = {(r['command'], r['input_index']): r['attribute'] for r in model['queries'] if r['command'] in (8, 10)}
    for row in native:
        require(row['return_code'] == 0, 'Native query failed')
        wanted = expected_native[(row['command'], row['input_index'])]
        require(all(value == wanted[key] for key, value in row['attribute'].items()), 'Actual native attribute differs from frozen query')
    require(initial['B']['packing_FPCR_at_initialization']['raw'] == 0
            and initial['B']['packing_FPCR_at_initialization']['supported'] is True, 'Initial FPCR differs')
    calls = run['calls']
    require(len(calls) == 8 and [r['method'] for r in calls] == list('ABBAABBA')
            and [r['order'] for r in calls] == list(range(8)), 'ABBA call order differs')
    baseline, outputs = None, []
    expected_outputs = {row['name']: tuple(row['shape']) for row in spec['outputs']}
    for order, call in enumerate(calls):
        method, record = call['method'], call['record']
        descriptor = call['output_artifact']
        path = Path(args.run).parent / descriptor['file']
        require(sha(path) == descriptor['sha256'] and path.stat().st_size == descriptor['bytes'], 'Saved output NPZ pin differs')
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == set(descriptor['arrays']) == set(expected_outputs), 'Output array set differs')
            values = {name: archive[name].copy() for name in archive.files}
        if baseline is None:
            baseline = values
        checked = {}
        for name, value in values.items():
            require(value.dtype == np.float32 and value.shape == expected_outputs[name]
                    and np.isfinite(value).all() and tensor_meta(value) == descriptor['arrays'][name], 'Saved output array metadata differs')
            different = int(np.count_nonzero(value.view(np.uint32) != baseline[name].view(np.uint32)))
            require(different == 0, 'Actual saved SDK A/B bits differ')
            checked[name] = dict(**tensor_meta(value), finite=True, different_first_A_bits=different)
        require(np.all((values['weight'] >= 0) & (values['weight'] <= 1))
                and any(np.any(value) for value in values.values()), 'Output weight/domain/nonzero failure')
        require(record['execution_status'] == 'completed' and record['mode'] == 'normal'
                and record['outputs_released'] is True and record['strict_FP32_gate_passed'] is None
                and record['strict_component_gate_passed'] is False
                and record['native_log']['RKNN_error_detected'] is False
                and record['native_log']['truncated'] is False, 'SDK call status/gate/release/log differs')
        timing = record['timing']
        require(timing['synchronous'] is True and all(math.isfinite(timing[k]) and timing[k] > 0 for k in TIMING_FIELDS), 'Timing scope differs')
        require(sum(timing[k] for k in TIMING_FIELDS[:-1]) <= timing['forward_seconds'] + 1e-9
                and timing['forward_seconds'] <= record['total_execute_seconds'] + 1e-9
                and record['total_execute_seconds'] <= call['actual_execute_wall_seconds'] + 1e-9, 'Nested timing levels differ')
        for index, item in enumerate(timing['input_records']):
            name = ('net', 'inp', 'corr')[index]
            require(item['index'] == index and item['name'] == name and item['owned'] is True
                    and item['logical_shape'] == run['input_arrays'][name]['shape'], 'Input ownership/name/CF shape differs')
            if method == 'B':
                attr = expected_native[(8, index)]
                require(item['pass_through'] == 1 and item['request_type'] == 1 and item['request_format'] == 2
                        and item['native_attribute_format'] == 2 and item['native_query_commands'] == [8, 10]
                        and item['bytes'] == attr['size_with_stride'] and item['buffer_shape'] == attr['shape']
                        and item['transport_dtype'] == 'float16' and item['buffer_storage_dtype'] == 'uint16'
                        and item['SDK_input_conversion_requested'] is False and item['padded_channels'] == (6 if name == 'corr' else 0), 'Native pass1/type1/fmt2 buffer record differs')
            else:
                require(item['pass_through'] == 0 and item['request_format'] == 1
                        and item['transport_dtype'] == 'float32' and item['bytes'] == run['input_arrays'][name]['bytes'], 'Original FP32 pass0 buffer differs')
        require(len(timing['input_records']) == 3, 'Exactly three pack records required')
        outputs.append(dict(order=order, method=method, file=descriptor['file'], file_sha256=descriptor['sha256'], arrays=checked))
    stats = {}
    for method in ('A', 'B'):
        own_calls = [row for row in calls if row['method'] == method]
        require([row['state_method_repeat_index'] for row in own_calls] == list(range(4))
                and [row['excluded_first_state_method_call'] for row in own_calls] == [True, False, False, False], 'Warm exclusion differs')
        warm = own_calls[1:]
        means = {key: sum(row['record']['timing'][key] for row in warm) / len(warm) for key in TIMING_FIELDS}
        means['actual_execute_wall_mean_seconds'] = sum(row['actual_execute_wall_seconds'] for row in warm) / len(warm)
        require(run['warm_statistics'][method]['n'] == 3
                and all(abs(means[key] - run['warm_statistics'][method][key]) < 1e-12 for key in means), 'Warm means differ')
        stats[method] = {key: value * 1000 for key, value in means.items()}
    owner = run['factory_owner_after_close']
    require(owner['closed'] is True and owner['requests'] == 0 and owner['cache']['closed'] is True
            and owner['cache']['created'] == owner['cache']['resident_contexts'] == 0
            and owner['cache']['cleanup_errors'] == [], 'Factory cleanup differs')
    A = stats['A']['actual_execute_wall_mean_seconds']
    B = stats['B']['actual_execute_wall_mean_seconds']
    return dict(schema='v43_saved_single_native_SDK_AB_independent_review', status='passed_saved_eight_SDK_calls_24_arrays_bitwise_same_feed',
        candidate_accepted=False, audit_native_or_SDK_or_model_execution=False, source_run_sha256=RUN_SHA,
        original_run_status_preserved=run['status'], feed_sha256=FEED_SHA, query_sha256=QUERY_SHA,
        conversion_receipt_sha256=CONVERSION_SHA, model_key=run['model_key'],
        model_sha256=spec['model_sha256'], compile_sha256=spec['compile_sha256'],
        actual_saved_files_verified=8, actual_saved_arrays_verified=24, all_saved_outputs_bitwise_equal_first_A=True,
        outputs=outputs, actual_native_query_attrs_checked=6, native_pack_records_checked=12,
        warm_means_recomputed_ms=stats, wall_reduction_ms=A-B, wall_reduction_fraction=(A-B)/A,
        sources_before_after_reported_stable=True, top_level_source_pins_checked=len(run['source_pins']),
        context_source_pins_checked={method: len(closed[method]['source_pins']) for method in ('A', 'B')},
        lifecycle_metadata=dict(contexts_destroyed=2, SDK_runs=8, completed_calls=8, releases_attempted=8, releases_succeeded=8,
            cleanup_errors=[], current_DMA_and_model_bytes_zero=True, contexts_actual_operator_coverage_not_audited=True),
        output_ownership_scope='Frozen transport retains input buffers through release and owns copied outputs; saved metadata and arrays checked, no pointer trace rerun',
        original_strict_FP32_gate_passed=False, same_call_original_Torch_teacher_used=False, recurrent_state_committed=False,
        validated_scope='One B2D6 actual observation34 saved feed, two normal SDK contexts, ABBAABBA; transport outputs match this SDK baseline only',
        whole_Update_or_online_replay_equivalence_validated=False, zero_copy_claim=False, SLAM_quality_or_20Hz_claim=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('run', 'feed', 'query-json', 'output'):
        parser.add_argument('--' + flag, required=True)
    args = parser.parse_args()
    require(not Path(args.output).exists(), 'Audit output already exists')
    result = audit(args)
    result['audit_script_sha256'] = sha(__file__)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], A_wall_ms=result['warm_means_recomputed_ms']['A']['actual_execute_wall_mean_seconds'],
                         B_wall_ms=result['warm_means_recomputed_ms']['B']['actual_execute_wall_mean_seconds'])))


if __name__ == '__main__':
    main()
