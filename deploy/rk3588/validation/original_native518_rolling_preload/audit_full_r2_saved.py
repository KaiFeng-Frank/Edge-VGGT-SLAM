#!/usr/bin/env python3
"""Independent JSON/NumPy/text audit of completed R123 R2 full-model ABBA.

No preprocessing, neural operations, ONNX, SDK, GPU or board calls. Exact
native-NPU reference equality verifies scheduling preservation, not FP32 quality.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

ROOT = Path('/dev/shm/edge-vggt-v123-review')
REF = Path('/path/to/research-artifacts/v111-native518-trajectory-quality-r1/batch0-r1')
REF_SHA = '2eb6f63d2c832c92631a55e527f4b49e3bf0b197685f3bdea8f76008aada1081'
MANIFEST_SHA = '232eb98e124b895fa25d4ba5083bb157d5adac206e923283edb8b86f9073ad77'
RUNNER_SHA = '534912cd7e60354e8d84f286d11fa8a5e2d8b3c6f4c6b32778fc2376e389d1b1'
RUNTIME_SHA = '99a08a3cb7823e5212a12184e487413318f206a10bedea59fe87110df6065f21'
OUTPUTS = ('pose_enc', 'extrinsic', 'intrinsic', 'depth', 'depth_conf')
COLD = dict(contexts_created=10, contexts_destroyed=10, runs_completed=18,
            output_buffers_acquired=23, output_buffers_released=23)
TOTAL = {k: 4*v for k, v in COLD.items()}
SUCCESS = 'completed_original_full_native518_ABBA_not_realtime_or_quality_accepted'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for piece in iter(lambda: stream.read(4 << 20), b''):
            value.update(piece)
    return value.hexdigest()


def raw(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def ledger(value, contexts, runs, buffers):
    require(value['contexts_created'] == value['contexts_destroyed'] == contexts
            and value['runs_completed'] == runs
            and value['output_buffers_acquired'] == value['output_buffers_released'] == buffers
            and value['closed'] and not value['failed'] and not value['live_contexts']
            and not value['live_output_buffers'] and not value['cleanup_errors'],
            'Unbalanced or failed completed SDK owner')


def rolling_timeline(owner):
    events = owner['timeline']
    require(events and all(math.isfinite(e['monotonic_seconds']) for e in events)
            and all(a['monotonic_seconds'] <= b['monotonic_seconds'] for a, b in zip(events, events[1:])),
            'Rolling event order differs')
    maximum = max(e['reserved_or_initializing_contexts'] for e in events)
    require(maximum == owner['maximum_reserved_or_initializing_contexts_observed']
            == owner['maximum_simultaneously_live_SDK_contexts'] == 2
            and owner['configured_live_context_limit'] == 2
            and owner['live_reserved_or_initializing_contexts'] == owner['live_owner_workers'] == 0,
            'Rolling reservation/terminal worker counts differ')
    require(owner['persistent_NPU_contexts'] is False
            and owner['init_payload_hash_destroy_in_complete_timer'] is True,
            'Full context lifetime must remain in the boundary')
    rows = []
    for group in range(4):
        selected = [e for e in events if e['group'] == group]
        require(selected and len({e['worker_thread'] for e in selected}) == 1,
                'Each context must use one owner thread')
        phases = {}
        for phase in ('init_begin', 'init_end', 'run_begin', 'run_end'):
            values = [e['monotonic_seconds'] for e in selected if e['phase'] == phase]
            require(len(values) == 1, 'Each group must initialize/run exactly once')
            phases[phase] = values[0]
        close_ends = [e['monotonic_seconds'] for e in selected if e['phase'] == 'close_end']
        require(close_ends and phases['init_begin'] <= phases['init_end'] <= phases['run_begin']
                <= phases['run_end'] <= min(close_ends)
                and not any(e['phase'].endswith('_failed') for e in selected),
                'Group initialization/run/destruction order differs')
        rows.append(dict(group=group, worker_thread=selected[0]['worker_thread'],
                         **phases, first_close_end=min(close_ends), terminal_close_end=max(close_ends)))
    for prior, following in zip(rows, rows[1:]):
        require(prior['run_end'] <= following['run_begin']
                and prior['first_close_end'] <= following['run_begin']
                and prior['run_begin'] <= following['init_begin'],
                'Runs must be serial and preload must follow actual current run start')
        following['init_overlap_with_prior_run_seconds'] = max(0.,
            min(prior['run_end'], following['init_end'])
            - max(prior['run_begin'], following['init_begin']))
    return dict(event_count=len(events), observed_reservation_upper_bound=maximum,
                owner_thread_consistent=True, SDK_runs_serial=True, terminal_workers=0,
                terminal_reservations=0, groups=rows,
                scope='Reservation includes an initializing slot; no unique physical-memory sum or device-parallelism inference')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--reference', type=Path, default=REF)
    parser.add_argument('--result-sha256', required=True)
    parser.add_argument('--control-sha256', required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'full-r2-saved-audit.json')
    args = parser.parse_args()
    files = []

    def record(path, digest=None):
        actual = sha(path)
        require(digest is None or actual == digest, 'Pinned file differs: '+str(path))
        files.append(dict(file=str(path), bytes=path.stat().st_size, sha256=actual))
        return path

    def receipt(path, digest=None):
        return json.loads(record(path, digest).read_text())

    def load(base, row):
        require(Path(row['file']).name == row['file'], 'NPY basename required')
        path = record(base/row['file'], row['sha256'])
        value = np.load(path, allow_pickle=False)
        require(path.stat().st_size == row['bytes'] and list(value.shape) == row['shape']
                and str(value.dtype) == row['dtype'] == 'float32' and raw(value) == row['raw_sha256']
                and bool(np.isfinite(value).all()) and row['finite'], 'NPY metadata/finite differs')
        files[-1].update(shape=list(value.shape), dtype='float32', raw_sha256=raw(value), finite=True)
        return value

    result = receipt(args.root/'full-r2/result.json', args.result_sha256)
    control = receipt(args.root/'control-r2/control.json', args.control_sha256)
    reference = receipt(args.reference/'result.json', REF_SHA)['pairs'][0]
    require(result['schema'] == 'v123_original_native518_full_model_rolling_preload_ABBA'
            and result['status'] == SUCCESS and len(result['calls']) == 4
            and [(c['index'], c['route']) for c in result['calls']] == list(enumerate('ABBA'))
            and result['cleanup_errors'] == [], 'Completed unchanged ABBA required')
    require(result['reference_receipt_sha256'] == REF_SHA
            and result['reference_output_metadata'] == reference['outputs']
            and result['source_images'] == reference['source_images']
            and result['source_indices'] == reference['source_indices'] == [32, 56]
            and result['reference_input_raw_sha256'] == reference['prepared_pair_raw_sha256']
            and result['shape'] == [2, 3, 392, 518] and result['core_mask'] == 7
            and result['previous_tail_cache_enabled'] is False
            and result['reference_intermediate_substituted'] is False
            and result['DINO_layers_per_frame'] == 24 and result['aggregator_blocks_per_pair'] == 48
            and result['camera_refinements'] == 4 and result['DPT_frames'] == 2
            and result['original_pose_codec'] and not result['SL4_backend_executed']
            and not result['candidate_accepted'], 'Original native scope/input differs')
    source = args.root/'executed-source-r2'
    manifest = receipt(source/'sources.json', MANIFEST_SHA)
    require(manifest == result['source_bundle'] and result['source_manifest_sha256'] == MANIFEST_SHA
            and manifest['run_rolling_full_model_board.py']['sha256'] == RUNNER_SHA
            and manifest['rolling_native_aggregator.py']['sha256'] == RUNTIME_SHA,
            'Actual R2 source identity differs')
    for name, row in manifest.items():
        path = record(source/name, row['sha256'])
        require(path.stat().st_size == row['bytes'], 'Executed source size differs')
        ast.parse(path.read_text())
    native = {key: load(args.reference, reference['outputs'][key]) for key in OUTPUTS}
    comparisons, timelines, expected_arrays = [], {}, set()
    for call in result['calls']:
        index, executor = call['index'], call['executor_call']
        require(executor['status'] == 'completed' and executor['complete_expected_execution']
                and executor['resources_balanced'] and executor['source_and_inputs_unchanged']
                and executor['cleanup_errors'] == [] and executor['SDK_totals'] == COLD
                and executor['encoder_frame_indices_computed'] == [0, 1]
                and executor['requested_core_mask'] == 7 and not executor['tail_cache_hit']
                and not executor['previous_tail_cache_enabled'] and not executor['prediction_or_aggregator_cache_used']
                and not executor['fixed_reference_pixel_substitution'] and not executor['intermediate_reference_substitution']
                and executor['actual_fresh_caller_pixels'] and executor['original_model_only']
                and executor['cache_after_call']['cache_entries'] == executor['cache_after_call']['cache_bytes'] == 0
                and call['original_input_raw_sha256'] == executor['input_raw_sha256'] == reference['prepared_pair_raw_sha256']
                and executor['input_frame_raw_sha256'] == reference['prepared_frame_raw_sha256'],
                'Complete cold actual-input execution differs')
        require(all(executor[k] for k in ('affinity_restored', 'environment_restored', 'sys_path_restored')),
                'Per-call restoration differs')
        require(set(call['outputs']) == set(OUTPUTS), 'Five canonical outputs required')
        compared = {}
        for key in OUTPUTS:
            row = call['outputs'][key]
            expected_arrays.add(row['file'])
            value = load(args.root/'full-r2', row)
            require(value.shape == native[key].shape and value.tobytes() == native[key].tobytes()
                    and call['byte_equal_frozen_full_model'][key]
                    and executor['final_outputs'][key]['raw_sha256'] == row['raw_sha256'],
                    'Actual canonical reference bytes differ')
            compared[key] = dict(byte_equal_R111=True, raw_sha256=raw(value),
                                 shape=list(value.shape), file_sha256_equal=row['sha256'] == reference['outputs'][key]['sha256'])
        owners = executor['resources']
        require(set(owners) == {'dino0', 'dino1', 'dino2', 'dino3', 'aggregator', 'heads'}, 'Owner set differs')
        for group in range(4):
            ledger(owners['dino'+str(group)], 1, 2, 2)
        aggregator = owners['aggregator']
        ledger(aggregator, 4, 4, 9)
        require(aggregator['pins_unchanged'] and aggregator['completed_original_blocks'] == 48
                and not aggregator['teacher_forcing'], 'Complete aggregator scope differs')
        require(len(aggregator['per_group_closed_contexts']) == 4, 'Four aggregator owners required')
        for group, owner in enumerate(aggregator['per_group_closed_contexts']):
            ledger(owner, 1, 1, 3 if group == 3 else 2)
            require(owner['model_sha256'] == reference['executor_call']['resources']['aggregator']
                    ['per_group_closed_contexts'][group]['model_sha256'], 'Original group artifact differs')
        heads = owners['heads']
        require(heads['closed'] and heads['resources_balanced'] and not heads['failed']
                and not heads['cleanup_errors'] and len(heads['contexts']) == 2
                and heads['completed_refinements'] == 4 and heads['completed_depth_frames'] == 2,
                'Original camera/depth completion differs')
        for owner, runs in zip(heads['contexts'], (4, 2)):
            ledger(owner, 1, runs, runs)
        for owner in ([owners['dino'+str(g)] for g in range(4)]
                      + aggregator['per_group_closed_contexts'] + heads['contexts']):
            require(owner['core_mask'] == 7 and owner['init_flags'] == 0, 'Normal core7 context required')
        if call['route'] == 'B':
            timelines[index] = rolling_timeline(aggregator)
        require(call['complete_prediction_seconds'] >= executor['complete_predict_boundary_seconds'] > 0
                and call['save_and_compare_seconds'] >= 0
                and math.isclose(call['image_read_to_prediction_seconds'],
                    call['preprocessing_seconds']+call['complete_prediction_seconds'], rel_tol=1e-12),
                'Outer complete timing differs')
        groups = executor['aggregator_timings']['groups']
        require([g['index'] for g in groups] == [0, 1, 2, 3]
                and all(g['complete_boundary_seconds'] >= g['SDK_run_seconds'] > 0 for g in groups),
                'Actual serial aggregator SDK timing differs')
        sdk_sum = sum(g['SDK_run_seconds'] for g in groups)
        require(executor['stage_seconds']['all48_aggregator_and_consumers_seconds'] >= sdk_sum,
                'Serial SDK runs must fit the complete aggregator wall boundary')
        comparisons.append(dict(index=index, route=call['route'], outputs=compared,
                                complete_prediction_seconds=call['complete_prediction_seconds'],
                                image_read_to_prediction_seconds=call['image_read_to_prediction_seconds'],
                                preprocessing_seconds=call['preprocessing_seconds'],
                                save_and_compare_seconds=call['save_and_compare_seconds'],
                                original_stage_seconds=executor['stage_seconds'],
                                aggregator_serial_SDK_run_sum_seconds=sdk_sum,
                                aggregator_per_group_SDK_run_seconds=[g['SDK_run_seconds'] for g in groups],
                                aggregator_per_group_init_seconds=[g['file_pin_and_context_init_seconds'] for g in groups],
                                aggregator_per_group_destroy_seconds=[g['context_destroy_seconds'] for g in groups],
                                aggregator_observed_preload_wait_seconds=[g.get('preload_wait_seconds') for g in groups]))
    require({p.name for p in (args.root/'full-r2').glob('*.npy')} == expected_arrays
            and len(expected_arrays) == 20, 'Exactly twenty final arrays required')
    final = result['executor_resources']
    require(final['closed'] and final['resources_balanced'] and final['cleanup_errors'] == []
            and final['live_contexts'] == final['live_output_buffers'] == 0 and final['SDK_totals'] == TOTAL
            and all(result[k] for k in ('source_and_inputs_unchanged', 'affinity_restored', 'environment_restored', 'sys_path_restored'))
            and result['memory_sampling_errors'] == [], 'Final ledger/restoration differs')
    samples = [json.loads(line) for line in record(args.root/'full-r2/memory-samples.jsonl').read_text().splitlines()]
    require(len(samples) > 1 and samples == result['memory_samples']
            and all(a['monotonic_seconds'] <= b['monotonic_seconds'] for a, b in zip(samples, samples[1:])),
            'Persisted memory observations differ from final receipt')
    log = record(args.root/'control-r2/worker.log').read_text()
    starts, logged, current = [], {}, None
    for line in log.splitlines():
        if not line.startswith('{'):
            continue
        row = json.loads(line)
        if row.get('stage') == 'complete_original_prediction_start':
            current = row['index']; starts.append((current, row['route']))
        if 'rolling_event' in row:
            logged.setdefault(current, []).append(row['rolling_event'])
    require(starts == list(enumerate('ABBA')) and set(logged) == {1, 2}, 'Actual ABBA worker progress differs')
    for index in (1, 2):
        require(logged[index] == result['calls'][index]['executor_call']['resources']['aggregator']['timeline'],
                'Persisted rolling phase events differ from terminal snapshots')
    require(control['child_returncode'] == 0 and control['restore_errors'] == []
            and control['baseline'] == control['restored'] and control['frequency_governor_limits_restored']
            and control['restored']['cur_freq'] == '300000000'
            and all(o['actual_frequency_hz'] == 1000000000 for o in control['observations']),
            'External completion/frequency restore differs')
    maximum_temperature = max(t for o in control['observations'] for t in o['temperatures_c'].values())
    require(maximum_temperature < control['temperature_limit_c'], 'Temperature limit reached')
    relocation = receipt(args.root/'export-relocation.json')
    require(result['storage_relocation']['receipt'] == relocation
            and result['storage_relocation']['receipt_sha256'] == sha(args.root/'export-relocation.json')
            and relocation['status'] == 'moved_verified_original_path_directory_symlink'
            and relocation['all_original_bytes_preserved'], 'Reversible storage receipt differs')
    means = {route: statistics.mean(c['complete_prediction_seconds'] for c in result['calls'] if c['route'] == route)
             for route in 'AB'}
    require(all(math.isclose(means[k], result['all_call_means_seconds'][k], rel_tol=1e-12) for k in means),
            'Reported complete prediction means differ')
    image_means = {route: statistics.mean(c['image_read_to_prediction_seconds'] for c in result['calls'] if c['route'] == route)
                   for route in 'AB'}
    aggregator_means = {route: statistics.mean(c['original_stage_seconds']['all48_aggregator_and_consumers_seconds']
                                              for c in comparisons if c['route'] == route) for route in 'AB'}
    sdk_means = {route: statistics.mean(c['aggregator_serial_SDK_run_sum_seconds']
                                      for c in comparisons if c['route'] == route) for route in 'AB'}
    memory = dict(sample_count=len(samples), min_MemAvailable_KiB=min(s['meminfo_KiB']['MemAvailable'] for s in samples),
                  maximum_Shmem_KiB=max(s['meminfo_KiB']['Shmem'] for s in samples),
                  minimum_zone_free_bytes={zone: min(s['zone_free_bytes'][zone] for s in samples)
                                            for zone in samples[0]['zone_free_bytes']})
    audit = dict(schema='v123_independent_completed_full_model_R2_saved_audit', status='verified_completed_native_NPU_byte_equivalent_ABBA',
                 auditor_sha256=sha(__file__), execution='Host JSON/NumPy/text/AST only; no inference/preprocessing/SDK/GPU/board calls',
                 checked_final_NPYs=20, reference_receipt_sha256=REF_SHA, all20_outputs_byte_equal_R111=True,
                 actual_input_source_indices=[32, 56], prepared_pair_raw_sha256=reference['prepared_pair_raw_sha256'],
                 input_byte_scope='Prepared pixel hashes are executed frozen preprocessing receipts; no host preprocessing rerun',
                 calls=comparisons, cold_per_call_SDK_totals=COLD, final_SDK_totals=TOTAL,
                 original_model=dict(DINO_layers_each_frame=24, frames_computed=2, aggregator_blocks=48,
                                     camera_refinements=4, DPT_frames=2, original_codec=True, previous_tail_cache=False),
                 complete_prediction_means_seconds=means, image_read_to_prediction_means_seconds=image_means,
                 B_prediction_time_reduction_percent=100*(1-means['B']/means['A']),
                 aggregator_complete_wall_means_seconds=aggregator_means,
                 aggregator_serial_SDK_run_sum_means_seconds=sdk_means,
                 timing_interpretation='All four calls are reported. First A also has longer DINO/head boundaries; pooled full-model difference is not a pure preload causal effect or sustained throughput estimate. Aggregator wall and serial SDK time are separate observations.',
                 first_and_repeat_per_route={route: [c['complete_prediction_seconds'] for c in result['calls'] if c['route'] == route]
                                            for route in 'AB'},
                 timing_scope=result['timing_scope'], rolling_owner_observations=timelines,
                 memory_observations=memory, reversible_storage_receipt_sha256=sha(args.root/'export-relocation.json'),
                 restored_worker_state=True, final_workers_and_reservations_zero=True,
                 external_control=dict(child_returncode=0, observed_NPU_hz=1000000000, restored_NPU_hz=300000000,
                                       exact_frequency_policy_restored=True, max_sampled_temperature_c=maximum_temperature,
                                       controller_seconds=control['seconds']),
                 limits=['R111 native NPU byte equality is scheduling equivalence, not original FP32 or trajectory quality acceptance',
                         'Two identical-image cold calls per route; no timestamp pacing, fresh-observation throughput, P95 or sustained thermal measurement',
                         'Original SL4/map backend was not run; no full SLAM, realtime,30min or8GB acceptance',
                         'Overlapped stage times cannot be summed as wall time; B includes diagnostic phase logging',
                         'SDK memory, RSS, reservations and system zone observations are different scopes; not additive physical memory',
                         'Relocation hashes are verified in the saved original relocation receipt; this audit does not rehash its512MB ONNX'],
                 files=files)
    args.output.write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=audit['status'], outputs=20, means_seconds=means,
                          reduction_percent=audit['B_prediction_time_reduction_percent'], SDK_totals=TOTAL)))


if __name__ == '__main__':
    main()
