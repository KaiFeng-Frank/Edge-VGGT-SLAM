#!/usr/bin/env python3
"""Audit saved original VGGT exclusive-session replay; no model or SDK execution.

Reuse existing saved timestamp/map/lifecycle and staged pageout checks;
also bind the explicit exclusive immutable-session contract. Only JSON, the original RGB index and saved NumPy are read.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SEGMENTS = ('arrival_to_decode_seconds', 'decode_seconds', 'queue_seconds',
            'prediction_seconds', 'SL4_and_points_seconds', 'publication_seconds')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path, expected=None):
    if expected is not None:
        require(sha(path) == expected, 'Saved receipt SHA differs')
    return json.loads(Path(path).read_text())


def mean(values):
    total = 0.
    for value in values:
        total += value
    return total / len(values)


def close(actual, expected, name):
    require(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9), name)


def accounting(source, count):
    require(source['source'] == 'original_timestamp_rgb_replay' and source['queue_capacity'] == 1
            and source['decode_policy'] == 'first_then_latest_due_original_timestamp'
            and source['error'] is None and source['physical_sensor_latency_validated'] is False,
            'Original one-slot source required')
    for key in ('original_frames', 'generated', 'delivered', 'overwritten', 'skipped_before_decode'):
        require(type(source[key]) is int and source[key] >= 0, 'Invalid source count')
    require(source['original_frames'] == source['generated'] + source['skipped_before_decode']
            and source['generated'] == source['delivered'] + source['overwritten']
            and source['delivered'] == count + 1, 'Source drop/delivery accounting differs')


def segment_checks(updates):
    residual_input, residual_processing = [], []
    for row in updates:
        values = [row[key] for key in SEGMENTS]
        require(all(math.isfinite(v) and v >= 0 for v in values)
                and math.isfinite(row['full_model_seconds']) and 0 <= row['full_model_seconds'] <= row['prediction_seconds']
                and math.isfinite(row['graph_error']), 'Invalid timing or graph error')
        full, processing = sum(values), sum(values[3:])
        close(row['input_to_pose_seconds'], full, 'Input-to-pose does not include all stages')
        close(row['processing_seconds'], processing, 'Processing stage sum differs')
        close(row['input_to_pose_seconds'], row['published_monotonic'] - row['captured_monotonic'], 'Latency endpoints differ')
        residual_input.append(abs(row['input_to_pose_seconds'] - full))
        residual_processing.append(abs(row['processing_seconds'] - processing))
    return max(residual_input), max(residual_processing)


def audit(args):
    directory = args.replay_directory.resolve(strict=True)
    report = read(directory / 'results.json', args.results_sha256)
    complete = read(args.complete_directory / 'result.json', args.complete_result_sha256)
    require(report['status'] == 'completed' and report['model'] == 'original_VGGT' and report['max_loops'] == 0
            and report['cleanup_errors'] == [] and report['candidate_accepted'] is False
            and report['trajectory_quality_validated'] is False and report['real_time_validated'] is False
            and report['physical_sensor_latency_validated'] is False and report['short_replay_not_30min'] is True
            and report['rknn_retained_aggregator'] is True and report['rknn_grouped_dino'] is True
            and report['rknn_paired_aggregator'] is True and report['patch_cache'] is True,
            'Completed original diagnostic required')
    require(sha(args.replay_source) == report['script_sha256'], 'Frozen replay source differs')
    for replay_key, complete_key in (
            ('rknn_manifest_sha256', 'manifest_sha256'), ('rknn_dino_manifest_sha256', 'dino_manifest_sha256'),
            ('rknn_depth_manifest_sha256', 'depth_manifest_sha256'), ('half_io_library_sha256', 'half_io_library_sha256'),
            ('model_threads', 'model_threads'),
            ('cpu_affinity', 'cpu_affinity')):
        require(report[replay_key] == complete[complete_key], 'Complete/replay execution identity differs: ' + replay_key)
    for name, digest in complete['runtime_source_sha256'].items():
        require(report['runtime_source_sha256'].get(name) == digest, 'Complete/replay runtime source differs: ' + name)
    require(set(report['runtime_source_sha256']) - set(complete['runtime_source_sha256']) <= {'runtime.py', 'checkpoint_identity.py', 'immutable_execution_session.py'}, 'Unexpected replay-only runtime source')
    for name, digest in report['runtime_source_sha256'].items():
        path = (args.source_root / 'vggt_slam' / name).resolve(strict=True)
        require(path.is_relative_to((args.source_root / 'vggt_slam').resolve()) and sha(path) == digest, 'Frozen runtime source differs: ' + name)
    stamps = [float(line.split()[0]) for line in args.rgb_index.read_text().splitlines()
              if line.strip() and not line.lstrip().startswith('#')]
    require(len(stamps) == report['original_frames'] and all(math.isfinite(v) for v in stamps)
            and all(a < b for a, b in zip(stamps, stamps[1:])), 'Original index timestamps invalid')
    rows, source = report['updates'], report['source']
    count = report['fresh_pose_count']
    require(report['immutable_execution'] is True
            and report['immutable_execution_contract'] == 'exclusive_immutable_model'
            and report['arbitrary_model_mutation_detection_preserved'] is False
            and report['immutable_execution_source_sha256'] == 'a731fb1f26ea82f31fa7123792625e1731e19d3e35723930aef18e331dbe6897'
            and report['runtime_source_sha256']['immutable_execution_session.py'] == report['immutable_execution_source_sha256'],
            'Explicit original immutable execution identity differs')
    session = report['immutable_session']
    require(session['contract'] == 'exclusive_immutable_model'
            and session['full_seal_validated'] and session['final_full_validated']
            and session['closed'] and not session['failed'] and session['cleanup_errors'] == []
            and session['fixed_tensors'] == 1343 and session['version_checks'] == 2*count
            and session['method_checks'] == 4*count and session['restored_overrides'] == 3
            and session['invalidations'] == 0
            and session['arbitrary_live_module_scalar_registry_mutation_detection'] is False
            and session['original_mutable_model_guard_semantics_equivalent'] is False,
            'Session seal/final validation, restoration or per-call evidence differs')
    require(type(count) is int and count > 0 and len(rows) == count and report['seed_timestamp'] == stamps[0], 'Fresh/seed counts differ')
    accounting(source, count)
    require(source['original_frames'] == len(stamps), 'Source index count differs')
    sequences = [row['sequence'] for row in rows]
    require(all(type(v) is int and 0 < v < len(stamps) for v in sequences)
            and all(a < b for a, b in zip(sequences, sequences[1:])) and sequences[-1] == len(stamps) - 1,
            'Fresh sequences or EOF tail differ')
    for i, row in enumerate(rows):
        require(row['revision'] == i + 1 and row['graph_nodes'] == 2 * (i + 1)
                and row['timestamp'] == stamps[row['sequence']]
                and row['anchor_timestamp'] == (rows[i-1]['timestamp'] if i else report['seed_timestamp']),
                'Fresh timestamp/anchor/map revision differs')
        require(row['captured_monotonic'] == source['start_monotonic'] + row['timestamp'] - stamps[0], 'Original scheduled arrival differs')
    require(all(a['published_monotonic'] < b['published_monotonic'] for a, b in zip(rows, rows[1:])), 'Publication timestamps not increasing')
    close(source['last_capture_monotonic'], rows[-1]['captured_monotonic'], 'EOF scheduled capture differs')
    duration = rows[-1]['published_monotonic'] - source['start_monotonic']
    close(report['fresh_wall_seconds_including_EOF_drain'], duration, 'Fresh wall denominator differs')
    close(report['fresh_pose_hz'], count / duration, 'Fresh rate denominator differs')
    p95 = float(np.percentile([row['input_to_pose_seconds'] for row in rows], 95, method='linear'))
    close(report['input_to_pose_p95_seconds'], p95, 'Full P95 differs')
    residual_input, residual_processing = segment_checks(rows)
    mapping = read(directory / 'map.json')
    require(mapping['format'] == 'vggt_camera_local_dense_chunks_with_SL4_graph' and mapping['units'] == 'model_units'
            and mapping['metric_scale_validated'] is False and mapping['revision'] == count
            and len(mapping['chunks']) == count and set(mapping['nodes']) == {str(i) for i in range(2*count)}, 'Map revision/node/chunk counts differ')
    for value in mapping['nodes'].values():
        matrix = np.asarray(value, dtype=np.float64)
        require(matrix.shape == (4, 4) and np.isfinite(matrix).all(), 'Invalid saved SL4 map node')
    artifacts, points = [], 0
    for i, chunk in enumerate(mapping['chunks']):
        require(chunk['submap_id'] == 2*i and chunk['nodes'] == [2*i, 2*i+1]
                and len(chunk['point_counts']) == 2, 'Map chunk node order differs')
        path = (directory / chunk['path']).resolve(strict=True)
        require(path.is_relative_to(directory / 'map_chunks'), 'Map chunk escapes saved directory')
        with np.load(path, allow_pickle=False) as saved:
            require(set(saved.files) == {'points_0', 'points_1', 'colors_0', 'colors_1'}, 'Map array keys differ')
            for frame, n in enumerate(chunk['point_counts']):
                p, c = saved[f'points_{frame}'], saved[f'colors_{frame}']
                require(type(n) is int and n >= 0 and p.shape == c.shape == (n, 3)
                        and p.dtype == np.float32 and c.dtype == np.uint8 and np.isfinite(p).all(), 'Map point/color shape/dtype/finite differs')
                points += n
        artifacts.append(dict(file=chunk['path'], bytes=path.stat().st_size, sha256=sha(path)))
    require({p.name for p in (directory/'map_chunks').glob('*.npz')} == {Path(a['file']).name for a in artifacts}, 'Saved map artifact set differs')
    poses = np.loadtxt(directory / 'poses.tum', dtype=np.float64, ndmin=2)
    require(poses.shape == (count+1, 8) and np.isfinite(poses).all()
            and np.array_equal(poses[:,0], [report['seed_timestamp']] + [r['timestamp'] for r in rows]), 'Saved pose timestamps/count/finite differs')
    lifecycle, total_runs = {}, 0
    for key, contexts, runs in (('rknn', 4, 4*count), ('dino_rknn', 6, 6*(count+1)), ('camera_rknn', 1, 4*count), ('depth_rknn', 1, 2*count)):
        stat = report[key]
        require(stat['contexts_created'] == stat['contexts_destroyed'] == contexts and stat['closed'] is True
                and stat['failed'] is False and stat['live_contexts'] == 0 and stat['cleanup_errors'] == []
                and stat.get('sdk_runs_succeeded', stat.get('npu_run_api_call_count')) == runs, 'SDK lifecycle/count differs: ' + key)
        if 'sdk_run_attempts' in stat:
            require(stat['sdk_run_attempts'] == runs, 'SDK attempt count differs')
        lifecycle[key] = dict(contexts_created_destroyed=contexts, successful_SDK_runs=runs, live_contexts=0, cleanup_errors=[])
        total_runs += runs
    agg, dino, camera, depth, cache = (report[k] for k in ('rknn', 'dino_rknn', 'camera_rknn', 'depth_rknn', 'cache'))
    require(agg['outputs_acquired'] == agg['outputs_released'] == 9*count and agg['completed_aggregator_calls'] == count
            and agg['completed_pairs'] == 24*count and agg['completed_blocks'] == 48*count
            and agg['group_calls'] == {f'group{i}': count for i in range(4)} and agg['target_tokens_source_global_index'] == 20
            and agg['original_forward_restored'] is True, 'Original retained aggregator calls/buffers differ')
    for stat, runs, outputs in ((agg, 4*count, 9*count), (dino, 6*(count+1), 6*(count+1))):
        half = stat['half_io']
        require(half['calls_attempted'] == half['calls_completed'] == half['input_conversion_calls'] == runs
                and half['output_conversion_calls'] == outputs, 'Owned half output conversion count differs')
    require(dino['batch2_feature_calls'] == 1 and dino['batch1_feature_calls'] == count-1
            and dino['group_frame_calls'] == {f'group{i}': count+1 for i in range(6)}
            and camera['completed_head_calls'] == count and len(camera['last_steps']) == 4
            and all(v['output_buffer_released'] is True for v in camera['last_steps'])
            and depth['completed_frames'] == 2*count, 'Original DINO/camera/depth call counts differ')
    require(cache['requests'] == count and cache['input_frames'] == 2*count and cache['hit_frames'] == count-1
            and cache['computed_frames'] == count+1 and cache['closed'] is True and cache['restored'] is True
            and cache['cache_entries'] == cache['cache_bytes'] == cache['errors'] == cache['invalidations'] == 0
            and cache['restore_errors'] == [], 'Cache lifecycle/count differs')
    result = dict(schema='v77_original_VGGT_exclusive_immutable_timestamp_replay_saved_audit', status='passed_saved_data_checks',
        scope='Saved JSON, original RGB index, poses and map NPZ only; no model, ORT, SDK, board or trajectory-quality execution.',
        auditor_source_sha256=sha(__file__), results_sha256=args.results_sha256, complete_result_sha256=args.complete_result_sha256,
        replay_source_sha256=report['script_sha256'], original_runtime_hashes_and_manifests_match_complete_candidate=True,
        immutable_session=session, immutable_source_sha256=report['immutable_execution_source_sha256'],
        original_mutable_guard_semantics_equivalent=False,
        rgb_index_sha256=sha(args.rgb_index), original_index_timestamps_independently_matched=True,
        fresh_pose_count=count, unique_sequence_count=len(set(sequences)), sequences=sequences,
        source=source, fresh_wall_seconds_including_EOF_drain=duration, fresh_pose_hz=count/duration,
        input_to_pose_p95_seconds=p95, last_input_to_pose_seconds=rows[-1]['input_to_pose_seconds'],
        EOF_drain_seconds_after_source_end=rows[-1]['published_monotonic']-source['end_monotonic'],
        segment_mean_seconds={k: mean([r[k] for r in rows]) for k in SEGMENTS},
        full_model_mean_seconds=mean([r['full_model_seconds'] for r in rows]),
        processing_mean_seconds=mean([r['processing_seconds'] for r in rows]),
        input_to_pose_segment_max_residual_seconds=residual_input, processing_segment_max_residual_seconds=residual_processing,
        full_model_nested_in_prediction_not_added_twice=True, map_sha256=sha(directory/'map.json'),
        map_chunk_count=count, graph_node_count=2*count, map_arrays_checked=4*count, point_count_total=points,
        map_chunks_finite_and_shape_counts_match=True, map_artifact_rows=artifacts,
        poses_count=count+1, poses_timestamps_match_updates_and_seed=True, poses_sha256=sha(directory/'poses.tum'),
        quaternion_unit_length_max_error=float(np.abs(np.linalg.norm(poses[:,4:], axis=1)-1).max()),
        lifecycle=lifecycle, total_contexts_created_destroyed=12, total_successful_SDK_runs=total_runs,
        aggregator_outputs_acquired_released=9*count, patch_cache_closed_restored=True,
        candidate_accepted=False, trajectory_quality_validated=False, physical_sensor_latency_validated=False, real_time_validated=False,
        limitations=['New selected timestamps are not the fixed-input complete comparison; no baseline trajectory quality is projected to this replay.',
                    'Software scheduled RGB arrival includes decode, queue, full prediction, SL4/points and dense map/pose publication; EOF drain is included.',
                    'Saved lifecycle counters and frozen finally paths support cleanup; this audit does not independently observe SDK frees.',
                    'Only aggregator buffers have full acquired/released counters; DINO completed owned conversions and camera last_steps are additional recorded evidence, not complete buffer instrumentation.',
                    'Exclusive immutable ownership defers arbitrary module/registry/config detection to final validation; it is not the original mutable-model guard contract.',
                    'Finite poses and maps do not establish trajectory/map quality, metric scale, physical-camera latency, 30-minute stability or 8GB operation.'])
    result.update(pageout=pageout(args, report, directory), frequency_controls=[frequency_control(p) for p in (args.complete_control, args.replay_control)])
    require(sha(args.controller_source) == 'cb991cf03abc872e3590dd92f26fd0f5ac548d620c5ca791f9354537a9f793b9', 'Frozen frequency controller source differs')
    platforms = [report['platform_start']] + [row['platform'] for row in rows]
    require(all(p['npu']['cur_freq'] == 1000000000 and p['npu']['governor'] == 'userspace' for p in platforms), 'Replay platform clock differs')
    result.update(controller_source_sha256=sha(args.controller_source), platform_sample_count=len(platforms), platform_max_temperature_c=max(t for p in platforms for t in p['temperature_c'].values()), startup_seconds=report['startup_seconds'], peak_rss_mib=report['peak_rss_mib'], candidate_accepted=False, eight_GB_hardware_validated=False)
    result['limitations'].append('Frequency evidence is sampled; complete and replay controls each restore effective300MHz. Fresh observations differ from the fixed complete-run pairs and previous replay schedules.')
    return result


def frequency_control(path):
    report = read(path)
    require(report['status'] == 'worker_completed_pending_result_review'
            and report['child_returncode'] == 0 and not report['restore_errors']
            and report['frequency_governor_limits_restored'] is True
            and report['requested_frequency_hz'] == 1000000000
            and report['candidate_accepted'] is False, 'Successful bounded frequency control required')
    old, restored, samples = report['baseline'], report['restored'], report['observations']
    require(old['cur_freq'] == restored['cur_freq'] == '300000000'
            and old['governor'] == restored['governor'] == 'userspace'
            and all(old[k] == restored[k] for k in ('min_freq', 'max_freq'))
            and restored['userspace/set_freq'] == '300000000', 'Effective300MHz/governor/limits restoration differs')
    require(samples and 0 < report['timeout_seconds'] <= 180
            and report['seconds'] < report['timeout_seconds'] + 10
            and report['temperature_limit_c'] <= 75, 'Frequency-control bound differs')
    sensors = set(report['baseline_temperatures_c'])
    require(len(sensors) == 7 and all(s['actual_frequency_hz'] == 1000000000
            and set(s['temperatures_c']) == sensors for s in samples), 'Running1GHz/sensor sample differs')
    require(all(a['seconds'] <= b['seconds'] for a, b in zip(samples, samples[1:])), 'Frequency sample order differs')
    peak = max(t for s in samples for t in s['temperatures_c'].values())
    require(peak < report['temperature_limit_c'], 'Thermal stop threshold exceeded')
    return dict(receipt_sha256=sha(path), observation_count=len(samples), all_running_samples_1GHz=True,
                max_running_sample_temperature_c=peak, child_returncode=0,
                effective300MHz_governor_and_limits_restored=True,
                original_userspace_request_was_undefined=report['userspace_request_was_undefined'],
                explicit_restore_request_hz=300000000, elapsed_seconds=report['seconds'], restore_errors=[])


def pageout(args, report, directory):
    require(report['checkpoint_pageout'] is True and report['rknn_camera_split4'] is True
            and report['checkpoint_residency_source_sha256'] == '1948fa3f49814d8fc1025516025df79adc7b1bd0e0a54a86f842b1590f445777'
            and report['eight_GB_hardware_validated'] is False, 'Original staged pageout contract differs')
    stages = ('aggregator_installed', 'camera_installed', 'DINO_installed', 'depth_installed')
    require(tuple(i['stage'] for i in report['checkpoint_pageout_stages']) == stages, 'Four stage order differs')
    receipts, snapshots, metadata, counts = [], [], [], []
    for item in report['checkpoint_pageout_stages']:
        path = (directory / item['receipt']).resolve(strict=True)
        require(path.is_relative_to(directory), 'Pageout receipt path escaped')
        r = read(path, item['sha256'])
        require(r['schema'] == 'v74_original_checkpoint_MADV_PAGEOUT_experiment'
                and r['status'] == item['status'] == 'advice_completed_not_memory_or_quality_acceptance'
                and r['stage'] == item['stage'] and r['script_sha256'] == report['checkpoint_residency_source_sha256']
                and not r['errors'] and r['advice'] == 21
                and r['checkpoint_sha256'] == 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
                and r['checkpoint_bytes'] == 5026874952, 'Pageout identity/status differs')
        require(r['tensor_metadata_before_sha256'] == r['tensor_metadata_after_sha256']
                and r['tensor_storage_VA_version_unchanged'] is True and r['checkpoint_VMA_layout_unchanged'] is True
                and item['tensor_storage_VA_version_unchanged'] is True and item['checkpoint_VMA_layout_unchanged'] is True
                and r['tensor_count'] == 1343 and r['checkpoint_backed_storage_count'] == 1341
                and r['checkpoint_rehashed_by_pageout_helper'] is False and r['tensor_values_read_by_helper'] is False,
                'Original pageout parameter metadata differs')
        require(r['source_sha256'] == {k: report['runtime_source_sha256'][k] for k in ('runtime.py', 'checkpoint_identity.py')}, 'Loader/checkpoint source pins differ')
        require(len(r['advice_calls']) == item['advice_calls'] == 1
                and all(c['returncode'] == c['errno'] == 0 and c['status'] == 'returned' for c in r['advice_calls']), 'Advice call failed')
        snapshots.extend((r['before'], r['after_immediate'])); metadata.extend((r['tensor_metadata_before_sha256'], r['tensor_metadata_after_sha256']))
        counts.append(r['tensor_count'])
        receipts.append(dict(stage=item['stage'], sha256=item['sha256'], advice_calls=1,
                             tensor_count=r['tensor_count'], checkpoint_backed_storage_count=r['checkpoint_backed_storage_count'],
                             original_parameter_storage_VA_and_version_unchanged=True, VMA_layout_unchanged=True))
    snapshots.extend((report['checkpoint_pageout_startup_memory'], report['checkpoint_pageout_precleanup_memory']))
    require(len(set(metadata)) == 1, 'Tensor metadata changed across installation stages')
    def vma_identity(snapshot):
        return [(v['start'],v['end'],v['permissions'],v['offset'],v['dev_major'],v['dev_minor'],v['inode'],v['path']) for v in snapshot['checkpoint_vmas']]
    identity = snapshots[0]['checkpoint_stat']; mapping = vma_identity(snapshots[0]); starttime = snapshots[0]['faults']['starttime_ticks']
    require(mapping and identity['bytes'] == 5026874952, 'Original mapping missing')
    for s in snapshots:
        require(s['checkpoint_stat'] == identity and vma_identity(s) == mapping
                and s['faults']['starttime_ticks'] == starttime, 'Saved checkpoint/process identity changed')
        require(all(all(v['metrics'][k] == 0 for k in ('Private_Dirty','Shared_Dirty','Anonymous','Swap','Locked')) for v in s['checkpoint_vmas']), 'Checkpoint mapping became dirty/anonymous/swapped/locked')
    require(all(a['monotonic'] <= b['monotonic'] and a['faults']['minor_faults'] <= b['faults']['minor_faults'] and a['faults']['major_faults'] <= b['faults']['major_faults'] for a,b in zip(snapshots,snapshots[1:])), 'Snapshot/fault chronology differs')
    startup, precleanup = snapshots[-2:]
    require(startup['monotonic'] <= report['source']['start_monotonic'] and precleanup['monotonic'] >= report['updates'][-1]['published_monotonic'], 'Memory snapshot enters timed fresh service')
    require(max(s['status_kB']['VmHWM']/1024 for s in snapshots) <= report['peak_rss_mib'], 'Saved HWM exceeds process peak')
    return dict(receipts=receipts, snapshot_count=len(snapshots), tensor_metadata_all_stages_unchanged=True,
                tensor_metadata_sha256=metadata[0], checkpoint_stat_and_VMA_layout_continuous=True,
                process_starttime_and_fault_counters_continuous=True,
                startup_RSS_MiB=startup['smaps_rollup_kB']['Rss']/1024,
                precleanup_RSS_MiB=precleanup['smaps_rollup_kB']['Rss']/1024,
                startup_peak_is_not_active_or_postcleanup_RSS=True,
                addresses_PID_inode_and_absolute_checkpoint_path_omitted=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('replay-directory', 'complete-directory', 'replay-source', 'rgb-index', 'output', 'source-root', 'complete-control', 'replay-control', 'controller-source'):
        parser.add_argument('--'+flag, type=Path, required=True)
    for flag in ('results-sha256', 'complete-result-sha256'):
        parser.add_argument('--'+flag, required=True)
    args = parser.parse_args()
    result = audit(args)
    require(not args.output.exists(), 'New output file required')
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('status', 'fresh_pose_count', 'fresh_pose_hz', 'input_to_pose_p95_seconds', 'total_contexts_created_destroyed', 'total_successful_SDK_runs')}))


if __name__ == '__main__':
    main()
