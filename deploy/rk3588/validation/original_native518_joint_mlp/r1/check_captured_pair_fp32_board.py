#!/usr/bin/env python3
"""Board-only same-input FP32 references for original frame0/global0 A/F/B.

Inputs are actual NPU DINO24 captures. These two-block CPU references are not
a full original Torch prediction or a trajectory reference. The caller owns
the 180-second CPU thermal controller; this worker never writes clocks.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time
import traceback

ROOT = Path('/tmp/edge-vggt-v126-basis-export-r2')
EXPORT_SHA = 'e23c8f7cdf6faa895a8bde34a41806bf2022e630f5361529d0271374742d740e'
CHECK_PATH = Path('/tmp/edge-vggt-v126-basis-check-r2/result.json')
CHECK_SHA = '48a7d5034b9eea0afa139e4a69cdb5b393c694dcec2b8e874a1c1f380f5f42b1'
ORIGINAL_WORKER_SHA = '73affeb0b78a794ed214cadcf0307d164ef74f79d52b779bc579e906a1089b70'
ORIGINAL_CONFIG_SHA = '442e22d5f8aaa033604610c92d5ce6a8b889005601f3d9f1cf78394c4f41842a'
CAPTURE_SOURCE_SHA = 'ed500643ef333dd824ccab81a126e87ff8616c3cb3ee09e3ec59f2a4b7bbf7e9'
CHECKPOINT = dict(bytes=5026874952, sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')
FRAME, GLOBAL = [2, 1041, 1024], [1, 2082, 1024]
GATE = dict(relative_l2_max=1e-5, max_abs_max=1e-3, rtol=1e-4, atol=1e-5)
SDK_COUNTS = dict(contexts_created=4, contexts_destroyed=4, runs_completed=16,
                  output_buffers_acquired=16, output_buffers_released=16)
PAIRS = [[0, 8], [16, 24], [48, 56], [64, 72], [96, 104], [112, 120]]
SUCCESS = 'checked_captured_pair_fixed_FP32_gate_passed'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            value.update(block)
    return value.hexdigest()


def digest_ok(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_capture(row):
    """Metadata-only validator; no model-package imports or execution."""
    require(row['schema'] == 'v128_original_native518_frame0_input_capture' and
            row['status'] == 'completed_original_native518_frame0_input_subbatch_capture' and
            row['script_sha256'] == CAPTURE_SOURCE_SHA, 'Frozen completed capture required')
    flags = ('original_model_only', 'actual_execution_on_board_only', 'component_data_capture_only',
             'checkpoint_verified_before_deserialization', 'checkpoint_final_SHA_verified',
             'full_six_pair_PNG_partitions_disjoint', 'all16_SDK_calls_completed', 'model_state_unchanged',
             'all_inputs_before_frame0', 'two_pair_subbatch_complete', 'all_NPU_contexts_closed',
             'affinity_restored', 'environment_restored', 'sys_path_restored', 'source_and_inputs_unchanged')
    require(all(row.get(key) is True for key in flags) and not row['cleanup_errors'], 'Capture completion/cleanup flags differ')
    require(row['checkpoint'] == CHECKPOINT and row['checkpoint_final_sha256'] == CHECKPOINT['sha256'], 'Original checkpoint lineage differs')
    require(row['pixel_shape'] == [2, 3, 392, 518] and row['logical_shape'] == FRAME and
            row['original_DINO_layers_per_image'] == 24 and row['original_images_per_pair'] == 2 and
            row['original_DINO_parameter_count'] == 336 and row['original_aggregator_blocks_executed'] == 0 and
            row['CPU_DINO_block_forwards_executed'] == 0 and row['CPU_original_pair_reference_executed'] is False and
            row['camera_or_DPT_executed'] is False and row['reuse_previous_tail'] is False and
            row['NPU_core_mask'] == 7 and row['NPU_init_flags'] == 0, 'Actual original native DINO-to-frame0 boundary required')
    require(row['SDK_totals'] == row['expected_SDK_totals'] == SDK_COUNTS and len(row['SDK_calls']) == 16 and
            row['maximum_live_NPU_contexts'] == 1 and len(row['owner_metrics']) == 4, 'Captured 4-context/16-run/16-buffer ledger differs')
    for owner in row['owner_metrics']:
        require(owner['closed'] and owner['live_contexts'] == owner['live_output_buffers'] == 0 and
                owner['resource_counts_balanced'] and not owner['cleanup_errors'] and
                {key: owner[key] for key in SDK_COUNTS} == dict(contexts_created=1, contexts_destroyed=1,
                runs_completed=4, output_buffers_acquired=4, output_buffers_released=4), 'Captured SDK owner not clean')
    batch = row['batch_index']
    require(type(batch) is int and batch in (0, 1, 2) and row['planned_all_pairs'] == PAIRS and
            row['planned_calibration_pair_count'] == 4 and row['planned_heldout_pair_count'] == 2 and
            row['saved_array_count'] == 2 and len(row['records']) == 2, 'Original predeclared two-pair batch differs')
    ids = set()
    for index, record in enumerate(row['records']):
        name = record['id']
        require(isinstance(name, str) and name and all(c.isalnum() or c in '_-' for c in name) and name not in ids,
                'Unique safe pair ID required')
        ids.add(name)
        require(record['source_pair_indices'] == PAIRS[2 * batch + index] and
                record['split'] == ('calibration' if batch < 2 else 'heldout') and
                all(record.get(k) is True for k in ('original_final_norm_and_role_concat_executed',
                    'input_is_before_original_frame0', 'actual_four_DINO_groups_executed')), 'Captured pair identity/boundary differs')
        require(record['input']['shape'] == FRAME and record['input']['dtype'] == 'float32' and
                record['input']['finite'] is True and digest_ok(record['input']['raw_sha256']) and
                len(record['DINO_group_outputs']) == 4, 'Captured original frame0 input differs')
        require([v['group'] for v in record['DINO_group_outputs']] == list(range(4)) and
                all(v['indices'] == list(range(6 * v['group'], 6 * v['group'] + 6)) and len(v['frames']) == 2
                    for v in record['DINO_group_outputs']), 'All 24 upstream DINO layers required')
    require(len({v['input']['raw_sha256'] for v in row['records']}) == 2, 'Distinct captured pair inputs required')
    require({(v['group'], v['pair_id'], v['frame']) for v in row['SDK_calls']} ==
            {(g, name, frame) for g in range(4) for name in ids for frame in range(2)}, 'Captured SDK call coverage differs')


def validate_original_pair(export, checked):
    require(export['schema'] == 'v126_original_native518_basis_pair_export' and
            export['status'] == 'exported_original_basis_pair_not_numerically_accepted' and
            checked['schema'] == 'v126_original_native518_basis_pair_check' and
            checked['status'] == 'checked_original_basis_pair_fixed_FP32_gate_passed', 'Original export/check receipt differs')
    for row in (export, checked):
        require(row['script_sha256'] == ORIGINAL_WORKER_SHA and row['config_sha256'] == ORIGINAL_CONFIG_SHA and
                row['checkpoint'] == CHECKPOINT and row['original_model_only'] and
                row['checkpoint_deserialized'] is False and row['original36_parameter_lineage_required'] and
                row['checkpoint_and_original_source_lineage_verified'] and row['fixed_gate'] == GATE and
                row['strict_elementwise_allclose_is_independent_diagnostic'] and row['source_and_inputs_unchanged'] and
                row['affinity_restored'] and row['environment_restored'] and not row['cleanup_errors'], 'Original source/36-parameter identity or numeric gate differs')
    require(export['all36_original_learned_values_recovered_and_verified'] and checked['FP32_gate_passed'] and
            checked['all6_calls_completed'] and checked['ORT_sessions_dropped'] and checked['ORT_session_dropped'] and
            checked['original_numeric_gate_unchanged'] and checked['strict_allclose_is_diagnostic'], 'Original FP32 proof incomplete')
    require(export['models'] == checked['models'] and export['original_parameters'] == checked['original_parameters'] and
            export['source_hashes'] == checked['source_hashes'], 'Original model/source records disagree')
    require(set(export['original_parameters']) == {'frame', 'global'} and
            all(len(v) == 18 and all(p['dtype'] == 'float32' and digest_ok(p['raw_sha256']) for p in v.values())
                for v in export['original_parameters'].values()), 'Both original 18-parameter blocks required')
    require([m['key'] for m in export['models']] == ['A', 'F', 'B'], 'Original three variants required')
    for model in export['models']:
        require(model['input_bindings'] == [dict(index=0, name='input', shape=FRAME, dtype='float32')] and
                model['output_bindings'] == [dict(index=i, name=name, shape=shape, dtype='float32')
                    for i, (name, shape) in enumerate((('frame_output', FRAME), ('global_output', GLOBAL)))] and
                all(model[k] is True for k in ('original36_learned_values_verified', 'canonical_outputs_restored',
                    'original_K4_partition_and_sum_node_order_preserved')) and
                model['basis_persistent_frame_to_global'] is (model['key'] == 'B'), 'Original canonical bindings/K4/basis proof differs')
        if model['key'] != 'A':
            require({p['role'] for p in model['fold_proofs']} == {'frame', 'global'}, 'Both original fold proofs required')
            for proof in model['fold_proofs']:
                original = export['original_parameters'][proof['role']]
                require(proof['original_learned_values_recovered_and_verified'] == 18 and
                        set(proof['original_parameter_metadata']) == set(original) and
                        all(all(proof['original_parameter_metadata'][key][k] == value[k]
                            for k in ('shape', 'dtype', 'tensor_bytes', 'raw_sha256')) for key, value in original.items()), 'Original learned-value fold proof differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-receipt', type=Path, required=True)
    parser.add_argument('--capture-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    require(platform.machine() == 'aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),
            'Board-only reference worker; host NN execution forbidden')
    require(digest_ok(args.capture_sha256), 'Exact capture receipt SHA required')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    old_env, old_affinity = dict(os.environ), set(os.sched_getaffinity(0))
    started = time.monotonic()
    watched, handlers, session = {}, {}, None
    report = dict(schema='v128_original_native518_captured_pair_FP32_check', status='running',
        script_sha256=sha(__file__), actual_execution_on_board_only=True, original_model_only=True,
        checkpoint_deserialized=False, RKNN_GPU_or_quantization_executed=False, candidate_accepted=False,
        full_model_quality_performance_realtime_or_8GB_accepted=False, system_or_frequency_writes=False,
        reference_scope='Original CPU ONNX frame0/global0 on actual captured NPU DINO24 frame0 inputs; not a complete original Torch prediction or trajectory reference',
        fixed_gate=GATE, fixed_gate_definition='relative_l2 <= 1e-5 and max_abs <= 1e-3, unchanged R126/R98 numeric gate',
        strict_allclose_is_diagnostic=True, maximum_simultaneous_ORT_sessions=1,
        caller_owns_180_second_CPU_thermal_controller=True, checks=[], cleanup_errors=[], sessions_created=0, sessions_dropped=0,
        saved_array_count=0, CPU_forward_times_are_not_NPU_performance_metrics=True)

    def progress(stage):
        report.update(stage=stage, elapsed_seconds=time.monotonic() - started,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        atomic(out / 'result.json', report)
        print(stage, flush=True)

    def pin(path, digest, size=None):
        path = Path(path).resolve(strict=True)
        require(digest_ok(digest) and sha(path) == digest and (size is None or path.stat().st_size == size),
                'Pinned file differs: ' + str(path))
        require(path not in watched or watched[path] == digest, 'Conflicting source identity')
        watched[path] = digest
        return path

    def relative_path(directory, file):
        require(isinstance(file, str) and not Path(file).is_absolute() and '..' not in Path(file).parts, 'Receipt-relative artifact required')
        return directory / file

    def metadata(value):
        return dict(shape=list(value.shape), dtype=str(value.dtype), tensor_bytes=value.nbytes,
            finite=bool(np.isfinite(value).all()), raw_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())

    def compare(actual, expected):
        x, y = actual.astype(np.float64), expected.astype(np.float64)
        difference = x - y
        norm = float(np.linalg.norm(y.reshape(-1)))
        relative = float(np.linalg.norm(difference.reshape(-1)) / norm) if norm > 0 else None
        maximum = float(np.max(np.abs(difference)))
        outside = int(np.count_nonzero(np.abs(difference) > GATE['atol'] + GATE['rtol'] * np.abs(y)))
        passed = relative is not None and relative <= GATE['relative_l2_max'] and maximum <= GATE['max_abs_max']
        return dict(relative_l2=relative, reference_norm=norm, max_abs=maximum,
            outside_count=outside, strict_pass=outside == 0, byte_equal=actual.tobytes() == expected.tobytes(),
            fixed_numeric_gate_passed=passed, fixed_gate_passed=passed, zero_reference_norm=norm == 0)

    progress('preflight')
    try:
        def interrupted(number, frame):
            raise InterruptedError('Controller termination signal ' + str(number))
        for number in (signal.SIGTERM, signal.SIGINT):
            handlers[number] = signal.signal(number, interrupted)
        pin(__file__, report['script_sha256'])
        capture_file = pin(args.capture_receipt, args.capture_sha256)
        capture = json.loads(capture_file.read_text())
        validate_capture(capture)
        export_file = pin(ROOT / 'result.json', EXPORT_SHA)
        check_file = pin(CHECK_PATH, CHECK_SHA)
        export, checked = json.loads(export_file.read_text()), json.loads(check_file.read_text())
        validate_original_pair(export, checked)
        require(capture['source_hashes'] == export['source_hashes'], 'Original upstream/pair source identities differ')
        require(any(Path(path).name == 'capture_frame0_inputs_board.py' and digest == CAPTURE_SOURCE_SHA
                    for path, digest in capture['pinned_files'].items()), 'Actual frozen capture source file pin required')
        for path, digest in capture['pinned_files'].items():
            pin(path, digest)
        models = []
        for model in export['models']:
            row = model['onnx']
            models.append(pin(relative_path(export_file.parent, row['file']), row['sha256'], row['bytes']))
        inputs = []
        for record in capture['records']:
            row = record['input']
            inputs.append(pin(relative_path(capture_file.parent, row['file']), row['sha256'], row['bytes']))
        report.update(capture_receipt=dict(file=str(capture_file), sha256=args.capture_sha256, script_sha256=CAPTURE_SOURCE_SHA),
            export_receipt=dict(file=str(export_file), sha256=EXPORT_SHA),
            original_check_receipt=dict(file=str(check_file), sha256=CHECK_SHA),
            batch_index=capture['batch_index'], checkpoint=CHECKPOINT, records=copy.deepcopy(capture['records']),
            models=copy.deepcopy(export['models']), source_hashes=export['source_hashes'],
            original_parameters=export['original_parameters'], all36_original_learned_values_verified=True,
            captured_upstream_SDK_totals=copy.deepcopy(capture['SDK_totals']), capture_completion_and_cleanup_verified=True)
        require({4, 5, 6, 7} <= old_affinity, 'A76 CPUs4..7 required')
        os.sched_setaffinity(0, {4, 5, 6, 7})
        os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
        import numpy as np
        import onnxruntime as ort
        report.update(numpy_version=np.__version__, ort_version=ort.__version__, execution_provider='CPUExecutionProvider',
            intra_op_num_threads=1, inter_op_num_threads=1)
        progress('original_capture_and_graph_pins_verified')
        for record, input_file in zip(capture['records'], inputs):
            x = np.load(input_file, allow_pickle=False)
            require(x.dtype == np.float32 and list(x.shape) == FRAME and x.flags.c_contiguous and
                    metadata(x)['raw_sha256'] == record['input']['raw_sha256'] and np.isfinite(x).all() and
                    x.nbytes == record['input']['tensor_bytes'], 'Saved captured frame0 input differs')
            input_hash = metadata(x)['raw_sha256']
            reference = None
            for model, model_file in zip(export['models'], models):
                row = dict(input_id=record['id'], variant=model['key'], source_pair_indices=record['source_pair_indices'],
                    split=record['split'], outputs=[], comparisons_vs_A={})
                report['checks'].append(row)
                options = ort.SessionOptions()
                options.intra_op_num_threads = options.inter_op_num_threads = 1
                options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                try:
                    session = ort.InferenceSession(str(model_file), sess_options=options, providers=['CPUExecutionProvider'])
                    report['sessions_created'] += 1
                    require(session.get_providers() == ['CPUExecutionProvider'], 'CPU-only ORT required')
                    require([(v.name, v.shape, v.type) for v in session.get_inputs()] == [('input', FRAME, 'tensor(float)')] and
                            [(v.name, v.shape, v.type) for v in session.get_outputs()] ==
                                [('frame_output', FRAME, 'tensor(float)'), ('global_output', GLOBAL, 'tensor(float)')], 'Original session logical bindings differ')
                    begin = time.monotonic()
                    values = session.run(['frame_output', 'global_output'], {'input': x})
                    row['CPU_forward_seconds'] = time.monotonic() - begin
                    for name, shape, value in zip(('frame_output', 'global_output'), (FRAME, GLOBAL), values):
                        value = np.array(value, copy=True, order='C')
                        artifact = metadata(value)
                        file = out / (record['id'] + '-' + model['key'] + '-' + name + '.npy')
                        with file.open('xb') as stream:
                            np.save(stream, value, allow_pickle=False)
                        artifact.update(name=name, file=file.name, bytes=file.stat().st_size, sha256=sha(file))
                        row['outputs'].append(artifact)
                        report['saved_array_count'] += 1
                        watched[file] = artifact['sha256']
                        require(value.dtype == np.float32 and list(value.shape) == shape and artifact['finite'], 'Original canonical output shape/dtype/finiteness differs')
                    require(len(values) == 2 and metadata(x)['raw_sha256'] == input_hash, 'Output count or captured input changed')
                    row['input_unchanged'] = True
                    row['complete'] = True
                    if model['key'] == 'A':
                        reference = [np.array(value, copy=True, order='C') for value in values]
                    for name, value, expected in zip(('frame_output', 'global_output'), values, reference):
                        row['comparisons_vs_A'][name] = compare(value, expected)
                    row['fixed_gate_passed'] = all(v['fixed_gate_passed'] for v in row['comparisons_vs_A'].values())
                    del values, value
                finally:
                    if session is not None:
                        session = None
                        gc.collect()
                        report['sessions_dropped'] += 1
                progress(record['id'] + '_' + model['key'] + '_saved_and_session_dropped')
            del x, reference
        report.update(all6_calls_completed=len(report['checks']) == 6 and all(v.get('complete') for v in report['checks']),
            FP32_gate_passed=all(v['fixed_gate_passed'] for v in report['checks']),
            canonical_outputs_restored=True, original_numeric_gate_unchanged=True)
        require(report['all6_calls_completed'] and report['saved_array_count'] == 12, 'All six original CPU pair calls and 12 outputs required')
        report['status'] = SUCCESS if report['FP32_gate_passed'] else 'failed_fixed_FP32_numeric_gate'
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        if session is not None:
            session = None
            gc.collect()
            report['sessions_dropped'] += 1
        try:
            os.sched_setaffinity(0, old_affinity)
        except BaseException as error:
            report['cleanup_errors'].append('affinity: ' + str(error))
        os.environ.clear()
        os.environ.update(old_env)
        for number, handler in handlers.items():
            signal.signal(number, handler)
        try:
            report['source_and_inputs_unchanged'] = all(path.is_file() and sha(path) == digest for path, digest in watched.items())
        except BaseException as error:
            report['source_and_inputs_unchanged'] = False
            report['cleanup_errors'].append('Final identity verification: ' + str(error))
        report.update(affinity_restored=set(os.sched_getaffinity(0)) == old_affinity,
            environment_restored=dict(os.environ) == old_env,
            ORT_sessions_dropped=session is None and report['sessions_created'] == report['sessions_dropped'],
            pinned_files={str(path): digest for path, digest in watched.items()},
            saved_array_bytes=sum(v['bytes'] for row in report['checks'] for v in row['outputs']))
        if report['cleanup_errors'] or not all(report[k] for k in ('source_and_inputs_unchanged', 'affinity_restored', 'environment_restored', 'ORT_sessions_dropped')):
            report['status'] = 'failed_cleanup_or_identity'
        progress('finished')
    return 0 if report['status'] == SUCCESS else 1


if __name__ == '__main__':
    raise SystemExit(main())
