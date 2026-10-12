#!/usr/bin/env python3
"""Board-only W8A8 PTQ of the unchanged original split4 CameraHead step graph."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import time
import traceback

BASE = Path('/mnt/vggt-data/vggt-slam2')
E = BASE / 'experiments'
GRAPH_SHA = 'f1b9eee30144ec643aa351db724c2f740a79d82a8f9df5468f4338e377600a7f'
GRAPH_BYTES = 864771004
CAMERA_ADAPTER_SHA = 'e20518d08c47746e1bf58e07bd6d0270bc7c7f6daa4d0858593f478ce6dfb09e'
EXPORT_SHA = '8ccf7ff593346962766b4eae6da7eb95ce43eeaa3183a8fc0e82bf134c172503'
PRIOR_COMPILE_SHA = 'a2cdad4a6d2af60c6b3244dfefb3d1a1fadead72c6b2374ef2c089912d318f74'
PRIOR_CHECK_SHA = '9c58744a713a8ad686e66e811163c260e82f6522f0a936dbcc92cb5fdfdc4c2a'
CALIBRATION_SHA = 'a1e6f1b2a89e1a96987befdea4f0a048aaf06066efa9fff72a370ac77d45507c'
TAIL_CAPTURE_SHA = '4b723e499763bb90565f6e11e16c551664634aaf72da46c7cdc493b84208ef67'
CAPTURE_SOURCE_SHA = '0337c6fbdb2ea3d70073f6cd08e8e6f8f230c4284b198ecdf6dfa48f0eaf00e8'
CHECKPOINT = dict(bytes=5026874952, sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')
CONFIG = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
    mean_values=None, std_values=None, quant_img_RGB2BGR=False, model_pruning=False,
    remove_reshape=False, quantize_weight=False, enable_flash_attention=False,
    quantized_dtype='w8a8', quantized_algorithm='normal', quantized_method='channel')
INPUTS = [('camera_tokens', [1, 2, 2048]), ('conditioning_pose', [1, 2, 9])]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            value.update(block)
    return value.hexdigest()


def pin(path, digest, watched, size=None):
    path = Path(path).resolve(strict=True)
    require(sha(path) == digest and (size is None or path.stat().st_size == size),
        'Frozen file identity differs: ' + str(path))
    watched[path] = digest
    return path


def artifact(directory, row, watched):
    root, relative = Path(directory).resolve(strict=True), Path(row['file'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'Relative artifact required')
    path = (root / relative).resolve(strict=True)
    path.relative_to(root)
    return pin(path, row['sha256'], watched, row['bytes'])


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare(args, report, watched):
    import numpy as np
    source = pin(args.source_root / 'vggt_slam/rknn_camera_head.py', CAMERA_ADAPTER_SHA, watched)
    spec = importlib.util.spec_from_file_location('_v82_original_camera_provenance', source)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)  # Only pure provenance helpers; no model/context construction.
    original_export = pin(args.source_root / 'deploy/rk3588/validation/vggt-v16-camera-step-export112-r1.json', EXPORT_SHA, watched)
    exported = json.loads(original_export.read_text())
    require(exported['status'] == 'exported_and_structurally_checked'
        and exported['source_and_inputs_unchanged'] is True and exported['cleanup_errors'] == []
        and exported['camera_iterations'] == exported['trunk_blocks_per_step'] == 4
        and exported['capture']['checkpoint']['sha256'] == CHECKPOINT['sha256']
        and exported['capture']['checkpoint']['bytes'] == CHECKPOINT['bytes'], 'Original camera-step export required')
    parameters = exported['weight_sharing']['parameters']
    require(len(parameters) == exported['weight_sharing']['original_parameters_verified'] == 68
        and len({x['parameter'] for x in parameters}) == 68
        and exported['weight_sharing']['weights_not_duplicated_per_iteration'] is True,
        'Original 68 graph parameters required')
    expected_parameters = {x['parameter']: dict(shape=x['shape'], dtype='float32',
        tensor_bytes=int(np.prod(x['shape']))*4, finite=True, raw_sha256=x['raw_sha256']) for x in parameters}
    empty = exported['outer_iteration_contract']['empty_weight']
    require(empty['shape'] == [1, 1, 9] and empty['dtype'] == 'float32' and empty['finite'] is True
        and exported['outer_iteration_contract']['all_four_original_iterations_array_equal'] is True,
        'Original learned empty token and four FP32 iterations required')
    expected_parameters['empty_pose_tokens'] = dict(shape=empty['shape'], dtype='float32',
        tensor_bytes=36, finite=True, raw_sha256=empty['sha256'])
    weights_for_parent_gate = {name: dict(sha256=value['raw_sha256']) for name, value in expected_parameters.items()}
    adapter._split4_artifact(args.source_root, weights_for_parent_gate)
    baseline_dir = E / 'vggt-v69-camera-step-split4-r1'
    prior_compile = pin(baseline_dir / 'result.json', PRIOR_COMPILE_SHA, watched)
    prior_check = pin(E / 'vggt-v69-head-ort-r1/result.json', PRIOR_CHECK_SHA, watched)
    validation = args.source_root / 'deploy/rk3588/validation/original_camera_partition'
    for name, digest in [('compile_camera_step_split_board.py', adapter.SPLIT_BUILDER_SHA),
            ('compile_camera_fc1_splits_board.py', adapter.SPLIT_HELPER_SHA),
            ('probe_camera_step_split_board.py', adapter.SPLIT_PROBE_SHA)]:
        pin(validation / name, digest, watched)
    graph = pin(baseline_dir / 'camera-step-split4.onnx', GRAPH_SHA, watched, GRAPH_BYTES)
    capture_path = pin(args.capture_dir / 'capture.json', args.capture_sha256, watched)
    require(args.capture_source_sha256 == CAPTURE_SOURCE_SHA, 'Frozen capture source required')
    capture_source = pin(args.capture_source, CAPTURE_SOURCE_SHA, watched)
    captured = json.loads(capture_path.read_text())
    prior_calibration = pin(E / 'vggt-v56-calibration-r1/capture.json', CALIBRATION_SHA, watched)
    tail_capture = pin(E / 'vggt-v53-capture-r1/capture.json', TAIL_CAPTURE_SHA, watched)
    original_calibration = json.loads(prior_calibration.read_text())
    require(captured['schema'] == 'v82_original_VGGT_camera_PTQ_calibration_capture'
        and captured['status'] == 'completed_original_FP32_camera_calibration_capture'
        and captured['script_sha256'] == args.capture_source_sha256 and captured['checkpoint'] == CHECKPOINT
        and captured['calibration_capture']['sha256'] == CALIBRATION_SHA
        and captured['original_tail_capture']['sha256'] == TAIL_CAPTURE_SHA
        and captured['parameters'] == expected_parameters and len(captured['samples']) == 8
        and captured['full_head_forward_count'] == 8 and captured['original_model_only'] is True
        and captured['conditioning_count'] == captured['delta_count'] == 32
        and captured['tail_blocks_forward_count'] == dict(frame23=8, global23=8)
        and captured['NPU_or_cache_installed'] is False and captured['model_state_unchanged'] is True
        and captured['all_final_poses_bitwise_equal_to_original_capture'] is True
        and captured['source_and_inputs_unchanged'] is True and captured['affinity_restored'] is True
        and captured['cleanup_errors'] == [], 'Completed original 69-parameter/eight-pair camera capture required')
    require(original_calibration['status'] == 'completed_original_FP32_calibration_capture'
        and original_calibration['checkpoint'] == CHECKPOINT and len(original_calibration['samples']) == 8
        and original_calibration['full_forward_completed'] == 8 and original_calibration['cleanup_errors'] == [],
        'Original eight-pair FP32 calibration lineage required')
    original_sources = exported['source_and_input_hashes']
    require(captured['source_hashes'], 'Original capture source map required')
    for relative, digest in captured['source_hashes'].items():
        require(relative.startswith('third_party/vggt/vggt/')
            and original_calibration['source_hashes'][relative] == digest, 'Original third-party source lineage differs')
        path = pin(args.source_root / relative, digest, watched)
        if str(path) in original_sources:
            require(original_sources[str(path)] == digest, 'Camera export/capture source differs')
    require(captured['source_hashes']['third_party/vggt/vggt/heads/camera_head.py']
        == original_sources[str(args.source_root / 'third_party/vggt/vggt/heads/camera_head.py')], 'Original CameraHead source required')
    for relative, digest in captured['runtime_source_hashes'].items():
        require(relative in ('vggt_slam/runtime.py', 'vggt_slam/checkpoint_identity.py'), 'Only original loading/runtime sources allowed')
        pin(args.source_root / relative, digest, watched)
    require(set(captured['runtime_source_hashes']) == {'vggt_slam/runtime.py', 'vggt_slam/checkpoint_identity.py'},
        'Original checkpoint loader source identities required')
    def load(row, shape):
        path = artifact(capture_path.parent, row, watched)
        value = np.load(path, allow_pickle=False)
        require(value.dtype == np.float32 and list(value.shape) == shape and value.flags.c_contiguous
            and np.isfinite(value).all() and row['dtype'] == 'float32' and row['shape'] == shape
            and row['tensor_bytes'] == value.nbytes
            and hashlib.sha256(memoryview(value).cast('B')).hexdigest() == row['raw_sha256'],
            'Captured original FP32 tensor identity differs')
        return path, value
    dataset_rows = []
    for i, (sample, original) in enumerate(zip(captured['samples'], original_calibration['samples'])):
        require(sample['sample_id'] == original['sample_id'] == 'calib%02d' % i
            and sample['source_pair_indices'] == original['indices'] == [i*96, i*96+48]
            and sample['images'] == original['images'] and len(sample['iterations']) == 4,
            'Actual original calibration pair identity differs')
        for image in sample['images']:
            pin(image['file'], image['sha256'], watched, image['bytes'])
        token_path, _ = load(sample['camera_tokens'], INPUTS[0][1])
        prediction = None
        for iteration, step in enumerate(sample['iterations']):
            require(step['index'] == iteration, 'Exactly four ordered original refinements required')
            condition_path, conditioning = load(step['conditioning_pose'], INPUTS[1][1])
            _, delta = load(step['delta'], [1, 2, 9])
            _, activated = load(step['activated_pose'], [1, 2, 9])
            if iteration == 0:
                require(conditioning[0, 0].tobytes() == conditioning[0, 1].tobytes()
                    and hashlib.sha256(conditioning[0, 0].tobytes()).hexdigest() == empty['sha256'],
                    'First conditioning is not the original learned empty pose')
                prediction = delta.copy()
            else:
                require(conditioning.tobytes() == prediction.tobytes(), 'Later conditioning must be previous unactivated FP32 prediction')
                prediction = np.add(prediction, delta, dtype=np.float32)
            if iteration == 3:
                require(step['activated_pose']['raw_sha256'] == original['canonical_outputs']['pose_enc']['raw_sha256'],
                    'Captured final pose differs from original full forward')
            dataset_rows.append(dict(sample_id=sample['sample_id'], source_pair_indices=sample['source_pair_indices'],
                iteration=iteration, inputs=[sample['camera_tokens'], step['conditioning_pose']],
                paths=[str(token_path), str(condition_path)]))
    require(len(dataset_rows) == 32 and all(not any(ch.isspace() for ch in path)
        for row in dataset_rows for path in row['paths']), 'Exactly 32 two-input whitespace-free calibration rows required')
    dataset = args.output_dir / 'dataset.txt'
    dataset.write_text(''.join(' '.join(row['paths']) + '\n' for row in dataset_rows))
    watched[dataset] = sha(dataset)
    report.update(checkpoint=CHECKPOINT, graph=dict(file=str(graph), bytes=GRAPH_BYTES, sha256=GRAPH_SHA),
        graph_sha256=GRAPH_SHA, original_ONNX_modified=False,
        original_export_sha256=EXPORT_SHA, prior_FP16_compile_receipt_sha256=PRIOR_COMPILE_SHA,
        prior_FP32_check_receipt_sha256=PRIOR_CHECK_SHA, original_parameters=captured['parameters'],
        required_outer_iterations=4, trunk_blocks_per_step=4, original_empty_and_FP32_outer_accumulation_unchanged=True,
        source_hashes=captured['source_hashes'], runtime_source_hashes=captured['runtime_source_hashes'],
        capture_source=dict(file=str(capture_source), sha256=args.capture_source_sha256),
        calibration=dict(capture=dict(file=str(capture_path), bytes=capture_path.stat().st_size, sha256=args.capture_sha256),
            source_calibration_sha256=CALIBRATION_SHA, original_tail_capture_sha256=TAIL_CAPTURE_SHA,
            pair_count=8, iterations_per_pair=4, samples=captured['samples'],
            dataset=dict(file='dataset.txt', bytes=dataset.stat().st_size, sha256=sha(dataset), input_count=2, line_count=32, rows=dataset_rows)),
        input_bindings=[dict(index=i, name=name, shape=shape, dtype='float32') for i, (name, shape) in enumerate(INPUTS)],
        output_bindings=[dict(index=0, name='pose_delta', shape=[1, 2, 9], dtype='float32')],
        configuration=CONFIG, build_options=dict(do_quantization=True, dataset=str(dataset), auto_hybrid=False))
    return graph, dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=BASE / 'src')
    p.add_argument('--capture-dir', type=Path, required=True)
    p.add_argument('--capture-sha256', required=True)
    p.add_argument('--capture-source', type=Path, required=True)
    p.add_argument('--capture-source-sha256', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--controller-source', type=Path)
    p.add_argument('--controller-source-sha256')
    args = p.parse_args()
    if bool(args.controller_source) != bool(args.controller_source_sha256):
        p.error('Controller source/path SHA must be provided together')
    args.source_root = args.source_root.resolve(strict=True)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    old_cwd, old_tmp, old_affinity = Path.cwd(), os.environ.get('TMPDIR'), os.sched_getaffinity(0)
    watched, sdk, started, failure = {}, None, time.monotonic(), None
    report = dict(schema='v82_original_CameraHead_step_split4_INT8_compile', status='running',
        script_sha256=sha(__file__), capture_sha256=args.capture_sha256,
        original_model_only=True, model_inference_executed=False, ORT_rerun=False,
        candidate_accepted=False, quality_accepted=False, realtime_validated=False,
        calibration_executed=False, operations=[], released=False, release_code=None, cleanup_errors=[],
        thermal_guard_external=True, thermal_stop_celsius=75, expected_NPU_hz=300000000,
        external_controller_receipt_finalized_after_child_exit=True)
    try:
        require(platform.machine().lower() in ('aarch64', 'arm64')
            and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(), 'RK3588S board only')
        pin(__file__, report['script_sha256'], watched)
        if args.controller_source:
            controller = pin(args.controller_source, args.controller_source_sha256, watched)
            report['controller_source'] = dict(file=str(controller), sha256=args.controller_source_sha256)
        os.sched_setaffinity(0, {4, 6})
        report['compile_cpus'] = [4, 6]
        graph, dataset = prepare(args, report, watched)
        os.chdir(args.output_dir)
        os.environ['TMPDIR'] = str(args.output_dir)
        report['toolkit_version'] = importlib.metadata.version('rknn-toolkit2')
        require(report['toolkit_version'] == '2.3.2', 'Frozen RKNN Toolkit2 2.3.2 required')
        write(args.output_dir / 'result.json', report)
        from rknn.api import RKNN
        sdk = RKNN(verbose=True, verbose_file=str(args.output_dir / 'sdk.log'))
        calls = [('config', lambda: sdk.config(**CONFIG)),
            ('load_onnx', lambda: sdk.load_onnx(model=str(graph))),
            ('build', lambda: sdk.build(do_quantization=True, dataset=str(dataset), auto_hybrid=False)),
            ('export_rknn', lambda: sdk.export_rknn(str(args.output_dir / 'camera-step-int8.rknn')))]
        for name, function in calls:
            operation = dict(name=name, status='running', code=None)
            report['operations'].append(operation)
            if name == 'build':
                report['calibration_build_api_invoked'] = True
                report['calibration_may_have_executed'] = True
            write(args.output_dir / 'result.json', report)
            tick = time.monotonic()
            try:
                code = function()
                operation['code'] = code
                require(code in (None, 0), name + ' failed: ' + str(code))
                operation['status'] = 'completed'
                if name == 'build':
                    report['calibration_executed'] = True
            except BaseException:
                operation['status'] = 'failed'
                raise
            finally:
                operation['seconds'] = time.monotonic()-tick
                write(args.output_dir / 'result.json', report)
        output = args.output_dir / 'camera-step-int8.rknn'
        require(output.stat().st_size > 0, 'Empty RKNN output')
        report.update(status='compiled_not_quality_accepted',
            rknn=dict(file=output.name, bytes=output.stat().st_size, sha256=sha(output)))
    except BaseException as error:
        failure = error
        report.update(status='failed', error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc())
    finally:
        if sdk is not None:
            try:
                code = sdk.release()
                report['release_code'] = code
                require(code in (None, 0), 'release failed: ' + str(code))
                report['released'] = True
            except BaseException as error:
                report['cleanup_errors'].append('release: ' + str(error))
        for name, action in [('cwd', lambda: os.chdir(old_cwd)), ('affinity', lambda: os.sched_setaffinity(0, old_affinity))]:
            try:
                action()
            except BaseException as error:
                report['cleanup_errors'].append(name + ': ' + str(error))
        try:
            if old_tmp is None:
                os.environ.pop('TMPDIR', None)
            else:
                os.environ['TMPDIR'] = old_tmp
            report['environment_restored'] = Path.cwd() == old_cwd and os.environ.get('TMPDIR') == old_tmp
            report['affinity_restored'] = os.sched_getaffinity(0) == old_affinity
        except BaseException as error:
            report['environment_restored'] = report['affinity_restored'] = False
            report['cleanup_errors'].append('restore verification: ' + str(error))
        try:
            report['source_and_inputs_unchanged'] = all(sha(path) == digest for path, digest in watched.items())
            report['pinned_files'] = {str(path): digest for path, digest in watched.items()}
        except BaseException as error:
            report['source_and_inputs_unchanged'] = False
            report['cleanup_errors'].append('pins: ' + str(error))
        if report['cleanup_errors'] or not report['environment_restored'] or not report['affinity_restored'] or not report['source_and_inputs_unchanged']:
            report['status'] = 'failed_cleanup'
        report['seconds'] = time.monotonic()-started
        try:
            if (args.output_dir / 'sdk.log').exists():
                report['sdk_log_sha256'] = sha(args.output_dir / 'sdk.log')
        except BaseException as error:
            report['cleanup_errors'].append('SDK log hash: ' + str(error))
            report['status'] = 'failed_cleanup'
        write(args.output_dir / 'result.json', report)
    if failure is not None:
        raise failure
    require(report['status'] == 'compiled_not_quality_accepted', 'Compilation cleanup/identity failed')
    print(json.dumps(dict(status=report['status'], result_sha256=sha(args.output_dir / 'result.json'))), flush=True)


if __name__ == '__main__':
    main()
