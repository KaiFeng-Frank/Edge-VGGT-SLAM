#!/usr/bin/env python3
"""Board-only W8A8 PTQ of the unchanged original 62-parameter DPT logits graph."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import traceback

BASE = Path('/mnt/vggt-data/vggt-slam2')
E = BASE / 'experiments'
PARENT_SHA = '9a6e5435df543b8484a4fc50da4b065995f32153233548829234dd149fbc7135'
EXPORT_SHA = '649a09ea323a2e7caa067c278f6f97c8ce13082689add2545ffd4653d592f292'
GRAPH_SHA = '5f885c944f6ceff2c3eb8f5030249b90a85f5040a9a5ba892f74b95803af2a5b'
GRAPH_BYTES = 135995381
CALIBRATION_SHA = 'a1e6f1b2a89e1a96987befdea4f0a048aaf06066efa9fff72a370ac77d45507c'
TAIL_CAPTURE_SHA = '4b723e499763bb90565f6e11e16c551664634aaf72da46c7cdc493b84208ef67'
DEPTH_CAPTURE_SHA = 'fa33dc6a05c2e286ca9c9a17c0f62df21ee2b19da0f3ea693fcff3718bc99033'
PRIOR_CHECK_SHA = '0e32292723b9b97dd5c177e08ec0579ca59b312d23634f3215b8e574b27c143b'
CHECKPOINT = dict(bytes=5026874952, sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')
CONFIG = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
    mean_values=None, std_values=None, quant_img_RGB2BGR=False, model_pruning=False,
    remove_reshape=False, quantize_weight=False, enable_flash_attention=False,
    quantized_dtype='w8a8', quantized_algorithm='normal', quantized_method='channel')
INPUT_NAMES = ['tokens_4', 'tokens_11', 'tokens_17', 'tokens_23']
INPUT_SHAPE = [1, 48, 2048]
OUTPUT_SHAPE = [1, 2, 84, 112]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def pin(path, digest, watched, size=None):
    path = Path(path).resolve(strict=True)
    require(len(digest) == 64 and sha(path) == digest
        and (size is None or path.stat().st_size == size), 'Frozen file differs: ' + str(path))
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
    parent_path = pin(args.parent_manifest, PARENT_SHA, watched)
    parent = json.loads(parent_path.read_text())
    export_path = pin(parent_path.parent / 'export.json', EXPORT_SHA, watched)
    exported = json.loads(export_path.read_text())
    require(parent['schema'] == 'v54_original_VGGT_depth_logits_RKNN_manifest'
        and parent['status'] == 'compiled_not_quality_accepted' and parent['checkpoint'] == CHECKPOINT
        and parent['export_manifest_sha256'] == EXPORT_SHA
        and exported['schema'] == 'v54_original_VGGT_depth_logits_ONNX_manifest'
        and exported['checkpoint'] == CHECKPOINT, 'Original depth graph ancestry required')
    for key in ['graph', 'parameters', 'positions', 'architecture', 'source_hashes']:
        require(parent[key] == exported[key], 'Original export/compiled metadata differs: ' + key)
    require(len(parent['parameters']) == 62 and len(parent['positions']) == 5
        and parent['architecture']['layers'] == [4, 11, 17, 23]
        and parent['architecture']['frames_chunk_size'] == 1
        and parent['all_62_original_parameter_values_preserved_in_graph'] is True
        and parent['original_inplace_relu_residual_skip_preserved'] is True
        and parent['all_five_original_align_corners_resizes_preserved'] is True,
        'All original DPT parameters, spatial operators and positions required')
    prior_config = {key: value for key, value in CONFIG.items() if not key.startswith('quantized_')}
    require(parent['configuration'] == prior_config, 'Parent FP16 configuration differs')
    g = parent['graph']
    require(g['sha256'] == GRAPH_SHA and g['bytes'] == GRAPH_BYTES
        and g['input_names'] == INPUT_NAMES and g['input_shapes'] == {name: INPUT_SHAPE for name in INPUT_NAMES}
        and g['output_name'] == 'logits' and g['output_shape'] == OUTPUT_SHAPE, 'Original graph IO differs')
    graph = artifact(parent_path.parent, g, watched)
    prior_check_path = pin(args.prior_fp32_check, PRIOR_CHECK_SHA, watched)
    prior_check = json.loads(prior_check_path.read_text())
    require(prior_check['schema'] == 'v54_original_depth_FP32_ORT_check'
        and prior_check['status'] == 'passed_original_FP32_logit_gate'
        and prior_check['export_sha256'] == EXPORT_SHA and prior_check['checkpoint'] == CHECKPOINT
        and prior_check['capture_sha256'] == DEPTH_CAPTURE_SHA
        and prior_check['rtol'] == 1e-4 and prior_check['atol'] == 1e-5
        and len(prior_check['calls']) == 2
        and all(row['fixed_gate_pass'] is True for row in prior_check['calls']),
        'Pinned previously passed original FP32 depth ONNX gate required')
    capture_path = pin(args.capture_dir / 'capture.json', args.capture_sha256, watched)
    capture_source = pin(args.capture_source, args.capture_source_sha256, watched)
    captured = json.loads(capture_path.read_text())
    calibration_path = pin(args.calibration_dir / 'capture.json', CALIBRATION_SHA, watched)
    calibration = json.loads(calibration_path.read_text())
    tail_path = pin(args.original_tail_dir / 'capture.json', TAIL_CAPTURE_SHA, watched)
    expected_parameters = {name: dict(value, finite=True) for name, value in parent['parameters'].items()}
    require(captured['schema'] == 'v85_original_VGGT_depth_PTQ_calibration_capture'
        and captured['status'] == 'completed_original_FP32_depth_calibration_capture'
        and captured['script_sha256'] == args.capture_source_sha256 and captured['checkpoint'] == CHECKPOINT
        and captured['calibration_capture']['sha256'] == CALIBRATION_SHA
        and captured['original_tail_capture']['sha256'] == TAIL_CAPTURE_SHA
        and captured['original_depth_capture']['sha256'] == DEPTH_CAPTURE_SHA
        and captured['original_depth_manifest_sha256'] == PARENT_SHA
        and captured['architecture'] == parent['architecture']
        and captured['positions'] == parent['positions']
        and captured['parameters'] == expected_parameters and len(captured['samples']) == 8,
        'Completed original eight-pair, 62-parameter depth capture required')
    for key in ['original_model_only', 'hooks_removed', 'model_state_unchanged',
            'source_and_inputs_unchanged', 'affinity_restored', 'environment_restored',
            'all_depth_and_conf_bitwise_equal_to_original_capture']:
        require(captured[key] is True, 'Required capture proof missing: ' + key)
    require(captured['NPU_or_cache_installed'] is False and captured['cleanup_errors'] == [],
        'Original CPU-only capture with clean restore required')
    require(calibration['status'] == 'completed_original_FP32_calibration_capture'
        and calibration['checkpoint'] == CHECKPOINT and len(calibration['samples']) == 8
        and calibration['full_forward_completed'] == 8 and calibration['cleanup_errors'] == [],
        'Original eight-pair FP32 calibration ancestry required')
    require(captured['source_hashes'], 'Capture source map required')
    for relative, digest in captured['source_hashes'].items():
        require(relative.startswith('third_party/vggt/vggt/')
            and calibration['source_hashes'][relative] == digest, 'Original source lineage differs')
        pin(args.source_root / relative, digest, watched)
    for relative, digest in parent['source_hashes'].items():
        pin(args.source_root / relative, digest, watched)
        if relative.startswith('third_party/'):
            require(captured['source_hashes'][relative] == digest, 'Depth source/export identity differs')
    require(set(captured['runtime_source_hashes']) == {'vggt_slam/runtime.py', 'vggt_slam/checkpoint_identity.py'},
        'Original pinned loader sources required')
    for relative, digest in captured['runtime_source_hashes'].items():
        require(calibration['source_hashes'][relative] == digest, 'Original loader changed')
        pin(args.source_root / relative, digest, watched)

    def load(row, shape):
        path = artifact(capture_path.parent, row, watched)
        value = np.load(path, allow_pickle=False)
        require(value.dtype == np.float32 and list(value.shape) == shape and value.flags.c_contiguous
            and np.isfinite(value).all() and row['dtype'] == 'float32' and row['shape'] == shape
            and row['finite'] is True and row['tensor_bytes'] == value.nbytes
            and hashlib.sha256(memoryview(value).cast('B')).hexdigest() == row['raw_sha256'],
            'Captured original FP32 tensor differs')
        return path

    rows = []
    for i, (sample, original) in enumerate(zip(captured['samples'], calibration['samples'])):
        require(sample['sample_id'] == original['sample_id'] == 'calib%02d' % i
            and sample['source_pair_indices'] == original['indices'] == [i*96, i*96+48]
            and sample['images'] == original['images'] and len(sample['frames']) == 2,
            'Eight ordered original calibration image pairs required')
        for image in sample['images']:
            pin(image['file'], image['sha256'], watched, image['bytes'])
        load(sample['depth'], [2, 84, 112, 1])
        load(sample['depth_conf'], [2, 84, 112])
        for name in ['depth', 'depth_conf']:
            require(sample[name]['raw_sha256'] == original['canonical_outputs'][name]['raw_sha256'],
                'Depth/conf reference is not the original complete forward')
        for frame_index, frame in enumerate(sample['frames']):
            require(frame['index'] == frame_index
                and [item['name'] for item in frame['inputs']] == INPUT_NAMES,
                'Original frame order and all four token layers required')
            paths = [str(load(item['artifact'], INPUT_SHAPE)) for item in frame['inputs']]
            load(frame['raw_logits'], OUTPUT_SHAPE)
            rows.append(dict(sample_id=sample['sample_id'], source_pair_indices=sample['source_pair_indices'],
                frame=frame_index, inputs=frame['inputs'], paths=paths))
    require(len(rows) == 16 and all(not any(ch.isspace() for ch in path)
        for row in rows for path in row['paths']), '16 four-input, whitespace-free dataset rows required')
    source_dataset = artifact(capture_path.parent, captured['dataset'], watched)
    expected = ''.join(' '.join(row['paths']) + '\n' for row in rows)
    require(captured['dataset']['line_count'] == 16 and captured['dataset']['input_count'] == 4
        and source_dataset.read_text() == expected, 'Capture dataset order/identity differs')
    dataset = args.output_dir / 'dataset.txt'
    dataset.write_text(expected)
    watched[dataset] = sha(dataset)
    report.update(checkpoint=CHECKPOINT, original_depth_manifest_sha256=PARENT_SHA,
        prior_FP16_compiled_manifest_sha256=PARENT_SHA,
        prior_FP32_check_receipt_sha256=PRIOR_CHECK_SHA,
        original_export_sha256=EXPORT_SHA, graph=dict(file=str(graph), bytes=GRAPH_BYTES, sha256=GRAPH_SHA),
        graph_sha256=GRAPH_SHA, original_ONNX_modified=False,
        original_parameters=expected_parameters, original_positions=parent['positions'],
        original_architecture=parent['architecture'], original_activate_head_CPU_unchanged=True,
        source_hashes=captured['source_hashes'], runtime_source_hashes=captured['runtime_source_hashes'],
        capture_source=dict(file=str(capture_source), sha256=args.capture_source_sha256),
        calibration=dict(capture=dict(file=str(capture_path), bytes=capture_path.stat().st_size, sha256=args.capture_sha256),
            source_calibration_sha256=CALIBRATION_SHA, original_tail_capture_sha256=TAIL_CAPTURE_SHA,
            pair_count=8, frames_per_pair=2, samples=captured['samples'],
            dataset=dict(file='dataset.txt', bytes=dataset.stat().st_size, sha256=sha(dataset),
                input_count=4, line_count=16, rows=rows)),
        input_bindings=[dict(index=i, name=name, shape=INPUT_SHAPE, dtype='float32') for i, name in enumerate(INPUT_NAMES)],
        output_bindings=[dict(index=0, name='logits', shape=OUTPUT_SHAPE, dtype='float32')],
        configuration=CONFIG, build_options=dict(do_quantization=True, dataset=str(dataset), auto_hybrid=False))
    return graph, dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=BASE / 'src')
    p.add_argument('--parent-manifest', type=Path, default=E / 'vggt-v54-depth-graphs-r1/compiled.json')
    p.add_argument('--calibration-dir', type=Path, default=E / 'vggt-v56-calibration-r1')
    p.add_argument('--original-tail-dir', type=Path, default=E / 'vggt-v53-capture-r1')
    p.add_argument('--prior-fp32-check', type=Path, default=E / 'vggt-v54-depth-ort-r1/check.json')
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
    report = dict(schema='v85_original_DPT_depth_logits_INT8_compile', status='running',
        script_sha256=sha(__file__), capture_sha256=args.capture_sha256,
        original_model_only=True, model_inference_executed=False, ORT_rerun=False,
        candidate_accepted=False, quality_accepted=False, realtime_validated=False,
        calibration_executed=False, operations=[], released=False, release_code=None, cleanup_errors=[],
        thermal_guard_external=True, thermal_pause_celsius=70, thermal_resume_celsius=63,
        thermal_stop_celsius=75, expected_NPU_hz=300000000,
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
            ('export_rknn', lambda: sdk.export_rknn(str(args.output_dir / 'depth-int8.rknn')))]
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
        output = args.output_dir / 'depth-int8.rknn'
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
