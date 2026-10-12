#!/usr/bin/env python3
"""Compile original pair00's gated Q/K layout with the original eight W8A8 inputs.

Route A is the pinned existing original W8A8 artifact. Only route B is built.
No original-model construction, tracing or ORT rerun is performed here.
"""
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
PARENT_SHA = '1872f2ed730f4b6d139562d831661126a088b1ac758249d6a1688cd7e9bfae42'
GRAPH_SHA = 'efbbaee8f3d6326565930e5a045eac0d500fdb35453fdf85ddac7481ec2fac65'
GRAPH_BYTES = 100914761
CAPTURE_SHA = 'a1e6f1b2a89e1a96987befdea4f0a048aaf06066efa9fff72a370ac77d45507c'
BASELINE_SHA = 'bdfe3757a7b4adef891dd622525d8a70ba948e9c0af9d0c1d9fffed41a74ea91'
BASELINE_LEAF_SHA = '46e83eb3d2b0e3ddfe7176d0d7de456190b53c48a9315a50bf0ea551315c9f8f'
BASELINE_GRAPH_ANCESTOR_SHA = '2e96b1d0187a69dbac7672574d07ff38a479b78923d6e5640e57e661bb0e0471'
BASELINE_MODEL_SHA = '9264eec3c2ad6d09663a34238a4d4169c94ec913a4be713affa8982c019ab3ba'
DATASET_SHA = '34789c52e8f625fe966cefab16ac3046c3a72ab71051ccd3e14e7d9374b12696'
CHECKPOINT = dict(bytes=5026874952, sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')
CONFIG = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
    mean_values=None, std_values=None, quant_img_RGB2BGR=False, model_pruning=False,
    remove_reshape=False, quantize_weight=False, enable_flash_attention=False,
    quantized_dtype='w8a8', quantized_algorithm='normal', quantized_method='channel')
INPUT_SHAPE = [2, 53, 1024]
OUTPUTS = [dict(index=0, name='frame_output', shape=[2, 53, 1024], dtype='float32'),
    dict(index=1, name='global_output', shape=[1, 106, 1024], dtype='float32')]
PROOF_FLAGS = ['all_original_initializers_protobuf_unchanged',
    'all_original_parameter_values_byte_equal', 'all_unrelated_nodes_protobuf_unchanged',
    'original_IO_protobuf_unchanged', 'global0_protobuf_unchanged',
    'original_qk_affine_and_LayerNorm_attributes_preserved',
    'original_RoPE_values_and_rotation_preserved', 'all_original_tokens_and_attention_reductions_preserved']


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


def bounded(directory, relative):
    root, relative = Path(directory).resolve(strict=True), Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Relative artifact required')
    path = (root / relative).resolve(strict=True)
    path.relative_to(root)
    return path


def artifact(directory, row, watched):
    return pin(bounded(directory, row['file']), row['sha256'], watched, row['bytes'])


def metadata(path):
    return dict(file=str(path), bytes=path.stat().st_size, sha256=sha(path))


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare(args, report, watched):
    import numpy as np
    parent_path = pin(args.parent_manifest, PARENT_SHA, watched)
    parent = json.loads(parent_path.read_text())
    require(parent['schema'] == 'v55_original_VGGT_24_frame_global_pair_RKNN_manifest'
        and parent['checkpoint'] == CHECKPOINT and len(parent['blocks']) == 24,
        'Original complete pair manifest/checkpoint required')
    row = parent['blocks'][0]
    require(row['role'] == 'pair' and row['index'] == 0 and row['sha256'] == GRAPH_SHA
        and row['bytes'] == GRAPH_BYTES and row['input_name'] == 'input'
        and row['input_shape'] == INPUT_SHAPE and row['output_bindings'] == OUTPUTS
        and row['all_36_original_parameter_values_preserved_by_parent_composition'] is True
        and set(row['parameter_raw_sha256']) == {'frame', 'global'}
        and all(len(row['parameter_raw_sha256'][role]) == 18 for role in ('frame', 'global')),
        'Complete original pair00 IO and 36 learned tensors required')
    original_graph = artifact(parent_path.parent, row, watched)
    for relative, digest in parent['source_hashes'].items():
        pin(bounded(args.source_root, relative), digest, watched)

    baseline_path = pin(args.baseline_manifest, BASELINE_SHA, watched)
    baseline = json.loads(baseline_path.read_text())
    leaf_path = pin(args.baseline_leaf, BASELINE_LEAF_SHA, watched)
    leaf = json.loads(leaf_path.read_text())
    old_row = baseline['blocks'][0]
    require(baseline['schema'] == 'v56_original_VGGT_calibrated_pair_RKNN_manifest'
        and baseline['status'] == leaf['status'] == 'compiled_not_quality_accepted'
        and baseline['compiled_count'] == 24 and baseline['checkpoint'] == CHECKPOINT
        and baseline['parent_manifest_sha256'] == PARENT_SHA
        and baseline['original_parent_manifest_sha256'] == BASELINE_GRAPH_ANCESTOR_SHA
        and baseline['capture_sha256'] == leaf['capture_sha256'] == CAPTURE_SHA
        and {k: v for k, v in old_row.items() if k != 'rknn'} == {k: v for k, v in row.items() if k != 'rknn'}
        and leaf['index'] == 0 and leaf['original_graph_sha256'] == GRAPH_SHA
        and leaf['configuration'] == CONFIG and leaf['dataset_sha256'] == DATASET_SHA
        and leaf['rknn'] == old_row['rknn']
        and [item['name'] for item in leaf['operations']] == ['config', 'load', 'build', 'export']
        and all(item['code'] == 0 for item in leaf['operations']),
        'Pinned original W8A8 pair00/configuration/calibration lineage differs')
    baseline_model = artifact(baseline_path.parent, old_row['rknn'], watched)
    require(old_row['rknn']['sha256'] == BASELINE_MODEL_SHA
        and old_row['rknn']['bytes'] == 27736654, 'Existing original W8A8 model differs')

    rewrite_source = pin(args.rewrite_source, args.rewrite_source_sha256, watched)
    export_path = pin(args.export, args.export_sha256, watched)
    exported = json.loads(export_path.read_text())
    proof = exported['proof']
    require(exported['schema'] == 'v86_original_pair00_frame_QK_rows_ONNX_rewrite'
        and exported['status'] == 'candidate_written_not_FP32_accepted'
        and exported['script_sha256'] == args.rewrite_source_sha256
        and exported['original_parent_manifest_sha256'] == PARENT_SHA
        and exported['original_pair_onnx_sha256'] == GRAPH_SHA
        and exported['original_checkpoint'] == CHECKPOINT and exported['parent'] == row
        and exported['source_and_inputs_unchanged'] is True
        and proof['original_parameter_tensors_verified'] == 36
        and proof['original_initializer_count'] == 63
        and proof['original_parameter_raw_sha256'] == row['parameter_raw_sha256']
        and all(proof[flag] is True for flag in PROOF_FLAGS),
        'Gated scoped layout proposal must preserve all original parameters/initializers/math')
    candidate_graph = artifact(export_path.parent, exported['candidate'], watched)
    check_source = pin(args.check_source, args.check_source_sha256, watched)
    check_path = pin(args.check, args.check_sha256, watched)
    checked = json.loads(check_path.read_text())
    require(checked['schema'] == 'v86_original_VGGT_pair00_frame_QK_rows_FP32_check'
        and checked['status'] == 'passed_original_FP32_ONNX_equivalence_not_quality_accepted'
        and checked['script_sha256'] == args.check_source_sha256
        and checked['original_parent_manifest_sha256'] == PARENT_SHA
        and checked['original_pair_onnx_sha256'] == GRAPH_SHA
        and checked['candidate_onnx_sha256'] == exported['candidate']['sha256']
        and checked['rewrite_receipt_sha256'] == args.export_sha256
        and checked['rewrite_source_sha256'] == args.rewrite_source_sha256
        and checked['strict_rtol'] == 1e-4 and checked['strict_atol'] == 1e-5
        and checked['all_candidate_vs_original_FP32_ONNX_strict_passed'] is True
        and checked['cleanup_errors'] == []
        and all(checked[key] is True for key in ['source_and_inputs_unchanged',
            'environment_restored', 'affinity_restored', 'sessions_dropped'])
        and [case['tag'] for case in checked['cases']] == ['v53', 'v56-calib00'],
        'Two-real-input, four complete output original FP32 gate required before SDK')

    def load(directory, item, shape):
        path = artifact(directory, item, watched)
        value = np.load(path, allow_pickle=False)
        require(value.dtype == np.float32 and list(value.shape) == shape and value.flags.c_contiguous
            and np.isfinite(value).all() and item['dtype'] == 'float32' and item['shape'] == shape
            and item['tensor_bytes'] == value.nbytes
            and hashlib.sha256(memoryview(value).cast('B')).hexdigest() == item['raw_sha256'],
            'Saved original/candidate FP32 array differs')
        return path, value

    for case in checked['cases']:
        load(check_path.parent, case['input'], INPUT_SHAPE)
        require(len(case['original_outputs']) == len(case['candidate_outputs']) == len(case['checks']) == 2,
            'Both complete original pair consumers required')
        for binding, original, candidate in zip(OUTPUTS, case['original_outputs'], case['candidate_outputs']):
            _, reference = load(check_path.parent, original, binding['shape'])
            _, value = load(check_path.parent, candidate, binding['shape'])
            a, b = value.astype(np.float64), reference.astype(np.float64)
            require(np.all(np.abs(a-b) <= 1e-5 + 1e-4*np.abs(b)),
                'Saved complete consumer independently fails fixed original FP32 gate')
        if case['original_Torch_outputs'] is not None:
            require(case['tag'] == 'v53' and len(case['original_Torch_outputs']) == 2,
                'Only v53 has independently captured original Torch output references')
            for item, binding in zip(case['original_Torch_outputs'], OUTPUTS):
                load(check_path.parent, item, binding['shape'])
        else:
            require(case['tag'] == 'v56-calib00', 'Missing v53 original Torch references')

    capture_path = pin(args.calibration_dir / 'capture.json', CAPTURE_SHA, watched)
    captured = json.loads(capture_path.read_text())
    require(captured['schema'] == 'v56_original_VGGT_calibration_capture'
        and captured['status'] == 'completed_original_FP32_calibration_capture'
        and captured['checkpoint'] == CHECKPOINT and captured['full_forward_completed'] == 8
        and len(captured['samples']) == 8 and captured['image_shape'] == [2, 3, 84, 112]
        and captured['NPU_or_cache_installed'] is False and captured['cleanup_errors'] == []
        and all(captured[key] is True for key in ['original_model_only', 'model_state_unchanged',
            'source_and_images_unchanged', 'hooks_removed', 'affinity_restored', 'environment_restored']),
        'Eight original CPU-only complete forward calibration pairs required')
    for relative, digest in parent['source_hashes'].items():
        require(captured['source_hashes'][relative] == digest
            and baseline['source_hashes'][relative] == digest, 'Original source identity differs')
    groups = [group for group in captured['groups']['aggregator'] if group['index'] == 0]
    require(len(groups) == 1 and len(groups[0]['samples']) == 8, 'One original frame0 calibration group required')
    group, paths = groups[0], []
    for index, (sample, origin) in enumerate(zip(group['samples'], captured['samples'])):
        require(sample['sample_id'] == origin['sample_id'] == 'calib%02d' % index
            and sample['source_pair_indices'] == origin['indices'] == [index*96, index*96+48]
            and origin['split'] == 'calibration' and origin['input_unchanged'] is True
            and sample['origin_batch_shape'] == INPUT_SHAPE and sample['origin_batch_row'] is None,
            'Original eight calibration pairs/input order differs')
        path, _ = load(capture_path.parent, sample['artifact'], INPUT_SHAPE)
        require(not any(char.isspace() for char in str(path)), 'Dataset path contains whitespace')
        paths.append(str(path))
        for image in origin['images']:
            pin(image['file'], image['sha256'], watched, image['bytes'])
    expected = ''.join(path + '\n' for path in paths)
    source_dataset = artifact(capture_path.parent, group['dataset'], watched)
    require(sha(source_dataset) == DATASET_SHA and source_dataset.read_text() == expected
        and group['dataset']['line_count'] == 8 and group['dataset']['input_count'] == 1,
        'Identical original A/B eight-row dataset required')
    dataset = args.output_dir / 'dataset.txt'
    dataset.write_text(expected)
    pin(dataset, DATASET_SHA, watched, source_dataset.stat().st_size)
    report.update(checkpoint=CHECKPOINT, original_parent_manifest_sha256=PARENT_SHA,
        original_pair_onnx_sha256=GRAPH_SHA, original_pair=row,
        graph=metadata(candidate_graph), graph_sha256=exported['candidate']['sha256'],
        original_graph=metadata(original_graph), graph_proof=proof,
        export_manifest_sha256=args.export_sha256, check_receipt_sha256=args.check_sha256,
        rewrite_source_sha256=args.rewrite_source_sha256, FP32_checker_source_sha256=args.check_source_sha256,
        candidate_export=metadata(export_path), original_FP32_check=metadata(check_path),
        rewrite_source=metadata(rewrite_source), FP32_checker_source=metadata(check_source),
        original_graph_is_baseline_A=True, candidate_graph_is_restructured_B=True,
        original_parameter_values_unchanged=True, FP32_four_outputs_gate_reverified_from_saved_arrays=True,
        baseline_reused_not_recompiled=True, baseline_manifest=metadata(baseline_path),
        baseline_leaf=metadata(leaf_path), baseline_rknn=metadata(baseline_model),
        baseline_parent_manifest_sha256=PARENT_SHA,
        baseline_original_parent_manifest_sha256=BASELINE_GRAPH_ANCESTOR_SHA,
        baseline_leaf_release_verification_available='released' in leaf,
        baseline_leaf_release_status=leaf.get('released'),
        baseline_missing_release_flag_not_assumed_successful=True,
        source_hashes=parent['source_hashes'], input_shape=INPUT_SHAPE, output_bindings=OUTPUTS,
        calibration=dict(capture=metadata(capture_path), capture_sha256=CAPTURE_SHA,
            samples=group['samples'], sample_count=8, original_image_pairs=captured['samples'],
            dataset=dict(file='dataset.txt', bytes=dataset.stat().st_size, sha256=DATASET_SHA,
                input_count=1, line_count=8), same_as_baseline_A=True),
        configuration=CONFIG, build_options=dict(do_quantization=True, dataset=str(dataset), auto_hybrid=False))
    return candidate_graph, dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role', choices=['B'], default='B')
    p.add_argument('--source-root', type=Path, default=BASE / 'src')
    p.add_argument('--parent-manifest', type=Path, default=E / 'vggt-v55-pair-graphs-r1/compiled.json')
    p.add_argument('--baseline-manifest', type=Path, default=E / 'vggt-v56-ptq-pairs-r1/compiled.json')
    p.add_argument('--baseline-leaf', type=Path, default=E / 'vggt-v56-ptq-pairs-r1/pair-00/compile.json')
    p.add_argument('--calibration-dir', type=Path, default=E / 'vggt-v56-calibration-r1')
    for name in ['export', 'check', 'rewrite-source', 'check-source']:
        p.add_argument('--' + name, type=Path, required=True)
        p.add_argument('--' + name + '-sha256', required=True)
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
    report = dict(schema='v86_original_pair00_QK_rows_W8A8_compile', status='running', role=args.role,
        script_sha256=sha(__file__), original_model_only=True, original_model_forward_executed=False,
        model_checkpoint_deserialized=False, ORT_rerun=False, calibration_executed=False,
        candidate_accepted=False, quality_accepted=False, full_model_or_realtime_validated=False,
        operations=[], released=False, release_code=None, cleanup_errors=[], thermal_guard_external=True,
        expected_NPU_hz=300000000, clocks_or_governors_modified=False)
    try:
        require(platform.machine().lower() in ('aarch64', 'arm64')
            and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(), 'RK3588S board only')
        pin(__file__, report['script_sha256'], watched)
        if args.controller_source:
            report['controller_source'] = metadata(pin(args.controller_source, args.controller_source_sha256, watched))
        os.sched_setaffinity(0, {4, 6})
        report['compile_cpus'] = [4, 6]
        graph, dataset = prepare(args, report, watched)
        os.chdir(args.output_dir)
        os.environ['TMPDIR'] = str(args.output_dir)
        report['toolkit_version'] = importlib.metadata.version('rknn-toolkit2')
        require(report['toolkit_version'].split('+')[0] == '2.3.2', 'RKNN Toolkit2 2.3.2 required')
        write(args.output_dir / 'result.json', report)
        from rknn.api import RKNN
        sdk = RKNN(verbose=True, verbose_file=str(args.output_dir / 'sdk.log'))
        output = args.output_dir / 'pair-qk-rows-int8.rknn'
        calls = [('config', lambda: sdk.config(**CONFIG)), ('load_onnx', lambda: sdk.load_onnx(model=str(graph))),
            ('build', lambda: sdk.build(do_quantization=True, dataset=str(dataset), auto_hybrid=False)),
            ('export_rknn', lambda: sdk.export_rknn(str(output)))]
        for name, function in calls:
            operation = dict(name=name, status='running', code=None)
            report['operations'].append(operation)
            if name == 'build':
                report['calibration_build_api_invoked'] = True
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
        require(output.stat().st_size > 0, 'Empty RKNN output')
        report.update(status='compiled_not_quality_accepted', rknn=dict(file=output.name,
            bytes=output.stat().st_size, sha256=sha(output)))
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
        if report['cleanup_errors'] or not all(report.get(key) for key in
                ['environment_restored', 'affinity_restored', 'source_and_inputs_unchanged']):
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
