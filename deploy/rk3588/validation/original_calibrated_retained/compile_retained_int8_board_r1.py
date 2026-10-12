#!/usr/bin/env python3
"""Board-only W8A8 build of one unchanged original VGGT retained graph."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
import traceback
from pathlib import Path

BASE = Path('/mnt/vggt-data/vggt-slam2')
PARENT_SHA = '771a9b82007d2feefd65ebbf3f4b0fb27070b79ff7f6d0faab1a8c866da90e43'
CAPTURE_SHA = 'a1e6f1b2a89e1a96987befdea4f0a048aaf06066efa9fff72a370ac77d45507c'
CHECKPOINT = dict(bytes=5026874952, sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')
RANGES = ((0, 4), (5, 11), (12, 17), (18, 23))
SHAPE = [2, 53, 1024]
CONFIG = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
    mean_values=None, std_values=None, quant_img_RGB2BGR=False, model_pruning=False,
    remove_reshape=False, quantize_weight=False, enable_flash_attention=False,
    quantized_dtype='w8a8', quantized_algorithm='normal', quantized_method='channel')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4 << 20), b''):
            digest.update(data)
    return digest.hexdigest()


def pin(path, expected, watched):
    path = Path(path).resolve(strict=True)
    require(sha(path) == expected, 'Frozen file SHA differs: ' + str(path))
    watched[path] = expected
    return path


def artifact(directory, row, watched):
    directory, relative = Path(directory).resolve(strict=True), Path(row['file'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'Relative artifact required')
    path = pin(directory / relative, row['sha256'], watched)
    path.relative_to(directory)
    require(path.stat().st_size == row['bytes'], 'Frozen artifact size differs')
    return path


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare(args, report, watched):
    import numpy as np
    parent_path = pin(args.manifest, PARENT_SHA, watched)
    capture_path = pin(args.capture, CAPTURE_SHA, watched)
    parent, capture = json.loads(parent_path.read_text()), json.loads(capture_path.read_text())
    require(parent['schema'] == 'v67_original_VGGT_retained_groups_RKNN_manifest'
        and parent['status'] == 'compiled_not_quality_accepted' and parent['cleanup_errors'] == []
        and parent['source_and_inputs_unchanged'] is True and parent['checkpoint'] == CHECKPOINT
        and parent['all_original_24_pairs_retained'] is True and len(parent['blocks']) == 4
        and parent['original_target_tokens_global_index'] == 20, 'Original complete retained parent required')
    require(capture['schema'] == 'v56_original_VGGT_calibration_capture'
        and capture['status'] == 'completed_original_FP32_calibration_capture'
        and capture['checkpoint'] == CHECKPOINT and capture['image_shape'] == [2, 3, 84, 112]
        and capture['full_forward_completed'] == 8 and len(capture['samples']) == 8
        and capture['original_model_only'] is True and capture['NPU_or_cache_installed'] is False
        and capture['model_state_unchanged'] is True and capture['source_and_images_unchanged'] is True
        and capture['cleanup_errors'] == [], 'Original complete FP32 calibration capture required')
    start, end = RANGES[args.group_index]
    row = parent['blocks'][args.group_index]
    require(row['index'] == args.group_index and row['indices'] == list(range(start, end + 1))
        and row['input_name'] == 'input' and row['input_shape'] == SHAPE
        and len(row['output_bindings']) == (3 if args.group_index == 3 else 2), 'Retained group scope differs')
    graph = artifact(parent_path.parent, row, watched)
    checked = artifact(parent_path.parent, row['check_receipt'], watched)
    compiled = artifact(parent_path.parent, row['compile_receipt'], watched)
    check, baseline = json.loads(checked.read_text()), json.loads(compiled.read_text())
    require(check['status'] == 'checked_separate_ONNX_equivalence_not_quality_accepted'
        and check['all_separate_ONNX_checks_byte_equal'] is True and check['group_index'] == args.group_index
        and check['graph_sha256'] == row['sha256'] and check['cleanup_errors'] == []
        and check['source_and_inputs_unchanged'] is True, 'Existing original FP32 chain proof required')
    require(baseline['status'] == 'compiled_not_quality_accepted' and baseline['group_index'] == args.group_index
        and baseline['graph_sha256'] == row['sha256'] and baseline['check_receipt_sha256'] == row['check_receipt']['sha256']
        and baseline['cleanup_errors'] == [] and baseline['released'] is True
        and baseline['configuration'] == {k: v for k, v in CONFIG.items() if not k.startswith('quantized_')},
        'Pinned original FP16 compile required')
    for relative, digest in parent['source_hashes'].items():
        require(capture['source_hashes'][relative] == digest, 'Original graph/capture source lineage differs')
        pin(args.source_root / relative, digest, watched)
    groups = {g['index']: g for g in capture['groups']['aggregator']}
    require(len(groups) == 24 and set(groups) == set(range(24)), '24 original frame-input capture groups required')
    group = groups[start]
    require(group['input_shape'] == SHAPE and len(group['samples']) == 8, 'Eight actual group-start inputs required')
    paths = []
    for i, (sample, source) in enumerate(zip(group['samples'], capture['samples'])):
        pair = [i * 96, i * 96 + 48]
        require(sample['sample_id'] == source['sample_id'] == 'calib%02d' % i
            and sample['source_pair_indices'] == source['indices'] == pair and source['split'] == 'calibration'
            and source['calls']['aggregator'] == list(range(24)) and source['input_unchanged'] is True
            and sample['origin_batch_shape'] == SHAPE and sample['origin_batch_row'] is None,
            'Actual original two-frame calibration identity differs')
        require([im['index'] for im in source['images']] == pair
            and all(im['bytes'] > 0 and len(im['sha256']) == 64 for im in source['images'])
            and source['images'][0]['timestamp'] < source['images'][1]['timestamp'], 'Source image provenance differs')
        item = sample['artifact']; path = artifact(capture_path.parent, item, watched)
        value = np.load(path, allow_pickle=False)
        require(value.dtype == np.float32 and list(value.shape) == SHAPE and value.flags.c_contiguous
            and np.isfinite(value).all() and item['finite'] is True and item['dtype'] == 'float32'
            and item['shape'] == SHAPE and item['tensor_bytes'] == value.nbytes
            and hashlib.sha256(memoryview(value).cast('B')).hexdigest() == item['raw_sha256'],
            'Captured original FP32 activation identity differs')
        paths.append(path)
    dataset = args.output_dir / 'dataset.txt'
    require(all(not any(ch.isspace() for ch in str(p)) for p in paths), 'RKNN dataset paths cannot contain whitespace')
    dataset.write_text(''.join(str(p) + '\n' for p in paths)); watched[dataset] = sha(dataset)
    report.update(parent_manifest_directory=str(parent_path.parent), calibration_directory=str(capture_path.parent),
        checkpoint=CHECKPOINT, indices=row['indices'], parent_group=row, source_hashes=parent['source_hashes'],
        graph_sha256=row['sha256'], graph=dict(file=str(graph), bytes=row['bytes'], sha256=row['sha256']),
        prior_check_receipt=row['check_receipt'], prior_FP16_compile_receipt=row['compile_receipt'],
        calibration=dict(capture=dict(file='capture.json', bytes=capture_path.stat().st_size, sha256=CAPTURE_SHA),
            group_start_index=start, samples=group['samples'], source_images=[s['images'] for s in capture['samples']],
            dataset=dict(file='dataset.txt', bytes=dataset.stat().st_size, sha256=sha(dataset), input_count=1, line_count=8)),
        configuration=CONFIG, build_options=dict(do_quantization=True, dataset=str(dataset), auto_hybrid=False))
    return graph, dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=BASE / 'experiments/vggt-v67-retained-groups-r1/compiled.json')
    p.add_argument('--capture', type=Path, default=BASE / 'experiments/vggt-v56-calibration-r1/capture.json')
    p.add_argument('--source-root', type=Path, default=BASE / 'src')
    p.add_argument('--group-index', type=int, choices=range(4), required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--controller-source', type=Path)
    p.add_argument('--controller-source-sha256')
    args = p.parse_args()
    if bool(args.controller_source) != bool(args.controller_source_sha256):
        p.error('--controller-source and its SHA must be provided together')
    args.output_dir = args.output_dir.resolve(); args.output_dir.mkdir(parents=True, exist_ok=False)
    old_cwd, old_tmp, old_affinity = Path.cwd(), os.environ.get('TMPDIR'), os.sched_getaffinity(0)
    watched, sdk, started, failure = {}, None, time.monotonic(), None
    report = dict(schema='v80_original_VGGT_retained_INT8_compile', status='running', group_index=args.group_index,
        script_sha256=sha(Path(__file__)), parent_manifest_sha256=PARENT_SHA, capture_sha256=CAPTURE_SHA,
        original_model_only=True, original_ONNX_modified=False, ORT_rerun=False,
        candidate_accepted=False, quality_accepted=False, realtime_validated=False, operations=[],
        released=False, release_code=None, cleanup_errors=[],
        external_controller_receipt_finalized_after_child_exit=True)
    try:
        require(platform.machine().lower() in ('aarch64', 'arm64')
            and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(), 'RK3588S board only')
        pin(Path(__file__), report['script_sha256'], watched)
        if args.controller_source:
            control = pin(args.controller_source, args.controller_source_sha256, watched)
            report['controller_source'] = dict(file=str(control), sha256=args.controller_source_sha256)
        os.sched_setaffinity(0, {4, 5, 6, 7})
        graph, dataset = prepare(args, report, watched)
        os.chdir(args.output_dir); os.environ['TMPDIR'] = str(args.output_dir)
        report['toolkit_version'] = importlib.metadata.version('rknn-toolkit2')
        require(report['toolkit_version'] == '2.3.2', 'Frozen RKNN Toolkit2 2.3.2 required')
        write(args.output_dir / 'result.json', report)
        from rknn.api import RKNN
        sdk = RKNN(verbose=True, verbose_file=str(args.output_dir / 'sdk.log'))
        calls = [('config', lambda: sdk.config(**CONFIG)), ('load_onnx', lambda: sdk.load_onnx(model=str(graph))),
            ('build', lambda: sdk.build(do_quantization=True, dataset=str(dataset), auto_hybrid=False)),
            ('export_rknn', lambda: sdk.export_rknn(str(args.output_dir / 'group.rknn')))]
        for name, function in calls:
            operation = dict(name=name, status='running', code=None); report['operations'].append(operation)
            write(args.output_dir / 'result.json', report); tick = time.monotonic()
            try:
                code = function(); operation['code'] = code
                require(code in (None, 0), name + ' failed: ' + str(code)); operation['status'] = 'completed'
            except BaseException:
                operation['status'] = 'failed'; raise
            finally:
                operation['seconds'] = time.monotonic() - tick; write(args.output_dir / 'result.json', report)
        output = args.output_dir / 'group.rknn'
        report.update(status='compiled_not_quality_accepted', rknn=dict(file='group.rknn', bytes=output.stat().st_size, sha256=sha(output)))
    except BaseException as error:
        failure = error; report.update(status='failed', error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc())
    finally:
        if sdk is not None:
            try:
                code = sdk.release(); report['release_code'] = code
                require(code in (None, 0), 'release failed: ' + str(code)); report['released'] = True
            except BaseException as error:
                report['cleanup_errors'].append('release: ' + str(error))
        for name, action in [('cwd', lambda: os.chdir(old_cwd)), ('affinity', lambda: os.sched_setaffinity(0, old_affinity))]:
            try: action()
            except BaseException as error: report['cleanup_errors'].append(name + ': ' + str(error))
        if old_tmp is None: os.environ.pop('TMPDIR', None)
        else: os.environ['TMPDIR'] = old_tmp
        report['environment_restored'] = Path.cwd() == old_cwd and os.environ.get('TMPDIR') == old_tmp
        report['affinity_restored'] = os.sched_getaffinity(0) == old_affinity
        try: report['source_and_inputs_unchanged'] = all(sha(path) == digest for path, digest in watched.items())
        except BaseException as error:
            report['source_and_inputs_unchanged'] = False; report['cleanup_errors'].append('pins: ' + str(error))
        if report['cleanup_errors'] or not report['environment_restored'] or not report['affinity_restored'] or not report['source_and_inputs_unchanged']:
            report['status'] = 'failed_cleanup'
        report['seconds'] = time.monotonic() - started
        if (args.output_dir / 'sdk.log').exists(): report['sdk_log_sha256'] = sha(args.output_dir / 'sdk.log')
        write(args.output_dir / 'result.json', report)
    if failure is not None: raise failure
    require(report['status'] == 'compiled_not_quality_accepted', 'Compilation cleanup/identity failed')
    print(json.dumps(dict(status=report['status'], group_index=args.group_index, result_sha256=sha(args.output_dir / 'result.json'))), flush=True)


if __name__ == '__main__':
    main()
