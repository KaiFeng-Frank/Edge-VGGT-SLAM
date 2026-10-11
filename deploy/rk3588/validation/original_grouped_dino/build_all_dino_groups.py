#!/usr/bin/env python3
"""Board-only sequential construction of six original four-layer DINO groups.

Reuse the compiled group0. For each other group, compare its FP32 ONNX against
the four unchanged parent ONNX graphs, then compile only if both inputs are
byte-identical. Original Torch FP32 threshold failures remain diagnostic
failures; neither the composition gate nor compilation accepts model quality.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

PARENT_SHA = 'd68d05f984688e2c793f2e9d8320531202a4f9871790d7c2d96c74b5f41c9325'
COMPOSER_SHA = 'ee2cbeebb6984fdc36b6a21b3957063323d1c6c571611584916317372f9e224c'
CHECKPOINT = {'sha256': 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0',
              'bytes': 5026874952}
SHAPE = [1, 53, 1024]
CONFIG = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
    mean_values=None, std_values=None, quant_img_RGB2BGR=False, model_pruning=False,
    remove_reshape=False, quantize_weight=False, enable_flash_attention=False)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for value in iter(lambda: stream.read(4 << 20), b''):
            digest.update(value)
    return digest.hexdigest()


def write(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def pinned(path, expected):
    path = Path(path).resolve(strict=True)
    if sha(path) != expected:
        raise ValueError('Pinned source/receipt changed: ' + str(path))
    return path


def artifact(directory, row):
    directory = Path(directory).resolve(strict=True)
    relative = Path(row['file'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Manifest-owned relative artifact required')
    path = (directory / relative).resolve(strict=True)
    if not path.is_relative_to(directory) or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
        raise ValueError('Artifact size/SHA mismatch: ' + str(path))
    return path


def load_parent(args):
    if args.manifest_sha256 != PARENT_SHA:
        raise ValueError('Only the pinned original complete DINO parent is supported')
    path = pinned(args.manifest, PARENT_SHA)
    parent = json.loads(path.read_text())
    if (parent['schema'] != 'v54_original_VGGT_24_DINO_block_RKNN_manifest'
            or parent['checkpoint'] != CHECKPOINT or parent['image_shape'] != [1, 3, 84, 112]
            or parent['depth'] != 24 or parent['chunked_blocks'] is not False
            or [(row['role'], row['index']) for row in parent['blocks']] != [('dino', i) for i in range(24)]):
        raise ValueError('Complete original 24-layer DINO parent required')
    return path, parent


def metadata(row, start, output, folder):
    result = dict(row)
    result.update(index=start // 4, start_index=start)
    for key in ('file',):
        result[key] = str((folder / row[key]).relative_to(output))
    return result


def copy_artifact(source, destination, expected):
    shutil.copyfile(source, destination)
    if destination.stat().st_size != expected['bytes'] or sha(destination) != expected['sha256']:
        raise ValueError('Artifact changed during copy')


def reuse_group0(args, parent, output):
    export_path = pinned(args.group0_export, args.group0_export_sha256)
    check_path = pinned(args.group0_check, args.group0_check_sha256)
    compile_path = pinned(args.group0_compile, args.group0_compile_sha256)
    exported, checked, compiled = (json.loads(path.read_text()) for path in (export_path, check_path, compile_path))
    row = exported['blocks'][0]
    if (exported['schema'] != 'v58_original_VGGT_4_DINO_block_group_ONNX_manifest'
            or exported['original_parent_manifest_sha256'] != PARENT_SHA
            or exported['script_sha256'] != COMPOSER_SHA or exported['checkpoint'] != CHECKPOINT
            or exported['boundary_mode'] != 'fp32' or row['indices'] != [0, 1, 2, 3]
            or row['parents'] != parent['blocks'][:4]
            or checked['status'] != 'checked_original_FP32_references'
            or checked['manifest_sha256'] != args.group0_export_sha256
            or checked['graph_sha256'] != row['sha256'] or checked['cleanup_errors'] != []
            or checked['all_separate_ONNX_checks_byte_equal'] is not True
            or [check['input'] for check in checked['checks']] != ['single0', 'single1']
            or not all(check['byte_equal_to_four_separate_FP32_ONNX'] is True for check in checked['checks'])
            or compiled['status'] != 'compiled_not_quality_accepted' or compiled['cleanup_errors'] != []
            or compiled['manifest_sha256'] != args.group0_export_sha256 or compiled['graph_sha256'] != row['sha256']
            or compiled['check_receipt_sha256'] != args.group0_check_sha256 or compiled['configuration'] != CONFIG
            or compiled['original_FP32_gate_passed'] is not checked['all_original_FP32_checks_passed']
            or compiled['candidate_accepted'] is not False
            or [item['name'] for item in compiled['operations']] != ['config', 'load_onnx', 'build', 'export_rknn']
            or any(item['code'] not in (None, 0) for item in compiled['operations'])):
        raise ValueError('Existing actual group0 export/check/compile lineage differs')
    graph = artifact(export_path.parent, row)
    model = artifact(compile_path.parent, compiled['rknn'])
    folder = output / 'group-00'
    folder.mkdir(exist_ok=False)
    copy_artifact(graph, folder / graph.name, row)
    copy_artifact(model, folder / 'group.rknn', compiled['rknn'])
    for path, name in ((export_path, 'export.json'), (check_path, 'ort-check.json'), (compile_path, 'compile.json')):
        shutil.copyfile(path, folder / name)
    result = metadata(row, 0, output, folder)
    result.update(rknn=dict(compiled['rknn'], file='group-00/group.rknn'),
        original_FP32_gate_passed=checked['all_original_FP32_checks_passed'],
        all_separate_ONNX_checks_byte_equal=True, checks=checked['checks'], configuration=dict(CONFIG),
        reused_actual_group0=True, original_group0_export_sha256=args.group0_export_sha256,
        original_group0_check_sha256=args.group0_check_sha256,
        original_group0_compile_sha256=args.group0_compile_sha256,
        candidate_accepted=False, cleanup_errors=[])
    return result


def load_reference(directory, row, np):
    value = np.load(artifact(directory, row), allow_pickle=False)
    if (list(value.shape) != SHAPE or value.dtype != np.float32 or not np.isfinite(value).all()
            or list(value.shape) != row['shape'] or str(value.dtype) != row['dtype']
            or value.nbytes != row['tensor_bytes']
            or hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast('B')).hexdigest() != row['raw_sha256']):
        raise ValueError('Original captured single-frame reference tensor differs')
    return value


def ort_checks(graph, rows, parent_path, parent, folder):
    import numpy as np
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
    group = ort.InferenceSession(str(graph), sess_options=options, providers=['CPUExecutionProvider'])
    separate = [ort.InferenceSession(str(artifact(parent_path.parent, row)), sess_options=options,
                providers=['CPUExecutionProvider']) for row in rows]
    reference = Path(parent['original_reference_capture_directory']).resolve(strict=True)
    checks = []
    for tag in ('single0', 'single1'):
        x = load_reference(reference, rows[0]['original_capture_references'][tag]['input'], np)
        y = load_reference(reference, rows[-1]['original_capture_references'][tag]['output'], np)
        begin = time.perf_counter()
        actual = group.run(['output'], {'input': x})[0]
        group_seconds = time.perf_counter() - begin
        begin = time.perf_counter()
        prior = x.copy()
        for session in separate:
            prior = session.run(['output'], {'input': prior})[0]
            if prior.dtype != np.float32 or list(prior.shape) != SHAPE or not np.isfinite(prior).all():
                raise ValueError('Separate original ONNX chain returned invalid tokens')
        chain_seconds = time.perf_counter() - begin
        if actual.dtype != np.float32 or list(actual.shape) != SHAPE or not np.isfinite(actual).all():
            raise ValueError('Composed original ONNX group returned invalid tokens')
        delta = actual.astype(np.float64) - y.astype(np.float64)
        failed = np.abs(delta) > 1e-5 + 1e-4 * np.abs(y.astype(np.float64))
        denominator = float(np.linalg.norm(y.astype(np.float64)))
        equal = actual.tobytes(order='C') == prior.tobytes(order='C')
        path = folder / (tag + '.npy')
        with path.open('xb') as stream:
            np.save(stream, actual, allow_pickle=False)
        checks.append(dict(input=tag, strict_rtol=1e-4, strict_atol=1e-5,
            strict_pass=not bool(failed.any()), failed_elements=int(failed.sum()),
            max_abs=float(np.abs(delta).max()), relative_l2=float(np.linalg.norm(delta)) / denominator if denominator else None,
            ORT_CPU_seconds=group_seconds, separate_ORT_CPU_seconds=chain_seconds,
            byte_equal_to_four_separate_FP32_ONNX=equal,
            versus_separate_max_abs=float(np.abs(actual.astype(np.float64) - prior.astype(np.float64)).max()),
            output=dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path),
                raw_sha256=hashlib.sha256(actual.tobytes(order='C')).hexdigest())))
    del group, separate
    gc.collect()
    return checks


def worker(args, output):
    parent_path, parent = load_parent(args)
    composer_path = pinned(args.composer, COMPOSER_SHA)
    spec = importlib.util.spec_from_file_location('_v58_original_DINO_group_composer', composer_path)
    composer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(composer)
    start = args.worker_start
    rows = parent['blocks'][start:start + 4]
    folder = output / f'group-{start // 4:02d}'
    folder.mkdir(exist_ok=False)
    report = dict(status='running', index=start // 4, start_index=start, indices=list(range(start, start + 4)),
        script_sha256=sha(__file__), composer_sha256=COMPOSER_SHA, original_parent_manifest_sha256=PARENT_SHA,
        candidate_accepted=False, configuration=dict(CONFIG), checks=[], operations=[], cleanup_errors=[])
    begin, api = time.perf_counter(), None
    saved_cwd, saved_tmpdir = Path.cwd(), os.environ.get('TMPDIR')
    original_paths = []
    try:
        original_paths = [artifact(parent_path.parent, row) for row in rows]
        import onnx
        tick = time.perf_counter()
        models = [onnx.load(str(path), load_external_data=False) for path in original_paths]
        model, preservation = composer.compose_original_dino_group(models, rows, boundary_mode='fp32')
        graph = folder / f'dino-group-{start:02d}-{start + 3:02d}.onnx'
        with graph.open('xb') as stream:
            stream.write(model.SerializeToString())
        report['compose_seconds'] = time.perf_counter() - tick
        report['graph'] = dict(file=str(graph.relative_to(output)), bytes=graph.stat().st_size, sha256=sha(graph))
        del model, models
        gc.collect()
        tick = time.perf_counter()
        report['checks'] = ort_checks(graph, rows, parent_path, parent, folder)
        report['ORT_check_seconds'] = time.perf_counter() - tick
        report['original_FP32_gate_passed'] = all(item['strict_pass'] for item in report['checks'])
        report['all_separate_ONNX_checks_byte_equal'] = all(item['byte_equal_to_four_separate_FP32_ONNX'] for item in report['checks'])
        write(folder / 'ort-check.json', report)
        if not report['all_separate_ONNX_checks_byte_equal']:
            raise ValueError('Composition differs from the four original ONNX parents; diagnostic compilation stopped')
        # Original Torch gate failures are deliberately retained above. This
        # compile gate accepts only graph composition identity, never accuracy.
        os.chdir(folder)
        os.environ['TMPDIR'] = str(folder)
        from rknn.api import RKNN
        api = RKNN(verbose=False, verbose_file=str(folder / 'sdk.log'))
        for name, function in (
            ('config', lambda: api.config(**CONFIG)),
            ('load_onnx', lambda: api.load_onnx(model=str(graph))),
            ('build', lambda: api.build(do_quantization=False)),
            ('export_rknn', lambda: api.export_rknn(export_path=str(folder / 'group.rknn')))):
            tick = time.perf_counter()
            code = function()
            report['operations'].append(dict(name=name, code=code, seconds=time.perf_counter() - tick))
            write(folder / 'build.json', report)
            if code not in (None, 0):
                raise RuntimeError(f'{name} failed: {code}')
        compiled = folder / 'group.rknn'
        report['group'] = dict(role='dino_group', index=start // 4, start_index=start,
            indices=list(range(start, start + 4)), **report['graph'],
            rknn=dict(file=str(compiled.relative_to(output)), bytes=compiled.stat().st_size, sha256=sha(compiled)),
            input_name='input', output_name='output', input_shape=SHAPE, output_shape=SHAPE,
            heads=16, boundary_mode='fp32', parents=rows,
            parameter_raw_sha256={str(row['index']): row['parameter_raw_sha256'] for row in rows},
            original_capture_references={tag: dict(input=rows[0]['original_capture_references'][tag]['input'],
                output=rows[-1]['original_capture_references'][tag]['output']) for tag in ('single0', 'single1', 'batch2')},
            configuration=dict(CONFIG), checks=report['checks'],
            original_FP32_gate_passed=report['original_FP32_gate_passed'],
            all_separate_ONNX_checks_byte_equal=True, reused_actual_group0=False, candidate_accepted=False,
            **preservation)
        report['status'] = 'compiled_not_quality_accepted'
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        raise
    finally:
        if api is not None:
            try:
                code = api.release()
                report['release_code'] = code
                if code not in (None, 0):
                    raise RuntimeError('RKNN release failed: ' + str(code))
            except BaseException as exc:
                report['cleanup_errors'].append(f'release: {type(exc).__name__}: {exc}')
        try:
            os.chdir(saved_cwd)
            if saved_tmpdir is None:
                os.environ.pop('TMPDIR', None)
            else:
                os.environ['TMPDIR'] = saved_tmpdir
            pinned(parent_path, PARENT_SHA)
            pinned(composer_path, COMPOSER_SHA)
            for path, row in zip(original_paths, rows):
                pinned(path, row['sha256'])
        except BaseException as exc:
            report['cleanup_errors'].append(f'restoration/source check: {type(exc).__name__}: {exc}')
        if report['cleanup_errors']:
            report['status'] = 'failed_cleanup'
        if 'group' in report:
            report['group']['cleanup_errors'] = list(report['cleanup_errors'])
        report['seconds'] = time.perf_counter() - begin
        write(folder / 'build.json', report)
        if report['cleanup_errors']:
            raise RuntimeError('Group build cleanup failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--composer', type=Path, default=Path(__file__).with_name('compose_original_dino_group.py'))
    for name in ('group0-export', 'group0-check', 'group0-compile'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--worker-start', type=int, choices=(4, 8, 12, 16, 20))
    args = parser.parse_args()
    if sys.platform != 'linux' or platform.machine() not in ('aarch64', 'arm64') or b'rk3588' not in Path('/proc/device-tree/compatible').read_bytes():
        parser.error('All actual graph execution and compilation must run on RK3588')
    saved_affinity = os.sched_getaffinity(0)
    report = None
    try:
        os.sched_setaffinity(0, {4, 5, 6, 7})
        output = args.output_dir.resolve()
        if args.worker_start is not None:
            worker(args, output)
            return
        output.mkdir(parents=True, exist_ok=False)
        report = dict(schema='v58_original_VGGT_6_DINO_group_RKNN_manifest', status='running',
            script_sha256=sha(__file__), composer_sha256=COMPOSER_SHA, groups=[], cleanup_errors=[],
            candidate_accepted=False, quality_or_realtime_validated=False,
            original_FP32_strict_rtol=1e-4, original_FP32_strict_atol=1e-5,
            actual_inference_and_compilation_on_board_only=True, calibration_executed=False,
            unchanged_exported_graph_identity_gate_is_not_original_FP32_quality_acceptance=True)
        begin = time.perf_counter()
        try:
            parent_path, parent = load_parent(args)
            pinned(args.composer, COMPOSER_SHA)
            for key in ('checkpoint', 'source_hashes', 'capture_sources', 'capture_sha256', 'image_shape',
                        'original_reference_capture_directory', 'original_non_block_parameter_raw_sha256',
                        'depth', 'embed_dim', 'heads', 'patch_size', 'num_register_tokens', 'chunked_blocks'):
                report[key] = parent[key]
            report['original_parent_manifest_sha256'] = PARENT_SHA
            report['original_parent_manifest_directory'] = str(parent_path.parent)
            report['groups'].append(reuse_group0(args, parent, output))
            write(output / 'compiled.json', report)
            print('Reused existing actual DINO group0 graph/model/check receipts', flush=True)
            script = Path(__file__).resolve()
            for start in (4, 8, 12, 16, 20):
                subprocess.run([sys.executable, '-B', '-u', str(script), *sys.argv[1:], '--worker-start', str(start)], check=True)
                build_path = output / f'group-{start // 4:02d}' / 'build.json'
                built = json.loads(build_path.read_text())
                if (built['status'] != 'compiled_not_quality_accepted' or built['cleanup_errors'] != []
                        or built['script_sha256'] != report['script_sha256']
                        or built['all_separate_ONNX_checks_byte_equal'] is not True):
                    raise ValueError('Group compilation or identity gate failed')
                group = built['group']
                artifact(output, group)
                artifact(output, group['rknn'])
                group['build_receipt_sha256'] = sha(build_path)
                report['groups'].append(group)
                write(output / 'compiled.json', report)
                print(json.dumps(dict(completed_groups=len(report['groups']), total=6, indices=group['indices'],
                                      seconds=built['seconds'], original_FP32_gate_passed=group['original_FP32_gate_passed'])), flush=True)
            if [group['indices'] for group in report['groups']] != [list(range(start, start + 4)) for start in range(0, 24, 4)]:
                raise ValueError('All original 24 DINO layers must be covered exactly once')
            report.update(status='compiled_not_quality_accepted',
                all_original_FP32_checks_passed=all(group['original_FP32_gate_passed'] for group in report['groups']),
                all_separate_ONNX_checks_byte_equal=True, preserved_original_DINO_blocks=24,
                original_prepare_positions_finalnorm_and_remaining_model_must_be_retained=True)
        except BaseException as exc:
            report.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
            raise
        finally:
            report['seconds'] = time.perf_counter() - begin
            write(output / 'compiled.json', report)
    finally:
        try:
            os.sched_setaffinity(0, saved_affinity)
        except BaseException as exc:
            receipt = (output / f'group-{args.worker_start // 4:02d}' / 'build.json'
                       if args.worker_start is not None else output / 'compiled.json')
            if receipt.is_file():
                failed = json.loads(receipt.read_text())
                failed['cleanup_errors'].append(f'affinity restore: {type(exc).__name__}: {exc}')
                failed['status'] = 'failed_cleanup'
                write(receipt, failed)
            raise


if __name__ == '__main__':
    main()
