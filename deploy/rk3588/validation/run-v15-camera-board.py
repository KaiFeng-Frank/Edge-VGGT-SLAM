#!/usr/bin/env python3
"""Run the entire four-iteration CameraHead RKNN graph on RK3588S only.

Private reuse of the pinned V12 SDK ABI/lifecycle, with explicitly different
input [1,2,2048] and output [1,2,9] shapes. Saves every owned FP32 pose output.
Cold plus three subsequent synchronous calls include all SDK I/O. Separate
--collect-perf-detail runs report actual CPU/NPU targets, not normal timings.
This is one deployed head; whole-model/capture-to-pose acceptance is separate.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes as ct
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
BASE = Path('/mnt/vggt-data/vggt-slam2')
INPUT_SHAPE = (1, 2, 2048)
OUTPUT_SHAPE = (1, 2, 9)
RUNTIME_SHA = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
CAPTURE_SHA = 'd50c217354b2f48f50626ff1ea2c98671d5611743ae0e97bed1c04b2991341a0'
EXPORT_SHA = '3f97e96e6e9f7beaf5a5de56569774aa79fa17eb7e30664e54cdbbfec224a033'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen(name, expected, module_name):
    path = Path(__file__).with_name(name)
    if sha(path) != expected:
        raise ValueError(f'Frozen helper SHA mismatch: {name}')
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def operator_targets(raw):
    rows = []
    for line in raw.splitlines():
        found = re.match(r'^\s*(\d+)\s+(\S+)\s+(\S+)\s+(CPU|NPU)\s+', line)
        if found:
            rows.append({'id': int(found[1]), 'operator_type': found[2],
                         'dtype': found[3], 'target': found[4]})
    internal = [row for row in rows if row['operator_type'] not in ('InputOperator', 'OutputOperator')]
    return {'rows': rows, 'reported_internal_operator_count': len(internal),
            'reported_internal_CPU_operators': [row for row in internal if row['target'] == 'CPU'],
            'reported_internal_targets_all_NPU': all(row['target'] == 'NPU' for row in internal) if internal else None,
            'scope': 'SDK-reported target rows only; CPU input/output conversion and postprocessing remain explicit'}


def runtime_for_camera(output, export_receipt, export_sha):
    base, base_path = load_frozen('run-v12-rknn-block-board.py', RUNTIME_SHA, '_v15_camera_rknn_ABI')
    capture, capture_path = load_frozen('capture-v15-camera-board.py', CAPTURE_SHA, '_v15_camera_capture_contract')
    exporter_path = Path(__file__).with_name('export-v15-camera-board.py')
    base.require_sha(exporter_path, EXPORT_SHA)
    base.SHAPE, base.CORE_MASK = INPUT_SHAPE, 7
    # RKNN packs per-iteration copies despite shared ONNX initializers.  The
    # measured full graph is 1,711,514,807 bytes; keep this diagnostic bounded.
    base.MAX_MODEL_BYTES = 2 * 1024 * 1024 * 1024
    base.__doc__ = __doc__
    old_validate = base.validate_attr
    artifacts = []

    def validate_attr(attr):
        name = bytes(attr.name).decode('utf-8')
        shapes = {'camera_tokens': INPUT_SHAPE, 'pose_enc': OUTPUT_SHAPE}
        if name not in shapes:
            raise ValueError(f'Unexpected complete CameraHead SDK tensor name: {name}')
        # The private module is single-threaded. Reuse the exact float/identity
        # descriptor gate with the appropriate shape, then restore its globals.
        previous = base.SHAPE
        try:
            base.SHAPE = shapes[name]
            return old_validate(attr)
        finally:
            base.SHAPE = previous

    def load_capture(directory, expected_sha):
        import numpy as np
        directory = Path(directory).resolve(strict=True)
        path = directory / 'capture.json'
        base.require_sha(path, expected_sha)
        metadata = json.loads(path.read_text())
        if (metadata.get('kind') != 'original_complete_CameraHead_CPU_capture'
                or metadata.get('status') != 'completed' or metadata.get('script_sha256') != CAPTURE_SHA
                or metadata.get('camera_iterations') != 4 or metadata.get('CPU_trunk_call_count') != 48
                or metadata.get('CPU_camera_forward_call_count') != 3 or metadata.get('npu_used') is not False
                or metadata.get('ORB_used') is not False or metadata.get('source_and_inputs_unchanged') is not True
                or metadata.get('cleanup_errors')):
            raise ValueError('Exact complete original four-iteration CPU capture is required')
        base.require_sha(export_receipt, export_sha)
        exported = json.loads(export_receipt.read_text())
        if (exported.get('status') != 'exported_and_structurally_checked'
                or exported.get('script_sha256') != EXPORT_SHA or exported.get('cleanup_errors')
                or exported.get('source_and_inputs_unchanged') is not True
                or exported.get('capture', {}).get('sha256') != expected_sha
                or exported.get('architecture') != metadata['architecture']
                or exported.get('weight_sharing', {}).get('original_parameters_verified') != 69
                or exported.get('weight_sharing', {}).get('weights_not_duplicated_per_iteration') is not True):
            raise ValueError('Frozen complete original CameraHead ONNX export provenance mismatch')
        pins = {str(path): expected_sha, str(export_receipt): export_sha,
                str(base_path): RUNTIME_SHA, str(capture_path): CAPTURE_SHA,
                str(exporter_path): EXPORT_SHA, str(Path(__file__).resolve()): sha(__file__)}
        arrays = []
        helper, helper_path = capture.base_capture()
        pins[str(helper_path)] = capture.BASE_CAPTURE_SHA
        for name, shape in (('input.npy', INPUT_SHAPE), ('output.npy', OUTPUT_SHAPE)):
            path = directory / name
            item = metadata['artifacts'][name]
            pins[str(path)] = base.require_sha(path, item['sha256'])
            array = np.load(path, allow_pickle=False)
            info = helper.array_info(array)
            if (tuple(array.shape) != shape or item['shape'] != list(shape)
                    or item['tensor_sha256'] != info['sha256'] or path.stat().st_size != item['bytes']):
                raise ValueError('Exact original raw camera token/reference shape/hash mismatch')
            arrays.append(array)
        metadata = dict(metadata, exported_ONNX=exported['onnx'], runtime_scope='complete four-iteration CameraHead')
        return arrays[0], arrays[1], metadata, pins

    def execute_call(api, context, inputs, input_format, timeout_ms, cleanup_errors, activity=None):
        import numpy as np
        request = (base.Input * 1)(base.Input(0, inputs.ctypes.data, inputs.nbytes, 0, 0, input_format))
        returned = (base.Output * 1)()
        returned[0].want_float = 1
        run = base.RunExtend(0, 0, timeout_ms, -1)
        received = base.OutputExtend()
        begin, got = time.perf_counter(), False
        row = {'want_float': 1, 'is_prealloc': 0, 'input_type': 'FP32', 'input_pass_through': 0,
               'input_format': input_format, 'synchronous': True, 'run_timeout_ms': timeout_ms}
        try:
            base.check_call(api.rknn_inputs_set(context, 1, request), 'rknn_inputs_set')
            set_end = time.perf_counter()
            if activity is not None:
                activity['npu_run_api_called'] = True
                activity['npu_run_api_call_count'] = activity.get('npu_run_api_call_count', 0) + 1
            base.check_call(api.rknn_run(context, ct.byref(run)), 'rknn_run')
            run_end = time.perf_counter()
            base.check_call(api.rknn_outputs_get(context, 1, returned, ct.byref(received)), 'rknn_outputs_get')
            got, get_end = True, time.perf_counter()
            if run.frame_id != received.frame_id:
                raise ValueError('Complete CameraHead synchronous frame identity mismatch')
            if not returned[0].buf or returned[0].size != math.prod(OUTPUT_SHAPE) * 4:
                raise ValueError('Invalid complete CameraHead FP32 pose output byte size/pointer')
            view = np.ctypeslib.as_array(ct.cast(returned[0].buf, ct.POINTER(ct.c_float)),
                                       shape=(math.prod(OUTPUT_SHAPE),))
            owned = view.copy().reshape(OUTPUT_SHAPE)
            copy_end = time.perf_counter()
            row.update(input_set_seconds=set_end-begin, run_api_seconds=run_end-set_end,
                       outputs_get_seconds=get_end-run_end, output_copy_seconds=copy_end-get_end,
                       runtime_frame_id=int(run.frame_id), output_bytes=int(returned[0].size))
        finally:
            if got or returned[0].buf:
                release_begin = time.perf_counter()
                code = api.rknn_outputs_release(context, 1, returned)
                end = time.perf_counter()
                row.update(outputs_release_seconds=end-release_begin, output_buffer_released=code == 0,
                           io_total_seconds=end-begin)
                if code:
                    cleanup_errors.append(f'rknn_outputs_release returned {code}')
                base.check_call(code, 'rknn_outputs_release')
        perf = base.PerfRun()
        code = api.rknn_query(context, 4, ct.byref(perf), ct.sizeof(perf))
        row.update(perf_run_query_code=int(code), sdk_run_duration_us=int(perf.run_duration) if code == 0 else None)
        path = output / f'output-{len(artifacts):02d}.npy'
        with path.open('xb') as stream:
            np.save(stream, owned, allow_pickle=False)
        artifacts.append({'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size,
                          'shape': list(OUTPUT_SHAPE), 'dtype': 'float32', 'finite': bool(np.isfinite(owned).all()),
                          'raw_sha256': hashlib.sha256(memoryview(owned).cast('B')).hexdigest()})
        return owned, row

    base.validate_attr, base.load_capture, base.execute_call = validate_attr, load_capture, execute_call
    return base, artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-sha256', required=True)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--capture-sha256', required=True)
    parser.add_argument('--export-receipt', type=Path, required=True)
    parser.add_argument('--export-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--core-mask', type=int, choices=(1, 7), default=7)
    parser.add_argument('--run-timeout-ms', type=int, default=30000)
    parser.add_argument('--collect-perf-detail', action='store_true')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(BASE / 'experiments/vggt-v15-camera') or os.path.lexists(args.output_dir):
        parser.error('Use a new output directory below board experiments/vggt-v15-camera')
    base, artifacts = runtime_for_camera(output, args.export_receipt.resolve(), args.export_sha256)
    base.require_board()
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'receipt.json'
    argv = ['--model', str(args.model), '--model-sha256', args.model_sha256,
            '--capture-dir', str(args.capture_dir), '--capture-sha256', args.capture_sha256,
            '--output', str(path), '--core-mask', str(args.core_mask), '--run-timeout-ms', str(args.run_timeout_ms)]
    if args.collect_perf_detail:
        argv.append('--collect-perf-detail')
    affinity = set(os.sched_getaffinity(0))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            code = base.main(argv)
    finally:
        os.sched_setaffinity(0, affinity)
    report = json.loads(path.read_text())
    report.update(kind='complete_original_four_iteration_CameraHead_RKNN_board_probe', camera_iterations=4,
                  input_shape=list(INPUT_SHAPE), output_shape=list(OUTPUT_SHAPE), ORB_used=False,
                  saved_outputs=artifacts, raw_output_saving_excluded_from_IO_timing=True,
                  cpu_affinity_restored=set(os.sched_getaffinity(0)) == affinity,
                  frozen_runtime_sha256=RUNTIME_SHA, wrapper_sha256=sha(__file__))
    report['timing_limitations'] = [
        'Four sequential calls are not statistical performance validation',
        'This is the complete CameraHead, not whole-model or capture-to-pose timing',
        'Shared ONNX initializers do not imply shared packed RKNN weights',
    ]
    if args.collect_perf_detail:
        report['timing_limitations'].append('SDK profiling changes execution cost; normal timings need a separate run')
    for row in report['calls']:
        if 'perf_detail' in row:
            row['operator_targets'] = operator_targets(row['perf_detail']['raw_text'])
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'error': report.get('error'),
                      'output': str(path), 'sha256': sha(path)}))
    return code


if __name__ == '__main__':
    sys.exit(main())
