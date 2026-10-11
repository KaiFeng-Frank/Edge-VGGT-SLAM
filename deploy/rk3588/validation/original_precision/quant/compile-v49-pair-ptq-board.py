#!/usr/bin/env python3
"""Original VGGT paired graph PTQ, actual calibration/build only on RK3588S.

Use the isolated ARM64 CPython 3.11 RKNN 2.3.2 toolchain with Torch <=2.2.0.
Eight new original frame0 capture tensors calibrate this one-input graph; two
disjoint original Torch references are excluded and reserved for later SDK
evaluation. This compilation stage does not call runtime/inference APIs or
adopt the candidate. The root agent alone runs actual board jobs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

sys.dont_write_bytecode = True
ENVIRONMENT = {'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1',
               'MKL_NUM_THREADS': '2', 'NUMEXPR_NUM_THREADS': '2'}
PACKAGES = ('rknn-toolkit2', 'torch', 'numpy', 'onnx', 'onnxruntime', 'protobuf')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def load_contract(path, expected):
    if sha(path) != expected:
        raise ValueError('PTQ provenance contracts SHA mismatch')
    spec = importlib.util.spec_from_file_location('_v49_original_pair_ptq_contracts', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_compile_operations(toolkit, graph, dataset, output, report, C):
    """A second board gate protects even direct calls; tests inject a fake API."""
    C.require_board()
    config = {k: v for k, v in C.CONFIGURATION.items() if k != 'do_quantization'}
    # Original captured tensors already contain the exact model prefix input.
    # Defaults None mean zero mean and unit std; no input transform is added.
    config.update(mean_values=None, std_values=None, quant_img_RGB2BGR=False)
    operations = [
        ('config', lambda: toolkit.config(**config)),
        ('load_onnx', lambda: toolkit.load_onnx(model=str(Path(graph).resolve()))),
        ('build', lambda: toolkit.build(do_quantization=True, dataset=str(Path(dataset).resolve()), auto_hybrid=False)),
        ('export_rknn', lambda: toolkit.export_rknn(str(Path(output).resolve()))),
    ]
    for name, operation in operations:
        print('START', name, flush=True)
        row = {'name': name, 'code': None, 'status': 'failed'}
        report['operations'].append(row)
        begin = time.monotonic_ns()
        if name == 'build':
            report['calibration_build_api_invoked'] = True
            report['calibration_may_have_executed'] = True
        try:
            code = operation()
            row.update(code=code, seconds=(time.monotonic_ns() - begin) / 1e9)
            if code is not None and (type(code) is not int or code != 0):
                raise RuntimeError(f'{name} returned {code}')
            row['status'] = 'completed'
            if name == 'build':
                report['calibration_executed'] = True
        except Exception as exc:
            row.setdefault('seconds', (time.monotonic_ns() - begin) / 1e9)
            row['error'] = f'{type(exc).__name__}: {exc}'
            raise


def release_toolkit(toolkit, report):
    if toolkit is None:
        return
    try:
        code = toolkit.release()
        report['release_code'] = code
        if code is not None and (type(code) is not int or code != 0):
            raise RuntimeError(f'RKNN release returned {code}')
        report['released'] = True
    except Exception as exc:
        report['released'] = False
        report['cleanup_errors'].append(f'RKNN release: {type(exc).__name__}: {exc}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('contracts', 'pair-contracts', 'pair-dir', 'board-ort-receipt',
                 'capture-contracts', 'capture-dir', 'toolchain-prefix', 'output-dir', 'work-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('contracts-sha256', 'capture-contracts-sha256', 'capture-sha256'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    # No captured arrays, Torch, ONNX, RKNN module or toolkit is loaded on host.
    if sys.platform != 'linux' or platform.machine().lower() not in ('aarch64', 'arm64'):
        parser.error('Actual quantizing compilation runs only on the Linux RK3588S board')
    C = load_contract(args.contracts, args.contracts_sha256)
    compatible = C.require_board()
    output_dir, work_dir = args.output_dir.resolve(), args.work_dir.resolve()
    if (Path.cwd().resolve() != C.BASE or output_dir == C.BASE / 'experiments'
            or not output_dir.is_relative_to(C.BASE / 'experiments')):
        parser.error('Board project cwd and new experiments output directory required')
    if work_dir == Path('/dev/shm') or not work_dir.is_relative_to('/dev/shm'):
        parser.error('New SDK cwd/TMPDIR must be confined beneath board /dev/shm')
    if Path(sys.prefix).resolve() != args.toolchain_prefix.resolve() or Path(sys.prefix).resolve() == C.BASE / 'venv':
        parser.error('Run with the explicitly selected isolated toolchain Python')
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'v49_original_VGGT_pair_board_PTQ_compile', 'status': 'failed',
              'script_sha256': sha(__file__), 'contracts_sha256': args.contracts_sha256,
              'pair_contracts_sha256': C.PAIR_CONTRACTS_SHA, 'pair_export_sha256': C.PAIR_EXPORT_SHA,
              'pair_graph_sha256': C.PAIR_GRAPH_SHA, 'passed_board_ORT_receipt_sha256': C.BOARD_ORT_SHA,
              'capture_contracts_sha256': args.capture_contracts_sha256,
              'capture_manifest_sha256': args.capture_sha256,
              'configuration': dict(C.CONFIGURATION), 'device_tree_compatible': compatible,
              'machine': platform.machine(), 'python': sys.version, 'toolchain_prefix': str(args.toolchain_prefix.resolve()),
              'board': {'machine': platform.machine(), 'device_tree_compatible': compatible, 'python': sys.version},
              'production_python_excluded': True, 'model_inference_on_host': False,
              'calibration_executed_on_board_only': True, 'calibration_executed': False,
              'calibration_build_api_invoked': False, 'calibration_may_have_executed': False,
              'runtime_inference_API_called': False, 'candidate_accepted': False,
              'full_model_run': False, 'real_time_validated': False, 'whole_model_quality_validated': False,
              'historical_FP32_rejections_retained': True, 'released': None, 'cleanup_errors': [],
              'operations': [], 'sdk_work_dir': str(work_dir),
              'calibration_scope': 'complete original frame0/global0 pair; one saved frame0 tensor per row',
              'input_transform': {'mean_values': None, 'std_values': None, 'quant_img_RGB2BGR': False,
                                  'transpose': False, 'extra_batch_dimension': False},
              'build_options': {'do_quantization': True, 'auto_hybrid': False},
              'limitations': ['Quantization/compilation alone does not prove actual SDK IO, operator precision or speed',
                              'Internal mixed floating/integer execution must be inspected after actual deployment',
                              'Calibration is eight pairs from one real sequence; generalization remains unproven',
                              'Heldout accuracy, full VGGT/SLAM ATE/RPE and real-time targets are not evaluated here',
                              'Frozen original pair CPU ORT proof covers its captured input; heldout numerical validation is separate']}
    pins = {str(Path(__file__).resolve()): report['script_sha256'],
            str(args.contracts.resolve()): args.contracts_sha256}
    original_cwd = Path.cwd()
    environment_keys = tuple(ENVIRONMENT) + ('TMPDIR', 'PATH')
    original_environment = {key: os.environ.get(key) for key in environment_keys}
    toolkit, error = None, None
    started = time.monotonic_ns()
    try:
        work_dir.mkdir(parents=True, exist_ok=False)
        os.environ.update(ENVIRONMENT)
        os.environ['TMPDIR'] = str(work_dir)
        os.environ['PATH'] = str(args.toolchain_prefix.resolve() / 'bin') + os.pathsep + os.environ.get('PATH', '')
        report['environment'] = {key: os.environ.get(key) for key in environment_keys}
        report['versions'] = {name: importlib.metadata.version(name) for name in PACKAGES}
        C.check_versions(report['versions'], sys.version_info, sys.prefix, args.toolchain_prefix)
        P = C.load_source(args.pair_contracts, C.PAIR_CONTRACTS_SHA, pins)
        pair, arrays, graph, bound = P.load_pair(args.pair_dir, C.PAIR_EXPORT_SHA)
        pins.update(bound)
        del arrays
        C.pin(args.board_ort_receipt, C.BOARD_ORT_SHA, pins)
        ort = json.loads(args.board_ort_receipt.read_text())
        C.check_board_ort(ort)
        capture_contracts = C.load_source(args.capture_contracts, args.capture_contracts_sha256, pins)
        captured, samples, bound = capture_contracts.load_capture(args.capture_dir, args.capture_sha256)
        pins.update(bound)
        if samples != captured['samples']:
            raise ValueError('Capture loader returned another sample set')
        if captured.get('script_sha256') != capture_contracts.CAPTURE_SOURCE_SHA:
            raise ValueError('Actual original capture source is not the frozen capture script')
        if captured.get('contracts_sha256') != args.capture_contracts_sha256:
            raise ValueError('Actual capture receipt binds another capture contracts source')
        dataset, dataset_row = C.prepare_dataset(args.capture_dir, captured, pair, output_dir, pins)
        report.update(capture_script_sha256=captured['script_sha256'], original_checkpoint=pair['checkpoint'],
                      capture_samples=captured['samples'], dataset=dataset_row,
                      heldout_sample_ids=dataset_row['heldout_samples_excluded'],
                      input={'file': 'pair.onnx', 'sha256': C.PAIR_GRAPH_SHA, 'bytes': graph.stat().st_size})
        import torch
        import numpy as np
        import onnx
        from rknn.api import RKNN
        modules = {'torch': torch, 'numpy': np, 'onnx': onnx}
        report['module_source_paths'] = {}
        for name, module in modules.items():
            source = Path(module.__file__).resolve()
            if not source.is_relative_to(args.toolchain_prefix.resolve()):
                raise ValueError(f'{name} leaked from outside the selected board toolchain prefix')
            report['module_source_paths'][name] = str(source)
            pins[str(source)] = C.sha(source)
        rknn_source = Path(inspect.getfile(RKNN)).resolve()
        if not rknn_source.is_relative_to(args.toolchain_prefix.resolve()):
            raise ValueError('RKNN compiler leaked from outside the selected board toolchain prefix')
        report['module_source_paths']['rknn'] = str(rknn_source)
        pins[str(rknn_source)] = C.sha(rknn_source)
        report['actual_module_versions'] = {name: str(module.__version__) for name, module in modules.items()}
        onnx.checker.check_model(str(graph), full_check=True)
        os.chdir(work_dir)
        toolkit = RKNN(verbose=True, verbose_file=str(output_dir / 'sdk.log'))
        output = output_dir / 'pair-int8.rknn'
        run_compile_operations(toolkit, graph, dataset, output, report, C)
        if not output.is_file() or output.stat().st_size <= 0:
            raise ValueError('SDK exported no nonempty original pair candidate')
        report['output'] = {'file': output.name, 'size': output.stat().st_size, 'sha256': C.sha(output)}
        report['status'] = 'compiled_not_runtime_validated'
    except BaseException as exc:
        error = exc
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        release_toolkit(toolkit, report)
        try:
            after, unchanged = C.recheck(pins)
            report.update(source_and_input_hashes=pins, source_and_input_hashes_after=after,
                          source_and_inputs_unchanged=unchanged)
            if not unchanged:
                report['cleanup_errors'].append('Original source/input/toolchain files changed during PTQ')
        except Exception as exc:
            report['cleanup_errors'].append(f'Source recheck: {type(exc).__name__}: {exc}')
        try:
            os.chdir(original_cwd)
            for key, value in original_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            report['cwd_restored'] = Path.cwd() == original_cwd
            report['environment_restored'] = {key: os.environ.get(key) for key in environment_keys} == original_environment
            if not report['cwd_restored'] or not report['environment_restored']:
                report['cleanup_errors'].append('Process cwd/environment restoration mismatch')
        except Exception as exc:
            report['cleanup_errors'].append(f'Process restoration: {type(exc).__name__}: {exc}')
        report['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        report['peak_rss_scope'] = 'compiler process high-water, not SLAM or 8 GB deployment validation'
        report['total_seconds'] = (time.monotonic_ns() - started) / 1e9
        if report['cleanup_errors'] or (report['status'] == 'compiled_not_runtime_validated' and report['released'] is not True):
            report['status'] = 'failed'
        with (output_dir / 'compile.json').open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
    print(json.dumps({'status': report['status'], 'error': report.get('error'),
                      'compile_sha256': C.sha(output_dir / 'compile.json')}, allow_nan=False), flush=True)
    return 0 if not error and report['status'] == 'compiled_not_runtime_validated' else 2


if __name__ == '__main__':
    raise SystemExit(main())
