#!/usr/bin/env python3
"""Compile the unchanged original VGGT pair with Flash Attention off/on.

Both builds are nonquantizing, in the same isolated RK3588S toolchain. They use
fresh SDK objects/workdirs and differ only in enable_flash_attention. This is
a diagnostic, never a full-model, numerical or real-time acceptance result.
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
BASE = Path('/mnt/vggt-data/vggt-slam2')
QUANT_CONTRACT_SHA = '47237b1a1e1614822c75180efd15be3e5c708b610f587f698fe9b37fcf1bc649'
SDK_SOURCE_SHA = '63e4506fb851214e79f933f9de0d27dab04ecf93f6f46959c07c5787433fd05e'
HELDOUT_ORT_SHA = '818ca23f2b1e5986d57d2046dfa734e067faac623a0c0eaf539c21ebbf961f3c'
ENV = {'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '2', 'NUMEXPR_NUM_THREADS': '2'}
CONFIG = {'target_platform': 'rk3588', 'float_dtype': 'float16', 'optimization_level': 3,
          'mean_values': None, 'std_values': None, 'quant_img_RGB2BGR': False,
          'model_pruning': False, 'remove_reshape': False, 'quantize_weight': False}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path, expected):
    if sha(path) != expected:
        raise ValueError('Frozen source SHA mismatch')
    spec = importlib.util.spec_from_file_location('_v50_flash_source', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def operations(api, graph, output, flash, row, board_guard):
    board_guard()
    if type(flash) is not bool:
        raise ValueError('Explicit Boolean Flash Attention choice required')
    config = dict(CONFIG, enable_flash_attention=flash)
    row['configuration'] = config
    row['build_options'] = {'do_quantization': False, 'dataset': None, 'auto_hybrid': False}
    steps = [('config', lambda: api.config(**config)),
             ('load_onnx', lambda: api.load_onnx(model=str(graph))),
             ('build', lambda: api.build(do_quantization=False, dataset=None, auto_hybrid=False)),
             ('export_rknn', lambda: api.export_rknn(str(output)))]
    for name, function in steps:
        entry = {'name': name, 'status': 'failed'}
        row['operations'].append(entry)
        started = time.monotonic_ns()
        try:
            code = function()
            entry['code'] = code
            if code is not None and (type(code) is not int or code != 0):
                raise RuntimeError(f'{name} returned {code}')
            entry['status'] = 'completed'
        except BaseException as exc:
            entry['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            entry['seconds'] = (time.monotonic_ns() - started) / 1e9


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['quant-contracts', 'pair-contracts', 'pair-dir', 'original-ort', 'heldout-ort',
                'toolchain-prefix', 'output-dir', 'work-dir']:
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'linux' or platform.machine().lower() not in ('aarch64', 'arm64'):
        parser.error('Actual SDK compilation runs only on RK3588S')
    C = load(args.quant_contracts, QUANT_CONTRACT_SHA)
    identity = C.require_board()
    prefix = args.toolchain_prefix.resolve()
    output, work = args.output_dir.resolve(), args.work_dir.resolve()
    if (Path.cwd() != BASE or Path(sys.prefix).resolve() != prefix or prefix == BASE / 'venv'
            or not output.is_relative_to(BASE / 'experiments') or output == BASE / 'experiments'
            or not work.is_relative_to('/dev/shm') or work == Path('/dev/shm')):
        parser.error('Isolated board toolchain, project cwd and new experiment/tmpfs directories required')
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'v50_original_pair_flash_config_board_compile', 'status': 'failed',
              'script_sha256': sha(__file__), 'board': identity, 'machine': platform.machine(),
              'toolchain_prefix': str(prefix), 'original_graph_unchanged': True,
              'actual_calibration_performed': False, 'runtime_API_called': False,
              'host_inference_performed': False, 'candidate_accepted': False,
              'full_model_quality_validated': False, 'real_time_validated': False,
              'variants': {}, 'cleanup_errors': []}
    pins = {str(Path(__file__).resolve()): sha(__file__), str(args.quant_contracts.resolve()): QUANT_CONTRACT_SHA}
    before = {k: os.environ.get(k) for k in [*ENV, 'PATH', 'TMPDIR']}
    cwd = Path.cwd()
    try:
        work.mkdir(parents=True, exist_ok=False)
        os.environ.update(ENV)
        os.environ['PATH'] = str(prefix / 'bin') + os.pathsep + os.environ.get('PATH', '')
        versions = {n: importlib.metadata.version(n) for n in ['rknn-toolkit2', 'torch', 'numpy', 'onnx', 'onnxruntime', 'protobuf']}
        C.check_versions(versions, sys.version_info, sys.prefix, prefix)
        report['versions'] = versions
        P = C.load_source(args.pair_contracts, C.PAIR_CONTRACTS_SHA, pins)
        pair, arrays, graph, bound = P.load_pair(args.pair_dir, C.PAIR_EXPORT_SHA)
        pins.update(bound)
        del arrays
        C.pin(args.original_ort, C.BOARD_ORT_SHA, pins)
        C.check_board_ort(json.loads(args.original_ort.read_text()))
        C.pin(args.heldout_ort, HELDOUT_ORT_SHA, pins)
        heldout = json.loads(args.heldout_ort.read_text())
        if heldout['status'] != 'numerical_rejected':
            raise ValueError('Preserve the exact new heldout ORT failure')
        report.update(pair_export_sha256=C.PAIR_EXPORT_SHA, pair_graph_sha256=C.PAIR_GRAPH_SHA,
                      original_checkpoint=pair['checkpoint'], new_heldout_ORT_status=heldout['status'],
                      new_heldout_ORT_sha256=HELDOUT_ORT_SHA)
        from rknn.api import RKNN
        source = Path(inspect.getfile(RKNN)).resolve()
        if not source.is_relative_to(prefix):
            raise ValueError('SDK module escaped isolated toolchain')
        C.pin(source, SDK_SOURCE_SHA, pins)
        for name, flash in [('ordinary', False), ('flash', True)]:
            folder, scratch = output / name, work / name
            folder.mkdir(); scratch.mkdir()
            os.chdir(scratch); os.environ['TMPDIR'] = str(scratch)
            row = {'status': 'failed', 'operations': [], 'released': False}
            report['variants'][name] = row
            api = None
            try:
                api = RKNN(verbose=True, verbose_file=str(folder / 'sdk.log'))
                model = folder / 'pair.rknn'
                operations(api, graph, model, flash, row, C.require_board)
                if not model.is_file() or model.stat().st_size <= 0:
                    raise ValueError('No nonempty compiled pair')
                row['output'] = {'file': str(model.relative_to(output)), 'bytes': model.stat().st_size, 'sha256': sha(model)}
                row['status'] = 'compiled_not_runtime_validated'
            finally:
                if api is not None:
                    code = api.release(); row['release_code'] = code
                    row['released'] = code is None or type(code) is int and code == 0
                    if not row['released']:
                        report['cleanup_errors'].append(name + ' SDK release failed')
        report['status'] = 'both_compiled_not_runtime_validated'
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        os.chdir(cwd)
        for key, value in before.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value
        report['environment_restored'] = {k: os.environ.get(k) for k in before} == before
        report['cwd_restored'] = Path.cwd() == cwd
        after, unchanged = C.recheck(pins)
        report.update(source_and_input_hashes=pins, source_and_input_hashes_after=after,
                      source_and_inputs_unchanged=unchanged)
        if not unchanged or not report['environment_restored'] or not report['cwd_restored']:
            report['cleanup_errors'].append('Source/input or process restoration failure')
        if report['cleanup_errors']: report['status'] = 'failed'
        report['peak_compiler_RSS_MiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        (output / 'compile.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'error': report.get('error'),
                      'receipt_sha256': sha(output / 'compile.json')}), flush=True)
    return 0 if report['status'] == 'both_compiled_not_runtime_validated' else 2


if __name__ == '__main__':
    raise SystemExit(main())
