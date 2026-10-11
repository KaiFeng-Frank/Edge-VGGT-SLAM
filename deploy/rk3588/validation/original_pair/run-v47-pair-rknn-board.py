#!/usr/bin/env python3
"""Original VGGT paired block SDK diagnostic, using the frozen original ABI.

One cold and three warm calls save outputs after the measured synchronous IO.
Profile uses a separate invocation. FP32 rejection remains rejection; this is
never a whole-model executor, automatic adoption, or fresh-pose benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_contract(path, expected):
    if sha(path) != expected:
        raise ValueError('Pair contracts SHA mismatch')
    spec = importlib.util.spec_from_file_location('_v47_sdk_contracts', path)
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    return C


def bind_pair_runtime(runtime, C, inputs, reference, metadata, pins, output_dir, saved):
    """Private module only; inherited SDK IO is unchanged, including cleanup."""
    if (tuple(inputs.shape) != C.INPUT_SHAPE or tuple(reference.shape) != C.OUTPUT_SHAPE
            or inputs.nbytes != reference.nbytes or inputs.nbytes != 4 * math.prod(C.OUTPUT_SHAPE)):
        raise ValueError('Rank-three pair IO must have the same original floating element count')
    # The inherited SDK copy uses SHAPE for output ownership. Logical input
    # shape is checked separately by its queried name; no global shape alias.
    runtime.SHAPE = C.OUTPUT_SHAPE
    runtime.CORE_MASK = 7
    runtime.validate_attr = lambda attr: C.validate_sdk_attr(runtime.attr_record(attr))
    runtime.comparison = C.compare

    def load_actual(directory, expected_sha):
        if (Path(directory).resolve() != Path(metadata['pair_dir']).resolve()
                or expected_sha != metadata['pair_export_sha256']):
            raise ValueError('Frozen runtime requested another pair/capture')
        return inputs, reference, metadata, dict(pins)

    runtime.load_capture = load_actual
    original_execute = runtime.execute_call

    def execute_and_save(*args, **kwargs):
        actual, timing = original_execute(*args, **kwargs)
        import numpy as np
        path = Path(output_dir) / f'actual_{len(saved):02d}.npy'
        np.save(path, actual, allow_pickle=False)
        saved.append({'file': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path),
                      'shape': list(actual.shape), 'dtype': str(actual.dtype),
                      'raw_sha256': hashlib.sha256(memoryview(actual).cast('B')).hexdigest()})
        return actual, timing

    runtime.execute_call = execute_and_save


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('pair-dir', 'contracts', 'runtime-helper', 'compile-receipt', 'model', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('pair-sha256', 'contracts-sha256', 'compile-sha256', 'compiler-sha256', 'model-sha256'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--collect-perf-detail', action='store_true')
    p.add_argument('--diagnostic-allow-FP32-rejected', action='store_true', required=True)
    a = p.parse_args(argv)
    C = load_contract(a.contracts, a.contracts_sha256)
    C.require_board()
    if Path.cwd().resolve() != C.BASE or not a.output_dir.resolve().is_relative_to(C.BASE / 'experiments'):
        p.error('Board project cwd and new experiments output required')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'v47_original_VGGT_frame_global_pair_RKNN_diagnostic',
              'status': 'failed', 'script_sha256': sha(__file__),
              'contracts_sha256': a.contracts_sha256, 'pair_export_sha256': a.pair_sha256,
              'compile_sha256': a.compile_sha256, 'compiler_sha256': a.compiler_sha256,
              'model_sha256': a.model_sha256, 'runtime_helper_sha256': C.RUNTIME_PARENT_SHA,
              'inference_on_board_only': True, 'candidate_accepted': False,
              'full_model_run': False, 'real_time_validated': False, 'whole_model_quality_validated': False,
              'historical_RKNN_FP32_rejections_retained': True, 'normal_benchmark': not a.collect_perf_detail,
              'diagnostic_allow_FP32_rejected': True, 'output_arrays': [], 'cleanup_errors': [],
              'all_internal_operators_on_NPU_verified': False,
              'input_route': 'frame0 original input -> complete frame0 -> original view -> complete global0'}
    pins = {str(Path(__file__).resolve()): report['script_sha256'],
            str(a.contracts.resolve()): a.contracts_sha256}
    affinity = os.sched_getaffinity(0)
    env_keys = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
    env = {k: os.environ.get(k) for k in env_keys}
    try:
        export, arrays, _, bound = C.load_pair(a.pair_dir, a.pair_sha256)
        pins.update(bound)
        C.pin(a.compile_receipt, a.compile_sha256, pins)
        compiled = json.loads(a.compile_receipt.read_text())
        C.check_compile(compiled, a.pair_sha256, a.model_sha256)
        if compiled.get('script_sha256') != a.compiler_sha256:
            raise ValueError('Pair compiler source SHA mismatch')
        C.pin(a.model, a.model_sha256, pins)
        if a.model.stat().st_size != compiled['output']['size']:
            raise ValueError('Pair compiled model size mismatch')
        C.pin(a.runtime_helper, C.RUNTIME_PARENT_SHA, pins)
        C.pin(Path('/usr/lib/librknnrt.so'), C.LIBRARY_SHA, pins)
        runtime = C.load_helper(a.runtime_helper, C.RUNTIME_PARENT_SHA)
        metadata = {'actual_kind': 'original_VGGT_complete_frame0_global0_pair',
                    'pair_dir': str(a.pair_dir.resolve()), 'pair_export_sha256': a.pair_sha256,
                    'profile_sha256': C.PROFILE_SHA, 'original_FP32_reference': True,
                    'input_shape': list(C.INPUT_SHAPE), 'output_shape': list(C.OUTPUT_SHAPE),
                    'original_parameter_tensors': 36, 'checkpoint': export['checkpoint']}
        bind_pair_runtime(runtime, C, arrays['input.npy'], arrays['output.npy'], metadata, pins,
                          a.output_dir, report['output_arrays'])
        raw_path = a.output_dir / 'driver.json'
        argv = ['--model', str(a.model), '--model-sha256', a.model_sha256,
                '--capture-dir', str(a.pair_dir), '--capture-sha256', a.pair_sha256,
                '--output', str(raw_path), '--core-mask', '7']
        if a.collect_perf_detail:
            argv.append('--collect-perf-detail')
        code = runtime.main(argv)
        raw = json.loads(raw_path.read_text())
        report.update(driver_receipt={'file': 'driver.json', 'sha256': sha(raw_path)},
                      raw_driver_kind_is_legacy_original_DINO_label=True,
                      npu_run_api_call_count=raw['npu_run_api_call_count'],
                      successful_synchronous_call_count=raw['successful_synchronous_call_count'],
                      context_destroyed=raw.get('context_destroyed'),
                      input_shape=list(C.INPUT_SHAPE), output_shape=list(C.OUTPUT_SHAPE),
                      cleanup_errors=list(raw['cleanup_errors']))
        if code != 0 or raw['status'] != 'completed' or raw.get('context_destroyed') is not True:
            raise ValueError(f'Pair SDK failed: {raw.get("error")}')
        gate = all(row['versus_cpu_reference']['strict_FP32_gate_passed'] for row in raw['calls'])
        report['strict_FP32_gate_passed'] = gate
        report['status'] = ('completed_FP32_pair_gate_diagnostic_not_adopted' if gate
                            else 'completed_FP32_rejected_diagnostic_not_adopted')
        report['warm_io_seconds'] = raw['subsequent_io_seconds']
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        try:
            os.sched_setaffinity(0, affinity)
            for k, v in env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            report['affinity_restored'] = os.sched_getaffinity(0) == affinity
            after, unchanged = C.recheck(pins)
            report.update(source_hashes=pins, source_hashes_after=after, source_and_inputs_unchanged=unchanged)
            if not unchanged or not report['affinity_restored']:
                report['cleanup_errors'].append('Source/input or CPU affinity restoration mismatch')
        except Exception as exc:
            report['cleanup_errors'].append(str(exc))
        if report['cleanup_errors']:
            report['status'] = 'failed'
        (a.output_dir / 'run.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'run_sha256': sha(a.output_dir / 'run.json'),
                      'error': report.get('error')}))
    return 0 if report['status'].startswith('completed_') else 2


if __name__ == '__main__':
    sys.exit(main())
