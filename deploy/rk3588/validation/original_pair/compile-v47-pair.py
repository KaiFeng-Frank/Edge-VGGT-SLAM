#!/usr/bin/env python3
"""Thin non-quantizing original VGGT pair compiler; never call inference APIs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def load_contract(path, expected):
    if sha(path) != expected:
        raise ValueError('Pair contracts SHA mismatch')
    spec = importlib.util.spec_from_file_location('_v47_compile_contracts', path)
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    return C


def check_ORT(row, pair_sha, allow_rejected=False):
    if (row.get('schema') != 'original_VGGT_pair_CPU_ORT_check_v47'
            or row.get('pair_export_sha256') != pair_sha
            or row.get('successful_session_run_count') != 4
            or row.get('session_released') is not True
            or row.get('source_and_inputs_unchanged') is not True
            or row.get('input_tensor_unchanged') is not True
            or row.get('cleanup_errors') or row.get('NPU_used') is not False
            or row.get('inference_on_board_only') is not True
            or row.get('rtol') != 1e-4 or row.get('atol') != 1e-5
            or row.get('strict_FP32_gate_passed') not in (True, False)):
        raise ValueError('Actual completed board pair ORT evidence required')
    expected_status = 'passed' if row['strict_FP32_gate_passed'] else 'numerical_rejected'
    if row.get('status') != expected_status or (not row['strict_FP32_gate_passed'] and not allow_rejected):
        raise ValueError('Pair ORT rejected; explicit diagnostic override required without adopting it')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('pair-dir', 'ort-receipt', 'contracts', 'parent-compiler', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('pair-sha256', 'ort-sha256', 'contracts-sha256'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--diagnostic-allow-ORT-rejected', action='store_true')
    a = p.parse_args(argv)
    C = load_contract(a.contracts, a.contracts_sha256)
    # SDK compiler intermediates must stay on tmpfs, including TMPDIR.
    if (not Path.cwd().resolve().is_relative_to('/dev/shm')
            or not Path(os.environ.get('TMPDIR', '/tmp')).resolve().is_relative_to('/dev/shm')):
        p.error('Set cwd and TMPDIR to existing /dev/shm compiler work directories')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    pins = {str(Path(__file__).resolve()): sha(__file__), str(a.contracts.resolve()): a.contracts_sha256}
    report = {'schema': 'v47_original_VGGT_pair_RKNN_compilation', 'status': 'failed',
              'script_sha256': sha(__file__), 'contracts_sha256': a.contracts_sha256,
              'parent_compiler_sha256': C.COMPILER_PARENT_SHA,
              'pair_export_sha256': a.pair_sha256, 'ORT_receipt_sha256': a.ort_sha256,
              'model_inference_executed': False, 'candidate_accepted': False,
              'historical_RKNN_FP32_rejections_retained': True, 'cleanup_errors': []}
    saved_argv = sys.argv
    try:
        export, arrays, graph, bound = C.load_pair(a.pair_dir, a.pair_sha256)
        pins.update(bound)
        del arrays
        C.pin(a.ort_receipt, a.ort_sha256, pins)
        ort = json.loads(a.ort_receipt.read_text())
        check_ORT(ort, a.pair_sha256, a.diagnostic_allow_ORT_rejected)
        report['board_ORT_gate_passed'] = ort['strict_FP32_gate_passed']
        report['diagnostic_allow_ORT_rejected'] = a.diagnostic_allow_ORT_rejected
        C.pin(a.parent_compiler, C.COMPILER_PARENT_SHA, pins)
        parent = C.load_helper(a.parent_compiler, C.COMPILER_PARENT_SHA)
        raw_dir = a.output_dir / 'compiler'
        sys.argv = [str(a.parent_compiler), '--onnx', str(graph), '--onnx-sha256', C.GRAPH_SHA,
                    '--output-dir', str(raw_dir)]
        code = parent.main()
        raw = json.loads((raw_dir / 'compile.json').read_text())
        report['parent_compile_receipt'] = {'file': 'compiler/compile.json',
                                           'sha256': sha(raw_dir / 'compile.json')}
        for key in ('configuration', 'operations', 'released', 'versions', 'peak_rss_mib'):
            report[key] = raw.get(key)
        report['input'] = {'file': 'pair.onnx', **export['onnx']}
        if code != 0 or raw.get('status') != 'compiled_not_runtime_validated' or raw.get('release_error'):
            raise ValueError(f'Original FP16 compiler failed: {raw.get("error", raw.get("release_error"))}')
        model = raw_dir / 'block.rknn'
        report['output'] = {'file': 'compiler/block.rknn', 'size': model.stat().st_size, 'sha256': sha(model)}
        report['status'] = 'compiled_not_runtime_validated'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        sys.argv = saved_argv
        after, unchanged = C.recheck(pins)
        report.update(source_hashes=pins, source_hashes_after=after, source_and_inputs_unchanged=unchanged)
        if not unchanged:
            report['cleanup_errors'].append('Source/input changed during compilation')
            report['status'] = 'failed'
        (a.output_dir / 'compile.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'compile_sha256': sha(a.output_dir / 'compile.json'),
                      'error': report.get('error')}))
    return 0 if report['status'] == 'compiled_not_runtime_validated' else 2


if __name__ == '__main__':
    sys.exit(main())
