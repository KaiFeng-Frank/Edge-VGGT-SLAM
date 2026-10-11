#!/usr/bin/env python3
"""Board-only two original-token FP32 ORT checks, fixed original-logit gate."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import depth_contract as C
import export_original_depth as E


def metrics(actual, reference):
    if (actual.dtype != np.float32 or reference.dtype != np.float32
            or actual.shape != C.OUTPUT_SHAPE or reference.shape != C.OUTPUT_SHAPE
            or not np.isfinite(actual).all() or not np.isfinite(reference).all()):
        raise ValueError('Original-reference finite FP32 per-frame logits required')
    expected = reference.astype(np.float64)
    difference = actual.astype(np.float64)-expected
    magnitude = np.linalg.norm(expected.ravel())
    tolerance = 1e-5+1e-4*np.abs(expected)
    return {'max_abs': float(np.abs(difference).max()),
            'rmse': float(np.sqrt(np.mean(difference*difference))),
            'relative_l2': float(np.linalg.norm(difference.ravel())/magnitude) if magnitude else None,
            'elements': actual.size, 'outside_fixed_gate': int(np.count_nonzero(np.abs(difference) > tolerance)),
            'maximum_tolerance_ratio': float(np.max(np.abs(difference)/tolerance)),
            'fixed_gate_pass': bool(np.all(np.abs(difference) <= tolerance)
                                    and np.allclose(actual, reference, rtol=1e-4, atol=1e-5))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--capture-sha256', required=True)
    parser.add_argument('--export-dir', type=Path, required=True)
    parser.add_argument('--export-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    compatible = Path('/proc/device-tree/compatible').read_bytes().decode(errors='replace')
    if sys.platform != 'linux' or platform.machine().lower() not in ('aarch64','arm64') or 'rockchip,rk3588' not in compatible:
        parser.error('All original ORT checks are restricted to Linux RK3588/S')
    directory, capture = E.load_capture(args.capture_dir, args.capture_sha256)
    exported_dir = args.export_dir.resolve(strict=True)
    receipt = exported_dir/'export.json'
    if C.sha256(receipt) != args.export_sha256:
        raise ValueError('Exact original static export SHA required')
    exported = json.loads(receipt.read_text())
    if (exported['schema'] != 'v54_original_VGGT_depth_logits_ONNX_manifest'
            or exported['capture_sha256'] != args.capture_sha256 or exported['checkpoint'] != C.CHECKPOINT
            or exported['graph']['input_names'] != list(C.INPUT_NAMES)
            or exported['graph']['input_shapes'] != {name: list(C.INPUT_SHAPE) for name in C.INPUT_NAMES}
            or exported['graph']['output_name'] != 'logits' or exported['graph']['output_shape'] != list(C.OUTPUT_SHAPE)):
        raise ValueError('Original exported graph contract differs')
    model_path = E.artifact_path(exported_dir, exported['graph'])
    tokens = E.load_archive(directory, capture['artifacts']['tokens'])
    references = E.load_archive(directory, capture['artifacts']['references'])
    if (set(tokens) != set(C.INPUT_NAMES) or any(value.shape != (1,2,53,2048) for value in tokens.values())
            or references['raw_logits'].shape != (2,2,84,112)):
        raise ValueError('Original retained input/reference frame order differs')
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    affinity = os.sched_getaffinity(0)
    report = {'schema': 'v54_original_depth_FP32_ORT_check', 'status': 'failed',
        'script_sha256': C.sha256(__file__), 'capture_sha256': args.capture_sha256,
        'export_sha256': args.export_sha256, 'checkpoint': dict(C.CHECKPOINT),
        'rtol': 1e-4, 'atol': 1e-5, 'calls': [], 'ORT_run_attempts': 0,
        'actual_inference_on_board_only': True, 'model_only_original_VGGT': True,
        'SDK_or_compiler_executed': False, 'candidate_accepted': False}
    try:
        if not {6,7} <= affinity:
            raise ValueError('Original A76 cores 6,7 unavailable')
        os.sched_setaffinity(0, {6,7})
        import onnxruntime as ort
        options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(str(model_path), options, providers=['CPUExecutionProvider'])
        report['ORT_version'] = ort.__version__
        actual = []
        for frame in range(2):
            feeds = {name: np.ascontiguousarray(value[:,frame:frame+1,5:].reshape(C.INPUT_SHAPE))
                     for name, value in tokens.items()}
            begin = time.perf_counter(); report['ORT_run_attempts'] += 1
            value = session.run(['logits'], feeds)[0]
            seconds = time.perf_counter()-begin
            row = metrics(value, references['raw_logits'][frame:frame+1])
            row.update(frame=frame, ORT_seconds=seconds,
                       original_input_raw_sha256={name: C.raw_sha256(item) for name,item in feeds.items()})
            report['calls'].append(row); actual.append(value)
        path = output/'ort-logits.npy'
        with path.open('xb') as stream:
            np.save(stream, np.concatenate(actual, axis=0), allow_pickle=False)
        report['private_output'] = {'file': path.name, 'bytes': path.stat().st_size, 'sha256': C.sha256(path)}
        report['status'] = 'passed_original_FP32_logit_gate' if all(row['fixed_gate_pass'] for row in report['calls']) else 'numerical_rejected'
    except BaseException as exc:
        report.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        try:
            os.sched_setaffinity(0, affinity)
            report['affinity_restored'] = os.sched_getaffinity(0) == affinity
        except BaseException as exc:
            report.update(status='failed', cleanup_error=f'affinity restore: {exc}', affinity_restored=False)
        path = output/'check.json'
        path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'status': report['status'], 'check_sha256': C.sha256(path)}), flush=True)
    return 0 if report['status'] == 'passed_original_FP32_logit_gate' else 1


if __name__ == '__main__':
    raise SystemExit(main())
