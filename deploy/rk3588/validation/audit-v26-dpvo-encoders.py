#!/usr/bin/env python3
"""Audit saved DPVO encoder arrays and receipts; no image/model execution."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

CHECKPOINT_SHA = '30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12'
MANIFEST_SHA = 'ce2ecaf96b086f69ef4d90a2e07e7b81afe8ae094656eb41260182000a69e375'
COMMIT = '0ac95b656d1fda91c271d2a106460d19ad966fc7'
SHAPES = {'fmap': [1, 1, 128, 116, 152], 'imap': [1, 1, 384, 116, 152]}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def meta(a):
    a = np.ascontiguousarray(a)
    return {'shape': list(a.shape), 'dtype': str(a.dtype), 'bytes': a.nbytes,
            'tensor_sha256': hashlib.sha256(a.tobytes()).hexdigest()}


def errors(actual, reference):
    d = actual.astype(np.float64) - reference.astype(np.float64)
    ref = reference.astype(np.float64)
    return {'failed_elements': int(np.count_nonzero(np.abs(d) > 1e-5 + 1e-4 * np.abs(ref))),
            'max_abs': float(np.max(np.abs(d))),
            'relative_l2': float(np.linalg.norm(d.ravel()) / max(np.linalg.norm(ref.ravel()), 1e-30)),
            'bitwise_equal': actual.tobytes() == reference.tobytes()}


def audit(capture_dir, export_dir, ort_dir, input_manifest):
    checks = []
    result = {'schema': 'v26_saved_encoder_array_review', 'status': 'failed', 'checks': checks,
              'scope': {'saved_arrays_only': True, 'images_decoded': False, 'model_forward_run': False,
                        'board_tasks_run': False, 'NPU_validation': False,
                        'complete_VO_or_20Hz_validated': False,
                        'independent_undistortion_pixel_comparison': False}}

    def check(ok, name):
        checks.append({'name': name, 'passed': bool(ok)})
        if not ok:
            raise ValueError(name)

    def receipt(directory, phase):
        path = directory / (phase + '.json')
        d = json.loads(path.read_text())
        result.setdefault('receipts', {})[phase] = {'file': path.name, 'sha256': sha(path)}
        return d

    def artifact(directory, record):
        check(Path(record['file']).name == record['file'], 'regular artifact basename')
        path = directory / record['file']
        check(path.is_file() and not path.is_symlink() and path.stat().st_size == record['bytes']
              and sha(path) == record['sha256'], 'artifact size/file SHA: ' + record['file'])
        return path

    def arrays(directory, record):
        path = artifact(directory, record['artifact'])
        with np.load(path, allow_pickle=False) as z:
            check(set(z.files) == set(SHAPES), 'exact feature keys: ' + path.name)
            out = {k: z[k].copy() for k in SHAPES}
        for k, a in out.items():
            check(a.dtype == np.float32 and list(a.shape) == SHAPES[k] and np.isfinite(a).all()
                  and np.any(a) and meta(a) == record['arrays'][k], 'feature shape/finite/raw SHA: ' + path.name + ':' + k)
        return out

    try:
        c = receipt(capture_dir, 'capture')
        e = receipt(export_dir, 'export')
        o = receipt(ort_dir, 'ort')
        plan = json.loads(input_manifest.read_text())
        plan_sha = sha(input_manifest)
        check(len(plan['frames']) == 3 and len({r['frame_id'] for r in plan['frames']}) == 3,
              'three distinct planned source frames')
        check([float(r['frame_id']) for r in plan['frames']] == sorted(float(r['frame_id']) for r in plan['frames']),
              'source timestamp order')
        for phase, r, status in [('capture', c, 'completed_encoder_cpu_probe_not_VO'),
                                  ('export', e, 'exported_not_runtime_validated'),
                                  ('ort', o, 'completed_CPU_ORT_gate')]:
            check(r['status'] == status and r['mode'] == phase and r['checkpoint_sha256'] == CHECKPOINT_SHA
                  and r['input_manifest_sha256'] == plan_sha, phase + ' completed source/checkpoint binding')
            check(r['state_unchanged'] and r['sources_and_inputs_unchanged'] and r['affinity_restored']
                  and r['cleanup_errors'] == [] and r['source_hashes_after'] == r['source_pins'],
                  phase + ' state/source/cleanup')
            loader = r['loader']
            coverage = loader['load_state']
            check(loader['commit'] == COMMIT and loader['manifest_sha256'] == MANIFEST_SHA
                  and coverage['strict'] is True and coverage['inference_state_keys'] == 94
                  and coverage['inference_parameters'] == 3384324
                  and coverage['released_state_keys'] == 98 and coverage['released_parameters'] == 3532549
                  and coverage['all_loaded_parameter_bytes_equal'], phase + ' full released state coverage')
            check(r['cpu_affinity'] == [4, 5, 6, 7] and r['torch_threads'] == 4
                  and r['encoder_only'] and not r['complete_VO_executed'] and not r['realtime_validated'],
                  phase + ' measured CPU scope')
            check(r['script_sha256'] == c['script_sha256'] and r['state_before'] == c['state_before'],
                  phase + ' exact same helper/weights')
        result['source_hashes'] = {'script_sha256': c['script_sha256'], 'checkpoint_sha256': CHECKPOINT_SHA,
                                   'official_manifest_sha256': MANIFEST_SHA, 'input_manifest_sha256': plan_sha}
        check(e['capture_sha256'] == result['receipts']['capture']['sha256']
              and o['capture_sha256'] == e['capture_sha256']
              and o['export_sha256'] == result['receipts']['export']['sha256'], 'receipt hash chain')
        check(len(c['inputs']) == 3 and [r['frame_index'] for r in c['inputs']] == [0, 1, 2], 'all normalized inputs')
        for i, row in enumerate(c['inputs']):
            check(row['source'] == plan['frames'][i] and row['frame_id'] == plan['frames'][i]['frame_id'],
                  'exact source identity: frame ' + str(i))
            x = np.load(artifact(capture_dir, row['input']), allow_pickle=False)
            check(x.dtype == np.float32 and list(x.shape) == [1, 1, 3, 464, 608]
                  and np.isfinite(x).all() and x.min() >= -.5 and x.max() <= 1.5
                  and meta(x) == row['array'], 'normalized input shape/range/raw SHA: ' + str(i))
            pixels = np.rint((x[0, 0] + np.float32(.5)) * np.float32(127.5)).astype(np.uint8)
            normalized = np.float32(2) * (pixels.astype(np.float32) / np.float32(255)) - np.float32(.5)
            check(meta(pixels) == row['cropped_BGR_uint8'] and normalized.tobytes() == x[0, 0].tobytes(),
                  'one FP32 normalization and cropped BGR hash: ' + str(i))
            check(row['pixel_intrinsics'] == [517.3, 516.5, 302.6, 247.3], 'crop intrinsic: ' + str(i))
        check([r['call_index'] for r in c['calls']] == list(range(6))
              and [r['frame_index'] for r in c['calls']] == [0, 0, 0, 0, 1, 2], 'six ordered CPU encoder pairs')
        representatives = {}
        for row in c['calls']:
            i = row['frame_index']
            check(row['frame_id'] == plan['frames'][i]['frame_id'], 'CPU call source identity')
            values = arrays(capture_dir, row)
            if i in representatives:
                check(all(values[k].tobytes() == representatives[i][k].tobytes() for k in SHAPES),
                      'repeat exact output: call ' + str(row['call_index']))
            else:
                representatives[i] = values
            check(row['fnet_seconds'] > 0 and row['inet_seconds'] > 0
                  and abs(row['total_encoder_seconds'] - row['fnet_seconds'] - row['inet_seconds']) < 1e-12,
                  'separate CNN time sum')
        warm = sum(r['total_encoder_seconds'] for r in c['calls'][1:4]) / 3
        check(abs(warm - c['first_frame_warm_mean_seconds']) < 1e-12, 'warm summary recomputation')
        check(e['input_shape'] == [1, 3, 464, 608]
              and e['output_shapes'] == [[1, 128, 116, 152], [1, 384, 116, 152]], 'singleton-view export contract')
        graph = artifact(export_dir, e['onnx'])
        check(o['onnx'] == e['onnx'] and o['providers'] == ['CPUExecutionProvider']
              and o['graph_optimization'] == 'ORT_DISABLE_ALL', 'same graph CPU ORT only')
        check(len(e['wrapper_checks']) == 3 and all(v['gate_passed'] and v['rtol'] == 1e-4 and v['atol'] == 1e-5
              for row in e['wrapper_checks'] for v in row.values()), 'original to singleton wrapper fixed gate')
        check([r['frame_index'] for r in o['pairs']] == [0, 1, 2], 'all three CPU ORT outputs')
        result['ORT_comparisons'] = []
        for row in o['pairs']:
            values = arrays(ort_dir, row)
            metrics = {}
            for key in SHAPES:
                measured = errors(values[key], representatives[row['frame_index']][key])
                stored = row['checks'][key]
                check(stored['rtol'] == 1e-4 and stored['atol'] == 1e-5 and stored['gate_passed']
                      and measured['failed_elements'] == 0, 'independent ORT gate: ' + key)
                check(all(measured[k] == stored[k] for k in ('failed_elements', 'max_abs', 'bitwise_equal'))
                      and math.isclose(measured['relative_l2'], stored['relative_l2'], rel_tol=1e-12, abs_tol=1e-15),
                      'independent ORT metrics: ' + key)
                metrics[key] = measured
            check(row['forward_seconds'] > 0, 'ORT positive actual time')
            result['ORT_comparisons'].append({'frame_index': row['frame_index'], 'metrics': metrics,
                                               'forward_seconds': row['forward_seconds']})
        result.update(status='passed_saved_array_review', actual_CPU_encoder_pair_calls=6,
                      actual_individual_CNN_forwards=12, CPU_first_pair_seconds=c['calls'][0]['total_encoder_seconds'],
                      CPU_warm_pair_mean_seconds=warm, onnx_sha256=sha(graph))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        result['error'] = str(exc)
    result['checked'] = len(checks)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['capture-dir', 'export-dir', 'ort-dir', 'input-manifest', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    result = audit(args.capture_dir, args.export_dir, args.ort_dir, args.input_manifest)
    result['auditor_sha256'] = sha(__file__)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'status': result['status'], 'checked': result['checked'], 'error': result.get('error')}))
    return 0 if result['status'] == 'passed_saved_array_review' else 1


if __name__ == '__main__':
    raise SystemExit(main())
