#!/usr/bin/env python3
"""Check exact-image DINO reuse on three pinned, overlapping original CPU pairs.

All real image/model work is board-only. The existing cache is installed only
for this diagnostic and restored afterwards. Timing is not a controlled speed
comparison or online SLAM test; all saved-reference differences are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HELPERS = {
    'capture-v17-resolution-pairs-board.py': '3706712dd3a7451fcbce98e079d4018989eca6f64eb98704a81b8f4a9acc03cb',
    'profile-v14-small-vggt-board.py': 'ec69e48e2cb5a953d8e134934af7ebf1f45a88dfa0ec5b2cf5967bb4027e8d22',
}
CACHE_SHA = 'cd86bcb580cbf55ca92b6b98b203bb6e789bf35adfa47644103efd6be592e6f9'
OUTPUT_KEYS = ('pose_enc', 'depth', 'depth_conf')
RTOL, ATOL = 1e-4, 1e-5


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def helper(name):
    path = HERE / name
    if sha(path) != HELPERS[name]:
        raise ValueError('Frozen dependency changed: ' + name)
    spec = importlib.util.spec_from_file_location('_v19_' + name.replace('-', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_reference(reference):
    expected = dict(status='completed_fixed_input_diagnostic',
                    script_sha256=HELPERS['capture-v17-resolution-pairs-board.py'],
                    original_weights=True, precision='float32', npu_used=False,
                    ORB_used=False, inference_on_board_only=True)
    if any(reference.get(k) != v for k, v in expected.items()) or reference.get('image_size') != 336:
        raise ValueError('Expected pinned original CPU FP32 v17 336 reference')
    pairs, selected = reference.get('pairs', []), reference.get('selected_inputs', [])
    if len(pairs) != 3 or len(selected) != 3:
        raise ValueError('All three original reference pairs are required')
    for i, (row, inputs) in enumerate(zip(pairs, selected), 1):
        ids, images = row.get('frame_ids', []), inputs.get('images', [])
        if (row.get('revision') != i or inputs.get('revision') != i
                or len(ids) != 2 or not ids[0] < ids[1] or ids != inputs.get('frame_ids')
                or len(images) != 2 or any(x.get('frame_id') != fid or x.get('samefile_validated') is not True
                                          for x, fid in zip(images, ids))):
            raise ValueError('Pair source identity/order mismatch')
        if row.get('input_tensor', {}).get('shape') != [2, 3, 252, 336]:
            raise ValueError('Original 336 two-frame tensor required')
        if Path(row['file']).name != row['file']:
            raise ValueError('Reference NPZ must be a basename')
        if i > 1:
            prior = selected[i - 2]['images'][1]
            if any(prior.get(k) != images[0].get(k) for k in ('frame_id', 'path', 'sha256')):
                raise ValueError('Consecutive pairs must share the exact previous last frame')
    return pairs, selected


def compare_arrays(actual, expected, np):
    if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.dtype != np.float32:
        raise ValueError('Output shape/precision changed')
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('Nonfinite prediction/reference')
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    norm = float(np.linalg.norm(expected.astype(np.float64)))
    return dict(array_equal=bool(np.array_equal(actual, expected)),
                allclose=bool(np.allclose(actual, expected, rtol=RTOL, atol=ATOL)),
                max_abs=float(np.max(np.abs(delta))), rmse=float(np.sqrt(np.mean(delta * delta))),
                relative_l2=float(np.linalg.norm(delta) / norm) if norm else None,
                outside_tolerance_elements=int(np.count_nonzero(np.abs(delta) > ATOL + RTOL * np.abs(expected.astype(np.float64)))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--reference-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    capture, profile = helper('capture-v17-resolution-pairs-board.py'), helper('profile-v14-small-vggt-board.py')
    board = profile.require_board()
    args.reference, args.output = args.reference.resolve(), args.output.resolve()
    if (not args.reference.is_relative_to(capture.BASE)
            or not args.output.is_relative_to(capture.BASE / 'experiments')
            or args.output == capture.BASE / 'experiments'):
        parser.error('Reference and a new experiment directory must be inside the board project')
    if sha(args.reference) != args.reference_sha256:
        parser.error('Reference SHA mismatch')
    reference = json.loads(args.reference.read_text())
    pairs, selected = validate_reference(reference)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(kind='original_VGGT_exact_image_DINO_tail_cache_diagnostic', status='started',
                  script_sha256=sha(__file__), helper_sha256=HELPERS, cache_sha256=CACHE_SHA,
                  reference_sha256=args.reference_sha256, board=board, pairs=[],
                  original_weights=True, ORB_used=False, npu_used=False, GT_used=False,
                  inference_on_board_only=True, traditional_pose_frontend_used=False,
                  architecture_changed=False, full_model_forward_each_pair=True,
                  cache_scope='previous main batch final frame DINO x_norm_patchtokens only',
                  full_trajectory_accuracy_validated=False, real_time_validated=False,
                  tolerance=dict(rtol=RTOL, atol=ATOL, predefined=True), cleanup_errors=[],
                  timing_scope='model(images) including cache guards, hashing, assembly; excludes decode, load, comparison and save',
                  timing_limit='Saved reference and cached runs have different thermal/runtime states; not a controlled speedup experiment')
    started, handle, model = time.monotonic(), None, None
    allowed = os.sched_getaffinity(0)
    try:
        if not capture.CPUS <= allowed:
            raise ValueError('Four big CPU cores unavailable')
        os.sched_setaffinity(0, capture.CPUS)
        os.environ.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4', MPLBACKEND='Agg')
        sys.path[:0] = [str(capture.SOURCE), str(capture.SOURCE / 'third_party/vggt')]
        pins = {capture.SOURCE / k: v for k, v in reference['source_hashes'].items()}
        pins[capture.SOURCE / 'vggt_slam/patch_cache.py'] = CACHE_SHA
        checkpoint, calibration = Path(reference['checkpoint']['path']), Path(reference['calibration']['path'])
        pins[args.reference] = args.reference_sha256
        pins[calibration] = reference['calibration']['sha256']
        for row, inputs in zip(pairs, selected):
            pins[args.reference.parent / row['file']] = row['sha256']
            for entry in inputs['images']:
                pins[Path(entry['path'])] = entry['sha256']
        if any(sha(path) != expected for path, expected in pins.items()):
            raise ValueError('Source/calibration/reference/image hash mismatch')
        stat = checkpoint.stat()
        if (stat.st_size != capture.CHECKPOINT_SIZE or sha(checkpoint) != capture.CHECKPOINT_SHA
                or reference['checkpoint']['sha256'] != capture.CHECKPOINT_SHA):
            raise ValueError('Original full checkpoint identity mismatch')
        report.update(checkpoint=reference['checkpoint'], source_hashes=reference['source_hashes'],
                      preprocessing_provenance=reference['preprocessing_provenance'],
                      selected_inputs=selected, image_size=336)
        import cv2
        import numpy as np
        import torch
        from vggt_slam.image_rectification import CalibrationRectifier
        from vggt_slam.edge_mapping import prepare_rectified_images
        from vggt_slam.runtime import load_model
        from vggt_slam.patch_cache import install_patch_cache
        cv2.setNumThreads(1)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        rectifier = CalibrationRectifier(json.loads(calibration.read_text()))
        if rectifier.metadata() != reference['rectification']:
            raise ValueError('Rectification differs from saved reference')
        report['runtime'] = dict(torch=torch.__version__, numpy=np.__version__, cpu_affinity=sorted(os.sched_getaffinity(0)),
                                 threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads())
        model = load_model(checkpoint, device='cpu', image_size=336, prune_unused_heads=True, offline=True)
        if model.training or any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError('Original eval CPU FP32 model required')
        handle = install_patch_cache(model)
        for row, inputs in zip(pairs, selected):
            images = prepare_rectified_images([x['path'] for x in inputs['images']], rectifier, 336)
            if capture.array_info(np.ascontiguousarray(images.numpy())) != row['input_tensor']:
                raise ValueError('Prepared input is not identical to saved reference')
            before, platform_before = handle.metrics(), profile.platform_record()
            profile.event('cached_forward_begin', revision=row['revision'])
            with torch.inference_mode():
                begin = time.perf_counter()
                prediction = model(images)
                seconds = time.perf_counter() - begin
            after = handle.metrics()
            arrays = {k: np.ascontiguousarray(prediction[k].detach().numpy()) for k in OUTPUT_KEYS}
            with np.load(args.reference.parent / row['file'], allow_pickle=False) as saved:
                comparisons = {k: compare_arrays(v, saved[k], np) for k, v in arrays.items()}
            expected_hits = 0 if row['revision'] == 1 else 1
            if (after['hit_frames'] - before['hit_frames'] != expected_hits
                    or after['computed_frames'] - before['computed_frames'] != 2 - expected_hits
                    or after['original_dino_calls'] - before['original_dino_calls'] != 1):
                raise ValueError('Actual cache hit/computation accounting differs from exact overlap')
            out = args.output / row['file']
            np.savez(out, **arrays, frame_ids=np.asarray(row['frame_ids'], np.float64))
            report['pairs'].append(dict(revision=row['revision'], frame_ids=row['frame_ids'],
                file=out.name, sha256=sha(out), bytes=out.stat().st_size,
                input_tensor=row['input_tensor'], arrays={k:capture.array_info(v) for k,v in arrays.items()},
                model_seconds=seconds, saved_reference_model_seconds=row['model_inference_seconds'],
                comparisons=comparisons, cache_before=before, cache_after=after,
                platform_before=platform_before, platform_after=profile.platform_record()))
            profile.event('cached_forward_end', revision=row['revision'], seconds=seconds,
                          array_equal=all(v['array_equal'] for v in comparisons.values()))
            del arrays, prediction, images
        report['input_and_source_hashes_unchanged'] = all(sha(path) == expected for path, expected in pins.items())
        if not report['input_and_source_hashes_unchanged'] or (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            raise ValueError('Input/source/checkpoint changed during diagnostic')
        report['all_outputs_array_equal'] = all(v['array_equal'] for p in report['pairs'] for v in p['comparisons'].values())
        report['all_outputs_within_tolerance'] = all(v['allclose'] for p in report['pairs'] for v in p['comparisons'].values())
        report['status'] = 'completed_cache_diagnostic'
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        if handle is not None:
            report['cache_before_close'] = handle.metrics()
            try:
                handle.close()
            except BaseException as exc:
                report['cleanup_errors'].append(str(exc))
            report['cache_after_close'] = handle.metrics()
        os.sched_setaffinity(0, allowed)
        report['total_seconds'] = time.monotonic() - started
        if report['cleanup_errors']:
            report['status'] = 'failed'
        path = args.output / 'cache.json'
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], receipt=str(path), sha256=sha(path), error=report.get('error'))))
    return 0 if report['status'] == 'completed_cache_diagnostic' else 1


if __name__ == '__main__':
    sys.exit(main())
