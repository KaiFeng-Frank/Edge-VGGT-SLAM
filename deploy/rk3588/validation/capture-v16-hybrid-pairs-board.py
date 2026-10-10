#!/usr/bin/env python3
"""Board-only, fixed-input VGGT diagnostic: RKNN camera + XNNPACK FP16 Linear.

The three ordered image pairs and every prepared-input digest are pinned to the
completed original CPU FP32 112 capture. Remaining VGGT computation is CPU; this
is neither an all-NPU model nor an online tracking/map benchmark. No GT is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import sys
import time
import traceback

sys.dont_write_bytecode = True
BASE = Path('/mnt/vggt-data/vggt-slam2')
SOURCE = BASE / 'src'
BASELINE_SHA = '76f39338bf634363ed6fa4a889147b4d2383feb6a4cf0ec8f16098c1b3530cfc'
BASELINE_SCRIPT_SHA = '64adbcbe7f6e013aa738b8bed57d21ef8fde10c4b63bd7c647e9fa2137a79d7c'
CHECKPOINT_BYTES = 5026874952
CHECKPOINT_SHA = 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
CPUS = {4, 5, 6, 7}
OUTPUT_KEYS = ('depth', 'depth_conf', 'intrinsic', 'extrinsic', 'pose_enc')
ADDITIONAL_SOURCES = (
    'vggt_slam/rknn_camera.py', 'vggt_slam/xnnpack_linear.py',
    'deploy/rk3588/validation/run-v16-camera-step-board.py',
    'deploy/rk3588/validation/export-v16-camera-step-board.py',
    'deploy/rk3588/validation/run-v12-rknn-block-board.py',
    'deploy/rk3588/reference/rknn_api_v2.3.2.h',
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(path, expected):
    path = Path(path)
    if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected) or sha(path) != expected:
        raise ValueError(f'File SHA256 mismatch: {path.name}')
    return expected


def array_info(value):
    import numpy as np
    if (not isinstance(value, np.ndarray) or value.dtype != np.float32 or not value.size
            or not value.flags.c_contiguous or not np.isfinite(value).all()):
        raise ValueError('Nonempty contiguous finite FP32 arrays required')
    return dict(shape=list(value.shape), dtype='float32', finite=True, bytes=value.nbytes,
                tensor_sha256=hashlib.sha256(memoryview(value).cast('B')).hexdigest())


def compare_arrays(actual, expected):
    import numpy as np
    array_info(actual)
    array_info(expected)
    if actual.shape != expected.shape:
        raise ValueError('Hybrid output shape differs from the original CPU baseline')
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    diff = a - b
    norm = float(np.linalg.norm(b))
    return dict(array_equal=bool(np.array_equal(actual, expected)),
                max_abs_error=float(np.max(np.abs(diff))), rmse=float(np.sqrt(np.mean(diff * diff))),
                relative_l2=float(np.linalg.norm(diff) / norm) if norm else None,
                relative_l2_defined=norm > 0, tolerance_applied=False, accuracy_pass_inferred=False)


def load_baseline(directory):
    """Verify receipt, original file identities, and all saved CPU tensors."""
    import numpy as np
    directory = Path(directory)
    require_sha(directory / 'capture.json', BASELINE_SHA)
    record = json.loads((directory / 'capture.json').read_text())
    if (record.get('status') != 'completed_fixed_input_diagnostic'
            or record.get('script_sha256') != BASELINE_SCRIPT_SHA
            or record.get('device') != 'cpu' or record.get('precision') != 'float32'
            or record.get('original_weights') is not True or record.get('image_size') != 112
            or record.get('ORB_used') is not False or record.get('npu_used') is not False
            or record.get('graph_run') is not False or record.get('GT_used') is not False
            or record.get('online_pipeline_run') is not False
            or len(record.get('selected_inputs', [])) != 3 or len(record.get('pairs', [])) != 3
            or record.get('checkpoint', {}).get('sha256') != CHECKPOINT_SHA
            or record.get('checkpoint', {}).get('bytes') != CHECKPOINT_BYTES):
        raise ValueError('Only the pinned completed original CPU FP32 112 three-pair baseline is supported')
    references = []
    for index, (selected, measured) in enumerate(zip(record['selected_inputs'], record['pairs']), 1):
        ids = selected['frame_ids']
        if (selected['revision'] != index or measured['revision'] != index
                or measured['file'] != f'predictions_{index:04d}.npz'
                or measured['frame_ids'] != ids or len(ids) != 2
                or not np.isfinite(ids).all() or ids[1] <= ids[0]
                or len(selected['images']) != 2):
            raise ValueError('Exact pair revision, ordering, and frame identities required')
        for fid, image in zip(ids, selected['images']):
            if (image['frame_id'] != fid or image['timestamp'] != fid
                    or image['schedule_token']['frame_id'] != fid
                    or image['schedule_token']['timestamp'] != fid
                    or not image['samefile_validated']
                    or not Path(image['path']).samefile(Path(image['schedule_token']['path']))):
                raise ValueError('Original same-file source timestamp binding differs')
            require_sha(image['path'], image['sha256'])
        path = directory / measured['file']
        require_sha(path, measured['sha256'])
        if path.stat().st_size != measured['bytes']:
            raise ValueError('Original CPU prediction file size mismatch')
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(OUTPUT_KEYS) | {'frame_ids'}:
                raise ValueError('All five original CPU outputs and frame IDs required')
            if not np.array_equal(archive['frame_ids'], np.asarray(ids, dtype=np.float64)):
                raise ValueError('Saved CPU frame identities differ from their recorded images')
            arrays = {key: archive[key].copy() for key in OUTPUT_KEYS}
        for key, array in arrays.items():
            if array_info(array) != measured['arrays'][key] or array.shape[:2] != (1, 2):
                raise ValueError('Original CPU tensor precision/SHA/batch identity mismatch')
        if measured['input_tensor']['shape'] != [2, 3, 84, 112] or measured['input_tensor']['dtype'] != 'float32':
            raise ValueError('Pinned two-frame 112 prepared input contract differs')
        references.append(arrays)
    for key in ('calibration', 'rgb_index'):
        require_sha(record[key]['path'], record[key]['sha256'])
    return record, references


def camera_configuration(a):
    config = {}
    for name in ('model', 'export_receipt', 'compile_receipt', 'capture_receipt'):
        path, expected = getattr(a, 'camera_' + name), getattr(a, 'camera_' + name + '_sha256')
        require_sha(path, expected)
        config['model_path' if name == 'model' else name] = str(path)
        config[{'model': 'model_sha256', 'export_receipt': 'export_sha256',
                'compile_receipt': 'compile_sha256', 'capture_receipt': 'capture_sha256'}[name]] = expected
    return config


def close_handles(linear, camera, record):
    """Every owned handle gets a cleanup attempt; a first error cannot skip the second."""
    for label, handle in (('xnnpack', linear), ('camera', camera)):
        if handle is None:
            continue
        try:
            handle.close()
        except BaseException as exc:
            record['cleanup_errors'].append(f'{label} close: {type(exc).__name__}: {exc}')
        try:
            record[label + '_runtime_final'] = handle.metrics()
        except BaseException as exc:
            record['cleanup_errors'].append(f'{label} final metrics: {type(exc).__name__}: {exc}')


def capture(a):
    record = dict(kind='fixed_rectified_original_VGGT_112_hybrid_pairs', status='started',
                  script_sha256=sha(__file__), CPU_baseline_receipt_sha256=BASELINE_SHA,
                  image_size=112, ordered_pairs=3, original_weights=True, ORB_used=False,
                  graph_run=False, GT_used=False, online_pipeline_run=False,
                  inference_on_board_only=True, npu_used=False, whole_model_on_NPU=False,
                  camera_backend='shared_RKNN_FP16_step_with_CPU_FP32_recurrence',
                  remaining_linear_backend='CPU_XNNPACK_FP16', remaining_non_linear_device='cpu',
                  real_time_validated=False, accuracy_validated=False,
                  timing_scope='fixed input, one model/context load, one forward per pair, no unreported warmup',
                  checkpoint_hash_before_load_warms_filesystem_cache=True,
                  cleanup_errors=[], pairs=[])
    start, allowed, model, camera, linear, pins = time.monotonic(), None, None, None, None, {}
    try:
        require_board()
        allowed = set(os.sched_getaffinity(0))
        if not CPUS <= allowed:
            raise ValueError('CPUs 4,5,6,7 must be allowed to match the CPU baseline')
        os.sched_setaffinity(0, CPUS)
        if set(os.sched_getaffinity(0)) != CPUS:
            raise ValueError('CPU affinity was silently intersected')
        environment = dict(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4',
                           MPLBACKEND='Agg', OMP_WAIT_POLICY='PASSIVE', GOMP_SPINCOUNT='0')
        record['environment_before'] = {key: os.environ.get(key) for key in environment}
        os.environ.update(environment)
        sys.path[:0] = [str(SOURCE), str(SOURCE / 'third_party/vggt')]
        baseline, references = load_baseline(a.baseline_capture_dir)
        config = camera_configuration(a)
        record.update(camera_configuration=config, selected_inputs=baseline['selected_inputs'],
                      CPU_baseline_model=baseline['model'], CPU_baseline_pair_inference_seconds=[
                          p['model_inference_seconds'] for p in baseline['pairs']])
        pins[str(a.baseline_capture_dir / 'capture.json')] = BASELINE_SHA
        for item in baseline['pairs']:
            pins[str(a.baseline_capture_dir / item['file'])] = item['sha256']
        for selected in baseline['selected_inputs']:
            for image in selected['images']:
                pins[image['path']] = image['sha256']
        for key in ('calibration', 'rgb_index'):
            pins[baseline[key]['path']] = baseline[key]['sha256']
        for name in ('model', 'export_receipt', 'compile_receipt', 'capture_receipt'):
            pins[str(getattr(a, 'camera_' + name))] = getattr(a, 'camera_' + name + '_sha256')
        source_names = sorted(set(baseline['source_hashes']) | set(ADDITIONAL_SOURCES))
        source_hashes = {name: sha(SOURCE / name) for name in source_names}
        differences = [name for name, expected in baseline['source_hashes'].items() if source_hashes[name] != expected]
        if any(name != 'vggt_slam/edge_mapping.py' for name in differences):
            raise ValueError('Original model, rectifier, or input preprocessing source changed')
        record.update(source_hashes=source_hashes, CPU_baseline_source_hashes=baseline['source_hashes'],
                      declared_authorized_source_differences=differences,
                      edge_mapping_change_scope='NPU camera integration; each prepared input must still match exact CPU tensor SHA')
        pins.update({str(SOURCE / name): digest for name, digest in source_hashes.items()})
        pins[str(Path(__file__).resolve())] = sha(__file__)
        checkpoint = Path(baseline['checkpoint']['path'])
        checkpoint_stat = checkpoint.stat()
        if checkpoint_stat.st_size != CHECKPOINT_BYTES:
            raise ValueError('Official checkpoint size mismatch')
        require_sha(checkpoint, CHECKPOINT_SHA)
        record['checkpoint'] = baseline['checkpoint']
        import cv2
        import numpy as np
        import torch
        from vggt_slam.edge_mapping import prepare_rectified_images, platform_sample
        from vggt_slam.image_rectification import CalibrationRectifier
        from vggt_slam.runtime import load_model
        from vggt_slam.rknn_camera import install_rknn_camera
        from vggt_slam.xnnpack_linear import install_xnnpack_linear
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        cv2.setNumThreads(1)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        record['runtime'] = dict(torch=torch.__version__, numpy=np.__version__, opencv=cv2.__version__,
                                 cpu_affinity=sorted(os.sched_getaffinity(0)), threads=torch.get_num_threads(),
                                 interop_threads=torch.get_num_interop_threads(), environment=environment)
        rectifier = CalibrationRectifier(json.loads(Path(baseline['calibration']['path']).read_text()))
        if rectifier.metadata() != baseline['rectification']:
            raise ValueError('Shared rectification differs from the exact CPU baseline')
        record['rectification'] = rectifier.metadata()
        record['platform_before_load'] = platform_sample()
        tick = time.monotonic()
        model = load_model(checkpoint, device='cpu', image_size=112, prune_unused_heads=True, offline=True)
        record['model_load_seconds'] = time.monotonic() - tick
        if (model.training or not all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in model.parameters())
                or sum(p.numel() for p in model.parameters()) != baseline['model']['parameters']
                or sorted(model.aggregator.retained_layer_indices) != baseline['model']['retained_layer_indices']
                or model.depth_frames_chunk_size != baseline['model']['depth_frames_chunk_size']):
            raise ValueError('Original eval CPU FP32 model parameter contract differs')
        record['platform_after_load'] = platform_sample()
        tick = time.monotonic()
        camera = install_rknn_camera(model, **config)
        record['camera_install_seconds'] = time.monotonic() - tick
        tick = time.monotonic()
        linear = install_xnnpack_linear(model, threads=4, dtype='float16', scope='all', cpu_affinity=tuple(sorted(CPUS)))
        record['xnnpack_install_seconds'] = time.monotonic() - tick
        record['platform_after_backend_install'] = platform_sample()
        for index, (selected, cpu_pair, reference) in enumerate(zip(baseline['selected_inputs'], baseline['pairs'], references), 1):
            attempt = dict(revision=index, frame_ids=selected['frame_ids'], status='started')
            record['pairs'].append(attempt)
            for image in selected['images']:
                require_sha(image['path'], image['sha256'])
            tick = time.monotonic()
            images = prepare_rectified_images([item['path'] for item in selected['images']], rectifier, 112)
            attempt['tensor_preparation_seconds'] = time.monotonic() - tick
            if images.device.type != 'cpu' or images.dtype != torch.float32 or images.shape != (2, 3, 84, 112):
                raise ValueError('Prepared RGB tensor CPU/precision/shape contract differs')
            input_array = np.ascontiguousarray(images.numpy())
            attempt['input_tensor'] = array_info(input_array)
            if attempt['input_tensor'] != cpu_pair['input_tensor']:
                raise ValueError('Actual rectified RGB input tensor differs from the pinned CPU pair')
            del input_array
            attempt.update(camera_runtime_before=camera.metrics(), xnnpack_runtime_before=linear.metrics(),
                           platform_before_forward=platform_sample())
            tick = time.monotonic()
            with torch.no_grad():
                prediction = model(images)
            attempt['model_inference_seconds'] = time.monotonic() - tick
            attempt.update(camera_runtime_after=camera.metrics(), xnnpack_runtime_after=linear.metrics(),
                           platform_after_forward=platform_sample())
            for key in ('sdk_run_attempts', 'sdk_calls', 'sdk_outputs_copied'):
                if attempt['camera_runtime_after'][key] - attempt['camera_runtime_before'][key] != 4:
                    raise ValueError('Each actual two-frame head must execute/copy exactly four SDK calls')
            if not attempt['camera_runtime_after']['npu_used']:
                raise ValueError('Requested RKNN camera did not actually run; no CPU fallback is permitted')
            record['npu_used'] = True
            tick = time.monotonic()
            extrinsic, intrinsic = pose_encoding_to_extri_intri(prediction['pose_enc'], images.shape[-2:])
            tensors = dict(depth=prediction['depth'], depth_conf=prediction['depth_conf'],
                           intrinsic=intrinsic, extrinsic=extrinsic, pose_enc=prediction['pose_enc'])
            arrays = {}
            for key, tensor in tensors.items():
                if tensor.device.type != 'cpu' or tensor.dtype != torch.float32 or tensor.shape[:2] != (1, 2):
                    raise ValueError('Hybrid output CPU FP32/batch identity contract differs: ' + key)
                arrays[key] = np.ascontiguousarray(tensor.detach().numpy())
            attempt['arrays'] = {key: array_info(value) for key, value in arrays.items()}
            attempt['versus_original_CPU'] = {key: compare_arrays(value, reference[key]) for key, value in arrays.items()}
            attempt['conversion_validation_seconds'] = time.monotonic() - tick
            filename = cpu_pair['file']
            tick = time.monotonic()
            with (a.output / filename).open('xb') as stream:
                np.savez(stream, **arrays, frame_ids=np.asarray(selected['frame_ids'], np.float64))
            save_seconds = time.monotonic() - tick
            tick = time.monotonic()
            file_digest = sha(a.output / filename)
            attempt.update(file=filename, bytes=(a.output / filename).stat().st_size,
                           sha256=file_digest, save_seconds=save_seconds, saved_file_hash_seconds=time.monotonic() - tick,
                           first_forward=index == 1, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                           status='completed')
            del images, prediction, tensors, arrays, intrinsic, extrinsic
        if (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns) != (checkpoint_stat.st_size, checkpoint_stat.st_mtime_ns):
            raise ValueError('Official checkpoint changed during inference')
        record['checkpoint_stat_unchanged'] = True
        record['status'] = 'completed_fixed_input_hybrid_diagnostic'
    except BaseException as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        close_handles(linear, camera, record)
        if camera is not None and 'camera_runtime_final' in record:
            record['npu_used'] = bool(record['camera_runtime_final'].get('npu_used', False))
        if pins:
            try:
                record['source_and_input_hashes'] = pins
                record['source_and_input_hashes_after'] = {path: sha(path) for path in pins}
                record['source_and_inputs_unchanged'] = record['source_and_input_hashes_after'] == pins
                if not record['source_and_inputs_unchanged']:
                    record['cleanup_errors'].append('Source/input SHA changed during inference')
            except BaseException as exc:
                record['cleanup_errors'].append(f'Source recheck: {type(exc).__name__}: {exc}')
        if allowed is not None:
            try:
                os.sched_setaffinity(0, allowed)
                record['affinity_restored'] = set(os.sched_getaffinity(0)) == allowed
                if not record['affinity_restored']:
                    raise RuntimeError('Affinity restoration mismatch')
            except BaseException as exc:
                record['cleanup_errors'].append(f'Affinity cleanup: {type(exc).__name__}: {exc}')
        del model
        record.update(total_seconds=time.monotonic() - start, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        if record['cleanup_errors']:
            record['status'] = 'failed'
        (a.output / 'capture.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    return record


def require_board():
    if platform.system() != 'Linux' or platform.machine().lower() not in ('aarch64', 'arm64'):
        raise RuntimeError('Model execution is restricted to the Linux ARM board; no host inference')
    compatible = Path('/proc/device-tree/compatible')
    if not compatible.exists() or b'rk3588' not in compatible.read_bytes():
        raise RuntimeError('RK3588 device tree is required')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-capture-dir', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    for name in ('model', 'export-receipt', 'compile-receipt', 'capture-receipt'):
        p.add_argument('--camera-' + name, required=True, type=Path)
        p.add_argument('--camera-' + name + '-sha256', required=True)
    a = p.parse_args(argv)
    try:
        require_board()
    except RuntimeError as exc:
        p.error(str(exc))
    for key in ('baseline_capture_dir', 'output', 'camera_model', 'camera_export_receipt',
                'camera_compile_receipt', 'camera_capture_receipt'):
        value = getattr(a, key).resolve()
        if not value.is_relative_to(BASE):
            p.error(key + ' must be inside the board data root')
        setattr(a, key, value)
    if not a.output.is_relative_to(BASE / 'experiments') or os.path.lexists(a.output):
        p.error('Output must be a NEW directory under board experiments')
    a.output.mkdir(parents=True, exist_ok=False)
    result = capture(a)
    print(json.dumps(dict(status=result['status'], sha256=sha(a.output / 'capture.json'), error=result.get('error'))))
    return 0 if result['status'] == 'completed_fixed_input_hybrid_diagnostic' else 1


if __name__ == '__main__':
    sys.exit(main())
