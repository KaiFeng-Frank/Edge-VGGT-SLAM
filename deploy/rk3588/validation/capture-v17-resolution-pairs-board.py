#!/usr/bin/env python3
"""V17 experimental 224/336 CPU FP32 VGGT diagnostic, on the ARM board only.

Run once per resolution; the same earliest recorded batches are selected without
using their quality. No GT, graph, frontend, XNNPACK or NPU is run. These timings
exclude online queues and delayed initialization and cannot establish realtime
performance. Hashing the checkpoint before loading warms the filesystem cache.
"""
from __future__ import annotations

import argparse
import ast
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
CHECKPOINT_SIZE = 5026874952
CHECKPOINT_SHA = 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
CPUS = {4, 5, 6, 7}
PREPROCESSING_SHA = '0353b56642d35e0f9f9752419287b0c111e3926357f5e09b6d5738f768f474df'
SOURCE_FILES = (
    'vggt_slam/runtime.py', 'vggt_slam/image_rectification.py',
    'vggt_slam/edge_mapping.py', 'vggt_slam/timestamp_replay.py',
    'third_party/vggt/vggt/models/vggt.py',
    'third_party/vggt/vggt/models/aggregator.py',
    'third_party/vggt/vggt/heads/camera_head.py',
    'third_party/vggt/vggt/heads/dpt_head.py',
    'third_party/vggt/vggt/layers/vision_transformer.py',
    'third_party/vggt/vggt/layers/block.py',
    'third_party/vggt/vggt/layers/attention.py',
    'third_party/vggt/vggt/utils/load_fn.py',
    'third_party/vggt/vggt/utils/pose_enc.py',
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def preprocessing_provenance(path):
    """Record the unchanged lossless bridge separately from mapper scheduling."""
    text = Path(path).read_text()
    matches = [node for node in ast.parse(text).body
               if isinstance(node, ast.FunctionDef) and node.name == 'prepare_rectified_images']
    if len(matches) != 1:
        raise ValueError('Expected exactly one original rectified preprocessing function')
    digest = hashlib.sha256(ast.get_source_segment(text, matches[0]).encode()).hexdigest()
    if digest != PREPROCESSING_SHA:
        raise ValueError('Original rectified preprocessing function changed')
    return dict(source='vggt_slam/edge_mapping.py', function='prepare_rectified_images',
                sha256=digest, hash_method='ast_get_source_segment_utf8_sha256',
                original_v16_function_unchanged=True)


def plan_pairs(schedule, rows, rectification, max_pairs):
    """Recover exact saved pairs without reading pixels or old predictions."""
    import numpy as np
    schedule = Path(schedule)
    metrics_path, events_path = schedule / 'metrics.json', schedule / 'events.jsonl'
    metrics = json.loads(metrics_path.read_text())
    if (metrics.get('status') != 'completed_experiment' or metrics.get('mapper_exitcode') != 0
            or metrics.get('cleanup_errors') != [] or metrics.get('ORB_used') is not False
            or metrics.get('model_backbone') != 'original_VGGT'):
        raise ValueError('Schedule must be a completed, clean original VGGT run without ORB')
    if metrics.get('common_rectification') != rectification:
        raise ValueError('Schedule and requested rectification metadata differ')
    if type(max_pairs) is not int or max_pairs < 1:
        raise ValueError('max_pairs must be a positive integer')
    batches = []
    for line in events_path.read_text().splitlines():
        event = json.loads(line)
        if event.get('kind') == 'map_submitted':
            batches.append(event['batch'])
    files = sorted(p for p in schedule.glob('predictions_*.npz')
                   if re.fullmatch(r'predictions_[0-9]{4,}\.npz', p.name))
    if not files or len(files) != len(batches) or metrics.get('map_updates') != len(files):
        raise ValueError('Saved prediction files do not match the complete submitted schedule')
    if [int(p.stem.split('_')[-1]) for p in files] != list(range(1, len(files) + 1)):
        raise ValueError('Saved prediction revisions must be consecutive from one')
    by_stamp = {row['timestamp']: row for row in rows}
    if len(by_stamp) != len(rows):
        raise ValueError('Original RGB index contains duplicate timestamps')
    pairs = []
    for revision, path in enumerate(files[:max_pairs], 1):
        with np.load(path, allow_pickle=False) as saved:
            if not {'frame_ids', 'depth', 'depth_conf', 'intrinsic', 'extrinsic'}.issubset(saved.files):
                raise ValueError('Schedule NPZ is missing required prediction/frame identity keys')
            ids = saved['frame_ids']
            if (ids.shape != (2,) or ids.dtype.kind not in 'fiu' or not np.isfinite(ids).all()
                    or float(ids[1]) <= float(ids[0])):
                raise ValueError('Schedule must contain two finite, increasing exact frame IDs')
            frame_ids = [float(v) for v in ids]
        batch = batches[revision - 1]
        if len(batch) != 2 or [item['frame_id'] for item in batch] != frame_ids:
            raise ValueError('NPZ frame IDs differ from the actual submitted batch')
        images = []
        for frame_id, token in zip(frame_ids, batch):
            if frame_id not in by_stamp or token['timestamp'] != frame_id:
                raise ValueError('Exact schedule timestamp is absent from the original RGB index')
            image = Path(by_stamp[frame_id]['path'])
            if not image.samefile(Path(token['path'])):
                raise ValueError('Recorded schedule path is not the same original RGB file')
            images.append(dict(path=str(image), frame_id=frame_id, timestamp=frame_id,
                               sha256=sha(image), schedule_token=token, samefile_validated=True))
        pairs.append(dict(revision=revision, schedule_file=str(path),
                          schedule_sha256=sha(path), frame_ids=frame_ids, images=images))
    return pairs, dict(metrics_sha256=sha(metrics_path), events_sha256=sha(events_path),
                      available_pairs=len(files), selected_pairs=len(pairs),
                      selection='earliest_saved_revision_order_without_quality_filtering',
                      original_backend=metrics['mapper_config']['backend'],
                      original_image_size=metrics['mapper_config']['image_size'])


def array_info(value):
    import numpy as np
    if (value.dtype != np.float32 or not value.size or not value.flags.c_contiguous
            or not np.isfinite(value).all()):
        raise ValueError('Original CPU prediction arrays must be nonempty contiguous finite FP32')
    return dict(shape=list(value.shape), dtype=str(value.dtype), finite=True, bytes=value.nbytes,
                tensor_sha256=hashlib.sha256(memoryview(value).cast('B')).hexdigest())


def capture(a):
    output = a.output
    receipt = dict(kind='fixed_rectified_original_VGGT_resolution_pairs', status='started',
                   script_sha256=sha(__file__), experiment_version='v17_resolution_probe',
                   image_size=a.image_size, max_pairs=a.max_pairs,
                   original_weights=True, device='cpu', precision='float32',
                   inference_on_board_only=True, ORB_used=False, npu_used=False,
                   graph_run=False, GT_used=False, online_pipeline_run=False,
                   real_time_validated=False, accuracy_validated=False,
                   timing_scope='fixed pairs, one model load; no warmup or online queue/initialization wait',
                   checkpoint_hash_before_load_warms_filesystem_cache=True,
                   preserve_original_prediction_batch_dimension=True, pairs=[])
    wall_started = time.monotonic()
    model = None
    try:
        allowed = set(os.sched_getaffinity(0))
        if not CPUS.issubset(allowed):
            raise ValueError(f'CPUs 4,5,6,7 must be allowed; actual allowed: {sorted(allowed)}')
        os.sched_setaffinity(0, CPUS)
        environment = dict(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4',
                           MPLBACKEND='Agg')
        receipt['environment_before'] = {key: os.environ.get(key) for key in environment}
        os.environ.update(environment)
        sys.path[:0] = [str(SOURCE), str(SOURCE / 'third_party/vggt')]
        receipt['source_hashes'] = {name: sha(SOURCE / name) for name in SOURCE_FILES}
        receipt['preprocessing_provenance'] = preprocessing_provenance(SOURCE / 'vggt_slam/edge_mapping.py')
        checkpoint_stat = a.checkpoint.stat()
        receipt['checkpoint'] = dict(path=str(a.checkpoint), bytes=checkpoint_stat.st_size)
        if checkpoint_stat.st_size != CHECKPOINT_SIZE:
            raise ValueError('Official checkpoint size mismatch')
        receipt['checkpoint']['sha256'] = sha(a.checkpoint)
        if receipt['checkpoint']['sha256'] != CHECKPOINT_SHA:
            raise ValueError('Official checkpoint SHA mismatch')
        receipt['rgb_index'] = dict(path=str(a.rgb_index), sha256=sha(a.rgb_index))
        receipt['calibration'] = dict(path=str(a.calibration), sha256=sha(a.calibration))
        imported = time.monotonic()
        import cv2
        import numpy as np
        import torch
        from vggt_slam.image_rectification import CalibrationRectifier
        from vggt_slam.timestamp_replay import read_rgb_index
        from vggt_slam.edge_mapping import prepare_rectified_images
        from vggt_slam.runtime import load_model
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        cv2.setNumThreads(1)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        receipt['import_seconds'] = time.monotonic() - imported
        receipt['runtime'] = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                                  opencv=cv2.__version__, cpu_affinity=sorted(os.sched_getaffinity(0)),
                                  torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                                  opencv_threads=cv2.getNumThreads(), environment={key: os.environ.get(key)
                                      for key in (*environment, 'OMP_WAIT_POLICY', 'GOMP_SPINCOUNT')})
        rectifier = CalibrationRectifier(json.loads(a.calibration.read_text()))
        receipt['rectification'] = rectifier.metadata()
        rows = read_rgb_index(a.rgb_index)
        pairs, schedule_info = plan_pairs(a.schedule_run_dir, rows, rectifier.metadata(), a.max_pairs)
        receipt['schedule'] = dict(path=str(a.schedule_run_dir), **schedule_info)
        receipt['selected_inputs'] = pairs
        loading = time.monotonic()
        model = load_model(a.checkpoint, device='cpu', image_size=a.image_size,
                           prune_unused_heads=True, offline=True)
        receipt['model_load_seconds'] = time.monotonic() - loading
        if not all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in model.parameters()):
            raise ValueError('Original model parameters must remain CPU FP32')
        receipt['model'] = dict(class_name=type(model).__name__, parameters=sum(p.numel() for p in model.parameters()),
                               eval=not model.training, prune_unused_heads=True,
                               retained_layer_indices=sorted(model.aggregator.retained_layer_indices),
                               depth_frames_chunk_size=model.depth_frames_chunk_size,
                               active_camera_depth_architecture_changed=False, linear_backend_installed=False)
        for pair in pairs:
            for image in pair['images']:
                if sha(image['path']) != image['sha256']:
                    raise ValueError('Original source image changed before preprocessing')
            prepared = time.monotonic()
            images = prepare_rectified_images([item['path'] for item in pair['images']], rectifier, a.image_size)
            prep_seconds = time.monotonic() - prepared
            if (images.device.type != 'cpu' or images.dtype != torch.float32 or images.shape[:2] != (2, 3)
                    or not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1):
                raise ValueError('Shared rectified preprocessing did not produce two CPU FP32 RGB images')
            started = time.monotonic()
            with torch.no_grad():
                prediction = model(images)
            inference_seconds = time.monotonic() - started
            converted = time.monotonic()
            extrinsic, intrinsic = pose_encoding_to_extri_intri(prediction['pose_enc'], images.shape[-2:])
            tensors = dict(depth=prediction['depth'], depth_conf=prediction['depth_conf'],
                           intrinsic=intrinsic, extrinsic=extrinsic, pose_enc=prediction['pose_enc'])
            arrays = {}
            for key, tensor in tensors.items():
                if tensor.device.type != 'cpu' or tensor.dtype != torch.float32 or tensor.shape[:2] != (1, 2):
                    raise ValueError(f'Unexpected original CPU tensor precision/batch identity: {key}')
                arrays[key] = np.ascontiguousarray(tensor.detach().numpy())
            infos = {key: array_info(value) for key, value in arrays.items()}
            input_array = np.ascontiguousarray(images.numpy())
            input_info = array_info(input_array)
            conversion_seconds = time.monotonic() - converted
            saved = time.monotonic()
            filename = Path(pair['schedule_file']).name
            np.savez(output / filename, **arrays, frame_ids=np.asarray(pair['frame_ids'], np.float64))
            save_seconds = time.monotonic() - saved
            receipt['pairs'].append(dict(revision=pair['revision'], frame_ids=pair['frame_ids'],
                file=filename, bytes=(output / filename).stat().st_size, sha256=sha(output / filename),
                input_tensor=input_info, arrays=infos, tensor_preparation_seconds=prep_seconds,
                model_inference_seconds=inference_seconds, conversion_validation_seconds=conversion_seconds,
                save_seconds=save_seconds, first_forward=not receipt['pairs'],
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
            for image in pair['images']:
                if sha(image['path']) != image['sha256']:
                    raise ValueError('Original source image changed during capture')
            del prediction, tensors, arrays, input_array, images, extrinsic, intrinsic
        if {name: sha(SOURCE / name) for name in SOURCE_FILES} != receipt['source_hashes']:
            raise ValueError('Inference source changed during capture')
        if sha(a.rgb_index) != receipt['rgb_index']['sha256'] or sha(a.calibration) != receipt['calibration']['sha256']:
            raise ValueError('RGB index or calibration changed during capture')
        if (a.checkpoint.stat().st_size, a.checkpoint.stat().st_mtime_ns) != (checkpoint_stat.st_size, checkpoint_stat.st_mtime_ns):
            raise ValueError('Official checkpoint changed during capture')
        receipt['status'] = 'completed_fixed_input_diagnostic'
    except BaseException as exc:
        receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        del model
        receipt['total_seconds'] = time.monotonic() - wall_started
        receipt['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        (output / 'capture.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('rgb-index', 'schedule-run-dir', 'checkpoint', 'calibration', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--image-size', type=int, choices=(224, 336), required=True)
    parser.add_argument('--max-pairs', type=int, default=3)
    a = parser.parse_args(argv)
    if platform.system() != 'Linux' or platform.machine().lower() not in ('aarch64', 'arm64'):
        parser.error('Model execution is restricted to the Linux ARM board; no host inference')
    if a.max_pairs < 1:
        parser.error('--max-pairs must be positive')
    for key in ('rgb_index', 'schedule_run_dir', 'checkpoint', 'calibration', 'output'):
        value = getattr(a, key).resolve()
        if not value.is_relative_to(BASE):
            parser.error(f'{key} must be inside the board data root {BASE}')
        setattr(a, key, value)
    if not a.output.is_relative_to(BASE / 'experiments'):
        parser.error('Output must be a new directory under board experiments')
    a.output.mkdir(parents=True, exist_ok=False)
    result = capture(a)
    print(json.dumps(dict(status=result['status'], receipt=str(a.output / 'capture.json'),
                          sha256=sha(a.output / 'capture.json'), error=result.get('error'))))
    return 0 if result['status'] == 'completed_fixed_input_diagnostic' else 1


if __name__ == '__main__':
    sys.exit(main())
