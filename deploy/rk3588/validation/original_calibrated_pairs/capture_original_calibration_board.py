#!/usr/bin/env python3
"""Eight real original FP32 full forwards; private activations, no SDK/cache."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np

PAIRS = ((0, 48), (96, 144), (192, 240), (288, 336),
         (384, 432), (480, 528), (576, 624), (672, 720))
EVALUATION_INDICES = (32, 80, 128, 176, 224, 272, 320, 368, 416,
                      464, 512, 560, 608, 656, 704, 752, 797)
CHECKPOINT = dict(bytes=5026874952,
    sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')

def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''): value.update(chunk)
    return value.hexdigest()

def file_info(path, root):
    return dict(file=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=sha(path))

def array_info(value, shape=None):
    value = np.ascontiguousarray(value)
    if value.dtype != np.float32 or not value.size or not np.isfinite(value).all():
        raise ValueError('Original nonempty finite FP32 activation/output required')
    if shape is not None and value.shape != shape: raise ValueError('Original tensor shape changed')
    return dict(shape=list(value.shape), dtype='float32', tensor_bytes=value.nbytes, finite=True,
        raw_sha256=hashlib.sha256(memoryview(value).cast('B')).hexdigest(),
        minimum=float(value.min()), maximum=float(value.max()))

def save(root, name, value, shape):
    array = np.ascontiguousarray(value); row = array_info(array, shape)
    path = root / name
    with path.open('xb') as stream: np.save(stream, array, allow_pickle=False)
    return dict(**file_info(path, root), **row)

def tensor_state(model):
    return tuple((name, id(t), t._version, t.dtype, t.device, tuple(t.shape), tuple(t.stride()),
                  t.data_ptr(), t.requires_grad) for name, t in
                 list(model.named_parameters()) + list(model.named_buffers()))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'checkpoint', 'rgb-index', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    board = Path('/proc/device-tree/compatible').read_bytes().replace(b'\0', b',').decode()
    if sys.platform != 'linux' or platform.machine().lower() not in ('aarch64', 'arm64') or 'rockchip,rk3588' not in board:
        p.error('Actual original-model capture requires the Linux RK3588S board')
    root = args.output_dir.resolve(); root.mkdir(parents=True, exist_ok=False)
    repo = args.repo.resolve(strict=True); checkpoint = args.checkpoint.resolve(strict=True)
    allowed = os.sched_getaffinity(0); old_path = sys.path[:]
    env_names = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
    old_env = {name: os.environ.get(name) for name in env_names}
    hooks = []; model = state = active = checkpoint_stat = None; source_paths = {}; image_pins = {}
    report = dict(schema='v56_original_VGGT_calibration_capture', status='failed', board=board,
        script_sha256=sha(__file__), checkpoint=CHECKPOINT, image_shape=[2, 3, 84, 112],
        source_hashes={}, samples=[], groups={'aggregator': [], 'dino': []}, cleanup_errors=[],
        evaluation_indices_excluded=list(EVALUATION_INDICES), full_forward_attempts=0,
        full_forward_completed=0, original_model_only=True, actual_inference_on_board_only=True,
        NPU_or_cache_installed=False, ORT_or_SDK_or_compiler_or_Solver_executed=False,
        dino_capture_mode='actual_B2_inputs_split_into_B1_rows; no_single_frame_forward',
        candidate_accepted=False, full_model_quality_or_realtime_validated=False)
    started = time.perf_counter()
    try:
        if not {4, 5, 6, 7} <= allowed: raise ValueError('CPU 4,5,6,7 required')
        os.sched_setaffinity(0, {4, 5, 6, 7}); os.environ.update({name: '4' for name in env_names})
        sys.path[:0] = [str(repo), str(repo / 'third_party/vggt')]
        originals = sorted((repo / 'third_party/vggt/vggt').rglob('*.py'))
        originals += [repo / 'vggt_slam' / name for name in ('runtime.py', 'checkpoint_identity.py', 'timestamp_replay.py')]
        source_paths = {str(path.relative_to(repo)): path for path in originals}
        report['source_hashes'] = {name: sha(path) for name, path in source_paths.items()}
        checkpoint_stat = checkpoint.stat()
        if checkpoint_stat.st_size != CHECKPOINT['bytes']: raise ValueError('Original checkpoint size differs')
        import torch
        from vggt_slam.runtime import load_model
        from vggt_slam.timestamp_replay import read_rgb_index
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        rows = read_rgb_index(args.rgb_index)
        report['rgb_index'] = dict(file=str(args.rgb_index.resolve()), sha256=sha(args.rgb_index), rows=len(rows))
        if len(rows) <= max(EVALUATION_INDICES): raise ValueError('TUM index has insufficient frames')
        if set(sum(PAIRS, ())) & set(EVALUATION_INDICES): raise ValueError('Calibration/evaluation index overlap')
        eval_hashes = {sha(rows[index]['path']) for index in EVALUATION_INDICES}
        load_started = time.perf_counter()
        model = load_model(checkpoint, device='cpu', image_size=112, prune_unused_heads=True, offline=True)
        report.update(model_load_seconds=time.perf_counter() - load_started, checkpoint_verified_before_deserialization=True)
        agg = model.aggregator; dino = agg.patch_embed
        if (len(agg.frame_blocks) != 24 or len(agg.global_blocks) != 24 or len(dino.blocks) != 24
                or dino.chunked_blocks or agg.aa_order != ['frame', 'global'] or agg.aa_block_size != 1
                or agg.patch_start_idx != 5 or dino.num_register_tokens != 4
                or model.camera_head is None or model.depth_head is None or model.depth_frames_chunk_size != 1
                or any(m.training for m in model.modules())
                or any(t.device.type != 'cpu' or t.dtype != torch.float32 for t in model.parameters())):
            raise ValueError('Complete unchanged original eval FP32 model required')
        for name, module in tuple(sys.modules.items()):
            if name.startswith(('vggt.', 'vggt_slam.runtime', 'vggt_slam.checkpoint_identity', 'vggt_slam.timestamp_replay')) and getattr(module, '__file__', None):
                path = Path(module.__file__).resolve()
                if not path.is_relative_to(repo): raise ValueError('Model/helper import escaped the selected source root')
        state = tensor_state(model)
        for role in ('aggregator', 'dino'):
            report['groups'][role] = [dict(index=i, samples=[], input_shape=[2 if role == 'aggregator' else 1, 53, 1024]) for i in range(24)]
            for i in range(24): (root / role / f'{i:02d}').mkdir(parents=True)
        def before(role, index):
            def hook(module, inputs, kwargs):
                x = inputs[0] if inputs else kwargs.get('x')
                if active is None or index != len(active['calls'][role]): raise ValueError('Original 24-layer call order changed')
                if not isinstance(x, torch.Tensor) or x.device.type != 'cpu' or x.dtype != torch.float32 or x.requires_grad:
                    raise ValueError('Original CPU FP32 block input required')
                array_info(x.detach().numpy(), (2, 53, 1024)); active['calls'][role].append(index)
                group = report['groups'][role][index]
                for frame in ((None,) if role == 'aggregator' else (0, 1)):
                    value = x if frame is None else x[frame:frame+1]
                    filename = f"{role}/{index:02d}/{active['sample_id']}" + ('' if frame is None else f'-frame{frame}') + '.npy'
                    item = save(root, filename, value.detach().numpy(), (2 if frame is None else 1, 53, 1024))
                    group['samples'].append(dict(sample_id=active['sample_id'], source_pair_indices=active['indices'],
                        origin_batch_shape=[2, 53, 1024], origin_batch_row=frame, artifact=item))
            return hook
        for i in range(24):
            hooks.append(agg.frame_blocks[i].register_forward_pre_hook(before('aggregator', i), with_kwargs=True))
            hooks.append(dino.blocks[i].register_forward_pre_hook(before('dino', i), with_kwargs=True))
        for n, indices in enumerate(PAIRS):
            active = dict(sample_id=f'calib{n:02d}', split='calibration', indices=list(indices),
                          calls={'aggregator': [], 'dino': []}, images=[])
            for index in indices:
                path = Path(rows[index]['path']); digest = sha(path)
                if digest in eval_hashes or digest in image_pins.values(): raise ValueError('Calibration image SHA overlaps evaluation/another calibration image')
                image_pins[str(path)] = digest
                active['images'].append(dict(index=index, timestamp=rows[index]['timestamp'], file=str(path), bytes=path.stat().st_size, sha256=digest))
            images = load_and_preprocess_images([item['file'] for item in active['images']], image_size=112, mode='crop')
            active['preprocessed_input'] = array_info(images.numpy(), (2, 3, 84, 112))
            report['full_forward_attempts'] += 1; begin = time.perf_counter()
            with torch.inference_mode(): raw = model(images, query_points=None, compute_similarity=False)
            report['full_forward_completed'] += 1
            active['full_forward_seconds_including_capture_IO'] = time.perf_counter() - begin
            if any(order != list(range(24)) for order in active['calls'].values()): raise ValueError('A complete original block sequence was not captured')
            extrinsic, intrinsic = pose_encoding_to_extri_intri(raw['pose_enc'], (84, 112))
            canonical = dict(pose_enc=raw['pose_enc'], extrinsic=extrinsic, intrinsic=intrinsic, depth=raw['depth'], depth_conf=raw['depth_conf'])
            active['canonical_outputs'] = {key: array_info(value.detach().numpy()) for key, value in canonical.items()}
            active['input_unchanged'] = array_info(images.numpy()) == active['preprocessed_input']
            if not active['input_unchanged']: raise ValueError('Original preprocessed input changed')
            report['samples'].append(active)
            print(json.dumps(dict(completed=n+1, total=8, sample_id=active['sample_id'])), flush=True)
            active = None; del raw, images, canonical, extrinsic, intrinsic
        for role, groups in report['groups'].items():
            for group in groups:
                if len(group['samples']) != (8 if role == 'aggregator' else 16): raise ValueError('Layer dataset row count differs')
                dataset = root / role / f"{group['index']:02d}" / 'dataset.txt'
                paths = [str(root / row['artifact']['file']) for row in group['samples']]
                if any(any(c.isspace() for c in path) for path in paths): raise ValueError('SDK dataset paths must contain no whitespace')
                dataset.write_text(''.join(path + '\n' for path in paths))
                group['dataset'] = dict(**file_info(dataset, root), line_count=len(paths), input_count=1)
        report['model_state_unchanged'] = tensor_state(model) == state
        if not report['model_state_unchanged']: raise ValueError('Original parameter/buffer identity changed')
        report['status'] = 'completed_original_FP32_calibration_capture'
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        report['hooks_removed'] = True
        for hook in hooks:
            try: hook.remove()
            except Exception as exc:
                report['hooks_removed'] = False; report['cleanup_errors'].append(f'hook: {exc}')
        try:
            report['source_and_images_unchanged'] = all(sha(path) == report['source_hashes'][name] for name, path in source_paths.items()) and all(sha(path) == digest for path, digest in image_pins.items())
            if source_paths and not report['source_and_images_unchanged']: raise ValueError('Original source/image bytes changed')
            if checkpoint_stat is not None:
                after = checkpoint.stat(); report['checkpoint_stat_unchanged'] = (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev) == (checkpoint_stat.st_size, checkpoint_stat.st_mtime_ns, checkpoint_stat.st_ino, checkpoint_stat.st_dev)
                if not report['checkpoint_stat_unchanged']: raise ValueError('Original checkpoint stat changed')
        except Exception as exc: report['cleanup_errors'].append(f'identity: {exc}')
        try:
            os.sched_setaffinity(0, allowed)
            report['affinity_restored'] = os.sched_getaffinity(0) == allowed
            if not report['affinity_restored']: raise ValueError('Affinity was not restored')
        except Exception as exc: report['cleanup_errors'].append(f'affinity: {exc}')
        for name, value in old_env.items():
            if value is None: os.environ.pop(name, None)
            else: os.environ[name] = value
        sys.path[:] = old_path
        report['environment_restored'] = all(os.environ.get(name) == value for name, value in old_env.items())
        report['seconds'] = time.perf_counter() - started
        if report['cleanup_errors']: report['status'] = 'failed'
        (root / 'capture.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], capture_sha256=sha(root / 'capture.json'))), flush=True)
    return 0 if report['status'] == 'completed_original_FP32_calibration_capture' else 1


if __name__ == '__main__':
    raise SystemExit(main())
