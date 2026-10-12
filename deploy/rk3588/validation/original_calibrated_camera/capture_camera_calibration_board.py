#!/usr/bin/env python3
"""Capture original CameraHead calibration by replaying only the saved FP32 tail.

No complete VGGT forward, NPU, cache, export or Solver. Capture timing includes
identity checks and saved arrays and is not an inference performance measurement.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

BASE = Path('/mnt/vggt-data/vggt-slam2')
CALIBRATION_SHA = 'a1e6f1b2a89e1a96987befdea4f0a048aaf06066efa9fff72a370ac77d45507c'
TAIL_SHA = '4b723e499763bb90565f6e11e16c551664634aaf72da46c7cdc493b84208ef67'
CAMERA_IDENTITY_SHA = '1c3234cfa65914a6f41634432153fa8ebb6dcf9c8d3eb3215f0955fc0ed000f1'
CHECKPOINT = dict(bytes=5026874952,
    sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')
PAIRS = ((0, 48), (96, 144), (192, 240), (288, 336),
         (384, 432), (480, 528), (576, 624), (672, 720))
RUNTIME_SOURCES = {
    'vggt_slam/runtime.py': 'f36bb5b87fbdbf03984ea3e1466e6ee27865a540ae3e5abd4ee07a45538e23f5',
    'vggt_slam/checkpoint_identity.py': '9fc0b927f6a68ac6ca55194616104febb1c488370593d4be60e8dba5bc336461'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def file_info(path, root):
    return dict(file=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=sha(path))


def array_info(value, shape=None):
    import numpy as np
    value = np.ascontiguousarray(value)
    if value.dtype != np.float32 or not value.size or not np.isfinite(value).all():
        raise ValueError('Original nonempty finite FP32 tensor required')
    if shape is not None and tuple(value.shape) != tuple(shape): raise ValueError('Original shape differs')
    return dict(shape=list(value.shape), dtype='float32', tensor_bytes=value.nbytes,
        finite=True, raw_sha256=hashlib.sha256(memoryview(value).cast('B')).hexdigest())


def save(root, name, value, shape):
    import numpy as np
    value = np.ascontiguousarray(value); info = array_info(value, shape)
    path = root / name
    with path.open('xb') as stream: np.save(stream, value, allow_pickle=False)
    return dict(**file_info(path, root), **info)


def tensor_state(model):
    return tuple((name, id(t), t._version, t.dtype, t.device, tuple(t.shape), tuple(t.stride()),
                  t.data_ptr(), t.requires_grad) for name, t in
                 list(model.named_parameters()) + list(model.named_buffers()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=BASE / 'src')
    p.add_argument('--checkpoint', type=Path, default=BASE / 'weights/model.pt')
    p.add_argument('--calibration-dir', type=Path, default=BASE / 'experiments/vggt-v56-calibration-r1')
    p.add_argument('--original-tail-dir', type=Path, default=BASE / 'experiments/vggt-v53-capture-r1')
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    board = next((x.read_bytes() for x in (Path('/proc/device-tree/compatible'),
        Path('/sys/firmware/devicetree/base/compatible')) if x.is_file()), b'')
    if sys.platform != 'linux' or platform.machine().lower() not in ('aarch64', 'arm64') or b'rk3588' not in board:
        p.error('Actual original-model capture is Linux RK3588/S board-only')
    repo = args.repo.resolve(strict=True); checkpoint = args.checkpoint.resolve(strict=True)
    calibration = args.calibration_dir.resolve(strict=True); tail_dir = args.original_tail_dir.resolve(strict=True)
    root = args.output_dir.resolve(); root.mkdir(parents=True, exist_ok=False)
    affinity = os.sched_getaffinity(0); old_path = sys.path[:]
    env_names = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
    old_env = {name: os.environ.get(name) for name in env_names}
    hooks, pins = [], {}; model = state = checkpoint_stat = None
    report = dict(schema='v82_original_VGGT_camera_PTQ_calibration_capture', status='failed',
        script_sha256=sha(__file__), checkpoint=CHECKPOINT, source_hashes={}, runtime_source_hashes=RUNTIME_SOURCES,
        original_model_only=True, actual_inference_on_board_only=True, NPU_or_cache_installed=False,
        full_VGGT_forward_executed=False, DINO_or_depth_executed=False,
        ORT_or_SDK_or_compiler_or_Solver_executed=False, samples=[], parameters={}, cleanup_errors=[],
        full_head_forward_count=0, tail_blocks_forward_count={'frame23': 0, 'global23': 0},
        conditioning_count=0, delta_count=0, all_final_poses_bitwise_equal_to_original_capture=False,
        candidate_accepted=False, full_model_quality_or_realtime_validated=False,
        capture_time_is_inference_performance=False)
    started = time.perf_counter()

    def pin(path, digest, size=None):
        path = Path(path).resolve(strict=True)
        if (size is not None and path.stat().st_size != size) or sha(path) != digest:
            raise ValueError('Frozen artifact/source changed: ' + str(path))
        pins[path] = digest
        return path

    def load_array(directory, item):
        import numpy as np
        path = (directory / item['file']).resolve(strict=True)
        if not path.is_relative_to(directory): raise ValueError('Array path escaped capture directory')
        value = np.load(pin(path, item['sha256'], item['bytes']), allow_pickle=False)
        raw = hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast('B')).hexdigest()
        if (list(value.shape) != item['shape'] or str(value.dtype) != item['dtype']
                or value.nbytes != item['tensor_bytes'] or raw != item['raw_sha256']
                or not value.size or not np.isfinite(value).all()):
            raise ValueError('Saved original activation identity differs')
        return np.ascontiguousarray(value)

    try:
        if not {4, 5, 6, 7} <= affinity: raise ValueError('Original CPU cores 4..7 unavailable')
        os.sched_setaffinity(0, {4, 5, 6, 7}); os.environ.update({name: '4' for name in env_names})
        calib_path = pin(calibration / 'capture.json', CALIBRATION_SHA)
        tail_path = pin(tail_dir / 'capture.json', TAIL_SHA)
        camera_path = pin(repo / 'deploy/rk3588/validation/vggt-v15-camera-capture112.json', CAMERA_IDENTITY_SHA)
        cap = json.loads(calib_path.read_text()); tail = json.loads(tail_path.read_text()); cam = json.loads(camera_path.read_text())
        if (cap['status'] != 'completed_original_FP32_calibration_capture' or cap['checkpoint'] != CHECKPOINT
                or cap['cleanup_errors'] or cap['full_forward_completed'] != 8 or cap['NPU_or_cache_installed']
                or not cap['model_state_unchanged'] or not cap['source_and_images_unchanged']
                or tail['status'] != 'completed_original_FP32_full_forward_and_48_block_capture'
                or tail['cleanup_errors'] or tail['checkpoint']['sha256'] != CHECKPOINT['sha256']
                or cam['status'] != 'completed' or cam['cleanup_errors'] or cam['camera_iterations'] != 4
                or cam['checkpoint']['sha256'] != CHECKPOINT['sha256']):
            raise ValueError('Completed pinned original captures required')
        if [tuple(s['indices']) for s in cap['samples']] != list(PAIRS): raise ValueError('Eight original calibration pairs differ')
        if set(sum(PAIRS, ())) & set(cap['evaluation_indices_excluded']): raise ValueError('Calibration/evaluation overlap')
        report.update(calibration_capture=file_info(calib_path, calibration),
            original_tail_capture=file_info(tail_path, tail_dir), original_camera_identity=file_info(camera_path, repo),
            evaluation_indices_excluded=cap['evaluation_indices_excluded'], image_shape=[2, 3, 84, 112],
            architecture=cam['architecture'], conditioning_semantics='first_original_empty; later_previous_unactivated_FP32_prediction')
        source_hashes = {name: digest for name, digest in cap['source_hashes'].items()
                         if name.startswith('third_party/vggt/vggt/')}
        for capture in (tail, cam):
            for name, digest in capture['source_hashes'].items():
                if name.startswith('third_party/vggt/vggt/') and source_hashes.get(name) != digest:
                    raise ValueError('Original source lineage differs: ' + name)
        for name, digest in dict(source_hashes, **RUNTIME_SOURCES).items(): pin(repo / name, digest)
        report['source_hashes'] = source_hashes
        checkpoint_stat = checkpoint.stat()
        if checkpoint_stat.st_size != CHECKPOINT['bytes']: raise ValueError('Original checkpoint size differs')
        sys.path[:0] = [str(repo), str(repo / 'third_party/vggt')]
        import numpy as np
        import torch
        from vggt_slam.runtime import load_model
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        if torch.__version__ != cam['runtime']['torch']: raise ValueError('Original board Torch version differs')
        report['runtime'] = dict(torch=torch.__version__, numpy=np.__version__, threads=4, interop_threads=1,
                                cpu_affinity=sorted(os.sched_getaffinity(0)))
        tick = time.perf_counter()
        model = load_model(checkpoint, device='cpu', image_size=112, prune_unused_heads=True, offline=True)
        report.update(model_load_seconds=time.perf_counter() - tick, checkpoint_verified_before_deserialization=True)
        agg = model.aggregator; head = model.camera_head
        if (len(agg.frame_blocks) != 24 or len(agg.global_blocks) != 24 or agg.aa_block_size != 1
                or agg.aa_order != ['frame', 'global'] or agg.patch_start_idx != 5 or head is None
                or len(head.trunk) != 4 or (head.trans_act, head.quat_act, head.fl_act) != ('linear', 'linear', 'relu')
                or any(m.training for m in model.modules())
                or any(t.device.type != 'cpu' or t.dtype != torch.float32 for t in model.parameters())):
            raise ValueError('Unchanged original eval FP32 model required')
        state = tensor_state(model)
        metadata = cam['artifacts']['camera-state.npz']['tensors']
        if len(metadata) != 69 or set(dict(head.named_parameters())) != set(metadata): raise ValueError('All 69 camera parameters required')
        for name, tensor in head.named_parameters():
            info = array_info(tensor.detach().numpy(), metadata[name]['shape'])
            if info['raw_sha256'] != metadata[name]['sha256']: raise ValueError('Original camera parameter differs: ' + name)
            report['parameters'][name] = info
        report['original_camera_parameters_verified'] = 69
        tail_rows = {}
        for role, block in (('frame', agg.frame_blocks[23]), ('global', agg.global_blocks[23])):
            row = next(r for r in tail['blocks'] if r['role'] == role and r['index'] == 23)
            if set(dict(block.named_parameters())) != set(row['parameters']): raise ValueError('Original tail parameter set differs')
            for name, tensor in block.named_parameters():
                if array_info(tensor.detach().numpy(), row['parameters'][name]['shape'])['raw_sha256'] != row['parameters'][name]['raw_sha256']:
                    raise ValueError('Original tail parameter differs: ' + role + '.' + name)
            tail_rows[role] = row
        report['original_tail_parameters_verified'] = 36
        frame_pos = load_array(tail_dir, tail_rows['frame']['positions'])
        global_pos = load_array(tail_dir, tail_rows['global']['positions'])
        if frame_pos.shape != (2, 53, 2) or global_pos.shape != (1, 106, 2) or frame_pos.dtype != np.int64 or not np.array_equal(frame_pos.reshape(1, 106, 2), global_pos):
            raise ValueError('Original frame/global positions differ')
        report['positions'] = {role: tail_rows[role]['positions'] for role in ('frame', 'global')}
        conditioning, deltas = [], []
        hooks.append(head.embed_pose.register_forward_pre_hook(lambda module, inputs: conditioning.append(np.ascontiguousarray(inputs[0].detach().numpy()).copy())))
        hooks.append(head.pose_branch.register_forward_hook(lambda module, inputs, value: deltas.append(np.ascontiguousarray(value.detach().numpy()).copy())))
        group = next(g for g in cap['groups']['aggregator'] if g['index'] == 23)
        if len(group['samples']) != 8: raise ValueError('Eight original frame23 inputs required')
        datasets = []; image_hashes = set()
        for n, original in enumerate(cap['samples']):
            item = group['samples'][n]
            if item['sample_id'] != original['sample_id'] or item['source_pair_indices'] != original['indices']:
                raise ValueError('Original tail input sample order differs')
            for image in original['images']:
                pin(image['file'], image['sha256'], image['bytes'])
                if image['sha256'] in image_hashes: raise ValueError('Calibration image content duplicated')
                image_hashes.add(image['sha256'])
            x = load_array(calibration, item['artifact']); array_info(x, (2, 53, 1024))
            sample_dir = root / original['sample_id']; sample_dir.mkdir()
            row = dict(sample_id=original['sample_id'], split='calibration', source_pair_indices=original['indices'],
                images=original['images'], original_frame23_input=item['artifact'], iterations=[],
                original_full_forward_pose_reference=original['canonical_outputs']['pose_enc'])
            conditioning.clear(); deltas.clear(); tick = time.perf_counter()
            with torch.inference_mode():
                frame, fi, intermediates = agg._process_frame_attention(torch.from_numpy(x), 1, 2, 53, 1024, 23, pos=torch.from_numpy(frame_pos))
                report['tail_blocks_forward_count']['frame23'] += 1
                glob, gi, global_intermediates = agg._process_global_attention(frame, 1, 2, 53, 1024, 23, pos=torch.from_numpy(global_pos))
                report['tail_blocks_forward_count']['global23'] += 1
                if fi != 24 or gi != 24 or len(intermediates) != 1 or len(global_intermediates) != 1: raise ValueError('Original tail call order differs')
                tokens = torch.cat([intermediates[0], global_intermediates[0]], dim=-1)
                poses = head([tokens], num_iterations=4)
                report['full_head_forward_count'] += 1
            row['tail_and_head_seconds_including_hooks'] = time.perf_counter() - tick
            if len(conditioning) != 4 or len(deltas) != 4 or len(poses) != 4: raise ValueError('Four actual original refinement inputs/outputs required')
            row['camera_tokens'] = save(root, f"{original['sample_id']}/camera_tokens.npy", tokens[:, :, 0].numpy(), (1, 2, 2048))
            cumulative = None
            for i in range(4):
                expected = head.empty_pose_tokens.expand(1, 2, -1).detach().numpy() if i == 0 else cumulative
                if conditioning[i].tobytes() != np.ascontiguousarray(expected).tobytes(): raise ValueError('Actual unactivated conditioning differs')
                cumulative = deltas[i] if i == 0 else cumulative + deltas[i]
                prefix = f"{original['sample_id']}/step{i}"
                iteration = dict(index=i, conditioning_pose=save(root, prefix + '-conditioning.npy', conditioning[i], (1, 2, 9)),
                    delta=save(root, prefix + '-delta.npy', deltas[i], (1, 2, 9)),
                    activated_pose=save(root, prefix + '-pose.npy', poses[i].numpy(), (1, 2, 9)))
                row['iterations'].append(iteration)
                datasets.append(str(root / row['camera_tokens']['file']) + ' ' + str(root / iteration['conditioning_pose']['file']))
            final = array_info(poses[-1].numpy(), (1, 2, 9))
            row['final_pose_bitwise_equal_to_original_capture'] = final['raw_sha256'] == original['canonical_outputs']['pose_enc']['raw_sha256']
            report['samples'].append(row); report['conditioning_count'] += 4; report['delta_count'] += 4
            if not row['final_pose_bitwise_equal_to_original_capture']: raise ValueError('Original tail+Camera final pose does not reproduce saved full original FP32 forward')
            if array_info(x)['raw_sha256'] != item['artifact']['raw_sha256']: raise ValueError('Original tail input mutated')
            print(json.dumps(dict(completed=n + 1, total=8, sample_id=original['sample_id'])), flush=True)
        if len(datasets) != 32 or any(any(c.isspace() for c in str(root / s['camera_tokens']['file'])) for s in report['samples']):
            raise ValueError('Exactly 32 two-input dataset rows without whitespace paths required')
        dataset = root / 'dataset.txt'; dataset.write_text(''.join(row + '\n' for row in datasets))
        report['dataset'] = dict(**file_info(dataset, root), line_count=32, input_count=2)
        report['all_final_poses_bitwise_equal_to_original_capture'] = True
        report['model_state_unchanged'] = tensor_state(model) == state
        if not report['model_state_unchanged']: raise ValueError('Original model parameter/buffer identity changed')
        report['status'] = 'completed_original_FP32_camera_calibration_capture'
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        report['hooks_removed'] = True
        for hook in hooks:
            try: hook.remove()
            except BaseException as exc:
                report['hooks_removed'] = False; report['cleanup_errors'].append('hook: ' + str(exc))
        try:
            report['source_and_inputs_unchanged'] = bool(pins) and all(sha(path) == digest for path, digest in pins.items())
            if not report['source_and_inputs_unchanged']: raise ValueError('Pinned source/input changed')
            if checkpoint_stat is not None:
                after = checkpoint.stat(); report['checkpoint_stat_unchanged'] = (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev) == (checkpoint_stat.st_size, checkpoint_stat.st_mtime_ns, checkpoint_stat.st_ino, checkpoint_stat.st_dev)
                if not report['checkpoint_stat_unchanged']: raise ValueError('Original checkpoint stat changed')
            if model is not None and state is not None:
                report['model_state_unchanged'] = tensor_state(model) == state
                if not report['model_state_unchanged']: raise ValueError('Original model identity changed')
        except BaseException as exc: report['cleanup_errors'].append('identity: ' + str(exc))
        try:
            os.sched_setaffinity(0, affinity); report['affinity_restored'] = os.sched_getaffinity(0) == affinity
            if not report['affinity_restored']: raise ValueError('Affinity not restored')
        except BaseException as exc: report['cleanup_errors'].append('affinity: ' + str(exc))
        for name, value in old_env.items():
            if value is None: os.environ.pop(name, None)
            else: os.environ[name] = value
        sys.path[:] = old_path
        report['environment_restored'] = all(os.environ.get(name) == value for name, value in old_env.items())
        report['seconds_including_hash_load_and_capture_IO'] = time.perf_counter() - started
        if report['cleanup_errors']: report['status'] = 'failed'
        temporary = root / 'capture.json.tmp'; temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        os.replace(temporary, root / 'capture.json')
    print(json.dumps(dict(status=report['status'], error=report.get('error'), capture_sha256=sha(root / 'capture.json'))), flush=True)
    return 0 if report['status'] == 'completed_original_FP32_camera_calibration_capture' else 1


if __name__ == '__main__':
    raise SystemExit(main())
