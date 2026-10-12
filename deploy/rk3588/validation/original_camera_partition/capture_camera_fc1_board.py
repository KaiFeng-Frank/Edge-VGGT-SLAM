#!/usr/bin/env python3
"""Capture four real original trunk0 FC1/GELU calls from the frozen CameraHead."""
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
CAPTURE_SHA = '1c3234cfa65914a6f41634432153fa8ebb6dcf9c8d3eb3215f0955fc0ed000f1'
CHECKPOINT_SHA = 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
CPUS = {4, 5, 6, 7}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def pin(path, digest, size=None):
    path = Path(path).resolve(strict=True)
    if (size is not None and path.stat().st_size != size) or sha(path) != digest:
        raise ValueError('Frozen original artifact changed: ' + str(path))
    return path


def tensor_info(array, np):
    if array.dtype != np.float32 or not array.flags.c_contiguous or not np.isfinite(array).all():
        raise ValueError('Finite contiguous original FP32 tensor required')
    return dict(shape=list(array.shape), dtype='float32', finite=True, bytes=array.nbytes,
                sha256=hashlib.sha256(memoryview(array).cast('B')).hexdigest())


def pose_check(actual, reference, np):
    if actual.shape != (1, 2, 9) or actual.dtype != np.float32 or not np.isfinite(actual).all():
        raise ValueError('Invalid original pose output')
    delta = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    allowed = 1e-5 + 1e-4 * np.abs(reference.astype(np.float64))
    return dict(bitwise_equal=actual.tobytes() == reference.tobytes(),
                strict_pass=bool((delta <= allowed).all()), rtol=1e-4, atol=1e-5,
                outside=int((delta > allowed).sum()), elements=18, max_abs=float(delta.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=BASE / 'src')
    parser.add_argument('--capture-dir', type=Path, default=BASE / 'experiments/vggt-v15-camera/capture112-r1')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    compatible = next((p.read_bytes() for p in (Path('/proc/device-tree/compatible'),
        Path('/sys/firmware/devicetree/base/compatible')) if p.is_file()), b'')
    if (sys.platform != 'linux' or platform.machine().lower() not in ('aarch64', 'arm64')
            or b'rk3588' not in compatible):
        raise RuntimeError('Actual original CameraHead execution is RK3588/S board-only')
    repo, capture = args.repo.resolve(strict=True), args.capture_dir.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    affinity = os.sched_getaffinity(0)
    hooks, pins = [], {}
    report = dict(schema='v69_original_CameraHead_trunk0_FC1_GELU_capture', status='failed',
        script_sha256=sha(__file__), original_capture_sha256=CAPTURE_SHA,
        original_capture_directory=str(capture), actual_execution_on_board_only=True,
        original_model_only=True, aggregator_reexecuted=False, SDK_used=False,
        original_camera_parameters_verified=0, full_head_refinements=4, trunk_index=0,
        input_shape=[1, 2, 2048], output_shape=[1, 2, 8192], gelu_approximate='none',
        output_semantics='original_trunk0_FC1_bias_then_exact_GELU',
        candidate_accepted=False, cases=[], artifacts={}, pose_checks=[], cleanup_errors=[],
        board=dict(machine=platform.machine(), device_tree_compatible=compatible.replace(b'\0', b',').decode()))
    try:
        if not CPUS <= affinity:
            raise ValueError('Original CPU cores 4..7 unavailable')
        os.sched_setaffinity(0, CPUS)
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[name] = '4'
        import numpy as np
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        cap_path = pin(capture / 'capture.json', CAPTURE_SHA)
        pins[cap_path] = CAPTURE_SHA
        cap = json.loads(cap_path.read_text())
        if (cap['kind'] != 'original_complete_CameraHead_CPU_capture' or cap['status'] != 'completed'
                or cap['camera_iterations'] != 4 or cap['CPU_camera_forward_call_count'] != 3
                or cap['CPU_trunk_call_count'] != 48 or cap['npu_used'] is not False
                or cap['cleanup_errors'] or not cap['source_and_inputs_unchanged']
                or cap['checkpoint']['sha256'] != CHECKPOINT_SHA
                or cap['checkpoint']['bytes'] != 5026874952
                or torch.__version__ != cap['runtime']['torch']):
            raise ValueError('Completed unchanged original capture and exact board Torch required')
        report['checkpoint'] = dict(sha256=CHECKPOINT_SHA, bytes=5026874952)
        source_hashes = {k: v for k, v in cap['source_hashes'].items()
            if k.startswith('third_party/vggt/vggt/layers/') or k in (
                'third_party/vggt/vggt/heads/camera_head.py', 'third_party/vggt/vggt/heads/head_act.py')}
        for name, digest in source_hashes.items():
            pins[pin(repo / name, digest)] = digest
        report['source_hashes'] = source_hashes
        sys.path[:0] = [str(repo), str(repo / 'third_party/vggt')]
        from vggt.heads.camera_head import CameraHead
        report['runtime'] = dict(torch=torch.__version__, numpy=np.__version__, threads=4,
                                interop_threads=1, cpu_affinity=sorted(os.sched_getaffinity(0)))

        def load_file(name):
            item = cap['artifacts'][name]
            path = pin(capture / name, item['sha256'], item['bytes'])
            pins[path] = item['sha256']
            return path

        raw = np.load(load_file('input.npy'), allow_pickle=False)
        if (raw.shape != (1, 2, 2048) or tensor_info(raw, np)['sha256'] != cap['artifacts']['input.npy']['tensor_sha256']):
            raise ValueError('Original raw camera token identity differs')
        with np.load(load_file('pose-iterations.npz'), allow_pickle=False) as archive:
            references = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
        if set(references) != {f'iteration_{i}' for i in range(4)}:
            raise ValueError('All four original activated poses required')
        for name, value in references.items():
            if tensor_info(value, np) != cap['artifacts']['pose-iterations.npz']['tensors'][name]:
                raise ValueError('Original pose reference differs')
        with np.load(load_file('camera-state.npz'), allow_pickle=False) as archive:
            state = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
        metadata = cap['artifacts']['camera-state.npz']['tensors']
        if len(state) != 69 or set(state) != set(metadata):
            raise ValueError('All 69 original CameraHead parameters required')
        for name, value in state.items():
            if tensor_info(value, np) != metadata[name]:
                raise ValueError('Original parameter differs: ' + name)
        with torch.device('meta'):
            head = CameraHead()
        head.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()}, strict=True, assign=True)
        head.eval().requires_grad_(False)
        for name, value in head.named_parameters():
            if tensor_info(value.detach().numpy(), np) != metadata[name]:
                raise ValueError('Assigned original parameter differs: ' + name)
        if (len(head.trunk) != 4 or head.trunk[0].mlp.fc1.in_features != 2048
                or head.trunk[0].mlp.fc1.out_features != 8192
                or head.trunk[0].mlp.act.approximate != 'none'
                or head.token_norm.eps != 1e-5 or head.adaln_norm.eps != 1e-6
                or (head.trans_act, head.quat_act, head.fl_act) != ('linear', 'linear', 'relu')):
            raise ValueError('Original head architecture changed')
        report['original_camera_parameters_verified'] = 69
        report['original_camera_state'] = dict(file='camera-state.npz', **cap['artifacts']['camera-state.npz'])
        report['original_parameter_raw_sha256'] = {k: metadata[k]['sha256'] for k in (
            'trunk.0.mlp.fc1.weight', 'trunk.0.mlp.fc1.bias')}
        inputs, outputs = [], []
        hooks.append(head.trunk[0].mlp.fc1.register_forward_pre_hook(
            lambda module, args: inputs.append(args[0].detach().clone().contiguous())))
        hooks.append(head.trunk[0].mlp.act.register_forward_hook(
            lambda module, args, value: outputs.append(value.detach().clone().contiguous())))
        tick = time.perf_counter()
        with torch.inference_mode():
            poses = head([torch.from_numpy(raw).unsqueeze(2)], num_iterations=4)
        report['original_head_seconds_including_capture_hooks'] = time.perf_counter() - tick
        if len(inputs) != 4 or len(outputs) != 4 or len(poses) != 4:
            raise ValueError('Exactly four original FC1/GELU refinements required')
        report['pose_checks'] = [dict(iteration=i, **pose_check(pose.numpy(), references[f'iteration_{i}'], np))
                                 for i, pose in enumerate(poses)]
        report['all_four_original_poses_bitwise_equal'] = all(x['bitwise_equal'] for x in report['pose_checks'])
        report['all_four_original_poses_strict_pass'] = all(x['strict_pass'] for x in report['pose_checks'])
        report['all_original_poses_bitwise_equal'] = report['all_four_original_poses_bitwise_equal']
        if not report['all_original_poses_bitwise_equal']:
            raise ValueError('Full original CameraHead does not bitwise reproduce all four archived poses; strict checks retained')

        def save(name, value, shape):
            value = np.ascontiguousarray(value.numpy())
            info = tensor_info(value, np)
            if tuple(value.shape) != shape:
                raise ValueError('Original captured FC1/GELU shape differs')
            path = output / name
            with path.open('xb') as stream:
                np.save(stream, value, allow_pickle=False)
            return dict(file=name, bytes=path.stat().st_size, sha256=sha(path),
                        shape=info['shape'], dtype=info['dtype'], tensor_bytes=value.nbytes,
                        raw_sha256=info['sha256'])

        for i, (x, y) in enumerate(zip(inputs, outputs)):
            x_item = save(f'input-{i}.npy', x, (1, 2, 2048))
            y_item = save(f'output-{i}.npy', y, (1, 2, 8192))
            report['artifacts'][x_item['file']] = x_item
            report['artifacts'][y_item['file']] = y_item
            report['cases'].append(dict(iteration=i, input=x_item, output=y_item))
        report['status'] = 'completed'
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        report['hooks_removed'] = True
        for hook in hooks:
            try:
                hook.remove()
            except BaseException as exc:
                report['hooks_removed'] = False
                report['cleanup_errors'].append('Hook cleanup: ' + str(exc))
        try:
            report['source_and_inputs_unchanged'] = bool(pins) and all(sha(p) == h for p, h in pins.items())
        except BaseException as exc:
            report['source_and_inputs_unchanged'] = False
            report['cleanup_errors'].append('Identity cleanup: ' + str(exc))
        report['affinity_restored'] = False
        try:
            os.sched_setaffinity(0, affinity)
            report['affinity_restored'] = os.sched_getaffinity(0) == affinity
        except BaseException as exc:
            report['cleanup_errors'].append('Affinity cleanup: ' + str(exc))
        if report['cleanup_errors'] or not report['source_and_inputs_unchanged'] or not report['affinity_restored']:
            report['status'] = 'failed_cleanup_or_identity'
        temp = output / 'capture.json.tmp'
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        os.replace(temp, output / 'capture.json')
    print(json.dumps(dict(status=report['status'], error=report.get('error'),
                         receipt_sha256=sha(output / 'capture.json'))), flush=True)
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
