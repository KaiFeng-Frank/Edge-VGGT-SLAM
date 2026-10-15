#!/usr/bin/env python3
"""Two original CPU CameraHead calls on saved native518 camera tokens.

First reproduce all four CPU-reference refinements exactly. Only after that
control passes, evaluate the same original FP32 head on actual saved NPU
aggregator tokens. This component diagnosis is not a deployed mixed executor,
full-model inference, trajectory evaluation, or performance acceptance.
"""
from __future__ import annotations

import argparse
import faulthandler
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time
import traceback

BASE = Path('/mnt/vggt-data/vggt-slam2')
CHECKPOINT = {'bytes': 5026874952,
              'sha256': 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'}
PINS = {
    'camera': '1c3234cfa65914a6f41634432153fa8ebb6dcf9c8d3eb3215f0955fc0ed000f1',
    'cpu': '969ab6f5611efa62d81b5d62625a15815248668e1e079db17ac4d9ed9c96932b',
    'npu': '4af7b9c4cbd38dbfd54c14b1308dd6035da25941f98f858a5ad13b781c3c8af6',
    'state': '8f4fd6accc1ca0899d86ecb2583659cdfdf3c8807888fdb1c0f827b552a5647c',
}
MATH_SOURCES = {
    'third_party/vggt/vggt/heads/camera_head.py':
        '78b870018b20821d27587122003e68afd4f6b977bc13e0952fe507c31d6d3da2',
    'third_party/vggt/vggt/heads/head_act.py':
        '428bdb087f457b809a66e22d7d01147c1cb4dab43bb0d98bc61c7f05d8383a1e',
    'third_party/vggt/vggt/utils/pose_enc.py':
        '845fc3a46cb851bebf0720b8fc2b9f34334ae8e11f1982376ba59f90ff18f22b',
    'third_party/vggt/vggt/utils/rotation.py':
        'ad3a904e243c6654f0aa5f1d0206ace5f2d61011bc1bf8f492b40e0b76cb2070',
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for piece in iter(lambda: stream.read(4 << 20), b''):
            digest.update(piece)
    return digest.hexdigest()


def identity(path):
    info = Path(path).stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=BASE / 'src')
    parser.add_argument('--camera-capture', type=Path,
                        default=BASE / 'experiments/vggt-v15-camera/capture112-r1/capture.json')
    parser.add_argument('--camera-state', type=Path,
                        default=BASE / 'experiments/vggt-v15-camera/capture112-r1/camera-state.npz')
    parser.add_argument('--cpu-reference', type=Path,
                        default=BASE / 'experiments/vggt-v99-head-reference-r1/capture.json')
    parser.add_argument('--npu-reference', type=Path,
                        default=Path('/dev/shm/edge-vggt-v100-native/full-model-sequential-core1-r1/result.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    require(platform.machine() == 'aarch64' and
            b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),
            'Original CameraHead execution is restricted to RK3588/S')
    faulthandler.enable(all_threads=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    usage_start = resource.getrusage(resource.RUSAGE_SELF)
    affinity, old_env, old_path = os.sched_getaffinity(0), dict(os.environ), list(sys.path)
    old_term = signal.getsignal(signal.SIGTERM)
    pinned, hooks, head, state = {}, [], None, None
    report = {
        'schema': 'v113_original_native518_CameraHead_precision_diagnosis',
        'status': 'running', 'completed': False, 'script_sha256': sha(__file__),
        'checkpoint': CHECKPOINT, 'original_model_only': True,
        'actual_execution_on_board_only': True, 'checkpoint_deserialization_performed': False,
        'SDK_ORT_NPU_GPU_executed': False, 'full_model_forward_completed': False,
        'method_override_or_output_substitution': False,
        'input_cut': 'Original CameraHead reads only last retained tensor[:, :, 0]; supplied raw camera tokens',
        'image_size': 518, 'image_size_hw': [392, 518], 'source_pair_indices': [32, 40],
        'camera_parameter_count': 69, 'learned_empty_seed_required': True,
        'counts': {'camera_calls': 0, 'camera_refinements': 0, 'pose_codec_calls': 0},
        'intra_op_threads': 1, 'inter_op_threads': 1, 'cases': [], 'artifacts': {},
        'timing_scope': 'Observed CPU component calls only; wall includes controller cooling pauses if any; excludes file IO and model loading',
        'trajectory_precision_realtime_or_8GB_accepted': False,
        'cleanup_errors': [], 'saved_array_bytes': 0,
    }

    def progress(stage):
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report.update(stage=stage, wall_seconds=time.perf_counter() - started,
                      process_user_seconds=usage.ru_utime - usage_start.ru_utime,
                      process_system_seconds=usage.ru_stime - usage_start.ru_stime,
                      peak_rss_mib=usage.ru_maxrss / 1024)
        atomic(output / 'result.json', report)
        print(json.dumps({'stage': stage, 'counts': report['counts'],
                          'wall_seconds': report['wall_seconds']}), flush=True)

    def terminate(signum, frame):
        raise RuntimeError('External bounded controller terminated CameraHead diagnosis')

    signal.signal(signal.SIGTERM, terminate)

    def pin(path, digest, size=None):
        path = Path(path).resolve(strict=True)
        before = identity(path)
        require((size is None or before[2] == size) and sha(path) == digest,
                'Pinned original source/input mismatch: ' + str(path))
        require(identity(path) == before, 'Pinned file changed while hashing: ' + str(path))
        pinned[path] = (digest, before)
        return path

    try:
        pin(__file__, report['script_sha256'])
        require({4, 5, 6, 7} <= affinity, 'Original A76 CPU affinity unavailable')
        os.sched_setaffinity(0, {4, 5, 6, 7})
        os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                          NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
        paths = {'camera': args.camera_capture, 'cpu': args.cpu_reference, 'npu': args.npu_reference}
        metadata = {role: json.loads(pin(path, PINS[role]).read_text()) for role, path in paths.items()}
        camera, cpu, npu = (metadata[role] for role in ('camera', 'cpu', 'npu'))
        for row in metadata.values():
            require({key: row['checkpoint'][key] for key in CHECKPOINT} == CHECKPOINT,
                    'Original checkpoint provenance differs')
            require(not row.get('cleanup_errors', []), 'Reference contains cleanup errors')
        require(camera['kind'] == 'original_complete_CameraHead_CPU_capture' and
                camera['status'] == 'completed' and camera['camera_iterations'] == 4 and
                camera['source_and_inputs_unchanged'] and camera['checkpoint_stat_unchanged'],
                'Original full CameraHead capture required')
        require(cpu['schema'] == 'v99_original_native518_head_reference_continuation' and
                cpu['status'] == 'completed_original_native518_head_reference_continuation' and
                cpu['completed'] and cpu['all_requested_final_outputs_bitwise_equal_v92'] and
                cpu['camera_original_parameter_identity_unchanged'] and cpu['hooks_removed'] and
                cpu['pinned_source_and_inputs_unchanged'] and
                cpu['counts']['camera_refinements'] == 4,
                'Completed unchanged original CPU native CameraHead reference required')
        require(npu['schema'] == 'v100_original_native518_image_to_model_heads' and
                npu['status'] == 'completed_actual_original_native_model_path_not_quality_or_realtime_accepted' and
                npu['prepared_image_HW'] == [392, 518] and npu['camera_refinements'] == 4 and
                npu['complete_expected_execution'] and npu['resources_balanced'] and
                npu['source_images'] == cpu['source_images'],
                'Actual same-image original native NPU reference required')
        parameters = camera['artifacts']['camera-state.npz']['tensors']
        require(len(parameters) == 69 and 'empty_pose_tokens' in parameters and
                parameters == cpu['camera_original_parameter_metadata'],
                'All 69 original camera parameters including learned empty seed required')
        state_path = pin(args.camera_state, PINS['state'], 864717236)
        sources = {**cpu['source_hashes'], **MATH_SOURCES}
        repo = args.repo.resolve(strict=True)
        for name, digest in sources.items():
            pin(repo / name, digest)
        sys.path[:0] = [str(repo / 'third_party/vggt'), str(repo)]
        import numpy as np
        import torch
        from vggt.heads.camera_head import CameraHead
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        require(torch.__version__ == cpu['versions']['torch'], 'Original board Torch version differs')
        report.update(source_hashes=sources, reference_sha256=PINS,
                      reference_paths={role: str(path) for role, path in paths.items()},
                      source_images=cpu['source_images'], versions={'torch': torch.__version__, 'numpy': np.__version__},
                      original_parameter_metadata=parameters, cpu_affinity=sorted(os.sched_getaffinity(0)))

        def raw_sha(array):
            return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()

        def load_array(folder, row, shape):
            require(Path(row['file']).name == row['file'], 'Reference array must be a basename')
            array = np.load(pin(folder / row['file'], row['sha256'], row['bytes']), allow_pickle=False)
            require(tuple(array.shape) == tuple(shape) == tuple(row['shape']) and
                    array.dtype == np.float32 and np.isfinite(array).all() and
                    raw_sha(array) == row['raw_sha256'], 'Original array contract differs: ' + row['file'])
            return np.array(array, copy=True, order='C')

        def save(name, value):
            array = np.ascontiguousarray(value.detach().numpy() if isinstance(value, torch.Tensor) else value)
            require(array.dtype == np.float32 and np.isfinite(array).all(), 'Finite FP32 capture required')
            path = output / (name + '.npy')
            with path.with_suffix('.npy.tmp').open('xb') as stream:
                np.save(stream, array, allow_pickle=False)
            path.with_suffix('.npy.tmp').replace(path)
            row = {'file': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path),
                   'shape': list(array.shape), 'dtype': 'float32', 'tensor_bytes': array.nbytes,
                   'raw_sha256': raw_sha(array), 'finite': True}
            report['artifacts'][name] = row
            report['saved_array_bytes'] += row['bytes']
            require(report['saved_array_bytes'] < (1 << 20), 'Tiny head diagnosis exceeded 1 MiB capture budget')
            return row

        def compare(actual, reference):
            require(actual.shape == reference.shape, 'Camera comparison shape differs')
            delta = actual.astype(np.float64) - reference.astype(np.float64)
            norm = float(np.linalg.norm(reference.astype(np.float64)))
            outside = int(np.count_nonzero(np.abs(delta) > 1e-5 + 1e-4 * np.abs(reference.astype(np.float64))))
            return {'bitwise_equal': raw_sha(actual) == raw_sha(reference),
                    'array_equal': bool(np.array_equal(actual, reference)), 'max_abs_error': float(np.abs(delta).max()),
                    'relative_l2_error': float(np.linalg.norm(delta) / norm) if norm else None,
                    'outside_fixed_element_gate': outside, 'strict_rtol_1e_4_atol_1e_5_pass': outside == 0}

        progress('loading_original_69_parameter_state')
        with np.load(state_path, allow_pickle=False) as archive:
            require(set(archive.files) == set(parameters), 'Original camera parameter names differ')
            state = {name: np.array(archive[name], copy=True, order='C') for name in archive.files}
        for name, array in state.items():
            row = parameters[name]
            require(list(array.shape) == row['shape'] and array.dtype == np.float32 and
                    array.nbytes == row['bytes'] and row['finite'] and np.isfinite(array).all() and
                    raw_sha(array) == row['sha256'], 'Original parameter bytes differ: ' + name)
        with torch.device('meta'):
            head = CameraHead()
        head.load_state_dict({name: torch.from_numpy(array) for name, array in state.items()}, strict=True, assign=True)
        head.eval().requires_grad_(False)
        require(set(dict(head.named_parameters())) == set(parameters) and not list(head.named_buffers()) and
                all(p.device.type == 'cpu' and p.dtype == torch.float32 and not p.requires_grad for p in head.parameters()) and
                not any(m.training for m in head.modules()) and
                head.trunk_depth == len(head.trunk) == 4 and head.target_dim == 9 and
                head.token_norm.eps == head.trunk_norm.eps == 1e-5 and head.adaln_norm.eps == 1e-6 and
                (head.trans_act, head.quat_act, head.fl_act) == ('linear', 'linear', 'relu') and
                all(block.mlp.act.approximate == 'none' for block in head.trunk),
                'Strict original CPU FP32 CameraHead architecture required')
        state = None
        parameter_identity = lambda: [(name, id(p), p._version, p.data_ptr()) for name, p in head.named_parameters()]
        initial_parameters = parameter_identity()
        normalized = []

        def observe_norm(module, inputs, result):
            normalized.append(result.detach().clone())

        hooks.append(head.token_norm.register_forward_hook(observe_norm))
        cpu_folder, npu_folder = args.cpu_reference.parent, args.npu_reference.parent
        raw_cpu = load_array(cpu_folder, cpu['artifacts']['camera_input'], (1, 2, 2048))
        normalized_cpu = load_array(cpu_folder, cpu['artifacts']['camera_normalized_input'], (1, 2, 2048))
        pose_cpu = [load_array(cpu_folder, cpu['artifacts']['camera_pose_iteration_' + str(i)], (1, 2, 9)) for i in range(4)]
        npu_outputs = npu['calls'][0]['actual_raw_group_outputs'][3]
        frame = load_array(npu_folder, npu_outputs[0], (2, 1041, 1024))
        glob = load_array(npu_folder, npu_outputs[1], (1, 2082, 1024))
        raw_npu = np.concatenate((frame.reshape(1, 2, 1041, 1024)[:, :, 0],
                                  glob.reshape(1, 2, 1041, 1024)[:, :, 0]), axis=-1).copy(order='C')
        require(raw_sha(raw_npu) == '703a919bab1661985a4029498ccd2d5ce62043db97f6aa58a74cfe781b077d43',
                'Actual pre-normalization NPU camera tokens differ')
        del frame, glob
        npu_final = load_array(npu_folder, npu['calls'][0]['artifacts']['pose_enc'], (1, 2, 9))
        npu_codec = {name: load_array(npu_folder, npu['calls'][0]['artifacts'][name], shape)
                     for name, shape in [('extrinsics', (1, 2, 3, 4)), ('intrinsics', (1, 2, 3, 3))]}
        for tag, raw in [('original_CPU_tokens_control', raw_cpu), ('actual_NPU_tokens_diagnostic', raw_npu)]:
            normalized.clear()
            tokens = torch.from_numpy(raw.copy()).unsqueeze(2)
            input_digest = raw_sha(tokens.numpy())
            progress('calling_' + tag)
            tick, before = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
            with torch.inference_mode():
                poses = head([tokens], num_iterations=4)
            after = resource.getrusage(resource.RUSAGE_SELF)
            timings = {'observed_head_wall_seconds': time.perf_counter() - tick,
                       'head_process_user_seconds': after.ru_utime - before.ru_utime,
                       'head_process_system_seconds': after.ru_stime - before.ru_stime}
            report['counts']['camera_calls'] += 1
            require(len(poses) == 4 and len(normalized) == 1 and
                    all(tuple(p.shape) == (1, 2, 9) and p.dtype == torch.float32 for p in poses),
                    'Original four refinements/one token norm required')
            report['counts']['camera_refinements'] += 4
            require(parameter_identity() == initial_parameters and raw_sha(tokens.numpy()) == input_digest,
                    'Original head state/input changed')
            case = {'tag': tag, 'input_raw_sha256': raw_sha(raw), 'timings': timings,
                    'pose_vs_original_CPU': [compare(p.numpy(), ref) for p, ref in zip(poses, pose_cpu)]}
            report['cases'].append(case)
            prefix = 'cpu_control' if len(report['cases']) == 1 else 'npu_tokens_cpu_head'
            case['input'] = save(prefix + '_input', raw)
            case['normalized_input'] = save(prefix + '_normalized_input', normalized[0])
            case['activated_poses'] = [save(prefix + '_pose' + str(i), pose) for i, pose in enumerate(poses)]
            tick = time.perf_counter()
            with torch.inference_mode():
                codecs = [pose_encoding_to_extri_intri(pose, image_size_hw=(392, 518)) for pose in poses]
            timings['pose_codec_wall_seconds'] = time.perf_counter() - tick
            report['counts']['pose_codec_calls'] += 4
            case['extrinsics'] = save(prefix + '_extrinsics', torch.stack([pair[0] for pair in codecs]))
            case['intrinsics'] = save(prefix + '_intrinsics', torch.stack([pair[1] for pair in codecs]))
            if prefix == 'cpu_control':
                case['normalized_input_vs_original_CPU'] = compare(normalized[0].numpy(), normalized_cpu)
                passed = case['normalized_input_vs_original_CPU']['bitwise_equal'] and all(
                    row['bitwise_equal'] for row in case['pose_vs_original_CPU'])
                report['original_CPU_control_all_four_poses_and_token_norm_bitwise_equal'] = passed
                progress('original_CPU_control_outputs_saved')
                require(passed, 'CPU control failed: stop before NPU-token diagnosis')
            else:
                case['final_pose_vs_actual_NPU_head_same_tokens'] = compare(poses[-1].numpy(), npu_final)
                case['final_codec_vs_actual_NPU_head_same_tokens'] = {
                    name: compare(codecs[-1][index].numpy(), npu_codec[name])
                    for index, name in enumerate(('extrinsics', 'intrinsics'))}
            progress(tag + '_complete')
        report.update(status='completed_original_CameraHead_precision_diagnosis_not_quality_accepted',
                      completed=True, original_parameter_storage_identity_unchanged=True,
                      saved_inputs_unchanged=raw_sha(raw_cpu) == cpu['artifacts']['camera_input']['raw_sha256'] and
                      raw_sha(raw_npu) == '703a919bab1661985a4029498ccd2d5ce62043db97f6aa58a74cfe781b077d43')
    except BaseException as exc:
        report.update(status='failed', completed=False, error=type(exc).__name__ + ': ' + str(exc),
                      traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        for hook in reversed(hooks):
            try:
                hook.remove()
            except BaseException as exc:
                report['cleanup_errors'].append('hook removal: ' + str(exc))
        report['hooks_removed'] = not report['cleanup_errors']
        head = state = None
        gc.collect()
        report['head_and_loading_state_released'] = True
        try:
            require(all(identity(path) == before and
                        (path.suffix in ('.npy', '.npz') or sha(path) == digest)
                        for path, (digest, before) in pinned.items()), 'Pinned source/input changed')
            report['pinned_source_and_inputs_unchanged'] = True
            report['large_input_final_check'] = 'NPY/NPZ file identity after pre-load file hash and exact tensor-byte checks'
        except BaseException as exc:
            report['cleanup_errors'].append('final pins: ' + str(exc))
            report['pinned_source_and_inputs_unchanged'] = False
        try:
            os.sched_setaffinity(0, affinity)
            os.environ.clear()
            os.environ.update(old_env)
            sys.path[:] = old_path
            signal.signal(signal.SIGTERM, old_term)
            report['affinity_restored'] = os.sched_getaffinity(0) == affinity
            report['environment_restored'] = dict(os.environ) == old_env
            require(report['affinity_restored'] and report['environment_restored'], 'Process state restoration failed')
        except BaseException as exc:
            report['cleanup_errors'].append('restore: ' + str(exc))
        if report['cleanup_errors']:
            report.update(status='failed', completed=False)
        progress('finished')
    return 0 if report['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
