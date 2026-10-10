#!/usr/bin/env python3
"""Capture the complete original four-iteration CameraHead on RK3588S.

Two original images are preprocessed at 112 and passed through the original
CPU aggregator. Save its raw [1,2,2048] camera tokens, all four CPU pose
iterations and the 69 original CameraHead parameter tensors. No NPU or ORB.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

sys.dont_write_bytecode = True
BASE = Path('/mnt/vggt-data/vggt-slam2')
SOURCE = BASE / 'src'
INPUT_SHAPE = (1, 2, 2048)
OUTPUT_SHAPE = (1, 2, 9)
CPUS = {4, 5, 6, 7}
CHECKPOINT_SHA = 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
CHECKPOINT_BYTES = 5026874952
BASE_CAPTURE_SHA = '0485a463ef851f2f41230c5af95a9df41cb6e8536a4dd52c19ace1f683214d7b'
EXTRA_PINS = {
    'vggt_slam/runtime.py': 'baf65a5afcfa98584e25b1ef85d7cae31507a7da2b4cc03f97d2015776796e8c',
    'third_party/vggt/vggt/models/vggt.py': '64ce0625effa8569cac5da527532defa22bb3d3f1e4facfaf54fbd329740bde6',
    'third_party/vggt/vggt/heads/camera_head.py': '78b870018b20821d27587122003e68afd4f6b977bc13e0952fe507c31d6d3da2',
    'third_party/vggt/vggt/heads/head_act.py': '428bdb087f457b809a66e22d7d01147c1cb4dab43bb0d98bc61c7f05d8383a1e',
}
IMAGE_PINS = {
    'frame_0001.jpg': 'c23e57b9612522a10e61b1723749acddfe5b96b3a3642ec8c9db06e604d7f511',
    'frame_0002.jpg': 'dbffc9baa67d44e75a96a81a5f7e6582fdc07e86f3ffe9aa9bca76d3b717ae6f',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require_board():
    if platform.system() != 'Linux' or platform.machine().lower() not in ('aarch64', 'arm64'):
        raise RuntimeError('Actual model execution requires RK3588S Linux aarch64; host inference is forbidden')
    for path in (Path('/proc/device-tree/compatible'), Path('/sys/firmware/devicetree/base/compatible')):
        if path.is_file():
            value = path.read_bytes().replace(b'\0', b',').decode(errors='replace')
            if 'rk3588' not in value:
                raise RuntimeError(f'Unexpected board: {value}')
            return value
    raise RuntimeError('RK3588 device-tree identity unavailable')


def base_capture():
    path = Path(__file__).with_name('capture-v12-dino-block-board.py')
    if sha(path) != BASE_CAPTURE_SHA:
        raise ValueError('Frozen capture provenance helper SHA mismatch')
    spec = importlib.util.spec_from_file_location('_v15_original_capture_provenance', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def camera_contract(head):
    """Read real module properties; never substitute DINO epsilon defaults."""
    import torch.nn as nn
    if (len(head.trunk) != 4 or head.target_dim != 9 or head.training
            or (head.trans_act, head.quat_act, head.fl_act) != ('linear', 'linear', 'relu')):
        raise ValueError('Original CameraHead four-block/pose activation contract mismatch')
    if (head.token_norm.eps, head.trunk_norm.eps, head.adaln_norm.eps) != (1e-5, 1e-5, 1e-6):
        raise ValueError('CameraHead normalization epsilon mismatch')
    if head.adaln_norm.elementwise_affine or not isinstance(head.poseLN_modulation[0], nn.SiLU):
        raise ValueError('Original non-affine AdaLN / SiLU modulation required')
    blocks = []
    for index, block in enumerate(head.trunk):
        if (block.norm1.eps, block.norm2.eps, block.mlp.act.approximate) != (1e-5, 1e-5, 'none'):
            raise ValueError('Camera trunk LayerNorm/Erf GELU mismatch')
        if (block.attn.num_heads != 16 or block.attn.head_dim != 128
                or block.attn.qkv.in_features != 2048 or block.mlp.fc1.out_features != 8192
                or not isinstance(block.attn.q_norm, nn.Identity)
                or not isinstance(block.attn.k_norm, nn.Identity)
                or block.attn.rope is not None or not block.attn.fused_attn
                or block.attn.attn_drop.p != 0 or block.attn.proj_drop.p != 0):
            raise ValueError('Original camera trunk attention/MLP contract mismatch')
        blocks.append({'index': index, 'norm1_eps': block.norm1.eps, 'norm2_eps': block.norm2.eps,
                       'num_heads': 16, 'head_dim': 128, 'mlp_hidden_dim': 8192,
                       'gelu_approximate': 'none', 'layerscale_shape': list(block.ls1.gamma.shape)})
    if head.pose_branch.act.approximate != 'none':
        raise ValueError('Original pose branch exact GELU required')
    return {'input_shape': list(INPUT_SHAPE), 'output_shape': list(OUTPUT_SHAPE),
            'trunk_depth': 4, 'iterations': 4, 'trunk_calls_per_forward': 16,
            'token_norm_eps': head.token_norm.eps, 'trunk_norm_eps': head.trunk_norm.eps,
            'adaln_eps': head.adaln_norm.eps, 'adaln_elementwise_affine': False,
            'qk_norm': False, 'rope': False, 'pose_encoding_type': 'absT_quaR_FoV',
            'target_dim': 9, 'trans_act': 'linear', 'quat_act': 'linear', 'fl_act': 'relu',
            'pose_branch_gelu_approximate': 'none', 'modulation_activation': 'SiLU',
            'blocks': blocks, 'all_original_iterations_retained': True}


def save_array(output, name, array, helper):
    import numpy as np
    info = helper.array_info(array)
    path = output / name
    with path.open('xb') as stream:
        np.save(stream, array, allow_pickle=False)
    return {'bytes': path.stat().st_size, 'sha256': sha(path), 'shape': info['shape'],
            'dtype': info['dtype'], 'finite': True, 'tensor_sha256': info['sha256']}


def capture(output):
    receipt = {'kind': 'original_complete_CameraHead_CPU_capture', 'status': 'failed',
               'script_sha256': sha(__file__), 'output': str(output), 'ORB_used': False,
               'npu_used': False, 'full_VGGT_forward_executed': False,
               'aggregator_forward_executed': True, 'camera_iterations': 4,
               'CPU_camera_forward_call_count': 0, 'CPU_trunk_call_count': 0,
               'export_trace_call_count': 0, 'real_time_validated': False,
               'full_model_quality_validated': False, 'production_modified': False,
               'artifacts': {}, 'cleanup_errors': []}
    start, allowed, pins, hooks = time.monotonic(), None, {}, []
    try:
        receipt['device_tree_compatible'] = require_board()
        helper, helper_path = base_capture()
        originals = {**helper.SOURCE_PINS, **EXTRA_PINS}
        receipt['source_hashes'] = {name: sha(SOURCE / name) for name in originals}
        if receipt['source_hashes'] != originals:
            raise ValueError('Frozen original aggregator/CameraHead/runtime source SHA mismatch')
        pins = {str(SOURCE / name): value for name, value in originals.items()}
        pins.update({str(helper_path): BASE_CAPTURE_SHA, str(Path(__file__).resolve()): sha(__file__)})
        images_paths = [BASE / 'datasets/office_loop' / name for name in IMAGE_PINS]
        receipt['input_images'] = [{'path': str(path), 'sha256': sha(path)} for path in images_paths]
        if any(item['sha256'] != IMAGE_PINS[path.name] for item, path in zip(receipt['input_images'], images_paths)):
            raise ValueError('Original frame1/frame2 image SHA mismatch')
        pins.update({item['path']: item['sha256'] for item in receipt['input_images']})
        checkpoint = BASE / 'weights/model.pt'
        stat = checkpoint.stat()
        if stat.st_size != CHECKPOINT_BYTES or sha(checkpoint) != CHECKPOINT_SHA:
            raise ValueError('Official original checkpoint size/SHA mismatch')
        receipt['checkpoint'] = {'path': str(checkpoint), 'bytes': CHECKPOINT_BYTES, 'sha256': CHECKPOINT_SHA}
        allowed = set(os.sched_getaffinity(0))
        if not CPUS <= allowed:
            raise ValueError('CPU 4,5,6,7 must be allowed')
        os.sched_setaffinity(0, CPUS)
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[key] = '4'
        os.environ['MPLBACKEND'] = 'Agg'
        sys.path[:0] = [str(SOURCE), str(SOURCE / 'third_party/vggt')]
        import numpy as np
        import torch
        from PIL import Image
        from vggt_slam.runtime import load_model
        from vggt.utils.load_fn import load_and_preprocess_images
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        receipt['runtime'] = {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                              'cpu_affinity': sorted(os.sched_getaffinity(0)), 'threads': 4,
                              'interop_threads': torch.get_num_interop_threads()}
        for item, path in zip(receipt['input_images'], images_paths):
            with Image.open(path) as original:
                item['raw_wh'] = list(original.size)
                if original.getexif().get(274) not in (None, 1):
                    raise ValueError('Unexpected EXIF orientation')
        tick = time.perf_counter()
        model = load_model(checkpoint, device='cpu', image_size=112, prune_unused_heads=True, offline=True)
        receipt['model_load_seconds'] = time.perf_counter() - tick
        head = model.camera_head
        receipt['architecture'] = camera_contract(head)
        if tuple(model.aggregator.patch_embed.pos_embed.shape) != (1, 1370, 1024):
            raise ValueError('Original 518 DINO positional weights must remain intact')
        images = load_and_preprocess_images([str(path) for path in images_paths], image_size=112, mode='crop')
        if tuple(images.shape) != (2, 3, 56, 112) or images.dtype != torch.float32 or not torch.isfinite(images).all():
            raise ValueError('Expected two real crop112 FP32 images')
        receipt['preprocessing'] = {'image_size': 112, 'mode': 'crop', 'image_shape': list(images.shape),
                                    'original_DINO_position_shape': [1, 1370, 1024]}
        with torch.inference_mode():
            tick = time.perf_counter()
            aggregated, patch_start, target_tokens, image_match_ratio = model.aggregator(images.unsqueeze(0), False)
            receipt['aggregator_CPU_seconds'] = time.perf_counter() - tick
            tokens = aggregated[-1][:, :, 0].detach().clone().contiguous()
            if tuple(tokens.shape) != INPUT_SHAPE or not torch.isfinite(tokens).all():
                raise ValueError('Unexpected original aggregator camera tokens')
            def counted(module, args, result):
                receipt['CPU_trunk_call_count'] += 1
            hooks = [block.register_forward_hook(counted) for block in head.trunk]
            times, exact = [], []
            for index in range(3):
                tick = time.perf_counter()
                poses = head(aggregated if index == 0 else [tokens.unsqueeze(2)], num_iterations=4)
                times.append(time.perf_counter() - tick)
                receipt['CPU_camera_forward_call_count'] += 1
                if len(poses) != 4 or any(tuple(value.shape) != OUTPUT_SHAPE or
                        value.dtype != torch.float32 or not torch.isfinite(value).all() for value in poses):
                    raise ValueError('Original four CameraHead pose outputs invalid')
                if index == 0:
                    reference = [value.detach().clone() for value in poses]
                exact.append(all(torch.equal(value, expected) for value, expected in zip(poses, reference)))
                if not exact[-1]:
                    raise ValueError('Raw-token wrapper changed original CPU CameraHead output')
        for hook in hooks:
            hook.remove()
        hooks.clear()
        if receipt['CPU_trunk_call_count'] != 48:
            raise ValueError('CameraHead must execute four blocks in all four iterations of each of three calls')
        # Keep only the real CameraHead and small captured tensors before large
        # state serialization. The aggregator/depth heads are not exported.
        del model, aggregated, target_tokens, image_match_ratio, images, poses
        gc.collect()
        receipt['artifacts']['input.npy'] = save_array(output, 'input.npy', np.ascontiguousarray(tokens.numpy()), helper)
        receipt['artifacts']['output.npy'] = save_array(output, 'output.npy', np.ascontiguousarray(reference[-1].numpy()), helper)
        refs = {f'iteration_{i}': np.ascontiguousarray(value.numpy()) for i, value in enumerate(reference)}
        path = output / 'pose-iterations.npz'
        with path.open('xb') as stream:
            np.savez(stream, **refs)
        receipt['artifacts'][path.name] = {'bytes': path.stat().st_size, 'sha256': sha(path),
                                           'tensors': {key: helper.array_info(value) for key, value in refs.items()}}
        state = {key: np.ascontiguousarray(value.detach().numpy()) for key, value in head.state_dict().items()}
        if len(state) != 69:
            raise ValueError('Expected all 69 original CameraHead parameter tensors')
        path = output / 'camera-state.npz'
        with path.open('xb') as stream:
            np.savez(stream, **state)
        receipt['artifacts'][path.name] = {'bytes': path.stat().st_size, 'sha256': sha(path),
                                           'checkpoint_key_prefix': 'camera_head.',
                                           'tensors': {key: helper.array_info(value) for key, value in state.items()}}
        receipt['unique_state_bytes'] = sum(value.nbytes for value in state.values())
        receipt['cpu_reference'] = {'device': 'cpu', 'dtype': 'float32', 'no_grad': True, 'eval': True,
                                    'finite': True, 'all_elementwise_equal': all(exact), 'repeats': 3,
                                    'seconds': times, 'mean_seconds': sum(times) / 3.,
                                    'first_call_uses_original_aggregator_output_list': True,
                                    'subsequent_calls_use_exact_raw_camera_tokens': True,
                                    'timing_includes_tiny_trunk_hooks': True,
                                    'timing_excludes_aggregator_load_export_IO': True}
        receipt['source_and_input_hashes'] = pins.copy()
        receipt['source_and_input_hashes_after'] = {path: sha(path) for path in pins}
        receipt['source_and_inputs_unchanged'] = receipt['source_and_input_hashes_after'] == pins
        after = checkpoint.stat()
        receipt['checkpoint_stat_unchanged'] = (after.st_size, after.st_mtime_ns, after.st_ino) == (stat.st_size, stat.st_mtime_ns, stat.st_ino)
        if not receipt['source_and_inputs_unchanged'] or not receipt['checkpoint_stat_unchanged']:
            raise ValueError('Source/input/checkpoint changed during capture')
        receipt['status'] = 'completed'
    except BaseException as exc:
        receipt.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        for hook in hooks:
            try:
                hook.remove()
            except BaseException as exc:
                receipt['cleanup_errors'].append(f'Hook cleanup: {exc}')
        if allowed is not None:
            try:
                os.sched_setaffinity(0, allowed)
                receipt['affinity_restored'] = set(os.sched_getaffinity(0)) == allowed
                if not receipt['affinity_restored']:
                    raise RuntimeError('CPU affinity restoration mismatch')
            except BaseException as exc:
                receipt['cleanup_errors'].append(f'Affinity cleanup: {exc}')
        if receipt['cleanup_errors']:
            receipt['status'] = 'failed'
        receipt['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.
        receipt['total_seconds_including_hash_load_IO'] = time.monotonic() - start
        with (output / 'capture.json').open('x') as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write('\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BASE / 'experiments/vggt-v15-camera/capture112-r1')
    args = parser.parse_args()
    output = args.output.resolve()
    if Path.cwd().resolve() != BASE or Path(__file__).resolve().parents[3] != SOURCE:
        parser.error('Run the deployed source from /mnt/vggt-data/vggt-slam2 on RK3588S')
    if not output.is_relative_to(BASE / 'experiments/vggt-v15-camera') or os.path.lexists(args.output):
        parser.error('Output must be a new directory below board experiments/vggt-v15-camera')
    require_board()
    output.mkdir(parents=True, exist_ok=False)
    receipt = capture(output)
    print(json.dumps({'status': receipt['status'], 'error': receipt.get('error'),
                      'output': str(output / 'capture.json'), 'sha256': sha(output / 'capture.json')}))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
