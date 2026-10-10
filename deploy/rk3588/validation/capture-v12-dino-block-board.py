#!/usr/bin/env python3
"""Capture official DINO block 0 on ARM; never run the remaining ViT/VGGT.

Source deploy/rk3588/env.sh, then run from /mnt/vggt-data/vggt-slam2.
Default execution is CPU FP32 only. --benchmark-current-linear separately
measures the existing RKNN Linear backend after saving the CPU artifacts.
This script does not export/build an RKNN graph or change production sources.
"""
from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping
from functools import partial
import hashlib
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
CHECKPOINT_SIZE = 5026874952
CHECKPOINT_SHA = 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
IMAGE = BASE / 'datasets/office_loop/frame_0001.jpg'
IMAGE_SHA = 'c23e57b9612522a10e61b1723749acddfe5b96b3a3642ec8c9db06e604d7f511'
PREFIX = 'aggregator.patch_embed.'
SOURCE_PINS = {
    'third_party/vggt/vggt/layers/__init__.py': 'e3c28c7b775165b6844000eb69f44d104d6c3803ac7da5d59f6ac3ecfb46d9d8',
    'third_party/vggt/vggt/layers/vision_transformer.py': 'f781186de8e9a52acc4dfc07820cde01bb182a3eb6835f630b863eaf2cde6bde',
    'third_party/vggt/vggt/layers/block.py': 'd8b8cb1bfe5f084d33810b9cc96626fa494a4beea2c3c175da197841070bbf17',
    'third_party/vggt/vggt/layers/attention.py': '9d80ff6c6f562efc29d806125fe47bd9b504a6356ce2d9b5d61e8ee4ea770a00',
    'third_party/vggt/vggt/layers/patch_embed.py': 'c2543ec293951d081db662b2125027625f6e632971630facaa7f911c2377d335',
    'third_party/vggt/vggt/layers/layer_scale.py': '83586eaa81ef11828b1001f3dd85747ba362df23ddb57c8454b2eba279f5a558',
    'third_party/vggt/vggt/layers/mlp.py': '255825c73b60a916dd00eb1e38aacbcdbf316e40d6a005efb46e245b7edb43aa',
    'third_party/vggt/vggt/layers/drop_path.py': 'b9f8236e86054b9d9a71275efcad2a9ecaa1f86b529d4b8d6109ddb5e806f67a',
    'third_party/vggt/vggt/layers/swiglu_ffn.py': 'af287ee5b7e6a2a5608ed78cf8d392a00191718825a3889c224ccaa0fd914089',
    'third_party/vggt/vggt/models/aggregator.py': '623a943847a3259292bfd5c8675d8b80b09abc845eefadba7622e6a901cb9cfd',
    'third_party/vggt/vggt/utils/load_fn.py': 'b0fe762f65497dca9a22069a3e4a37fd3508398c55383a8c1cb85c403fb3e9dd',
}
RKNN_SOURCE_PIN = '69056a2cb44ba69a2d311265b71ff919e4758fcaa150834a0b0cfe85dbcaef7d'
BLOCK_KEYS = {
    'norm1.weight', 'norm1.bias', 'attn.qkv.weight', 'attn.qkv.bias',
    'attn.proj.weight', 'attn.proj.bias', 'ls1.gamma', 'norm2.weight',
    'norm2.bias', 'mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight',
    'mlp.fc2.bias', 'ls2.gamma',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def array_info(array):
    if array.dtype.name != 'float32' or not array.flags.c_contiguous:
        raise ValueError('Captured values must be contiguous CPU FP32')
    import numpy as np
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite captured tensor')
    return {'shape': list(array.shape), 'dtype': 'float32', 'finite': True,
            'bytes': array.nbytes,
            'sha256': hashlib.sha256(memoryview(array).cast('B')).hexdigest()}


def normalization_constants():
    """Read the pinned aggregator constants without importing a full model."""
    values = {}
    tree = ast.parse((SOURCE / 'third_party/vggt/vggt/models/aggregator.py').read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('_RESNET_MEAN', '_RESNET_STD'):
                    values[target.id] = ast.literal_eval(node.value)
    if set(values) != {'_RESNET_MEAN', '_RESNET_STD'}:
        raise ValueError('Missing original aggregator normalization constants')
    return values['_RESNET_MEAN'], values['_RESNET_STD']


def current_linear_benchmark(block, tokens, reference, torch, np):
    """Optional existing mixed CPU/RKNN baseline; CPU artifacts stay unchanged."""
    from vggt_slam.rknn_linear import install_rknn_linear
    handle = None
    result = {'requested': True, 'status': 'started', 'first_seconds': None,
              'warm_seconds': [], 'errors': [], 'cleanup_errors': [],
              'core_mask': 1, 'weight_layout': 'native', 'max_contexts': 8,
              'max_rows': 1024, 'reference': 'CPU FP32 original SDPA block output'}
    original_error = None
    try:
        started = time.perf_counter()
        handle = install_rknn_linear(block, core_mask=1, weight_layout='native',
                                     max_contexts=8, max_rows=1024)
        result['install_seconds'] = time.perf_counter() - started
        for index in range(4):
            started = time.perf_counter()
            with torch.no_grad():
                actual = block(tokens)
            seconds = time.perf_counter() - started
            array = np.ascontiguousarray(actual.numpy())
            array_info(array)
            delta = array.astype(np.float64) - reference.astype(np.float64)
            error = {'max_abs': float(np.max(np.abs(delta))),
                     'mean_abs': float(np.mean(np.abs(delta))),
                     'rmse': float(np.sqrt(np.mean(delta * delta))),
                     'relative_l2': float(np.linalg.norm(delta) /
                                          max(np.linalg.norm(reference.astype(np.float64)), 1e-30))}
            result['errors'].append(error)
            if index == 0:
                result['first_seconds'] = seconds
            else:
                result['warm_seconds'].append(seconds)
            del actual, array, delta
        result['status'] = 'completed'
    except Exception as exc:
        original_error = exc
        result['status'] = 'failed'
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if handle is not None:
            for name in ('restore', 'close'):
                try:
                    getattr(handle, name)()
                except Exception as exc:
                    result['cleanup_errors'].append(f'{name}: {type(exc).__name__}: {exc}')
            try:
                result['npu'] = handle.metrics()
            except Exception as exc:
                result['cleanup_errors'].append(f'metrics: {type(exc).__name__}: {exc}')
    result['npu_used'] = bool(result.get('npu', {}).get('npu_calls', 0))
    if result['cleanup_errors'] or not result['npu_used'] or result.get('npu', {}).get('npu_errors', 0):
        result['status'] = 'failed'
    return result, original_error


def capture(output, benchmark_current_linear):
    receipt = {'kind': 'official_dino_block0_CPU_capture', 'status': 'started',
               'output': str(output), 'script_sha256': sha(__file__),
               'source_root': str(SOURCE), 'full_VGGT_constructed': False,
               'original_DINO_depth': 24, 'constructed_DINO_depth': 1,
               'executed_blocks': [0], 'model_inference_scope': 'token preparation and DINO block 0 only',
               'no_npu': not benchmark_current_linear, 'npu_used': False,
               'inference_on_board_only': True, 'real_time_validated': False,
               'accuracy_validated': False, 'artifacts': {},
               'timing_scope': 'three block-only CPU forwards; excludes imports/load/preprocess/token preparation/capture/I/O'}
    started = time.monotonic()
    try:
        pins = dict(SOURCE_PINS)
        if benchmark_current_linear:
            pins['vggt_slam/rknn_linear.py'] = RKNN_SOURCE_PIN
        before = {name: sha(SOURCE / name) for name in pins}
        receipt['source_hashes'] = before
        if before != pins:
            raise ValueError('Pinned original DINO/preprocessing source hash mismatch')
        checkpoint = BASE / 'weights/model.pt'
        checkpoint_stat = checkpoint.stat()
        receipt['checkpoint'] = {'path': str(checkpoint), 'bytes': checkpoint_stat.st_size}
        if checkpoint_stat.st_size != CHECKPOINT_SIZE:
            raise ValueError('Official checkpoint size mismatch')
        receipt['checkpoint']['sha256'] = sha(checkpoint)
        if receipt['checkpoint']['sha256'] != CHECKPOINT_SHA:
            raise ValueError('Official checkpoint SHA mismatch')
        receipt['image'] = {'path': str(IMAGE), 'sha256': sha(IMAGE)}
        if receipt['image']['sha256'] != IMAGE_SHA:
            raise ValueError('Original frame_0001.jpg SHA mismatch')
        allowed = set(os.sched_getaffinity(0))
        if not {6, 7}.issubset(allowed):
            raise ValueError(f'CPU 6,7 must be allowed; actual allowed CPUs: {sorted(allowed)}')
        os.sched_setaffinity(0, {6, 7})
        env_values = {'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1',
                      'MKL_NUM_THREADS': '2', 'MPLBACKEND': 'Agg'}
        receipt['environment_before'] = {key: os.environ.get(key) for key in env_values}
        os.environ.update(env_values)
        sys.path[:0] = [str(SOURCE), str(SOURCE / 'third_party/vggt')]
        import numpy as np
        import torch
        from PIL import Image
        from vggt.layers import MemEffAttention, NestedTensorBlock
        from vggt.layers.vision_transformer import DinoVisionTransformer
        from vggt.utils.load_fn import load_and_preprocess_images
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        receipt['runtime'] = {'python': sys.version, 'platform': platform.platform(),
                              'torch': torch.__version__, 'numpy': np.__version__,
                              'torch_threads': torch.get_num_threads(),
                              'torch_interop_threads': torch.get_num_interop_threads(),
                              'cpu_affinity': sorted(os.sched_getaffinity(0)),
                              'environment': {key: os.environ.get(key) for key in
                                              (*env_values, 'OMP_WAIT_POLICY', 'GOMP_SPINCOUNT', 'LD_LIBRARY_PATH')},
                              'torch_config': torch.__config__.show()}
        with Image.open(IMAGE) as original:
            receipt['image']['raw_wh'] = list(original.size)
            receipt['image']['exif_orientation'] = original.getexif().get(274)
            if receipt['image']['exif_orientation'] not in (None, 1):
                raise ValueError('Unexpected EXIF orientation in pinned original image')
        load_started = time.perf_counter()
        state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        if isinstance(state, Mapping):
            for key in ('state_dict', 'model'):
                if key in state and isinstance(state[key], Mapping):
                    state = state[key]
                    break
        if not isinstance(state, Mapping) or 'aggregator.camera_token' not in state:
            raise ValueError('Checkpoint is not the original VGGT state dictionary')
        with torch.device('meta'):
            dino = DinoVisionTransformer(
                img_size=518, patch_size=14, in_chans=3, embed_dim=1024,
                depth=1, num_heads=16, mlp_ratio=4., qkv_bias=True,
                ffn_bias=True, proj_bias=True, drop_path_rate=0., init_values=1.,
                block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention),
                ffn_layer='mlp', block_chunks=0, num_register_tokens=4,
                interpolate_antialias=True, interpolate_offset=0., qk_norm=False)
        expected_keys = set(dino.state_dict())
        # Only token preparation and block 0 are materialized. The two tiny
        # final-norm tensors are loaded for strict constructor state coverage,
        # but that norm and the other 23 blocks are never executed.
        selected = {key: state[PREFIX + key] for key in sorted(expected_keys)}
        selected_info = {}
        for key, tensor in selected.items():
            if tensor.device.type != 'cpu' or tensor.dtype != torch.float32:
                raise ValueError(f'Official CPU FP32 tensor required: {key}')
            selected_info[PREFIX + key] = array_info(np.ascontiguousarray(tensor.numpy()))
        dino.load_state_dict(selected, strict=True, assign=True)
        del selected, state
        if any(value.is_meta for value in (*dino.parameters(), *dino.buffers())):
            raise ValueError('Strict assignment left meta tensors')
        dino.eval()
        block = dino.blocks[0]
        receipt['checkpoint_load_seconds'] = time.perf_counter() - load_started
        receipt['loaded_state_tensors'] = selected_info
        receipt['loaded_state_bytes'] = sum(value['bytes'] for value in selected_info.values())
        receipt['architecture'] = {
            'embed_dim': 1024, 'num_heads': 16, 'head_dim': 64, 'patch_size': 14,
            'mlp_hidden_dim': 4096, 'norm1_eps': block.norm1.eps, 'norm2_eps': block.norm2.eps,
            'gelu_approximate': block.mlp.act.approximate, 'layerscale_initial_value': 1.,
            'qk_norm': False, 'rope': False, 'dropout': 0., 'drop_path': 0., 'eval': True,
            'num_register_tokens': 4, 'cls_tokens': 1, 'patch_tokens': 777, 'sequence_length': 782,
            'attention': 'torch.nn.functional.scaled_dot_product_attention (original CPU dispatch)',
            'position_included_in_input': True, 'interpolate_antialias': True, 'interpolate_offset': 0.,
        }
        if (block.norm1.eps, block.norm2.eps, block.mlp.act.approximate) != (1e-6, 1e-6, 'none'):
            raise ValueError('Original block normalization/GELU contract mismatch')
        mean, std = normalization_constants()
        preprocess_started = time.perf_counter()
        images = load_and_preprocess_images([str(IMAGE)], image_size=518, mode='crop')
        if images.shape != (1, 3, 294, 518) or images.dtype != torch.float32:
            raise ValueError(f'Expected original 294x518 input, got {tuple(images.shape)} {images.dtype}')
        # Keep the original aggregator's [B,S,C,H,W] subtraction/division and
        # then its view(B*S,C,H,W), rather than alternate normalizations.
        image5 = images.unsqueeze(0)
        mean5 = torch.FloatTensor(mean).view(1, 1, 3, 1, 1)
        std5 = torch.FloatTensor(std).view(1, 1, 3, 1, 1)
        normalized = ((image5 - mean5) / std5).view(1, 3, 294, 518)
        receipt['preprocess_seconds'] = time.perf_counter() - preprocess_started
        receipt['preprocessing'] = {'mode': 'crop', 'image_size': 518, 'normalized_shape': list(normalized.shape),
                                    'mean': mean, 'std': std, 'source': 'original load_and_preprocess_images'}
        with torch.no_grad():
            token_started = time.perf_counter()
            tokens = dino.prepare_tokens_with_masks(normalized, masks=None)
            receipt['token_preparation_seconds'] = time.perf_counter() - token_started
            if tokens.shape != (1, 782, 1024) or tokens.dtype != torch.float32:
                raise ValueError(f'Unexpected block 0 input shape/dtype: {tuple(tokens.shape)} {tokens.dtype}')
            capture_started = time.perf_counter()
            reference = block(tokens)
            receipt['initial_block_capture_seconds'] = time.perf_counter() - capture_started
            times = []
            exact = []
            for _ in range(3):
                timer = time.perf_counter()
                actual = block(tokens)
                times.append(time.perf_counter() - timer)
                exact.append(bool(torch.equal(actual, reference)))
                if not torch.isfinite(actual).all() or not exact[-1]:
                    raise ValueError('Repeated CPU reference is nonfinite or not elementwise identical')
            del actual
        input_array = np.ascontiguousarray(tokens.numpy())
        output_array = np.ascontiguousarray(reference.numpy())
        if output_array.shape != input_array.shape:
            raise ValueError('Original block output shape differs from input')
        state_arrays = {key: np.ascontiguousarray(value.detach().numpy())
                        for key, value in block.state_dict().items()}
        if set(state_arrays) != BLOCK_KEYS:
            raise ValueError(f'Unexpected original block state keys: {sorted(state_arrays)}')
        tensor_info = {key: array_info(value) for key, value in state_arrays.items()}
        for filename, array in (('input.npy', input_array), ('output.npy', output_array)):
            metadata = array_info(array)
            path = output / filename
            with path.open('xb') as stream:
                np.save(stream, array, allow_pickle=False)
            receipt['artifacts'][filename] = {'bytes': path.stat().st_size, 'sha256': sha(path),
                                              'shape': metadata['shape'], 'dtype': metadata['dtype'],
                                              'finite': True, 'tensor_sha256': metadata['sha256']}
        state_path = output / 'block-state.npz'
        with state_path.open('xb') as stream:
            np.savez(stream, **state_arrays)
        receipt['artifacts']['block-state.npz'] = {'bytes': state_path.stat().st_size,
                                                  'sha256': sha(state_path), 'tensors': tensor_info,
                                                  'checkpoint_key_prefix': PREFIX + 'blocks.0.'}
        receipt['cpu_reference'] = {'seconds': times, 'mean_seconds': sum(times) / len(times),
                                    'repeats': 3, 'capture_call_excluded': True,
                                    'all_elementwise_equal': all(exact), 'finite': True,
                                    'no_grad': True, 'eval': True, 'dtype': 'float32', 'device': 'cpu'}
        if benchmark_current_linear:
            mixed, mixed_error = current_linear_benchmark(block, tokens, output_array, torch, np)
            receipt['current_linear_benchmark'] = mixed
            receipt['npu_used'] = mixed['npu_used']
            if mixed['status'] != 'completed':
                raise RuntimeError('Optional current RKNN Linear baseline failed') from mixed_error
        after = {name: sha(SOURCE / name) for name in pins}
        receipt['source_hashes_after'] = after
        receipt['source_unchanged'] = after == before
        receipt['image_unchanged'] = sha(IMAGE) == IMAGE_SHA
        stat_after = checkpoint.stat()
        receipt['checkpoint_stat_unchanged'] = (
            stat_after.st_size, stat_after.st_mtime_ns, stat_after.st_ino) == (
                checkpoint_stat.st_size, checkpoint_stat.st_mtime_ns, checkpoint_stat.st_ino)
        if not all(receipt[key] for key in ('source_unchanged', 'image_unchanged', 'checkpoint_stat_unchanged')):
            raise ValueError('A source or input changed during capture')
        receipt['status'] = 'completed'
    except Exception as exc:
        receipt['status'] = 'failed'
        receipt['error'] = f'{type(exc).__name__}: {exc}'
        receipt['traceback'] = traceback.format_exc()
    finally:
        receipt['total_seconds_including_hash_load_IO'] = time.monotonic() - started
        receipt['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.
        receipt['rss_scope'] = 'this process high-water mark, includes imports/hash/load/capture and any inherited pre-exec history'
        with (output / 'capture.json').open('x') as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write('\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BASE / 'experiments/orb-v12-dino-block/capture-r1')
    parser.add_argument('--benchmark-current-linear', action='store_true',
                        help='after CPU capture, separately measure existing mixed RKNN Linear block')
    args = parser.parse_args()
    if platform.machine() != 'aarch64' or Path.cwd().resolve() != BASE:
        parser.error('Run only on the RK3588S board from /mnt/vggt-data/vggt-slam2')
    if Path(__file__).resolve().parents[3] != SOURCE:
        parser.error('Use the deployed script under /mnt/vggt-data/vggt-slam2/src')
    output = args.output.resolve()
    if not output.is_relative_to(BASE / 'experiments') or output == BASE / 'experiments':
        parser.error('--output must be a new directory strictly below the board experiments directory')
    if os.path.lexists(args.output):
        parser.error('Refusing an existing output directory or symlink')
    output.mkdir(parents=True, exist_ok=False)
    receipt = capture(output, args.benchmark_current_linear)
    print(json.dumps({'status': receipt['status'], 'receipt': str(output / 'capture.json'),
                      'sha256': sha(output / 'capture.json'), 'error': receipt.get('error')}))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
