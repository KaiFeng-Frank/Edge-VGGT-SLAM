#!/usr/bin/env python3
"""Capture original DINO block 0 at actual crop112 on the RK3588S only.

The constructor and positional weights remain the original 518 configuration.
Original token preparation interpolates them to 32 patches plus CLS/registers;
this is not a slice of the previous 782-token capture. No production changes.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from functools import partial
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
SHAPE = (1, 37, 1024)
HELPER_SHA = '0485a463ef851f2f41230c5af95a9df41cb6e8536a4dd52c19ace1f683214d7b'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_helper():
    path = Path(__file__).with_name('capture-v12-dino-block-board.py')
    if sha(path) != HELPER_SHA:
        raise ValueError('Frozen V12 CPU capture helper SHA mismatch')
    spec = importlib.util.spec_from_file_location('_v14_small_capture_base', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def capture(output):
    receipt = {
        'kind': 'official_dino_block0_CPU_capture', 'status': 'started',
        'experiment': 'v14_original_DINO_crop112', 'output': str(output),
        'script_sha256': sha(__file__), 'source_root': str(SOURCE),
        'full_VGGT_constructed': False, 'original_DINO_depth': 24,
        'constructed_DINO_depth': 1, 'executed_blocks': [0],
        'model_inference_scope': 'original token preparation and DINO block 0 only',
        'no_npu': True, 'npu_used': False, 'orb_used': False,
        'inference_on_board_only': True, 'real_time_validated': False,
        'accuracy_validated': False, 'full_model_quality_validated': False,
        'artifacts': {}, 'cleanup_errors': [],
        'timing_scope': 'three CPU block-only calls including first; excludes load, token preparation and I/O',
    }
    started = time.monotonic()
    original_affinity = None
    try:
        helper, helper_path = load_helper()
        receipt['frozen_helper'] = {'path': str(helper_path), 'sha256': HELPER_SHA}
        before = {name: sha(SOURCE / name) for name in helper.SOURCE_PINS}
        receipt['source_hashes'] = before
        if before != helper.SOURCE_PINS:
            raise ValueError('Pinned original DINO/preprocessing source hash mismatch')
        checkpoint = BASE / 'weights/model.pt'
        checkpoint_stat = checkpoint.stat()
        receipt['checkpoint'] = {'path': str(checkpoint), 'bytes': checkpoint_stat.st_size}
        if checkpoint_stat.st_size != helper.CHECKPOINT_SIZE:
            raise ValueError('Official checkpoint size mismatch')
        receipt['checkpoint']['sha256'] = sha(checkpoint)
        if receipt['checkpoint']['sha256'] != helper.CHECKPOINT_SHA:
            raise ValueError('Official checkpoint SHA mismatch')
        receipt['image'] = {'path': str(helper.IMAGE), 'sha256': sha(helper.IMAGE)}
        if receipt['image']['sha256'] != helper.IMAGE_SHA:
            raise ValueError('Original frame_0001.jpg SHA mismatch')
        original_affinity = set(os.sched_getaffinity(0))
        if not {6, 7}.issubset(original_affinity):
            raise ValueError('CPU 6,7 must be allowed')
        os.sched_setaffinity(0, {6, 7})
        env = {'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1',
               'MKL_NUM_THREADS': '2', 'MPLBACKEND': 'Agg'}
        receipt['environment_before'] = {key: os.environ.get(key) for key in env}
        os.environ.update(env)
        sys.path[:0] = [str(SOURCE), str(SOURCE / 'third_party/vggt')]
        import numpy as np
        import torch
        from PIL import Image
        from vggt.layers import MemEffAttention, NestedTensorBlock
        from vggt.layers.vision_transformer import DinoVisionTransformer
        from vggt.utils.load_fn import load_and_preprocess_images
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        receipt['runtime'] = {
            'python': sys.version, 'platform': platform.platform(),
            'torch': torch.__version__, 'numpy': np.__version__,
            'torch_threads': torch.get_num_threads(),
            'torch_interop_threads': torch.get_num_interop_threads(),
            'cpu_affinity': sorted(os.sched_getaffinity(0)),
            'environment': {key: os.environ.get(key) for key in
                            (*env, 'OMP_WAIT_POLICY', 'GOMP_SPINCOUNT', 'LD_LIBRARY_PATH')},
            'torch_config': torch.__config__.show(),
        }
        with Image.open(helper.IMAGE) as image:
            receipt['image']['raw_wh'] = list(image.size)
            receipt['image']['exif_orientation'] = image.getexif().get(274)
            if receipt['image']['exif_orientation'] not in (None, 1):
                raise ValueError('Unexpected EXIF orientation')
        load_started = time.perf_counter()
        state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        if isinstance(state, Mapping):
            for key in ('state_dict', 'model'):
                if key in state and isinstance(state[key], Mapping):
                    state = state[key]
                    break
        if not isinstance(state, Mapping) or 'aggregator.camera_token' not in state:
            raise ValueError('Not an original VGGT state dictionary')
        with torch.device('meta'):
            dino = DinoVisionTransformer(
                img_size=518, patch_size=14, in_chans=3, embed_dim=1024,
                depth=1, num_heads=16, mlp_ratio=4., qkv_bias=True,
                ffn_bias=True, proj_bias=True, drop_path_rate=0., init_values=1.,
                block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention),
                ffn_layer='mlp', block_chunks=0, num_register_tokens=4,
                interpolate_antialias=True, interpolate_offset=0., qk_norm=False)
        selected = {key: state[helper.PREFIX + key] for key in sorted(dino.state_dict())}
        loaded = {}
        for key, value in selected.items():
            if value.device.type != 'cpu' or value.dtype != torch.float32:
                raise ValueError(f'Original CPU FP32 weight required: {key}')
            loaded[helper.PREFIX + key] = helper.array_info(np.ascontiguousarray(value.numpy()))
        dino.load_state_dict(selected, strict=True, assign=True)
        del selected, state
        if any(value.is_meta for value in (*dino.parameters(), *dino.buffers())):
            raise ValueError('Strict assignment left meta tensors')
        dino.eval()
        block = dino.blocks[0]
        receipt['checkpoint_load_seconds'] = time.perf_counter() - load_started
        receipt['loaded_state_tensors'] = loaded
        receipt['loaded_state_bytes'] = sum(item['bytes'] for item in loaded.values())
        if (block.norm1.eps, block.norm2.eps, block.mlp.act.approximate) != (1e-6, 1e-6, 'none'):
            raise ValueError('Original normalization/GELU contract mismatch')
        receipt['architecture'] = {
            'embed_dim': 1024, 'num_heads': 16, 'head_dim': 64, 'patch_size': 14,
            'mlp_hidden_dim': 4096, 'norm1_eps': block.norm1.eps, 'norm2_eps': block.norm2.eps,
            'gelu_approximate': 'none', 'layerscale_initial_value': 1.,
            'qk_norm': False, 'rope': False, 'dropout': 0., 'drop_path': 0., 'eval': True,
            'num_register_tokens': 4, 'cls_tokens': 1, 'patch_tokens': 32, 'sequence_length': 37,
            'attention': 'original torch.nn.functional.scaled_dot_product_attention',
            'position_included_in_input': True, 'interpolate_antialias': True,
            'interpolate_offset': 0., 'constructor_image_size': 518,
            'original_position_weights_shape': list(dino.pos_embed.shape),
        }
        if tuple(dino.pos_embed.shape) != (1, 1370, 1024):
            raise ValueError('Original 518 positional weights were not preserved')
        mean, std = helper.normalization_constants()
        timer = time.perf_counter()
        images = load_and_preprocess_images([str(helper.IMAGE)], image_size=112, mode='crop')
        if tuple(images.shape) != (1, 3, 56, 112) or images.dtype != torch.float32:
            raise ValueError(f'Unexpected crop112 image: {tuple(images.shape)} {images.dtype}')
        image5 = images.unsqueeze(0)
        mean5 = torch.FloatTensor(mean).view(1, 1, 3, 1, 1)
        std5 = torch.FloatTensor(std).view(1, 1, 3, 1, 1)
        normalized = ((image5 - mean5) / std5).view(1, 3, 56, 112)
        receipt['preprocess_seconds'] = time.perf_counter() - timer
        receipt['preprocessing'] = {'mode': 'crop', 'image_size': 112,
                                    'normalized_shape': list(normalized.shape), 'mean': mean, 'std': std,
                                    'source': 'original load_and_preprocess_images and aggregator normalization'}
        with torch.no_grad():
            timer = time.perf_counter()
            tokens = dino.prepare_tokens_with_masks(normalized, masks=None)
            receipt['token_preparation_seconds'] = time.perf_counter() - timer
            if tuple(tokens.shape) != SHAPE or tokens.dtype != torch.float32:
                raise ValueError(f'Unexpected original block0 input: {tuple(tokens.shape)} {tokens.dtype}')
            times, exact = [], []
            for index in range(3):
                timer = time.perf_counter()
                actual = block(tokens)
                times.append(time.perf_counter() - timer)
                if index == 0:
                    reference = actual
                exact.append(bool(torch.equal(actual, reference)))
                if not torch.isfinite(actual).all() or not exact[-1]:
                    raise ValueError('CPU reference repeated output mismatch/nonfinite')
        arrays = {'input.npy': np.ascontiguousarray(tokens.numpy()),
                  'output.npy': np.ascontiguousarray(reference.numpy())}
        for filename, array in arrays.items():
            if array.shape != SHAPE:
                raise ValueError('Original input/output shape mismatch')
            metadata = helper.array_info(array)
            path = output / filename
            with path.open('xb') as stream:
                np.save(stream, array, allow_pickle=False)
            receipt['artifacts'][filename] = {
                'bytes': path.stat().st_size, 'sha256': sha(path), 'shape': metadata['shape'],
                'dtype': metadata['dtype'], 'finite': True, 'tensor_sha256': metadata['sha256']}
        state_arrays = {key: np.ascontiguousarray(value.detach().numpy())
                        for key, value in block.state_dict().items()}
        if set(state_arrays) != helper.BLOCK_KEYS:
            raise ValueError('Original block state must contain exactly the 14 pinned keys')
        tensor_info = {key: helper.array_info(value) for key, value in state_arrays.items()}
        state_path = output / 'block-state.npz'
        with state_path.open('xb') as stream:
            np.savez(stream, **state_arrays)
        receipt['artifacts']['block-state.npz'] = {
            'bytes': state_path.stat().st_size, 'sha256': sha(state_path), 'tensors': tensor_info,
            'checkpoint_key_prefix': helper.PREFIX + 'blocks.0.'}
        receipt['cpu_reference'] = {
            'seconds': times, 'first_seconds': times[0], 'subsequent_seconds': times[1:],
            'mean_seconds': sum(times) / 3., 'repeats': 3, 'capture_call_excluded': False,
            'all_elementwise_equal': all(exact), 'finite': True, 'no_grad': True, 'eval': True,
            'dtype': 'float32', 'device': 'cpu'}
        receipt['source_hashes_after'] = {name: sha(SOURCE / name) for name in helper.SOURCE_PINS}
        receipt['source_unchanged'] = receipt['source_hashes_after'] == before
        receipt['image_unchanged'] = sha(helper.IMAGE) == helper.IMAGE_SHA
        stat_after = checkpoint.stat()
        receipt['checkpoint_stat_unchanged'] = (
            stat_after.st_size, stat_after.st_mtime_ns, stat_after.st_ino) == (
                checkpoint_stat.st_size, checkpoint_stat.st_mtime_ns, checkpoint_stat.st_ino)
        receipt['frozen_helper_unchanged'] = sha(helper_path) == HELPER_SHA
        if not all(receipt[key] for key in ('source_unchanged', 'image_unchanged',
                                           'checkpoint_stat_unchanged', 'frozen_helper_unchanged')):
            raise ValueError('A source/input/helper changed during capture')
        receipt['status'] = 'completed'
    except Exception as exc:
        receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        if original_affinity is not None:
            try:
                os.sched_setaffinity(0, original_affinity)
                receipt['affinity_restored'] = set(os.sched_getaffinity(0)) == original_affinity
                if not receipt['affinity_restored']:
                    raise RuntimeError('Affinity restoration mismatch')
            except Exception as exc:
                receipt['cleanup_errors'].append(f'{type(exc).__name__}: {exc}')
                receipt['status'] = 'failed'
        receipt['total_seconds_including_hash_load_IO'] = time.monotonic() - started
        receipt['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.
        with (output / 'capture.json').open('x') as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write('\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BASE / 'experiments/vggt-realtime-1hz/capture112-r1')
    args = parser.parse_args()
    if platform.machine() != 'aarch64' or platform.system() != 'Linux' or Path.cwd().resolve() != BASE:
        parser.error('Run only on the RK3588S board from /mnt/vggt-data/vggt-slam2')
    if Path(__file__).resolve().parents[3] != SOURCE:
        parser.error('Use the deployed script under /mnt/vggt-data/vggt-slam2/src')
    output = args.output.resolve()
    if not output.is_relative_to(BASE / 'experiments') or output == BASE / 'experiments':
        parser.error('--output must be strictly below board experiments')
    if os.path.lexists(args.output):
        parser.error('Refusing an existing output directory or symlink')
    output.mkdir(parents=True, exist_ok=False)
    receipt = capture(output)
    print(json.dumps({'status': receipt['status'], 'receipt': str(output / 'capture.json'),
                      'sha256': sha(output / 'capture.json'), 'error': receipt.get('error')}))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
