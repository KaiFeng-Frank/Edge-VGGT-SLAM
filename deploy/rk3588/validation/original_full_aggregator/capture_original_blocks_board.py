#!/usr/bin/env python3
"""Board-only original FP32 full forward and 48-block capture; no ONNX or SDK."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import original_block_graph as G


def save_array(directory, filename, value):
    array = np.ascontiguousarray(value)
    if array.dtype not in (np.float32, np.int64) or not array.size or not np.isfinite(array).all():
        raise ValueError('Finite FP32/int64 original tensor required')
    path = directory / filename
    with path.open('xb') as stream:
        np.save(stream, array, allow_pickle=False)
    return {'file': filename, 'bytes': path.stat().st_size, 'sha256': G.sha256(path),
        'shape': list(array.shape), 'dtype': str(array.dtype), 'tensor_bytes': array.nbytes,
        'raw_sha256': G.raw_sha256(array)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--images', type=Path, nargs=2, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    compatible = Path('/proc/device-tree/compatible').read_bytes().replace(b'\x00', b',').decode()
    if platform.machine() != 'aarch64' or 'rockchip,rk3588' not in compatible:
        raise RuntimeError('All original-model capture is restricted to RK3588/S board')
    repo = args.repo.resolve(strict=True)
    sys.path.insert(0, str(repo)); sys.path.insert(0, str(repo / 'third_party/vggt'))
    original_affinity = os.sched_getaffinity(0)
    if not {4, 5, 6, 7} <= original_affinity:
        raise RuntimeError('Four original A76 cores unavailable')
    directory = args.output_dir.resolve(); directory.mkdir(parents=True, exist_ok=False)
    hooks = []
    report = {'schema': 'v53_original_VGGT_48_block_capture', 'status': 'failed',
        'script_sha256': G.sha256(__file__), 'builder_source_sha256': G.sha256(G.__file__),
        'board': compatible, 'checkpoint': dict(G.CHECKPOINT), 'image_shape': list(G.IMAGE_SHAPE),
        'frames': 2, 'tokens': 53, 'dim': 1024, 'heads': 16, 'depth': 24,
        'aa_order': ['frame', 'global'], 'aa_block_size': 1, 'patch_start_idx': 5,
        'full_model_forward_attempts': 0, 'full_model_forward_completed': False,
        'actual_inference_on_board_only': True, 'original_model_only': True,
        'ORT_or_SDK_or_compiler_executed': False, 'candidate_accepted': False,
        'full_model_quality_or_realtime_validated': False, 'blocks': [], 'cleanup_errors': []}
    try:
        os.sched_setaffinity(0, {4, 5, 6, 7})
        import torch
        from vggt_slam.runtime import load_model
        from vggt.utils.load_fn import load_and_preprocess_images
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        print('Loading verified original VGGT checkpoint', flush=True)
        model = load_model(args.checkpoint, device='cpu', image_size=112,
                           prune_unused_heads=True, offline=True)
        agg = model.aggregator
        if (len(agg.frame_blocks) != 24 or len(agg.global_blocks) != 24 or agg.depth != 24
                or list(agg.aa_order) != ['frame', 'global'] or agg.aa_block_size != 1
                or agg.patch_start_idx != 5 or agg.patch_size != 14
                or model.camera_head is None or model.depth_head is None
                or any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters())):
            raise ValueError('Complete original CPU FP32 architecture required')
        images = load_and_preprocess_images([str(p) for p in args.images], image_size=112)
        if tuple(images.shape) != G.IMAGE_SHAPE or images.dtype != torch.float32:
            raise ValueError('Original TUM pair must preprocess to [2,3,84,112]')
        report['source_images'] = [{'file': str(p.resolve(strict=True)), 'bytes': p.stat().st_size,
                                    'sha256': G.sha256(p)} for p in args.images]
        report['images'] = save_array(directory, 'images.npy', images.numpy())
        sources = {'capture_original_blocks_board.py': G.sha256(__file__),
                   'original_block_graph.py': G.sha256(G.__file__)}
        for module_name in ('vggt_slam.runtime', 'vggt_slam.checkpoint_identity', 'vggt.models.vggt',
                            'vggt.models.aggregator', 'vggt.layers.block', 'vggt.layers.attention',
                            'vggt.layers.rope', 'vggt.layers.mlp', 'vggt.layers.layer_scale',
                            'vggt.utils.load_fn'):
            path = Path(sys.modules[module_name].__file__).resolve()
            key = str(path.relative_to(repo)) if path.is_relative_to(repo) else module_name
            sources[key] = G.sha256(path)
        report['source_hashes'] = sources
        expected_positions = G.expected_frame_positions()
        positions = {}; records = {}
        def make_pre(role, index):
            def before(module, args, kwargs):
                key = (role, index)
                if key in records:
                    raise ValueError('Each original block must execute exactly once')
                x = args[0]; pos = kwargs.get('pos', args[1] if len(args) > 1 else None)
                shape = G.block_shapes()[role]
                target = expected_positions if role == 'frame' else expected_positions.reshape(1, 106, 2)
                if (tuple(x.shape) != shape or x.dtype != torch.float32 or pos is None
                        or pos.dtype != torch.int64 or not np.array_equal(pos.numpy(), target)):
                    raise ValueError('Original block input/position layout mismatch')
                if role not in positions:
                    positions[role] = save_array(directory, role + '-positions.npy', pos.numpy())
                names = dict(module.named_parameters())
                if set(names) != set(G.PARAMETER_NAMES):
                    raise ValueError('Exact 18 original block parameters required')
                records[key] = {'role': role, 'index': index, 'positions': positions[role],
                    'input': save_array(directory, f'{role}-{index:02d}-input.npy', x.detach().numpy()),
                    'heads': module.attn.num_heads,
                    'epsilons': {'norm1': module.norm1.eps, 'norm2': module.norm2.eps,
                                 'attn.q_norm': module.attn.q_norm.eps, 'attn.k_norm': module.attn.k_norm.eps},
                    'gelu_approximate': module.mlp.act.approximate,
                    'rope_frequency': module.attn.rope.base_frequency,
                    'rope_scaling_factor': module.attn.rope.scaling_factor}
            return before
        def make_post(role, index):
            def after(module, args, output):
                if (not isinstance(output, torch.Tensor) or output.dtype != torch.float32
                        or tuple(output.shape) != G.block_shapes()[role]):
                    raise ValueError('Original block output shape/type mismatch')
                records[(role, index)]['output'] = save_array(
                    directory, f'{role}-{index:02d}-output.npy', output.detach().numpy())
            return after
        for index in range(24):
            for role in ('frame', 'global'):
                block = getattr(agg, role + '_blocks')[index]
                hooks.append(block.register_forward_pre_hook(make_pre(role, index), with_kwargs=True))
                hooks.append(block.register_forward_hook(make_post(role, index)))
        report['full_model_forward_attempts'] = 1
        print('Capturing one complete original FP32 forward, CameraHead and Depth enabled', flush=True)
        started = time.perf_counter()
        with torch.inference_mode():
            raw = model(images, query_points=None, compute_similarity=False)
        report['hooked_forward_seconds_including_capture_IO'] = time.perf_counter() - started
        report['full_model_forward_completed'] = True
        if len(records) != 48 or any('output' not in value for value in records.values()):
            raise ValueError('All 48 original blocks must complete')
        report['references'] = {key: save_array(directory, 'reference-' + key + '.npy', raw[key].detach().numpy())
                                for key in ('pose_enc', 'depth', 'depth_conf')}
        # The actual original forward populated this exact table-cache key.
        table_key = (32, 9, torch.device('cpu'), torch.float32)
        if table_key not in agg.rope.frequency_cache:
            raise ValueError('Original forward did not populate the expected RoPE tables')
        cosine, sine = agg.rope.frequency_cache[table_key]
        report['rope_cos'] = save_array(directory, 'rope-cos.npy', cosine.numpy())
        report['rope_sin'] = save_array(directory, 'rope-sin.npy', sine.numpy())
        for index in range(24):
            for role in ('frame', 'global'):
                block = getattr(agg, role + '_blocks')[index]
                state = {name: np.ascontiguousarray(value.detach().numpy())
                         for name, value in block.named_parameters()}
                path = directory / f'{role}-{index:02d}-weights.npz'
                with path.open('xb') as stream:
                    np.savez(stream, **state)
                row = records[(role, index)]
                row['weights'] = {'file': path.name, 'bytes': path.stat().st_size, 'sha256': G.sha256(path)}
                row['parameters'] = {name: {'shape': list(value.shape), 'dtype': str(value.dtype),
                    'tensor_bytes': value.nbytes, 'raw_sha256': G.raw_sha256(value)} for name, value in state.items()}
                report['blocks'].append(row)
                del state
        report['status'] = 'completed_original_FP32_full_forward_and_48_block_capture'
        print('All 48 original parameter sets and references saved', flush=True)
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        for hook in hooks:
            hook.remove()
        report['hooks_removed'] = True
        os.sched_setaffinity(0, original_affinity)
        report['affinity_restored'] = os.sched_getaffinity(0) == original_affinity
        with (directory / 'capture.json').open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
        print(json.dumps({'status': report['status'], 'capture_sha256': G.sha256(directory / 'capture.json')}), flush=True)


if __name__ == '__main__':
    main()
