#!/usr/bin/env python3
"""Capture every original VGGT DINO block on RK3588; no ORT/SDK/compiler.

The original checkpoint's complete internal DINO is loaded strictly. Two real
images use the original preprocessing and aggregator normalization. Capture
one original batch-two forward plus two original single-frame forwards,
including all block inputs/outputs and final-normalized patch features.
"""
from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import original_dino_graph as G


def require_board():
    if platform.machine().lower() not in ('aarch64', 'arm64'):
        raise RuntimeError('Original actual capture is restricted to RK3588 aarch64')
    identity = Path('/proc/device-tree/compatible').read_bytes().replace(b'\0', b',').decode()
    if 'rockchip,rk3588' not in identity:
        raise RuntimeError('RK3588 device-tree identity required')
    return identity


def normalization_constants(path):
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('_RESNET_MEAN', '_RESNET_STD'):
                    result[target.id] = ast.literal_eval(node.value)
    if set(result) != {'_RESNET_MEAN', '_RESNET_STD'}:
        raise ValueError('Original aggregator normalization constants required')
    return result['_RESNET_MEAN'], result['_RESNET_STD']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--images', type=Path, nargs=2, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    board = require_board()
    repo = args.repo.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema='v54_original_VGGT_24_DINO_block_capture', status='failed', board=board,
        script_sha256=G.sha(__file__), graph_builder_source_sha256=G.sha(G.__file__),
        checkpoint=dict(G.CHECKPOINT), image_shape=[2, 3, 84, 112], single_image_shape=list(G.IMAGE_SHAPE),
        depth=24, embed_dim=1024, heads=16, tokens=53, patch_size=14, num_register_tokens=4,
        actual_inference_on_board_only=True, original_model_only=True, full_VGGT_forward_executed=False,
        ORT_or_SDK_or_compiler_executed=False, candidate_accepted=False,
        full_model_quality_or_realtime_validated=False, blocks=[], references={}, cleanup_errors=[],
        original_batch_forward_attempts=0, original_single_forward_attempts=0)
    allowed, hooks, mode = None, [], 'batch2'
    before_env = {name: os.environ.get(name) for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}
    try:
        if checkpoint.stat().st_size != G.CHECKPOINT['bytes'] or G.sha(checkpoint) != G.CHECKPOINT['sha256']:
            raise ValueError('Pinned complete original VGGT checkpoint required before deserialization')
        checkpoint_stat = checkpoint.stat()
        report['source_images'] = [dict(file=str(path.resolve(strict=True)), bytes=path.stat().st_size,
                                       sha256=G.sha(path)) for path in args.images]
        originals = ['third_party/vggt/vggt/layers/' + name for name in (
            '__init__.py', 'vision_transformer.py', 'block.py', 'attention.py', 'patch_embed.py',
            'layer_scale.py', 'mlp.py', 'drop_path.py', 'swiglu_ffn.py')]
        originals += ['third_party/vggt/vggt/models/aggregator.py', 'third_party/vggt/vggt/utils/load_fn.py']
        report['source_hashes'] = {name: G.sha(repo / name) for name in originals}
        report['capture_sources'] = {'capture_original_dino_board.py': G.sha(__file__),
                                     'original_dino_graph.py': G.sha(G.__file__)}
        allowed = set(os.sched_getaffinity(0))
        if not {4, 5, 6, 7} <= allowed:
            raise ValueError('CPU 4,5,6,7 required')
        os.sched_setaffinity(0, {4, 5, 6, 7})
        os.environ.update({name: '4' for name in before_env})
        sys.path[:0] = [str(repo), str(repo / 'third_party/vggt')]
        import numpy as np
        import torch
        from vggt.layers.vision_transformer import vit_large
        from vggt.utils.load_fn import load_and_preprocess_images
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        report['runtime'] = dict(torch=torch.__version__, numpy=np.__version__,
                               python=sys.version, cpu_affinity=sorted(os.sched_getaffinity(0)))
        state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        if isinstance(state, Mapping):
            for name in ('state_dict', 'model'):
                if name in state and isinstance(state[name], Mapping):
                    state = state[name]
                    break
        if not isinstance(state, Mapping) or 'aggregator.camera_token' not in state:
            raise ValueError('Original VGGT state dictionary required')
        with torch.device('meta'):
            dino = vit_large(img_size=518, patch_size=14, num_register_tokens=4,
                             interpolate_antialias=True, interpolate_offset=0., block_chunks=0, init_values=1.)
        expected = set(dino.state_dict())
        prefix = 'aggregator.patch_embed.'
        actual = {name[len(prefix):]: tensor for name, tensor in state.items() if name.startswith(prefix)}
        if set(actual) != expected or any(t.device.type != 'cpu' or t.dtype != torch.float32 for t in actual.values()):
            raise ValueError('Complete original DINO checkpoint tensor set/FP32 dtype required')
        dino.load_state_dict(actual, strict=True, assign=True)
        del state, actual
        dino.eval().requires_grad_(False)
        if (dino.chunked_blocks or len(dino.blocks) != 24 or dino.n_blocks != 24
                or dino.embed_dim != 1024 or dino.num_heads != 16 or dino.num_register_tokens != 4
                or dino.patch_size != 14 or any(t.is_meta for t in (*dino.parameters(), *dino.buffers()))):
            raise ValueError('Complete original 24-layer DINO architecture required')
        outside = {name: tensor for name, tensor in dino.named_parameters() if not name.startswith('blocks.')}
        report['non_block_parameter_raw_sha256'] = {name: G.raw_sha(t.detach().numpy()) for name, t in outside.items()}
        images = load_and_preprocess_images([str(path) for path in args.images], image_size=112, mode='crop')
        if images.shape != (2, 3, 84, 112) or images.dtype != torch.float32:
            raise ValueError('Original TUM pair preprocessing must produce [2,3,84,112] FP32')
        mean, std = normalization_constants(repo / 'third_party/vggt/vggt/models/aggregator.py')
        normalized = ((images.unsqueeze(0) - torch.FloatTensor(mean).view(1, 1, 3, 1, 1))
                      / torch.FloatTensor(std).view(1, 1, 3, 1, 1)).view(2, 3, 84, 112)
        report['images'] = G.save_array(output, 'images.npy', images.numpy())
        report['normalized_images'] = G.save_array(output, 'normalized-images.npy', normalized.numpy())
        report['normalization'] = dict(mean=mean, std=std, original_batch_view=[1, 2, 3, 84, 112])
        seen = {name: [] for name in ('batch2', 'single0', 'single1')}
        def before(index):
            def hook(module, args, kwargs):
                x = args[0] if args else kwargs.get('x')
                n = 2 if mode == 'batch2' else 1
                if (not isinstance(x, torch.Tensor) or tuple(x.shape) != (n, 53, 1024)
                        or x.dtype != torch.float32 or x.device.type != 'cpu'
                        or kwargs.get('pos') is not None or kwargs.get('viz_attention', False)):
                    raise ValueError('Original DINO token shape/type/no-RoPE contract required')
                if index != len(seen[mode]):
                    raise ValueError('Original all-24-layer order changed')
                seen[mode].append(index)
                report['blocks'][index]['references'][mode] = dict(
                    input=G.save_array(output, f'dino-{index:02d}-{mode}-input.npy', x.detach().numpy()))
            return hook
        def after(index):
            def hook(module, args, result):
                n = 2 if mode == 'batch2' else 1
                if not isinstance(result, torch.Tensor) or tuple(result.shape) != (n, 53, 1024):
                    raise ValueError('Original DINO single tensor output required')
                report['blocks'][index]['references'][mode]['output'] = G.save_array(
                    output, f'dino-{index:02d}-{mode}-output.npy', result.detach().numpy())
            return hook
        for index, block in enumerate(dino.blocks):
            params = dict(block.named_parameters())
            if (set(params) != set(G.PARAMETER_NAMES) or block.norm1.eps != 1e-6 or block.norm2.eps != 1e-6
                    or block.attn.num_heads != 16 or block.attn.head_dim != 64 or block.attn.rope is not None
                    or not isinstance(block.attn.q_norm, torch.nn.Identity)
                    or not isinstance(block.attn.k_norm, torch.nn.Identity)
                    or block.mlp.act.approximate != 'none' or block.attn.attn_drop.p != 0
                    or block.attn.proj_drop.p != 0 or block.sample_drop_ratio != 0):
                raise ValueError('Original 14-parameter DINO block semantics required')
            weights = {name: np.ascontiguousarray(tensor.detach().numpy()) for name, tensor in params.items()}
            path = output / f'dino-{index:02d}-weights.npz'
            with path.open('xb') as stream:
                np.savez(stream, **weights)
            report['blocks'].append(dict(role='dino', index=index, heads=16,
                epsilons={'norm1': 1e-6, 'norm2': 1e-6}, gelu_approximate='none', qk_norm=False, rope=None,
                original_batch_shape=[2, 53, 1024], single_frame_shape=list(G.SHAPE),
                weights=dict(file=path.name, bytes=path.stat().st_size, sha256=G.sha(path)),
                parameters={name: dict(shape=list(a.shape), dtype=str(a.dtype), tensor_bytes=a.nbytes,
                                       raw_sha256=G.raw_sha(a)) for name, a in weights.items()}, references={}))
            del weights
            hooks += [block.register_forward_pre_hook(before(index), with_kwargs=True),
                      block.register_forward_hook(after(index))]
        begin = time.perf_counter()
        with torch.inference_mode():
            for current, batch in [('batch2', normalized), ('single0', normalized[:1]), ('single1', normalized[1:])]:
                mode = current
                report['original_batch_forward_attempts' if mode == 'batch2' else 'original_single_forward_attempts'] += 1
                result = dino.forward_features(batch)
                if seen[mode] != list(range(24)):
                    raise ValueError('Every original DINO layer must execute exactly once per forward')
                report['references'][mode] = {name: G.save_array(output, f'{mode}-{name}.npy', result[name].numpy())
                    for name in ('x_norm_clstoken', 'x_norm_regtokens', 'x_norm_patchtokens', 'x_prenorm')}
                print(f'Captured complete original DINO {mode}: 24 layers', flush=True)
        report['capture_forward_seconds_including_hooks_and_disk_IO'] = time.perf_counter() - begin
        report['source_hashes_after'] = {name: G.sha(repo / name) for name in originals}
        report['source_and_inputs_unchanged'] = (report['source_hashes_after'] == report['source_hashes']
            and all(G.sha(item['file']) == item['sha256'] for item in report['source_images']))
        report['checkpoint_stat_unchanged'] = checkpoint.stat() == checkpoint_stat
        if not report['source_and_inputs_unchanged'] or not report['checkpoint_stat_unchanged']:
            raise ValueError('Original checkpoint/source/real image changed during capture')
        report['status'] = 'completed_original_FP32_24_DINO_block_capture'
    except BaseException as exc:
        report.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        for hook in hooks:
            try:
                hook.remove()
            except BaseException as exc:
                report['cleanup_errors'].append(str(exc))
        report['hooks_removed'] = not report['cleanup_errors']
        if allowed is not None:
            try:
                os.sched_setaffinity(0, allowed)
                report['affinity_restored'] = set(os.sched_getaffinity(0)) == allowed
            except BaseException as exc:
                report['cleanup_errors'].append(str(exc))
        for name, old in before_env.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old
        report['environment_restored'] = {name: os.environ.get(name) for name in before_env} == before_env
        if report['cleanup_errors'] or report.get('affinity_restored') is not True:
            report['status'] = 'failed'
        with (output / 'capture.json').open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(dict(status=report['status'], error=report.get('error'),
                          capture_sha256=G.sha(output / 'capture.json'))), flush=True)
    return 0 if report['status'] == 'completed_original_FP32_24_DINO_block_capture' else 1


if __name__ == '__main__':
    raise SystemExit(main())
