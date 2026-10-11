#!/usr/bin/env python3
"""One original board FP32 forward; private DPT state/tokens/logits capture."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import depth_contract as C


def save_npz(directory, filename, arrays):
    for value in arrays.values():
        if value.dtype != np.float32 or not value.flags.c_contiguous or not np.isfinite(value).all():
            raise ValueError('Private captured tensors must be contiguous finite FP32')
    path = directory / filename
    with path.open('xb') as stream:
        np.savez(stream, **arrays)
    return {'file': filename, 'bytes': path.stat().st_size, 'sha256': C.sha256(path),
            'tensors': {name: C.array_info(value) for name, value in arrays.items()}}


def parameter_identity(model):
    return [(name, id(value), value._version, str(value.device), str(value.dtype), tuple(value.shape))
            for name, value in model.named_parameters()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--images', type=Path, nargs=2, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-threads', type=int, choices=(2, 4), default=4)
    args = parser.parse_args()
    compatible = Path('/proc/device-tree/compatible').read_bytes().replace(b'\0', b',').decode()
    if sys.platform != 'linux' or platform.machine().lower() not in ('aarch64', 'arm64') or 'rockchip,rk3588' not in compatible:
        parser.error('Original model capture is restricted to Linux RK3588/S')
    repo = args.repo.resolve(strict=True)
    for relative, expected in C.SOURCE_HASHES.items():
        if C.sha256(repo / relative) != expected:
            raise ValueError('Fixed original depth-head source changed: '+relative)
    original_affinity = os.sched_getaffinity(0)
    cpus = {6, 7} if args.model_threads == 2 else {4, 5, 6, 7}
    if not cpus <= original_affinity:
        raise ValueError('Selected original A76 cores unavailable')
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'v54_original_VGGT_depth_capture', 'status': 'failed',
        'script_sha256': C.sha256(__file__), 'contract_source_sha256': C.sha256(C.__file__),
        'checkpoint': dict(C.CHECKPOINT), 'source_hashes': dict(C.SOURCE_HASHES),
        'board': compatible, 'image_shape': [1, 2, 3, 84, 112], 'original_model_only': True,
        'actual_inference_on_board_only': True, 'ORT_or_SDK_or_compiler_executed': False,
        'full_model_forward_attempts': 0, 'full_model_forward_completed': False,
        'original_depth_head_calls': 0, 'original_raw_logits_calls': 0,
        'candidate_accepted': False, 'full_model_quality_or_realtime_validated': False,
        'model_threads': args.model_threads, 'cleanup_errors': [], 'artifacts': {}}
    hooks, model = [], None
    try:
        os.sched_setaffinity(0, cpus)
        report['cpu_affinity'] = sorted(os.sched_getaffinity(0))
        sys.path[:0] = [str(repo), str(repo / 'third_party/vggt')]
        import torch
        from vggt_slam.runtime import load_model
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.heads import utils
        torch.set_num_threads(args.model_threads)
        torch.set_num_interop_threads(1)
        model = load_model(args.checkpoint, device='cpu', image_size=112, prune_unused_heads=True, offline=True)
        head = model.depth_head
        report['architecture'] = C.head_contract(head)
        if (model.depth_frames_chunk_size != 1 or model.aggregator.patch_start_idx != 5
                or any(module.training for module in model.modules())
                or any(value.device.type != 'cpu' or value.dtype != torch.float32 for value in model.parameters())):
            raise ValueError('Unmodified original CPU FP32 model with one-frame DPT chunks required')
        identity_before = parameter_identity(model)
        state = {name: np.ascontiguousarray(value.detach().numpy()) for name, value in head.named_parameters()}
        C.verify_arrays(state, C.PARAMETER_SHAPES)
        report['artifacts']['state'] = save_npz(directory, 'original-depth-state.npz', state)
        del state
        images = load_and_preprocess_images([str(path) for path in args.images], image_size=112)
        if tuple(images.shape) != (2, 3, 84, 112) or images.dtype != torch.float32 or not torch.isfinite(images).all():
            raise ValueError('Original image preprocessing must produce [2,3,84,112] FP32')
        report['source_images'] = [{'file': str(path.resolve(strict=True)), 'bytes': path.stat().st_size,
                                    'sha256': C.sha256(path)} for path in args.images]
        report['artifacts']['images'] = save_npz(directory, 'images.npz', {'images': images.numpy()})
        for module_name in ('vggt_slam.runtime', 'vggt_slam.checkpoint_identity', 'vggt.models.vggt',
                            'vggt.models.aggregator', 'vggt.utils.load_fn', 'vggt.heads.dpt_head',
                            'vggt.heads.head_act', 'vggt.heads.utils'):
            path = Path(sys.modules[module_name].__file__).resolve(strict=True)
            if not path.is_relative_to(repo):
                raise ValueError('Original loaded source escapes the production repo')
            report['source_hashes'][str(path.relative_to(repo))] = C.sha256(path)
        token_arrays, raw_logits = {}, []
        def depth_before(module, args, kwargs):
            if report['original_depth_head_calls']:
                raise ValueError('Exactly one complete original depth-head call required')
            tokens = args[0]
            input_images = kwargs.get('images', args[1] if len(args) > 1 else None)
            patch_start = kwargs.get('patch_start_idx', args[2] if len(args) > 2 else None)
            if (patch_start != 5 or tuple(input_images.shape) != (1,2,3,84,112)
                    or kwargs.get('frames_chunk_size') != 1 or len(tokens) != 24):
                raise ValueError('Original full-pair token/chunk/patch order differs')
            for layer, name in zip(C.LAYERS, C.INPUT_NAMES):
                value = tokens[layer]
                if tuple(value.shape) != (1,2,53,2048) or value.dtype != torch.float32:
                    raise ValueError('Original retained DPT intermediate layout differs')
                token_arrays[name] = np.array(value.detach().numpy(), copy=True, order='C')
            report['original_depth_head_calls'] += 1
        def logits_after(module, args, output):
            if tuple(output.shape) != C.OUTPUT_SHAPE or output.dtype != torch.float32 or len(raw_logits) >= 2:
                raise ValueError('Original per-frame raw-logit output differs')
            raw_logits.append(np.array(output.detach().numpy(), copy=True, order='C'))
            report['original_raw_logits_calls'] += 1
        hooks.append(head.register_forward_pre_hook(depth_before, with_kwargs=True))
        hooks.append(head.scratch.output_conv2.register_forward_hook(logits_after))
        report['full_model_forward_attempts'] = 1
        print('Capturing one complete original FP32 forward with unchanged DPT chunks', flush=True)
        begin = time.perf_counter()
        with torch.inference_mode():
            prediction = model(images, query_points=None, compute_similarity=False)
        report['hooked_forward_seconds_including_capture_copies'] = time.perf_counter()-begin
        report['full_model_forward_completed'] = True
        if report['original_depth_head_calls'] != 1 or len(raw_logits) != 2:
            raise ValueError('Both original DPT frame chunks must complete')
        report['artifacts']['tokens'] = save_npz(directory, 'original-retained-tokens.npz', token_arrays)
        references = {'raw_logits': np.concatenate(raw_logits, axis=0)}
        for name, shape in (('depth', (1,2,84,112,1)), ('depth_conf', (1,2,84,112))):
            value = prediction[name]
            if tuple(value.shape) != shape or value.dtype != torch.float32:
                raise ValueError('Original activated depth output differs: '+name)
            references[name] = np.ascontiguousarray(value.detach().numpy())
        report['artifacts']['references'] = save_npz(directory, 'original-depth-references.npz', references)
        constants = {}
        # Execute the exact original source utilities AFTER the untouched full
        # forward. Keep their FP64 sin/cos intermediates and final FP32 cast;
        # these are geometry constants, not approximated NumPy trigonometry.
        with torch.inference_mode():
            for name, (_, channels, height, width) in C.POSITION_SHAPES.items():
                grid = utils.create_uv_grid(width, height, aspect_ratio=112/84,
                                           dtype=torch.float32, device=torch.device('cpu'))
                embedding = utils.position_grid_to_embed(grid, channels) * 0.1
                constants[name] = np.ascontiguousarray(embedding.permute(2,0,1)[None].numpy())
        C.verify_arrays(constants, C.POSITION_SHAPES)
        report['artifacts']['positions'] = save_npz(directory, 'original-position-constants.npz', constants)
        report['position_constant_provenance'] = 'exact original source utilities after full forward; not subtracted from rounded activations'
        report['original_position_utility_calls_after_forward'] = 5
        report['model_parameter_identity_unchanged'] = parameter_identity(model) == identity_before
        if not report['model_parameter_identity_unchanged']:
            raise ValueError('Original parameters changed during capture')
        report['status'] = 'completed_original_FP32_depth_capture'
    except BaseException as exc:
        report.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        for hook in reversed(hooks):
            try:
                hook.remove()
            except BaseException as exc:
                report['cleanup_errors'].append(f'hook removal: {exc}')
        report['hooks_removed'] = not report['cleanup_errors']
        try:
            os.sched_setaffinity(0, original_affinity)
            report['affinity_restored'] = os.sched_getaffinity(0) == original_affinity
        except BaseException as exc:
            report['affinity_restored'] = False
            report['cleanup_errors'].append(f'affinity restore: {exc}')
        if report['cleanup_errors']:
            report['status'] = 'failed'
        path = directory / 'capture.json'
        path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'status': report['status'], 'capture_sha256': C.sha256(path)}), flush=True)
    return 0 if report['status'] == 'completed_original_FP32_depth_capture' else 1


if __name__ == '__main__':
    raise SystemExit(main())
