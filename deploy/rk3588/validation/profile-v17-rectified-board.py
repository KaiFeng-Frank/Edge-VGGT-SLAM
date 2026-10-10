#!/usr/bin/env python3
"""Board-only original VGGT stage timings and three first-block captures.

Use a completed v17 CPU pair receipt. Two full forwards use its exact first
pair: one uninstrumented, then one with timing and tensor-copy hooks. Neither
is an online throughput measurement. Model weights/architecture stay unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PINS = {
    'capture-v17-resolution-pairs-board.py': '3706712dd3a7451fcbce98e079d4018989eca6f64eb98704a81b8f4a9acc03cb',
    'profile-v14-small-vggt-board.py': 'ec69e48e2cb5a953d8e134934af7ebf1f45a88dfa0ec5b2cf5967bb4027e8d22',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4 << 20), b''):
            h.update(part)
    return h.hexdigest()


def dependency(name):
    path = HERE / name
    if sha(path) != PINS[name]:
        raise ValueError(f'Frozen helper changed: {name}')
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def array_record(a):
    import numpy as np
    if a.dtype not in (np.dtype('float32'), np.dtype('int64')) or not a.size or not np.isfinite(a).all():
        raise ValueError('Capture requires finite, nonempty FP32 or int64 arrays')
    if not a.flags.c_contiguous:
        raise ValueError('Capture array must be contiguous')
    return dict(shape=list(a.shape), dtype=str(a.dtype), bytes=a.nbytes,
                sha256=hashlib.sha256(memoryview(a).cast('B')).hexdigest())


class BlockCapture:
    """Detach/copy only; hooks never replace inputs or outputs."""
    def __init__(self, modules):
        self.handles, self.values, self.calls = [], {}, {}
        try:
            for name, module in modules.items():
                self.calls[name] = 0
                self.handles.append(module.register_forward_pre_hook(self.before(name), with_kwargs=True))
                self.handles.append(module.register_forward_hook(self.after(name)))
        except BaseException:
            self.close()
            raise

    def before(self, name):
        def hook(module, args, kwargs):
            if self.calls[name] or name in self.values:
                raise ValueError(f'First block executed more than once: {name}')
            if kwargs.get('viz_attention', False):
                raise ValueError('Attention visualization is not a capture forward')
            x = args[0]
            if x.device.type != 'cpu' or str(x.dtype) != 'torch.float32':
                raise ValueError('Capture input must be original CPU FP32')
            values = {'input': x.detach().clone().contiguous()}
            pos = kwargs.get('pos', args[1] if len(args) > 1 else None)
            if pos is not None:
                if pos.device.type != 'cpu' or str(pos.dtype) != 'torch.int64':
                    raise ValueError('Original RoPE positions must be CPU int64')
                values['pos'] = pos.detach().clone().contiguous()
            self.values[name] = values
        return hook

    def after(self, name):
        def hook(module, args, output):
            if output.device.type != 'cpu' or str(output.dtype) != 'torch.float32':
                raise ValueError('Capture output must be original CPU FP32')
            self.values[name]['output'] = output.detach().clone().contiguous()
            self.calls[name] += 1
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def validate_reference(reference, size):
    if (reference.get('status') != 'completed_fixed_input_diagnostic'
            or reference.get('script_sha256') != PINS['capture-v17-resolution-pairs-board.py']
            or reference.get('image_size') != size
            or reference.get('original_weights') is not True
            or reference.get('precision') != 'float32'
            or reference.get('npu_used') is not False
            or reference.get('ORB_used') is not False):
        raise ValueError('Expected completed original CPU FP32 v17 resolution reference')
    pairs, selected = reference.get('pairs', []), reference.get('selected_inputs', [])
    if not pairs or not selected or pairs[0]['revision'] != 1 or selected[0]['revision'] != 1:
        raise ValueError('Reference must include the first actual schedule pair')
    if pairs[0]['frame_ids'] != selected[0]['frame_ids']:
        raise ValueError('Reference pair identity mismatch')
    return pairs[0], selected[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--reference-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    c = dependency('capture-v17-resolution-pairs-board.py')
    p = dependency('profile-v14-small-vggt-board.py')
    board = p.require_board()
    args.reference, args.output = args.reference.resolve(), args.output.resolve()
    if not args.reference.is_relative_to(c.BASE) or not args.output.is_relative_to(c.BASE / 'experiments'):
        parser.error('Inputs and new experiment output must be on the board data volume')
    if sha(args.reference) != args.reference_sha256:
        parser.error('Reference receipt SHA mismatch')
    reference = json.loads(args.reference.read_text())
    if reference.get('image_size') not in (224, 336):
        parser.error('Only original v17 224/336 references are supported')
    first, selected = validate_reference(reference, reference['image_size'])
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(kind='original_VGGT_rectified_stage_and_block_capture', status='started',
                  script_sha256=sha(__file__), helper_hashes=PINS, board=board,
                  reference_sha256=args.reference_sha256, image_size=reference['image_size'],
                  ORB_used=False, npu_used=False, inference_on_board_only=True,
                  original_weights=True, architecture_changed=False, GT_used=False,
                  real_time_validated=False, accuracy_validated=False,
                  timing_scope='one unhooked full forward; one hooked forward including tensor clones and stage logs; no online pipeline',
                  stages_are_inclusive_and_nested=True, blocks={}, cleanup_errors=[])
    started, hooks, captures, model = time.monotonic(), None, None, None
    allowed = os.sched_getaffinity(0)
    try:
        if not c.CPUS <= allowed:
            raise ValueError('Four big CPU cores are not available')
        os.sched_setaffinity(0, c.CPUS)
        os.environ.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4', MPLBACKEND='Agg')
        sys.path[:0] = [str(c.SOURCE), str(c.SOURCE / 'third_party/vggt')]
        current_sources = {name: sha(c.SOURCE / name) for name in reference['source_hashes']}
        if current_sources != reference['source_hashes']:
            raise ValueError('Original source differs from CPU reference')
        report['source_hashes'] = dict(current_sources)
        for name in ('third_party/vggt/vggt/layers/rope.py', 'third_party/vggt/vggt/layers/mlp.py',
                     'third_party/vggt/vggt/layers/layer_scale.py'):
            report['source_hashes'][name] = sha(c.SOURCE / name)
        report['preprocessing_provenance'] = c.preprocessing_provenance(c.SOURCE / 'vggt_slam/edge_mapping.py')
        checkpoint = Path(reference['checkpoint']['path'])
        checkpoint_stat = checkpoint.stat()
        if checkpoint_stat.st_size != c.CHECKPOINT_SIZE or sha(checkpoint) != c.CHECKPOINT_SHA:
            raise ValueError('Official checkpoint identity mismatch')
        report['checkpoint'] = reference['checkpoint']
        report['checkpoint_hash_warms_filesystem_cache'] = True
        import cv2
        import numpy as np
        import torch
        from vggt_slam.image_rectification import CalibrationRectifier
        from vggt_slam.edge_mapping import prepare_rectified_images
        from vggt_slam.runtime import load_model
        cv2.setNumThreads(1)
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        report['runtime'] = dict(torch=torch.__version__, numpy=np.__version__, cpu_affinity=sorted(os.sched_getaffinity(0)),
                                 threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads())
        calibration = Path(reference['calibration']['path'])
        if sha(calibration) != reference['calibration']['sha256']:
            raise ValueError('Calibration file changed')
        rectifier = CalibrationRectifier(json.loads(calibration.read_text()))
        if rectifier.metadata() != reference['rectification']:
            raise ValueError('Rectification differs from reference')
        for item in selected['images']:
            if sha(item['path']) != item['sha256']:
                raise ValueError('Selected source image changed')
        images = prepare_rectified_images([x['path'] for x in selected['images']], rectifier, reference['image_size'])
        if c.array_info(np.ascontiguousarray(images.numpy())) != first['input_tensor']:
            raise ValueError('Input tensor is not bitwise identical to reference')
        report['input_tensor'], report['selected_pair'] = first['input_tensor'], selected
        model = load_model(checkpoint, device='cpu', image_size=reference['image_size'], prune_unused_heads=True, offline=True)
        if model.training or any(x.dtype != torch.float32 or x.device.type != 'cpu' for x in model.parameters()):
            raise ValueError('Model must remain original CPU FP32 in eval mode')
        report['platform_before'] = p.platform_record()
        p.event('unhooked_forward_begin')
        with torch.inference_mode():
            begin = time.perf_counter()
            prediction = model(images)
            report['unhooked_forward_seconds'] = time.perf_counter() - begin
        unhooked = p.prediction_values(prediction, torch)
        del prediction
        p.event('unhooked_forward_end', seconds=report['unhooked_forward_seconds'])
        old_path = args.reference.parent / first['file']
        if sha(old_path) != first['sha256']:
            raise ValueError('Saved reference prediction changed')
        with np.load(old_path, allow_pickle=False) as saved:
            report['unhooked_vs_prior_capture_array_equal'] = {key: bool(np.array_equal(value.numpy(), saved[key])) for key, value in unhooked.items()}
        if not all(report['unhooked_vs_prior_capture_array_equal'].values()):
            raise ValueError('Original unhooked output differs from saved CPU reference')
        modules = dict(dino0=model.aggregator.patch_embed.blocks[0], frame0=model.aggregator.frame_blocks[0],
                       global0=model.aggregator.global_blocks[0])
        hooks = p.StageHooks(model)
        captures = BlockCapture(modules)
        with torch.inference_mode():
            begin = time.perf_counter()
            prediction = model(images)
            report['hooked_forward_seconds'] = time.perf_counter() - begin
        report['hooked_vs_unhooked'] = p.compare(p.prediction_values(prediction, torch), unhooked)
        del prediction
        if not all(v['array_equal'] for v in report['hooked_vs_unhooked'].values()):
            raise ValueError('Capture hooks changed model output')
        report['stages'], report['stage_summaries'] = hooks.rows, hooks.summaries()
        report['platform_after'] = p.platform_record()
        if any(hooks.stacks.values()) or any(v != 1 for v in captures.calls.values()):
            raise ValueError('Unbalanced hooks or unexpected block invocation count')
        for name, module in modules.items():
            tensors = dict(captures.values[name])
            rope = module.attn.rope
            if rope is not None:
                pos = tensors['pos']
                cos, sin = rope._compute_frequency_components(module.attn.head_dim // 2, int(pos.max()) + 1, pos.device, torch.float32)
                tensors.update(rope_cos=cos, rope_sin=sin)
            arrays = {key: np.ascontiguousarray(value.detach().numpy()) for key, value in tensors.items()}
            weights = {key: np.ascontiguousarray(value.detach().numpy()) for key, value in module.state_dict().items()}
            row = dict(arrays={key: array_record(value) for key, value in arrays.items()},
                       weights={key: array_record(value) for key, value in weights.items()},
                       module_class=type(module).__name__, num_heads=module.attn.num_heads, head_dim=module.attn.head_dim,
                       norm1_eps=module.norm1.eps, norm2_eps=module.norm2.eps,
                       q_norm_eps=getattr(module.attn.q_norm, 'eps', None), k_norm_eps=getattr(module.attn.k_norm, 'eps', None),
                       gelu_approximate=module.mlp.act.approximate,
                       rope_frequency=getattr(rope, 'base_frequency', None),
                       rope_scaling_factor=getattr(rope, 'scaling_factor', None), artifacts={})
            for suffix, values in (('tensors', arrays), ('weights', weights)):
                path = args.output / f'{name}-{suffix}.npz'
                np.savez(path, **values)
                row['artifacts'][suffix] = dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path))
            report['blocks'][name] = row
        if {name: sha(c.SOURCE / name) for name in report['source_hashes']} != report['source_hashes']:
            raise ValueError('Source changed during diagnostic')
        if (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns) != (checkpoint_stat.st_size, checkpoint_stat.st_mtime_ns):
            raise ValueError('Checkpoint changed during diagnostic')
        for item in selected['images']:
            if sha(item['path']) != item['sha256']:
                raise ValueError('Source image changed during diagnostic')
        if sha(args.reference) != args.reference_sha256 or sha(calibration) != reference['calibration']['sha256']:
            raise ValueError('Reference or calibration changed during diagnostic')
        report['status'] = 'completed_original_CPU_profile_and_capture'
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        for handle in (captures, hooks):
            if handle is not None:
                try:
                    handle.close()
                except BaseException as exc:
                    report['cleanup_errors'].append(str(exc))
        os.sched_setaffinity(0, allowed)
        report['hooks_removed'] = (captures is None or not captures.handles) and (hooks is None or not hooks.hooks)
        if report['cleanup_errors']:
            report['status'] = 'failed'
        report['total_seconds'] = time.monotonic() - started
        path = args.output / 'profile.json'
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], receipt=str(path), sha256=sha(path), error=report.get('error'))))
    return 0 if report['status'] == 'completed_original_CPU_profile_and_capture' else 1


if __name__ == '__main__':
    sys.exit(main())
