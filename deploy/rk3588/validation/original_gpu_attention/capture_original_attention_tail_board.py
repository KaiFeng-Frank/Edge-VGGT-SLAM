#!/usr/bin/env python3
"""Board-only capture of original frame0/global0 QKNorm, RoPE and SDPA tails.

Replay two saved real original FP32 block inputs; no complete VGGT forward,
NPU, cache, export or Solver. Capture IO/checks are not inference performance.
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
TAIL_SHA = '4b723e499763bb90565f6e11e16c551664634aaf72da46c7cdc493b84208ef67'
CALIBRATION_SHA = 'a1e6f1b2a89e1a96987befdea4f0a048aaf06066efa9fff72a370ac77d45507c'
CHECKPOINT = dict(bytes=5026874952,
    sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')
RUNTIME_SOURCES = {
    'vggt_slam/runtime.py': 'f36bb5b87fbdbf03984ea3e1466e6ee27865a540ae3e5abd4ee07a45538e23f5',
    'vggt_slam/checkpoint_identity.py': '9fc0b927f6a68ac6ca55194616104febb1c488370593d4be60e8dba5bc336461'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def info(value, shape=None, dtype='float32'):
    import numpy as np
    value = np.ascontiguousarray(value)
    if (str(value.dtype) != dtype or not value.size or not np.isfinite(value).all()
            or (shape is not None and tuple(value.shape) != tuple(shape))):
        raise ValueError('Finite original tensor shape/dtype differs')
    return dict(shape=list(value.shape), dtype=dtype, tensor_bytes=value.nbytes, finite=True,
                raw_sha256=hashlib.sha256(memoryview(value).cast('B')).hexdigest())


def save(root, name, value, shape=None, dtype='float32'):
    import numpy as np
    value = np.ascontiguousarray(value); metadata = info(value, shape, dtype)
    path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream: np.save(stream, value, allow_pickle=False)
    return dict(file=str(path.relative_to(root)), bytes=path.stat().st_size,
                sha256=sha(path), **metadata)


def compare(actual, reference):
    import numpy as np
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise ValueError('Original reference shape/dtype differs')
    a = actual.astype(np.float64); b = reference.astype(np.float64)
    error = np.abs(a - b); tolerance = 1e-5 + 1e-4 * np.abs(b)
    return dict(rtol=1e-4, atol=1e-5, strict_passed=bool(np.all(error <= tolerance)),
                outside_tolerance=int(np.count_nonzero(error > tolerance)),
                byte_equal=bool(actual.tobytes() == reference.tobytes()),
                max_abs_error=float(error.max()),
                relative_l2=float(np.linalg.norm(a-b) / max(np.linalg.norm(b), np.finfo(np.float64).tiny)))


def tensor_state(model):
    return tuple((name, id(t), t._version, t.dtype, t.device, tuple(t.shape), tuple(t.stride()),
                  t.data_ptr(), t.requires_grad) for name, t in
                 list(model.named_parameters()) + list(model.named_buffers()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=BASE / 'src')
    p.add_argument('--checkpoint', type=Path, default=BASE / 'weights/model.pt')
    p.add_argument('--original-tail-dir', type=Path, default=BASE / 'experiments/vggt-v53-capture-r1')
    p.add_argument('--calibration-dir', type=Path, default=BASE / 'experiments/vggt-v56-calibration-r1')
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    board = next((x.read_bytes() for x in (Path('/proc/device-tree/compatible'),
        Path('/sys/firmware/devicetree/base/compatible')) if x.is_file()), b'')
    if sys.platform != 'linux' or platform.machine().lower() not in ('aarch64', 'arm64') or b'rk3588' not in board:
        p.error('Actual original-model capture is Linux RK3588/S board-only')
    repo = args.repo.resolve(strict=True); checkpoint = args.checkpoint.resolve(strict=True)
    tail_dir = args.original_tail_dir.resolve(strict=True); calibration = args.calibration_dir.resolve(strict=True)
    root = args.output_dir.resolve(); root.mkdir(parents=True, exist_ok=False)
    affinity = os.sched_getaffinity(0); old_path = sys.path[:]
    env_names = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
    old_env = {name: os.environ.get(name) for name in env_names}
    hooks, pins = [], {}; model = state = checkpoint_stat = None
    report = dict(schema='v83_original_VGGT_attention_tail_capture', status='failed',
        script_sha256=sha(__file__), checkpoint=CHECKPOINT, source_hashes={}, runtime_source_hashes=RUNTIME_SOURCES,
        original_model_only=True, actual_inference_on_board_only=True, NPU_or_cache_installed=False,
        full_VGGT_forward_executed=False, DINO_or_heads_executed=False,
        ORT_or_SDK_or_compiler_or_Solver_executed=False, capture_time_is_inference_performance=False,
        image_shape=[2, 3, 84, 112], cases=[], parameters={}, original_block_forward_count=0,
        cleanup_errors=[], candidate_accepted=False, full_model_quality_or_realtime_validated=False)
    started = time.perf_counter()

    def write():
        tmp = root / 'capture.json.tmp'; tmp.write_text(json.dumps(report, indent=2) + '\n'); tmp.replace(root / 'capture.json')

    def pin(path, digest, size=None):
        path = Path(path).resolve(strict=True)
        if (size is not None and path.stat().st_size != size) or sha(path) != digest:
            raise ValueError('Frozen source/artifact changed: ' + str(path))
        pins[path] = digest
        return path

    def load_array(directory, item):
        import numpy as np
        path = (directory / item['file']).resolve(strict=True)
        if not path.is_relative_to(directory): raise ValueError('Original array path escaped capture')
        value = np.load(pin(path, item['sha256'], item['bytes']), allow_pickle=False)
        actual = info(value, item['shape'], item['dtype'])
        if actual['tensor_bytes'] != item['tensor_bytes'] or actual['raw_sha256'] != item['raw_sha256']:
            raise ValueError('Saved original array identity differs')
        return np.ascontiguousarray(value)

    try:
        if not {4, 5, 6, 7} <= affinity: raise ValueError('Original cores 4..7 unavailable')
        os.sched_setaffinity(0, {4, 5, 6, 7}); os.environ.update({name: '4' for name in env_names})
        tail_path = pin(tail_dir / 'capture.json', TAIL_SHA)
        calib_path = pin(calibration / 'capture.json', CALIBRATION_SHA)
        tail = json.loads(tail_path.read_text()); cap = json.loads(calib_path.read_text())
        if (tail['status'] != 'completed_original_FP32_full_forward_and_48_block_capture'
                or cap['status'] != 'completed_original_FP32_calibration_capture'
                or tail['checkpoint'] != CHECKPOINT or cap['checkpoint'] != CHECKPOINT
                or tail['cleanup_errors'] or cap['cleanup_errors'] or cap['NPU_or_cache_installed']
                or not cap['model_state_unchanged'] or not cap['source_and_images_unchanged']):
            raise ValueError('Completed fixed original FP32 captures required')
        source_hashes = {n: d for n, d in cap['source_hashes'].items() if n.startswith('third_party/vggt/vggt/')}
        for name, digest in tail['source_hashes'].items():
            if name.startswith('third_party/vggt/vggt/') and source_hashes.get(name) != digest:
                raise ValueError('Original source lineage differs: ' + name)
        for name, digest in dict(source_hashes, **RUNTIME_SOURCES).items(): pin(repo / name, digest)
        report.update(source_hashes=source_hashes, original_tail_capture_sha256=TAIL_SHA,
                      source_calibration_sha256=CALIBRATION_SHA)
        checkpoint_stat = checkpoint.stat()
        if checkpoint_stat.st_size != CHECKPOINT['bytes']: raise ValueError('Original checkpoint size differs')
        sys.path[:0] = [str(repo), str(repo / 'third_party/vggt')]
        import numpy as np
        import torch
        from vggt_slam.runtime import load_model
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        if torch.__version__ != '2.3.1': raise ValueError('Pinned original board Torch 2.3.1 required')
        report['runtime'] = dict(torch=torch.__version__, numpy=np.__version__, threads=4,
                                interop_threads=1, cpu_affinity=sorted(os.sched_getaffinity(0)))
        tick = time.perf_counter()
        model = load_model(checkpoint, device='cpu', image_size=112, prune_unused_heads=True, offline=True)
        report.update(model_load_seconds=time.perf_counter()-tick, checkpoint_verified_before_deserialization=True)
        agg = model.aggregator
        if (len(agg.frame_blocks) != 24 or len(agg.global_blocks) != 24 or agg.aa_block_size != 1
                or agg.aa_order != ['frame', 'global'] or agg.patch_start_idx != 5
                or any(m.training or m._forward_hooks or m._forward_pre_hooks for m in model.modules())
                or any(t.device.type != 'cpu' or t.dtype != torch.float32 for t in model.parameters())):
            raise ValueError('Unmodified original eval FP32 model required')
        state = tensor_state(model); rows = {}; positions = {}
        for role, block in (('frame', agg.frame_blocks[0]), ('global', agg.global_blocks[0])):
            row = next(r for r in tail['blocks'] if r['role'] == role and r['index'] == 0)
            if set(dict(block.named_parameters())) != set(row['parameters']): raise ValueError('All original block parameters required')
            attn = block.attn
            if (attn.num_heads != 16 or attn.head_dim != 64 or attn.scale != .125
                    or attn.q_norm.eps != 1e-5 or attn.k_norm.eps != 1e-5
                    or tuple(attn.q_norm.normalized_shape) != (64,) or tuple(attn.k_norm.normalized_shape) != (64,)
                    or not attn.fused_attn or attn.attn_drop.p != 0 or attn.rope is None
                    or attn.rope.base_frequency != 100 or attn.rope.scaling_factor != 1.0):
                raise ValueError('Original QKNorm/RoPE/SDPA architecture differs')
            report['parameters'][role] = {}
            for name, tensor in block.named_parameters():
                actual = info(tensor.detach().numpy(), row['parameters'][name]['shape'])
                if actual['raw_sha256'] != row['parameters'][name]['raw_sha256']:
                    raise ValueError('Original parameter differs: ' + role + '.' + name)
                report['parameters'][role][name] = actual
            rows[role] = row; positions[role] = load_array(tail_dir, row['positions'])
        if (positions['frame'].shape != (2, 53, 2) or positions['global'].shape != (1, 106, 2)
                or not np.array_equal(positions['frame'].reshape(1, 106, 2), positions['global'])):
            raise ValueError('Original frame/global position layout differs')
        report.update(original_parameters_verified=36,
            attention_contract=dict(heads=16, head_dim=64, q_norm_eps=1e-5, k_norm_eps=1e-5,
                head_scale=.125, original_SDPA_scale_argument=None, mask=None, dropout_p=0.0, is_causal=False,
                tensor_dtype='float32', softmax_accumulation='Original Torch CPU SDPA implementation; not independently overridden',
                rope_frequency=100, rope_scaling_factor=1.0, rope_component_dimension=32,
                rope_rotate='Within each 32-dimensional vertical/horizontal half: concatenate(-last16,first16)',
                rope_indexed_cos_sin_layout='[B,1,S,32]; y=positions[...,0], x=positions[...,1]',
                attention_tail_layout='[B,16,S,64]', proj_input_layout='[B,S,1024]'))
        group0 = next(g for g in cap['groups']['aggregator'] if g['index'] == 0)
        group1 = next(g for g in cap['groups']['aggregator'] if g['index'] == 1)
        calibration0 = next(s for s in group0['samples'] if s['sample_id'] == 'calib00')
        calibration1 = next(s for s in group1['samples'] if s['sample_id'] == 'calib00')
        source0 = next(s for s in cap['samples'] if s['sample_id'] == 'calib00')
        if source0['indices'] != [0, 48] or calibration0['source_pair_indices'] != [0, 48] or calibration1['source_pair_indices'] != [0, 48]:
            raise ValueError('Fixed real calibration pair differs')
        cases = [('v53', tail_dir, rows['frame']['input'], tail['source_images']),
                 ('v56-calib00', calibration, calibration0['artifact'], source0['images'])]
        for tag, directory, artifact, images in cases:
            for image in images: pin(image['file'], image['sha256'], image['bytes'])
            x = load_array(directory, artifact); info(x, (2, 53, 1024))
            case = dict(tag=tag, source_images=images, original_frame_input=artifact, roles={})
            report['cases'].append(case); write()
            for role, block in (('frame', agg.frame_blocks[0]), ('global', agg.global_blocks[0])):
                if role == 'global': x = x.reshape(1, 106, 1024)
                b, n, _ = x.shape; attn = block.attn; captured = {}; rope_calls = []
                def pre(key):
                    def hook(module, values):
                        if key in captured: raise ValueError('Original hook called twice: ' + key)
                        captured[key] = values[0].detach()
                    return hook
                def post(key):
                    def hook(module, values, output):
                        if key in captured: raise ValueError('Original hook called twice: ' + key)
                        captured[key] = output.detach()
                    return hook
                def rope_hook(module, values, output): rope_calls.append((values[0].detach(), values[1].detach(), output.detach()))
                hooks.extend([attn.qkv.register_forward_pre_hook(pre('qkv_input')),
                    attn.qkv.register_forward_hook(post('qkv_output')),
                    attn.q_norm.register_forward_pre_hook(pre('q_pre_norm')),
                    attn.k_norm.register_forward_pre_hook(pre('k_pre_norm')),
                    attn.q_norm.register_forward_hook(post('q_post_norm')),
                    attn.k_norm.register_forward_hook(post('k_post_norm')),
                    attn.rope.register_forward_hook(rope_hook),
                    attn.proj.register_forward_pre_hook(pre('proj_input')),
                    attn.register_forward_hook(post('attention_output'))])
                input_hash = info(x)['raw_sha256']; tick = time.perf_counter()
                with torch.inference_mode(): output = block(torch.from_numpy(x), pos=torch.from_numpy(positions[role]))
                report['original_block_forward_count'] += 1
                elapsed = time.perf_counter()-tick
                for hook in hooks: hook.remove()
                hooks.clear()
                if len(rope_calls) != 2 or set(captured) != {'qkv_input','qkv_output','q_pre_norm','k_pre_norm','q_post_norm','k_post_norm','proj_input','attention_output'}:
                    raise ValueError('Original attention hook call order/count differs')
                if info(x)['raw_sha256'] != input_hash: raise ValueError('Original block input mutated')
                q, k, v = captured['qkv_output'].reshape(b, n, 3, 16, 64).permute(2,0,3,1,4).unbind(0)
                for expected, actual in ((q,captured['q_pre_norm']), (k,captured['k_pre_norm']),
                    (captured['q_post_norm'],rope_calls[0][0]), (captured['k_post_norm'],rope_calls[1][0])):
                    if not np.array_equal(expected.numpy(), actual.numpy()): raise ValueError('Original QKV/Norm/RoPE ordering differs')
                if any(not np.array_equal(call[1].numpy(), positions[role]) for call in rope_calls):
                    raise ValueError('Actual RoPE positions differ')
                captured.update(v=v, q_post_rope=rope_calls[0][2], k_post_rope=rope_calls[1][2],
                    attention_tail=captured['proj_input'].reshape(b,n,16,64).transpose(1,2), block_output=output.detach())
                role_row = dict(block_index=0, original_input_raw_sha256=input_hash,
                    forward_seconds_including_hooks_not_performance=elapsed, artifacts={})
                case['roles'][role] = role_row
                prefix = tag + '/' + role + '/'
                for name, value in captured.items():
                    shape = (b,n,3072) if name == 'qkv_output' else (b,n,1024) if name in ('qkv_input','proj_input','attention_output','block_output') else (b,16,n,64)
                    role_row['artifacts'][name] = save(root, prefix+name+'.npy', value.numpy(), shape)
                role_row['artifacts']['positions'] = save(root, prefix+'positions.npy', positions[role], (b,n,2), 'int64')
                for key, tensor in (('q_norm_weight',attn.q_norm.weight),('q_norm_bias',attn.q_norm.bias),
                                    ('k_norm_weight',attn.k_norm.weight),('k_norm_bias',attn.k_norm.bias)):
                    role_row['artifacts'][key] = save(root, prefix+key+'.npy', tensor.detach().numpy(), (64,))
                cache_key = (32, int(positions[role].max())+1, torch.device('cpu'), torch.float32)
                cos, sin = attn.rope.frequency_cache[cache_key]
                for name, table in (('cos',cos),('sin',sin)):
                    previous = load_array(tail_dir, tail['rope_'+name])
                    if not np.array_equal(table.numpy(), previous): raise ValueError('Original RoPE source table differs')
                    role_row['artifacts']['rope_'+name+'_table'] = save(root,prefix+'rope_'+name+'_table.npy',table.numpy(),(9,32))
                    for axis, coordinate in (('y',0),('x',1)):
                        indexed = table.numpy()[positions[role][...,coordinate]][:,None,:,:]
                        role_row['artifacts'][name+'_'+axis] = save(root,prefix+name+'_'+axis+'.npy',indexed,(b,1,n,32))
                x = np.ascontiguousarray(output.detach().numpy())
                original_reference = rows[role]['output'] if tag == 'v53' else calibration1['artifact'] if role == 'global' else None
                role_row['original_block_reference'] = original_reference
                if original_reference is not None:
                    reference = load_array(tail_dir if tag == 'v53' else calibration, original_reference)
                    if tag != 'v53': reference = reference.reshape(1,106,1024)
                    role_row['reference_check'] = compare(x,reference)
                    if not role_row['reference_check']['strict_passed']: raise ValueError('Original block fixed FP32 reference gate failed')
                else: role_row['reference_check'] = None
                role_row['attention_tail_is_new_original_CPU_reference_not_preexisting_saved_reference'] = True
                write()
        report['status'] = 'completed_original_FP32_attention_tail_capture'
    except BaseException as exc:
        report['error'] = dict(type=type(exc).__name__, message=str(exc)); raise
    finally:
        for hook in hooks:
            try: hook.remove()
            except Exception as exc: report['cleanup_errors'].append(str(exc))
        report['hooks_removed'] = not model or all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
        report['model_state_unchanged'] = state is not None and tensor_state(model) == state
        report['checkpoint_stat_unchanged'] = checkpoint_stat is not None and checkpoint.stat() == checkpoint_stat
        report['source_and_inputs_unchanged'] = all(sha(path) == digest for path,digest in pins.items())
        sys.path[:] = old_path
        for name,value in old_env.items():
            if value is None: os.environ.pop(name,None)
            else: os.environ[name] = value
        os.sched_setaffinity(0,affinity)
        report['environment_restored'] = all(os.environ.get(n) == old_env[n] for n in env_names) and sys.path == old_path
        report['affinity_restored'] = os.sched_getaffinity(0) == affinity
        if not all(report[k] for k in ('hooks_removed','model_state_unchanged','checkpoint_stat_unchanged',
            'source_and_inputs_unchanged','environment_restored','affinity_restored')) or report['cleanup_errors']:
            report['status'] = 'failed'
        report['seconds_including_identity_load_capture_IO_not_performance'] = time.perf_counter()-started
        write()
        print(json.dumps(dict(status=report['status'], original_block_forward_count=report['original_block_forward_count'],
                              cleanup_errors=report['cleanup_errors']), sort_keys=True))
    if report['status'] != 'completed_original_FP32_attention_tail_capture': raise RuntimeError('Original capture cleanup/identity failed')


if __name__ == '__main__': main()
