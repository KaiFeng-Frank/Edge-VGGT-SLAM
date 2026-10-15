#!/usr/bin/env python3
"""Three CPU controls of the original native518 DINO0, never an FP16 fallback.

1. Original FP32 must reproduce the pinned original checkpoints byte for byte.
2. Independently round original input/14 weights FP32->Half->FP32, then execute
   the unchanged original class in FP32 (storage_rounding_only_FP32).
3. A fresh original class receives Half input/weights and attempts its complete
   CPU forward. Unsupported Half operators are recorded, never replaced.
All observers leave results unchanged. This is a component precision diagnosis;
Torch CPU kernel accumulation precision is not inferred from tensor dtype.
"""
import argparse
import faulthandler
from functools import partial
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
CAPTURE_SHA = '17d77cfe3d8cdb3ff88e15de4a06634683899348b53b24fbadad6eef1584e91d'
CAPTURE_SOURCE_SHA = '587c9bd0c71098b2534e8c73ff72f3c06b862d540142f3a8141a468f234d33d6'
TAP_SHA = '1f89eac69b85ec84ea76f9e8ca0a50ca7d4c5f19620c4159f41eaf6f3661225a'
INPUT_RAW_SHA = '706d7e06efb45e7c90d3fc990e3479a4f566e93cb8c44f42ef57622b037f1efd'
CHECKPOINT = {'bytes': 5026874952, 'sha256': 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'}
SHAPE = (1, 1041, 1024)
POINTS = ('norm1', 'attn.qkv', 'attention', 'attn.proj.input', 'output')
ROUTES = ('original_FP32', 'storage_rounding_only_FP32', 'CPU_float16')
SUCCESS = 'completed_original_DINO0_CPU_precision_controls_not_quality_accepted'
BYTE_LIMIT = 48 * 1024 * 1024


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def identity(path):
    s = Path(path).stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=BASE / 'src')
    p.add_argument('--capture', type=Path, default=BASE / 'experiments/vggt-v100-dino0-intermediates-r1/capture.json')
    p.add_argument('--capture-sha256', default=CAPTURE_SHA)
    p.add_argument('--weights', type=Path, default=BASE / 'experiments/vggt-v99-dino-reference-r1/dino-00-weights.npz')
    p.add_argument('--tap-result', type=Path, default=BASE / 'experiments/vggt-v100-dino-taps-core1-r1/result.json')
    p.add_argument('--tap-result-sha256', default=TAP_SHA)
    p.add_argument('--tap-payload', type=Path, help='Defaults to the pinned call0.npz beside tap-result')
    p.add_argument('--output-dir', type=Path, default=Path('/tmp/vggt-v116-half-control-r1'))
    a = p.parse_args()
    require(platform.machine() == 'aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(), 'RK3588/S board CPU execution only')
    require(a.capture_sha256 == CAPTURE_SHA and a.tap_result_sha256 == TAP_SHA, 'Exact original CPU/NPU diagnostic receipts required')
    faulthandler.enable(all_threads=True)
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    env, affinity, old_path = dict(os.environ), set(os.sched_getaffinity(0)), list(sys.path)
    old_term = signal.getsignal(signal.SIGTERM)
    started, cpu_started = time.perf_counter(), time.process_time()
    pins, hooks, owners, saved_bytes = {}, [], [], 0
    functional = original_sdpa = current = None
    report = dict(schema='v116_original_native518_DINO0_CPU_precision_controls', status='running', script_sha256=sha(__file__),
        checkpoint=CHECKPOINT, capture_sha256=CAPTURE_SHA, tap_result_sha256=TAP_SHA, calls=[], cleanup_errors=[],
        original_class_forward_source_edited=False, observation_hooks_return_None=True, SDPA_observer_forwards_arguments_and_returns_original_result=True,
        FP16_failure_fallback_enabled=False, full_checkpoint_deserialized=False, SDK_ORT_NPU_GPU_executed=False,
        actual_execution_on_board_only=True, full_encoder_or_model_forward_completed=False, performance_or_trajectory_acceptance=False,
        storage_rounding_control_is_FP16_execution=False, payload_byte_limit=BYTE_LIMIT, intra_op_threads=1, inter_op_threads=1,
        timing_scope='Single observed CPU forwards include hook copies, hashes, file IO and any external pauses; not a performance benchmark',
        precision_scope='Half tensor dtype does not prove the internal accumulation dtype of Torch CPU kernels or RKNN numerical equivalence')

    def terminate(number, frame):
        raise InterruptedError('External CPU-only timeout terminated precision control')
    signal.signal(signal.SIGTERM, terminate)

    def pin(path, digest, size=None):
        path = Path(path).resolve(strict=True); before = identity(path)
        require((size is None or before[2] == size) and sha(path) == digest and identity(path) == before, 'Pinned source/payload changed: ' + str(path))
        pins[path] = (digest, before)
        return path

    def progress(stage):
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report.update(stage=stage, elapsed_seconds=time.perf_counter()-started, process_cpu_seconds=time.process_time()-cpu_started,
                      process_user_seconds=usage.ru_utime, process_system_seconds=usage.ru_stime, peak_rss_mib=usage.ru_maxrss/1024, saved_array_bytes=saved_bytes)
        atomic(out / 'result.json', report)
        print(json.dumps(dict(stage=stage, route=current['route'] if current else None, saved_array_bytes=saved_bytes)), flush=True)

    try:
        pin(__file__, report['script_sha256']); capture_path = pin(a.capture, CAPTURE_SHA)
        c = json.loads(capture_path.read_text()); tap_path = pin(a.tap_result, TAP_SHA); t = json.loads(tap_path.read_text())
        require(c['schema'] == 'v100_original_native518_DINO0_intermediate_capture' and c['status'] == 'completed_original_DINO0_intermediates' and
                c['script_sha256'] == CAPTURE_SOURCE_SHA and c['checkpoint'] == CHECKPOINT and c['calls'] == 1 and c['cleanup_errors'] == [] and
                c['hooks_removed'] is True and c['source_and_inputs_unchanged'] is True and c['original_parameter_state_unchanged'] is True and
                c['final_comparison']['byte_equal_v99'] is True, 'Clean unchanged original CPU reference required')
        require(t['schema'] == 'v100_original_native518_DINO0_three_output_diagnostic' and t['status'] == 'completed_diagnostic_baseline_equal' and
                t['capture_sha256'] == CAPTURE_SHA and t['original_input_raw_sha256'] == INPUT_RAW_SHA and t['final_baseline_byte_equal_all_calls'] is True and
                t['resources_balanced'] is True and t['pins_unchanged'] is True and t['cleanup_errors'] == [], 'Actual NPU tap final must have untapped baseline byte parity')
        require({4, 5, 6, 7} <= affinity, 'A76 CPUs4/5/6/7 required')
        os.sched_setaffinity(0, {4, 5, 6, 7})
        os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
        repo = a.repo.resolve(strict=True)
        for name, digest in c['source_hashes'].items():
            pin(repo / name, digest)
        report.update(source_hashes=c['source_hashes'], source_images=c['source_images'], architecture=c['architecture'],
                      original_parameters=c['original_parameters'], original_weights=c['original_weights'], original_input=c['artifacts']['input'])
        sys.path[:0] = [str(repo / 'third_party/vggt'), str(repo)]
        import numpy as np
        import torch
        from vggt.layers.block import NestedTensorBlock
        from vggt.layers.attention import MemEffAttention
        from vggt.layers import block as block_source, attention as attention_source
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        require(torch.__version__ == c['versions']['torch'] and not block_source.XFORMERS_AVAILABLE and not attention_source.XFORMERS_AVAILABLE, 'Original Torch CPU path/version required')
        require(Path(block_source.__file__).resolve() == (repo / 'third_party/vggt/vggt/layers/block.py').resolve() and
                Path(attention_source.__file__).resolve() == (repo / 'third_party/vggt/vggt/layers/attention.py').resolve(), 'Imports must use pinned original classes')
        report['versions'] = dict(torch=torch.__version__, numpy=np.__version__)
        functional = torch.nn.functional; original_sdpa = functional.scaled_dot_product_attention

        def raw(v):
            return hashlib.sha256(memoryview(np.ascontiguousarray(v)).cast('B')).hexdigest()

        def array(row):
            rel = Path(row['file']); require(not rel.is_absolute() and '..' not in rel.parts, 'Capture-relative reference required')
            v = np.load(pin(capture_path.parent / rel, row['sha256'], row['bytes']), allow_pickle=False)
            require(v.dtype == np.float32 and list(v.shape) == row['shape'] and v.nbytes == row['tensor_bytes'] and np.isfinite(v).all() and raw(v) == row['raw_sha256'], 'Reference raw values changed')
            return np.ascontiguousarray(v)

        x = array(c['artifacts']['input']); refs = {n: array(c['artifacts'][n]) for n in POINTS if n != 'attention'}
        refs['attention'] = np.ascontiguousarray(refs['attn.proj.input'].reshape(1, 1041, 16, 64).transpose(0, 2, 1, 3))
        require(x.shape == SHAPE and raw(x) == INPUT_RAW_SHA, 'Original native518 complete1041 token input required')
        row = t['calls'][0]['artifact']; payload = pin(a.tap_payload or tap_path.parent / row['file'], row['sha256'], row['bytes'])
        with np.load(payload, allow_pickle=False) as archive:
            npu = np.array(archive['output'], copy=True, order='C')
        require(npu.shape == SHAPE and npu.dtype == np.float32 and np.isfinite(npu).all() and raw(npu) == row['arrays']['output']['raw_sha256'], 'Actual NPU final output identity changed')
        report['actual_NPU_reference'] = dict(receipt_sha256=TAP_SHA, payload=row, selected_array='output', attribution_supported_by_final_baseline_byte_parity=True)
        wr = c['original_weights']; weight_path = pin(a.weights, wr['sha256'], wr['bytes'])
        with np.load(weight_path, allow_pickle=False) as archive:
            require(len(archive.files) == 14 and set(archive.files) == set(c['original_parameters']), 'All14 original learned parameters required')
            weights = {n: np.array(archive[n], copy=True, order='C') for n in archive.files}
        for name, v in weights.items():
            r = c['original_parameters'][name]
            require(v.dtype == np.float32 and list(v.shape) == r['shape'] and v.nbytes == r['tensor_bytes'] and np.isfinite(v).all() and raw(v) == r['raw_sha256'], 'Original parameter changed: ' + name)

        def compare(v, ref):
            av, rv = v.astype(np.float64), ref.astype(np.float64); d = av-rv
            an, rn = float(np.linalg.norm(av)), float(np.linalg.norm(rv)); bad = np.abs(d) > (1e-5+1e-4*np.abs(rv))
            return dict(relative_l2=float(np.linalg.norm(d))/rn if rn else None, reference_l2=rn, max_abs=float(np.abs(d).max()),
                        cosine=float(np.sum(av*rv)/(an*rn)) if an and rn else None, strict_rtol=1e-4, strict_atol=1e-5,
                        strict_pass=not bool(bad.any()), outside_count=int(bad.sum()), byte_equal=v.dtype == ref.dtype and v.tobytes() == ref.tobytes())

        def state(block):
            return [(n, id(v), v._version, v.data_ptr(), str(v.dtype)) for n, v in block.named_parameters()] + [(n, m.training) for n, m in block.named_modules()]

        def observe(name, tensor):
            nonlocal saved_bytes
            require(name not in current['observations'] and tensor.device.type == 'cpu' and str(tensor.dtype) == current['tensor_dtype'], 'One CPU observation with declared dtype required: ' + name)
            v = np.array(tensor.detach().numpy(), copy=True, order='C')
            expected = (1, 16, 1041, 64) if name == 'attention' else (1, 1041, 3072) if name == 'attn.qkv' else SHAPE
            require(v.shape == expected and np.isfinite(v).all(), 'Complete finite original checkpoint shape required: ' + name)
            r = dict(shape=list(v.shape), dtype=str(v.dtype), tensor_bytes=v.nbytes, raw_sha256=raw(v), finite=True, saved=False)
            if current['route'] != 'original_FP32':
                path = out / (current['route'] + '-' + name.replace('.', '_') + '.npy')
                require(saved_bytes + v.nbytes + (1 << 20) < BYTE_LIMIT, 'Bounded capture payload required')
                with path.open('xb') as f:
                    np.save(f, v, allow_pickle=False)
                r.update(file=path.name, bytes=path.stat().st_size, sha256=sha(path), saved=True); saved_bytes += r['bytes']
            current['observations'][name] = r; current['comparisons'][name] = compare(v, refs[name])
            if name == 'output':
                current['output_vs_actual_NPU'] = compare(v, npu)
                current['residual_update_same_original_input_vs_CPU'] = compare(v.astype(np.float64)-x, refs['output'].astype(np.float64)-x)
                current['residual_update_same_original_input_vs_actual_NPU'] = compare(v.astype(np.float64)-x, npu.astype(np.float64)-x)
            progress(current['route'] + '_captured_' + name)

        def sdpa_observer(*args, **kwargs):
            require(len(args) == 3 and set(kwargs) == {'dropout_p'} and kwargs['dropout_p'] == 0.0 and 'SDPA' not in current, 'Exact original single unmasked noncausal SDPA call required')
            require(all(tuple(v.shape) == (1, 16, 1041, 64) and str(v.dtype) == current['tensor_dtype'] and v.device.type == 'cpu' for v in args), 'Original completeQ/K/V CPU dtype/shape required')
            current['SDPA'] = dict(QKV_shapes=[list(v.shape) for v in args], QKV_strides=[list(v.stride()) for v in args],
                                   tensor_dtype=current['tensor_dtype'], kwargs=kwargs, mask=None, is_causal=False, default_scale=0.125,
                                   arguments_forwarded_unchanged=True, original_return_preserved=True)
            result = original_sdpa(*args, **kwargs)
            observe('attention', result)
            return result

        report['actual_NPU_vs_original_CPU'] = compare(npu, refs['output'])
        report['actual_NPU_residual_update_vs_original_CPU'] = compare(npu.astype(np.float64)-x, refs['output'].astype(np.float64)-x)
        report['reference_attention_layout'] = 'Original attn.proj.input reshape[1,1041,16,64] transpose[0,2,1,3]; no arithmetic'
        for route in ROUTES:
            dtype = torch.float16 if route == 'CPU_float16' else torch.float32
            current = dict(route=route, tensor_dtype=str(dtype), forward_attempts=0, forward_completed=False, observations={}, comparisons={})
            report['calls'].append(current)
            with torch.device('meta'):
                block = NestedTensorBlock(dim=1024, num_heads=16, mlp_ratio=4, qkv_bias=True, proj_bias=True, ffn_bias=True, init_values=1.,
                    qk_norm=False, rope=None, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), attn_class=MemEffAttention, drop=0., attn_drop=0., drop_path=0.)
            def conversion(v):
                tensor = torch.from_numpy(v)
                if route != 'original_FP32':
                    tensor = tensor.to(torch.float16)
                return tensor.to(torch.float32) if route == 'storage_rounding_only_FP32' else tensor
            assigned = {n: conversion(v) for n, v in weights.items()}; tx = conversion(x)
            block.load_state_dict(assigned, strict=True, assign=True); block.eval().requires_grad_(False)
            require(type(block) is NestedTensorBlock and type(block.attn) is MemEffAttention and block.attn.fused_attn and
                    block.norm1.eps == block.norm2.eps == 1e-6 and block.attn.num_heads == 16 and block.attn.head_dim == 64 and block.attn.scale == 0.125 and
                    isinstance(block.attn.q_norm, torch.nn.Identity) and isinstance(block.attn.k_norm, torch.nn.Identity) and block.attn.rope is None and
                    block.mlp.act.approximate == 'none' and block.attn.attn_drop.p == block.attn.proj_drop.p == block.sample_drop_ratio == 0. and
                    not block.ls1.inplace and not block.ls2.inplace and set(dict(block.named_parameters())) == set(weights) and not list(block.buffers()) and
                    all(v.dtype == dtype and v.device.type == 'cpu' and not v.requires_grad and bool(torch.isfinite(v).all()) for v in block.parameters()) and
                    tx.dtype == dtype and tx.device.type == 'cpu' and tuple(tx.shape) == SHAPE, 'Fresh original class/complete14 parameter declared dtype required')
            before = state(block); owners.append((block, before)); current['input_raw_sha256'] = raw(tx.numpy())
            current['parameter_metadata'] = {n: dict(shape=list(v.shape), dtype=str(v.dtype), raw_sha256=raw(v.detach().numpy())) for n, v in block.named_parameters()}
            hooks = [block.norm1.register_forward_hook(lambda m, args, result: observe('norm1', result)),
                     block.attn.qkv.register_forward_hook(lambda m, args, result: observe('attn.qkv', result)),
                     block.attn.proj.register_forward_pre_hook(lambda m, args: observe('attn.proj.input', args[0]))]
            functional.scaled_dot_product_attention = sdpa_observer
            current['forward_attempts'] = 1; progress(route + '_forward_start'); tick, cpu_tick = time.perf_counter(), time.process_time()
            with torch.inference_mode():
                result = block(tx)
            current['forward_completed'] = True; observe('output', result)
            current.update(observed_forward_wall_seconds=time.perf_counter()-tick, observed_forward_process_cpu_seconds=time.process_time()-cpu_tick,
                           model_parameter_state_unchanged=state(block) == before, input_unchanged=raw(tx.numpy()) == current['input_raw_sha256'])
            require(set(current['observations']) == set(POINTS) and current['model_parameter_state_unchanged'] and current['input_unchanged'], 'All original points/state/input must remain unchanged')
            for hook in reversed(hooks):
                hook.remove()
            hooks.clear(); functional.scaled_dot_product_attention = original_sdpa
            if route == 'original_FP32':
                require(all(v['byte_equal'] for v in current['comparisons'].values()), 'Original FP32 wholeblock/all observed points must byte-match pinned CPU reference before other controls')
                report['original_FP32_control_all_points_byte_equal'] = True
        report['status'] = SUCCESS
    except BaseException as exc:
        unsupported = (current is not None and current['route'] == 'CPU_float16' and current['forward_attempts'] == 1 and not current['forward_completed'] and
            isinstance(exc, (RuntimeError, NotImplementedError)) and any(v in str(exc).lower() for v in ('not implemented', 'not supported', 'unsupported')))
        report.update(status='failed_CPU_float16_operator_unsupported_no_fallback' if unsupported else 'failed',
                      error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        if current is not None:
            current['error'] = report['error']
        traceback.print_exc()
    finally:
        for hook in reversed(hooks):
            try:
                hook.remove()
            except BaseException as exc:
                report['cleanup_errors'].append('Hook removal: ' + str(exc))
        report['hooks_removed'] = not report['cleanup_errors']
        if functional is not None and original_sdpa is not None:
            functional.scaled_dot_product_attention = original_sdpa
            report['original_SDPA_function_restored'] = functional.scaled_dot_product_attention is original_sdpa
        for path, (digest, before) in pins.items():
            try:
                require(identity(path) == before and sha(path) == digest, 'Source/reference/payload changed: ' + str(path))
            except BaseException as exc:
                report['cleanup_errors'].append(str(exc))
        report['source_and_inputs_unchanged'] = not report['cleanup_errors']
        if owners:
            report['all_instantiated_parameter_states_unchanged'] = all(state(block) == before for block, before in owners)
            require_states = report['all_instantiated_parameter_states_unchanged']
            if not require_states:
                report['cleanup_errors'].append('Parameter state changed')
        try:
            os.sched_setaffinity(0, affinity); os.environ.clear(); os.environ.update(env); sys.path[:] = old_path; signal.signal(signal.SIGTERM, old_term)
            report.update(affinity_restored=set(os.sched_getaffinity(0)) == affinity, environment_restored=dict(os.environ) == env, sys_path_restored=sys.path == old_path)
            require(report['affinity_restored'] and report['environment_restored'] and report['sys_path_restored'], 'Process settings restore failed')
        except BaseException as exc:
            report['cleanup_errors'].append(str(exc))
        report['forward_attempts_by_control'] = {row['route']: row['forward_attempts'] for row in report['calls']}
        report['completed_forwards_by_control'] = {row['route']: int(row['forward_completed']) for row in report['calls']}
        if report['cleanup_errors']:
            report['status'] = 'failed_cleanup'
        progress('finished')
        if saved_bytes + (out / 'result.json').stat().st_size >= BYTE_LIMIT:
            report.update(status='failed', error='Private capture exceeded48MiB including receipt'); atomic(out / 'result.json', report)
    return 0 if report['status'] == SUCCESS else 1


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
