#!/usr/bin/env python3
"""Original CPU SDPA on identical actual NPU QKV: FP32 byte control then Half.

Two component calls only. No block/model construction, checkpoint or weights,
SDK, ORT, GPU, autocast or fallback. Original reshape/permute/unbind, strides,
dropout0, absent mask, noncausal default and default scale are preserved.
Half tensor dtype does not establish the kernels' internal accumulation dtype.
"""
import argparse
import ast
import faulthandler
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
E = BASE / 'experiments'
CAPTURE_SHA = '17d77cfe3d8cdb3ff88e15de4a06634683899348b53b24fbadad6eef1584e91d'
TAP_SHA = '1f89eac69b85ec84ea76f9e8ca0a50ca7d4c5f19620c4159f41eaf6f3661225a'
CF_SHA = '03e9a5797ca21c542e467a68057f7290f0a881a689b273db74f7f7d5d3dea94d'
DUMP_SHA = '565054e051afd2565bcd07d6487e02cedeca62456e09e1214021f30c9c4f46d4'
ATTENTION_SHA = '9d80ff6c6f562efc29d806125fe47bd9b504a6356ce2d9b5d61e8ee4ea770a00'
QKV_RAW = '1f09065df446dbbc56ca249b96156f7ec2c78b87555461c3ed33852e40f91936'
CF_ATTENTION_RAW = 'ca6f8763d7d9c4fad20161b20989e42e61b1cafc90fea33d7dafafd43ac57369'
SHAPE = (1, 16, 1041, 64)
STRIDE = (3197952, 64, 3072, 1)
SUCCESS = 'completed_original_CPU_same_actual_QKV_SDPA_controls_not_quality_accepted'


def require(ok, message):
    if not ok: raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''): h.update(chunk)
    return h.hexdigest()


def atomic(path, value):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=BASE / 'src')
    p.add_argument('--capture', type=Path, default=E / 'vggt-v100-dino0-intermediates-r1/capture.json')
    p.add_argument('--tap-result', type=Path, default=E / 'vggt-v100-dino-taps-core1-r1/result.json')
    p.add_argument('--counterfactual-result', type=Path, default=E / 'vggt-v100-qkv-counterfactual-r1/result.json')
    p.add_argument('--dump-result', type=Path, default=Path('/tmp/vggt-v117-dump-r1/result.json'))
    p.add_argument('--output-dir', type=Path, default=Path('/tmp/vggt-v118-same-qkv-r1'))
    a = p.parse_args()
    require(platform.machine() == 'aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(), 'Actual CPU SDPA calls are RK3588/S-board-only')
    faulthandler.enable(all_threads=True)
    out = a.output_dir.resolve(); require(out.is_relative_to('/tmp'), 'Use fresh private rootfs /tmp output')
    out.mkdir(parents=True, exist_ok=False)
    env, affinity = dict(os.environ), set(os.sched_getaffinity(0)); old_term = signal.getsignal(signal.SIGTERM)
    started, cpu_started = time.perf_counter(), time.process_time(); pins = {}; current = None
    report = dict(schema='v118_original_native518_CPU_SDPA_same_actual_QKV_precision_controls', status='running', script_sha256=sha(__file__),
        capture_sha256=CAPTURE_SHA, tap_result_sha256=TAP_SHA, counterfactual_result_sha256=CF_SHA, dump_result_sha256=DUMP_SHA,
        calls=[], cleanup_errors=[], actual_execution_on_board_only=True, CPU_only=True, original_SDPA_function_or_results_overridden=False,
        block_or_model_constructed=False, checkpoint_or_weights_deserialized=False, SDK_ORT_NPU_GPU_executed=False,
        silent_fallback_enabled=False, frequency_or_system_writes=False, performance_or_trajectory_acceptance=False,
        shape=list(SHAPE), original_QKV_stride_elements=list(STRIDE), intra_op_threads=1, inter_op_threads=1,
        original_call=dict(dropout_p=0.0, attn_mask=None, is_causal=False, scale=None, effective_default_scale=0.125),
        timing_scope='Observed two individual CPU SDPA calls with owned output validation/copies; not a performance benchmark',
        precision_scope='Identical QKV values in FP32/Half tensors; Torch CPU and RKNN internal accumulation/approximations need not match')

    def pin(path, digest, size=None):
        path = Path(path).resolve(strict=True)
        require(sha(path) == digest and (size is None or path.stat().st_size == size), 'Pinned source/reference changed: ' + str(path))
        pins[path] = digest; return path
    def progress(stage):
        u = resource.getrusage(resource.RUSAGE_SELF)
        report.update(stage=stage, elapsed_seconds=time.perf_counter()-started, process_cpu_seconds=time.process_time()-cpu_started,
                      process_user_seconds=u.ru_utime, process_system_seconds=u.ru_stime, peak_rss_mib=u.ru_maxrss/1024)
        atomic(out / 'result.json', report); print(json.dumps(dict(stage=stage)), flush=True)
    def terminate(number, frame):
        raise InterruptedError('External CPU-only180s timeout interrupted control')
    signal.signal(signal.SIGTERM, terminate)
    try:
        pin(__file__, report['script_sha256']); cp = pin(a.capture, CAPTURE_SHA); c = json.loads(cp.read_text())
        tp = pin(a.tap_result, TAP_SHA); t = json.loads(tp.read_text()); fp = pin(a.counterfactual_result, CF_SHA); cf = json.loads(fp.read_text())
        dp = pin(a.dump_result, DUMP_SHA); d = json.loads(dp.read_text())
        require(c['status'] == 'completed_original_DINO0_intermediates' and c['versions']['torch'] == '2.3.1' and
                c['cleanup_errors'] == [] and c['final_comparison']['byte_equal_v99'] is True, 'Clean original CPU DINO0 provenance required')
        require(t['status'] == 'completed_diagnostic_baseline_equal' and t['capture_sha256'] == CAPTURE_SHA and t['final_baseline_byte_equal_all_calls'] is True and
                cf['status'] == 'completed_QKV_counterfactual_diagnostic_not_model_acceptance' and cf['capture_sha256'] == CAPTURE_SHA and cf['tap_result_sha256'] == TAP_SHA and
                cf['baseline_all_points_byte_equal_original_CPU'] is True and cf['QKV_substitution_calls'] == 1 and cf['hooks_removed'] is True and cf['cleanup_errors'] == [], 'Exact same-QKV original CPU reference required')
        require(d['status'] == 'completed_debug_dump_final_byte_equal_not_root_cause_proven' and d['final_baseline_byte_equal'] is True and
                d['output']['raw_sha256'] == 'e52d53b8abcbb15da0c129457f93e3eabcd94d48ae48eb5eaef0bb0edb900554' and
                d['model_sha256'] == 'fe8acac49eb10b1215edf1d4c348b69da3ddfed53f16c2e202b0e3f6428b32c0' and d['cleanup_errors'] == [], 'Unchanged final-compatible NPU dump required')
        relative = 'third_party/vggt/vggt/layers/attention.py'; require(c['source_hashes'][relative] == ATTENTION_SHA, 'Original attention source identity differs')
        source = pin(a.repo / relative, ATTENTION_SHA); tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Attention')
        forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        calls = [n for n in ast.walk(forward) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'scaled_dot_product_attention']
        require(len(calls) == 1 and ast.unparse(calls[0].func) == 'F.scaled_dot_product_attention' and [ast.unparse(n) for n in calls[0].args] == ['q', 'k', 'v'] and
                [n.arg for n in calls[0].keywords] == ['dropout_p'] and ast.unparse(calls[0].keywords[0].value) == 'self.attn_drop.p if self.training else 0.0', 'Pinned original SDPA call signature/defaults differ')
        report.update(checkpoint=c['checkpoint'], source_images=c['source_images'], original_attention_source_sha256=ATTENTION_SHA,
                      verified_original_source_call=ast.unparse(calls[0]), actual_QKV_raw_sha256=QKV_RAW)
        require({4, 5, 6, 7} <= affinity, 'A76 CPUs4/5/6/7 required'); os.sched_setaffinity(0, {4, 5, 6, 7})
        os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        import numpy as np
        import torch
        import torch.nn.functional as F
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        require(torch.__version__ == c['versions']['torch'], 'Original reference Torch2.3.1 required')
        report['versions'] = dict(torch=torch.__version__, numpy=np.__version__)
        def raw(v): return hashlib.sha256(memoryview(np.ascontiguousarray(v)).cast('B')).hexdigest()
        def array(path, row):
            v = np.load(pin(path, row['sha256'], row['bytes']), allow_pickle=False)
            require(v.dtype == np.float32 and list(v.shape) == row['shape'] and np.isfinite(v).all() and raw(v) == row['raw_sha256'], 'Saved reference raw values differ')
            return np.ascontiguousarray(v)
        row = t['calls'][0]['artifact']; npz = pin(tp.parent / row['file'], row['sha256'], row['bytes'])
        with np.load(npz, allow_pickle=False) as z: logical = np.array(z['attn.qkv'], copy=True, order='C')
        require(logical.shape == (1, 1041, 3072) and logical.dtype == np.float32 and np.isfinite(logical).all() and raw(logical) == QKV_RAW and
                cf['actual_NPU_tap_artifact']['arrays']['attn.qkv']['raw_sha256'] == QKV_RAW, 'All actual QKV values must be unchanged')
        ref = array(fp.parent / cf['artifacts']['attn.proj.input']['file'], cf['artifacts']['attn.proj.input'])
        require(raw(ref) == CF_ATTENTION_RAW and ref.shape == (1, 1041, 1024), 'Original same-QKV CPU SDPA consumer reference required')
        npu_row = next(r for r in d['dump_files'] if r['file'].startswith('007_exSDPAttention_'))
        native = array(dp.parent / 'dumps' / npu_row['file'], npu_row); require(native.shape == (16, 64, 1, 1041), 'Native compiled head layout required')
        npu = np.ascontiguousarray(native[:, :, 0, :].transpose(0, 2, 1)[None])
        half_values = logical.astype(np.float16); require(half_values.astype(np.float32).tobytes() == logical.tobytes(), 'Actual QKV Half conversion must preserve every original FP32 byte')
        report['actual_QKV_Half_roundtrip_byte_equal'] = True
        ref_heads = np.ascontiguousarray(ref.reshape(1, 1041, 16, 64).transpose(0, 2, 1, 3))
        def compare(v, expected):
            av, rv = v.astype(np.float64), expected.astype(np.float64); delta = av-rv; an, rn = float(np.linalg.norm(av)), float(np.linalg.norm(rv))
            bad = np.abs(delta) > 1e-5+1e-4*np.abs(rv)
            return dict(relative_l2=float(np.linalg.norm(delta))/rn if rn else None, max_abs=float(np.abs(delta).max()), cosine=float(np.sum(av*rv)/(an*rn)) if an and rn else None,
                        strict_rtol=1e-4, strict_atol=1e-5, strict_pass=not bool(bad.any()), outside_count=int(bad.sum()), byte_equal=v.dtype == expected.dtype and v.tobytes() == expected.tobytes())
        report['NPU_vs_same_QKV_CPU_FP32'] = compare(npu, ref_heads)
        for name, values, dtype in (('original_CPU_FP32', logical, torch.float32), ('original_CPU_float16', half_values, torch.float16)):
            current = dict(name=name, tensor_dtype=str(dtype), attempts=0, completed=False); report['calls'].append(current)
            original = torch.from_numpy(values); before = raw(values)
            q, k, v = original.reshape(1, 1041, 3, 16, 64).permute(2, 0, 3, 1, 4).unbind(0)
            require(all(tuple(z.shape) == SHAPE and tuple(z.stride()) == STRIDE and z.device.type == 'cpu' and z.dtype == dtype for z in (q, k, v)), 'Original QKV shape/strides/CPU dtype required')
            current['QKV_strides'] = [list(z.stride()) for z in (q, k, v)]; current['attempts'] = 1; progress(name + '_SDPA_start')
            tick, cpu_tick = time.perf_counter(), time.process_time()
            with torch.inference_mode(): result = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            current['completed'] = True
            require(result.device.type == 'cpu' and result.dtype == dtype and tuple(result.shape) == SHAPE, 'Original SDPA must return complete declared CPU dtype')
            actual = np.array(result.detach().numpy(), copy=True, order='C'); require(np.isfinite(actual).all() and raw(values) == before, 'Finite owned output/unchanged input required')
            current.update(observed_wall_seconds=time.perf_counter()-tick, observed_process_cpu_seconds=time.process_time()-cpu_tick, input_unchanged=True,
                           versus_same_QKV_original_CPU_FP32=compare(actual, ref_heads), versus_actual_NPU_SDPA=compare(actual, npu))
            path = out / (name + '-attention.npy')
            with path.open('xb') as f: np.save(f, actual, allow_pickle=False)
            consumer = np.ascontiguousarray(actual.transpose(0, 2, 1, 3).reshape(1, 1041, 1024))
            current['output'] = dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path), raw_sha256=raw(actual), shape=list(actual.shape), dtype=str(actual.dtype), tensor_bytes=actual.nbytes, finite=True,
                                     consumer_shape=list(consumer.shape), consumer_raw_sha256=raw(consumer))
            progress(name + '_saved')
            if name == 'original_CPU_FP32':
                require(consumer.tobytes() == ref.tobytes() and raw(consumer) == CF_ATTENTION_RAW, 'Direct original FP32 SDPA must byte-match old counterfactual before Half call')
                report['original_FP32_reference_byte_gate_passed'] = True
        require(sum(row['output']['bytes'] for row in report['calls']) < 8 << 20, 'Bounded complete-output payload required')
        report.update(status=SUCCESS, interpretation='Same actual QKV values and original CPU SDPA parameters were used for two dtypes. Half-vs-NPU differences still compare distinct internal kernels/accumulation, not a same-hardware-precision bug or a deployable acceleration.')
    except BaseException as exc:
        unsupported = current is not None and current['name'] == 'original_CPU_float16' and current['attempts'] == 1 and not current['completed'] and isinstance(exc, (RuntimeError, NotImplementedError)) and any(s in str(exc).lower() for s in ('not implemented', 'not supported', 'unsupported'))
        report.update(status='failed_CPU_float16_SDPA_unsupported_no_fallback' if unsupported else 'failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()); traceback.print_exc()
    finally:
        for path, digest in pins.items():
            try: require(sha(path) == digest, 'Pinned source/input changed: ' + str(path))
            except BaseException as exc: report['cleanup_errors'].append(str(exc))
        report['source_and_inputs_unchanged'] = not report['cleanup_errors']
        try:
            os.sched_setaffinity(0, affinity); os.environ.clear(); os.environ.update(env); signal.signal(signal.SIGTERM, old_term)
            report.update(affinity_restored=set(os.sched_getaffinity(0)) == affinity, environment_restored=dict(os.environ) == env)
            require(report['affinity_restored'] and report['environment_restored'], 'Process settings restoration failed')
        except BaseException as exc: report['cleanup_errors'].append(str(exc))
        if report['cleanup_errors']: report['status'] = 'failed_cleanup'
        progress('finished')
    return 0 if report['status'] == SUCCESS else 1


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
