#!/usr/bin/env python3
"""One flags0 call of the untouched native DINO0 RKNN with runtime debug dump.

RKNN_LOG_LEVEL=5 is documented; RKNN_DUMP_DIR is present in the SHA-pinned
2.3.2 library and its actual destination must be checked. No graph conversion,
extra public tensor, CPU reference forward or performance acceptance occurs.
Dump attribution requires the final output to byte-match the untapped baseline.
"""
import argparse
import faulthandler
import hashlib
import importlib.util
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
OWNED_SHA = '0bfcb295ad5490e8196e8fcc1244e11a99b07bfb3ba7d5c350cb8c81d3ff3be4'
MODEL_SHA = 'fe8acac49eb10b1215edf1d4c348b69da3ddfed53f16c2e202b0e3f6428b32c0'
CAPTURE_SHA = '17d77cfe3d8cdb3ff88e15de4a06634683899348b53b24fbadad6eef1584e91d'
INPUT_SHA = 'aae907d5105f68d4b6d74bb733871e0971fb2c639ec8be8cb46a82176a706aa8'
INPUT_RAW_SHA = '706d7e06efb45e7c90d3fc990e3479a4f566e93cb8c44f42ef57622b037f1efd'
BASELINE_RAW_SHA = 'e52d53b8abcbb15da0c129457f93e3eabcd94d48ae48eb5eaef0bb0edb900554'
TAP_SHA = '1f89eac69b85ec84ea76f9e8ca0a50ca7d4c5f19620c4159f41eaf6f3661225a'
SHAPE = (1, 1041, 1024)


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
    p.add_argument('--source-root', type=Path, default=BASE / 'src')
    p.add_argument('--owned-source', type=Path, default=E / 'vggt-v100-source-r1/native_heads_runtime.py')
    p.add_argument('--model', type=Path, default=E / 'vggt-v99-dino-compile-r1/native-block.rknn')
    p.add_argument('--capture', type=Path, default=E / 'vggt-v100-dino0-intermediates-r1/capture.json')
    p.add_argument('--tap-result', type=Path, default=E / 'vggt-v100-dino-taps-core1-r1/result.json')
    p.add_argument('--output-dir', type=Path, default=Path('/tmp/vggt-v117-dump-r1'))
    a = p.parse_args()
    require(platform.machine() == 'aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(), 'RK3588/S board SDK execution only')
    faulthandler.enable(all_threads=True)
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False); dump = out / 'dumps'; dump.mkdir()
    require(str(out).startswith('/tmp/'), 'Use a fresh rootfs /tmp destination, not the nearly-full dataset filesystem')
    env, affinity = dict(os.environ), set(os.sched_getaffinity(0)); old_limit = resource.getrlimit(resource.RLIMIT_FSIZE)
    old_term, old_xfsz = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGXFSZ)
    pins = {}; owner = None; started = time.perf_counter()
    default_dump = Path('/userdata/dumps')

    def disk():
        s = os.statvfs(out); return dict(available_bytes=s.f_bavail*s.f_frsize, free_bytes=s.f_bfree*s.f_frsize)
    def default_snapshot():
        return dict(exists=default_dump.exists(), files=[dict(file=str(f.relative_to(default_dump)), bytes=f.stat().st_size, mtime_ns=f.stat().st_mtime_ns)
            for f in sorted(default_dump.rglob('*')) if f.is_file()] if default_dump.exists() else [])
    report = dict(schema='v117_original_native518_DINO0_unchanged_RKNN_runtime_dump', status='running', script_sha256=sha(__file__),
        model_sha256=MODEL_SHA, init_flags=0, core_mask=1, calls_expected=1, forward_calls=0, cleanup_errors=[],
        graph_recompiled_or_outputs_appended=False, NN_CPU_ORt_GPU_executed=False, candidate_accepted=False, performance_measurement=False,
        debug_duration_is_not_normal_performance=True, final_baseline_byte_parity_required_for_attribution=True,
        dump_path_override_officially_documented=False, dump_path_override_found_in_exact_runtime_binary=True,
        dump_directory=str(dump), default_dump_before=default_snapshot(), disk_before=disk(), dump_files=[],
        disk_estimate=dict(scope='Estimate, not a hard total-byte bound; fused16-row profile output streams and original weights only',
                           profile_output_streams_FP32_bytes=106598400, original_FP32_learned_weight_bytes=50376704,
                           one_materialized_FP32_score_bytes=69355584, fused_internal_score_dumps_not_guaranteed=True),
        file_size_soft_limit_bytes=256 << 20, required_preflight_available_bytes=1 << 30)

    def progress(stage):
        report.update(stage=stage, diagnostic_elapsed_seconds=time.perf_counter()-started, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        atomic(out / 'result.json', report); print(json.dumps(dict(stage=stage)), flush=True)
    def pin(path, digest, size=None):
        path = Path(path).resolve(strict=True)
        require(sha(path) == digest and (size is None or path.stat().st_size == size), 'Pinned original source/payload changed: ' + str(path))
        pins[path] = digest; return path
    def interrupted(number, frame):
        raise InterruptedError('External timeout or per-file256MiB limit interrupted diagnostic: signal ' + str(number))
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGXFSZ, interrupted)
    try:
        require(report['disk_before']['available_bytes'] >= 1 << 30, 'Need at least1GiB available on actual dump rootfs')
        require(not os.path.lexists(default_dump), 'Default dump destination must be absent; never risk overwriting earlier records')
        require({4, 5, 6, 7} <= affinity, 'A76 CPUs4/5/6/7 required')
        os.sched_setaffinity(0, {4, 5, 6, 7})
        limit = 256 << 20
        if old_limit[0] != resource.RLIM_INFINITY: limit = min(limit, old_limit[0])
        resource.setrlimit(resource.RLIMIT_FSIZE, (limit, old_limit[1])); report['actual_file_size_soft_limit_bytes'] = limit
        pin(__file__, report['script_sha256']); cp = pin(a.capture, CAPTURE_SHA); c = json.loads(cp.read_text())
        require(c['status'] == 'completed_original_DINO0_intermediates' and c['calls'] == 1 and len(c['original_parameters']) == 14 and
                c['hooks_removed'] is True and c['source_and_inputs_unchanged'] is True and c['original_parameter_state_unchanged'] is True and
                c['final_comparison']['byte_equal_v99'] is True and c['cleanup_errors'] == [], 'Complete original14-parameter native CPU reference required')
        tp = pin(a.tap_result, TAP_SHA); t = json.loads(tp.read_text())
        require(t['status'] == 'completed_diagnostic_baseline_equal' and t['capture_sha256'] == CAPTURE_SHA and
                t['final_baseline_byte_equal_all_calls'] is True and t['original_input_raw_sha256'] == INPUT_RAW_SHA and t['cleanup_errors'] == [], 'Known untapped-final-equivalent NPU baseline required')
        # Environment is set before importing the owned helper or loading the SDK.
        os.environ.update(RKNN_LOG_LEVEL='5', RKNN_DUMP_DIR=str(dump), CUDA_VISIBLE_DEVICES='',
                          OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        os.environ.pop('RKNN_DUMP_QUANT', None); os.environ.pop('RKNN_DUMP_TXT_TENSOR', None)
        source = pin(a.owned_source, OWNED_SHA)
        spec = importlib.util.spec_from_file_location('_v117_frozen_owned_context', source); H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
        import numpy as np
        def raw(v): return hashlib.sha256(memoryview(np.ascontiguousarray(v)).cast('B')).hexdigest()
        def array(row):
            path = pin(cp.parent / row['file'], row['sha256'], row['bytes']); v = np.load(path, allow_pickle=False)
            require(v.shape == SHAPE and v.dtype == np.float32 and np.isfinite(v).all() and raw(v) == row['raw_sha256'], 'Original finite raw array identity differs')
            return v
        x = array(c['artifacts']['input']); cpu_ref = array(c['artifacts']['output'])
        require(c['artifacts']['input']['sha256'] == INPUT_SHA and raw(x) == INPUT_RAW_SHA, 'Use identical actual native prepared input')
        row = t['calls'][0]['artifact']; path = pin(tp.parent / row['file'], row['sha256'], row['bytes'])
        with np.load(path, allow_pickle=False) as archive: baseline = np.array(archive['output'], copy=True, order='C')
        require(baseline.dtype == np.float32 and baseline.shape == SHAPE and raw(baseline) == BASELINE_RAW_SHA, 'Exact untapped baseline final required')
        report.update(checkpoint=c['checkpoint'], original_parameter_count=14, original_parameters=c['original_parameters'], input=c['artifacts']['input'],
                      original_CPU_reference=c['artifacts']['output'], NPU_baseline_payload=row, owned_source_sha256=OWNED_SHA)
        model = pin(a.model, MODEL_SHA, 26517686)
        progress('ordinary_untapped_RKNN_init_with_debug_environment')
        try:
            owner = H.OwnedContext(model, MODEL_SHA, input_shapes=[SHAPE], output_shapes=[SHAPE], input_names=['input'], output_names=['output'],
                                   core_mask=1, flags=0, source_root=a.source_root)
        except BaseException as exc:
            owner = getattr(exc, 'owned_context', owner); raise
        progress('one_actual_call')
        y = owner.run([x])[0]; report['forward_calls'] = 1
        report['SDK_IO_timings_diagnostic_only'] = owner.last_timings
        yp = out / 'output.npy'
        with yp.open('xb') as f: np.save(f, y, allow_pickle=False)
        report['output'] = dict(file=yp.name, bytes=yp.stat().st_size, sha256=sha(yp), raw_sha256=raw(y), shape=list(y.shape), dtype=str(y.dtype), finite=bool(np.isfinite(y).all()))
        parity = y.tobytes() == baseline.tobytes() and raw(y) == BASELINE_RAW_SHA
        report.update(final_baseline_byte_equal=parity, original_CPU_comparison=H.module(H.EXPERIMENT / 'vggt-v99-source-r1/run_native_block_r1.py', H.HELPER_SHA, '_v117_metric_only').comparison(np, y, cpu_ref))
        owner.close(); report['resources'] = owner.metrics()
        require(report['resources']['closed'] and report['resources']['resource_counts_balanced'] and report['resources']['contexts_created'] == 1 and
                report['resources']['runs_completed'] == 1 and report['resources']['output_buffers_acquired'] == 1 and report['resources']['cleanup_errors'] == [], 'Onecontext/onerun/onebuffer ledger must close')
        for f in sorted(dump.rglob('*')):
            if not f.is_file(): continue
            r = dict(file=str(f.relative_to(dump)), bytes=f.stat().st_size, sha256=sha(f))
            if f.suffix.lower() == '.npy':
                try:
                    v = np.load(f, allow_pickle=False, mmap_mode='r')
                    r.update(shape=list(v.shape), dtype=str(v.dtype), tensor_bytes=v.nbytes, finite=bool(np.isfinite(v).all()), raw_sha256=raw(v))
                    del v
                except BaseException as exc: r['array_parse_error'] = str(exc)
            report['dump_files'].append(r)
        valid = [r for r in report['dump_files'] if 'raw_sha256' in r]
        report.update(dump_total_file_bytes=sum(r['bytes'] for r in report['dump_files']), valid_NPY_dump_count=len(valid),
                      default_dump_after=default_snapshot(), dump_layout_not_assumed_logical_BHSD=True,
                      fused_layer_dump_does_not_guarantee_separate_QK_softmax_PV_values=True)
        report['default_dump_unchanged'] = report['default_dump_before'] == report['default_dump_after']
        report['attribution_preconditions_met'] = parity and bool(valid) and report['default_dump_unchanged']
        report['status'] = 'completed_debug_dump_final_byte_equal_not_root_cause_proven' if report['attribution_preconditions_met'] else 'completed_debug_dump_no_attribution'
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()); traceback.print_exc()
    finally:
        if owner is not None:
            try: owner.close()
            except BaseException as exc: report['cleanup_errors'].append(str(exc))
            report['resources'] = owner.metrics()
        for path, digest in pins.items():
            if not path.is_file() or sha(path) != digest: report['cleanup_errors'].append('Pinned input/source changed: ' + str(path))
        report['source_and_inputs_unchanged'] = not report['cleanup_errors']
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, old_limit); os.sched_setaffinity(0, affinity); os.environ.clear(); os.environ.update(env)
            signal.signal(signal.SIGTERM, old_term); signal.signal(signal.SIGXFSZ, old_xfsz)
            report.update(environment_restored=dict(os.environ) == env, affinity_restored=set(os.sched_getaffinity(0)) == affinity,
                          file_size_limit_restored=resource.getrlimit(resource.RLIMIT_FSIZE) == old_limit)
            require(all(report[k] for k in ('environment_restored', 'affinity_restored', 'file_size_limit_restored')), 'Process settings cleanup failed')
        except BaseException as exc: report['cleanup_errors'].append(str(exc))
        report['disk_after'] = disk()
        if report['cleanup_errors']: report['status'] = 'failed_cleanup'
        progress('finished')
    return 1 if report['status'].startswith('failed') else 0


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
