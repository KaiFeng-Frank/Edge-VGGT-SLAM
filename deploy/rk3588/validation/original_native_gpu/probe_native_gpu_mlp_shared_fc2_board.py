#!/usr/bin/env python3
"""Original full frame0 MLP: compiled/core7 versus GPU Erf + shared-A FC2 parts.

Actual execution is RK3588S-only. All 106 rows and original learned values are
retained. This is a component diagnostic, with no fallback or model acceptance.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait
import ctypes as ct
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

BASE = Path('/mnt/vggt-data/vggt-slam2')
E = BASE / 'experiments'
HERE = Path(__file__).resolve().parent
NATIVE_HELPER_SHA = '08d9b84af256c91b60a2f6dcd8100bf68d2a2d0209fdde1ff90fd9e4fbb09c6d'
SHARED_FC2_SOURCE_SHA = '4a7e8da5c981e68ed577f7dc9f774e9c4b48eea40bc7066ce30da3a23786b9d9'
GPU_SOURCE_SHA = 'f06486299101fb233c68abc9908e8c2629c4cb1f7dca6241133e6e222634acc2'
GPU_LIBRARY_SHA = 'd57e53240b5078eee5cfbc79e103544c090a7825c09fdf3877fee45084897dda'
RANGES = ((0, 352, 1), (352, 704, 2), (704, 1024, 4))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=BASE / 'src')
    p.add_argument('--native-helper', type=Path, default=HERE / 'probe_native_gpu_mlp_observed_board.py')
    p.add_argument('--capture-dir', type=Path, default=E / 'vggt-v53-capture-r1')
    p.add_argument('--saved-dir', type=Path, default=E / 'vggt-v68-native-compiled-rank3-ab-r1')
    p.add_argument('--runtime-library', type=Path, default=Path('/usr/lib/librknnrt.so'))
    p.add_argument('--neon-library', type=Path, default=E / 'vggt-v56-neon-build-r1/fp16_neon.so')
    p.add_argument('--gpu-bridge-source', type=Path, default=HERE / 'native_erf_gelu_opencl.py')
    p.add_argument('--gpu-library', type=Path, default=E / 'libmali-valhall-g610-g6p0-dummy.so')
    p.add_argument('--expected-npu-hz', type=int, choices=(300000000, 1000000000), default=1000000000)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if platform.machine() != 'aarch64' or b'rk3588' not in Path('/proc/device-tree/compatible').read_bytes():
        raise RuntimeError('Actual MLP/SDK/GPU execution is board-only')
    if sha(args.native_helper) != NATIVE_HELPER_SHA:
        raise ValueError('Frozen successful native/GPU helper differs')
    n = module(args.native_helper, '_v81_observed_native_helpers')
    repo, saved, capture = (x.resolve(strict=True) for x in (args.repo, args.saved_dir, args.capture_dir))
    hp = repo / 'deploy/rk3588/validation/original_native_mlp/native_mlp_rank3_probe_board.py'
    if sha(hp) != n.HELPER_SHA:
        raise ValueError('Frozen original capture helper differs')
    h = module(hp, '_v81_complete_mlp_capture_helpers')
    pins = {args.native_helper.resolve(): NATIVE_HELPER_SHA, hp: n.HELPER_SHA}
    def pin(path, digest, size=None):
        file = h.pin(path, digest, size)
        pins[file] = digest
        return file
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='v81_original_frame0_MLP_GPU_Erf_shared_A_FC2_diagnostic', status='failed',
        source_sha256=sha(__file__), derived_shared_FC2_source_sha256=SHARED_FC2_SOURCE_SHA,
        original_model_only=True, checkpoint=n.CHECKPOINT, candidate_accepted=False,
        full_model_or_SLAM_validated=False, strict_rtol=1e-4, strict_atol=1e-5,
        shape=[106, 1024, 4096, 1024], no_rows_tokens_layers_or_learned_values_omitted=True,
        expected_npu_hz=args.expected_npu_hz, frequency_written=False,
        routes={'A': 'original_rank3_compiled_FP16_core7',
            'E': 'native_AC1_FC1_core1_GPU_original_Erf_shared_A_FC2_parts_cores1_2_4'},
        native_AC_layout=1, native_B_layout=1, native_type=1,
        FC2_output_channel_ranges=[list(x) for x in RANGES], FC2_K_reduction_partitioned=False,
        FC1_bias_in_GPU_once=True, FC2_bias_added_once_after_each_complete_K_product=True,
        shared_A_owner='unused_unsplit_FC2_context', shared_A_allocations_consumed_by_parts=1,
        unused_part_owned_A_allocations_retained=3, shared_A_cross_context_support_assumed=False,
        GPU_managed_host_consistency=True, no_CPU_writes_to_A2_during_concurrent_runs=True,
        first_calls_retained=True, warm_calls_per_route=12,
        timing_scope='Complete input conversion/packing/FC1/sync/GPU bias-Erf-half-A8 write/FC2 owner sync, shared bindings, concurrent complete-K products, all output sync/owned copies/bias/concatenation/finite checks. Comparison, first-call diagnostics, hashes and file saves are outside.',
        limits=['Original MLP component only; this is not full VGGT or fresh SLAM pose throughput.',
            'Compiled core7 lowering and explicit native/GPU execution may round differently; fixed original FP32 checks are recorded.',
            'Combined native MatMul mask7 is not used. Individual masks1/2/4 and shared A support must succeed; no fallback.',
            'GPU synchronization retains the previously validated managed import policy; same-FC1 first-call checks do not replace full-model quality validation.'],
        calls=[], cleanup_errors=[])
    original_affinity = os.sched_getaffinity(0)
    contexts, counters, compiled, bridge, executor = [], [], None, None, None
    error = None
    try:
        if not {4, 5, 6, 7} <= original_affinity:
            raise RuntimeError('CPUs4..7 unavailable')
        os.sched_setaffinity(0, {4, 5, 6, 7})
        import numpy as np
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        report['versions'] = dict(numpy=np.__version__, torch=torch.__version__)
        report['cpu_affinity'], report['torch_threads'] = [4, 5, 6, 7], 1
        old = json.loads(pin(saved / 'result.json', n.SAVED_SHA).read_text())
        cap = json.loads(pin(capture / 'capture.json', n.CAPTURE_SHA).read_text())
        if (old['status'] != 'completed_component_diagnostic_not_quality_accepted'
                or old['checkpoint'] != n.CHECKPOINT or old['capture_sha256'] != n.CAPTURE_SHA
                or old['cleanup_errors'] or not old['reconstructed_block_vs_original_full_forward']['bitwise_equal']
                or cap['checkpoint'] != n.CHECKPOINT or cap['cleanup_errors']
                or cap['full_model_forward_completed'] is not True):
            raise ValueError('Completed original capture/reference required')
        for name, item in old['sources'].items():
            pin(repo / name, item['sha256'], item['bytes'])
        def array(directory, item):
            value = np.load(pin(h.child(directory, item['file']), item['sha256'], item['bytes']), allow_pickle=False)
            if (str(value.dtype) != item['dtype'] or list(value.shape) != item['shape']
                    or h.raw_sha(value) != item['raw_sha256'] or not value.flags.c_contiguous
                    or not np.isfinite(value).all()):
                raise ValueError('Saved original array changed')
            return value
        x, reference, original_fc1, original_gelu = (array(saved, old['artifacts'][k])
            for k in ('input', 'torch_output', 'torch_fc1', 'torch_fc2_input'))
        if (x.dtype != np.float32 or x.shape != (106, 1024) or reference.shape != x.shape
                or original_fc1.shape != (106, 4096) or original_gelu.shape != original_fc1.shape):
            raise ValueError('All original 106-row MLP arrays required')
        row = next(item for item in cap['blocks'] if item['role'] == 'frame' and item['index'] == 0)
        if row['gelu_approximate'] != 'none':
            raise ValueError('Original exact Erf GELU required')
        wp = pin(h.child(capture, row['weights']['file']), row['weights']['sha256'], row['weights']['bytes'])
        with np.load(wp, allow_pickle=False) as archive:
            weights = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
        if set(weights) != set(row['parameters']) or len(weights) != 18:
            raise ValueError('All original frame0 parameter identities required')
        for key, value in weights.items():
            meta = row['parameters'][key]
            if (value.dtype != np.float32 or list(value.shape) != meta['shape']
                    or h.raw_sha(value) != meta['raw_sha256'] or not np.isfinite(value).all()):
                raise ValueError('Original parameter changed: ' + key)
        for key, shape in {'mlp.fc1.weight': (4096, 1024), 'mlp.fc1.bias': (4096,),
                'mlp.fc2.weight': (1024, 4096), 'mlp.fc2.bias': (1024,)}.items():
            if weights[key].shape != shape:
                raise ValueError('Original MLP parameter shape changed')
        report['original_parameter_metadata'] = row['parameters']
        report['reference'] = dict(saved_receipt_sha256=n.SAVED_SHA, capture_sha256=n.CAPTURE_SHA,
            input=old['artifacts']['input'], output=old['artifacts']['torch_output'])
        slices = [(np.ascontiguousarray(weights['mlp.fc2.weight'][start:end]),
            np.ascontiguousarray(weights['mlp.fc2.bias'][start:end])) for start, end, _ in RANGES]
        if (np.concatenate([v[0] for v in slices]).tobytes() != weights['mlp.fc2.weight'].tobytes()
                or np.concatenate([v[1] for v in slices]).tobytes() != weights['mlp.fc2.bias'].tobytes()):
            raise ValueError('Original FC2 slices failed exact reconstruction')
        report['FC2_weight_and_bias_slices_reconstruct_bytewise'] = True
        pin(args.runtime_library, n.RUNTIME_SHA)
        neon = ct.CDLL(str(pin(args.neon_library, n.NEON_SHA)))
        neon.f32_to_f16.argtypes, neon.f32_to_f16.restype = [ct.c_void_p, ct.c_void_p, ct.c_size_t], None
        sys.path[:0] = [str(repo), str(repo / 'third_party/vggt')]
        abi = n.load_module(pin(repo / 'vggt_slam/rknn_linear.py', n.ABI_SHA), '_v81_shared_mlp_abi')
        if (ct.sizeof(abi.RKNNMatmulInfo), ct.sizeof(abi.RKNNMatmulIOAttr), ct.sizeof(abi.RKNNTensorMem)) != (64, 996, 40):
            raise RuntimeError('aarch64 typed ABI mismatch')
        api = abi._API(str(args.runtime_library))
        started = time.perf_counter()
        cp = pin(repo / 'deploy/rk3588/validation/original_native_mlp/compiled_mlp_rank3.py', n.COMPILED_SHA)
        compiled = n.load_module(cp, '_v81_shared_mlp_compiled').CompiledMLP(repo, core_mask=7)
        definitions = [('FC1', 1024, 4096, 1, weights['mlp.fc1.weight']),
            ('FC2_owner', 4096, 1024, 1, weights['mlp.fc2.weight'])] + [
            (f'FC2_part{i}', 4096, end-start, mask, slices[i][0]) for i, (start, end, mask) in enumerate(RANGES)]
        for label, k, channels, mask, weight in definitions:
            calls = n.Counted(api)
            counters.append(calls)
            context = n.NativeContext(abi, calls, np, k, channels, True, mask)
            contexts.append(context)
            context.record['label'] = label
            context.open(weight)
        first, owner, *parts = contexts
        common_A = owner.memories['A']
        if any(part.record['A'] != owner.record['A'] for part in parts):
            raise RuntimeError('Shared native A descriptors differ')
        report['shared_A_descriptor'] = owner.record['A']
        gpu = module(pin(args.gpu_bridge_source, GPU_SOURCE_SHA), '_v81_shared_fc2_gpu_bridge')
        library = pin(args.gpu_library, GPU_LIBRARY_SHA)
        bridge = gpu.NativeErfGeluBridge.__new__(gpu.NativeErfGeluBridge)
        bridge.__init__(first.memories['C'], common_A, weights['mlp.fc1.bias'],
            rows=106, channels=4096, library=str(library), managed_host_consistency=True)
        executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix='original-mlp-fc2')
        report['setup_seconds'] = time.perf_counter() - started
        gelu = torch.nn.GELU(approximate='none').eval()
        def run_part(context, bias):
            start = time.perf_counter()
            abi._check(context.api.rknn_matmul_set_io_mem(context.ctx, common_A,
                ct.byref(context.attrs.A)), 'FC2 shared GPU-A binding')
            run_start = time.perf_counter()
            abi._check(context.api.rknn_matmul_run(context.ctx), 'FC2 partition MatMul')
            run_end = time.perf_counter()
            value = context.owned()
            value += bias
            if not np.isfinite(value).all():
                raise RuntimeError('Nonfinite original FC2 partition')
            end = time.perf_counter()
            return value, dict(core_mask=context.mask, shared_A_bind_seconds=run_start-start,
                MatMul_run_seconds=run_end-run_start, C_sync_owned_bias_finite_seconds=end-run_end,
                worker_start_monotonic=start, run_start_monotonic=run_start,
                run_end_monotonic=run_end, worker_end_monotonic=end)
        def native_gpu_parts():
            begin = time.perf_counter()
            first.submit(x, neon)
            stages = dict(FC1_conversion_pack_sync_run_seconds=time.perf_counter()-begin)
            begin = time.perf_counter()
            gpu_times = bridge.run(before_input=lambda: first.sync('C', abi.RKNN_MEMORY_SYNC_FROM_DEVICE),
                before_output=lambda: owner.sync('A', abi.RKNN_MEMORY_SYNC_TO_DEVICE),
                after_output=lambda: owner.sync('A', abi.RKNN_MEMORY_SYNC_FROM_DEVICE))
            stages['GPU_bias_Erf_native_A8_and_sync_seconds'] = time.perf_counter()-begin
            stages['GPU_bridge_breakdown'] = gpu_times
            begin = time.perf_counter()
            # GPU completion/invalidation precedes the sole post-GPU input flush.
            # No CPU writes or next GPU output occur until all three runs end.
            owner.sync('A', abi.RKNN_MEMORY_SYNC_TO_DEVICE)
            stages['FC2_owner_A_post_GPU_TO_DEVICE_seconds'] = time.perf_counter()-begin
            begin = time.perf_counter()
            futures = [executor.submit(run_part, part, slices[i][1]) for i, part in enumerate(parts)]
            wait(futures)
            failures = [f.exception() for f in futures if f.exception() is not None]
            if failures:
                raise RuntimeError('Explicit shared-A FC2 partition failed: ' + '; '.join(str(e) for e in failures)) from failures[0]
            results = [future.result() for future in futures]
            value = np.concatenate([result[0] for result in results], axis=1)
            stages['FC2_dispatch_run_owned_bias_concat_seconds'] = time.perf_counter()-begin
            stages['FC2_workers'] = [result[1] for result in results]
            return value, stages
        def save_array(name, value):
            value = np.ascontiguousarray(value)
            if not np.isfinite(value).all():
                raise RuntimeError('Nonfinite owned snapshot')
            file = out / name
            with file.open('xb') as stream:
                np.save(stream, value, allow_pickle=False)
            return dict(file=name, bytes=file.stat().st_size, sha256=sha(file),
                raw_sha256=h.raw_sha(value), shape=list(value.shape), dtype=str(value.dtype))
        schedule = ['A', 'E'] + [route for j in range(12) for route in ('AE' if j % 2 == 0 else 'EA')]
        report['schedule'] = schedule
        previous = {}
        for index, route in enumerate(schedule):
            before = n.platform_observation(args.expected_npu_hz)
            begin = time.perf_counter()
            value, stages = (compiled.run(x), {}) if route == 'A' else native_gpu_parts()
            if (value.dtype != np.float32 or value.shape != (106, 1024)
                    or not value.flags.c_contiguous or not np.isfinite(value).all()):
                raise RuntimeError('Finite owned dense original MLP output required')
            elapsed = time.perf_counter()-begin
            after = n.platform_observation(args.expected_npu_hz)
            first_call = route not in previous
            item = dict(index=index, route=route, first_for_route=first_call,
                complete_route_seconds=elapsed, stages_seconds=stages,
                platform_before=before, platform_after=after, vs_original_CPU_FP32=h.compare(value, reference),
                bitwise_equal_first_same_route=None if first_call else value.tobytes() == previous[route].tobytes(),
                vs_first_compiled_output=h.compare(value, previous.get('A', value)))
            if route == 'E' and first_call:
                diagnostic_begin = time.perf_counter()
                fc1_matmul = first.owned()
                biased_fc1 = np.ascontiguousarray(fc1_matmul + weights['mlp.fc1.bias'])
                with torch.inference_mode():
                    cpu_gelu = np.ascontiguousarray(gelu(torch.from_numpy(biased_fc1)).numpy())
                owner.sync('A', abi.RKNN_MEMORY_SYNC_FROM_DEVICE)
                gpu_a8 = owner.views['A'].copy()
                gpu_half = np.ascontiguousarray(gpu_a8.transpose(1, 0, 2).reshape(106, 4096))
                gpu_value = gpu_half.astype(np.float32)
                report['GPU_same_NPU_FC1_validation'] = dict(
                    scope='First E after normal elapsed stops; owned reads only, no diagnostic is substituted into timed inference.',
                    seconds=time.perf_counter()-diagnostic_begin,
                    FC1_vs_original_CPU=h.compare(biased_fc1, original_fc1),
                    CPU_GELU_same_FC1_vs_original_CPU=h.compare(cpu_gelu, original_gelu),
                    GPU_GELU_half_vs_original_CPU=h.compare(gpu_value, original_gelu),
                    GPU_GELU_vs_same_FC1_CPU_GELU_once_FP16_rounded=h.compare(gpu_value, cpu_gelu.astype(np.float16).astype(np.float32)),
                    artifacts=dict(FC1_matmul=save_array('same-fc1-matmul.npy', fc1_matmul),
                        FC1_biased=save_array('same-fc1-biased.npy', biased_fc1),
                        CPU_GELU=save_array('same-fc1-cpu-gelu.npy', cpu_gelu),
                        GPU_native_A8=save_array('same-fc1-gpu-a8.npy', gpu_a8),
                        GPU_GELU_half=save_array('same-fc1-gpu-gelu-half.npy', gpu_half)))
            item['output'] = save_array(f'call-{index:02d}-{route}.npy', value)
            report['calls'].append(item)
            if first_call:
                previous[route] = value.copy()
            print(json.dumps(dict(index=index, route=route, seconds=elapsed)), flush=True)
        report['summary_seconds'] = {route: dict(all_count=13, warm_count=12,
            all_mean=statistics.mean(c['complete_route_seconds'] for c in report['calls'] if c['route'] == route),
            warm_mean=statistics.mean(c['complete_route_seconds'] for c in report['calls'] if c['route'] == route and not c['first_for_route']),
            warm_median=statistics.median(c['complete_route_seconds'] for c in report['calls'] if c['route'] == route and not c['first_for_route'])) for route in 'AE'}
        report['status'] = 'completed_original_MLP_component_not_quality_accepted'
    except BaseException as exc:
        error = exc
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['exception_notes'] = list(getattr(exc, '__notes__', []))
    finally:
        workers_quiescent = executor is None
        if executor is not None:
            try:
                executor.shutdown(wait=True, cancel_futures=True)
                workers_quiescent = True
            except BaseException as exc:
                report['cleanup_errors'].append('Worker shutdown: ' + str(exc))
        gpu_released = bridge is None
        if bridge is not None:
            close_ok = False
            try:
                bridge.close()
                close_ok = True
            except BaseException as exc:
                report['cleanup_errors'].append('GPU close: ' + str(exc))
            try:
                metrics = bridge.metrics()
                report['GPU_metrics_after_close'] = metrics
                gpu_released = (close_ok and metrics['fully_released']
                    and metrics['live_OpenCL_objects'] == metrics['live_owned_DMA_fds'] == 0
                    and metrics['events_created'] == metrics['events_released'])
                report['cleanup_errors'].extend(metrics['cleanup_errors'])
                if report['status'].startswith('completed') and metrics['calls_completed'] != 13:
                    report['cleanup_errors'].append('GPU completed-call count differs')
            except BaseException as exc:
                report['cleanup_errors'].append('GPU metrics: ' + str(exc))
        report['contexts_after_close'] = []
        for context in contexts[::-1]:
            label = context.record.get('label', 'partial')
            owner_with_live_parts = label == 'FC2_owner' and any(c.ctx.value for c in contexts[2:])
            if not gpu_released or not workers_quiescent or owner_with_live_parts:
                state = dict(label=label, live_context=int(bool(context.ctx.value)),
                    unresolved_memory_count=len(context.memories),
                    cleanup_errors=['Native close skipped while GPU/workers/shared-A child lifetime is unresolved'])
            else:
                try:
                    state = context.close()
                    state['label'] = label
                except BaseException as exc:
                    state = dict(label=label, live_context=int(bool(context.ctx.value)),
                        unresolved_memory_count=len(context.memories), cleanup_errors=['Native close: ' + str(exc)])
            report['contexts_after_close'].append(state)
            report['cleanup_errors'].extend(state['cleanup_errors'])
        if compiled is not None:
            try:
                compiled.close()
            except BaseException as exc:
                report['cleanup_errors'].append('Compiled close: ' + str(exc))
            try:
                metrics = compiled.metrics()
                report['compiled_metrics_after_close'] = metrics
                report['cleanup_errors'].extend(metrics['cleanup_errors'])
                if report['status'].startswith('completed') and not (metrics['closed'] and metrics['live_contexts'] == 0
                        and metrics['contexts_created'] == metrics['contexts_destroyed'] == 1
                        and metrics['outputs_acquired'] == metrics['outputs_released'] == 13
                        and metrics['calls_completed'] == metrics['sdk_runs_succeeded'] == 13):
                    report['cleanup_errors'].append('Compiled lifecycle counts differ')
            except BaseException as exc:
                report['cleanup_errors'].append('Compiled metrics: ' + str(exc))
        report['SDK_API'] = [dict(calls=c.calls, successes=c.successes, return_codes=c.codes) for c in counters]
        report['native_lifecycle_balanced'] = (len(counters) == 5 and len(report['contexts_after_close']) == 5
            and all(c.successes.get('rknn_matmul_create', 0) == c.successes.get('rknn_matmul_destroy', 0) == 1
                and c.successes.get('rknn_create_mem', 0) == c.successes.get('rknn_destroy_mem', 0) == 3
                and c.successes.get('rknn_matmul_run', 0) == runs
                and all(code == 0 for codes in c.codes.values() for code in codes)
                for c, runs in zip(counters, (13, 0, 13, 13, 13)))
            and not any(c['live_context'] or c['unresolved_memory_count'] for c in report['contexts_after_close']))
        if report['status'].startswith('completed') and not report['native_lifecycle_balanced']:
            report['cleanup_errors'].append('Expected native5 contexts/15 memories/52 runs differ')
        try:
            os.sched_setaffinity(0, original_affinity)
            report['affinity_restored'] = os.sched_getaffinity(0) == original_affinity
        except BaseException as exc:
            report['affinity_restored'] = False
            report['cleanup_errors'].append('Affinity restore: ' + str(exc))
        try:
            report['sources_and_inputs_unchanged'] = all(sha(file) == digest for file, digest in pins.items())
            report['pinned_files'] = {str(file): digest for file, digest in pins.items()}
        except BaseException as exc:
            report['sources_and_inputs_unchanged'] = False
            report['cleanup_errors'].append('Rehash: ' + str(exc))
        if report['cleanup_errors'] or not report['affinity_restored'] or not report['sources_and_inputs_unchanged']:
            report['status'] = 'failed_cleanup_or_identity'
        n.atomic(out / 'result.json', report)
        print(json.dumps(dict(status=report['status'], result_sha256=sha(out / 'result.json'))), flush=True)
    if error:
        raise error
    if report['status'] != 'completed_original_MLP_component_not_quality_accepted':
        raise RuntimeError(report['status'])


if __name__ == '__main__':
    main()
