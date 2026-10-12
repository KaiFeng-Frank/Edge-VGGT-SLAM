#!/usr/bin/env python3
"""Original FC2: single core versus three complete-K parts sharing one native A.

Source-only preparation; actual ctypes/MLP execution is RK3588S-only. The
Three parts have N=352/352/320, K=4096, M=106 and explicit masks 1/2/4.
Shared-A support is an actual SDK diagnostic, never an assumed capability.
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
NATIVE_HELPER_SHA = 'b53521e8db68d5f46467364cd2712f7498834569beee7ee39c7eef5539540abf'
CAPTURE_HELPER_SHA = '17be747bf865268c3e6182abe9d4852f7f0199ddead1e3c2218dfb55111b395b'
RANGES = ((0, 352, 1), (352, 704, 2), (704, 1024, 4))


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=BASE / 'src')
    p.add_argument('--native-helper', type=Path, default=HERE / 'probe_native_gpu_mlp_board_r2.py')
    p.add_argument('--capture-dir', type=Path, default=E / 'vggt-v53-capture-r1')
    p.add_argument('--saved-dir', type=Path, default=E / 'vggt-v68-native-compiled-rank3-ab-r1')
    p.add_argument('--runtime-library', type=Path, default=Path('/usr/lib/librknnrt.so'))
    p.add_argument('--neon-library', type=Path, default=E / 'vggt-v56-neon-build-r1/fp16_neon.so')
    p.add_argument('--expected-npu-hz', type=int, choices=(300000000, 1000000000), default=1000000000)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    compatible = Path('/proc/device-tree/compatible').read_bytes()
    if platform.machine() != 'aarch64' or b'rk3588' not in compatible:
        raise RuntimeError('Actual native FC2 execution is board-only')
    if sha(args.native_helper) != NATIVE_HELPER_SHA:
        raise ValueError('Frozen native A/C helper differs')
    n = module(args.native_helper, '_v81_frozen_native_context')
    repo, saved, capture = (x.resolve(strict=True) for x in (args.repo, args.saved_dir, args.capture_dir))
    hp = repo / 'deploy/rk3588/validation/original_native_mlp/native_mlp_rank3_probe_board.py'
    if sha(hp) != CAPTURE_HELPER_SHA:
        raise ValueError('Frozen v68 capture helper differs')
    h = module(hp, '_v81_fc2_capture_helpers')
    pins = {args.native_helper.resolve(): NATIVE_HELPER_SHA, hp: CAPTURE_HELPER_SHA}
    def pin(path, digest, size=None):
        file = h.pin(path, digest, size)
        pins[file] = digest
        return file
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='v81_original_frame0_FC2_three_single_core_shared_A_partitions', status='failed',
        source_sha256=sha(__file__), original_model_only=True, checkpoint=n.CHECKPOINT,
        M=106, K=4096, N=1024, B_layout=1, AC_layout=1, type=1,
        candidate_accepted=False, full_model_or_SLAM_validated=False,
        source_original_FP32_input_used=True, original_bias_added_once=True,
        K_reduction_partitioned=False, output_channel_ranges=[list(x) for x in RANGES],
        all_106_rows_retained=True, no_padding=True, frequency_written=False,
        expected_npu_hz=args.expected_npu_hz, first_calls_retained=True,
        routes={'A': 'unsplit_native_AC1_mask1', 'B': 'three_native_AC1_parts_masks1_2_4_shared_A'},
        shared_A_owner='unsplit_context', shared_A_allocations_consumed_by_parts=1,
        unused_part_owned_A_allocations_retained=3, shared_A_cross_context_support_assumed=False,
        no_CPU_writes_during_parallel_runs=True, shared_A_input_copy_count_per_B_call=1,
        shared_A_input_TO_DEVICE_sync_count_per_B_call=1,
        timing_scope='FP16 conversion once per call; B copies to owner A once and syncs once, then three shared-pointer binds/runs/C sync/owned copies/bias, dispatch/join, concatenation and finite checks; validation, hashes and file writes outside.',
        limits=['This is an original FC2 component, not full original VGGT or SLAM throughput.',
            'Output-channel partition preserves complete K products and learned values; actual NPU rounding need not be bitwise identical.',
            'Three explicit single-core contexts do not use the unsupported combined-core MatMul mask; no fallback.',
            'Cross-context A binding is tested, not guaranteed. The owner remains live until all part contexts are closed.'],
        calls=[], cleanup_errors=[])
    contexts, counted, executor = [], [], None
    original_affinity = os.sched_getaffinity(0)
    error = None
    try:
        if not {4, 5, 6, 7} <= original_affinity:
            raise RuntimeError('CPUs4..7 unavailable')
        os.sched_setaffinity(0, {4, 5, 6, 7})
        import numpy as np
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
            file = pin(h.child(directory, item['file']), item['sha256'], item['bytes'])
            value = np.load(file, allow_pickle=False)
            if (str(value.dtype) != item['dtype'] or list(value.shape) != item['shape']
                    or h.raw_sha(value) != item['raw_sha256'] or not value.flags.c_contiguous
                    or not np.isfinite(value).all()):
                raise ValueError('Saved original array changed')
            return value
        x = array(saved, old['artifacts']['torch_fc2_input'])
        reference = array(saved, old['artifacts']['torch_output'])
        if x.dtype != np.float32 or x.shape != (106, 4096) or reference.shape != (106, 1024):
            raise ValueError('All original FC2 rows/channels required')
        row = next(item for item in cap['blocks'] if item['role'] == 'frame' and item['index'] == 0)
        wp = pin(h.child(capture, row['weights']['file']), row['weights']['sha256'], row['weights']['bytes'])
        with np.load(wp, allow_pickle=False) as archive:
            if set(archive.files) != set(row['parameters']):
                raise ValueError('Original parameter archive changed')
            weight, bias = (np.ascontiguousarray(archive[key]) for key in ('mlp.fc2.weight', 'mlp.fc2.bias'))
        for key, value, shape in (('mlp.fc2.weight', weight, (1024, 4096)), ('mlp.fc2.bias', bias, (1024,))):
            meta = row['parameters'][key]
            if value.dtype != np.float32 or value.shape != shape or h.raw_sha(value) != meta['raw_sha256'] or not np.isfinite(value).all():
                raise ValueError('Original FC2 parameter changed')
        report['reference'] = dict(saved_receipt_sha256=n.SAVED_SHA, capture_sha256=n.CAPTURE_SHA,
            input=old['artifacts']['torch_fc2_input'], output=old['artifacts']['torch_output'],
            original_FC2_parameters={key: row['parameters'][key] for key in ('mlp.fc2.weight', 'mlp.fc2.bias')})
        slices = [(np.ascontiguousarray(weight[start:end]), np.ascontiguousarray(bias[start:end])) for start, end, _ in RANGES]
        if np.concatenate([v[0] for v in slices]).tobytes() != weight.tobytes() or np.concatenate([v[1] for v in slices]).tobytes() != bias.tobytes():
            raise ValueError('Original channel slices failed exact reconstruction')
        report['original_weight_and_bias_slices_reconstruct_bytewise'] = True
        pin(args.runtime_library, n.RUNTIME_SHA)
        neon = ct.CDLL(str(pin(args.neon_library, n.NEON_SHA)))
        neon.f32_to_f16.argtypes, neon.f32_to_f16.restype = [ct.c_void_p, ct.c_void_p, ct.c_size_t], None
        sys.path[:0] = [str(repo), str(repo / 'third_party/vggt')]
        abi = n.load_module(pin(repo / 'vggt_slam/rknn_linear.py', n.ABI_SHA), '_v81_fc2_native_abi')
        if (ct.sizeof(abi.RKNNMatmulInfo), ct.sizeof(abi.RKNNMatmulIOAttr), ct.sizeof(abi.RKNNTensorMem)) != (64, 996, 40):
            raise RuntimeError('aarch64 ABI size mismatch')
        api = abi._API(str(args.runtime_library))
        started = time.perf_counter()
        # Independent counters are accessed by one context at a time, so SDK
        # calls may overlap without racing shared Python accounting.
        for name, channels, mask, w in [('unsplit', 1024, 1, weight)] + [
                (f'part{i}', end-start, mask, slices[i][0]) for i, (start, end, mask) in enumerate(RANGES)]:
            calls = n.Counted(api)
            counted.append(calls)
            context = n.NativeContext(abi, calls, np, 4096, channels, True, mask)
            contexts.append(context)  # partial creation/allocation always remains closeable
            context.open(w)
            context.record['label'] = name
        report['setup_seconds'] = time.perf_counter() - started
        common_A = contexts[0].memories['A']
        owner_attr = contexts[0].record['A']
        if any(context.record['A'] != owner_attr for context in contexts[1:]):
            raise RuntimeError('All shared A descriptors must be identical')
        report['shared_A_descriptor'] = owner_attr
        report['shared_A_owner_memory'] = dict(size=int(common_A.contents.size),
            fd=int(common_A.contents.fd), offset=int(common_A.contents.offset))
        executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix='original-fc2')
        def run_part(context, prepared_native_a, original_bias, use_shared=False):
            start = last = time.perf_counter()
            if use_shared:
                abi._check(context.api.rknn_matmul_set_io_mem(context.ctx, common_A,
                    ct.byref(context.attrs.A)), 'FC2 shared owner-A binding')
                times = dict(shared_A_bind_seconds=time.perf_counter() - last,
                    A_copy_seconds=0., A_TO_DEVICE_sync_count=0, shared_A_used=True)
            else:
                np.copyto(context.views['A'], prepared_native_a, casting='no')
                times = dict(A_copy_seconds=time.perf_counter() - last, shared_A_used=False)
                last = time.perf_counter()
                context.sync('A', abi.RKNN_MEMORY_SYNC_TO_DEVICE)
                context.bind('A')
                times['A_sync_bind_seconds'] = time.perf_counter() - last
            run_start = time.perf_counter()
            abi._check(context.api.rknn_matmul_run(context.ctx), 'FC2 matmul_run')
            run_end = time.perf_counter()
            times['MatMul_run_seconds'] = run_end - run_start
            last = time.perf_counter()
            value = context.owned()
            value += original_bias
            if not np.isfinite(value).all():
                raise RuntimeError('Nonfinite original FC2 result')
            times.update(C_sync_owned_bias_finite_seconds=time.perf_counter() - last,
                worker_start_monotonic=start, run_start_monotonic=run_start, run_end_monotonic=run_end,
                worker_end_monotonic=time.perf_counter(), core_mask=context.mask)
            return value, times
        previous = {}
        schedule = list('AB' + 'ABBA' * 6)
        report['schedule'] = schedule
        for index, route in enumerate(schedule):
            before = n.platform_observation(args.expected_npu_hz)
            begin = time.perf_counter()
            half = np.empty(x.shape, np.float16)
            neon.f32_to_f16(x.ctypes.data, half.ctypes.data, x.size)
            if not np.isfinite(half).all():
                raise RuntimeError('Original input FP16 overflow')
            native_a = half.reshape(106, 512, 8).transpose(1, 0, 2)
            half.flags.writeable = False
            conversion_seconds = time.perf_counter() - begin
            if route == 'A':
                value, times = run_part(contexts[0], native_a, bias)
                workers = [times]
                shared_copy_seconds = shared_sync_seconds = 0.
            else:
                # The sole CPU write/flush completes before any part is bound or
                # launched. Do not touch owner A again until every future ends.
                last = time.perf_counter()
                np.copyto(contexts[0].views['A'], native_a, casting='no')
                shared_copy_seconds = time.perf_counter() - last
                last = time.perf_counter()
                contexts[0].sync('A', abi.RKNN_MEMORY_SYNC_TO_DEVICE)
                shared_sync_seconds = time.perf_counter() - last
                futures = [executor.submit(run_part, contexts[i+1], None, slices[i][1], True) for i in range(3)]
                wait(futures)  # every submitted call finishes before any failure escapes or cleanup starts
                errors = [future.exception() for future in futures if future.exception() is not None]
                if errors:
                    raise RuntimeError('Explicit FC2 partition failed: ' + '; '.join(str(e) for e in errors)) from errors[0]
                results = [future.result() for future in futures]
                value = np.concatenate([result[0] for result in results], axis=1)
                workers = [result[1] for result in results]
            if value.dtype != np.float32 or value.shape != (106, 1024) or not value.flags.c_contiguous or not np.isfinite(value).all():
                raise RuntimeError('Finite owned FP32 full FC2 output required')
            seconds = time.perf_counter() - begin
            after = n.platform_observation(args.expected_npu_hz)
            first = route not in previous
            file = out / f'call-{index:02d}-{route}.npy'
            with file.open('xb') as stream:
                np.save(stream, value, allow_pickle=False)
            report['calls'].append(dict(index=index, route=route, first_for_route=first,
                complete_route_seconds=seconds, input_conversion_once_seconds=conversion_seconds,
                shared_A_copy_seconds=shared_copy_seconds, shared_A_TO_DEVICE_sync_seconds=shared_sync_seconds,
                workers=workers, platform_before=before, platform_after=after,
                vs_original_CPU_FP32=h.compare(value, reference),
                vs_first_unsplit=h.compare(value, previous.get('A', value)),
                bitwise_equal_previous_same_route=None if first else value.tobytes() == previous[route].tobytes(),
                output=dict(file=file.name, bytes=file.stat().st_size, sha256=sha(file),
                    raw_sha256=h.raw_sha(value), shape=list(value.shape), dtype='float32')))
            previous[route] = value.copy()
            print(json.dumps(dict(index=index, route=route, seconds=seconds)), flush=True)
        report['summary_seconds'] = {route: dict(all_count=13, warm_count=12,
            warm_mean=statistics.mean(c['complete_route_seconds'] for c in report['calls'] if c['route'] == route and not c['first_for_route']),
            warm_median=statistics.median(c['complete_route_seconds'] for c in report['calls'] if c['route'] == route and not c['first_for_route']),
            all_mean=statistics.mean(c['complete_route_seconds'] for c in report['calls'] if c['route'] == route)) for route in 'AB'}
        report['status'] = 'completed_original_FC2_component_not_quality_accepted'
    except BaseException as exc:
        error = exc
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['exception_notes'] = list(getattr(exc, '__notes__', []))
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        report['contexts_after_close'] = []
        for context in contexts[::-1]:
            try:
                state = context.close()
                report['contexts_after_close'].append(state)
                report['cleanup_errors'].extend(state['cleanup_errors'])
            except BaseException as exc:
                report['cleanup_errors'].append(str(exc))
        report['SDK_API'] = [dict(calls=c.calls, successes=c.successes, return_codes=c.codes) for c in counted]
        if report['status'].startswith('completed'):
            for counters, runs in zip(counted, (13, 13, 13, 13)):
                if not (counters.successes.get('rknn_matmul_create', 0) == counters.successes.get('rknn_matmul_destroy', 0) == 1
                        and counters.successes.get('rknn_create_mem', 0) == counters.successes.get('rknn_destroy_mem', 0) == 3
                        and counters.successes.get('rknn_matmul_run', 0) == runs
                        and all(code == 0 for codes in counters.codes.values() for code in codes)):
                    report['cleanup_errors'].append('Expected context/memory/run counts differ')
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
    if report['status'] != 'completed_original_FC2_component_not_quality_accepted':
        raise RuntimeError(report['status'])


if __name__ == '__main__':
    main()
