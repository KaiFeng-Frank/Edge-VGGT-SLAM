#!/usr/bin/env python3
"""Original frame0 MLP: compiled/core7, normal/core1, native/core1, optional GPU Erf.

Actual execution is RK3588S-only. No fallback, changed model, padding or hidden
warmup. Every route retains its first call and measures twelve subsequent calls.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

BASE = Path('/mnt/vggt-data/vggt-slam2')
EXPERIMENTS = BASE / 'experiments'
CAPTURE_SHA = '4b723e499763bb90565f6e11e16c551664634aaf72da46c7cdc493b84208ef67'
SAVED_SHA = 'b3918176bd02002831b92158b148bbd36fc208132c00ac8e0f44723db1af00a3'
HELPER_SHA = '17be747bf865268c3e6182abe9d4852f7f0199ddead1e3c2218dfb55111b395b'
COMPILED_SHA = 'f055e45eec073557e85f00f6ce3fab30e6ee212c1f32d928d9ac0c40e6055bc3'
ABI_SHA = '69056a2cb44ba69a2d311265b71ff919e4758fcaa150834a0b0cfe85dbcaef7d'
RUNTIME_SHA = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
NEON_SHA = 'd08147c438461669ad8ebd3c6483e0a7908053e95cc6f83d0fe7eb7a5cd668c9'
CHECKPOINT = dict(sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0', bytes=5026874952)
M = 106


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic(path, report):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def platform_observation(expected_hz):
    dev = Path('/sys/class/devfreq/fdab0000.npu')
    hz = int((dev / 'cur_freq').read_text())
    if hz != expected_hz:
        raise RuntimeError(f'Observed NPU frequency {hz}, expected {expected_hz}; no frequency writes')
    return dict(npu_hz=hz, temperature_c={p.parent.name: int(p.read_text()) / 1000.
        for p in sorted(Path('/sys/class/thermal').glob('thermal_zone*/temp'))})


class Counted:
    def __init__(self, api):
        self.api, self.calls, self.successes, self.codes = api, {}, {}, {}

    def __getattr__(self, name):
        function = getattr(self.api, name)
        def call(*args):
            self.calls[name] = self.calls.get(name, 0) + 1
            result = function(*args)
            pointer = name == 'rknn_create_mem'
            if not pointer:
                self.codes.setdefault(name, []).append(int(result))
            if bool(result) if pointer else int(result) == 0:
                self.successes[name] = self.successes.get(name, 0) + 1
            return result
        return call


class NativeContext:
    """Pinned ctypes ABI, with queried native A8/C4 instead of normal A/C."""
    def __init__(self, abi, api, np, k, n, native, mask):
        self.abi, self.api, self.np = abi, api, np
        self.k, self.n, self.native, self.mask = k, n, native, mask
        self.ctx, self.attrs = abi.RKNNContext(), abi.RKNNMatmulIOAttr()
        self.memories, self.views, self.record = {}, {}, {}
        self.cleanup_errors = []

    def open(self, weight):
        abi, api, np = self.abi, self.api, self.np
        info = abi.RKNNMatmulInfo(M=M, K=self.k, N=self.n,
            type=abi.RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32, B_layout=1,
            AC_layout=int(self.native), AC_quant_type=0)
        abi._check(api.rknn_matmul_create(ct.byref(self.ctx), ct.byref(info), ct.byref(self.attrs)), 'matmul_create')
        if not self.ctx.value:
            raise RuntimeError('Null MatMul context')
        abi._check(api.rknn_matmul_set_core_mask(self.ctx, self.mask), 'matmul_set_core_mask')
        dims = dict(A=(self.k // 8, M, 8) if self.native else (M, self.k),
            B=(self.n // 16, self.k // 32, 16, 32),
            C=(self.n // 4, M, 4) if self.native else (M, self.n))
        for name in 'ABC':
            attr = getattr(self.attrs, name)
            shape = tuple(attr.dims[:attr.n_dims])
            dtype = np.float32 if name == 'C' else np.float16
            expected_bytes = math.prod(dims[name]) * np.dtype(dtype).itemsize
            if shape != dims[name] or attr.type != (0 if name == 'C' else 1) or attr.size < expected_bytes:
                raise RuntimeError(f'Unsupported actual {name} descriptor: {shape}/{attr.type}/{attr.size}')
            self.record[name] = dict(shape=list(shape), type=int(attr.type), size=int(attr.size), logical_bytes=expected_bytes)
            pointer = api.rknn_create_mem(self.ctx, attr.size)
            if not pointer:
                raise RuntimeError(f'Null {name} allocation')
            self.memories[name] = pointer
            memory = pointer.contents
            if not memory.virt_addr or memory.size < attr.size:
                raise RuntimeError(f'Invalid {name} memory')
            if name != 'C':
                ct.memset(memory.virt_addr, 0, attr.size)
            raw = (ct.c_uint8 * attr.size).from_address(memory.virt_addr)
            self.views[name] = np.frombuffer(raw, dtype=dtype, count=math.prod(shape)).reshape(shape)
            self.bind(name)
        packed = np.ascontiguousarray(weight.T.astype(np.float16).reshape(
            self.k // 32, 32, self.n // 16, 16).transpose(2, 0, 3, 1))
        if not np.isfinite(packed).all():
            raise RuntimeError('Original weight FP16 overflow')
        np.copyto(self.views['B'], packed, casting='no')
        self.sync('B', abi.RKNN_MEMORY_SYNC_TO_DEVICE)
        self.bind('B')

    def sync(self, name, direction):
        self.abi._check(self.api.rknn_mem_sync(self.ctx, self.memories[name], direction), f'mem_sync {name}/{direction}')

    def bind(self, name):
        self.abi._check(self.api.rknn_matmul_set_io_mem(self.ctx, self.memories[name],
            ct.byref(getattr(self.attrs, name))), 'set_io_mem ' + name)

    def submit(self, value, neon):
        np = self.np
        half = np.empty((M, self.k), np.float16)
        neon.f32_to_f16(value.ctypes.data, half.ctypes.data, value.size)
        if not np.isfinite(half).all():
            raise RuntimeError('Activation FP16 overflow')
        layout = half.reshape(M, self.k // 8, 8).transpose(1, 0, 2) if self.native else half
        np.copyto(self.views['A'], layout, casting='no')
        self.run_prepared()

    def run_prepared(self):
        self.sync('A', self.abi.RKNN_MEMORY_SYNC_TO_DEVICE)
        self.bind('A')
        self.abi._check(self.api.rknn_matmul_run(self.ctx), 'matmul_run')

    def owned(self):
        self.sync('C', self.abi.RKNN_MEMORY_SYNC_FROM_DEVICE)
        view = self.views['C']
        normal = view.transpose(1, 0, 2).reshape(M, self.n) if self.native else view
        return normal.copy(order='C')

    def close(self):
        self.views.clear()
        for name, pointer in list(self.memories.items())[::-1]:
            code = self.api.rknn_destroy_mem(self.ctx, pointer)
            if code:
                self.cleanup_errors.append(f'destroy_mem {name} returned {code}')
            else:
                del self.memories[name]
        if self.ctx.value:
            code = self.api.rknn_matmul_destroy(self.ctx)
            if code:
                self.cleanup_errors.append(f'matmul_destroy returned {code}')
            else:
                self.ctx.value = 0
        return dict(core_mask=self.mask, AC_layout=int(self.native), descriptors=self.record,
            live_context=int(bool(self.ctx.value)), unresolved_memory_count=len(self.memories), cleanup_errors=self.cleanup_errors)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=BASE / 'src')
    p.add_argument('--capture-dir', type=Path, default=EXPERIMENTS / 'vggt-v53-capture-r1')
    p.add_argument('--saved-dir', type=Path, default=EXPERIMENTS / 'vggt-v68-native-compiled-rank3-ab-r1')
    p.add_argument('--runtime-library', type=Path, default=Path('/usr/lib/librknnrt.so'))
    p.add_argument('--neon-library', type=Path, default=EXPERIMENTS / 'vggt-v56-neon-build-r1/fp16_neon.so')
    p.add_argument('--gpu-bridge-source', type=Path)
    p.add_argument('--gpu-bridge-sha256')
    p.add_argument('--gpu-library', type=Path)
    p.add_argument('--gpu-library-sha256')
    p.add_argument('--expected-npu-hz', type=int, choices=(300000000, 1000000000), default=1000000000)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    gpu_args = (args.gpu_bridge_source, args.gpu_bridge_sha256, args.gpu_library, args.gpu_library_sha256)
    if any(gpu_args) and not all(gpu_args):
        p.error('GPU route requires source/library paths and both SHA256 values')
    compatible = Path('/proc/device-tree/compatible').read_bytes().replace(b'\0', b',').decode()
    if platform.machine() != 'aarch64' or 'rockchip,rk3588' not in compatible:
        raise RuntimeError('Actual SDK/MLP execution is board-only')
    repo, capture, saved = (x.resolve(strict=True) for x in (args.repo, args.capture_dir, args.saved_dir))
    helper_path = repo / 'deploy/rk3588/validation/original_native_mlp/native_mlp_rank3_probe_board.py'
    import hashlib
    if hashlib.sha256(helper_path.read_bytes()).hexdigest() != HELPER_SHA:
        raise ValueError('Frozen saved-capture helper differs')
    h = load_module(helper_path, '_v81_original_v68_helpers')
    pins = {helper_path: HELPER_SHA}
    def pin(path, digest, size=None):
        file = h.pin(path, digest, size)
        pins[file] = digest
        return file
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='v81_original_frame0_MLP_native_AC_GPU_Erf_diagnostic', status='failed',
        script_sha256=h.sha(__file__), original_model_only=True, checkpoint=CHECKPOINT,
        candidate_accepted=False, full_model_or_SLAM_validated=False, strict_rtol=1e-4, strict_atol=1e-5,
        shape=[M, 1024, 4096, 1024], no_rows_tokens_or_layers_omitted=True,
        expected_npu_hz=args.expected_npu_hz, frequency_written=False, first_calls_retained=True,
        warm_calls_per_route=12, calls=[], cleanup_errors=[], GPU_requested=bool(args.gpu_library),
        timing_scope='Each call includes conversions/packing/sync/bind/MatMul/GELU/owned output/bias/finite; hashes, comparisons and file writes are outside. One-time setup separate.',
        routes=dict(A='original_rank3_compiled_FP16_core7', B='native_B_normal_AC_core1_original_CPU_GELU',
            C='native_B_native_AC1_core1_original_CPU_GELU'),
        limits=['Normal/core1 to native/core1 isolates A/C layout; compiled baseline uses core7.',
            'core_mask7 request does not prove three-core utilization. Native and compiled arithmetic/rounding can differ.',
            'OpenCL Erf must be validated against original CPU outputs; no fast GELU, silent CPU fallback or equivalence claim from static math.'])
    original_affinity = os.sched_getaffinity(0)
    contexts, compiled, bridge, counted, np = [], None, None, None, None
    previous = {}
    error = None
    try:
        if not {4, 5, 6, 7} <= original_affinity:
            raise RuntimeError('Requested process CPU affinity unavailable')
        os.sched_setaffinity(0, {4, 5, 6, 7})
        import numpy as np
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        report['versions'] = dict(numpy=np.__version__, torch=torch.__version__)
        report['cpu_affinity'], report['torch_threads'] = [4, 5, 6, 7], 1
        old = json.loads(pin(saved / 'result.json', SAVED_SHA).read_text())
        cap = json.loads(pin(capture / 'capture.json', CAPTURE_SHA).read_text())
        if (old['status'] != 'completed_component_diagnostic_not_quality_accepted'
                or old['capture_sha256'] != CAPTURE_SHA or old['checkpoint'] != CHECKPOINT
                or not old['reconstructed_block_vs_original_full_forward']['bitwise_equal']
                or cap['checkpoint'] != CHECKPOINT or cap['cleanup_errors']
                or cap['full_model_forward_completed'] is not True):
            raise ValueError('Completed original capture/reference identity required')
        for name, meta in old['sources'].items():
            pin(repo / name, meta['sha256'], meta['bytes'])
        row = next(x for x in cap['blocks'] if x['role'] == 'frame' and x['index'] == 0)
        if row['gelu_approximate'] != 'none':
            raise ValueError('Original exact Erf GELU required')
        def array(directory, item):
            file = pin(h.child(directory, item['file']), item['sha256'], item['bytes'])
            value = np.load(file, allow_pickle=False)
            if (str(value.dtype) != item['dtype'] or list(value.shape) != item['shape']
                    or not value.flags.c_contiguous or h.raw_sha(value) != item['raw_sha256']
                    or not np.isfinite(value).all()):
                raise ValueError('Captured array changed')
            return value
        x, reference = (array(saved, old['artifacts'][k]) for k in ('input', 'torch_output'))
        if x.dtype != np.float32 or x.shape != (M, 1024) or reference.shape != x.shape:
            raise ValueError('Original all-106-row input/output required')
        report['reference'] = dict(saved_receipt_sha256=SAVED_SHA, capture_sha256=CAPTURE_SHA,
            input=old['artifacts']['input'], original_CPU_output=old['artifacts']['torch_output'])
        wp = pin(h.child(capture, row['weights']['file']), row['weights']['sha256'], row['weights']['bytes'])
        with np.load(wp, allow_pickle=False) as archive:
            weights = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
        if set(weights) != set(row['parameters']) or len(weights) != 18:
            raise ValueError('All original frame0 parameter identities required')
        for key, value in weights.items():
            meta = row['parameters'][key]
            if value.dtype != np.float32 or list(value.shape) != meta['shape'] or h.raw_sha(value) != meta['raw_sha256'] or not np.isfinite(value).all():
                raise ValueError('Original parameter changed: ' + key)
        report['original_parameter_metadata'] = row['parameters']
        for key, shape in {'mlp.fc1.weight': (4096, 1024), 'mlp.fc1.bias': (4096,),
                'mlp.fc2.weight': (1024, 4096), 'mlp.fc2.bias': (1024,)}.items():
            if weights[key].shape != shape:
                raise ValueError('Original MLP shape changed')
        pin(args.runtime_library, RUNTIME_SHA)
        neon = ct.CDLL(str(pin(args.neon_library, NEON_SHA)))
        for name in ('f32_to_f16', 'f16_to_f32'):
            f = getattr(neon, name)
            f.argtypes, f.restype = [ct.c_void_p, ct.c_void_p, ct.c_size_t], None
        sys.path[:0] = [str(repo), str(repo / 'third_party/vggt')]
        abi = load_module(pin(repo / 'vggt_slam/rknn_linear.py', ABI_SHA), '_v81_native_abi')
        if (ct.sizeof(abi.RKNNMatmulInfo), ct.sizeof(abi.RKNNMatmulIOAttr), ct.sizeof(abi.RKNNTensorMem)) != (64, 996, 40):
            raise RuntimeError('aarch64 typed ABI size mismatch')
        counted = Counted(abi._API(str(args.runtime_library)))
        tick = time.perf_counter()
        cp = pin(repo / 'deploy/rk3588/validation/original_native_mlp/compiled_mlp_rank3.py', COMPILED_SHA)
        compiled = load_module(cp, '_v81_compiled_mlp').CompiledMLP(repo, core_mask=7)
        pairs = {}
        for route, native, mask in (('B', False, 1), ('C', True, 1)):
            pair = []
            for i, (k, n) in enumerate(((1024, 4096), (4096, 1024)), 1):
                context = NativeContext(abi, counted, np, k, n, native, mask)
                contexts.append(context)
                context.open(weights[f'mlp.fc{i}.weight'])
                pair.append(context)
            pairs[route] = pair
        report['setup_seconds'] = time.perf_counter() - tick
        if args.gpu_library:
            module = load_module(pin(args.gpu_bridge_source, args.gpu_bridge_sha256), '_v81_gpu_bridge')
            library = pin(args.gpu_library, args.gpu_library_sha256)
            bridge = module.NativeErfGeluBridge(pairs['C'][0].memories['C'], pairs['C'][1].memories['A'],
                weights['mlp.fc1.bias'], rows=M, channels=4096, library=str(library), managed_host_consistency=True)
            report['routes']['D'] = 'native_B_native_AC1_core1_GPU_original_Erf_GELU'
        gelu = torch.nn.GELU(approximate='none').eval()
        def native_cpu(pair):
            stages = {}
            begin = time.perf_counter()
            pair[0].submit(x, neon)
            first = pair[0].owned()
            first += weights['mlp.fc1.bias']
            stages['fc1_conversion_pack_sync_run_owned_bias'] = time.perf_counter() - begin
            begin = time.perf_counter()
            with torch.inference_mode():
                middle = np.ascontiguousarray(gelu(torch.from_numpy(first)).numpy())
            stages['original_CPU_Erf_GELU'] = time.perf_counter() - begin
            begin = time.perf_counter()
            pair[1].submit(middle, neon)
            value = pair[1].owned()
            value += weights['mlp.fc2.bias']
            stages['fc2_conversion_pack_sync_run_owned_bias'] = time.perf_counter() - begin
            return value, stages
        def native_gpu():
            first, second = pairs['C']
            begin = time.perf_counter()
            first.submit(x, neon)
            stages = dict(fc1_conversion_pack_sync_run=time.perf_counter() - begin)
            begin = time.perf_counter()
            bridge.run(before_input=lambda: first.sync('C', abi.RKNN_MEMORY_SYNC_FROM_DEVICE),
                before_output=lambda: second.sync('A', abi.RKNN_MEMORY_SYNC_TO_DEVICE),
                after_output=lambda: second.sync('A', abi.RKNN_MEMORY_SYNC_FROM_DEVICE))
            stages['GPU_bias_Erf_pack_and_all_sync'] = time.perf_counter() - begin
            begin = time.perf_counter()
            second.run_prepared()
            value = second.owned()
            value += weights['mlp.fc2.bias']
            stages['fc2_sync_run_owned_bias'] = time.perf_counter() - begin
            return value, stages
        routes = list(report['routes'])
        schedule = routes + [route for j in range(12) for route in (routes if j % 2 == 0 else routes[::-1])]
        report['schedule'] = schedule
        for index, route in enumerate(schedule):
            before = platform_observation(args.expected_npu_hz)
            started = time.perf_counter()
            if route == 'A':
                y, stages = compiled.run(x), {}
            elif route in ('B', 'C'):
                y, stages = native_cpu(pairs[route])
            else:
                y, stages = native_gpu()
            if y.dtype != np.float32 or y.shape != (M, 1024) or not y.flags.c_contiguous or not np.isfinite(y).all():
                raise RuntimeError('Finite owned dense FP32 output required')
            elapsed = time.perf_counter() - started
            after = platform_observation(args.expected_npu_hz)
            first_call = route not in previous
            item = dict(index=index, route=route, first_for_route=first_call, complete_route_seconds=elapsed,
                stages_seconds=stages, platform_before=before, platform_after=after,
                vs_original_CPU_FP32=h.compare(y, reference), output_raw_sha256=h.raw_sha(y),
                bitwise_equal_previous_same_route=None if first_call else y.tobytes() == previous[route].tobytes())
            if route != 'A':
                item['vs_first_compiled_output'] = h.compare(y, previous['A'])
            if route == 'D':
                item['vs_native_AC1_CPU_GELU'] = h.compare(y, previous['C'])
            file = out / f'call-{index:02d}-{route}.npy'
            with file.open('xb') as stream:
                np.save(stream, y, allow_pickle=False)
            item['output'] = dict(file=file.name, bytes=file.stat().st_size, sha256=h.sha(file),
                raw_sha256=h.raw_sha(y), shape=list(y.shape), dtype='float32')
            report['calls'].append(item)
            previous[route] = y.copy()
            print(json.dumps(dict(index=index, route=route, seconds=elapsed)), flush=True)
        report['summary_seconds'] = {route: dict(
            all_count=13, warm_count=12,
            all_mean=statistics.mean(x['complete_route_seconds'] for x in report['calls'] if x['route'] == route),
            warm_mean=statistics.mean(x['complete_route_seconds'] for x in report['calls'] if x['route'] == route and not x['first_for_route']),
            warm_median=statistics.median(x['complete_route_seconds'] for x in report['calls'] if x['route'] == route and not x['first_for_route'])) for route in routes}
        report['status'] = 'completed_component_diagnostic_not_quality_accepted'
    except BaseException as exc:
        error = exc
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['exception_notes'] = list(getattr(exc, '__notes__', []))
    finally:
        if bridge is not None:
            try:
                bridge.close()
                report['GPU_metrics_after_close'] = bridge.metrics()
            except BaseException as exc:
                report['cleanup_errors'].append('GPU close: ' + str(exc))
        report['native_contexts_after_close'] = []
        for context in contexts[::-1]:
            try:
                state = context.close()
                report['native_contexts_after_close'].append(state)
                report['cleanup_errors'].extend(state['cleanup_errors'])
            except BaseException as exc:
                report['cleanup_errors'].append('Native close: ' + str(exc))
        if compiled is not None:
            try:
                compiled.close()
            except BaseException as exc:
                report['cleanup_errors'].append('Compiled close: ' + str(exc))
            report['compiled_metrics_after_close'] = compiled.metrics()
        if counted is not None:
            report['native_API'] = dict(calls=counted.calls, successes=counted.successes, return_codes=counted.codes)
            expected_runs = 2 * sum(c['route'] != 'A' for c in report['calls'])
            report['native_lifecycle_balanced'] = (
                counted.successes.get('rknn_matmul_create', 0) == counted.successes.get('rknn_matmul_destroy', 0) == 4
                and counted.successes.get('rknn_create_mem', 0) == counted.successes.get('rknn_destroy_mem', 0) == 12
                and counted.successes.get('rknn_matmul_run', 0) == expected_runs
                and all(code == 0 for codes in counted.codes.values() for code in codes)
                and not any(x['live_context'] or x['unresolved_memory_count'] for x in report['native_contexts_after_close']))
            if report['status'].startswith('completed') and not report['native_lifecycle_balanced']:
                report['cleanup_errors'].append('Native lifecycle counts differ')
        try:
            os.sched_setaffinity(0, original_affinity)
            report['affinity_restored'] = os.sched_getaffinity(0) == original_affinity
        except BaseException as exc:
            report['affinity_restored'] = False
            report['cleanup_errors'].append('Affinity restore: ' + str(exc))
        try:
            report['sources_and_inputs_unchanged'] = all(h.sha(file) == digest for file, digest in pins.items())
            report['pinned_files'] = {str(file): digest for file, digest in pins.items()}
        except BaseException as exc:
            report['sources_and_inputs_unchanged'] = False
            report['cleanup_errors'].append('Rehash: ' + str(exc))
        if report['cleanup_errors'] or not report['affinity_restored'] or not report['sources_and_inputs_unchanged']:
            report['status'] = 'failed_cleanup_or_identity'
        atomic(out / 'result.json', report)
        print(json.dumps(dict(status=report['status'], result_sha256=h.sha(out / 'result.json'))), flush=True)
    if error:
        raise error
    if report['status'] != 'completed_component_diagnostic_not_quality_accepted':
        raise RuntimeError(report['status'])


if __name__ == '__main__':
    main()
