#!/usr/bin/env python3
"""Board-only original frame0 FC1 -> original GELU -> FC2, two persistent native Bs."""
from __future__ import annotations
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

CAPTURE_SHA = '4b723e499763bb90565f6e11e16c551664634aaf72da46c7cdc493b84208ef67'
CHECKPOINT = {'sha256': 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0', 'bytes': 5026874952}
ABI_SHA = '69056a2cb44ba69a2d311265b71ff919e4758fcaa150834a0b0cfe85dbcaef7d'
RUNTIME_SHA = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
NEON_SHA = 'd08147c438461669ad8ebd3c6483e0a7908053e95cc6f83d0fe7eb7a5cd668c9'
NEON_SOURCE_SHA = '77707b735a6291c417a997ed64acbe4e1cd60cc5f9dd150d83fe1836868bd5e3'
COMPILED_HELPER_SHA = 'f055e45eec073557e85f00f6ce3fab30e6ee212c1f32d928d9ac0c40e6055bc3'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def pin(path, expected, size=None):
    path = Path(path).resolve(strict=True)
    if (size is not None and path.stat().st_size != size) or sha(path) != expected:
        raise ValueError('Identity changed: ' + str(path))
    return path


def child(root, name):
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Artifact outside its capture')
    return path


def raw_sha(array):
    return hashlib.sha256(memoryview(array).cast('B')).hexdigest()


def compare(actual, reference):
    import numpy as np
    if actual.shape != reference.shape or actual.dtype != np.float32 or not np.isfinite(actual).all():
        raise ValueError('Invalid comparison array')
    delta = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    allowance = 1e-5 + 1e-4 * np.abs(reference.astype(np.float64))
    return {'rtol': 1e-4, 'atol': 1e-5, 'strict_pass': bool((delta <= allowance).all()),
            'outside': int((delta > allowance).sum()), 'elements': int(actual.size),
            'max_abs': float(delta.max()), 'relative_l2': float(np.linalg.norm(delta) /
            max(np.linalg.norm(reference.astype(np.float64)), np.finfo(np.float64).tiny)),
            'bitwise_equal': actual.tobytes() == reference.tobytes()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--capture-dir', type=Path, required=True)
    p.add_argument('--runtime-library', type=Path, required=True)
    p.add_argument('--neon-library', type=Path)
    p.add_argument('--compiled-mlp', type=Path)
    p.add_argument('--compiled-mlp-sha256')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--threads', type=int, default=1, choices=(1, 2, 4))
    p.add_argument('--cpu-cores', type=int, nargs='+', default=[6, 7])
    p.add_argument('--core-mask', type=int, default=1, choices=(0, 1, 2, 4))
    a = p.parse_args()
    if bool(a.compiled_mlp) != bool(a.compiled_mlp_sha256):
        p.error('--compiled-mlp and --compiled-mlp-sha256 must be provided together')
    if a.compiled_mlp and (a.compiled_mlp_sha256 != COMPILED_HELPER_SHA or a.core_mask != 1):
        p.error('Frozen compiled helper requires its pinned SHA and shared core-mask=1')
    compatible = Path('/proc/device-tree/compatible').read_bytes().replace(b'\0', b',').decode()
    if platform.machine() != 'aarch64' or 'rockchip,rk3588' not in compatible:
        raise RuntimeError('Original model/MatMul execution is RK3588/S-board-only')
    repo, capture = a.repo.resolve(strict=True), a.capture_dir.resolve(strict=True)
    original_affinity = os.sched_getaffinity(0)
    if not set(a.cpu_cores) <= original_affinity:
        raise ValueError('Requested CPU cores unavailable')
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'v68_original_VGGT_frame0_persistent_native_MLP', 'status': 'failed',
        'script_sha256': sha(__file__), 'capture_sha256': CAPTURE_SHA, 'checkpoint': CHECKPOINT,
        'board': {'machine': platform.machine(), 'device_tree_compatible': compatible},
        'candidate_accepted': False, 'full_model_or_replay_validated': False,
        'actual_execution_on_board_only': True, 'calibration_or_training': False,
        'original_model_only': True, 'threads': a.threads, 'cpu_affinity': a.cpu_cores,
        'type': 1, 'B_layout': 1, 'AC_layout': 0, 'core_mask': a.core_mask,
        'M': 106, 'no_token_or_row_padding': True, 'contexts': [], 'calls': [],
        'cleanup_errors': [], 'closed_contexts': 0, 'sources': {}, 'inputs': {}}
    contexts, hooks, counted, compiled = [], [], None, None
    try:
        os.sched_setaffinity(0, set(a.cpu_cores))
        import numpy as np
        import torch
        torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
        report['versions'] = {'torch': torch.__version__, 'numpy': np.__version__}
        manifest_path = pin(capture / 'capture.json', CAPTURE_SHA)
        original = json.loads(manifest_path.read_text())
        if (original['schema'] != 'v53_original_VGGT_48_block_capture'
                or original['status'] != 'completed_original_FP32_full_forward_and_48_block_capture'
                or original['checkpoint'] != CHECKPOINT or not original['full_model_forward_completed']
                or original['cleanup_errors'] or not original['actual_inference_on_board_only']):
            raise ValueError('Completed original pinned capture required')
        rows = [r for r in original['blocks'] if r['role'] == 'frame' and r['index'] == 0]
        if len(rows) != 1:
            raise ValueError('Unique original frame0 required')
        row = rows[0]
        # Inspect exact source files before importing the original class.
        sources = {'vggt_slam/rknn_linear.py': ABI_SHA}
        sources.update({k: v for k, v in original['source_hashes'].items()
                        if k.startswith('third_party/vggt/vggt/layers/')})
        for name, digest in sources.items():
            report['sources'][name] = {'sha256': digest, 'bytes': pin(repo / name, digest).stat().st_size}
        report['runtime_library'] = {'sha256': RUNTIME_SHA,
            'bytes': pin(a.runtime_library, RUNTIME_SHA).stat().st_size}
        neon = None
        if a.neon_library:
            pin(repo / 'vggt_slam/native/fp16_neon.c', NEON_SOURCE_SHA)
            lib = pin(a.neon_library, NEON_SHA)
            neon = ct.CDLL(str(lib))
            neon.f32_to_f16.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t]
            neon.f32_to_f16.restype = None
            report['conversion'] = {'route': 'frozen_NEON', 'library_sha256': NEON_SHA,
                'source_sha256': NEON_SOURCE_SHA, 'library_bytes': lib.stat().st_size}
        else:
            report['conversion'] = {'route': 'numpy_float16_cast'}
        sys.path[:0] = [str(repo), str(repo / 'third_party/vggt')]
        from vggt.layers.block import Block
        from vggt.layers.rope import RotaryPositionEmbedding2D
        spec = importlib.util.spec_from_file_location('_v68_frozen_matmul_abi', repo / 'vggt_slam/rknn_linear.py')
        abi = importlib.util.module_from_spec(spec); spec.loader.exec_module(abi)
        if (ct.sizeof(abi.RKNNMatmulInfo), ct.sizeof(abi.RKNNMatmulIOAttr), ct.sizeof(abi.RKNNTensorMem)) != (64, 996, 40):
            raise ValueError('Unexpected aarch64 ABI sizes')

        def load(item):
            file = pin(child(capture, item['file']), item['sha256'], item['bytes'])
            x = np.load(file, allow_pickle=False)
            if (str(x.dtype) != item['dtype'] or list(x.shape) != item['shape']
                    or not x.flags.c_contiguous or raw_sha(x) != item['raw_sha256'] or not np.isfinite(x).all()):
                raise ValueError('Captured tensor changed')
            report['inputs'][item['file']] = dict(item)
            return x

        def save(name, x):
            x = np.ascontiguousarray(x, dtype=np.float32)
            if not np.isfinite(x).all():
                raise ValueError('Nonfinite result')
            file = out / name
            with file.open('xb') as f:
                np.save(f, x, allow_pickle=False)
            return {'file': name, 'bytes': file.stat().st_size, 'sha256': sha(file),
                'shape': list(x.shape), 'dtype': 'float32', 'raw_sha256': raw_sha(x), 'tensor_bytes': x.nbytes}

        weights_path = pin(child(capture, row['weights']['file']), row['weights']['sha256'], row['weights']['bytes'])
        report['inputs'][row['weights']['file']] = row['weights']
        with np.load(weights_path, allow_pickle=False) as packed:
            weights = {key: np.ascontiguousarray(packed[key]) for key in packed.files}
        if set(weights) != set(row['parameters']) or len(weights) != 18:
            raise ValueError('All 18 original block parameters required')
        for key, value in weights.items():
            meta = row['parameters'][key]
            if (value.dtype != np.float32 or list(value.shape) != meta['shape']
                    or raw_sha(value) != meta['raw_sha256'] or not np.isfinite(value).all()):
                raise ValueError('Original parameter changed: ' + key)
        report['original_parameters'] = row['parameters']
        rope = RotaryPositionEmbedding2D(frequency=row['rope_frequency'], scaling_factor=row['rope_scaling_factor'])
        rope.frequency_cache[(32, 9, torch.device('cpu'), torch.float32)] = tuple(
            torch.from_numpy(load(original[key])) for key in ('rope_cos', 'rope_sin'))
        block = Block(dim=1024, num_heads=16, mlp_ratio=4, qk_norm=True, init_values=.01, rope=rope).eval()
        block.load_state_dict({key: torch.from_numpy(value) for key, value in weights.items()}, strict=True)
        for name, value in block.named_parameters():
            if raw_sha(value.detach().numpy()) != row['parameters'][name]['raw_sha256']:
                raise ValueError('Assigned original parameter changed')
        eps = {'norm1': block.norm1.eps, 'norm2': block.norm2.eps,
               'attn.q_norm': block.attn.q_norm.eps, 'attn.k_norm': block.attn.k_norm.eps}
        if eps != row['epsilons'] or block.mlp.act.approximate != 'none' or row['gelu_approximate'] != 'none':
            raise ValueError('Original normalization/GELU changed')
        captured = {}
        hooks.append(block.mlp.register_forward_pre_hook(lambda m, args: captured.update(input=args[0].detach().clone())))
        hooks.append(block.mlp.fc1.register_forward_hook(lambda m, args, y: captured.update(fc1=y.detach().clone())))
        hooks.append(block.mlp.fc2.register_forward_pre_hook(lambda m, args: captured.update(fc2_input=args[0].detach().clone())))
        hooks.append(block.mlp.register_forward_hook(lambda m, args, y: captured.update(output=y.detach().clone())))
        started = time.perf_counter()
        with torch.inference_mode():
            reconstructed = block(torch.from_numpy(load(row['input'])), pos=torch.from_numpy(load(row['positions'])))
        report['original_block_capture_seconds_including_hooks'] = time.perf_counter() - started
        for hook in hooks: hook.remove()
        hooks.clear()
        report['reconstructed_block_vs_original_full_forward'] = compare(reconstructed.numpy(), load(row['output']))
        if not report['reconstructed_block_vs_original_full_forward']['strict_pass']:
            raise ValueError('Original block reconstruction failed fixed FP32 gate; SDK not initialized')
        x = np.ascontiguousarray(captured['input'].numpy().reshape(106, 1024))
        if tuple(captured['input'].shape) != (2, 53, 1024):
            raise ValueError('All two-frame original tokens required')
        ref = np.ascontiguousarray(captured['output'].numpy().reshape(106, 1024))
        report['artifacts'] = {'input': save('input.npy', x), 'torch_output': save('torch-output.npy', ref),
            'torch_fc1': save('torch-fc1.npy', captured['fc1'].numpy().reshape(106, 4096)),
            'torch_fc2_input': save('torch-fc2-input.npy', captured['fc2_input'].numpy().reshape(106, 4096))}
        report['capture_original_input_shape'] = [2, 53, 1024]
        # A comparison that rounds operands to FP16 but accumulates on CPU in FP32;
        # this does not establish the NPU's internal arithmetic/rounding mode.
        w1, w2 = (torch.from_numpy(weights[f'mlp.fc{i}.weight'].astype(np.float16).astype(np.float32)) for i in (1, 2))
        b1, b2 = (torch.from_numpy(weights[f'mlp.fc{i}.bias']) for i in (1, 2))
        with torch.inference_mode():
            first = torch.from_numpy(x.astype(np.float16).astype(np.float32)) @ w1.T + b1
            middle = block.mlp.drop(block.mlp.act(first))
            emulated = torch.from_numpy(middle.numpy().astype(np.float16).astype(np.float32)) @ w2.T + b2
            emulated = block.mlp.drop(emulated).numpy()
        report['artifacts']['rounded_operands_emulation'] = save('rounded-operands-output.npy', emulated)
        report['rounded_operands_vs_original_FP32'] = compare(emulated, ref)

        class Counted:
            def __init__(self, api): self.api, self.counts, self.codes, self.succeeded = api, {}, {}, {}
            def __getattr__(self, name):
                function = getattr(self.api, name)
                def call(*args):
                    self.counts[name] = self.counts.get(name, 0) + 1
                    result = function(*args)
                    if name != 'rknn_create_mem':
                        self.codes.setdefault(name, []).append(int(result))
                    if (bool(result) if name == 'rknn_create_mem' else int(result) == 0):
                        self.succeeded[name] = self.succeeded.get(name, 0) + 1
                    return result
                return call
        counted = Counted(abi._API(str(a.runtime_library)))
        for i, (k, n) in enumerate(((1024, 4096), (4096, 1024)), 1):
            start = time.perf_counter()
            # W is original [N,K]; B is normal [K,N], native [N/16,K/32,16,32].
            b = weights[f'mlp.fc{i}.weight'].T.astype(np.float16)
            native = np.ascontiguousarray(b.reshape(k // 32, 32, n // 16, 16).transpose(2, 0, 3, 1))
            if not np.isfinite(native).all(): raise ValueError('FP16 weight overflow')
            packing = time.perf_counter() - start
            start = time.perf_counter()
            context = abi._MatmulContext(counted, (106, k, n), a.core_mask, 'native')
            contexts.append(context); context.allocate()
            np.copyto(context.views['B'], native, casting='no')
            context.sync('B', abi.RKNN_MEMORY_SYNC_TO_DEVICE); context.bind('B')
            report['contexts'].append({'name': f'fc{i}', 'shape': [106, k, n], 'weight_uploads': 1,
                'B_set_io_mem_calls_during_setup': 2, 'initial_zero_B_bind_in_allocate': True,
                'weight_upload_bytes': native.nbytes, 'weight_pack_seconds': packing,
                'create_allocate_upload_bind_seconds': time.perf_counter() - start,
                'device_io_bytes': context.device_bytes, 'queried_attrs': {
                    key: {'name': bytes(getattr(context.attrs, key).name).split(b'\0')[0].decode(),
                        'shape': list(getattr(context.attrs, key).dims[:getattr(context.attrs, key).n_dims]),
                        'type': getattr(context.attrs, key).type, 'size': getattr(context.attrs, key).size}
                    for key in 'ABC'}})
            del b, native

        def linear(context, value, bias):
            times = {}; started = last = time.perf_counter()
            half = np.empty(value.shape, np.float16)
            if neon: neon.f32_to_f16(value.ctypes.data, half.ctypes.data, value.size)
            else: np.copyto(half, value)
            if not np.isfinite(half).all(): raise ValueError('FP16 activation overflow/nonfinite')
            now = time.perf_counter(); times['fp16_conversion'] = now - last; last = now
            np.copyto(context.views['A'], half, casting='no')
            now = time.perf_counter(); times['A_copy'] = now - last; last = now
            context.sync('A', abi.RKNN_MEMORY_SYNC_TO_DEVICE); context.bind('A')
            now = time.perf_counter(); times['A_sync_bind'] = now - last; last = now
            abi._check(counted.rknn_matmul_run(context.ctx), 'rknn_matmul_run')
            now = time.perf_counter(); times['run'] = now - last; last = now
            context.sync('C', abi.RKNN_MEMORY_SYNC_FROM_DEVICE)
            now = time.perf_counter(); times['C_sync'] = now - last; last = now
            result = context.views['C'].copy()
            now = time.perf_counter(); times['C_owned_copy'] = now - last; last = now
            result += bias
            now = time.perf_counter(); times['bias'] = now - last; times['total'] = now - started
            return result, times

        previous = {}
        with torch.inference_mode():
            for index, route in enumerate('ABBAABBA'):
                started = time.perf_counter(); stages = {}
                if route == 'A':
                    y = block.mlp(torch.from_numpy(x)).numpy().reshape(106, 1024)
                else:
                    first, stages['fc1'] = linear(contexts[0], x, weights['mlp.fc1.bias'])
                    start = time.perf_counter()
                    middle = block.mlp.drop(block.mlp.act(torch.from_numpy(first))).numpy()
                    stages['original_GELU_drop'] = time.perf_counter() - start
                    y, stages['fc2'] = linear(contexts[1], middle, weights['mlp.fc2.bias'])
                    y = block.mlp.drop(torch.from_numpy(y)).numpy()
                seconds = time.perf_counter() - started
                y = np.ascontiguousarray(y)
                quality = compare(y, ref)
                repeat = None if route not in previous else y.tobytes() == previous[route].tobytes()
                previous[route] = y.copy()
                report['calls'].append({'index': index, 'route': route, 'first_for_route': index == (0 if route == 'A' else 1),
                    'complete_route_seconds': seconds, 'stages_seconds': stages, 'vs_original_FP32': quality,
                    'vs_rounded_operands_emulation': compare(y, emulated), 'bitwise_equal_previous_same_route': repeat,
                    'output': save(f'call-{index:02d}-{route}.npy', y)})
        report['artifacts']['native_output'] = save('native-output.npy', previous['B'])
        if a.compiled_mlp:
            helper = pin(a.compiled_mlp, a.compiled_mlp_sha256)
            report['compiled_helper'] = {'file': helper.name, 'sha256': a.compiled_mlp_sha256,
                'bytes': helper.stat().st_size}
            spec = importlib.util.spec_from_file_location('_v68_compiled_mlp', helper)
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            report['compiled_lifecycle_verified'] = False
            compiled = module.CompiledMLP(repo, core_mask=a.core_mask)
            report['compiled_comparison'] = {'routes': {'A': 'compiled_MLP', 'B': 'native_MLP'}, 'calls': []}
            for index, route in enumerate('ABBAABBA'):
                started = time.perf_counter(); stages = {}
                if route == 'A':
                    y = compiled.run(x)
                else:
                    with torch.inference_mode():
                        first, stages['fc1'] = linear(contexts[0], x, weights['mlp.fc1.bias'])
                        start = time.perf_counter()
                        middle = block.mlp.drop(block.mlp.act(torch.from_numpy(first))).numpy()
                        stages['original_GELU_drop'] = time.perf_counter() - start
                        y, stages['fc2'] = linear(contexts[1], middle, weights['mlp.fc2.bias'])
                        y = block.mlp.drop(torch.from_numpy(y)).numpy()
                seconds = time.perf_counter() - started
                y = np.ascontiguousarray(y)
                report['compiled_comparison']['calls'].append({'index': index, 'route': route,
                    'first_for_route': index == (0 if route == 'A' else 1), 'complete_route_seconds': seconds,
                    'stages_seconds': stages, 'vs_original_FP32': compare(y, ref),
                    'vs_native_first_segment': compare(y, previous['B']),
                    'output': save(f'compiled-ab-{index:02d}-{route}.npy', y)})
            report['compiled_comparison']['summary_seconds'] = {route: {
                'all_mean': float(np.mean([c['complete_route_seconds'] for c in report['compiled_comparison']['calls'] if c['route'] == route])),
                'subsequent_mean': float(np.mean([c['complete_route_seconds'] for c in report['compiled_comparison']['calls'] if c['route'] == route and not c['first_for_route']]))}
                for route in 'AB'}
        report['timing_scope'] = 'Warm routes include conversion/all A sync+bind/run/C sync+owned copy/bias/original GELU+drop; no file hashing/saving. B weight setup is separate and happens once per context. First calls retained.'
        report['summary_seconds'] = {route: {'all_mean': float(np.mean([c['complete_route_seconds'] for c in report['calls'] if c['route'] == route])),
            'subsequent_mean': float(np.mean([c['complete_route_seconds'] for c in report['calls'] if c['route'] == route and not c['first_for_route']]))}
            for route in 'AB'}
        report['status'] = 'completed_component_diagnostic_not_quality_accepted'
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['exception_notes'] = list(getattr(exc, '__notes__', []))
        raise
    finally:
        report['hooks_removed'] = True
        for hook in hooks:
            try: hook.remove()
            except BaseException as exc:
                report['hooks_removed'] = False; report['cleanup_errors'].append(str(exc))
        for context in reversed(contexts):
            try: context.close(); report['closed_contexts'] += 1
            except BaseException as exc: report['cleanup_errors'].append(str(exc))
        if compiled:
            try: compiled.close()
            except BaseException as exc: report['cleanup_errors'].append(str(exc))
            try:
                metrics = compiled.metrics(); report['compiled_metrics_after_close'] = metrics
                report['compiled_lifecycle_verified'] = (metrics['contexts_created'] == metrics['contexts_destroyed'] == 1
                    and metrics['outputs_acquired'] == metrics['outputs_released'] == 4
                    and metrics['calls_completed'] == metrics['sdk_runs_succeeded'] == 4
                    and metrics['live_contexts'] == 0 and metrics['closed'] and not metrics['cleanup_errors'])
                if report['status'].startswith('completed') and not report['compiled_lifecycle_verified']:
                    report['cleanup_errors'].append('Compiled context/output/run count invariant failed')
            except BaseException as exc: report['cleanup_errors'].append(str(exc))
        if counted:
            report['SDK_call_counts'], report['SDK_return_codes'] = counted.counts, counted.codes
            report['SDK_successful_call_counts'] = counted.succeeded
            for name in ('rknn_destroy_mem', 'rknn_matmul_destroy'):
                for code in counted.codes.get(name, []):
                    if code != 0: report['cleanup_errors'].append(f'{name} returned {code}; release unverified')
            expected_runs = 16 if a.compiled_mlp else 8
            report['native_lifecycle_verified'] = (report['closed_contexts'] == 2
                and counted.succeeded.get('rknn_matmul_create') == counted.succeeded.get('rknn_matmul_destroy') == 2
                and counted.succeeded.get('rknn_create_mem') == counted.succeeded.get('rknn_destroy_mem') == 6
                and counted.succeeded.get('rknn_matmul_run') == expected_runs
                and all(code == 0 for codes in counted.codes.values() for code in codes))
            if report['status'].startswith('completed') and not report['native_lifecycle_verified']:
                report['cleanup_errors'].append('Native context/memory/run count invariant failed')
        try:
            report['source_and_inputs_unchanged'] = all(sha(repo / key) == value['sha256'] for key, value in report['sources'].items())
            report['source_and_inputs_unchanged'] &= all(sha(child(capture, key)) == value['sha256'] for key, value in report['inputs'].items())
            report['source_and_inputs_unchanged'] &= sha(capture / 'capture.json') == CAPTURE_SHA
            report['source_and_inputs_unchanged'] &= sha(a.runtime_library) == RUNTIME_SHA
            if a.neon_library:
                report['source_and_inputs_unchanged'] &= sha(a.neon_library) == NEON_SHA
                report['source_and_inputs_unchanged'] &= sha(repo / 'vggt_slam/native/fp16_neon.c') == NEON_SOURCE_SHA
            if a.compiled_mlp:
                report['source_and_inputs_unchanged'] &= sha(a.compiled_mlp) == a.compiled_mlp_sha256
            if 'artifacts' in report:
                report['input_array_unchanged'] = raw_sha(x) == report['artifacts']['input']['raw_sha256']
                report['source_and_inputs_unchanged'] &= report['input_array_unchanged']
        except BaseException as exc:
            report['source_and_inputs_unchanged'] = False; report['cleanup_errors'].append(str(exc))
        report['affinity_restored'] = False
        try:
            os.sched_setaffinity(0, original_affinity)
            report['affinity_restored'] = os.sched_getaffinity(0) == original_affinity
        except BaseException as exc: report['cleanup_errors'].append(str(exc))
        if report['cleanup_errors'] or not report['source_and_inputs_unchanged'] or not report['affinity_restored']:
            report['status'] = 'failed_cleanup_or_identity'
        temp = out / 'result.json.tmp'; temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        os.replace(temp, out / 'result.json')
        print(json.dumps({'status': report['status'], 'result_sha256': sha(out / 'result.json')}), flush=True)
    if report['status'] != 'completed_component_diagnostic_not_quality_accepted':
        raise RuntimeError(report['status'])


if __name__ == '__main__':
    main()
