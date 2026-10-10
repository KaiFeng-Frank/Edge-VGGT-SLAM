#!/usr/bin/env python3
"""Original aggregator block with CPU surrounding ops and persistent NPU SDPA.

Only SDPA is replaced. Four CPU/hybrid pairs use AB/BA order in one process.
Packing, worker scheduling, synchronous RKNN IO and concatenation are included
in hybrid block time. Hashes, error metrics, saving and perf queries are outside
the timed forwards. This is an isolated board experiment, not a model backend.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
from functools import partial
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
BASE = Path('/mnt/vggt-data/vggt-slam2')
SOURCE = BASE / 'src'
R3_NAME = 'run-v19-attention-board-r3.py'
R3_SHA = 'be4185ef2ea137bf67b1ed52898b3298ab8626274b67fe33cfdca4b6030f0a1e'
ARRAY_KEYS = ('query', 'key', 'value')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def load_attention():
    path = HERE / R3_NAME
    if sha(path) != R3_SHA:
        raise ValueError('Frozen R3 native transport SHA256 mismatch')
    spec = importlib.util.spec_from_file_location('_v20_frozen_attention_r3', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_base()


def sdk_errors(text):
    return [line for line in text.splitlines()
            if 'E RKNN' in line or 'normalize error(' in line or 'unsupported src layout' in line]


class NativeLog:
    """Capture both native fd streams, with independent restoration on failure."""
    def __init__(self, path):
        self.path = Path(path)
        self.saved, self.fd = {}, None
        self.restored = False
        self.libc = ct.CDLL(None)
        self.libc.fflush.argtypes, self.libc.fflush.restype = [ct.c_void_p], ct.c_int

    def flush(self):
        sys.stdout.flush()
        sys.stderr.flush()
        self.libc.fflush(None)

    def __enter__(self):
        self.flush()
        try:
            self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            for stream in (1, 2):
                self.saved[stream] = os.dup(stream)
            for stream in (1, 2):
                os.dup2(self.fd, stream)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def inspect(self):
        self.flush()
        if self.path.stat().st_size > 64 << 20:
            raise RuntimeError('Native log exceeded the diagnostic 64MiB bound')
        errors = sdk_errors(self.path.read_text(errors='replace'))
        return dict(error_line_count=len(errors), error_lines=errors[:64],
                    checked=True, error_detection='E RKNN / normalize error / unsupported src layout',
                    file=self.path.name, bytes=self.path.stat().st_size, sha256=sha(self.path))

    def __exit__(self, exc_type, exc, tb):
        failures = []
        try:
            self.flush()
        except BaseException as error:
            failures.append(error)
        for stream, saved in self.saved.items():
            try:
                os.dup2(saved, stream)
            except BaseException as error:
                failures.append(error)
            finally:
                try:
                    os.close(saved)
                except BaseException as error:
                    failures.append(error)
        self.saved.clear()
        if self.fd is not None:
            try:
                os.close(self.fd)
            except BaseException as error:
                failures.append(error)
            self.fd = None
        self.restored = not failures
        if failures:
            raise RuntimeError(f'Native stream restoration failed: {failures}')
        return False


class AttentionBridge:
    """Single isolated block owner; original CPU dispatch or three-context NPU."""
    def __init__(self, functional, torch, expected_shape, native_call):
        self.functional, self.torch = functional, torch
        self.shape, self.native_call = tuple(expected_shape), native_call
        self.original = functional.scaled_dot_product_attention
        self.wrapper = self.intercept
        self.restored = False
        self.row, self.operands, self.output = None, None, None

    def begin(self, mode, row):
        if mode not in ('cpu', 'hybrid'):
            raise ValueError('Unknown forward mode')
        self.mode, self.row = mode, row
        self.calls, self.operands, self.output = 0, None, None

    def __enter__(self):
        if self.functional.scaled_dot_product_attention is not self.original:
            raise RuntimeError('SDPA callable changed before isolated block experiment')
        self.functional.scaled_dot_product_attention = self.wrapper
        return self

    def intercept(self, query, key, value, *args, **kwargs):
        if self.calls or args or kwargs != {'dropout_p': 0.0} or self.row is None:
            raise ValueError('Exactly one original unmasked eval SDPA call per block required')
        for tensor in (query, key, value):
            if tuple(tensor.shape) != self.shape or tensor.device.type != 'cpu' or tensor.dtype != self.torch.float32:
                raise ValueError('Original captured CPU FP32 Q/K/V shape required')
        self.calls += 1
        self.operands = (query, key, value)  # References only; hashes and diagnostic copies occur later.
        begin = time.perf_counter()
        self.row['attention_begin_monotonic'] = time.monotonic()
        if self.mode == 'cpu':
            output = self.original(query, key, value, **kwargs)
        else:
            arrays = {name: tensor.detach().contiguous().numpy() for name, tensor in zip(ARRAY_KEYS, self.operands)}
            flat = {name: array.reshape(1, self.shape[0]*self.shape[1], self.shape[2], self.shape[3])
                    for name, array in arrays.items()}
            self.row['tensor_to_numpy_seconds'] = time.perf_counter()-begin
            group = {}
            self.row['native_attention'] = group
            actual = self.native_call(flat, self.shape, group)
            tick = time.perf_counter()
            output = self.torch.from_numpy(actual)
            self.row['owned_numpy_to_torch_seconds'] = time.perf_counter()-tick
        self.output = output
        self.row.update(attention_seconds=time.perf_counter()-begin,
                        attention_end_monotonic=time.monotonic(), sdpa_calls=self.calls)
        return output

    def __exit__(self, exc_type, exc, tb):
        self.functional.scaled_dot_product_attention = self.original
        self.restored = self.functional.scaled_dot_product_attention is self.original
        return False


def rebuild_block(torch, row, arrays, weights, loader):
    from vggt.layers.block import Block
    from vggt.layers.rope import RotaryPositionEmbedding2D
    rope = RotaryPositionEmbedding2D(frequency=row['rope_frequency'], scaling_factor=row['rope_scaling_factor'])
    rope.frequency_cache[(row['head_dim']//2, int(arrays['pos'].max())+1, torch.device('cpu'), torch.float32)] = (
        torch.from_numpy(arrays['rope_cos']), torch.from_numpy(arrays['rope_sin']))
    with torch.device('meta'):
        block = Block(dim=1024, num_heads=16, mlp_ratio=4., qkv_bias=True, proj_bias=True, ffn_bias=True,
                      drop=0., attn_drop=0., init_values=.01, drop_path=0., act_layer=torch.nn.GELU,
                      norm_layer=partial(torch.nn.LayerNorm, eps=row['norm1_eps']), qk_norm=True,
                      fused_attn=True, rope=rope)
    block.load_state_dict({name: torch.from_numpy(value) for name, value in weights.items()}, strict=True, assign=True)
    block.eval()
    import numpy as np
    if ({name: loader.array_record(np.ascontiguousarray(value.detach().numpy()))
         for name, value in block.state_dict().items()} != row['weights']
            or block.training or block.attn.training or block.attn.attn_drop.p != 0.
            or block.attn.proj_drop.p != 0. or block.sample_drop_ratio != 0.
            or block.mlp.act.approximate != 'none' or block.ls1.inplace or block.ls2.inplace
            or any(value.is_meta or value.device.type != 'cpu' or value.dtype != torch.float32
                   for value in (*block.parameters(), *block.buffers()))):
        raise ValueError('Original 18-weight eval CPU Block reconstruction mismatch')
    return block


def validate_capture_pair(profile, row, block, capture):
    if (capture.get('block') != block or capture.get('profile_sha256') != sha(Path(profile['_path']))
            or capture.get('original_block_array_metadata') != row['arrays']
            or capture.get('original_block_parameter_metadata') != row['weights']
            or capture.get('selected_pair') != profile['selected_pair']
            or capture.get('checkpoint') != profile['checkpoint']):
        raise ValueError('Block profile and attention capture must describe the identical original input/weights')


def order_for_round(index):
    return ('cpu', 'hybrid') if index % 2 == 0 else ('hybrid', 'cpu')


def cleanup_native(base, runtime, api, pool, sessions, report):
    try:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        report['workers_joined'] = True
    except BaseException as error:
        report['workers_joined'] = False
        report['cleanup_errors'].append(f'Worker shutdown: {error}')
    finally:
        if api is not None:
            base.destroy_sessions(api, runtime, sessions, report['cleanup_errors'])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('baseline', 'parallel'), default='parallel')
    parser.add_argument('--block', choices=('frame0', 'global0'), required=True)
    for name in ('profile-dir', 'capture-dir', 'export-dir', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('profile-sha256', 'capture-sha256', 'export-sha256'):
        parser.add_argument('--'+name, required=True)
    for name in ('models', 'compile-receipts'):
        parser.add_argument('--'+name, type=Path, nargs='+', required=True)
    for name in ('model-sha256', 'compile-sha256'):
        parser.add_argument('--'+name, nargs='+', required=True)
    parser.add_argument('--library', type=Path, default=Path('/usr/lib/librknnrt.so'))
    parser.add_argument('--run-timeout-ms', type=int, default=30000)
    parser.add_argument('--collect-perf-detail', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    base = load_attention()
    runtime = base.load_helper('run-v12-rknn-block-board.py', base.RUNTIME_SHA)
    compatible = runtime.require_board()  # Must precede output creation, Torch or native library load.
    if Path.cwd().resolve() != BASE or Path(__file__).resolve().parents[3] != SOURCE:
        raise ValueError('Run only the deployed board helper from the project base')
    output = args.output.resolve()
    if (os.path.lexists(args.output) or not output.is_relative_to(BASE/'experiments')
            or output == BASE/'experiments' or not 1 <= args.run_timeout_ms <= 60000):
        raise ValueError('New board experiments output and bounded SDK timeout required')
    if any(output.is_relative_to(path.resolve()) or path.resolve().is_relative_to(output)
           for path in (args.profile_dir, args.capture_dir, args.export_dir)):
        raise ValueError('Output must not overlap any frozen capture/export')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(kind='original_VGGT_CPU_block_NPU_attention_hybrid_probe', status='failed',
        script_sha256=sha(__file__), R3_attention_sha256=R3_SHA, mode=args.mode, block=args.block,
        inference_on_board_only=True, production_changed=False, original_weights=True, architecture_changed=False,
        full_model_run=False, ORB_used=False, npu_run_api_called=False, all_operators_on_npu_verified=False,
        CPU_operators=['qkv', 'norm1', 'QK norm', 'RoPE', 'projection', 'layer scales', 'norm2', 'GELU', 'MLP', 'residuals'],
        NPU_requested_operators=['complete-key scaled dot-product attention'],
        predeclared_FP32_gate=dict(rtol=1e-4, atol=1e-5), init_flags=8 if args.collect_perf_detail else 0,
        perf_detail_collection_enabled=args.collect_perf_detail, normal_benchmark_timing=not args.collect_perf_detail,
        round_order=[list(order_for_round(i)) for i in range(4)], unmeasured_warmup_calls=0,
        whole_block_endpoint='original block input to complete block output, including attention conversions/queue/IO/concat',
        timing_excludes=['hashing', 'error metrics', 'file writes', 'context initialization', 'SDK perf queries'],
        timing_limits=['CPU pass-through timing observer is included', 'AB/BA four pairs are not a thermal/statistical benchmark',
                       'Profile timings include extra instrumentation and are not normal benchmarks',
                       'Host SDK overlap does not prove simultaneous kernel execution'],
        whole_model_quality_validated=False, realtime_validated=False, device_tree_compatible=compatible,
        contexts=[], forwards=[], output_arrays=[], cleanup_errors=[])
    affinity = os.sched_getaffinity(0)
    env_keys = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
    old_env = {key: os.environ.get(key) for key in env_keys}
    sessions, pins, api, pool, bridge, native_log = [], {}, None, None, None, None
    try:
        if not {6, 7} <= affinity:
            raise RuntimeError('CPUs 6,7 not in process allowed affinity')
        os.sched_setaffinity(0, {6, 7})
        os.environ.update({key: '2' for key in env_keys})
        exporter = base.load_helper('export-v19-attention.py', base.EXPORTER_SHA)
        capture, exported, reference_flat, specs, pins = base.load_inputs(args, exporter, runtime)
        loader = exporter.frozen_loader()
        profile, row, arrays, weights, profile_pins = loader.load_capture(args.profile_dir, args.profile_sha256, args.block)
        profile['_path'] = str(args.profile_dir/'profile.json')
        validate_capture_pair(profile, row, args.block, capture)
        pins.update(profile_pins)
        pins[str(Path(__file__).resolve())] = report['script_sha256']
        library = args.library.resolve(strict=True)
        base.require_sha(library, base.LIBRARY_SHA)
        pins[str(library)] = base.LIBRARY_SHA
        report.update(profile_sha256=args.profile_sha256, capture_sha256=args.capture_sha256,
                      export_sha256=args.export_sha256, checkpoint=profile['checkpoint'], selected_pair=profile['selected_pair'],
                      block_shape=list(arrays['input'].shape), attention_shape=exported['original_shape'],
                      core_masks=[item['core_mask'] for item in specs], source_and_input_hashes=dict(pins),
                      block_parameter_metadata=row['weights'], native_library_sha256=base.LIBRARY_SHA,
                      abi=runtime.abi_record(), original_position_tables_used=True)
        sys.path[:0] = [str(SOURCE), str(SOURCE/'third_party/vggt')]
        import numpy as np
        import torch
        import torch.nn.functional as functional
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        report['torch'] = dict(version=str(torch.__version__), threads=torch.get_num_threads(),
                               interop_threads=torch.get_num_interop_threads(), cpu_affinity=sorted(os.sched_getaffinity(0)),
                               environment={key: os.environ.get(key) for key in (*env_keys, 'OMP_WAIT_POLICY', 'GOMP_SPINCOUNT')})
        block = rebuild_block(torch, row, arrays, weights, loader)
        tokens, positions = torch.from_numpy(arrays['input']), torch.from_numpy(arrays['pos'])
        expected_shape = tuple(exported['original_shape'])
        expected_attention = reference_flat['output'].reshape(expected_shape)
        native_log = NativeLog(output/'sdk.log')
        with native_log:
            try:
                api = runtime.bind_api(ct.CDLL(str(library)))
                for item in specs:
                    context_row = {key: value for key, value in item.items() if key != 'model'}
                    context_row.update(model_file=str(item['model']), context_destroyed=None)
                    report['contexts'].append(context_row)
                    base.create_session(api, runtime, item, report['init_flags'], sessions, context_row)
                if len(sessions) == 3:
                    pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix='v20-npu-attention')

                def native_call(flat, shape, group):
                    return base.execute_round(api, runtime, sessions, flat, shape, args.run_timeout_ms, group, pool)

                bridge = AttentionBridge(functional, torch, expected_shape, native_call)
                with bridge, torch.inference_mode():
                    for i in range(4):
                        for mode in order_for_round(i):
                            forward = dict(round_index=i, phase='first' if i == 0 else 'subsequent', mode=mode)
                            report['forwards'].append(forward)
                            bridge.begin(mode, forward)
                            tick = time.perf_counter()
                            forward['begin_monotonic'] = time.monotonic()
                            actual_tensor = block(tokens, pos=positions)
                            forward.update(end_monotonic=time.monotonic(), whole_block_seconds=time.perf_counter()-tick)
                            if bridge.calls != 1 or bridge.output is None:
                                raise ValueError('One complete attention call is required per original block')
                            actual = np.ascontiguousarray(actual_tensor.detach().numpy())
                            attention = np.ascontiguousarray(bridge.output.detach().numpy())
                            forward['QKV_exact_captured_raw_hashes'] = {}
                            for name, tensor in zip(ARRAY_KEYS, bridge.operands):
                                value = np.ascontiguousarray(tensor.detach().numpy())
                                metadata = exporter.array_record(value)
                                if metadata != capture['artifacts']['attention.npz']['arrays'][name]:
                                    raise ValueError(f'Live original QKV differs from frozen capture: {name}')
                                forward['QKV_exact_captured_raw_hashes'][name] = metadata['sha256']
                            forward['versus_original_CPU_block'] = base.comparison(runtime, actual, arrays['output'])
                            forward['attention_versus_original_CPU_SDPA'] = base.comparison(runtime, attention, expected_attention)
                            for label, value in (('block', actual), ('attention', attention)):
                                path = output/f'{mode}_{i:02d}_{label}.npy'
                                with path.open('xb') as stream:
                                    np.save(stream, value, allow_pickle=False)
                                report['output_arrays'].append(dict(mode=mode, round_index=i, role=label,
                                    file=path.name, bytes=path.stat().st_size, sha256=sha(path), array=exporter.array_record(value)))
                            report['native_log'] = native_log.inspect()
                            if report['native_log']['error_line_count']:
                                raise RuntimeError('Native RKNN error log detected, regardless of successful C API return')
                            if mode == 'hybrid' and forward['attention_versus_original_CPU_SDPA']['output_all_zero']:
                                raise RuntimeError('All-zero hybrid attention is a transport/execution failure, not quality success')
                            if mode == 'cpu' and (not forward['versus_original_CPU_block']['elementwise_gate_passed']
                                                  or not forward['attention_versus_original_CPU_SDPA']['elementwise_gate_passed']):
                                raise ValueError('Original CPU block fails unchanged predefined FP32 reference gate')
                            if mode == 'hybrid':
                                for session, call in zip(sessions, forward['native_attention']['calls']):
                                    perf = runtime.PerfRun()
                                    code = api.rknn_query(session['context'], 4, ct.byref(perf), ct.sizeof(perf))
                                    call.update(perf_run_query_code=int(code), sdk_run_duration_us=int(perf.run_duration) if code == 0 else None)
                                    if args.collect_perf_detail:
                                        call['perf_detail'] = runtime.perf_detail(api, session['context'])
                for mode in ('cpu', 'hybrid'):
                    selected = [item for item in report['forwards'] if item['mode'] == mode]
                    warm = [item['whole_block_seconds'] for item in selected[1:]]
                    report[mode+'_timing'] = dict(first_seconds=selected[0]['whole_block_seconds'],
                        subsequent_seconds=warm, subsequent_mean_seconds=float(np.mean(warm)),
                        attention_subsequent_mean_seconds=float(np.mean([item['attention_seconds'] for item in selected[1:]])))
                report['hybrid_strict_FP32_block_gate_all_rounds_passed'] = all(
                    item['versus_original_CPU_block']['elementwise_gate_passed'] for item in report['forwards'] if item['mode'] == 'hybrid')
                report['hybrid_strict_FP32_attention_gate_all_rounds_passed'] = all(
                    item['attention_versus_original_CPU_SDPA']['elementwise_gate_passed'] for item in report['forwards'] if item['mode'] == 'hybrid')
                report['execution_completed_does_not_mean_quality_passed'] = True
                report['status'] = 'completed'
            finally:
                cleanup_native(base, runtime, api, pool, sessions, report)
                report['native_log'] = native_log.inspect()
                if report['native_log']['error_line_count']:
                    raise RuntimeError('RKNN native log contains errors; execution cannot be accepted')
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
    finally:
        report['sdpa_restored'] = bridge is None or bridge.restored
        report['native_streams_restored'] = native_log is None or native_log.restored
        if pins:
            try:
                after = {path: sha(path) for path in pins}
                report.update(source_and_input_hashes_after=after, source_and_inputs_unchanged=after == pins)
                if after != pins:
                    report['cleanup_errors'].append('Frozen source/input changed')
            except BaseException as error:
                report['cleanup_errors'].append(f'Source recheck: {error}')
        try:
            os.sched_setaffinity(0, affinity)
            report['cpu_affinity_restored'] = os.sched_getaffinity(0) == affinity
        except BaseException as error:
            report['cleanup_errors'].append(f'Affinity restoration: {error}')
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        report['memory_final'] = runtime.process_memory()
        calls = [call for row in report['forwards'] for call in row.get('native_attention', {}).get('calls', [])]
        report['npu_run_api_call_count'] = sum(call.get('run_api_called', False) for call in calls)
        report['npu_run_api_called'] = report['npu_run_api_call_count'] > 0
        report['successful_synchronous_call_count'] = sum(call.get('output_copied_before_release', False)
                                                        and call.get('output_buffer_released', False) for call in calls)
        report['output_release_attempt_count'] = sum('outputs_release_code' in call for call in calls)
        report['total_seconds'] = time.monotonic()-started
        if report['cleanup_errors'] or not report['sdpa_restored'] or not report['native_streams_restored']:
            report['status'] = 'failed'
        path = output/'run.json'
        with path.open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
    print(json.dumps(dict(status=report['status'], error=report.get('error'), output=str(path), sha256=sha(path))), flush=True)
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
