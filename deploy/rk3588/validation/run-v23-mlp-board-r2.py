#!/usr/bin/env python3
"""R2 board-only original MLP, using rank-three MatMul/Add and exact Erf GELU.

The hidden slices are independent branches of one original MLP, not temporal
iterations. Three partial outputs are summed in FP32 before the one original
FC2 bias, LayerScale and residual. No model hooks/default backend are changed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = Path('/mnt/vggt-data/vggt-slam2')
RUNTIME_SHA = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
LIBRARY_SHA = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
COMPILER_SHA = 'a40bf09e53ef9b378db4781be18f9e53f824afcc809c9009e0a65f4f0f91b4e5'
CAPTURE_SHA = '3664f5c932602a3c6538ea6013ea0389625c42bdcb7dcf10a9fbceb9304abeb1'
RTOL, ATOL = 1e-4, 1e-5
SLICES = [(0, 4096), (0, 1344), (1344, 2688), (2688, 4096)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()


def require_sha(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or sha(path) != expected:
        raise ValueError(f'SHA256 mismatch: {path}')


def array_record(a):
    return dict(shape=list(a.shape), dtype=str(a.dtype), bytes=a.nbytes,
                sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())


def runtime_module():
    path = HERE / 'run-v12-rknn-block-board.py'
    require_sha(path, RUNTIME_SHA)
    spec = importlib.util.spec_from_file_location('_v23_frozen_runtime', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def comparison(runtime, actual, expected):
    import numpy as np
    row = runtime.comparison(actual, expected)
    close = np.isclose(actual, expected, rtol=RTOL, atol=ATOL)
    row.update(element_count=int(close.size), failed_elements=int(np.count_nonzero(~close)),
               elementwise_gate_passed=bool(close.all()), rtol=RTOL, atol=ATOL,
               output_all_zero=bool(not np.any(actual)))
    return row


def padded_input(value, bucket):
    import numpy as np
    if (value.dtype != np.float32 or value.ndim != 3 or value.shape[0] != 1
            or value.shape[1] > bucket or not np.isfinite(value).all()):
        raise ValueError('Finite original FP32 rows fitting the bucket required')
    result = np.zeros((1, bucket, value.shape[2]), dtype=np.float32)
    result[:, :value.shape[1]] = value
    return result


def combine(values, bias, rows, partial):
    import numpy as np
    if len(values) != (3 if partial else 1):
        raise ValueError('Exactly three partial outputs or one complete output required')
    shape = values[0].shape
    if (len(shape) != 3 or shape[0] != 1 or not 0 < rows <= shape[1]
            or bias.shape != (shape[2],) or bias.dtype != np.float32
            or not np.isfinite(bias).all()):
        raise ValueError('Invalid reduction shape/bias')
    if any(v.shape != shape or v.dtype != np.float32 or not np.isfinite(v).all() for v in values):
        raise ValueError('Partial outputs must be finite identical-shape FP32 buffers')
    result = np.array(values[0][:, :rows], dtype=np.float32, order='C', copy=True)
    if partial:
        for value in values[1:]:
            np.add(result, value[:, :rows], out=result)
        np.add(result, bias, out=result)
    return result


def observations():
    """Read-only non-atomic sample, outside all inference intervals."""
    result = dict(monotonic=time.monotonic(), thermal={}, cpufreq={})
    for path in sorted(Path('/sys/class/thermal').glob('thermal_zone*/temp')):
        try:
            result['thermal'][path.parent.name] = dict(type=(path.parent/'type').read_text().strip(),
                                                       millidegrees_C=int(path.read_text()))
        except OSError:
            pass
    for path in sorted(Path('/sys/devices/system/cpu/cpufreq').glob('policy*')):
        row = {}
        for name in ('scaling_cur_freq', 'scaling_max_freq', 'scaling_governor'):
            try:
                row[name] = (path/name).read_text().strip()
            except OSError:
                pass
        result['cpufreq'][path.name] = row
    return result


def load_inputs(args):
    import numpy as np
    capture_path = args.capture_dir / 'capture.json'
    require_sha(capture_path, CAPTURE_SHA)
    capture = json.loads(capture_path.read_text())
    if capture['status'] != 'captured_original_FP32_stages_bitwise_bound':
        raise ValueError('Original verified stage capture required')
    pins = {str(capture_path): CAPTURE_SHA, str(Path(__file__).resolve()): sha(__file__),
            str(HERE/'run-v12-rknn-block-board.py'): RUNTIME_SHA}
    data = {}
    for filename, metadata_key in (('activations.npz', 'arrays'), ('mlp-state.npz', 'global0_state')):
        path = args.capture_dir/filename
        entry = capture['artifacts'][filename]
        require_sha(path, entry['sha256'])
        if path.stat().st_size != entry['bytes']:
            raise ValueError('Capture byte size mismatch')
        pins[str(path)] = entry['sha256']
        with np.load(path, allow_pickle=False) as archive:
            for name in archive.files:
                value = archive[name]
                if array_record(value) != capture[metadata_key][name]:
                    raise ValueError(f'Original array metadata mismatch: {name}')
                data[name] = value
    if data['norm2'].shape != (1, 874, 1024):
        raise ValueError('This probe requires the original 336-resolution 874 rows')
    if len(args.exports) != 4 or len(args.export_sha256) != 4:
        raise ValueError('Four ordered baseline/shard export directories and SHA pins required')
    specs = []
    for i, (directory, digest, interval) in enumerate(zip(args.exports, args.export_sha256, SLICES)):
        path = directory/'export.json'
        require_sha(path, digest)
        receipt = json.loads(path.read_text())
        if receipt.get('exporter_name') != 'export-v23-mlp-r2.py':
            raise ValueError('R2 rank-three exporter required')
        for key, name, expected in (
            ('parent_exporter', 'export-v23-mlp.py', '56476c724cde537fd35466be0b41d300592821be59f2a840519be6c117c5146e'),
            ('gelu_pattern_source', 'export-v12-dino-block-gelu.py', '59f5ea1be93812571947f7afb71ea9d908057cf277c3cbc2cfb9bbaaf10e5a78'),
        ):
            if receipt.get(key) != dict(file=name, sha256=expected):
                raise ValueError('R2 frozen source provenance mismatch')
            require_sha(HERE/name, expected)
            pins[str(HERE/name)] = expected
        for key, expected in dict(capture_sha256=CAPTURE_SHA, original_rows=874,
                                  input_name='input', output_name='output',
                                  hidden_slice=list(interval), includes_fc2_bias=i == 0).items():
            if receipt.get(key) != expected:
                raise ValueError(f'Export contract mismatch: {key}')
        if receipt['input_shape'] != [1, receipt['bucket'], 1024] or receipt['bucket'] < 874:
            raise ValueError('Original rows must fit the static bucket without truncation')
        expected_params = {key: capture['global0_state'][key] for key in (
            'mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight', 'mlp.fc2.bias')}
        if receipt['original_parameter_metadata'] != expected_params:
            raise ValueError('Original four-parameter provenance mismatch')
        require_sha(HERE/'export-v23-mlp-r2.py', receipt['script_sha256'])
        entry = receipt['onnx']
        if Path(entry['file']).name != entry['file']:
            raise ValueError('Simple local ONNX filename required')
        onnx_path = directory/entry['file']
        require_sha(onnx_path, entry['sha256'])
        if onnx_path.stat().st_size != entry['bytes']:
            raise ValueError('ONNX byte size mismatch')
        pins.update({str(path): digest, str(onnx_path): entry['sha256'],
                     str(HERE/'export-v23-mlp-r2.py'): receipt['script_sha256']})
        specs.append(dict(index=i, name='baseline' if i == 0 else f'shard_{i-1}',
                          shape=receipt['input_shape'], onnx=str(onnx_path), onnx_sha256=entry['sha256'],
                          export_sha256=digest, core_mask=[7, 1, 2, 4][i], hidden_slice=list(interval)))
    if any(s['shape'] != specs[0]['shape'] for s in specs):
        raise ValueError('All baseline/shards must use the same row bucket')
    return capture, data, specs, pins


def run_ort(runtime, specs, data, output, report):
    import numpy as np
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    padded = padded_input(data['norm2'], specs[0]['shape'][1])
    values = []
    report['output_arrays'] = []
    report.update(onnxruntime_version=ort.__version__, provider='CPUExecutionProvider',
                  graph_optimization='ORT_DISABLE_ALL', model_inference_on_board=True)
    for spec in specs:
        session = ort.InferenceSession(spec['onnx'], sess_options=options, providers=['CPUExecutionProvider'])
        ins, outs = session.get_inputs(), session.get_outputs()
        if (len(ins) != 1 or len(outs) != 1 or ins[0].name != 'input' or outs[0].name != 'output'
                or ins[0].shape != spec['shape'] or outs[0].shape != spec['shape']
                or ins[0].type != 'tensor(float)' or outs[0].type != 'tensor(float)'):
            raise ValueError('ORT IO contract mismatch')
        value = session.run(['output'], {'input': padded})[0]
        if value.shape != padded.shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('ORT invalid output')
        values.append(value)
        path = output/(spec['name']+'.npy')
        np.save(path, value, allow_pickle=False)
        report['output_arrays'].append(dict(file=path.name, bytes=path.stat().st_size,
                                            sha256=sha(path), array=array_record(value)))
        del session
    full = combine(values[:1], data['mlp.fc2.bias'], 874, False)
    parts = combine(values[1:], data['mlp.fc2.bias'], 874, True)
    report['comparisons'] = {name: comparison(runtime, a, b) for name, a, b in (
        ('baseline_vs_original', full, data['mlp_output']),
        ('shards_vs_original', parts, data['mlp_output']), ('shards_vs_baseline', parts, full))}
    report['exact_graph_reference_gate_passed'] = all(v['elementwise_gate_passed'] for v in report['comparisons'].values())
    if not report['exact_graph_reference_gate_passed']:
        raise ValueError('FP32 ORT reference gate failed; do not proceed to RKNN compilation')


class Session:
    def __init__(self, api, runtime, spec, profile, row):
        self.api, self.runtime, self.row = api, runtime, row
        self.context = runtime.CONTEXT()
        self.spec, self.profile = spec, profile
        try:
            filename = ct.create_string_buffer(os.fsencode(spec['model']))
            tick = time.perf_counter()
            runtime.check_call(api.rknn_init(ct.byref(self.context), ct.cast(filename, ct.c_void_p),
                                           0, 8 if profile else 0, None), 'rknn_init')
            row['init_seconds'] = time.perf_counter()-tick
            runtime.check_call(api.rknn_set_core_mask(self.context, spec['core_mask']), 'rknn_set_core_mask')
            version = runtime.query(api, self.context, 5, runtime.SDKVersion())
            row.update(sdk_version=bytes(version.api_version).decode(), driver_version=bytes(version.drv_version).decode())
            if not row['sdk_version'].startswith('2.3.2'):
                raise ValueError('RKNN runtime 2.3.2 required')
            counts = runtime.query(api, self.context, 0, runtime.InputOutputNum())
            if counts.n_input != 1 or counts.n_output != 1:
                raise ValueError('One input and one output required')
            for name, command in (('input', 1), ('output', 2)):
                attr = runtime.query(api, self.context, command, runtime.TensorAttr(index=0))
                record = runtime.validate_attr(attr)
                # Rank-three original order: no implicit NCHW/NHWC conversion.
                if record['format'] != 3 or record['name'] != name:
                    raise ValueError(f'Only explicit UNDEFINED rank-three layout implemented: {record}')
                row[name+'_attr'] = record
            memory = runtime.query(api, self.context, 6, runtime.MemSize())
            row['memory_bytes'] = {name: int(getattr(memory, name)) for name, _ in memory._fields_ if name != 'reserved'}
            if row['memory_bytes']['total_dma_allocated_size'] > 512 * 1024**2:
                raise ValueError('Single MLP context exceeds 512 MiB DMA research limit')
        except BaseException:
            self.close()
            raise

    def call(self, value, barrier=None):
        if barrier is not None:
            barrier.wait(timeout=10)
        start = time.monotonic()
        cleanup = []
        actual, row = self.runtime.execute_call(self.api, self.context, value, 3, 30000, cleanup)
        row.update(begin_monotonic=start, end_monotonic=time.monotonic(),
                   context_index=self.spec['index'], core_mask=self.spec['core_mask'], cleanup_errors=cleanup)
        if self.profile:
            row['perf_detail'] = self.runtime.perf_detail(self.api, self.context)
        return actual, row

    def close(self):
        if self.context.value:
            code = self.api.rknn_destroy(self.context)
            self.row['destroy_code'] = int(code)
            self.row['context_destroyed'] = code == 0
            self.context.value = 0
            self.runtime.check_call(code, 'rknn_destroy')


def run_npu(args, runtime, data, specs, pins, output, report):
    import numpy as np
    if len(args.compile_dirs or []) != 4 or len(args.compile_sha256 or []) != 4:
        raise ValueError('Exactly four compile receipts required')
    require_sha(args.ort_receipt, args.ort_sha256)
    ort = json.loads(args.ort_receipt.read_text())
    if (ort['status'] != 'completed' or not ort['exact_graph_reference_gate_passed']
            or ort['export_sha256'] != args.export_sha256 or ort['capture_sha256'] != CAPTURE_SHA):
        raise ValueError('Passing board ORT precheck for these exact exports required')
    pins[str(args.ort_receipt)] = args.ort_sha256
    references = []
    for spec, entry in zip(specs, ort['output_arrays'], strict=True):
        if entry['file'] != spec['name']+'.npy':
            raise ValueError('Ordered board ORT output artifacts required')
        path = args.ort_receipt.parent/entry['file']
        require_sha(path, entry['sha256'])
        value = np.load(path, allow_pickle=False)
        if array_record(value) != entry['array'] or list(value.shape) != spec['shape']:
            raise ValueError('ORT reference artifact metadata mismatch')
        references.append(value)
        pins[str(path)] = entry['sha256']
    for spec, directory, digest in zip(specs, args.compile_dirs, args.compile_sha256):
        path = directory/'compile.json'
        require_sha(path, digest)
        receipt = json.loads(path.read_text())
        if (receipt.get('status') != 'compiled_not_runtime_validated'
                or receipt['script_sha256'] != COMPILER_SHA
                or receipt['input']['sha256'] != spec['onnx_sha256']
                or receipt['configuration'] != dict(target_platform='rk3588', float_dtype='float16',
                                                     optimization_level=3, do_quantization=False)):
            raise ValueError('Frozen FP16/no-quantization compile contract mismatch')
        model = directory/'block.rknn'
        require_sha(model, receipt['output']['sha256'])
        if model.stat().st_size != receipt['output']['size'] or model.stat().st_size > 128*1024**2:
            raise ValueError('Compiled MLP size mismatch or limit exceeded')
        pins.update({str(path): digest, str(model): receipt['output']['sha256']})
        spec.update(model=str(model), model_sha256=receipt['output']['sha256'], model_bytes=model.stat().st_size)
    require_sha(args.library, LIBRARY_SHA)
    pins[str(args.library)] = LIBRARY_SHA
    runtime.SHAPE = tuple(specs[0]['shape'])  # One fixed shape for every owned context.
    api = runtime.bind_api(ct.CDLL(str(args.library)))
    sessions = []
    try:
        for spec in specs:
            row = dict(spec)
            report['contexts'].append(row)
            sessions.append(Session(api, runtime, spec, args.profile, row))
        report['sum_context_dma_bytes'] = sum(s['memory_bytes']['total_dma_allocated_size'] for s in report['contexts'])
        report['memory_with_contexts'] = runtime.process_memory()
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix='v23-mlp') as workers:
            first = {}
            schedule = ['A', 'B', 'A', 'B', 'B', 'A', 'A', 'B', 'B', 'A']
            report['schedule'] = schedule
            for i, kind in enumerate(schedule):
                row = dict(index=i, variant=kind, phase='warmup' if i < 2 else 'measured', sysfs=observations())
                report['rounds'].append(row)
                start = time.perf_counter()
                value = padded_input(data['norm2'], specs[0]['shape'][1])
                packed = time.perf_counter()
                if kind == 'A':
                    outputs = [sessions[0].call(value)]
                else:
                    barrier = threading.Barrier(3)
                    futures = [workers.submit(s.call, value, barrier) for s in sessions[1:]]
                    # Join every worker before propagating an error or freeing any context.
                    outputs, errors = [], []
                    for future in futures:
                        try:
                            outputs.append(future.result())
                        except Exception as exc:
                            errors.append(exc)
                    if errors:
                        raise errors[0]
                joined = time.perf_counter()
                actual = combine([item[0] for item in outputs], data['mlp.fc2.bias'], 874, kind == 'B')
                reduced = time.perf_counter()
                block = actual * data['ls2.gamma'] + data['residual1']
                finished = time.perf_counter()
                row.update(pack_seconds=packed-start, calls=[item[1] for item in outputs],
                           execution_join_seconds=joined-packed, reduction_crop_seconds=reduced-joined,
                           mlp_endpoint_seconds=reduced-start, ls2_residual_seconds=finished-reduced,
                           stage_endpoint_seconds=finished-start)
                row['mlp_vs_original'] = comparison(runtime, actual, data['mlp_output'])
                row['block_vs_original'] = comparison(runtime, block, data['block_output'])
                indices = [0] if kind == 'A' else [1, 2, 3]
                for (value_out, call), index in zip(outputs, indices):
                    call['versus_board_ORT'] = comparison(runtime, value_out, references[index])
                    if call['versus_board_ORT']['output_all_zero']:
                        raise ValueError(f'All-zero NPU branch {index} rejected independently of SDK success codes')
                if kind not in first:
                    first[kind] = actual.copy()
                row['repeat'] = comparison(runtime, actual, first[kind])
                if row['mlp_vs_original']['output_all_zero']:
                    raise ValueError('All-zero NPU result rejected independently of SDK success codes')
                path = output/f'actual_{i:02d}.npz'
                np.savez(path, mlp=actual, block=block,
                         **{f'partial_{j}': item[0] for j, item in enumerate(outputs)})
                row['artifact'] = dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path))
                print(json.dumps(dict(round=i, variant=kind, ms=1000*row['stage_endpoint_seconds'],
                                      relative_l2=row['block_vs_original']['relative_l2'])), flush=True)
        report['variant_comparison'] = comparison(runtime, first['B'], first['A'])
        report['measured_summary'] = {}
        for kind in ('A', 'B'):
            times = [r['stage_endpoint_seconds'] for r in report['rounds'] if r['phase'] == 'measured' and r['variant'] == kind]
            report['measured_summary'][kind] = dict(values=times, mean=float(np.mean(times)),
                                                   min=min(times), max=max(times), count=len(times))
    finally:
        for session in reversed(sessions):
            try:
                session.close()
            except Exception as exc:
                report['cleanup_errors'].append(str(exc))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('ort', 'npu'), required=True)
    p.add_argument('--capture-dir', type=Path, required=True)
    p.add_argument('--exports', type=Path, nargs=4, required=True)
    p.add_argument('--export-sha256', nargs=4, required=True)
    p.add_argument('--compile-dirs', type=Path, nargs=4)
    p.add_argument('--compile-sha256', nargs=4)
    p.add_argument('--ort-receipt', type=Path)
    p.add_argument('--ort-sha256')
    p.add_argument('--library', type=Path, default=Path('/usr/lib/librknnrt.so'))
    p.add_argument('--profile', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    runtime = runtime_module()
    compatible = runtime.require_board()
    output = args.output.resolve()
    if not output.is_relative_to(BASE/'experiments') or output == BASE/'experiments':
        p.error('New output must be below board experiments')
    protected = [args.capture_dir, *args.exports, *(args.compile_dirs or [])]
    if any(output.is_relative_to(v.resolve()) or v.resolve().is_relative_to(output) for v in protected):
        p.error('Output must not overlap input artifacts')
    output.mkdir(parents=True, exist_ok=False)
    start, affinity = time.monotonic(), os.sched_getaffinity(0)
    report = dict(kind='original_VGGT_MLP_hidden_shard_board_probe', status='failed', mode=args.mode,
                  script_sha256=sha(__file__), capture_sha256=CAPTURE_SHA, export_sha256=args.export_sha256,
                  device_tree_compatible=compatible, runtime_sha256=RUNTIME_SHA, abi=runtime.abi_record(),
                  profile=args.profile, normal_benchmark_timing=not args.profile,
                  original_weights=True, hidden_neurons_dropped=0, GELU='original exact Erf formula',
                  ORB_used=False, host_model_inference_executed=False, production_changed=False,
                  full_model_run=False, realtime_validated=False, quality_validated=False,
                  endpoint='FP32 padding/copy + submit/join + synchronous SDK IO + ordered FP32 sum + single bias + crop + original LS2/residual',
                  endpoint_excludes=['context initialization', 'error metrics', 'hashing', 'file writes', 'sysfs samples'],
                  endpoint_includes_informational_perf_queries=True, contexts=[], rounds=[], cleanup_errors=[])
    pins = {}
    try:
        if not {4, 5, 6, 7} <= affinity:
            raise ValueError('Four big CPU cores unavailable')
        os.sched_setaffinity(0, {4, 5, 6, 7})
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            os.environ[key] = '4'
        capture, data, specs, pins = load_inputs(args)
        report.update(selected_pair=capture['selected_pair'], checkpoint=capture['checkpoint'],
                      original_parameter_metadata={k: capture['global0_state'][k] for k in capture['mlp_state_keys']},
                      shape=specs[0]['shape'], cpu_affinity=sorted(os.sched_getaffinity(0)))
        if args.mode == 'ort':
            run_ort(runtime, specs, data, output, report)
        else:
            run_npu(args, runtime, data, specs, pins, output, report)
        report['status'] = 'completed'
    except Exception as exc:
        report.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        report['source_and_input_hashes'] = pins
        try:
            report['sources_and_inputs_unchanged'] = all(sha(path) == digest for path, digest in pins.items())
            if not report['sources_and_inputs_unchanged']:
                report['cleanup_errors'].append('Source/input hash changed')
            os.sched_setaffinity(0, affinity)
            report['affinity_restored'] = os.sched_getaffinity(0) == affinity
        except Exception as exc:
            report['cleanup_errors'].append(str(exc))
        report.update(total_seconds=time.monotonic()-start, memory_final=runtime.process_memory())
        if report['cleanup_errors']:
            report['status'] = 'failed'
        (output/'run.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'], error=report.get('error'), report=str(output/'run.json'))), flush=True)
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
