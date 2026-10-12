#!/usr/bin/env python3
"""Board-only original pair00 versus the same graph with two pre-GELU taps.

Normal ABBAABBA and fresh flags=8 profiles have separate contexts and timings.
FP32 submission/retrieval includes both extra taps; this component experiment
does not establish whole-model speed, numerical acceptance or SLAM accuracy.
"""
import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time
import traceback

BASE = Path('/mnt/vggt-data/vggt-slam2')
BASELINE_SHA = '1872f2ed730f4b6d139562d831661126a088b1ac758249d6a1688cd7e9bfae42'
PAIR_GRAPH_SHA = 'efbbaee8f3d6326565930e5a045eac0d500fdb35453fdf85ddac7481ec2fac65'
CAPTURE_SHA = '4b723e499763bb90565f6e11e16c551664634aaf72da46c7cdc493b84208ef67'
CHECKPOINT = dict(sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0', bytes=5026874952)
SOURCE_PINS = {
    'vggt_slam/rknn_aggregator_blocks.py': '4fea665334e9d12a3d55839c3150680e269bab9794074d698bed2ad0a6e41c93',
    'vggt_slam/rknn_aggregator_pairs.py': '08270da911213d0d22c21c9a7f001aaa7e16d7f7b32220f8a2b0c1ddc4b09b88',
    'deploy/rk3588/validation/run-v12-rknn-block-board.py': '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608',
    'deploy/rk3588/reference/rknn_api_v2.3.2.h': 'c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a',
}
ORDER = ('A', 'B', 'B', 'A', 'A', 'B', 'B', 'A')
BINDINGS = [dict(index=i, name=name, shape=shape, dtype='float32') for i, (name, shape) in enumerate([
    ('frame_output', [2, 53, 1024]), ('global_output', [1, 106, 1024]),
    ('frame0/mlp.fc1', [2, 53, 4096]), ('global0/mlp.fc1', [1, 106, 4096])])]
CONFIGURATION = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
    mean_values=None, std_values=None, quant_img_RGB2BGR=False, model_pruning=False,
    remove_reshape=False, quantize_weight=False, enable_flash_attention=False)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''): digest.update(chunk)
    return digest.hexdigest()


def pin(path, expected, pins):
    path = Path(path).resolve(strict=True)
    if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected) or sha(path) != expected:
        raise ValueError(f'Frozen SHA256 mismatch: {path}')
    pins[path] = expected
    return path


def compare(np, actual, reference):
    if actual.shape != reference.shape or actual.dtype != np.float32 or not np.isfinite(reference).all():
        raise ValueError('Original reference shape/dtype/finite mismatch')
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(delta); failed = absolute > 1e-5 + 1e-4 * np.abs(reference.astype(np.float64))
    denominator = float(np.linalg.norm(reference.astype(np.float64))); error = float(np.linalg.norm(delta))
    return dict(bitwise_equal=actual.tobytes() == reference.tobytes(), max_abs=float(absolute.max()),
        relative_l2=error / denominator if denominator else (0. if error == 0 else None),
        strict_rtol=1e-4, strict_atol=1e-5, failed_elements=int(failed.sum()), strict_pass=not bool(failed.any()))


def execute(base, api, np, context, array, fmt, bindings, counters, cleanup_errors):
    """Own every output before release, including partially allocated failures."""
    if array.dtype != np.float32 or not array.flags.c_contiguous or not np.isfinite(array).all():
        raise ValueError('Dense finite FP32 SDK input required')
    count = len(bindings)
    inputs = (base.Input * 1)(base.Input(0, array.ctypes.data, array.nbytes, 0, 0, fmt))
    outputs = (base.Output * count)()
    for index in range(count): outputs[index].index, outputs[index].want_float = index, 1
    run, received = base.RunExtend(0, 0, 30000, -1), base.OutputExtend()
    timing = dict(input_api_type=0, input_pass_through=0, input_format=fmt, output_want_float=1,
                  output_count=count, tap_output_copy_seconds=0., extra_tap_output_bytes=0,
                  output_buffer_released=False)
    got = False; begin = time.perf_counter()
    try:
        base.check_call(api.rknn_inputs_set(context, 1, inputs), 'rknn_inputs_set')
        set_end = time.perf_counter(); counters['sdk_run_attempts'] += 1
        code = api.rknn_run(context, ct.byref(run)); counters['sdk_runs_succeeded'] += int(code == 0)
        base.check_call(code, 'rknn_run'); run_end = time.perf_counter()
        counters['output_sets_get_attempts'] += 1
        code = api.rknn_outputs_get(context, count, outputs, ct.byref(received))
        counters['outputs_acquired'] += sum(bool(output.buf) for output in outputs)
        counters['output_sets_acquired'] += int(code == 0)
        base.check_call(code, 'rknn_outputs_get'); got = True; get_end = time.perf_counter()
        if run.frame_id != received.frame_id: raise ValueError('Synchronous frame ID mismatch')
        owned, copy_seconds, output_bytes = [], [], []
        for output, binding in zip(outputs, bindings):
            size = math.prod(binding['shape'])
            if not output.buf or output.size != size * 4: raise ValueError('Logical FP32 output pointer/size mismatch')
            started = time.perf_counter()
            view = np.ctypeslib.as_array(ct.cast(output.buf, ct.POINTER(ct.c_float)), shape=(size,))
            value = np.empty(binding['shape'], dtype=np.float32)
            np.copyto(value.reshape(-1), view); owned.append(value)
            copy_seconds.append(time.perf_counter() - started); output_bytes.append(int(output.size))
        copy_end = time.perf_counter()
        timing.update(input_set_seconds=set_end-begin, run_api_seconds=run_end-set_end,
            outputs_get_seconds=get_end-run_end, all_owned_output_copy_seconds=copy_end-get_end,
            output_copy_seconds_by_index=copy_seconds, output_bytes_by_index=output_bytes,
            original_output_copy_seconds=sum(copy_seconds[:2]), tap_output_copy_seconds=sum(copy_seconds[2:]),
            extra_tap_output_bytes=sum(output_bytes[2:]), runtime_frame_id=int(run.frame_id))
    finally:
        if got or any(output.buf for output in outputs):
            acquired = sum(bool(output.buf) for output in outputs)
            counters['output_sets_release_attempts'] += 1
            release_begin = time.perf_counter()
            code = api.rknn_outputs_release(context, count, outputs)
            timing.update(outputs_release_seconds=time.perf_counter()-release_begin,
                          output_buffer_released=code == 0)
            if code == 0:
                counters['output_sets_released'] += 1; counters['outputs_released'] += acquired
            else: cleanup_errors.append(f'rknn_outputs_release returned {code}')
            base.check_call(code, 'rknn_outputs_release')
        timing['io_total_seconds'] = time.perf_counter()-begin
    # Finite checks, comparisons, hashing, saving and profiling are outside IO.
    if any(not value.flags.owndata or not np.isfinite(value).all() for value in owned):
        raise ValueError('All SDK outputs, including taps, must be owned and finite')
    return owned, timing


def summarize(calls):
    result = {}
    for role in ('A', 'B'):
        rows = [row for row in calls if row['role'] == role]
        if len(rows) != 4: raise ValueError('Exactly four normal calls per route required')
        result[role] = dict(cold_io_seconds=rows[0]['timing']['io_total_seconds'],
            all_count=4, all_mean_io_seconds=sum(x['timing']['io_total_seconds'] for x in rows)/4,
            excluding_first_count=3,
            excluding_first_mean_io_seconds=sum(x['timing']['io_total_seconds'] for x in rows[1:])/3,
            excluding_first_mean_tap_copy_seconds=sum(x['timing']['tap_output_copy_seconds'] for x in rows[1:])/3)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=BASE/'src')
    p.add_argument('--baseline-manifest', type=Path, default=BASE/'experiments/vggt-v55-pair-graphs-r1/compiled.json')
    p.add_argument('--capture-dir', type=Path, default=BASE/'experiments/vggt-v53-capture-r1')
    p.add_argument('--candidate-manifest', type=Path, required=True)
    p.add_argument('--candidate-manifest-sha256', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if not re.fullmatch('[0-9a-f]{64}', args.candidate_manifest_sha256): p.error('Lowercase SHA256 required')
    if (sys.platform != 'linux' or platform.machine().lower() not in ('aarch64', 'arm64')
            or b'rockchip,rk3588' not in Path('/proc/device-tree/compatible').read_bytes()):
        p.error('Actual SDK execution is restricted to Linux RK3588S')
    output_dir = args.output_dir.resolve(); output_dir.mkdir(parents=True, exist_ok=False)
    affinity, old_path = os.sched_getaffinity(0), sys.path[:]
    contexts, pins, api, base = [], {}, None, None
    report = dict(schema='v61_original_VGGT_pair00_FC1_tap_SDK_probe', status='failed',
        script_sha256=sha(__file__), original_model_only=True, candidate_accepted=False,
        full_model_or_trajectory_or_realtime_validated=False, core_mask=7, normal_init_flags=0,
        profile_init_flags=8, normal_order=list(ORDER), hidden_warmup_calls=0, normal=[], profiles=[],
        contexts_created=0, contexts_destroyed=0, cleanup_errors=[],
        counters={name: 0 for name in ('sdk_run_attempts', 'sdk_runs_succeeded', 'output_sets_get_attempts',
            'output_sets_acquired', 'output_sets_release_attempts', 'output_sets_released', 'outputs_acquired', 'outputs_released')},
        timing_scope='Normal SDK input/run/output retrieval, all owned output copies and release; excludes diagnostics/NPZ/profile',
        extra_tap_IO_included_not_subtracted=True, tap_only_finite_no_FP32_reference=True)
    try:
        if not {4, 5, 6, 7} <= affinity: raise ValueError('Model CPUs4..7 required')
        os.sched_setaffinity(0, {4, 5, 6, 7}); report['cpu_affinity'] = sorted(os.sched_getaffinity(0))
        repo = args.repo.resolve(strict=True)
        for name, digest in SOURCE_PINS.items(): pin(repo/name, digest, pins)
        spec = importlib.util.spec_from_file_location('_v61_original_ABI', repo/'deploy/rk3588/validation/run-v12-rknn-block-board.py')
        base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
        report['board'] = base.require_board(); sys.path.insert(0, str(repo))
        from vggt_slam.rknn_aggregator_blocks import artifact, floating_attr, LIBRARY_SHA
        from vggt_slam.rknn_aggregator_pairs import pair_output_attr
        import numpy as np
        for name in ('vggt_slam/rknn_aggregator_blocks.py', 'vggt_slam/rknn_aggregator_pairs.py'):
            if Path(sys.modules[name[:-3].replace('/', '.')].__file__).resolve() != repo/name:
                raise ValueError('Frozen helper import escaped selected source root')
        baseline_path = pin(args.baseline_manifest, BASELINE_SHA, pins); baseline = json.loads(baseline_path.read_text())
        candidate_path = pin(args.candidate_manifest, args.candidate_manifest_sha256, pins); candidate = json.loads(candidate_path.read_text())
        capture_path = pin(args.capture_dir/'capture.json', CAPTURE_SHA, pins); capture = json.loads(capture_path.read_text())
        if (baseline['schema'] != 'v55_original_VGGT_24_frame_global_pair_RKNN_manifest' or baseline['checkpoint'] != CHECKPOINT
                or baseline['capture_sha256'] != CAPTURE_SHA or len(baseline['blocks']) != 24
                or candidate['schema'] != 'v61_original_VGGT_pair00_FC1_tap_RKNN_manifest'
                or candidate['status'] != 'compiled_not_numerically_accepted' or len(candidate['blocks']) != 1
                or candidate['original_baseline_manifest_sha256'] != BASELINE_SHA
                or candidate['original_pair_onnx_sha256'] != PAIR_GRAPH_SHA or candidate['checkpoint'] != CHECKPOINT
                or candidate['configuration'] != CONFIGURATION
                or candidate['build_options'] != dict(do_quantization=False, dataset=None, auto_hybrid=False)
                or candidate['released'] is not True or candidate['cleanup_errors']
                or candidate['candidate_accepted'] is not False
                or [x['name'] for x in candidate['operations']] != ['config', 'load_onnx', 'build', 'export_rknn', 'release']
                or any(x['status'] != 'returned' or x['code'] not in (0, None) for x in candidate['operations'])
                or capture['checkpoint'] != CHECKPOINT or capture['cleanup_errors']
                or capture['full_model_forward_completed'] != 1 or len(capture['blocks']) != 48):
            raise ValueError('Frozen original parent/candidate/capture compile provenance differs')
        original, tapped = baseline['blocks'][0], candidate['blocks'][0]
        if (original['index'] != 0 or original['role'] != 'pair' or original['sha256'] != PAIR_GRAPH_SHA
                or original['output_bindings'] != BINDINGS[:2] or tapped['role'] != 'pair' or tapped['index'] != 0
                or tapped['output_bindings'] != BINDINGS or tapped['input_shape'] != [2, 53, 1024]
                or tapped['input_name'] != 'input'
                or any(tapped[key] != original[key] for key in ('parents', 'parameter_raw_sha256', 'positions'))
                or any(tapped[key] is not True for key in ('only_two_pre_GELU_outputs_appended',
                    'removing_two_appended_outputs_recovers_original_protobuf_bytes', 'all_original_nodes_and_initializers_bytewise_unchanged'))):
            raise ValueError('Only two pre-GELU outputs may be appended to the complete original pair00')
        for name, digest in baseline['source_hashes'].items():
            path = (repo/name).resolve(strict=True)
            if not path.is_relative_to(repo): raise ValueError('Original source path escapes selected root')
            pin(path, digest, pins)
        def bound_artifact(directory, row):
            path = artifact(directory, row); pins[path] = row['sha256']; return path
        bound_artifact(baseline_path.parent, original); bound_artifact(candidate_path.parent, tapped)
        models = {'A': bound_artifact(baseline_path.parent, original['rknn']),
                  'B': bound_artifact(candidate_path.parent, tapped['rknn'])}
        def load(row, shape):
            value = np.load(bound_artifact(args.capture_dir.resolve(strict=True), row), allow_pickle=False)
            if (value.shape != tuple(shape) or value.dtype != np.float32 or not value.flags.c_contiguous
                    or not np.isfinite(value).all() or row['shape'] != shape or row['dtype'] != 'float32'
                    or row['tensor_bytes'] != value.nbytes or hashlib.sha256(memoryview(value).cast('B')).hexdigest() != row['raw_sha256']):
                raise ValueError('Captured original tensor raw identity differs')
            return value
        frame, global_ = capture['blocks'][:2]
        if (frame['role'], frame['index'], global_['role'], global_['index']) != ('frame', 0, 'global', 0):
            raise ValueError('Original frame0/global0 reference order differs')
        array = load(frame['input'], [2, 53, 1024])
        references = [load(frame['output'], BINDINGS[0]['shape']), load(global_['output'], BINDINGS[1]['shape'])]
        input_sha = frame['input']['raw_sha256']
        sdk_path = pin('/usr/lib/librknnrt.so', LIBRARY_SHA, pins); api = base.bind_api(ct.CDLL(str(sdk_path)))
        report.update(checkpoint=CHECKPOINT, baseline_manifest_sha256=BASELINE_SHA, original_pair_onnx_sha256=PAIR_GRAPH_SHA,
            candidate_manifest_sha256=args.candidate_manifest_sha256, capture_sha256=CAPTURE_SHA,
            input=frame['input'], original_FP32_references=[frame['output'], global_['output']], library_sha256=LIBRARY_SHA,
            candidate_export_manifest_sha256=candidate['export_manifest_sha256'],
            candidate_check_receipt_sha256=candidate['check_receipt_sha256'],
            candidate_compile_receipt_sha256=candidate['compile_receipt_sha256'], context_queries=[])
        def init(role, flags):
            context = base.CONTEXT(); contexts.append(context)
            filename = ct.create_string_buffer(str(models[role]).encode())
            code = api.rknn_init(ct.byref(context), ct.cast(filename, ct.c_void_p), 0, flags, None)
            report['contexts_created'] += int(bool(context.value)); base.check_call(code, 'rknn_init')
            if not context.value: raise ValueError('Empty RKNN context')
            base.check_call(api.rknn_set_core_mask(context, 7), 'rknn_set_core_mask')
            version = base.query(api, context, 5, base.SDKVersion())
            if not re.match(r'^2\.3\.2(?:\D|$)', bytes(version.api_version).decode()): raise ValueError('SDK2.3.2 required')
            bindings = BINDINGS[:2] if role == 'A' else BINDINGS
            counts = base.query(api, context, 0, base.InputOutputNum())
            if (counts.n_input, counts.n_output) != (1, len(bindings)): raise ValueError('Logical input/output count differs')
            inp = floating_attr(base, base.query(api, context, 1, base.TensorAttr(index=0)), (2, 53, 1024), 'input')
            outs = [pair_output_attr(base, base.query(api, context, 2, base.TensorAttr(index=binding['index'])), binding) for binding in bindings]
            report['context_queries'].append(dict(role=role, flags=flags, model_sha256=sha(models[role]),
                sdk_api_version=bytes(version.api_version).decode(), sdk_driver_version=bytes(version.drv_version).decode(), input=inp, outputs=outs))
            return context, inp['format'], bindings
        def close(context):
            if context.value:
                base.check_call(api.rknn_destroy(context), 'rknn_destroy'); context.value = 0; report['contexts_destroyed'] += 1
        def save(tag, values):
            path = output_dir/(tag+'.npz')
            with path.open('xb') as stream: np.savez(stream, **{f'output{i}': x for i, x in enumerate(values)})
            return dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path), outputs=[dict(
                index=i, shape=list(x.shape), dtype='float32', tensor_bytes=x.nbytes,
                raw_sha256=hashlib.sha256(memoryview(x).cast('B')).hexdigest(), finite=True) for i, x in enumerate(values)])
        routes = {role: init(role, 0) for role in ('A', 'B')}; first = {}
        for sequence, role in enumerate(ORDER):
            context, fmt, bindings = routes[role]
            values, timing = execute(base, api, np, context, array, fmt, bindings, report['counters'], report['cleanup_errors'])
            cold = role not in first
            if cold: first[role] = values
            report['normal'].append(dict(sequence=sequence, role=role, cold=cold, timing=timing,
                original_FP32=[compare(np, x, y) for x, y in zip(values[:2], references)],
                same_input_first_A=[compare(np, x, y) for x, y in zip(values[:2], first['A'][:2])],
                repeat_of_first_role=[compare(np, x, y) for x, y in zip(values[:2], first[role][:2])],
                artifact=save(f'normal-{sequence:02d}-{role}', values)))
        report['normal_summary'] = summarize(report['normal'])
        for context, _, _ in routes.values(): close(context)
        for role in ('A', 'B'):
            context = None
            try:
                context, fmt, bindings = init(role, 8)
                values, timing = execute(base, api, np, context, array, fmt, bindings, report['counters'], report['cleanup_errors'])
                detail = base.query(api, context, 3, base.PerfDetail())
                if not detail.perf_data or not 0 < detail.data_len <= 8 << 20:
                    raise ValueError('Invalid SDK profile pointer/length')
                raw = ct.string_at(detail.perf_data, detail.data_len)
                path = output_dir/f'profile-{role}.txt'; path.write_bytes(raw)
                report['profiles'].append(dict(role=role, flags=8, timing_informational_not_normal=timing,
                    original_FP32=[compare(np, x, y) for x, y in zip(values[:2], references)],
                    same_input_first_normal_role=[compare(np, x, y) for x, y in zip(values[:2], first[role][:2])],
                    artifact=save(f'profile-{role}', values), perf_detail=dict(query_command=3,
                        bytes=len(raw), raw_sha256=hashlib.sha256(raw).hexdigest(), operator_targets_interpreted_by_probe=False),
                    raw_profile=dict(file=path.name, bytes=len(raw), sha256=sha(path))))
            finally:
                if context is not None: close(context)
        if hashlib.sha256(memoryview(array).cast('B')).hexdigest() != input_sha: raise ValueError('Frozen input mutated')
        if (report['counters']['sdk_run_attempts'] != 10 or report['counters']['sdk_runs_succeeded'] != 10
                or report['counters']['outputs_acquired'] != 30 or report['counters']['outputs_released'] != 30
                or report['counters']['output_sets_acquired'] != 10 or report['counters']['output_sets_released'] != 10):
            raise ValueError('Exactly8 normal+2 profile calls and30 owned/released outputs required')
        report['status'] = 'completed_component_diagnostic_not_adopted'
    except BaseException as exc:
        report.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        for context in reversed(contexts):
            if context.value and api is not None:
                try:
                    base.check_call(api.rknn_destroy(context), 'rknn_destroy')
                    context.value = 0; report['contexts_destroyed'] += 1
                except BaseException as exc: report['cleanup_errors'].append('context destroy: '+str(exc))
        report['live_contexts'] = sum(bool(x.value) for x in contexts)
        try:
            if any(sha(path) != digest for path, digest in pins.items()): raise ValueError('Frozen source/artifact changed')
            report['frozen_pins_unchanged'] = True
        except BaseException as exc: report['cleanup_errors'].append('identity: '+str(exc))
        try:
            os.sched_setaffinity(0, affinity)
            if os.sched_getaffinity(0) != affinity: raise ValueError('CPU affinity was not restored')
            report['affinity_restored'] = True
        except BaseException as exc: report['cleanup_errors'].append('affinity: '+str(exc))
        sys.path[:] = old_path
        if report['cleanup_errors'] or report['contexts_created'] != report['contexts_destroyed']: report['status'] = 'failed'
        (output_dir/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'], result_sha256=sha(output_dir/'result.json'))), flush=True)
    return 0 if report['status'] == 'completed_component_diagnostic_not_adopted' else 1


if __name__ == '__main__':
    raise SystemExit(main())
