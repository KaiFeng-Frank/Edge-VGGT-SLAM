#!/usr/bin/env python3
"""Board-only ABBA of unchanged native global0 FP16 and calibrated W8A8.

Each phase runs both heldout inputs, with a fresh context per call. SDK normal
FP32 pass0 input / want_float1 output performs conversion; no manual quantizer,
native packed memory, alternative model, or neural reference execution is used.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import time
import traceback

BASE = Path('/mnt/vggt-data/vggt-slam2')
E = BASE / 'experiments'
OWNED_SHA = '0bfcb295ad5490e8196e8fcc1244e11a99b07bfb3ba7d5c350cb8c81d3ff3be4'
HELPER_SHA = '56ce45635a5d010e862bf59db2f9fcc6e8ec642b2b023c8c40cad0bd5808050f'
BASE_COMPILE_SHA = '711ee6dcc45134b4695ee8b24ec47ee0c7f43356b9ce047c09051e1ed108a33c'
BASE_MODEL_SHA = '8f9ca55e5f4f4b8646b6a5dbf041a590c168b0e1e6d390e4ce9d594dddd5654d'
GRAPH_SHA = 'c44094793c3b590c6066cbdd634cf0c4d8bd5b75987c8b491db7b17b2201f55d'
CHECKPOINT = dict(bytes=5026874952, sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')
SHAPE = (1, 2082, 1024)
CALIBRATION_IDS = ['calib00', 'calib01', 'calib02', 'calib03']
EVALUATION_IDS = ['heldout00', 'heldout01']
FIXED_GATE = dict(relative_l2_max=1e-5, max_abs_max=1e-3, rtol=1e-4, atol=1e-5)
FLOAT_CONFIG = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
    mean_values=None, std_values=None, quant_img_RGB2BGR=False, model_pruning=False,
    remove_reshape=False, quantize_weight=False, enable_flash_attention=False)
QUANT_CONFIG = dict(FLOAT_CONFIG, quantized_dtype='w8a8', quantized_algorithm='normal', quantized_method='channel')
COUNTERS = ('contexts_created', 'contexts_destroyed', 'runs_completed', 'output_buffers_acquired', 'output_buffers_released')
SUCCESS = 'completed_original_global0_W8A8_component_ABBA_not_quality_accepted'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for piece in iter(lambda: stream.read(4 << 20), b''):
            digest.update(piece)
    return digest.hexdigest()


def raw_sha(value):
    return hashlib.sha256(value.tobytes(order='C')).hexdigest()


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def floating_or_int8_context(parent):
    class QuantLogicalContext(parent):
        def _descriptors(self, command, shapes, names):
            rows = [self.base.attr_record(self.base.query(self.api, self.context, command,
                    self.base.TensorAttr(index=i))) for i in range(len(shapes))]
            require(len(rows) == 1 and names is not None and len(names) == 1 and
                    rows[0]['index'] == 0 and rows[0]['name'] == names[0] and tuple(shapes[0]) == SHAPE,
                    'Exact original one-input/one-output global0 ABI required')
            for row, shape in zip(rows, shapes):
                identity = row['type'] == 1 and row['quantization_type'] == 2 and row['zero_point'] == 0 and row['scale'] == 1.
                floating = row['type'] in (0, 1) and (row['quantization_type'] == 0 or identity)
                int8 = row['type'] == 2 and row['quantization_type'] == 2 and -128 <= row['zero_point'] <= 127 and row['scale'] > 0
                item_bytes = {0: 4, 1: 2, 2: 1}.get(row['type'], 0)
                require((floating or int8) and math.isfinite(row['scale']) and row['pass_through'] == 0 and
                        row['format'] in (0, 3) and row['shape'] == list(shape) and
                        row['n_elems'] == math.prod(shape) and row['size'] == math.prod(shape) * item_bytes,
                        'Unsupported floating/signed-affine-INT8 logical descriptor: ' + str(row))
                row['caller_API'] = 'FP32 pass_through0 input / want_float1 owned FP32 output; SDK conversions'
            return rows
    return QuantLogicalContext


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture', 'capture-source', 'candidate-compile', 'compiler-source'):
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--baseline-compile', type=Path, default=E / 'vggt-v98-global-fc2k4-compile-r1/result.json')
    parser.add_argument('--owned-runtime', type=Path, default=E / 'vggt-v100-source-r1/native_heads_runtime.py')
    parser.add_argument('--runtime-helper', type=Path, default=E / 'vggt-v99-source-r1/run_native_block_r1.py')
    parser.add_argument('--source-root', type=Path, default=BASE / 'src')
    parser.add_argument('--library', type=Path, default=Path('/usr/lib/librknnrt.so'))
    parser.add_argument('--core-mask', type=int, choices=(1, 7), default=7)
    parser.add_argument('--timeout-seconds', type=float, default=170.)
    parser.add_argument('--output-dir', type=Path, required=True, help='Use /tmp on rootfs for the four saved outputs')
    a = parser.parse_args()
    require(platform.machine() == 'aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),
            'Actual SDK execution is RK3588/S only')
    require(0 < a.timeout_seconds <= 170, 'Use existing 180s/75C/1GHz controller with cleanup reserve')
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    receipt = out / 'result.json'
    started = time.monotonic()
    deadline = started + a.timeout_seconds
    env, affinity = dict(os.environ), os.sched_getaffinity(0)
    handlers, pins, owners = {}, {}, []
    r = dict(schema='v115_original_native518_global0_W8A8_ABBA', status='running', script_sha256=sha(__file__),
        original_model_only=True, component_only=True, graph_sha256=GRAPH_SHA, checkpoint=CHECKPOINT,
        schedule=list('ABBA'), heldout_order=EVALUATION_IDS, calls_expected=8, init_flags=0, core_mask=a.core_mask,
        expected_NPU_hz=1000000000, maximum_live_contexts=1, original_layers_tokens_and_resolution_preserved=True,
        input_API='Contiguous FP32/pass0; conversion performed by SDK', output_API='want_float1; owned FP32 copy before release',
        manual_input_quantization=False, native_packed_IO=False, all_INT8_or_all_NPU_placement_verified=False,
        profile_collected=False, candidate_accepted=False, full_model_trajectory_realtime_or_8GB_accepted=False,
        system_or_frequency_writes=False, fixed_FP32_graph_gate=FIXED_GATE, calls=[], records=[], cleanup_errors=[],
        timing_scope='Complete fresh-context lifecycle: frozen constructor source/model/library hashes, init/query, normal FP32 SDK IO/conversions/run, owned output copy/release, finite checks and destruction.',
        outside_timer='Startup provenance/array checks, platform samples, output hashes/saves, repeat and CPU-reference/residual-update statistics.',
        sample_policy='First invocation of each route separate; remaining three pooled. Each heldout has two calls/route; every call creates/destroys its context.',
        output_storage_policy='Save first A/B output per heldout only (four NPY). Every call retains its own raw hash and repeat comparison. A differing repeat has no new NPY and is not misidentified as the first saved payload.')
    atomic(receipt, r)

    def pin(path, digest, size=None):
        path = Path(path).resolve(strict=True)
        require(len(digest) == 64 and all(c in '0123456789abcdef' for c in digest) and
                (size is None or path.stat().st_size == size) and sha(path) == digest,
                'Pinned source/artifact differs: ' + str(path))
        pins[path] = digest
        return path

    def artifact(folder, row, absolute=False):
        relative = Path(row['file'])
        require(absolute or (not relative.is_absolute() and '..' not in relative.parts), 'Safe original artifact path required')
        path = relative if relative.is_absolute() else folder / relative
        return pin(path, row['sha256'], row['bytes'])

    try:
        def interrupted(number, frame):
            raise InterruptedError('External bounded controller stopped global0 ABBA')
        for number in (signal.SIGTERM, signal.SIGINT):
            handlers[number] = signal.signal(number, interrupted)
        require({4, 5, 6, 7} <= affinity, 'Original A76 CPU affinity unavailable')
        os.sched_setaffinity(0, {4, 5, 6, 7})
        os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
        pin(__file__, r['script_sha256'])
        cp = pin(a.capture, a.capture_sha256)
        candidate_path = pin(a.candidate_compile, a.candidate_compile_sha256)
        capture = json.loads(cp.read_text())
        candidate = json.loads(candidate_path.read_text())
        pin(a.capture_source, a.capture_source_sha256)
        pin(a.compiler_source, a.compiler_source_sha256)
        require(capture['schema'] == 'v115_original_native518_global0_representative_capture' and
                capture['status'] == 'completed_original_native518_global0_input_capture' and
                capture['script_sha256'] == a.capture_source_sha256 and capture['checkpoint'] == CHECKPOINT and
                capture['calibration_ids'] == CALIBRATION_IDS and capture['evaluation_ids'] == EVALUATION_IDS and
                capture['original_CPU_global0_forward_count'] == 6 and len(capture['original_parameters']) == 18 and
                not capture['cleanup_errors'], 'Completed original four-calibration/two-heldout capture required')
        for key in ('source_and_inputs_unchanged', 'model_state_unchanged', 'hooks_removed', 'affinity_restored',
                    'environment_restored', 'sys_path_restored', 'all_NPU_contexts_closed', 'resource_counts_balanced'):
            require(capture[key] is True, 'Capture cleanup/provenance failed: ' + key)
        require(candidate['schema'] == 'v115_original_native518_global0_W8A8_compile' and
                candidate['status'] == 'compiled_original_global0_W8A8_not_quality_accepted' and
                candidate['script_sha256'] == a.compiler_source_sha256 and candidate['checkpoint'] == CHECKPOINT and
                candidate['graph_sha256'] == candidate['onnx']['sha256'] == GRAPH_SHA and candidate['onnx']['bytes'] == 51469885 and
                candidate['capture']['sha256'] == a.capture_sha256 and candidate['capture_source']['sha256'] == a.capture_source_sha256 and
                candidate['original_parameters'] == capture['original_parameters'] and candidate['fixed_gate'] == FIXED_GATE and
                candidate['configuration'] == QUANT_CONFIG and candidate['evaluation_ids'] == EVALUATION_IDS and
                candidate['original_FP32_gate_passed'] and candidate['sdk_released'] and candidate['sdk_release_code'] in (None, 0) and
                candidate['all_operations_succeeded'] and candidate['session_dropped'] and not candidate['cleanup_errors'],
                'Clean unchanged-graph calibrated candidate with both board FP32 gates required')
        require(candidate['build_options'] == dict(do_quantization=True,
                dataset=str(candidate_path.parent / 'dataset.txt'), auto_hybrid=False), 'Exact four-input quantization build required')
        for key in ('source_and_inputs_unchanged', 'affinity_restored', 'environment_restored', 'cwd_restored'):
            require(candidate[key] is True, 'Compiler cleanup/provenance failed: ' + key)
        require([row['operation'] for row in candidate['calls']] == ['config', 'load_onnx', 'build', 'export_rknn'] and
                all(row['status'] == 'completed' and row['returncode'] in (None, 0) for row in candidate['calls']),
                'All four compiler operations must have completed successfully')
        baseline_path = pin(a.baseline_compile, BASE_COMPILE_SHA)
        baseline = json.loads(baseline_path.read_text())
        require(baseline['schema'] == 'original_native518_single_block_FP16_compile' and baseline['status'] == 'compiled_not_quality_accepted' and
                baseline['onnx']['sha256'] == GRAPH_SHA and baseline['rknn']['sha256'] == BASE_MODEL_SHA and
                baseline['configuration'] == FLOAT_CONFIG and baseline['build_options'] == dict(do_quantization=False, dataset=None, auto_hybrid=False) and
                baseline['sdk_released'] and not baseline['cleanup_errors'], 'Frozen original complete global0 FP16 baseline required')
        O = module(pin(a.owned_runtime, OWNED_SHA), '_v115_frozen_owned')
        H = module(pin(a.runtime_helper, HELPER_SHA), '_v115_frozen_normal_IO')
        IO = floating_or_int8_context(O.OwnedContext)
        pin(a.library, O.SDK_SHA)
        pin(a.source_root / 'deploy/rk3588/validation/run-v12-rknn-block-board.py', O.ABI_SHA)
        import numpy as np

        def load(folder, row, absolute=False):
            value = np.load(artifact(folder, row, absolute), allow_pickle=False)
            require(value.shape == SHAPE and value.dtype == np.float32 and np.isfinite(value).all() and
                    row['shape'] == list(SHAPE) and row['dtype'] == 'float32' and row['finite'] and
                    row['tensor_bytes'] == value.nbytes and raw_sha(value) == row['raw_sha256'], 'Original FP32 array differs')
            return np.array(value, copy=True, order='C')

        def compare(actual, reference):
            actual, reference = actual.astype(np.float64), reference.astype(np.float64)
            delta = actual - reference
            error, rn, an = [float(np.linalg.norm(v.ravel())) for v in (delta, reference, actual)]
            outside = int(np.count_nonzero(np.abs(delta) > 1e-5 + 1e-4 * np.abs(reference)))
            return dict(relative_l2=error / rn if rn else None, l2_error=error, reference_l2_norm=rn,
                max_abs=float(np.abs(delta).max()), cosine=float(actual.ravel() @ reference.ravel()) / (an * rn) if an and rn else None,
                outside=outside, strict_allclose=outside == 0, rtol=1e-4, atol=1e-5)

        records = capture['records']
        require([row['id'] for row in records] == CALIBRATION_IDS + EVALUATION_IDS and
                [row['role'] for row in records] == ['calibration'] * 4 + ['heldout'] * 2,
                'Exact original capture case order/partition required')
        values, references, input_paths = {}, {}, []
        for row in records:
            value = load(cp.parent, row['input'])
            input_paths.append(str(artifact(cp.parent, row['input'])))
            if row['role'] == 'heldout':
                require(row['reference']['saved'] is True, 'Actual heldout CPU reference payload required')
                values[row['id']] = value
                references[row['id']] = load(cp.parent, row['reference'])
            r['records'].append(dict(id=row['id'], role=row['role'], source_pair_indices=row['source_pair_indices'],
                source_images=row['source_images'], input=copy.deepcopy(row['input']), reference=copy.deepcopy(row['reference'])))
        require(len({row['input']['raw_sha256'] for row in records}) == 6, 'Calibration/heldout activations must be distinct')
        captured_dataset = artifact(cp.parent, capture['dataset'])
        require(capture['dataset']['line_count'] == 4 and capture['dataset']['input_count'] == 1 and
                captured_dataset.read_text().splitlines() == input_paths[:4], 'Only the four calibration inputs belong to the dataset')
        dataset = artifact(candidate_path.parent, candidate['dataset'])
        require(candidate['dataset']['line_count'] == 4 and candidate['dataset']['input_count'] == 1 and
                candidate['dataset']['calibration_ids'] == CALIBRATION_IDS and dataset.read_text().splitlines() == input_paths[:4] and
                candidate['dataset']['rows'] == [dict(id=row['id'], input_sha256=row['input']['sha256'], path=path)
                    for row, path in zip(records[:4], input_paths[:4])], 'Compiler calibration lineage differs')
        require([row['id'] for row in candidate['FP32_ONNX_checks']] == EVALUATION_IDS, 'Both heldout FP32 gates required')
        for checked, record in zip(candidate['FP32_ONNX_checks'], records[4:]):
            require(checked['input_sha256'] == record['input']['sha256'] and checked['reference_sha256'] == record['reference']['sha256'] and
                    checked['comparison']['fixed_gate_passed'], 'Correct same-input heldout FP32 lineage required')
            actual = load(candidate_path.parent, checked['output'], absolute=True)
            error = compare(actual, references[record['id']])
            require(error['strict_allclose'] and error['relative_l2'] is not None and error['relative_l2'] <= 1e-5 and
                    error['max_abs'] <= 1e-3, 'Saved heldout FP32 graph gate does not independently pass')
        models = {'A': artifact(baseline_path.parent, baseline['rknn']), 'B': artifact(candidate_path.parent, candidate['rknn'])}
        digests = {'A': baseline['rknn']['sha256'], 'B': candidate['rknn']['sha256']}
        r.update(capture_sha256=a.capture_sha256, capture_source_sha256=a.capture_source_sha256,
            candidate_compile_sha256=a.candidate_compile_sha256, compiler_source_sha256=a.compiler_source_sha256,
            baseline_compile_sha256=BASE_COMPILE_SHA, models={route: dict(file=str(path), sha256=digests[route]) for route, path in models.items()},
            configuration=candidate['configuration'], build_options=candidate['build_options'], original_parameters=capture['original_parameters'])

        def observe():
            require(time.monotonic() < deadline, 'Bounded component deadline reached')
            sample = H.sample()
            require(sample['npu_hz'] == 1000000000 and sample['temperatures_c'] and
                    max(sample['temperatures_c'].values()) < 75, 'Clock/thermal guard failed')
            return sample

        first_outputs, saved_outputs, route_count = {}, {}, {'A': 0, 'B': 0}
        for phase, route in enumerate('ABBA'):
            for record in records[4:]:
                ident = record['id']
                x = values[ident]
                call = dict(route=route, input_id=ident, phase=phase, route_ordinal=route_count[route],
                    first_of_route=route_count[route] == 0, input_raw_sha256=record['input']['raw_sha256'], platform_before=observe())
                r['calls'].append(call)
                owner = None
                tick = time.perf_counter()
                try:
                    try:
                        owner = IO(models[route], digests[route], input_shapes=[SHAPE], output_shapes=[SHAPE],
                            input_names=['input'], output_names=['output'], core_mask=a.core_mask, flags=0, timeout_ms=30000,
                            source_root=a.source_root, library=a.library, runtime_helper=a.runtime_helper)
                    except BaseException as exc:
                        owner = getattr(exc, 'owned_context', None)
                        raise
                    output, = owner.run([x])
                finally:
                    if owner is not None:
                        owners.append(owner)
                        owner.close()
                call['elapsed_seconds'] = time.perf_counter() - tick
                call['resources'] = owner.metrics()
                call['SDK_IO_timings'] = copy.deepcopy(owner.last_timings)
                require(raw_sha(x) == record['input']['raw_sha256'], 'Original input was changed')
                raw = raw_sha(output)
                key = (route, ident)
                if key not in first_outputs:
                    first_outputs[key] = output
                    file = out / (route + '-' + ident + '.npy')
                    with file.open('xb') as stream:
                        np.save(stream, output, allow_pickle=False)
                    saved_outputs[key] = dict(file=file.name, bytes=file.stat().st_size, sha256=sha(file), raw_sha256=raw,
                        shape=list(SHAPE), dtype='float32', tensor_bytes=output.nbytes, finite=True)
                same = raw == saved_outputs[key]['raw_sha256']
                call.update(output=copy.deepcopy(saved_outputs[key]) if same else dict(raw_sha256=raw, shape=list(SHAPE), dtype='float32', finite=True, saved=False),
                    repeat=dict(raw_sha256_equal_to_first=same, comparison=compare(output, first_outputs[key])), platform_after=observe(),
                    comparison=dict(whole_output_vs_CPU=compare(output, references[ident]),
                                    residual_update_vs_CPU=compare(output.astype(np.float64) - x.astype(np.float64),
                                        references[ident].astype(np.float64) - x.astype(np.float64))))
                route_count[route] += 1
                atomic(receipt, r)
                print(json.dumps(dict(route=route, input_id=ident, phase=phase, elapsed_seconds=call['elapsed_seconds'])), flush=True)
        r['candidate_vs_baseline_error'] = {}
        for ident in EVALUATION_IDS:
            aa, bb, ref, x = first_outputs[('A', ident)], first_outputs[('B', ident)], references[ident], values[ident]
            r['candidate_vs_baseline_error'][ident] = dict(A=compare(aa, ref), B=compare(bb, ref), B_vs_A=compare(bb, aa),
                residual_A=compare(aa.astype(np.float64) - x.astype(np.float64), ref.astype(np.float64) - x.astype(np.float64)),
                residual_B=compare(bb.astype(np.float64) - x.astype(np.float64), ref.astype(np.float64) - x.astype(np.float64)))
        def summary(calls):
            times = [row['elapsed_seconds'] for row in calls]
            return dict(count=len(times), mean_seconds=sum(times) / len(times), min_seconds=min(times), max_seconds=max(times))
        r.update(status=SUCCESS, all_repeat_byte_equal=all(row['repeat']['raw_sha256_equal_to_first'] for row in r['calls']),
            summary={route: dict(first=summary([row for row in r['calls'] if row['route'] == route and row['first_of_route']]),
                                 subsequent=summary([row for row in r['calls'] if row['route'] == route and not row['first_of_route']]),
                                 all=summary([row for row in r['calls'] if row['route'] == route])) for route in 'AB'},
            saved_output_files=4, saved_output_bytes=sum(row['bytes'] for row in saved_outputs.values()),
            quality_acceptance_threshold_added=False,
            residual_diagnostic_scope='Same exact x; compare y-x against original_CPU_y-x. This is a component/update diagnostic, not trajectory quality.')
        require(r['saved_output_bytes'] < 34 * 1024 * 1024, 'Four-output storage bound exceeded')
    except BaseException as exc:
        r.update(status='failed', error=type(exc).__name__ + ': ' + str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        for owner in reversed(owners):
            try:
                owner.close()
            except BaseException as exc:
                r['cleanup_errors'].append('Owned context close: ' + repr(exc))
        r['resources'] = [owner.metrics() for owner in owners]
        r['SDK_totals'] = {key: sum(row[key] for row in r['resources']) for key in COUNTERS}
        r['resources_balanced'] = all(row['closed'] and row['resource_counts_balanced'] and
            not row['live_contexts'] and not row['live_output_buffers'] and not row['cleanup_errors'] for row in r['resources'])
        r['complete_expected_execution'] = r['SDK_totals'] == dict(contexts_created=8, contexts_destroyed=8,
            runs_completed=8, output_buffers_acquired=8, output_buffers_released=8) and len(r['calls']) == 8
        try:
            os.sched_setaffinity(0, affinity)
            r['affinity_restored'] = os.sched_getaffinity(0) == affinity
            os.environ.clear()
            os.environ.update(env)
            r['environment_restored'] = dict(os.environ) == env
            for number, handler in handlers.items():
                signal.signal(number, handler)
        except BaseException as exc:
            r['cleanup_errors'].append('Process restoration: ' + repr(exc))
        errors = []
        for path, digest in pins.items():
            try:
                require(sha(path) == digest, 'Pinned source/input changed: ' + str(path))
            except BaseException as exc:
                errors.append(str(exc))
        r.update(source_and_inputs_unchanged=not errors, pin_errors=errors,
            source_and_input_sha256={str(path): digest for path, digest in pins.items()}, elapsed_seconds=time.monotonic() - started,
            peak_process_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        if errors or r['cleanup_errors'] or not r['resources_balanced'] or not r.get('affinity_restored') or not r.get('environment_restored') or (r['status'] == SUCCESS and not r['complete_expected_execution']):
            r['status'] = 'failed_cleanup'
        atomic(receipt, r)
    return 0 if r['status'] == SUCCESS else 1


if __name__ == '__main__':
    raise SystemExit(main())
