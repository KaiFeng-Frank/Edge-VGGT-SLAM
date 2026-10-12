#!/usr/bin/env python3
"""Compare all four original CameraHead refinements with the FC1 split candidate."""
from __future__ import annotations
import argparse
import ctypes as ct
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

BASE = Path('/mnt/vggt-data/vggt-slam2')
ABI_SHA = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
STEP_SHA = 'dbe4fc933891581099084a569e2e847e15c582082680bf31bcf5dacc86210109'
HELPER_SHA = 'b26efe9c6cd62a2b7977e6e9c096cf0bd9bbf63076a063b9a332ef59565af258'
CAPTURE_SHA = '1c3234cfa65914a6f41634432153fa8ebb6dcf9c8d3eb3215f0955fc0ed000f1'
EXPORT_SHA = '8ccf7ff593346962766b4eae6da7eb95ce43eeaa3183a8fc0e82bf134c172503'
COMPILE_SHA = 'c8771d743d9840bcd32b22c6894f791ed239b4ad3795db703e7f7122329fbc7a'
BASELINE_ONNX_SHA = 'f89fb0e66f43a3d3b94a76ec0124d4f521fa1ce9a6c93efab5e362d3a614eb9f'
BASELINE_MODEL_SHA = '0821683fe1f9526c52777b0ef1ba45ba4c744375f9fd731f0f3d2068e4355af9'
RUNTIME_SHA = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
CONFIG = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
    mean_values=None, std_values=None, quant_img_RGB2BGR=False, model_pruning=False,
    remove_reshape=False, quantize_weight=False, enable_flash_attention=False)
BUILD = dict(do_quantization=False, dataset=None, auto_hybrid=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4 << 20), b''):
            h.update(data)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=BASE / 'src')
    p.add_argument('--capture-dir', type=Path, default=BASE / 'experiments/vggt-v15-camera/capture112-r1')
    p.add_argument('--export-dir', type=Path, default=BASE / 'experiments/vggt-v16-camera-step/export112-r1')
    p.add_argument('--baseline-model', type=Path, default=BASE / 'experiments/vggt-v16-camera-step/camera-step-fp16.rknn')
    p.add_argument('--candidate-dir', type=Path, required=True)
    p.add_argument('--candidate-sha256', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--check-onnx', action='store_true', help='Only validate both graphs against four frozen original CPU deltas')
    p.add_argument('--check-receipt', type=Path)
    p.add_argument('--check-sha256')
    p.add_argument('--profile', action='store_true')
    a = p.parse_args()
    if a.check_onnx and a.profile:
        p.error('--check-onnx and --profile are separate modes')
    if not a.check_onnx and (a.check_receipt is None or not a.check_sha256):
        p.error('SDK execution requires --check-receipt and --check-sha256 from a passed --check-onnx run')
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    mode = 'check_onnx' if a.check_onnx else ('profile' if a.profile else 'normal')
    r = dict(schema='v69_original_CameraHead_step_split_probe', status='failed', mode=mode,
        script_sha256=sha(__file__), original_model_only=True, candidate_accepted=False,
        full_model_or_realtime_validated=False, clock_writes=False, core_mask=7,
        init_flags=8 if a.profile else 0, normal_benchmark_timing=mode == 'normal',
        capture_sha256=CAPTURE_SHA, export_sha256=EXPORT_SHA, candidate_receipt_sha256=a.candidate_sha256,
        actual_execution_on_board_only=True, Torch_used=False, calls=[], graphs=[],
        cleanup_errors=[], contexts_created=0, contexts_destroyed=0, outputs_acquired=0,
        outputs_released=0, sdk_run_attempts=0, sdk_runs_succeeded=0,
        npu_run_api_called=False, npu_run_api_call_count=0,
        timing_scope='Four sequential SDK steps, original FP32 unactivated accumulation and all four activations; file saves/comparisons excluded. Profile query costs occur only in the separate profile mode.')
    contexts, pins, api, base, affinity = [], {}, None, None, os.sched_getaffinity(0)

    def pin(path, digest, size=None):
        path = Path(path).resolve(strict=True)
        require((size is None or path.stat().st_size == size) and sha(path) == digest, 'Artifact changed: ' + str(path))
        pins[str(path)] = digest
        return path

    def load_source(name, digest):
        path = pin(a.repo / 'deploy/rk3588/validation' / name, digest)
        spec = importlib.util.spec_from_file_location('_v69_' + name.replace('-', '_').replace('.', '_'), path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module

    def child(directory, name):
        root = Path(directory).resolve(strict=True); path = (root / name).resolve(strict=True)
        require(path.is_relative_to(root) and path != root, 'Artifact escaped its directory')
        return path

    try:
        base = load_source('run-v12-rknn-block-board.py', ABI_SHA)
        r['board'] = base.require_board()
        require({4, 5, 6, 7} <= affinity, 'CPUs 4..7 unavailable')
        os.sched_setaffinity(0, {4, 5, 6, 7})
        os.environ.update({key: '4' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
        r['cpu_affinity'] = sorted(os.sched_getaffinity(0))
        import numpy as np
        r['NumPy_version'] = np.__version__
        step = load_source('run-v16-camera-step-board.py', STEP_SHA)
        helper = load_source('export-v16-camera-step-board.py', HELPER_SHA)

        def raw_sha(value):
            return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast('B')).hexdigest()

        def compare(value, reference):
            require(value.shape == reference.shape and value.dtype == reference.dtype == np.float32
                    and np.isfinite(value).all() and np.isfinite(reference).all(), 'Invalid comparison array')
            d = np.abs(value.astype(np.float64) - reference.astype(np.float64))
            outside = int((d > 1e-5 + 1e-4 * np.abs(reference.astype(np.float64))).sum())
            return dict(bitwise_equal=value.tobytes() == reference.tobytes(), strict_pass=outside == 0,
                outside=outside, elements=value.size, max_abs=float(d.max()), rtol=1e-4, atol=1e-5,
                relative_l2=float(np.linalg.norm(d) / max(np.linalg.norm(reference.astype(np.float64)), np.finfo(np.float64).tiny)))

        def array_info(value):
            require(value.dtype == np.float32 and value.flags.c_contiguous and np.isfinite(value).all(), 'Finite contiguous FP32 required')
            return dict(shape=list(value.shape), dtype='float32', finite=True, bytes=value.nbytes, sha256=raw_sha(value))

        cap = json.loads(pin(a.capture_dir / 'capture.json', CAPTURE_SHA).read_text())
        export = json.loads(pin(a.export_dir / 'export.json', EXPORT_SHA).read_text())
        original_compile = json.loads(pin(a.repo / 'deploy/rk3588/validation/vggt-v16-camera-step-compile112-r1.json', COMPILE_SHA).read_text())
        require(cap['status'] == 'completed' and cap['camera_iterations'] == 4 and not cap['cleanup_errors']
            and cap['checkpoint']['sha256'] == 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
            and export['status'] == 'exported_and_structurally_checked' and not export['cleanup_errors']
            and export['capture']['sha256'] == CAPTURE_SHA and export['onnx']['sha256'] == BASELINE_ONNX_SHA
            and export['weight_sharing']['original_parameters_verified'] == 68
            and export['outer_iteration_contract']['all_four_original_iterations_array_equal'] is True
            and original_compile['status'] == 'compiled_not_runtime_validated' and original_compile['released']
            and original_compile['export_receipt_sha256'] == EXPORT_SHA, 'Original capture/export/compile lineage differs')
        item = cap['artifacts']['input.npy']
        raw = np.load(pin(a.capture_dir / 'input.npy', item['sha256'], item['bytes']), allow_pickle=False)
        require(raw.shape == (1, 2, 2048) and array_info(raw)['sha256'] == item['tensor_sha256'], 'Original raw camera tokens differ')
        arrays = {}
        for name in ('conditioning-seed.npy', 'empty-pose-token.npy', 'step-iterations.npz'):
            item = export['artifacts'][name]
            path = pin(a.export_dir / name, item['sha256'], item['bytes'])
            if name.endswith('.npz'):
                with np.load(path, allow_pickle=False) as saved:
                    arrays = {key: np.ascontiguousarray(saved[key]) for key in saved.files}
                require(set(arrays) == {f'{key}_{i}' for key in ('conditioning', 'delta', 'preactivation', 'activation') for i in range(4)}, 'All original iterations required')
                for key, value in arrays.items():
                    helper.pose_array(value)
                    require(array_info(value) == item['tensors'][key], 'Original iteration reference changed')
            elif name == 'conditioning-seed.npy': seed = np.load(path, allow_pickle=False)
            else: empty = np.load(path, allow_pickle=False)
        helper.pose_array(seed)
        require(array_info(empty) == export['outer_iteration_contract']['empty_weight']
            and seed.tobytes() == np.broadcast_to(empty, (1, 2, 9)).tobytes(), 'Original learned first conditioning differs')
        candidate = json.loads(pin(a.candidate_dir / 'result.json', a.candidate_sha256).read_text())
        require(candidate['schema'] == 'v69_original_CameraHead_step_FC1_split_compile'
            and candidate['status'] == 'compiled_not_quality_accepted' and candidate['chunks'] == 4
            and candidate['trunk_blocks_per_step'] == candidate['required_outer_iterations'] == 4
            and candidate['baseline_onnx_sha256'] == BASELINE_ONNX_SHA
            and candidate['configuration'] == CONFIG and candidate['build_options'] == BUILD
            and candidate['released'] is True and candidate['release_code'] in (None, 0)
            and not candidate['cleanup_errors'] and candidate['source_and_inputs_unchanged']
            and candidate['environment_restored'], 'Candidate original nonquantized compile contract differs')
        require([x['name'] for x in candidate['operations']] == ['config', 'load_onnx', 'build', 'export_rknn']
            and all(x['code'] in (None, 0) and x['status'] == 'completed' for x in candidate['operations']), 'Candidate SDK compilation incomplete')
        graphs = [dict(route='A', onnx=pin(a.export_dir / 'camera-step.onnx', BASELINE_ONNX_SHA, export['onnx']['bytes']),
            rknn=pin(a.baseline_model, BASELINE_MODEL_SHA, 434452453))]
        graphs.append(dict(route='B', **{key: pin(child(a.candidate_dir, candidate[key]['file']), candidate[key]['sha256'], candidate[key]['bytes']) for key in ('onnx', 'rknn')}))
        r['graphs'] = [dict(route=g['route'], onnx_sha256=pins[str(g['onnx'])], rknn_sha256=pins[str(g['rknn'])]) for g in graphs]
        if a.check_onnx:
            import onnxruntime as ort
            r['ORT_version'], r['onnx_checks'] = ort.__version__, []
            for graph in graphs:
                opt = ort.SessionOptions(); opt.intra_op_num_threads = opt.inter_op_num_threads = 1
                opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                session = ort.InferenceSession(str(graph['onnx']), sess_options=opt, providers=['CPUExecutionProvider'])
                session.disable_fallback()
                require(session.get_providers() == ['CPUExecutionProvider'], 'ORT provider differs')
                require({v.name: (v.shape, v.type) for v in session.get_inputs()} == {
                    'camera_tokens': ([1, 2, 2048], 'tensor(float)'), 'conditioning_pose': ([1, 2, 9], 'tensor(float)')}
                    and [(v.name, v.shape, v.type) for v in session.get_outputs()] == [('pose_delta', [1, 2, 9], 'tensor(float)')], 'Original ONNX IO differs')
                for i in range(4):
                    value, = session.run(['pose_delta'], {'camera_tokens': raw, 'conditioning_pose': arrays[f'conditioning_{i}']})
                    comparison = compare(value, arrays[f'delta_{i}'])
                    file = f'ort-{graph["route"]}-{i}.npy'
                    with (out / file).open('xb') as stream:
                        np.save(stream, value, allow_pickle=False)
                    r['onnx_checks'].append(dict(route=graph['route'], iteration=i, comparison=comparison,
                        artifact=dict(file=file, bytes=(out / file).stat().st_size, sha256=sha(out / file),
                                      tensor=array_info(value))))
                del session; gc.collect()
            require(all(x['comparison']['strict_pass'] for x in r['onnx_checks']), 'Original FP32 ONNX fixed-conditioning gate failed; SDK not loaded')
            r['status'] = 'passed_original_FP32_fixed_conditioning_ONNX_gate'
        else:
            check = json.loads(pin(a.check_receipt, a.check_sha256).read_text())
            require(check['schema'] == r['schema'] and check['mode'] == 'check_onnx'
                and check['status'] == 'passed_original_FP32_fixed_conditioning_ONNX_gate'
                and check['candidate_receipt_sha256'] == a.candidate_sha256 and check['graphs'] == r['graphs']
                and check['capture_sha256'] == CAPTURE_SHA and check['export_sha256'] == EXPORT_SHA
                and len(check['onnx_checks']) == 8 and all(x['comparison']['strict_pass'] for x in check['onnx_checks'])
                and check['source_and_inputs_unchanged'] and not check['cleanup_errors']
                and check['contexts_created'] == 0 and not check['npu_run_api_called'], 'Passed same-graph ONNX receipt required')
            r['passed_onnx_receipt_sha256'] = a.check_sha256
            library = base.bind_api(ct.CDLL(str(pin('/usr/lib/librknnrt.so', RUNTIME_SHA))))
            class Counted:
                def __getattr__(self, name):
                    function = getattr(library, name)
                    def call(*args):
                        released = sum(bool(args[2][i].buf) for i in range(args[1])) if name == 'rknn_outputs_release' else 0
                        if name == 'rknn_run': r['sdk_run_attempts'] += 1
                        code = function(*args)
                        if name == 'rknn_run': r['sdk_runs_succeeded'] += int(code == 0)
                        if name == 'rknn_outputs_get': r['outputs_acquired'] += sum(bool(args[2][i].buf) for i in range(args[1]))
                        if name == 'rknn_outputs_release': r['outputs_released'] += released * int(code == 0)
                        return code
                    return call
            api = Counted()
            for graph in graphs:
                context = base.CONTEXT(); contexts.append(context)
                filename = ct.create_string_buffer(os.fsencode(graph['rknn']))
                code = api.rknn_init(ct.byref(context), ct.cast(filename, ct.c_void_p), 0, r['init_flags'], None)
                r['contexts_created'] += int(bool(context.value)); base.check_call(code, 'rknn_init')
                require(bool(context.value), 'Empty context')
                base.check_call(api.rknn_set_core_mask(context, 7), 'rknn_set_core_mask')
                version = base.query(api, context, 5, base.SDKVersion())
                require(bytes(version.api_version).decode().startswith('2.3.2'), 'SDK version differs')
                count = base.query(api, context, 0, base.InputOutputNum())
                require((count.n_input, count.n_output) == (2, 1), 'Original two-input, one-delta ABI required')
                attrs = [base.query(api, context, 1, base.TensorAttr(index=i)) for i in range(2)]
                output_attr = base.query(api, context, 2, base.TensorAttr(index=0))
                graph['bindings'] = step.validate_bindings(base, attrs, output_attr)
                memory = base.query(api, context, 6, base.MemSize())
                r['graphs'][len(contexts)-1].update(inputs=[base.attr_record(x) for x in attrs], output=base.attr_record(output_attr),
                    SDK_version=bytes(version.api_version).decode(), driver_version=bytes(version.drv_version).decode(),
                    memory=dict(weight_bytes=int(memory.total_weight_size), internal_bytes=int(memory.total_internal_size),
                                dma_bytes=int(memory.total_dma_allocated_size)))
            seen = {}
            schedule = 'AB' if a.profile else 'ABBAABBA'
            for index, route in enumerate(schedule):
                graph, context = graphs[route == 'B'], contexts[route == 'B']
                captured, steps, prediction = {}, [], None
                begin = time.perf_counter()
                for i in range(4):
                    conditioning = seed if prediction is None else prediction
                    delta, timing = step.execute_step(base, api, context, raw, conditioning,
                        graph['bindings'], 30000, r['cleanup_errors'], r)
                    prediction = helper.accumulate_delta(prediction, delta)
                    activation = helper.activate_numpy_pose(prediction)
                    captured.update({f'conditioning_{i}': conditioning, f'delta_{i}': delta,
                                     f'preactivation_{i}': prediction, f'activation_{i}': activation})
                    steps.append(dict(iteration=i, **timing))
                    if a.profile:
                        detail = base.perf_detail(api, context); payload = detail.pop('raw_text').encode('utf-8')
                        require(len(payload) == detail['bytes'] and hashlib.sha256(payload).hexdigest() == detail['raw_sha256'], 'Profile text encoding changed raw SDK bytes')
                        file = f'profile-{route}-{i}.txt'; (out / file).write_bytes(payload)
                        steps[-1]['profile'] = dict(file=file, **detail)
                seconds = time.perf_counter() - begin
                artifact_path = out / f'head-{index:02d}-{route}.npz'
                with artifact_path.open('xb') as stream: np.savez(stream, **captured)
                record = dict(index=index, route=route, first_for_route=route not in seen,
                    complete_head_seconds=seconds, sdk_io_seconds=sum(x['io_total_seconds'] for x in steps), steps=steps,
                    output={key: compare(value, arrays[key]) for key, value in captured.items()},
                    artifact=dict(file=artifact_path.name, bytes=artifact_path.stat().st_size, sha256=sha(artifact_path),
                                  tensors={key: array_info(value) for key, value in captured.items()}))
                if route in seen: record['bitwise_equal_previous_same_route'] = all(captured[k].tobytes() == seen[route][k].tobytes() for k in captured)
                seen[route] = captured; r['calls'].append(record)
            if not a.profile:
                r['summary_seconds'] = {route: dict(calls=4, warm_count=3,
                    first=next(c['complete_head_seconds'] for c in r['calls'] if c['route'] == route),
                    warm_mean=float(np.mean([c['complete_head_seconds'] for c in r['calls'] if c['route'] == route and not c['first_for_route']])),
                    warm_sdk_io_mean=float(np.mean([c['sdk_io_seconds'] for c in r['calls'] if c['route'] == route and not c['first_for_route']]))) for route in 'AB'}
            r['status'] = 'completed_full_CameraHead_diagnostic_not_quality_accepted'
    except BaseException as exc:
        r['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        for context in reversed(contexts):
            if context.value:
                try:
                    base.check_call(api.rknn_destroy(context), 'rknn_destroy')
                    context.value = 0; r['contexts_destroyed'] += 1
                except BaseException as exc: r['cleanup_errors'].append('Context cleanup: ' + str(exc))
        r['live_contexts'] = sum(bool(context.value) for context in contexts)
        if r['status'].startswith('completed'):
            calls = 8 if a.profile else 32
            r['lifecycle_verified'] = (r['contexts_created'] == r['contexts_destroyed'] == 2
                and r['outputs_acquired'] == r['outputs_released'] == calls
                and r['sdk_run_attempts'] == r['sdk_runs_succeeded'] == r['npu_run_api_call_count'] == calls)
            if not r['lifecycle_verified']: r['cleanup_errors'].append('Context/output/run count invariant failed')
        try:
            r['source_and_inputs_unchanged'] = bool(pins) and all(sha(path) == h for path, h in pins.items())
        except BaseException as exc:
            r['source_and_inputs_unchanged'] = False; r['cleanup_errors'].append('Identity cleanup: ' + str(exc))
        r['affinity_restored'] = False
        try:
            os.sched_setaffinity(0, affinity); r['affinity_restored'] = os.sched_getaffinity(0) == affinity
        except BaseException as exc: r['cleanup_errors'].append('Affinity cleanup: ' + str(exc))
        if r['cleanup_errors'] or r['live_contexts'] or not r['source_and_inputs_unchanged'] or not r['affinity_restored']:
            r['status'] = 'failed_cleanup_or_identity'
        temp = out / 'result.json.tmp'; temp.write_text(json.dumps(r, indent=2, allow_nan=False) + '\n')
        os.replace(temp, out / 'result.json')
        print(json.dumps(dict(status=r['status'], error=r.get('error'), receipt_sha256=sha(out / 'result.json'))), flush=True)
    return 0 if r['status'].startswith(('completed', 'passed')) else 1


if __name__ == '__main__':
    sys.exit(main())
