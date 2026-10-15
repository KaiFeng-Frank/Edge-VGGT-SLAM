#!/usr/bin/env python3
"""Calibration pair0 only: flags0/flags8 placement diagnostic, not a benchmark.

Each route owns two fresh contexts. Both outputs are saved before profile
queries; attribution requires flags8 byte parity with that route's flags0.
No heldout array or same-input CPU reference is loaded by this worker.
An exact finite-output guard failure is retained as invalid numeric evidence;
all other errors remain fatal. No nonfinite value is rewritten or accepted.
"""
import argparse
import copy
import ctypes as ct
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import sys
import time
import traceback

NORMAL_SOURCE_SHA = 'd3157c5d1cecc4de541e9ff9a0d2adbf02e4a0269018892f24cddecbe47fce30'
STEP2_SOURCE_SHA = 'f1c94014ad6f7555c32ac0f83d489ade46295cef96362ed7f7c9878f01cf9780'
PROFILE_OWNER_SHA = '5aa75a7630ecb9793773553da1f7351a97ca5bac728d4f01f9b74d83d3ca2426'
JOINT_COMPILER_SHA = '4edc92073654e88fa73862b0f50e23bc987c1ec54f70b1c756a0a4d5519e41c3'
JOINT_EXPORT_SHA = 'f8f935e43db97ffb5cee683d8eee2a8fe4ceb728a96f10c37ad8b6772a76ea49'
JOINT_GRAPH_SHA = 'fd684d7b0e3c3b43f61a14125e300f3e06bd806f7d5e8577c48a83e4498251d9'
SUCCESS = 'completed_calibration_profile_diagnostic_not_performance_or_quality_accepted'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for variant in ('A', 'F', 'B'):
        p.add_argument('--compile-' + variant, type=Path, required=True)
        p.add_argument('--compile-' + variant + '-sha256', required=True)
    p.add_argument('--normal-source', type=Path, required=True)
    p.add_argument('--step2-source', type=Path, required=True)
    p.add_argument('--profile-owned-runtime', type=Path, default=Path(__file__).with_name('profile_owned_context_r2.py'))
    p.add_argument('--baseline-compile', type=Path, default=Path('/dev/shm/edge-vggt-v127-compile-A-r1/result.json'))
    p.add_argument('--capture-receipt', type=Path, default=Path('/dev/shm/edge-vggt-v128-capture-batch0-r1/result.json'))
    p.add_argument('--export-receipt', type=Path, default=Path('/tmp/edge-vggt-v126-basis-export-r2/result.json'))
    p.add_argument('--joint-compile', type=Path)
    p.add_argument('--joint-compile-sha256')
    p.add_argument('--joint-compiler-source', type=Path, default=Path(__file__).with_name('compile_joint_mlp_fp16_board.py'))
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args(); sys.dont_write_bytecode = True
    if platform.machine() != 'aarch64' or b'rockchip,rk3588' not in Path('/proc/device-tree/compatible').read_bytes():
        raise RuntimeError('RK3588 board only')
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    env, affinity, handlers, watched, owners = dict(os.environ), set(os.sched_getaffinity(0)), {}, {}, []
    began = time.monotonic()
    report = dict(schema='v128_original_native518_calibration_pair_flags8_placement_diagnostic', status='running',
        profiling_only=True, input_id='pair0', input_split='calibration', heldout_arrays_accessed=False,
        core_mask=1, flags_per_route=[0, 8], maximum_live_contexts=1, candidate_accepted=False,
        no_normal_performance_claim=True, protected_placement_requires_actual_profile_rows=True,
        system_or_frequency_writes=False, calls=[], cleanup_errors=[],
        numerical_failures_are_diagnostic_not_accepted=True, production_finite_guard_modified=False)

    def require(ok, message):
        if not ok: raise RuntimeError(message)

    def sha(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(4 << 20), b''): h.update(block)
        return h.hexdigest()

    def pin(path, digest, size=None):
        path = Path(path).resolve(strict=True)
        require(sha(path) == digest and (size is None or path.stat().st_size == size), 'Pinned file differs: ' + str(path))
        watched[path] = digest; return path

    def relative(folder, row):
        name = Path(row['file']); require(not name.is_absolute() and '..' not in name.parts, 'Receipt-relative artifact required')
        return pin(folder / name, row['sha256'], row['bytes'])

    def module(path, digest, name):
        spec = importlib.util.spec_from_file_location(name, pin(path, digest))
        value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value

    def progress(stage):
        report.update(stage=stage, seconds=time.monotonic() - began)
        temp = out / 'result.tmp'; temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n'); temp.replace(out / 'result.json')
        print(stage, flush=True)

    def record(path):
        return dict(file=str(path.relative_to(out)), bytes=path.stat().st_size, sha256=sha(path))

    def clean(row):
        return not row['cleanup_errors'] and all(row[k] for k in ('source_and_inputs_unchanged', 'affinity_restored', 'environment_restored'))

    def interrupted(number, frame): raise InterruptedError('External controller signal ' + str(number))

    report['script_sha256'] = sha(__file__); watched[Path(__file__).resolve()] = report['script_sha256']; progress('preflight')
    try:
        for number in (signal.SIGTERM, signal.SIGINT): handlers[number] = signal.signal(number, interrupted)
        require({4, 6} <= affinity, 'A76 affinity required'); os.sched_setaffinity(0, {4, 6})
        os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        import numpy as np
        M = module(a.normal_source, NORMAL_SOURCE_SHA, '_v128_calibration_logical_IO')
        pin(a.step2_source, STEP2_SOURCE_SHA)
        ep = pin(a.export_receipt, M.EXPORT_SHA); exported = json.loads(ep.read_text())
        require(exported['schema'] == 'v126_original_native518_basis_pair_export' and
            exported['status'] == 'exported_original_basis_pair_not_numerically_accepted' and clean(exported) and
            set(exported['original_parameters']) == {'frame', 'global'} and
            all(len(exported['original_parameters'][k]) == 18 for k in ('frame', 'global')), 'Original 36-parameter pair export required')
        cp = pin(a.capture_receipt, M.CAPTURE_SHAS[0]); capture = json.loads(cp.read_text())
        require(capture['schema'] == 'v128_original_native518_frame0_input_capture' and
            capture['status'] == 'completed_original_native518_frame0_input_subbatch_capture' and capture['script_sha256'] == M.CAPTURE_SOURCE_SHA and
            capture['batch_index'] == 0 and [r['id'] for r in capture['records']] == ['pair0', 'pair1'] and
            all(r['split'] == 'calibration' for r in capture['records']) and clean(capture) and capture['all16_SDK_calls_completed'] and
            capture['all_inputs_before_frame0'] and capture['all_NPU_contexts_closed'] and capture['model_state_unchanged'] and
            capture['checkpoint_final_SHA_verified'] and capture['checkpoint'] == exported['checkpoint'] and
            capture['source_hashes'] == exported['source_hashes'] and capture['SDK_totals'] == dict(contexts_created=4, contexts_destroyed=4,
                runs_completed=16, output_buffers_acquired=16, output_buffers_released=16), 'Clean original calibration subbatch required')
        row = capture['records'][0]
        require(row['source_pair_indices'] == [0, 8] and row['input_is_before_original_frame0'] and
            row['actual_four_DINO_groups_executed'] and row['original_final_norm_and_role_concat_executed'], 'True calibration pair0 boundary required')
        artifact = row['input']; x = np.load(relative(cp.parent, artifact), allow_pickle=False)
        require(x.dtype == np.float32 and list(x.shape) == artifact['shape'] == M.FRAME and artifact['dtype'] == 'float32' and
            x.flags.c_contiguous and artifact['finite'] and np.isfinite(x).all() and x.nbytes == artifact['tensor_bytes'] and
            hashlib.sha256(x.tobytes()).hexdigest() == artifact['raw_sha256'], 'Captured pair0 bytes differ')
        x.setflags(write=False)
        bp = pin(a.baseline_compile, M.FP16_COMPILE_SHA); baseline = json.loads(bp.read_text())
        graph_a = next(r for r in exported['models'] if r['key'] == 'A')
        require(baseline['schema'] == 'v127_original_native518_basis_pair_FP16_compile' and baseline['status'] == 'compiled_not_runtime_accepted' and
            baseline['variant'] == 'A' and baseline['script_sha256'] == M.FP16_COMPILER_SHA and baseline['configuration'] == M.FLOAT_CONFIG and
            baseline['build_options'] == M.FLOAT_BUILD and baseline['export_receipt']['sha256'] == M.EXPORT_SHA and
            baseline['check_receipt']['sha256'] == M.ORIGINAL_CHECK_SHA and baseline['onnx']['sha256'] == graph_a['onnx']['sha256'] and
            baseline['rknn']['sha256'] == M.FP16_MODEL_SHA and baseline['all_operations_succeeded'] and baseline['sdk_released'] and
            baseline['sdk_release_code'] in (0, None) and clean(baseline) and baseline['cwd_restored'], 'Frozen original FP16 A required')
        input_rows = [dict(index=0, name='input', shape=M.FRAME, dtype='float32')]
        output_rows = [dict(index=i, name=n, shape=s, dtype='float32') for i, (n, s) in enumerate(zip(M.NAMES, (M.FRAME, M.GLOBAL)))]
        require(baseline['input_bindings'] == input_rows and baseline['output_bindings'] == output_rows, 'Original complete IO differs')
        models = {'originalFP16_A': dict(baseline['rknn'], file=str(relative(bp.parent, baseline['rknn'])))}
        compiles = {'originalFP16_A': dict(file=str(bp), sha256=M.FP16_COMPILE_SHA)}
        for variant in ('A', 'F', 'B'):
            path = pin(getattr(a, 'compile_' + variant), getattr(a, 'compile_' + variant + '_sha256')); r = json.loads(path.read_text())
            graph = next(g for g in exported['models'] if g['key'] == variant)
            require(r['schema'] == 'v128_original_native518_basis_pair_manual_hybrid_step2' and r['status'] == 'compiled_manual_hybrid_pair_not_runtime_or_quality_accepted' and
                r['variant'] == variant and r['script_sha256'] == STEP2_SOURCE_SHA and r['selector_source_sha256'] == '25db1ca6c66fcea6d84b921c3f8f8382d0d16eeaa9d8f6f75d7ae5b9b804da72' and
                r['configuration'] == M.QUANT_CONFIG and r['graph']['sha256'] == graph['onnx']['sha256'] and r['graph']['bytes'] == graph['onnx']['bytes'] and
                r['input_bindings'] == input_rows and r['output_bindings'] == output_rows and r['original_parameters'] == exported['original_parameters'] and
                r['checkpoint'] == exported['checkpoint'] and r['actual_generated_selector_mapping_recomputed'] and
                r['original_quantization_section_byte_preserved'] and not r['update_branch_inputs_selected'] and r['requested_policy_only'] and
                r['all_SDK_operations_succeeded'] and r['sdk_released'] and r['sdk_release_code'] in (0, None) and clean(r) and r['cwd_restored'], 'Actual clean manual-hybrid compile differs')
            require([v['capture']['sha256'] for v in r['three_clean_capture_check_receipts']] == M.CAPTURE_SHAS and
                [v['check']['sha256'] for v in r['three_clean_capture_check_receipts']] == M.CHECK_SHAS and
                r['step1_receipt']['script_sha256'] == M.STEP1_SOURCE_SHA and not r['runtime_inference_invoked'] and not r['additional_calibration_invoked'], 'Original capture/check/step1 lineage differs')
            key = 'mixed' + variant; models[key] = dict(r['rknn'], file=str(relative(path.parent, r['rknn'])))
            compiles[key] = dict(file=str(path), sha256=getattr(a, 'compile_' + variant + '_sha256'), selector_policy=r['selector_policy'],
                residual_role_mapping=r['residual_role_mapping'], basis_role_mapping=r['basis_role_mapping'])
        require(bool(a.joint_compile) == bool(a.joint_compile_sha256), 'Joint receipt and checksum must be supplied together')
        if a.joint_compile:
            pin(a.joint_compiler_source, JOINT_COMPILER_SHA); jp = pin(a.joint_compile, a.joint_compile_sha256); r = json.loads(jp.read_text())
            require(r['schema'] == 'v129_original_native518_joint_MLP_pair_FP16_compile' and
                r['status'] == 'compiled_original_joint_MLP_pair_not_runtime_or_quality_accepted' and r['script_sha256'] == JOINT_COMPILER_SHA and
                r['configuration'] == M.FLOAT_CONFIG and r['build_options'] == M.FLOAT_BUILD and r['export_receipt']['sha256'] == JOINT_EXPORT_SHA and
                r['onnx']['sha256'] == JOINT_GRAPH_SHA and r['original_parameters'] == exported['original_parameters'] and r['checkpoint'] == exported['checkpoint'] and
                r['input_bindings'] == input_rows and r['output_bindings'] == output_rows and r['six_new_original_FP32_gates_passed'] and
                r['all_operations_succeeded'] and r['sdk_released'] and r['sdk_release_code'] in (0, None) and clean(r) and r['cwd_restored'] and
                r['proof']['whole_parent_inverse_ModelProto_byte_equal'], 'Actual complete joint-MLP FP16 compile required')
            models['joint_MLP'] = dict(r['rknn'], file=str(relative(jp.parent, r['rknn'])))
            compiles['joint_MLP'] = dict(file=str(jp), sha256=a.joint_compile_sha256)
        helper_path = M.E / 'vggt-v99-source-r1/run_native_block_r1.py'
        heads = module(a.profile_owned_runtime, PROFILE_OWNER_SHA, '_v128_calibration_owned'); H = module(helper_path, M.HELPER_SHA, '_v128_calibration_platform')
        IO = M.logical_context(heads.OwnedContext); abi = pin(M.B / 'src/deploy/rk3588/validation/run-v12-rknn-block-board.py', M.ABI_SHA); library = pin('/usr/lib/librknnrt.so', M.SDK_SHA)
        require(shutil.disk_usage(out).free >= len(models) * 4 * 8528000 + (128 << 20), 'Output storage reserve insufficient')
        counts = dict(contexts_created=2 * len(models), contexts_destroyed=2 * len(models), runs_completed=2 * len(models),
            output_buffers_acquired=4 * len(models), output_buffers_released=4 * len(models))
        report.update(models=models, compile_receipts=compiles, routes=list(models), expected_resource_counts=counts,
            owned_IO_source_sha256=PROFILE_OWNER_SHA, original_owned_IO_source_sha256=M.HEADS_SHA,
            capture_receipt=dict(file=str(cp), sha256=M.CAPTURE_SHAS[0]), input=artifact, source_pair_indices=[0, 8],
            export_receipt=dict(file=str(ep), sha256=M.EXPORT_SHA), original_parameters=exported['original_parameters'], checkpoint=exported['checkpoint'])
        for route, model in models.items():
            normal = None
            for flags in (0, 8):
                sample = H.sample(); require(sample['npu_hz'] == 1000000000 and sample['temperatures_c'] and max(sample['temperatures_c'].values()) < 75, 'External 1GHz/thermal guard differs')
                call = dict(index=len(report['calls']), route=route, init_flags=flags, status='running', platform_before=sample); report['calls'].append(call)
                progress(route + '_flags' + str(flags) + '_init'); owner = None
                try:
                    try:
                        owner = IO(model['file'], model['sha256'], input_shapes=[M.FRAME], output_shapes=[M.FRAME, M.GLOBAL],
                            input_names=['input'], output_names=M.NAMES, core_mask=1, flags=flags, abi_source=abi, library=library, runtime_helper=helper_path)
                    except BaseException as error:
                        owner = getattr(error, 'owned_context', None)
                        if owner is not None: owners.append(owner)
                        raise
                    owners.append(owner); descriptors = owner.metrics()
                    call['input_descriptors'] = copy.deepcopy(descriptors['input_descriptors']); call['output_descriptors'] = copy.deepcopy(descriptors['output_descriptors'])
                    call['invalid_numeric'] = False
                    try:
                        ys = owner.run([x])
                    except RuntimeError as error:
                        failed = getattr(owner, 'last_failed_outputs', None)
                        complete = isinstance(failed, list) and len(failed) == 2 and all(isinstance(y, np.ndarray) and
                            y.dtype == np.float32 and list(y.shape) == shape and y.flags.c_contiguous
                            for y, shape in zip(failed, (M.FRAME, M.GLOBAL)))
                        if type(error) is not RuntimeError or str(error) != 'Nonfinite owned SDK output' or not complete or not any(not np.isfinite(y).all() for y in failed):
                            raise
                        state = owner.metrics()
                        require(state['runs_completed'] == state['output_get_successes'] == 1 and state['live_output_buffers'] == 0 and
                            state['output_buffers_acquired'] == state['output_buffers_released'] == 2 and not owner.active,
                            'Invalid-output diagnostic requires complete owned copies and successful original release')
                        ys = failed; call.update(invalid_numeric=True, numerical_guard_error='Nonfinite owned SDK output',
                            raw_failed_outputs_preserved=True, original_output_guard_raised=True)
                    call['outputs'] = []
                    for name, y in zip(M.NAMES, ys):
                        path = out / (route + '-flags' + str(flags) + '-' + name + '.npy')
                        with path.open('xb') as stream: np.save(stream, y, allow_pickle=False)
                        call['outputs'].append(dict(record(path), name=name, shape=list(y.shape), dtype=str(y.dtype), tensor_bytes=y.nbytes,
                            raw_sha256=hashlib.sha256(y.tobytes()).hexdigest(), finite=bool(np.isfinite(y).all()),
                            finite_count=int(np.count_nonzero(np.isfinite(y))), nan_count=int(np.count_nonzero(np.isnan(y))),
                            positive_inf_count=int(np.count_nonzero(np.isposinf(y))), negative_inf_count=int(np.count_nonzero(np.isneginf(y)))))
                    call['all_outputs_finite'] = all(row['finite'] for row in call['outputs'])
                    require(call['invalid_numeric'] == (not call['all_outputs_finite']), 'Finite guard and preserved raw payload disagree')
                    if flags == 0: normal = ys
                    else:
                        call['all_outputs_byte_equal_to_flags0'] = all(y.dtype == ref.dtype and y.shape == ref.shape and y.tobytes() == ref.tobytes() for y, ref in zip(ys, normal))
                    progress(route + '_flags' + str(flags) + '_outputs_saved_before_queries')
                    if flags == 8:
                        detail = owner.base.query(owner.api, owner.context, 3, owner.base.PerfDetail())
                        require(detail.perf_data and 0 < detail.data_len <= 16 << 20, 'Invalid profile pointer/length')
                        payload = ct.string_at(detail.perf_data, int(detail.data_len)).rstrip(b'\x00')
                        path = out / (route + '-perf-detail.txt'); path.write_bytes(payload)
                        perf = owner.base.query(owner.api, owner.context, 4, owner.base.PerfRun())
                        call.update(profile=record(path), SDK_reported_run_duration_us=int(perf.run_duration), pointer_copied_before_next_query=True)
                    memory = owner.base.query(owner.api, owner.context, 6, owner.base.MemSize())
                    call['SDK_model_memory_bytes'] = {name: int(getattr(memory, name)) for name, _ in memory._fields_ if name != 'reserved'}
                finally:
                    if owner is not None:
                        owner.close(); call['owner_metrics'] = owner.metrics()
                call['status'] = 'completed_invalid_numeric_diagnostic' if call['invalid_numeric'] else 'completed'
                progress(route + '_flags' + str(flags) + '_closed')
        require(hashlib.sha256(x.tobytes()).hexdigest() == artifact['raw_sha256'], 'Captured input changed')
        report.update(status=SUCCESS, all_calls_completed=True, saved_array_count=4 * len(models),
            attribution_byte_parity_passed=all(r['all_outputs_byte_equal_to_flags0'] for r in report['calls'] if r['init_flags'] == 8),
            all_outputs_finite=all(r['all_outputs_finite'] for r in report['calls']),
            invalid_numeric_routes=list(dict.fromkeys(r['route'] for r in report['calls'] if r['invalid_numeric'])),
            numerical_acceptance=False, nonfinite_values_rewritten=False)
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc()); traceback.print_exc()
    finally:
        for owner in owners:
            if not owner.closed:
                try: owner.close()
                except BaseException as error: report['cleanup_errors'].append('Owner close: ' + str(error))
        metrics = [owner.metrics() for owner in owners]
        report['resource_totals'] = {k: sum(r[k] for r in metrics) for k in ('contexts_created', 'contexts_destroyed', 'runs_completed', 'output_buffers_acquired', 'output_buffers_released')}
        report['all_resources_closed'] = all(r['closed'] and r['resource_counts_balanced'] and not r['cleanup_errors'] for r in metrics)
        if report['status'] == SUCCESS and report['resource_totals'] != report['expected_resource_counts']: report['cleanup_errors'].append('Resource counts differ')
        try: os.sched_setaffinity(0, affinity)
        except BaseException as error: report['cleanup_errors'].append('Affinity restore: ' + str(error))
        os.environ.clear(); os.environ.update(env)
        for number, handler in handlers.items(): signal.signal(number, handler)
        report.update(affinity_restored=set(os.sched_getaffinity(0)) == affinity, environment_restored=dict(os.environ) == env,
            pinned_files={str(k): v for k, v in watched.items()})
        try: report['source_and_inputs_unchanged'] = all(p.exists() and sha(p) == s for p, s in watched.items())
        except BaseException as error: report['source_and_inputs_unchanged'] = False; report['cleanup_errors'].append(str(error))
        if report['cleanup_errors'] or not all(report[k] for k in ('all_resources_closed', 'affinity_restored', 'environment_restored', 'source_and_inputs_unchanged')): report['status'] = 'failed_cleanup_or_identity'
        progress('finished')
    return 0 if report['status'] == SUCCESS else 1


if __name__ == '__main__': raise SystemExit(main())
