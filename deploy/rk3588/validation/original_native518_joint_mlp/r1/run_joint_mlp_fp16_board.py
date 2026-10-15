#!/usr/bin/env python3
"""Normal core1/flags0 original FP16 A versus complete joint-MLP pair.

Fresh A/J/J/A contexts each consume both real heldouts. All eight calls count;
their complete owned IO times and context lifecycle are reported separately.
"""
import argparse
import copy
import faulthandler
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import sys
import time
import traceback

B = Path('/mnt/vggt-data/vggt-slam2'); E = B / 'experiments'
COMPILE_SOURCE_SHA = '4edc92073654e88fa73862b0f50e23bc987c1ec54f70b1c756a0a4d5519e41c3'
HEADS_SHA = '0bfcb295ad5490e8196e8fcc1244e11a99b07bfb3ba7d5c350cb8c81d3ff3be4'
HELPER_SHA = '56ce45635a5d010e862bf59db2f9fcc6e8ec642b2b023c8c40cad0bd5808050f'
ABI_SHA = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
LIBRARY_SHA = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
BASELINE_SHA = '71b8fc05994e15cf3a658d01f57b89f909172617ff513dea97971dc13d18bb5c'
BASELINE_SOURCE_SHA = '2bbf180a9a70444ba592b7898fa23b96b11f36da3859f07cc5b435f0a25acb11'
BASELINE_MODEL_SHA = '803f7e53c5cf574ccb86e1bdd5b9091332f21ec48aef8864fdee52bc72017666'
CAPTURE_SHA = '5560b34755bfbca54cc179c119991792b63d6135e839ad15ac8db95e16965660'
CAPTURE_SOURCE_SHA = 'ed500643ef333dd824ccab81a126e87ff8616c3cb3ee09e3ec59f2a4b7bbf7e9'
REFERENCE_SHA = '17f9aec6d7c93189c7ba74c33c3101160c4170f0a38e2b6ae91d77eeb4fdc250'
REFERENCE_SOURCE_SHA = 'b0f4bd489d1e6ec8eb28869f609d7e48469aa9d19b2120d7906cd9284059bc4c'
JOINT_CHECK_SHA = '666908dfbf47c6f9641f1d3d74588bb1abe5298e09dc83dc2205f209bd516490'
EXPORT_SHA = 'f8f935e43db97ffb5cee683d8eee2a8fe4ceb728a96f10c37ad8b6772a76ea49'
FRAME, GLOBAL = [2, 1041, 1024], [1, 2082, 1024]
SCHEDULE = ['originalFP16_A', 'joint_MLP', 'joint_MLP', 'originalFP16_A']
SUCCESS = 'completed_original_joint_MLP_pair_FP16_component_not_accepted'


def require(ok, message):
    if not ok: raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''): h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--joint-compile', type=Path, required=True); p.add_argument('--joint-compile-sha256', required=True)
    p.add_argument('--compiler-source', type=Path, default=Path(__file__).with_name('compile_joint_mlp_fp16_board.py'))
    p.add_argument('--baseline-compile', type=Path, default=Path('/dev/shm/edge-vggt-v127-compile-A-r1/result.json'))
    p.add_argument('--capture-receipt', type=Path, default=Path('/dev/shm/edge-vggt-v128-capture-batch2-r1/result.json'))
    p.add_argument('--reference-receipt', type=Path, default=Path('/dev/shm/edge-vggt-v128-fp32-batch2-r1/result.json'))
    p.add_argument('--joint-check', type=Path, default=Path('/dev/shm/edge-vggt-v129-joint-mlp-check2-r1/result.json'))
    p.add_argument('--owned-runtime', type=Path, default=E / 'vggt-v100-source-r1/native_heads_runtime.py')
    p.add_argument('--runtime-helper', type=Path, default=E / 'vggt-v99-source-r1/run_native_block_r1.py')
    p.add_argument('--abi-source', type=Path, default=B / 'src/deploy/rk3588/validation/run-v12-rknn-block-board.py')
    p.add_argument('--runtime-library', type=Path, default=Path('/usr/lib/librknnrt.so'))
    p.add_argument('--output-dir', type=Path, required=True); a = p.parse_args(); sys.dont_write_bytecode = True
    require(platform.machine() == 'aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(), 'RK3588S board-only runtime')
    faulthandler.enable(); out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    old_env, old_affinity = dict(os.environ), set(os.sched_getaffinity(0)); source = Path(__file__).resolve()
    watched, handlers, owners = {source: sha(source)}, {}, []; began = time.monotonic()
    report = dict(schema='v129_original_native518_joint_MLP_pair_FP16_runtime', status='running', script_sha256=watched[source],
        core_mask=1, init_flags=0, schedule=SCHEDULE, heldout_order=['pair4', 'pair5'], calls=[], phases=[], cleanup_errors=[],
        component_only=True, candidate_accepted=False, full_model_trajectory_realtime_or_8GB_accepted=False,
        maximum_live_contexts=1, system_or_frequency_writes=False, calibration_or_manual_quantization_invoked=False,
        precision_scope='Original complete frame0/global0 on the same captured pre-frame0 input. Upstream DINO is the original deployed FP16 route; this is not a full image-to-pose or trajectory comparison.',
        timing_scope='Outer OwnedContext.run including finite/input preparation, SDK set/run/get, two owned copies/release. Init/query/destroy separately in phase lifecycle. Comparison/hash/save and platform observations are outside these timers.')

    def progress(stage):
        report.update(stage=stage, elapsed_seconds=time.monotonic() - began, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        temp = out / 'result.tmp'; temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n'); temp.replace(out / 'result.json'); print(stage, flush=True)

    def pin(path, digest, size=None):
        path = Path(path).resolve(strict=True); require(sha(path) == digest and (size is None or path.stat().st_size == size), 'Pinned file differs: ' + str(path)); watched[path] = digest; return path

    def relative(folder, row):
        file = Path(row['file']); require(not file.is_absolute() and '..' not in file.parts, 'Receipt-relative artifact required'); return pin(folder / file, row['sha256'], row['bytes'])

    def module(path, digest, name):
        spec = importlib.util.spec_from_file_location(name, pin(path, digest)); value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value

    def array(folder, row, shape):
        value = np.load(relative(folder, row), allow_pickle=False)
        require(value.dtype == np.float32 and list(value.shape) == shape == row['shape'] and value.flags.c_contiguous and
            np.isfinite(value).all() and hashlib.sha256(value.tobytes()).hexdigest() == row['raw_sha256'], 'Complete saved FP32 artifact differs')
        value.setflags(write=False); return value

    def compare(x, y):
        delta = x.astype(np.float64) - y.astype(np.float64); norm = float(np.linalg.norm(y.astype(np.float64)))
        l2 = float(np.linalg.norm(delta)) / norm if norm else None; maximum = float(np.abs(delta).max())
        strict = bool(np.all(np.abs(delta) <= 1e-5 + 1e-4 * np.abs(y.astype(np.float64))))
        return dict(relative_l2=l2, max_abs=maximum, strict_pass=strict, rtol=1e-4, atol=1e-5,
            fixed_numeric_gate_passed=l2 is not None and l2 <= 1e-5 and maximum <= 1e-3,
            byte_equal=x.tobytes() == y.tobytes(), original_tolerances_unchanged=True)

    def observe(stage):
        sample = H.sample(); report.setdefault('platform_samples', []).append(dict(stage=stage, monotonic=time.monotonic(), **sample))
        require(sample['npu_hz'] == 1000000000 and sample['temperatures_c'] and max(sample['temperatures_c'].values()) < 75, 'External 1GHz/thermal guard differs')

    progress('preflight')
    try:
        def interrupted(number, frame): raise InterruptedError('Controller signal ' + str(number))
        for number in (signal.SIGTERM, signal.SIGINT): handlers[number] = signal.signal(number, interrupted)
        require({4, 6} <= old_affinity, 'A76 affinity required'); os.sched_setaffinity(0, {4, 6})
        os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
        C = module(a.compiler_source, COMPILE_SOURCE_SHA, 'v129_frozen_FP16_compile_metadata')
        jp, bp = pin(a.joint_compile, a.joint_compile_sha256), pin(a.baseline_compile, BASELINE_SHA)
        joint, baseline = json.loads(jp.read_text()), json.loads(bp.read_text())
        require(joint['schema'] == 'v129_original_native518_joint_MLP_pair_FP16_compile' and joint['status'] == C.SUCCESS and
            joint['script_sha256'] == COMPILE_SOURCE_SHA and joint['configuration'] == C.CONFIG and joint['build_options'] == C.BUILD and
            joint['export_receipt']['sha256'] == EXPORT_SHA and [r['sha256'] for r in joint['check_receipts']] == list(C.CHECK_SHAS) and
            joint['six_new_original_FP32_gates_passed'] and joint['all_operations_succeeded'] and joint['sdk_released'], 'Actual clean same-configuration joint compile required')
        require(baseline['schema'] == 'v127_original_native518_basis_pair_FP16_compile' and baseline['status'] == 'compiled_not_runtime_accepted' and
            baseline['variant'] == 'A' and baseline['script_sha256'] == BASELINE_SOURCE_SHA and baseline['configuration'] == C.CONFIG and baseline['build_options'] == C.BUILD and
            baseline['onnx']['sha256'] == C.PARENT_GRAPH_SHA and baseline['rknn']['sha256'] == BASELINE_MODEL_SHA and baseline['all_operations_succeeded'] and baseline['sdk_released'], 'Original FP16 A complete parent required')
        for receipt in (joint, baseline): require(not receipt['cleanup_errors'] and all(receipt[k] for k in ('source_and_inputs_unchanged', 'affinity_restored', 'environment_restored', 'cwd_restored')) and receipt['sdk_release_code'] in (0, None), 'Clean immutable compiler receipt required')
        expected_inputs = [dict(index=0, name='input', shape=FRAME, dtype='float32')]
        expected_outputs = [dict(index=0, name='frame_output', shape=FRAME, dtype='float32'), dict(index=1, name='global_output', shape=GLOBAL, dtype='float32')]
        require(all(r['input_bindings'] == expected_inputs and r['output_bindings'] == expected_outputs for r in (joint, baseline)), 'Fair complete canonical pair bindings required')
        cap_path, ref_path, jref_path = pin(a.capture_receipt, CAPTURE_SHA), pin(a.reference_receipt, REFERENCE_SHA), pin(a.joint_check, JOINT_CHECK_SHA)
        cap, ref, jref = [json.loads(path.read_text()) for path in (cap_path, ref_path, jref_path)]
        require(cap['status'] == 'completed_original_native518_frame0_input_subbatch_capture' and cap['script_sha256'] == CAPTURE_SOURCE_SHA and
            [r['id'] for r in cap['records']] == ['pair4', 'pair5'] and all(r['split'] == 'heldout' for r in cap['records']) and cap['all_NPU_contexts_closed'] and not cap['cleanup_errors'], 'Two actual heldout before-frame0 inputs required')
        require(ref['status'] == 'checked_captured_pair_fixed_FP32_gate_passed' and ref['script_sha256'] == REFERENCE_SOURCE_SHA and ref['FP32_gate_passed'] and
            ref['capture_receipt']['sha256'] == CAPTURE_SHA and ref['records'] == cap['records'] and ref['original_parameters'] == joint['original_parameters'], 'Same-input original A FP32 references required')
        require(jref['schema'] == 'v129_original_native518_joint_MLP_pair_FP32_check' and jref['script_sha256'] == C.SOURCE_SHA and jref['status'] == 'checked_original_joint_MLP_pair_fixed_FP32_gate_passed_not_NPU_accepted' and
            jref['original_FP32_reference_receipt']['sha256'] == REFERENCE_SHA and jref['export_receipt']['sha256'] == EXPORT_SHA and jref['FP32_gate_passed'] and jref['all2_calls_completed'] and
            jref['original_parameters'] == joint['original_parameters'] and jref['proof'] == joint['proof'] and [r['input_id'] for r in jref['checks']] == ['pair4', 'pair5'], 'Same-input own joint FP32 references required')
        for r in (cap, ref, jref): require(not r['cleanup_errors'] and all(r[k] for k in ('source_and_inputs_unchanged', 'affinity_restored', 'environment_restored')), 'Clean heldout provenance required')
        import numpy as np
        inputs = {r['id']: array(cap_path.parent, r['input'], FRAME) for r in cap['records']}
        original_refs = {r['input_id']: [array(ref_path.parent, v, s) for v, s in zip(r['outputs'], (FRAME, GLOBAL))] for r in ref['checks'] if r['variant'] == 'A'}
        joint_refs = {r['input_id']: [array(jref_path.parent, v, s) for v, s in zip(r['outputs'], (FRAME, GLOBAL))] for r in jref['checks']}
        require(shutil.disk_usage(out).free >= 16 * 8528000 + (128 << 20), 'Output storage reserve insufficient')
        models = {'originalFP16_A': relative(bp.parent, baseline['rknn']), 'joint_MLP': relative(jp.parent, joint['rknn'])}
        model_rows = {'originalFP16_A': baseline['rknn'], 'joint_MLP': joint['rknn']}
        H = module(a.runtime_helper, HELPER_SHA, 'v129_frozen_platform'); heads = module(a.owned_runtime, HEADS_SHA, 'v129_frozen_owned_IO')
        abi, library = pin(a.abi_source, ABI_SHA), pin(a.runtime_library, LIBRARY_SHA)
        report.update(compile_receipts=dict(originalFP16_A=dict(file=str(bp), sha256=BASELINE_SHA), joint_MLP=dict(file=str(jp), sha256=a.joint_compile_sha256)),
            capture_receipt=dict(file=str(cap_path), sha256=CAPTURE_SHA), original_FP32_reference_receipt=dict(file=str(ref_path), sha256=REFERENCE_SHA),
            joint_FP32_reference_receipt=dict(file=str(jref_path), sha256=JOINT_CHECK_SHA), configuration=C.CONFIG, build_options=C.BUILD,
            expected_resource_counts=dict(contexts_created=4, contexts_destroyed=4, runs_completed=8, output_buffers_acquired=16, output_buffers_released=16))
        first_hashes = {}
        for phase, route in enumerate(SCHEDULE):
            observe('before_phase' + str(phase)); row = dict(index=phase, route=route); report['phases'].append(row); owner = None; payloads = []; tick = time.perf_counter()
            try:
                try: owner = heads.OwnedContext(models[route], model_rows[route]['sha256'], input_shapes=[FRAME], output_shapes=[FRAME, GLOBAL], input_names=['input'], output_names=['frame_output', 'global_output'], core_mask=1, flags=0, abi_source=abi, library=library, runtime_helper=a.runtime_helper)
                except BaseException as error:
                    owner = getattr(error, 'owned_context', None)
                    if owner is not None: owners.append(owner)
                    raise
                owners.append(owner); row['init_seconds'] = time.perf_counter() - tick
                for case, x in inputs.items():
                    call_tick = time.perf_counter(); ys = owner.run([x]); elapsed = time.perf_counter() - call_tick
                    call = dict(index=len(report['calls']), phase=phase, route=route, input_id=case, complete_seconds=elapsed, SDK_stages=copy.deepcopy(owner.last_timings)); report['calls'].append(call); payloads.append((call, ys))
            finally:
                if owner is not None:
                    close_tick = time.perf_counter()
                    try: owner.close()
                    finally: row.update(destroy_seconds=time.perf_counter() - close_tick, lifecycle_seconds_excluding_save_compare=time.perf_counter() - tick, owner_metrics=owner.metrics())
            observe('after_phase' + str(phase))
            for call, ys in payloads:
                case = call['input_id']; own = original_refs if route == 'originalFP16_A' else joint_refs; call['outputs'] = []
                call['comparisons_vs_original_A_FP32'] = {}; call['comparisons_vs_own_graph_FP32'] = {}
                for name, y, original, own_y in zip(('frame_output', 'global_output'), ys, original_refs[case], own[case]):
                    file = out / ('call%02d-%s.npy' % (call['index'], name))
                    with file.open('xb') as stream: np.save(stream, y, allow_pickle=False)
                    digest = hashlib.sha256(y.tobytes()).hexdigest(); key = (route, case, name); first_hashes.setdefault(key, digest)
                    call['outputs'].append(dict(name=name, file=file.name, bytes=file.stat().st_size, sha256=sha(file), raw_sha256=digest, shape=list(y.shape), dtype=str(y.dtype), finite=bool(np.isfinite(y).all()), repeat_raw_sha256_equal=digest == first_hashes[key]))
                    call['comparisons_vs_original_A_FP32'][name] = compare(y, original); call['comparisons_vs_own_graph_FP32'][name] = compare(y, own_y)
                call['repeat_raw_sha256_equal'] = all(r['repeat_raw_sha256_equal'] for r in call['outputs'])
            require(all(hashlib.sha256(inputs[r['id']].tobytes()).hexdigest() == r['input']['raw_sha256'] for r in cap['records']), 'Immutable actual heldout input changed')
            progress('phase' + str(phase) + '_closed_and_saved')
        report.update(status=SUCCESS, all8_calls_completed=True, saved_array_count=16,
            timing_summary={route: dict(all4_call_seconds=[r['complete_seconds'] for r in report['calls'] if r['route'] == route],
                all4_call_mean_seconds=sum(r['complete_seconds'] for r in report['calls'] if r['route'] == route) / 4) for route in dict.fromkeys(SCHEDULE)},
            timing_limits='All first and second calls of both fresh visits included; no warmup, P95, sustained throughput or full-model inference claim.')
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc()); traceback.print_exc()
    finally:
        for owner in owners:
            if not owner.closed:
                try: owner.close()
                except BaseException as error: report['cleanup_errors'].append('Owner close: ' + str(error))
        metrics = [owner.metrics() for owner in owners]; keys = ('contexts_created', 'contexts_destroyed', 'runs_completed', 'output_buffers_acquired', 'output_buffers_released')
        report['owner_metrics'] = metrics; report['resource_totals'] = {k: sum(r[k] for r in metrics) for k in keys}
        report['all_contexts_and_pending_outputs_closed'] = all(r['closed'] and r['live_contexts'] == r['live_output_buffers'] == 0 and r['resource_counts_balanced'] for r in metrics)
        if report['status'] == SUCCESS and (report['resource_totals'] != report['expected_resource_counts'] or not report['all_contexts_and_pending_outputs_closed']): report['cleanup_errors'].append('SDK resource ledger incomplete')
        try: os.sched_setaffinity(0, old_affinity)
        except BaseException as error: report['cleanup_errors'].append('Affinity restore: ' + str(error))
        os.environ.clear(); os.environ.update(old_env)
        for number, handler in handlers.items(): signal.signal(number, handler)
        try: report['source_and_inputs_unchanged'] = all(p.is_file() and sha(p) == digest for p, digest in watched.items())
        except BaseException as error: report['source_and_inputs_unchanged'] = False; report['cleanup_errors'].append('Identity: ' + str(error))
        report.update(affinity_restored=set(os.sched_getaffinity(0)) == old_affinity, environment_restored=dict(os.environ) == old_env,
            pinned_files={str(p): digest for p, digest in watched.items()})
        if report['cleanup_errors'] or any(r['cleanup_errors'] for r in metrics) or not report['all_contexts_and_pending_outputs_closed'] or not all(report[k] for k in ('source_and_inputs_unchanged', 'affinity_restored', 'environment_restored')): report['status'] = 'failed_cleanup_or_identity'
        progress('finished')
    return 0 if report['status'] == SUCCESS else 1


if __name__ == '__main__':
    raise SystemExit(main())
