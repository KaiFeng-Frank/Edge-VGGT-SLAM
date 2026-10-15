#!/usr/bin/env python3
"""Board-only A/F/B/B/F/A diagnostic of complete native frame0/global0 pairs.

One context per phase, two pinned inputs, owned FP32 SDK IO. FP16 errors are
reported without a new acceptance threshold; this is not full-model quality.
"""
import argparse
import copy
import faulthandler
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time
import traceback

B = Path('/mnt/vggt-data/vggt-slam2'); E = B / 'experiments'
HEADS_SHA = '0bfcb295ad5490e8196e8fcc1244e11a99b07bfb3ba7d5c350cb8c81d3ff3be4'
HELPER_SHA = '56ce45635a5d010e862bf59db2f9fcc6e8ec642b2b023c8c40cad0bd5808050f'
ABI_SHA = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
SDK_SHA = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
COMPILER_SHA = '2bbf180a9a70444ba592b7898fa23b96b11f36da3859f07cc5b435f0a25acb11'
EXPORT_SHA = 'e23c8f7cdf6faa895a8bde34a41806bf2022e630f5361529d0271374742d740e'
CHECK_SHA = '48a7d5034b9eea0afa139e4a69cdb5b393c694dcec2b8e874a1c1f380f5f42b1'
CONFIG_SHA = '442e22d5f8aaa033604610c92d5ce6a8b889005601f3d9f1cf78394c4f41842a'
CONFIG = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
    mean_values=None, std_values=None, quant_img_RGB2BGR=False, model_pruning=False,
    remove_reshape=False, quantize_weight=False, enable_flash_attention=False)
BUILD = dict(do_quantization=False, dataset=None, auto_hybrid=False)
FRAME, GLOBAL = [2, 1041, 1024], [1, 2082, 1024]
SCHEDULE = ['A', 'F', 'B', 'B', 'F', 'A']


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for variant in ('A', 'F', 'B'):
        p.add_argument('--compile-' + variant, type=Path, required=True)
        p.add_argument('--compile-' + variant + '-sha256', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--export-receipt', type=Path, default=Path('/tmp/edge-vggt-v126-basis-export-r2/result.json'))
    p.add_argument('--check-receipt', type=Path, default=Path('/tmp/edge-vggt-v126-basis-check-r2/result.json'))
    p.add_argument('--config', type=Path)
    p.add_argument('--heads-source', type=Path, default=E / 'vggt-v100-source-r1/native_heads_runtime.py')
    p.add_argument('--runtime-helper', type=Path, default=E / 'vggt-v99-source-r1/run_native_block_r1.py')
    p.add_argument('--abi-source', type=Path, default=B / 'src/deploy/rk3588/validation/run-v12-rknn-block-board.py')
    p.add_argument('--runtime-library', type=Path, default=Path('/usr/lib/librknnrt.so'))
    p.add_argument('--expected-npu-hz', type=int, choices=[300000000, 1000000000], default=1000000000)
    p.add_argument('--cpu-affinity', type=int, nargs='+', default=[4, 6])
    a = p.parse_args()
    require(platform.machine() == 'aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(), 'RK3588S-only runtime; no host execution')
    faulthandler.enable()
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve(); watched = {source: sha(source)}
    old_env, old_affinity = dict(os.environ), set(os.sched_getaffinity(0))
    owners, handlers, inputs = [], {}, {}; began = time.monotonic()
    report = dict(schema='v127_original_native518_basis_pair_FP16_runtime', status='running',
        script_sha256=watched[source], compiler_source_sha256=COMPILER_SHA, schedule=SCHEDULE,
        core_mask=1, init_flags=0, candidate_accepted=False, FP16_hard_acceptance_gate_defined=False,
        full_model_quality_performance_realtime_or_8GB_accepted=False, system_or_frequency_writes=False,
        input_API='Logical FP32, SDK pass_through0; want_float1 outputs; owned copies before release',
        timing_scope='Per-call outer OwnedContext.run includes preparation, SDK FP32 conversion/set/run/get, all copies/release/finite. Phase lifecycle includes model rehash, init, two calls and destroy; excludes platform samples, comparison/hash and saving.',
        phases=[], calls=[], platform_samples=[], cleanup_errors=[])

    def progress(stage):
        report.update(stage=stage, total_wall_seconds=time.monotonic() - began,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        temp = out / 'result.tmp'; temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n'); temp.replace(out / 'result.json')
        print(json.dumps(dict(stage=stage, status=report['status'])), flush=True)

    def pin(path, digest, size=None):
        path = Path(path).resolve(strict=True)
        require(len(digest) == 64 and sha(path) == digest and (size is None or path.stat().st_size == size), 'Pinned file differs: ' + str(path))
        watched[path] = digest; return path

    def module(path, digest, name):
        spec = importlib.util.spec_from_file_location(name, pin(path, digest))
        value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value

    def array(path, row, shape):
        value = np.load(pin(path, row['sha256'], row['bytes']), allow_pickle=False)
        require(value.dtype == np.float32 and list(value.shape) == shape == row['shape'] and
                value.flags.c_contiguous and np.isfinite(value).all() and raw(value) == row['raw_sha256'], 'Logical FP32 artifact differs')
        value.setflags(write=False); return value

    def raw(value):
        return hashlib.sha256(value.tobytes(order='C')).hexdigest()

    def save(name, value):
        file = out / name
        with file.open('xb') as stream:
            np.save(stream, value, allow_pickle=False)
        return dict(file=name, bytes=file.stat().st_size, sha256=sha(file), shape=list(value.shape),
                    dtype=str(value.dtype), tensor_bytes=value.nbytes, raw_sha256=raw(value), finite=bool(np.isfinite(value).all()))

    def comparison(actual, reference):
        x, y = actual.astype(np.float64), reference.astype(np.float64); delta = x - y
        norm = float(np.linalg.norm(y)); outside = int(np.count_nonzero(np.abs(delta) > 1e-5 + 1e-4 * np.abs(y)))
        return dict(relative_l2=float(np.linalg.norm(delta)) / norm if norm > 0 else None,
                    zero_reference_norm=norm == 0, max_abs=float(np.abs(delta).max()),
                    outside_count=outside, strict_pass=outside == 0, rtol=1e-4, atol=1e-5,
                    strict_is_diagnostic_not_FP16_acceptance=True, byte_equal=actual.tobytes(order='C') == reference.tobytes(order='C'))

    def observe(stage):
        value = H.sample(); report['platform_samples'].append(dict(stage=stage, monotonic=time.monotonic(), **value))
        require(value['npu_hz'] == a.expected_npu_hz and value['temperatures_c'] and max(value['temperatures_c'].values()) < 75, 'External frequency/thermal guard differs')

    def interrupted(number, frame):
        raise InterruptedError('External controller signal ' + str(number))

    progress('preflight')
    try:
        for number in (signal.SIGTERM, signal.SIGINT):
            handlers[number] = signal.signal(number, interrupted)
        require(a.cpu_affinity and set(a.cpu_affinity) <= old_affinity & {4, 5, 6, 7}, 'A76 affinity unavailable')
        os.sched_setaffinity(0, set(a.cpu_affinity))
        os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        ep, cp = pin(a.export_receipt, EXPORT_SHA), pin(a.check_receipt, CHECK_SHA)
        e, c = json.loads(ep.read_text()), json.loads(cp.read_text())
        require(e['schema'] == 'v126_original_native518_basis_pair_export' and e['status'] == 'exported_original_basis_pair_not_numerically_accepted' and
                c['schema'] == 'v126_original_native518_basis_pair_check' and c['status'] == 'checked_original_basis_pair_fixed_FP32_gate_passed' and
                c['FP32_gate_passed'] and c['all6_calls_completed'] and c['models'] == e['models'] and c['export_receipt']['sha256'] == EXPORT_SHA and
                e['source_and_inputs_unchanged'] and c['source_and_inputs_unchanged'] and not e['cleanup_errors'] and not c['cleanup_errors'], 'Pinned completed R126 gate required')
        cfg_path = a.config or next(Path(path) for path, digest in c['input_and_source_SHA_pins'].items()
                                   if Path(path).name == 'basis-pair-inputs-r1.json' and digest == CONFIG_SHA)
        cfg = json.loads(pin(cfg_path, CONFIG_SHA).read_text())
        require(e['input_cases'] == c['input_cases'] == cfg['inputs'] and [row['id'] for row in cfg['inputs']] == ['cpu_prefix', 'npu_prefix'], 'Actual input contract differs')
        import numpy as np
        models, compiles, references = {}, {}, {}
        for variant in ('A', 'F', 'B'):
            compile_path = pin(getattr(a, 'compile_' + variant), getattr(a, 'compile_' + variant + '_sha256'))
            receipt = json.loads(compile_path.read_text()); graph = next(m for m in e['models'] if m['key'] == variant)
            require(receipt['schema'] == 'v127_original_native518_basis_pair_FP16_compile' and receipt['status'] == 'compiled_not_runtime_accepted' and
                    receipt['variant'] == variant and receipt['script_sha256'] == COMPILER_SHA and receipt['configuration'] == CONFIG and receipt['build_options'] == BUILD and
                    receipt['export_receipt']['sha256'] == EXPORT_SHA and receipt['check_receipt']['sha256'] == CHECK_SHA and
                    receipt['onnx']['sha256'] == graph['onnx']['sha256'] and receipt['onnx']['bytes'] == graph['onnx']['bytes'] and
                    receipt['input_bindings'] == graph['input_bindings'] and receipt['output_bindings'] == graph['output_bindings'] and
                    receipt['all_operations_succeeded'] and receipt['sdk_released'] and receipt['sdk_release_code'] in (None, 0) and
                    all(receipt[k] for k in ('source_and_inputs_unchanged', 'affinity_restored', 'environment_restored', 'cwd_restored')) and not receipt['cleanup_errors'], 'Actual clean FP16 compile lineage differs')
            row = receipt['rknn']; model = pin(compile_path.parent / row['file'], row['sha256'], row['bytes'])
            models[variant] = dict(file=str(model), bytes=row['bytes'], sha256=row['sha256'])
            compiles[variant] = dict(file=str(compile_path), sha256=getattr(a, 'compile_' + variant + '_sha256'))
        for case in cfg['inputs']:
            inputs[case['id']] = array(case['input']['file'], case['input'], FRAME)
        require(raw(inputs['cpu_prefix']) != raw(inputs['npu_prefix']), 'Two distinct real prefix distributions required')
        for row in c['checks']:
            references[(row['input_id'], row['variant'])] = [array(cp.parent / artifact['file'], artifact, shape)
                                                         for artifact, shape in zip(row['outputs'], (FRAME, GLOBAL))]
        require(len(references) == 6 and all(len(v) == 2 for v in references.values()), 'Complete matched FP32 endpoints required')
        heads = module(a.heads_source, HEADS_SHA, '_v127_frozen_owned_IO')
        H = module(a.runtime_helper, HELPER_SHA, '_v127_frozen_platform_IO')
        abi, library = pin(a.abi_source, ABI_SHA), pin(a.runtime_library, SDK_SHA)
        report.update(export_receipt=dict(file=str(ep), sha256=EXPORT_SHA), check_receipt=dict(file=str(cp), sha256=CHECK_SHA),
                      models=models, compile_receipts=compiles, inputs=cfg['inputs'], input_scope=cfg['scope'],
                      checkpoint=e['checkpoint'], configuration=CONFIG, build_options=BUILD,
                      expected_resource_counts=dict(contexts_created=6, contexts_destroyed=6, runs_completed=12,
                                                    output_buffers_acquired=24, output_buffers_released=24))
        baseline_a, first_hashes = {}, {}; phase_owner = None
        for phase, variant in enumerate(SCHEDULE):
            observe('before_phase_' + str(phase)); phase_row = dict(index=phase, variant=variant, status='running')
            report['phases'].append(phase_row); progress('phase_' + str(phase) + '_init')
            phase_owner = None; payloads = []; lifecycle_tick = time.perf_counter(); init_tick = lifecycle_tick
            try:
                try:
                    phase_owner = heads.OwnedContext(models[variant]['file'], models[variant]['sha256'], input_shapes=[FRAME],
                        output_shapes=[FRAME, GLOBAL], input_names=['input'], output_names=['frame_output', 'global_output'],
                        core_mask=1, flags=0, abi_source=abi, library=library, runtime_helper=a.runtime_helper)
                except BaseException as error:
                    phase_owner = getattr(error, 'owned_context', None)
                    if phase_owner is not None: owners.append(phase_owner)
                    raise
                owners.append(phase_owner); phase_row['init_seconds'] = time.perf_counter() - init_tick
                for case_id, x in inputs.items():
                    tick = time.perf_counter(); ys = phase_owner.run([x]); elapsed = time.perf_counter() - tick
                    call = dict(index=len(report['calls']), phase=phase, variant=variant, input_id=case_id,
                                elapsed_seconds=elapsed, SDK_stages=copy.deepcopy(phase_owner.last_timings), status='owned_outputs_copied_and_released')
                    report['calls'].append(call); payloads.append((call, ys))
            finally:
                if phase_owner is not None:
                    close_tick = time.perf_counter()
                    try:
                        phase_owner.close()
                    except BaseException as error:
                        report['cleanup_errors'].append('Phase close: ' + str(error)); raise
                    finally:
                        phase_row.update(destroy_seconds=time.perf_counter() - close_tick,
                            lifecycle_seconds_excluding_save_compare=time.perf_counter() - lifecycle_tick,
                            owner_metrics=phase_owner.metrics())
            phase_row['status'] = 'closed_before_next_context'; observe('after_phase_' + str(phase))
            progress('phase_' + str(phase) + '_closed')
            for call, ys in payloads:
                case_id = call['input_id']; own_ref = references[(case_id, variant)]; canonical_ref = references[(case_id, 'A')]
                if phase == 0: baseline_a[case_id] = ys
                output_rows, comparisons = [], {}
                for index, (name, y) in enumerate(zip(('frame_output', 'global_output'), ys)):
                    output_rows.append(save('call%02d-%s.npy' % (call['index'], name), y))
                    comparisons[name + '_vs_own_FP32'] = comparison(y, own_ref[index])
                    comparisons[name + '_vs_original_A_FP32'] = comparison(y, canonical_ref[index])
                    comparisons[name + '_vs_first_A_FP16_same_input'] = comparison(y, baseline_a[case_id][index])
                    key = (variant, case_id, name); digest = raw(y)
                    if key not in first_hashes: first_hashes[key] = digest
                    output_rows[-1]['repeat_raw_sha256_equal'] = digest == first_hashes[key]
                call.update(outputs=output_rows, comparisons=comparisons,
                            repeat_raw_sha256_equal=all(row['repeat_raw_sha256_equal'] for row in output_rows))
            require(all(raw(x) == case['input']['raw_sha256'] for case, x in zip(cfg['inputs'], inputs.values())), 'Original input bytes changed')
            progress('phase_' + str(phase) + '_saved')
        report.update(status='completed_basis_pair_FP16_component_diagnostic_not_accepted',
                      all12_calls_completed=True, all_output_repetitions_byte_equal=all(row['repeat_raw_sha256_equal'] for row in report['calls']),
                      timing_summary={v: dict(call_count=4, all_call_seconds=[row['elapsed_seconds'] for row in report['calls'] if row['variant'] == v],
                                             all_call_mean_seconds=sum(row['elapsed_seconds'] for row in report['calls'] if row['variant'] == v) / 4,
                                             cold_phase_lifecycle_seconds=[row['lifecycle_seconds_excluding_save_compare'] for row in report['phases'] if row['variant'] == v]) for v in ('A', 'F', 'B')},
                      timing_limit='Two fresh contexts per variant; both inputs are the same image pair with different prefix provenance. No steady-stream rate or P95 claim.')
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc()); traceback.print_exc()
    finally:
        for owner in owners:
            if not owner.closed:
                try:
                    owner.close()
                except BaseException as error:
                    report['cleanup_errors'].append('Final owner close: ' + str(error))
        metrics = [owner.metrics() for owner in owners]
        keys = ('contexts_created', 'contexts_destroyed', 'runs_completed', 'output_buffers_acquired', 'output_buffers_released')
        report['owner_metrics'] = metrics; report['resource_totals'] = {key: sum(row[key] for row in metrics) for key in keys}
        report['all_contexts_and_pending_outputs_closed'] = all(row['closed'] and row['live_contexts'] == row['live_output_buffers'] == 0 and row['resource_counts_balanced'] for row in metrics)
        if report['status'].startswith('completed_'):
            if not (report['resource_totals'] == report['expected_resource_counts'] and report['all_contexts_and_pending_outputs_closed'] and
                    all(not row['cleanup_errors'] for row in metrics)):
                report['cleanup_errors'].append('Actual SDK resources incomplete/unbalanced')
        try:
            os.sched_setaffinity(0, old_affinity)
        except BaseException as error:
            report['cleanup_errors'].append('Affinity restore: ' + str(error))
        os.environ.clear(); os.environ.update(old_env)
        for number, handler in handlers.items(): signal.signal(number, handler)
        report.update(affinity_restored=set(os.sched_getaffinity(0)) == old_affinity, environment_restored=dict(os.environ) == old_env,
                      source_and_inputs_unchanged=all(path.is_file() and sha(path) == expected for path, expected in watched.items()),
                      input_and_source_SHA_pins={str(path): expected for path, expected in watched.items()})
        if report['cleanup_errors'] or not report['all_contexts_and_pending_outputs_closed'] or not all(report[k] for k in ('affinity_restored', 'environment_restored', 'source_and_inputs_unchanged')):
            report['status'] = 'failed_cleanup_or_identity'
        progress('finished')
    return 0 if report['status'] == 'completed_basis_pair_FP16_component_diagnostic_not_accepted' else 1


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
