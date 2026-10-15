#!/usr/bin/env python3
"""Two fresh B-attention calibration contexts, flags0/8; no acceptance.

Only the saved real pair0 is loaded. Preserve the existing finite-output guard
and its already-owned failed copies. Profile attribution requires final raw
byte parity. No additional baseline, heldout or FP32 model is executed.
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

COMPILER_SHA = '95e2e56c2c090e5f5de389466e704aa39353972044c96c372b320ec8ff47c8dc'
PARENT_COMPILER_SHA = 'f1c94014ad6f7555c32ac0f83d489ade46295cef96362ed7f7c9878f01cf9780'
NORMAL_SHA = 'd3157c5d1cecc4de541e9ff9a0d2adbf02e4a0269018892f24cddecbe47fce30'
OWNED_SHA = '5aa75a7630ecb9793773553da1f7351a97ca5bac728d4f01f9b74d83d3ca2426'
PROFILE_SHA = '4013651f463a39475560b7c2e0544b3ce0fb813ece5a73e343cae9299e84d91b'
PROFILE_SOURCE_SHA = 'e5671ade47ca65244c529b3ddb325c6aac1d6234a215e43a4652ae18ce135a79'
CAPTURE_SHA = 'cc2dd7fbddef8ee12de00b7a5cc88a16ba9d531783ab09ae8e66abe50df43408'
SELECTOR_SHA = 'f4684d9c40d5a2fec133d3fe73d75672de62d4edcc004a7f0b3f1e6b77459509'
POLICY_SHA = 'febe36eae962b52bcb0742676ae951ae0eb0b8347f9825e768af558015cfc70f'
STEP1_B_SHA = 'b334bf5abe5935d6d5d316d3f46fc583fdba02bb0d6c4286321bc7514e8baa69'
GRAPH_SHA = 'dacaec9a1dee72b83460c3109540c11c67683ba8e5ccc97428ed3f8bb1da719e'
FRAME, GLOBAL = [2, 1041, 1024], [1, 2082, 1024]
NAMES = ['frame_output', 'global_output']
SUCCESS = 'completed_B_attention_calibration_diagnostic_not_performance_or_quality_accepted'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile-receipt', type=Path, required=True)
    parser.add_argument('--compile-sha256', required=True)
    parser.add_argument('--compiler-source', type=Path, required=True)
    parser.add_argument('--normal-source', type=Path, default=Path(__file__).with_name('run_hybrid_pair_board.py'))
    parser.add_argument('--owned-source', type=Path, default=Path(__file__).with_name('profile_owned_context_r2.py'))
    parser.add_argument('--profile-receipt', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(); sys.dont_write_bytecode = True
    if platform.machine() != 'aarch64' or b'rockchip,rk3588' not in Path('/proc/device-tree/compatible').read_bytes():
        raise RuntimeError('RK3588 board only')
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    environment, affinity, handlers, watched, owners = dict(os.environ), set(os.sched_getaffinity(0)), {}, {}, []
    began = time.monotonic()
    report = dict(schema='v128_original_B_attention_FP16_calibration_flags8_diagnostic', status='running',
        input_id='pair0', source_pair_indices=[0, 8], input_split='calibration', heldout_arrays_accessed=False,
        core_mask=1, flags_per_context=[0, 8], baseline_NPU_calls=0, maximum_live_contexts=1,
        profiling_only=True, candidate_accepted=False, numerical_acceptance=False,
        requested_FP16_attention_and_eligible_INT8_MLP_not_actual_placement_proof=True,
        system_or_frequency_writes=False, normal_performance_claim=False,
        production_finite_guard_modified=False, nonfinite_values_rewritten=False, calls=[], cleanup_errors=[])

    def require(ok, message):
        if not ok: raise RuntimeError(message)

    def sha(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(4 << 20), b''): h.update(block)
        return h.hexdigest()

    def pin(path, digest, size=None):
        path = Path(path).resolve(strict=True)
        require(len(digest) == 64 and sha(path) == digest and (size is None or path.stat().st_size == size), 'Pinned file differs: ' + str(path))
        watched[path] = digest; return path

    def relative(folder, row):
        name = Path(row['file']); require(not name.is_absolute() and '..' not in name.parts, 'Owned relative artifact required')
        return pin(folder / name, row['sha256'], row['bytes'])

    def module(path, digest, name):
        spec = importlib.util.spec_from_file_location(name, pin(path, digest))
        value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value

    def clean(row):
        return not row['cleanup_errors'] and all(row[k] for k in ('source_and_inputs_unchanged', 'affinity_restored', 'environment_restored'))

    def progress(stage):
        report.update(stage=stage, diagnostic_seconds=time.monotonic() - began)
        temporary = output / 'result.tmp'; temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n'); temporary.replace(output / 'result.json')
        print(stage, flush=True)

    def interrupted(number, frame): raise InterruptedError('External controller signal ' + str(number))

    report['script_sha256'] = sha(__file__); watched[Path(__file__).resolve()] = report['script_sha256']; progress('preflight')
    try:
        for number in (signal.SIGTERM, signal.SIGINT): handlers[number] = signal.signal(number, interrupted)
        require({4, 6} <= affinity, 'A76 affinity required'); os.sched_setaffinity(0, {4, 6})
        os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        import numpy as np
        pin(args.compiler_source, COMPILER_SHA)
        M = module(args.normal_source, NORMAL_SHA, '_v128_B_attention_logical_IO')
        owned = module(args.owned_source, OWNED_SHA, '_v128_B_attention_owned_IO')
        helper_path = M.E / 'vggt-v99-source-r1/run_native_block_r1.py'
        H = module(helper_path, M.HELPER_SHA, '_v128_B_attention_platform')
        abi = pin(M.B / 'src/deploy/rk3588/validation/run-v12-rknn-block-board.py', M.ABI_SHA)
        library = pin('/usr/lib/librknnrt.so', M.SDK_SHA)
        original_path = pin(args.profile_receipt, PROFILE_SHA); prior = json.loads(original_path.read_text())
        require(prior['schema'] == 'v128_original_native518_calibration_pair_flags8_placement_diagnostic' and
            prior['status'] == 'completed_calibration_profile_diagnostic_not_performance_or_quality_accepted' and
            prior['script_sha256'] == PROFILE_SOURCE_SHA and prior['owned_IO_source_sha256'] == OWNED_SHA and
            prior['input_id'] == 'pair0' and prior['source_pair_indices'] == [0, 8] and prior['input_split'] == 'calibration' and
            not prior['heldout_arrays_accessed'] and prior['all_calls_completed'] and prior['all_resources_closed'] and clean(prior), 'Original saved calibration lineage required')
        cap_path = pin(prior['capture_receipt']['file'], CAPTURE_SHA); capture = json.loads(cap_path.read_text())
        require(capture['schema'] == 'v128_original_native518_frame0_input_capture' and
            capture['status'] == 'completed_original_native518_frame0_input_subbatch_capture' and
            capture['script_sha256'] == M.CAPTURE_SOURCE_SHA and capture['all_NPU_contexts_closed'] and clean(capture), 'Clean original input capture required')
        record = next(r for r in capture['records'] if r['id'] == 'pair0')
        require(record['input'] == prior['input'] and record['split'] == 'calibration' and record['source_pair_indices'] == [0, 8] and
            record['input_is_before_original_frame0'] and record['actual_four_DINO_groups_executed'] and record['original_final_norm_and_role_concat_executed'], 'Actual original frame0 input required')
        x = np.load(relative(cap_path.parent, record['input']), allow_pickle=False)
        require(x.dtype == np.float32 and list(x.shape) == FRAME == record['input']['shape'] and x.flags.c_contiguous and
            np.isfinite(x).all() and x.nbytes == record['input']['tensor_bytes'] and
            hashlib.sha256(x.tobytes()).hexdigest() == record['input']['raw_sha256'], 'Prepared real input bytes differ')
        x.setflags(write=False)
        compile_path = pin(args.compile_receipt, args.compile_sha256); compiled = json.loads(compile_path.read_text())
        require(compiled['schema'] == 'v128_original_native518_B_attention_FP16_MLP_hybrid_step2' and
            compiled['status'] == 'compiled_B_attention_FP16_MLP_hybrid_not_runtime_or_quality_accepted' and
            compiled['script_sha256'] == COMPILER_SHA and compiled['variant'] == 'B' and compiled['configuration'] == M.QUANT_CONFIG and
            compiled['graph']['sha256'] == GRAPH_SHA and compiled['graph']['bytes'] == 106579200 and
            compiled['original_parameters'] == prior['original_parameters'] and compiled['checkpoint'] == prior['checkpoint'] and
            all(len(compiled['original_parameters'][r]) == 18 for r in ('frame', 'global')) and
            compiled['all_SDK_operations_succeeded'] and compiled['sdk_released'] and compiled['sdk_release_code'] in (0, None) and
            clean(compiled) and compiled['cwd_restored'], 'Actual unchanged B-graph compile required')
        require(compiled['parent_source_sha256'] == PARENT_COMPILER_SHA and
            compiled['base_selector_source_sha256'] == '25db1ca6c66fcea6d84b921c3f8f8382d0d16eeaa9d8f6f75d7ae5b9b804da72' and
            compiled['selector_source_sha256'] == SELECTOR_SHA and compiled['selector_policy']['sha256'] == POLICY_SHA and
            compiled['step1_receipt']['sha256'] == STEP1_B_SHA and compiled['step1_receipt']['script_sha256'] == M.STEP1_SOURCE_SHA and
            compiled['selector_count'] == len(compiled['requested_float16_layers']) == 80 and len(compiled['requested_attention_outputs']) == 57 and
            len(compiled['attention_role_mapping']) == len(compiled['MLP_eligible_role_mapping']) == len(compiled['basis_role_mapping']) == 3 and
            compiled['actual_generated_selector_mapping_recomputed'] and compiled['original_quantization_section_byte_preserved'] and
            compiled['attention_update_inputs_selected'] and compiled['update_branch_inputs_selected'] and compiled['MLP_update_outputs_not_selected'] and
            compiled['requested_policy_only'] and not compiled['protected_precision_placement_established'] and
            not compiled['runtime_inference_invoked'] and not compiled['additional_calibration_invoked'], 'Exact requested attention/MLP policy differs')
        require([r['capture']['sha256'] for r in compiled['three_clean_capture_check_receipts']] == M.CAPTURE_SHAS and
            [r['check']['sha256'] for r in compiled['three_clean_capture_check_receipts']] == M.CHECK_SHAS, 'Original six-case FP32/capture lineage differs')
        ins = [dict(index=0, name='input', shape=FRAME, dtype='float32')]
        outs = [dict(index=i, name=n, shape=s, dtype='float32') for i, (n, s) in enumerate(zip(NAMES, (FRAME, GLOBAL)))]
        require(compiled['input_bindings'] == ins and compiled['output_bindings'] == outs, 'Full canonical pair source IO differs')
        model_path = relative(compile_path.parent, compiled['rknn'])
        report.update(compile_receipt=dict(file=str(compile_path), sha256=args.compile_sha256), compiler_source_sha256=COMPILER_SHA,
            profile_r2_receipt=dict(file=str(original_path), sha256=PROFILE_SHA), capture_receipt=dict(file=str(cap_path), sha256=CAPTURE_SHA),
            input=record['input'], model=compiled['rknn'], graph=compiled['graph'], original_parameters=compiled['original_parameters'],
            checkpoint=compiled['checkpoint'], selector_policy_sha256=POLICY_SHA, selector_source_sha256=SELECTOR_SHA,
            selector_count=80, requested_attention_selector_count=57, base_highway_selector_count=23,
            actual_internal_FP16_INT8_placement_established=False, owned_IO_source_sha256=OWNED_SHA, normal_source_sha256=NORMAL_SHA)
        require(shutil.disk_usage(output).free >= 128 << 20, 'Four owned outputs need at least128MiB free output storage')
        IO = M.logical_context(owned.OwnedContext); normal = None
        for flags in (0, 8):
            sample = H.sample(); require(sample['npu_hz'] == 1000000000 and sample['temperatures_c'] and max(sample['temperatures_c'].values()) < 75, 'External 1GHz/thermal guard differs')
            call = dict(index=len(report['calls']), init_flags=flags, status='running', platform_before=sample); report['calls'].append(call)
            owner = None; progress('flags' + str(flags) + '_init')
            try:
                try:
                    owner = IO(model_path, compiled['rknn']['sha256'], input_shapes=[FRAME], output_shapes=[FRAME, GLOBAL], input_names=['input'],
                        output_names=NAMES, core_mask=1, flags=flags, abi_source=abi, library=library, runtime_helper=helper_path)
                except BaseException as error:
                    owner = getattr(error, 'owned_context', None)
                    if owner is not None: owners.append(owner)
                    raise
                owners.append(owner); state = owner.metrics()
                call.update(input_descriptors=copy.deepcopy(state['input_descriptors']), output_descriptors=copy.deepcopy(state['output_descriptors']), invalid_numeric=False)
                try: ys = owner.run([x])
                except RuntimeError as error:
                    ys = getattr(owner, 'last_failed_outputs', None)
                    complete = isinstance(ys, list) and len(ys) == 2 and all(isinstance(y, np.ndarray) and y.dtype == np.float32 and
                        list(y.shape) == shape and y.flags.c_contiguous for y, shape in zip(ys, (FRAME, GLOBAL)))
                    if type(error) is not RuntimeError or str(error) != 'Nonfinite owned SDK output' or not complete or not any(not np.isfinite(y).all() for y in ys): raise
                    state = owner.metrics(); require(state['runs_completed'] == state['output_get_successes'] == 1 and not owner.active and
                        state['live_output_buffers'] == 0 and state['output_buffers_acquired'] == state['output_buffers_released'] == 2, 'Failed payload requires completed original owned IO/release')
                    call.update(invalid_numeric=True, original_output_guard_raised=True, raw_failed_outputs_preserved=True, numerical_guard_error=str(error))
                call['outputs'] = []
                for name, y in zip(NAMES, ys):
                    path = output / ('flags' + str(flags) + '-' + name + '.npy')
                    with path.open('xb') as stream: np.save(stream, y, allow_pickle=False)
                    call['outputs'].append(dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path), name=name,
                        shape=list(y.shape), dtype=str(y.dtype), tensor_bytes=y.nbytes, raw_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
                        finite=bool(np.isfinite(y).all()), finite_count=int(np.isfinite(y).sum()), nan_count=int(np.isnan(y).sum()),
                        positive_inf_count=int(np.isposinf(y).sum()), negative_inf_count=int(np.isneginf(y).sum())))
                call['all_outputs_finite'] = all(r['finite'] for r in call['outputs'])
                require(call['invalid_numeric'] == (not call['all_outputs_finite']), 'Original guard and retained numeric evidence disagree')
                if flags == 0: normal = ys
                else:
                    call['all_outputs_byte_equal_to_flags0'] = all(y.shape == ref.shape and y.dtype == ref.dtype and y.tobytes() == ref.tobytes() for y, ref in zip(ys, normal))
                progress('flags' + str(flags) + '_owned_outputs_saved')
                if flags == 8:
                    detail = owner.base.query(owner.api, owner.context, 3, owner.base.PerfDetail())
                    require(detail.perf_data and 0 < detail.data_len <= 16 << 20, 'Invalid profile pointer/length')
                    data = ct.string_at(detail.perf_data, int(detail.data_len)).rstrip(b'\x00')
                    path = output / 'perf-detail.txt'; path.write_bytes(data)
                    perf = owner.base.query(owner.api, owner.context, 4, owner.base.PerfRun())
                    call.update(profile=dict(file=path.name, bytes=len(data), sha256=sha(path)),
                        pointer_copied_before_next_query=True, SDK_reported_run_duration_us=int(perf.run_duration),
                        profile_attribution_allowed=call['all_outputs_byte_equal_to_flags0'])
                memory = owner.base.query(owner.api, owner.context, 6, owner.base.MemSize())
                call['SDK_model_memory_bytes'] = {n: int(getattr(memory, n)) for n, _ in memory._fields_ if n != 'reserved'}
                call['status'] = 'completed_invalid_numeric_diagnostic' if call['invalid_numeric'] else 'completed'
            finally:
                if owner is not None:
                    owner.close(); call['owner_metrics'] = owner.metrics()
                progress('flags' + str(flags) + '_closed')
        report.update(all_calls_completed=True, saved_array_count=4,
            all_outputs_finite=all(c['all_outputs_finite'] for c in report['calls']),
            attribution_byte_parity_passed=report['calls'][1]['all_outputs_byte_equal_to_flags0'])
        report['status'] = SUCCESS if report['attribution_byte_parity_passed'] else 'completed_B_attention_calibration_no_profile_attribution'
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc()); traceback.print_exc()
    finally:
        for owner in owners:
            try: owner.close()
            except BaseException as error: report['cleanup_errors'].append('Owned close: ' + str(error))
        states = [o.metrics() for o in owners]
        totals = {k: sum(s[k] for s in states) for k in ('contexts_created', 'contexts_destroyed', 'runs_completed', 'output_buffers_acquired', 'output_buffers_released')}
        report.update(resource_totals=totals, owner_metrics=states, all_resources_closed=all(s['closed'] and s['resource_counts_balanced'] and not s['cleanup_errors'] for s in states))
        if report['status'].startswith('completed') and (totals != dict(contexts_created=2, contexts_destroyed=2, runs_completed=2, output_buffers_acquired=4, output_buffers_released=4) or not report['all_resources_closed']):
            report['cleanup_errors'].append('Expected two-context/two-run/four-output ledger differs')
        try: report['source_and_inputs_unchanged'] = all(p.exists() and sha(p) == s for p, s in watched.items())
        except BaseException as error: report['source_and_inputs_unchanged'] = False; report['cleanup_errors'].append('Identity: ' + str(error))
        try: os.sched_setaffinity(0, affinity)
        except BaseException as error: report['cleanup_errors'].append('Affinity: ' + str(error))
        os.environ.clear(); os.environ.update(environment)
        for number, handler in handlers.items(): signal.signal(number, handler)
        report.update(environment_restored=dict(os.environ) == environment, affinity_restored=set(os.sched_getaffinity(0)) == affinity,
            pinned_files={str(p): s for p, s in watched.items()})
        if report['cleanup_errors'] or not all(report[k] for k in ('source_and_inputs_unchanged', 'environment_restored', 'affinity_restored')):
            report['status'] = 'failed_cleanup_or_identity'
        progress('finished')
    return 1 if report['status'].startswith('failed') else 0


if __name__ == '__main__': raise SystemExit(main())
