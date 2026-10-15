#!/usr/bin/env python3
"""One untouched mixedA/pair0 flags0 call with process-local SDK debug dump.

Preserve raw invalid outputs through the diagnostic owner, retaining its finite
guard. Dump attribution requires both final outputs to byte-match actual r2
flags0 payloads. SDK filenames/shapes are observed, not assumed logical BHSD.
"""
import argparse
import faulthandler
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import resource
import signal
import sys
import time
import traceback

PROFILE_SHA = '4013651f463a39475560b7c2e0544b3ce0fb813ece5a73e343cae9299e84d91b'
PROFILE_SOURCE_SHA = 'e5671ade47ca65244c529b3ddb325c6aac1d6234a215e43a4652ae18ce135a79'
OWNED_SHA = '5aa75a7630ecb9793773553da1f7351a97ca5bac728d4f01f9b74d83d3ca2426'
NORMAL_SHA = 'd3157c5d1cecc4de541e9ff9a0d2adbf02e4a0269018892f24cddecbe47fce30'
MODEL_SHA = '48621d9d6db93779215d35856f58d8db378a0779e359b3ea0063e28bc141b6d8'
SUCCESS = 'completed_mixedA_debug_dump_final_byte_equal_not_root_cause_or_quality_accepted'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile-receipt', type=Path, required=True)
    p.add_argument('--normal-source', type=Path, default=Path(__file__).with_name('run_hybrid_pair_board.py'))
    p.add_argument('--owned-source', type=Path, default=Path(__file__).with_name('profile_owned_context_r2.py'))
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args(); sys.dont_write_bytecode = True
    if platform.machine() != 'aarch64' or b'rockchip,rk3588' not in Path('/proc/device-tree/compatible').read_bytes():
        raise RuntimeError('RK3588 board-only SDK dump')
    faulthandler.enable(all_threads=True)
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False); dump = out / 'dumps'; dump.mkdir()
    env, affinity, limit = dict(os.environ), set(os.sched_getaffinity(0)), resource.getrlimit(resource.RLIMIT_FSIZE)
    watched, handlers, owner = {}, {}, None; began = time.monotonic(); default = Path('/userdata/dumps')
    report = dict(schema='v128_original_mixedA_calibration_pair_untouched_RKNN_debug_dump', status='running',
        input_id='pair0', input_split='calibration', init_flags=0, core_mask=1, expected_forward_calls=1,
        model_sha256=MODEL_SHA, candidate_accepted=False, numerical_acceptance=False, performance_measurement=False,
        graph_recompiled_or_outputs_appended=False, CPU_NN_ORt_GPU_executed=False, heldout_arrays_accessed=False,
        production_finite_guard_modified=False, nonfinite_values_rewritten=False, system_or_frequency_writes=False,
        debug_duration_is_not_normal_performance=True, final_raw_byte_parity_required_for_attribution=True,
        dump_path_override_officially_documented=False, dump_path_override_found_in_pinned_runtime_binary=True,
        file_size_soft_limit_bytes=256 << 20, minimum_free_bytes=1 << 30, cleanup_errors=[], dump_files=[])

    def require(ok, message):
        if not ok: raise RuntimeError(message)

    def sha(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(4 << 20), b''): h.update(block)
        return h.hexdigest()

    def pin(path, digest, size=None):
        path = Path(path).resolve(strict=True)
        require(sha(path) == digest and (size is None or path.stat().st_size == size), 'Pinned source/artifact differs: ' + str(path))
        watched[path] = digest; return path

    def relative(folder, row):
        name = Path(row['file']); require(not name.is_absolute() and '..' not in name.parts, 'Receipt-relative artifact required')
        return pin(folder / name, row['sha256'], row['bytes'])

    def module(path, digest, name):
        spec = importlib.util.spec_from_file_location(name, pin(path, digest))
        value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value

    def progress(stage):
        report.update(stage=stage, diagnostic_seconds=time.monotonic() - began, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        temp = out / 'result.tmp'; temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n'); temp.replace(out / 'result.json')
        print(stage, flush=True)

    def counts(value):
        return dict(finite=bool(np.isfinite(value).all()), finite_count=int(np.count_nonzero(np.isfinite(value))),
            nan_count=int(np.count_nonzero(np.isnan(value))), positive_inf_count=int(np.count_nonzero(np.isposinf(value))),
            negative_inf_count=int(np.count_nonzero(np.isneginf(value))))

    def load(folder, row, shape):
        value = np.load(relative(folder, row), allow_pickle=False)
        require(value.dtype == np.float32 and list(value.shape) == shape == row['shape'] and value.flags.c_contiguous and
            hashlib.sha256(value.tobytes()).hexdigest() == row['raw_sha256'], 'Captured FP32 payload differs')
        value.setflags(write=False); return value

    def interrupted(number, frame): raise InterruptedError('Controller or per-file limit signal ' + str(number))

    report['script_sha256'] = sha(__file__); watched[Path(__file__).resolve()] = report['script_sha256']; progress('preflight')
    try:
        require(str(out).startswith('/dev/shm/') and os.statvfs(out).f_bavail * os.statvfs(out).f_frsize >= 1 << 30, 'Fresh tmpfs dump destination needs at least1GiB free')
        require(not os.path.lexists(default), 'Preserve existing /userdata/dumps; default destination must be absent')
        report['default_dump_absent_before'] = True
        for number in (signal.SIGTERM, signal.SIGINT, signal.SIGXFSZ): handlers[number] = signal.signal(number, interrupted)
        require({4, 6} <= affinity, 'A76 affinity required'); os.sched_setaffinity(0, {4, 6})
        soft = min(256 << 20, limit[0]) if limit[0] != resource.RLIM_INFINITY else 256 << 20
        resource.setrlimit(resource.RLIMIT_FSIZE, (soft, limit[1])); report['actual_file_size_soft_limit_bytes'] = soft
        os.environ.update(RKNN_LOG_LEVEL='5', RKNN_DUMP_DIR=str(dump), CUDA_VISIBLE_DEVICES='',
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        for key in ('RKNN_DUMP_QUANT', 'RKNN_DUMP_TXT_TENSOR'): os.environ.pop(key, None)
        # Set the debug environment before importing helpers or loading SDK2.3.2.
        M = module(a.normal_source, NORMAL_SHA, '_v128_dump_logical_IO'); heads = module(a.owned_source, OWNED_SHA, '_v128_dump_owned_IO')
        H = module(M.E / 'vggt-v99-source-r1/run_native_block_r1.py', M.HELPER_SHA, '_v128_dump_platform')
        import numpy as np
        rp = pin(a.profile_receipt, PROFILE_SHA); prior = json.loads(rp.read_text())
        require(prior['schema'] == 'v128_original_native518_calibration_pair_flags8_placement_diagnostic' and
            prior['status'] == 'completed_calibration_profile_diagnostic_not_performance_or_quality_accepted' and
            prior['script_sha256'] == PROFILE_SOURCE_SHA and prior['owned_IO_source_sha256'] == OWNED_SHA and prior['input_id'] == 'pair0' and
            not prior['heldout_arrays_accessed'] and prior['all_calls_completed'] and prior['attribution_byte_parity_passed'] and
            prior['all_resources_closed'] and prior['source_and_inputs_unchanged'] and not prior['cleanup_errors'], 'Actual completed raw calibration diagnostic required')
        rows = [r for r in prior['calls'] if r['route'] == 'mixedA' and r['init_flags'] == 0]
        require(len(rows) == 1 and rows[0]['invalid_numeric'] and [r['name'] for r in rows[0]['outputs']] == M.NAMES, 'Exact invalid mixedA flags0 baseline required')
        baseline = [load(rp.parent, row, shape) for row, shape in zip(rows[0]['outputs'], (M.FRAME, M.GLOBAL))]
        cap_row = prior['capture_receipt']; cp = pin(cap_row['file'], cap_row['sha256']); cap = json.loads(cp.read_text())
        require(cap_row['sha256'] == M.CAPTURE_SHAS[0] and cap['script_sha256'] == M.CAPTURE_SOURCE_SHA and
            cap['status'] == 'completed_original_native518_frame0_input_subbatch_capture' and cap['all_NPU_contexts_closed'] and not cap['cleanup_errors'], 'Pinned original calibration capture required')
        row = next(r for r in cap['records'] if r['id'] == 'pair0'); require(row['split'] == 'calibration' and row['input'] == prior['input'], 'Same pair0 input required')
        x = load(cp.parent, row['input'], M.FRAME); require(np.isfinite(x).all(), 'Original input must be finite')
        model_row = prior['models']['mixedA']; model = pin(model_row['file'], MODEL_SHA, model_row['bytes'])
        abi = pin(M.B / 'src/deploy/rk3588/validation/run-v12-rknn-block-board.py', M.ABI_SHA); library = pin('/usr/lib/librknnrt.so', M.SDK_SHA)
        report.update(profile_receipt=dict(file=str(rp), sha256=PROFILE_SHA), baseline_outputs=rows[0]['outputs'],
            capture_receipt=cap_row, input=row['input'], original_parameters=prior['original_parameters'], checkpoint=prior['checkpoint'],
            owned_source_sha256=OWNED_SHA, normal_source_sha256=NORMAL_SHA, model=model_row, dump_directory=str(dump))
        sample = H.sample(); require(sample['npu_hz'] == 1000000000 and sample['temperatures_c'] and max(sample['temperatures_c'].values()) < 75, 'External 1GHz/thermal guard differs')
        report['platform_before'] = sample; IO = M.logical_context(heads.OwnedContext); progress('untouched_mixedA_init_with_debug_environment')
        try:
            owner = IO(model, MODEL_SHA, input_shapes=[M.FRAME], output_shapes=[M.FRAME, M.GLOBAL], input_names=['input'],
                output_names=M.NAMES, core_mask=1, flags=0, abi_source=abi, library=library, runtime_helper=M.E / 'vggt-v99-source-r1/run_native_block_r1.py')
        except BaseException as error:
            owner = getattr(error, 'owned_context', None); raise
        report['input_descriptors'] = owner.metrics()['input_descriptors']; report['output_descriptors'] = owner.metrics()['output_descriptors']; progress('one_actual_flags0_call')
        try:
            ys = owner.run([x]); report['original_output_guard_raised'] = False
        except RuntimeError as error:
            ys = getattr(owner, 'last_failed_outputs', None)
            complete = isinstance(ys, list) and len(ys) == 2 and all(isinstance(y, np.ndarray) and y.dtype == np.float32 and
                list(y.shape) == shape and y.flags.c_contiguous for y, shape in zip(ys, (M.FRAME, M.GLOBAL)))
            if type(error) is not RuntimeError or str(error) != 'Nonfinite owned SDK output' or not complete or not any(not np.isfinite(y).all() for y in ys): raise
            state = owner.metrics(); require(state['runs_completed'] == state['output_get_successes'] == 1 and not owner.active and
                state['live_output_buffers'] == 0 and state['output_buffers_acquired'] == state['output_buffers_released'] == 2, 'Diagnostic payload requires successful original release')
            report.update(original_output_guard_raised=True, numerical_guard_error=str(error))
        report['outputs'] = []
        for name, y, reference in zip(M.NAMES, ys, baseline):
            path = out / (name + '.npy')
            with path.open('xb') as stream: np.save(stream, y, allow_pickle=False)
            report['outputs'].append(dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path), raw_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
                name=name, shape=list(y.shape), dtype=str(y.dtype), tensor_bytes=y.nbytes, byte_equal_r2_flags0=y.tobytes() == reference.tobytes(), **counts(y)))
        report.update(final_baseline_byte_equal=all(r['byte_equal_r2_flags0'] for r in report['outputs']),
            invalid_numeric=not all(r['finite'] for r in report['outputs']), SDK_IO_timings_diagnostic_only=owner.last_timings)
        progress('final_owned_outputs_saved'); owner.close(); report['status'] = 'completed_dump_pending_inventory'
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc()); traceback.print_exc()
    finally:
        if owner is not None:
            try: owner.close()
            except BaseException as error: report['cleanup_errors'].append('Owned close: ' + str(error))
            report['resources'] = owner.metrics(); report['forward_calls'] = report['resources']['runs_completed']
        # Inventory even an interrupted/failed run; never delete generated tensors.
        try:
            for folder, label in ((dump, 'dumps'), (default, 'unexpected_default_dumps')):
                if folder == default and not report.get('default_dump_absent_before'): continue
                if not folder.exists(): continue
                for path in sorted(folder.rglob('*')):
                    if not path.is_file(): continue
                    item = dict(file=str(Path(label) / path.relative_to(folder)), bytes=path.stat().st_size, sha256=sha(path), unexpected_default_destination=folder == default)
                    match = re.match(r'^(\d+)_([^_]+)_(.*?)_shape_', path.name)
                    if match: item.update(filename_operator_index=int(match[1]), filename_op_type=match[2], filename_tensor_label=match[3], filename_only_binding_not_proven=True)
                    if path.suffix.lower() == '.npy':
                        try:
                            value = np.load(path, allow_pickle=False, mmap_mode='r')
                            item.update(shape=list(value.shape), dtype=str(value.dtype), tensor_bytes=value.nbytes,
                                raw_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest(), **counts(value)); del value
                        except BaseException as error: item['array_parse_error'] = type(error).__name__ + ': ' + str(error)
                    report['dump_files'].append(item)
            report.update(default_dump_absent_after=not os.path.lexists(default), dump_total_file_bytes=sum(r['bytes'] for r in report['dump_files']),
                valid_NPY_dump_count=sum('raw_sha256' in r for r in report['dump_files']),
                dump_layout_not_assumed_logical_BHSD=True, fused_dump_does_not_guarantee_separate_QK_softmax_PV_values=True)
        except BaseException as error: report['cleanup_errors'].append('Dump inventory: ' + str(error))
        try: report['source_and_inputs_unchanged'] = all(p.exists() and sha(p) == s for p, s in watched.items())
        except BaseException as error: report['source_and_inputs_unchanged'] = False; report['cleanup_errors'].append('Identity: ' + str(error))
        for name, action in (('file_limit', lambda: resource.setrlimit(resource.RLIMIT_FSIZE, limit)), ('affinity', lambda: os.sched_setaffinity(0, affinity))):
            try: action()
            except BaseException as error: report['cleanup_errors'].append(name + ': ' + str(error))
        os.environ.clear(); os.environ.update(env)
        for number, handler in handlers.items(): signal.signal(number, handler)
        report.update(environment_restored=dict(os.environ) == env, affinity_restored=set(os.sched_getaffinity(0)) == affinity,
            file_size_limit_restored=resource.getrlimit(resource.RLIMIT_FSIZE) == limit, pinned_files={str(p): s for p, s in watched.items()})
        if report['status'] == 'completed_dump_pending_inventory':
            expected = dict(contexts_created=1, contexts_destroyed=1, runs_completed=1, output_buffers_acquired=2, output_buffers_released=2)
            state = report['resources']; require_counts = {k: state[k] for k in expected} == expected and state['closed'] and state['resource_counts_balanced'] and not state['cleanup_errors']
            report['resource_counts_verified'] = require_counts
            if not require_counts: report['cleanup_errors'].append('Expected one-context/one-run/two-output ledger failed')
            report['attribution_preconditions_met'] = report['final_baseline_byte_equal'] and report['valid_NPY_dump_count'] > 0 and report['default_dump_absent_after']
            report['status'] = SUCCESS if report['attribution_preconditions_met'] else 'completed_debug_dump_no_attribution'
        if report['cleanup_errors'] or not all(report[k] for k in ('source_and_inputs_unchanged', 'environment_restored', 'affinity_restored', 'file_size_limit_restored')): report['status'] = 'failed_cleanup_or_identity'
        progress('finished')
    return 1 if report['status'].startswith('failed') else 0


if __name__ == '__main__': raise SystemExit(main())
