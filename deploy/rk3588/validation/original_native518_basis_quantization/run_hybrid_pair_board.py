#!/usr/bin/env python3
"""Board-only heldout diagnostic: original FP16 A versus manual-hybrid A/F/B.

Eight separately owned contexts each run pair4 and pair5. Ordinary logical
FP32 SDK IO includes SDK conversion, complete owned copies and release. This
does not establish internal mixed precision, trajectory quality or deployment.
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
HEADS_SHA = '0bfcb295ad5490e8196e8fcc1244e11a99b07bfb3ba7d5c350cb8c81d3ff3be4'
HELPER_SHA = '56ce45635a5d010e862bf59db2f9fcc6e8ec642b2b023c8c40cad0bd5808050f'
ABI_SHA = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
SDK_SHA = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
FP16_COMPILER_SHA = '2bbf180a9a70444ba592b7898fa23b96b11f36da3859f07cc5b435f0a25acb11'
FP16_COMPILE_SHA = '71b8fc05994e15cf3a658d01f57b89f909172617ff513dea97971dc13d18bb5c'
FP16_MODEL_SHA = '803f7e53c5cf574ccb86e1bdd5b9091332f21ec48aef8864fdee52bc72017666'
STEP1_SOURCE_SHA = 'a9b6f6b591e7ec7fef39f04ece167115445f445eecd87f996cdb5921035a20f6'
CAPTURE_SOURCE_SHA = 'ed500643ef333dd824ccab81a126e87ff8616c3cb3ee09e3ec59f2a4b7bbf7e9'
CHECKER_SHA = 'b0f4bd489d1e6ec8eb28869f609d7e48469aa9d19b2120d7906cd9284059bc4c'
EXPORT_SHA = 'e23c8f7cdf6faa895a8bde34a41806bf2022e630f5361529d0271374742d740e'
ORIGINAL_CHECK_SHA = '48a7d5034b9eea0afa139e4a69cdb5b393c694dcec2b8e874a1c1f380f5f42b1'
CAPTURE_SHAS = ['cc2dd7fbddef8ee12de00b7a5cc88a16ba9d531783ab09ae8e66abe50df43408',
    '34ee3b510014f55c34525c6a96a3527603ba6350c896fae7498ae5493987df69',
    '5560b34755bfbca54cc179c119991792b63d6135e839ad15ac8db95e16965660']
CHECK_SHAS = ['70caf6de746831cf7950fb4971359a1341d2dd4d845f3f3eb5a1da0e1b32072c',
    'b975a0eb82fa3d2ce275a8e487aed583e37bfc16bfb2afc5b6e42f7fc0402618',
    '17f9aec6d7c93189c7ba74c33c3101160c4170f0a38e2b6ae91d77eeb4fdc250']
FLOAT_CONFIG = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
    mean_values=None, std_values=None, quant_img_RGB2BGR=False, model_pruning=False,
    remove_reshape=False, quantize_weight=False, enable_flash_attention=False)
QUANT_CONFIG = dict(FLOAT_CONFIG, quantized_dtype='w8a8', quantized_algorithm='normal', quantized_method='channel')
FLOAT_BUILD = dict(do_quantization=False, dataset=None, auto_hybrid=False)
FRAME, GLOBAL = [2, 1041, 1024], [1, 2082, 1024]
NAMES = ['frame_output', 'global_output']; EVALUATION_IDS = ['pair4', 'pair5']
SCHEDULE = ['originalFP16_A', 'mixedA', 'mixedF', 'mixedB', 'mixedB', 'mixedF', 'mixedA', 'originalFP16_A']
GATE = dict(relative_l2_max=1e-5, max_abs_max=1e-3, rtol=1e-4, atol=1e-5)
TRIAGE_SHA = '55c54aa9f1d1890ac2033f37bf6bb706cde11a781141c508c829cd4e527ac6ff'
SUCCESS = 'completed_basis_pair_manual_hybrid_component_diagnostic_not_accepted'


def require(ok, message):
    if not ok: raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''): h.update(block)
    return h.hexdigest()


def logical_context(parent):
    """Only broaden the frozen descriptor guard; inherit its exact owned IO."""
    class MixedContext(parent):
        def _descriptors(self, command, shapes, names):
            b = self.base
            rows = [b.attr_record(b.query(self.api, self.context, command, b.TensorAttr(index=i))) for i in range(len(shapes))]
            require([r['index'] for r in rows] == list(range(len(shapes))) and names is not None and
                len(names) == len(rows) and len(set(names)) == len(names) and {r['name'] for r in rows} == set(names), 'Logical pair IO names/indices differ')
            rows = [next(r for r in rows if r['name'] == name) for name in names]
            for row, shape in zip(rows, shapes):
                identity = row['type'] == 1 and row['quantization_type'] == 2 and row['zero_point'] == 0 and row['scale'] == 1.
                floating = row['type'] in (0, 1) and (row['quantization_type'] == 0 or identity)
                int8 = row['type'] == 2 and row['quantization_type'] == 2 and -128 <= row['zero_point'] <= 127 and row['scale'] > 0
                require((floating or int8) and math.isfinite(row['scale']) and row['shape'] == list(shape) and
                    row['n_elems'] == math.prod(shape) and row['size'] == math.prod(shape) * {0: 4, 1: 2, 2: 1}.get(row['type'], 0) and
                    row['format'] in (0, 3) and row['pass_through'] == 0, 'Unsupported floating/signed-affine INT8 logical descriptor: ' + str(row))
                row['logical_endpoint_is_floating'] = floating
                row['caller_API'] = 'FP32/pass0 input; want_float1 owned output; automatic SDK conversions'
            return rows
    return MixedContext


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for variant in ('A', 'F', 'B'):
        p.add_argument('--compile-' + variant, type=Path, required=True)
        p.add_argument('--compile-' + variant + '-sha256', required=True)
    p.add_argument('--step2-source', type=Path, required=True)
    p.add_argument('--step2-source-sha256', required=True)
    p.add_argument('--baseline-compile', type=Path, default=Path('/dev/shm/edge-vggt-v127-compile-A-r1/result.json'))
    p.add_argument('--heldout-check-receipt', type=Path, required=True)
    p.add_argument('--heldout-capture-receipt', type=Path)
    p.add_argument('--triage-policy', type=Path, required=True)
    p.add_argument('--export-receipt', type=Path, default=Path('/tmp/edge-vggt-v126-basis-export-r2/result.json'))
    p.add_argument('--owned-runtime', type=Path, default=E / 'vggt-v100-source-r1/native_heads_runtime.py')
    p.add_argument('--runtime-helper', type=Path, default=E / 'vggt-v99-source-r1/run_native_block_r1.py')
    p.add_argument('--abi-source', type=Path, default=B / 'src/deploy/rk3588/validation/run-v12-rknn-block-board.py')
    p.add_argument('--runtime-library', type=Path, default=Path('/usr/lib/librknnrt.so'))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--cpu-affinity', type=int, nargs='+', default=[4, 6])
    p.add_argument('--expected-npu-hz', type=int, choices=[300000000, 1000000000], default=1000000000)
    a = p.parse_args(); sys.dont_write_bytecode = True
    require(platform.machine() == 'aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(), 'RK3588S-only SDK runtime; no host execution')
    faulthandler.enable()
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve(); watched = {source: sha(source)}
    old_env, old_affinity = dict(os.environ), set(os.sched_getaffinity(0))
    handlers, owners, inputs = {}, [], {}; began = time.monotonic()
    report = dict(schema='v128_original_native518_basis_pair_manual_hybrid_runtime', status='running', script_sha256=watched[source],
        step2_source_sha256=a.step2_source_sha256, schedule=SCHEDULE, heldout_order=EVALUATION_IDS,
        core_mask=1, init_flags=0, maximum_live_contexts=1, candidate_accepted=False,
        component_only=True, original_frame0_and_global0_complete=True, original_layers_tokens_and_resolution_preserved=True,
        requested_policy_only=True, protected_precision_placement_established=False, all_INT8_or_all_NPU_placement_verified=False,
        performance_quality_trajectory_realtime_or_8GB_accepted=False, system_or_frequency_writes=False,
        manual_quantization=False, native_packed_IO=False,
        input_API='Logical FP32/pass0 input; want_float1 output; owned copies of both endpoints before release',
        timing_scope='Per-call outer OwnedContext.run includes finite/preparation, SDK input conversion/set/run/get, both owned copies/release/output finite. Phase lifecycle includes model/source/library rehash, init/query, two calls and destroy. Samples, comparison, hashes and saves are outside these timers.',
        precision_scope='Same heldout input versus its original A CPU FP32 graph and own-variant FP32 graph, plus matched original A FP16. Inputs originate from original native DINO FP16; this is not an end-to-end FP32 or trajectory comparison.',
        phases=[], calls=[], platform_samples=[], cleanup_errors=[])

    def progress(stage):
        report.update(stage=stage, total_wall_seconds=time.monotonic() - began, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        temp = out / 'result.tmp'; temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n'); temp.replace(out / 'result.json')
        print(json.dumps(dict(stage=stage, status=report['status'])), flush=True)

    def pin(path, digest, size=None):
        path = Path(path).resolve(strict=True)
        require(isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest) and
            sha(path) == digest and (size is None or path.stat().st_size == size), 'Pinned file differs: ' + str(path))
        require(path not in watched or watched[path] == digest, 'Conflicting file identity'); watched[path] = digest
        return path

    def relative(folder, file):
        value = Path(file); require(not value.is_absolute() and '..' not in value.parts, 'Receipt-relative artifact required')
        return folder / value

    def module(path, digest, name):
        spec = importlib.util.spec_from_file_location(name, pin(path, digest))
        value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value

    def raw(value): return hashlib.sha256(value.tobytes(order='C')).hexdigest()

    def array(folder, row, shape):
        value = np.load(pin(relative(folder, row['file']), row['sha256'], row['bytes']), allow_pickle=False)
        require(value.dtype == np.float32 and list(value.shape) == shape == row['shape'] and row['dtype'] == 'float32' and
            row['tensor_bytes'] == value.nbytes and row['finite'] and value.flags.c_contiguous and np.isfinite(value).all() and raw(value) == row['raw_sha256'], 'Actual canonical FP32 artifact differs')
        value.setflags(write=False); return value

    def save(name, value):
        file = out / name
        with file.open('xb') as stream: np.save(stream, value, allow_pickle=False)
        return dict(file=name, bytes=file.stat().st_size, sha256=sha(file), shape=list(value.shape), dtype=str(value.dtype),
            tensor_bytes=value.nbytes, raw_sha256=raw(value), finite=bool(np.isfinite(value).all()))

    def comparison(actual, reference):
        x, y = actual.astype(np.float64), reference.astype(np.float64); delta = x - y
        an, rn, dn = [float(np.linalg.norm(v)) for v in (x, y, delta)]
        outside = int(np.count_nonzero(np.abs(delta) > 1e-5 + 1e-4 * np.abs(y)))
        return dict(relative_l2=dn / rn if rn else None, l2_error=dn, reference_norm=rn, zero_reference_norm=rn == 0,
            max_abs=float(np.abs(delta).max()), cosine=float(x.ravel() @ y.ravel()) / (an * rn) if an and rn else None,
            outside_count=outside, strict_pass=outside == 0, rtol=1e-4, atol=1e-5,
            strict_is_diagnostic_not_mixed_precision_acceptance=True, byte_equal=actual.tobytes(order='C') == reference.tobytes(order='C'))

    def observe(stage):
        value = H.sample(); report['platform_samples'].append(dict(stage=stage, monotonic=time.monotonic(), **value))
        require(value['npu_hz'] == a.expected_npu_hz and value['temperatures_c'] and max(value['temperatures_c'].values()) < 75, 'External frequency/thermal guard differs')

    def interrupted(number, frame): raise InterruptedError('External bounded controller signal ' + str(number))

    progress('preflight')
    try:
        for number in (signal.SIGTERM, signal.SIGINT): handlers[number] = signal.signal(number, interrupted)
        require(a.cpu_affinity and set(a.cpu_affinity) <= old_affinity & {4, 5, 6, 7}, 'A76 affinity unavailable')
        os.sched_setaffinity(0, set(a.cpu_affinity))
        os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        triage_path = pin(a.triage_policy, TRIAGE_SHA); triage = json.loads(triage_path.read_text())
        require(triage['schema'] == 'v128_pre_runtime_component_triage' and triage['status'] == 'declared_before_any_R128_candidate_build_or_runtime' and
            triage['heldout_inputs'] == EVALUATION_IDS and triage['no_optimization_on_heldouts'], 'Predeclared post-run descriptive screen required')
        report['pre_runtime_triage_policy'] = dict(file=str(triage_path), sha256=TRIAGE_SHA, policy=triage)
        ep, cp = pin(a.export_receipt, EXPORT_SHA), pin(a.heldout_check_receipt, CHECK_SHAS[2])
        e, c = json.loads(ep.read_text()), json.loads(cp.read_text())
        capture_path = pin(a.heldout_capture_receipt or c['capture_receipt']['file'], CAPTURE_SHAS[2]); cap = json.loads(capture_path.read_text())
        require(e['schema'] == 'v126_original_native518_basis_pair_export' and e['status'] == 'exported_original_basis_pair_not_numerically_accepted' and
            e['source_and_inputs_unchanged'] and not e['cleanup_errors'] and cap['schema'] == 'v128_original_native518_frame0_input_capture' and
            cap['status'] == 'completed_original_native518_frame0_input_subbatch_capture' and cap['script_sha256'] == CAPTURE_SOURCE_SHA and cap['batch_index'] == 2 and
            all(cap[k] for k in ('all16_SDK_calls_completed', 'all_inputs_before_frame0', 'all_NPU_contexts_closed', 'source_and_inputs_unchanged', 'model_state_unchanged',
                'checkpoint_final_SHA_verified', 'affinity_restored', 'environment_restored', 'sys_path_restored')) and not cap['cleanup_errors'], 'Original clean heldout capture required')
        require(c['schema'] == 'v128_original_native518_captured_pair_FP32_check' and c['status'] == 'checked_captured_pair_fixed_FP32_gate_passed' and
            c['script_sha256'] == CHECKER_SHA and c['capture_receipt']['sha256'] == CAPTURE_SHAS[2] and c['capture_receipt']['script_sha256'] == CAPTURE_SOURCE_SHA and
            c['records'] == cap['records'] and c['models'] == e['models'] and c['source_hashes'] == cap['source_hashes'] == e['source_hashes'] and
            c['original_parameters'] == e['original_parameters'] and c['checkpoint'] == cap['checkpoint'] == e['checkpoint'] and
            c['export_receipt']['sha256'] == EXPORT_SHA and c['original_check_receipt']['sha256'] == ORIGINAL_CHECK_SHA and c['fixed_gate'] == GATE and
            c['strict_allclose_is_diagnostic'] and c['FP32_gate_passed'] and c['all6_calls_completed'] and c['saved_array_count'] == 12 and
            c['ORT_sessions_dropped'] and c['sessions_created'] == c['sessions_dropped'] == 6 and c['source_and_inputs_unchanged'] and
            c['affinity_restored'] and c['environment_restored'] and not c['cleanup_errors'], 'Matched same-input original A/F/B FP32 gate required')
        require([r['id'] for r in cap['records']] == EVALUATION_IDS and [r['split'] for r in cap['records']] == ['heldout'] * 2 and
            [r['source_pair_indices'] for r in cap['records']] == [[96, 104], [112, 120]] and
            [(r['input_id'], r['variant']) for r in c['checks']] == [(case, v) for case in EVALUATION_IDS for v in ('A', 'F', 'B')], 'Two new heldout pairs/all variants required')
        import numpy as np
        inputs = {row['id']: array(capture_path.parent, row['input'], FRAME) for row in cap['records']}
        require(len({raw(x) for x in inputs.values()}) == 2, 'Heldout inputs must be distinct')
        for record in cap['records']:
            require(record['input_is_before_original_frame0'] and record['actual_four_DINO_groups_executed'] and record['original_final_norm_and_role_concat_executed'], 'True frame0 entry boundary required')
            for image in record['source_images']: pin(image['file'], image['sha256'], image['bytes'])
        references = {}
        for row in c['checks']:
            require(row['complete'] and row['input_unchanged'] and row['fixed_gate_passed'] and len(row['outputs']) == 2 and
                [v['name'] for v in row['outputs']] == NAMES, 'Two complete canonical FP32 endpoints required')
            require(set(row['comparisons_vs_A']) == set(NAMES) and all(m['fixed_gate_passed'] and math.isfinite(m['relative_l2']) and math.isfinite(m['max_abs']) and
                m['relative_l2'] <= GATE['relative_l2_max'] and m['max_abs'] <= GATE['max_abs_max'] for m in row['comparisons_vs_A'].values()), 'Unchanged numeric FP32 gate failed')
            references[(row['input_id'], row['variant'])] = [array(cp.parent, artifact, shape) for artifact, shape in zip(row['outputs'], (FRAME, GLOBAL))]
        require(shutil.disk_usage(out).free >= 32 * 8528000 + (128 << 20), 'Insufficient space for all32 output NPYs and receipt reserve')
        baseline_path = pin(a.baseline_compile, FP16_COMPILE_SHA); baseline = json.loads(baseline_path.read_text()); graph_a = next(m for m in e['models'] if m['key'] == 'A')
        require(baseline['schema'] == 'v127_original_native518_basis_pair_FP16_compile' and baseline['status'] == 'compiled_not_runtime_accepted' and baseline['variant'] == 'A' and
            baseline['script_sha256'] == FP16_COMPILER_SHA and baseline['configuration'] == FLOAT_CONFIG and baseline['build_options'] == FLOAT_BUILD and
            baseline['export_receipt']['sha256'] == EXPORT_SHA and baseline['check_receipt']['sha256'] == ORIGINAL_CHECK_SHA and
            baseline['onnx']['sha256'] == graph_a['onnx']['sha256'] and baseline['onnx']['bytes'] == graph_a['onnx']['bytes'] and
            baseline['input_bindings'] == graph_a['input_bindings'] and baseline['output_bindings'] == graph_a['output_bindings'] and
            baseline['rknn']['sha256'] == FP16_MODEL_SHA and baseline['rknn']['bytes'] == 66378284 and baseline['all_operations_succeeded'] and
            baseline['sdk_released'] and baseline['sdk_release_code'] in (0, None) and not baseline['cleanup_errors'] and
            all(baseline[k] for k in ('source_and_inputs_unchanged', 'affinity_restored', 'environment_restored', 'cwd_restored')), 'Frozen original A FP16 compile/model lineage differs')
        row = baseline['rknn']; models = {'originalFP16_A': dict(row, file=str(pin(relative(baseline_path.parent, row['file']), row['sha256'], row['bytes'])))}
        compiles = {'originalFP16_A': dict(file=str(baseline_path), sha256=FP16_COMPILE_SHA)}
        step2_source = pin(a.step2_source, a.step2_source_sha256)
        for variant in ('A', 'F', 'B'):
            path = pin(getattr(a, 'compile_' + variant), getattr(a, 'compile_' + variant + '_sha256')); receipt = json.loads(path.read_text())
            graph = next(m for m in e['models'] if m['key'] == variant)
            require(receipt['schema'] == 'v128_original_native518_basis_pair_manual_hybrid_step2' and receipt['status'] == 'compiled_manual_hybrid_pair_not_runtime_or_quality_accepted' and
                receipt['variant'] == variant and receipt['script_sha256'] == a.step2_source_sha256 and receipt['configuration'] == QUANT_CONFIG and
                receipt['graph']['sha256'] == graph['onnx']['sha256'] and receipt['graph']['bytes'] == graph['onnx']['bytes'] and
                receipt['input_bindings'] == graph['input_bindings'] and receipt['output_bindings'] == graph['output_bindings'] and
                receipt['checkpoint'] == e['checkpoint'] and receipt['original_parameters'] == e['original_parameters'] and
                receipt['step1_receipt']['script_sha256'] == STEP1_SOURCE_SHA and receipt['requested_policy_only'] is True and
                receipt['protected_precision_placement_established'] is False and receipt['all_SDK_operations_succeeded'] and receipt['sdk_release_called'] and
                receipt['sdk_released'] and receipt['sdk_release_code'] in (0, None) and not receipt['cleanup_errors'] and
                all(receipt[k] for k in ('source_and_inputs_unchanged', 'affinity_restored', 'environment_restored', 'cwd_restored')), 'Actual clean manual-hybrid step2 lineage differs')
            rows = receipt['three_clean_capture_check_receipts']
            require(len(rows) == 3 and [r['batch_index'] for r in rows] == [0, 1, 2] and
                [r['capture']['sha256'] for r in rows] == CAPTURE_SHAS and [r['check']['sha256'] for r in rows] == CHECK_SHAS, 'All six captured inputs/new18 FP32 gates required')
            s = receipt['step1_receipt']; sfile = pin(s['file'], s['sha256']); step1 = json.loads(sfile.read_text())
            require(step1['schema'] == 'v128_original_native518_basis_pair_manual_hybrid_step1' and step1['status'] == 'generated_manual_hybrid_step1_not_compiled_or_accepted' and
                step1['script_sha256'] == STEP1_SOURCE_SHA and step1['variant'] == variant and step1['configuration'] == QUANT_CONFIG and
                step1['graph']['sha256'] == graph['onnx']['sha256'] and step1['three_clean_capture_check_receipts'] == rows and
                [r['id'] for r in step1['calibration_records']] == ['pair0', 'pair1', 'pair2', 'pair3'] and
                [{k: r[k] for k in ('id', 'split', 'input')} for r in step1['heldout_records']] == [{k: r[k] for k in ('id', 'split', 'input')} for r in cap['records']] and
                step1['heldouts_excluded_from_dataset'] and step1['calibration_evaluation_disjoint'] and step1['dataset']['line_count'] == 4 and
                step1['dataset']['canonical_unrotated_inputs'] and step1['sdk_released'] and step1['all_SDK_operations_succeeded'] and not step1['cleanup_errors'], 'Four calibration/two heldout step1 contract differs')
            policy = receipt['selector_policy']; pin(policy['file'], policy['sha256'])
            row = receipt['rknn']; key = 'mixed' + variant
            models[key] = dict(row, file=str(pin(relative(path.parent, row['file']), row['sha256'], row['bytes'])))
            compiles[key] = dict(file=str(path), sha256=getattr(a, 'compile_' + variant + '_sha256'), step1_receipt=s, selector_policy=policy,
                generated_inputs=receipt['generated_inputs'], requested_policy_only=True)
        heads = module(a.owned_runtime, HEADS_SHA, '_v128_frozen_owned_IO'); H = module(a.runtime_helper, HELPER_SHA, '_v128_frozen_platform_IO')
        IO = logical_context(heads.OwnedContext); abi, library = pin(a.abi_source, ABI_SHA), pin(a.runtime_library, SDK_SHA)
        report.update(export_receipt=dict(file=str(ep), sha256=EXPORT_SHA), heldout_check_receipt=dict(file=str(cp), sha256=CHECK_SHAS[2]),
            heldout_capture_receipt=dict(file=str(capture_path), sha256=CAPTURE_SHAS[2]), step2_source=dict(file=str(step2_source), sha256=a.step2_source_sha256),
            checkpoint=e['checkpoint'], original_parameters=e['original_parameters'], original_source_hashes=e['source_hashes'], records=cap['records'],
            models=models, compile_receipts=compiles, calibration_ids=['pair0', 'pair1', 'pair2', 'pair3'], evaluation_ids=EVALUATION_IDS,
            mixed_configuration=QUANT_CONFIG, original_FP16_configuration=FLOAT_CONFIG,
            expected_resource_counts=dict(contexts_created=8, contexts_destroyed=8, runs_completed=16, output_buffers_acquired=32, output_buffers_released=32))
        baseline_a, first_hashes = {}, {}
        for phase, route in enumerate(SCHEDULE):
            observe('before_phase_' + str(phase)); phase_row = dict(index=phase, route=route, status='running'); report['phases'].append(phase_row)
            progress('phase_' + str(phase) + '_init'); owner = None; payloads = []; lifecycle_tick = time.perf_counter()
            try:
                try:
                    owner = IO(models[route]['file'], models[route]['sha256'], input_shapes=[FRAME], output_shapes=[FRAME, GLOBAL],
                        input_names=['input'], output_names=NAMES, core_mask=1, flags=0, abi_source=abi, library=library, runtime_helper=a.runtime_helper)
                except BaseException as error:
                    owner = getattr(error, 'owned_context', None)
                    if owner is not None: owners.append(owner)
                    raise
                owners.append(owner); phase_row['init_seconds'] = time.perf_counter() - lifecycle_tick
                descriptors = copy.deepcopy(owner.metrics()); endpoints = descriptors['input_descriptors'] + descriptors['output_descriptors']
                phase_row.update(input_descriptors=descriptors['input_descriptors'], output_descriptors=descriptors['output_descriptors'],
                    all_logical_endpoints_observed_floating=all(r['logical_endpoint_is_floating'] for r in endpoints),
                    floating_endpoint_policy_mismatch=not all(r['logical_endpoint_is_floating'] for r in endpoints),
                    descriptor_precision_does_not_establish_internal_policy=True)
                for case_id, x in inputs.items():
                    tick = time.perf_counter(); ys = owner.run([x]); elapsed = time.perf_counter() - tick
                    call = dict(index=len(report['calls']), phase=phase, route=route, input_id=case_id, elapsed_seconds=elapsed,
                        SDK_stages=copy.deepcopy(owner.last_timings), status='both_owned_outputs_copied_and_released')
                    report['calls'].append(call); payloads.append((call, ys))
            finally:
                if owner is not None:
                    close_tick = time.perf_counter()
                    try: owner.close()
                    except BaseException as error: report['cleanup_errors'].append('Phase close: ' + str(error)); raise
                    finally: phase_row.update(destroy_seconds=time.perf_counter() - close_tick,
                        lifecycle_seconds_excluding_save_compare=time.perf_counter() - lifecycle_tick, owner_metrics=owner.metrics())
            phase_row['status'] = 'closed_before_next_context'; observe('after_phase_' + str(phase)); progress('phase_' + str(phase) + '_closed')
            for call, ys in payloads:
                case_id = call['input_id']; variant = 'A' if route == 'originalFP16_A' else route[-1]
                own_ref, ref = references[(case_id, variant)], references[(case_id, 'A')]
                if phase == 0: baseline_a[case_id] = ys
                output_rows, comparisons = [], {}
                for i, (name, y) in enumerate(zip(NAMES, ys)):
                    output_rows.append(dict(save('call%02d-%s.npy' % (call['index'], name), y), name=name))
                    comparisons[name + '_vs_original_A_CPU_FP32'] = comparison(y, ref[i])
                    comparisons[name + '_vs_own_variant_CPU_FP32'] = comparison(y, own_ref[i])
                    comparisons[name + '_vs_first_original_A_FP16'] = comparison(y, baseline_a[case_id][i])
                    key = (route, case_id, name); digest = raw(y)
                    if key not in first_hashes: first_hashes[key] = digest
                    output_rows[-1]['repeat_raw_sha256_equal'] = digest == first_hashes[key]
                x = inputs[case_id]
                call.update(outputs=output_rows, comparisons=comparisons, local_residuals=dict(
                    frame_update_vs_A_CPU_FP32=comparison(ys[0] - x, ref[0] - x),
                    global_update_vs_A_CPU_FP32=comparison(ys[1] - ys[0].reshape(GLOBAL), ref[1] - ref[0].reshape(GLOBAL)),
                    complete_pair_update_vs_A_CPU_FP32=comparison(ys[1] - x.reshape(GLOBAL), ref[1] - x.reshape(GLOBAL))),
                    residual_scope='FP32 canonical subtraction before FP64 norms; global update uses each route\'s own frame endpoint. No denominator floor; zero reference norm is unavailable. Component diagnostic, not trajectory.',
                    repeat_raw_sha256_equal=all(r['repeat_raw_sha256_equal'] for r in output_rows))
            require(all(raw(inputs[r['id']]) == r['input']['raw_sha256'] for r in cap['records']), 'Immutable actual frame0 inputs changed')
            progress('phase_' + str(phase) + '_saved')
        report.update(status=SUCCESS, all16_calls_completed=True, saved_array_count=32,
            all_output_repetitions_byte_equal=all(r['repeat_raw_sha256_equal'] for r in report['calls']),
            timing_summary={route: dict(call_count=4, all_call_seconds=[r['elapsed_seconds'] for r in report['calls'] if r['route'] == route],
                all_call_mean_seconds=sum(r['elapsed_seconds'] for r in report['calls'] if r['route'] == route) / 4,
                cold_phase_lifecycle_seconds=[r['lifecycle_seconds_excluding_save_compare'] for r in report['phases'] if r['route'] == route]) for route in dict.fromkeys(SCHEDULE)},
            timing_limit='Two fresh contexts per route, two heldout inputs each; all calls visible. No persistent-stream interval, P95, whole-model or trajectory claim.')
        screens = {}; base_mean = report['timing_summary']['originalFP16_A']['all_call_mean_seconds']
        for route in ('mixedA', 'mixedF', 'mixedB'):
            fidelity = []
            for case_id in EVALUATION_IDS:
                baseline_calls = [r for r in report['calls'] if r['route'] == 'originalFP16_A' and r['input_id'] == case_id]
                candidate_calls = [r for r in report['calls'] if r['route'] == route and r['input_id'] == case_id]
                for name in NAMES:
                    key = name + '_vs_original_A_CPU_FP32'; base_error = baseline_calls[0]['comparisons'][key]['relative_l2']
                    errors = [r['comparisons'][key]['relative_l2'] for r in candidate_calls]
                    available = base_error is not None and all(v is not None for v in errors)
                    fidelity.append(dict(input_id=case_id, endpoint=name, baseline_relative_l2_all_calls=[r['comparisons'][key]['relative_l2'] for r in baseline_calls],
                        baseline_reference='First matched originalFP16_A call; both baseline calls retained', candidate_relative_l2_all_calls=errors,
                        allowable_relative_l2=1.2 * base_error if base_error is not None else None,
                        candidate_to_baseline_ratios=[v / base_error if v is not None and base_error else None for v in errors],
                        zero_baseline_error=base_error == 0, screen_available=available,
                        screen_passed=all(v <= 1.2 * base_error for v in errors) if available else None))
            mean = report['timing_summary'][route]['all_call_mean_seconds']; reduction = 1 - mean / base_mean if base_mean else None
            screens[route] = dict(fidelity=fidelity, component_fidelity_screen_passed=all(r['screen_passed'] is True for r in fidelity),
                original_FP16_mean_seconds=base_mean, candidate_mean_seconds=mean, mean_time_reduction_fraction=reduction,
                component_speed_screen_passed=reduction >= .1 if reduction is not None else None,
                protected_precision_placement_established=False, candidate_accepted=False,
                scope='Post-run declared component screen only; no output suppression or new FP32 gate. Internal policy remains unverified; no whole-model expansion/quality/realtime acceptance from this report.')
        report['post_run_component_screens'] = screens
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc()); traceback.print_exc()
    finally:
        for owner in owners:
            if not owner.closed:
                try: owner.close()
                except BaseException as error: report['cleanup_errors'].append('Final owner close: ' + str(error))
        metrics = [owner.metrics() for owner in owners]; keys = ('contexts_created', 'contexts_destroyed', 'runs_completed', 'output_buffers_acquired', 'output_buffers_released')
        report['owner_metrics'] = metrics; report['resource_totals'] = {k: sum(r[k] for r in metrics) for k in keys}
        report['all_contexts_and_pending_outputs_closed'] = all(r['closed'] and r['live_contexts'] == r['live_output_buffers'] == 0 and r['resource_counts_balanced'] for r in metrics)
        if report['status'] == SUCCESS and (report['resource_totals'] != report['expected_resource_counts'] or not report['all_contexts_and_pending_outputs_closed'] or any(r['cleanup_errors'] for r in metrics)):
            report['cleanup_errors'].append('Actual SDK resources incomplete/unbalanced')
        try: os.sched_setaffinity(0, old_affinity)
        except BaseException as error: report['cleanup_errors'].append('Affinity restore: ' + str(error))
        os.environ.clear(); os.environ.update(old_env)
        for number, handler in handlers.items(): signal.signal(number, handler)
        try: report['source_and_inputs_unchanged'] = all(path.is_file() and sha(path) == digest for path, digest in watched.items())
        except BaseException as error: report['source_and_inputs_unchanged'] = False; report['cleanup_errors'].append('Identity: ' + str(error))
        report.update(affinity_restored=set(os.sched_getaffinity(0)) == old_affinity, environment_restored=dict(os.environ) == old_env,
            input_and_source_SHA_pins={str(path): digest for path, digest in watched.items()})
        if report['cleanup_errors'] or not report['all_contexts_and_pending_outputs_closed'] or not all(report[k] for k in ('affinity_restored', 'environment_restored', 'source_and_inputs_unchanged')):
            report['status'] = 'failed_cleanup_or_identity'
        progress('finished')
    return 0 if report['status'] == SUCCESS else 1


if __name__ == '__main__':
    raise SystemExit(main())
