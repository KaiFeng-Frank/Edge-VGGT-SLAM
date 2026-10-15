#!/usr/bin/env python3
"""Saved-only NumPy audit of B full-attention calibration flags0/8.

No NN, SDK, board, heldout array or model deserialization is invoked. Compare
the same real pair0 against original A and own B FP32 conditional references;
profile durations are diagnostic SDK fields, not normal performance results.
"""
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
import re
import sys

import numpy as np

W = Path(__file__).resolve().parent
PINS = {
    'B-attention-calibration-r1/result.json': '202bf7c1b46b34c901ac78f973bd11807a53de7ec0d7b38ef49f52ce6f8d0ece',
    'B-attention-calibration-control-r1/control.json': '84ac8c1c2a6ef6609c03a4215488287f934dd2c7002a3f50ba39d79f50130a81',
    'B-attention-step2-r1/result.json': '3c952601fa1cad925c37c1dd4a9a5edb169dea2ab6356ab5abd8c91fa5004317',
    'calibration-profile-r2/result.json': '4013651f463a39475560b7c2e0544b3ce0fb813ece5a73e343cae9299e84d91b',
    'fp32-batch0-r1/result.json': '70caf6de746831cf7950fb4971359a1341d2dd4d845f3f3eb5a1da0e1b32072c',
    'B-attention-compiled-placement.json': 'b6bdd072e032df0fe8e0e3e538200e854f42117142b54aee86678e8b84db2a81',
    'executed-B-attention-calibration-source-r1/source-manifest.json': 'd5167efab3b38c8747dafdf18b169bfd640a8f00d80ee99d9e81c1639333dbc9',
}
HELPER_SHA = '4270f2ce4ec741ee6e1a4c3dd6f24524c2eb0d3f3550ec007de4e767130f4552'
WORKER_SHA = '0e2c56cbcbea97c360051e1c420f4ab912b82c9e94ce74a6a63020b1dacbae69'
EXPECTED = dict(contexts_created=2, contexts_destroyed=2, runs_completed=2,
    output_buffers_acquired=4, output_buffers_released=4)
FRAME, GLOBAL = [2, 1041, 1024], [1, 2082, 1024]
NAMES = ['frame_output', 'global_output']
GATE = dict(relative_l2_max=1e-5, max_abs_max=1e-3, rtol=1e-4, atol=1e-5)


def require(ok, message):
    if not ok: raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(4 << 20), b''): h.update(b)
    return h.hexdigest()


def compare(actual, reference):
    require(actual.shape == reference.shape, 'Comparison shapes differ')
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        return dict(finite=False, relative_l2=None, max_abs=None, strict_pass=False, fixed_gate_passed=False)
    x, y = actual.astype(np.float64), reference.astype(np.float64)
    d = x - y; norm = float(np.sqrt(np.sum(y*y))); error = float(np.sqrt(np.sum(d*d)))
    relative = error/norm if norm else (0.0 if error == 0 else None)
    maximum = float(np.max(np.abs(d))); outside = int(np.count_nonzero(np.abs(d) > GATE['atol'] + GATE['rtol'] * np.abs(y)))
    return dict(finite=True, relative_l2=relative, reference_norm=norm, zero_reference_norm=norm == 0,
        max_abs=maximum, strict_outside_count=outside, strict_pass=outside == 0,
        fixed_gate_passed=relative is not None and relative <= GATE['relative_l2_max'] and maximum <= GATE['max_abs_max'],
        raw_byte_equal=actual.tobytes() == reference.tobytes())


def main():
    sys.dont_write_bytecode = True
    helper = W / 'audit_calibration_profile_r2_saved.py'; require(sha(helper) == HELPER_SHA, 'Frozen saved-audit helper differs')
    spec = importlib.util.spec_from_file_location('_v128_frozen_saved_helpers', helper)
    H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
    docs = {}
    for name, digest in PINS.items():
        require(sha(W / name) == digest, 'Receipt/metadata identity differs: ' + name)
        docs[name] = json.loads((W / name).read_text())
    r = docs['B-attention-calibration-r1/result.json']; c = docs['B-attention-calibration-control-r1/control.json']
    compiled = docs['B-attention-step2-r1/result.json']; prior = docs['calibration-profile-r2/result.json']
    ref = docs['fp32-batch0-r1/result.json']; placement = docs['B-attention-compiled-placement.json']
    require(r['schema'] == 'v128_original_B_attention_FP16_calibration_flags8_diagnostic' and
        r['status'] == 'completed_B_attention_calibration_diagnostic_not_performance_or_quality_accepted' and r['script_sha256'] == WORKER_SHA,
        'Executed worker identity differs')
    require(r['input_id'] == prior['input_id'] == 'pair0' and r['source_pair_indices'] == prior['source_pair_indices'] == [0, 8] and
        r['input_split'] == 'calibration' and not r['heldout_arrays_accessed'] and r['baseline_NPU_calls'] == 0 and
        r['input'] == prior['input'] and r['input'] == ref['records'][0]['input'], 'Same calibration-only input lineage required')
    require(r['compile_receipt']['sha256'] == PINS['B-attention-step2-r1/result.json'] and
        r['profile_r2_receipt']['sha256'] == PINS['calibration-profile-r2/result.json'] and
        r['graph'] == compiled['graph'] and r['model'] == compiled['rknn'] and
        r['original_parameters'] == compiled['original_parameters'] == ref['original_parameters'] and
        r['checkpoint'] == compiled['checkpoint'] == ref['checkpoint'], 'Original graph/parameters/model lineage differs')
    H.relative(W / 'B-attention-step2-r1', compiled['rknn'])
    manifest = docs['executed-B-attention-calibration-source-r1/source-manifest.json']
    for item in manifest['files']: H.relative(W / 'executed-B-attention-calibration-source-r1', item)
    require(ref['FP32_gate_passed'] and ref['all6_calls_completed'] and ref['fixed_gate'] == GATE and
        all(ref[k] for k in ('source_and_inputs_unchanged', 'affinity_restored', 'environment_restored', 'ORT_sessions_dropped')) and
        not ref['cleanup_errors'], 'Clean unchanged conditional FP32 gates required')
    require(r['all_calls_completed'] and r['all_resources_closed'] and r['resource_totals'] == EXPECTED and
        r['saved_array_count'] == 4 and not r['cleanup_errors'] and
        all(r[k] for k in ('source_and_inputs_unchanged', 'environment_restored', 'affinity_restored')) and
        not any(r[k] for k in ('candidate_accepted', 'numerical_acceptance', 'nonfinite_values_rewritten', 'production_finite_guard_modified')),
        'Worker ownership/cleanup/acceptance scope differs')
    require(c['child_returncode'] == 0 and c['frequency_governor_limits_restored'] and not c['restore_errors'] and
        c['requested_frequency_hz'] == 1000000000 and c['timeout_seconds'] == 180 and c['temperature_limit_c'] == 75 and
        c['restored'] == c['baseline'], 'Restoring controller differs')
    maximum_temp = max(t for observation in c['observations'] for t in observation['temperatures_c'].values())
    require(maximum_temp < 75 and all(o['actual_frequency_hz'] == 1000000000 for o in c['observations']), 'Observed clock/thermal scope differs')
    arrays, identities = {}, []

    def load(folder, row, shape, label):
        raw, info = H.read_array(folder, row, shape); info['role'] = label; identities.append(info)
        return np.frombuffer(raw, dtype=np.float32).reshape(shape)

    calls = []
    require([x['init_flags'] for x in r['calls']] == [0, 8], 'Exact flags0/8 calls required')
    for call in r['calls']:
        require([x['name'] for x in call['outputs']] == NAMES, 'Original output ordering differs')
        flags = call['init_flags']
        arrays[flags] = {n: load(W / 'B-attention-calibration-r1', row, shape, 'new_flags'+str(flags)+'/'+n)
            for n, row, shape in zip(NAMES, call['outputs'], (FRAME, GLOBAL))}
        finite = all(np.isfinite(x).all() for x in arrays[flags].values())
        require(finite == call['all_outputs_finite'] and call['invalid_numeric'] == (not finite), 'Finite guard accounting differs')
        s = call['owner_metrics']
        require(s['closed'] and s['resource_counts_balanced'] and s['live_output_buffers'] == 0 and not s['cleanup_errors'] and
            s['contexts_created'] == s['contexts_destroyed'] == s['runs_completed'] == 1 and
            s['output_buffers_acquired'] == s['output_buffers_released'] == 2, 'Individual owner is incomplete')
        calls.append(dict(init_flags=flags, finite=finite, logical_descriptors=H.descriptors(call),
            SDK_model_memory_bytes=call['SDK_model_memory_bytes']))
    parity = all(arrays[0][n].tobytes() == arrays[8][n].tobytes() for n in NAMES)
    require(parity and r['attribution_byte_parity_passed'] and r['calls'][1]['all_outputs_byte_equal_to_flags0'], 'Raw byte parity differs')
    require(calls[0]['logical_descriptors'] == calls[1]['logical_descriptors'], 'Normal/profile descriptors differ')
    references = {}
    for variant in ('A', 'B'):
        rows = [x for x in ref['checks'] if x['input_id'] == 'pair0' and x['variant'] == variant]
        require(len(rows) == 1 and rows[0]['split'] == 'calibration' and rows[0]['source_pair_indices'] == [0, 8], 'Exact same-input reference required')
        references[variant] = {n: load(W / 'fp32-batch0-r1', row, shape, 'FP32_'+variant+'/'+n)
            for n, row, shape in zip(NAMES, rows[0]['outputs'], (FRAME, GLOBAL))}
    routes = {'new_B_full_attention_FP16': arrays[0]}
    for route in ('mixedB', 'originalFP16_A'):
        row = next(x for x in prior['calls'] if x['route'] == route and x['init_flags'] == 0)
        routes[route] = {n: load(W / 'calibration-profile-r2', item, shape, 'prior_'+route+'/'+n)
            for n, item, shape in zip(NAMES, row['outputs'], (FRAME, GLOBAL))}
    comparisons = {route: {n: {v: compare(outputs[n], refs[n]) for v, refs in references.items()} for n in NAMES}
        for route, outputs in routes.items()}
    profile_path = H.relative(W / 'B-attention-calibration-r1', r['calls'][1]['profile'])
    rows = H.profile_rows(profile_path); lines = profile_path.read_text().splitlines()
    header = next(line for line in lines if line.startswith('ID   OpType'))
    for row in rows:
        line = lines[row['source_line']-1]
        match = re.search(r'\s+\d+/\d+/\d+\s+(\d+)\s+', line); require(match is not None, 'Missing diagnostic operator time')
        row['time_us'] = int(match[1])
        row['input_shape_text'] = line[header.index('InputShape'):header.index('OutputShape')].strip()
        row['output_shape_text'] = line[header.index('OutputShape'):header.index('Cycles(')].strip()
    signature = lambda x: (x['op_type'], x['dtype'], x['target'], x['full_name'])
    require([signature(x) for x in rows] == [(x['op'], x['dtype'], x['target'], x['name']) for x in placement['all_operators']], 'Runtime-profile/compiler placement differs')
    require(len(rows) == 84 and Counter((x['dtype'], x['target']) for x in rows) ==
        Counter({('FLOAT16', 'NPU'): 63, ('INT8', 'NPU'): 17, ('FLOAT16', 'CPU'): 4}), 'Complete actual profile dtype/target scope differs')
    sdpa = [x for x in rows if x['op_type'] == 'exSDPAttention']
    require(len(sdpa) == 3 and [x['output_shape_text'] for x in sdpa] == ['(16,64,1,1041)', '(16,64,1,1041)', '(16,64,1,2082)'] and
        all(x['dtype'] == 'FLOAT16' and x['target'] == 'NPU' for x in sdpa), 'Original full-head/query shapes differ')
    total = int(re.search(r'Total Operator Elapsed Per Frame Time\(us\):\s*(\d+)', profile_path.read_text())[1])
    require(total == sum(x['time_us'] for x in rows), 'Diagnostic operator time accounting differs')
    profile = dict(file_sha256=r['calls'][1]['profile']['sha256'], raw_byte_attribution_passed=parity,
        operator_count=84, operator_sum_us=total, SDK_PERF_RUN_us=r['calls'][1]['SDK_reported_run_duration_us'],
        diagnostic_not_normal_performance=True, dtype_target_counts=placement['dtype_target_counts'],
        SDPA_rows=sdpa, CPU_rows=[x for x in rows if x['target'] == 'CPU'],
        time_by_op_us={k: sum(x['time_us'] for x in rows if x['op_type'] == k) for k in sorted({x['op_type'] for x in rows})})
    new_error = comparisons['new_B_full_attention_FP16']['global_output']['A']['relative_l2']
    old_error = comparisons['mixedB']['global_output']['A']['relative_l2']
    baseline_error = comparisons['originalFP16_A']['global_output']['A']['relative_l2']
    report = dict(schema='v128_saved_B_full_attention_calibration_audit', status='saved_evidence_verified_candidate_rejected',
        script_sha256=sha(__file__), helper_sha256=HELPER_SHA, receipt_pins=PINS, input_id='pair0', input_split='calibration',
        heldout_arrays_accessed=False, NN_SDK_or_board_executed=False, reference_scope='Same-input conditional original A and own B pair FP32 outputs, not complete-model accuracy.',
        fixed_gate=GATE, strict_allclose_is_independent=True, new_payloads_verified=4, all_verified_payloads=len(identities),
        saved_arrays=identities, flags0_8_raw_byte_equal=parity, all_new_outputs_finite=all(x['finite'] for x in calls),
        comparisons=comparisons, calls=calls, actual_flags8_profile=profile, resource_totals=EXPECTED,
        controller=dict(seconds=c['seconds'], sampled_max_temperature_c=maximum_temp, prior=c['baseline'], restored=c['restored'], restoration_verified=True),
        decision=dict(prior_B_global_relative_l2=old_error, new_B_global_relative_l2=new_error,
            original_FP16_A_global_relative_l2=baseline_error, error_reduced_not_fixed=new_error < old_error and new_error > baseline_error,
            original_fixed_and_strict_gates_passed=False, candidate_accepted=False,
            no_normal_performance_or_trajectory_claim=True, no_sole_operator_error_cause_established=True))
    target = W / 'B-attention-calibration-saved-audit.json'; target.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(file=target.name, sha256=sha(target), global_relative_l2=new_error,
        prior_B_global_relative_l2=old_error, raw_byte_parity=parity, candidate_accepted=False)))


if __name__ == '__main__': main()
