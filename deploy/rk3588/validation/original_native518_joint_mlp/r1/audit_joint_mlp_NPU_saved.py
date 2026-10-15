#!/usr/bin/env python3
"""Saved-only independent R129 normal ABBA audit; imports only NumPy/stdlib."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import numpy as np

HELPER_SHA = 'bcc4751d7ea5f036d1b52b401e0d359e631b4e8d14952c664972cee7eaeae6c6'
SOURCE_SHA = '4b9f83cead499cc120cae590bdf48d538a9dae308273dbd65a88b5e73505540e'
RESULT_SHA = 'c6603cef8afcf1b199f892393cb51fd36381b6da023dca273b53be5422090531'
CONTROL_SHA = '6c405ce85dcb6972226f5d497bb4e50b8ce6de7f7db41e4b55e274e20a5ebad3'
COMPILE_SHA = '01f9034d151e0b0388c98b1fbf1ccf81163dace668c4afa13ccce9ecf433388f'
TRIAGE_SHA = '8421935e34be185cfeeecbd066c92b7fd644ee9e42c057d98f3a36e6ceb150a2'
OBSERVATION_SHA = '3c49a48ebd0c356fa0f8f0ebdc256fa166a7b2c638abd6dc28ef6ccdfe467018'
BASE_MODEL_SHA = '803f7e53c5cf574ccb86e1bdd5b9091332f21ec48aef8864fdee52bc72017666'
JOINT_MODEL_SHA = '9401562374b463a40523b01ac6a5aaec4e01f581412eb2e2a2e773702a71c610'
SCHEDULE = ['originalFP16_A', 'joint_MLP', 'joint_MLP', 'originalFP16_A']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workdir', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--output', type=Path)
    a = p.parse_args(); w = a.workdir
    helper = Path(__file__).resolve().with_name('audit_joint_mlp_saved.py')
    def sha(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for piece in iter(lambda: stream.read(4 << 20), b''): h.update(piece)
        return h.hexdigest()
    if sha(helper) != HELPER_SHA: raise RuntimeError('Frozen saved-only numeric helper differs')
    spec = importlib.util.spec_from_file_location('_r129_saved_only_numeric', helper)
    h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
    require = h.require
    r = h.receipt(w / 'runtime-r1/result.json', RESULT_SHA); c = h.receipt(w / 'runtime-control-r1/control.json', CONTROL_SHA)
    triage = h.receipt(w / 'pre-runtime-triage.json', TRIAGE_SHA); compile_row = h.receipt(w / 'compile-r1/result.json', COMPILE_SHA)
    observation = h.receipt(w / 'compiled-schedule-observation.json', OBSERVATION_SHA)
    require(sha(w / 'run_joint_mlp_fp16_board.py') == SOURCE_SHA and r['script_sha256'] == SOURCE_SHA and
        r['schema'] == 'v129_original_native518_joint_MLP_pair_FP16_runtime' and r['status'] == 'completed_original_joint_MLP_pair_FP16_component_not_accepted' and
        r['schedule'] == SCHEDULE and r['core_mask'] == 1 and r['init_flags'] == 0 and
        r['all8_calls_completed'] and r['saved_array_count'] == 16 and not r['candidate_accepted'] and not r['cleanup_errors'], 'Complete pinned normal A/J/J/A runtime required')
    require(triage['schema'] == 'v129_pre_runtime_component_triage' and triage['status'] == 'declared_before_R129_NPU_inference' and triage['heldout_inputs'] == ['pair4', 'pair5'] and
        triage['fidelity']['factor'] == 1.2 and triage['speed']['minimum_mean_time_reduction_fraction'] == .1, 'Predeclared triage changed')
    cap = h.receipt(w.parent / 'capture-batch2-r1/result.json', h.CAP_SHAS[2]); ref = h.receipt(w.parent / 'fp32-batch2-r1/result.json', h.REF_SHAS[2])
    own = h.receipt(w / 'check2-r1/result.json', h.CHECK_SHAS[2])
    require(r['capture_receipt']['sha256'] == h.CAP_SHAS[2] and r['original_FP32_reference_receipt']['sha256'] == h.REF_SHAS[2] and
        r['joint_FP32_reference_receipt']['sha256'] == h.CHECK_SHAS[2] and ref['records'] == cap['records'] and
        [v['id'] for v in cap['records']] == ['pair4', 'pair5'], 'Same-input original and joint FP32 references required')
    references, inputs = {}, {}
    for record in cap['records']:
        case = record['id']; inputs[case] = h.array(w.parent / 'capture-batch2-r1', record['input'], h.SHAPES[0])
        original = next(v for v in ref['checks'] if v['input_id'] == case and v['variant'] == 'A')
        joint = next(v for v in own['checks'] if v['input_id'] == case)
        references[case] = [h.array(w.parent / 'fp32-batch2-r1', v, shape) for v, shape in zip(original['outputs'], h.SHAPES)]
        require(all(h.array(w / 'check2-r1', v, shape).tobytes() == y.tobytes() for v, shape, y in zip(joint['outputs'], h.SHAPES, references[case])), 'Own joint FP32 endpoints differ from original A')
    require([(v['phase'], v['route'], v['input_id']) for v in r['calls']] == [(i, route, case) for i, route in enumerate(SCHEDULE) for case in ('pair4', 'pair5')], 'Exact interleaved schedule required')
    calls, first, actuals, filenames = [], {}, {}, set()
    for row in r['calls']:
        require(len(row['outputs']) == 2 and [v['name'] for v in row['outputs']] == h.NAMES, 'Both canonical outputs required')
        ys, endpoints = [], []
        for artifact, shape, y in zip(row['outputs'], h.SHAPES, references[row['input_id']]):
            metadata = dict(artifact, tensor_bytes=math.prod(shape) * 4)
            x = h.array(w / 'runtime-r1', metadata, shape); ys.append(x); metric = h.compare(x, y)
            for key in ('comparisons_vs_original_A_FP32', 'comparisons_vs_own_graph_FP32'):
                reported = row[key][artifact['name']]
                require(reported['rtol'] == 1e-4 and reported['atol'] == 1e-5 and
                    all(math.isclose(metric[k], reported[k], rel_tol=1e-10, abs_tol=1e-12) if isinstance(metric[k], float) else metric[k] == reported[k]
                        for k in ('relative_l2', 'max_abs', 'strict_pass', 'fixed_numeric_gate_passed', 'byte_equal')), 'Reported FP32-reference metrics differ')
            key = (row['route'], row['input_id'], artifact['name'])
            if key not in first: first[key] = x
            repeated = x.tobytes() == first[key].tobytes()
            require(repeated == artifact['repeat_raw_sha256_equal'], 'Repeat claim differs from actual bytes')
            endpoints.append(dict(artifact=metadata, vs_original_A_CPU_FP32=metric, byte_equal_first_route_repeat=repeated))
            filenames.add(artifact['file'])
        require(row['repeat_raw_sha256_equal'] == all(v['byte_equal_first_route_repeat'] for v in endpoints), 'Call repeat claim differs')
        x = inputs[row['input_id']]; reference = references[row['input_id']]
        residuals = dict(frame_update=h.compare(ys[0] - x, reference[0] - x),
            global_update=h.compare(ys[1] - ys[0].reshape(h.SHAPES[1]), reference[1] - reference[0].reshape(h.SHAPES[1])),
            complete_pair_update=h.compare(ys[1] - x.reshape(h.SHAPES[1]), reference[1] - x.reshape(h.SHAPES[1])))
        require(row['complete_seconds'] >= row['SDK_stages']['complete_boundary_seconds'], 'Outer full IO timer excludes inner boundary')
        calls.append(dict(index=row['index'], phase=row['phase'], route=row['route'], input_id=row['input_id'],
            complete_seconds=row['complete_seconds'], SDK_stages=row['SDK_stages'], endpoints=endpoints, canonical_residuals_vs_A_CPU_FP32=residuals))
        actuals[row['index']] = ys
    require(len(filenames) == 16 and {p.name for p in (w / 'runtime-r1').glob('*.npy')} == filenames, 'Missing/extra saved outputs')
    memory, totals = {}, dict.fromkeys(('contexts_created', 'contexts_destroyed', 'runs_completed', 'output_buffers_acquired', 'output_buffers_released'), 0)
    require(len(r['owner_metrics']) == len(r['phases']) == 4, 'Four owned contexts required')
    for phase, route, owner in zip(r['phases'], SCHEDULE, r['owner_metrics']):
        require(owner['closed'] and not owner['failed'] and owner['live_contexts'] == owner['live_output_buffers'] == 0 and owner['resource_counts_balanced'] and
            not owner['cleanup_errors'] and owner['core_mask'] == 1 and owner['init_flags'] == 0 and owner['runs_completed'] == 2 and
            owner['contexts_created'] == owner['contexts_destroyed'] == 1 and owner['output_buffers_acquired'] == owner['output_buffers_released'] == 4 and
            owner['model_sha256'] == (BASE_MODEL_SHA if route == 'originalFP16_A' else JOINT_MODEL_SHA), 'Actual owned context/model ledger differs')
        for key in totals: totals[key] += owner[key]
        if route in memory: require(memory[route] == owner['SDK_memory'], 'Per-route SDK memory fields changed')
        memory[route] = owner['SDK_memory']
    require(totals == r['resource_totals'] == r['expected_resource_counts'] == dict(contexts_created=4, contexts_destroyed=4, runs_completed=8,
        output_buffers_acquired=16, output_buffers_released=16) and r['all_contexts_and_pending_outputs_closed'] and
        all(r[k] for k in ('source_and_inputs_unchanged', 'affinity_restored', 'environment_restored')), 'Restoration/resource totals incomplete')
    times = {route: [v['complete_seconds'] for v in calls if v['route'] == route] for route in dict.fromkeys(SCHEDULE)}
    means = {route: sum(v) / len(v) for route, v in times.items()}
    require(all(math.isclose(means[route], r['timing_summary'][route]['all4_call_mean_seconds'], rel_tol=1e-12) for route in means), 'Timing summary differs')
    reduction = 1 - means['joint_MLP'] / means['originalFP16_A']
    fidelity = []
    for case in ('pair4', 'pair5'):
        for name in h.NAMES:
            base_call = next(v for v in calls if v['route'] == 'originalFP16_A' and v['input_id'] == case)
            base = next(v['vs_original_A_CPU_FP32']['relative_l2'] for v in base_call['endpoints'] if v['artifact']['name'] == name)
            candidate_errors = [next(v['vs_original_A_CPU_FP32']['relative_l2'] for v in call['endpoints'] if v['artifact']['name'] == name)
                for call in calls if call['route'] == 'joint_MLP' and call['input_id'] == case]
            available = base is not None and all(v is not None for v in candidate_errors)
            fidelity.append(dict(input_id=case, endpoint=name, first_original_A_relative_l2=base, candidate_relative_l2_all_calls=candidate_errors,
                candidate_to_baseline_ratios=[v / base if v is not None and base else None for v in candidate_errors], zero_baseline_error=base == 0,
                screen_passed=all(v <= 1.2 * base for v in candidate_errors) if available else None))
    stdout = w / 'vggt-v83-native518joint129fp16r1-r1.log'
    require(sha(stdout) == observation['compiler_stdout_sha256'] and observation['compile_receipt_sha256'] == COMPILE_SHA, 'Compiler listing observation lineage differs')
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', stdout.read_text(errors='replace'))
    for row in observation['actual_compiled_MLP_rows']:
        require(any(row['name'] in line and re.search(r'\b' + str(row['index']) + r'\b', line) and
            all(v in line for v in (row['op_type'], row['dtype'], row['target'])) for line in text.splitlines()), 'Compiler listing row absent')
    require(observation['all_four_FC1_before_any_FC2'] and not observation['source_interleaving_preserved'], 'Compiled listing scope differs')
    require(c['child_returncode'] == 0 and c['frequency_governor_limits_restored'] and not c['restore_errors'] and c['timeout_seconds'] == 180 and
        c['temperature_limit_c'] == 75 and c['requested_frequency_hz'] == 1000000000, 'Actual controller failed/restoration incomplete')
    temperatures = [t for sample in c['observations'] for t in sample['temperatures_c'].values()]
    require(temperatures and max(temperatures) < 75, 'Thermal guard exceeded')
    cross = [dict(input_id=case, endpoint=name, byte_equal=first[('originalFP16_A', case, name)].tobytes() == first[('joint_MLP', case, name)].tobytes())
        for case in ('pair4', 'pair5') for name in h.NAMES]
    report = dict(schema='v129_joint_MLP_independent_saved_NPU_audit', status='passed_saved_audit_candidate_rejected_by_predeclared_component_speed_screen',
        auditor_sha256=sha(__file__), runtime_receipt_sha256=RESULT_SHA, source_sha256=SOURCE_SHA, compile_receipt_sha256=COMPILE_SHA,
        predeclared_triage_sha256=TRIAGE_SHA, controller_sha256=CONTROL_SHA, schedule=SCHEDULE,
        fixed_FP32_numeric_gate=dict(relative_l2_max=1e-5, max_abs_max=1e-3), strict_rtol=1e-4, strict_atol=1e-5, strict_is_diagnostic=True,
        calls=calls, times_seconds=times, means_seconds=means, mean_complete_time_reduction_fraction=reduction,
        phases=[{k: v[k] for k in ('index', 'route', 'init_seconds', 'destroy_seconds', 'lifecycle_seconds_excluding_save_compare')} for v in r['phases']],
        fidelity_screen=fidelity, component_fidelity_screen_passed=all(v['screen_passed'] is True for v in fidelity),
        component_speed_screen_passed=reduction >= .1, candidate_accepted=False,
        byte_identity_original_A_vs_joint=cross, all_route_repeats_byte_equal=True, audited_output_count=16,
        resource_totals=totals, SDK_per_context_memory_bytes=memory,
        SDK_memory_delta_J_minus_A={k: memory['joint_MLP'][k] - memory['originalFP16_A'][k] for k in ('weight_bytes', 'internal_bytes', 'dma_bytes')},
        memory_scope='Actual SDK per-context query fields; not physical unique allocations, peak whole-model memory or 8GB acceptance.',
        compiler_listing=dict(receipt_sha256=OBSERVATION_SHA, stdout_sha256=observation['compiler_stdout_sha256'], rows=observation['actual_compiled_MLP_rows'],
            source_interleaving_preserved=False, all_four_FC1_listed_before_FC2=True, runtime_hardware_cycle_order_or_hidden_lifetime_measured=False),
        controller=dict(returncode=0, seconds=c['seconds'], max_observed_temperature_c=max(temperatures), baseline=c['baseline'], restored=c['restored'], frequency_governor_limits_restored=True),
        residual_scope='Canonical FP32 subtraction before FP64 norms; each route uses its own frame endpoint for the global update. Zero norms unavailable, no denominator floor. Conditional component reference, not trajectory.',
        NN_SDK_ONNX_or_board_operations_by_auditor=False, full_model_quality_realtime_ATE_or_RPE_accepted=False)
    output = a.output or w / 'joint-MLP-NPU-saved-audit.json'
    with output.open('x') as stream: json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(file=output.name, bytes=output.stat().st_size, sha256=sha(output), reduction_percent=reduction * 100,
        fidelity_screen=report['component_fidelity_screen_passed'], speed_screen=report['component_speed_screen_passed'], all_cross_route_bytes=all(v['byte_equal'] for v in cross))))


if __name__ == '__main__': main()
