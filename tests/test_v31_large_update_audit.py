"""Tiny saved-array fixtures; no runtime/model/image execution."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
HELPER = HERE / 'audit-v31-large-update.py'
if not HELPER.exists():
    HELPER = HERE.parent / 'deploy/rk3588/validation/dpvo_dynamic_update/audit-v31-large-update.py'
spec = importlib.util.spec_from_file_location('_v31_large_audit_synthetic', HELPER)
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


def test_threshold_adjacent_FP32_values_are_not_relaxed():
    reference = np.array([0, 0], np.float32)
    actual = np.array([np.float32(1e-5), np.nextafter(np.float32(1e-5), np.float32(np.inf))], np.float32)
    result = audit.numeric(actual, reference)
    assert result['failed_elements'] == 1
    assert result['first_failures'][0]['index'] == [1]
    assert result['rtol'] == 1e-4 and result['atol'] == 1e-5


def test_nonfinite_summary_never_serializes_NaN():
    with np.errstate(invalid='ignore'):
        result = audit.numeric(np.array([np.nan, np.inf], np.float32), np.array([0, 1], np.float32))
    assert not result['finite'] and result['failed_elements'] == 2
    assert result['RMS'] is None
    json.dumps(result, allow_nan=False)


def test_micro_negative_weight_domain_is_independent_of_strict_gate():
    values = dict(net=np.ones((1, 1, 384), np.float32), delta=np.zeros((1, 1, 2), np.float32),
                  weight=np.full((1, 1, 2), -1e-7, np.float32))
    assert audit.numeric(values['weight'], np.zeros_like(values['weight']))['failed_elements'] == 0
    assert not audit.checks(values)['weight_domain_passed']


@pytest.fixture
def tiny_run(tmp_path, monkeypatch):
    runner, dynamic = tmp_path / 'runner.py', tmp_path / 'dynamic.py'
    runner.write_text('# synthetic source only\n')
    dynamic.write_text('# synthetic source only\n')
    monkeypatch.setattr(audit, 'DRIVER_SHA', audit.sha(runner))
    monkeypatch.setattr(audit, 'DYNAMIC_SHA', audit.sha(dynamic))
    contract = tmp_path / 'contract.json'
    rows = [dict(role='before', sha256='1' * 64, active_edges=2),
            dict(role='final', sha256='2' * 64, active_edges=2)]
    contract.write_text(json.dumps(dict(actual_saved_states=rows)))
    monkeypatch.setattr(audit, 'CONTRACT_SHA', audit.sha(contract))
    pins = {str(runner): audit.sha(runner), str(dynamic): audit.sha(dynamic)}
    weights = {str(i): {'tensor_sha256': '3' * 64} for i in range(94)}
    d = dict(calls=[], attempted_calls=6, completed_calls=6, failed_calls=0, closed=True,
             topology_cache=dict(closed=True, current_payload_bytes=0, current_entries=0))
    native = dict(attempted_calls=2, completed_calls=2, current_context_open=False,
                  generations=[dict(preparation=dict(context_bytes=1024))],
                  closed_contexts=[dict(closed=True, current_context_bytes=0, failed_calls=0, completed_calls=2)])
    run = dict(schema='v31_DPVO_saved_large_Update_comparison', states=[], fixed_gate=dict(rtol=1e-4, atol=1e-5),
               script_sha256=audit.DRIVER_SHA, dynamic_source_sha256=audit.DYNAMIC_SHA,
               reconstruction_contract_sha256=audit.CONTRACT_SHA, order=[list(v) for v in audit.ORDERS],
               source_pins=pins, source_hashes_after=pins.copy(), sources_and_inputs_unchanged=True,
               original_weights_before=weights, original_weights_after=weights.copy(), original_weights_unchanged=True,
               dynamic_metrics=d, correlation_metrics=native, encoder_metrics=dict(closed=True, completed_frames=17, total_encode_seconds=.1),
               reconstructed_frames=[{} for _ in range(17)], cleanup_errors=[], affinity_restored=True,
               NPU_used=False, BA_executed=False, valid_pose_output=False, realtime_validated=False,
               candidate_accepted=True, status='completed_numerical_accepted', reconstruction_seconds=.2,
               peak_rss_mib=1, actual_driver_elapsed_seconds=20)
    artifacts = {}
    for i, row in enumerate(rows):
        values = dict(net=np.ones((1, 2, 384), np.float32), delta=np.ones((1, 2, 2), np.float32),
                      weight=np.full((1, 2, 2), .5, np.float32))
        reps = {}
        for method in ('Torch', 'ORT'):
            path = tmp_path / ('state-%d-%s.npz' % (i, method))
            np.savez(path, **values)
            reps[method] = dict(file=path.name, bytes=path.stat().st_size, sha256=audit.sha(path),
                                arrays={k: audit.meta(v) for k, v in values.items()})
            artifacts[i, method] = path
        numerics = {k: dict(failed_elements=0, bitwise_equal=True, rtol=1e-4, atol=1e-5,
                           max_abs=0., relative_l2=0., RMS=0.) for k in audit.KEYS}
        calls = []
        for repeat, order in enumerate(audit.ORDERS):
            for method in order:
                if method == 'Torch':
                    timing = dict(total_seconds=1.2, feed_copy_tensor_seconds=.1, original_Update_seconds=.9, output_copy_seconds=.2)
                else:
                    b = dict(index=len(d['calls']), input_validation_seconds=.01, topology_seconds=.01,
                             feed_copy_seconds=.04, ORT_seconds=.9, output_copy_check_seconds=.04,
                             total_seconds=1., group_shapes={'kk': [1632, 17], 'ij': [245, 96]})
                    d['calls'].append(b)
                    timing = dict(total_seconds=1.1, runtime_breakdown=b)
                calls.append(dict(method=method, repeat=repeat, order=list(order), outputs=reps[method]['arrays'],
                                  repeat_bitwise_equal=True, timing=timing,
                                  output_checks=dict(finite=True, weight_domain_passed=True, weight_min=.5, weight_max=.5)))
        run['states'].append(dict(role=row['role'], snapshot_sha256=row['sha256'], edges=2,
            representative_outputs=reps, comparisons=[dict(repeat=j, comparison=dict(gate_passed=True, outputs=numerics)) for j in range(3)],
            calls=calls, numerical_gate_passed=True, inputs_unchanged=True, original_state_net_reset_each_call=True,
            historical_target_and_weight_are_not_reference=True,
            timing={m: dict(first_seconds=t, warm_seconds=[t, t], warm_mean_seconds=t) for m, t in (('Torch', 1.2), ('ORT', 1.1))},
            reproject_seconds=.01, correlation_complete_seconds=.02, context_gather_seconds=.01))

    def execute():
        (tmp_path / 'run.json').write_text(json.dumps(run))
        return audit.audit(tmp_path, contract, runner, dynamic)

    return run, artifacts, execute


def test_clean_tiny_saved_fixture_passes_accounting(tiny_run):
    _, _, execute = tiny_run
    result = execute()
    assert result['status'] == 'passed' and result['candidate_accepted']
    assert result['input_files']['comparison_run.json']['file'] == 'run.json'
    assert 'run.json' not in result['input_files']


def test_false_accept_rejected_even_if_modified_output_hashes_are_updated(tiny_run):
    run, artifacts, execute = tiny_run
    path = artifacts[0, 'ORT']
    with np.load(path) as z:
        values = {k: z[k].copy() for k in z.files}
    values['delta'][0, 0, 0] += .01
    np.savez(path, **values)
    item = run['states'][0]['representative_outputs']['ORT']
    item.update(sha256=audit.sha(path), bytes=path.stat().st_size, arrays={k: audit.meta(v) for k, v in values.items()})
    for c in run['states'][0]['calls']:
        if c['method'] == 'ORT':
            c['outputs'] = item['arrays']
    result = execute()
    assert result['status'] == 'failed' and not result['candidate_accepted']
    assert 'overall_no_false_acceptance' in result['checks']['failures']
    assert result['states'][0]['numeric']['delta']['failed_elements'] == 1


def test_stale_archive_SHA_detected_even_if_arrays_match(tiny_run):
    run, _, execute = tiny_run
    run['states'][0]['representative_outputs']['Torch']['sha256'] = '0' * 64
    result = execute()
    assert result['status'] == 'failed'
    assert 'artifact_SHA:state-0-Torch.npz' in result['checks']['failures']


def test_timing_summary_tamper_detected(tiny_run):
    run, _, execute = tiny_run
    run['states'][1]['timing']['ORT']['warm_mean_seconds'] = .1
    result = execute()
    assert result['status'] == 'failed'
    assert 'warm_mean:1:ORT' in result['checks']['failures']
