#!/usr/bin/env python3
"""Independent saved-array audit; no model, SDK, calibration or device calls."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def compare(actual, reference):
    x, y = actual.astype(np.float64), reference.astype(np.float64)
    error = x - y
    norm = float(np.sqrt(np.sum(y * y)))
    require(norm > 0, 'Nonzero reference norm required')
    maximum = float(np.max(np.abs(error)))
    relative = float(np.sqrt(np.sum(error * error)) / norm)
    strict = bool(np.all(np.abs(error) <= 1e-5 + 1e-4 * np.abs(y)))
    return dict(relative_l2=relative, max_abs=maximum,
                strict_pass=strict,
                fixed_gate_passed=relative <= 1e-5 and maximum <= 1e-3 and strict)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--controller', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    c, r, control = [json.loads(f.read_text()) for f in (a.capture, a.runtime, a.controller)]
    checked = {}

    def array(root, meta):
        name = meta['file']
        require(Path(name).name == name, 'Direct saved-array file required')
        path = root / name
        require(path.stat().st_size == meta['bytes'] and sha(path) == meta['sha256'], 'File identity differs: ' + name)
        value = np.load(path, allow_pickle=False)
        require(value.dtype == np.float32 and value.shape == (1, 2082, 1024)
                and value.flags.c_contiguous and np.isfinite(value).all(), 'Complete native FP32 logical tensor required')
        digest = hashlib.sha256(value.tobytes()).hexdigest()
        require(digest == meta['raw_sha256'], 'Raw array identity differs: ' + name)
        checked[str(path)] = dict(bytes=path.stat().st_size, sha256=meta['sha256'], raw_sha256=digest)
        return value

    records = {row['id']: row for row in r['records'] if row['role'] == 'heldout'}
    require(set(records) == {'heldout00', 'heldout01'}, 'Exactly two preselected heldout cases required')
    capture_records = {row['id']: row for row in c['records']}
    require(set(capture_records) == {'calib00', 'calib01', 'calib02', 'calib03', *records}, 'Frozen 4/2 data split required')
    data = {}
    for key, row in records.items():
        source = capture_records[key]
        require(source['role'] == row['role'] == 'heldout' and source['input'] == row['input'], 'Heldout capture lineage differs')
        reference = source.get('reference', source.get('output'))
        require(reference == row['reference'], 'Heldout original CPU reference lineage differs')
        data[key] = (array(a.capture.parent, row['input']), array(a.capture.parent, row['reference']))
    calls = r['calls']
    require(len(calls) == 8 and [v['route'] for v in calls] == list('AABBBBAA'), 'ABBA phases, two heldout cases each required')
    require([v['input_id'] for v in calls] == ['heldout00', 'heldout01'] * 4, 'Identical heldout ordering required')
    by_case, by_route, results = {}, {'A': [], 'B': []}, []
    outputs = {}
    for call in calls:
        key = (call['route'], call['input_id'])
        x, ref = data[key[1]]
        y = array(a.runtime.parent, call['output'])
        whole = compare(y, ref)
        # Keep differences in FP64 so cancellation is not rounded through FP32.
        increment = compare(y.astype(np.float64) - x, ref.astype(np.float64) - x)
        for field, measured in (('whole_output_vs_CPU', whole), ('residual_update_vs_CPU', increment)):
            recorded = call['comparison'][field]
            require(all(np.isclose(measured[k], recorded[k], rtol=1e-10, atol=1e-13)
                        for k in ('relative_l2', 'max_abs'))
                    and measured['strict_pass'] == recorded['strict_allclose'],
                    'Stored comparison does not recompute: ' + field)
        if key in outputs:
            require(np.array_equal(outputs[key], y), 'Repeated output differs')
        else:
            outputs[key] = y
        elapsed = call['elapsed_seconds']
        require(isinstance(elapsed, (int, float)) and 0 < elapsed < 180, 'Actual positive bounded elapsed required')
        by_case.setdefault(key, []).append(elapsed)
        by_route[key[0]].append(elapsed)
        results.append(dict(route=key[0], input_id=key[1], elapsed_seconds=elapsed,
                            whole_output_vs_CPU=whole, residual_update_vs_CPU=increment))
    require(len(outputs) == 4 and all(len(v) == 2 for v in by_case.values()), 'Four independently repeated outputs required')
    require(r['status'] == 'completed_original_global0_W8A8_component_ABBA_not_quality_accepted'
            and r['resources_balanced'] and r['complete_expected_execution'] and not r['cleanup_errors']
            and r['source_and_inputs_unchanged'] and r['affinity_restored'] and r['environment_restored'],
            'Runtime completion and cleanup required')
    counters = ('contexts_created', 'contexts_destroyed', 'runs_completed',
                'output_buffers_acquired', 'output_buffers_released')
    require(all(sum(row[k] for row in r['resources']) == r['SDK_totals'][k] == 8 for k in counters)
            and all(row['closed'] and not row['live_contexts'] and not row['live_output_buffers']
                    and not row['cleanup_errors'] for row in r['resources']), 'Balanced eight-call resource ledger required')
    require(control['status'] == 'worker_completed_pending_result_review' and control['child_returncode'] == 0
            and control['frequency_governor_limits_restored'] and not control['restore_errors'], 'Controller completion/restoration failed')
    require(control['observations'], 'Actual controller observations required')
    require(control['baseline'] == control['restored'], 'Exact frequency policy restoration required')
    require(all(v['actual_frequency_hz'] == 1000000000 and v['temperatures_c']
                and max(v['temperatures_c'].values()) < 75 for v in control['observations']),
            'Observed fixed clock/thermal bound failed')
    summaries = {route: dict(first_seconds=values[0], subsequent_count=len(values)-1,
                            subsequent_mean_seconds=float(np.mean(values[1:])),
                            all_call_mean_seconds=float(np.mean(values))) for route, values in by_route.items()}
    report = dict(schema='v115_independent_saved_global0_w8a8_audit', status='passed',
                  execution='Saved arrays only; no neural execution, SDK, ONNX, calibration or device calls',
                  inputs={str(f): sha(f) for f in (a.capture, a.runtime, a.controller)},
                  auditor_sha256=sha(__file__), checked_arrays=checked, calls=results,
                  summary=summaries, all_repetitions_byte_equal=True,
                  controller=dict(seconds=control['seconds'], observed_samples=len(control['observations']),
                                  maximum_temperature_c=max(max(v['temperatures_c'].values()) for v in control['observations']),
                                  exact_frequency_policy_restored=True),
                  candidate_accepted=False, full_model_quality_or_realtime_accepted=False)
    report['subsequent_lifecycle_B_over_A'] = summaries['B']['subsequent_mean_seconds'] / summaries['A']['subsequent_mean_seconds']
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], summary=summaries,
                          B_over_A=report['subsequent_lifecycle_B_over_A'])))


if __name__ == '__main__':
    main()
