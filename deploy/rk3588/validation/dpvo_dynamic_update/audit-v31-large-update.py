#!/usr/bin/env python3
"""Portable NumPy-only audit of saved v31 large-E Update comparison outputs.

No Torch, ORT, image decoding, native execution or BA. Audit consistency may
pass while the candidate fails the unchanged numerical gate. Only first-call
representatives were saved; warm-repeat hashes are recorded evidence, not
independently available repeat arrays. Missing optional inputs/sources are
explicitly unknown and cannot become a reconstructed-feature certificate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.dont_write_bytecode = True
DRIVER_SHA = '32083ec282643a5149ddc38c374d8919f76182d0b4a2698d17040eb2d632826d'
DYNAMIC_SHA = '2e313911b4cf2ac18373cbd3b72dbfe81644918b92c4ffee32f1109e238aa692'
CONTRACT_SHA = '45ba417a63a813a3af0c33ed080fc404d9eb7052ddae309cb0dba02d4216c45f'
KEYS = ('net', 'delta', 'weight')
ORDERS = (('Torch', 'ORT'), ('ORT', 'Torch'), ('Torch', 'ORT'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def meta(a):
    return dict(shape=list(a.shape), dtype=str(a.dtype), bytes=a.nbytes,
                tensor_sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())


def numeric(actual, expected):
    if actual.dtype != np.float32 or expected.dtype != np.float32 or actual.shape != expected.shape:
        raise ValueError('Matching FP32 output shape required')
    finite = np.isfinite(actual) & np.isfinite(expected)
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    difference = a - b
    mask = ~finite | (np.abs(difference) > np.float64(1e-5) + np.float64(1e-4) * np.abs(b))
    good = bool(finite.all())
    denominator = float(np.linalg.norm(b.ravel())) if good else None
    return dict(failed_elements=int(mask.sum()), finite=good, rtol=1e-4, atol=1e-5,
                max_abs=float(np.abs(difference).max()) if good else None,
                RMS=float(np.sqrt(np.mean(difference * difference))) if good else None,
                relative_l2=float(np.linalg.norm(difference.ravel())) / denominator
                if good and denominator else None,
                bitwise_equal=actual.tobytes(order='C') == expected.tobytes(order='C'),
                first_failures=[dict(index=[int(i) for i in loc], actual=float(a[tuple(loc)])
                                    if np.isfinite(a[tuple(loc)]) else None,
                                    reference=float(b[tuple(loc)]) if np.isfinite(b[tuple(loc)]) else None,
                                    absolute_error=float(abs(difference[tuple(loc)]))
                                    if np.isfinite(difference[tuple(loc)]) else None,
                                    tolerance=float(1e-5 + 1e-4 * abs(b[tuple(loc)]))
                                    if np.isfinite(b[tuple(loc)]) else None)
                                for loc in np.argwhere(mask)[:8]])


def checks(a):
    finite = {k: bool(np.isfinite(v).all()) for k, v in a.items()}
    w = a['weight']
    return dict(finite=all(finite.values()),
                nonfinite_elements={k: int(np.count_nonzero(~np.isfinite(v))) for k, v in a.items()},
                weight_domain_passed=bool(finite['weight'] and ((w >= 0) & (w <= 1)).all()),
                weight_min=float(w.min()) if finite['weight'] else None,
                weight_max=float(w.max()) if finite['weight'] else None)


def audit(run_dir, contract_path, runner_path, dynamic_path, *, snapshot_dir=None,
          update_manifest=None, source_map=None):
    count, failures = 0, []
    input_files = {}

    def check(value, name):
        nonlocal count
        count += 1
        if not value:
            failures.append(name)

    def read(path, digest=None, role=None):
        path = Path(path)
        actual = sha(path)
        input_files[role or path.name] = dict(file=path.name, sha256=actual, bytes=path.stat().st_size)
        if digest is not None:
            check(actual == digest, 'file_SHA:' + path.name)
        return json.loads(path.read_text())

    def arrays(folder, artifact):
        name = artifact['file']
        if Path(name).name != name:
            raise ValueError('Artifact basename required')
        path = Path(folder) / name
        check(sha(path) == artifact['sha256'], 'artifact_SHA:' + name)
        check(path.stat().st_size == artifact['bytes'], 'artifact_bytes:' + name)
        with np.load(path, allow_pickle=False) as z:
            values = {k: z[k].copy() for k in z.files}
        check({k: meta(v) for k, v in values.items()} == artifact['arrays'], 'raw_arrays:' + name)
        input_files[name] = dict(sha256=sha(path), bytes=path.stat().st_size)
        return values

    run_dir = Path(run_dir)
    run = read(run_dir / 'run.json', role='comparison_run.json')
    contract = read(contract_path, CONTRACT_SHA)
    for path, digest in ((runner_path, DRIVER_SHA), (dynamic_path, DYNAMIC_SHA)):
        actual = sha(path)
        input_files[Path(path).name] = dict(sha256=actual, bytes=Path(path).stat().st_size)
        check(actual == digest, 'frozen_actual_source:' + Path(path).name)
    check(run['schema'] == 'v31_DPVO_saved_large_Update_comparison' and len(run['states']) == 2, 'run_schema_two_states')
    check(run['script_sha256'] == DRIVER_SHA and run['dynamic_source_sha256'] == DYNAMIC_SHA
          and run['reconstruction_contract_sha256'] == CONTRACT_SHA, 'source_receipt_binding')
    check(run['fixed_gate'] == dict(rtol=1e-4, atol=1e-5), 'fixed_gate')
    check(run['order'] == [list(v) for v in ORDERS], 'declared_alternating_order')
    pins = run['source_pins']
    check(run['source_hashes_after'] == pins and run['sources_and_inputs_unchanged'], 'recorded_source_before_after')
    actual_sources, unknown_sources = [], []
    for board_path, expected in pins.items():
        mapped = (source_map or {}).get(board_path)
        path = Path(mapped) if mapped is not None else Path(board_path)
        if path.is_file():
            value = sha(path)
            check(value == expected, 'independent_source_bytes:' + path.name)
            actual_sources.append(dict(file=Path(board_path).name, sha256=value))
        else:
            unknown_sources.append(dict(file=Path(board_path).name, recorded_sha256=expected))
    check(len(run['original_weights_before']) == 94 and run['original_weights_before'] == run['original_weights_after']
          and run['original_weights_unchanged'], 'recorded_original94_weight_metadata_before_after')
    if update_manifest is not None:
        manifest = read(update_manifest, run['dynamic_metrics']['manifest_sha256'])
        check(manifest['original_weights_before'] == run['original_weights_before']
              and manifest['original_weights_before'] == manifest['original_weights_after'], 'export_original94_weight_binding')
        m = manifest['models']['final']
        path = Path(update_manifest).parent / m['file']
        check(Path(m['file']).name == m['file'] and sha(path) == m['sha256']
              and path.stat().st_size == m['bytes']
              and m['sha256'] == run['dynamic_metrics']['model_sha256'], 'actual_dynamic_graph_artifact')
        input_files[m['file']] = dict(sha256=sha(path), bytes=path.stat().st_size)
    state_results = []
    snapshots = []
    if snapshot_dir is not None:
        original_run = read(Path(snapshot_dir) / 'run.json', contract['provenance']['v30_run_sha256'], role='v30_original_run.json')
        for row in contract['actual_saved_states']:
            snapshots.append(arrays(snapshot_dir, original_run[row['role']]['artifact']))
        check(original_run['original_weights_before'] == run['original_weights_before'], 'v30_original_weight_metadata_binding')
    ort_index = 0
    for index, (state, original) in enumerate(zip(run['states'], contract['actual_saved_states'])):
        check(state['role'] == original['role'] and state['snapshot_sha256'] == original['sha256']
              and state['edges'] == original['active_edges'], 'state_identity:%d' % index)
        if snapshots:
            for k in ('net', 'ii', 'jj', 'kk'):
                check(meta(snapshots[index][k]) == state['input_arrays'][k], 'original_saved_input:%d:%s' % (index, k))
        inputs_item = state.get('inputs_artifact')
        input_status = 'unknown_not_available_locally'
        if inputs_item and (run_dir / inputs_item['file']).is_file():
            values = arrays(run_dir, inputs_item)
            check(all(meta(values[k]) == v for k, v in state['input_arrays'].items()), 'full_saved_input_metadata:%d' % index)
            check(meta(values['coords']) == state['coordinates'], 'saved_projection_metadata:%d' % index)
            input_status = 'saved_single_input_bundle_SHA_and_raw_arrays_checked'
            del values
        representative = {method: arrays(run_dir, state['representative_outputs'][method]) for method in ('Torch', 'ORT')}
        for method, values in representative.items():
            check(tuple(values) == KEYS and all(v.dtype == np.float32 and v.shape == (1, state['edges'], c)
                  for v, c in zip(values.values(), (384, 2, 2))), 'output_contract:%d:%s' % (index, method))
        result = {k: numeric(representative['ORT'][k], representative['Torch'][k]) for k in KEYS}
        domains = {method: checks(values) for method, values in representative.items()}
        gate = all(r['failed_elements'] == 0 for r in result.values()) \
            and all(d['finite'] and d['weight_domain_passed'] for d in domains.values())
        for repeat, recorded in enumerate(state['comparisons']):
            check(recorded['repeat'] == repeat and recorded['comparison']['gate_passed'] == gate, 'recorded_gate:%d:%d' % (index, repeat))
            for k, computed in result.items():
                previous = recorded['comparison']['outputs'][k]
                check(all(previous[field] == computed[field] for field in ('failed_elements', 'bitwise_equal', 'rtol', 'atol'))
                      and all(np.isclose(previous[field], computed[field], rtol=1e-12, atol=1e-14)
                              for field in ('max_abs', 'relative_l2', 'RMS')), 'independent_numeric:%d:%d:%s' % (index, repeat, k))
        check(len(state['calls']) == 6 and len(state['comparisons']) == 3, 'fixed_call_population:%d' % index)
        for pos, call in enumerate(state['calls']):
            repeat, within = divmod(pos, 2)
            method = ORDERS[repeat][within]
            check(call['method'] == method and call['repeat'] == repeat and call['order'] == list(ORDERS[repeat]),
                  'actual_alternating_order:%d:%d' % (index, pos))
            check(call['outputs'] == {k: meta(v) for k, v in representative[method].items()}
                  and call['repeat_bitwise_equal'], 'recorded_repeat_raw_SHA:%d:%d' % (index, pos))
            check(call['output_checks']['finite'] == domains[method]['finite']
                  and call['output_checks']['weight_domain_passed'] == domains[method]['weight_domain_passed']
                  and call['output_checks']['weight_min'] == domains[method]['weight_min']
                  and call['output_checks']['weight_max'] == domains[method]['weight_max'], 'recorded_output_domain:%d:%d' % (index, pos))
            t = call['timing']
            if method == 'Torch':
                parts = [t[k] for k in ('feed_copy_tensor_seconds', 'original_Update_seconds', 'output_copy_seconds')]
                check(all(np.isfinite(x) and x >= 0 for x in parts) and abs(sum(parts) - t['total_seconds']) < 1e-8,
                      'Torch_timing_sum:%d:%d' % (index, pos))
            else:
                b = t['runtime_breakdown']
                check(b == run['dynamic_metrics']['calls'][ort_index] and b['index'] == ort_index, 'ORT_counter_breakdown:%d' % ort_index)
                ort_index += 1
                parts = [b[k] for k in ('input_validation_seconds', 'topology_seconds', 'feed_copy_seconds', 'ORT_seconds', 'output_copy_check_seconds')]
                check(all(np.isfinite(x) and x >= 0 for x in parts) and abs(sum(parts) - b['total_seconds']) < 1e-8
                      and t['total_seconds'] >= b['total_seconds'], 'ORT_timing_sum:%d:%d' % (index, pos))
                check(b['group_shapes'] == {'kk': [1632, 17], 'ij': [245, 96]}, 'group_capacity:%d:%d' % (index, pos))
        for method in ('Torch', 'ORT'):
            ts = [c['timing']['total_seconds'] for c in state['calls'] if c['method'] == method]
            expected = dict(first_seconds=ts[0], warm_seconds=ts[1:], warm_mean_seconds=float(np.mean(ts[1:])))
            check(state['timing'][method] == expected, 'warm_mean:%d:%s' % (index, method))
        check(state['numerical_gate_passed'] == gate and state['inputs_unchanged']
              and state['original_state_net_reset_each_call'] and state['historical_target_and_weight_are_not_reference'],
              'state_decision_input_scope:%d' % index)
        state_results.append(dict(role=state['role'], edges=state['edges'], independently_recomputed_gate=gate,
                                  numeric=result, domains=domains, timing=state['timing'], input_bundle_status=input_status,
                                  reproject_seconds=state['reproject_seconds'], correlation_complete_seconds=state['correlation_complete_seconds'],
                                  context_gather_seconds=state['context_gather_seconds'],
                                  saved_outputs={k: state['representative_outputs'][k] for k in ('Torch', 'ORT')}))
        del representative
    accepted = all(s['independently_recomputed_gate'] for s in state_results)
    check(run['candidate_accepted'] == accepted and run['status'] ==
          ('completed_numerical_accepted' if accepted else 'completed_numerical_rejected'), 'overall_no_false_acceptance')
    d = run['dynamic_metrics']
    tc = d['topology_cache']
    check(d['attempted_calls'] == d['completed_calls'] == 6 and d['failed_calls'] == 0
          and d['closed'] and tc['closed'] and tc['current_payload_bytes'] == tc['current_entries'] == 0,
          'dynamic_calls_and_cache_closed')
    c = run['correlation_metrics']
    check(c['attempted_calls'] == c['completed_calls'] == 2 and len(c['generations']) == 1
          and not c['current_context_open'] and len(c['closed_contexts']) == 1, 'native_shared_generation_and_closed')
    check(all(x['closed'] and x['current_context_bytes'] == 0 and x['failed_calls'] == 0
              and x['completed_calls'] == 2 for x in c['closed_contexts']), 'native_context_bytes_zero')
    check(run['encoder_metrics']['closed'] and run['encoder_metrics']['completed_frames'] == len(run['reconstructed_frames']) == 17,
          'singleton_encoder_17_and_closed')
    check(not run['cleanup_errors'] and run['affinity_restored'] and not run['NPU_used']
          and not run['BA_executed'] and not run['valid_pose_output'] and not run['realtime_validated'], 'cleanup_and_no_VO_claim')
    return dict(schema='v31_large_Update_independent_saved_array_review', status='passed' if not failures else 'failed',
                auditor_sha256=sha(__file__), checks=dict(count=count, failures=failures), input_files=input_files,
                candidate_accepted=accepted, original_runner_status=run['status'], states=state_results,
                source_scope=dict(independently_hashed_local_files=actual_sources, unavailable_local_files=unknown_sources,
                                  trusted_runner_source_sha256=DRIVER_SHA, trusted_dynamic_source_sha256=DYNAMIC_SHA,
                                  original94_weight_metadata_before_after_bound_to_export=update_manifest is not None,
                                  actual_parameter_tensors_reloaded=False),
                lifecycle=dict(dynamic=d, encoder_closed=run['encoder_metrics']['closed'], native_context_bytes_at_close=[x['current_context_bytes'] for x in c['closed_contexts']],
                               native_prepared_bytes=c['generations'][0]['preparation']['context_bytes'], cleanup_errors=run['cleanup_errors']),
                reconstruction_seconds=run['reconstruction_seconds'], encoder_seconds=run['encoder_metrics']['total_encode_seconds'],
                peak_rss_mib=run['peak_rss_mib'], driver_seconds=run['actual_driver_elapsed_seconds'],
                limitations=['Numerical rejection is preserved; audit passed means saved-artifact/accounting consistency, not accepted deployment or VO.',
                             'Warm-repeat arrays were not retained; their recorded raw SHA values match actual saved representatives, but each warm array cannot be independently recomputed.',
                             'Same input is supported by frozen runner/static copy/reset wiring and snapshot input hashes; per-call actual feeds were not separately retained.',
                             'No saved original feature byte certificate; no re-encoding, reproject, correlation, ORT/Torch execution or BA was performed by this auditor.',
                             'Source and weights checks distinguish locally hashed bytes from unavailable recorded metadata; no original parameter tensors were loaded.',
                             'Timings are recorded same-process observations, not thermal-controlled estimates or full-VO/20Hz acceptance.'])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'contract', 'runner-source', 'dynamic-source', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--snapshot-dir', type=Path)
    p.add_argument('--update-manifest', type=Path)
    p.add_argument('--source-map', type=Path, help='Optional JSON: exact recorded board filename -> actual local filename')
    a = p.parse_args(argv)
    result = audit(a.run_dir, a.contract, a.runner_source, a.dynamic_source,
                   snapshot_dir=a.snapshot_dir, update_manifest=a.update_manifest,
                   source_map=json.loads(a.source_map.read_text()) if a.source_map else None)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], candidate_accepted=result['candidate_accepted'],
                          checks=result['checks'], output=a.output.name, sha256=sha(a.output)), allow_nan=False))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
