#!/usr/bin/env python3
"""Saved NumPy/JSON audit of one complete captured B8/D8/M48 SDK graph.

No Torch, ONNX Runtime, native SDK, model loading or forward execution.
SDK lifecycle and remote source checks are recorded evidence; saved arrays,
local source/artifact hashes and inverse edge mapping are independently checked.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

MANIFEST_SHA = '0e182dcb42825525fe1e7e6b84ffd5af50c3dc02a33f183087bff3963a653331'
RUNTIME_SHA = '55d7b04e782a687fedb6475b8388a12d52361cc56b1a7f260537bd3e1d679d1d'
COMPILER_SHA = 'fb3a6e9eed9a4bc3f8490f0cf61c7ef5d5f4f4eff065bb9d9f057de97fb9cdd1'
CONTRACTS_SHA = 'aa9e6af938f64ae4a4a82ffc84b487321fff208547703b0d65646c72b96d85c9'
CANONICAL = ('net', 'delta', 'weight')
SDK_NAMES = ('net_out', 'delta', 'weight')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def numerical(value, reference):
    a, b = value.astype(np.float64), reference.astype(np.float64)
    if value.shape != reference.shape or value.dtype != reference.dtype or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Comparable finite FP32 arrays required')
    difference = a-b
    failed = int(np.count_nonzero(np.abs(difference) > 1e-5+1e-4*np.abs(b)))
    denominator = float(np.linalg.norm(b.ravel()))
    return dict(finite=True, gate_passed=failed == 0, failed_elements=failed,
        relative_L2=float(np.linalg.norm(difference.ravel())/denominator) if denominator else None,
        max_abs=float(np.max(np.abs(difference))), nonfinite_elements=0, rtol=1e-4, atol=1e-5)


def audit(args):
    failures, count, artifacts = [], 0, []
    def check(condition, name):
        nonlocal count
        count += 1
        if not condition:
            failures.append(name)
    def arrays(directory, row, names=None):
        relative = Path(row['file'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe artifact path')
        path = directory/relative
        check(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'artifact_file:'+relative.name)
        with np.load(path, allow_pickle=False) as archive:
            check(set(archive.files) == set(row['arrays']), 'artifact_keys:'+relative.name)
            values = {name: archive[name] for name in (names or archive.files)}
        for name, value in values.items():
            meta = row['arrays'][name]
            check(list(value.shape) == meta['shape'] and str(value.dtype) == meta['dtype']
                and value.nbytes == meta['bytes']
                and hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() == meta['tensor_sha256'],
                'array_metadata:'+relative.name+':'+name)
        artifacts.append(dict(file=relative.as_posix(), bytes=row['bytes'], sha256=sha(path)))
        return values
    def same_metrics(actual, recorded, name):
        for key, value in actual.items():
            other = recorded[key]
            exact = value == other
            # ARM/x86 diagnostic scalar reductions may round differently. This
            # does not alter the elementwise gate or its exact failure counts.
            check(exact or (key == 'relative_L2' and value is not None and other is not None
                  and np.isclose(value, other, rtol=1e-10, atol=1e-14)), name+':'+key)
    manifest = json.loads((args.export_dir/'manifest.json').read_text())
    check(sha(args.export_dir/'manifest.json') == MANIFEST_SHA, 'original_manifest_SHA')
    check(manifest['candidate_accepted'] is False and manifest['strict_FP32_gate_passed'] is False,
          'original_CPU_ORT_rejection_retained')
    case = manifest['cases'][0]
    check(case['observation_id'] == 13 and case['edges'] == 3072 and len(case['groups']) == 1
          and case['groups'][0]['model_key'] == 'B8_D8_M48', 'whole_initial_graph')
    original = arrays(args.capture_dir, case['original_inputs_artifact'], ('net', 'inp', 'corr', 'ii', 'jj', 'kk'))
    references = dict(original_Torch=arrays(args.capture_dir, case['original_gold_artifact']),
        grouped_Torch=arrays(args.export_dir, case['grouped_Torch_artifact']),
        grouped_ORT=arrays(args.export_dir, case['grouped_ORT_artifact']))
    feed = arrays(args.export_dir, case['groups'][0]['inputs_artifact'])
    ii, jj, kk = (original[name] for name in ('ii', 'jj', 'kk'))
    check(all(v.dtype == np.int64 and v.shape == (3072,) for v in (ii, jj, kk)), 'original_int64_edges')
    check(np.array_equal(ii, kk//48), 'original_patch_source_IDs')
    # Independent lexicographic inverse map; do not reuse the driver's scatter.
    order = np.lexsort((kk % 48, jj, ii))
    index = order.reshape(8, 8, 48)
    check(np.array_equal(np.sort(order), np.arange(3072)), 'each_original_edge_exactly_once')
    sources = np.unique(ii)
    for view, source in enumerate(sources):
        check(np.all(ii[index[view]] == source) and len(np.unique(jj[index[view]])) == 8,
              'eight_complete_targets:'+str(view))
        check(np.array_equal((kk[index[view]] % 48), np.broadcast_to(np.arange(48), (8, 48))),
              'all_48_patches_each_target:'+str(view))
    for name in ('net', 'inp', 'corr'):
        expected = original[name][0, index, :].transpose(0, 3, 1, 2)
        check(np.array_equal(expected, feed[name]), 'exact_captured_CF_input:'+name)
    packed_references = {kind: {sdk: values[canonical][0, index, :].transpose(0, 3, 1, 2)
        for sdk, canonical in zip(SDK_NAMES, CANONICAL)} for kind, values in references.items()}
    source_hashes = {name: sha(args.sdk_dir/name) for name in
        ('compile-v37-grouped-update.py', 'v37_grouped_contracts.py', 'v37_grouped_rknn.py')}
    check(list(source_hashes.values()) == [COMPILER_SHA, CONTRACTS_SHA, RUNTIME_SHA], 'frozen_local_SDK_sources')
    source_hashes[args.driver.name] = sha(args.driver)
    compiled = json.loads((args.compile_dir/'compile.json').read_text())
    check(compiled['script_sha256'] == COMPILER_SHA and compiled['contracts_sha256'] == CONTRACTS_SHA
          and compiled['manifest_sha256'] == MANIFEST_SHA and compiled['model_key'] == 'B8_D8_M48'
          and compiled['input']['sha256'] == manifest['models']['B8_D8_M48']['model']['sha256'], 'compile_ancestor_chain')
    check(compiled['configuration'] == dict(target_platform='rk3588', float_dtype='float16', optimization_level=3,
          do_quantization=False) and compiled['released'] is True and not compiled.get('release_error')
          and compiled['candidate_accepted'] is False and compiled['export_strict_FP32_gate_passed'] is False
          and compiled['diagnostic_allow_fp32_rejected'] is True, 'compiler_config_and_diagnostic_scope')
    check(compiled['source_pins'] == compiled['source_hashes_after'] and compiled['sources_and_inputs_unchanged'],
          'reported_compile_source_lifetime')
    for directory, artifact in ((args.compile_dir, compiled['output']),
                                (args.export_dir, manifest['models']['B8_D8_M48']['model'])):
        path = directory/artifact['file']
        check(path.stat().st_size == artifact['bytes'] and sha(path) == artifact['sha256'], 'local_model_SHA:'+path.name)
    calls, receipts, all_values, lifecycle = [], {}, [], {}
    profile_rows = None
    for role, directory, expected_calls in (('normal', args.normal_dir, 4), ('profile', args.profile_dir, 1)):
        path = directory/'run.json'; report = json.loads(path.read_text())
        receipts[role] = dict(file=path.name, sha256=sha(path))
        check(report['status'] == 'completed_initial_group_RKNN_diagnostic_not_VO'
            and report['script_sha256'] == source_hashes[args.driver.name]
            and report['runtime_sha256'] == RUNTIME_SHA and report['export_manifest_sha256'] == MANIFEST_SHA
            and report['compile_sha256'] == sha(args.compile_dir/'compile.json')
            and report['model_sha256'] == compiled['output']['sha256'], role+':run_chain')
        check(len(report['calls']) == expected_calls and report['candidate_accepted'] is False
            and report['VO_state_committed'] is False and report['inherited_CPU_ORT_FP32_gate_passed'] is False
            and report['actual_original_FP32_gate_passed'] is False and report['collect_perf_detail'] == (role == 'profile'), role+':scope')
        check(report['cleanup_errors'] == [] and report['affinity_restored'] and report['sources_and_inputs_unchanged']
              and report['source_pins'] == report['source_hashes_after'], role+':reported_owner_cleanup_and_sources')
        session = report['session']
        check(session['closed'] and session['context_destroyed'] and session['cleanup_errors'] == []
              and session['current_DMA_bytes'] == session['current_model_host_bytes'] == 0
              and session['sdk_run_api_calls'] == session['completed_calls'] == session['SDK_outputs_releases'] == expected_calls
              and session['source_pins'] == session['source_hashes_after'] and session['sources_and_inputs_unchanged']
              and session['core_mask'] == 7 and session['init_flags'] == (8 if role == 'profile' else 0), role+':reported_SDK_lifecycle')
        check(session['input_attributes'] == (lifecycle.get('normal') or session)['input_attributes']
              and session['output_attributes'] == (lifecycle.get('normal') or session)['output_attributes'], role+':same_attrs')
        lifecycle[role] = session
        for number, row in enumerate(report['calls']):
            label = role+':'+str(number)
            check(row['execution_status'] == 'completed' and row['candidate_accepted'] is False
                  and row['outputs_released'] and row['outputs_finite'] and row['weight_domain_valid']
                  and not row['all_outputs_zero'] and not row['strict_component_gate_passed'], label+':reported_call_scope')
            values = arrays(directory, row['CF_artifact']); full = arrays(directory, row['full_order_artifact'])
            all_values.append(values)
            check(set(values) == set(SDK_NAMES) and set(full) == set(CANONICAL), label+':output_names')
            check(all(v.dtype == np.float32 and np.isfinite(v).all() for v in values.values())
                  and np.all((values['weight'] >= 0) & (values['weight'] <= 1)), label+':finite_domain')
            for sdk, canonical in zip(SDK_NAMES, CANONICAL):
                inverse = values[sdk].transpose(0, 2, 3, 1).reshape(3072, -1)[np.argsort(order)][None]
                check(np.array_equal(inverse, full[canonical]), label+':independent_inverse_scatter:'+canonical)
            comparisons = {kind: {name: numerical(value, ref[name]) for name, value in values.items()}
                           for kind, ref in packed_references.items()}
            for kind, comparison in comparisons.items():
                for name, computed in comparison.items():
                    same_metrics(computed, row['versus_saved_references'][kind][name], label+':'+kind+':'+name)
            for name in CANONICAL:
                same_metrics(numerical(full[name], references['original_Torch'][name]),
                             row['full_versus_original'][name], label+':full:'+name)
            gate = all(v['gate_passed'] for v in comparisons['original_Torch'].values())
            check(gate == row['strict_FP32_gate_passed'], label+':fixed_gate_result')
            timing = row['timing']
            stages = ('pack_seconds','input_set_seconds','run_api_seconds','outputs_get_seconds','owned_copy_seconds','outputs_release_seconds')
            check(all(timing[name] >= 0 for name in stages) and sum(timing[name] for name in stages) <= timing['forward_seconds']
                  and timing['forward_seconds'] <= row['total_execute_seconds'] and timing['synchronous'], label+':nested_timing')
            for name, record in zip(('net','inp','corr'), timing['input_records']):
                check(record['name'] == name and record['logical_shape'] == list(feed[name].shape)
                      and record['buffer_shape'] == list(feed[name].transpose(0,2,3,1).shape)
                      and record['bytes'] == feed[name].nbytes and record['request_format'] == record['native_attribute_format'] == 1
                      and record['pass_through'] == 0 and record['owned'] and record['explicit_NHWC_pack'], label+':actual_input_record:'+name)
            native = row['native_log']; raw = native['raw_text'].encode('utf-8')
            check(len(raw) == native['bytes'] and hashlib.sha256(raw).hexdigest() == native['raw_sha256']
                  and not native['truncated'] and not native['RKNN_error_detected'] and 'E RKNN' not in native['raw_text'], label+':native_log')
            calls.append(dict(mode=role,index=number,forward_seconds=timing['forward_seconds'],
                restore_edge_order_seconds=row['restore_edge_order_seconds'],timing=timing,versus_references=comparisons))
            if role == 'profile':
                performance = row['profile']; text = performance['raw_text']; data = text.encode('utf-8')
                check(len(data) == performance['bytes'] and hashlib.sha256(data).hexdigest() == performance['raw_sha256'], 'actual_profile_raw_SHA')
                pattern = re.compile(r'^\s*(\d+)\s+(\S+)\s+(\S+)\s+(CPU|NPU|GPU)\s+(.+?)\s+(\d+/\d+/\d+)\s+(\d+)\s+(.*?)\s+(\d+(?:\.\d+)?%/\d+(?:\.\d+)?%/\d+(?:\.\d+)?%)\s+(\d+)\s+(.+?)\s*$')
                profile_rows = []
                for line in text.splitlines():
                    if re.match(r'^\s*\d+\s', line):
                        found = pattern.match(line)
                        if found is None:
                            raise ValueError('Unparsed actual profile operator row')
                        profile_rows.append(dict(id=int(found[1]),op=found[2],dtype=found[3],target=found[4],
                            microseconds=int(found[7]),workload=found[9],name=found[11]))
                check([v['id'] for v in profile_rows] == list(range(1,len(profile_rows)+1)), 'complete_profile_row_sequence')
                total = re.search(r'^Total\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$', text, re.M)
                expected = [sum(v['microseconds'] for v in profile_rows if v['target'] == target) for target in ('CPU','GPU','NPU')]
                check(total is not None and list(map(int,total.groups())) == expected+[sum(expected)], 'profile_time_summary')
    repeat = all(np.array_equal(all_values[0][name], value[name]) for value in all_values for name in SDK_NAMES)
    check(repeat, 'all_five_normal_profile_outputs_bitwise_equal')
    normal = [row for row in calls if row['mode'] == 'normal']; warm = normal[1:]
    mean = float(np.mean([row['forward_seconds'] for row in warm]))
    saved_mean = json.loads((args.normal_dir/'run.json').read_text())['timing_summary']['warm_mean_forward_seconds']
    check(mean == saved_mean, 'ordinary_warm_mean_recomputed')
    internal_cpu = [row for row in profile_rows if row['target'] == 'CPU' and row['op'] not in ('InputOperator','OutputOperator')]
    public_lifecycle = {role: {key:value[key] for key in ('sdk_run_api_calls','completed_calls','SDK_outputs_releases','closed',
        'context_destroyed','current_DMA_bytes','current_model_host_bytes','memory','cleanup_errors')} for role,value in lifecycle.items()}
    return dict(status='passed_saved_grouped_initial_identity_FP32_rejected' if not failures else 'failed',
        script_sha256=sha(__file__),checks=dict(count=count,failed=len(failures),failures=failures),candidate_accepted=False,
        identity=dict(model_key='B8_D8_M48',complete_original_edges=3072,manifest_sha256=MANIFEST_SHA,
            compile_sha256=sha(args.compile_dir/'compile.json'),model_sha256=compiled['output']['sha256'],local_source_sha256=source_hashes),
        raw_receipts=receipts,saved_artifacts=artifacts,numerical=dict(original_FP32_gate_passed=False,
            all_five_normal_profile_outputs_bitwise_equal=repeat,first_call_versus_references=calls[0]['versus_references']),
        layout=dict(logical_inputs=manifest['models']['B8_D8_M48']['inputs'],native_input_attributes=lifecycle['normal']['input_attributes'],
            native_output_attributes=lifecycle['normal']['output_attributes'],input_CF_to_NHWC_pack_verified=True,
            all_five_inverse_scatter_bitwise_verified=True,want_float_owned_copies=True),
        timing=dict(normal_forward_seconds=[v['forward_seconds'] for v in normal],ordinary_warm_mean_forward_seconds=mean,
            ordinary_warm_mean_stages={key:float(np.mean([v['timing'][key] for v in warm])) for key in stages},
            normal_restore_edge_order_seconds=[v['restore_edge_order_seconds'] for v in normal],
            profile_forward_seconds=[v['forward_seconds'] for v in calls if v['mode']=='profile'],
            scope='Normal forward includes input finite scan/owned NHWC packing/set/run/get/owned CF copy/release; excludes topology, init, saved-reference comparisons and original-E scatter. Profile is separate diagnostic.'),
        operator_coverage=dict(profile_raw_sha256=performance['raw_sha256'],rows=len(profile_rows),
            target_counts={target:sum(v['target']==target for v in profile_rows) for target in ('CPU','NPU','GPU')},
            summed_microseconds={target:sum(v['microseconds'] for v in profile_rows if v['target']==target) for target in ('CPU','NPU','GPU')},
            internal_CPU=internal_cpu,CPU_IO_count=sum(v['op'] in ('InputOperator','OutputOperator') and v['target']=='CPU' for v in profile_rows),
            hottest=sorted(profile_rows,key=lambda v:v['microseconds'],reverse=True)[:8],all_internal_operators_NPU=False),
        lifecycle=public_lifecycle,limits=[
            'One complete initial graph only; 13 other shapes, evolving graphs and bounded-cache hardware eviction remain untested.',
            'Both inherited CPU/ORT and actual SDK strict FP32 gates remain rejected; no recurrent VO commit or adoption.',
            'No BA, fresh-image encoder, trajectory or 20Hz/100ms whole-path acceptance from this component timing.',
            'Remote before/after source pins and SDK calls/release/destroy are recorded evidence; host rehashes available selected model/source/artifact files, not unavailable board source paths.',
            'Profile operator summed times are diagnostic and are not added to ordinary forward timing.',
            'ARM/x86 relative-L2 scalar rounding tolerance is separate from the unchanged exact elementwise strict gate.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('normal-dir','profile-dir','export-dir','capture-dir','compile-dir','sdk-dir','driver','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('New audit receipt required')
    report = audit(args)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],checks=report['checks'],ordinary_warm_seconds=report['timing']['ordinary_warm_mean_forward_seconds'],
        target_counts=report['operator_coverage']['target_counts'],output=args.output.name)),flush=True)
    return int(report['status']=='failed')


if __name__ == '__main__':
    raise SystemExit(main())
