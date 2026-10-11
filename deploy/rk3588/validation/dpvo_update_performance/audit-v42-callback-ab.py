#!/usr/bin/env python3
"""Independent NumPy saved-array audit of the board callback ABBA experiment."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
PARENT_RUN_SHA = '78f15e36e5444dfe17a70610eb4410f0605c262eb7102bc2fee39ba5cfdf427b'
HARNESS_SHA = '434e7d8bbd13c14833368cbc7c42da5266c48353936b7a0c741dfe8a0b81fea2'
INDEX_SHA = '8c3f1c3b92e14a6742a984f750a5eb4c1bbf6397ca31440b96b0768333dc21fa'
CALLBACK_SHA = dict(A='c7caef0c76673cd4511aa1a4c8a8a5427bfee691e1964227d782d0d6f26554eb',
                    B='4152c78b9f5942115fd3f754744fff234c788550806296e16dfb53b062038054')
INPUTS = ('net','inp','corr','ii','jj','kk')
OUTPUTS = ('net','delta','weight')
OBS = (34,127,191)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for value in iter(lambda: stream.read(1 << 20), b''):digest.update(value)
    return digest.hexdigest()


def meta(value):
    return dict(shape=list(value.shape),dtype=str(value.dtype),bytes=value.nbytes,
                tensor_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())


def literal_equal(actual, reference):
    return actual.shape == reference.shape and actual.dtype == reference.dtype and actual.tobytes(order='C') == reference.tobytes(order='C')


def validate_schedule(rows):
    if len(rows) != 24:raise ValueError('Exactly 24 fixed calls required')
    expected = []
    for obs in OBS:
        seen = dict(A=0,B=0)
        for cycle in range(2):
            for position,method in enumerate('ABBA'):
                repeat = seen[method];seen[method] += 1
                expected.append(dict(observation_id=obs,cycle=cycle,position=position,method=method,
                    state_method_repeat_index=repeat,excluded_first_state_method_call=repeat == 0))
    for row,wanted in zip(rows,expected):
        if any(row.get(k) != v for k,v in wanted.items()):raise ValueError('Fixed ABBA schedule/cold selection differs')


def audit(args):
    checks = 0;files = {}
    def check(value,reason):
        nonlocal checks
        checks += 1
        if not value:raise ValueError(reason)
    def read(path,expected,label):
        check(sha(path) == expected,'Receipt SHA: '+label);files[label] = expected
        return json.loads(path.read_text())
    def arrays(directory,artifact,label):
        check(Path(artifact['file']).name == artifact['file'],'Artifact basename: '+label)
        path = directory/artifact['file']
        check(not path.is_symlink() and path.stat().st_size == artifact['bytes'] and sha(path) == artifact['sha256'], 'Artifact file bytes/SHA: '+label)
        files[label] = artifact['sha256']
        with np.load(path,allow_pickle=False) as bundle:
            check(set(bundle.files) == set(artifact['arrays']),'Artifact keys: '+label)
            result = {n:np.array(bundle[n],copy=True,order='C') for n in bundle.files}
        for n,v in result.items():check(meta(v) == artifact['arrays'][n] and np.isfinite(v).all(),'Raw metadata/finite: '+label+'/'+n)
        return result
    run = read(args.run_dir/'run.json',args.run_sha256,'AB/run.json')
    parent = read(args.inputs_dir/'run.json',PARENT_RUN_SHA,'original/run.json')
    check(sha(args.harness) == HARNESS_SHA,'Frozen actual harness bytes')
    check(run['schema'] == 'v42_saved_three_state_callback_metadata_ABBA_diagnostic'
          and run['status'] == 'completed_AB_bitwise_equal_diagnostic_not_adopted'
          and run['script_sha256'] == HARNESS_SHA and run['source_run_sha256'] == PARENT_RUN_SHA
          and run['compile_index_sha256'] == INDEX_SHA and run['callback_sha256'] == CALLBACK_SHA,
          'Exact experiment/script/callback/index lineage')
    check(run['candidate_accepted'] is False and run['historical_SDK_strict_FP32_rejected'] is True
          and run['strict_FP32_equivalence_validated'] is False and run['CPU_teacher_used'] is False
          and run['recurrent_feedback'] is False and run['BA_executed'] is False and run['VO_state_committed'] is False
          and run['realtime_validated'] is False and run['collect_perf_detail'] is False,'Diagnostic parity is not original FP32 or online acceptance')
    check(run['cleanup_errors'] == [] and run['affinity_restored'] is True
          and run['sources_and_inputs_unchanged'] is True and run['source_pins'] == run['source_hashes_after'],
          'Recorded experiment source/cleanup lifecycle')
    captures = parent['update_captures']
    check([r['frame_index'] for r in captures] == list(OBS),'Three frozen owning captures required')
    originals = {};input_summary = {}
    for actual,source in zip(run['input_captures'],captures):
        obs = source['frame_index'];values = arrays(args.inputs_dir,source['input'],f'original/input-{obs}.npz')
        check(actual['observation_id'] == obs and actual['saved_update_index'] == source['update_index']
              and actual['edges'] == source['edges'] and actual['input_artifact'] == source['input']
              and actual['selected_inputs'] == list(INPUTS)
              and actual['ignored_verified_artifact_keys'] == sorted(set(values)-set(INPUTS)), 'Original same-state six-input binding')
        check(set(INPUTS) <= set(values) and np.array_equal(values['ii'],values['kk']//16),'Original M16 topology association')
        originals[obs] = values
        input_summary[str(obs)] = dict(saved_update_index=source['update_index'],edges=source['edges'],file_sha256=source['input']['sha256'])
    validate_schedule(run['calls']);validate_schedule(run['schedule'])
    check(len(run['input_captures']) == 3,'Exactly three input descriptors')
    first = {};sdk_counts=Counter();shapes = dict(A=Counter(),B=Counter());local_index = Counter();first_hashes = {}
    for order,row in enumerate(run['calls']):
        obs,method = row['observation_id'],row['method'];record = row['callback_record'];full = originals[obs];e = full['ii'].size
        artifact = row['output_artifact']
        check(artifact['file'] == f'call-{order:03d}-{method}-obs{obs:03d}.npz','Fixed saved-output file/order')
        values = arrays(args.run_dir,artifact,f'AB/{artifact["file"]}')
        check(set(values) == set(OUTPUTS),'Three saved canonical outputs')
        for name,c in zip(OUTPUTS,(384,2,2)):
            check(values[name].dtype == np.float32 and values[name].shape == (1,e,c),'Actual complete original edge-order shape: '+name)
        check(((values['weight'] >= 0)&(values['weight'] <= 1)).all() and any(np.any(v) for v in values.values()),'Finite nonzero/unclipped weight domain')
        prior = first.setdefault(obs,{n:v.tobytes(order='C') for n,v in values.items()})
        check(all(values[n].tobytes(order='C') == prior[n] for n in OUTPUTS)
              and row['bitwise_equal_first_A_for_state'] is True,'Independent literal same-state A/B/repeat equality')
        first_hashes.setdefault(str(obs),{n:meta(v)['tensor_sha256'] for n,v in values.items()})
        check(row['CPU_reference_supplied'] is False and row['original_strict_FP32_gate_passed'] is False
              and row['candidate_accepted'] is False and record['index'] == local_index[method]
              and record['status'] == 'completed_diagnostic_not_adopted' and record['route'] == 'full'
              and record['NPU_used'] is True and record['Torch_fallback'] is False and record['fullgraph_returned'] is True
              and record['outputs_committed_by_callback'] is False and record['strict_FP32_gate_passed'] is None,
              'Actual diagnostic full call without teacher or state commit')
        local_index[method] += 1
        check(record['inputs'] == {n:dict(shape=list(full[n].shape),dtype=str(full[n].dtype),bytes=full[n].nbytes) for n in INPUTS},
              'Actual callback full input descriptor matches frozen NPZ')
        details = record['SDK_execution'];groups = details['groups'];timing = row['timing']
        check(details['sequential_groups'] == len(groups) and details['padding'] is False and details['dropped_edges'] is False
              and details['versus_reference'] is None and record['model_keys'] == [g['model_key'] for g in groups], 'No dropped/padded/reference graph groups')
        fields = tuple(timing['sum_SDK_intervals']);sums = {field:0. for field in fields}
        for group in groups:
            log = group['native_log'];t = group['timing'];key = group['model_key']
            check(group['execution_status'] == 'completed' and group['mode'] == 'normal' and group['outputs_released'] is True
                  and group['outputs_finite'] is True and group['weight_domain_valid'] is True and group['all_outputs_zero'] is False
                  and t['synchronous'] is True and log['truncated'] is False and log['RKNN_error_detected'] is False,
                  'SDK normal finite/log-free synchronous copied/released group')
            for field in fields:
                check(np.isfinite(t[field]) and t[field] >= 0,'Finite SDK timing');sums[field] += t[field]
            sdk_counts[method] += 1;shapes[method][key] += 1
        check(timing['SDK_calls'] == len(groups) and sums == timing['sum_SDK_intervals']
              and timing['full_graph_seconds'] == details['full_graph_seconds']
              and timing['internal_callback_seconds'] == record['callback_seconds']
              and 0 <= sums['forward_seconds'] <= timing['full_graph_seconds']+1e-6
              and timing['full_graph_seconds'] <= timing['internal_callback_seconds']+1e-6
              and timing['internal_callback_seconds'] <= timing['actual_callback_wall_seconds']+1e-6,
              'Recomputed nested SDK intervals; do not add containing clocks')
    warm = []
    for obs in OBS:
        summary = dict(observation_id=obs,methods={})
        recorded = next(v for v in run['warm_statistics'] if v['observation_id'] == obs)
        for method in ('A','B'):
            rows = [r for r in run['calls'] if r['observation_id'] == obs and r['method'] == method and not r['excluded_first_state_method_call']]
            check(len(rows) == 3,'Exactly three declared warm samples')
            def mean(key):return float(np.mean([r['timing'][key] for r in rows]))
            stats = dict(warm_calls=3,actual_callback_wall_mean_seconds=mean('actual_callback_wall_seconds'),
                internal_callback_mean_seconds=mean('internal_callback_seconds'),full_graph_mean_seconds=mean('full_graph_seconds'),
                sum_SDK_forward_mean_seconds=float(np.mean([r['timing']['sum_SDK_intervals']['forward_seconds'] for r in rows])))
            check(all(np.isclose(v,recorded['methods'][method][k],rtol=1e-12,atol=1e-15) for k,v in stats.items()),'Independent warm means match recorded numbers')
            summary['methods'][method] = stats
        warm.append(summary)
    lifecycle = {}
    for method in ('A','B'):
        owner = run['final_callbacks'][method];cache = owner['cache'];histories = cache['histories']
        check(owner['requests'] == owner['full_attempts'] == owner['full_completed'] == 12 and owner['motion_probe_attempts'] == 0
              and owner['failed_calls'] == 0 and owner['closed'] is True and owner['failed'] is False
              and owner['shape_histogram'] == dict(shapes[method]), 'Completed owner/call/shape counts')
        check(cache['closed'] is True and cache['resident_contexts'] == 0 and cache['current_DMA_bytes'] == cache['current_model_host_bytes'] == 0
              and cache['cleanup_errors'] == [] and len(histories) == cache['created'],'Closed cache capacity/lifecycle')
        calls,releases = 0,0
        for history in histories:
            m = history['metrics']
            check(m['synthetic'] is False and m['npu_used'] is True and m['closed'] is True and m['context_destroyed'] is True
                  and m['cleanup_errors'] == [] and m['current_DMA_bytes'] == m['current_model_host_bytes'] == 0
                  and m['collect_perf_detail'] is False and m['sources_and_inputs_unchanged'] is True
                  and m['sdk_run_api_calls'] == m['completed_calls'] == m['SDK_outputs_release_attempts'] == m['SDK_outputs_releases'],
                  'Actual native context completion/releases/destroy/source declarations')
            calls += m['sdk_run_api_calls'];releases += m['SDK_outputs_releases']
        check(calls == releases == sdk_counts[method] and run['lifecycle'][method]['sdk_run_api_calls'] == calls
              and run['lifecycle'][method]['outputs_released'] == releases and run['lifecycle'][method]['contexts_destroyed'] == len(histories),
              'Independent per-group and native lifecycle totals agree')
        lifecycle[method] = dict(callbacks=12,sdk_calls=calls,outputs_released=releases,contexts_destroyed=len(histories),
            current_DMA_bytes=0,current_model_host_bytes=0,peak_contexts=cache['peak_resident_contexts'],
            peak_DMA_bytes=cache['peak_resident_DMA_bytes'],peak_model_bytes=cache['peak_resident_model_bytes'])
    return dict(schema='v42_callback_AB_saved_independent_review',status='passed_saved_AB_identity_not_adopted',
        auditor_sha256=sha(__file__),checks=dict(count=checks,failures=0),file_sha256=files,
        callbacks=24,outputs=24,all_same_state_outputs_literal_bitwise_equal=True,inputs=input_summary,
        first_A_output_raw_sha256=first_hashes,warm_statistics=warm,lifecycle=lifecycle,
        candidate_accepted=False,historical_SDK_strict_FP32_rejected=True,
        limitations=['Only saved arrays/JSON; no model/ORT/SDK/BA execution in this auditor.',
                    'Each source NPZ and all 24 output NPZ file/raw hashes are independently checked. Per-call input contents are associated through frozen files and callback descriptors, not independent transport dumps.',
                    'Large board source-pin equality, native counters and cleanup are recorded declarations cross-checked against all group records; the auditor does not rehash every board file.',
                    'Warm n=3 per method/state; ABBA and observed platform samples do not imply controlled thermals, statistical significance or online speed.',
                    'A/B literal equality is not original CPU FP32 equivalence; historical rejection and no adoption remain.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir','inputs-dir','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--run-sha256',required=True)
    parser.add_argument('--harness',type=Path,default=Path(__file__).with_name('run-v42-callback-ab-board.py'))
    args = parser.parse_args(argv)
    if args.output.exists():raise ValueError('New output required')
    try:result = audit(args)
    except Exception as error:result = dict(status='failed_saved_audit',error=repr(error),auditor_sha256=sha(__file__))
    with args.output.open('x') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result.get(k) for k in ('status','checks','error')}))
    return int(result['status'] == 'failed_saved_audit')


if __name__ == '__main__':raise SystemExit(main())
