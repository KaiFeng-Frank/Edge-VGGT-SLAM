#!/usr/bin/env python3
"""Independent NumPy v43 saved full-state native-input ABBA audit; no inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
PARENT_RUN_SHA = '78f15e36e5444dfe17a70610eb4410f0605c262eb7102bc2fee39ba5cfdf427b'
PARENT_AUDITOR_SHA = 'cfa2812bfd369bae49e88280bbb164b5d60d33e1c6fdaf11c24a38464e3ff106'
HARNESS_SHA = '4c056f07828db7b5fa74914148e5a196840af2275988eb4f4104a459488fa9c2'
INDEX_SHA = 'eeb9241c9eab52039aa64fcf6546ce4624d1388756a9e919f42c22de4083a85c'
CALLBACK_SHA = dict.fromkeys(('A','B'),'4f4a5457dbdec5aebe1ddd6204a37c9073896ac99ed90abd3fbfe3d1f78db566')
MODEL_KEYS = {f'B{b}_D{d}_M16' for b in (1,2) for d in range(4,8)} | {'B1_D3_M16'}
NATIVE = 'native_FP16_NC1HWC2_C2_8_pass_through_1'
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



def dimensions(key):
    if key not in MODEL_KEYS: raise ValueError('Unknown nine-catalogue model key')
    b,d,m = (int(v[1:]) for v in key.split('_'))
    return b,d,m


def validate_native_queries(metrics,key):
    b,d,m = dimensions(key)
    queries = metrics['native_input_queries']
    if len(queries)!=6 or {(v['command'],v['input_index']) for v in queries}!={(q,i) for q in (8,10) for i in range(3)}:
        raise ValueError('Each initialized native context requires six actual queries')
    for index,(name,c) in enumerate(zip(('net','inp','corr'),(384,384,882))):
        pairs=[v for v in queries if v['input_index']==index]
        x,y=(v['attribute'] for v in pairs)
        expected=[b,(c+7)//8,d,m,8]; elements=int(np.prod(expected))
        if (any(v['return_code']!=0 for v in pairs) or x!=y or x['index']!=index
                or x['name']!=name or x['n_dims']!=5 or x['shape']!=expected
                or x['dims'][:5]!=expected or any(x['dims'][5:])
                or x['fmt']!=2 or x['type']!=1 or x['qnt_type']!=2
                or x['zp']!=0 or x['scale']!=1. or x['scale_finite'] is not True
                or x['n_elems']!=elements or x['size']!=elements*2
                or x['size_with_stride']!=elements*2 or x['w_stride']!=16 or x['h_stride']!=0):
            raise ValueError('Native query attrs differ from exact C2=8 FP16 contract')
        binding=next(v for v in metrics['input_bindings'] if v['name']==name)
        if (binding['native_input_attribute']!=x or binding['pass_through']!=1
                or binding['transport_format']!=2 or binding['transport_dtype']!='float16'
                or binding['native_query_commands']!=[8,10] or binding['transport_bytes']!=elements*2):
            raise ValueError('Native binding/query identity differs')
    log=metrics['native_input_query_log'];fpcr=metrics['packing_FPCR_at_initialization']
    if (metrics['native_inputs_validated'] is not True or metrics['input_transport']!=NATIVE
            or metrics['diagnostic_allow_native_fp16'] is not True or metrics['transport_equivalence_validated'] is not False
            or metrics['native_query_ancestor_sha256']!='89f0cead99df024b25ed3d2638e412a075f91d6736ecdb2b35ace2304eef0642'
            or log['RKNN_error_detected'] is not False or log['truncated'] is not False
            or fpcr['supported'] is not True or fpcr['FPCR_modified_by_packer'] is not False):
        raise ValueError('Native query/packing guards or diagnostic labels differ')
    return len(queries)


def validate_pack_records(group,method):
    b,d,m=dimensions(group['model_key']); rows=group['timing']['input_records']
    if len(rows)!=3:raise ValueError('Exactly three actual input pack descriptors required')
    for index,(row,name,c) in enumerate(zip(rows,('net','inp','corr'),(384,384,882))):
        if (row['index']!=index or row['name']!=name or row['logical_shape']!=[b,c,d,m]
                or row['source_dtype']!='float32' or row['owned'] is not True):
            raise ValueError('Logical pack shape/index/ownership differs')
        if method=='A':
            if (row['buffer_shape']!=[b,d,m,c] or row['native_attribute_format']!=1
                    or row['request_format']!=1 or row['transport_dtype']!='float32'
                    or row['pass_through']!=0 or row['bytes']!=b*c*d*m*4
                    or row['explicit_NHWC_pack'] is not True):
                raise ValueError('Original FP32 NHWC SDK conversion contract differs')
        elif method=='B':
            shape=[b,(c+7)//8,d,m,8]
            if (row['buffer_shape']!=shape or row['logical_attribute_format']!=1
                    or row['native_attribute_format']!=2 or row['native_query_commands']!=[8,10]
                    or row['request_format']!=2 or row['request_type']!=1
                    or row['transport_dtype']!='float16' or row['buffer_storage_dtype']!='uint16'
                    or row['pass_through']!=1 or row['bytes']!=int(np.prod(shape))*2
                    or row['input_transport']!=NATIVE or row['padded_channels']!=shape[1]*8-c
                    or row['tail_padding_value']!='FP16_positive_zero'
                    or row['SDK_input_conversion_requested'] is not False):
                raise ValueError('Native FP16 NC1HWC2 pass-through descriptor differs')
        else:raise ValueError('Unknown AB method')
    if (group['candidate_accepted'] is not False or group['strict_FP32_gate_passed'] is not None
            or group['strict_component_gate_passed'] is not False):
        raise ValueError('Input transport parity cannot change numerical rejection')
    return len(rows)


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
    check(run['schema'] == 'v43_native_three_complete_saved_state_ABBA_diagnostic'
          and run['status'] == 'completed_AB_bitwise_equal_diagnostic_not_adopted'
          and run['script_sha256'] == HARNESS_SHA and run['source_run_sha256'] == PARENT_RUN_SHA
          and run['compile_index_sha256'] == INDEX_SHA and run['callback_sha256'] == CALLBACK_SHA,
          'Exact experiment/script/callback/index lineage')
    check(run['parent_harness_sha256']=='434e7d8bbd13c14833368cbc7c42da5266c48353936b7a0c741dfe8a0b81fea2'
        and run['factory_sha256']=='e15ff495ac626e7a1a7271a918b39854b4cc45989b77e334db56d8234ea5f300'
        and run['same_callback_for_both_methods'] is True and run['only_input_transport_changed'] is True
        and run['single_AB_prerequisite_sha256']=='28185202278d0446b62525721338345ce279a88773969d45e9efb8f3f9fbcba2'
        and run['native_transport_bitwise_for_three_saved_states_only'] is True
        and run['methods']==dict(A='original_FP32_NHWC_pass_through_0',B=NATIVE),
        'Same pinned metadata callback/nine-model factory; only input transport changed')
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
    pack_counts=Counter();native_queries=0
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
            pack_counts[method] += validate_pack_records(group,method);checks += 1
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
            interval_means={field:float(np.mean([r['timing']['sum_SDK_intervals'][field] for r in rows])) for field in rows[0]['timing']['sum_SDK_intervals']}
            check(all(np.isclose(v,recorded['methods'][method]['sum_SDK_interval_means'][k],rtol=1e-12,atol=1e-15) for k,v in interval_means.items()), 'Independent warm individual SDK interval means')
            stats['sum_SDK_interval_means']=interval_means
            summary['methods'][method] = stats
        warm.append(summary)
    lifecycle = {}
    for method in ('A','B'):
        owner = run['final_callbacks'][method];cache = owner['cache'];histories = cache['histories']
        check(set(owner['model_keys'])==MODEL_KEYS and cache['max_contexts']==9,'Same nine catalogue entries; actual states use a subset')
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
            check(m['source_pins']==m['source_hashes_after'] and m['diagnostic_allow_fp32_rejected'] is True
                and m['candidate_accepted'] is False,'Context source/unchanged strict-rejection policy')
            if method=='B':native_queries += validate_native_queries(m,history['model_key']);checks += 1
            calls += m['sdk_run_api_calls'];releases += m['SDK_outputs_releases']
        check(calls == releases == sdk_counts[method] and run['lifecycle'][method]['sdk_run_api_calls'] == calls
              and run['lifecycle'][method]['outputs_released'] == releases and run['lifecycle'][method]['contexts_destroyed'] == len(histories),
              'Independent per-group and native lifecycle totals agree')
        lifecycle[method] = dict(callbacks=12,sdk_calls=calls,outputs_released=releases,contexts_destroyed=len(histories),
            current_DMA_bytes=0,current_model_host_bytes=0,peak_contexts=cache['peak_resident_contexts'],
            peak_DMA_bytes=cache['peak_resident_DMA_bytes'],peak_model_bytes=cache['peak_resident_model_bytes'])
    return dict(schema='v43_native_complete_callback_AB_saved_independent_review',status='passed_saved_AB_identity_not_adopted',
        auditor_sha256=sha(__file__),checks=dict(count=checks,failures=0),file_sha256=files,
        parent_auditor_sha256=PARENT_AUDITOR_SHA,callbacks=24,outputs=24,output_raw_arrays=72,
        original_saved_input_NPZ=3,nine_catalogue_models=sorted(MODEL_KEYS),
        same_metadata_callback_for_both_methods=True,native_query_calls=native_queries,
        pack_records_by_method=dict(pack_counts),
        all_same_state_outputs_literal_bitwise_equal=True,inputs=input_summary,
        first_A_output_raw_sha256=first_hashes,warm_statistics=warm,lifecycle=lifecycle,
        candidate_accepted=False,historical_SDK_strict_FP32_rejected=True,
        limitations=['Only saved arrays/JSON; no model/ORT/SDK/BA execution in this auditor.',
                    'Each source NPZ and all 24 output NPZ file/raw hashes are independently checked. Per-call input contents are associated through frozen files and callback descriptors, not independent transport dumps.',
                    'Large board source-pin equality, native counters and cleanup are recorded declarations cross-checked against all group records; the auditor does not rehash every board file.',
                    'Warm n=3 per method/state; ABBA and observed platform samples do not imply controlled thermals, statistical significance or online speed.',
                    'A/B literal equality is not original CPU FP32 equivalence; historical rejection and no adoption remain.',
                    'Six initialized contexts per method are checked. The nine-entry catalogue includes D3, but this experiment executes only the source34/127/191 six-model subset.',
                    'All recorded native query and pass-through pack descriptors are checked. Native packed payload bytes are not saved by this full AB experiment and are not independently re-converted here.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir','inputs-dir','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--run-sha256',required=True)
    parser.add_argument('--harness',type=Path,default=Path(__file__).with_name('run-v43-native-full-ab-board.py'))
    args = parser.parse_args(argv)
    if args.output.exists():raise ValueError('New output required')
    try:result = audit(args)
    except Exception as error:result = dict(status='failed_saved_audit',error=repr(error),auditor_sha256=sha(__file__))
    with args.output.open('x') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result.get(k) for k in ('status','checks','error')}))
    return int(result['status'] == 'failed_saved_audit')


if __name__ == '__main__':raise SystemExit(main())
