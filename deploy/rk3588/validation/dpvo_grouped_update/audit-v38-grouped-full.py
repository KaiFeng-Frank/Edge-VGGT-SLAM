#!/usr/bin/env python3
"""Read saved full-graph NumPy arrays/JSON only; never execute a neural backend.

Independent complete-source indices recover CF group bytes from each full
output. Remote SDK/source lifecycle is recorded evidence, not host observation.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np

MANIFEST_SHA = '0e182dcb42825525fe1e7e6b84ffd5af50c3dc02a33f183087bff3963a653331'
INDEX_SHA = 'f92ea7663db60702790110e9ac446f9fafd78e5645050451565587e31beb1894'
DRIVER_SHA = '6da1b7431ff17626f592d4815ecddd883884437c94ec06d19dde76804f7a82b7'
SOURCES = {'v37_grouped_contracts.py': 'aa9e6af938f64ae4a4a82ffc84b487321fff208547703b0d65646c72b96d85c9',
    'compile-v37-grouped-update.py': 'fb3a6e9eed9a4bc3f8490f0cf61c7ef5d5f4f4eff065bb9d9f057de97fb9cdd1',
    'v37_grouped_rknn.py': '55d7b04e782a687fedb6475b8388a12d52361cc56b1a7f260537bd3e1d679d1d'}
NAMES = ('net', 'delta', 'weight')
SDK_NAMES = ('net_out', 'delta', 'weight')
STAGES = ('pack_seconds', 'input_set_seconds', 'run_api_seconds', 'outputs_get_seconds',
          'owned_copy_seconds', 'outputs_release_seconds')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def raw(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes,
        tensor_sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())


def numerical(actual, reference):
    if actual.shape != reference.shape or actual.dtype != reference.dtype or actual.dtype != np.float32:
        raise ValueError('Comparable FP32 output shapes required')
    finite = bool(np.isfinite(actual).all() and np.isfinite(reference).all())
    if not finite:
        return dict(finite=False, gate_passed=False, failed_elements=None, relative_L2=None,
            max_abs=None, nonfinite_elements=int((~np.isfinite(actual)).sum()), rtol=1e-4, atol=1e-5)
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    error = a-b; failures = int(np.count_nonzero(np.abs(error) > 1e-5+1e-4*np.abs(b)))
    norm = float(np.linalg.norm(b.ravel()))
    return dict(finite=True, gate_passed=failures == 0, failed_elements=failures,
        relative_L2=float(np.linalg.norm(error.ravel())/norm) if norm else None,
        max_abs=float(np.abs(error).max()), nonfinite_elements=0, rtol=1e-4, atol=1e-5)


def independent_groups(ii, jj, kk, patches=48):
    """Sort complete source/target/patch tuples, preserving original edge indices."""
    if not all(v.dtype == np.int64 and v.ndim == 1 and v.shape == ii.shape for v in (ii,jj,kk)):
        raise ValueError('Int64 original edge vectors required')
    if ii.size == 0 or not np.array_equal(ii, kk//patches) or min(ii.min(), jj.min(), kk.min()) < 0:
        raise ValueError('Original patch/source identity differs')
    by_degree = {}
    for source in np.unique(ii):
        edges = np.flatnonzero(ii == source)
        order = edges[np.lexsort((kk[edges] % patches, jj[edges]))]
        targets = np.unique(jj[edges]); degree = targets.size
        if order.size != degree*patches:
            raise ValueError('Incomplete source or duplicate patch/target')
        indices = order.reshape(degree, patches)
        if not np.array_equal(jj[indices], np.broadcast_to(targets[:,None], indices.shape)) or not np.array_equal(
                kk[indices] % patches, np.broadcast_to(np.arange(patches), indices.shape)):
            raise ValueError('Incomplete source target/patch order')
        by_degree.setdefault(degree, []).append((int(source), targets.tolist(), indices))
    result = []
    for degree, members in sorted(by_degree.items()):
        result.append(dict(key=f'B{len(members)}_D{degree}_M{patches}', sources=[v[0] for v in members],
            targets=[v[1] for v in members], indices=np.stack([v[2] for v in members])))
    covered = np.concatenate([v['indices'].ravel() for v in result])
    if not np.array_equal(np.sort(covered), np.arange(ii.size)):
        raise ValueError('Original edges omitted or duplicated')
    return result


def parse_profile(record):
    text = record['raw_text']; data = text.encode('utf-8')
    if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['raw_sha256']:
        raise ValueError('Actual SDK profile text SHA differs')
    if 'ID   OpType' not in text or 'Time(us)' not in text or 'WorkLoad(0/1/2)' not in text:
        raise ValueError('Unknown profile table header')
    pattern = re.compile(r'^\s*(\d+)\s+(\S+)\s+(\S+)\s+(CPU|NPU|GPU)\s+(.+?)\s+(\d+/\d+/\d+)\s+(\d+)\s+(.*?)\s+(\d+(?:\.\d+)?%/\d+(?:\.\d+)?%/\d+(?:\.\d+)?%)\s+(\d+)\s+(.+?)\s*$')
    rows = []
    for line in text.splitlines():
        if not re.match(r'^\s*\d+\s', line): continue
        match = pattern.match(line)
        if match is None: raise ValueError('Unparsed actual profile row')
        rows.append(dict(id=int(match[1]),op=match[2],dtype=match[3],target=match[4],
            microseconds=int(match[7]),workload=match[9],name=match[11]))
    if not rows or [v['id'] for v in rows] != list(range(1,len(rows)+1)):
        raise ValueError('Incomplete profile row sequence')
    total = re.search(r'^Total\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$', text, re.M)
    sums = [sum(v['microseconds'] for v in rows if v['target'] == t) for t in ('CPU','GPU','NPU')]
    if total is None or list(map(int,total.groups())) != sums+[sum(sums)]:
        raise ValueError('Profile total disagrees with parsed rows')
    internal = [v for v in rows if v['op'] not in ('InputOperator','OutputOperator')]
    return dict(raw_sha256=record['raw_sha256'],rows=len(rows),
        internal_target_counts=dict(Counter(v['target'] for v in internal)),
        CPU_IO_rows=sum(v['op'] in ('InputOperator','OutputOperator') for v in rows),
        target_microseconds=dict(zip(('CPU','GPU','NPU'),sums)),total_microseconds=sum(sums),
        internal_CPU=[v for v in internal if v['target']=='CPU'],
        largest_NPU=sorted((v for v in internal if v['target']=='NPU'),key=lambda v:v['microseconds'],reverse=True)[:3])


def audit(args):
    failures, count, artifacts = [], 0, []
    def check(value, name):
        nonlocal count
        count += 1
        if not value: failures.append(name)
    def read_arrays(directory, artifact):
        rel = Path(artifact['file'])
        if rel.is_absolute() or '..' in rel.parts: raise ValueError('Unsafe artifact path')
        path = directory/rel
        check(path.stat().st_size == artifact['bytes'] and sha(path) == artifact['sha256'], 'artifact:'+rel.name)
        with np.load(path,allow_pickle=False) as archive:
            check(set(archive.files) == set(artifact['arrays']), 'array_keys:'+rel.name)
            result = {k:archive[k] for k in archive.files}
        for name, value in result.items():
            meta = raw(value)
            check(all(meta[k] == artifact['arrays'][name][k] for k in meta), 'array_bytes:'+rel.name+':'+name)
            if 'finite_elements' in artifact['arrays'][name]:
                check(int(np.isfinite(value).sum()) == artifact['arrays'][name]['finite_elements']
                    and int((~np.isfinite(value)).sum()) == artifact['arrays'][name]['nonfinite_elements'], 'finite_counts:'+rel.name+':'+name)
        artifacts.append(dict(file=rel.as_posix(),bytes=artifact['bytes'],sha256=sha(path)))
        return result
    def match_metrics(actual, recorded, label):
        for key, value in actual.items():
            other = recorded[key]
            check(value == other or (key == 'relative_L2' and value is not None and other is not None
                and np.isclose(value,other,rtol=1e-10,atol=1e-14)),label+':'+key)
    def log(record, label):
        data = record['raw_text'].encode('utf-8')
        check(len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['raw_sha256']
            and not record['truncated'] and not record['RKNN_error_detected'] and 'E RKNN' not in record['raw_text'],label)
    mp = args.export_dir/'manifest.json'; manifest = json.loads(mp.read_text())
    ip = args.compile_index; index = json.loads(ip.read_text())
    check(sha(mp) == MANIFEST_SHA and sha(ip) == INDEX_SHA,'frozen_manifest_and_index')
    check(manifest['strict_FP32_gate_passed'] is False and manifest['candidate_accepted'] is False
        and manifest['original_parameters_unchanged'] and manifest['original_weights_before'] == manifest['original_weights_after'], 'rejected_original_parameter_ancestor')
    check(index['manifest_sha256'] == MANIFEST_SHA and index['compiler_sha256'] == SOURCES['compile-v37-grouped-update.py']
        and index['contracts_sha256'] == SOURCES['v37_grouped_contracts.py'] and len(index['models']) == 14
        and set(index['models']) == set(manifest['models']), 'fourteen_actual_models')
    local_sources = {name:sha(args.sdk_dir/name) for name in SOURCES}
    check(local_sources == SOURCES and sha(args.driver) == DRIVER_SHA, 'frozen_local_sources')
    models = {}
    for key, item in index['models'].items():
        directory = args.reused_compile_dir if key == 'B8_D8_M48' else args.compile_root/Path(item['compile_dir']).name
        cp = directory/'compile.json'; compiled = json.loads(cp.read_text()); model = directory/item['model']['file']
        check(sha(cp) == item['compile_sha256'] and compiled['model_key'] == key and compiled['output'] == item['model']
            and compiled['manifest_sha256'] == MANIFEST_SHA and compiled['input']['sha256'] == manifest['models'][key]['model']['sha256'], 'compile_chain:'+key)
        check(compiled['configuration'] == dict(target_platform='rk3588',float_dtype='float16',optimization_level=3,do_quantization=False)
            and compiled['script_sha256'] == SOURCES['compile-v37-grouped-update.py'] and compiled['released']
            and not compiled.get('release_error') and compiled['candidate_accepted'] is False
            and compiled['source_pins'] == compiled['source_hashes_after'] and compiled['sources_and_inputs_unchanged'], 'compile_scope:'+key)
        check(model.stat().st_size == item['model']['bytes'] and sha(model) == item['model']['sha256'],'compiled_model_bytes:'+key)
        models[key] = dict(compile_sha256=sha(cp),model_sha256=sha(model),model_bytes=item['model']['bytes'])
    reports, receipt_hashes, cache_summary, timing, profiles = {}, {}, {}, {}, []
    for role in ('normal','profile'):
        directory = getattr(args,role+'_dir'); rp = directory/'run.json'; report = json.loads(rp.read_text())
        reports[role] = report; receipt_hashes[role] = sha(rp)
        expected_calls = [(o,n) for o in (13,127,191) for n in range(3 if role=='normal' else 1)]
        check([(r['observation_id'],r['repeat_index']) for r in report['calls']] == expected_calls,role+':complete_fixed_schedule')
        check(report['status'] == 'completed_three_full_graphs_diagnostic_not_VO' and report['script_sha256'] == DRIVER_SHA
            and report['runtime_sha256'] == SOURCES['v37_grouped_rknn.py'] and report['export_manifest_sha256'] == MANIFEST_SHA
            and report['compile_index_sha256'] == INDEX_SHA and report['compiled_models'] == index['models'],role+':source_chain')
        check(report['candidate_accepted'] is False and report['VO_state_committed'] is False and report['actual_Torch_OR_ORT_calls'] == 0
            and report['actual_original_FP32_gate_passed'] is False and report['inherited_CPU_ORT_FP32_gate_passed'] is False
            and report['collect_perf_detail'] == (role=='profile'),role+':scope_and_rejection')
        check(report['cleanup_errors'] == [] and report['affinity_restored'] and report['sources_and_inputs_unchanged']
            and report['source_pins'] == report['source_hashes_after'],role+':reported_sources_and_cleanup')
        keys = [g['model_key'] for r in report['calls'] for g in r['groups']]; calls_per_key = Counter(keys)
        cache = report['cache']; history = {h['model_key']:h['metrics'] for h in cache['histories']}
        check(len(history) == len(cache['histories']) == len(models) == cache['created'] == 14
            and cache['hits'] == len(keys)-14 and cache['evicted'] == 0,role+':cache_sequence_counts')
        check(cache['closed'] and cache['cleanup_errors'] == [] and cache['resident'] == [] and cache['sessions'] == {}
            and cache['resident_contexts'] == cache['current_DMA_bytes'] == cache['current_model_host_bytes'] == 0,role+':cache_closed_zero')
        check(cache['peak_resident_contexts'] == 14 and cache['peak_resident_model_bytes'] == sum(v['model_bytes']+1 for v in models.values())
            and cache['peak_resident_DMA_bytes'] == sum(h['memory']['total_dma_allocated_size'] for h in history.values())
            and cache['peak_resident_DMA_bytes'] <= cache['max_dma_bytes'] and cache['peak_resident_model_bytes'] <= cache['max_model_bytes'],role+':resident_memory')
        for key,h in history.items():
            check(h['sdk_run_api_calls'] == h['completed_calls'] == h['SDK_outputs_releases'] == h['SDK_outputs_release_attempts'] == calls_per_key[key]
                and h['closed'] and h['context_destroyed'] and h['cleanup_errors'] == [] and h['npu_used'] and not h['synthetic']
                and h['current_DMA_bytes'] == h['current_model_host_bytes'] == 0 and h['core_mask'] == 7 and h['init_flags'] == (8 if role=='profile' else 0)
                and h['source_pins'] == h['source_hashes_after'] and h['sources_and_inputs_unchanged'],role+':session_lifecycle:'+key)
            log(h['initialization_native_log'],role+':init_log:'+key)
            for output,attr in zip(manifest['models'][key]['outputs'],h['output_attributes']):
                check(attr['name'] == output['name'] and attr['shape'] == output['shape'] and attr['format'] == 0
                    and attr['type'] == 1 and attr['n_elems'] == int(np.prod(output['shape'])) and attr['size'] == attr['n_elems']*2
                    and attr['size_with_stride'] >= attr['size'],role+':native_output_attr:'+key+':'+output['name'])
            models[key][role+'_memory'] = h['memory']
        check(set(history) == set(models) and len(report['initialization_events']) == 14 and all(v['status']=='initialized'
            and v['memory']==history[v['model_key']]['memory'] for v in report['initialization_events']),role+':initialization_events')
        cache_summary[role] = {k:cache[k] for k in ('created','hits','evicted','peak_resident_contexts','peak_resident_model_bytes',
            'peak_resident_DMA_bytes','peak_transient_init_DMA_bytes','closed','resident_contexts','current_DMA_bytes','current_model_host_bytes')}
        cache_summary[role].update(sdk_calls=len(keys),outputs_releases=sum(v['SDK_outputs_releases'] for v in history.values()),
            destroyed_contexts=sum(v['context_destroyed'] for v in history.values()),peak_RSS_MiB=report['peak_rss_mib'])
    repeat, representative = {}, {}
    for case in manifest['cases']:
        observation = case['observation_id']; original = read_arrays(args.capture_dir,case['original_inputs_artifact'])
        refs = dict(original_Torch=read_arrays(args.capture_dir,case['original_gold_artifact']),
            grouped_Torch=read_arrays(args.export_dir,case['grouped_Torch_artifact']),grouped_ORT=read_arrays(args.export_dir,case['grouped_ORT_artifact']))
        groups = independent_groups(*(original[k] for k in ('ii','jj','kk')))
        check([v['key'] for v in groups] == [v['model_key'] for v in case['groups']], 'independent_groups:'+str(observation))
        for group,saved in zip(groups,case['groups']):
            check(raw(group['indices']) == saved['canonical_edge_indices'] and group['sources'] == saved['topology']['source_slots']
                and group['targets'] == saved['topology']['target_slots'], 'original_edge_mapping:'+str(observation)+':'+group['key'])
            feed = read_arrays(args.export_dir,saved['inputs_artifact'])
            for name in ('net','inp','corr'):
                check(np.array_equal(original[name][0,group['indices'],:].transpose(0,3,1,2),feed[name]),'same_complete_group_input:'+str(observation)+':'+group['key']+':'+name)
        first = None; repeat[observation] = True; representative[observation] = {}; timing[observation] = {}
        for role in ('normal','profile'):
            rows = [v for v in reports[role]['calls'] if v['observation_id'] == observation]
            time_rows = []
            for row in rows:
                label = role+':'+str(observation)+':'+str(row['repeat_index']); values = read_arrays(getattr(args,role+'_dir'),row['full_order_artifact'])
                check(set(values)==set(NAMES) and all(v.dtype==np.float32 and v.shape==(1,case['edges'],c) and np.isfinite(v).all()
                    for v,c in zip((values[k] for k in NAMES),(384,2,2))) and np.all((values['weight']>=0)&(values['weight']<=1)),label+':finite_shapes_domain')
                if first is None: first = values
                else: repeat[observation] &= all(np.array_equal(first[k],values[k]) for k in NAMES)
                comparisons = {kind:{k:numerical(values[k],reference[k]) for k in NAMES} for kind,reference in refs.items()}
                for kind,metrics in comparisons.items():
                    for k,v in metrics.items(): match_metrics(v,row['versus_saved_references'][kind][k],label+':'+kind+':'+k)
                check(row['strict_FP32_gate_passed'] == all(v['gate_passed'] for v in comparisons['original_Torch'].values()),label+':unchanged_element_gate')
                check(row['degree_plan'] == case['degree_plan'] and row['snapshot_identity'] == case['snapshot_identity'] and row['full_graph_returned']
                    and row['status']=='completed_diagnostic_not_adopted' and row['candidate_accepted'] is False and row['VO_state_committed'] is False
                    and not row['padding'] and not row['dropped_edges'] and row['outputs_finite'] and row['weight_domain_valid']
                    and row['sequential_groups']==len(groups)==len(row['groups']),label+':complete_atomic_diagnostic')
                sums = Counter(); forward = 0; cache_call = 0
                for group,saved in zip(groups,row['groups']):
                    record = saved['record']; t = record['timing']
                    check(saved['model_key']==group['key'] and record['execution_status']=='completed' and record['outputs_released']
                        and record['outputs_finite'] and record['weight_domain_valid'] and not record['all_outputs_zero']
                        and record['strict_FP32_gate_passed'] is None and record['candidate_accepted'] is False,label+':group_execution:'+group['key'])
                    for sdk,canonical in zip(SDK_NAMES,NAMES):
                        recovered = values[canonical][0,group['indices'],:].transpose(0,3,1,2); meta = raw(recovered)
                        check(all(meta[k]==saved['raw_outputs'][sdk][k] for k in meta),label+':independent_group_restore_raw_SHA:'+group['key']+':'+sdk)
                    for input_name,transport in zip(('net','inp','corr'),t['input_records']):
                        shape = manifest['models'][group['key']]['inputs'][('net','inp','corr').index(input_name)]['shape']
                        check(transport['name']==input_name and transport['logical_shape']==shape and transport['buffer_shape']==[shape[i] for i in (0,2,3,1)]
                            and transport['native_attribute_format']==transport['request_format']==1 and transport['source_dtype']==transport['transport_dtype']=='float32'
                            and transport['pass_through']==0 and transport['bytes']==4*int(np.prod(shape)) and transport['owned'] and transport['explicit_NHWC_pack'],label+':input_pack:'+group['key']+':'+input_name)
                    check(t['synchronous'] and all(t[s]>=0 for s in STAGES) and sum(t[s] for s in STAGES)<=t['forward_seconds']
                        and t['forward_seconds']<=record['total_execute_seconds']<=record['cache_call_seconds'],label+':nested_group_timing:'+group['key'])
                    log(record['native_log'],label+':native_log:'+group['key'])
                    for s in STAGES: sums[s] += t[s]
                    forward += t['forward_seconds']; cache_call += record['cache_call_seconds']
                    if role=='profile': profiles.append(dict(observation_id=observation,model_key=group['key'],**parse_profile(record['profile'])))
                check(forward==row['summed_nested_SDK_forward_seconds'] and cache_call==row['summed_nested_cache_call_seconds']
                    and cache_call<=row['full_graph_seconds']<=row['full_wall_seconds'],label+':whole_vs_nested_timing')
                time_rows.append(dict(repeat_index=row['repeat_index'],full_wall_seconds=row['full_wall_seconds'],
                    nested_SDK_forward_seconds=forward,nested_cache_seconds=cache_call,nested_SDK_stages=dict(sums),
                    new_contexts=len(row['new_initializations']),outside_reference_statistics_seconds=row['outside_forward_statistics_seconds'],outside_save_seconds=row['outside_forward_save_seconds']))
                if role=='normal' and row['repeat_index']==0: representative[observation] = comparisons
            timing[observation][role] = time_rows
            if role=='normal':
                mean = sum(v['full_wall_seconds'] for v in time_rows[1:])/2
                check(mean==reports[role]['timing_summary'][str(observation)]['warm_mean_full_wall_seconds'],'warm_mean:'+str(observation))
                timing[observation]['ordinary_warm_mean_full_wall_seconds'] = mean
                timing[observation]['ordinary_warm_mean_nested_SDK_stages'] = {s:sum(v['nested_SDK_stages'][s] for v in time_rows[1:])/2 for s in STAGES}
        check(repeat[observation],'all_four_normal_profile_full_outputs_bitwise:'+str(observation))
    return dict(status='passed_saved_full_graph_identity_FP32_rejected' if not failures else 'failed',script_sha256=sha(__file__),
        checks=dict(count=count,failed=len(failures),failures=failures),candidate_accepted=False,VO_state_committed=False,
        identity=dict(manifest_sha256=MANIFEST_SHA,compile_index_sha256=INDEX_SHA,driver_sha256=DRIVER_SHA,local_SDK_sources=local_sources,
            compiled_models=models,total_compiled_model_bytes=sum(v['model_bytes'] for v in models.values())),
        raw_receipts=receipt_hashes,saved_artifacts=artifacts,
        numerical=dict(strict_FP32_gate_passed=False,thresholds=dict(rtol=1e-4,atol=1e-5),
            four_normal_profile_outputs_bitwise_by_observation=repeat,representative_versus_all_three_references=representative),
        cache_and_lifecycle=cache_summary,timing=timing,actual_profiles=profiles,
        limitations=['Only three fixed captured original Update graphs; no recurrent feedback, BA, trajectory or 20Hz acceptance.',
            'Numerical rejection is preserved; finite/domain-valid SDK execution is not original FP32 quality acceptance.',
            'Independent indices reconstruct per-group CF raw hashes from saved full arrays; separate group payloads were not saved.',
            'SDK copy/release/destroy, remote source stability and memory are recorded board evidence; host independently verifies saved bytes and local compile/source hashes.',
            'Profile operator sums describe separate diagnostic calls and are not normal wholewall latency or additive acceleration promises.',
            'Host compile directories are explicitly mapped from staged board basenames; this does not imply board paths exist on host.',
            'Diagnostic relative-L2 scalar reductions may round across ARM/x86; strict element thresholds and failed counts are unchanged.'])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('normal-dir','profile-dir','export-dir','capture-dir','compile-index','compile-root','reused-compile-dir','sdk-dir','driver','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args(argv)
    if args.output.exists(): raise FileExistsError(args.output)
    try: report=audit(args)
    except Exception as error: report=dict(status='failed',error=repr(error),script_sha256=sha(__file__),candidate_accepted=False)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report.get(k) for k in ('status','checks','error')},allow_nan=False))
    return int(report['status']=='failed')


if __name__=='__main__': raise SystemExit(main())
