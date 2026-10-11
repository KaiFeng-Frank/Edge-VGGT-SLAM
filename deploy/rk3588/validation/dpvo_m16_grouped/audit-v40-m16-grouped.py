#!/usr/bin/env python3
"""Audit saved M16 arrays, receipts and profiles; never execute a neural backend.

The frozen v38 NumPy comparison/group/profile readers are reused by SHA. Local
payloads are independently checked; remote SDK lifecycle remains board evidence.
DPVO MIT attribution: Copyright (c) 2022 Princeton University.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
PARENT_AUDITOR_SHA = '4b369517e80892fea48744098d66d6148f1b2c3b1165ab0099e401a6fda32156'
MANIFEST_SHA = '5a9d6c571b672fbb174cf6adf9f8f578689fdd9e5f58ee4ec327320b55b62bfb'
ORIGINAL_RUN_SHA = 'dd9f37524b999c1e586182a92c226481fcfc7f4da851d5e88d62d82be1029ecd'
POLICY_SHA = 'cc6a31f89bdcff0b43c0d51266ad62c48d0e297346845a98d42bad5c8874ef4a'
SOURCES = {
    'v40_m16_grouped_contracts.py': '7be6bf750dabb8a3d88464fd1502bde3bf7690596a39791710cf41a9f8cb62fe',
    'compile-v40-m16-grouped-update.py': '3ccd9002247bf3ff15e87bf04db2e3462db782c6d5a637f50b98094bcd584294',
    'v40_m16_grouped_rknn.py': 'd4cdd1f9f21b04e44b6369c8ad7aea1fe4fbd43855b13993c6fdf4550f7bf0d7',
    'run-v40-m16-grouped-board.py': '49670a57dc19f4e053c937b5c2490a34d3b327cb18fe1f2ff28111858d511940',
    'v37_grouped_contracts.py': 'aa9e6af938f64ae4a4a82ffc84b487321fff208547703b0d65646c72b96d85c9',
    'compile-v37-grouped-update.py': 'fb3a6e9eed9a4bc3f8490f0cf61c7ef5d5f4f4eff065bb9d9f057de97fb9cdd1',
    'v37_grouped_rknn.py': '55d7b04e782a687fedb6475b8388a12d52361cc56b1a7f260537bd3e1d679d1d',
}
NAMES = ('net', 'delta', 'weight')
SDK_NAMES = ('net_out', 'delta', 'weight')
OBSERVATIONS = (34, 127, 191)
MODEL_KEYS = ('B2_D4_M16','B2_D5_M16','B2_D6_M16','B2_D7_M16',
              'B3_D6_M16','B1_D5_M16','B5_D6_M16','B1_D7_M16')
STAGES = ('pack_seconds','input_set_seconds','run_api_seconds','outputs_get_seconds',
          'owned_copy_seconds','outputs_release_seconds')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1 << 20), b''):
            digest.update(data)
    return digest.hexdigest()


def load_parent(directory):
    path = Path(directory)/'audit-v38-grouped-full.py'
    if sha(path) != PARENT_AUDITOR_SHA:
        raise ValueError('Frozen NumPy auditor ancestor SHA differs')
    spec = importlib.util.spec_from_file_location('_v40_saved_numpy_parent', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Checks:
    """A bounded receipt collector; no backend imports or execution."""
    def __init__(self, parent):
        self.parent, self.count, self.failures, self.artifacts = parent, 0, [], []

    def check(self, value, label):
        self.count += 1
        if not value:
            self.failures.append(label)

    def read_arrays(self, directory, row):
        relative = Path(row['file'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe artifact path')
        path = Path(directory)/relative
        if path.resolve().is_relative_to(Path(directory).resolve()) is False:
            raise ValueError('Artifact escapes its directory')
        self.check(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'artifact:'+relative.name)
        with np.load(path,allow_pickle=False) as archive:
            self.check(set(archive.files) == set(row['arrays']), 'array_keys:'+relative.name)
            values = {name:archive[name] for name in archive.files}
        for name,value in values.items():
            meta, saved = self.parent.raw(value), row['arrays'][name]
            self.check(all(meta[k] == saved[k] for k in meta), 'raw_bytes:'+relative.name+':'+name)
            if 'finite_elements' in saved:
                self.check(int(np.isfinite(value).sum()) == saved['finite_elements']
                    and int((~np.isfinite(value)).sum()) == saved['nonfinite_elements'], 'finite_counts:'+relative.name+':'+name)
        self.artifacts.append(dict(file=relative.as_posix(),sha256=sha(path),bytes=path.stat().st_size))
        return values

    def metrics(self, computed, recorded, label):
        for key in ('finite','gate_passed','failed_elements','max_abs','relative_L2','rtol','atol'):
            a,b = computed[key],recorded[key]
            self.check(a == b or (key == 'relative_L2' and a is not None and b is not None
                and np.isclose(a,b,rtol=1e-10,atol=1e-14)),label+':'+key)
        if 'nonfinite_elements' in recorded:
            self.check(computed['nonfinite_elements'] == recorded['nonfinite_elements'],label+':nonfinite_elements')

    def native_log(self, row, label):
        data = row['raw_text'].encode('utf-8')
        self.check(len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['raw_sha256']
            and not row['truncated'] and not row['RKNN_error_detected'] and 'E RKNN' not in row['raw_text'],label)


def native_attr_valid(attr, logical, index, output=False):
    """Independently validate the actually observed NHWC input/CF output layout."""
    shape = logical['shape']; elements = int(np.prod(shape))
    expected_shape = shape if output else [shape[i] for i in (0,2,3,1)]
    neutral = attr['quantization_type'] == 0 or (attr['quantization_type'] == 2
        and attr['scale'] == 1 and attr['zero_point'] == 0 and attr['fractional_length'] == 0)
    native_bytes = elements*(2 if attr['type'] == 1 else 4)
    return (attr['index'] == index and attr['name'] == logical['name'] and attr['shape'] == expected_shape
        and attr['format'] == (0 if output else 1) and attr['type'] in (0,1) and neutral
        and attr['n_elems'] == elements and attr['size'] == native_bytes
        and (attr['size_with_stride'] >= native_bytes if output else attr['size_with_stride'] in (0,native_bytes)))


def audit(args):
    parent = load_parent(args.parent_dir); tests = Checks(parent); check = tests.check
    paths = {name:((args.contract_dir if name in ('v40_m16_grouped_contracts.py','compile-v40-m16-grouped-update.py')
        else args.runtime_dir if name.startswith(('v40_','run-v40')) else args.parent_dir)/name) for name in SOURCES}
    local_sources = {name:sha(path) for name,path in paths.items()}
    check(local_sources == SOURCES,'frozen_local_sources')
    mp = args.export_dir/'manifest.json'; manifest = json.loads(mp.read_text())
    check(sha(mp) == MANIFEST_SHA,'actual_M16_manifest_SHA')
    original_run = args.capture_dir/'run.json'
    check(sha(original_run) == ORIGINAL_RUN_SHA and manifest['original_run_sha256'] == ORIGINAL_RUN_SHA
        and manifest['policy_sha256'] == POLICY_SHA,'original_capture_and_policy')
    check(manifest['strict_FP32_gate_passed'] and not manifest['candidate_accepted']
        and manifest['original_parameters_unchanged'] and len(manifest['original_Update_parameters']) == 50
        and len(manifest['original_weights_before']) == 94
        and manifest['original_weights_before'] == manifest['original_weights_after']
        and {'update.'+k:v for k,v in manifest['original_Update_parameters'].items()}
        == {k:v for k,v in manifest['original_weights_before'].items() if k.startswith('update.')}
        and manifest['source_pins'] == manifest['source_hashes_after'] and manifest['cleanup_errors'] == [],'original_50_parameters_and_CPU_ORT_ancestor')
    ip = args.compile_index; index = json.loads(ip.read_text())
    check(sha(ip) == args.compile_index_sha256 and index['schema'] == 'v40_M16_grouped_Update_compile_index'
        and index['status'] == 'staged_all_actual_shapes' and index['manifest_sha256'] == MANIFEST_SHA
        and index['contracts_sha256'] == SOURCES['v40_m16_grouped_contracts.py']
        and index['compiler_sha256'] == SOURCES['compile-v40-m16-grouped-update.py']
        and set(index['models']) == set(manifest['models']) == set(MODEL_KEYS)
        and not index['candidate_accepted'] and not index['host_model_inference_executed']
        and not index['host_calibration_executed'],'eight_actual_nonquantized_models')
    models = {}
    for key,row in index['models'].items():
        # Explicit host root maps the staged board's shape-directory basename.
        directory = args.compile_root/Path(row['compile_dir']).name
        cp = directory/'compile.json'; compiled = json.loads(cp.read_text())
        model = directory/'block.rknn'; selected = manifest['models'][key]
        check(sha(cp) == row['compile_sha256'] and compiled['model_key'] == key
            and compiled['manifest_sha256'] == MANIFEST_SHA and compiled['input'] == selected['model']
            and compiled['output'] == row['model'] and compiled['logical_inputs'] == selected['inputs']
            and compiled['logical_outputs'] == selected['outputs'],'compiled_IO_and_graph:'+key)
        graph = args.export_dir/selected['model']['file']
        check(sha(graph) == compiled['input']['sha256'] and graph.stat().st_size == compiled['input']['bytes']
            and sha(model) == row['model']['sha256'] and model.stat().st_size == row['model']['bytes'],'compiled_payload_bytes:'+key)
        check(compiled['script_sha256'] == SOURCES['compile-v40-m16-grouped-update.py']
            and compiled['contracts_sha256'] == SOURCES['v40_m16_grouped_contracts.py']
            and compiled['parent_compiler_sha256'] == SOURCES['compile-v37-grouped-update.py']
            and compiled['parent_contracts_sha256'] == SOURCES['v37_grouped_contracts.py']
            and compiled['configuration'] == dict(target_platform='rk3588',float_dtype='float16',optimization_level=3,do_quantization=False)
            and compiled['versions']['rknn-toolkit2'] == '2.3.2' and compiled['released'] and not compiled.get('release_error')
            and compiled['source_pins'] == compiled['source_hashes_after'] and compiled['sources_and_inputs_unchanged']
            and not compiled['candidate_accepted'] and not compiled['model_inference_executed']
            and not compiled['NPU_execution_or_placement_validated'],'compile_lineage_and_scope:'+key)
        check([v['name'] for v in compiled['operations']] == ['config','load_onnx','build','export_rknn']
            and all(v['code'] == 0 for v in compiled['operations']),'compiler_operations:'+key)
        models[key] = dict(compile_sha256=sha(cp),model_sha256=sha(model),model_bytes=model.stat().st_size)
    reports, receipt_hashes, lifecycle, profiles = {}, {}, {}, []
    for role in ('normal','profile'):
        rp = getattr(args,role+'_dir')/'run.json'; report = json.loads(rp.read_text())
        reports[role],receipt_hashes[role] = report,sha(rp)
        check(sha(rp) == getattr(args,role+'_sha256'),'receipt_pin:'+role)
        expected = [(o,n) for o in OBSERVATIONS for n in range(3 if role == 'normal' else 1)]
        check([(r['observation_id'],r['repeat_index']) for r in report['calls']] == expected,'complete_schedule:'+role)
        check(report['schema'] == 'v40_three_complete_M16_grouped_Update_RKNN_diagnostic'
            and report['status'] == 'completed_three_full_M16_graphs_diagnostic_not_VO'
            and report['script_sha256'] == SOURCES['run-v40-m16-grouped-board.py']
            and report['runtime_sha256'] == SOURCES['v37_grouped_rknn.py']
            and report['M16_runtime_binding_sha256'] == SOURCES['v40_m16_grouped_rknn.py']
            and report['M16_contracts_sha256'] == SOURCES['v40_m16_grouped_contracts.py']
            and report['compiler_sha256'] == SOURCES['compile-v40-m16-grouped-update.py']
            and report['original_run_sha256'] == ORIGINAL_RUN_SHA and report['policy_sha256'] == POLICY_SHA
            and report['export_manifest_sha256'] == MANIFEST_SHA and report['compile_index_sha256'] == args.compile_index_sha256
            and report['compiled_models'] == index['models'],'run_lineage:'+role)
        check(not report['candidate_accepted'] and not report['VO_state_committed']
            and report['actual_Torch_OR_ORT_calls'] == 0 and report['inherited_CPU_ORT_FP32_gate_passed'] is True
            and not report['whole_v39_workload_quality_accepted'] and not report['whole_online_shape_catalogue_supported']
            and not report['motion_probe_D1_supported'] and report['collect_perf_detail'] == (role == 'profile')
            and not report['source_group_padding'],'diagnostic_scope:'+role)
        check(report['cleanup_errors'] == [] and report['affinity_restored'] and report['sources_and_inputs_unchanged']
            and report['source_pins'] == report['source_hashes_after'],'recorded_cleanup_sources:'+role)
        keys = [g['model_key'] for row in report['calls'] for g in row['groups']]; counts = Counter(keys)
        cache = report['cache']; histories = {h['model_key']:h['metrics'] for h in cache['histories']}
        check(len(histories) == len(cache['histories']) == cache['created'] == 8
            and cache['hits'] == len(keys)-8 and cache['evicted'] == 0 and set(histories) == set(models),'cache_counts:'+role)
        check(cache['closed'] and not cache['cleanup_errors'] and not cache['resident'] and not cache['sessions']
            and cache['resident_contexts'] == cache['current_DMA_bytes'] == cache['current_model_host_bytes'] == 0,'cache_closed_zero:'+role)
        check(cache['peak_resident_contexts'] == 8 and cache['peak_resident_model_bytes'] == sum(v['model_bytes']+1 for v in models.values())
            and cache['peak_resident_DMA_bytes'] == sum(h['memory']['total_dma_allocated_size'] for h in histories.values())
            and cache['peak_resident_DMA_bytes'] <= cache['max_dma_bytes'] == 512<<20
            and cache['peak_resident_model_bytes'] <= cache['max_model_bytes'] == 256<<20,'memory_bounds:'+role)
        for key,h in histories.items():
            check(h['sdk_run_api_calls'] == h['completed_calls'] == h['SDK_outputs_releases'] == h['SDK_outputs_release_attempts'] == counts[key]
                and h['closed'] and h['context_destroyed'] and h['cleanup_errors'] == [] and h['npu_used'] and not h['synthetic']
                and h['current_DMA_bytes'] == h['current_model_host_bytes'] == 0 and h['core_mask'] == 7
                and h['init_flags'] == (8 if role == 'profile' else 0) and h['export_strict_FP32_gate_passed']
                and h['source_pins'] == h['source_hashes_after'] and h['sources_and_inputs_unchanged'],'session_lifecycle:'+role+':'+key)
            tests.native_log(h['initialization_native_log'],'native_init_log:'+role+':'+key)
            for output in (False,True):
                attrs,logical = h['output_attributes' if output else 'input_attributes'],manifest['models'][key]['outputs' if output else 'inputs']
                check(len(attrs) == len(logical) == 3 and all(native_attr_valid(a,l,i,output) for i,(a,l) in enumerate(zip(attrs,logical))),
                      'actual_native_attrs:'+role+':'+key+':'+str(output))
            models[key][role+'_memory'] = h['memory']
        check(len(report['initialization_events']) == 8 and {v['model_key'] for v in report['initialization_events']} == set(models)
            and all(v['status'] == 'initialized' and v['memory'] == histories[v['model_key']]['memory'] for v in report['initialization_events']),
              'initialization_events:'+role)
        lifecycle[role] = {k:cache[k] for k in ('created','hits','evicted','peak_resident_contexts','peak_resident_model_bytes',
            'peak_resident_DMA_bytes','closed','resident_contexts','current_DMA_bytes','current_model_host_bytes')}
        lifecycle[role].update(sdk_calls=len(keys),outputs_releases=sum(v['SDK_outputs_releases'] for v in histories.values()),
            destroyed_contexts=sum(v['context_destroyed'] for v in histories.values()),peak_RSS_MiB=report['peak_rss_mib'])
    inherited_count, repeat, representative, timing, actual_gates = 0, {}, {}, {}, []
    for case in manifest['cases']:
        observation = case['observation_id']; original = tests.read_arrays(args.capture_dir,case['original_inputs_artifact'])
        refs = dict(original_Torch=tests.read_arrays(args.capture_dir,case['original_gold_artifact']),
            grouped_Torch=tests.read_arrays(args.export_dir,case['grouped_Torch_artifact']),
            grouped_ORT=tests.read_arrays(args.export_dir,case['grouped_ORT_artifact']))
        groups = parent.independent_groups(*(original[k] for k in ('ii','jj','kk')),patches=16)
        check([g['key'] for g in groups] == [g['model_key'] for g in case['groups']],'independent_M16_complete_groups:'+str(observation))
        pairs = (('grouped_Torch','original_Torch','Torch_vs_original_full_gold'),
                 ('grouped_ORT','original_Torch','ORT_vs_original_full_gold'),
                 ('grouped_ORT','grouped_Torch','ORT_vs_grouped_Torch'))
        for a,b,label in pairs:
            for name in NAMES:
                result = parent.numerical(refs[a][name],refs[b][name]); inherited_count += 1
                tests.metrics(result,case[label][name],'inherited_full:'+str(observation)+':'+label+':'+name)
                check(result['gate_passed'],'CPU_ORT_full_gate:'+str(observation)+':'+label+':'+name)
        for group,saved in zip(groups,case['groups']):
            check(parent.raw(group['indices']) == saved['canonical_edge_indices'] and group['sources'] == saved['topology']['source_slots']
                and group['targets'] == saved['topology']['target_slots'],'independent_edge_order:'+str(observation)+':'+group['key'])
            feed = tests.read_arrays(args.export_dir,saved['inputs_artifact'])
            for name in ('net','inp','corr'):
                check(np.array_equal(original[name][0,group['indices'],:].transpose(0,3,1,2),feed[name]),'identical_group_input:'+group['key']+':'+name)
            for a,b,label in pairs:
                saved_label = label.replace('_full_','_group_')
                for name in NAMES:
                    result = parent.numerical(refs[a][name][0,group['indices'],:].transpose(0,3,1,2),
                        refs[b][name][0,group['indices'],:].transpose(0,3,1,2)); inherited_count += 1
                    tests.metrics(result,saved[saved_label][name],'inherited_group:'+str(observation)+':'+group['key']+':'+saved_label+':'+name)
                    check(result['gate_passed'],'CPU_ORT_group_gate:'+str(observation)+':'+group['key']+':'+saved_label+':'+name)
        first = None; repeat[observation] = True; timing[observation] = {}; representative[observation] = {}
        for role in ('normal','profile'):
            rows = [r for r in reports[role]['calls'] if r['observation_id'] == observation]; times = []
            for row in rows:
                label = role+':'+str(observation)+':'+str(row['repeat_index'])
                values = tests.read_arrays(getattr(args,role+'_dir'),row['full_order_artifact'])
                check(set(values) == set(NAMES) and all(values[n].dtype == np.float32 and values[n].shape == (1,case['edges'],c)
                    and np.isfinite(values[n]).all() for n,c in zip(NAMES,(384,2,2)))
                    and np.all((values['weight'] >= 0)&(values['weight'] <= 1)),label+':finite_shape_domain')
                if first is None: first = values
                else: repeat[observation] &= all(np.array_equal(first[k],values[k]) for k in NAMES)
                comparisons = {kind:{k:parent.numerical(values[k],ref[k]) for k in NAMES} for kind,ref in refs.items()}
                for kind,metrics in comparisons.items():
                    for k,v in metrics.items(): tests.metrics(v,row['versus_saved_references'][kind][k],label+':'+kind+':'+k)
                gate = all(v['gate_passed'] for v in comparisons['original_Torch'].values()); actual_gates.append(gate)
                check(row['strict_FP32_gate_passed'] == gate,label+':unchanged_FP32_gate')
                check(row['degree_plan'] == case['degree_plan'] and row['snapshot_identity'] == case['snapshot_identity']
                    and row['snapshot_identity'] == dict(saved_update_index=case['original_call_index'],online_revisions_available=False)
                    and row['full_graph_returned'] and row['status'] == 'completed_diagnostic_not_adopted'
                    and not row['candidate_accepted'] and not row['VO_state_committed'] and not row['padding'] and not row['dropped_edges']
                    and row['outputs_finite'] and row['weight_domain_valid'] and row['sequential_groups'] == len(groups) == len(row['groups']),label+':atomic_original_E_result')
                sums = Counter(); forward = cache_time = 0
                for group,saved in zip(groups,row['groups']):
                    record,t = saved['record'],saved['record']['timing']
                    check(saved['model_key'] == group['key'] and record['execution_status'] == 'completed' and record['outputs_released']
                        and record['outputs_finite'] and record['weight_domain_valid'] and not record['all_outputs_zero']
                        and record['strict_FP32_gate_passed'] is None and not record['candidate_accepted'],label+':SDK_group:'+group['key'])
                    for sdk,name in zip(SDK_NAMES,NAMES):
                        recovered = values[name][0,group['indices'],:].transpose(0,3,1,2); meta = parent.raw(recovered)
                        check(all(meta[k] == saved['raw_outputs'][sdk][k] for k in meta),label+':inverse_scatter_SHA:'+group['key']+':'+sdk)
                    for inp,transport in zip(manifest['models'][group['key']]['inputs'],t['input_records']):
                        shape = inp['shape']
                        check(transport['name'] == inp['name'] and transport['logical_shape'] == shape
                            and transport['buffer_shape'] == [shape[i] for i in (0,2,3,1)]
                            and transport['native_attribute_format'] == transport['request_format'] == 1
                            and transport['source_dtype'] == transport['transport_dtype'] == 'float32'
                            and transport['bytes'] == 4*int(np.prod(shape)) and transport['pass_through'] == 0
                            and transport['owned'] and transport['explicit_NHWC_pack'],label+':NHWC_pack:'+group['key']+':'+inp['name'])
                    check(t['synchronous'] and all(t[s] >= 0 for s in STAGES) and sum(t[s] for s in STAGES) <= t['forward_seconds']
                        <= record['total_execute_seconds'] <= record['cache_call_seconds'],label+':nested_timing:'+group['key'])
                    tests.native_log(record['native_log'],label+':native_log:'+group['key'])
                    for s in STAGES: sums[s] += t[s]
                    forward += t['forward_seconds']; cache_time += record['cache_call_seconds']
                    if role == 'profile': profiles.append(dict(observation_id=observation,model_key=group['key'],**parent.parse_profile(record['profile'])))
                check(forward == row['summed_nested_SDK_forward_seconds'] and cache_time == row['summed_nested_cache_call_seconds']
                    and cache_time <= row['full_graph_seconds'] <= row['full_wall_seconds'],label+':fullwall_vs_nested')
                times.append(dict(repeat_index=row['repeat_index'],full_wall_seconds=row['full_wall_seconds'],
                    nested_SDK_forward_seconds=forward,nested_cache_seconds=cache_time,nested_SDK_stages=dict(sums),
                    new_contexts=len(row['new_initializations']),outside_statistics_seconds=row['outside_forward_statistics_seconds'],
                    outside_save_seconds=row['outside_forward_save_seconds']))
                if role == 'normal' and row['repeat_index'] == 0: representative[observation] = comparisons
            timing[observation][role] = times
            if role == 'normal':
                mean = sum(t['full_wall_seconds'] for t in times[1:])/2
                check(mean == reports[role]['timing_summary'][str(observation)]['warm_mean_full_wall_seconds'],'warm_mean:'+str(observation))
                timing[observation]['ordinary_warm_mean_full_wall_seconds'] = mean
                timing[observation]['ordinary_warm_mean_nested_SDK_stages'] = {s:sum(t['nested_SDK_stages'][s] for t in times[1:])/2 for s in STAGES}
        check(repeat[observation],'three_normal_plus_separate_profile_bitwise:'+str(observation))
    check(inherited_count == manifest['strict_comparison_count'] == 135,'all_135_original_CPU_ORT_comparisons')
    for role,report in reports.items():
        gates = [r['strict_FP32_gate_passed'] for r in report['calls']]
        check(report['actual_original_FP32_gate_passed'] == all(gates),'run_component_gate:'+role)
    coverage = {}
    for key in MODEL_KEYS:
        rows = [p for p in profiles if p['model_key'] == key]
        check(bool(rows) and all(p['internal_target_counts'] == rows[0]['internal_target_counts']
            and p['CPU_IO_rows'] == rows[0]['CPU_IO_rows'] for p in rows),'actual_profile_coverage:'+key)
        if rows:
            coverage[key] = dict(calls=len(rows),internal_target_counts=rows[0]['internal_target_counts'],CPU_IO_rows=rows[0]['CPU_IO_rows'],
                target_microseconds_by_call=[p['target_microseconds'] for p in rows],largest_NPU=rows[0]['largest_NPU'],internal_CPU=rows[0]['internal_CPU'])
    return dict(status='passed_saved_M16_identity_not_adopted' if not tests.failures else 'failed',script_sha256=sha(__file__),
        checks=dict(count=tests.count,failed=len(tests.failures),failures=tests.failures),candidate_accepted=False,VO_state_committed=False,
        identity=dict(manifest_sha256=MANIFEST_SHA,original_run_sha256=ORIGINAL_RUN_SHA,compile_index_sha256=sha(ip),
            parent_NumPy_auditor_sha256=PARENT_AUDITOR_SHA,local_source_sha256=local_sources,compiled_models=models,
            total_compiled_model_bytes=sum(v['model_bytes'] for v in models.values())),raw_receipts=receipt_hashes,saved_artifacts=tests.artifacts,
        numerical=dict(inherited_CPU_ORT_comparisons=135,inherited_CPU_ORT_FP32_gate_passed=True,
            actual_SDK_FP32_gate_passed=all(actual_gates),thresholds=dict(rtol=1e-4,atol=1e-5),
            three_normal_plus_profile_bitwise_by_observation=repeat,representative_versus_all_three_references=representative),
        cache_and_lifecycle=lifecycle,timing=timing,profile_coverage_by_actual_shape=coverage,actual_profiles=profiles,
        limitations=['Three fixed captured M16 full graphs only; no recurrent feedback, BA, trajectory, all-online-shape or 20Hz acceptance.',
            'CPU/ORT component acceptance and finite SDK execution do not withdraw the v39 whole-workload accuracy rejection or adopt FP16 output.',
            'Per-group CF raw hashes are independently recovered from original-E full payloads; separate SDK group payloads were not saved.',
            'Remote copy/release/destroy, source stability, immutable snapshot identity and DMA are recorded board evidence; host verifies saved bytes and local provenance.',
            'Profile operator sums are separate diagnostic calls, not ordinary fullwall latency or additive acceleration promises.',
            'Relative-L2 reductions may round on ARM/x86; the original FP32 thresholds and failed-element counts are unchanged.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('normal-dir','profile-dir','export-dir','capture-dir','compile-index','compile-root',
                 'parent-dir','contract-dir','runtime-dir','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('compile-index','normal','profile'):
        parser.add_argument('--'+name+'-sha256',required=True)
    args = parser.parse_args(argv)
    if args.output.exists(): raise FileExistsError(args.output)
    try: report = audit(args)
    except Exception as error:
        report = dict(status='failed',script_sha256=sha(__file__),error=repr(error),candidate_accepted=False)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report.get(k) for k in ('status','checks','error')},allow_nan=False))
    return int(report['status'] == 'failed')


if __name__ == '__main__': raise SystemExit(main())
