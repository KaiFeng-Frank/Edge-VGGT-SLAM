#!/usr/bin/env python3
"""Independent saved NumPy audit: fused core7 versus native three-core DINO0.

No neural, ONNX, SDK, GPU, preprocessing or board execution. Parallel worker
durations are observations, never added as the complete block latency.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import numpy as np

ROOT = Path('/dev/shm/edge-vggt-v124-review')
PRIOR = Path('/path/to/research-artifacts/v117-native518-attention-dump-r1/old-reference-r100')
METRICS = Path('/dev/shm/edge-vggt-v120-review/audit_dino0_shared_block_saved.py')
METRICS_SHA = '67424215747b36d495e27ffc31448c53bdb2a1660523393c9c9465f39da3a7a9'
RESULT_SHA = 'a3201f914ebfde5cd70df376a6fd10155cae08eb02368bb4ef866349743da6bd'
CONTROL_SHA = '8ba92daeabcd269a4c20c90bd3efa205d7dfc1ed6ffb102587776a0fc6569086'
MANIFEST_SHA = 'b2e383e03c4c514ca332b55185bdfc4e0c8be71c4b50eff1180c98ca093e1cf2'
CAPTURE_SHA = '17d77cfe3d8cdb3ff88e15de4a06634683899348b53b24fbadad6eef1584e91d'
OLD_SHA = {'R121':'baa246379666a5b2320d62b7935ae553d398feef8be21ef58e27d727f5a295ff',
           'R122':'a76489d0f276b272c3a5aa97b0ac005b09008198890dec6649ec54f4e4f023ed'}
RAW_A = 'e52d53b8abcbb15da0c129457f93e3eabcd94d48ae48eb5eaef0bb0edb900554'
RAW_B = '70de59362e3bb03eb875bc3ee9be56a7af538b4098e662d9fd6b817d5c3c3e0d'
LINEARS = {'qkv':(1024,3072,(1024,1024,1024)), 'proj':(1024,1024,(352,352,320)),
           'fc1':(1024,4096,(1376,1376,1344)), 'fc2':(4096,1024,(352,352,320))}
RANGES = ((0,6),(6,11),(11,16))
PRODUCTS = {name+str(lane):(k,parts[lane]) for name,(k,_,parts) in LINEARS.items() for lane in range(3)}
for head in range(16):
    PRODUCTS['qk'+str(head)] = (64,1056)
    PRODUCTS['pv'+str(head)] = (1056,64)
GPU_STAGES = ('norm1','pack_all','softmax_all','merge_all','proj_residual_norm','erf_gelu','final_residual')
LAUNCH = {'norm1':(266496,5,40), 'pack_all':(1081344,103,824), 'softmax_all':(4263936,65,520),
          'merge_all':(1069056,34,272), 'proj_norm':(266496,14,112), 'gelu':(4263936,9,72),
          'final':(1065984,10,80)}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4<<20), b''):
            value.update(block)
    return value.hexdigest()


def span(rows):
    return max(r['SDK_run_end_monotonic'] for r in rows)-min(r['SDK_run_start_monotonic'] for r in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--reference',type=Path,default=PRIOR)
    parser.add_argument('--reference121',type=Path,default=Path('/dev/shm/edge-vggt-v121-review'))
    parser.add_argument('--reference122',type=Path,default=Path('/dev/shm/edge-vggt-v122-review'))
    parser.add_argument('--metrics-source',type=Path,default=METRICS)
    parser.add_argument('--output',type=Path,default=ROOT/'native-threecore-saved-audit.json')
    args = parser.parse_args(); files=[]
    require(sha(args.metrics_source)==METRICS_SHA,'Frozen saved-only NumPy helper differs')
    spec=importlib.util.spec_from_file_location('_v124_saved_numpy_metrics',args.metrics_source)
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)

    def record(path,digest=None):
        actual=sha(path);require(digest is None or actual==digest,'Pinned file differs: '+str(path))
        files.append(dict(file=str(path),bytes=path.stat().st_size,sha256=actual));return path

    def receipt(path,digest=None):
        return json.loads(record(path,digest).read_text())

    def load(base,row):
        require(Path(row['file']).name==row['file'],'Relative NPY basename required')
        path=record(base/row['file'],row['sha256']);value=np.load(path,allow_pickle=False)
        require(path.stat().st_size==row['bytes'] and list(value.shape)==row['shape']
                and str(value.dtype)==row['dtype'] and h.raw(value)==row['raw_sha256']
                and np.isfinite(value).all(),'NPY shape/dtype/file/raw/finite differs')
        files[-1].update(raw_sha256=h.raw(value),shape=list(value.shape),dtype=str(value.dtype),finite=True)
        return value

    result=receipt(args.root/'block-r1/result.json',RESULT_SHA)
    control=receipt(args.root/'control-r1/control.json',CONTROL_SHA)
    require(result['schema']=='v124_original_native518_complete_DINO0_three_core_native_ABBA'
            and result['status']=='completed_original_DINO0_three_core_ABBA_not_full_model_accepted'
            and not result['cleanup_errors'] and all(result[k] for k in ('original_input_unchanged',
                'source_and_inputs_unchanged','affinity_restored','environment_restored'))
            and not result['candidate_accepted'] and not result['reference_intermediate_substituted']
            and result['baseline_core_mask']==7 and result['candidate_native_core_masks']==[1,2,4],
            'Clean unchanged complete component execution required')
    source=args.root/'executed-source-r1'
    require(receipt(source/'sources.json',MANIFEST_SHA)==result['source_bundle']
            and result['source_manifest_sha256']==MANIFEST_SHA,'Actual source bundle differs')
    for name,row in result['source_bundle'].items():
        path=record(source/name,row['sha256']);require(path.stat().st_size==row['bytes'],'Source length differs')
        if path.suffix=='.py':ast.parse(path.read_text())
    cap=receipt(args.reference/'capture.json',CAPTURE_SHA)
    require(result['original_parameters']==cap['original_parameters'] and len(result['original_parameters'])==14
            and result['original_input']==cap['artifacts']['input']
            and result['original_output_reference']==cap['artifacts']['output'],'Original input/14parameter metadata differs')
    x=load(args.reference,cap['artifacts']['input']);ref=load(args.reference,cap['artifacts']['output'])
    require(x.shape==ref.shape==(1,1041,1024) and x.dtype==ref.dtype==np.float32,'Original native input/output required')
    old={tag:receipt(base/'block-r1/result.json',OLD_SHA[tag])
         for tag,base in (('R121',args.reference121),('R122',args.reference122))}
    old_bases={'R121':args.reference121/'block-r1','R122':args.reference122/'block-r1'}
    owners=result['candidate_cleanup']['owner_metrics']['fixed_owner_workers']
    require([o['lane'] for o in owners]==[0,1,2],'Three fixed owners required')
    owner_by_mask={o['core_mask']:o['owner_thread_id'] for o in owners}
    require(set(owner_by_mask)=={1,2,4} and len(set(owner_by_mask.values()))==3,'Distinct fixed owner threads required')
    rows=result['calls'];require(len(rows)==8 and [r['index'] for r in rows]==list(range(8))
            and ''.join(r['route'] for r in rows)=='ABBAABBA','Complete ABBAABBA required')
    values={};comparison=[];timings=[];cross=[];expected_files=set();gpu_events=0
    for row in rows:
        route=row['route'];index=row['index']
        require(row['first_for_route']==(index==(0 if route=='A' else 1)),'First-route exclusion differs')
        y=load(args.root/'block-r1',row['output']);expected_files.add(row['output']['file'])
        require(y.shape==x.shape and y.dtype==np.float32 and h.raw(y)==(RAW_A if route=='A' else RAW_B),
                'Whole output shape/canonical bytes differ')
        if route in values:require(y.tobytes()==values[route].tobytes(),'Repeat output differs')
        else:values[route]=y
        checks={}
        for tag,previous in old.items():
            prior=previous['calls'][index]
            require(prior['route']==route,'Corresponding historical route differs')
            z=load(old_bases[tag],prior['output'])
            require(y.tobytes()==z.tobytes(),'Corresponding historical output differs')
            checks[tag]=dict(byte_equal=True,file_sha256_equal=row['output']['sha256']==prior['output']['sha256'])
        cross.append(dict(file=row['output']['file'],references=checks))
        whole,update=h.compare(y,ref),h.compare(y-x,ref-x)
        for actual,reported in ((whole,row['vs_original_CPU']),(update,row['residual_update_vs_original_CPU'])):
            h.agree(actual,reported)
            require(actual['fixed_gate_passed']==reported['fixed_reference_gate_passed']
                    and reported['strict_rtol']==1e-4 and reported['strict_atol']==1e-5,'Fixed gate/tolerances differ')
        comparison.append(dict(index=index,route=route,whole_output=whole,residual_update=update))
        t=dict(index=index,route=route,complete_block_seconds=row['complete_block_seconds'])
        if route=='A':
            t['fused_owned_IO_boundary_seconds']=row['stages']['complete_boundary_seconds']
            require(t['complete_block_seconds']>=t['fused_owned_IO_boundary_seconds'],'A outer boundary differs')
            t['SDK_run_seconds']=row['stages']['SDK_run_seconds']
        else:
            inner=row['stages'];s=inner['stages'];heads=s['heads']
            require([v['head'] for v in heads]==list(range(16)),'All16 full attention heads required')
            part_rows=[p for key in LINEARS for p in s[key]['parts']]+[v[k] for v in heads for k in ('QK','PV')]
            require(len(part_rows)==44 and len({v['product'] for v in part_rows})==44,'All44 native products required')
            for part in part_rows:
                require(part['owner_thread_id']==owner_by_mask[part['core_mask']]
                        and math.isclose(part['SDK_run_seconds'],part['SDK_run_end_monotonic']-part['SDK_run_start_monotonic'],rel_tol=1e-10)
                        and part['worker_start_monotonic']<=part['SDK_run_start_monotonic']<=part['SDK_run_end_monotonic'],
                        'Fixed owner/native SDK timestamps differ')
            weighted={}
            for key in LINEARS:
                stage=s[key];parts=stage['parts'];sdk_span=span(parts)
                require([p['product'] for p in parts]==[key+str(i) for i in range(3)]
                        and [p['core_mask'] for p in parts]==[1,2,4]
                        and stage['complete_seconds']>=sdk_span
                        and math.isclose(stage['measured_blocking_SDK_run_span_seconds'],sdk_span,rel_tol=1e-12)
                        and stage['overlapping_worker_durations_are_not_summed_as_chain_latency'],
                        'Weighted dispatch/span differs')
                overlaps=[dict(core_masks=[a['core_mask'],b['core_mask']],seconds=max(0.,
                    min(a['SDK_run_end_monotonic'],b['SDK_run_end_monotonic'])
                    -max(a['SDK_run_start_monotonic'],b['SDK_run_start_monotonic'])))
                    for lane,a in enumerate(parts) for b in parts[lane+1:]]
                weighted[key]=dict(wall_seconds=stage['complete_seconds'],blocking_SDK_span_seconds=sdk_span,
                    per_worker_SDK_seconds=[p['SDK_run_seconds'] for p in parts],host_call_pairwise_overlap=overlaps)
            attention={}
            for kind,parallel in (('QK','QK_parallel'),('PV','PV_parallel')):
                parts=[v[kind] for v in heads];stage=s[parallel];sdk_span=span(parts)
                require(stage['owner_groups']==[list(v) for v in RANGES] and stage['complete_seconds']>=sdk_span,
                        'Attention dispatch/span differs')
                for mask,(lo,hi) in zip((1,2,4),RANGES):
                    selected=parts[lo:hi]
                    require(all(p['core_mask']==mask for p in selected)
                            and all(a['SDK_run_end_monotonic']<=b['SDK_run_start_monotonic']
                                    for a,b in zip(selected,selected[1:])), 'Per-owner head calls must be serial')
                attention[kind]=dict(wall_seconds=stage['complete_seconds'],blocking_SDK_span_seconds=sdk_span)
            gpu=[s[key] for key in GPU_STAGES]
            for stage in gpu:
                g=stage['GPU'];q=g['GPU_event_clocks_ns'];global_size,argc,size=LAUNCH[g['method']]
                require(g['global_size']==[global_size] and g['local_size']==[256]
                        and q['queued']<=q['submit']<=q['start']<=q['end']
                        and math.isclose(g['GPU_kernel_seconds'],(q['end']-q['start'])/1e9,rel_tol=1e-12),
                        'GPU launch/event timing differs')
                if g['method']=='softmax_all':require(g['finite_error_bits']==0,'Softmax flag differs')
                gpu_events+=1
            subtotal=s['input_upload_seconds']+s['owned_output_and_finite_seconds']+sum(s[k]['complete_seconds']
                for k in LINEARS)+s['QK_parallel']['complete_seconds']+s['PV_parallel']['complete_seconds']+sum(g['complete_seconds'] for g in gpu)
            require(subtotal<=inner['complete_native_block_seconds']+1e-10
                    and inner['complete_native_block_seconds']<=t['complete_block_seconds'], 'Joined stage boundaries exceed block')
            t.update(weighted_stage_observations=weighted,attention_stage_observations=attention,
                weighted_wall_seconds=sum(s[k]['complete_seconds'] for k in LINEARS),
                QK_dispatch_wall_seconds=s['QK_parallel']['complete_seconds'],PV_dispatch_wall_seconds=s['PV_parallel']['complete_seconds'],
                GPU_stage_wall_seconds=sum(g['complete_seconds'] for g in gpu),
                GPU_softmax_wall_seconds=s['softmax_all']['complete_seconds'],
                contained_GPU_kernel_seconds=sum(g['GPU']['GPU_kernel_seconds'] for g in gpu),
                owned_output_seconds=s['owned_output_and_finite_seconds'],inner_other_seconds=inner['complete_native_block_seconds']-subtotal,
                outer_other_seconds=t['complete_block_seconds']-inner['complete_native_block_seconds'])
        timings.append(t)
    diagnostics=rows[1]['diagnostics_outside_timing']
    require(set(diagnostics)=={p+v for p in LINEARS for v in 'AC'},'Eight first-B diagnostic arrays required')
    for name,row in diagnostics.items():
        value=load(args.root/'block-r1',row);expected_files.add(row['file']);k,n,_=LINEARS[name[:-1]]
        require(value.shape==((k//8,1041,8) if name.endswith('A') else (n//4,1041,4))
                and value.dtype==(np.float16 if name.endswith('A') else np.float32),'Reconstructed native AC layout differs')
        checks={}
        for tag,previous in old.items():
            prior=previous['calls'][1]['diagnostics_outside_timing'][name];z=load(old_bases[tag],prior)
            require(value.tobytes()==z.tobytes(),'Corresponding historical native diagnostic differs')
            checks[tag]=dict(byte_equal=True,file_sha256_equal=row['sha256']==prior['sha256'])
        cross.append(dict(file=row['file'],references=checks))
    require(len(expected_files)==16 and {p.name for p in (args.root/'block-r1').glob('*.npy')}==expected_files,
            'Exactly16 actual NPYs required')
    expected=dict(rknn_matmul_create=44,rknn_matmul_set_core_mask=44,rknn_create_mem=132,
        rknn_matmul_run=176,rknn_destroy_mem=132,rknn_matmul_destroy=44,rknn_matmul_set_io_mem=460,rknn_mem_sync=1020)
    require(result['MatMul_attempts']==result['MatMul_successes']==expected,'Native API accounting differs')
    clean=result['candidate_cleanup'];terminal=clean['owner_metrics']
    require(clean['native_close_permitted_after_GPU_release'] and clean['borrowed_A_bindings_released']
            and clean['shared_A_leader_destroyed_only_after_borrower_contexts'] and not clean['cleanup_errors']
            and clean['calls_attempted']==clean['calls_completed']==4 and set(clean['resources'])==set(PRODUCTS)
            and terminal['closed'] and not terminal['failed'] and terminal['head_ranges']==[list(v) for v in RANGES]
            and all(clean[k]==terminal[k]==0 for k in ('live_owner_workers','live_contexts','live_memories'))
            and clean['live_owner_threads']==0,'Candidate worker/shared-A cleanup differs')
    worker_totals={key:0 for key in expected}
    for lane,owner in enumerate(owners):
        require(owner['core_mask']==1<<lane and owner['cleanup_history']==[],'Fixed worker identity/history differs')
        require(owner['SDK_API']['attempts']==owner['SDK_API']['successes'],'Worker API failure differs')
        for key in expected:worker_totals[key]+=owner['SDK_API']['successes'][key]
        for name,value in owner['resources'].items():
            require(value['live_context']==value['live_memories']==0 and (value['K'],value['output_N'])==PRODUCTS[name],
                    'Terminal native layout/owner differs')
    require(worker_totals==expected,'Independent per-worker total differs')
    for name,value in clean['resources'].items():
        require(not value['cleanup_errors'] and value['live_context']==value['live_memories']==0
                and (value['K'],value['output_N'])==PRODUCTS[name],'Native final owner remains live')
    baseline=result['baseline_resources']
    require(baseline['core_mask']==7 and baseline['init_flags']==0 and baseline['closed']
            and not baseline['failed'] and not baseline['cleanup_errors'] and baseline['resource_counts_balanced']
            and baseline['contexts_created']==baseline['contexts_destroyed']==1
            and baseline['runs_completed']==baseline['output_buffers_acquired']==baseline['output_buffers_released']==4
            and baseline['live_contexts']==baseline['live_output_buffers']==0,'Fused core7 ownership differs')
    g=result['GPU_after_close'];objects=dict(Context=1,CommandQueue=1,Program=1,MemObject=126,Kernel=7)
    require(gpu_events==28 and all(g[k]==28 for k in ('calls_attempted','calls_completed','events_created','events_released'))
            and g['objects_created']==g['objects_released']==objects and g['DMA_import_count']==112
            and g['fully_released'] and g['queue_idle'] and g['closed'] and not g['failed'] and not g['cleanup_errors']
            and g['live_OpenCL_objects']==g['live_events']==g['live_owned_DMA_fds']==g['live_host_IO_owners']==0,
            'GPU objects/events/import cleanup differs')
    require(g['device_max_parameter_size_bytes']==1024 and g['actual_build_options']=='-cl-std=CL1.2',
            'GPU parameter limit/build flags differ')
    for method,(total,argc,size) in LAUNCH.items():
        m=g['methods'][method]
        require(m['calls_attempted']==m['calls_completed']==4 and m['kernel_arguments']==argc
                and m['kernel_parameter_bytes']==size and m['global_size']==[total] and m['local_size']==[256],
                'Actual kernel ABI/launch/count differs')
    require(g['parameter_vector_raw_sha256']=={k:result['original_parameters'][k]['raw_sha256'] for k in g['parameter_vector_raw_sha256']}
            and len(g['parameter_vector_raw_sha256'])==10,'Original parameter vectors differ')
    indices={route:[i for i,c in enumerate(rows) if c['route']==route and not c['first_for_route']] for route in 'AB'}
    means={route:statistics.mean(rows[i]['complete_block_seconds'] for i in ids) for route,ids in indices.items()}
    require(all(len(indices[k])==3 and math.isclose(means[k],result['subsequent_three_means_seconds'][k],rel_tol=1e-12)
                for k in means),'Warm summary differs')
    b_stages={key:statistics.mean(timings[i][key] for i in indices['B']) for key in
              ('weighted_wall_seconds','QK_dispatch_wall_seconds','PV_dispatch_wall_seconds','GPU_stage_wall_seconds',
               'GPU_softmax_wall_seconds','contained_GPU_kernel_seconds','owned_output_seconds','inner_other_seconds','outer_other_seconds')}
    lifecycle=sum(result['initialization_seconds'].values())+result['destroy_seconds']+sum(c['complete_block_seconds'] for c in rows)
    require(math.isclose(lifecycle,result['summed_eight_call_lifecycle_seconds'],rel_tol=1e-12),'Separate lifecycle accounting differs')
    require(control['child_returncode']==0 and not control['restore_errors'] and control['baseline']==control['restored']
            and control['frequency_governor_limits_restored'] and control['restored']['cur_freq']=='300000000'
            and all(v['actual_frequency_hz']==1000000000 for v in control['observations']),'Control restoration differs')
    maximum=max(t for v in control['observations'] for t in v['temperatures_c'].values())
    require(maximum<control['temperature_limit_c'],'Thermal limit reached')
    audit=dict(schema='v124_independent_saved_original_DINO0_three_core_audit',status='verified_saved_component_evidence_not_accepted',
        auditor_sha256=sha(__file__),execution='Host JSON/NumPy/text/AST only; no neural/SDK/GPU/board calls',candidate_accepted=False,
        checked_actual_NPYs=16,all16_arrays_byte_equal_corresponding_R121_and_R122=True,cross_run_comparisons=cross,
        route_raw_sha256={k:h.raw(v) for k,v in values.items()},within_each_route_all4_outputs_byte_equal=True,
        original_FP32_whole_output={k:h.compare(v,ref) for k,v in values.items()},
        original_FP32_residual_update={k:h.compare(v-x,ref-x) for k,v in values.items()},per_call_comparisons=comparison,
        baseline_core_mask=7,candidate_core_masks=[1,2,4],call_timings=timings,subsequent_indices=indices,
        subsequent_three_mean_seconds=means,B_slowdown_percent=100*(means['B']/means['A']-1),
        B_subsequent_stage_wall_means_seconds=b_stages,initialization_seconds=result['initialization_seconds'],
        destroy_seconds=result['destroy_seconds'],summed_eight_call_lifecycle_seconds=lifecycle,
        final_ledger=dict(native_contexts=44,native_memories=132,native_runs=176,bindings=460,sync_calls=1020,
            outside_timer_diagnostic_FROM_sync_calls=16,GPU_imports=112,OpenCL_objects_created_released=136,
            GPU_kernels_events_created_released=28,baseline_contexts=1,baseline_runs=4,baseline_output_buffers=4,
            live_workers=0,live_contexts=0,live_memories=0,shared_A_borrowers_released=True),
        worker_close_scope='Pinned source joins borrower destroy jobs before leader/shared-A close; terminal per-worker ledgers agree. No per-destroy timestamps captured.',
        peak_rss_mib_reported=result['peak_rss_mib'],GPU_policy_before=result['platform_before']['GPU'],
        GPU_policy_after=result['platform_after']['GPU'],control=dict(child_exit=0,observed_NPU_hz=1000000000,
            restored_NPU_hz=300000000,exact_policy_restored=True,max_sampled_temperature_c=maximum),
        timing_scope=result['timing_scope'],limits=['Audit verifies stored evidence and cleanup, not component/full-model precision or performance acceptance',
            'Comparator is original fused core7 in the same ABBA; historical core1 is not the performance baseline',
            'Parallel worker durations are not summed as block latency; weighted/heads dispatch wall and SDK spans are separately recorded',
            'Overlapping blocking host SDK calls do not prove parallel device execution, core utilization or bandwidth',
            'First A/B calls excluded; three subsequent samples per route do not measure P95 or sustained throughput',
            'GPU profiling events are contained in stage wall time; endpoint GPU auto-governor readings do not prove constant frequency',
            'Init/destroy and first-B diagnostics/file/reference work are outside persistent call timers',
            'One DINO0 block only; no full24 encoder, aggregator, original SL4, trajectory or realtime acceptance',
            'Original14 FP32 weight identity is saved board metadata, not a new host weight/inference check'],files=files)
    args.output.write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=audit['status'],arrays=16,means_seconds=means,
        B_slowdown_percent=audit['B_slowdown_percent'],B_relative_L2=audit['original_FP32_whole_output']['B']['relative_l2'])))


if __name__=='__main__':
    main()
