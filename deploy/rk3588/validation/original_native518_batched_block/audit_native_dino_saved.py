#!/usr/bin/env python3
"""Saved NumPy audit of original DINO0 native ABBA; no neural/device execution."""
import argparse
import ast
import importlib.util
import json
import math
from pathlib import Path
import statistics
import numpy as np

ROOT = Path('/dev/shm/edge-vggt-v122-review')
REFERENCE121 = Path('/dev/shm/edge-vggt-v121-review')
PRIOR = Path('/path/to/research-artifacts/v117-native518-attention-dump-r1')
METRICS = Path(__file__).resolve().parents[1]/'original_native518_shared_block/audit_dino0_shared_block_saved.py'
METRICS_SHA = 'c390494e5515d83539a65545e19b6bccaa0ad08907a1cd63759e84a8cb39fc71'
RESULT_SHA = 'a76489d0f276b272c3a5aa97b0ac005b09008198890dec6649ec54f4e4f023ed'
RESULT121_SHA = 'baa246379666a5b2320d62b7935ae553d398feef8be21ef58e27d727f5a295ff'
CAPTURE_SHA = '17d77cfe3d8cdb3ff88e15de4a06634683899348b53b24fbadad6eef1584e91d'
BASE_RAW = 'e52d53b8abcbb15da0c129457f93e3eabcd94d48ae48eb5eaef0bb0edb900554'
SHAPE = (1,1041,1024)
PRODUCTS = {'qkv':(1024,3072),
    'proj':(1024,1024),'fc1':(1024,4096),'fc2':(4096,1024)}
for head in range(16):
    PRODUCTS['qk'+str(head)] = (64,1056)
    PRODUCTS['pv'+str(head)] = (1056,64)
METHODS = {'norm1':(4,266496),'pack_all':(4,1081344),'softmax_all':(4,4263936),
    'merge_all':(4,1069056),'proj_norm':(4,266496),'gelu':(4,4263936),'final':(4,1065984)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--prior',type=Path,default=PRIOR)
    p.add_argument('--reference121',type=Path,default=REFERENCE121)
    p.add_argument('--metrics-source',type=Path,default=METRICS)
    p.add_argument('--output',type=Path,default=ROOT/'native-dino-r1-saved-audit.json')
    a=p.parse_args()
    import hashlib
    require=lambda ok,msg: None if ok else (_ for _ in ()).throw(ValueError(msg))
    require(hashlib.sha256(a.metrics_source.read_bytes()).hexdigest()==METRICS_SHA,'Frozen saved-only metric helper differs')
    spec=importlib.util.spec_from_file_location('_v121_saved_numpy_metrics',a.metrics_source)
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    files=[]
    def record(path,digest=None):
        actual=h.sha(path);require(digest is None or actual==digest,'Pinned identity differs: '+str(path))
        files.append(dict(file=str(path),bytes=path.stat().st_size,sha256=actual));return path
    def receipt(path,digest=None):return json.loads(record(path,digest).read_text())
    def load(base,row):
        require(Path(row['file']).name==row['file'],'NPY must be relative basename')
        path=base/row['file'];record(path,row['sha256']);v=np.load(path,allow_pickle=False)
        require(path.stat().st_size==row['bytes'] and list(v.shape)==row['shape'] and str(v.dtype)==row['dtype']
            and h.raw(v)==row['raw_sha256'] and np.isfinite(v).all(),'NPY bytes/raw/shape/dtype/finite differs')
        files[-1].update(raw_sha256=h.raw(v),shape=list(v.shape),dtype=str(v.dtype),finite=True);return v
    r=receipt(a.root/'block-r1/result.json',RESULT_SHA);c=receipt(a.root/'control-r1/control.json')
    require(r['schema']=='v122_original_native518_complete_DINO0_batched_native_ABBA' and
        r['status']=='completed_original_DINO0_ABBA_not_full_model_accepted' and not r['cleanup_errors'] and
        all(r[k] for k in ('source_and_inputs_unchanged','affinity_restored','environment_restored','original_input_unchanged'))
        and r['reference_intermediate_substituted'] is False and r['candidate_accepted'] is False,'Clean component execution required')
    require(receipt(a.root/'sources.json',r['source_manifest_sha256'])==r['source_bundle'],'Actual source bundle differs')
    for name,row in r['source_bundle'].items():
        path=record(a.root/name,row['sha256']);require(path.stat().st_size==row['bytes'],'Source size differs')
        if path.suffix=='.py':ast.parse(path.read_text())
    cap=receipt(a.prior/'old-reference-r100/capture.json',CAPTURE_SHA)
    old=receipt(a.reference121/'block-r1/result.json',RESULT121_SHA)
    require(r['original_input']==cap['artifacts']['input'] and r['original_output_reference']==cap['artifacts']['output']
        and r['original_parameters']==cap['original_parameters'] and len(r['original_parameters'])==14,'Original reference/parameter metadata differs')
    x=load(a.prior/'old-reference-r100',cap['artifacts']['input']);ref=load(a.prior/'old-reference-r100',cap['artifacts']['output'])
    require(x.shape==ref.shape==SHAPE and x.dtype==ref.dtype==np.float32,'Native original FP32 input/output differs')
    rows=r['calls'];require(len(rows)==8 and [v['index'] for v in rows]==list(range(8)) and
        ''.join(v['route'] for v in rows)=='ABBAABBA','Exact ABBA schedule required')
    values,timings,comparisons={},[],[];gpu_events=0;cross_run=[]
    for row in rows:
        route=row['route'];require(row['first_for_route']==(row['index']==(0 if route=='A' else 1)),'First-route exclusion differs')
        y=load(a.root/'block-r1',row['output']);require(y.shape==SHAPE and y.dtype==np.float32,'Complete output differs')
        prior_row=old['calls'][row['index']];require(prior_row['route']==route,'R121 source route differs')
        z=load(a.reference121/'block-r1',prior_row['output'])
        require(y.tobytes()==z.tobytes(),'Whole output differs from corresponding R121 capture')
        cross_run.append(dict(file=row['output']['file'],array_bytes_equal=True,file_sha256_equal=row['output']['sha256']==prior_row['output']['sha256']))
        if route in values:require(y.tobytes()==values[route].tobytes(),'Within-route repeat bytes differ')
        else:values[route]=y
        if route=='A':require(h.raw(y)==BASE_RAW,'A differs from original fused baseline')
        whole,update=h.compare(y,ref),h.compare(y-x,ref-x)
        h.agree(whole,row['vs_original_CPU']);h.agree(update,row['residual_update_vs_original_CPU'])
        for actual,reported in ((whole,row['vs_original_CPU']),(update,row['residual_update_vs_original_CPU'])):
            require(actual['fixed_gate_passed']==reported['fixed_reference_gate_passed'] and
                reported['strict_rtol']==1e-4 and reported['strict_atol']==1e-5,'Reported fixed gate/tolerances differ')
        comparisons.append(dict(index=row['index'],route=route,whole_output=whole,residual_update=update))
        t=dict(index=row['index'],route=route,complete_block_seconds=row['complete_block_seconds'])
        if route=='A':
            t['fused_boundary_seconds']=row['stages']['complete_boundary_seconds'];subtotal=t['fused_boundary_seconds']
            for key,value in row['stages'].items():t['fused_'+key]=value
        else:
            inner=row['stages'];s=inner['stages'];heads=s['heads']
            require([v['head'] for v in heads]==list(range(16)),'All16 heads required')
            native=[s[k] for k in ('qkv','proj','fc1','fc2')]+[v[k] for v in heads for k in ('QK','PV')]
            gpu=[s[k] for k in ('norm1','pack_all','softmax_all','merge_all','proj_residual_norm','erf_gelu','final_residual')]
            require(len(native)==36 and len(gpu)==7,'36 native products/7 GPU kernels per block required')
            t.update(input_upload_seconds=s['input_upload_seconds'],owned_output_seconds=s['owned_output_and_finite_seconds'],
                native_linear_seconds=sum(s[k]['complete_seconds'] for k in ('qkv','proj','fc1','fc2')),
                native_QK_seconds=sum(v['QK']['complete_seconds'] for v in heads),native_PV_seconds=sum(v['PV']['complete_seconds'] for v in heads),
                GPU_norm_residual_Erf_seconds=sum(s[k]['complete_seconds'] for k in ('norm1','proj_residual_norm','erf_gelu','final_residual')),
                GPU_head_pack_seconds=s['pack_all']['complete_seconds'],GPU_softmax_seconds=s['softmax_all']['complete_seconds'],
                GPU_head_merge_seconds=s['merge_all']['complete_seconds'],
                native_SDK_run_seconds=sum(v['SDK_run_seconds'] for v in native),
                native_sync_bind_seconds=sum(v['input_sync_bind_seconds']+v['output_sync_seconds'] for v in native),
                GPU_bridge_wall_seconds=sum(v['bridge_wall_seconds'] for v in gpu),GPU_sync_seconds=sum(v['prewrite_seconds']+v['postwrite_seconds'] for v in gpu),
                contained_GPU_kernel_seconds=sum(v['GPU']['GPU_kernel_seconds'] for v in gpu),
                GPU_kernel_finish_wall_seconds=sum(v['GPU']['kernel_finish_seconds'] for v in gpu),
                GPU_profiling_and_release_seconds=sum(v['GPU']['profiling_and_event_release_seconds'] for v in gpu))
            for v in gpu:
                g=v['GPU'];clocks=g['GPU_event_clocks_ns'];method=g['method']
                require(g['global_size']==[METHODS[method][1]] and g['local_size']==[256] and
                    clocks['queued']<=clocks['submit']<=clocks['start']<=clocks['end'] and
                    math.isclose(g['GPU_kernel_seconds'],(clocks['end']-clocks['start'])/1e9,rel_tol=1e-12),'GPU launch/event evidence differs')
                if method=='softmax_all':require(g['finite_error_bits']==0,'Softmax finite flag differs')
                gpu_events+=1
            subtotal=s['input_upload_seconds']+s['owned_output_and_finite_seconds']+sum(v['complete_seconds'] for v in native+gpu)
            require(subtotal<=inner['complete_native_block_seconds']+1e-10,'Stages exceed inner boundary')
            t['inner_other_seconds']=inner['complete_native_block_seconds']-subtotal
            t['outer_other_seconds']=t['complete_block_seconds']-inner['complete_native_block_seconds']
        if route=='A':t['outer_other_seconds']=t['complete_block_seconds']-subtotal
        require(t['outer_other_seconds']>=0,'Stages exceed complete boundary');timings.append(t)
    first=rows[1]['diagnostics_outside_timing'];require(set(first)=={p+v for p in ('qkv','proj','fc1','fc2') for v in 'AC'},'Exact8 diagnostic arrays required')
    diag={}
    for name,row in first.items():
        v=load(a.root/'block-r1',row);k,n=PRODUCTS[name[:-1]];field=name[-1]
        shape=(k//8,1041,8) if field=='A' else (n//4,1041,4)
        require(v.shape==shape and v.dtype==(np.float16 if field=='A' else np.float32),'Native diagnostic layout/type differs')
        prior_row=old['calls'][1]['diagnostics_outside_timing'][name]
        z=load(a.reference121/'block-r1',prior_row)
        require(v.tobytes()==z.tobytes(),'Native diagnostic differs from corresponding R121 capture')
        cross_run.append(dict(file=row['file'],array_bytes_equal=True,file_sha256_equal=row['sha256']==prior_row['sha256']))
        diag[name]=v
    require({p.name for p in (a.root/'block-r1').glob('*.npy')}=={v['output']['file'] for v in rows}|{v['file'] for v in first.values()},'Exactly16 saved actual NPYs required')
    norm=load(a.prior/'old-reference-r100',cap['artifacts']['norm1'])
    logical=np.ascontiguousarray(diag['qkvA'].transpose(1,0,2).reshape(SHAPE),dtype=np.float32)
    norm_comparison=dict(actual_norm1_Half_vs_original_FP32=h.compare(logical,norm),
        actual_norm1_Half_vs_original_FP32_rounded_Half=h.compare(logical,norm.astype(np.float16).astype(np.float32)))
    indices={route:[v['index'] for v in rows if v['route']==route and not v['first_for_route']] for route in 'AB'}
    means={route:{k:statistics.mean(timings[i][k] for i in ids) for k in timings[ids[0]] if k not in ('index','route')} for route,ids in indices.items()}
    for route in 'AB':require(len(indices[route])==3 and math.isclose(means[route]['complete_block_seconds'],r['subsequent_three_means_seconds'][route],rel_tol=1e-12),'Reported warm mean differs')
    expected=dict(rknn_matmul_create=36,rknn_matmul_set_core_mask=36,rknn_create_mem=108,rknn_matmul_set_io_mem=384,
        rknn_mem_sync=972,rknn_matmul_run=144,rknn_destroy_mem=108,rknn_matmul_destroy=36)
    require(r['MatMul_attempts']==r['MatMul_successes']==expected,'Native ledger differs')
    clean=r['candidate_cleanup'];require(clean['native_close_permitted_after_GPU_release'] and not clean['cleanup_errors'] and
        clean['calls_attempted']==clean['calls_completed']==4 and set(clean['resources'])==set(PRODUCTS),'Candidate cleanup differs')
    for name,v in clean['resources'].items():
        require(not v['cleanup_errors'] and v['live_context']==v['live_memories']==0 and (v['K'],v['output_N'])==PRODUCTS[name],'Native owner remains live')
    b=r['baseline_resources'];require(b['closed'] and not b['failed'] and b['resource_counts_balanced'] and not b['cleanup_errors'] and
        b['contexts_created']==b['contexts_destroyed']==1 and b['runs_completed']==4 and
        b['output_buffers_acquired']==b['output_buffers_released']==4 and b['live_contexts']==b['live_output_buffers']==0,'Baseline owner imbalance')
    g=r['GPU_after_close'];require(gpu_events==28 and all(g[k]==28 for k in ('calls_attempted','calls_completed','events_created','events_released')) and
        g['objects_created']==g['objects_released']==dict(Context=1,CommandQueue=1,Program=1,MemObject=118,Kernel=7) and
        g['fully_released'] and g['queue_idle'] and g['closed'] and not g['failed'] and not g['cleanup_errors'] and
        g['live_OpenCL_objects']==g['live_events']==g['live_owned_DMA_fds']==g['live_host_IO_owners']==0,'GPU ownership ledger differs')
    for method,(count,total) in METHODS.items():require(g['methods'][method]['calls_attempted']==g['methods'][method]['calls_completed']==count,'Per-kernel count differs')
    require(g['device_max_parameter_size_bytes']==1024,'Actual device parameter capacity differs')
    for method,argc,size in (('pack_all',99,792),('softmax_all',65,520),('merge_all',34,272)):
        q=g['methods'][method];require(q['kernel_arguments']==argc and q['kernel_parameter_bytes']==size and size<=1024,'Actual batch kernel argument contract differs')
    require(g['parameter_vector_raw_sha256']=={k:r['original_parameters'][k]['raw_sha256'] for k in g['parameter_vector_raw_sha256']} and
        len(g['parameter_vector_raw_sha256'])==10 and g['actual_build_options']=='-cl-std=CL1.2','Original vector/build-options identity differs')
    lifecycle=sum(r['initialization_seconds'].values())+r['destroy_seconds']+sum(v['complete_block_seconds'] for v in rows)
    require(math.isclose(lifecycle,r['summed_eight_call_lifecycle_seconds'],rel_tol=1e-12),'Lifecycle accounting differs')
    require(c['child_returncode']==0 and not c['restore_errors'] and c['baseline']==c['restored'] and c['frequency_governor_limits_restored'] and
        c['restored']['cur_freq']=='300000000' and all(v['actual_frequency_hz']==1000000000 for v in c['observations']),'Frequency/control restoration differs')
    maximum=max(t for v in c['observations'] for t in v['temperatures_c'].values());require(maximum<75,'Observed thermal stop exceeded')
    am,bm=(means[v]['complete_block_seconds'] for v in 'AB')
    late_kernels={}
    for key in ('proj_residual_norm','erf_gelu','final_residual'):
        durations={str(row['index']):row['stages']['stages'][key]['GPU']['GPU_kernel_seconds'] for row in rows if row['route']=='B'}
        others=statistics.mean(v for i,v in durations.items() if i!='6')
        late_kernels[key]=dict(event_seconds_by_call=durations,call6_vs_other_B_mean_ratio=durations['6']/others)
    result=dict(schema='v122_independent_saved_complete_batched_native_DINO0_R1_audit',status='passed',auditor_sha256=h.sha(__file__),
        execution='Saved NumPy, JSON and AST only; no neural/SDK/GPU/board calls',candidate_accepted=False,
        checked_actual_NPYs=16,all8_comparison_records_independently_recomputed=True,within_each_route_all4_outputs_byte_equal=True,
        A4_byte_equal_original_fused_baseline=True,route_raw_sha256={k:h.raw(v) for k,v in values.items()},
        whole_output_vs_original_FP32={k:h.compare(v,ref) for k,v in values.items()},
        residual_update_vs_original_FP32={k:h.compare(v-x,ref-x) for k,v in values.items()},B_vs_A=h.compare(values['B'],values['A']),
        norm1_same_input_diagnostic=norm_comparison,all8_native_diagnostic_layouts_finite=True,
        all16_actual_arrays_byte_equal_corresponding_R121_captures=True,R121_cross_run_array_comparisons=cross_run,
        device_max_parameter_size_bytes=1024,batched_kernel_arity_and_parameter_bytes={'pack_all':[99,792],'softmax_all':[65,520],'merge_all':[34,272]},
        per_call_comparisons=comparisons,subsequent_indices=indices,subsequent_three_mean_seconds=means,call_timings=timings,
        B_slowdown_percent=(bm-am)/am*100,initialization_seconds=r['initialization_seconds'],destroy_seconds=r['destroy_seconds'],summed_eight_call_lifecycle_seconds=lifecycle,
        final_ledger=dict(native_contexts=36,native_memories=108,native_MatMul_runs=144,bindings=384,sync_calls=972,
            outside_timer_diagnostic_FROM_DEVICE_sync_calls=8,GPU_kernels_events_created_released=28,OpenCL_objects_created_released=128,
            baseline_contexts=1,baseline_runs=4,baseline_output_buffers_acquired_released=4,live=0),
        peak_worker_rss_mib_reported=r['peak_rss_mib'],late_call6_GPU_kernel_event_durations=late_kernels,
        GPU_policy_before=r['platform_before']['GPU'],GPU_policy_after=r['platform_after']['GPU'],
        control=dict(child_exit=0,observed_NPU_hz=1000000000,restored_NPU_hz=300000000,exact_policy_restored=True,max_sampled_temperature_c=maximum),
        limits=['Audit passes data consistency, not precision/performance acceptance; both routes retain original fixed strict results',
            'One original native518 DINO0 input; no complete encoder, full model, trajectory or realtime acceptance',
            'Residual updates use executed FP32 y-x; norms and element checks use FP64',
            'Only first B native buffers captured; diagnostics outside timing, never reused as input',
            'Norm1 comparison has the same original input but includes explicit Half storage; remaining intermediate differences are not local error attribution',
            'Context init/destroy costs separate from persistent ABBA timing; GPU events are contained in wall time',
            'B call6 final norm2/Erf/final kernel events are faster; GPU auto governor and endpoint readings do not prove a constant frequency or the cause',
            'R121 versus R122 timing is across separate runs; numerical byte equality is established, not a matched scheduling speedup measurement',
            'No claim about internal NPU accumulation precision, physical DDR traffic or constant GPU kernel frequency',
            'Original weight NPZ was verified by executed board source; no host weight payload was pulled or recomputed'],files=files)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],arrays=16,A_mean=am,B_mean=bm,B_slowdown_percent=result['B_slowdown_percent'],
        B_relative_l2=result['whole_output_vs_original_FP32']['B']['relative_l2'],B_update_relative_l2=result['residual_update_vs_original_FP32']['B']['relative_l2'])))


if __name__=='__main__':main()
