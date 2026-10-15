#!/usr/bin/env python3
"""Saved-only audit of shared-memory softmax attention and controlled ABBA.

No Torch/model/SDK/ORT/GPU/board calls. Recompute fixed comparisons and the
CPU probability formula from saved scores; retain platform rounding differences.
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

R = Path('/dev/shm/edge-vggt-v119-review')
D117 = Path('/path/to/research-artifacts/v117-native518-attention-dump-r1')
D118 = Path('/path/to/research-artifacts/v118-native518-native-attention-r1')
DRIVER_SHA = '10b15ffa734866283aede7ac865ab698114d204dd4f691d42cb0e0450059b092'
BRIDGE_SHA = '73ff0c5ad33e0c334485553e9d8ab0764e6eb816a8f743b399531a58b619155e'
KERNEL_SHA = '5cbb36e03e09c38db9e6769f067e3168e1cf7f67d719009601c4456e9f2b150a'
HELPER_SHA = '480ea456c56d79c1f9660a3f1d34bc063c86f4000bd819e1c50cc2b726d9d12b'
CF_SHA = '03e9a5797ca21c542e467a68057f7290f0a881a689b273db74f7f7d5d3dea94d'
DUMP_SHA = '565054e051afd2565bcd07d6487e02cedeca62456e09e1214021f30c9c4f46d4'
TAP_SHA = '1f89eac69b85ec84ea76f9e8ca0a50ca7d4c5f19620c4159f41eaf6f3661225a'
A_RAW = '00545f97739c7b0a0748918cf766fcd808a9060e2edf6772ec0e88ed502e4fff'
B_RAW = 'ec16194888f33dd49e6712f38336b412b831c29d91b0780fa7ba438b6472004c'


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4<<20), b''): h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=R); p.add_argument('--archive117', type=Path, default=D117)
    p.add_argument('--archive118', type=Path, default=D118); p.add_argument('--output', type=Path, default=R/'saved-shared-attention-audit.json')
    a = p.parse_args(); root, prior, native_prior = a.root, a.archive117, a.archive118; files = []; checked = 0
    def record(path, digest=None):
        actual=sha(path);require(digest is None or actual==digest,'Pinned identity differs: '+str(path))
        files.append(dict(file=str(path),bytes=path.stat().st_size,sha256=actual));return path
    def receipt(path, digest=None): return json.loads(record(path,digest).read_text())
    helper=record(Path(__file__).resolve().parents[1]/'original_native518_native_attention/audit_same_qkv_native_saved.py',HELPER_SHA)
    spec=importlib.util.spec_from_file_location('_v119_frozen_saved_metrics',helper);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    def load(path, row, new=False):
        nonlocal checked
        require(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'NPY file identity differs')
        v=np.load(path,allow_pickle=False)
        require(list(v.shape)==row['shape'] and str(v.dtype)==row['dtype'] and np.isfinite(v).all() and h.raw(v)==row['raw_sha256'],'NPY shape/dtype/finite/raw differs')
        files.append(dict(file=str(path),bytes=path.stat().st_size,sha256=row['sha256'],raw_sha256=h.raw(v),shape=list(v.shape),dtype=str(v.dtype),finite=True));checked+=int(new);return v
    for name,digest in (('run_shared_attention_board.py',DRIVER_SHA),('native_attention_softmax_opencl.py',BRIDGE_SHA),('attention_softmax.cl',KERNEL_SHA)):
        path=record(root/name,digest)
        if path.suffix=='.py':ast.parse(path.read_text())
    data=receipt(root/'shared-attention-r1/result.json');control=receipt(root/'control-r1/control.json')
    require(data['script_sha256']==DRIVER_SHA and data['bridge_sha256']==BRIDGE_SHA and data['kernel_sha256']==KERNEL_SHA and
            data['status']=='completed_shared_attention_component_not_full_model_accepted' and data['cleanup_errors']==[] and
            all(data[k] for k in ('source_and_inputs_unchanged','affinity_restored','environment_restored','native_close_permitted_after_GPU_release','resource_counts_balanced')),'Clean actual execution/source required')
    require(data['queries']==data['keys']==1041 and data['heads']==16 and data['head_dim']==64 and data['padded_storage_keys']==1056 and
            data['all_original_keys_kept'] and data['padding_excluded_from_softmax'] and data['candidate_accepted'] is False,'Original dimensions/scope required')
    cf=receipt(prior/'qkv-counterfactual-r100/result.json',CF_SHA);dump=receipt(prior/'dump-r1/result.json',DUMP_SHA)
    tap=receipt(prior/'old-reference-r100/npu-result.json',TAP_SHA)
    require(cf['baseline_all_points_byte_equal_original_CPU'] and cf['QKV_substitution_calls']==1 and cf['cleanup_errors']==[] and dump['final_baseline_byte_equal'],'Original CPU/fused reference lineage required')
    ref=load(prior/'qkv-counterfactual-r100/attn_proj_input.npy',cf['artifacts']['attn.proj.input'])
    drow=next(r for r in dump['dump_files'] if r['file'].startswith('008_'));z=load(prior/'dump-r1/dumps'/drow['file'],drow)
    fused=np.ascontiguousarray(z[:,:,0,:].transpose(0,2,1));trow=tap['calls'][0]['artifact']
    archive=record(prior/'old-reference-r100/npu-call0.npz',trow['sha256'])
    with np.load(archive,allow_pickle=False) as z:qkv=np.array(z['attn.qkv'],copy=True,order='C')
    require(qkv.shape==(1,1041,3072) and qkv.dtype==np.float32 and h.raw(qkv)==data['QKV_raw_sha256']==cf['actual_NPU_tap_artifact']['arrays']['attn.qkv']['raw_sha256'] and
            qkv.astype(np.float16).astype(np.float32).tobytes()==qkv.tobytes(),'Identical actual Half-representable QKV required')
    require(h.raw(ref)==data['CPU_reference_raw_sha256'] and h.raw(fused)==data['fused_reference_raw_sha256'],'Reference logical identities differ')
    old=receipt(native_prior/'native-attention-r1/result.json');old_y=load(native_prior/'native-attention-r1/call0.npy',old['calls'][0]['output'])
    require(h.raw(old_y)==A_RAW,'Frozen original CPU-softmax native baseline required')
    rows=data['calls'];require(len(rows)==8 and [r['index'] for r in rows]==list(range(8)) and ''.join(r['route'] for r in rows)=='ABBAABBA','Ordered complete ABBA required')
    values={};timings=[];comparison_count=0
    for r in rows:
        route=r['route'];require(r['first_call_for_route']==(r['index']==(0 if route=='A' else 1)) and [x['head'] for x in r['heads']]==list(range(16)),'First-route/all-head identities differ')
        v=load(root/'shared-attention-r1'/r['output']['file'],r['output'],True)
        require(v.shape==(1,1041,1024) and v.dtype==np.float32 and h.raw(v)==(A_RAW if route=='A' else B_RAW),'Complete owned route output identity differs')
        if route in values:require(v.tobytes()==values[route].tobytes(),'Within-route repeat byte equality failed')
        else:values[route]=v
        if route=='A':require(v.tobytes()==old_y.tobytes(),'ABBA baseline differs from frozen original native output')
        h.agree(h.compare(v,ref),r['vs_original_CPU_same_QKV']);h.agree(h.compare(v,fused),r['vs_unchanged_fused_same_QKV']);comparison_count+=2
        t=dict(index=r['index'],route=route,complete_attention_seconds=r['complete_attention_seconds'])
        if route=='A':
            for product in ('QK','PV'):
                for key in ('pack_upload_bind_seconds','SDK_run_seconds','sync_owned_output_finite_seconds','complete_product_seconds'):
                    t[product+'_'+key]=sum(x[product][key] for x in r['heads'])
            t['CPU_softmax_crop_scale_probability_store_seconds']=sum(x['softmax_crop_scale_probability_store_seconds'] for x in r['heads'])
            subtotal=t['QK_complete_product_seconds']+t['PV_complete_product_seconds']+t['CPU_softmax_crop_scale_probability_store_seconds']
        else:
            for key in ('QK_pack_upload_bind_seconds','QK_SDK_run_seconds','PV_upload_bind_seconds','PV_SDK_run_seconds','PV_sync_owned_finite_seconds'):
                t[key]=sum(x[key] for x in r['heads'])
            for key in ('QK_C_FROM_DEVICE_seconds','PV_A_prewrite_TO_DEVICE_seconds','finite_flag_reset_seconds','kernel_enqueue_seconds','kernel_finish_seconds',
                        'finite_flag_read_check_seconds','PV_A_postwrite_FROM_DEVICE_seconds','profiling_and_event_release_seconds','complete_bridge_seconds','GPU_kernel_seconds'):
                t['shared_'+key]=sum(x['shared_softmax'][key] for x in r['heads'])
            for x in r['heads']:
                s=x['shared_softmax'];clock=s['GPU_event_clocks_ns'];require(s['finite_error_bits']==0 and clock['queued']<=clock['submit']<=clock['start']<=clock['end'],'GPU event/error gate differs')
                require(math.isclose(s['GPU_kernel_seconds'],(clock['end']-clock['start'])/1e9,rel_tol=1e-12),'GPU duration/event clocks differ')
                sequential=sum(s[k] for k in ('QK_C_FROM_DEVICE_seconds','PV_A_prewrite_TO_DEVICE_seconds','finite_flag_reset_seconds','kernel_enqueue_seconds','kernel_finish_seconds',
                           'finite_flag_read_check_seconds','PV_A_postwrite_FROM_DEVICE_seconds','profiling_and_event_release_seconds'))
                require(sequential<=s['complete_bridge_seconds']+1e-10 and s['GPU_kernel_seconds']<=s['kernel_finish_seconds']+1e-6,'Bridge sequential versus contained event timing differs')
            subtotal=sum(t[k] for k in ('QK_pack_upload_bind_seconds','QK_SDK_run_seconds','PV_upload_bind_seconds','PV_SDK_run_seconds','PV_sync_owned_finite_seconds','shared_complete_bridge_seconds'))
        t['outer_other_seconds']=t['complete_attention_seconds']-subtotal;require(t['outer_other_seconds']>=0,'Stages exceed complete timer');timings.append(t)
    validation=data['same_scores_validation'];require(validation['head']==15 and validation['after_call']==1 and validation['inside_benchmark_timing'] is False,'Same-score capture scope differs')
    base=root/'shared-attention-r1';scores=load(base/validation['scores']['file'],validation['scores'],True)
    gp=load(base/validation['GPU_probabilities']['file'],validation['GPU_probabilities'],True);cp=load(base/validation['CPU_probabilities_once_Half']['file'],validation['CPU_probabilities_once_Half'],True)
    require(scores.shape==gp.shape==(1041,1056) and scores.dtype==np.float32 and gp.dtype==cp.dtype==np.float16 and cp.shape==(1041,1041),'Native-score/probability shapes/dtypes differ')
    require(not np.any(scores[:,1041:]) and gp[:,1041:].tobytes()==np.zeros((1041,15),np.float16).tobytes() and
            np.all(gp>=0) and np.all(gp<=1) and np.all(cp>=0) and np.all(cp<=1),'Zero padding/valid probability values required')
    logits=np.ascontiguousarray(scores[:,:1041]);logits*=np.float32(.125);logits-=logits.max(axis=1,keepdims=True)
    np.exp(logits,out=logits);logits/=logits.sum(axis=1,keepdims=True,dtype=np.float32);recomputed=logits.astype(np.float16)
    cpu_formula=h.compare(recomputed,cp);require(cpu_formula['strict_pass'],'Saved CPU probability formula failed original fixed element gate')
    probability=h.compare(gp[:,:1041],cp);h.agree(probability,validation['GPU_Half_vs_CPU_softmax_once_Half']);require(probability['strict_pass'],'Same-score GPU probability fixed gate failed')
    comparison_count+=1;require(checked==11,'All11 new NPYs must be checked')
    indices={route:[r['index'] for r in rows if r['route']==route and not r['first_call_for_route']] for route in 'AB'}
    means={route:{k:statistics.mean(timings[i][k] for i in ids) for k in timings[ids[0]] if k not in ('index','route')} for route,ids in indices.items()}
    for route in 'AB':require(len(indices[route])==3 and math.isclose(means[route]['complete_attention_seconds'],data['subsequent_three_means_seconds'][route],rel_tol=1e-12),'Reported three-sample mean differs')
    expected=dict(rknn_matmul_create=2,rknn_matmul_set_core_mask=2,rknn_create_mem=6,rknn_matmul_set_io_mem=518,rknn_mem_sync=898,rknn_matmul_run=256,rknn_destroy_mem=6,rknn_matmul_destroy=2)
    require(data['SDK_attempts']==data['SDK_successes']==expected and len(data['resources'])==2 and all(r['live_context']==r['live_memories']==0 and r['cleanup_errors']==[] for r in data['resources']),'Native256-products/2-contexts/6-memories ledger differs')
    g=data['GPU_after_close'];require(all(g[k]==64 for k in ('calls_attempted','calls_completed','events_created','events_released')) and
        g['objects_created']==g['objects_released']==dict(Context=1,CommandQueue=1,Program=1,Kernel=1,MemObject=3) and g['cleanup_errors']==[] and
        g['closed'] and g['fully_released'] and g['queue_idle'] and not g['failed'] and g['live_events']==g['live_owned_DMA_fds']==g['live_OpenCL_objects']==0,'Final OpenCL/event/import ownership differs')
    require(g['managed_host_consistency'] and g['actual_build_options']=='-cl-std=CL1.2' and all(g[name]['imported_via_dma_fd'] and not g[name]['copied_to_GPU_staging_buffer'] for name in ('QK_C','PV_A')),'Actual shared-memory/build-option contract differs')
    lifecycle=data['init_seconds']+data['destroy_seconds']+sum(r['complete_attention_seconds'] for r in rows)
    require(math.isclose(lifecycle,data['summed_eight_call_lifecycle_seconds'],rel_tol=1e-12),'Init/calls/destroy accounting differs')
    require(control['child_returncode']==0 and control['restore_errors']==[] and control['baseline']==control['restored'] and control['frequency_governor_limits_restored'] and
            control['restored']['cur_freq']=='300000000' and all(r['actual_frequency_hz']==1000000000 for r in control['observations']),'Controller frequency/policy restoration failed')
    maximum=max(v for r in control['observations'] for v in r['temperatures_c'].values());require(maximum<75,'Observed thermal stop threshold exceeded')
    am,bm=[means[r]['complete_attention_seconds'] for r in 'AB']
    result=dict(schema='v119_independent_saved_shared_attention_ABBA_audit',status='passed',auditor_sha256=sha(__file__),execution='Saved NumPy, JSON and AST only; no Torch/model/SDK/ORT/GPU/board calls',
        checked_new_NPYs=11,all8_output_comparison_records_recomputed=True,reported_metric_objects_recomputed=comparison_count,
        within_each_route_all4_outputs_byte_equal=True,A_outputs_byte_equal_frozen_native118=True,route_output_raw_sha256={'A':A_RAW,'B':B_RAW},
        identical_actual_QKV_Half_roundtrip_byte_equal=True,precision_vs_original_CPU_same_QKV={route:h.compare(v,ref) for route,v in values.items()},
        precision_vs_unchanged_fused_same_QKV={route:h.compare(v,fused) for route,v in values.items()},B_vs_A=h.compare(values['B'],values['A']),
        same_score_probabilities_GPU_vs_saved_CPU_Half=probability,saved_CPU_probability_formula_recomputed=cpu_formula,
        host_CPU_formula_note='Host NumPy exp/sum gives minor Half rounding differences from saved board CPU probabilities; original fixed rtol1e-4/atol1e-5 passes. No altered tolerance or byte-equality claim.',
        all15_probability_padding_columns_positive_zero=True,all15_native_score_padding_columns_zero=True,
        probability_Half_row_sum_ranges={route:[float(v.astype(np.float64).sum(axis=1).min()),float(v.astype(np.float64).sum(axis=1).max())] for route,v in [('CPU',cp),('GPU',gp)]},
        subsequent_indices=indices,subsequent_three_mean_seconds=means,complete_attention_call_seconds=[r['complete_attention_seconds'] for r in rows],
        component_ABBA_mean_reduction_percent=(am-bm)/am*100,component_mean_speed_ratio=am/bm,
        init_seconds=data['init_seconds'],destroy_seconds=data['destroy_seconds'],summed_eight_call_lifecycle_seconds=lifecycle,call_timings=timings,
        final_resource_ledger=dict(native_contexts=2,native_memories=6,native_MatMul_products=256,GPU_softmax_calls=64,GPU_events_created_released=64,OpenCL_objects_created_released=7,live=0),
        initial_nested_metric_ledger_not_independent_snapshot=True,initial_metric_note='bridge_initial nested objects_released aliases mutable bridge counters and was updated by close; no assertion of initial releases0. Only final ledger and actual per-call records are accepted.',
        control=dict(exit_code=0,observed_NPU_hz=1000000000,restored_NPU_hz=300000000,restored_exact_policy=True,sampled_max_temperature_c=maximum),
        GPU_policy=data['platform_before']['GPU'],GPU_policy_after=data['platform_after']['GPU'],
        limits=['One original DINO0 actual-QKV case; no whole-model or trajectory acceptance','GPU simple_ondemand policy and endpoint readings do not prove a fixed frequency throughout kernels',
                'Kernel event durations are contained in bridge wall time; do not add them twice','Shared DMA imports avoid explicit GPU matrix staging; no physical DDR traffic, end-to-end zero-copy or device-parallelism claim',
                'Same-score validation covers captured final head15 on first B call; remaining heads have source/finite/output checks, not per-head probability captures',
                'Native/type1 FP32 output and Half inputs do not prove internal accumulator precision','SDK/event ownership is verified from saved ledgers and records, not new hardware observations'],files=files)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],arrays=checked,A_mean=am,B_mean=bm,reduction_percent=result['component_ABBA_mean_reduction_percent'],probability_strict=probability['strict_pass'])))


if __name__=='__main__':
    main()
