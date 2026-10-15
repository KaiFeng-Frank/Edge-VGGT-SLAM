#!/usr/bin/env python3
"""Independent NumPy saved-data audit; no model, SDK, Torch or GPU execution."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
PINS={
 'board-abba-r1/result.json':'80d7bf6a556dcada5dfe740f3ab81e9524265d6e52de5f14450564e4f0260285',
 'board-abba-control-r1/control.json':'0c15ece0fa5a9a4abfab9db48dc34855c5fd622379312c92f5ec09233de0d31a',
 'board-export-r1/export.json':'0b1bb2df87b6b115e43677625adcdf9a5b9233a6cb3921591fbe6909881fa81e',
 'board-check-r1/result.json':'7099c22a5faced696cfc1708907366ef77c9169eaf3844bd14e789e054062603',
 'board-compile-r1/result.json':'7b5095ad821e42395fbf517ac65d07681f685c2a7029ffdfcbba90048ec5c9a0',
 'run_global0_split_abba.py':'9b35689eed8f282b7f77022721f83a80e84b7abd046d6b589ebd2e54bbbad820',
}
REF=Path('/dev/shm/edge-vggt-v98-native518/reference-r1/capture.json')
REF_SHA='4ebbc4adaaed5e3c3cf8558b4947ec3fc090d56e82f8c61c9509951eb34dd35d'
GLOBAL=(1,2082,1024)
GATE=dict(relative_l2_max=1e-5,max_abs_max=1e-3,rtol=1e-4,atol=1e-5)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise AssertionError(message)
def near(a,b):return math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-14)
def mean(values):return sum(values)/len(values)


def compare(actual,expected):
    diff=actual.astype(np.float64)-expected.astype(np.float64)
    rel=float(np.linalg.norm(diff.ravel())/np.linalg.norm(expected.astype(np.float64).ravel()))
    maximum=float(np.max(np.abs(diff)));strict=bool(np.allclose(actual,expected,rtol=1e-4,atol=1e-5))
    return dict(strict_rtol=1e-4,strict_atol=1e-5,strict_pass=strict,byte_equal=actual.tobytes()==expected.tobytes(),
        relative_L2=rel,max_abs=maximum,fixed_gate_passed=bool(strict and rel<=1e-5 and maximum<=1e-3))


def metric_matches(saved,actual):
    require(all(near(saved[k],actual[k]) for k in ('strict_rtol','strict_atol','relative_L2','max_abs'))
        and all(saved[k]==actual[k] for k in ('strict_pass','byte_equal','fixed_gate_passed')),'Saved numerical metric differs')


def main():
    for name,digest in PINS.items():require(sha(ROOT/name)==digest,'Pinned receipt/source differs: '+name)
    require(sha(REF)==REF_SHA,'Pinned original native reference differs')
    r=json.loads((ROOT/'board-abba-r1/result.json').read_text());control=json.loads((ROOT/'board-abba-control-r1/control.json').read_text())
    gated=json.loads((ROOT/'board-check-r1/result.json').read_text());compiled=json.loads((ROOT/'board-compile-r1/result.json').read_text())
    native=json.loads(REF.read_text());cache={};payloads={}
    def load(parent,meta):
        file=Path(meta['file']);require(not file.is_absolute() and '..' not in file.parts,'Direct receipt-relative NPY required')
        p=(parent/file).resolve(strict=True)
        require(p.is_relative_to(parent) and p.stat().st_size==meta['bytes'] and sha(p)==meta['sha256'],'Saved NPY bytes/hash differ')
        value=np.load(p,allow_pickle=False)
        require(value.shape==GLOBAL and value.dtype==np.float32 and value.flags.c_contiguous and np.isfinite(value).all()
            and list(value.shape)==meta['shape'] and meta['dtype']=='float32'
            and hashlib.sha256(value.tobytes()).hexdigest()==meta['raw_sha256'],'Saved original tensor identity differs')
        return value
    x=load(REF.parent,native['artifacts']['global0_input']);reference=load(REF.parent,native['artifacts']['global0_output'])
    require(r['original_input_raw_sha256']==hashlib.sha256(x.tobytes()).hexdigest(),'Original input identity differs')
    require(r['status']=='completed_original_global0_split_ABBA_not_quality_or_integration_accepted'
        and r['script_sha256']==PINS['run_global0_split_abba.py'] and r['fixed_gate']==GATE
        and r['split_export_sha256']==PINS['board-export-r1/export.json'] and r['split_gate_sha256']==PINS['board-check-r1/result.json']
        and r['split_compile_sha256']==PINS['board-compile-r1/result.json'] and r['original_reference_sha256']==REF_SHA
        and r['source_and_inputs_unchanged'] and r['input_unchanged'] and r['resources_balanced'] and r['complete_expected_execution']
        and r['affinity_restored'] and r['environment_restored'] and not r['pin_errors'] and not r['cleanup_errors']
        and r['suffix_residual_is_original_input'] and r['no_reference_QKV_attention_or_suffix_input_reset']
        and not r['candidate_accepted'],'Completed actual original lifecycle/provenance required')
    require(gated['fixed_gate']==GATE and gated['original_FP32_gate_passed'] and all(v['fixed_gate_passed'] for v in gated['checks'].values())
        and all(v['comparison']['fixed_gate_passed'] for v in gated['head_part_checks'])
        and gated['checks']['complete_composed_vs_parent']['byte_equal'],'Actual complete original FP32 graph gate differs')
    for row in compiled['models']:
        p=ROOT/'board-compile-r1'/row['rknn']['file']
        require(p.stat().st_size==row['rknn']['bytes'] and sha(p)==row['rknn']['sha256'],'Compiled cut payload differs')
    require(len(r['calls'])==8 and r['schedule']==list('ABBA') and len(r['phases'])==4,'Eight complete lifecycle calls required')
    first={};subsequent={'A':[],'B':[]};stage_rows={'A':[],'B':[]};platform=[];comparisons={};timing_checks=[]
    model_sizes={'A':[29798710],'B':[r['cut_models']['prefix']['rknn']['bytes']]+[
        r['head_models'][key]['rknn']['bytes'] for key in ('heads6','heads5','heads5')]+[r['cut_models']['suffix']['rknn']['bytes']]}
    fixed_bytes=sum(v['bytes']*v['hash_calls'] for key,v in r['constructor_hash_files'].items() if isinstance(v,dict))
    for n,call in enumerate(r['calls']):
        phase,ordinal=divmod(n,2);route='ABBA'[phase];route_index=sum(c['route']==route for c in r['calls'][:n])
        require((call['phase'],call['index'],call['route'],call['route_ordinal'],call['first_invocation_of_route'])
            ==(phase,ordinal,route,route_index,route_index==0),'First-per-route lifecycle sampling/order differs')
        meta=call['output'];file=meta['file']
        if file not in cache:cache[file]=load(ROOT/'board-abba-r1',meta);payloads[file]=meta
        else:require(meta==payloads[file],'Repeated path has conflicting output metadata')
        y=cache[file]
        if route not in first:first[route]=y
        require(np.array_equal(y,first['A']) and np.array_equal(y,first[route]),'Actual A/B or repeat arrays differ')
        comparisons[route]=compare(y,reference);metric_matches(call['vs_original_CPU_FP32'],comparisons[route])
        metric_matches(call['vs_first_actual_A_FP16'],compare(y,first['A']));metric_matches(call['repeat_same_route'],compare(y,first[route]))
        detail=call['execution'];t=detail['stage_timings'];outer=call['complete_lifecycle_boundary_seconds']
        require(outer>0 and math.isfinite(outer),'Finite full lifecycle time required')
        expected_hash=fixed_bytes*len(model_sizes[route])+sum(model_sizes[route])
        require(detail['logical_constructor_hashed_bytes']==expected_hash,'Frozen constructor logical hash-byte accounting differs')
        if route=='A':
            minimum=sum(t[k] for k in ('original_input_residual_owned_copy_seconds','complete_block_initialization_seconds','complete_block_destruction_seconds'))+t['complete_block_IO']['complete_boundary_seconds']
            require(len(detail['weighted_contexts'])==1 and not detail['head_contexts'],'A must execute one whole weighted context')
        else:
            minimum=sum(t[k] for k in ('original_input_residual_owned_copy_seconds','prefix_initialization_seconds','prefix_destruction_seconds',
                'head_worker_initialization_seconds','actual_prefix_QKV_slice_copy_seconds','head_dispatch_and_join_seconds',
                'actual_attention_concat_and_finite_seconds','head_worker_destruction_and_shutdown_seconds','suffix_initialization_seconds','suffix_destruction_seconds'))
            minimum+=t['prefix_IO']['complete_boundary_seconds']+t['suffix_IO']['complete_boundary_seconds']
            require(len(detail['weighted_contexts'])==2 and len(detail['head_contexts'])==3,'B must execute both weighted cuts and all3 heads')
            head=t['head_IO'];ids=[row['owner_thread_id'] for row in head]
            span=max(row['worker_boundary_end'] for row in head)-min(row['worker_boundary_start'] for row in head)
            require(len(set(ids))==3 and t['head_dispatch_and_join_seconds']+1e-6>=span,'Three independent head workers/complete join span required')
        require(outer+1e-6>=minimum,'Full boundary omits a recorded lifecycle/copy/IO stage')
        require(all(row['closed'] and not row['live_contexts'] and not row['live_output_buffers'] for row in detail['weighted_contexts']+detail['head_contexts']),
                'Every call must return after all contexts/output releases')
        timing_checks.append(dict(call=n,route=route,complete_boundary_seconds=outer,scoped_stage_sum_seconds=minimum,passed=True))
        if not call['first_invocation_of_route']:subsequent[route].append(outer);stage_rows[route].append(t)
        platform.extend([call['platform_before'],call['platform_after']])
    require(len(cache)==r['saved_unique_output_files']==2 and sum(meta['bytes'] for meta in payloads.values())==r['saved_unique_output_bytes'],'Exactly two actual unique full outputs required')
    summary={route:dict(count=len(v),mean_seconds=mean(v),min_seconds=min(v),max_seconds=max(v)) for route,v in subsequent.items()}
    for route in 'AB':
        require(summary[route]['count']==r['summary'][route]['subsequent_lifecycle_count']==3,'Three post-first lifecycle samples per route required')
        for key in ('mean_seconds','min_seconds','max_seconds'):require(near(summary[route][key],r['summary'][route][key]),'Recorded lifecycle aggregate differs')
        expected=fixed_bytes*len(model_sizes[route])+sum(model_sizes[route]);h=r['constructor_hash_summary'][route]
        require(h==dict(calls=4,logical_hashed_bytes=4*expected,per_call_logical_bytes=[expected]*4),'Route logical hash-byte totals differ')
    ratio=summary['B']['mean_seconds']/summary['A']['mean_seconds'];require(near(ratio,r['B_to_A_subsequent_mean_ratio']),'Complete block ratio differs')
    def average(route,path):
        vals=[]
        for row in stage_rows[route]:
            value=row
            for key in path:value=value[key]
            vals.append(value)
        return mean(vals)
    stages=dict(A_IO_seconds=average('A',['complete_block_IO','complete_boundary_seconds']),
        A_initialization_seconds=average('A',['complete_block_initialization_seconds']),A_destruction_seconds=average('A',['complete_block_destruction_seconds']),
        B_prefix_IO_seconds=average('B',['prefix_IO','complete_boundary_seconds']),B_head_dispatch_join_seconds=average('B',['head_dispatch_and_join_seconds']),
        B_suffix_IO_seconds=average('B',['suffix_IO','complete_boundary_seconds']))
    for key in ('prefix_initialization_seconds','head_worker_initialization_seconds','suffix_initialization_seconds',
                'prefix_destruction_seconds','head_worker_destruction_and_shutdown_seconds','suffix_destruction_seconds'):
        stages['B_'+key]=average('B',[key])
    stages['B_partition_IO_sum_seconds']=sum(stages[key] for key in ('B_prefix_IO_seconds','B_head_dispatch_join_seconds','B_suffix_IO_seconds'))
    stages['B_initialization_sum_seconds']=sum(stages['B_'+key] for key in ('prefix_initialization_seconds','head_worker_initialization_seconds','suffix_initialization_seconds'))
    stages['B_destruction_shutdown_sum_seconds']=sum(stages['B_'+key] for key in ('prefix_destruction_seconds','head_worker_destruction_and_shutdown_seconds','suffix_destruction_seconds'))
    rows=r['resources'];require(len(rows)==24,'All24 owner ledgers required')
    for row in rows:
        require(row['contexts_created']==row['contexts_destroyed']==row['runs_completed']==1 and row['closed'] and not row['failed']
            and row['output_buffers_acquired']==row['output_buffers_released']==len(row['output_descriptors'])
            and row['resource_counts_balanced'] and not row['live_contexts'] and not row['live_output_buffers'] and not row['cleanup_errors'],
            'SDK owner/resource ledger differs')
        if 'part' in row:
            require(row['core_mask']==row['part']['core_mask'] and row['SDK_memory']['weight_bytes']==0 and row['worker_pool_shutdown']
                and row['SDK_thread_ownership_preserved'] and row['run_thread_ids']==[row['owner_thread_id']]
                and row['close_thread_ids']==[row['owner_thread_id']],'Unweighted dedicated worker ownership differs')
        else:require(row['core_mask']==7,'All whole/prefix/suffix weighted contexts use7')
        for desc in row['input_descriptors']+row['output_descriptors']:
            semantic=desc.get('original_semantic_shape',desc.get('original_BHSD_shape'));fmt=desc['format']
            expected=semantic if len(semantic)==3 or fmt!=1 else [semantic[0],semantic[2],semantic[3],semantic[1]]
            require(desc['shape']==expected and desc['type']==1 and desc['quantization_type']==2 and desc['zero_point']==0
                and desc['scale']==1. and desc['pass_through']==0 and desc['size']==int(np.prod(semantic))*2,
                'Exact rank3/rank4 logical float descriptor differs')
    totals={key:sum(row[key] for row in rows) for key in ('contexts_created','contexts_destroyed','runs_completed','output_buffers_acquired','output_buffers_released')}
    require(totals==r['SDK_totals']==dict(contexts_created=24,contexts_destroyed=24,runs_completed=24,output_buffers_acquired=32,output_buffers_released=32),
            'Expected full actual pipeline SDK counts differ')
    require(all(p['npu_hz']==1000000000 and max(p['temperatures_c'].values())<75. for p in platform),'Call clock/thermal evidence differs')
    observations=control['observations'];maximum=max(v for p in observations for v in p['temperatures_c'].values())
    require(control['status']=='worker_completed_pending_result_review' and control['child_returncode']==0 and not control['restore_errors']
        and control['seconds']<180 and control['requested_frequency_hz']==1000000000 and maximum<75.
        and all(p['actual_frequency_hz']==1000000000 for p in observations) and control['restored']==control['baseline']
        and control['restored']['cur_freq']=='300000000' and control['frequency_governor_limits_restored'],'Controller completion/restoration differs')
    require(r['all_actual_A_B_byte_equal'] and r['all_same_route_repeats_byte_equal'] and not r['all_original_FP32_runtime_gates_passed']
        and all(not v['fixed_gate_passed'] for v in comparisons.values()),'Actual output parity/strict FP32 failure flags differ')
    report=dict(schema='v114_independent_saved_complete_global0_split_audit',status='audited_completed_rejected_schedule',audit_passed=True,
        auditor_source_sha256=sha(__file__),receipt_source_pins=PINS,original_reference=dict(file=str(REF),sha256=REF_SHA),
        SDK_NN_Torch_GPU_board_execution_by_auditor=False,payloads_verified=payloads,all8_output_records_validated=True,
        all_A_B_and_repeats_byte_equal=True,original_FP32_graph_gate_passed=True,original_FP32_runtime_fixed_gate_passed=False,
        saved_numeric_metric_recompute_tolerance=dict(rtol=1e-11,atol=1e-14),tensor_fixed_gate_unchanged=True,
        original_FP32_L2_recompute_minus_board_record={route:comparisons[route]['relative_L2']-next(c['vs_original_CPU_FP32']['relative_L2'] for c in r['calls'] if c['route']==route) for route in 'AB'},
        original_FP32_metrics=comparisons,fixed_gate=GATE,summary=summary,B_to_A_lifecycle_mean_ratio=ratio,
        observed_percent_slower=(ratio-1)*100,subsequent3_stage_means=stages,timing_scope_checks=timing_checks,
        logical_constructor_hash_summary=r['constructor_hash_summary'],hash_bytes_are_not_measured_DDR_or_cold_pagecache=True,
        SDK_totals=totals,all_contexts_outputs_workers_closed=True,rank3_rank4_API_layout_and_owner_fields_verified=True,
        controller=dict(sha256=PINS['board-abba-control-r1/control.json'],seconds=control['seconds'],maximum_temperature_c=maximum,
            clock_samples=len(observations),all_samples_1GHz=True,prior_configuration_restored_300MHz=True),
        decision='Reject this fresh-context split schedule; no production integration.',
        generic_head_parallelism_impossible_claim=False,persistent_context_or_fullmodel_performance_measured=False,
        limits=['Partition IO sums are extracted recorded stages with different compiler boundaries; not a measured persistent-context counterfactual.',
                'Only one fixed original native input and eight complete-block calls, not a live trajectory or fresh-stream rate.'])
    p=ROOT/'saved-global0-split-audit.json';p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(audit_passed=True,sha256=sha(p),summary=summary,stages=stages,FP32=comparisons['A'])));return 0


if __name__=='__main__':raise SystemExit(main())
