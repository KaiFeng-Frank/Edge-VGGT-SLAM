#!/usr/bin/env python3
"""Independent saved-array/ledger audit only; no SDK, NN, Torch or GPU calls."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
PINS={
 'board-abba-r2/result.json':'03844530afb100568ee1fb7f5707e79dab3d76223f6f95e05957e97cc171ecab',
 'board-abba-control-r2/control.json':'347033d1ebf8348006073d46d39fb7239b9c291e95dc8104429a15f9bbea0333',
 'board-compile-r1/result.json':'27ce62e53665801a821efe9cf47ba8359cbcce0171cb7d0c2fc142f935807e0a',
 'board-check-r1/result.json':'e39c12a5748cb7c8952e8fc9af77c783b6a46fb30d9eaa008a8db248ec11e34c',
 'board-export-r1/export.json':'e09acb577228c66d3bbae9a14ce5f22115f8c1f4ea7b128cafbbc61709dcaa6e',
 'run_sdpa_head_parallel_abba.py':'e74ad2969551f90ef7d6f6eabebd3a1aa67bbb4dd819738f161a37189ee429f6',
}
CAPTURE=Path('/dev/shm/edge-vggt-v103-native-attention/capture-r1/capture.json')
CAPTURE_SHA='e4ffaea2b11df9ac558c119cb4854d490b0030c02b7706eee344b8894cefa686'
BHSD=(1,16,2082,64);PROJ=(1,2082,1024)
GATE=dict(relative_l2_max=1e-5,max_abs_max=1e-3,rtol=1e-4,atol=1e-5)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise AssertionError(message)
def near(a,b):return math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-15)


def comparison(actual,expected):
    diff=actual.astype(np.float64)-expected.astype(np.float64)
    relative=float(np.linalg.norm(diff.ravel())/np.linalg.norm(expected.astype(np.float64).ravel()))
    max_abs=float(np.max(np.abs(diff)));strict=bool(np.allclose(actual,expected,rtol=1e-4,atol=1e-5))
    return dict(relative_L2=relative,max_abs=max_abs,strict_rtol=1e-4,strict_atol=1e-5,strict_pass=strict,
        byte_equal=actual.tobytes()==expected.tobytes(),fixed_gate_passed=bool(strict and relative<=1e-5 and max_abs<=1e-3),
        strict_outlier_count=int(np.count_nonzero(np.abs(diff)>1e-5+1e-4*np.abs(expected.astype(np.float64)))))


def verify_metric(record,actual):
    for key in ('relative_L2','max_abs','strict_rtol','strict_atol'):require(near(record[key],actual[key]),'Recorded numerical metric differs: '+key)
    for key in ('strict_pass','byte_equal','fixed_gate_passed'):require(record[key]==actual[key],'Recorded numerical gate differs: '+key)


def main():
    for name,digest in PINS.items():require(sha(ROOT/name)==digest,'Pinned source/receipt differs: '+name)
    require(sha(CAPTURE)==CAPTURE_SHA,'Original actual capture differs')
    runtime=json.loads((ROOT/'board-abba-r2/result.json').read_text())
    control=json.loads((ROOT/'board-abba-control-r2/control.json').read_text())
    compiled=json.loads((ROOT/'board-compile-r1/result.json').read_text())
    gated=json.loads((ROOT/'board-check-r1/result.json').read_text())
    captured=json.loads(CAPTURE.read_text());loaded={};payload_records={}

    def load(parent,metadata,shape):
        relative=Path(metadata['file']);require(not relative.is_absolute() and '..' not in relative.parts,'Captured NPY path must be receipt-relative')
        path=(parent/relative).resolve(strict=True);require(path.is_relative_to(parent),'NPY escapes receipt directory')
        require(path.stat().st_size==metadata['bytes'] and sha(path)==metadata['sha256'],'NPY file hash/size differs')
        value=np.load(path,allow_pickle=False)
        require(value.shape==shape and value.dtype==np.float32 and value.flags.c_contiguous and np.isfinite(value).all()
            and metadata['shape']==list(shape) and metadata['dtype']=='float32' and value.nbytes==metadata['tensor_bytes']
            and hashlib.sha256(value.tobytes()).hexdigest()==metadata['raw_sha256'] and metadata['finite'],'NPY tensor identity differs')
        return value

    refs=[load(CAPTURE.parent,captured['artifacts'][name],shape) for name,shape in [('attention_tail',BHSD),('proj_input',PROJ)]]
    for i,name in enumerate(('q_post_rope','k_post_rope','v')):
        value=load(CAPTURE.parent,captured['artifacts'][name],BHSD)
        require(runtime['original_input_raw_sha256'][i]==hashlib.sha256(value.tobytes()).hexdigest(),'Original QKV raw identity differs')
    require(np.array_equal(refs[0].transpose(0,2,1,3).reshape(PROJ),refs[1]),'Original captured consumer layout differs')
    for model in compiled['models']:
        p=ROOT/'board-compile-r1'/model['rknn']['file']
        require(p.stat().st_size==model['rknn']['bytes'] and sha(p)==model['rknn']['sha256'],'Compiled payload identity differs')
    require(runtime['status']=='completed_original_SDPA_head_parallel_ABBA_not_quality_or_integration_accepted'
        and runtime['runtime_revision']==2 and runtime['script_sha256']==PINS['run_sdpa_head_parallel_abba.py']
        and runtime['fixed_gate']==GATE and runtime['schedule']==list('ABBA') and runtime['init_flags']==0
        and runtime['baseline_core_mask']==7 and not runtime['candidate_accepted'] and runtime['resources_balanced']
        and runtime['source_and_inputs_unchanged'] and runtime['input_unchanged'] and not runtime['pin_errors']
        and runtime['affinity_restored'] and runtime['environment_restored'] and not runtime['cleanup_errors'],
        'Runtime completion/source/cleanup provenance differs')
    require(runtime['compile_receipt_sha256']==PINS['board-compile-r1/result.json']
        and runtime['gate_receipt_sha256']==PINS['board-check-r1/result.json']
        and runtime['export_receipt_sha256']==PINS['board-export-r1/export.json'] and runtime['capture_sha256']==CAPTURE_SHA,
        'Actual original export/gate/compile/capture linkage differs')
    require(gated['fixed_gate']==GATE and gated['original_FP32_gate_passed']
        and all(x['comparison']['fixed_gate_passed'] for x in gated['checks'])
        and all(x['fixed_gate_passed'] for x in gated['reassembly_checks'].values()),'Original FP32 graph gate not passed')
    require(len(runtime['calls'])==16 and len(runtime['phases'])==4,'Expected16 full ABBA calls required')
    route_first={};warm={'A':[],'B':[]};comparisons={};timing_checks=[];platform=[]
    for n,call in enumerate(runtime['calls']):
        phase,ordinal=divmod(n,4);route='ABBA'[phase]
        require((call['phase'],call['index'],call['route'],call['recorded_first'])==(phase,ordinal,route,ordinal==0),
                'First+3 per-phase ABBA order differs')
        ys=[]
        for role,meta,shape in zip(('attention_values','proj_input'),call['outputs'],(BHSD,PROJ)):
            file=meta['file']
            if file not in loaded:
                loaded[file]=load(ROOT/'board-abba-r2',meta,shape);payload_records[file]=meta
            else:require(meta==payload_records[file],'Same saved path has conflicting call metadata')
            ys.append(loaded[file])
        require(np.array_equal(ys[0].transpose(0,2,1,3).reshape(PROJ),ys[1]),'Actual head-to-consumer layout differs')
        if route not in route_first:route_first[route]=ys
        for i,value in enumerate(ys):
            actual=comparison(value,refs[i]);verify_metric(call['vs_original_CPU_FP32'][i],actual)
            verify_metric(call['vs_first_actual_A_FP16'][i],comparison(value,route_first['A'][i]))
            verify_metric(call['repeat_first_same_route'][i],comparison(value,route_first[route][i]))
            require(np.array_equal(value,route_first['A'][i]) and np.array_equal(value,route_first[route][i]),'Actual A/B or repeat bytes differ')
            comparisons[route+'-'+('head' if i==0 else 'proj')]=actual
        t=call['timings'];dt=call['complete_boundary_seconds'];require(math.isfinite(dt) and dt>0,'Invalid outer boundary')
        context_rows=t['SDK_context_timings'];inner=[row if route=='A' else row['timings'] for row in context_rows]
        require(len(inner)==(1 if route=='A' else 3),'All actual context timing rows required')
        for row in inner:
            require(all(math.isfinite(v) and v>=0 for v in row.values()),'Invalid SDK timing')
            require(row['complete_boundary_seconds']+1e-6>=row['frozen_owned_API_boundary_seconds']
                +row['original_BHSD_to_logical_API_owned_copy_seconds']+row['logical_API_to_original_BHSD_owned_copy_finite_seconds'],
                'Adapter timing omits layout copies')
            require(dt+1e-6>=row['complete_boundary_seconds'],'Outer boundary omits a complete context')
        if route=='A':
            minimum=t['owned_input_copy_seconds']+inner[0]['complete_boundary_seconds']+t['original_consumer_owned_reformat_and_finite_seconds']
        else:
            minimum=sum(t[key] for key in ('owned_input_slice_copy_seconds','three_worker_dispatch_seconds','all_worker_join_seconds',
                'ordered_head_concat_seconds','original_consumer_owned_reformat_and_finite_seconds'))
            ids=[x['owner_thread_id'] for x in context_rows];require(len(set(ids))==3,'Three independent B worker owners required')
            span=max(x['worker_boundary_end'] for x in context_rows)-min(x['worker_boundary_start'] for x in context_rows)
            require(t['three_worker_dispatch_seconds']+t['all_worker_join_seconds']+1e-6>=span,'B dispatch/join omits worker execution span')
        require(dt+1e-6>=minimum and call['SDK_output_buffers_released_before_return'],'Outer boundary omits copy/join/release/consumer work')
        timing_checks.append(dict(call=n,route=route,complete_boundary_seconds=dt,lower_scoped_seconds=minimum,passed=True))
        if ordinal:warm[route].append(dt)
        platform.extend([call['platform_before'],call['platform_after']])
    require(len(loaded)==runtime['saved_unique_output_files']==4 and sum(x['bytes'] for x in payload_records.values())==runtime['saved_unique_output_bytes'],
            'Expected four unique saved output files required')
    summary={route:dict(count=len(v),mean_seconds=sum(v)/len(v),min_seconds=min(v),max_seconds=max(v)) for route,v in warm.items()}
    for route in 'AB':
        require(summary[route]['count']==runtime['summary'][route]['steady_count']==6,'Six recorded steady calls per route required')
        for key in ('mean_seconds','min_seconds','max_seconds'):require(near(summary[route][key],runtime['summary'][route][key]),'Steady boundary aggregate differs')
    ratio=summary['B']['mean_seconds']/summary['A']['mean_seconds'];require(near(ratio,runtime['B_to_A_steady_mean_ratio']),'ABBA ratio differs')
    rows=runtime['resources'];require(len(rows)==8,'All eight context ledgers required')
    for row in rows:
        require(row['contexts_created']==row['contexts_destroyed']==1 and row['runs_completed']==4
            and row['output_buffers_acquired']==row['output_buffers_released']==4 and row['closed']
            and not row['failed'] and not row['live_contexts'] and not row['live_output_buffers']
            and row['resource_counts_balanced'] and not row['cleanup_errors'] and row['init_flags']==0,'SDK lifecycle ledger differs')
        heads=row.get('part',{}).get('heads',16)
        for d in row['input_descriptors']:
            require(d['shape']==d['logical_API_shape']==[1,2082,64,heads] and d['format']==1
                and d['original_BHSD_shape']==[1,heads,2082,64] and d['type']==1 and d['quantization_type']==2
                and d['zero_point']==0 and d['scale']==1. and d['pass_through']==0 and d['size']==heads*2082*64*2,
                'Observed exact NHWC floating input descriptor differs')
        d=row['output_descriptors'][0];require(d['shape']==d['original_BHSD_shape']==[1,heads,2082,64]
            and d['format']==0 and d['type']==1 and d['quantization_type']==2 and d['zero_point']==0 and d['scale']==1.,
            'Observed exact NCHW float output descriptor differs')
        if 'part' in row:
            require(row['core_mask']==row['part']['core_mask'] and row['SDK_thread_ownership_preserved'] and row['worker_pool_shutdown']
                and row['run_thread_ids']==[row['owner_thread_id']]*4 and row['close_thread_ids']==[row['owner_thread_id']],
                'Dedicated construction/run/close owner ledger differs')
    totals={key:sum(row[key] for row in rows) for key in ('contexts_created','contexts_destroyed','runs_completed','output_buffers_acquired','output_buffers_released')}
    require(totals==runtime['SDK_totals']==dict(contexts_created=8,contexts_destroyed=8,runs_completed=32,output_buffers_acquired=32,output_buffers_released=32),
            'Full expected SDK counters differ')
    require([x['core_mask'] for x in rows[:2]]==[7,7] and [x['core_mask'] for x in rows[2:]]==[1,2,4,1,2,4],'Observed A/B core masks differ')
    require(all(v['npu_hz']==1000000000 and v['temperatures_c'] and max(v['temperatures_c'].values())<75. for v in platform),'Call clock/thermal evidence differs')
    observations=control['observations'];max_temp=max(v for obs in observations for v in obs['temperatures_c'].values())
    require(control['status']=='worker_completed_pending_result_review' and control['child_returncode']==0 and control['restore_errors']==[]
        and control['requested_frequency_hz']==1000000000 and control['temperature_limit_c']==75. and control['timeout_seconds']==180.
        and control['seconds']<180 and all(x['actual_frequency_hz']==1000000000 for x in observations) and max_temp<75.
        and control['restored']==control['baseline'] and control['restored']['cur_freq']=='300000000'
        and control['frequency_governor_limits_restored'],'Controller completion/frequency restoration differs')
    require(runtime['all_actual_A_vs_B_byte_equal'] and runtime['all_same_route_repeats_byte_equal']
        and not runtime['all_original_FP32_runtime_gates_passed'] and all(not x['fixed_gate_passed'] for x in comparisons.values()),
        'Actual byte-parity or unchanged FP32 failure flags differ')
    report=dict(schema='v112_independent_saved_SDPA_head_parallel_r2_audit',status='audited_completed_component_not_quality_accepted',audit_passed=True,
        auditor_source_sha256=sha(__file__),runtime_source_sha256=PINS['run_sdpa_head_parallel_abba.py'],receipt_pins=PINS,
        original_capture=dict(file=str(CAPTURE),sha256=CAPTURE_SHA),actual_SDK_NN_Torch_GPU_board_execution_by_auditor=False,
        payloads_verified=payload_records,all16calls32output_records_validated=True,all_A_B_and_repeats_byte_equal=True,
        original_FP32_graph_gate_passed=True,original_FP32_runtime_fixed_gate_passed=False,fixed_gate=GATE,
        independent_comparisons=comparisons,summary=summary,B_to_A_steady_mean_ratio=ratio,observed_percent_lower_boundary=(1-ratio)*100,
        timing_scope_checks=timing_checks,SDK_totals=totals,all_contexts_outputs_workers_closed=True,
        exact_observed_input_NHWC_output_NCHW_layouts_validated=True,dedicated_worker_owner_ids_validated=True,
        controller=dict(sha256=PINS['board-abba-control-r2/control.json'],seconds=control['seconds'],clock_samples=len(observations),
            maximum_temperature_c=max_temp,all_samples_1GHz=True,restored_300MHz_and_prior_configuration=True),
        candidate_or_fullblock_fullmodel_trajectory_realtime_accepted=False,
        limits=['A393ms is standalone complete IO, not the parent fused SDPA245ms profiling row.',
                'Context initialization/destruction and numerical comparison/saving are outside call timers.',
                'SDK internal mapping was not profiled by this flags0 trial; no whole-block integration benefit measured.'])
    output=ROOT/'saved-abba-r2-audit.json';output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(audit_passed=True,output=str(output),sha256=sha(output),summary=summary,
        relative_L2=comparisons['A-head']['relative_L2'],max_abs=comparisons['A-head']['max_abs'])));return 0


if __name__=='__main__':raise SystemExit(main())
