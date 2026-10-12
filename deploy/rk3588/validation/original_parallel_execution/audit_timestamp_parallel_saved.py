#!/usr/bin/env python3
"""Saved JSON/map NumPy audit only; no model, Torch, ORT, SDK or board calls."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import numpy as np

SCHEMAS = {
    'v79_original_VGGT_parallel_shared_timestamp_replay': 'replay_parallel_shared_board.py',
    'v79_original_VGGT_parallel_shared_timestamp_replay_two_lanes': 'replay_parallel_two_lanes_board.py',
}
PHASES = ('arrival_to_decode_seconds','decode_seconds','latest_slot_seconds','worker_queue_seconds',
          'preprocessing_seconds','full_model_and_codec_seconds','owned_copy_seconds',
          'result_queue_seconds','ordered_commit_wait_seconds','SL4_and_points_seconds','publication_seconds')

def require(ok, message):
    if not ok: raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4<<20), b''): h.update(chunk)
    return h.hexdigest()

def close(a,b):
    return math.isclose(a,b,rel_tol=1e-11,abs_tol=2e-8)

def audit(replay_dir, control_path, source_dir):
    rp=replay_dir/'results.json'; cp=control_path/'control.json' if control_path.is_dir() else control_path
    r=json.loads(rp.read_text()); c=json.loads(cp.read_text()); schema=r['schema']
    require(schema in SCHEMAS,'Unsupported replay evidence schema')
    lanes=len(r['worker_core_masks']); require(lanes in (2,3) and r['worker_core_masks']==[1,2,4][:lanes],'Declared lanes/masks differ')
    require(r['independent_original_model_cache_sessions']==lanes,'Independent original model count differs')
    bindings={SCHEMAS[schema]:r['script_sha256'],'full_parallel_shared_board.py':r['inherited_full_runner_sha256'],
              'commit_parallel_predictions_board.py':r['saved_executor_source_sha256']}
    for name,digest in bindings.items(): require(sha(source_dir/name)==digest,'Frozen source SHA differs: '+name)
    registry=source_dir/'mask-shared-buffer-overlay/vggt_slam/shared_context_registry.py'
    require(sha(registry)==r['shared_registry_source_sha256'],'Shared registry source SHA differs')
    observations=c['observations']; require(observations,'Missing frequency observations')
    freqs=sorted({s['actual_frequency_hz'] for s in observations})
    maximum=max(v for s in observations for v in s['temperatures_c'].values())
    require(freqs==[1000000000] and maximum<75,'Frequency/thermal controller failed')
    require(c['frequency_governor_limits_restored'] is True and not c['restore_errors'],'Frequency restoration failed')
    require(int(c['restored']['cur_freq'])==300000000 and int(c['restored']['userspace/set_freq'])==300000000,'Effective/raw request300MHz not restored')
    require(all(c['restored'][k]==c['baseline'][k] for k in ('governor','min_freq','max_freq')),'Governor/limits changed')
    out=dict(schema='v79_saved_parallel_timestamp_audit',auditor_sha256=sha(__file__),
        experiment_schema=schema,declared_lane_count=lanes,result_sha256=sha(rp),control_sha256=sha(cp),
        frozen_sources=bindings,shared_registry_source_sha256=r['shared_registry_source_sha256'],
        source_identity_verified=True,actual_model_SDK_board_calls_by_auditor=0,
        controller=dict(status=c['status'],child_returncode=c['child_returncode'],sample_count=len(observations),
            frequencies_hz=freqs,maximum_temperature_c=maximum,restored_effective_hz=300000000,
            restored_raw_request_hz=300000000,governor_limits_restored=True))
    if r.get('status')!='completed' or c['child_returncode']!=0:
        require(c['child_returncode']!=0,'Noncompleted result with successful controller needs investigation')
        log=cp.parent/'worker.log'; text=log.read_text() if log.is_file() else ''
        out.update(status='verified_failed_run_no_replay_acceptance',experiment_completed=False,
            saved_partial_result_status=r.get('status'),saved_partial_fresh_pose_count=r.get('fresh_pose_count'),
            published_pose_count_authoritative=False,SDK_context_cleanup_verified=False,
            thread_join_verified=False,SLAM_throughput_hz=None,latency_P95_seconds=None,
            fatal_SIGSEGV=(c['child_returncode']==-11),log_sha256=sha(log) if log.is_file() else None,
            SDK_allocation_failure_in_log=bool(re.search(r'failed to (?:malloc npu memory|allocate handle)',text)),
            SDK_allocation_sizes_in_log=[int(v) for v in re.findall(r'failed to malloc npu memory, size:\s*(\d+)',text)],
            limitations=['Partial running JSON is not a successful run or cleanup receipt.','Process termination does not prove explicit SDK destruction.'])
        return out
    require(c['status']=='worker_completed_pending_result_review' and not r['cleanup_errors'] and r['all_worker_threads_joined'] is True,'Successful replay cleanup/controller differs')
    s=r['source']; u=r['updates']; n=len(u)
    require(n>0 and n==r['fresh_pose_count']==r['dispatched_pairs']==len(r['dispatches']),'Fresh/dispatch counts differ')
    require(r['EOF_tail_drained'] is True and s['error'] is None,'EOF/error state differs')
    require(s['original_frames']==r['original_frames']==798 and s['generated']+s['skipped_before_decode']==798,'Original798 accounting differs')
    require(s['generated']==s['delivered']+s['overwritten'] and s['delivered']==n+1,'Latest-slot/seed accounting differs')
    require(s['queue_capacity']==r['queue_capacity']==r['per_worker_outstanding_capacity']==1 and r['total_uncommitted_capacity']==lanes,'Queue contract differs')
    require(r['unique_parent_SL4_solver'] is True and r['SL4_executed'] is True,'Original unique SL4 absent')
    previous=(r['seed_sequence'],r['seed_timestamp'],None); timeline=[]; sum_error=0.; latencies=[]
    for i,(row,dispatch) in enumerate(zip(u,r['dispatches'])):
        case=dispatch['case']; lane=dispatch['worker']
        require(case['case_index']==i and case['pair']==row['pair']==row['revision']==i+1,'Commit order/revision differs')
        require(0<=lane<lanes and row['worker']==lane and row['core_mask']==r['worker_core_masks'][lane],'Lane/core identity differs')
        require(row['source_indices']==case['source_indices'] and row['timestamps']==case['timestamps'] and row['image_paths']==case['image_paths'],'Completed source identities differ')
        require(case['source_indices'][0]==previous[0] and case['timestamps'][0]==previous[1],'Anchor is not previous dispatched target')
        require(previous[2] is None or case['image_paths'][0]==previous[2],'Overlap RGB path differs')
        require(0<=case['source_indices'][0]<case['source_indices'][1]<798 and case['timestamps'][0]<case['timestamps'][1],'Source order/range differs')
        require(row['sequence']==case['source_indices'][1] and row['timestamp']==case['timestamps'][1] and row['anchor_timestamp']==previous[1],'Fresh target/seed inclusion differs')
        require(all(float(Path(path).stem)==stamp for path,stamp in zip(case['image_paths'],case['timestamps'])),'TUM filenames/timestamps differ')
        clocks=[row[k] for k in ('captured_monotonic','dispatch_monotonic','worker_start_monotonic','worker_end_monotonic','result_received_monotonic','commit_start_monotonic','published_monotonic')]
        require(all(math.isfinite(v) for v in clocks) and clocks==sorted(clocks),'Arrival/service/commit clock order differs')
        require(close(dispatch['dispatch_monotonic'],row['dispatch_monotonic']),'Dispatch clock identity differs')
        require(all(math.isfinite(row[k]) and row[k]>=0 for k in PHASES),'Negative/nonfinite phase')
        latency=clocks[-1]-clocks[0]; require(close(latency,row['input_to_pose_seconds']),'Latency does not include complete arrival/publication')
        error=abs(sum(row[k] for k in PHASES)-latency);require(error<2e-8,'Phase durations do not sum to full latency')
        sum_error=max(sum_error,error);latencies.append(latency)
        if r['seed_sequence']==0:
            require(close(clocks[0],s['start_monotonic']+row['timestamp']-r['seed_timestamp']),'Original scheduled timestamp clock differs')
        require(np.asarray(row['pose']).shape==(4,4) and np.isfinite(row['pose']).all(),'Nonfinite published world pose')
        timeline.extend([(row['dispatch_monotonic'],1,lane),(row['published_monotonic'],-1,lane)])
        previous=(row['sequence'],row['timestamp'],case['image_paths'][1])
    active=set(); peak=0
    for t,change,lane in sorted(timeline,key=lambda x:(x[0],x[1])):
        if change==1:require(lane not in active,'Multiple uncommitted jobs on one lane');active.add(lane)
        else:require(lane in active,'Unmatched commit');active.remove(lane)
        peak=max(peak,len(active));require(peak<=lanes,'Inflight bound exceeded')
    require(not active,'EOF left uncommitted jobs')
    duration=u[-1]['published_monotonic']-s['start_monotonic']; hz=n/duration;p95=float(np.percentile(latencies,95))
    require(close(duration,r['fresh_wall_seconds_including_EOF_drain']) and close(hz,r['fresh_pose_hz']) and close(p95,r['input_to_pose_p95_seconds']),'Hz/P95/full EOF denominator differs')
    require(u[-1]['published_monotonic']>=s['last_capture_monotonic'],'EOF tail excluded')
    gaps=np.diff([row['published_monotonic'] for row in u])
    initial_delay=u[0]['published_monotonic']-s['start_monotonic']
    gap_p95=float(np.percentile(gaps,95)) if len(gaps) else None
    gap_max=float(np.max(gaps)) if len(gaps) else None
    cadence_ok=bool(initial_delay<=1 and (not len(gaps) or np.max(gaps)<=1))
    m=json.loads((replay_dir/'map.json').read_text());require(len(m['chunks'])==m['revision']==n and len(m['nodes'])==2*n,'Map chunk/node counts differ')
    require(all(np.asarray(H).shape==(4,4) and np.isfinite(H).all() for H in m['nodes'].values()),'Nonfinite map homography')
    point_count=0;array_count=0
    for chunk in m['chunks']:
        path=(replay_dir/chunk['path']).resolve();require(path.is_relative_to(replay_dir.resolve()),'Map path escaped saved directory')
        require(path.stat().st_size==chunk['bytes'] and sha(path)==chunk['sha256'],'Map NPZ artifact changed')
        with np.load(path,allow_pickle=False) as archive:
            require(set(archive.files)=={'points_0','points_1','colors_0','colors_1'},'Original two-frame chunk fields differ')
            for j,count in enumerate(chunk['point_counts']):
                points,colors=archive[f'points_{j}'],archive[f'colors_{j}']
                require(points.dtype==np.float32 and colors.dtype==np.uint8 and points.shape==colors.shape==(count,3) and np.isfinite(points).all(),'Chunk shapes/dtypes/finite/count differ')
                array_count+=2;point_count+=count
    poses=np.asarray([[float(x) for x in line.split()] for line in (replay_dir/'poses.tum').read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')])
    require(poses.shape==(n+1,8) and np.isfinite(poses).all(),'Pose trajectory count/shape/finite differs')
    require(np.all(np.diff(poses[:,0])>0) and np.allclose(poses[:,0],[r['seed_timestamp']]+[row['timestamp'] for row in u],rtol=0,atol=6e-7),'Trajectory timestamp order differs')
    f=r['shared_context_registry_final'];require(f['actual_contexts_created']==f['actual_contexts_destroyed']==12*lanes and not f['live_contexts'] and not f['cleanup_errors'] and f['all_registered_contexts_final_zero'] is True and not f['retained_CPU_model_bytes'],'Physical registry release differs')
    require(f['share_init_requests']==12*(lanes-1),'Expected shared init count differs')
    init=[e for e in f['events'] if e['operation']=='initialize'];destroy=[e for e in f['events'] if e['operation']=='physical_destroy']
    require(len(init)==len(destroy)==12*lanes and all(e['code']==0 and e['context_created'] for e in init) and all(e['code']==0 for e in destroy),'Actual SDK initialize/destroy results differ')
    require(sum(e['actual_flags']==32 and e['model_input']=='owned_CPU_buffer' for e in init)==12*(lanes-1),'Child shared CPU-buffer boundary differs')
    workers={};allruns=0
    for key,w in r['workers'].items():
        lane=int(key);require(w['status']=='closed' and not w['cleanup_errors'] and w['actual_core_mask']==r['worker_core_masks'][lane],'Worker final ownership/status differs')
        count=sum(row['worker']==lane for row in u);metrics=w['closed_metrics']
        require(metrics['cache']['closed'] and metrics['cache']['requests']==count and metrics['immutable']['final_full_validated'] and metrics['immutable']['restored_overrides']==3,'Cache/session final verification differs')
        for name,ctxs in [('aggregator',4),('DINO',6),('camera',1),('depth',1)]:
            v=metrics[name];require(v['contexts_created']==v['contexts_destroyed']==ctxs and v['closed'] and not v['live_contexts'] and not v['cleanup_errors'],'Backend cleanup differs: '+name)
        a,d,cam,depth=[metrics[k] for k in ('aggregator','DINO','camera','depth')]
        require(a['sdk_run_attempts']==a['sdk_runs_succeeded']==4*count and a['outputs_acquired']==a['outputs_released']==9*count,'Aggregator call/output count differs')
        require(cam['completed_head_calls']==count and cam['completed_refinements']==cam['npu_run_api_call_count']==4*count,'Camera four refinements differ')
        require(depth['sdk_run_attempts']==depth['sdk_runs_succeeded']==2*count and d['sdk_run_attempts']==d['sdk_runs_succeeded']==6*metrics['cache']['computed_frames'],'DINO/depth complete original calls differ')
        runs=a['sdk_runs_succeeded']+d['sdk_runs_succeeded']+cam['npu_run_api_call_count']+depth['sdk_runs_succeeded'];allruns+=runs
        require(len(w['pageout_stages'])==4,'Four pageout stages missing')
        for stage in w['pageout_stages']:
            path=replay_dir/stage['file']; page=json.loads(path.read_text());require(sha(path)==stage['sha256'] and page['tensor_storage_VA_version_unchanged'] and page['checkpoint_VMA_layout_unchanged'],'Pageout saved identity differs')
        workers[key]=dict(complete_original_pairs=count,contexts=12,SDK_runs=runs,cache_hits=metrics['cache']['hit_frames'],cache_computed_frames=metrics['cache']['computed_frames'],immutable_final_validated=True,pageout_receipts_verified=4)
    out.update(status='passed_saved_success_audit',experiment_completed=True,original_frames=798,
        source_accounting={k:s[k] for k in ('generated','delivered','overwritten','skipped_before_decode')},
        seed_sequence=r['seed_sequence'],fresh_count=n,unique_fresh_targets=True,ordered_overlap_verified=True,
        maximum_uncommitted=peak,per_lane_outstanding_one_verified=True,EOF_tail_drained=True,
        fresh_wall_seconds_including_EOF_drain=duration,fresh_pose_hz=hz,input_to_pose_p95_seconds=p95,
        maximum_phase_sum_error_seconds=sum_error,startup_seconds=r['startup_seconds'],peak_rss_mib=r['peak_rss_mib'],
        publication_interval_count=len(gaps),publication_interval_p95_seconds=gap_p95,
        maximum_publication_gap_seconds=gap_max,
        publication_intervals_greater_than_one_second=int(np.count_nonzero(gaps>1)),
        initial_fresh_publication_delay_seconds=initial_delay,
        mean_fresh_rate_at_least_one_hz=hz>=1,strict_at_least_one_update_each_second=cadence_ok,
        cadence_scope='First fresh publication after replay start and every subsequent consecutive publication gap; mean Hz alone is not cadence acceptance.',
        memory_scope=r['memory_scope'],map_chunks=n,graph_nodes=2*n,map_arrays_verified=array_count,
        map_points=point_count,poses=n+1,map_sha256=sha(replay_dir/'map.json'),poses_sha256=sha(replay_dir/'poses.tum'),
        actual_contexts_created_destroyed=12*lanes,share_init_requests=f['share_init_requests'],
        SDK_runs=allruns,SDK_context_cleanup_verified=True,thread_join_verified=True,workers=workers,
        limitations=['Software timestamp replay; no physical camera latency validation.','Short run; no30min or physical8GB acceptance.','Original FP32 quality acceptance is not established by this runtime audit.'])
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay-dir',type=Path,required=True)
    p.add_argument('--control-dir',type=Path,required=True);p.add_argument('--source-dir',type=Path,default=Path(__file__).parent)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.replay_dir,a.control_dir,a.source_dir)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result['status'],a.output.name,sha(a.output))
    return 0

if __name__=='__main__':raise SystemExit(main())
