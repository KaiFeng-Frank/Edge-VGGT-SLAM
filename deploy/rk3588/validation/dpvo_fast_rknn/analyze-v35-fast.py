#!/usr/bin/env python3
"""Saved JSON + integer topology only. Never imports Torch, ORT, SDK or BA."""
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

PUBLIC = Path('/dev/shm/vggt-v15-public-snapshot')
RUN = Path('/dev/shm/vggt-v34-output/vggt-v34-dpvo-fast-r1/run.json')
PLAN = PUBLIC/'deploy/rk3588/validation/dpvo_source_frame/source_frame_plan.py'
OUT = Path(__file__).with_name('fast-critical-path-review.json')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, reason):
    if not bool(ok):
        raise ValueError(reason)


def stats(values):
    x = np.asarray(values, np.float64)
    require(x.size and np.isfinite(x).all(), 'Finite nonempty stats')
    return dict(count=int(x.size), sum_seconds=float(x.sum()), mean_seconds=float(x.mean()),
                p50_seconds=float(np.median(x)), p95_seconds=float(np.quantile(x,.95)),
                max_seconds=float(x.max()))


def metadata(x):
    return dict(shape=list(x.shape), dtype=str(x.dtype), bytes=x.nbytes,
                tensor_sha256=hashlib.sha256(x.tobytes(order='C')).hexdigest())


def main():
    require(digest(RUN)=='a0f18920b156f9f819630b591fa71af25f5e869a9a160ccd5c7d4b471218162e','Frozen FAST run changed')
    require(digest(PLAN)=='04e240c2c35bd1359dbe5649fb038490a0a7ff99654d8d112ad7c83acc87ac2e','Frozen plan changed')
    spec=importlib.util.spec_from_file_location('_v35_numpy_source_plan',PLAN)
    plan=importlib.util.module_from_spec(spec);sys.modules[spec.name]=plan;spec.loader.exec_module(plan)
    r=json.loads(RUN.read_text());M=r['config']['patches_per_frame'];life=r['config']['patch_lifetime']
    require(M==48 and r['frames_completed']==192 and not r['cleanup_errors'],'Completed original FAST48 required')
    empty=lambda:np.empty(0,np.int64)
    ii,jj,kk=empty(),empty(),empty();ai,aj,ak=empty(),empty(),empty();frames=[]
    u=r['update_calls'];c=r['correlation']['calls'];gens=r['correlation']['generations']
    position=0;seen_generations=set();rows=[];degree_occurrences=collections.Counter();last_update_plans=[]
    for f in r['frames']:
        result=f['result'];begin=position;prep=0.
        def consume(edges):
            nonlocal position,prep
            require(u[position]['index']==c[position]['index']==position,'Call order differs')
            require(u[position]['edges']==c[position]['edges']==edges,'Actual call edge count differs')
            g=c[position]['generation']
            if g not in seen_generations:
                prep+=gens[g]['preparation']['total_seconds'];seen_generations.add(g)
            position+=1
        if result['motion_probe'] is not None:
            consume(M)
        degrees=None
        if result['admitted']:
            frames.append(result['observation_id']);n=len(frames)
            old=np.arange(M*max(n-life,0),M*(n-1),dtype=np.int64)
            new=np.arange(M*(n-1),M*n,dtype=np.int64);targets=np.arange(max(n-life,0),n,dtype=np.int64)
            added=np.concatenate((old,np.repeat(new,len(targets))))
            ii=np.concatenate((ii,added//M));kk=np.concatenate((kk,added))
            jj=np.concatenate((jj,np.full(len(old),n-1,np.int64),np.tile(targets,len(new))))
            if result['updates']:
                plans=plan.build_source_frame_plan(ii,jj,kk,patches_per_frame=M)
                degrees=collections.Counter(p.tensor_shape_without_batch_channels[0] for p in plans)
                for d,count in degrees.items():degree_occurrences[d]+=count*result['updates']
                last_update_plans=[p.to_metadata() for p in plans]
                for _ in range(result['updates']):consume(len(kk))
            keyframe=result.get('keyframe')
            if keyframe is not None:
                removed=keyframe['removed_observation_id']
                if removed is not None:
                    k=n-r['config']['keyframe_index'];require(frames[k]==removed,'Recorded removed identity differs')
                    mask=(ii==k)|(jj==k);ii,jj,kk=ii[~mask],jj[~mask],kk[~mask]
                    later=ii>k;kk=kk-later.astype(np.int64)*M;ii=ii-later.astype(np.int64);jj=jj-(jj>k).astype(np.int64)
                    frames.pop(k)
                mask=(kk//M)<len(frames)-r['config']['removal_window']
                ai=np.concatenate((ai,ii[mask]));aj=np.concatenate((aj,jj[mask]));ak=np.concatenate((ak,kk[mask]))
                ii,jj,kk=ii[~mask],jj[~mask],kk[~mask]
            require(len(kk)==result['active_edges'] and len(frames)==result['retained_frames'],'Recorded graph size differs')
        dt=sum(x['seconds'] for x in u[begin:position]);ct=sum(x['total_seconds'] for x in c[begin:position])
        nq=sum(x['native_query']['total_seconds'] for x in c[begin:position])
        require(abs(f['processing_seconds']-sum(f[k] for k in ('decode_undistort_crop_seconds','encode_seconds','stream_seconds')))<1e-8,'Outer interval nesting differs')
        remainder=f['stream_seconds']-dt-ct;require(remainder>=-1e-7,'Nested times exceed stream interval')
        rows.append(dict(index=f['index'],fresh=result['fresh_pose_valid'],updates=result['updates'],
            decode=f['decode_undistort_crop_seconds'],encoder_outer=f['encode_seconds'],stream=f['stream_seconds'],
            Update=dt,correlation_outer=ct,cache_preparation_subset=prep,native_query_subset=nq,
            stream_remainder=remainder,processing=f['processing_seconds'],
            before_Update_source_blocks=None if degrees is None else sum(degrees.values()),
            degree_histogram=None if degrees is None else dict(sorted(degrees.items()))))
    online_position=position
    require(position==203,'190 original Updates +13 motion probes expected')
    artifact=r['snapshot_before_termination']['artifact'];matches={}
    for name,a in dict(ii=ii,jj=jj,kk=kk,ii_inactive=ai,jj_inactive=aj,kk_inactive=ak).items():
        matches[name]=metadata(a)==artifact['arrays'][name]
    require(all(matches.values()),'Integer reconstructed topology does not match saved rawSHA')
    require(frames==r['snapshot_before_termination']['metadata']['retained_observation_ids'],'Retained observation identities differ')
    final_plans=[p.to_metadata() for p in plan.build_source_frame_plan(ii,jj,kk,patches_per_frame=M)]
    for _ in range(r['termination']['offline_updates']):consume(len(kk))
    require(position==len(u)==len(c) and len(seen_generations)==len(gens),'All actual calls accounted')
    populations={}
    for name,selected in [('all_source',rows),('fresh', [x for x in rows if x['fresh']]),
                          ('steady_fresh_excluding_initialization',[x for x in rows if x['fresh'] and x['updates']==1])]:
        populations[name]={k:stats([x[k] for x in selected]) for k in ('decode','encoder_outer','stream','Update','correlation_outer','cache_preparation_subset','native_query_subset','stream_remainder','processing')}
    fresh=[x for x in rows if x['fresh']];mu=populations['fresh']['processing']['mean_seconds'];p95=populations['fresh']['processing']['p95_seconds']
    counterfactual={}
    for label,remove in [('Update_zero',('Update',)),('encoder_zero',('encoder_outer',)),
                         ('encoder_and_Update_zero',('encoder_outer','Update')),
                         ('encoder_Update_and_corr_zero',('encoder_outer','Update','correlation_outer'))]:
        counterfactual[label]=stats([x['processing']-sum(x[k] for k in remove) for x in fresh])
    report=dict(schema='v35_saved_FAST_critical_path_and_integer_topology_review',status='saved_numeric_source_analysis_not_realtime_acceptance',
        input_run_sha256=digest(RUN),analysis_sha256=digest(__file__),source_plan_sha256=digest(PLAN),
        frames=192,fresh=179,online_neural_calls_including_motion=online_position,EOF_neural_calls=12,
        timing_populations=populations,
        independent_all_source_encoder_inner_totals=r['encoder_metrics'],
        timing_contracts={
          'outer':'processing=decode_undistort_crop+encoder_outer+stream; replay also includes printing/progress outside per-frame intervals',
          'encoder':'encoder_outer includes normalization/ORT/patch sampling, lock/report bookkeeping; first3 additionally saved-reference checks. Fresh per-frame ORT split not recorded, only all192 totals and final-frame inner timers.',
          'Update':'Timer includes finite input scan, Torch views, original Update, owned NumPy output copies and finite/domain guard. Weight min/max reporting occurs just after recorded interval.',
          'corr':'correlation_outer includes closing/rebuilding cache, feature concatenation, pool/layout, ID remap, native query and outer finite output scan. cache_preparation and native_query are subsets, never additive to outer.',
          'stream_remainder':'Includes BA, reproject, context gather, state fork and graph copies, flow/removal and guards. BA is not independently timed; remainder is not BA time.',
          'EOF':'12 offline updates and reconstruction excluded from fresh timing/populations.'},
        targets={
          'serial_service_budget_20Hz_seconds':.05,'pose_latency_P95_budget_seconds':.1,
          'fresh_mean_speed_factor_to_50ms':mu/.05,'fresh_mean_required_reduction_fraction':1-.05/mu,
          'fresh_P95_speed_factor_to_100ms':p95/.1,'fresh_P95_required_reduction_fraction':1-.1/p95,
          'whole_replay_fresh_Hz':179/r['replay_elapsed_seconds'],
          'whole_replay_speed_factor_for20fresh_per_second':20*r['replay_elapsed_seconds']/179},
        fixed_trace_counterfactuals=counterfactual,
        topology=dict(method='Integer append/remove/remap only from all recorded admissions and removed IDs, exact six-array rawSHA/shape/dtype/bytes match; no NPZ read or geometry execution.',
          before_and_final_topology_equal=all(artifact['arrays'][k]==r['final_snapshot']['artifact']['arrays'][k] for k in matches),
          six_saved_array_metadata_matches=matches,retained_observation_ids=frames,final_edges=len(kk),
          last_pre_Update_plans=last_update_plans,final_plans=final_plans,
          degree_block_occurrences_online_updates=dict(sorted(degree_occurrences.items())),
          final_snapshot_NPZ_available_on_host=False,
          recurrent_state_and_corr_values_verified_here=False),
        representative_fresh_rows=[x for x in rows if x['index'] in (13,14,63,127,178,191)],
        limitations=[
          'Counterfactuals set recorded stage cost to zero with all other costs fixed; not hardware theoretical bounds or measured optimized pipeline results.',
          'Historical CPU run is unpaced and thermally uncontrolled. Software file admission is not physical capture or queued sensor latency.',
          'Topology reconstruction does not reconstruct net/corr/patch numeric values; strict M48 fullgraph recurrent CPU/ORT/RKNN comparisons still require board snapshots.',
          'Published source-frame RKNN D10/M96 rejects original FP32 gate. Its partial-block timing cannot certify the M48 fullgraph or become a fresh pose.'],
        model_forward_calls=0,ORT_calls=0,SDK_calls=0,BA_calls=0,images_read=0,snapshot_NPZ_reads=0,board_jobs=0)
    with OUT.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':report['status'],'output_sha256':digest(OUT),'fresh_mean_seconds':mu,
        'fresh_P95_seconds':p95,'last_pre_Update_degree_histogram':rows[-1]['degree_histogram'],
        'final_degree_histogram':dict(collections.Counter(p['degree'] for p in final_plans))}))


if __name__=='__main__':main()
