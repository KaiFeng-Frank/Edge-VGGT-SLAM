#!/usr/bin/env python3
"""Saved JSON/NumPy/TUM-only CPU-vs-native precision and map disagreement audit."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
METRICS_SHA='f7841599c6f0e1d58503444c4b90b369acd581fc697469f8fdbf01ecab55ff65'
CPU_SHA='23e29fa0679e80755153b055702129309f39e495cf3349a9d080fa3980a5190c'
NATIVE_SHA='78cb60f56df0b98879a987b34db1172c62cd0d51b5364cb1c3c81fd6f3ac1fe9'
EXECUTOR_SHA='e153a5f71ef0b10119207b2e43589b628a2e85ab1abeaeb72d2df2a7dd6fb638'
CONSUMER_SHA='a0d37e54b1b1655bf4443f5fa80c07043a1ada9c12dbf1d0cb851b41dc7af271'
GT_SHA='aac0319a6ef4e1cdf61e779d2152b95aa7e9f7b1749d6d18717b43ddabffede2'
RGB_SHA='532884d60bef0c9a05e14bf311998883baf59a2fa47009d1021f2c7937bd69a3'
CHECKPOINT={'bytes':5026874952,'sha256':'d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'}
INDICES=[32+24*i for i in range(17)]
SHAPES={'pose_enc':[2,9],'extrinsic':[2,3,4],'intrinsic':[2,3,3],'depth':[2,392,518,1],'depth_conf':[2,392,518]}
COUNT_KEYS=('contexts_created','contexts_destroyed','runs_completed','output_buffers_acquired','output_buffers_released')


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4<<20),b''):h.update(chunk)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cpu-result',type=Path,required=True);ap.add_argument('--cpu-sha256',required=True)
    ap.add_argument('--native-results',type=Path,nargs=4,required=True);ap.add_argument('--native-sha256',nargs=4,required=True)
    ap.add_argument('--cpu-sl4-result',type=Path,required=True);ap.add_argument('--cpu-sl4-sha256',required=True)
    ap.add_argument('--native-sl4-result',type=Path,required=True);ap.add_argument('--native-sl4-sha256',required=True)
    ap.add_argument('--groundtruth',type=Path,default=ROOT/'data/groundtruth.txt')
    ap.add_argument('--source-images',type=Path,default=ROOT/'data/rgb')
    ap.add_argument('--trajectory-helper',type=Path,default=ROOT/'evaluate_original_trajectory.py')
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();oldenv=dict(os.environ);pins={};array_count=0
    report=dict(schema='v111_independent_saved_native518_trajectory_quality_audit',status='failed',auditor_sha256=sha(__file__),
        saved_data_only=True,NN_ORT_SDK_GPU_GTSAM_execution=False,original_model_only=True,
        quality_realtime_map_accuracy_30min_or_8GB_accepted=False,strict_rtol=1e-4,strict_atol=1e-5)
    try:
        os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
        import numpy as np
        def pin(path,digest):
            path=Path(path).resolve(strict=True);require(sha(path)==digest,'File SHA differs: '+str(path));pins[path]=digest;return path
        def receipt(path,digest):
            path=pin(path,digest);return json.loads(path.read_text()),path.parent
        def artifact(root,row):
            rel=Path(row['file']);require(not rel.is_absolute() and '..' not in rel.parts,'Relative artifact required')
            path=pin(root/rel,row['sha256']);require(path.is_relative_to(root) and path.stat().st_size==row['bytes'],'Artifact path/size differs');return path
        def array(root,row,dtype,shape):
            nonlocal array_count
            path=artifact(root,row);v=np.load(path,allow_pickle=False)
            require(v.dtype==dtype and list(v.shape)==row['shape']==shape and v.flags.c_contiguous and np.isfinite(v).all()
                and v.nbytes==row['tensor_bytes'] and hashlib.sha256(v.tobytes()).hexdigest()==row['raw_sha256'] and row['finite'], 'Array metadata/raw/finite differs')
            array_count+=1;return v
        def source(row):return {k:row[k] for k in ('index','timestamp','bytes','sha256')}
        pin(__file__,report['auditor_sha256'])
        cpu,cr=receipt(a.cpu_result,a.cpu_sha256)
        require(cpu['schema']=='v111_original_native518_host_CPU_FP32_predictions'
            and cpu['status']=='completed_original_native518_host_FP32_prediction_capture' and cpu['script_sha256']==CPU_SHA
            and cpu['execution_device']=='host_cpu' and not cpu['host_GPU_NPU_SDK_ORt_execution'] and not cpu['prune_unused_heads']
            and not cpu['forward_hooks_installed'] and not cpu['patch_or_feature_cache_installed'] and not cpu['original_model_methods_overridden']
            and cpu['all_original_heads_retained'] and cpu['full_original_forward_count']==cpu['full_model_forward_calls_completed']==16
            and cpu['model_state_unchanged'] and cpu['depth_frames_chunk_size']==1,'Complete unchanged original host FP32 reference required')
        logical=[source(v) for v in cpu['source_images']];stamps=[v['timestamp'] for v in logical]
        require([v['index'] for v in logical]==INDICES and len(set(stamps))==17 and len({v['sha256'] for v in logical})==17,'Exact17 original observations required')
        for row in cpu['source_images']:
            path=pin(a.source_images/Path(row['path']).name,row['sha256']);require(path.stat().st_size==row['bytes'],'Original source PNG size differs')
        native={};native_totals=dict.fromkeys(COUNT_KEYS,0)
        for path,digest in zip(a.native_results,a.native_sha256):
            r,rd=receipt(path,digest);z=r['executor_resources']
            require(r['schema']=='v111_original_native518_trajectory_prediction_batch'
                and r['status']=='completed_actual_native518_four_pair_batch_not_quality_or_realtime_accepted'
                and r['script_sha256']==NATIVE_SHA and r['executor_sha256']==EXECUTOR_SHA and r['core_mask']==7
                and r['pair_count']==4 and r['start_pair'] in (0,4,8,12) and [source(v) for v in r['source_images']]==logical
                and z['closed'] and z['resources_balanced'] and not z['live_contexts'] and not z['live_output_buffers'] and not z['cleanup_errors']
                and z['SDK_totals']==dict(zip(COUNT_KEYS,(40,40,60,80,80))),'Complete native batch identity/cleanup differs')
            for key in COUNT_KEYS:native_totals[key]+=z['SDK_totals'][key]
            for p in r['pairs']:
                require(p['index'] not in native,'Duplicate native pair');native[p['index']]=(p,rd)
            require(r['checkpoint']==CHECKPOINT and r['source_indices']==INDICES and r['max_loops']==0 and r['rgb_index_sha256']==RGB_SHA
                and r['source_and_inputs_unchanged'] and not r['cleanup_errors'] and r['environment_restored'] and r['affinity_restored'] and r['sys_path_restored'],'Original native provenance/restoration differs')
        require(set(native)==set(range(16)) and len(cpu['pairs'])==16 and cpu['checkpoint']==CHECKPOINT and cpu['source_indices']==INDICES
            and cpu['source_and_inputs_unchanged'] and not cpu['cleanup_errors'] and cpu['rgb_index_sha256']==RGB_SHA,'Incomplete matching prediction routes')
        def diagnostics(actual,expected):
            x,y=actual.astype(np.float64),expected.astype(np.float64);delta=x-y
            sq=float(np.sum(delta*delta));ref=float(np.sum(y*y));absolute=math.sqrt(sq);norm=math.sqrt(ref)
            return dict(relative_l2=absolute/norm if norm else (0. if not absolute else None),reference_norm_zero=norm==0,
                absolute_l2=absolute,rmse=math.sqrt(sq/actual.size),mean_abs=float(np.mean(np.abs(delta))),max_abs=float(np.max(np.abs(delta))),
                outside_strict_tolerance=int(np.count_nonzero(np.abs(delta)>1e-5+1e-4*np.abs(y))),
                strict_pass=bool(np.allclose(actual,expected,rtol=1e-4,atol=1e-5)),raw_byte_equal=actual.tobytes()==expected.tobytes()),sq,ref
        per_pair=[];sums={k:dict(squared_error=0.,reference_squared=0.,elements=0,strict_passed_arrays=0,max_abs=0.) for k in SHAPES}
        for i in range(16):
            c=cpu['pairs'][i];n,nr=native[i]
            require(c['index']==n['index']==i and c['source_indices']==n['source_indices']==INDICES[i:i+2]
                and c['timestamps']==n['timestamps']==stamps[i:i+2] and [source(v) for v in c['source_images']]==[source(v) for v in n['source_images']]==logical[i:i+2]
                and c['prepared_image_shape']==n['prepared_image_shape']==[2,3,392,518]
                and c['preprocessed_images_raw_sha256']==n['prepared_pair_raw_sha256']
                and c['prepared_frame_raw_sha256']==n['prepared_frame_raw_sha256'] and c['camera_refinement_count']==4,'Matched original pair/prepared bytes differ')
            if i:require(c['prepared_frame_raw_sha256'][0]==cpu['pairs'][i-1]['prepared_frame_raw_sha256'][1],'Cross-batch overlap differs')
            d={}
            for key,shape in SHAPES.items():
                x=array(nr,n['outputs'][key],np.float32,shape);y=array(cr,c['outputs'][key],np.float32,shape)
                d[key],sq,ref=diagnostics(x,y);z=sums[key];z['squared_error']+=sq;z['reference_squared']+=ref;z['elements']+=x.size
                z['strict_passed_arrays']+=int(d[key]['strict_pass']);z['max_abs']=max(z['max_abs'],d[key]['max_abs']);del x,y
            call=n['executor_call'];local=i%4;owners=[call['resources'][f'dino{j}'] for j in range(4)]+call['resources']['aggregator']['per_group_closed_contexts']+call['resources']['heads']['contexts']
            totals={k:sum(v[k] for v in owners) for k in COUNT_KEYS}
            require(totals==dict(zip(COUNT_KEYS,(10,10,14 if local else 18,19 if local else 23,19 if local else 23)))==call['SDK_totals']
                and all(v['closed'] and not v['live_contexts'] and not v['live_output_buffers'] and not v['cleanup_errors'] and v['core_mask']==7 for v in owners)
                and call['input_raw_sha256']==n['prepared_pair_raw_sha256'] and call['input_frame_raw_sha256']==n['prepared_frame_raw_sha256'],'Native per-pair input/resource ledger differs')
            per_pair.append(dict(index=i,source_indices=c['source_indices'],timestamps=c['timestamps'],five_outputs_native_vs_original_FP32=d,
                host_original_full_model_and_codec_seconds=c['original_full_model_and_pose_codec_seconds'],native_prediction_seconds=n['prediction_seconds']))
        for z in sums.values():
            z['relative_l2']=math.sqrt(z['squared_error']/z['reference_squared']) if z['reference_squared'] else (0. if not z['squared_error'] else None)
            z['rmse']=math.sqrt(z['squared_error']/z['elements']);z['strict_failed_arrays']=16-z['strict_passed_arrays']
            del z['squared_error'],z['reference_squared']
        helper=pin(a.trajectory_helper,METRICS_SHA);spec=importlib.util.spec_from_file_location('_v111_saved_TUM_metrics',helper)
        metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
        gt=metrics.read_tum(pin(a.groundtruth,GT_SHA));sl4={};trajectory={};map_arrays={}
        manifest_pins={'original_fp32':{a.cpu_sha256},'native':set(a.native_sha256)}
        for name,path,digest in (('original_fp32',a.cpu_sl4_result,a.cpu_sl4_sha256),('native',a.native_sl4_result,a.native_sl4_sha256)):
            r,rd=receipt(path,digest)
            require(r['schema']=='v111_original_native518_saved_prediction_SL4_replay' and r['status']=='completed_saved_original_native518_16_pair_SL4_replay_not_quality_accepted'
                and r['script_sha256']==CONSUMER_SHA and r['prediction_kind']==name and [source(v) for v in r['source_images']]==logical
                and {v['sha256'] for v in r['prediction_manifests']}==manifest_pins[name] and r['checkpoint']==CHECKPOINT
                and [r[k] for k in ('pair_count','submap_count','node_count','factor_count','loop_closure_count','distinct_source_observation_count','frame_occurrence_count','saved_executor_calls')]==[16,16,32,32,0,17,32,16]
                and r['saved_executor_closed'] and not r['cleanup_errors'] and r['source_and_inputs_unchanged'] and r['environment_restored'] and r['affinity_restored'] and r['sys_path_restored']
                and not r['NN_SDK_GPU_executed'] and not r['model_loaded'] and r['samples_are_not_confidence_filtered'],'Original complete saved SL4 replay required')
            poses=array(rd,r['artifacts']['poses_tum_rows'],np.float64,[17,8]);text=metrics.read_tum(artifact(rd,r['poses_tum']))
            require(np.array_equal(poses,text) and np.array_equal(poses[:,0],stamps),'Original17 pose timestamps/rows differ')
            trajectory[name]=metrics.evaluate_tum(poses,gt,max_difference=0.02,offset=0.,delta_frames=1)
            require(trajectory[name]['association']['matched_count']==17,'All17 fixed GT associations required')
            h=array(rd,r['artifacts']['homographies'],np.float64,[32,4,4]);require(np.isfinite(np.linalg.det(h)).all() and np.all(np.linalg.det(h)!=0),'Saved SL4 homography not invertible')
            data={k:array(rd,r['artifacts'][k],dtype,shape) for k,dtype,shape in (
                ('sample_flat_indices',np.int64,[4062]),('sampled_world_points',np.float64,[32,4062,3]),
                ('sampled_confidence_masks',np.bool_,[32,4062]),('full_confidence_masks',np.bool_,[32,392,518]),
                ('frame_occurrence_source_indices',np.int64,[32]),('frame_occurrence_node_ids',np.int64,[32]),('frame_occurrence_timestamps',np.float64,[32]))}
            require(np.array_equal(data['sample_flat_indices'],np.arange(0,392*518,50,dtype=np.int64))
                and np.array_equal(data['frame_occurrence_source_indices'],np.asarray([v for i in range(16) for v in INDICES[i:i+2]]))
                and np.array_equal(data['frame_occurrence_timestamps'],np.asarray([v for i in range(16) for v in stamps[i:i+2]]))
                and np.array_equal(data['sampled_confidence_masks'],data['full_confidence_masks'].reshape(32,-1)[:,data['sample_flat_indices']]),'Fixed same-pixel original map samples/masks differ')
            require(all(p['saved_output_raw_sha256']=={k:(cpu['pairs'][i] if name=='original_fp32' else native[i][0])['outputs'][k]['raw_sha256'] for k in SHAPES}
                and p['prepared_pair_raw_sha256']==cpu['pairs'][i]['preprocessed_images_raw_sha256'] for i,p in enumerate(r['pairs'])),'SL4 consumed another prediction/input')
            sl4[name]=r;map_arrays[name]=data
        require(trajectory['original_fp32']['association']['matches']==trajectory['native']['association']['matches'],'GT match correspondence differs')
        metric_changes={};screens={'10_percent':{},'20_percent':{}}
        for metric in ('ate_translation','rpe_translation','rpe_rotation'):
            metric_changes[metric]={}
            for statistic in ('rmse','maximum'):
                b=trajectory['original_fp32'][metric][statistic];c=trajectory['native'][metric][statistic]
                metric_changes[metric][statistic]=dict(original_FP32_absolute_error=b,native_absolute_error=c,units=trajectory['native'][metric]['units'],
                    absolute_change=c-b,relative_change_percent=100*(c/b-1) if b else None,original_baseline_zero=b==0,
                    zero_baseline_policy='Percentage undefined at exactly zero; report absolute difference, no epsilon flooring.')
                for margin in (10,20):screens[f'{margin}_percent'][metric+'_'+statistic]=bool(c<=b*(1+margin/100)) if b else c==0
        for name,data in map_arrays.items():
            t=trajectory[name]['alignment'];xyz=data['sampled_world_points'];data['aligned_xyz']=t['scale']*(xyz@np.asarray(t['rotation']).T)+np.asarray(t['translation'])
        left,right=map_arrays['original_fp32'],map_arrays['native']
        occurrence_keys=('occurrence','pair_index','frame_in_pair','submap_id','node_id','source_index','timestamp','full_point_count','sampled_point_count')
        require(np.array_equal(left['frame_occurrence_node_ids'],right['frame_occurrence_node_ids'])
            and [{k:v[k] for k in occurrence_keys} for v in sl4['original_fp32']['per_frame_geometry']]
                ==[{k:v[k] for k in occurrence_keys} for v in sl4['native']['per_frame_geometry']], 'Frame/pixel correspondence metadata differs')
        distances=np.linalg.norm(right['aligned_xyz']-left['aligned_xyz'],axis=-1)
        def distance_stats(v):
            v=np.asarray(v);return dict(count=v.size,rmse=float(np.sqrt(np.mean(v*v))),mean=float(v.mean()),median=float(np.median(v)),p95=float(np.percentile(v,95)),maximum=float(v.max())) if v.size else dict(count=0)
        def mask_stats(x,y):
            inter=int(np.count_nonzero(x&y));union=int(np.count_nonzero(x|y));return dict(original_count=int(x.sum()),native_count=int(y.sum()),intersection=inter,union=union,IoU=inter/union if union else 1.,different_pixels=int(np.count_nonzero(x!=y)),empty_union=union==0)
        frames=[]
        for i in range(32):frames.append(dict(occurrence=i,source_index=int(left['frame_occurrence_source_indices'][i]),
            pair_index=i//2,frame_in_pair=i%2,same_pixel_reference_disagreement=distance_stats(distances[i]),
            full_confidence_masks=mask_stats(left['full_confidence_masks'][i],right['full_confidence_masks'][i])))
        both=left['sampled_confidence_masks']&right['sampled_confidence_masks']
        controls={}
        for start in (0,4,8,12):
            path=ROOT/f'control-batch{start}-r1/control.json';path=pin(path,sha(path));control=json.loads(path.read_text())
            obs=control['observations'];temperatures=[v for o in obs for v in o['temperatures_c'].values()]
            require(control['child_returncode']==0 and not control['restore_errors'] and control['frequency_governor_limits_restored']
                and control['restored']==control['baseline'] and control['requested_frequency_hz']==1000000000
                and obs and all(o['actual_frequency_hz']==1000000000 for o in obs) and max(temperatures)<75.,'Board prediction controller not restored')
            controls[str(start)]=dict(observations=len(obs),maximum_temperature_c=max(temperatures),restored_NPU_hz=int(control['restored']['cur_freq']))
        report.update(status='completed_independent_saved_native518_quality_diagnostic',source_indices=INDICES,source_timestamps=stamps,
            matched_source_PNG_and_prepared_pixel_identity=True,prepared_arrays_independently_preprocessed_on_host=False,
            per_pair_five_output_diagnostics=per_pair,aggregate_five_output_diagnostics=sums,original_SL4_counts={n:dict(nodes=32,factors=32,submaps=16,observations=17) for n in sl4},
            native_SDK_totals=native_totals,board_prediction_controls=controls,trajectory_vs_GT=trajectory,trajectory_metric_changes=metric_changes,
            relative_trajectory_error_diagnostic_screens=screens,all_RMSE_and_maximum_within_10_percent=all(screens['10_percent'].values()),
            all_RMSE_and_maximum_within_20_percent=all(screens['20_percent'].values()),
            map_reference_disagreement=dict(scope='Original CPU map as reference, not ground-truth map accuracy. Each route uses its own trajectory-to-same-GT positive Sim3; all32 occurrences and fixed pixels retained.',
                units='distance in each GT-fitted similarity gauge; metric map scale not independently validated',unfiltered=distance_stats(distances),
                common_confidence_subset=distance_stats(distances[both]),full_confidence_masks=mask_stats(left['full_confidence_masks'],right['full_confidence_masks']),per_occurrence=frames),
            verified_array_count=array_count,groundtruth_sha256=GT_SHA,trajectory_helper_sha256=METRICS_SHA,
            limits=['Strict FP32 tensor failures rule out numerical equivalence, not automatically the user trajectory-quality criterion.',
                'One12.8s/17-observation trajectory is a matched precision diagnostic, not broad deployment/30min/8GB acceptance.',
                'Offline host CPU timing is not board timing; four board batches are not paced fresh-20Hz/P95 or2-5s map cadence.',
                'No ground-truth map surface was supplied; fixed-pixel point disagreement and confidence IoU measure reference consistency only.'])
    except BaseException as exc:report['error']=type(exc).__name__+': '+str(exc)
    finally:
        report['file_sha256']={str(p):h for p,h in pins.items()};changed=[str(p) for p,h in pins.items() if not p.is_file() or sha(p)!=h]
        report['saved_files_unchanged']=not changed
        if changed:report.update(status='failed',changed_files=changed)
        os.environ.clear();os.environ.update(oldenv)
        with a.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:report.get(k) for k in ('status','error','all_RMSE_and_maximum_within_10_percent','all_RMSE_and_maximum_within_20_percent')}))
    return 0 if report['status'].startswith('completed_') else 1


if __name__=='__main__':raise SystemExit(main())
