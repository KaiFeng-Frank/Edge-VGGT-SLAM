"""Saved JSON/NumPy v43 complete replay parity audit; no inference or board work.

Compare all192 observations, both final states, recorded RNG digests and
declared captures against the frozen completed v42 run. Both EOF populations
remain separate from fresh outputs. DPVO-derived conventions, MIT LICENSE.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

OLD_SHA = 'c0546eb004e79fe9d953901a882d51ff60a620cc124cad7c9eed42365a339fcf'
PARENT_AUDITOR_SHA256 = '97e6047d86fdaf1503250f71b5cd087c5c61d6e60383f940b891dc2ce78e170a'
DRIVER_SHA = '176d3cb37abcf38818ac37a38e9ba1924b84caf4b3071784850058755ca96d65'
CLOCK_FIELDS = {'captured_monotonic', 'process_seconds'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata(value):
    value=np.ascontiguousarray(value)
    return dict(shape=list(value.shape),dtype=str(value.dtype),bytes=value.nbytes,
        tensor_sha256=hashlib.sha256(value.tobytes()).hexdigest())


def compare_nonclock(left,right,path='',mismatches=None,clocks=None):
    mismatches=[] if mismatches is None else mismatches
    clocks=[] if clocks is None else clocks
    if type(left) is not type(right):mismatches.append(path+':type');return mismatches,clocks
    if isinstance(left,dict):
        if left.keys()!=right.keys():mismatches.append(path+':keys');return mismatches,clocks
        for k in left:
            if k in CLOCK_FIELDS:
                if not np.isfinite([left[k],right[k]]).all() or min(left[k],right[k])<0:
                    mismatches.append(path+'.'+k+':invalid_clock')
                elif left[k]!=right[k]:clocks.append(path+'.'+k)
            else:compare_nonclock(left[k],right[k],path+'.'+k,mismatches,clocks)
    elif isinstance(left,list):
        if len(left)!=len(right):mismatches.append(path+':length');return mismatches,clocks
        for i,(a,b) in enumerate(zip(left,right)):compare_nonclock(a,b,path+f'[{i}]',mismatches,clocks)
    elif left!=right:mismatches.append(path)
    return mismatches,clocks


def verify_final_RNG(new,old):
    """Compare recorded final state descriptors/digests, not nonexistent raw files."""
    values=[]
    for run in (new,old):
        value=run.get('random_generator_after')
        if not isinstance(value,dict) or set(value)!=set(('shape','dtype','bytes','tensor_sha256')):
            raise ValueError('Both successful final RNG descriptors required')
        digest=value['tensor_sha256']
        if (value['shape']!=[5056] or value['dtype']!='uint8' or value['bytes']!=5056
                or not isinstance(digest,str) or len(digest)!=64
                or any(c not in '0123456789abcdef' for c in digest)
                or run.get('random_depth_draw_calls')!=192):
            raise ValueError('Original successful RNG descriptor/draw count differs')
        values.append(value)
    return dict(same_metadata_and_digest=values[0]==values[1],raw_RNG_bytes_reread=False,
        new=values[0],old=values[1],depth_draw_calls_each=192)


def audit(new_directory,old_directory,*,new_sha256,old_sha256):
    new_directory,old_directory=Path(new_directory),Path(old_directory)
    pins={};checks=0;mismatches=[];clock_changes=[]
    def check(ok,message):
        nonlocal checks
        checks+=1
        if not ok:raise ValueError(message)
    def pin(path,digest):
        check(sha(path)==digest,'File SHA differs: '+str(path));pins[str(path)]=digest
    pin(new_directory/'run.json',new_sha256);pin(old_directory/'run.json',old_sha256)
    check(old_sha256==OLD_SHA,'Only the declared frozen completed v42 reference is allowed')
    new,old=(json.loads((d/'run.json').read_text()) for d in (new_directory,old_directory))
    check(new['schema']=='v43_cached_undistort_metadata_only_online_RKNN_research_replay'
        and new['script_sha256']==DRIVER_SHA
        and new['original_v42_online_driver_sha256']=='17d8e26018b1499ae576b1474946ee393b5ce701c562987d8f9060f7b181960d'
        and new['metadata_only_reporting'] is True and new['native_input_transport_added'] is False
        and new['original_v41_online_driver_sha256']=='73f5a4c011d957f95d8d8e54687a167709f22cc6104611a7d62bf7b4f68ea6d9'
        and new['status']=='completed_unpaced_half_online_RKNN_research_not_realtime',
        'New completed metadata-only v43 source/lifecycle differs')
    check(old['schema']=='v42_cached_undistort_online_RKNN_research_replay'
        and old['script_sha256']=='17d8e26018b1499ae576b1474946ee393b5ce701c562987d8f9060f7b181960d'
        and old['status']=='completed_unpaced_half_online_RKNN_research_not_realtime',
        'Old completed run must retain its actual schema/status')
    for label,run,d in (('v43',new,new_directory),('v42',old,old_directory)):
        check(run['frames_completed']==run['frames_requested']==len(run['frames'])==192,label+': complete192 required')
        check(run['cleanup_errors']==[] and run['sources_and_inputs_unchanged'] is True
            and run['source_pins']==run['source_hashes_after'] and run['original_weights_unchanged'] is True
            and run['original_weights_before']==run['original_weights_after'] and len(run['original_weights_before'])==94,
            label+': original94/source/cleanup report differs')
        check(run['candidate_accepted'] is False and run['historical_SDK_strict_FP32_rejected'] is True
            and run['SDK_FP32_equivalence_validated'] is False and run['FP32_thresholds_relaxed'] is False,
            label+': historical rejection must be preserved')
        pin(d/'input-manifest.json',run['input_manifest_sha256'])
    check(new['input_manifest_sha256']==old['input_manifest_sha256']
        and new['policy']==old['policy'] and new['policy_sha256']==old['policy_sha256']
        and new['original_weights_before']==old['original_weights_before']
        and new['seed']==old['seed'] and new['random_generator_before']==old['random_generator_before'],
        'Source plan/policy/original94/initial RNG differs')
    check(new['metrics_before_termination']==old['metrics_before_termination'],
        'Before-EOF complete online lifecycle differs')
    for i,(a,b) in enumerate(zip(new['frames'],old['frames'])):
        check(a['index']==b['index']==i and a['frame_id']==b['frame_id'] and a['timestamp']==b['timestamp'],
            'Original frame/source identity differs')
        compare_nonclock(a['result'],b['result'],f'frame[{i}].result',mismatches,clock_changes)
        if i<3:
            check(all(a[k]==b[k] for k in ('original_full_crop_bytes_match','resized_BGR','normalized_half_input')),
                'First3 original pixels/normalized-input metadata differs')
    compare_nonclock(new['snapshot_before_termination']['metadata'],old['snapshot_before_termination']['metadata'],
        'before_EOF.metadata',mismatches,clock_changes)
    check(new['stream_metrics']==old['stream_metrics'],'Completed EOF lifecycle metrics differ')
    compare_nonclock(new['termination'],old['termination'],'EOF',mismatches,clock_changes)
    compare_nonclock(new['final_snapshot']['metadata'],old['final_snapshot']['metadata'],
        'final.metadata',mismatches,clock_changes)
    rng=verify_final_RNG(new,old);check(rng['same_metadata_and_digest'],'Successful final RNG digest differs')
    descriptors=[('before-termination',new['snapshot_before_termination']['artifact'],old['snapshot_before_termination']['artifact']),
        ('final-state',new['final_snapshot']['artifact'],old['final_snapshot']['artifact'])]
    check([r['frame_index'] for r in new['update_captures']]==[r['frame_index'] for r in old['update_captures']]==[34,127,191],
        'All three predeclared captures required')
    for a,b in zip(new['update_captures'],old['update_captures']):
        check({k:v for k,v in a.items() if k not in ('input','output')}=={k:v for k,v in b.items() if k not in ('input','output')},
            'Capture identity/scope differs')
        for kind in ('input','output'):descriptors.append((f'capture-{a["frame_index"]}-{kind}',a[kind],b[kind]))
    array_pairs=0;array_evidence=[]
    for label,a,b in descriptors:
        paths=[]
        for desc,d in ((a,new_directory),(b,old_directory)):
            check(Path(desc['file']).name==desc['file'],'Artifact path escape')
            path=d/desc['file'];pin(path,desc['sha256']);check(path.stat().st_size==desc['bytes'],'NPZ bytes differ')
            paths.append(path)
        with np.load(paths[0],allow_pickle=False) as x,np.load(paths[1],allow_pickle=False) as y:
            check(set(x.files)==set(y.files)==set(a['arrays'])==set(b['arrays']),'NPZ array names differ')
            for name in x.files:
                v,w=x[name],y[name];vm,wm=metadata(v),metadata(w)
                check(vm==a['arrays'][name] and wm==b['arrays'][name],'Saved raw array descriptor differs')
                check(np.isfinite(v).all() and np.isfinite(w).all(),'Saved array is nonfinite')
                equal=v.dtype==w.dtype and v.shape==w.shape and v.tobytes(order='C')==w.tobytes(order='C')
                if not equal:mismatches.append(label+'.'+name+':bytes')
                array_pairs+=1;array_evidence.append(dict(artifact=label,array=name,bitwise_equal=equal,new=vm,old=wm))
    term=new['termination'];before=new['metrics_before_termination'];after=new['stream_metrics']
    check(term['fresh_pose_valid'] is False and term['online_publication'] is False
        and term['reconstructed_poses_counted_as_fresh'] is False and term['offline_updates']==12
        and after['offline_updates']==12 and after['fresh_poses']==before['fresh_poses']
        and after['update_calls']==before['update_calls']+12 and after['BA_iterations']==before['BA_iterations']+24,
        'New EOF must be separate from fresh output')
    state,cache=new['online_metrics'],new['online_metrics']['cache']
    check(state['full_completed']==state['full_attempts']==181 and state['motion_probe_completed']==34
        and state['requests']==215 and state['failed_calls']==0 and state['closed'] is True and state['failed'] is False
        and cache['closed'] is True and cache['cleanup_errors']==[]
        and all(cache[k]==0 for k in ('resident_contexts','current_model_host_bytes','current_DMA_bytes')),
        'Complete new callback/cleanup counters differ')
    fresh=[r for r in new['frames'] if r['result']['fresh_pose_valid']]
    check(len(fresh)==before['fresh_poses']==158,'Fresh source coverage differs')
    check(new['SDK_activity']==old['SDK_activity'],'Actual reported SDK activity differs')
    oldstate=old['online_metrics'];oldcache=oldstate['cache']
    check(all(oldstate[k]==state[k] for k in ('requests','full_attempts','full_completed','motion_probe_attempts',
        'motion_probe_completed','failed_calls','closed','failed','shape_histogram','model_keys'))
        and oldcache['closed'] is True and oldcache['cleanup_errors']==[]
        and all(oldcache[k]==0 for k in ('resident_contexts','current_model_host_bytes','current_DMA_bytes')),
        'Old completed callback/cleanup differs')
    check(new['random_depth_draw_calls']==192,'New successful source RNG draw count differs')
    check(all(sha(p)==v for p,v in pins.items()),'Audited file changed during saved-only comparison')
    return dict(schema='v43_metadata_only_complete_saved_array_parity',
        status='passed_exact_complete_replay_metadata_only_parity' if not mismatches else 'failed_complete_replay_parity',
        script_sha256=sha(__file__),parent_auditor_sha256=PARENT_AUDITOR_SHA256,checks_passed=checks,pins=pins,new_run_sha256=new_sha256,old_run_sha256=old_sha256,
        original_run_statuses=dict(v43=new['status'],v42=old['status']),
        nonclock_frame_results_compared=192,clock_field_allowlist=sorted(CLOCK_FIELDS),changed_clock_fields=len(clock_changes),
        mismatches=mismatches,actual_NPZ_pairs_reread=len(descriptors),actual_array_pairs_compared=array_pairs,
        array_evidence=array_evidence,original94_weights_same_metadata=True,initial_seed_and_RNG_same_metadata=True,
        old_final_RNG_recorded=True,final_RNG_comparison=rng,final_state_compared=True,
        actual_final_state_array_pairs_compared=sum(v['artifact']=='final-state' for v in array_evidence),
        new_final_RNG_reported=new['random_generator_after'],new_random_depth_draw_calls=192,
        fresh_count=len(fresh),fresh_source_coverage=len(fresh)/192,replay_elapsed_seconds=new['replay_elapsed_seconds'],
        fresh_Hz=len(fresh)/new['replay_elapsed_seconds'],fresh_P95_publication_ms=float(np.percentile([r['admission_to_publication_seconds'] for r in fresh],95)*1000),
        EOF=dict(offline_updates=12,offline_BA_steps=24,seconds=new['termination_seconds'],
            counted_as_fresh=False,old_EOF_was_failed=False,compared_to_old_EOF=True),
        SDK_activity_reported=new['SDK_activity'],callback_counts=dict(requests=215,full_completed=181,motion_probe_completed=34,failed_calls=0),
        cache_closed=True,current_DMA_bytes=0,cleanup_errors=[],peak_process_RSS_mib=new['peak_rss_mib'],
        historical_FP32_and_workload_quality_rejections_retained=True,candidate_accepted=False,
        limitations=['Initial and successful final RNG metadata/digests are compared. Raw RNG bytes were not saved separately and are not reread here.',
            'Eight declared NPZ pairs and their raw arrays are reread; images, models, original checkpoint bytes and SDK operator placement are not executed or independently checked here.',
            'Both completed EOF outputs and final states are compared; EOF remains offline and cannot count as fresh output.',
            'Unpaced separate-run timing is not a controlled speedup,20Hz,physical-camera,30-minute or map/trajectory-accuracy acceptance.'],
        model_calls=0,ORT_calls=0,SDK_calls=0,BA_calls=0,board_jobs=0,images_read=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('new-run-dir','old-run-dir','output'):parser.add_argument('--'+key,type=Path,required=True)
    for key in ('new-run-sha256','old-run-sha256'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();report=audit(args.new_run_dir,args.old_run_dir,new_sha256=args.new_run_sha256,old_sha256=args.old_run_sha256)
    with args.output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:report[k] for k in ('status','checks_passed','nonclock_frame_results_compared','actual_array_pairs_compared','mismatches')}))
    return 0 if not report['mismatches'] else 2


if __name__=='__main__':raise SystemExit(main())
