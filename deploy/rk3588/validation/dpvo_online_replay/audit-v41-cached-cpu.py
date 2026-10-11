"""Compare saved original/cached CPU replay results; no model or image work."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--original-dir',type=Path,required=True);p.add_argument('--cached-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(a.original_dir/'run.json')=='dd9f37524b999c1e586182a92c226481fcfc7f4da851d5e88d62d82be1029ecd'
assert sha(a.cached_dir/'run.json')=='c1f0a4636601fb1cd702ca57018af5fd937ecdbb9e722b0293428810bf66d05d'
x=json.loads((a.original_dir/'run.json').read_text());y=json.loads((a.cached_dir/'run.json').read_text())
assert y['status']=='completed_unpaced_half_Torch_CPU_research_not_realtime' and y['frames_completed']==192
assert y['input_manifest_sha256']==x['input_manifest_sha256'] and y['policy_sha256']==x['policy_sha256']
assert y['original_weights_before']==y['original_weights_after']==x['original_weights_before']
assert y['sources_and_inputs_unchanged'] and y['source_pins']==y['source_hashes_after'] and y['affinity_restored'] and not y['cleanup_errors']
changed=[]
def compare(left,right,path):
    assert type(left)==type(right),path
    if isinstance(left,dict):
        assert left.keys()==right.keys(),path
        for k in left:
            if k in ('captured_monotonic','process_seconds'):
                if left[k]!=right[k]:changed.append(path+'.'+k)
            else:compare(left[k],right[k],path+'.'+k)
    elif isinstance(left,list):
        assert len(left)==len(right),path
        for i,(m,n) in enumerate(zip(left,right)):compare(m,n,path+f'[{i}]')
    else:assert left==right,path
for i,(m,n) in enumerate(zip(x['frames'],y['frames'])):compare(m['result'],n['result'],f'frame[{i}]')
compare(x['termination'],y['termination'],'termination')
assert x['random_generator_after']==y['random_generator_after'] and x['random_depth_draw_calls']==y['random_depth_draw_calls']
files=['before-termination.npz','final-state.npz']+[f'update-{i:03d}-{kind}.npz' for i in (34,127,191) for kind in ('input','gold')]
arrays=0
for file in files:
    with np.load(a.original_dir/file,allow_pickle=False) as left,np.load(a.cached_dir/file,allow_pickle=False) as right:
        assert set(left.files)==set(right.files)
        for key in left.files:
            assert left[key].shape==right[key].shape and left[key].dtype==right[key].dtype and left[key].tobytes()==right[key].tobytes(),(file,key)
            arrays+=1
fresh=[r for r in y['frames'] if r['result']['fresh_pose_valid']]
assert len(fresh)==158 and y['undistort_cache']['completed_calls']==192 and y['undistort_cache']['closed'] and y['undistort_cache']['current_map_bytes']==0
report=dict(schema='v41_cached_CPU_full_replay_saved_parity',status='passed_exact_nonclock_results_and_saved_arrays',original_sha256=sha(a.original_dir/'run.json'),cached_sha256=sha(a.cached_dir/'run.json'),frame_results_checked=192,changed_clock_fields=len(changed),clock_field_allowlist=['captured_monotonic','process_seconds'],actual_NPZ_pairs_reread=len(files),actual_array_pairs_compared=arrays,original94_weights_preserved=True,original_policy_and_RNG_preserved=True,new_full_quality_claim=False,prior_workload_quality_rejection_retained=True,
    fresh_count=len(fresh),elapsed_seconds=y['replay_elapsed_seconds'],fresh_Hz=len(fresh)/y['replay_elapsed_seconds'],fresh_P95_ms=float(np.percentile([r['admission_to_publication_seconds'] for r in fresh],95)*1000),
    all192_mean_stages_ms={label:{k:float(np.mean([f[k] for f in run['frames']])*1000) for k in ('decode_undistort_crop_resize_seconds','encode_seconds','stream_seconds')} for label,run in [('original_v39',x),('cached_v41',y)]},
    limitations=['Separate runs had uncontrolled temperature/frequency; faster preprocessing does not imply faster total replay.','Old +77% common-fresh ATE cost of half workload remains; cached preprocessing does not repair accuracy.','Unpaced192-file replay, no physical camera/queue/endurance or realtime acceptance.'])
assert not a.output.exists();a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
