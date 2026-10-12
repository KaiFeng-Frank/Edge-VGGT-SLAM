from pathlib import Path
import hashlib,json
import numpy as np
R=Path('/dev/shm/edge-vggt-v82');D=R/'camera-calibration-r1'
P=Path('/home/frank/.local/share/edge-vggt-private/v80/capture.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(P)=='a1e6f1b2a89e1a96987befdea4f0a048aaf06066efa9fff72a370ac77d45507c'
f=D/'capture.json';c=json.loads(f.read_text());original=json.loads(P.read_text());count=0
assert c['status']=='completed_original_FP32_camera_calibration_capture' and not c['cleanup_errors']
assert c['original_camera_parameters_verified']==69 and c['original_tail_parameters_verified']==36
assert c['full_head_forward_count']==8 and c['tail_blocks_forward_count']==dict(frame23=8,global23=8)
assert c['conditioning_count']==c['delta_count']==32
assert all(c[k] for k in ['model_state_unchanged','source_and_inputs_unchanged','hooks_removed','affinity_restored','environment_restored','checkpoint_stat_unchanged'])
def array(meta):
 global count
 p=D/meta['file'];assert p.resolve().is_relative_to(D.resolve())
 assert sha(p)==meta['sha256'] and p.stat().st_size==meta['bytes']
 x=np.load(p,allow_pickle=False);assert x.dtype==np.float32 and list(x.shape)==meta['shape'] and x.flags.c_contiguous and np.isfinite(x).all()
 assert x.nbytes==meta['tensor_bytes'] and hashlib.sha256(x.tobytes()).hexdigest()==meta['raw_sha256']
 count+=1;return x
rows=[];final_hashes=[]
for i,(s,o) in enumerate(zip(c['samples'],original['samples'])):
 assert s['source_pair_indices']==o['indices']==[96*i,96*i+48] and len(s['iterations'])==4
 assert not set(s['source_pair_indices'])&set(c['evaluation_indices_excluded'])
 array(s['camera_tokens']);prediction=None
 for j,step in enumerate(s['iterations']):
  assert j==step['index'];condition,delta,pose=[array(step[k]) for k in ('conditioning_pose','delta','activated_pose')]
  if j:
   assert condition.tobytes()==prediction.tobytes();prediction=np.add(prediction,delta,dtype=np.float32)
  else:
   assert condition[0,0].tobytes()==condition[0,1].tobytes()
   assert hashlib.sha256(condition[0,0].tobytes()).hexdigest()==c['parameters']['empty_pose_tokens']['raw_sha256'];prediction=delta.copy()
  rows.append(' '.join('/mnt/vggt-data/vggt-slam2/experiments/vggt-v82-camera-calibration-r1/'+meta['file'] for meta in (s['camera_tokens'],step['conditioning_pose'])))
 assert hashlib.sha256(pose.tobytes()).hexdigest()==o['canonical_outputs']['pose_enc']['raw_sha256']
 final_hashes.append(hashlib.sha256(pose.tobytes()).hexdigest())
assert len(c['samples'])==len(original['samples'])==8 and count==104
assert (D/'dataset.txt').read_text().splitlines()==rows and sha(D/'dataset.txt')==c['dataset']['sha256']
control=json.loads((R/'camera-capture-control-r1.json').read_text())
assert control['status']=='completed' and control['returncode']==0 and not control['cleanup_errors']
assert all(x['npu_hz']==300000000 for x in control['samples'])
a={'schema':'v82_saved_original_camera_calibration_audit','capture_sha256':sha(f),'status':'passed_saved_array_and_original_full_pose_identity_checks','actual_host_model_or_GPU_or_SDK_execution':False,'saved_arrays_verified':count,'calibration_pairs':8,'step_input_pairs':32,'original_full_FP32_final_poses_byte_equal':8,'calibration_and_17_evaluation_indices_disjoint':True,'conditioning_is_previous_unactivated_FP32_sum':True,'first_conditioning_original_empty_token_raw_sha256':c['parameters']['empty_pose_tokens']['raw_sha256'],'final_pose_raw_sha256':final_hashes,'actual_model_parameter_rehash_on_host':False,'parameter_identity_scope':'Board capture checks 69 camera and 36 final-block parameters; this host audit verifies saved arrays and reference hashes only.','capture_control_sha256':sha(R/'camera-capture-control-r1.json'),'NPU_hz':300000000,'max_temperature_c':max(t for row in control['samples'] for t in row['temperatures_c'].values()),'wall_seconds':control['seconds'],'thermal_scheduler_events':len(control['compile_scheduler_events']),'realtime_or_quality_acceptance':False}
(R/'camera-calibration-saved-audit.json').write_text(json.dumps(a,indent=2)+'\n');print(json.dumps({k:a[k] for k in ['status','saved_arrays_verified','calibration_pairs','original_full_FP32_final_poses_byte_equal']}))
