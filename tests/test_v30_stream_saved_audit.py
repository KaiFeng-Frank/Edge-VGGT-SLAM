"""Small synthetic saved-value guards; no model, image, geometry or BA calls."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
ENTRY = HERE / "audit-v30-dpvo-stream.py"
if not ENTRY.exists():
    ENTRY = HERE.parent / "deploy/rk3588/validation/dpvo_stream/audit-v30-dpvo-stream.py"
SPEC = importlib.util.spec_from_file_location("v30_saved_audit", ENTRY)
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)


def diagnostic(edges, n):
    interval = [max(n-10,1), n]
    return dict(edges=edges,iterations=2,optimized_pose_interval=interval,
        diagnostic=dict(optimized_pose_interval=interval,upstream_eff_impl_requested=False,
            iterations=[dict(iteration=i,active_edges=edges,weighted_squared_residual_before=.1,
                pose_update_norm=.01,inverse_depth_update_norm=.01) for i in range(2)]))


def lifecycle_fixture():
    """Hand-specified 10 observations: skip1, initialize8, delete source6 at9."""
    inputs = [dict(frame_id=str(100+i),timestamp=float(100+i)) for i in range(10)]
    rows=[]
    for i in range(10):
        skip = i == 1
        n = 1 if i <= 1 else i
        initialized = i >= 8
        updates = 12 if i == 8 else 1 if i == 9 else 0
        edges = 96 * n * n
        result=dict(frame_id=inputs[i]['frame_id'],timestamp=inputs[i]['timestamp'],
            observation_id=i,captured_monotonic=float(10*i),admitted=not skip,
            initialized=initialized,fresh_pose_valid=bool(updates),
            pose_w2c=[0,0,0,0,0,0,1] if updates else None,
            motion_probe=None if i in (0,9) else 1. if skip else 3.,
            updates=updates,process_seconds=.1,reason='initial_motion_below_2' if skip else
                'updated_current_observation' if updates else 'initialization_wait')
        if not skip:
            result.update(update_diagnostics=[diagnostic(edges,n) for _ in range(updates)],
                keyframe=dict(mean_bidirectional_flow=0.,removed_observation_id=6) if i==9 else None,
                retained_frames=8 if i==9 else n,active_edges=6144 if i==9 else edges)
        rows.append(dict(index=i,frame_id=inputs[i]['frame_id'],timestamp=inputs[i]['timestamp'],
            arrival_monotonic=float(10*i),pose_available_monotonic=float(10*i)+.75,
            decode_undistort_crop_seconds=.25,encode_seconds=.25,stream_seconds=.25,
            processing_seconds=.75,result=result))
    return rows,inputs


def test_original_counts_and_id_remap_from_saved_rows():
    rows,inputs=lifecycle_fixture()
    s=A.lifecycle(rows,inputs,A.DEFAULT)
    assert s['init_index']==8 and s['fresh']==[8,9] and s['skipped']==[1]
    assert s['retained']==[0,2,3,4,5,7,8,9] and s['removed']==[6]
    assert s['relative_parents']=={1:0,6:5}
    assert len(s['ii'])==6144 and np.array_equal(s['ii'],s['kk']//96)
    assert s['source_updates']==[8]*12+[9]
    assert len(s['expected_calls'])==21
    assert max(s['kk'])==8*96-1 and max(s['jj'])==7


@pytest.mark.parametrize('mutant', ['waiting_fresh','skipped_fresh','init_count','next_count',
    'wrong_removed_id','source_id','capture','time_scope','missing_row'])
def test_bad_saved_lifecycle_rejected(mutant):
    rows,inputs=lifecycle_fixture()
    if mutant=='waiting_fresh':rows[0]['result']['fresh_pose_valid']=True
    elif mutant=='skipped_fresh':rows[1]['result']['fresh_pose_valid']=True
    elif mutant=='init_count':rows[8]['result']['updates']=1
    elif mutant=='next_count':rows[9]['result']['updates']=12
    elif mutant=='wrong_removed_id':rows[9]['result']['keyframe']['removed_observation_id']=5
    elif mutant=='source_id':rows[8]['result']['frame_id']='old'
    elif mutant=='capture':rows[8]['result']['captured_monotonic']=70.
    elif mutant=='time_scope':rows[8]['pose_available_monotonic']+=1.
    else:rows.pop()
    with pytest.raises(ValueError):A.lifecycle(rows,inputs,A.DEFAULT)


def test_npz_file_pin_and_raw_metadata_guard(tmp_path):
    path=tmp_path/'tiny.npz';array=np.array([0.,-0.,1.],np.float32)
    np.savez(path,values=array)
    item=dict(file=path.name,bytes=path.stat().st_size,sha256=A.sha(path),arrays={'values':A.meta(array)})
    assert A.exact(A.artifact(tmp_path,item)['values'],array)
    bad=copy.deepcopy(item);bad['arrays']['values']['tensor_sha256']='0'*64
    with pytest.raises(ValueError,match='raw array'):A.artifact(tmp_path,bad)
    np.savez(path,values=array+1.)
    with pytest.raises(ValueError,match='file SHA'):A.artifact(tmp_path,item)


def test_source_index_order_guard(tmp_path,monkeypatch):
    path=tmp_path/'rgb.txt';path.write_text('# metadata\n100.0 rgb/100.png\n101.0 rgb/101.png\n')
    monkeypatch.setattr(A,'RGB_SHA',A.sha(path))
    manifest=dict(schema='v30_first_contiguous_TUM_RGB_frames',GT_read=False,rgb_index_sha256=A.sha(path),
        frames=[dict(frame_id=str(i)+'.0',timestamp=float(i),relative_path='rgb/'+str(i)+'.png',bytes=3,sha256='0'*64)
            for i in (100,101)])
    assert len(A.source_rows(path,manifest,2))==2
    manifest['frames'].reverse()
    with pytest.raises(ValueError,match='identity/order'):A.source_rows(path,manifest,2)


def cache_fixture():
    inputs=[dict(frame_id=str(i)) for i in range(4)]
    key=dict(source=['1','3'],target=['3'],patches_per_frame=96)
    generation=dict(index=0,key=json.dumps(key),source_frame_ids=key['source'],target_frame_ids=key['target'])
    corr=dict(attempted_calls=2,completed_calls=2,current_context_open=False,generations=[generation],
        calls=[dict(index=i,edges=192,generation=0) for i in range(2)],
        closed_contexts=[dict(closed=True,current_context_bytes=0,failed_calls=0)])
    calls=[dict(edges=192,source_ids=[1,3],target_ids=[3]) for _ in range(2)]
    return corr,calls,inputs


def test_same_source_members_can_reuse_cache_after_slot_compression():
    corr,calls,inputs=cache_fixture()
    A.correlation_check(corr,calls,inputs)


@pytest.mark.parametrize('mutant',['wrong_member','wrong_order','open_context'])
def test_native_cache_saved_identity_and_close_guards(mutant):
    corr,calls,inputs=cache_fixture()
    if mutant=='wrong_member':calls[-1]['source_ids']=[0,3]
    elif mutant=='wrong_order':corr['generations'][0]['source_frame_ids'].reverse()
    else:corr['closed_contexts'][0]['current_context_bytes']=64
    with pytest.raises(ValueError):A.correlation_check(corr,calls,inputs)


def test_offline_inverse_and_reconstruction_are_not_fresh():
    rows,inputs=lifecycle_fixture();s=A.lifecycle(rows,inputs,A.DEFAULT)
    pose=np.array([0,0,0,0,0,0,1],np.float32)
    final={'poses':np.tile(pose,(1,8,1))};before={'poses':final['poses'].copy()}
    result=dict(offline_updates=12,fresh_pose_valid=False,online_publication=False,initialized=True,
        trajectory_state='initialized_offline_reconstruction',reconstructed_poses_counted_as_fresh=False,
        reconstructed_observation_ids=[1,6],update_diagnostics=[diagnostic(6144,8) for _ in range(12)],
        observations=inputs,pose_w2c=np.tile(pose,(10,1)).tolist(),pose_c2w=np.tile(pose,(10,1)).tolist())
    relative={'1':[0,pose.tolist()],'6':[5,pose.tolist()]}
    A.termination_check(result,before,final,s,inputs,rows,relative)
    result['fresh_pose_valid']=True
    with pytest.raises(ValueError,match='counted as fresh'):A.termination_check(result,before,final,s,inputs,rows,relative)
    result['fresh_pose_valid']=False;result['pose_c2w'][6][0]=1.
    with pytest.raises(ValueError,match='inverse-pose'):A.termination_check(result,before,final,s,inputs,rows,relative)
