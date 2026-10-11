"""Toy numeric trajectories only; no image, model, ORT, SDK or BA execution."""
import ast
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
VALIDATION = REPO/'deploy/rk3588/validation'
ROOT = VALIDATION/'dpvo_fast_rknn'
PARENT = VALIDATION/'dpvo_fast/evaluate-v34-fast-trajectory.py'
spec = importlib.util.spec_from_file_location('v34_saved_trajectory_synthetic', ROOT/'evaluate-v35-fast-rknn-trajectory.py')
ev = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ev
spec.loader.exec_module(ev)


def identity_pose(center):
    return np.r_[-np.asarray(center), 0., 0., 0., 1.].tolist()


def save_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False) + '\n')


@pytest.fixture
def pair(tmp_path):
    times=np.arange(9)*.25
    centers=np.stack((times,.15*times**2,.1*np.sin(times)),axis=1)
    gt=tmp_path/'groundtruth.txt'
    np.savetxt(gt,np.column_stack((times,centers,np.tile([0.,0.,0.,1.],(9,1)))))
    sources=[dict(frame_id=str(t),timestamp=float(t),relative_path=f'rgb/generated-{i}.png',bytes=0,sha256='synthetic-not-image') for i,t in enumerate(times)]
    dirs=[]
    for name,schema,scale,offset,indices,cfg in [
        ('CPU_FAST',ev.CPU_RUN_SCHEMA,2.,np.array([10.,-3.,2.]),set(range(9)),(48,11,16,7)),
        ('NPU_encoder_FAST',ev.NPU_RUN_SCHEMA,3.,np.array([1.,2.,-4.]),{0,2,4,5,6,8},(48,11,16,7))]:
        folder=tmp_path/name;folder.mkdir();dirs.append(folder)
        save_json(folder/'input-manifest.json',{'schema':'v30_first_contiguous_TUM_RGB_frames','frames':sources,'GT_read':False})
        frames=[];pc=centers*scale+offset
        for i,t in enumerate(times):
            cap=100.+i*.01;reply=cap+.005;fresh=i in indices
            result=dict(observation_id=i,frame_id=str(t),timestamp=float(t),captured_monotonic=cap,
                        fresh_pose_valid=fresh,admitted=fresh,initialized=fresh,updates=int(fresh),
                        reason='updated_current_observation' if fresh else 'initial_motion_below_2',
                        pose_w2c=identity_pose(pc[i]) if fresh else None)
            frames.append(dict(index=i,frame_id=str(t),timestamp=float(t),arrival_monotonic=cap,
                               pose_available_monotonic=reply,processing_seconds=reply-cap,result=result))
        term=dict(observations=[dict(observation_id=i,frame_id=str(t),timestamp=float(t)) for i,t in enumerate(times)],
                  fresh_pose_valid=False,online_publication=False,reconstructed_poses_counted_as_fresh=False,
                  trajectory_state='initialized_offline_reconstruction',initialized=True,offline_updates=12,
                  reconstructed_observation_ids=[],pose_w2c=[identity_pose(p) for p in pc],
                  pose_c2w=[np.r_[p,0.,0.,0.,1.].tolist() for p in pc])
        run=dict(schema=schema,status='synthetic_completed',frames_requested=9,frames_completed=9,
                 input_manifest_sha256=ev.sha(folder/'input-manifest.json'),frames=frames,
                 metrics_before_termination={'fresh_poses':len(indices),'observations':9},termination=term,
                 config=dict(zip(('patches_per_frame','patch_lifetime','removal_window','optimization_window'),cfg),
                             capacity=64,loop_closure=False,classic_loop_closure=False),
                 GT_or_sensor_depth_input=False,candidate_accepted=False,original_VGGT_default_unchanged=True,
                 diagnostic_allow_fp32_rejected=True,historical_encoder_FP32_gate_passed=False,
                 FP32_thresholds_relaxed=False,output_clipping=False,CPU_encoder_fallback=False,
                 first_three_FP32_gate_passed=False,
                 FAST_config_source={'default_yaml_not_applied':True,'merge_order':['dpvo/config.py','config/fast.yaml']})
        save_json(folder/'run.json',run)
    return dirs,gt


def compare(pair):
    dirs,gt=pair
    return ev.compare(*dirs,gt,ev.sha(dirs[0]/'run.json'),ev.sha(dirs[1]/'run.json'),ev.sha(gt),PARENT)


def test_frozen_parent_math_and_association_functions_unchanged():
    parent={n.name:n for n in ast.parse(PARENT.read_text()).body if isinstance(n,ast.FunctionDef)}
    child={n.name:n for n in ast.parse((ROOT/'evaluate-v35-fast-rknn-trajectory.py').read_text()).body if isinstance(n,ast.FunctionDef)}
    for name in ('matrices','inverse','summary','nearest_association','endpoint_plan',
                 'fit_positive_sim3_centers','evaluate_population'):
        assert ast.dump(parent[name],include_attributes=False)==ast.dump(child[name],include_attributes=False)
    assert ev.sha(PARENT)==ev.PARENT_EVALUATOR_SHA256


def test_common_indices_fixed_plan_single_positive_alignment_and_EOF_separate(pair):
    result=compare(pair);common=result['additional_predeclared_common_fresh']
    assert common['common_fresh_source_indices']==[0,2,4,5,6,8]
    assert common['common_GT_associated_count']==6
    assert result['fixed_1second_all_source_plan']==[(0,4),(1,5),(2,6),(3,7),(4,8)]
    assert common['common_endpoint_pairs']==[[0,4],[2,6],[4,8]]
    assert common['CPU_FAST']['ATE_alignment']['scale']==pytest.approx(.5)
    assert common['NPU_encoder_FAST']['ATE_alignment']['scale']==pytest.approx(1/3)
    for label in ('CPU_FAST','NPU_encoder_FAST'):
        assert common[label]['ATE_sim3_aligned_meters']['max']<1e-12
        assert common[label]['RPE_translation_GT_scale_meters']['max']<1e-12
        assert common[label]['RPE_rotation_degrees']['max']==0
        assert common[label]['RPE_endpoint_coverage']==3/5
        assert result['complete_populations'][label]['EOF_reoptimized_and_reconstructed']['counted_as_fresh'] is False
        assert result['complete_populations'][label]['EOF_reoptimized_and_reconstructed']['evaluation']['poses']==9
    assert result['complete_populations']['CPU_FAST']['fresh_online']['poses']==9
    assert result['complete_populations']['NPU_encoder_FAST']['fresh_online']['poses']==6
    assert result['complete_populations']['NPU_encoder_FAST']['original_run_schema']==ev.NPU_RUN_SCHEMA


def test_wrong_FAST_schema_rejected_without_relabeling_original_run(pair):
    dirs,_=pair;p=dirs[1]/'run.json';d=json.loads(p.read_text());d['schema']=ev.CPU_RUN_SCHEMA;save_json(p,d)
    with pytest.raises(ValueError,match='schema'):compare(pair)
    assert json.loads(p.read_text())['schema']==ev.CPU_RUN_SCHEMA


def test_different_all_source_schedule_rejected_even_identical_timestamps(pair):
    dirs,_=pair;p=dirs[1]/'input-manifest.json';d=json.loads(p.read_text());d['frames'][8]['sha256']='different-pixels';save_json(p,d)
    run=dirs[1]/'run.json';d=json.loads(run.read_text());d['input_manifest_sha256']=ev.sha(p);save_json(run,d)
    with pytest.raises(ValueError,match='complete original source plans'):compare(pair)


def test_missing_GT_and_degenerate_common_population_are_unavailable_not_zero_error():
    gt=np.broadcast_to(np.eye(4),(4,4,4)).copy();poses={i:np.eye(4) for i in range(4)}
    row=ev.common_fresh_analysis(poses,{1:np.eye(4),3:np.eye(4)}, {1:1},gt,[(0,2),(1,3)],['a','b','c','d'])
    assert row['common_fresh_count']==2 and row['common_GT_associated_count']==1
    assert row['common_endpoint_pair_count']==0
    for label in ('CPU_FAST','NPU_encoder_FAST'):
        assert row[label]['ATE_alignment'] is None
        assert row[label]['ATE_sim3_aligned_meters']['rmse'] is None
        assert row[label]['RPE_rotation_degrees']['count']==0


def test_known_rotation_translation_and_TUM_xyzw_convention():
    # A genuine 90-degree z rotation in Tcw; GT is inverse camera-to-world.
    current=ev.matrices([[1.,2.,0.,0.,0.,np.sqrt(.5),np.sqrt(.5)]])[0]
    expected_Tcw=np.array([[0.,-1.,0.,1.],[1.,0.,0.,2.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
    np.testing.assert_allclose(current,expected_Tcw,rtol=0,atol=1e-15)
    expected_Twc=np.array([[0.,1.,0.,-2.],[-1.,0.,0.,1.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
    # GT is hand-derived rather than generated with the evaluator inverse.
    poses={0:np.eye(4),1:current};gt=np.stack([np.eye(4),expected_Twc])
    result=ev.evaluate_population(poses,{0:0,1:1},gt,[(0,1)])
    assert result['RPE_rotation_degrees']['max']==0
    # arccos of parallel vectors may round just below one; this is 2e-6 deg.
    assert result['RPE_translation_direction_degrees_nonzero_vectors']['max']<2e-6


def test_GT20ms_boundary_and_unique_source_order_matches_are_preserved():
    assert ev.nearest_association(np.array([0.,1.]),np.array([.019,1.021]))=={0:0}
    assert ev.nearest_association(np.array([.009,.011]),np.array([.01]))=={0:0}


@pytest.mark.parametrize('field,value',[('diagnostic_allow_fp32_rejected',False),('CPU_encoder_fallback',True),('historical_encoder_FP32_gate_passed',True),('FP32_thresholds_relaxed',True)])
def test_NPU_rejection_and_no_fallback_contract_required(pair,field,value):
    dirs,_=pair;path=dirs[1]/'run.json';run=json.loads(path.read_text());run[field]=value;save_json(path,run)
    with pytest.raises(ValueError,match='diagnostic'):compare(pair)
