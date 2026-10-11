"""Small generated JSON/math guards; no actual model/ORT/SDK/BA/image work."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

TEST=Path(__file__).resolve().parent
HERE=TEST if (TEST/'evaluate-v42-online.py').is_file() else Path(os.environ.get('V42_ONLINE_SOURCE',TEST.parent/'deploy/rk3588/validation/dpvo_online_replay'))


def load(name):
    spec=importlib.util.spec_from_file_location('_v42_saved_toy_'+name.replace('-','_'),HERE/name)
    value=importlib.util.module_from_spec(spec);sys.modules[spec.name]=value;spec.loader.exec_module(value);return value


audit=load('audit-v42-online-prefix.py');evaluator=load('evaluate-v42-online.py')


def completed():
    before=dict(status='running',failure=None,offline_updates=0,observations=192,admitted=165,
        motion_skipped=27,fresh_poses=158,motion_probe_calls=34,update_calls=169,BA_calls=169,BA_iterations=338)
    after=dict(before,status='terminated',offline_updates=12,update_calls=181,BA_calls=181,BA_iterations=362)
    state=dict(closed=True,failed=False,failed_calls=0,full_attempts=181,full_completed=181,
        motion_probe_attempts=34,motion_probe_completed=34,requests=215,
        calls=[dict(index=i,status='completed_diagnostic_not_adopted') for i in range(215)],
        cache=dict(closed=True,cleanup_errors=[],resident_contexts=0,current_DMA_bytes=0,current_model_host_bytes=0))
    return dict(schema=evaluator.SCHEMAS['online_RKNN'],status='completed_unpaced_half_online_RKNN_research_not_realtime',
        frames_requested=192,frames_completed=192,frames=[{} for _ in range(192)],cleanup_errors=[],
        metrics_before_termination=before,stream_metrics=after,online_metrics=state,
        termination=dict(offline_updates=12,fresh_pose_valid=False,online_publication=False,
            reconstructed_poses_counted_as_fresh=False,initialized=True,trajectory_state='initialized_offline_reconstruction'),
        candidate_accepted=False,historical_SDK_strict_FP32_rejected=True,SDK_FP32_equivalence_validated=False,
        FP32_thresholds_relaxed=False,complete_Update_Torch_fallback=False)


def test_nonclock_comparison_only_allows_declared_clock_changes():
    before=dict(captured_monotonic=1.,process_seconds=.1,pose=[1.,2.],updates=1)
    after=copy.deepcopy(before)
    after.update(captured_monotonic=2.,process_seconds=.2)
    mismatches,clocks=audit.compare_nonclock(before,after)
    assert not mismatches and len(clocks)==2
    after['pose'][0]=3.
    assert audit.compare_nonclock(before,after)[0]==['.pose[0]']


def test_nonclock_missing_fields_cannot_be_ignored():
    assert audit.compare_nonclock(dict(updates=1),{})[0]==[':keys']


def test_nonfinite_clock_is_not_a_parity_exemption():
    assert audit.compare_nonclock(dict(process_seconds=.1),dict(process_seconds=float('nan')))[0]


def test_completed_EOF_uses_separate_nonfresh_population():
    row=evaluator.verify_completed_EOF(completed())
    assert row['offline_updates_completed']==12 and row['EOF_counted_as_fresh'] is False


@pytest.mark.parametrize('mutation',('failed','fresh_EOF','missing_Update','context_leak','prior_callback_failure'))
def test_incomplete_or_mislabelled_lifecycle_rejected(mutation):
    row=completed()
    if mutation=='failed':row['status']='failed'
    elif mutation=='fresh_EOF':row['termination']['fresh_pose_valid']=True
    elif mutation=='missing_Update':row['online_metrics']['full_completed']=180
    elif mutation=='context_leak':row['online_metrics']['cache']['current_DMA_bytes']=1
    elif mutation=='prior_callback_failure':row['online_metrics']['calls'][4]['status']='failed_no_partial_fullgraph'
    with pytest.raises(ValueError):evaluator.verify_completed_EOF(row)


def test_actual_v41_failed_schema_is_not_relabelled_as_v42(tmp_path):
    run=dict(schema='v41_cached_undistort_online_RKNN_research_replay',status='failed')
    path=tmp_path/'run.json';path.write_text(json.dumps(run))
    with pytest.raises(ValueError,match='actual run schema differs'):
        evaluator.load_run(tmp_path,evaluator.sha(path),'online_RKNN',None,None,None)


def test_failed_new_schema_has_no_failed_EOF_exception(tmp_path):
    run=dict(schema=evaluator.SCHEMAS['online_RKNN'],status='failed')
    path=tmp_path/'run.json';path.write_text(json.dumps(run))
    with pytest.raises(ValueError,match='no failed-run exception'):
        evaluator.load_run(tmp_path,evaluator.sha(path),'online_RKNN',None,None,None)


def test_old_baseline_schemas_remain_actual_original_schemas():
    assert evaluator.SCHEMAS['cached_half_CPU']=='v41_cached_undistort_original_Torch_CPU_research_replay'
    assert evaluator.SCHEMAS['full_resolution_FAST_CPU']=='v34_DPVO_original_FAST_fresh_frame_CPU_replay'
