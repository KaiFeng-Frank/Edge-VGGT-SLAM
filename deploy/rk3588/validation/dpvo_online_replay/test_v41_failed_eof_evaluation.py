"""Generated JSON tests for the narrowly opted-in failed EOF boundary."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
path = HERE / 'evaluate-v41-online.py'
if not path.is_file():
    path = HERE.parent / 'deploy/rk3588/validation/dpvo_online_update/evaluate-v41-online.py'
spec = importlib.util.spec_from_file_location('_v41_failed_eof_evaluator_test', path)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def generated_failure():
    error = "ValueError('Unknown complete online shape(s): B1_D3_M16')"
    before = dict(status='running', failure=None, observations=192, fresh_poses=158,
                  update_calls=169, BA_calls=169, BA_iterations=338, offline_updates=0,
                  admitted=165, motion_skipped=27, motion_probe_calls=34)
    after = dict(before, status='failed', update_calls=170,
                 failure=dict(phase='offline_termination', type='OnlineUpdateError', reason=error))
    calls = [dict(index=i, route='motion_probe' if i < 34 else 'full',
                  status='completed_diagnostic_not_adopted') for i in range(204)]
    calls[-1].update(status='failed_no_partial_fullgraph', error=error,
                    fullgraph_returned=False, outputs_committed_by_callback=False,
                    degree_plan={'batches': [dict(sources=1, degree=3, patches=16)]},
                    model_keys=['B1_D3_M16'])
    final = dict(requests=204, full_attempts=170, full_completed=169, failed_calls=1,
                 motion_probe_attempts=34, motion_probe_completed=34, failed=True, closed=True,
                 calls=calls, model_keys=[f'B{b}_D{d}_M16' for b in [1, 2] for d in [4, 5, 6, 7]],
                 cache=dict(closed=True, cleanup_errors=[], resident_contexts=0,
                            current_model_host_bytes=0, current_DMA_bytes=0))
    preclose = copy.deepcopy(final)
    preclose['closed'] = False
    frames = [dict(result=dict(updates=12 if i == 34 else int(i > 34))) for i in range(192)]
    return dict(status='failed', active_stage='termination_offline', frames_requested=192,
                frames_completed=192, frames=frames, cleanup_errors=[], termination=None,
                metrics_before_termination=before, stream_metrics_at_exit=after,
                online_metrics=final, online_metrics_before_close=preclose,
                candidate_accepted=False, historical_SDK_strict_FP32_rejected=True,
                SDK_FP32_equivalence_validated=False, FP32_thresholds_relaxed=False,
                complete_Update_Torch_fallback=False)


def test_generated_failure_keeps_overall_failure_and_no_eof_pose():
    result = evaluation.verify_failed_eof_only(generated_failure())
    assert result['overall_run_success'] is False
    assert result['state'] == 'failed_not_completed'
    assert result['counted_as_fresh'] is False
    assert result['full_online_updates_completed'] == 169
    assert result['EOF_trajectory_evaluation_available'] is False


def test_default_still_rejects_failed_run(tmp_path):
    run = generated_failure()
    run['schema'] = evaluation.SCHEMAS['online_RKNN']
    path = tmp_path / 'run.json'
    path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match='explicit opt-in'):
        evaluation.load_run(tmp_path, evaluation.sha(path), 'online_RKNN', {}, None, None)


@pytest.mark.parametrize('field,value', [
    ('active_stage', 'stream'), ('frames_completed', 191), ('termination', {'fake': True}),
])
def test_earlier_or_incomplete_failure_is_not_admitted(field, value):
    run = generated_failure()
    run[field] = value
    with pytest.raises(ValueError):
        evaluation.verify_failed_eof_only(run)


def test_hidden_prior_callback_failure_is_not_admitted():
    run = generated_failure()
    run['online_metrics']['calls'][10]['status'] = 'failed_no_partial_fullgraph'
    run['online_metrics_before_close']['calls'][10]['status'] = 'failed_no_partial_fullgraph'
    with pytest.raises(ValueError, match='prior online callback'):
        evaluation.verify_failed_eof_only(run)


def test_additional_failure_or_missing_online_call_is_not_admitted():
    for field, value in [('failed_calls', 2), ('full_completed', 168)]:
        run = generated_failure()
        run['online_metrics'][field] = value
        with pytest.raises(ValueError, match='completion/failure counters'):
            evaluation.verify_failed_eof_only(run)


def test_source_plan_ignores_description_but_rejects_changed_image_identity():
    original = dict(source=[dict(frame_id='toy', timestamp=1., sha256='a')],
                    schedule=dict(schema='new', dataset='toy', rgb_index_sha256='b', frame_selection='new description'))
    reference = copy.deepcopy(original)
    reference['schedule'].update(schema='old', frame_selection='old description')
    data = dict(online_RKNN=original, full_resolution_FAST_CPU=reference)
    evaluation.verify_same_source_plan(data)
    reference['source'][0]['sha256'] = 'changed'
    with pytest.raises(ValueError, match='image pins'):
        evaluation.verify_same_source_plan(data)
