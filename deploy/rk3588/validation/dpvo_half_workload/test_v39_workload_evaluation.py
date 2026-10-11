"""Generated pose/timing/ratio checks only, without dataset GT or inference."""
import copy
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

SOURCE = Path(__file__).with_name('evaluate-v39-workload.py')
spec = importlib.util.spec_from_file_location('v39_evaluation_tested', SOURCE)
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)


def example():
    source = [{'frame_id': 'synthetic0', 'timestamp': 100.}]
    result = dict(observation_id=0, frame_id='synthetic0', timestamp=100.,
                  captured_monotonic=10., fresh_pose_valid=True, admitted=True,
                  initialized=True, updates=1, reason='updated_current_observation',
                  pose_w2c=[0., 0., 0., 0., 0., 0., 1.])
    row = dict(index=0, frame_id='synthetic0', timestamp=100.,
               arrival_monotonic=10., processed_monotonic=10.8,
               pose_available_monotonic=11., processing_seconds=.8,
               admission_to_publication_seconds=1., publication_seconds=.2,
               result=result)
    run = dict(frames=[row], frames_completed=1,
               metrics_before_termination=dict(observations=1, fresh_poses=1))
    math = SimpleNamespace(matrices=lambda pose: np.eye(4)[None])
    return run, source, math


def test_half_publication_includes_flush_and_clock_tamper_rejects():
    run, source, math = example()
    poses, processing, publication = evaluate.fresh_rows(run, source, math, half=True)
    assert len(poses) == 1 and processing == pytest.approx([.8])
    assert publication == [1.]
    run['frames'][0]['publication_seconds'] = 0.
    with pytest.raises(ValueError, match='Publication'):
        evaluate.fresh_rows(run, source, math, half=True)


@pytest.mark.parametrize('field,value', [('initialized', False), ('updates', 0),
                                       ('reason', 'EOF_reconstructed')])
def test_wait_or_EOF_cannot_be_labeled_fresh(field, value):
    run, source, math = example()
    run['frames'][0]['result'][field] = value
    with pytest.raises(ValueError, match='labeled fresh'):
        evaluate.fresh_rows(run, source, math, half=True)


def test_ratios_preserve_zero_baseline_and_different_population_as_unavailable():
    names = ('ATE_sim3_aligned_meters', 'RPE_rotation_degrees',
             'RPE_translation_GT_scale_meters',
             'RPE_translation_direction_degrees_nonzero_vectors')
    reference = {key: dict(count=3, mean=2., median=2., p95=2., rmse=2.) for key in names}
    candidate = copy.deepcopy(reference)
    for key in names:
        candidate[key].update(mean=3., median=3., p95=3., rmse=3.)
    ratios = evaluate.ratios(reference, candidate)
    assert ratios[names[0]]['statistics']['rmse']['candidate_over_baseline'] == 1.5
    assert ratios[names[0]]['statistics']['rmse']['error_change_percent'] == 50.
    reference[names[0]]['rmse'] = 0.
    candidate[names[-1]]['count'] = 2
    ratios = evaluate.ratios(reference, candidate)
    assert ratios[names[0]]['statistics']['rmse']['candidate_over_baseline'] is None
    assert ratios[names[-1]]['statistics']['median']['candidate_over_baseline'] is None


def test_frozen_math_common_plan_uses_same_generated_GT_endpoints():
    helper = os.environ.get('EDGE_DPVO_EVALUATOR_HELPER')
    assert helper, 'Set EDGE_DPVO_EVALUATOR_HELPER to the frozen v34 helper; no download occurs'
    math = evaluate.load_math(helper)
    source_stamps = np.arange(6, dtype=np.float64)
    gt_pose = np.zeros((6, 7));gt_pose[:, 6] = 1.
    gt_pose[:, 0] = source_stamps
    gt_pose[:, 1] = .1 * source_stamps**2
    gt = math.matrices(gt_pose)
    poses = {i: math.inverse(gt[i]) for i in range(6)}
    candidate = {i: value for i, value in poses.items() if i != 2}
    plan = math.endpoint_plan(source_stamps)
    common = math.common_fresh_analysis(poses, candidate, dict(enumerate(range(6))),
                  gt, plan, [str(i) for i in range(6)])
    assert common['common_fresh_source_indices'] == [0, 1, 3, 4, 5]
    assert common['fixed_all_source_plan_pairs'] == 5
    assert common['default']['used_endpoint_pairs'] == common['FAST']['used_endpoint_pairs']
    assert common['common_endpoint_pairs'] == [[0, 1], [3, 4], [4, 5]]
    assert common['EOF_poses_included'] is False
