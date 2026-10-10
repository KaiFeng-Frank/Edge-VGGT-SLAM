"""Pure saved-geometry tests; never run VGGT, images or a tracker."""
import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import evaluate_edge_trajectory as evaluation


def pose(t):
    angle = np.radians(t * 20)
    T = np.eye(4)
    T[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    T[:3, 3] = [t, .1*t, 0]
    return T


def row(t, gauge=0, anchor=0, scale=1):
    relative = evaluation.inverse(pose(t)) @ pose(anchor)
    relative[:3, 3] /= scale
    return dict(timestamp=t, frame_id=t, sequence=int(t), pose_valid=True,
                historical_replay=False, gauge_id=gauge, anchor_frame_id=anchor,
                T_current_from_anchor=relative.tolist(), status='tracking')


def fixture(tmp_path, scale=1):
    gt = tmp_path / 'groundtruth.txt'
    lines = []
    for t in [0., 1., 2.]:
        a = np.radians(t*20) / 2
        lines.append(f'{t} {t} {.1*t} 0 0 0 {np.sin(a)} {np.cos(a)}')
    gt.write_text('# synthetic camera-to-world poses\n' + '\n'.join(lines) + '\n')
    run = tmp_path / 'saved'; run.mkdir()
    rows = [row(t, scale=scale) for t in [0., 1., 2.]]
    (run / 'tracking.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    (run / 'metrics.json').write_text(json.dumps({'status': 'completed_experiment', 'ORB_used': False,
        'mapper_config': {'backend': 'synthetic_saved_geometry'}, 'source': {'original_frames': 3, 'delivered': 3},
        'pose_timing': {'fresh_valid_poses': 3, 'first_valid_seconds_after_start': 0}}))
    (run / 'events.jsonl').write_text(json.dumps({'kind': 'map_submitted', 'batch': [{'frame_id': 0.}, {'frame_id': 2.}]})+'\n')
    np.savez(run / 'predictions_0001.npz', frame_ids=np.array([0., 2.]),
             extrinsic=np.stack([evaluation.inverse(pose(t))[:3] for t in [0, 2]]),
             intrinsic=np.stack([np.eye(3), np.eye(3)]))
    return argparse.Namespace(run_dir=[run], groundtruth=gt, max_time_difference=.02,
        rpe_time_tolerance=.02, minimum_translation=.01, fit_gauge_scale=False)


def test_TUM_xyzw_normalization_and_optical_pose_direction():
    q = np.array([0, 0, np.sqrt(.5), np.sqrt(.5)])
    expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert np.allclose(evaluation.quaternion_xyzw(q*2), expected)
    assert np.allclose(evaluation.quaternion_xyzw(-q), expected)
    relative = evaluation.inverse(pose(2)) @ pose(0)
    assert evaluation.errors(relative, relative, .01)['rotation_error_degrees'] < 1e-10
    # Reversing current-from-anchor into camera-to-world is observably wrong.
    assert evaluation.errors(evaluation.inverse(relative), relative, .01)['rotation_error_degrees'] > 70


def test_manual_camera_center_and_point_example_fixes_Tcw_direction():
    # World: anchor camera at x=1, identity orientation. Current camera at
    # x=2, rotated +90 degrees around z. This expected transform is written
    # explicitly, independently of the evaluator's inverse/quaternion helpers.
    anchor = np.eye(4); anchor[0, 3] = 1
    current = np.array([[0., -1, 0, 2], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    expected = np.array([[0., 1, 0, 0], [-1, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
    relative = evaluation.inverse(current) @ anchor
    assert np.array_equal(relative, expected)
    # Anchor point [2,1,5] is world [3,1,5], hence current camera [1,-1,5].
    assert np.array_equal(relative @ [2, 1, 5, 1], [1, -1, 5, 1])
    assert np.array_equal(-relative[:3, :3].T @ relative[:3, 3], [1, 0, 0])
    q = [0, 0, np.sqrt(.5), np.sqrt(.5)]
    assert np.allclose(evaluation.quaternion_xyzw(q), current[:3, :3])


def test_rotation_pi_and_nonrigid_rejection():
    assert evaluation.rotation_degrees(np.diag([-1, -1, 1])) == pytest.approx(180)
    bad = np.eye(4); bad[0, 0] = -1
    with pytest.raises(ValueError, match='proper SE3'):
        evaluation.rigid(bad, 'reflection')


def test_translation_direction_positive_scale_invariant_and_sign_sensitive():
    a = np.eye(4); a[:3, 3] = [1, 2, 3]
    b = a.copy(); b[:3, 3] *= 17
    assert evaluation.errors(a, b, .01)['translation_direction_error_degrees'] < 1e-5
    b[:3, 3] *= -1
    assert evaluation.errors(a, b, .01)['translation_direction_error_degrees'] == pytest.approx(180)


def test_zero_or_small_translation_is_unknown_not_perfect():
    result = evaluation.errors(np.eye(4), np.eye(4), .01)
    assert result['translation_direction_error_degrees'] is None
    assert result['translation_direction_unavailable_reason'] == 'GT_translation_below_diagnostic_minimum'


def test_nearest_GT_tie_and_twenty_ms_gate():
    times = np.array([0., 1., 2.])
    assert evaluation.nearest(times, .5, .5) == 0
    assert evaluation.nearest(times, 1.019, .02) == 1
    assert evaluation.nearest(times, 1.0201, .02) is None


def test_one_second_RPE_never_crosses_gauge():
    rows = [row(0), row(1), row(2, gauge=1, anchor=2), row(3, gauge=1, anchor=2)]
    pairs, counts = evaluation.fixed_rpe_pairs(rows, .02)
    assert [(a['timestamp'], b['timestamp']) for a, b in pairs] == [(0, 1), (2, 3)]
    assert counts['same_gauge_pairs'] == 2


def test_GT_duplicate_timestamp_and_zero_quaternion_rejected(tmp_path):
    path = tmp_path / 'gt.txt'
    path.write_text('0 0 0 0 0 0 0 1\n0 0 0 0 0 0 0 1\n')
    with pytest.raises(ValueError, match='increase strictly'):
        evaluation.read_gt(path)
    path.write_text('0 0 0 0 0 0 0 0\n1 0 0 0 0 0 0 1\n')
    with pytest.raises(ValueError, match='nonzero'):
        evaluation.read_gt(path)


@pytest.mark.parametrize('change', ['invalid_nonnull', 'historical', 'changed_anchor'])
def test_tracking_identity_and_missing_pose_guards(tmp_path, change):
    rows = [row(0), row(1)]
    if change == 'invalid_nonnull': rows[0]['pose_valid'] = False
    elif change == 'historical': rows[0]['historical_replay'] = True
    else: rows[1]['anchor_frame_id'] = .5
    path = tmp_path / 'tracking.jsonl'; path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    with pytest.raises(ValueError): evaluation.read_tracking(path)


def test_saved_geometry_full_evaluation_has_zero_error_and_input_provenance(tmp_path):
    result = evaluation.evaluate(fixture(tmp_path))
    run = result['runs'][0]
    assert result['model_inference_executed'] is False
    assert result['metric_pose_validated'] is False
    assert run['coverage']['fresh_valid_local_poses'] == 3
    assert run['local_anchor_relative']['rotation_error_degrees']['max'] < 1e-10
    assert run['fixed_one_second_RPE']['samples'] == 2
    assert run['fixed_one_second_RPE']['rotation_error_degrees']['max'] < 1e-10
    assert run['saved_model_two_frame_relative'][0]['diagnostic']['rotation_error_degrees'] < 1e-10
    assert run['source_and_inputs_unchanged'] is True
    assert run['input_sha256']


def test_optional_GT_scale_is_positive_evaluation_only(tmp_path):
    args = fixture(tmp_path, scale=2); args.fit_gauge_scale = True
    result = evaluation.evaluate(args)
    scale = result['runs'][0]['local_anchor_relative']['per_gauge'][0]['GT_fitted_scale_evaluation_only']
    assert scale['positive_scale'] == pytest.approx(2)
    assert scale['fit_and_score_use_same_samples'] is True
    assert scale['anchor_translation_residual_m']['max'] < 1e-10
    assert result['GT_used_in_model_or_tracking'] is False


def test_full_coverage_retains_missing_pose(tmp_path):
    args = fixture(tmp_path)
    path = args.run_dir[0] / 'tracking.jsonl'
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    rows[1].update(pose_valid=False, T_current_from_anchor=None, status='lost')
    path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    p = args.run_dir[0] / 'metrics.json'; metadata = json.loads(p.read_text())
    metadata['pose_timing']['fresh_valid_poses'] = 2; p.write_text(json.dumps(metadata))
    run = evaluation.evaluate(args)['runs'][0]
    assert run['coverage']['valid_fraction_of_source'] == pytest.approx(2/3)
    assert run['fixed_one_second_RPE']['samples'] == 0


def test_unbound_prediction_batch_rejected(tmp_path):
    args = fixture(tmp_path)
    (args.run_dir[0] / 'events.jsonl').write_text('')
    with pytest.raises(ValueError, match='exact actual submitted batch'):
        evaluation.evaluate(args)


def test_tolerance_cannot_be_relaxed(tmp_path):
    args = fixture(tmp_path); args.max_time_difference = .021
    with pytest.raises(ValueError, match='<=20ms'): evaluation.evaluate(args)
