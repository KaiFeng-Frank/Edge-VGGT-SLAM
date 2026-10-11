#!/usr/bin/env python3
"""Generated geometry only; no dataset replay or model execution."""
import json
from pathlib import Path
import tempfile

import numpy as np

from evaluate_original_trajectory import (associate_timestamps, evaluate_tum, fit_sim3,
                                         poses_from_tum, read_tum, relative_pose_errors,
                                         validated_rows)


def yaw_rows(points, angles=None, times=None):
    count = len(points)
    angles = np.zeros(count) if angles is None else np.asarray(angles)
    times = np.arange(count, dtype=np.float64) if times is None else np.asarray(times)
    return np.column_stack((times, points, np.zeros((count, 2)), np.sin(angles/2), np.cos(angles/2)))


def must_reject(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError('Generated invalid geometry was accepted')


def main():
    checks = []
    points = np.array([[0.,0.,0.], [1.,0.,0.], [1.,1.,0.], [0.,1.,1.], [0.,0.,1.]])
    angles = np.array([0., .1, .2, .3, .4])
    ground = yaw_rows(points, angles)
    identity = evaluate_tum(ground, ground)
    for key in ('ate_translation', 'rpe_translation', 'rpe_rotation'):
        assert identity[key]['rmse'] < 1e-12
    assert identity['ate_translation']['count'] == 5 and identity['rpe_translation']['count'] == 4
    checks.append('identical generated poses: zero ATE and RPE, exact denominators')

    angle, scale, translation = .47, 2.75, np.array([.4, -2., .9])
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0.,0.,1.]])
    estimated_points = (points-translation) @ rotation / scale
    estimate = yaw_rows(estimated_points, angles-angle)
    aligned = evaluate_tum(estimate, ground)
    np.testing.assert_allclose(aligned['alignment']['scale'], scale, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(aligned['alignment']['rotation'], rotation, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(aligned['alignment']['translation'], translation, rtol=1e-12, atol=1e-12)
    for key in ('ate_translation', 'rpe_translation', 'rpe_rotation'):
        assert aligned[key]['rmse'] < 1e-11
    checks.append('known nontrivial positive Sim3 recovered, orientations transformed, shared RPE scale')

    drifted = yaw_rows(points, np.arange(5)*.1)
    reference = yaw_rows(points)
    drift = evaluate_tum(drifted, reference)
    assert drift['ate_translation']['rmse'] < 1e-12
    np.testing.assert_allclose(drift['rpe_rotation']['rmse'], np.degrees(.1), rtol=1e-12)
    checks.append('known 0.1 rad relative yaw error with zero positional ATE')

    gp, ep = poses_from_tum(reference), poses_from_tum(drifted)
    actual_translation, actual_angle = relative_pose_errors(ep, gp, delta_frames=2)
    expected_translation, expected_angle = [], []
    for i in range(3):
        error = np.linalg.inv(np.linalg.inv(gp[i]) @ gp[i+2]) @ (np.linalg.inv(ep[i]) @ ep[i+2])
        expected_translation.append(np.linalg.norm(error[:3,3]))
        expected_angle.append(np.arccos(np.clip((np.trace(error[:3,:3])-1.)/2., -1., 1.)))
    np.testing.assert_allclose(actual_translation, expected_translation, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_angle, expected_angle, rtol=1e-12, atol=1e-12)
    checks.append('RPE checked independently against full 4x4 inverses, delta=2')

    half_turn = ep.copy()
    half_turn[1:,:3,:3] = np.diag([-1.,-1.,1.])
    _, half_angle = relative_pose_errors(half_turn, gp)
    np.testing.assert_allclose(half_angle[0], np.pi, atol=1e-12)
    checks.append('SO3 half-turn boundary gives pi radians')

    offset_rows = ground.copy()
    offset_rows[:,0] -= .011
    ei, gi = associate_timestamps(offset_rows, ground, offset=.011, max_difference=.001)
    np.testing.assert_array_equal(ei, np.arange(5))
    np.testing.assert_array_equal(gi, np.arange(5))
    extra = np.insert(offset_rows, 2, yaw_rows([[5.,5.,5.]], times=[1.5])[0], axis=0)
    # The inserted timestamp lies between 0.989 and 1.989 and remains unmatched.
    ei, gi = associate_timestamps(extra, ground, offset=.011, max_difference=.001)
    assert len(ei) == 5 and len(set(gi)) == 5 and 2 not in ei
    metrics = evaluate_tum(extra, ground, offset=.011, max_difference=.001)
    assert metrics['association']['unmatched_estimated_count'] == 1
    np.testing.assert_allclose(metrics['association']['estimated_matched_fraction'], 5/6)
    checks.append('timestamp offset, one-to-one matches and dropped-estimate coverage')

    mirrored = points.copy(); mirrored[:,0] *= -1.
    fitted_scale, fitted_rotation, _ = fit_sim3(mirrored, points)
    assert fitted_scale > 0 and np.linalg.det(fitted_rotation) > .999999999
    checks.append('reflection excluded from Sim3 rotation')

    must_reject(fit_sim3, np.zeros((5,3)), points)
    must_reject(fit_sim3, np.column_stack((np.arange(5), np.zeros((5,2)))), points)
    bad = ground.copy(); bad[1,0] = bad[0,0]
    must_reject(validated_rows, bad)
    bad = ground.copy(); bad[0,4:8] = 0.
    must_reject(validated_rows, bad)
    bad = ground.copy(); bad[0,1] = np.nan
    must_reject(validated_rows, bad)
    must_reject(associate_timestamps, ground, ground+np.array([10.,0.,0.,0.,0.,0.,0.,0.]))
    must_reject(relative_pose_errors, ep, gp, delta_frames=0)
    checks.append('degenerate fits, invalid rows, insufficient matches and invalid RPE interval rejected')

    with tempfile.TemporaryDirectory(prefix='generated-tum-metrics-') as temporary:
        path = Path(temporary)/'generated.tum'
        path.write_text('# Generated fixture\n'+'\n'.join(' '.join(map(str,row)) for row in estimate)+'\n')
        np.testing.assert_allclose(read_tum(path), estimate)
        path.write_text('1 2 3\n')
        must_reject(read_tum, path)
    checks.append('TUM text format and camera-to-world quaternion conversion')

    print(json.dumps(dict(status='passed', generated_check_count=len(checks), checks=checks,
                         scope='generated geometry only; no model, SDK, board or real dataset replay'), indent=2))


if __name__ == '__main__':
    main()
