#!/usr/bin/env python3
"""Tiny camera/topology checks only; no images, model, solver, native or SDK.

The discrete edge rules are derived from Princeton-VL/DPVO (MIT), pinned
commit 0ac95b656d1fda91c271d2a106460d19ad966fc7. These checks do not predict
learned motion, keyframe deletion, numerical accuracy or execution speed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append(pairs, n, lifetime):
    first = max(n - lifetime, 0)
    return pairs + [(i, n - 1) for i in range(first, n - 1)] + [
        (n - 1, j) for j in range(first, n)]


def simulate(removal, deletion, observations=40):
    """Tiny frame-pair graph; deletion policy is generated, never a pose result."""
    n, pairs, rows = 0, [], []
    for observation in range(observations):
        n += 1
        pairs = append(pairs, n, 4)
        assert len(set(pairs)) == len(pairs)
        if n >= 8:
            invalid = [(i, j) for i, j in pairs if i < n - removal - 1]
            if invalid:
                return rows, dict(observation=observation, n=n, offending=invalid)
        if observation == 7:
            assert n == 8 and len(pairs) == 44
        if observation >= 8:
            i, j, removed = n - 3, n - 1, n - 2
            assert (i, j) in pairs and (j, i) in pairs
            if deletion(observation):
                pairs = [(a - (a > removed), b - (b > removed))
                         for a, b in pairs if a != removed and b != removed]
                n -= 1
            pairs = [(a, b) for a, b in pairs if a >= n - removal]
        assert len(set(pairs)) == len(pairs)
        # Every retained frame-pair means all M patches; no source/target group
        # is truncated by the plan. Synthetic IDs allow checking E ordering.
        edges = [(i, j, i * 16 + k) for i, j in pairs for k in range(16)]
        assert all(i == k // 16 for i, j, k in edges)
        for i, j in pairs:
            assert {k % 16 for a, b, k in edges if (a, b) == (i, j)} == set(range(16))
        rows.append(dict(observation=observation, retained=n,
                         frame_pairs=len(pairs), edges=len(edges)))
    return rows, None


def run(policy):
    assert policy['stream_config']['patches_per_frame'] == 16
    assert policy['stream_config']['patch_lifetime'] == 4
    assert policy['stream_config']['removal_window'] == 8
    assert policy['stream_config']['keyframe_index'] == 2
    assert policy['image_size_hw'] == [232, 304]
    K = np.array([[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1.]])
    crop = np.array([[1., 0, -16], [0, 1, -8], [0, 0, 1]])
    half = np.array([[.5, 0, -.25], [0, .5, -.25], [0, 0, 1]])
    expected = half @ crop @ K
    fx, fy, cx, cy = policy['intrinsics']
    declared = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
    assert np.allclose(expected, declared, rtol=0, atol=3e-14)
    rays = np.array([[.2, -.1, 1.], [-.4, .3, 2.], [0., 0., 1.]])
    raw = rays @ K.T
    raw /= raw[:, 2:]
    resized = raw @ crop.T @ half.T
    actual = rays @ declared.T
    actual /= actual[:, 2:]
    assert np.allclose(actual, resized, rtol=0, atol=3e-14)
    naive = crop @ K
    naive[:2] *= .5
    naive_pixels = rays @ naive.T
    naive_pixels /= naive_pixels[:, 2:]
    assert np.allclose(naive_pixels[:, :2] - actual[:, :2], .25, atol=3e-14)
    failures = {}
    for removal, expected_observation in ((6, 7), (7, 8)):
        rows, error = simulate(removal, lambda i: False)
        assert error is not None and error['observation'] == expected_observation
        assert error['offending'][0][0] == 0
        failures[str(removal)] = error
    simulations = {}
    for name, deletes in [('no_keyframe_deletion', lambda i: False),
                          ('delete_each_postinit', lambda i: True),
                          ('generated_alternating', lambda i: i % 2 == 0)]:
        rows, error = simulate(8, deletes)
        assert error is None and len(rows) == 40
        assert rows[7]['edges'] == 704
        simulations[name] = {'observations': len(rows), 'initial_edges': 704,
                             'last': rows[-1]}
    assert simulations['no_keyframe_deletion']['last']['edges'] == 800
    return dict(status='passed_tiny_camera_and_discrete_topology_checks',
                check_groups=8, failed_checks=0,
                halfpixel_naive_principal_point_error_pixels=.25,
                rejected_removal_boundary_cases=failures,
                legal_R8_generated_topologies=simulations,
                limits='Generated topology only: no actual flow/Update/BA/RNG/image/model evaluation; R8 legality is not trajectory accuracy or latency validation.',
                model_forward_calls=0, ORT_calls=0, SDK_calls=0, BA_calls=0,
                native_calls=0, board_jobs=0, real_images_read=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--policy-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert sha(args.policy) == args.policy_sha256
    report = run(json.loads(args.policy.read_text()))
    report.update(script_sha256=sha(__file__), policy_sha256=sha(args.policy))
    assert report['policy_sha256'] == args.policy_sha256
    assert not args.output.exists()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
