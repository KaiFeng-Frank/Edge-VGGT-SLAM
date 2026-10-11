"""Tiny topology/SE3 fixtures derived from pinned official DPVO, no neural execution.

Source: princeton-vl/DPVO@0ac95b656d1fda91c271d2a106460d19ad966fc7.
Upstream MIT notice is preserved in lifecycle-oracle.LICENSE alongside this file.
This is an audit fixture, not a streaming runtime or a benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def appended_edges(n: int, patches_per_frame: int, lifetime: int = 13):
    """Official forward append followed by backward append, ij meshgrid order."""
    def product(kk, jj):
        return np.repeat(kk, len(jj)), np.tile(jj, len(kk))
    m = patches_per_frame
    fk, fj = product(np.arange(m * max(n - lifetime, 0), m * (n - 1)),
                     np.arange(n - 1, n))
    bk, bj = product(np.arange(m * (n - 1), m * n),
                     np.arange(max(n - lifetime, 0), n))
    kk, jj = np.r_[fk, bk], np.r_[fj, bj]
    return kk // m, jj, kk


def run_checks():
    checks = []
    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append({"name": name, "passed": True})

    ii, jj, kk = appended_edges(2, 2)
    check("forward_then_backward_patch_outer_target_inner",
          list(zip(ii, jj, kk)) == [(0, 1, 0), (0, 1, 1), (1, 0, 2),
                                  (1, 1, 2), (1, 0, 3), (1, 1, 3)])
    edges = [appended_edges(n, 96) for n in range(1, 9)]
    ii, jj, kk = (np.concatenate([e[k] for e in edges]) for k in range(3))
    check("eight_admitted_frames_have_6144_edges_including_768_self_edges",
          len(kk) == 6144 and np.count_nonzero(ii == jj) == 768)
    check("initialization_topology_has_every_patch_target_pair_once",
          set(zip(kk, jj)) == {(k, j) for k in range(768) for j in range(8)}
          and len(set(zip(kk, jj))) == len(kk))
    check("initialization_append_counts",
          [len(e[0]) for e in edges] == [96, 288, 480, 672, 864, 1056, 1248, 1440])

    ii, jj, kk = appended_edges(14, 2)
    check("lifetime13_window_uses_slots1_through13",
          ii.min() == 1 and jj.min() == 1 and ii.max() == jj.max() == 13
          and len(kk) == 50)
    old = np.arange(12).reshape(3, 4)
    new = np.r_[old, np.zeros((len(kk), 4), dtype=old.dtype)]
    check("append_preserves_old_recurrent_rows_and_zeros_new_rows",
          np.array_equal(new[:3], old) and not new[3:].any())

    all_edges = [appended_edges(n, 2) for n in range(1, 10)]
    ii, jj, kk = (np.concatenate([e[k] for e in all_edges]) for k in range(3))
    k = 5
    keep = (ii != k) & (jj != k)
    ii, jj, kk = ii[keep], jj[keep], kk[keep]
    kk = kk - 2 * (ii > k)
    ii, jj = ii - (ii > k), jj - (jj > k)
    source_ids = np.delete(np.arange(9), k)
    check("middle_delete_remaps_patch_source_target_and_preserves_source_identity",
          np.array_equal(ii, kk // 2) and source_ids[-1] == 8
          and set(zip(kk, jj)) == {(p, j) for p in range(16) for j in range(8)})
    ages = np.array([7, 8, 9])
    check("inactive_source_age_removal_is_strict_less_than_n_minus22",
          (ages < (30 - 22)).tolist() == [True, False, False])

    values = np.array([1., 2., 3., 4.])
    check("depth_torch_median_is_lower_middle_not_numpy_median",
          np.sort(values)[(len(values)-1)//2] == 2. and np.median(values) == 2.5)
    observed_times = [0., 1., 4., 5.]  # observation 1 was rejected; 0 and 4 retained
    a, b, c = observed_times[-3:]
    fac = (c-b)/(b-a)
    check("motion_dt_uses_all_observations_even_rejected",
          fac == 1/3 and fac != (5.-4.)/(4.-0.))

    def pose(angle, translation):
        c, s = np.cos(angle), np.sin(angle)
        out = np.eye(4)
        out[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        out[:3, 3] = translation
        return out
    parent = pose(.7, [.3, -.2, .1])
    deleted = pose(-.4, [-.1, .4, .2])
    relative = deleted @ np.linalg.inv(parent)
    check("deleted_world_to_camera_reconstruction_left_multiplies_parent",
          np.allclose(relative @ parent, deleted, rtol=0, atol=1e-15)
          and not np.allclose(parent @ relative, deleted))
    check("translation_only_flow_retains_relative_translation_after_pose_composition",
          np.allclose(relative[:3, 3], deleted[:3, 3] - relative[:3, :3] @ parent[:3, 3])
          and not np.allclose(relative[:3, 3], deleted[:3, 3] - parent[:3, 3]))

    return {"status": "passed_tiny_synthetic_only", "checks": checks,
            "check_count": len(checks), "model_forward_calls": 0,
            "real_input_used": False, "board_jobs": 0,
            "scope": "Integer topology, median convention, timestamp choice and tiny SE3 matrix algebra only."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_checks()
    result["fixture_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps({"status": result["status"], "checks": result["check_count"]}))


if __name__ == "__main__":
    main()
