"""Tiny generated JSON/numeric cases; no dataset/GT/model/backend execution."""
import copy
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
PUBLIC = HERE.parent if (HERE.parent / 'deploy').is_dir() else HERE.parent / 'public-snapshot'
EVALUATOR = HERE / 'evaluate-v41-online.py'
if not EVALUATOR.is_file():
    EVALUATOR = PUBLIC / 'deploy/rk3588/validation/dpvo_online_update/evaluate-v41-online.py'
spec = importlib.util.spec_from_file_location('_v41_saved_evaluator_test', EVALUATOR)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
v39_path = Path(os.environ.get('V41_V39_EVALUATOR',
    PUBLIC / 'deploy/rk3588/validation/dpvo_half_workload/evaluate-v39-workload.py'))
math_path = Path(os.environ.get('V41_MATH_HELPER',
    PUBLIC / 'deploy/rk3588/validation/dpvo_fast/evaluate-v34-fast-trajectory.py'))
v39, math = evaluation.load_frozen(v39_path, math_path)


def toy():
    source = [{'frame_id': 'toy-frame', 'timestamp': 1.0}]
    result = dict(observation_id=0, frame_id='toy-frame', timestamp=1.0,
                  captured_monotonic=10.0, fresh_pose_valid=True, admitted=True,
                  initialized=True, updates=1, reason='updated_current_observation',
                  pose_w2c=[1., 2., 3., 0., 0., 0., 1.])
    row = dict(index=0, frame_id='toy-frame', timestamp=1.0, result=result,
               arrival_monotonic=10.0, processed_monotonic=10.25,
               pose_available_monotonic=10.5, processing_seconds=.25,
               admission_to_publication_seconds=.5, publication_seconds=.25)
    run = dict(frames_completed=1, frames=[row],
               metrics_before_termination=dict(observations=1, fresh_poses=1))
    return run, source


def test_generated_fresh_clock_includes_publication():
    run, source = toy()
    poses, processing, publication = v39.fresh_rows(run, source, math, half=True)
    assert processing == [.25] and publication == [.5]
    np.testing.assert_array_equal(poses[0][:3, 3], [1., 2., 3.])


def test_stale_observation_cannot_be_fresh():
    run, source = toy()
    run['frames'][0]['result']['observation_id'] = 1
    with pytest.raises(ValueError, match='identity'):
        v39.fresh_rows(run, source, math, half=True)


def test_nonfresh_pose_is_rejected():
    run, source = toy()
    run['frames'][0]['result']['fresh_pose_valid'] = False
    with pytest.raises(ValueError, match='non-fresh'):
        v39.fresh_rows(run, source, math, half=True)


def test_eof_cannot_count_as_fresh():
    data = {'run': {'termination': {'fresh_pose_valid': True}}}
    with pytest.raises(ValueError, match='EOF'):
        v39.evaluate_eof(data, math, {}, np.empty((0, 4, 4)), [])


def test_cpu_schema_cannot_masquerade_as_rknn(tmp_path):
    run = dict(schema=evaluation.SCHEMAS['cached_half_CPU'])
    path = tmp_path / 'run.json'
    path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match='actual run schema'):
        evaluation.load_run(tmp_path, evaluation.sha(path), 'online_RKNN', {}, v39, math)
