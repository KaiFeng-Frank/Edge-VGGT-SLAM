"""Generated JSON/RNG lifecycle guards; no datasets, inference, SDK or BA."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

TEST = Path(__file__).resolve().parent
HERE = TEST if (TEST/'evaluate-v45-online.py').is_file() else Path(os.environ.get(
    'V45_ONLINE_SOURCE', TEST.parent/'deploy/rk3588/validation/dpvo_online_replay'))


def load(name):
    spec = importlib.util.spec_from_file_location('_v45_saved_toy_'+name.replace('-', '_'), HERE/name)
    value = importlib.util.module_from_spec(spec); sys.modules[spec.name] = value
    spec.loader.exec_module(value); return value


audit = load('audit-v45-online-parity.py')
evaluator = load('evaluate-v45-online.py')


def rng():
    return dict(random_generator_after=dict(shape=[5056], dtype='uint8', bytes=5056,
        tensor_sha256='a'*64), random_depth_draw_calls=192)


def completed(schema=None):
    before = dict(status='running', failure=None, offline_updates=0, observations=192,
        admitted=165, motion_skipped=27, fresh_poses=158, motion_probe_calls=34,
        update_calls=169, BA_calls=169, BA_iterations=338)
    after = dict(before, status='terminated', offline_updates=12, update_calls=181,
        BA_calls=181, BA_iterations=362)
    state = dict(closed=True, failed=False, failed_calls=0, full_attempts=181,
        full_completed=181, motion_probe_attempts=34, motion_probe_completed=34,
        requests=215, calls=[dict(index=i, status='completed_diagnostic_not_adopted') for i in range(215)],
        cache=dict(closed=True, cleanup_errors=[], resident_contexts=0,
            current_DMA_bytes=0, current_model_host_bytes=0))
    return dict(schema=schema or evaluator.SCHEMAS['online_RKNN'],
        status='completed_unpaced_half_online_RKNN_research_not_realtime',
        frames_requested=192, frames_completed=192, frames=[{} for _ in range(192)],
        cleanup_errors=[], metrics_before_termination=before, stream_metrics=after,
        online_metrics=state, termination=dict(offline_updates=12, fresh_pose_valid=False,
            online_publication=False, reconstructed_poses_counted_as_fresh=False,
            initialized=True, trajectory_state='initialized_offline_reconstruction'),
        candidate_accepted=False, historical_SDK_strict_FP32_rejected=True,
        SDK_FP32_equivalence_validated=False, FP32_thresholds_relaxed=False,
        complete_Update_Torch_fallback=False)


def test_nonclock_allowlist_never_hides_geometry_changes():
    left = dict(captured_monotonic=1., process_seconds=.1, pose=[1., 2.], updates=1)
    right = dict(left, captured_monotonic=2., process_seconds=.2)
    mismatches, clocks = audit.compare_nonclock(left, right)
    assert not mismatches and len(clocks) == 2
    right['pose'] = [3., 2.]
    assert audit.compare_nonclock(left, right)[0] == ['.pose[0]']
    assert audit.compare_nonclock(dict(updates=1), {})[0] == [':keys']


def test_nonfinite_or_negative_clock_is_not_exempt():
    for invalid in (float('nan'), float('inf'), -1.):
        assert audit.compare_nonclock(dict(process_seconds=.1), dict(process_seconds=invalid))[0]


def test_final_RNG_equality_is_metadata_only_and_changed_digest_is_not_equal():
    left = rng(); right = copy.deepcopy(left)
    result = audit.verify_final_RNG(left, right)
    assert result['same_metadata_and_digest'] and result['raw_RNG_bytes_reread'] is False
    right['random_generator_after']['tensor_sha256'] = 'b'*64
    assert audit.verify_final_RNG(left, right)['same_metadata_and_digest'] is False


@pytest.mark.parametrize('mutation', ('missing', 'dtype', 'digest', 'draws'))
def test_invalid_final_RNG_descriptor_or_draw_count_rejected(mutation):
    right = rng()
    if mutation == 'missing': del right['random_generator_after']
    elif mutation == 'dtype': right['random_generator_after']['dtype'] = 'float32'
    elif mutation == 'digest': right['random_generator_after']['tensor_sha256'] = 'g'*64
    else: right['random_depth_draw_calls'] = 191
    with pytest.raises(ValueError): audit.verify_final_RNG(rng(), right)


@pytest.mark.parametrize('label', ('online_RKNN', 'previous_online_RKNN'))
def test_both_actual_completed_schemas_keep_EOF_separate(label):
    result = evaluator.verify_completed_EOF(completed(evaluator.SCHEMAS[label]))
    assert result['offline_updates_completed'] == 12 and result['EOF_counted_as_fresh'] is False
    assert result['full_Update_calls'] == 181


def test_fresh_EOF_or_adoption_or_context_leak_is_rejected():
    for mutation in ('fresh_EOF', 'adoption', 'context_leak'):
        row = completed()
        if mutation == 'fresh_EOF': row['termination']['fresh_pose_valid'] = True
        elif mutation == 'adoption': row['candidate_accepted'] = True
        else: row['online_metrics']['cache']['current_DMA_bytes'] = 1
        with pytest.raises(ValueError): evaluator.verify_completed_EOF(row)


def test_old_callback_failure_cannot_be_hidden_by_completed_status():
    row = completed(); row['online_metrics']['calls'][4]['status'] = 'failed_no_partial_fullgraph'
    with pytest.raises(ValueError): evaluator.verify_completed_EOF(row)


def test_failed_v45_has_no_failed_EOF_exception(tmp_path):
    path = tmp_path/'run.json'
    path.write_text(json.dumps(dict(schema=evaluator.SCHEMAS['online_RKNN'], status='failed')))
    with pytest.raises(ValueError, match='no failed-run exception'):
        evaluator.load_run(tmp_path, evaluator.sha(path), 'online_RKNN', None, None, None)


def test_previous_actual_schema_cannot_be_relabelled_as_new(tmp_path):
    path = tmp_path/'run.json'
    path.write_text(json.dumps(dict(schema=evaluator.SCHEMAS['previous_online_RKNN'],
        status='completed_unpaced_half_online_RKNN_research_not_realtime')))
    with pytest.raises(ValueError, match='actual run schema differs'):
        evaluator.load_run(tmp_path, evaluator.sha(path), 'online_RKNN', None, None, None)
    assert evaluator.SCHEMAS['cached_half_CPU'] == 'v41_cached_undistort_original_Torch_CPU_research_replay'
    assert evaluator.SCHEMAS['full_resolution_FAST_CPU'] == 'v34_DPVO_original_FAST_fresh_frame_CPU_replay'
