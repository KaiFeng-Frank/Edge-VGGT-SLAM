"""Synthetic saved arrays only; no image, checkpoint or model inference."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope='module')
def comparator():
    path = Path(__file__).resolve().parents[1] / 'deploy/rk3588/validation/compare-v22-come-fullmodel.py'
    spec = importlib.util.spec_from_file_location('come_fullmodel_comparison', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metadata(module, arrays):
    return {key: dict(shape=list(value.shape), dtype=str(value.dtype), bytes=value.nbytes,
                      sha256=module.tensor_sha(value)) for key, value in arrays.items()}


@pytest.fixture
def saved(comparator, tmp_path):
    original = tmp_path / 'original'; original.mkdir()
    merged = tmp_path / 'fullmodel'; merged.mkdir()
    shape = [2, 3, 14, 336]
    selected, pairs, values = [], [], []
    for pair in range(3):
        ids = [10.125 + pair, 10.25 + pair]
        images = [dict(path=f'/private/synthetic/image-{fid}.png', sha256=f'{pair * 2 + j + 1:064x}',
                       frame_id=fid, timestamp=fid, samefile_validated=True,
                       schedule_token=dict(frame_id=fid, timestamp=fid)) for j, fid in enumerate(ids)]
        selected.append(dict(revision=pair + 1, frame_ids=ids, images=images))
        extrinsic = np.broadcast_to(np.eye(4, dtype=np.float32)[None, None, :3], (1, 2, 3, 4)).copy()
        extrinsic[0, 1, 0, 3] = -.2 * (pair + 1)
        K = np.array([[200., 0., 167.5], [0., 200., 6.5], [0., 0., 1.]], np.float32)
        arrays = dict(depth=np.full((1, 2, 14, 336, 1), 2. + pair, np.float32),
                      depth_conf=np.zeros((1, 2, 14, 336), np.float32),
                      extrinsic=extrinsic, intrinsic=np.broadcast_to(K, (1, 2, 3, 3)).copy(),
                      pose_enc=np.zeros((1, 2, 9), np.float32))
        values.append(arrays)
        file = original / f'predictions_{pair + 1:04d}.npz'
        np.savez(file, **arrays, frame_ids=np.asarray(ids, dtype=np.float64))
        pairs.append(dict(revision=pair + 1, frame_ids=ids, file=file.name,
                          sha256=comparator.sha(file), bytes=file.stat().st_size,
                          arrays=metadata(comparator, arrays), input_tensor=dict(shape=shape,
                          tensor_sha256=f'{pair + 10:064x}')))
    reference = dict(status='completed_fixed_input_diagnostic', image_size=336, max_pairs=3,
                     original_weights=True, device='cpu', precision='float32', ORB_used=False, npu_used=False,
                     checkpoint=dict(sha256=comparator.CHECKPOINT_SHA, bytes=comparator.CHECKPOINT_BYTES),
                     pairs=pairs, selected_inputs=selected, rectification=dict(image_hw=[14, 336],
                     R=np.eye(3).tolist(), crop=None), source_hashes={'original.py': 'a' * 64})
    (original / 'capture.json').write_text(json.dumps(reference))
    reference_sha = comparator.sha(original / 'capture.json')
    records = []
    for order, (pair, method) in enumerate(comparator.ORDER):
        arrays = {key: value.copy() for key, value in values[pair].items()}
        if method == 'merged':
            arrays['depth'] *= 2
            arrays['extrinsic'][..., 3] *= 2
        file = merged / f'call-{order:02d}-pair-{pair}-{method}.npz'
        np.savez(file, **arrays, frame_ids=np.asarray(pairs[pair]['frame_ids'], dtype=np.float64))
        records.append(dict(pair_index=pair, method=method, order_index=order,
                            artifact=dict(file=file.name, sha256=comparator.sha(file), bytes=file.stat().st_size),
                            arrays=metadata(comparator, arrays), frame_ids=pairs[pair]['frame_ids'],
                            input_tensor_sha256=pairs[pair]['input_tensor']['tensor_sha256'],
                            pair_source_images=[dict(path=image['path'], sha256=image['sha256'])
                                                for image in selected[pair]['images']], forward_seconds=1. + order * .1))
    for row in records:
        if row['method'] == 'merged':
            row['research_metrics'] = dict(closed=True, active_scene=False, cleanup_errors=[],
                original_weight_hashes_unchanged=True, scenes_started=1, scenes_completed=1, scenes_failed=0,
                predictor_calls=1, plan_builds=1, mlp_calls={'dino': 9, 'frame': 24, 'global': 24})
            row['scene_record'] = dict(status='completed', mlp_calls={'dino': 9, 'frame': 24, 'global': 24},
                synthetic_plan_callback_used=False, synthetic_predictor_callback_used=False)
    receipt = dict(kind='isolated_original_weight_MLP_only_CoMe_fullmodel_research', status='completed',
                   device='cpu', precision='float32', original_weights=True, npu_used=False, ORB_used=False,
                   GT_used=False, inference_on_board_only=True, full_CoMe_method_implemented=False,
                   full_model_forward_attempts=8, full_model_forward_count=8, warmup_forwards=0, cleanup_errors=[],
                   sources_and_inputs_unchanged=True, cpu_affinity_restored=True,
                   all_research_contexts_closed=True, original_forward_identities_and_hooks_restored=True,
                   actual_research_predictor_calls=4, actual_research_MLP_calls=228,
                   research_contexts=[row['research_metrics'] for row in records if row['method'] == 'merged'],
                   reference_capture_sha256=reference_sha,
                   image_size=336, checkpoint=reference['checkpoint'], selected_inputs=selected,
                   rectification=reference['rectification'], source_hashes=reference['source_hashes'],
                   call_records=records, timing_scope='synthetic saved numbers only')
    (merged / 'fullmodel.json').write_text(json.dumps(receipt))
    return merged, original, reference_sha


def analyze(module, saved):
    full, original, pin = saved
    return module.analyze(full, original, module.sha(full / 'fullmodel.json'), expected_reference_sha256=pin)


def mutate_receipt(saved, mutation):
    file = saved[0] / 'fullmodel.json'
    value = json.loads(file.read_text())
    mutation(value)
    file.write_text(json.dumps(value))


def rewrite_call(module, saved, index, change):
    file = saved[0] / 'fullmodel.json'
    value = json.loads(file.read_text())
    row = value['call_records'][index]
    path = saved[0] / row['artifact']['file']
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    change(arrays)
    np.savez(path, **arrays)
    row['artifact'].update(sha256=module.sha(path), bytes=path.stat().st_size)
    row['arrays'] = metadata(module, {key: arrays[key] for key in module.KEYS})
    file.write_text(json.dumps(value))


def test_complete_cohort_is_anonymized_and_bound_to_frozen_arrays(comparator, saved):
    result = analyze(comparator, saved)
    assert len(result['calls']) == 8 and len(result['pairs']) == 3
    assert result['cohort']['original_calls_bitwise_bound_to_frozen_reference']
    assert result['scope']['GT_used'] is False and result['scope']['realtime_validated'] is False
    assert result['scope']['host_model_inference_executed'] is False
    assert '/private/' not in json.dumps(result)
    for call in result['calls']:
        if call['method'] == 'merged':
            assert call['all_pixel_depth']['positive_scale_merged_to_original'] == pytest.approx(.5)
            assert call['all_pixel_depth']['raw']['relative_l2'] == pytest.approx(1)
            assert call['all_pixel_depth']['scale_aligned']['relative_l2'] == pytest.approx(0)
            assert call['relative_pose']['translation_direction_difference_degrees'] == pytest.approx(0)
        else:
            assert all(error['bitwise_equal'] for error in call['errors_against_frozen_original'].values())
    assert result['pairs'][0]['repeats']['merged']['repeated_stability_exercised']
    assert not result['pairs'][1]['repeats']['merged']['repeated_stability_exercised']


def test_one_shared_two_view_scale_never_masks_borders_or_low_confidence(comparator):
    actual = np.ones((1, 2, 2, 2, 1), dtype=np.float32)
    original = np.ones_like(actual) * 2
    original[:, 1] = 4
    result = comparator.compare_depth(actual, original)
    assert result['pixels'] == 8 and result['excluded_pixels'] == 0
    assert result['positive_scale_merged_to_original'] == pytest.approx(3)
    assert result['scale_aligned']['rmse'] == pytest.approx(1)
    assert result['scale_fit_and_score_use_same_pixels']
    actual[0, 0, 0, 0, 0] = 100
    changed = comparator.compare_depth(actual, original)
    assert changed['pixels'] == 8 and changed['raw']['max_absolute'] == pytest.approx(98)


def test_nonpositive_depth_is_reported_and_not_silently_filtered(comparator):
    original = np.ones((1, 2, 2, 2, 1), np.float32)
    actual = -original
    result = comparator.compare_depth(actual, original)
    assert result['merged_nonpositive_pixels'] == 8 and result['excluded_pixels'] == 0
    assert result['positive_scale_merged_to_original'] is None and result['scale_aligned'] is None
    assert result['raw']['relative_l2'] == pytest.approx(2)
    zero = comparator.numerical_error(original, np.zeros_like(original))
    assert zero['relative_l2'] is None and zero['relative_l2_unavailable_reason'] == 'zero_reference_norm'


def test_relative_transform_uses_world_to_camera_convention(comparator):
    E = np.zeros((1, 2, 3, 4), np.float32)
    E[0, 0, :, :3] = np.eye(3)
    E[0, 1, :, :3] = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    E[0, 0, :, 3] = [1., 2., 0.]
    E[0, 1, :, 3] = [4., 5., 0.]
    relative = comparator.relative_pose(E)
    assert np.array_equal(relative[:3, 3], [6., 4., 0.])
    assert np.array_equal(relative[:3, :3], E[0, 1, :, :3])
    zero = comparator.compare_pose(E, E)
    assert zero['rotation_difference_degrees'] == pytest.approx(0)
    E2 = E.copy(); E2[0, 1, :, :3] = -E2[0, 1, :, :3]
    with pytest.raises(ValueError, match='proper rotation'):
        comparator.relative_pose(E2)


def test_pi_rotation_and_opposite_translation_are_not_hidden(comparator):
    original = np.broadcast_to(np.eye(4, dtype=np.float32)[None, None, :3], (1, 2, 3, 4)).copy()
    actual = original.copy()
    original[0, 1, 0, 3] = 1
    actual[0, 1, 0, 3] = -1
    actual[0, 1, :, :3] = np.diag([-1., -1., 1.])
    result = comparator.compare_pose(actual, original)
    assert result['rotation_difference_degrees'] == pytest.approx(180)
    assert result['translation_direction_difference_degrees'] == pytest.approx(180)
    original[0, 1, 0, 3] = 0
    assert comparator.compare_pose(actual, original)['translation_direction_difference_degrees'] is None


@pytest.mark.parametrize('field', ['pair_index', 'method', 'order_index', 'frame_ids', 'input_tensor_sha256', 'pair_source_images'])
def test_call_identity_and_order_mismatches_are_fatal(comparator, saved, field):
    def change(value):
        row = value['call_records'][1]
        if field == 'pair_index': row[field] = 1
        elif field == 'method': row[field] = 'original'
        elif field == 'order_index': row[field] = 2
        elif field == 'frame_ids': row[field][1] += .001
        elif field == 'input_tensor_sha256': row[field] = 'f' * 64
        else: row[field][0]['sha256'] = 'f' * 64
    mutate_receipt(saved, change)
    with pytest.raises(ValueError):
        analyze(comparator, saved)


@pytest.mark.parametrize('field', ['checkpoint', 'source_hashes', 'rectification', 'selected_inputs', 'reference_capture_sha256'])
def test_cross_run_provenance_mismatches_are_fatal(comparator, saved, field):
    def change(value):
        if field == 'checkpoint': value[field]['sha256'] = 'b' * 64
        elif field == 'source_hashes': value[field]['original.py'] = 'b' * 64
        elif field == 'rectification': value[field]['crop'] = 1
        elif field == 'selected_inputs': value[field][0]['images'][0]['sha256'] = 'b' * 64
        else: value[field] = 'b' * 64
    mutate_receipt(saved, change)
    with pytest.raises(ValueError):
        analyze(comparator, saved)


def test_modified_file_and_changed_original_baseline_cannot_be_silently_scored(comparator, saved):
    full = saved[0]
    row = json.loads((full / 'fullmodel.json').read_text())['call_records'][1]
    with (full / row['artifact']['file']).open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='file SHA/size'):
        analyze(comparator, saved)
    rewrite_call(comparator, saved, 1, lambda values: None)
    rewrite_call(comparator, saved, 0, lambda values: values['depth'].__iadd__(1))
    with pytest.raises(ValueError, match='original call differs bitwise'):
        analyze(comparator, saved)


@pytest.mark.parametrize('mutation', ['frame_dtype', 'NaN', 'metadata_SHA', 'extra_key', 'canonical_dtype'])
def test_bad_array_or_header_is_rejected(comparator, saved, mutation):
    if mutation == 'metadata_SHA':
        mutate_receipt(saved, lambda value: value['call_records'][1]['arrays']['depth'].__setitem__('sha256', 'f' * 64))
    else:
        def change(values):
            if mutation == 'frame_dtype': values['frame_ids'] = values['frame_ids'].astype(np.float32)
            elif mutation == 'NaN': values['depth'][0, 0, 0, 0, 0] = np.nan
            elif mutation == 'extra_key': values['hidden_filter'] = np.array([1])
            else: values['depth_conf'] = values['depth_conf'].astype(np.float64)
        rewrite_call(comparator, saved, 1, change)
    with pytest.raises(ValueError):
        analyze(comparator, saved)


def test_missing_call_is_fatal(comparator, saved):
    mutate_receipt(saved, lambda value: value['call_records'].pop())
    with pytest.raises(ValueError, match='eight'):
        analyze(comparator, saved)


def test_receipt_pin_and_duplicate_JSON_keys_are_required(comparator, saved):
    with pytest.raises(ValueError, match='supplied measurement pin'):
        comparator.analyze(saved[0], saved[1], 'f' * 64, expected_reference_sha256=saved[2])
    file = saved[0] / 'fullmodel.json'
    file.write_text('{"call_records":[],"call_records":[]}')
    with pytest.raises(ValueError, match='Duplicate JSON key'):
        analyze(comparator, saved)


def test_conflicting_raw_hashes_and_path_escape_are_rejected(comparator, saved):
    mutate_receipt(saved, lambda value: value['call_records'][1]['arrays']['depth'].__setitem__('tensor_sha256', 'b' * 64))
    with pytest.raises(ValueError, match='Conflicting raw'):
        analyze(comparator, saved)
    with pytest.raises(ValueError, match='local NPZ basenames'):
        comparator.artifact_path(saved[0], '../elsewhere.npz')


@pytest.mark.parametrize('field', ['status', 'cleanup_errors', 'full_model_forward_count', 'actual_research_MLP_calls'])
def test_incomplete_or_unclean_execution_cannot_be_reported_completed(comparator, saved, field):
    def change(value):
        value[field] = 'failed' if field == 'status' else (['failure'] if field == 'cleanup_errors' else 7)
    mutate_receipt(saved, change)
    with pytest.raises(ValueError, match='completed board-only'):
        analyze(comparator, saved)
