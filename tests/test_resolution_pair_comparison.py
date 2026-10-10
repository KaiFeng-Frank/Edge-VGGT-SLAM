"""Synthetic saved numbers only: no image decoding, tracking or inference."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def comparison():
    path = Path(__file__).resolve().parents[1] / 'deploy/rk3588/validation/compare-v16-resolution-pairs.py'
    spec = importlib.util.spec_from_file_location('resolution_pair_comparison', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def saved(comparison, tmp_path):
    raw_hw = [12, 16]
    K = np.array([[14., 0., 7.5], [0., 14., 5.5], [0., 0., 1.]])
    selected = []
    for i in range(3):
        images = [dict(path=f'/private/synthetic/rgb/{j}.png', frame_id=float(j), timestamp=float(j),
                       sha256=f'{j + 1:064x}', samefile_validated=True,
                       schedule_token=dict(frame_id=float(j), timestamp=float(j))) for j in (i, i + 1)]
        selected.append(dict(revision=i + 1, frame_ids=[float(i), float(i + 1)], images=images))
    outputs = []
    for size, scale in ((112, 1.), (518, 3.)):
        path = tmp_path / str(size); path.mkdir()
        new_h = max(14, round(raw_hw[0] * size / raw_hw[1] / 14) * 14)
        shape = [2, 3, new_h, size]
        A, _ = comparison.pixel_affine(raw_hw, shape, size)
        records = []
        for i in range(3):
            extrinsic = np.repeat(np.eye(4, dtype=np.float32)[None, None, :3], 2, axis=1)
            extrinsic[0, 1, 0, 3] = -.2 * scale
            arrays = dict(depth=np.full((1, 2, new_h, size, 1), 2 * scale, np.float32),
                depth_conf=np.ones((1, 2, new_h, size), np.float32),
                intrinsic=np.broadcast_to(A @ K, (1, 2, 3, 3)).astype(np.float32).copy(),
                extrinsic=extrinsic, pose_enc=np.zeros((1, 2, 9), np.float32))
            filename = f'predictions_{i + 1:04d}.npz'
            np.savez_compressed(path / filename, **arrays, frame_ids=selected[i]['frame_ids'])
            records.append(dict(revision=i + 1, frame_ids=selected[i]['frame_ids'], file=filename,
                sha256=comparison.geometry.sha(path / filename), input_tensor=dict(shape=shape),
                arrays={name: dict(shape=list(a.shape), bytes=a.nbytes, tensor_sha256=comparison.tensor_sha(a))
                        for name, a in arrays.items()}, model_inference_seconds=1. + size / 100.))
        metadata = dict(status='completed_fixed_input_diagnostic', image_size=size, max_pairs=3,
            script_sha256=comparison.CAPTURE_SCRIPT_SHA, device='cpu', precision='float32', ORB_used=False,
            npu_used=False, checkpoint=dict(sha256=comparison.CHECKPOINT_SHA, bytes=5026874952),
            source_hashes={'third_party/vggt/vggt/utils/load_fn.py': comparison.LOAD_FN_SHA},
            rectification=dict(R=np.eye(3).tolist(), crop=None, image_hw=raw_hw, output_K=K.tolist()),
            pairs=records, selected_inputs=selected, model={'class_name': 'VGGT'},
            rgb_index={'sha256': '4' * 64}, calibration={'sha256': '5' * 64},
            schedule=dict(selection='earliest_saved_revision_order_without_quality_filtering',
                          metrics_sha256='6' * 64, events_sha256='7' * 64))
        (path / 'capture.json').write_text(json.dumps(metadata))
        outputs.append(path)
    gt = tmp_path / 'gt.txt'
    gt.write_text('\n'.join(f'{i} {i} 0 0 0 0 0 1' for i in range(4)) + '\n')
    return (*outputs, gt)


def test_pixel_affine_uses_measured_392_not_guessed_378(comparison):
    A, info = comparison.pixel_affine([480, 640], [2, 3, 392, 518], 518)
    assert A[0, 0] == pytest.approx(518 / 640)
    assert A[1, 1] == pytest.approx(392 / 480)
    assert A[0, 0] != A[1, 1]
    assert info['actual_input_hw'] == [392, 518] and info['crop_y'] == 0
    with pytest.raises(ValueError, match='Recorded input shape'):
        comparison.pixel_affine([480, 640], [2, 3, 378, 518], 518)


def test_portrait_crop_and_half_pixel_K_restore(comparison):
    A, info = comparison.pixel_affine([640, 480], [2, 3, 112, 112], 112)
    assert info['resized_hw_before_crop'] == [154, 112] and info['crop_y'] == 21
    uv = np.array([199.3, 320.1, 1.])
    mapped = A @ uv
    assert mapped[1] == pytest.approx((uv[1] + .5) * 154 / 640 - .5 - 21)
    K = np.array([[200., 0., 239.5], [0., 190., 319.5], [0., 0., 1.]])
    assert np.allclose(comparison.camera_stats(A @ K, A, K)['K_original_rectified_pixels'], K)


def test_bilinear_sampling_is_finite_positive_and_requires_complete_neighbours(comparison):
    values = np.arange(12, dtype=float).reshape(3, 4) + 1
    result = comparison.bilinear_sample(values, np.array([[.5, .5], [3., 1.], [-.01, 1.]]))
    assert result[0] == pytest.approx(3.5)
    assert np.isnan(result[1:]).all()
    values[1, 1] = 0
    assert np.isnan(comparison.bilinear_sample(values, np.array([[.5, .5]]))[0])


def test_three_pair_GT_direction_and_single_depth_scale_are_separated(comparison, saved):
    result = comparison.analyze(*saved)
    assert len(result['pairs']) == 3 and result['input_identity_verified']
    assert not result['model_inference_executed'] and not result['full_dataset_ATE_RPE_validated']
    assert '/private/' not in json.dumps(result)
    for pair in result['pairs']:
        depth = pair['depth']
        assert depth['positive_scale_112_to_518'] == pytest.approx(3)
        assert depth['unaligned']['relative_l2'] == pytest.approx(2 / 3)
        assert depth['scale_aligned']['relative_l2'] < 1e-12
        assert depth['scale_fit_and_score_use_same_pixels'] and not depth['depth_GT_used']
        for prediction in pair['predictions'].values():
            gt = prediction['GT_pair_diagnostic']
            assert gt['rotation_error_degrees'] == pytest.approx(0)
            assert gt['translation_direction_error_degrees'] == pytest.approx(0)
    assert result['pairs'][0]['predictions']['112']['relative_Tcw'][0][3] == pytest.approx(-.2)


@pytest.mark.parametrize('field', ['source_hashes', 'calibration', 'selected_inputs', 'schedule'])
def test_cross_capture_identity_mismatch_is_rejected(comparison, saved, field):
    low, high, gt = saved
    path = high / 'capture.json'; metadata = json.loads(path.read_text())
    if field == 'source_hashes': metadata[field]['untracked_source.py'] = '8' * 64
    elif field == 'calibration': metadata[field]['sha256'] = '8' * 64
    elif field == 'selected_inputs': metadata[field][0]['images'][0]['sha256'] = '8' * 64
    else: metadata[field]['metrics_sha256'] = '8' * 64
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='differ'):
        comparison.analyze(low, high, gt)


def test_changed_prediction_file_cannot_be_silently_rescored(comparison, saved):
    low, high, gt = saved
    with (high / 'predictions_0001.npz').open('ab') as stream:
        stream.write(b'changed saved artifact')
    with pytest.raises(ValueError, match='SHA differs'):
        comparison.analyze(low, high, gt)


def test_missing_GT_or_tiny_motion_does_not_count_perfect_direction(comparison, saved):
    low, high, gt = saved
    gt.write_text('0 0 0 0 0 0 0 1\n1 .001 0 0 0 0 0 1\n2 .002 0 0 0 0 0 1\n3 .003 0 0 0 0 0 1\n')
    result = comparison.analyze(low, high, gt)
    for pair in result['pairs']:
        for row in pair['predictions'].values():
            assert row['GT_pair_diagnostic']['translation_direction_error_degrees'] is None
    gt.write_text('10 0 0 0 0 0 0 1\n11 1 0 0 0 0 0 1\n')
    result = comparison.analyze(low, high, gt)
    assert result['pairs'][0]['predictions']['112']['GT_pair_diagnostic'] is None


def test_each_batch_gets_one_scale_not_separate_frame_scales(comparison):
    low = dict(arrays={'depth': np.ones((1, 2, 2, 2, 1))}, A=np.eye(3), record={'frame_ids': [0., 1.]})
    high_depth = np.ones((1, 2, 3, 3, 1)) * 2
    high_depth[:, 1] *= 2
    high = dict(arrays={'depth': high_depth}, A=np.eye(3))
    result = comparison.compare_depth(low, high)
    assert result['positive_scale_112_to_518'] == pytest.approx(3)
    assert result['scale_aligned']['rmse'] == pytest.approx(1)


def test_pi_rotation_and_opposite_translation_are_not_hidden(comparison):
    a, b = np.eye(4), np.eye(4)
    a[0, 3], b[0, 3] = 1., -1.
    assert comparison.geometry.errors(a, b, .01)['translation_direction_error_degrees'] == pytest.approx(180)
    assert comparison.geometry.rotation_degrees(np.diag([-1., -1., 1.])) == pytest.approx(180)
