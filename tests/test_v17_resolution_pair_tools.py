"""Synthetic metadata/array checks; no real images, models or board calls."""
import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'deploy/rk3588/validation' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def capture_driver():
    return load('capture_resolution_v17', 'capture-v17-resolution-pairs-board.py')


@pytest.mark.parametrize('name', ['sha', 'plan_pairs', 'array_info'])
def test_original_capture_identity_and_array_logic_is_unchanged(capture_driver, name):
    original = ROOT / 'deploy/rk3588/validation/capture-v16-resolution-pairs-board.py'
    assert capture_driver.sha(original) == '64adbcbe7f6e013aa738b8bed57d21ef8fde10c4b63bd7c647e9fa2137a79d7c'
    texts = [original.read_text(), Path(capture_driver.__file__).read_text()]
    functions = []
    for text in texts:
        node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
        functions.append(ast.get_source_segment(text, node))
    assert functions[0] == functions[1]


@pytest.mark.parametrize('image_size', [224, 336])
def test_host_capture_is_rejected_before_model_and_output(capture_driver, monkeypatch, tmp_path, image_size):
    monkeypatch.setattr(capture_driver.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(capture_driver, 'capture', lambda _: pytest.fail('Host cannot run a model'))
    output = tmp_path / 'must-not-exist'
    argv = ['--rgb-index', str(tmp_path / 'rgb.txt'), '--schedule-run-dir', str(tmp_path),
            '--checkpoint', str(tmp_path / 'model.pt'), '--calibration', str(tmp_path / 'K.json'),
            '--image-size', str(image_size), '--output', str(output)]
    with pytest.raises(SystemExit) as exc:
        capture_driver.main(argv)
    assert exc.value.code == 2 and not output.exists()


@pytest.mark.parametrize('image_size', [112, 128, 518])
def test_v17_does_not_silently_expand_resolution_choices(capture_driver, tmp_path, image_size):
    argv = ['--rgb-index', str(tmp_path / 'rgb.txt'), '--schedule-run-dir', str(tmp_path),
            '--checkpoint', str(tmp_path / 'model.pt'), '--calibration', str(tmp_path / 'K.json'),
            '--image-size', str(image_size), '--output', str(tmp_path / 'unused')]
    with pytest.raises(SystemExit) as exc:
        capture_driver.main(argv)
    assert exc.value.code == 2


def test_only_mapper_scheduling_may_change_around_the_lossless_bridge(capture_driver, tmp_path):
    source = (ROOT / 'vggt_slam/edge_mapping.py').read_text()
    file = tmp_path / 'synthetic_source.py'
    file.write_text(source + '\n# scheduling change outside the image bridge\n')
    metadata = capture_driver.preprocessing_provenance(file)
    assert metadata['sha256'] == capture_driver.PREPROCESSING_SHA
    file.write_text(source.replace('cv2.IMWRITE_PNG_COMPRESSION, 1', 'cv2.IMWRITE_PNG_COMPRESSION, 9'))
    with pytest.raises(ValueError, match='function changed'):
        capture_driver.preprocessing_provenance(file)
    file.write_text(source + '\ndef prepare_rectified_images():\n    pass\n')
    with pytest.raises(ValueError, match='exactly one'):
        capture_driver.preprocessing_provenance(file)


@pytest.fixture
def comparison():
    return load('compare_resolution_v17', 'compare-v17-resolution-pairs.py')


@pytest.fixture(params=[224, 336])
def saved(comparison, tmp_path, request):
    size = request.param
    raw_hw = [480, 640]
    K = np.array([[517.3, 0., 318.6], [0., 516.5, 255.3], [0., 0., 1.]])
    selected = []
    for i in range(3):
        images = [dict(path=f'/private/synthetic/rgb/{j}.png', frame_id=float(j), timestamp=float(j),
                       sha256=f'{j + 1:064x}', samefile_validated=True,
                       schedule_token=dict(frame_id=float(j), timestamp=float(j))) for j in (i, i + 1)]
        selected.append(dict(revision=i + 1, frame_ids=[float(i), float(i + 1)], images=images))
    paths = []
    for resolution, scale, producer in ((size, 1., comparison.V17_CAPTURE_SHA),
                                         (518, 3., comparison.V16_CAPTURE_SHA)):
        path = tmp_path / str(resolution)
        path.mkdir()
        height = max(14, round(raw_hw[0] * resolution / raw_hw[1] / 14) * 14)
        shape = [2, 3, height, resolution]
        A, _ = comparison.pixel_affine(raw_hw, shape, resolution)
        records = []
        for i in range(3):
            extrinsic = np.repeat(np.eye(4, dtype=np.float32)[None, None, :3], 2, axis=1)
            extrinsic[0, 1, 0, 3] = -.2 * scale
            arrays = dict(depth=np.full((1, 2, height, resolution, 1), 2 * scale, np.float32),
                depth_conf=np.ones((1, 2, height, resolution), np.float32),
                intrinsic=np.broadcast_to(A @ K, (1, 2, 3, 3)).astype(np.float32).copy(),
                extrinsic=extrinsic, pose_enc=np.zeros((1, 2, 9), np.float32))
            filename = f'predictions_{i + 1:04d}.npz'
            np.savez_compressed(path / filename, **arrays, frame_ids=selected[i]['frame_ids'])
            records.append(dict(revision=i + 1, frame_ids=selected[i]['frame_ids'], file=filename,
                sha256=comparison.geometry.sha(path / filename), input_tensor=dict(shape=shape),
                arrays={name: dict(shape=list(a.shape), bytes=a.nbytes, tensor_sha256=comparison.tensor_sha(a))
                        for name, a in arrays.items()}, model_inference_seconds=1. + resolution / 100.))
        sources = {name: 'a' * 64 for name in comparison.capture.SOURCE_FILES}
        sources['third_party/vggt/vggt/utils/load_fn.py'] = comparison.v16.LOAD_FN_SHA
        sources[comparison.EDGE_MAPPING] = comparison.V16_EDGE_MAPPING_SHA
        metadata = dict(status='completed_fixed_input_diagnostic', image_size=resolution, max_pairs=3,
            script_sha256=producer, device='cpu', precision='float32', ORB_used=False,
            npu_used=False, checkpoint=dict(sha256=comparison.v16.CHECKPOINT_SHA, bytes=5026874952),
            source_hashes=sources,
            rectification=dict(R=np.eye(3).tolist(), crop=None, image_hw=raw_hw, output_K=K.tolist()),
            pairs=records, selected_inputs=selected, model={'class_name': 'VGGT'},
            rgb_index={'sha256': '4' * 64}, calibration={'sha256': '5' * 64},
            schedule=dict(selection='earliest_saved_revision_order_without_quality_filtering',
                          metrics_sha256='6' * 64, events_sha256='7' * 64))
        if resolution == size:
            metadata['preprocessing_provenance'] = dict(source=comparison.EDGE_MAPPING,
                function='prepare_rectified_images', sha256=comparison.capture.PREPROCESSING_SHA,
                hash_method='ast_get_source_segment_utf8_sha256', original_v16_function_unchanged=True)
        (path / 'capture.json').write_text(json.dumps(metadata))
        paths.append(path)
    gt = tmp_path / 'gt.txt'
    gt.write_text('\n'.join(f'{i} {i} 0 0 0 0 0 1' for i in range(4)) + '\n')
    return (*paths, gt, size)


def test_both_candidate_sizes_use_correct_names_half_pixels_and_same_batch_scale(comparison, saved):
    result = comparison.analyze(*saved)
    size = saved[-1]
    assert result['candidate_image_size'] == size and result['reference_image_size'] == 518
    assert result['input_identity_verified'] and result['source_identity']['source_hashes_exact_match']
    assert not result['model_inference_executed'] and not result['full_dataset_ATE_RPE_validated']
    assert '/private/' not in json.dumps(result)
    for pair in result['pairs']:
        assert pair['mapping_candidate']['actual_input_hw'] == [168 if size == 224 else 252, size]
        assert pair['mapping_reference518']['actual_input_hw'] == [392, 518]
        assert set(pair['predictions']) == {'candidate', 'reference518'}
        assert pair['predictions']['candidate']['image_size'] == size
        assert pair['depth']['positive_scale_candidate_to_reference518'] == pytest.approx(3.)
        assert pair['depth']['scale_aligned']['relative_l2'] < 1e-12
        assert pair['depth']['unaligned']['relative_l2'] == pytest.approx(2 / 3)
        assert 'median_ratio_518_over_112' not in pair['depth']['unaligned']
        for row in pair['predictions'].values():
            assert row['GT_pair_diagnostic']['rotation_error_degrees'] == pytest.approx(0.)
            assert row['GT_pair_diagnostic']['translation_direction_error_degrees'] == pytest.approx(0.)
            assert np.allclose(row['K_original_pixels'][0]['K_original_rectified_pixels'],
                               [[517.3, 0., 318.6], [0., 516.5, 255.3], [0., 0., 1.]], rtol=1e-6)


def test_authorized_mapper_difference_is_explicit_and_does_not_claim_byte_identity(comparison, saved):
    candidate, reference, gt, size = saved
    path = candidate / 'capture.json'
    metadata = json.loads(path.read_text())
    metadata['source_hashes'][comparison.EDGE_MAPPING] = 'b' * 64
    path.write_text(json.dumps(metadata))
    result = comparison.analyze(candidate, reference, gt, size)
    evidence = result['source_identity']
    assert not evidence['source_hashes_exact_match']
    assert evidence['original_model_sources_exact_match'] and evidence['lossless_preprocessing_function_unchanged']
    assert set(evidence['allowed_source_differences']) == {comparison.EDGE_MAPPING}
    assert evidence['allowed_source_differences'][comparison.EDGE_MAPPING]['reference518'] == comparison.V16_EDGE_MAPPING_SHA


@pytest.mark.parametrize('bad', ['producer', 'model_source', 'bridge', 'missing_bridge', 'reference_mapper',
                                'image_sha', 'frame_id', 'schedule', 'missing_source', 'prediction_sha'])
def test_cross_producer_identity_and_unapproved_changes_are_rejected(comparison, saved, bad):
    candidate, reference, gt, size = saved
    path = candidate / 'capture.json'
    metadata = json.loads(path.read_text())
    if bad == 'producer': metadata['script_sha256'] = comparison.V16_CAPTURE_SHA
    if bad == 'model_source': metadata['source_hashes']['third_party/vggt/vggt/heads/camera_head.py'] = 'b' * 64
    if bad == 'bridge': metadata['preprocessing_provenance']['sha256'] = 'b' * 64
    if bad == 'missing_bridge': del metadata['preprocessing_provenance']
    if bad == 'reference_mapper':
        metadata['source_hashes'][comparison.EDGE_MAPPING] = 'b' * 64
        ref_path = reference / 'capture.json'
        ref = json.loads(ref_path.read_text())
        ref['source_hashes'][comparison.EDGE_MAPPING] = 'c' * 64
        ref_path.write_text(json.dumps(ref))
    if bad == 'image_sha': metadata['selected_inputs'][0]['images'][0]['sha256'] = 'b' * 64
    if bad == 'frame_id': metadata['pairs'][0]['frame_ids'][1] += .00001
    if bad == 'schedule': metadata['schedule']['events_sha256'] = 'b' * 64
    if bad == 'missing_source': del metadata['source_hashes']['vggt_slam/runtime.py']
    if bad == 'prediction_sha':
        with (candidate / 'predictions_0001.npz').open('ab') as stream:
            stream.write(b'altered saved numbers')
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        comparison.analyze(candidate, reference, gt, size)


def test_depth_uses_one_scale_for_both_frames_without_GT_or_quality_selection(comparison):
    candidate = dict(arrays={'depth': np.ones((1, 2, 2, 2, 1))}, A=np.eye(3), record={'frame_ids': [0., 1.]})
    depth = np.ones((1, 2, 3, 3, 1)) * 2
    depth[:, 1] *= 2
    reference = dict(arrays={'depth': depth}, A=np.eye(3))
    result = comparison.compare_depth(candidate, reference)
    assert result['positive_scale_candidate_to_reference518'] == pytest.approx(3.)
    assert result['scale_aligned']['rmse'] == pytest.approx(1.)
    assert result['scale_fit_and_score_use_same_pixels'] and not result['depth_GT_used']
