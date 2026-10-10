"""Synthetic schedule identities only; no images are decoded or models run."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def driver():
    path = Path(__file__).resolve().parents[1] / 'deploy/rk3588/validation/capture-v16-resolution-pairs-board.py'
    spec = importlib.util.spec_from_file_location('resolution_pair_capture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def schedule(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    rows = []
    for i in range(4):
        path = raw / f'{100.125 + i:.3f}.png'
        path.write_bytes(f'not an image; identity fixture {i}'.encode())
        rows.append(dict(timestamp=100.125 + i, path=str(path)))
    planned = tmp_path / 'schedule'
    planned.mkdir()
    rectification = {'calibration_sha256': 'synthetic-config', 'output_K': [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]}
    metrics = dict(status='completed_experiment', mapper_exitcode=0, cleanup_errors=[],
                   ORB_used=False, model_backbone='original_VGGT', common_rectification=rectification,
                   map_updates=3, mapper_config=dict(backend='xnnpack-f16', image_size=112))
    events = []
    for i in range(3):
        ids = [rows[i]['timestamp'], rows[i + 1]['timestamp']]
        np.savez(planned / f'predictions_{i + 1:04d}.npz', frame_ids=ids,
                 depth=np.zeros(1), depth_conf=np.zeros(1), intrinsic=np.zeros(1), extrinsic=np.zeros(1))
        events.append(dict(kind='map_submitted', batch=[dict(path=row['path'], frame_id=row['timestamp'],
                              timestamp=row['timestamp'], sequence=j, captured_monotonic=10. + j)
                             for j, row in enumerate(rows[i:i + 2], i)]))
    (planned / 'metrics.json').write_text(json.dumps(metrics))
    (planned / 'events.jsonl').write_text(''.join(json.dumps(event) + '\n' for event in events))
    return planned, rows, rectification, metrics, events


def test_default_plan_is_earliest_three_exact_overlapping_pairs(driver, schedule):
    path, rows, rectification, _, _ = schedule
    pairs, metadata = driver.plan_pairs(path, rows, rectification, 3)
    assert [pair['frame_ids'] for pair in pairs] == [[100.125, 101.125], [101.125, 102.125], [102.125, 103.125]]
    assert all(image['samefile_validated'] for pair in pairs for image in pair['images'])
    assert all(image['sha256'] == driver.sha(image['path']) for pair in pairs for image in pair['images'])
    assert metadata['selected_pairs'] == metadata['available_pairs'] == 3
    assert 'without_quality_filtering' in metadata['selection']
    assert metadata['original_backend'] == 'xnnpack-f16'
    assert driver.plan_pairs(path, rows, rectification, 1)[0] == pairs[:1]


@pytest.mark.parametrize('bad', ['calibration', 'completed', 'frame_id', 'timestamp', 'path', 'count', 'missing', 'revision'])
def test_schedule_identity_and_complete_metadata_are_not_inferred(driver, schedule, tmp_path, bad):
    path, rows, rectification, metrics, events = schedule
    if bad == 'calibration': rectification = {'calibration_sha256': 'other'}
    if bad == 'completed': metrics['status'] = 'running'
    if bad == 'frame_id': events[0]['batch'][0]['frame_id'] += .0001
    if bad == 'timestamp': events[0]['batch'][0]['timestamp'] += .0001
    if bad == 'path':
        other = tmp_path / Path(rows[0]['path']).name
        other.write_bytes(b'different original file with the same basename')
        events[0]['batch'][0]['path'] = str(other)
    if bad == 'count': metrics['map_updates'] = 4
    if bad == 'missing': (path / 'predictions_0003.npz').unlink()
    if bad == 'revision': (path / 'predictions_0003.npz').rename(path / 'predictions_0004.npz')
    (path / 'metrics.json').write_text(json.dumps(metrics))
    (path / 'events.jsonl').write_text(''.join(json.dumps(event) + '\n' for event in events))
    with pytest.raises(ValueError):
        driver.plan_pairs(path, rows, rectification, 3)


def test_unlisted_npz_frame_and_nonpair_are_rejected(driver, schedule):
    path, rows, rectification, _, _ = schedule
    for ids in ([100.125, 999.125], [100.125], [101.125, 100.125], [100.125, float('nan')]):
        np.savez(path / 'predictions_0001.npz', frame_ids=ids,
                 depth=np.zeros(1), depth_conf=np.zeros(1), intrinsic=np.zeros(1), extrinsic=np.zeros(1))
        with pytest.raises(ValueError):
            driver.plan_pairs(path, rows, rectification, 3)


def test_host_main_is_rejected_before_model_or_output(driver, monkeypatch, tmp_path):
    monkeypatch.setattr(driver.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(driver, 'capture', lambda _: pytest.fail('Host must not invoke capture'))
    output = tmp_path / 'unused-output'
    argv = ['--rgb-index', str(tmp_path / 'rgb.txt'), '--schedule-run-dir', str(tmp_path),
            '--checkpoint', str(tmp_path / 'model.pt'), '--calibration', str(tmp_path / 'K.json'),
            '--image-size', '112', '--output', str(output)]
    with pytest.raises(SystemExit) as exc:
        driver.main(argv)
    assert exc.value.code == 2 and not output.exists()


def test_array_receipt_preserves_shape_and_rejects_nonfinite_or_cast_precision(driver):
    value = np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2)
    info = driver.array_info(value)
    assert info['shape'] == [1, 2, 2, 2] and info['dtype'] == 'float32' and info['finite']
    for bad in (value.astype(np.float16), np.full_like(value, np.inf), value[:, :, :, ::-1]):
        with pytest.raises(ValueError):
            driver.array_info(bad)
