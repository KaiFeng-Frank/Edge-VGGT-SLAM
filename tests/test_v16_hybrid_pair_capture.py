"""Synthetic values/files only: no images are decoded and no models run."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def driver():
    path = Path(__file__).resolve().parents[1] / 'deploy/rk3588/validation/capture-v16-hybrid-pairs-board.py'
    spec = importlib.util.spec_from_file_location('v16_hybrid_capture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def baseline(driver, tmp_path, monkeypatch):
    images = []
    for i in range(4):
        path = tmp_path / f'{100.125 + i}.png'
        path.write_bytes(f'synthetic identity, not image {i}'.encode())
        images.append(dict(path=str(path), frame_id=100.125 + i, timestamp=100.125 + i,
                           sha256=driver.sha(path), samefile_validated=True,
                           schedule_token=dict(path=str(path), frame_id=100.125 + i,
                                               timestamp=100.125 + i, captured_monotonic=10. + i, sequence=i)))
    directory = tmp_path / 'CPU-capture'
    directory.mkdir()
    selected, measured = [], []
    for i in range(3):
        ids = [images[i]['frame_id'], images[i + 1]['frame_id']]
        arrays = {key: np.full((1, 2, 3), 1 + i, dtype=np.float32) for key in driver.OUTPUT_KEYS}
        name = f'predictions_{i+1:04d}.npz'
        np.savez(directory / name, **arrays, frame_ids=np.asarray(ids, dtype=np.float64))
        selected.append(dict(revision=i+1, frame_ids=ids, images=images[i:i+2]))
        measured.append(dict(revision=i+1, frame_ids=ids, file=name, bytes=(directory/name).stat().st_size,
                             sha256=driver.sha(directory/name), arrays={k: driver.array_info(v) for k,v in arrays.items()},
                             input_tensor=driver.array_info(np.zeros((2, 3, 84, 112), dtype=np.float32))))
    metadata = {}
    for name in ('calibration', 'rgb_index'):
        p = tmp_path / (name + '.json')
        p.write_text('{}')
        metadata[name] = dict(path=str(p), sha256=driver.sha(p))
    record = dict(status='completed_fixed_input_diagnostic', script_sha256=driver.BASELINE_SCRIPT_SHA,
                  device='cpu', precision='float32', original_weights=True, image_size=112,
                  ORB_used=False, npu_used=False, graph_run=False, GT_used=False, online_pipeline_run=False,
                  selected_inputs=selected, pairs=measured,
                  checkpoint=dict(bytes=driver.CHECKPOINT_BYTES, sha256=driver.CHECKPOINT_SHA), **metadata)
    def save():
        (directory / 'capture.json').write_text(json.dumps(record))
        monkeypatch.setattr(driver, 'BASELINE_SHA', driver.sha(directory / 'capture.json'))
    save()
    return directory, record, save


def test_exact_ordered_three_pair_baseline_and_independent_cpu_arrays(driver, baseline):
    directory, expected, _ = baseline
    record, arrays = driver.load_baseline(directory)
    assert record == expected
    assert [x['frame_ids'] for x in record['selected_inputs']] == [[100.125, 101.125], [101.125, 102.125], [102.125, 103.125]]
    assert len(arrays) == 3 and set(arrays[0]) == set(driver.OUTPUT_KEYS)
    assert all(a.dtype == np.float32 for pair in arrays for a in pair.values())


@pytest.mark.parametrize('mutation', ['running', 'npu', 'frame', 'inputshape', 'missingpair', 'array_sha', 'calibration_sha'])
def test_baseline_semantics_identity_tensor_and_configuration_are_not_inferred(driver, baseline, mutation):
    directory, record, save = baseline
    if mutation == 'running': record['status'] = 'running'
    if mutation == 'npu': record['npu_used'] = True
    if mutation == 'frame': record['selected_inputs'][0]['images'][0]['frame_id'] += .01
    if mutation == 'inputshape': record['pairs'][0]['input_tensor']['shape'] = [2, 3, 112, 112]
    if mutation == 'missingpair': record['pairs'].pop()
    if mutation == 'array_sha': record['pairs'][0]['arrays']['depth']['tensor_sha256'] = '0'*64
    if mutation == 'calibration_sha': record['calibration']['sha256'] = '0'*64
    save()
    with pytest.raises(ValueError):
        driver.load_baseline(directory)


def test_raw_image_mutation_and_cpu_file_mutation_are_rejected(driver, baseline):
    directory, record, _ = baseline
    path = Path(record['selected_inputs'][0]['images'][0]['path'])
    original = path.read_bytes()
    path.write_bytes(original + b'changed')
    with pytest.raises(ValueError, match='SHA256'):
        driver.load_baseline(directory)
    path.write_bytes(original)
    (directory / 'predictions_0001.npz').write_bytes(b'changed CPU reference')
    with pytest.raises(ValueError, match='SHA256'):
        driver.load_baseline(directory)


def test_error_comparison_never_turns_difference_into_accuracy_pass(driver):
    expected = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    actual = expected + np.float32(.125)
    result = driver.compare_arrays(actual, expected)
    assert result['max_abs_error'] == .125 and not result['array_equal']
    assert result['tolerance_applied'] is result['accuracy_pass_inferred'] is False
    assert np.array_equal(actual, expected + .125)
    assert driver.compare_arrays(np.zeros_like(expected), np.zeros_like(expected))['relative_l2'] is None
    for bad in (actual.astype(np.float16), np.full_like(actual, np.nan), actual[..., ::-1]):
        with pytest.raises(ValueError):
            driver.compare_arrays(bad, expected)


def test_full_camera_configuration_preserves_exact_api_keywords_and_sha(driver, tmp_path):
    a = SimpleNamespace()
    for name in ('model', 'export_receipt', 'compile_receipt', 'capture_receipt'):
        path = tmp_path / name
        path.write_bytes(name.encode())
        setattr(a, 'camera_' + name, path)
        setattr(a, 'camera_' + name + '_sha256', driver.sha(path))
    result = driver.camera_configuration(a)
    assert set(result) == {'model_path', 'model_sha256', 'export_receipt', 'export_sha256',
                           'compile_receipt', 'compile_sha256', 'capture_receipt', 'capture_sha256'}
    a.camera_model_sha256 = '0'*64
    with pytest.raises(ValueError, match='SHA256'):
        driver.camera_configuration(a)


def test_cleanup_attempts_both_handles_and_reads_historical_metrics_after_failure(driver):
    events = []
    class Handle:
        def __init__(self, name, fail=False): self.name, self.fail, self.closed = name, fail, False
        def close(self):
            events.append('close:' + self.name)
            self.closed = True
            if self.fail: raise RuntimeError('synthetic close failure')
        def metrics(self):
            events.append('metrics:' + self.name)
            return dict(closed=self.closed, sdk_calls=12)
    record = dict(cleanup_errors=[], error='original inference failure')
    driver.close_handles(Handle('linear', True), Handle('camera'), record)
    assert events == ['close:linear', 'metrics:linear', 'close:camera', 'metrics:camera']
    assert len(record['cleanup_errors']) == 1 and record['error'] == 'original inference failure'
    assert record['camera_runtime_final'] == dict(closed=True, sdk_calls=12)


def test_host_main_rejects_before_output_or_model_and_capture_itself_is_guarded(driver, monkeypatch, tmp_path):
    monkeypatch.setattr(driver.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(driver, 'capture', lambda _: pytest.fail('Host cannot call capture'))
    output = tmp_path / 'not_created'
    args = ['--baseline-capture-dir', str(tmp_path), '--output', str(output)]
    for name in ('model', 'export-receipt', 'compile-receipt', 'capture-receipt'):
        args.extend(['--camera-' + name, str(tmp_path / name), '--camera-' + name + '-sha256', '0'*64])
    with pytest.raises(SystemExit) as exc:
        driver.main(args)
    assert exc.value.code == 2 and not output.exists()
    with pytest.raises(RuntimeError, match='no host inference'):
        driver.require_board()
