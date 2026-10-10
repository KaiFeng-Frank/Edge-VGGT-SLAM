"""Pure synthetic file/timestamp contracts; no image decoding or model calls."""
from argparse import Namespace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def driver():
    path = ROOT / 'deploy/rk3588/validation/capture-v18-quality-pairs-board.py'
    spec = importlib.util.spec_from_file_location('quality_capture_v18', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def schedule(driver, tmp_path, monkeypatch):
    protocol = driver.load_protocol()
    stamps = np.interp(np.arange(798), driver.SOURCE_SEQUENCES + (797,),
                       driver.FRAME_IDS + (1305031128.747363,)).tolist()
    rows = []
    for sequence, stamp in enumerate(stamps):
        file = tmp_path / f'{sequence}.png'
        if sequence in driver.SOURCE_SEQUENCES:
            file.write_bytes(f'not an image; synthetic identity {sequence}'.encode())
        rows.append(dict(timestamp=stamp, path=str(file)))
    path = tmp_path / 'schedule'
    path.mkdir()
    rectification = dict(calibration_sha256=driver.CALIBRATION_SHA)
    start = 63007.579881606
    metrics = dict(status='completed_experiment', mapper_exitcode=0, cleanup_errors=[],
                   ORB_used=False, model_backbone='original_VGGT', common_rectification=rectification,
                   map_updates=6, mapper_config=dict(backend='xnnpack-f16', image_size=112),
                   source=dict(original_frames=798, error=None, start_monotonic=start))
    events = []
    for index in range(6):
        ids = list(driver.FRAME_IDS[index:index + 2])
        np.savez(path / f'predictions_{index + 1:04d}.npz', frame_ids=np.array(ids),
                 depth=np.zeros(1), depth_conf=np.zeros(1), intrinsic=np.zeros(1), extrinsic=np.zeros(1))
        batch = [dict(path=rows[seq]['path'], frame_id=stamp, timestamp=stamp,
                      sequence=seq, captured_monotonic=start + stamp - stamps[0])
                 for stamp, seq in zip(ids, driver.SOURCE_SEQUENCES[index:index + 2])]
        events.extend([dict(kind='map_submitted', batch=batch),
                       dict(kind='map', revision=index + 1, graph_node_id=2 * index + 1, anchor=batch[-1]),
                       dict(kind='depth_adoption', revision=index + 1, graph_node_id=2 * index + 1,
                            anchor_frame_id=ids[-1], anchor_captured_monotonic=batch[-1]['captured_monotonic'])])
    def write():
        (path / 'metrics.json').write_text(json.dumps(metrics))
        (path / 'events.jsonl').write_text(''.join(json.dumps(event) + '\n' for event in events))
    def repin():
        monkeypatch.setattr(driver, 'SCHEDULE_FILES',
                            {name: driver.sha(path / name) for name in driver.SCHEDULE_FILES})
    write()
    repin()
    return protocol, path, rows, rectification, metrics, events, write, repin


def test_frozen_protocol_and_all_six_overlaps_have_exact_timestamp_anchor_identity(driver, schedule):
    protocol, path, rows, rectification, *_ = schedule
    pairs, metadata = driver.validate_schedule(protocol, path, rows, rectification)
    assert len(pairs) == metadata['available_pairs'] == metadata['selected_pairs'] == 6
    assert [p['frame_ids'] for p in pairs] == [list(driver.FRAME_IDS[i:i + 2]) for i in range(6)]
    assert [a['graph_node_id'] for a in metadata['anchors']] == [1, 3, 5, 7, 9, 11]
    assert [a['source_sequence'] for a in metadata['anchors']] == [8, 151, 289, 430, 593, 771]
    assert metadata['source_frames'] == 798 and not metadata['old_prediction_values_used']
    assert all(pairs[i]['images'][-1] == pairs[i + 1]['images'][0] for i in range(5))


@pytest.mark.parametrize('bad', ['checksum', 'extra', 'truncated', 'anchor', 'sequence', 'capture',
                               'npz_ids', 'count', 'calibration', 'index'])
def test_incomplete_or_rebound_schedule_is_rejected_without_quality_selection(driver, schedule, bad):
    protocol, path, rows, rectification, metrics, events, write, repin = schedule
    if bad == 'checksum':
        metrics['unused_note'] = 'changed'; write()
    elif bad == 'extra':
        (path / 'predictions_0007.npz').write_bytes(b'extra')
    elif bad == 'truncated':
        (path / 'predictions_0006.npz').unlink()
    else:
        if bad == 'anchor': events[-1]['anchor_frame_id'] += .01
        if bad == 'sequence': events[0]['batch'][1]['sequence'] -= 1
        if bad == 'capture': events[0]['batch'][0]['captured_monotonic'] += .000001
        if bad == 'npz_ids':
            np.savez(path / 'predictions_0006.npz', frame_ids=[1., 2.], depth=np.zeros(1),
                     depth_conf=np.zeros(1), intrinsic=np.zeros(1), extrinsic=np.zeros(1))
        if bad == 'count': metrics['map_updates'] = 5
        if bad == 'calibration': rectification['calibration_sha256'] = '0' * 64
        if bad == 'index': rows.pop()
        write(); repin()
    with pytest.raises((ValueError, FileNotFoundError)):
        driver.validate_schedule(protocol, path, rows, rectification)


@pytest.mark.parametrize('image_size', [112, 336, 518])
def test_host_is_rejected_before_output_or_model(driver, monkeypatch, tmp_path, image_size):
    monkeypatch.setattr(driver.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(driver, 'capture', lambda _: pytest.fail('Host model execution forbidden'))
    output = tmp_path / 'output-must-not-exist'
    argv = ['--rgb-index', str(tmp_path / 'rgb.txt'), '--schedule-run-dir', str(tmp_path),
            '--checkpoint', str(tmp_path / 'model.pt'), '--calibration', str(tmp_path / 'K.json'),
            '--output', str(output), '--image-size', str(image_size)]
    with pytest.raises(SystemExit) as exc:
        driver.main(argv)
    assert exc.value.code == 2 and not output.exists()


def test_new_wrapper_delegates_unchanged_capture_and_preserves_quality_only_scope(driver, monkeypatch, tmp_path):
    class FakeProtocol:
        def capture(self, args):
            assert args.max_pairs == 6 and args.image_size == 518
            return dict(status='completed_fixed_input_diagnostic',
                        script_sha256=driver.PROTOCOL_FILES['capture-v17-resolution-pairs-board.py'],
                        selected_inputs=[], pairs=[dict(revision=i) for i in range(1, 7)])
    monkeypatch.setattr(driver, 'load_protocol', lambda: FakeProtocol())
    monkeypatch.setattr(driver, 'input_guard', lambda *_: dict(selected_inputs=[]))
    result = driver.capture(Namespace(output=tmp_path, image_size=518))
    assert result['status'] == 'completed_fixed_input_diagnostic'
    assert result['max_pairs'] == 6 and result['quality_inputs_unchanged']
    assert result['base_capture_script_sha256'] == driver.PROTOCOL_FILES['capture-v17-resolution-pairs-board.py']
    assert result['script_sha256'] == driver.sha(driver.__file__)
    assert result['quality_only'] and result['model_batch_frames'] == 2
    assert result['paper_evaluation_batch_frames'] == 32 and not result['reproduces_paper_evaluation']
    for name in ('GT_used', 'npu_used', 'graph_run', 'online_pipeline_run',
                 'accuracy_validated', 'real_time_validated', 'full_dataset_ATE_RPE_validated'):
        assert result[name] is False
    assert json.loads((tmp_path / 'capture.json').read_text()) == result


def test_primary_capture_failure_is_kept_when_after_guard_also_fails(driver, monkeypatch, tmp_path):
    class FakeProtocol:
        def capture(self, args):
            return dict(status='failed', error='original model failed', pairs=[])
    monkeypatch.setattr(driver, 'load_protocol', lambda: FakeProtocol())
    calls = []
    def guard(*_):
        calls.append(1)
        if len(calls) > 1:
            raise ValueError('source changed afterwards')
        return dict(selected_inputs=[])
    monkeypatch.setattr(driver, 'input_guard', guard)
    result = driver.capture(Namespace(output=tmp_path, image_size=112))
    assert result['status'] == 'failed' and result['error'] == 'original model failed'
    assert result['quality_guard_errors'] == ['ValueError: source changed afterwards']
    assert result['quality_only']
