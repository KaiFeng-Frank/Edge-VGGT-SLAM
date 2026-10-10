#!/usr/bin/env python3
"""Capture all six frozen r11 pairs at 112/336/518, CPU FP32 on RK3588 only.

This is an offline quality-only cache, without tracking, GT, graph optimization,
XNNPACK or NPU execution. S=2 original-model batches are this project's baseline;
they do not reproduce the paper's S=32 evaluation. Model latency is recorded but
cannot be omitted from, or counted as evidence for, an online pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback

sys.dont_write_bytecode = True
BASE = Path('/mnt/vggt-data/vggt-slam2')
SOURCE = BASE / 'src'
PROTOCOL_FILES = {
    'capture-v17-resolution-pairs-board.py': '3706712dd3a7451fcbce98e079d4018989eca6f64eb98704a81b8f4a9acc03cb',
    'capture-v16-resolution-pairs-board.py': '64adbcbe7f6e013aa738b8bed57d21ef8fde10c4b63bd7c647e9fa2137a79d7c',
}
RGB_INDEX_SHA = '532884d60bef0c9a05e14bf311998883baf59a2fa47009d1021f2c7937bd69a3'
CALIBRATION_SHA = '29b6ca7e7a19570f0821fbb28b729d8ff8a73f239ff74efa31525f293a92cbaa'
SCHEDULE_FILES = {
    'metrics.json': '2d70baea0f27d924af09cf3177f630e94133039b286933be0857b5f8766e3245',
    'events.jsonl': 'b9302c78afce78f765502f56382441b5ffe8eb0efb14a32b0d2b19b6234e6df5',
    'predictions_0001.npz': '5086948b40e00995e7ee135fc47f3675b4eae23a3a31577ccbc573e9d6004f17',
    'predictions_0002.npz': '0fd1860a3272d159fbc1b3bfbf056ebd964601a6883778a47616380d09af2e73',
    'predictions_0003.npz': '0586f6bdad32fec3e419d11dc783a5763fe10a46bade085d8bda7694123832ba',
    'predictions_0004.npz': 'e5763ccce49b78c562a9236638f5925fed407a401a0ca3c91d0d3c7df5d7e35e',
    'predictions_0005.npz': '18803564900184b0e6ed676932d04b4e1a323324d7e54dc6659e49980636718d',
    'predictions_0006.npz': '9284606a5db3b1fd1b8b19dd4fe6f99ec78c422beea811fcb4351906e463dc1f',
}
FRAME_IDS = (1305031102.175304, 1305031102.443271, 1305031107.211358,
             1305031111.811406, 1305031116.511205, 1305031121.947288,
             1305031127.879380)
SOURCE_SEQUENCES = (0, 8, 151, 289, 430, 593, 771)
PAIR_COUNT = 6


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def load_protocol():
    directory = Path(__file__).resolve().parent
    for name, expected in PROTOCOL_FILES.items():
        if sha(directory / name) != expected:
            raise ValueError(f'Frozen capture protocol changed: {name}')
    path = directory / 'capture-v17-resolution-pairs-board.py'
    spec = importlib.util.spec_from_file_location('quality_capture_v17_protocol', path)
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    return protocol


def validate_schedule(protocol, schedule, rows, rectification):
    """Validate all saved batches and their source/anchor identity, not quality."""
    schedule = Path(schedule)
    actual = {name: sha(schedule / name) for name in SCHEDULE_FILES}
    if actual != SCHEDULE_FILES:
        raise ValueError('Frozen r11 schedule SHA mismatch')
    expected_names = [f'predictions_{i:04d}.npz' for i in range(1, PAIR_COUNT + 1)]
    if sorted(p.name for p in schedule.glob('predictions_*.npz')) != expected_names:
        raise ValueError('Exactly the six complete r11 prediction files are required')
    if (len(rows) != 798 or any(not math.isfinite(row['timestamp']) for row in rows)
            or any(b['timestamp'] <= a['timestamp'] for a, b in zip(rows, rows[1:]))):
        raise ValueError('The complete 798-frame strictly increasing RGB index is required')
    if rectification.get('calibration_sha256') != CALIBRATION_SHA:
        raise ValueError('Frozen r11 normalized calibration mismatch')
    pairs, information = protocol.plan_pairs(schedule, rows, rectification, PAIR_COUNT)
    if information['selected_pairs'] != PAIR_COUNT or information['available_pairs'] != PAIR_COUNT:
        raise ValueError('Quality capture must include every r11 pair without truncation')
    metrics = json.loads((schedule / 'metrics.json').read_text())
    source = metrics['source']
    if source['original_frames'] != len(rows) or source.get('error') is not None:
        raise ValueError('Frozen source index/count does not match the r11 replay')
    start = source['start_monotonic']
    if isinstance(start, bool) or not isinstance(start, (float, int)) or not math.isfinite(start):
        raise ValueError('Frozen source start must be finite')
    events = [json.loads(line) for line in (schedule / 'events.jsonl').read_text().splitlines()]
    maps = [event for event in events if event.get('kind') == 'map']
    adoptions = [event for event in events if event.get('kind') == 'depth_adoption']
    if len(maps) != PAIR_COUNT or len(adoptions) != PAIR_COUNT:
        raise ValueError('Every recorded r11 pair must have its exact map and anchor record')
    anchors = []
    for index, (pair, mapped, adopted) in enumerate(zip(pairs, maps, adoptions)):
        revision = index + 1
        if pair['frame_ids'] != list(FRAME_IDS[index:index + 2]):
            raise ValueError('Frozen r11 frame IDs or complete pair order changed')
        for offset, image in enumerate(pair['images']):
            token = image['schedule_token']
            sequence = SOURCE_SEQUENCES[index + offset]
            stamp = FRAME_IDS[index + offset]
            # Match TimestampReplay's original FP64 operation order exactly.
            captured = start + stamp - rows[0]['timestamp']
            if (rows[sequence]['timestamp'] != stamp or type(token['sequence']) is not int
                    or token['sequence'] != sequence or token['captured_monotonic'] != captured):
                raise ValueError('Original source sequence/timestamp/capture identity changed')
        token = pair['images'][-1]['schedule_token']
        node = 2 * revision - 1
        if (mapped['revision'] != revision or mapped['graph_node_id'] != node
                or mapped['anchor'] != token or adopted['revision'] != revision
                or adopted['graph_node_id'] != node or adopted['anchor_frame_id'] != token['frame_id']
                or adopted['anchor_captured_monotonic'] != token['captured_monotonic']):
            raise ValueError('Recorded map/depth anchor is not the actual last input of this pair')
        if index and pairs[index - 1]['images'][-1] != pair['images'][0]:
            raise ValueError('Consecutive pairs must share the exact original overlap observation')
        anchors.append(dict(revision=revision, graph_node_id=node, source_sequence=token['sequence'],
                            frame_id=token['frame_id'], captured_monotonic=token['captured_monotonic']))
    information.update(selection='all_six_frozen_r11_pairs_without_quality_filtering',
                       source_frames=798, anchors=anchors, frozen_file_sha256=actual,
                       old_prediction_values_used=False, graph_geometry_used=False)
    return pairs, information


def input_guard(a, protocol):
    if sha(a.rgb_index) != RGB_INDEX_SHA:
        raise ValueError('Frozen original TUM RGB index SHA mismatch')
    sys.path.insert(0, str(SOURCE))
    from vggt_slam.timestamp_replay import read_rgb_index
    rows = read_rgb_index(a.rgb_index)
    calibration = json.loads(a.calibration.read_text())
    normalized = dict(K=[[float(v) for v in row] for row in calibration['K']],
                      distortion=[float(v) for v in calibration['distortion']],
                      image_hw=calibration['image_hw'], source=calibration['source'])
    digest = hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(',', ':'),
                                      ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    if digest != CALIBRATION_SHA or set(calibration) != set(normalized):
        raise ValueError('Requested calibration differs from the frozen r11 calibration')
    rectification = json.loads((a.schedule_run_dir / 'metrics.json').read_text())['common_rectification']
    pairs, information = validate_schedule(protocol, a.schedule_run_dir, rows, rectification)
    return dict(selected_inputs=pairs, schedule=information, rgb_index_sha256=sha(a.rgb_index),
                calibration_file_sha256=sha(a.calibration))


def capture(a):
    started = time.monotonic()
    producer_sha = sha(__file__)
    result = dict(status='started', pairs=[])
    before = None
    environment = dict(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='4',
                       MPLBACKEND='Agg')
    environment_before = {key: os.environ.get(key) for key in environment}
    try:
        # plan_pairs imports NumPy; establish the original capture's BLAS policy
        # before that import, rather than after a library has created its pool.
        os.environ.update(environment)
        protocol = load_protocol()
        before = input_guard(a, protocol)
        a.max_pairs = PAIR_COUNT
        result = protocol.capture(a)  # Original model/preprocessing/FP32 execution, unchanged.
        if result.get('selected_inputs') is not None and result['selected_inputs'] != before['selected_inputs']:
            raise ValueError('Capture selected inputs differ from the complete prevalidated schedule')
        after = input_guard(a, protocol)
        load_protocol()  # Includes both frozen producer files in the after guard.
        if after != before or sha(__file__) != producer_sha:
            raise ValueError('Quality input schedule/source images or helper changed during capture')
        result['quality_inputs_unchanged'] = True
        if result['status'] == 'completed_fixed_input_diagnostic' and len(result['pairs']) != PAIR_COUNT:
            raise ValueError('Completed capture did not save all six original pairs')
    except BaseException as exc:
        if result.get('status') == 'failed':
            result.setdefault('quality_guard_errors', []).append(f'{type(exc).__name__}: {exc}')
        else:
            result.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        if 'script_sha256' in result:
            result['base_capture_script_sha256'] = result['script_sha256']
        result.update(kind='fixed_r11_original_VGGT_quality_only_pairs',
                      experiment_version='v18_quality_only', script_sha256=producer_sha,
                      capture_protocol_sha256=dict(PROTOCOL_FILES), quality_only=True,
                      image_size=a.image_size, max_pairs=PAIR_COUNT,
                      original_weights=True, device='cpu', precision='float32',
                      inference_on_board_only=True, ORB_used=False, npu_used=False,
                      GT_used=False, graph_run=False, online_pipeline_run=False,
                      real_time_validated=False, accuracy_validated=False,
                      full_dataset_ATE_RPE_validated=False, model_batch_frames=2,
                      paper_evaluation_batch_frames=32, reproduces_paper_evaluation=False,
                      baseline_scope='same six project S=2 pairs; not the paper S=32 benchmark',
                      environment_before=environment_before,
                      quality_input_guard=before,
                      quality_capture_total_seconds=time.monotonic() - started)
        (a.output / 'capture.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('rgb-index', 'schedule-run-dir', 'checkpoint', 'calibration', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--image-size', type=int, choices=(112, 336, 518), required=True)
    a = parser.parse_args(argv)
    if platform.system() != 'Linux' or platform.machine().lower() not in ('aarch64', 'arm64'):
        parser.error('Model execution is restricted to the Linux ARM RK3588 board; no host inference')
    compatible = Path('/proc/device-tree/compatible')
    if not compatible.is_file() or b'rk3588' not in compatible.read_bytes():
        parser.error('The board must identify as RK3588/RK3588S')
    for key in ('rgb_index', 'schedule_run_dir', 'checkpoint', 'calibration', 'output'):
        value = getattr(a, key).resolve()
        if not value.is_relative_to(BASE):
            parser.error(f'{key} must be inside the board data root {BASE}')
        setattr(a, key, value)
    if not a.output.is_relative_to(BASE / 'experiments'):
        parser.error('Output must be a new directory under board experiments')
    a.output.mkdir(parents=True, exist_ok=False)
    result = capture(a)
    print(json.dumps(dict(status=result['status'], receipt=str(a.output / 'capture.json'),
                          sha256=sha(a.output / 'capture.json'), error=result.get('error'))))
    return 0 if result['status'] == 'completed_fixed_input_diagnostic' else 1


if __name__ == '__main__':
    sys.exit(main())
