"""Original VGGT board PTQ provenance and dataset preparation, no SDK import."""
from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import sys

BASE = Path('/mnt/vggt-data/vggt-slam2')
INPUT_SHAPE = (2, 437, 1024)
OUTPUT_SHAPE = (1, 874, 1024)
POSITIONS_SHAPE = (2, 437, 2)
PAIR_EXPORT_SHA = '149450ffc675ce59950c6f89cf84adb41e388cd050eeb731b6d09a6fe16e9fe1'
PAIR_GRAPH_SHA = '2cf1f4de19c91b42b5bf685ab5e949bd98b08a64fc2e78b7f0c8f07992a009fc'
PAIR_CONTRACTS_SHA = '4ec6c0e827ab9669c942b589d2c504f0c9de306f76f4f4be0799af14ae322cd7'
BOARD_ORT_SHA = 'f93e1939be9628b7e28df8f91cf06d28002f0b3871a0d0a46adca1780ffa5fc0'
BOARD_ORT_SOURCE_SHA = '7b6571ffd74778c05c3dca15305ee248a234b0329bb1e144d7cd83284c2af916'
CHECKPOINT_SHA = 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
CONFIGURATION = {'target_platform': 'rk3588', 'float_dtype': 'float16', 'optimization_level': 3,
                 'quantized_dtype': 'w8a8', 'quantized_algorithm': 'normal',
                 'quantized_method': 'channel', 'do_quantization': True}
EXPECTED_VERSIONS = {'rknn-toolkit2': '2.3.2', 'torch': '2.2.0', 'numpy': '1.26.4',
                     'onnx': '1.16.1', 'onnxruntime': '1.16.0', 'protobuf': '3.20.3'}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def pin(path, expected, pins):
    path = Path(path).resolve(strict=True)
    if (not path.is_file() or not isinstance(expected, str) or len(expected) != 64
            or any(c not in '0123456789abcdef' for c in expected) or sha(path) != expected):
        raise ValueError(f'Frozen SHA mismatch: {path}')
    pins[str(path)] = expected
    return path


def load_source(path, expected, pins):
    path = pin(path, expected, pins)
    spec = importlib.util.spec_from_file_location('_v49_ptq_' + path.stem.replace('-', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_board():
    if (sys.platform != 'linux' or platform.machine().lower() not in ('aarch64', 'arm64')
            or sys.byteorder != 'little' or ctypes.sizeof(ctypes.c_void_p) != 8):
        raise ValueError('Actual calibration/build requires Linux ARM64 RK3588 with little-endian LP64')
    for path in (Path('/proc/device-tree/compatible'), Path('/sys/firmware/devicetree/base/compatible')):
        if path.is_file():
            value = path.read_bytes()
            if b'rk3588' not in value:
                raise ValueError('Actual calibration/build requires RK3588 device-tree identity')
            return value.replace(b'\0', b',').decode(errors='replace')
    raise ValueError('RK3588 device-tree identity unavailable')


def check_versions(versions, python_version, prefix, expected_prefix):
    if tuple(python_version[:2]) != (3, 11):
        raise ValueError('Use isolated CPython 3.11 ARM64 board toolchain')
    if (Path(prefix).resolve() != Path(expected_prefix).resolve()
            or Path(prefix).resolve() == BASE / 'venv'):
        raise ValueError('PTQ toolchain must be the explicitly selected isolated prefix')
    if versions.get('rknn-toolkit2') != '2.3.2':
        raise ValueError('Original pair PTQ requires RKNN-Toolkit2 2.3.2')
    value = versions.get('torch', '')
    match = re.fullmatch(r'(\d+)\.(\d+)\.(\d+)(?:[+.-][A-Za-z0-9_.+-]+)?', value)
    if not match or tuple(int(x) for x in match.groups()) > (2, 2, 0):
        raise ValueError('Use isolated Torch <= 2.2.0; production Torch 2.3.1 is excluded')
    if any(versions.get(name, '').split('+', 1)[0] != expected for name, expected in EXPECTED_VERSIONS.items()):
        raise ValueError('Use the explicitly frozen RKNN 2.3.2 ARM64 dependency versions')
    if any(not isinstance(versions.get(name), str) or not versions[name]
           for name in ('numpy', 'onnx', 'onnxruntime', 'protobuf')):
        raise ValueError('All actual toolkit dependency versions must be recorded')


def check_board_ort(row):
    if (row.get('schema') != 'original_VGGT_pair_CPU_ORT_check_v47'
            or row.get('script_sha256') != BOARD_ORT_SOURCE_SHA
            or row.get('pair_export_sha256') != PAIR_EXPORT_SHA
            or row.get('status') != 'passed' or row.get('strict_FP32_gate_passed') is not True
            or row.get('successful_session_run_count') != 4
            or row.get('session_released') is not True or row.get('cleanup_errors')
            or row.get('source_and_inputs_unchanged') is not True
            or row.get('input_tensor_unchanged') is not True
            or row.get('NPU_used') is not False or row.get('inference_on_board_only') is not True
            or row.get('rtol') != 1e-4 or row.get('atol') != 1e-5):
        raise ValueError('Frozen paired graph passed board CPU ORT gate required')


def artifact_path(capture_dir, item):
    relative = Path(item['file'])
    if relative.is_absolute() or '..' in relative.parts or any(c.isspace() for c in str(relative)):
        raise ValueError('Capture artifact must use a confined relative path without whitespace')
    path = (Path(capture_dir).resolve() / relative).resolve(strict=True)
    if not path.is_relative_to(Path(capture_dir).resolve()) or not path.is_file():
        raise ValueError('Captured artifact path escapes its original capture directory')
    if any(c.isspace() for c in str(path)):
        raise ValueError('SDK dataset input path cannot contain whitespace')
    return path


def validate_saved_array(path, item, shape, dtype):
    import numpy as np
    value = np.load(path, allow_pickle=False, mmap_mode='r')
    if (value.shape != shape or value.dtype != np.dtype(dtype) or not value.flags.c_contiguous
            or item['shape'] != list(shape) or item['dtype'] != str(np.dtype(dtype))
            or value.nbytes != item['tensor_bytes'] or not np.isfinite(value).all()
            or hashlib.sha256(memoryview(value).cast('B')).hexdigest() != item['raw_sha256']):
        raise ValueError('Actual finite original saved tensor metadata/raw byte identity mismatch')


def prepare_dataset(capture_dir, capture, pair, output_dir, pins):
    """Capture loader must first verify its complete frozen original schema."""
    frozen = capture.get('frozen_pair', {})
    if (frozen.get('export_sha256') != PAIR_EXPORT_SHA or frozen.get('onnx_sha256') != PAIR_GRAPH_SHA
            or frozen.get('original_input_raw_sha256') != pair['artifacts']['input.npy']['raw_sha256']
            or frozen.get('original_positions_raw_sha256') != pair['artifacts']['positions.npy']['raw_sha256']
            or capture.get('checkpoint', {}).get('sha256') != CHECKPOINT_SHA
            or capture.get('checkpoint', {}).get('bytes') != 5026874952):
        raise ValueError('Capture must bind the same original pair graph/checkpoint')
    samples = capture.get('samples', [])
    calib = [sample for sample in samples if sample.get('split') == 'calibration']
    heldout = [sample for sample in samples if sample.get('split') == 'evaluation']
    if len(samples) != 10 or len(calib) != 8 or len(heldout) != 2:
        raise ValueError('Exactly eight distinct calibration and two separate heldout pairs required')
    rows = []
    identities = {'calibration': set(), 'evaluation': set()}
    images = {'calibration': set(), 'evaluation': set()}
    ids = set()
    for sample in samples:
        split, sample_id = sample['split'], sample['sample_id']
        if not isinstance(sample_id, str) or not sample_id or sample_id in ids:
            raise ValueError('Unique original sample_id required')
        ids.add(sample_id)
        item = sample['artifacts']['input.npy']
        if item['shape'] != list(INPUT_SHAPE) or item['dtype'] != 'float32':
            raise ValueError('Original rank-three FP32 frame0 input required without conversion')
        raw = item['raw_sha256']
        if raw in identities[split] or raw == frozen['original_input_raw_sha256']:
            raise ValueError('Calibration/heldout cannot repeat another input or frozen evaluation input')
        identities[split].add(raw)
        position = sample['artifacts']['positions.npy']
        if (position['shape'] != list(POSITIONS_SHAPE) or position['dtype'] != 'int64'
                or position['raw_sha256'] != frozen['original_positions_raw_sha256']):
            raise ValueError('Every sample must retain the original token position order')
        image_rows = sample['images']
        if len(image_rows) != 2 or len({image['sha256'] for image in image_rows}) != 2:
            raise ValueError('Two distinct original source images required for each pair')
        image_ids = {image['sha256'] for image in image_rows}
        if images[split] & image_ids:
            raise ValueError('Source image overlap within a calibration/heldout split')
        images[split].update(image_ids)
        path = artifact_path(capture_dir, item)
        pin(path, item['sha256'], pins)
        if path.stat().st_size != item['bytes']:
            raise ValueError('Original captured input file byte count mismatch')
        validate_saved_array(path, item, INPUT_SHAPE, 'float32')
        position_path = artifact_path(capture_dir, position)
        pin(position_path, position['sha256'], pins)
        if position_path.stat().st_size != position['bytes']:
            raise ValueError('Original captured position file byte count mismatch')
        validate_saved_array(position_path, position, POSITIONS_SHAPE, 'int64')
        if split == 'calibration':
            if set(sample['artifacts']) != {'input.npy', 'positions.npy'}:
                raise ValueError('Calibration samples contain only original frame0 input and position')
            rows.append({'sample_id': sample_id, 'path': str(path), 'input_sha256': item['sha256'],
                         'input_raw_sha256': raw, 'input_shape': list(INPUT_SHAPE)})
        elif set(sample['artifacts']) != {'input.npy', 'positions.npy', 'frame-reference.npy', 'output.npy'}:
            raise ValueError('Heldout pairs require actual original frame0/global0 references')
    if identities['calibration'] & identities['evaluation'] or images['calibration'] & images['evaluation']:
        raise ValueError('Calibration/heldout images and actual input tensors must be disjoint')
    if (images['calibration'] | images['evaluation']) & {image['sha256'] for image in pair['selected_pair']['images']}:
        raise ValueError('New calibration/heldout must exclude frozen graph evaluation source images')
    dataset = Path(output_dir) / 'dataset.txt'
    content = ''.join(row['path'] + '\n' for row in rows)
    with dataset.open('x', encoding='utf-8') as stream:
        stream.write(content)
    dataset_sha = sha(dataset)
    pins[str(dataset.resolve())] = dataset_sha
    return dataset, {'file': dataset.name, 'sha256': dataset_sha, 'bytes': dataset.stat().st_size,
                     'input_count': 1, 'line_count': 8, 'rows': rows,
                     'input_layout': 'original [2,437,1024] unchanged; no transpose or extra batch dimension',
                     'heldout_samples_excluded': [sample['sample_id'] for sample in heldout]}


def recheck(pins):
    after = {path: sha(path) if Path(path).is_file() else None for path in pins}
    return after, after == pins
