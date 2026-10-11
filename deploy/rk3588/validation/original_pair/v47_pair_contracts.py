"""Fixed original VGGT frame0/global0 pair provenance and floating SDK IO."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys

INPUT_SHAPE = (2, 437, 1024)
OUTPUT_SHAPE = (1, 874, 1024)
EXPORTER_SHA = 'ab8e1548369d402448c29be19bfa6c47f7466f27a02f8f6c753fe5ebb5ed89f0'
PROFILE_SHA = '6c3d2cb51af75410188c062e83a26eae4d4c5d69c632a6eeae95993eba004820'
GRAPH_SHA = '2cf1f4de19c91b42b5bf685ab5e949bd98b08a64fc2e78b7f0c8f07992a009fc'
COMPILER_PARENT_SHA = 'a40bf09e53ef9b378db4781be18f9e53f824afcc809c9009e0a65f4f0f91b4e5'
RUNTIME_PARENT_SHA = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
LIBRARY_SHA = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
BASE = Path('/mnt/vggt-data/vggt-slam2')
RTOL, ATOL = 1e-4, 1e-5


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def pin(path, expected, pins):
    if (not isinstance(expected, str) or len(expected) != 64
            or any(x not in '0123456789abcdef' for x in expected) or sha(path) != expected):
        raise ValueError(f'Frozen SHA mismatch: {path}')
    pins[str(Path(path).resolve())] = expected


def load_helper(path, expected):
    pin(path, expected, {})
    name = '_v47_private_' + Path(path).stem.replace('-', '_')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_board():
    if sys.platform != 'linux' or platform.machine().lower() not in ('aarch64', 'arm64'):
        raise ValueError('Original VGGT probe runs only on Linux RK3588S ARM64')
    for path in (Path('/proc/device-tree/compatible'), Path('/sys/firmware/devicetree/base/compatible')):
        if path.exists() and b'rk3588' in path.read_bytes():
            return
    raise ValueError('RK3588 device-tree binding missing')


def recheck(pins):
    after = {path: sha(path) if Path(path).is_file() else None for path in pins}
    return after, after == pins


def load_pair(directory, expected_sha):
    """Only saved NPY/JSON/hash checks; no ONNX, Torch, image or SDK execution."""
    import numpy as np
    directory = Path(directory).resolve()
    pins = {}
    pin(directory / 'export.json', expected_sha, pins)
    report = json.loads((directory / 'export.json').read_text())
    required = {'schema': 'v47_original_VGGT_frame0_global0_static_pair',
                'status': 'exported_and_structurally_checked_not_numerically_validated',
                'script_sha256': EXPORTER_SHA, 'profile_sha256': PROFILE_SHA,
                'candidate_accepted': False, 'paired_board_FP32_gate_passed': None,
                'host_model_inference_executed': False, 'host_ORT_executed': False,
                'original_parameter_tensors': 36, 'original_parameter_FP32_bytes': 100788224,
                'original_weights_preserved': True, 'source_and_inputs_unchanged': True,
                'historical_RKNN_FP32_rejections_retained': True,
                'input_shape': list(INPUT_SHAPE), 'output_shape': list(OUTPUT_SHAPE),
                'input_name': 'input', 'output_name': 'output', 'node_count': 87, 'initializer_count': 63}
    if any(report.get(k) != v for k, v in required.items()):
        raise ValueError('Original pair export contract mismatch')
    if (report.get('onnx', {}).get('file') != 'pair.onnx'
            or report['onnx'].get('sha256') != GRAPH_SHA
            or report['onnx'].get('bytes') != 101700877):
        raise ValueError('Original pair graph identity mismatch')
    graph = directory / 'pair.onnx'
    pin(graph, GRAPH_SHA, pins)
    if graph.stat().st_size != report['onnx']['bytes']:
        raise ValueError('Pair ONNX byte count mismatch')
    arrays = {}
    shapes = {'input.npy': INPUT_SHAPE, 'output.npy': OUTPUT_SHAPE,
              'frame-reference.npy': INPUT_SHAPE, 'positions.npy': (2, 437, 2)}
    if set(report.get('artifacts', {})) != set(shapes):
        raise ValueError('All four original input/position/reference artifacts required')
    for name, shape in shapes.items():
        item = report['artifacts'][name]
        path = directory / name
        pin(path, item['sha256'], pins)
        value = np.load(path, allow_pickle=False)
        dtype = np.dtype('int64' if name == 'positions.npy' else 'float32')
        if (item['file'] != name or path.stat().st_size != item['bytes']
                or value.dtype != dtype or value.shape != shape or not value.flags.c_contiguous
                or not np.isfinite(value).all() or item['shape'] != list(shape)
                or item['dtype'] != str(dtype) or item['tensor_bytes'] != value.nbytes
                or hashlib.sha256(memoryview(value).cast('B')).hexdigest() != item['raw_sha256']):
            raise ValueError(f'Original saved array metadata mismatch: {name}')
        arrays[name] = value
    if (report['captured_connection']['raw_sha256']
            != report['artifacts']['frame-reference.npy']['raw_sha256']):
        raise ValueError('Original sequential frame-global view binding mismatch')
    return report, arrays, graph, pins


def compare(actual, reference):
    import numpy as np
    if (actual.dtype != np.float32 or actual.shape != reference.shape
            or not np.isfinite(actual).all() or not np.isfinite(reference).all()):
        raise ValueError('Finite FP32 output with exact reference shape required')
    a, r = actual.astype(np.float64), reference.astype(np.float64)
    diff = a - r
    absolute, tolerance = np.abs(diff), ATOL + RTOL * np.abs(r)
    rn, an = float(np.linalg.norm(r)), float(np.linalg.norm(a))
    return {'finite': True, 'array_equal': bool(np.array_equal(actual, reference)),
            'max_abs_error': float(absolute.max()), 'rmse': float(np.sqrt(np.mean(diff * diff))),
            'relative_l2': float(np.linalg.norm(diff) / rn) if rn else None,
            'cosine_similarity': float(np.sum(a * r) / (rn * an)) if rn and an else None,
            'rtol': RTOL, 'atol': ATOL, 'strict_FP32_gate_passed': bool(np.all(absolute <= tolerance)),
            'outside_tolerance_elements': int(np.count_nonzero(absolute > tolerance)),
            'max_tolerance_ratio': float(np.max(absolute / tolerance)),
            'output_raw_sha256': hashlib.sha256(memoryview(actual).cast('B')).hexdigest()}


def validate_sdk_attr(record):
    role = record.get('name')
    if role not in ('input', 'output'):
        raise ValueError('SDK logical IO must name the original pair input/output')
    shape = INPUT_SHAPE if role == 'input' else OUTPUT_SHAPE
    identity = (record.get('type') == 1 and record.get('quantization_type') == 2
                and record.get('zero_point') == 0 and record.get('scale') == 1.0)
    if (record.get('shape') != list(shape) or record.get('n_elems') != math.prod(shape)
            or record.get('type') not in (0, 1)
            or (record.get('quantization_type') != 0 and not identity)
            or record.get('format') != 3
            or not math.isfinite(record.get('scale', float('nan')))):
        raise ValueError(f'Unsupported logical rank-three floating SDK descriptor: {record}')
    if record.get('size') != math.prod(shape) * (4 if record['type'] == 0 else 2):
        raise ValueError('SDK floating logical byte count mismatch')
    return record


def check_compile(row, export_sha, model_sha):
    if (row.get('schema') != 'v47_original_VGGT_pair_RKNN_compilation'
            or row.get('status') != 'compiled_not_runtime_validated'
            or row.get('pair_export_sha256') != export_sha
            or row.get('parent_compiler_sha256') != COMPILER_PARENT_SHA
            or row.get('input', {}).get('sha256') != GRAPH_SHA
            or row.get('output', {}).get('sha256') != model_sha
            or row.get('model_inference_executed') is not False
            or row.get('released') is not True or row.get('cleanup_errors')
            or row.get('source_and_inputs_unchanged') is not True
            or row.get('candidate_accepted') is not False
            or row.get('configuration') != {'target_platform': 'rk3588', 'float_dtype': 'float16',
                                            'optimization_level': 3, 'do_quantization': False}):
        raise ValueError('New original pair non-quantized compile provenance mismatch')
    operations = row.get('operations', [])
    if ([r.get('name') for r in operations] != ['config', 'load_onnx', 'build', 'export_rknn']
            or any(r.get('code') not in (None, 0) for r in operations)):
        raise ValueError('Compiler did not complete all four original operations')
