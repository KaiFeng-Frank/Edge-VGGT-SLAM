"""Frozen original VGGT frame0/global0 lineage and logical floating SDK IO.

Functions here inspect JSON, file hashes and saved arrays only. No model, ORT,
SDK or compiler is imported or executed by this module.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

BASE = Path('/mnt/vggt-data/vggt-slam2')
FRAME_SHAPE = (2, 437, 1024)
GLOBAL_SHAPE = (1, 874, 1024)
PAIR_EXPORT_SHA = '149450ffc675ce59950c6f89cf84adb41e388cd050eeb731b6d09a6fe16e9fe1'
PAIR_CONTRACTS_SHA = '4ec6c0e827ab9669c942b589d2c504f0c9de306f76f4f4be0799af14ae322cd7'
PAIR_COMPILER_SHA = '459bbbb81aa43b3ca7562e1f9dc5c946086b276b89f68730123383df84e4ff8a'
PAIR_ORT_SOURCE_SHA = '7b6571ffd74778c05c3dca15305ee248a234b0329bb1e144d7cd83284c2af916'
PARENT_RUNTIME_SHA = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
PARENT_COMPILER_SHA = 'a40bf09e53ef9b378db4781be18f9e53f824afcc809c9009e0a65f4f0f91b4e5'
SDK_LIBRARY_SHA = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
SDK_HEADER_SHA = 'c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a'
CHECKPOINT_SHA = 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
PROFILE_SHA = '6c3d2cb51af75410188c062e83a26eae4d4c5d69c632a6eeae95993eba004820'
PARENTS = {
    'frame0': {
        'export_sha256': '07d139bc51dc5ab244920020e7728e774ee62f055a464c107da292664ae24eaa',
        'ort_sha256': 'e65ea97ef477709bc0a346658fdf7adeca1ff6e4c8efcbeaaffca3076889b3ca',
        'graph_sha256': '20912c4580eec7d3935900437cbd2733013bd1f92229befac48bad5d860cb1ce',
        'compile_sha256': '4d44169d783ea2bdb840d07fb2bac895672c01ce03218e4548c695d2acb81673',
        'model_sha256': '0471b1a39629d61f4ef75285768e537af34862f66ebad6495c9bbe46380ed778',
        'model_bytes': 27989748, 'shape': FRAME_SHAPE,
    },
    'global0': {
        'export_sha256': 'f851432a0dd8e9c2d66b4dfee2a76dfb674836bfb710a2ccbea09972f5e3f33d',
        'ort_sha256': '1929248412b695bffc5e12d89efdca478413c94a80b3ed68b31354df00fbfa87',
        'graph_sha256': 'f67c2210f6a53a7ff88865e4607c6d1760cf23c221941339c204c7414cda2778',
        'compile_sha256': 'ac86341dfde8cc3e8af20c5de0970a87a93f292a1280aeb512c657a0605aa29a',
        'model_sha256': 'ba201c49ba4849e5e4812142bfc92c3dda98ec3fda5566e2a80040389c0375a9',
        'model_bytes': 27411252, 'shape': GLOBAL_SHAPE,
    },
}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def pin(path, expected, pins):
    path = Path(path).resolve(strict=True)
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in '0123456789abcdef' for c in expected)
            or not path.is_file() or sha(path) != expected):
        raise ValueError(f'Frozen SHA mismatch: {path}')
    pins[str(path)] = expected
    return path


def load_json(path, expected, pins):
    return json.loads(pin(path, expected, pins).read_text())


def load_source(path, expected, pins):
    path = pin(path, expected, pins)
    spec = importlib.util.spec_from_file_location('_v48_ab_' + path.stem.replace('-', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw_sha(array):
    return hashlib.sha256(memoryview(array).cast('B')).hexdigest()


def validate_sdk_attr(record, role, direction):
    if role not in ('frame0', 'global0', 'fused') or direction not in ('input', 'output'):
        raise ValueError('Unknown original block SDK IO role')
    shape = FRAME_SHAPE if role == 'frame0' or (role == 'fused' and direction == 'input') else GLOBAL_SHAPE
    identity = (record.get('type') == 1 and record.get('quantization_type') == 2
                and record.get('zero_point') == 0 and record.get('scale') == 1.0)
    if (record.get('index') != 0 or record.get('name') != direction
            or record.get('shape') != list(shape) or record.get('n_elems') != math.prod(shape)
            or record.get('type') not in (0, 1)
            or (record.get('quantization_type') != 0 and not identity)
            or record.get('format') != 3 or record.get('pass_through') != 0
            or not math.isfinite(record.get('scale', float('nan')))):
        raise ValueError(f'Unsupported logical rank-three floating {role}/{direction}: {record}')
    if record.get('size') != math.prod(shape) * (4 if record['type'] == 0 else 2):
        raise ValueError('Logical SDK shape/type byte size mismatch')
    return record


def check_parent(role, export, ort, compiled, pair):
    row = PARENTS[role]
    shape = list(row['shape'])
    if pair['parent_graphs'][role] != {
        'export_sha256': row['export_sha256'], 'onnx_sha256': row['graph_sha256'],
        'ort_sha256': row['ort_sha256'], 'input_shape': shape,
    }:
        raise ValueError('Pair and separate graph lineage differ')
    required = dict(status='exported_and_structurally_checked', profile_sha256=PROFILE_SHA,
                    block=role, input_shape=shape, input_name='input', output_name='output',
                    original_weights=True, parameter_count=18, position_and_tables_fixed=True,
                    host_trigonometry_recomputed=False, host_model_inference_executed=False,
                    host_ORT_executed=False, external_data=False, external_data_used=False)
    if (any(export.get(k) != v for k, v in required.items())
            or export['onnx']['sha256'] != row['graph_sha256']
            or export['architecture'] != pair['architecture']
            or export['checkpoint'] != pair['checkpoint']
            or pair['checkpoint'].get('sha256') != CHECKPOINT_SHA
            or pair['checkpoint'].get('bytes') != 5026874952):
        raise ValueError('Original separate export/checkpoint binding failed')
    original_parameters = {k: v['original'] for k, v in pair['original_parameter_bindings'][role].items()}
    if (len(original_parameters) != 18
            or export['captured_parameter_metadata'] != original_parameters):
        raise ValueError('Separate and fused original parameter identities differ')
    expected_input = 'input.npy' if role == 'frame0' else 'frame-reference.npy'
    expected_output = 'frame-reference.npy' if role == 'frame0' else 'output.npy'
    for direction, pair_key in [('input', expected_input), ('output', expected_output)]:
        if (export['artifacts'][direction + '.npy']['raw_sha256'] != pair['artifacts'][pair_key]['raw_sha256']
                or export['artifacts'][direction + '.npy']['shape'] != shape
                or export['artifacts'][direction + '.npy']['dtype'] != 'float32'):
            raise ValueError('Original separate input/reference order differs from pair')
    if export['captured_array_metadata']['pos']['sha256'] != pair['artifacts']['positions.npy']['raw_sha256']:
        raise ValueError('Original token position order differs from pair')
    if (ort.get('kind') != 'original_VGGT_aggregator_block_CPU_ORT_check'
            or ort.get('status') != 'passed' or ort.get('block') != role
            or ort.get('profile_sha256') != PROFILE_SHA
            or ort.get('export_sha256') != row['export_sha256']
            or ort.get('successful_session_run_count') != 4
            or ort.get('all_runs_within_tolerance') is not True
            or ort.get('session_released') is not True
            or ort.get('input_tensor_unchanged') is not True
            or ort.get('inference_on_board_only') is not True
            or ort.get('tolerance', {}).get('rtol') != 1e-4
            or ort.get('tolerance', {}).get('atol') != 1e-5
            or ort.get('identity', {}).get('checkpoint') != pair['checkpoint']):
        raise ValueError('Original separate board CPU ORT gate evidence failed')
    config = dict(target_platform='rk3588', float_dtype='float16', optimization_level=3, do_quantization=False)
    operations = compiled.get('operations', [])
    if (compiled.get('status') != 'compiled_not_runtime_validated'
            or compiled.get('script_sha256') != PARENT_COMPILER_SHA
            or compiled.get('model_inference_executed') is not False
            or compiled.get('input', {}).get('sha256') != row['graph_sha256']
            or compiled.get('output', {}).get('sha256') != row['model_sha256']
            or compiled.get('output', {}).get('size') != row['model_bytes']
            or compiled.get('released') is not True or compiled.get('release_error')
            or compiled.get('configuration') != config
            or [o.get('name') for o in operations] != ['config', 'load_onnx', 'build', 'export_rknn']
            or any(o.get('code') not in (None, 0) for o in operations)):
        raise ValueError('Original separate nonquantized compilation evidence failed')


def check_fused_ort(row, expected_sha, compiled):
    if (compiled.get('ORT_receipt_sha256') != expected_sha
            or compiled.get('board_ORT_gate_passed') is not True
            or row.get('schema') != 'original_VGGT_pair_CPU_ORT_check_v47'
            or row.get('script_sha256') != PAIR_ORT_SOURCE_SHA
            or row.get('pair_export_sha256') != PAIR_EXPORT_SHA
            or row.get('status') != 'passed' or row.get('strict_FP32_gate_passed') is not True
            or row.get('successful_session_run_count') != 4
            or row.get('session_released') is not True or row.get('cleanup_errors')
            or row.get('source_and_inputs_unchanged') is not True
            or row.get('input_tensor_unchanged') is not True
            or row.get('NPU_used') is not False or row.get('inference_on_board_only') is not True
            or row.get('rtol') != 1e-4 or row.get('atol') != 1e-5):
        raise ValueError('Fused actual board CPU ORT gate must pass before controlled SDK A/B')


def recheck(pins):
    after = {path: sha(path) if Path(path).is_file() else None for path in pins}
    return after, after == pins
