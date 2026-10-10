#!/usr/bin/env python3
"""Isolated DA3-SMALL three-output RKNN board probe, with official CPU codec.

One normalized FP32 NCHW two-view input enters the learned RKNN graph. All
three SDK-owned outputs are copied and released before the original DA3
quaternion/FOV codec runs on CPU. This is a different model from VGGT and a
fixed-input experiment, not a SLAM, quality, or real-time validation.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

sys.dont_write_bytecode = True
BASE = Path('/mnt/vggt-data/vggt-slam2')
SHAPE = (2, 3, 252, 336)
OUTPUTS = [dict(name='depth', reference_key='depth', shape=[2, 252, 336]),
           dict(name='depth_conf', reference_key='conf', shape=[2, 252, 336]),
           dict(name='pose_enc', reference_key='pose', shape=[2, 9])]
ABI_SHA = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
HEADER_SHA = 'c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a'
COMPILER_SHA = 'a40bf09e53ef9b378db4781be18f9e53f824afcc809c9009e0a65f4f0f91b4e5'
LIBRARY_SHA = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
REFERENCE_SHA = '8a5b9e8a7f94757fa906fbcc1b2d37ae3d4751076d69e966dd626a15a6c63598'
INPUT_MANIFEST_SHA = '3a104c1f30cf3ed97294852a1d8cabccaf4c8a38d2b8297086eb1503c79a65a2'
COMMIT = '3d835ec1a5802d64a8b8b15f817a1ab54809bfe4'
WEIGHTS_SHA = '364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf'
CONFIG_SHA = 'a486e29e82b7ab4a7d4cefc1ea4526cfe2ae438a572c8ca98917cfbcde7447d2'
RTOL, ATOL = 1e-4, 1e-5


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def pin(path, expected, pins):
    path = Path(path).resolve(strict=True)
    require(isinstance(expected, str) and len(expected) == 64
            and all(c in '0123456789abcdef' for c in expected), 'Invalid SHA256 pin')
    require(sha(path) == expected, 'SHA256 mismatch: ' + path.name)
    require(str(path) not in pins or pins[str(path)] == expected, 'Conflicting source pin')
    pins[str(path)] = expected
    return path


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (
        _ for _ in ()).throw(ValueError('Nonfinite JSON: ' + value)))


def load_module(path, expected, name, pins):
    path = pin(path, expected, pins)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def metadata(value):
    value = np.ascontiguousarray(value)
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=int(value.nbytes),
                tensor_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())


def validate_array(value, info, shape, dtype=np.float32):
    require(value.dtype == dtype and value.shape == tuple(shape) and np.isfinite(value).all(),
            'Array dtype/shape/finite mismatch')
    actual = metadata(value)
    require(all(info.get(key) == actual[key] for key in ('shape', 'dtype', 'bytes')),
            'Array metadata header mismatch')
    digests = [info[k] for k in ('sha256', 'tensor_sha256') if k in info]
    require(digests and all(d == actual['tensor_sha256'] for d in digests), 'Array raw SHA mismatch')


def comparison(actual, expected):
    require(actual.shape == expected.shape and actual.dtype == expected.dtype == np.float32
            and np.isfinite(actual).all() and np.isfinite(expected).all(), 'Finite matching FP32 outputs required')
    a, b = actual.astype(np.float64).ravel(), expected.astype(np.float64).ravel()
    delta = a - b
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    close = np.isclose(actual, expected, rtol=RTOL, atol=ATOL)
    return dict(max_abs=float(np.max(np.abs(delta))), rms=float(np.sqrt(np.mean(delta * delta))),
                relative_l2=float(np.linalg.norm(delta) / norm_b) if norm_b else None,
                cosine=float(np.dot(a / norm_a, b / norm_b)) if norm_a and norm_b else None,
                array_equal=bool(np.array_equal(actual, expected)),
                C_order_bytes_equal=actual.tobytes(order='C') == expected.tobytes(order='C'),
                element_count=int(close.size), failed_elements=int(np.count_nonzero(~close)),
                elementwise_gate_passed=bool(close.all()), rtol=RTOL, atol=ATOL,
                actual_all_zero=bool(np.all(actual == 0)), finite=True)


def local_artifact(root, item, pins):
    name = item['file']
    require(isinstance(name, str) and Path(name).name == name, 'Simple local artifact filename required')
    path = pin(Path(root) / name, item['sha256'], pins)
    require(path.stat().st_size == item['bytes'], 'Artifact file size mismatch')
    return path


def validate_attribute(abi, attr, name, logical_shape, *, is_input):
    value = abi.attr_record(attr)
    require(value['name'] == name, 'RKNN tensor name/order mismatch: ' + str(value))
    require(value['type'] in (0, 1) and (value['quantization_type'] == 0 or (
        value['type'] == 1 and value['quantization_type'] == 2
        and value['zero_point'] == 0 and value['scale'] == 1.0)), 'Only unquantized FP32/FP16 descriptors supported')
    require(math.isfinite(value['scale']), 'Nonfinite SDK scale')
    require(value['n_elems'] == math.prod(logical_shape)
            and value['size'] == value['n_elems'] * (4 if value['type'] == 0 else 2),
            'SDK tensor elements/logical bytes mismatch')
    if is_input:
        permitted = {0: logical_shape, 1: (logical_shape[0], logical_shape[2], logical_shape[3], logical_shape[1])}
        require(value['format'] in permitted and tuple(value['shape']) == permitted[value['format']],
                'Only explicit NCHW or explicitly packed native NHWC input supported')
    else:
        require(value['format'] in (0, 3) and tuple(value['shape']) == tuple(logical_shape),
                'Output must have the exact logical shape and NCHW/undefined layout')
    return value


def pack_input(images, native_format):
    require(images.dtype == np.float32 and images.shape == SHAPE and np.isfinite(images).all(),
            'Normalized FP32 fixed-shape input required')
    if native_format == 1:
        return np.array(images.transpose(0, 2, 3, 1), copy=True, order='C')
    require(native_format == 0, 'Unsupported input layout')
    return np.array(images, copy=True, order='C')


def execute_multi(api, abi, context, images, native_format, timeout_ms, cleanup, activity):
    """One blocking SDK call; independently owned copies of all three outputs.

    Includes input packing, SDK set/run/get, copies and release in sdk_io_seconds.
    Release is attempted for any partial allocation, even when outputs_get fails.
    A cleanup failure never masks an already active primary exception.
    """
    start = time.perf_counter()
    packed = pack_input(images, native_format)
    packed_at = time.perf_counter()
    inputs = (abi.Input * 1)(abi.Input(0, packed.ctypes.data, packed.nbytes, 0, 0, native_format))
    outputs = (abi.Output * 3)()
    for i in range(3):
        outputs[i].want_float, outputs[i].is_prealloc, outputs[i].index = 1, 0, i
    extension = abi.RunExtend(0, 0, timeout_ms, -1)
    received = abi.OutputExtend()
    row = dict(input_api_format=native_format, input_pass_through=0, input_api_dtype='float32',
               input_logical_shape=list(images.shape), input_buffer_shape=list(packed.shape),
               input_packing_seconds=packed_at - start, output_api_want_float=1,
               synchronous=True, run_timeout_ms=timeout_ms)
    got = False
    try:
        abi.check_call(api.rknn_inputs_set(context, 1, inputs), 'rknn_inputs_set')
        set_at = time.perf_counter()
        activity['sdk_run_api_calls'] = activity.get('sdk_run_api_calls', 0) + 1
        abi.check_call(api.rknn_run(context, ct.byref(extension)), 'rknn_run')
        run_at = time.perf_counter()
        abi.check_call(api.rknn_outputs_get(context, 3, outputs, ct.byref(received)), 'rknn_outputs_get')
        got = True
        get_at = time.perf_counter()
        require(extension.frame_id == received.frame_id, 'SDK run/get frame identity mismatch')
        actual = []
        for value, expected in zip(outputs, OUTPUTS):
            elems = math.prod(expected['shape'])
            require(bool(value.buf) and value.size == elems * 4, 'Invalid SDK FP32 output buffer size/pointer')
            view = np.ctypeslib.as_array(ct.cast(value.buf, ct.POINTER(ct.c_float)), shape=(elems,))
            actual.append(view.copy().reshape(expected['shape']))
        copied_at = time.perf_counter()
        row.update(input_set_seconds=set_at - packed_at, run_api_seconds=run_at - set_at,
                   outputs_get_seconds=get_at - run_at, output_copy_seconds=copied_at - get_at,
                   runtime_frame_id=int(extension.frame_id), output_bytes=[int(o.size) for o in outputs])
    finally:
        if got or any(o.buf for o in outputs):
            active_error = sys.exc_info()[0] is not None
            release_at = time.perf_counter()
            try:
                code = api.rknn_outputs_release(context, 3, outputs)
                row.update(outputs_release_seconds=time.perf_counter() - release_at,
                           output_buffers_released=code == 0, sdk_io_seconds=time.perf_counter() - start)
                if code:
                    cleanup.append('rknn_outputs_release returned ' + str(code))
                    if not active_error:
                        abi.check_call(code, 'rknn_outputs_release')
            except Exception as error:
                if not cleanup or cleanup[-1] != 'rknn_outputs_release returned ' + str(locals().get('code')):
                    cleanup.append('rknn_outputs_release: ' + repr(error))
                if not active_error:
                    raise
    return actual, row


def validate_outputs(actual):
    require(len(actual) == 3, 'Three output arrays required')
    for value, spec in zip(actual, OUTPUTS):
        require(value.shape == tuple(spec['shape']) and value.dtype == np.float32
                and np.isfinite(value).all() and np.any(value != 0),
                'Nonfinite or all-zero SDK output: ' + spec['name'])
    require(np.all(actual[0] > 0) and np.all(actual[1] >= 1), 'Positive depth and exp+1 confidence required')
    require(np.all(np.sum(actual[2][:, 3:7].astype(np.float64) ** 2, axis=1) > 1e-12),
            'Invalid DA3 quaternion cannot be decoded')


def canonical_output(actual, codec):
    validate_outputs(actual)
    start = time.perf_counter()
    E, K = codec(actual[2])
    arrays = dict(depth=actual[0][None, ..., None], depth_conf=actual[1][None], pose_enc=actual[2][None],
                  extrinsic=E[None], intrinsic=K[None])
    seconds = time.perf_counter() - start
    expected = dict(depth=(1, 2, 252, 336, 1), depth_conf=(1, 2, 252, 336), pose_enc=(1, 2, 9),
                    extrinsic=(1, 2, 3, 4), intrinsic=(1, 2, 3, 3))
    for name, value in arrays.items():
        require(value.dtype == np.float32 and value.shape == expected[name] and np.isfinite(value).all(),
                'Canonical official codec output mismatch: ' + name)
    return arrays, seconds


def prepare_codec(source_dir, common):
    common.install_official_import_path(source_dir)
    import torch
    from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
    from depth_anything_3.utils.geometry import affine_inverse

    def codec(pose):
        with torch.no_grad(), torch.autocast(device_type='cpu', enabled=False):
            c2w, K = pose_encoding_to_extri_intri(torch.from_numpy(pose).unsqueeze(0), (252, 336))
            E = affine_inverse(c2w)
            return E.squeeze(0).numpy().copy(), K.squeeze(0).numpy().copy()
    return codec


def load_inputs(args, common, pins):
    """Bind external CPU capture, export, ORT gate, compiled model and raw pixels."""
    exported = read_json(pin(args.export_dir / 'export.json', args.export_sha256, pins))
    compiled = read_json(pin(args.compile_dir / 'compile.json', args.compile_sha256, pins))
    ort = read_json(pin(args.ort_dir / 'ort.json', args.ort_sha256, pins))
    captured = read_json(pin(args.capture_dir / 'capture.json', args.capture_sha256, pins))
    inputs = read_json(pin(args.input_manifest, INPUT_MANIFEST_SHA, pins))
    require(exported['status'] == 'exported_not_runtime_validated'
            and exported['script_sha256'] == args.exporter_sha256 and exported['outputs'] == OUTPUTS
            and exported['input'] == dict(name='images', shape=list(SHAPE), dtype='float32', layout='NCHW; N=view count')
            and exported['external_data_used'] is False and exported['original_weights_unchanged'] is True,
            'Matching original DA3 fixed-shape export required')
    require(ort['status'] == 'completed' and ort['exact_graph_reference_gate_passed'] is True
            and ort['export_sha256'] == args.export_sha256 and ort['script_sha256'] == args.exporter_sha256
            and ort['onnx'] == exported['onnx'] and ort['outputs'] == OUTPUTS
            and ort['CPUExecutionProvider_only'] is True, 'Bound three-pair CPU ORT gate required')
    require(compiled['status'] == 'compiled_not_runtime_validated' and compiled['released'] is True
            and not compiled.get('release_error') and compiled['script_sha256'] == COMPILER_SHA
            and compiled['input']['sha256'] == exported['onnx']['sha256']
            and compiled['configuration'] == dict(target_platform='rk3588', float_dtype='float16',
                optimization_level=3, do_quantization=False)
            and [o['name'] for o in compiled['operations']] == ['config', 'load_onnx', 'build', 'export_rknn']
            and all(o['code'] in (0, None) for o in compiled['operations'])
            and compiled['versions']['rknn-toolkit2'] == '2.3.2', 'Successful pinned compiler path required')
    require(captured['status'] == 'completed_fixed_input_diagnostic'
            and captured['model_name'] == 'da3-small' and captured['official_commit'] == COMMIT
            and captured['source_code_sha256']['run-da3-small-board.py'] == args.capture_script_sha256
            and captured['source_code_sha256']['da3_probe_common.py'] == args.common_sha256
            and captured['checkpoint']['sha256'] == WEIGHTS_SHA and captured['checkpoint']['bytes'] == 137248940
            and captured['config_sha256'] == CONFIG_SHA and captured['npu_used'] is False
            and captured['GT_used'] is False and captured['ORB_used'] is False
            and captured['camera_conditions'] == dict(extrinsics=None, intrinsics=None)
            and captured['use_ray_pose'] is False and captured['precision'] == 'float32'
            and captured['input_manifest_sha256'] == INPUT_MANIFEST_SHA
            and captured['reference_capture_sha256'] == REFERENCE_SHA,
            'Unconditioned pinned external CPU reference required')
    norm = captured['normalization']
    require(norm['mean'] == [.485, .456, .406] and norm['std'] == [.229, .224, .225]
            and norm['applications'] == 1 and norm['uint8_roundtrip'] is False
            and captured['geometric_preprocessing_changed'] is False
            and captured['selected_inputs'] == inputs['pairs'] and inputs['reference_sha256'] == REFERENCE_SHA,
            'Same original RGB geometry and one official normalization required')
    require(len(captured['records']) == len(exported['pairs']) == len(ort['pairs']) == len(inputs['pairs']) == 3,
            'All three fixed pairs required')
    for receipt in (exported, ort):
        require(receipt['sources_and_inputs_unchanged'] is True and not receipt['cleanup_errors'],
                'Export/ORT source or cleanup validation failed')
        for path, digest in receipt['source_and_input_hashes'].items():
            pin(path, digest, pins)
    require(captured['source_unchanged'] is True and captured['weights_unchanged'] is True
            and captured['model_released'] is True and not captured['cleanup_errors'],
            'CPU reference source/state/cleanup validation failed')
    provenance = exported['official_model_provenance']
    require(provenance['official_commit'] == COMMIT and provenance['checkpoint']['sha256'] == WEIGHTS_SHA
            and provenance['checkpoint']['bytes'] == 137248940
            and provenance['released_model_config']['sha256'] == CONFIG_SHA,
            'Export weights/config provenance differs from external CPU reference')
    checked_source = common.verify_official_source(args.source_dir, args.source_manifest, args.source_manifest_sha256)
    require(checked_source['source_hashes'] == provenance['source_hashes']
            == captured['load_provenance']['source_hashes'], 'Official source closures differ')
    for relative, digest in checked_source['source_hashes'].items():
        pin(args.source_dir / relative, digest, pins)
    pin(args.source_manifest, args.source_manifest_sha256, pins)
    pin(args.source_dir.parent / 'model-metadata/config.json', CONFIG_SHA, pins)
    model = pin(args.compile_dir / Path(compiled['output']['path']).name, compiled['output']['sha256'], pins)
    require(model.stat().st_size == compiled['output']['size'] and 0 < model.stat().st_size <= 512 * 1024 ** 2,
            'Compiled model byte size mismatch or exceeds isolated 512MiB limit')
    local_artifact(args.export_dir, exported['onnx'], pins)
    rows = []
    for i, (seed, exp, cpu, check) in enumerate(zip(inputs['pairs'], exported['pairs'], captured['records'], ort['pairs'])):
        require(seed['pair_index'] == exp['index'] == cpu['pair_index'] == check['index'] == i
                and seed['frame_ids'] == cpu['frame_ids'] and len(seed['frame_ids']) == 2,
                'Exact pair source/sequence identity mismatch')
        for source in seed['source_images']:
            pin(source['path'], source['sha256'], pins)
        raw_path = pin(seed['images'], seed['sha256'], pins)
        require(raw_path.stat().st_size == seed['bytes'], 'Original RGB file bytes differ')
        raw = np.load(raw_path, allow_pickle=False)
        validate_array(raw, cpu['input_tensor'], SHAPE)
        input_item = exported['artifacts'][exp['input_artifact']]
        image_path = local_artifact(args.export_dir, input_item, pins)
        images = np.load(image_path, allow_pickle=False)
        validate_array(images, input_item['array'], SHAPE)
        normalized = cpu['normalized_input_tensor']
        normalized_path = pin(args.capture_dir / normalized['file'], normalized['file_sha256'], pins)
        external_images = np.load(normalized_path, allow_pickle=False)
        validate_array(external_images, normalized, (1,) + SHAPE)
        require(images.tobytes(order='C') == external_images[0].tobytes(order='C'),
                'Export and external CPU reference normalized input bytes differ')
        cpu_path = local_artifact(args.capture_dir, cpu['artifact'], pins)
        with np.load(cpu_path, allow_pickle=False) as saved:
            canonical = {k: saved[k].copy() for k in ('depth', 'depth_conf', 'pose_enc', 'extrinsic', 'intrinsic')}
            require(saved['frame_ids'].dtype == np.float64 and np.array_equal(saved['frame_ids'], seed['frame_ids']),
                    'CPU canonical source IDs mismatch')
        expected_shapes = dict(depth=(1, 2, 252, 336, 1), depth_conf=(1, 2, 252, 336),
                               pose_enc=(1, 2, 9), extrinsic=(1, 2, 3, 4), intrinsic=(1, 2, 3, 3))
        for key, value in canonical.items():
            validate_array(value, cpu['arrays'][key], expected_shapes[key])
        reference_item = exported['artifacts'][exp['reference_artifact']]
        reference_path = local_artifact(args.export_dir, reference_item, pins)
        with np.load(reference_path, allow_pickle=False) as saved:
            baseline = {k: saved[k].copy() for k in ('depth', 'conf', 'pose', 'extrinsic', 'intrinsic')}
        for item in OUTPUTS:
            validate_array(baseline[item['reference_key']], reference_item['arrays'][item['reference_key']], item['shape'])
        require(all(comparison(baseline[k], canonical[c][0, ..., 0] if c == 'depth' else canonical[c][0])
                    ['elementwise_gate_passed'] for k, c in (('depth', 'depth'), ('conf', 'depth_conf'),
                    ('pose', 'pose_enc'), ('extrinsic', 'extrinsic'), ('intrinsic', 'intrinsic'))),
                'Export reference differs from the external complete CPU model')
        ort_item = ort['artifacts'][f'actual-pair-{i}.npz']
        local_artifact(args.ort_dir, ort_item, pins)
        require(all(v['elementwise_gate_passed'] is True for group in ('comparisons', 'codec_comparisons')
                    for v in check[group].values()), 'Individual ORT/reference gate failed')
        rows.append(dict(images=np.ascontiguousarray(images), canonical=canonical, seed=seed, cpu=cpu))
    return model, captured, exported, rows


def save_call(output, arrays, row, index):
    values = dict(arrays, frame_ids=np.asarray(row['seed']['frame_ids'], np.float64))
    path = output / f'call-{index:02d}-pair-{row["seed"]["pair_index"]}.npz'
    with path.open('xb') as stream:
        np.savez(stream, **values)
    return dict(artifact=dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path)),
                arrays={key: metadata(value) for key, value in values.items()})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('capture-dir', 'export-dir', 'compile-dir', 'ort-dir', 'common', 'source-dir',
                 'source-manifest', 'input-manifest', 'abi-helper', 'header', 'output-dir'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('capture-sha256', 'capture-script-sha256', 'export-sha256', 'exporter-sha256',
                 'compile-sha256', 'ort-sha256', 'common-sha256', 'source-manifest-sha256'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--library', type=Path, default=Path('/usr/lib/librknnrt.so'))
    p.add_argument('--core-mask', type=int, choices=(1, 2, 4, 7), default=7)
    p.add_argument('--cpus', default='4,5,6,7')
    p.add_argument('--threads', type=int, choices=(1, 2, 3, 4), default=4)
    p.add_argument('--run-timeout-ms', type=int, default=60000)
    p.add_argument('--collect-perf-detail', action='store_true')
    args = p.parse_args(argv)
    pins = {}
    abi = load_module(args.abi_helper, ABI_SHA, '_v24_frozen_rknn_abi', pins)
    compatible = abi.require_board()
    output = args.output_dir.resolve()
    require(output.is_relative_to(BASE / 'experiments') and output != BASE / 'experiments',
            'New output must be below board experiments')
    protected = [getattr(args, name) for name in ('capture_dir', 'export_dir', 'compile_dir', 'ort_dir',
        'common', 'source_dir', 'source_manifest', 'input_manifest', 'abi_helper', 'header', 'library')]
    require(not any(output == path.resolve() or output.is_relative_to(path.resolve())
                    or path.resolve().is_relative_to(output) for path in protected), 'Output overlaps an input/source')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    original_affinity = os.sched_getaffinity(0)
    context, api = abi.CONTEXT(), None
    previous_threads = None
    report = dict(schema='da3_small_fixed_input_rknn_probe_v24.1', status='failed', model_name='da3-small',
        script_sha256=sha(__file__), official_commit=COMMIT, different_model_from_VGGT=True,
        original_VGGT_weights_used=False, production_changed=False, ORB_used=False, GT_used=False,
        no_GT_conditions=True, camera_conditions=dict(extrinsics=None, intrinsics=None), use_ray_pose=False,
        graph_run=False, online_pipeline_run=False, inference_on_board_only=True, device='rknn_and_cpu_codec',
        precision='RKNN unquantized FP16, FP32 IO; official CPU FP32 pose codec',
        checkpoint_sha256=WEIGHTS_SHA, config_sha256=CONFIG_SHA, input_manifest_sha256=INPUT_MANIFEST_SHA,
        source_manifest_sha256=args.source_manifest_sha256, reference_capture_sha256=REFERENCE_SHA,
        CPU_reference_capture_sha256=args.capture_sha256, export_sha256=args.export_sha256,
        compile_sha256=args.compile_sha256, ort_sha256=args.ort_sha256, sdk_run_api_calls=0,
        all_operators_on_npu_verified=False, quality_validated=False, real_time_validated=False,
        collect_perf_detail=args.collect_perf_detail, init_flags=8 if args.collect_perf_detail else 0,
        device_tree_compatible=compatible, abi=abi.abi_record(), cleanup_errors=[], records=[], call_records=[],
        timing_semantics=dict(forward='input layout packing + SDK set/run/get/copy/release + official CPU pose codec',
            excluded=['import/load/init', 'attribute/performance queries', 'finite/error comparisons', 'hashing', 'saving'],
            profile_separate_from_normal=True, normal_calls=6, profile_calls=3,
            limitations='Fixed inputs; three warm repeats are not statistical or online performance validation'))
    try:
        cpus = {int(value) for value in args.cpus.split(',')}
        require(cpus and cpus <= original_affinity and args.run_timeout_ms > 0,
                'Available CPU affinity and positive timeout required')
        os.sched_setaffinity(0, cpus)
        report['cpu_affinity'] = sorted(os.sched_getaffinity(0))
        pin(__file__, report['script_sha256'], pins)
        pin(args.header, HEADER_SHA, pins)
        pin(args.library, LIBRARY_SHA, pins)
        common = load_module(args.common, args.common_sha256, '_v24_official_common', pins)
        model, captured, exported, rows = load_inputs(args, common, pins)
        import torch
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(args.threads)
        report.update(torch_version=torch.__version__, torch_threads=torch.get_num_threads(), numpy_version=np.__version__)
        codec = prepare_codec(args.source_dir, common)
        # Preserve the already verified B1 normalized file bytes for saved-array
        # consumers; this is file I/O before every timed SDK/codec interval.
        for row in rows:
            info = row['cpu']['normalized_input_tensor']
            target = output / info['file']
            with (args.capture_dir / info['file']).open('rb') as source, target.open('xb') as destination:
                for chunk in iter(lambda: source.read(1 << 20), b''):
                    destination.write(chunk)
            require(sha(target) == info['file_sha256'], 'Saved normalized input copy differs')
        report.update(selected_inputs=captured['selected_inputs'], rectification=captured['rectification'],
                      normalization=captured['normalization'], geometric_preprocessing_changed=False,
                      checkpoint=captured['checkpoint'], source_code_sha256={Path(__file__).name: report['script_sha256'],
                        'run-da3-small-board.py': args.capture_script_sha256, 'da3_probe_common.py': args.common_sha256},
                      compiled_model_sha256=sha(model), compiled_model_bytes=model.stat().st_size,
                      memory_before_sdk=abi.process_memory())
        api = abi.bind_api(ct.CDLL(str(args.library.resolve())))
        filename = ct.create_string_buffer(os.fsencode(model))
        tick = time.perf_counter()
        abi.check_call(api.rknn_init(ct.byref(context), ct.cast(filename, ct.c_void_p), 0, report['init_flags'], None), 'rknn_init')
        report['init_seconds'] = time.perf_counter() - tick
        abi.check_call(api.rknn_set_core_mask(context, args.core_mask), 'rknn_set_core_mask')
        report['core_mask'] = args.core_mask
        version = abi.query(api, context, 5, abi.SDKVersion())
        report.update(sdk_version=bytes(version.api_version).decode(errors='replace'),
                      driver_version=bytes(version.drv_version).decode(errors='replace'))
        require(report['sdk_version'].startswith('2.3.2'), 'Pinned RKNN runtime 2.3.2 required')
        counts = abi.query(api, context, 0, abi.InputOutputNum())
        require(counts.n_input == 1 and counts.n_output == 3, 'Exactly one input and three outputs required')
        input_attr = abi.query(api, context, 1, abi.TensorAttr(index=0))
        output_attrs = [abi.query(api, context, 2, abi.TensorAttr(index=i)) for i in range(3)]
        report['input_attr'] = abi.attr_record(input_attr)
        report['output_attrs'] = [abi.attr_record(attr) for attr in output_attrs]
        validate_attribute(abi, input_attr, 'images', SHAPE, is_input=True)
        for attr, item in zip(output_attrs, OUTPUTS):
            validate_attribute(abi, attr, item['name'], tuple(item['shape']), is_input=False)
        memory = abi.query(api, context, 6, abi.MemSize())
        report['model_memory_bytes'] = {name: int(getattr(memory, name)) for name, _ in memory._fields_ if name != 'reserved'}
        require(memory.total_dma_allocated_size <= 2 * 1024 ** 3, 'Isolated SDK DMA allocation exceeds 2GiB safety budget')
        for row in rows:
            repeat = 1 if args.collect_perf_detail or row['seed']['pair_index'] else 4
            for repetition in range(repeat):
                index = len(report['call_records'])
                actual, call = execute_multi(api, abi, context, row['images'], int(input_attr.fmt),
                                            args.run_timeout_ms, report['cleanup_errors'], report)
                validate_outputs(actual)
                arrays, codec_seconds = canonical_output(actual, codec)
                call.update(call_index=index, pair_index=row['seed']['pair_index'], repetition=repetition,
                    timing_role='profile' if args.collect_perf_detail else ('cold' if index == 0 else 'warm'),
                    frame_ids=row['seed']['frame_ids'], input_tensor=row['cpu']['input_tensor'],
                    normalized_input_tensor=row['cpu']['normalized_input_tensor'],
                    official_codec_seconds=codec_seconds, forward_seconds=call['sdk_io_seconds'] + codec_seconds,
                    versus_CPU_reference={key: comparison(value, row['canonical'][key]) for key, value in arrays.items()},
                    raw_outputs={item['name']: metadata(value) for item, value in zip(OUTPUTS, actual)},
                    **save_call(output, arrays, row, index))
                perf = abi.PerfRun()
                code = api.rknn_query(context, 4, ct.byref(perf), ct.sizeof(perf))
                call.update(perf_run_query_code=int(code), sdk_run_duration_us=int(perf.run_duration) if code == 0 else None)
                if args.collect_perf_detail:
                    call['perf_detail'] = abi.perf_detail(api, context)
                report['call_records'].append(call)
                if repetition == 0:
                    report['records'].append(dict(call))
                print(json.dumps(dict(event='call_completed', index=index, pair=call['pair_index'],
                      forward_seconds=call['forward_seconds'], strict_FP32_pass=all(v['elementwise_gate_passed']
                        for v in call['versus_CPU_reference'].values()))), flush=True)
        report['exact_FP32_reference_gate_passed'] = all(v['elementwise_gate_passed'] for c in report['call_records']
                                                        for v in c['versus_CPU_reference'].values())
        if not args.collect_perf_detail:
            report['first_pair_warm_forward_seconds'] = [c['forward_seconds'] for c in report['call_records'][1:4]]
            report['first_pair_warm_mean_seconds'] = float(np.mean(report['first_pair_warm_forward_seconds']))
        require(len(report['call_records']) == (3 if args.collect_perf_detail else 6), 'Prescribed call schedule incomplete')
        report['import_origins'] = common.check_import_origins(args.source_dir, common.verify_official_source(
            args.source_dir, args.source_manifest, args.source_manifest_sha256)['source_hashes'])
        report['status'] = 'completed_fixed_input_diagnostic'
    except Exception as error:
        report.update(error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
    finally:
        if api is not None and context.value:
            try:
                abi.check_call(api.rknn_destroy(context), 'rknn_destroy')
                report['context_destroyed'] = True
            except Exception as error:
                report['context_destroyed'] = False
                report['cleanup_errors'].append('rknn_destroy: ' + repr(error))
        else:
            report['context_destroyed'] = None
        try:
            report['source_and_input_hashes_after'] = {path: sha(path) for path in pins}
            report['sources_and_inputs_unchanged'] = report['source_and_input_hashes_after'] == pins
            if not report['sources_and_inputs_unchanged']:
                report['cleanup_errors'].append('Source/input/runtime SHA changed')
        except Exception as error:
            report['cleanup_errors'].append('Source recheck: ' + repr(error))
        try:
            if previous_threads is not None:
                torch.set_num_threads(previous_threads)
            os.sched_setaffinity(0, original_affinity)
            report['affinity_restored'] = os.sched_getaffinity(0) == original_affinity
            require(report['affinity_restored'], 'CPU affinity restoration failed')
        except Exception as error:
            report['cleanup_errors'].append('Thread/affinity cleanup: ' + repr(error))
        report.update(source_and_input_hashes=pins, memory_final=abi.process_memory(),
                      successful_synchronous_call_count=len(report['call_records']), total_seconds=time.monotonic() - started)
        if report['cleanup_errors']:
            report['status'] = 'failed'
        (output / 'run.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], error=report.get('error'), receipt_sha256=sha(output / 'run.json'))), flush=True)
    return 0 if report['status'] == 'completed_fixed_input_diagnostic' else 1


if __name__ == '__main__':
    raise SystemExit(main())
