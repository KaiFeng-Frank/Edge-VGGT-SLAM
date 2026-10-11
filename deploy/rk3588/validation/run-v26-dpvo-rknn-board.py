#!/usr/bin/env python3
"""Board-only RKNN probe of the two original-weight DPVO CNN encoders.

This executes no correlation, learned update, bundle adjustment or VO. Inputs
are the hash-bound original BGR/undistort/crop/normalization capture, and both
feature outputs already include the official division by four. No CPU model
or fallback is loaded. FP16 errors are reported against an independently
validated FP32 CPU/ORT reference; a failed FP32 gate is not a runtime success
claim or a fatal research error. Nonfinite/all-zero outputs are fatal.
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
INPUT_SHAPE = (1, 3, 464, 608)
OUTPUTS = (('fmap', (1, 128, 116, 152)), ('imap', (1, 384, 116, 152)))
ABI_SHA = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
HEADER_SHA = 'c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a'
LIBRARY_SHA = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
COMPILER_SHA = 'a40bf09e53ef9b378db4781be18f9e53f824afcc809c9009e0a65f4f0f91b4e5'
COMMON_SHA = 'ba47164a4273e0d392f8e23892f17be1aaf89b7ef42c3ebd746e3a7b3fd946a9'
COMMIT = '0ac95b656d1fda91c271d2a106460d19ad966fc7'
SOURCE_MANIFEST_SHA = 'ce2ecaf96b086f69ef4d90a2e07e7b81afe8ae094656eb41260182000a69e375'
CHECKPOINT_SHA = '30d02dc2b88a321cf99aad8e4ea1152a44d791b5b65bf95ad036922819c0ff12'
PROBE_SHA = 'e2f56fac7a6432761bf67ea2703a7b72b9d695577302d8f02d5cc8af0c0db9a5'
INPUT_MANIFEST_SHA = 'bffcd221ee1e7c4b17045bf9b642df7272747ee000e67d5e682610b635804539'
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


def load_abi(path, pins):
    path = pin(path, ABI_SHA, pins)
    spec = importlib.util.spec_from_file_location('_v26_frozen_rknn_abi', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def metadata(value):
    value = np.ascontiguousarray(value)
    return dict(shape=list(value.shape), dtype=str(value.dtype), bytes=int(value.nbytes),
                tensor_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())


def validate_array(value, info, shape):
    require(value.dtype == np.float32 and value.shape == tuple(shape)
            and np.isfinite(value).all(), 'Array dtype/shape/finite mismatch')
    require(info == metadata(value), 'Array raw SHA or metadata mismatch')


def comparison(actual, expected):
    require(actual.shape == expected.shape and actual.dtype == expected.dtype == np.float32
            and np.isfinite(actual).all() and np.isfinite(expected).all(),
            'Finite matching FP32 outputs required')
    a, b = actual.astype(np.float64).ravel(), expected.astype(np.float64).ravel()
    delta = a - b
    norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    failures = np.abs(delta) > ATOL + RTOL * np.abs(b)
    return dict(rtol=RTOL, atol=ATOL, failed_elements=int(failures.sum()),
                element_count=int(failures.size), gate_passed=not bool(failures.any()),
                max_abs=float(np.max(np.abs(delta))), rms=float(np.sqrt(np.mean(delta * delta))),
                relative_l2=float(np.linalg.norm(delta) / norm_b) if norm_b else None,
                cosine=float(np.dot(a / norm_a, b / norm_b)) if norm_a and norm_b else None,
                array_equal=bool(np.array_equal(actual, expected)),
                bitwise_equal=actual.tobytes(order='C') == expected.tobytes(order='C'),
                finite=True, actual_all_zero=bool(np.all(actual == 0)))


def local_artifact(directory, item, pins):
    name = item['file']
    require(isinstance(name, str) and Path(name).name == name, 'Simple local filename required')
    path = pin(Path(directory) / name, item['sha256'], pins)
    require(path.stat().st_size == item['bytes'], 'Artifact file size mismatch')
    return path


def validate_compilation(compiled, exported):
    require(compiled.get('script_sha256') == COMPILER_SHA
            and compiled.get('configuration') == dict(target_platform='rk3588', float_dtype='float16',
                optimization_level=3, do_quantization=False), 'Exact frozen v12 compiler/config required')
    require(compiled.get('status') == 'compiled_not_runtime_validated'
            and compiled.get('released') is True and not compiled.get('release_error'),
            'Compiler did not finish and release successfully')
    operations = compiled['operations']
    require([row['name'] for row in operations] == ['config', 'load_onnx', 'build', 'export_rknn']
            and all(row['code'] is None or type(row['code']) is int and row['code'] == 0
                    for row in operations), 'Compiler operation schedule or return code differs')
    require(compiled['versions']['rknn-toolkit2'] == '2.3.2'
            and compiled['input']['sha256'] == exported['onnx']['sha256'],
            'Compiler version or ONNX source differs')


def validate_receipts(captured, exported, ort):
    expected_status = ('completed_encoder_cpu_probe_not_VO', 'exported_not_runtime_validated',
                       'completed_CPU_ORT_gate')
    for receipt, mode, status in zip((captured, exported, ort), ('capture', 'export', 'ort'), expected_status):
        require(receipt.get('schema') == 'v26_DPVO_original_encoders_board_probe'
                and receipt.get('mode') == mode and receipt.get('status') == status,
                'Unexpected or unsuccessful encoder receipt')
        require(receipt.get('checkpoint_sha256') == CHECKPOINT_SHA
                and receipt.get('state_unchanged') is True
                and receipt.get('sources_and_inputs_unchanged') is True
                and receipt.get('affinity_restored') is True and not receipt.get('cleanup_errors'),
                'Original checkpoint/source/state/cleanup validation failed')
        require(receipt.get('script_sha256') == captured['script_sha256'] == PROBE_SHA
                and receipt.get('input_manifest_sha256') == captured['input_manifest_sha256'] == INPUT_MANIFEST_SHA
                and receipt.get('preprocessing') == captured['preprocessing']
                and receipt.get('state_before') == captured['state_before'],
                'Producer, preprocessing, input identity or loaded state differs')
        loader = receipt['loader']
        require(loader['repo'] == 'princeton-vl/DPVO' and loader['commit'] == COMMIT
                and loader['manifest_sha256'] == SOURCE_MANIFEST_SHA
                and loader['helper_sha256'] == COMMON_SHA
                and loader['checkpoint'] == dict(sha256=CHECKPOINT_SHA, bytes=14167743),
                'Pinned original DPVO source/weights/helper required')
        state = loader['load_state']
        require(state['strict'] is True and state['inference_parameter_coverage'] == 1.0
                and state['released_state_keys'] == 98 and state['inference_state_keys'] == 94
                and state['inference_parameters'] == 3384324
                and state['all_loaded_parameter_bytes_equal'] is True,
                'Full strict original inference parameter coverage required')
        loaded = state['loaded_tensors']
        stored = captured['state_before']
        require(len(loaded) == len(stored) == 94 and loaded.keys() == stored.keys(),
                'Strict loader and CPU captured parameter keys differ')
        for name, item in stored.items():
            original = loaded[name]
            require(item['dtype'] == 'float32' and original['dtype'] == 'torch.float32'
                    and all(item[key] == original[key] for key in ('shape', 'bytes', 'tensor_sha256'))
                    and original['numel'] == math.prod(item['shape'])
                    and original['bytes'] == original['numel'] * 4,
                    'Strict loader and CPU captured parameter bytes differ: ' + name)
    require(captured['preprocessing'] == dict(decode='cv2.imread BGR uint8',
        K=[517.3, 516.5, 318.6, 255.3], distortion=[.2624, -.9531, -.0054, .0026, 1.1633],
        crop=[8, 8, 16, 16], shape=[1, 1, 3, 464, 608],
        normalization='2 * (image / 255.0) - 0.5, FP32', GT_or_sensor_depth_input=False),
        'Exact original TUM BGR/undistort/crop/once-normalization required')
    require(exported['input_shape'] == list(INPUT_SHAPE)
            and exported['output_shapes'] == [list(shape) for _, shape in OUTPUTS]
            and ort['providers'] == ['CPUExecutionProvider']
            and ort['graph_optimization'] == 'ORT_DISABLE_ALL'
            and ort['exact_reference_gate_passed'] is True,
            'Fixed graph shape or independent CPU ORT gate differs')


def load_inputs(args, pins):
    receipt_paths = [pin(directory / filename, digest, pins) for directory, filename, digest in (
        (args.capture_dir, 'capture.json', args.capture_sha256),
        (args.export_dir, 'export.json', args.export_sha256),
        (args.ort_dir, 'ort.json', args.ort_sha256),
        (args.compile_dir, 'compile.json', args.compile_sha256))]
    captured, exported, ort, compiled = [read_json(path) for path in receipt_paths]
    validate_receipts(captured, exported, ort)
    require(exported['capture_sha256'] == ort['capture_sha256'] == args.capture_sha256
            and ort['export_sha256'] == args.export_sha256 and ort['onnx'] == exported['onnx'],
            'Capture/export/ORT lineage differs')
    validate_compilation(compiled, exported)
    for receipt in (captured, exported, ort):
        require(receipt['source_hashes_after'] == receipt['source_pins'], 'Saved source pin recheck differs')
        for path, digest in receipt['source_pins'].items():
            require(Path(path).is_absolute(), 'Absolute original source/input path required')
            pin(path, digest, pins)
    local_artifact(args.export_dir, exported['onnx'], pins)
    model_item = compiled['output']
    model = pin(args.compile_dir / Path(model_item['path']).name, model_item['sha256'], pins)
    require(model.stat().st_size == model_item['size'] and 0 < model.stat().st_size <= 128 * 1024 ** 2,
            'Compiled encoder artifact size differs or exceeds isolated budget')
    require(len(captured['inputs']) == len(ort['pairs']) == 3
            and [c['frame_index'] for c in captured['calls']] == [0, 0, 0, 0, 1, 2],
            'Exactly three source frames and prescribed CPU reference calls required')
    rows = []
    previous_id = -math.inf
    for i, (seed, checked) in enumerate(zip(captured['inputs'], ort['pairs'])):
        frame_id = seed['frame_id']
        require(seed['frame_index'] == checked['frame_index'] == i
                and isinstance(frame_id, str) and math.isfinite(float(frame_id)) and float(frame_id) > previous_id
                and seed['source']['frame_id'] == frame_id,
                'Frame index/timestamp order differs')
        previous_id = float(frame_id)
        images = np.load(local_artifact(args.capture_dir, seed['input'], pins), allow_pickle=False)
        validate_array(images, seed['array'], (INPUT_SHAPE[0], 1) + INPUT_SHAPE[1:])
        require(np.min(images) >= -.5 and np.max(images) <= 1.5,
                'Expected once-normalized original BGR range [-.5,1.5]')
        references = [row for row in captured['calls'] if row['frame_index'] == i]
        require(all(row['frame_id'] == frame_id for row in references), 'CPU call source timestamp differs')
        baseline = None
        for row in references:
            with np.load(local_artifact(args.capture_dir, row['artifact'], pins), allow_pickle=False) as saved:
                require(set(saved.files) == {name for name, _ in OUTPUTS}, 'CPU NPZ output keys differ')
                current = {name: saved[name].copy() for name, _ in OUTPUTS}
            for name, shape in OUTPUTS:
                validate_array(current[name], row['arrays'][name], (shape[0], 1) + shape[1:])
                require(np.any(current[name] != 0), 'All-zero CPU reference')
                if baseline is not None:
                    require(current[name].tobytes(order='C') == baseline[name].tobytes(order='C'),
                            'Repeated original CPU reference bytes differ')
            baseline = current
        with np.load(local_artifact(args.ort_dir, checked['artifact'], pins), allow_pickle=False) as saved:
            require(set(saved.files) == {name for name, _ in OUTPUTS}, 'ORT NPZ output keys differ')
            ort_values = {name: saved[name].copy() for name, _ in OUTPUTS}
        for name, shape in OUTPUTS:
            validate_array(ort_values[name], checked['arrays'][name], (shape[0], 1) + shape[1:])
            gate = checked['checks'][name]
            require(gate['rtol'] == RTOL and gate['atol'] == ATOL and gate['gate_passed'] is True
                    and comparison(ort_values[name], baseline[name])['gate_passed'],
                    'Independent saved CPU ORT reference comparison failed')
        rows.append(dict(seed=seed, images=np.ascontiguousarray(images[:, 0]),
                         reference=baseline, ort_reference=ort_values))
    return model, captured, exported, compiled, rows


def validate_attribute(abi, attr, index, name, logical_shape, *, is_input):
    value = abi.attr_record(attr)
    require(value['index'] == index and value['name'] == name, 'RKNN tensor index/name/order mismatch')
    require(value['type'] in (0, 1) and math.isfinite(value['scale'])
            and (value['quantization_type'] == 0 or value['type'] == 1
                 and value['quantization_type'] == 2 and value['zero_point'] == 0 and value['scale'] == 1.0),
            'Only unquantized FP32/FP16 or exact FP16 identity-affine descriptor supported')
    require(value['n_elems'] == math.prod(logical_shape)
            and value['size'] == value['n_elems'] * (4 if value['type'] == 0 else 2),
            'SDK tensor elements/logical byte size differs')
    if is_input:
        permitted = {0: tuple(logical_shape),
                     1: (logical_shape[0], logical_shape[2], logical_shape[3], logical_shape[1])}
        require(value['format'] in permitted and tuple(value['shape']) == permitted[value['format']],
                'Explicit native NCHW or NHWC input shape required')
    else:
        require(value['format'] == 0 and tuple(value['shape']) == tuple(logical_shape),
                'Feature output must have exact NCHW index/name/shape; no blind layout reshape')
    return value


def pack_input(images, native_format):
    require(images.dtype == np.float32 and images.shape == INPUT_SHAPE and np.isfinite(images).all(),
            'Fixed-shape normalized FP32 BGR input required')
    if native_format == 1:
        return np.array(images.transpose(0, 2, 3, 1), copy=True, order='C')
    require(native_format == 0, 'Unsupported native input layout')
    return np.array(images, copy=True, order='C')


def execute_call(api, abi, context, images, native_format, timeout_ms, cleanup, activity):
    """Pack, synchronous SDK call, owned FP32 copies, then release; no model codec."""
    tick = time.perf_counter()
    packed = pack_input(images, native_format)
    packed_at = time.perf_counter()
    inputs = (abi.Input * 1)(abi.Input(0, packed.ctypes.data, packed.nbytes, 0, 0, native_format))
    outputs = (abi.Output * len(OUTPUTS))()
    for i, output in enumerate(outputs):
        output.want_float, output.is_prealloc, output.index = 1, 0, i
    extension, received = abi.RunExtend(0, 0, timeout_ms, -1), abi.OutputExtend()
    row = dict(input_api_format=native_format, input_pass_through=0, input_api_dtype='float32',
               input_logical_shape=list(images.shape), input_buffer_shape=list(packed.shape),
               input_packing_seconds=packed_at - tick, output_api_want_float=1,
               synchronous=True, run_timeout_ms=timeout_ms)
    got = False
    try:
        abi.check_call(api.rknn_inputs_set(context, 1, inputs), 'rknn_inputs_set')
        set_at = time.perf_counter()
        activity['sdk_run_api_calls'] = activity.get('sdk_run_api_calls', 0) + 1
        abi.check_call(api.rknn_run(context, ct.byref(extension)), 'rknn_run')
        run_at = time.perf_counter()
        abi.check_call(api.rknn_outputs_get(context, len(outputs), outputs, ct.byref(received)), 'rknn_outputs_get')
        got = True
        get_at = time.perf_counter()
        require(extension.frame_id == received.frame_id, 'SDK run/get frame identity mismatch')
        actual = {}
        for output, (name, shape) in zip(outputs, OUTPUTS):
            elems = math.prod(shape)
            require(bool(output.buf) and output.size == elems * 4, 'Invalid SDK-owned FP32 output pointer/size')
            view = np.ctypeslib.as_array(ct.cast(output.buf, ct.POINTER(ct.c_float)), shape=(elems,))
            actual[name] = view.copy().reshape(shape)
        copied_at = time.perf_counter()
        row.update(input_set_seconds=set_at - packed_at, run_api_seconds=run_at - set_at,
                   outputs_get_seconds=get_at - run_at, output_copy_seconds=copied_at - get_at,
                   runtime_frame_id=int(extension.frame_id), output_bytes=[int(o.size) for o in outputs])
    finally:
        if got or any(output.buf for output in outputs):
            primary_active = sys.exc_info()[0] is not None
            release_at = time.perf_counter()
            try:
                code = api.rknn_outputs_release(context, len(outputs), outputs)
                row.update(outputs_release_seconds=time.perf_counter() - release_at,
                           output_buffers_released=code == 0, sdk_io_seconds=time.perf_counter() - tick)
                if code != 0:
                    cleanup.append('rknn_outputs_release returned ' + str(code))
                    if not primary_active:
                        abi.check_call(code, 'rknn_outputs_release')
            except Exception as error:
                if not cleanup or cleanup[-1] != 'rknn_outputs_release returned ' + str(locals().get('code')):
                    cleanup.append('rknn_outputs_release: ' + repr(error))
                if not primary_active:
                    raise
    return actual, row


def canonical_outputs(actual):
    require(set(actual) == {name for name, _ in OUTPUTS}, 'Both feature arrays required')
    canonical = {}
    for name, shape in OUTPUTS:
        value = actual[name]
        require(value.dtype == np.float32 and value.shape == shape
                and np.isfinite(value).all() and np.any(value != 0),
                'Nonfinite or all-zero SDK feature output: ' + name)
        # Outputs own their copied storage. Inserting only the known view axis
        # keeps the original CPU encoder's B,S,C,H,W contract unchanged.
        canonical[name] = value[:, None]
    return canonical


def call_schedule(collect_perf_detail):
    return (0, 1, 2) if collect_perf_detail else (0, 0, 0, 0, 1, 2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture-dir', 'export-dir', 'ort-dir', 'compile-dir', 'abi-helper', 'header', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('capture', 'export', 'ort', 'compile'):
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--library', type=Path, default=Path('/usr/lib/librknnrt.so'))
    parser.add_argument('--core-mask', type=int, choices=(1, 2, 4, 7), default=7)
    parser.add_argument('--cpus', default='4,5,6,7')
    parser.add_argument('--run-timeout-ms', type=int, default=60000)
    parser.add_argument('--collect-perf-detail', action='store_true')
    args = parser.parse_args(argv)
    pins = {}
    abi = load_abi(args.abi_helper, pins)
    compatible = abi.require_board()
    output = args.output_dir.resolve()
    require(output.is_relative_to(BASE / 'experiments') and output != BASE / 'experiments',
            'Output must be a new experiment directory on the board')
    protected = [getattr(args, name) for name in ('capture_dir', 'export_dir', 'ort_dir', 'compile_dir',
                                                 'abi_helper', 'header', 'library')]
    require(not any(output == path.resolve() or output.is_relative_to(path.resolve())
                    or path.resolve().is_relative_to(output) for path in protected),
            'Output overlaps an input/source')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    prior_affinity = os.sched_getaffinity(0)
    context, api = abi.CONTEXT(), None
    report = dict(schema='v26_DPVO_original_encoders_RKNN_probe', status='failed',
        script_sha256=sha(__file__), official_commit=COMMIT, upstream_license='MIT',
        checkpoint_sha256=CHECKPOINT_SHA, original_DPVO_weights=True, original_VGGT_weights=False,
        different_model_from_VGGT=True, ORB_used=False, production_changed=False,
        encoder_only=True, correlation_executed=False, update_executed=False, BA_executed=False,
        complete_VO_executed=False, realtime_validated=False, quality_validated=False,
        actual_inference_on_RK3588S=True, all_operators_on_npu_verified=False,
        precision='Unquantized RKNN FP16; original FP32 IO; both original encoder /4 outputs',
        capture_sha256=args.capture_sha256, export_sha256=args.export_sha256,
        ort_sha256=args.ort_sha256, compile_sha256=args.compile_sha256,
        device_tree_compatible=compatible, abi=abi.abi_record(),
        collect_perf_detail=args.collect_perf_detail, init_flags=8 if args.collect_perf_detail else 0,
        sdk_run_api_calls=0, calls=[], cleanup_errors=[],
        timing_semantics=dict(forward='input native-layout packing + SDK set/run/get + owned FP32 copies/release',
            excludes=['imports', 'model load/init', 'preprocessing', 'attribute/profile queries',
                      'numeric comparisons', 'hashing', 'saving'],
            profile_separate_from_normal=True, normal_calls=6, profile_calls=3,
            limitation='Fixed three inputs; CNN features only; not frame-to-pose or a real-time validation'),
        environment={key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'LD_LIBRARY_PATH')})
    try:
        cpus = {int(value) for value in args.cpus.split(',')}
        require(cpus and cpus <= prior_affinity and args.run_timeout_ms > 0,
                'Available CPU affinity and positive timeout required')
        os.sched_setaffinity(0, cpus)
        report['cpu_affinity'] = sorted(os.sched_getaffinity(0))
        pin(__file__, report['script_sha256'], pins)
        pin(args.header, HEADER_SHA, pins)
        pin(args.library, LIBRARY_SHA, pins)
        model, captured, exported, compiled, rows = load_inputs(args, pins)
        report.update(preprocessing=captured['preprocessing'], original_loader_provenance=captured['loader'],
                      input_manifest_sha256=captured['input_manifest_sha256'],
                      source_frames=[row['seed']['source'] for row in rows],
                      inputs=[row['seed'] for row in rows],
                      compiled_model_sha256=sha(model), compiled_model_bytes=model.stat().st_size,
                      compiler_configuration=compiled['configuration'], numpy_version=np.__version__,
                      memory_before_sdk=abi.process_memory())
        api = abi.bind_api(ct.CDLL(str(args.library.resolve())))
        filename = ct.create_string_buffer(os.fsencode(model))
        tick = time.perf_counter()
        abi.check_call(api.rknn_init(ct.byref(context), ct.cast(filename, ct.c_void_p), 0, report['init_flags'], None),
                       'rknn_init')
        report['init_seconds'] = time.perf_counter() - tick
        abi.check_call(api.rknn_set_core_mask(context, args.core_mask), 'rknn_set_core_mask')
        report['core_mask'] = args.core_mask
        version = abi.query(api, context, 5, abi.SDKVersion())
        report.update(sdk_version=bytes(version.api_version).decode(errors='replace'),
                      driver_version=bytes(version.drv_version).decode(errors='replace'))
        require(report['sdk_version'].startswith('2.3.2'), 'Pinned RKNN runtime 2.3.2 required')
        counts = abi.query(api, context, 0, abi.InputOutputNum())
        require(counts.n_input == 1 and counts.n_output == len(OUTPUTS), 'Exactly one input/two feature outputs required')
        input_attr = abi.query(api, context, 1, abi.TensorAttr(index=0))
        output_attrs = [abi.query(api, context, 2, abi.TensorAttr(index=i)) for i in range(len(OUTPUTS))]
        report['input_attr'] = validate_attribute(abi, input_attr, 0, 'normalized_bgr', INPUT_SHAPE, is_input=True)
        report['output_attrs'] = [validate_attribute(abi, attr, i, name, shape, is_input=False)
                                 for i, (attr, (name, shape)) in enumerate(zip(output_attrs, OUTPUTS))]
        memory = abi.query(api, context, 6, abi.MemSize())
        report['model_memory_bytes'] = {name: int(getattr(memory, name)) for name, _ in memory._fields_ if name != 'reserved'}
        require(memory.total_dma_allocated_size <= 1024 ** 3, 'Isolated SDK DMA allocation exceeds 1GiB budget')
        for index, frame_index in enumerate(call_schedule(args.collect_perf_detail)):
            row = rows[frame_index]
            actual, call = execute_call(api, abi, context, row['images'], int(input_attr.fmt),
                                        args.run_timeout_ms, report['cleanup_errors'], report)
            canonical = canonical_outputs(actual)
            artifact = output / f'call-{index:02d}-frame-{frame_index}.npz'
            with artifact.open('xb') as stream:
                np.savez(stream, **canonical)
            call.update(call_index=index, frame_index=frame_index, frame_id=row['seed']['frame_id'],
                timing_role='profile' if args.collect_perf_detail else ('first_call' if index == 0 else
                    'warm_same_frame' if index < 4 else 'other_frame_single'),
                input=row['seed']['input'], input_array=row['seed']['array'],
                forward_seconds=call['sdk_io_seconds'],
                artifact=dict(file=artifact.name, bytes=artifact.stat().st_size, sha256=sha(artifact)),
                arrays={name: metadata(value) for name, value in canonical.items()},
                versus_CPU_reference={name: comparison(value, row['reference'][name]) for name, value in canonical.items()},
                versus_CPU_ORT={name: comparison(value, row['ort_reference'][name]) for name, value in canonical.items()})
            perf = abi.PerfRun()
            code = api.rknn_query(context, 4, ct.byref(perf), ct.sizeof(perf))
            call.update(perf_run_query_code=int(code), sdk_run_duration_us=int(perf.run_duration) if code == 0 else None)
            if args.collect_perf_detail:
                call['perf_detail'] = abi.perf_detail(api, context)
            report['calls'].append(call)
            print(json.dumps(dict(event='encoder_call_completed', index=index, frame_index=frame_index,
                forward_seconds=call['forward_seconds'], strict_FP32_pass=all(v['gate_passed']
                for v in call['versus_CPU_reference'].values()))), flush=True)
        report['exact_FP32_reference_gate_passed'] = all(v['gate_passed'] for call in report['calls']
                                                       for v in call['versus_CPU_reference'].values())
        if not args.collect_perf_detail:
            warm = [call['forward_seconds'] for call in report['calls'][1:4]]
            report.update(first_frame_warm_forward_seconds=warm, first_frame_warm_mean_seconds=float(np.mean(warm)))
        require(len(report['calls']) == (3 if args.collect_perf_detail else 6), 'Prescribed SDK schedule incomplete')
        report['status'] = 'completed_encoder_RKNN_probe_not_VO'
    except Exception as error:
        report['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
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
            report['source_hashes_after'] = {path: sha(path) for path in pins}
            report['sources_and_inputs_unchanged'] = report['source_hashes_after'] == pins
            require(report['sources_and_inputs_unchanged'], 'Source/input/runtime SHA changed')
        except Exception as error:
            report['cleanup_errors'].append('Source recheck: ' + repr(error))
        try:
            os.sched_setaffinity(0, prior_affinity)
            report['affinity_restored'] = os.sched_getaffinity(0) == prior_affinity
            require(report['affinity_restored'], 'Affinity restoration failed')
        except Exception as error:
            report['cleanup_errors'].append('Affinity cleanup: ' + repr(error))
        report.update(source_pins=pins, memory_final=abi.process_memory(),
                      successful_synchronous_call_count=len(report['calls']), total_seconds=time.monotonic() - started)
        if report['cleanup_errors']:
            report['status'] = 'failed'
        (output / 'run.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], error=report.get('error'),
                         receipt_sha256=sha(output / 'run.json'))), flush=True)
    return 0 if report['status'] == 'completed_encoder_RKNN_probe_not_VO' else 1


if __name__ == '__main__':
    raise SystemExit(main())
