#!/usr/bin/env python3
"""Static original-weight MLP export; no host model/ORT/image execution.

The graph consumes already-normalized FP32 rows and contains only the original
FC1, exact Erf GELU and FC2. Norm2, LayerScale and residuals remain outside.
An optional hidden slice omits FC2 bias: callers sum all disjoint partial
outputs in FP32, then add the original FC2 bias once. This is real-arithmetic
equivalence; compiled FP16 precision and useful speed require board evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

sys.dont_write_bytecode = True
CAPTURE_SHA = '3664f5c932602a3c6538ea6013ea0389625c42bdcb7dcf10a9fbceb9304abeb1'
CAPTURE_SCRIPT_SHA = '59c1b346bdda45fe1130eab0f869f47732755cc95761c16c43d3c0346805bc14'
CHECKPOINT_SHA = 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
CHECKPOINT_BYTES = 5026874952
CHANNELS, HIDDEN = 1024, 4096
ORIGINAL_SHAPE = (1, 874, CHANNELS)
PARAMETER_KEYS = ('mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight', 'mlp.fc2.bias')
ARCHIVE_KEYS = (*PARAMETER_KEYS, 'ls2.gamma')
BUCKETS = (874, 1024, 512)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def require_sha(path, expected):
    if (not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected)
            or sha(path) != expected):
        raise ValueError(f'SHA256 mismatch: {Path(path).name}')
    return expected


def array_record(value):
    if (not isinstance(value, np.ndarray) or value.dtype != np.float32
            or not value.size or not value.flags.c_contiguous or not np.isfinite(value).all()):
        raise ValueError('Finite nonempty C-contiguous original FP32 values are required')
    return dict(shape=list(value.shape), dtype='float32', bytes=int(value.nbytes),
                sha256=hashlib.sha256(memoryview(value).cast('B')).hexdigest())


def _artifact(directory, capture, name):
    item = capture['artifacts'][name]
    path = directory / name
    if path.is_symlink() or not path.is_file():
        raise ValueError('Captured artifacts must be owned regular files')
    digest = require_sha(path, item['sha256'])
    if path.stat().st_size != item['bytes']:
        raise ValueError(f'Captured artifact byte size mismatch: {name}')
    return path, digest


def load_capture(directory, expected_capture_sha=CAPTURE_SHA):
    """Read/verify arrays only, without ONNX/Torch; fixed original global0 run.

    Returns (capture, {'input': norm2, 'output': full_mlp}, four_weights, pins).
    The fifth archive tensor LS2 is verified but is never put into this graph.
    Source identity comes from the pinned historical receipt; this loader does
    not claim current checkout sources, images or the checkpoint were rehashed.
    """
    if expected_capture_sha != CAPTURE_SHA:
        raise ValueError('Only the frozen original global0 capture is supported')
    directory = Path(directory).resolve(strict=True)
    receipt = directory / 'capture.json'
    require_sha(receipt, expected_capture_sha)
    capture = json.loads(receipt.read_text())
    architecture = capture.get('architecture', {})
    # The exact receipt SHA above binds every metadata field. Model identity is
    # checked through the original checkpoint, architecture and tensor pins;
    # a historical experiment label is not a model identity.
    if (capture.get('status') != 'captured_original_FP32_stages_bitwise_bound'
            or capture.get('script_sha256') != CAPTURE_SCRIPT_SHA
            or capture.get('original_weights') is not True
            or capture.get('architecture_changed') is not False
            or capture.get('npu_used') is not False or capture.get('ORB_used') is not False
            or capture.get('GT_used') is not False
            or capture.get('host_inference_executed') is not False
            or capture.get('inference_on_board_only') is not True
            or capture.get('full_model_forward_count') != 1
            or capture.get('cleanup_errors') != [] or capture.get('hooks_removed') is not True
            or capture.get('sources_and_inputs_unchanged') is not True
            or capture.get('checkpoint', {}).get('sha256') != CHECKPOINT_SHA
            or capture['checkpoint'].get('bytes') != CHECKPOINT_BYTES
            or architecture.get('mlp_class') != 'Mlp'
            or architecture.get('gelu_approximate') != 'none'
            or architecture.get('dropout') != 0. or architecture.get('channels') != CHANNELS
            or architecture.get('views') != 2 or architecture.get('special_tokens') != 5
            or architecture.get('grid_hw') != [18, 24]
            or capture.get('mlp_state_keys') != list(ARCHIVE_KEYS)
            or capture.get('global0_vs_profile') != dict(input=True, output=True, pos=True)
            or capture.get('full_predictions_vs_original') !=
                dict(depth=True, depth_conf=True, extrinsic=True, intrinsic=True, pose_enc=True)):
        raise ValueError('Frozen original MLP capture identity/architecture/reference gate mismatch')
    state_path, state_sha = _artifact(directory, capture, 'mlp-state.npz')
    activation_path, activation_sha = _artifact(directory, capture, 'activations.npz')
    expected_shapes = {'mlp.fc1.weight': (HIDDEN, CHANNELS), 'mlp.fc1.bias': (HIDDEN,),
                       'mlp.fc2.weight': (CHANNELS, HIDDEN), 'mlp.fc2.bias': (CHANNELS,),
                       'ls2.gamma': (CHANNELS,)}
    with np.load(state_path, allow_pickle=False) as saved:
        if set(saved.files) != set(ARCHIVE_KEYS):
            raise ValueError('Exactly the four original MLP tensors plus excluded LS2 are required')
        state = {key: saved[key] for key in ARCHIVE_KEYS}
    for key, value in state.items():
        if value.shape != expected_shapes[key] or array_record(value) != capture['global0_state'][key]:
            raise ValueError(f'Original parameter shape/raw SHA differs: {key}')
    arrays = {}
    with np.load(activation_path, allow_pickle=False) as saved:
        for name, captured_name in (('input', 'norm2'), ('output', 'mlp_output')):
            value = saved[captured_name]
            if value.shape != ORIGINAL_SHAPE or array_record(value) != capture['arrays'][captured_name]:
                raise ValueError(f'Original {captured_name} shape/raw SHA differs')
            arrays[name] = value
    pins = {str(receipt): expected_capture_sha, str(state_path): state_sha,
            str(activation_path): activation_sha}
    for path, digest in pins.items():
        require_sha(path, digest)
    return capture, arrays, {key: state[key] for key in PARAMETER_KEYS}, pins


def validate_state(state):
    """Generic positive dimensions only for tiny synthetic graph tests."""
    if set(state) != set(PARAMETER_KEYS):
        raise ValueError('Exactly four original MLP parameters are required')
    weight = state['mlp.fc1.weight']
    if weight.ndim != 2 or min(weight.shape) < 1:
        raise ValueError('FC1 must be an original [hidden,channels] weight')
    hidden, channels = weight.shape
    shapes = {'mlp.fc1.weight': (hidden, channels), 'mlp.fc1.bias': (hidden,),
              'mlp.fc2.weight': (channels, hidden), 'mlp.fc2.bias': (channels,)}
    for key, value in state.items():
        if value.shape != shapes[key]:
            raise ValueError(f'Original MLP parameter dimensions mismatch: {key}')
        array_record(value)
    return channels, hidden


def validate_slice(hidden_slice, hidden, *, alignment=1):
    if hidden_slice is None:
        return (0, hidden), True
    if (not isinstance(hidden_slice, (tuple, list)) or len(hidden_slice) != 2
            or any(type(n) is not int for n in hidden_slice)
            or not 0 <= hidden_slice[0] < hidden_slice[1] <= hidden
            or hidden_slice[0] % alignment or hidden_slice[1] % alignment):
        raise ValueError(f'Hidden slice must be increasing in range and aligned to {alignment}')
    return tuple(hidden_slice), False


def build_mlp_onnx(state, bucket, *, hidden_slice=None):
    """Pure static graph construction; never evaluates original weights."""
    import onnx
    from onnx import TensorProto, checker, helper, numpy_helper, shape_inference
    channels, hidden = validate_state(state)
    if type(bucket) is not int or bucket < 1:
        raise ValueError('Static row bucket must be a positive integer')
    (begin, end), includes_bias = validate_slice(hidden_slice, hidden)
    nodes, initializers, bindings = [], [], []

    def constant(name, value):
        initializers.append(numpy_helper.from_array(np.array(value, copy=True, order='C'), name))
        return name

    def parameter(key, value, slicing):
        value = np.ascontiguousarray(value)
        bindings.append(dict(parameter=key, initializer=key, original=array_record(state[key]),
                             hidden_slice=slicing, embedded=array_record(value)))
        return constant(key, value)

    def node(kind, inputs, name, **attrs):
        nodes.append(helper.make_node(kind, inputs, [name], name=name, **attrs))
        return name

    shape = [1, bucket, channels]
    flat = node('Reshape', ['input', constant('flat_shape', np.array([bucket, channels], np.int64))], 'rows')
    w1 = parameter('mlp.fc1.weight', state['mlp.fc1.weight'][begin:end], [begin, end])
    b1 = parameter('mlp.fc1.bias', state['mlp.fc1.bias'][begin:end], [begin, end])
    fc1 = node('Gemm', [flat, w1, b1], 'fc1', alpha=1., beta=1., transA=0, transB=1)
    inv_sqrt2 = constant('gelu_inv_sqrt_two', np.array(1 / math.sqrt(2), np.float32))
    one = constant('gelu_one', np.array(1, np.float32))
    half = constant('gelu_half', np.array(.5, np.float32))
    erf = node('Erf', [node('Mul', [fc1, inv_sqrt2], 'gelu_scaled')], 'gelu_erf')
    gelu = node('Mul', [node('Mul', [fc1, half], 'gelu_half_input'),
                        node('Add', [erf, one], 'gelu_erf_plus_one')], 'gelu')
    w2 = parameter('mlp.fc2.weight', state['mlp.fc2.weight'][:, begin:end], [begin, end])
    fc2_inputs = [gelu, w2]
    if includes_bias:
        fc2_inputs.append(parameter('mlp.fc2.bias', state['mlp.fc2.bias'], None))
    fc2 = node('Gemm', fc2_inputs, 'fc2', alpha=1., beta=1., transA=0, transB=1)
    node('Reshape', [fc2, constant('output_shape', np.array(shape, np.int64))], 'output')
    graph = helper.make_graph(nodes, 'original_VGGT_MLP_static_bucket',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)], initializer=initializers)
    model = helper.make_model(graph, producer_name='Edge-VGGT-SLAM/v23-static-original-MLP',
                              opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 8
    model.doc_string = ('Only original FC1/exact Erf GELU/FC2 on normalized rows. '
        'No norm, attention, LayerScale or residual. Hidden partial graphs omit FC2 bias; '
        'sum all disjoint partial outputs then add original bias once outside graph. '
        'Static validation does not establish compiled precision, NPU coverage or acceleration.')
    checker.check_model(model, full_check=True)
    model = shape_inference.infer_shapes(model, check_type=True, strict_mode=True, data_prop=True)
    checker.check_model(model, full_check=True)
    return model, bindings


def parse_slice(value):
    if not re.fullmatch(r'[0-9]+:[0-9]+', value):
        raise argparse.ArgumentTypeError('Use BEGIN:END for a hidden slice')
    return tuple(int(part) for part in value.split(':'))


def export(args):
    import onnx
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError('Export output already exists')
    if args.bucket not in BUCKETS:
        raise ValueError('Only explicitly supported 874/1024/512 buckets may be exported')
    capture, arrays, weights, pins = load_capture(args.capture_dir, args.capture_sha256)
    hidden_slice, includes_bias = validate_slice(args.hidden_slice, HIDDEN, alignment=64)
    model, bindings = build_mlp_onnx(weights, args.bucket, hidden_slice=args.hidden_slice)
    args.output.mkdir(parents=True, exist_ok=False)
    onnx_path = args.output / 'mlp.onnx'
    with onnx_path.open('xb') as stream:
        stream.write(model.SerializeToString())
    onnx.checker.check_model(str(onnx_path), full_check=True)
    artifacts = {}
    for name, value in (('input.npy', arrays['input']), ('reference.npy', arrays['output']),
                         ('fc2-bias.npy', weights['mlp.fc2.bias'])):
        path = args.output / name
        with path.open('xb') as stream:
            np.save(stream, value, allow_pickle=False)
        artifacts[name] = dict(file=name, bytes=path.stat().st_size, sha256=sha(path), array=array_record(value))
    for path, digest in pins.items():
        require_sha(path, digest)
    shape = [1, args.bucket, CHANNELS]
    report = dict(kind='original_VGGT_MLP_static_export', status='exported_and_structurally_checked',
        script_sha256=sha(__file__), capture_sha256=args.capture_sha256,
        capture_artifacts=capture['artifacts'], original_rows=ORIGINAL_SHAPE[1], bucket=args.bucket,
        input_shape=shape, output_shape=shape, input_name='input', output_name='output',
        hidden_slice=list(hidden_slice), includes_fc2_bias=includes_bias,
        cpu_sum_then_bias_required=not includes_bias,
        original_parameter_metadata={key:array_record(value) for key,value in weights.items()},
        embedded_parameter_bindings=bindings, original_parameter_tensors=4,
        graph_parameter_tensors=len(bindings), excluded_parameter_metadata={'ls2.gamma':capture['global0_state']['ls2.gamma']},
        onnx=dict(file=onnx_path.name, bytes=onnx_path.stat().st_size, sha256=sha(onnx_path)),
        artifacts=artifacts, original_source_hashes=capture['source_hashes'], input_pins=pins,
        selected_pair=capture['selected_pair'], checkpoint=capture['checkpoint'],
        exact_GELU_equation='0.5*x*(1+erf(x/sqrt(2))); approximate=none',
        hidden_slice_contract='FC1 output and bias rows [begin:end], FC2 input columns [begin:end]; partial FC2 bias absent',
        reconstruction_contract='Disjoint slices cover [0,4096) exactly; sum partial outputs in a fixed order in FP32, add original fc2 bias exactly once',
        row_contract='Actual M must be <=bucket; zero-pad normalized input rows, execute, crop first M output rows. Padding is row-independent; do not silently truncate.',
        original_capture_fits_bucket=ORIGINAL_SHAPE[1] <= args.bucket,
        saved_input_is_original_unpadded_capture=True,
        saved_reference_is_original_full_MLP_not_a_partial_slice=True,
        real_arithmetic_partition_equivalence_only=not includes_bias,
        norms_in_graph=False, LayerScale_in_graph=False, residual_in_graph=False,
        original_FP32_weights=True, source_and_capture_artifacts_unchanged=True,
        current_checkout_sources_or_checkpoint_rehashed=False,
        external_data_used=False, node_count=len(model.graph.node),
        op_counts=dict(Counter(n.op_type for n in model.graph.node)), onnx_version=onnx.__version__,
        host_model_inference_executed=False, host_ORT_executed=False, host_images_read=False,
        production_changed=False, npu_execution_validated=False, all_operators_on_npu_verified=False,
        whole_model_quality_validated=False, real_time_validated=False)
    receipt = args.output / 'export.json'
    with receipt.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return report, receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--capture-sha256', default=CAPTURE_SHA)
    parser.add_argument('--bucket', type=int, choices=BUCKETS, default=874)
    parser.add_argument('--hidden-slice', type=parse_slice)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report, receipt = export(args)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps(dict(status=report['status'], onnx=report['onnx'],
                          output=str(args.output), export_sha256=sha(receipt))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
