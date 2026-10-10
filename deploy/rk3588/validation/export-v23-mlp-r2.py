#!/usr/bin/env python3
"""Rank-three original MLP graph with the frozen exact GELU fusion pattern.

Only ONNX construction runs on host. No original model/weights inference runs.
The GELU equation remains exact Erf, but FP32 operation ordering changes from
R1; board ORT/numerical/operator checks remain mandatory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
PARENT_SHA = '56476c724cde537fd35466be0b41d300592821be59f2a840519be6c117c5146e'
GELU_PATTERN_SHA = '59f5ea1be93812571947f7afb71ea9d908057cf277c3cbc2cfb9bbaaf10e5a78'
CAPTURE_SHA = '3664f5c932602a3c6538ea6013ea0389625c42bdcb7dcf10a9fbceb9304abeb1'
BUCKETS = (874, 1024, 512)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            digest.update(part)
    return digest.hexdigest()


def load_parent():
    path = Path(__file__).with_name('export-v23-mlp.py')
    if sha(path) != PARENT_SHA:
        raise ValueError('Frozen R1 MLP exporter source changed')
    spec = importlib.util.spec_from_file_location('v23_mlp_r2_frozen_parent', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_capture(directory, expected_capture_sha=CAPTURE_SHA):
    return load_parent().load_capture(directory, expected_capture_sha)


def build_mlp_onnx(state, bucket, *, hidden_slice=None):
    """Static construction only; generic dimensions support synthetic tests."""
    import onnx
    from onnx import TensorProto, checker, helper, numpy_helper, shape_inference
    parent = load_parent()
    channels, hidden = parent.validate_state(state)
    if type(bucket) is not int or bucket < 1:
        raise ValueError('Static row bucket must be a positive integer')
    (begin, end), includes_bias = parent.validate_slice(hidden_slice, hidden)
    nodes, initializers, bindings = [], [], []

    def constant(name, value):
        initializers.append(numpy_helper.from_array(np.array(value, copy=True, order='C'), name))
        return name

    def parameter(key, value, slicing, transposed=False):
        value = np.ascontiguousarray(value)
        bindings.append(dict(parameter=key, initializer=key, original=parent.array_record(state[key]),
                             hidden_slice=slicing, embedded=parent.array_record(value),
                             weight_transposed_in_initializer=transposed))
        return constant(key, value)

    def node(kind, inputs, name):
        nodes.append(helper.make_node(kind, inputs, [name], name=name))
        return name

    w1 = parameter('mlp.fc1.weight', state['mlp.fc1.weight'][begin:end].T, [begin, end], True)
    b1 = parameter('mlp.fc1.bias', state['mlp.fc1.bias'][begin:end], [begin, end])
    fc1 = node('Add', [node('MatMul', ['input', w1], 'mlp.fc1.matmul'), b1], 'mlp.fc1')
    sqrt2 = constant('gelu_sqrt_two', np.array(math.sqrt(2), np.float32))
    one = constant('gelu_one', np.array(1, np.float32))
    half = constant('gelu_half', np.array(.5, np.float32))
    scaled = node('Div', [fc1, sqrt2], 'gelu_scaled_input')
    erf = node('Erf', [scaled], 'gelu_erf')
    plus_one = node('Add', [erf, one], 'gelu_erf_plus_one')
    before_half = node('Mul', [fc1, plus_one], 'gelu_before_half')
    gelu = node('Mul', [before_half, half], 'gelu')
    w2 = parameter('mlp.fc2.weight', state['mlp.fc2.weight'][:, begin:end].T, [begin, end], True)
    if includes_bias:
        b2 = parameter('mlp.fc2.bias', state['mlp.fc2.bias'], None)
        node('Add', [node('MatMul', [gelu, w2], 'mlp.fc2.matmul'), b2], 'output')
    else:
        node('MatMul', [gelu, w2], 'output')
    shape = [1, bucket, channels]
    graph = helper.make_graph(nodes, 'original_VGGT_MLP_rank3_exact_GELU',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)], initializer=initializers)
    model = helper.make_model(graph, producer_name='Edge-VGGT-SLAM/v23-r2-original-MLP',
                             opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 8
    model.doc_string = ('Original normalized rows -> FC1 -> exact Erf GELU -> FC2. '
        'Rank3 MatMul/Add, pretransposed original weights, no runtime Transpose. '
        'Partial hidden graphs omit FC2 bias. Static construction is not NPU coverage/quality evidence.')
    checker.check_model(model, full_check=True)
    model = shape_inference.infer_shapes(model, check_type=True, strict_mode=True, data_prop=True)
    checker.check_model(model, full_check=True)
    return model, bindings


def export(args):
    """Reuse R1 capture/output framework; publish R2 identity in final receipt."""
    parent = load_parent()
    parent_path = Path(__file__).with_name('export-v23-mlp.py')
    pattern_path = Path(__file__).with_name('export-v12-dino-block-gelu.py')
    if sha(pattern_path) != GELU_PATTERN_SHA:
        raise ValueError('Frozen exact GELU pattern source changed')
    own_sha = sha(__file__)
    source_pins = {str(Path(__file__).resolve()): own_sha,
                   str(parent_path.resolve()): PARENT_SHA,
                   str(pattern_path.resolve()): GELU_PATTERN_SHA}
    parent.build_mlp_onnx = build_mlp_onnx
    report, receipt = parent.export(args)
    report.update(script_sha256=own_sha, exporter_name=Path(__file__).name,
                  parent_exporter=dict(file=parent_path.name, sha256=PARENT_SHA),
                  gelu_pattern_source=dict(file=pattern_path.name, sha256=GELU_PATTERN_SHA),
                  exact_GELU_equation='0.5*(x*(1+erf(x/sqrt(2)))); approximate=none',
                  GELU_node_pattern='Div -> Erf -> Add(1) -> Mul(hidden) -> Mul(0.5)',
                  graph_layout='Rank3 MatMul/Add; transposed original weights embedded once; no Gemm/Reshape/Transpose',
                  fp32_operation_order_changed_from_r1=True,
                  fp32_numerical_equivalence_validated=False,
                  parent_and_pattern_sources_unchanged=True)
    report['input_pins'].update(source_pins)
    for path, digest in report['input_pins'].items():
        parent.require_sha(path, digest)
    # This file was exclusively created by this call, never a preexisting receipt.
    with receipt.open('w') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return report, receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--capture-sha256', default=CAPTURE_SHA)
    parser.add_argument('--bucket', type=int, choices=BUCKETS, default=874)
    parser.add_argument('--hidden-slice', type=load_parent().parse_slice)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report, receipt = export(args)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps(dict(status=report['status'], onnx=report['onnx'], output=str(args.output),
                          export_sha256=sha(receipt))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
