#!/usr/bin/env python3
"""V19 static-only original aggregator graph with a two-dimensional MLP.

The frozen R3 capture validation, 18 parameters, attention, RoPE, exact Erf
GELU, LayerScale and residuals are retained. Only the two MLP MatMul/Add pairs
become alpha=beta=1 Gemm between explicit flatten/restore Reshapes. No Torch,
real-image processing, model inference, ORT or host trigonometry is executed.
Compilation and synthetic equivalence do not establish hardware execution.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
R3_PATH = HERE / 'export-v17-aggregator-block-r3.py'
R3_SHA = '6724aba806806ec36d8ba097f0c2076f3703e106f23c83d4272ae2e3a3fc2874'
_R3 = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def r3_module():
    global _R3
    if _R3 is None:
        if sha(R3_PATH) != R3_SHA:
            raise ValueError('Frozen R3 aggregator exporter SHA mismatch')
        spec = importlib.util.spec_from_file_location('_v19_private_r3', R3_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _R3 = module
    return _R3


SOURCE_PINS = dict(r3_module().SOURCE_PINS)
GRAPH_HELPERS = {**r3_module().GRAPH_HELPERS, R3_PATH.name: R3_SHA}
PROFILE_SCRIPT_SHA = r3_module().PROFILE_SCRIPT_SHA


def load_capture(directory, expected_profile_sha, block):
    values = r3_module().load_capture(directory, expected_profile_sha, block)
    values[-1][str(R3_PATH)] = R3_SHA
    values[-1][str(Path(__file__).resolve())] = sha(__file__)
    return values


def _static_shape(value):
    dimensions = value.type.tensor_type.shape.dim
    if len(dimensions) != 3 or any(not d.HasField('dim_value') or d.dim_value <= 0 for d in dimensions):
        raise ValueError('Fixed positive rank-three block input is required')
    return tuple(int(d.dim_value) for d in dimensions)


def rewrite_mlp2d(original):
    """Return a new graph; all existing initializers/unrelated nodes stay exact."""
    from onnx import checker, helper, numpy_helper, shape_inference

    model = copy.deepcopy(original)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError('One original block input and output are required')
    shape = _static_shape(model.graph.input[0])
    batch, tokens, channels = shape
    patterns = {
        'mlp.fc1.matmul': ('MatMul', ['norm2', 'mlp.fc1.weight_transposed'], ['mlp.fc1.matmul']),
        'mlp.fc1': ('Add', ['mlp.fc1.matmul', 'mlp.fc1.bias'], ['mlp.fc1']),
        'mlp.fc2.matmul': ('MatMul', ['gelu', 'mlp.fc2.weight_transposed'], ['mlp.fc2.matmul']),
        'mlp.fc2': ('Add', ['mlp.fc2.matmul', 'mlp.fc2.bias'], ['mlp.fc2']),
    }
    by_name = {node.name: node for node in model.graph.node}
    if len(by_name) != len(model.graph.node):
        raise ValueError('Unique original node names are required')
    for name, (kind, inputs, outputs) in patterns.items():
        node = by_name.get(name)
        if (node is None or node.op_type != kind or list(node.input) != inputs
                or list(node.output) != outputs or node.attribute):
            raise ValueError(f'Frozen R3 MLP pattern mismatch: {name}')
    for output, expected in (('mlp.fc1.matmul', 'mlp.fc1'), ('mlp.fc2.matmul', 'mlp.fc2')):
        if [node.name for node in model.graph.node if output in node.input] != [expected]:
            raise ValueError('MLP MatMul intermediate has an unexpected consumer')
        if output in [item.name for item in model.graph.output]:
            raise ValueError('MLP MatMul intermediate must not be a graph output')
    reserved = {'mlp2d_flat_shape', 'mlp2d_restore_shape', 'mlp2d_input', 'mlp2d_output', 'mlp2d_restore'}
    tensor_names = {name for node in model.graph.node for name in list(node.input) + list(node.output)}
    if reserved & (tensor_names | set(by_name)):
        raise ValueError('V19 MLP tensor/node names already exist')
    arrays = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    if len(arrays) != len(model.graph.initializer) or reserved & set(arrays):
        raise ValueError('Unique unchanged initializer names are required')
    required_shapes = {'mlp.fc1.weight_transposed': (channels, 4 * channels),
                       'mlp.fc1.bias': (4 * channels,),
                       'mlp.fc2.weight_transposed': (4 * channels, channels),
                       'mlp.fc2.bias': (channels,)}
    for name, expected in required_shapes.items():
        value = arrays.get(name)
        if value is None or value.shape != expected or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f'Unchanged original FP32 MLP initializer required: {name}')
    original_initializers = [item.SerializeToString() for item in model.graph.initializer]
    original_unrelated = {node.name: node.SerializeToString() for node in model.graph.node if node.name not in patterns}
    replacements = {
        'mlp.fc1.matmul': helper.make_node('Reshape', ['norm2', 'mlp2d_flat_shape'],
                                         ['mlp2d_input'], name='mlp2d_input'),
        'mlp.fc1': helper.make_node('Gemm', ['mlp2d_input', 'mlp.fc1.weight_transposed', 'mlp.fc1.bias'],
                                   ['mlp.fc1'], name='mlp.fc1', alpha=1., beta=1., transA=0, transB=0),
        'mlp.fc2.matmul': helper.make_node('Gemm', ['gelu', 'mlp.fc2.weight_transposed', 'mlp.fc2.bias'],
                                          ['mlp2d_output'], name='mlp.fc2', alpha=1., beta=1., transA=0, transB=0),
        'mlp.fc2': helper.make_node('Reshape', ['mlp2d_output', 'mlp2d_restore_shape'],
                                   ['mlp.fc2'], name='mlp2d_restore'),
    }
    nodes = [replacements.get(node.name, node) for node in model.graph.node]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    for name, values in (('mlp2d_flat_shape', [batch * tokens, channels]),
                         ('mlp2d_restore_shape', list(shape))):
        model.graph.initializer.append(numpy_helper.from_array(np.array(values, dtype=np.int64), name))
    # GELU remains the exact same five nodes; it now receives an equivalent 2D tensor.
    # Clear previous intermediate ranks before strict ONNX shape inference.
    del model.graph.value_info[:]
    checker.check_model(model, full_check=True)
    model = shape_inference.infer_shapes(model, strict_mode=True, check_type=True, data_prop=True)
    checker.check_model(model, full_check=True)
    if original_initializers != [item.SerializeToString() for item in model.graph.initializer[:-2]]:
        raise ValueError('MLP rewrite changed an original initializer')
    remaining = {node.name: node.SerializeToString() for node in model.graph.node
                 if node.name not in ('mlp2d_input', 'mlp.fc1', 'mlp.fc2', 'mlp2d_restore')}
    if original_unrelated != remaining:
        raise ValueError('MLP rewrite changed attention, GELU, LayerScale or residual nodes')
    model.producer_name = 'vggt-slam2-rk3588s/v19-aggregator-mlp2d-static-export'
    model.doc_string += (' V19 only flattens the MLP rows, uses two alpha=beta=1/transA=transB=0 Gemm '
                         'with unchanged FP32 initializers, and restores the original rank before ls2/residual. '
                         'No K splitting or activation approximation. Actual board execution is not established.')
    return model


def build_aggregator_onnx(*args, **kwargs):
    return rewrite_mlp2d(r3_module().build_aggregator_onnx(*args, **kwargs))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--profile-sha256', required=True)
    parser.add_argument('--block', choices=('frame0', 'global0'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--r3-onnx', type=Path)
    parser.add_argument('--r3-onnx-sha256')
    args = parser.parse_args(argv)
    if (args.r3_onnx is None) != (args.r3_onnx_sha256 is None):
        parser.error('--r3-onnx and --r3-onnx-sha256 must be supplied together')
    import onnx
    from onnx import checker

    base = r3_module().r2_module().base_parts()[0]
    profile, row, arrays, weights, pins = load_capture(args.capture, args.profile_sha256, args.block)
    shape = tuple(arrays['input'].shape)
    original = r3_module().build_aggregator_onnx(weights, shape, arrays['pos'], arrays['rope_cos'], arrays['rope_sin'],
        heads=row['num_heads'], epsilons={name: row[field] for name, field in base.EPS_FIELDS.items()})
    original_serialized = original.SerializeToString()
    if args.r3_onnx is not None:
        base.require_sha(args.r3_onnx, args.r3_onnx_sha256)
        if args.r3_onnx.read_bytes() != original_serialized:
            raise ValueError('Supplied R3 ONNX is not the exact graph rebuilt from the frozen capture')
        pins[str(args.r3_onnx.resolve())] = args.r3_onnx_sha256
    graph = rewrite_mlp2d(original)
    for name, expected in GRAPH_HELPERS.items():
        pins[str(HERE / name)] = expected
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    onnx_path = output / 'block.onnx'
    with onnx_path.open('xb') as stream:
        stream.write(graph.SerializeToString())
    checker.check_model(str(onnx_path), full_check=True)
    artifacts = {}
    for key in ('input', 'output'):
        path = output / f'{key}.npy'
        with path.open('xb') as stream:
            np.save(stream, arrays[key], allow_pickle=False)
        metadata = base.array_record(arrays[key])
        artifacts[path.name] = dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path),
            shape=metadata['shape'], dtype=metadata['dtype'], raw_sha256=metadata['sha256'])
    for path, expected in pins.items():
        base.require_sha(path, expected)
    report = dict(kind='original_aggregator_MLP_2D_Gemm_static_export',
        status='exported_and_structurally_checked', script_sha256=sha(__file__),
        profile_sha256=args.profile_sha256, block=args.block, input_shape=list(shape),
        input_name='input', output_name='output',
        onnx=dict(file=onnx_path.name, bytes=onnx_path.stat().st_size, sha256=sha(onnx_path)),
        capture_directory=str(args.capture.resolve()), capture_artifacts=row['artifacts'], artifacts=artifacts,
        architecture={key: row[key] for key in ('num_heads', 'head_dim', 'norm1_eps', 'norm2_eps',
            'q_norm_eps', 'k_norm_eps', 'gelu_approximate', 'rope_frequency', 'rope_scaling_factor')},
        captured_array_metadata=row['arrays'], captured_parameter_metadata=row['weights'],
        parameter_count=18, original_weights=True, source_pins=SOURCE_PINS, frozen_helpers=GRAPH_HELPERS,
        input_pins=pins, selected_pair=profile['selected_pair'], checkpoint=profile['checkpoint'],
        original_R3_graph_sha256=hashlib.sha256(original_serialized).hexdigest(),
        rewrite=dict(mlp_rows=shape[0] * shape[1], flat_input_shape=[shape[0] * shape[1], shape[2]],
            restored_output_shape=list(shape), Gemm_attributes=dict(alpha=1., beta=1., transA=0, transB=0),
            original_initializer_count=len(original.graph.initializer), original_initializers_byte_identical=True,
            added_shape_initializers=2, unrelated_nodes_byte_identical=True,
            exact_Erf_GELU_nodes_unchanged=True, weights_changed=False, attention_changed=False, K_split=False),
        node_count=len(graph.graph.node), op_counts=dict(Counter(node.op_type for node in graph.graph.node)),
        onnx_version=onnx.__version__, position_and_tables_fixed=True, host_trigonometry_recomputed=False,
        host_model_inference_executed=False, host_ORT_executed=False, external_data=False, external_data_used=False,
        production_changed=False, all_operators_on_npu_verified=False, real_time_validated=False,
        whole_model_quality_validated=False,
        numerical_scope='Synthetic ReferenceEvaluator checks are separate; real tensors are only loaded/copied/exported here.')
    report_path = output / 'export.json'
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], onnx=report['onnx'], receipt=str(report_path), sha256=sha(report_path))))
    return 0


if __name__ == '__main__':
    sys.exit(main())
