#!/usr/bin/env python3
"""Static composition of four consecutive, unchanged original DINO blocks.

No Torch, ORT, calibration or SDK execution. ``fp32`` preserves the parent
ONNX operations; ``fp16-roundtrip`` additionally requests rounding at the three
internal block boundaries. Those Casts are not compiler barriers and neither
mode proves equivalence to the separately compiled FP16 blocks. Original token
preparation, position interpolation and final norm remain outside this graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SHAPE = [1, 53, 1024]
CHECKPOINT = {'sha256': 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0',
              'bytes': 5026874952}
PARAMETERS = (
    'norm1.weight', 'norm1.bias', 'attn.qkv.weight', 'attn.qkv.bias',
    'attn.proj.weight', 'attn.proj.bias', 'ls1.gamma', 'norm2.weight',
    'norm2.bias', 'mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight',
    'mlp.fc2.bias', 'ls2.gamma')
TRANSPOSED = {'attn.qkv.weight', 'attn.proj.weight', 'mlp.fc1.weight', 'mlp.fc2.weight'}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def artifact(directory, row):
    relative = Path(row['file'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Original manifest-relative graph required')
    path = (directory / relative).resolve(strict=True)
    if not path.is_relative_to(directory) or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
        raise ValueError('Original ONNX artifact size/SHA mismatch: ' + str(relative))
    return path


def compose_original_dino_group(models, rows, *, boundary_mode='fp32'):
    """Return (ModelProto, provenance) for exactly four ordered original layers."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    if boundary_mode not in ('fp32', 'fp16-roundtrip') or len(models) != 4 or len(rows) != 4:
        raise ValueError('Exactly four parents and an explicit supported boundary mode required')
    indices = [row['index'] for row in rows]
    if any(type(index) is not int for index in indices) or indices != list(range(indices[0], indices[0] + 4)):
        raise ValueError('Four consecutive original DINO indices required')
    if not 0 <= indices[0] <= 20:
        raise ValueError('Group must lie inside original DINO layers 0..23')
    first = models[0]
    opsets = [entry.SerializeToString() for entry in first.opset_import]
    nodes, initializers, infos, bindings = [], [], [], {}
    wires = 'input'
    parent_nodes = 0
    for ordinal, (model, row) in enumerate(zip(models, rows)):
        graph, index = model.graph, row['index']
        if (model.ir_version != first.ir_version
                or [entry.SerializeToString() for entry in model.opset_import] != opsets
                or model.functions or model.training_info or graph.sparse_initializer
                or len(graph.input) != 1 or len(graph.output) != 1):
            raise ValueError('Same static IR/opsets and simple complete parent graphs required')
        if (row['role'] != 'dino' or row['input_shape'] != SHAPE or row['output_shape'] != SHAPE
                or row['heads'] != 16 or row['epsilons'] != {'norm1': 1e-6, 'norm2': 1e-6}
                or row['gelu_approximate'] != 'none' or row['qk_norm'] is not False or row['rope'] is not None
                or row['all_14_caller_parameter_values_preserved_in_graph'] is not True
                or set(row['parameter_raw_sha256']) != set(PARAMETERS)):
            raise ValueError('Original 14-parameter DINO contract differs')
        for value, expected_name in ((graph.input[0], row['input_name']), (graph.output[0], row['output_name'])):
            dims = value.type.tensor_type.shape.dim
            if (value.name != expected_name or value.type.tensor_type.elem_type != TensorProto.FLOAT
                    or any(not dim.HasField('dim_value') for dim in dims)
                    or [dim.dim_value for dim in dims] != SHAPE):
                raise ValueError('Each original parent must retain its fixed FP32 BNC IO')
        if any(tensor.external_data or tensor.data_location == TensorProto.EXTERNAL for tensor in graph.initializer):
            raise ValueError('Only the original embedded ONNX tensors are supported')
        if any(attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
               for node in graph.node for attribute in node.attribute):
            raise ValueError('Nested control-flow graphs are outside this minimal composer')

        prefix = f'dino{index:02d}/'
        output_name = 'output' if ordinal == 3 else prefix + 'output'
        overrides = {graph.input[0].name: wires, graph.output[0].name: output_name}

        def rename(value):
            return overrides.get(value, prefix + value) if value else value

        tensors = {tensor.name: tensor for tensor in graph.initializer}
        parameter_bindings = {}
        for parameter in PARAMETERS:
            name = parameter + '_transposed' if parameter in TRANSPOSED else parameter
            value = numpy_helper.to_array(tensors[name])
            original_value = value.T if parameter in TRANSPOSED else value
            raw = np.ascontiguousarray(original_value)
            if (raw.dtype != np.float32 or not np.isfinite(raw).all()
                    or hashlib.sha256(memoryview(raw).cast('B')).hexdigest() != row['parameter_raw_sha256'][parameter]):
                raise ValueError(f'dino{index}: original parameter payload differs: {parameter}')
            parameter_bindings[parameter] = dict(initializer=rename(name),
                original_raw_sha256=row['parameter_raw_sha256'][parameter],
                original_shape=list(raw.shape), onnx_shape=list(value.shape))
        bindings[str(index)] = parameter_bindings

        # Copy every original node/attribute and initializer payload; change only
        # the namespace and wire names. No pattern rewrite or operator removal.
        for source in graph.node:
            target = onnx.NodeProto()
            target.CopyFrom(source)
            target.name = prefix + source.name if source.name else ''
            target.input[:] = [rename(value) for value in source.input]
            target.output[:] = [rename(value) for value in source.output]
            nodes.append(target)
        for source in graph.initializer:
            target = onnx.TensorProto()
            target.CopyFrom(source)
            target.name = rename(source.name)
            initializers.append(target)
        for source in list(graph.value_info) + [graph.output[0]]:
            target = onnx.ValueInfoProto()
            target.CopyFrom(source)
            target.name = rename(source.name)
            if target.name not in ('input', 'output'):
                infos.append(target)
        parent_nodes += len(graph.node)
        wires = output_name
        if ordinal != 3 and boundary_mode == 'fp16-roundtrip':
            half, wide = prefix + 'boundary_fp16', prefix + 'boundary_fp32'
            nodes.extend((helper.make_node('Cast', [wires], [half], name=half, to=TensorProto.FLOAT16),
                          helper.make_node('Cast', [half], [wide], name=wide, to=TensorProto.FLOAT)))
            infos.extend((helper.make_tensor_value_info(half, TensorProto.FLOAT16, SHAPE),
                          helper.make_tensor_value_info(wide, TensorProto.FLOAT, SHAPE)))
            wires = wide

    # Value_info may already contain a parent's public output. Deduplicate it.
    unique_infos = {}
    for value in infos:
        previous = unique_infos.setdefault(value.name, value)
        if previous.SerializeToString() != value.SerializeToString():
            raise ValueError('Conflicting original tensor metadata after composition')
    result = helper.make_model(helper.make_graph(nodes,
        f'original_DINO_group_{indices[0]:02d}_{indices[-1]:02d}',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, SHAPE)],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, SHAPE)],
        initializer=initializers, value_info=list(unique_infos.values())),
        producer_name='Edge-VGGT-SLAM/original-v58-static-DINO-group',
        opset_imports=list(first.opset_import))
    result.ir_version = first.ir_version
    expected_casts = 6 if boundary_mode == 'fp16-roundtrip' else 0
    if len(result.graph.node) != parent_nodes + expected_casts:
        raise ValueError('Every original node plus only the requested boundary Casts must remain')
    onnx.checker.check_model(result, full_check=True)
    return result, dict(parameter_bindings=bindings, original_node_count=parent_nodes,
        added_boundary_cast_count=expected_casts, initializer_count=len(initializers),
        all_original_parent_nodes_attributes_and_initializer_payloads_retained=True,
        original_FP32_parent_graph_math_preserved=boundary_mode == 'fp32',
        compiled_boundary_rounding_or_internal_precision_preservation_proven=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--boundary-mode', choices=('fp32', 'fp16-roundtrip'), default='fp32')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve(strict=True)
    if sha(manifest_path) != args.manifest_sha256:
        raise ValueError('Pinned original DINO manifest required')
    parent = json.loads(manifest_path.read_text())
    if (parent['schema'] not in ('v54_original_VGGT_24_DINO_block_ONNX_manifest',
                                  'v54_original_VGGT_24_DINO_block_RKNN_manifest')
            or parent['checkpoint'] != CHECKPOINT or parent['image_shape'] != [1, 3, 84, 112]
            or parent['depth'] != 24 or parent['chunked_blocks'] is not False
            or [(row['role'], row['index']) for row in parent['blocks']] != [('dino', index) for index in range(24)]
            or not 0 <= args.start_index <= 20):
        raise ValueError('Complete original 24-layer DINO manifest and in-range group required')
    rows = parent['blocks'][args.start_index:args.start_index + 4]
    paths = [artifact(manifest_path.parent, row) for row in rows]
    import onnx
    models = [onnx.load(str(path), load_external_data=False) for path in paths]
    model, preservation = compose_original_dino_group(models, rows, boundary_mode=args.boundary_mode)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    graph_path = output / f'dino-group-{args.start_index:02d}-{args.start_index + 3:02d}.onnx'
    with graph_path.open('xb') as stream:
        stream.write(model.SerializeToString())
    if sha(manifest_path) != args.manifest_sha256:
        raise ValueError('Parent manifest changed during composition')
    for path, row in zip(paths, rows):
        if sha(path) != row['sha256']:
            raise ValueError('Parent graph changed during composition')
    row = dict(role='dino_group', index=args.start_index, indices=[item['index'] for item in rows],
        file=graph_path.name, bytes=graph_path.stat().st_size, sha256=sha(graph_path),
        input_name='input', output_name='output', input_shape=SHAPE, output_shape=SHAPE,
        heads=16, boundary_mode=args.boundary_mode, parents=rows,
        parameter_raw_sha256={str(item['index']): item['parameter_raw_sha256'] for item in rows},
        original_capture_references={key: dict(input=rows[0]['original_capture_references'][key]['input'],
            output=rows[-1]['original_capture_references'][key]['output']) for key in ('single0', 'single1', 'batch2')},
        **preservation)
    fields = ('checkpoint', 'source_hashes', 'capture_sources', 'capture_sha256', 'image_shape',
              'original_reference_capture_directory', 'original_non_block_parameter_raw_sha256')
    report = {key: parent[key] for key in fields}
    report.update(schema='v58_original_VGGT_4_DINO_block_group_ONNX_manifest',
        status='static_export_only_not_runtime_validated', script_sha256=sha(__file__),
        original_parent_manifest_sha256=args.manifest_sha256, boundary_mode=args.boundary_mode,
        complete_DINO_depth=24, exported_group_size=4, exported_start_index=args.start_index,
        blocks=[row], candidate_accepted=False,
        calibration_or_ORT_or_SDK_or_model_execution_by_builder=False,
        original_prepare_position_interpolation_final_norm_and_other_20_layers_must_be_retained=True,
        compiler_may_fold_casts_fuse_operators_or_change_precision=True,
        full_original_outputs_and_separate_FP16_outputs_require_board_comparison=True,
        required_original_FP32_rtol=1e-4, required_original_FP32_atol=1e-5,
        numerical_or_trajectory_quality_or_realtime_validated=False)
    with (output / 'export.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=report['status'], group_indices=row['indices'],
                         export_sha256=sha(output / 'export.json'))), flush=True)


if __name__ == '__main__':
    main()
