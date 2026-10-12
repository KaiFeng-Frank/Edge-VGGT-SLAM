#!/usr/bin/env python3
"""Compose only original aggregator pair0+pair1, retaining all four outputs.

Static ONNX copying only: no Torch, ORT, RKNN, calibration or model execution.
Every original node and initializer survives reversible namespace changes.
Only the original global0-to-frame1 token view is added. This does not prove
that an RKNN compiler preserves the two separately compiled FP16 boundaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PARENT_SHA = '1872f2ed730f4b6d139562d831661126a088b1ac758249d6a1688cd7e9bfae42'
GRAPH_SHAS = ('efbbaee8f3d6326565930e5a045eac0d500fdb35453fdf85ddac7481ec2fac65',
              '3ea408f67436f6b509ac8c54fd87ce6c0155808d59e5f23135b51f69bfb2b65f')
CHECKPOINT = dict(sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0',
                  bytes=5026874952)
FRAME, GLOBAL = [2, 53, 1024], [1, 106, 1024]
OUTPUT_BINDINGS = [dict(index=2 * i + j, name=f'pair{i}_{role}_output',
                        shape=shape, dtype='float32')
                   for i in (0, 1) for j, (role, shape) in
                   enumerate((('frame', FRAME), ('global', GLOBAL)))]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def artifact(directory, row):
    directory = Path(directory).resolve(strict=True)
    relative = Path(row['file'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Original manifest-relative artifact required')
    path = (directory / relative).resolve(strict=True)
    if (not path.is_relative_to(directory) or path.stat().st_size != row['bytes']
            or sha(path) != row['sha256']):
        raise ValueError('Original artifact bytes/SHA changed: ' + str(relative))
    return path


def compose_original_pair_group(models, rows):
    """Return the fixed two-pair graph and reversible node/tensor proofs."""
    import onnx
    from onnx import TensorProto, helper

    if len(models) != 2 or len(rows) != 2 or [r['index'] for r in rows] != [0, 1]:
        raise ValueError('Only unchanged original pair0 and pair1 are supported')
    first = models[0]
    opsets = [p.SerializeToString() for p in first.opset_import]
    nodes, initializers, infos, outputs, proofs = [], [], [], [], []
    public_names = {'input', *(b['name'] for b in OUTPUT_BINDINGS)}
    for index, (model, row) in enumerate(zip(models, rows)):
        graph = model.graph
        expected_outputs = [dict(index=0, name='frame_output', shape=FRAME, dtype='float32'),
                            dict(index=1, name='global_output', shape=GLOBAL, dtype='float32')]
        if (model.ir_version != first.ir_version
                or [p.SerializeToString() for p in model.opset_import] != opsets
                or model.functions or model.training_info or graph.sparse_initializer
                or len(graph.input) != 1 or len(graph.output) != 2
                or row['role'] != 'pair' or row['input_name'] != 'input'
                or row['input_shape'] != FRAME or row['output_bindings'] != expected_outputs
                or row['heads'] != 16
                or row['all_36_original_parameter_values_preserved_by_parent_composition'] is not True
                or row['parent_nodes_plus_one_reshape'] is not True
                or row['original_aggregator_concat_must_be_retained'] is not True
                or set(row['parameter_raw_sha256']) != {'frame', 'global'}
                or any(len(v) != 18 for v in row['parameter_raw_sha256'].values())):
            raise ValueError('Complete original pair IR/opsets/parameter/IO contract differs')
        for value, name, shape in [(graph.input[0], 'input', FRAME),
                                  (graph.output[0], 'frame_output', FRAME),
                                  (graph.output[1], 'global_output', GLOBAL)]:
            dims = value.type.tensor_type.shape.dim
            if (value.name != name or value.type.tensor_type.elem_type != TensorProto.FLOAT
                    or any(not d.HasField('dim_value') for d in dims)
                    or [d.dim_value for d in dims] != shape):
                raise ValueError('Original fixed FP32 BNC tensor binding differs')
        if (any(t.external_data or t.data_location == TensorProto.EXTERNAL for t in graph.initializer)
                or any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                       for n in graph.node for a in n.attribute)):
            raise ValueError('Only embedded original tensors and flat static graphs are supported')

        prefix = f'pair{index}/'
        overrides = {'input': 'input' if index == 0 else 'pair1_input',
                     'frame_output': OUTPUT_BINDINGS[2 * index]['name'],
                     'global_output': OUTPUT_BINDINGS[2 * index + 1]['name']}
        tensor_names = {v.name for v in (*graph.input, *graph.output, *graph.value_info, *graph.initializer)}
        tensor_names.update(v for n in graph.node for v in (*n.input, *n.output) if v)
        renames = {v: overrides.get(v, prefix + v) for v in tensor_names}
        inverse = {v: k for k, v in renames.items()}
        if len(inverse) != len(renames):
            raise ValueError('Namespaces must be reversible without tensor collisions')
        rename = lambda name: renames[name] if name else name
        parent_nodes, parent_tensors = [], []
        for source in graph.node:
            target = onnx.NodeProto(); target.CopyFrom(source)
            target.name = prefix + source.name if source.name else ''
            target.input[:] = [rename(v) for v in source.input]
            target.output[:] = [rename(v) for v in source.output]
            recovered = onnx.NodeProto(); recovered.CopyFrom(target)
            recovered.name = recovered.name[len(prefix):] if recovered.name else ''
            recovered.input[:] = [inverse[v] if v else v for v in recovered.input]
            recovered.output[:] = [inverse[v] if v else v for v in recovered.output]
            if recovered.SerializeToString() != source.SerializeToString():
                raise ValueError('Original node/attribute protobuf changed')
            nodes.append(target); parent_nodes.append(recovered.SerializeToString())
        for source in graph.initializer:
            target = onnx.TensorProto(); target.CopyFrom(source); target.name = rename(source.name)
            recovered = onnx.TensorProto(); recovered.CopyFrom(target); recovered.name = inverse[target.name]
            if recovered.SerializeToString() != source.SerializeToString():
                raise ValueError('Original initializer payload/metadata changed')
            initializers.append(target); parent_tensors.append(recovered.SerializeToString())
        for source in (*graph.value_info, *graph.input, *graph.output):
            target = onnx.ValueInfoProto(); target.CopyFrom(source); target.name = rename(source.name)
            if target.name not in public_names:
                infos.append(target)
        for source in graph.output:
            target = onnx.ValueInfoProto(); target.CopyFrom(source); target.name = rename(source.name)
            outputs.append(target)
        proofs.append(dict(index=index, original_node_count=len(graph.node),
            original_initializer_count=len(graph.initializer),
            original_nodes_sha256=hashlib.sha256(b''.join(parent_nodes)).hexdigest(),
            original_initializers_sha256=hashlib.sha256(b''.join(parent_tensors)).hexdigest(),
            inverse_namespace_recovers_every_node_and_initializer_protobuf_bytewise=True))
        if index == 0:
            nodes.append(helper.make_node('Reshape', ['pair0_global_output', 'pair0_to_pair1_shape'],
                ['pair1_input'], name='global0_to_frame1_original_view', allowzero=0))
            initializers.append(helper.make_tensor('pair0_to_pair1_shape', TensorProto.INT64, [3], FRAME))

    unique_infos = {}
    for value in infos:
        previous = unique_infos.setdefault(value.name, value)
        if previous.SerializeToString() != value.SerializeToString():
            raise ValueError('Conflicting tensor metadata after original graph composition')
    result = onnx.ModelProto(); result.CopyFrom(first)
    result.graph.CopyFrom(helper.make_graph(nodes, 'original_aggregator_pair0_pair1_group',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, FRAME)], outputs,
        initializer=initializers, value_info=list(unique_infos.values())))
    result.producer_name = 'Edge-VGGT-SLAM/original-v64-static-two-pair-group'
    if (len(result.graph.node) != sum(len(m.graph.node) for m in models) + 1
            or len(result.graph.initializer) != sum(len(m.graph.initializer) for m in models) + 1
            or [v.name for v in result.graph.output] != [b['name'] for b in OUTPUT_BINDINGS]):
        raise ValueError('All original nodes/initializers and four outputs must remain')
    onnx.checker.check_model(result, full_check=True)
    return result, dict(parent_preservation_proofs=proofs,
        original_node_count=sum(len(m.graph.node) for m in models), added_reshape_count=1,
        original_initializer_count=sum(len(m.graph.initializer) for m in models), added_shape_initializer_count=1,
        all_original_parent_nodes_attributes_and_initializer_payloads_retained=True,
        all_72_original_parameter_values_preserved_by_parent_composition=True,
        original_FP32_parent_graph_math_preserved=True,
        only_global0_to_frame1_original_view_added=True,
        all_four_original_pair_outputs_retained=True,
        original_aggregator_concat_must_be_retained=True,
        compiled_boundary_rounding_or_internal_precision_preservation_proven=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    manifest_path = args.manifest.resolve(strict=True)
    if args.manifest_sha256 != PARENT_SHA or sha(manifest_path) != PARENT_SHA:
        raise ValueError('Exact original v55 compiled parent manifest required')
    parent = json.loads(manifest_path.read_text())
    if (parent['schema'] != 'v55_original_VGGT_24_frame_global_pair_RKNN_manifest'
            or parent['checkpoint'] != CHECKPOINT or parent['image_shape'] != [2, 3, 84, 112]
            or parent['depth'] != 24 or parent['aa_order'] != ['frame', 'global']
            or parent['aa_block_size'] != 1 or len(parent['blocks']) != 24
            or [r['index'] for r in parent['blocks']] != list(range(24))):
        raise ValueError('Complete unchanged original 24-pair manifest required')
    rows = parent['blocks'][:2]
    if tuple(r['sha256'] for r in rows) != GRAPH_SHAS:
        raise ValueError('Only the exact original pair0 and pair1 graphs may be composed')
    paths = [artifact(manifest_path.parent, row) for row in rows]
    import onnx
    model, preservation = compose_original_pair_group(
        [onnx.load(str(path), load_external_data=False) for path in paths], rows)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    graph_path = output / 'pair-group-00-01.onnx'
    with graph_path.open('xb') as stream:
        stream.write(model.SerializeToString())
    if sha(manifest_path) != PARENT_SHA or any(sha(path) != row['sha256'] for path, row in zip(paths, rows)):
        raise ValueError('Original manifest/graphs changed during static composition')
    row = dict(role='pair_group', index=0, indices=[0, 1], file=graph_path.name,
        bytes=graph_path.stat().st_size, sha256=sha(graph_path), input_name='input', input_shape=FRAME,
        output_bindings=OUTPUT_BINDINGS, heads=16, parents=rows,
        positions={str(i): r['positions'] for i, r in enumerate(rows)},
        parameter_raw_sha256={str(i): r['parameter_raw_sha256'] for i, r in enumerate(rows)}, **preservation)
    fields = ('checkpoint', 'source_hashes', 'capture_sources', 'capture_sha256', 'image_shape',
              'depth', 'heads', 'frames', 'tokens', 'dim', 'patch_start_idx', 'aa_order', 'aa_block_size')
    report = {field: parent[field] for field in fields}
    report.update(schema='v64_original_VGGT_pair0_pair1_group_ONNX_manifest',
        status='static_export_only_not_runtime_validated', script_sha256=sha(__file__),
        original_parent_manifest_sha256=PARENT_SHA, original_parent_manifest_directory=str(manifest_path.parent),
        original_full_block_parent_manifest_sha256=parent['original_parent_manifest_sha256'],
        blocks=[row], boundary_mode='fp32', exported_pair_indices=[0, 1], candidate_accepted=False,
        calibration_or_ORT_or_SDK_or_model_execution_by_builder=False,
        full_original_model_and_original_aggregator_concat_must_be_retained=True,
        numerical_or_trajectory_quality_or_realtime_validated=False,
        required_original_FP32_rtol=1e-4, required_original_FP32_atol=1e-5)
    with (output / 'export.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=report['status'], export_sha256=sha(output / 'export.json'),
                         graph_sha256=row['sha256'])), flush=True)


if __name__ == '__main__':
    main()
