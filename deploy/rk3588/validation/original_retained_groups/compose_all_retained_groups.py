#!/usr/bin/env python3
"""Compose all 24 original aggregator pairs at their retained consumer exits.

Static ONNX copying only: no Torch, ORT, RKNN, calibration or model execution.
Every original node and initializer survives reversible namespace changes.
Only original interpair token views are added. All internal frame and global
tensors remain computation tensors. The last group also exposes raw global20
for the original CPU target-token clone and normalization. This does not prove
that an RKNN compiler preserves the separately compiled FP16 boundaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PARENT_SHA = '1872f2ed730f4b6d139562d831661126a088b1ac758249d6a1688cd7e9bfae42'
CHECKPOINT = dict(sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0',
                  bytes=5026874952)
FRAME, GLOBAL = [2, 53, 1024], [1, 106, 1024]
GROUPS = ((0, 4), (5, 11), (12, 17), (18, 23))


def output_bindings(indices):
    bindings = [dict(index=0, name='frame_output', shape=list(FRAME), dtype='float32'),
                dict(index=1, name='global_output', shape=list(GLOBAL), dtype='float32')]
    if indices[0] == 18:
        bindings.append(dict(index=2, name='target_global_output', shape=list(GLOBAL), dtype='float32'))
    return bindings


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


def compose_retained_group(models, rows):
    """Return one allowed original group and reversible node/tensor proofs."""
    import onnx
    from onnx import TensorProto, helper

    indices = [r['index'] for r in rows]
    if (not indices or any(type(i) is not int for i in indices) or len(models) != len(rows)
            or (indices[0], indices[-1]) not in GROUPS
            or indices != list(range(indices[0], indices[-1] + 1))):
        raise ValueError('Only the four complete original retained-consumer groups are supported')
    start, end = indices[0], indices[-1]
    bindings = output_bindings(indices)
    first = models[0]
    opsets = [p.SerializeToString() for p in first.opset_import]
    nodes, initializers, infos, outputs, targets, proofs = [], [], [], [], [], []
    public_names = {'input', *(b['name'] for b in bindings)}
    for ordinal, (model, row) in enumerate(zip(models, rows)):
        index = row['index']
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
        overrides = {'input': 'input' if ordinal == 0 else f'pair{index}_input',
                     'frame_output': 'frame_output' if index == end else f'pair{index}_frame_output',
                     'global_output': ('global_output' if index == end else
                                       'target_global_output' if index == 20 else f'pair{index}_global_output')}
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
        if index == end:
            for source in graph.output:
                target = onnx.ValueInfoProto(); target.CopyFrom(source); target.name = rename(source.name)
                outputs.append(target)
        if index == 20:
            target = onnx.ValueInfoProto(); target.CopyFrom(graph.output[1])
            target.name = rename(target.name); targets.append(target)
        proofs.append(dict(index=index, original_node_count=len(graph.node),
            original_initializer_count=len(graph.initializer),
            original_nodes_sha256=hashlib.sha256(b''.join(parent_nodes)).hexdigest(),
            original_initializers_sha256=hashlib.sha256(b''.join(parent_tensors)).hexdigest(),
            inverse_namespace_recovers_every_node_and_initializer_protobuf_bytewise=True))
        if index < end:
            shape_name = f'pair{index}_to_pair{index + 1}_shape'
            nodes.append(helper.make_node('Reshape', [overrides['global_output'], shape_name],
                [f'pair{index + 1}_input'], name=f'global{index}_to_frame{index + 1}_original_view', allowzero=0))
            initializers.append(helper.make_tensor(shape_name, TensorProto.INT64, [3], FRAME))

    unique_infos = {}
    for value in infos:
        previous = unique_infos.setdefault(value.name, value)
        if previous.SerializeToString() != value.SerializeToString():
            raise ValueError('Conflicting tensor metadata after original graph composition')
    outputs.extend(targets)  # Terminal frame/global first, raw global20 last.
    result = onnx.ModelProto(); result.CopyFrom(first)
    result.graph.CopyFrom(helper.make_graph(nodes, f'original_aggregator_retained_pair_group_{start:02d}_{end:02d}',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, FRAME)], outputs,
        initializer=initializers, value_info=list(unique_infos.values())))
    result.producer_name = 'Edge-VGGT-SLAM/original-v67-static-retained-groups'
    if (len(result.graph.node) != sum(len(m.graph.node) for m in models) + len(rows) - 1
            or len(result.graph.initializer) != sum(len(m.graph.initializer) for m in models) + len(rows) - 1
            or [v.name for v in result.graph.output] != [b['name'] for b in bindings]):
        raise ValueError('All original nodes/initializers and required retained/target exits must remain')
    onnx.checker.check_model(result, full_check=True)
    return result, dict(parent_preservation_proofs=proofs,
        original_node_count=sum(len(m.graph.node) for m in models), added_reshape_count=len(rows) - 1,
        original_initializer_count=sum(len(m.graph.initializer) for m in models), added_shape_initializer_count=len(rows) - 1,
        all_original_parent_nodes_attributes_and_initializer_payloads_retained=True,
        all_original_parameter_values_preserved_by_parent_composition=True,
        original_parameter_value_count=36 * len(rows),
        original_FP32_parent_graph_math_preserved=True,
        only_original_interpair_reshapes_added=True,
        all_original_frame_and_global_computation_tensors_retained=True,
        only_retained_consumer_outputs_public=True,
        public_output_sources=[dict(index=0, name='frame_output', pair_index=end, role='frame'),
                               dict(index=1, name='global_output', pair_index=end, role='global')]
            + ([dict(index=2, name='target_global_output', pair_index=20, role='global')] if start == 18 else []),
        target_global_output_source_pair_index=20 if start == 18 else None,
        target_global_output_is_unnormalized=True if start == 18 else None,
        retained_consumer_indices=[4, 11, 17, 23],
        original_retained_concats_required=True,
        unconsumed_concat_allocations_may_be_omitted=True,
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
            or [(r['role'], r['index']) for r in parent['blocks']] != [('pair', i) for i in range(24)]):
        raise ValueError('Complete unchanged original 24-pair manifest required')
    directory = manifest_path.parent
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    # Runtime positions remain the exact original small NPY files, not recomputed.
    copied_positions = {}
    for row in parent['blocks']:
        for metadata in row['positions'].values():
            key = metadata['file']
            if key in copied_positions:
                if copied_positions[key] != metadata:
                    raise ValueError('Conflicting original position metadata')
                continue
            source = artifact(directory, metadata)
            destination = output / key
            if destination.parent != output:
                raise ValueError('Original position files must be manifest-root relative')
            with source.open('rb') as src, destination.open('xb') as dst:
                for chunk in iter(lambda: src.read(4 << 20), b''):
                    dst.write(chunk)
            if destination.stat().st_size != metadata['bytes'] or sha(destination) != metadata['sha256']:
                raise ValueError('Original position bytes changed during copying')
            copied_positions[key] = metadata
    import onnx
    blocks, graph_paths = [], []
    for group_index, (start, end) in enumerate(GROUPS):
        rows = parent['blocks'][start:end + 1]
        paths = [artifact(directory, row) for row in rows]
        model, preservation = compose_retained_group(
            [onnx.load(str(path), load_external_data=False) for path in paths], rows)
        graph_path = output / f'retained-pair-group-{start:02d}-{end:02d}.onnx'
        with graph_path.open('xb') as stream:
            stream.write(model.SerializeToString())
        del model
        blocks.append(dict(role='retained_pair_group', index=group_index, indices=list(range(start, end + 1)),
            file=graph_path.name, bytes=graph_path.stat().st_size, sha256=sha(graph_path), input_name='input',
            input_shape=list(FRAME), output_bindings=output_bindings(list(range(start, end + 1))), heads=16,
            parents=rows, positions={str(r['index']): r['positions'] for r in rows},
            parameter_raw_sha256={str(r['index']): r['parameter_raw_sha256'] for r in rows}, **preservation))
        graph_paths.extend(zip(paths, rows))
        print(f'Composed original pair group {start}..{end}: {group_index + 1}/4', flush=True)
    if ([i for row in blocks for i in row['indices']] != list(range(24))
            or sum(row['original_parameter_value_count'] for row in blocks) != 864
            or sum(row['added_reshape_count'] for row in blocks) != 20
            or [len(row['output_bindings']) for row in blocks] != [2, 2, 2, 3]):
        raise ValueError('All 24 pairs, 864 original parameter tensors and nine consumer outputs required')
    if sha(manifest_path) != PARENT_SHA or any(sha(path) != row['sha256'] for path, row in graph_paths):
        raise ValueError('Original manifest/graphs changed during static composition')
    for metadata in copied_positions.values():
        artifact(directory, metadata); artifact(output, metadata)
    fields = ('checkpoint', 'source_hashes', 'capture_sources', 'capture_sha256', 'image_shape',
              'depth', 'heads', 'frames', 'tokens', 'dim', 'patch_start_idx', 'aa_order', 'aa_block_size')
    report = {field: parent[field] for field in fields}
    report.update(schema='v67_original_VGGT_retained_groups_ONNX_manifest',
        status='static_export_only_not_runtime_validated', script_sha256=sha(__file__),
        original_parent_manifest_sha256=PARENT_SHA, original_parent_manifest_directory=str(directory),
        original_full_block_parent_manifest_sha256=parent['original_parent_manifest_sha256'],
        blocks=blocks, boundary_mode='fp32', exported_pair_indices=list(range(24)), candidate_accepted=False,
        retained_consumer_indices=[4, 11, 17, 23], original_retained_concats_required=True,
        unconsumed_concat_allocations_may_be_omitted=True, original_target_tokens_global_index=20,
        original_target_token_clone_and_norm_must_be_retained=True, public_target_output_group_index=3,
        public_target_output_index=2, public_target_output_name='target_global_output',
        public_target_output_shape=list(GLOBAL), all_original_24_pairs_retained=True,
        original_parameter_value_count=864, original_small_position_files_copied_bytewise=True,
        calibration_or_ORT_or_SDK_or_model_execution_by_builder=False,
        original_token_preparation_all_other_original_modules_and_SL4_must_be_retained=True,
        full_model_target_token_preservation_validated=False,
        numerical_or_trajectory_quality_or_realtime_validated=False,
        required_original_FP32_rtol=1e-4, required_original_FP32_atol=1e-5)
    with (output / 'export.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=report['status'], export_sha256=sha(output / 'export.json'),
                         group_count=len(blocks), pair_count=24, consumer_output_count=9)), flush=True)


if __name__ == '__main__':
    main()
