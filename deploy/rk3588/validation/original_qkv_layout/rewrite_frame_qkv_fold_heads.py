#!/usr/bin/env python3
"""Fold independent frame/head axes inside original frame0 attention only."""
import hashlib
import importlib.util
from pathlib import Path

BASE_SHA = 'cd6db3cd589b383050d4f63a1087340b5d5acf20116f811fc12fd9e1d3e3b194'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def rewrite(model, parent):
    import numpy as np
    import onnx
    from onnx import helper as h, numpy_helper as nh
    source = Path(__file__).with_name('rewrite_frame_qkv_head_major.py')
    require(hashlib.sha256(source.read_bytes()).hexdigest() == BASE_SHA, 'Frozen head-major helper differs')
    spec = importlib.util.spec_from_file_location('_v75_qkv_head_major_base', source)
    q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
    candidate, proof = q.rewrite(model, parent)
    prefix = 'frame0/'
    nodes = {n.name: n for n in candidate.graph.node}
    tensors = {t.name: t for t in candidate.graph.initializer}
    occupied = set(nodes) | set(tensors) | {v for n in candidate.graph.node for v in (*n.input, *n.output)}
    fold, fold_shape, restore, restore_shape = [prefix + name for name in
        ('qkv_fold_frame_heads', 'qkv_fold_frame_heads_shape', 'attention_restore_frame_heads', 'attention_restore_frame_heads_shape')]
    require(not occupied.intersection((fold, fold_shape, restore, restore_shape)), 'New view name collision')
    transpose = nodes[prefix + 'qkv_transpose']
    final = nodes[prefix + 'attention_heads_transpose']
    require(final.op_type == 'Transpose' and list(final.input) == [prefix + 'attention_values']
        and len(final.attribute) == 1 and final.attribute[0].name == 'perm'
        and list(final.attribute[0].ints) == [0, 2, 1, 3], 'Original attention head restoration differs')
    for name in ('query', 'key', 'value'):
        node = nodes[prefix + name]
        require(node.op_type == 'Gather' and node.input[0] == transpose.output[0], 'Original QKV consumer differs')
        node.input[0] = fold
    final.input[0] = restore
    changed_nodes = {prefix + name for name in ('qkv_transpose', 'query', 'key', 'value', 'attention_heads_transpose')}
    added_nodes = {
        fold: h.make_node('Reshape', [transpose.output[0], fold_shape], [fold], name=fold),
        restore: h.make_node('Reshape', [prefix + 'attention_values', restore_shape], [restore], name=restore),
    }
    rebuilt = []
    for node in candidate.graph.node:
        if node.name == final.name:
            rebuilt.append(added_nodes[restore])
        rebuilt.append(node)
        if node.name == transpose.name:
            rebuilt.append(added_nodes[fold])
    del candidate.graph.node[:]; candidate.graph.node.extend(rebuilt)
    candidate.graph.initializer.extend([
        nh.from_array(np.asarray([3, 1, 32, 53, 64], np.int64), fold_shape),
        nh.from_array(np.asarray([2, 16, 53, 64], np.int64), restore_shape),
    ])
    # The fixed frame positions use the same grid for both frames. Keep every
    # original cosine/sine value, with a singleton broadcast over folded heads.
    rope_proofs = []
    for suffix in ('rope_full_cos', 'rope_full_sin'):
        name = prefix + suffix
        old = nh.to_array(tensors[name])
        require(old.dtype == np.float32 and old.shape == (2, 1, 53, 64)
            and np.isfinite(old).all() and old[0].tobytes() == old[1].tobytes(),
            'Both original frame RoPE grids must be byte-identical')
        new = np.ascontiguousarray(old[:1])
        require(np.broadcast_to(new, old.shape).tobytes() == old.tobytes(), 'Original RoPE broadcast changed')
        tensors[name].CopyFrom(nh.from_array(new, name))
        rope_proofs.append(dict(name=name, old_shape=list(old.shape), new_shape=list(new.shape),
            original_raw_sha256=hashlib.sha256(old.tobytes()).hexdigest(),
            broadcast_restored_raw_sha256=hashlib.sha256(np.broadcast_to(new, old.shape).tobytes()).hexdigest()))
    changed_infos = {prefix + 'qkv_reshape'}
    for info in candidate.graph.value_info:
        dims = info.type.tensor_type.shape.dim
        shape = [d.dim_value for d in dims]
        if info.name.startswith(prefix) and len(shape) == 4 and shape[:2] == [2, 16]:
            dims[0].dim_value, dims[1].dim_value = 1, 32
            changed_infos.add(info.name)
        elif info.name in (prefix + 'rope_full_cos', prefix + 'rope_full_sin'):
            require(shape == [2, 1, 53, 64], 'Unexpected RoPE ValueInfo')
            dims[0].dim_value = 1
            changed_infos.add(info.name)
    require(prefix + 'attention_values' in changed_infos
        and all(prefix + n in changed_infos for n in ('query', 'key', 'value')),
        'All original QKV/attention intermediate shape records required')
    candidate.graph.value_info.extend([
        h.make_tensor_value_info(fold, onnx.TensorProto.FLOAT, [3, 1, 32, 53, 64]),
        h.make_tensor_value_info(restore, onnx.TensorProto.FLOAT, [2, 16, 53, 64]),
    ])
    changed_initializers = set(proof['changed_initializers']) | {p['name'] for p in rope_proofs}
    original_nodes = {n.name: n.SerializeToString() for n in model.graph.node}
    original_tensors = {t.name: t.SerializeToString() for t in model.graph.initializer}
    original_infos = {v.name: v.SerializeToString() for v in model.graph.value_info}
    require(set(n.name for n in candidate.graph.node) == set(original_nodes) | set(added_nodes), 'Original nodes missing/duplicated')
    for node in candidate.graph.node:
        if node.name not in changed_nodes | set(added_nodes):
            require(node.SerializeToString() == original_nodes[node.name], 'Other original node changed')
    for tensor in candidate.graph.initializer:
        if tensor.name not in changed_initializers | {fold_shape, restore_shape}:
            require(tensor.SerializeToString() == original_tensors[tensor.name], 'Other original initializer changed')
    for info in candidate.graph.value_info:
        if info.name not in changed_infos | {fold, restore}:
            require(info.SerializeToString() == original_infos[info.name], 'Other original ValueInfo changed')
    require([v.SerializeToString() for v in (*model.graph.input, *model.graph.output)] ==
        [v.SerializeToString() for v in (*candidate.graph.input, *candidate.graph.output)], 'Original public IO changed')
    onnx.checker.check_model(candidate, full_check=True)
    proof.update(variant='head_major_with_frame_head_view_folding', base_rewrite_helper_sha256=BASE_SHA,
        changed_nodes=sorted(changed_nodes), changed_initializers=sorted(changed_initializers),
        changed_value_info=sorted(changed_infos), added_nodes=[fold, restore],
        added_nonparameter_shape_initializers=[fold_shape, restore_shape],
        internal_attention_shape_before=[2, 16, 53, 64], internal_attention_shape_after=[1, 32, 53, 64],
        restored_before_original_attention_heads_transpose=True, original_attention_per_head_matrix_dimensions_preserved=True,
        original_frame_and_head_order_preserved=True, rope_broadcast_proofs=rope_proofs,
        all_original_learned_parameter_values_preserved=True,
        initializer_value_count_unchanged=False,
        nonparameter_RoPE_duplicate_storage_compacted_without_value_change=True,
        candidate_node_count=len(candidate.graph.node))
    return candidate, proof
