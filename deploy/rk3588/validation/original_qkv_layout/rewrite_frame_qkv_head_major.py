#!/usr/bin/env python3
"""Reorder only original pair00 frame0 QKV channels; no execution at import.

The caller must run all real graph checks/builds on RK3588S. This helper keeps
each dot product's K order, every original parameter value and all public IO.
It does not establish RKNN placement, FP16 equality or a performance benefit.
"""
import hashlib

PARENT_SHA = '1872f2ed730f4b6d139562d831661126a088b1ac758249d6a1688cd7e9bfae42'
GRAPH_SHA = 'efbbaee8f3d6326565930e5a045eac0d500fdb35453fdf85ddac7481ec2fac65'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def channel_permutation(heads=16, head_dim=64):
    """New channel j reads old channel permutation[j]; inverse restores bytes."""
    import numpy as np
    require(type(heads) is int and type(head_dim) is int and heads > 0 and head_dim > 0,
            'Positive integer heads/head_dim required')
    order = np.arange(3 * heads * head_dim, dtype=np.int64).reshape(3, heads, head_dim)
    permutation = order.transpose(1, 0, 2).reshape(-1).copy()
    inverse = np.argsort(permutation)
    require(np.array_equal(permutation[inverse], np.arange(permutation.size)),
            'QKV permutation is not bijective')
    return permutation, inverse


def rewrite(model, parent):
    """Return (fresh candidate ModelProto, proof), accepting only pinned pair00."""
    import numpy as np
    import onnx
    from onnx import numpy_helper as nh

    require(hashlib.sha256(model.SerializeToString()).hexdigest() == GRAPH_SHA,
            'Exact frozen original pair00 protobuf required')
    require(parent.get('role') == 'pair' and parent.get('index') == 0
            and parent.get('sha256') == GRAPH_SHA, 'Pinned original pair00 row required')
    candidate = type(model)()
    candidate.CopyFrom(model)
    nodes = {n.name: n for n in candidate.graph.node}
    tensors = {t.name: t for t in candidate.graph.initializer}
    require(len(nodes) == len(candidate.graph.node)
            and len(tensors) == len(candidate.graph.initializer), 'Unique original names required')
    prefix = 'frame0/'
    wn, bn, sn = [prefix + s for s in ('attn.qkv.weight_transposed', 'attn.qkv.bias', 'qkv_shape')]
    mm, add, reshape, transpose = [nodes[prefix + s] for s in
                                 ('attn.qkv.matmul', 'attn.qkv', 'qkv_reshape', 'qkv_transpose')]
    require(mm.op_type == 'MatMul' and list(mm.input) == [prefix + 'norm1', wn]
            and list(mm.output) == [mm.name]
            and add.op_type == 'Add' and list(add.input) == [mm.name, bn]
            and list(add.output) == [add.name]
            and reshape.op_type == 'Reshape' and list(reshape.input) == [add.name, sn]
            and list(reshape.output) == [reshape.name]
            and transpose.op_type == 'Transpose' and list(transpose.input) == [reshape.name]
            and list(transpose.output) == [transpose.name], 'Original frame QKV pattern differs')
    require(len(transpose.attribute) == 1 and transpose.attribute[0].name == 'perm'
            and list(transpose.attribute[0].ints) == [2, 0, 3, 1, 4], 'Original QKV transpose differs')
    for index, name in enumerate(('query', 'key', 'value')):
        gather = nodes[prefix + name]
        value = nh.to_array(tensors[prefix + name + '_index'])
        require(gather.op_type == 'Gather' and list(gather.input) == [transpose.name, prefix + name + '_index']
                and list(gather.output) == [gather.name] and len(gather.attribute) == 1
                and gather.attribute[0].name == 'axis' and gather.attribute[0].i == 0
                and value.dtype == np.int64 and value.shape == () and int(value) == index,
                'Original Q/K/V extraction differs')
    consumers = {name: [n.name for n in candidate.graph.node if name in n.input] for name in (wn, bn, sn)}
    require(consumers == {wn: [mm.name], bn: [add.name], sn: [reshape.name]},
            'Reordered QKV parameter/shape has another consumer')
    wt, bias, shape = [nh.to_array(tensors[name]) for name in (wn, bn, sn)]
    old_shape, new_shape = [2, 53, 3, 16, 64], [2, 53, 16, 3, 64]
    require(wt.shape == (1024, 3072) and bias.shape == (3072,)
            and wt.dtype == bias.dtype == np.float32 and np.isfinite(wt).all() and np.isfinite(bias).all()
            and shape.dtype == np.int64 and np.array_equal(shape, old_shape),
            'Original QKV parameter/shape differs')
    raw = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    original = {'attn.qkv.weight': raw(wt.T), 'attn.qkv.bias': raw(bias)}
    require(all(parent['parameter_raw_sha256']['frame'][name] == value for name, value in original.items()),
            'Original QKV checkpoint parameter values differ')
    permutation, inverse = channel_permutation()
    new_wt, new_bias = np.ascontiguousarray(wt[:, permutation]), np.ascontiguousarray(bias[permutation])
    reconstructed = {'attn.qkv.weight': raw(new_wt[:, inverse].T), 'attn.qkv.bias': raw(new_bias[inverse])}
    require(original == reconstructed, 'Inverse QKV parameter reconstruction failed')
    require(np.array_equal(new_wt[:, inverse], wt) and np.array_equal(new_bias[inverse], bias),
            'Inverse QKV values differ')
    for name, array in ((wn, new_wt), (bn, new_bias), (sn, np.asarray(new_shape, np.int64))):
        tensors[name].CopyFrom(nh.from_array(array, name))
    del transpose.attribute[0].ints[:]
    transpose.attribute[0].ints.extend([3, 0, 2, 1, 4])
    infos = [v for v in candidate.graph.value_info if v.name == reshape.name]
    require(len(infos) == 1 and [d.dim_value for d in infos[0].type.tensor_type.shape.dim] == old_shape,
            'Original QKV intermediate ValueInfo differs')
    for dim, size in zip(infos[0].type.tensor_type.shape.dim, new_shape):
        dim.dim_value = size
    require(len(model.graph.node) == len(candidate.graph.node)
            and len(model.graph.initializer) == len(candidate.graph.initializer)
            and len(model.graph.value_info) == len(candidate.graph.value_info), 'Original graph counts changed')
    for before, after in zip(model.graph.node, candidate.graph.node):
        if before.name != transpose.name:
            require(before.SerializeToString() == after.SerializeToString(), 'Other original node changed')
    for before, after in zip(model.graph.initializer, candidate.graph.initializer):
        if before.name not in (wn, bn, sn):
            require(before.SerializeToString() == after.SerializeToString(), 'Other original initializer changed')
    for before, after in zip(model.graph.value_info, candidate.graph.value_info):
        if before.name != reshape.name:
            require(before.SerializeToString() == after.SerializeToString(), 'Other original ValueInfo changed')
    require([v.SerializeToString() for v in (*model.graph.input, *model.graph.output)]
            == [v.SerializeToString() for v in (*candidate.graph.input, *candidate.graph.output)],
            'Original public IO changed')
    require(sum(int(np.prod(t.dims)) for t in model.graph.initializer)
            == sum(int(np.prod(t.dims)) for t in candidate.graph.initializer), 'Initializer value count changed')
    onnx.checker.check_model(candidate, full_check=True)
    require(hashlib.sha256(model.SerializeToString()).hexdigest() == GRAPH_SHA, 'Source model mutated')
    proof = dict(original_parent_manifest_sha256=PARENT_SHA, original_pair_onnx_sha256=GRAPH_SHA,
                 role='frame', index=0, old_order='QKV_head_component', new_order='head_QKV_component',
                 changed_initializers=[wn, bn, sn], changed_nodes=[transpose.name],
                 changed_value_info=[reshape.name], old_qkv_shape=old_shape, new_qkv_shape=new_shape,
                 old_qkv_transpose_perm=[2, 0, 3, 1, 4], new_qkv_transpose_perm=[3, 0, 2, 1, 4],
                 original_parameter_raw_sha256=original, inverse_reconstructed_parameter_raw_sha256=reconstructed,
                 all_other_original_nodes_protobuf_unchanged=True,
                 all_other_original_initializers_protobuf_unchanged=True,
                 all_other_original_value_info_protobuf_unchanged=True, original_IO_protobuf_unchanged=True,
                 global0_protobuf_unchanged=True, original_query_key_value_order_restored=True,
                 original_dot_K_reduction_order_unchanged=True,
                 all_original_parameter_values_preserved_by_inverse_permutation=True,
                 initializer_value_count_unchanged=True, original_node_count=len(model.graph.node),
                 candidate_node_count=len(candidate.graph.node), additional_outputs=0,
                 FP16_numerical_equivalence_validated=False, performance_validated=False,
                 candidate_accepted=False)
    return candidate, proof
