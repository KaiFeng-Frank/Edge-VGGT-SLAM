#!/usr/bin/env python3
"""Write a bounded original pair00 frame0 QKV/attention layout candidate.

Only the root operator runs this entry point on RK3588S. This script performs
static ONNX work, not Torch/ORT/SDK inference. The original QKV dot products,
separate Q/K 64-channel LayerNorms, RoPE tree, all 53-key attention reductions,
and all original learned/constant initializer values are retained. Independent
FP32 execution and complete-pair RKNN timing are required before adoption.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import sys

sys.dont_write_bytecode = True
BASE = Path('/mnt/vggt-data/vggt-slam2')
PARENT_SHA = '1872f2ed730f4b6d139562d831661126a088b1ac758249d6a1688cd7e9bfae42'
GRAPH_SHA = 'efbbaee8f3d6326565930e5a045eac0d500fdb35453fdf85ddac7481ec2fac65'
PROPOSAL_SHA = 'fece257e672ee9ce229b1823bcc676aa75be83da4b776380d62a220ee91b0c4f'
CHECKPOINT = dict(bytes=5026874952,
                  sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')
SCHEMA = 'v86_original_pair00_frame_QK_rows_ONNX_rewrite'
SUCCESS = 'candidate_written_not_FP32_accepted'
OUTPUT_BINDINGS = [dict(index=0, name='frame_output', shape=[2, 53, 1024], dtype='float32'),
                   dict(index=1, name='global_output', shape=[1, 106, 1024], dtype='float32')]
PREFIX = 'frame0/'
REMOVED_SUFFIXES = ('qkv_reshape', 'qkv_transpose')
ROW_SUFFIXES = ('query', 'key', 'value', 'attn.q_norm', 'query_rope_permuted',
    'query_rope_rotated', 'query_rope_cosine', 'query_rope_sine', 'query_rope',
    'attn.k_norm', 'key_rope_permuted', 'key_rope_rotated', 'key_rope_cosine',
    'key_rope_sine', 'key_rope')
SCORE_SUFFIXES = ('attention_scores', 'attention_scaled_scores', 'attention_probabilities')
MODIFIED_SUFFIXES = ('query', 'key', 'value', 'query_rope_permuted', 'key_rope_permuted',
    'query_rope_cosine', 'query_rope_sine', 'key_rope_cosine', 'key_rope_sine',
    'key_transpose', 'attention_values')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def protobuf_sha(value):
    return hashlib.sha256(value.SerializeToString()).hexdigest()


def artifact(path):
    path = Path(path)
    return dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path))


def raw(np, array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def attributes(onnx, node):
    require(len({item.name for item in node.attribute}) == len(node.attribute),
            'Duplicate original attributes: ' + node.name)
    return {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}


def dims(info):
    shape = info.type.tensor_type.shape.dim
    require(all(item.HasField('dim_value') and item.dim_value > 0 for item in shape),
            'Static positive original shape required: ' + info.name)
    return [item.dim_value for item in shape]


def generated_layout_check(np):
    """Integer labels establish every row/channel identity, not model accuracy."""
    labels = np.arange(2 * 53 * 3072, dtype=np.int64).reshape(2, 53, 3072)
    old = labels.reshape(2, 53, 3, 16, 64).transpose(2, 0, 3, 1, 4)
    for index, operand in enumerate(np.split(labels, 3, axis=-1)):
        new = operand.reshape(2, 53, 16, 64).transpose(0, 2, 1, 3).reshape(32, 53, 64)
        require(np.array_equal(old[index].reshape(32, 53, 64), new),
                'QKV row/channel identity differs')
        require(np.array_equal(old[index], new.reshape(2, 16, 53, 64)),
                'Restored original head-major identity differs')
    return dict(integer_element_identity_passed=True, original_QKV_shape=[2, 53, 3072],
                original_operand_shape=[2, 16, 53, 64], candidate_operand_shape=[32, 53, 64],
                batch_heads=32, tokens_per_head=53, normalization_channels=64,
                score_K_reduction_channels=64, probability_V_reduction_tokens=53,
                actual_model_math_executed=False)


def rewrite(model, parent):
    """Return (fresh ModelProto, exact scoped proof) for the pinned original pair."""
    import numpy as np
    import onnx
    from onnx import numpy_helper as nh

    require(protobuf_sha(model) == GRAPH_SHA, 'Exact frozen original pair00 protobuf required')
    require(parent.get('role') == 'pair' and parent.get('index') == 0
            and parent.get('sha256') == GRAPH_SHA and parent.get('output_bindings') == OUTPUT_BINDINGS,
            'Pinned original pair00 row and complete public outputs required')
    require(model.ir_version == 8 and [(o.domain, o.version) for o in model.opset_import] == [('', 17)],
            'Original ONNX IR/opset contract differs')
    require(not model.graph.sparse_initializer and not model.functions,
            'Unexpected original sparse/function graph')
    nodes = {n.name: n for n in model.graph.node}
    inits = {t.name: t for t in model.graph.initializer}
    infos = {v.name: v for v in model.graph.value_info}
    require(len(nodes) == len(model.graph.node) == 87
            and len(inits) == len(model.graph.initializer) == 63
            and len(infos) == len(model.graph.value_info), 'Original unique graph catalog differs')
    require(all(t.data_location != onnx.TensorProto.EXTERNAL and not t.external_data
                for t in model.graph.initializer), 'Embedded original initializers required')
    require(len(model.graph.input) == 1 and model.graph.input[0].name == 'input'
            and dims(model.graph.input[0]) == [2, 53, 1024]
            and model.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT,
            'Original public input differs')
    require([(v.name, dims(v), v.type.tensor_type.elem_type) for v in model.graph.output]
            == [(b['name'], b['shape'], onnx.TensorProto.FLOAT) for b in OUTPUT_BINDINGS],
            'Original complete public outputs differ')

    parameter_hashes = {}
    for role in ('frame', 'global'):
        parameters = parent['parameter_raw_sha256'][role]
        require(len(parameters) == 18, 'All eighteen original block parameters required')
        parameter_hashes[role] = {}
        for name, digest in parameters.items():
            transpose = name in ('attn.qkv.weight', 'attn.proj.weight', 'mlp.fc1.weight', 'mlp.fc2.weight')
            tensor = role + '0/' + name + ('_transposed' if transpose else '')
            require(tensor in inits, 'Missing original learned initializer: ' + tensor)
            value = nh.to_array(inits[tensor])
            require(value.dtype == np.float32 and np.isfinite(value).all(),
                    'Original FP32 learned initializer differs: ' + tensor)
            actual = raw(np, value.T if transpose else value)
            require(actual == digest, 'Original learned parameter bytes differ: ' + tensor)
            parameter_hashes[role][name] = actual
    require(sum(len(rows) for rows in parameter_hashes.values()) == 36,
            'All thirty-six original parameters required')

    def original(suffix, op, inputs, attrs=None):
        name = PREFIX + suffix
        require(name in nodes, 'Missing original node: ' + name)
        node = nodes[name]
        require(node.domain == '' and node.op_type == op and list(node.input) == inputs
                and list(node.output) == [name], 'Original node boundary differs: ' + name)
        if attrs is not None:
            require(attributes(onnx, node) == attrs, 'Original node attributes differ: ' + name)
        return node

    original('attn.qkv.matmul', 'MatMul', [PREFIX + 'norm1', PREFIX + 'attn.qkv.weight_transposed'], {})
    original('attn.qkv', 'Add', [PREFIX + 'attn.qkv.matmul', PREFIX + 'attn.qkv.bias'], {})
    original('qkv_reshape', 'Reshape', [PREFIX + 'attn.qkv', PREFIX + 'qkv_shape'], {})
    original('qkv_transpose', 'Transpose', [PREFIX + 'qkv_reshape'], dict(perm=[2, 0, 3, 1, 4]))
    require(np.array_equal(nh.to_array(inits[PREFIX + 'qkv_shape']), np.asarray([2, 53, 3, 16, 64], np.int64)),
            'Original QKV head shape differs')
    for index, name in enumerate(('query', 'key', 'value')):
        original(name, 'Gather', [PREFIX + 'qkv_transpose', PREFIX + name + '_index'], dict(axis=0))
        value = nh.to_array(inits[PREFIX + name + '_index'])
        require(value.dtype == np.int64 and value.shape == () and int(value) == index,
                'Original scalar QKV index differs')

    norm_attributes = {}
    for operand, short in (('query', 'q'), ('key', 'k')):
        name = 'attn.' + short + '_norm'
        node = original(name, 'LayerNormalization', [PREFIX + operand,
            PREFIX + name + '.weight', PREFIX + name + '.bias'])
        attrs = attributes(onnx, node)
        require(set(attrs) == {'axis', 'epsilon', 'stash_type'} and attrs['axis'] == -1
                and attrs['stash_type'] == 1 and abs(attrs['epsilon'] - 1e-5) < 1e-12,
                'Original independent last64 FP32-stash LayerNorm differs')
        norm_attributes[short] = attrs
        for affine in ('weight', 'bias'):
            value = nh.to_array(inits[PREFIX + name + '.' + affine])
            require(value.dtype == np.float32 and value.shape == (64,) and np.isfinite(value).all(),
                    'Original separate64-channel affine differs')
        original(operand + '_rope_permuted', 'Gather', [PREFIX + name, PREFIX + 'rope_permutation'], dict(axis=3))
        original(operand + '_rope_rotated', 'Mul', [PREFIX + operand + '_rope_permuted', PREFIX + 'rope_rotation_sign'], {})
        original(operand + '_rope_cosine', 'Mul', [PREFIX + name, PREFIX + 'rope_full_cos'], {})
        original(operand + '_rope_sine', 'Mul', [PREFIX + operand + '_rope_rotated', PREFIX + 'rope_full_sin'], {})
        original(operand + '_rope', 'Add', [PREFIX + operand + '_rope_cosine', PREFIX + operand + '_rope_sine'], {})
    expected_rotation = np.asarray(list(range(16, 32)) + list(range(16))
                                   + list(range(48, 64)) + list(range(32, 48)), np.int64)
    require(np.array_equal(nh.to_array(inits[PREFIX + 'rope_permutation']), expected_rotation),
            'Original two32-channel rotation permutation differs')
    expected_sign = np.asarray([-1.0] * 16 + [1.0] * 16 + [-1.0] * 16 + [1.0] * 16, np.float32)
    require(np.array_equal(nh.to_array(inits[PREFIX + 'rope_rotation_sign']), expected_sign),
            'Original two32-channel rotation sign differs')
    table_proof = {}
    for suffix in ('rope_full_cos', 'rope_full_sin'):
        table = nh.to_array(inits[PREFIX + suffix])
        require(table.dtype == np.float32 and table.shape == (2, 1, 53, 64)
                and np.isfinite(table).all() and table[0].tobytes() == table[1].tobytes(),
                'Original frame RoPE tables cannot use identical independent head-row broadcast')
        require(np.array_equal(np.broadcast_to(table, (2, 16, 53, 64)).reshape(32, 53, 64),
                               np.broadcast_to(table[0], (32, 53, 64))),
                'Original every-row RoPE table value differs')
        table_proof[suffix] = dict(original_shape=list(table.shape), derived_shape=[1, 53, 64],
            original_raw_sha256=raw(np, table), derived_raw_sha256=raw(np, table[0]),
            both_original_frame_tables_byte_equal=True, every_head_row_table_equal=True,
            derivation='Gather axis=0 using unchanged original scalar query_index=0')

    original('key_transpose', 'Transpose', [PREFIX + 'key_rope'], dict(perm=[0, 1, 3, 2]))
    original('attention_scores', 'MatMul', [PREFIX + 'query_rope', PREFIX + 'key_transpose'], {})
    original('attention_scaled_scores', 'Mul', [PREFIX + 'attention_scores', PREFIX + 'attention_scale'], {})
    original('attention_probabilities', 'Softmax', [PREFIX + 'attention_scaled_scores'], dict(axis=-1))
    original('attention_values', 'MatMul', [PREFIX + 'attention_probabilities', PREFIX + 'value'], {})
    original('attention_heads_transpose', 'Transpose', [PREFIX + 'attention_values'], dict(perm=[0, 2, 1, 3]))
    original('attention_heads_flatten', 'Reshape', [PREFIX + 'attention_heads_transpose', PREFIX + 'output_shape'], {})
    scale = nh.to_array(inits[PREFIX + 'attention_scale'])
    require(scale.dtype == np.float32 and scale.shape == () and float(scale) == 0.125,
            'Original all64-channel attention scale differs')
    require(np.array_equal(nh.to_array(inits[PREFIX + 'output_shape']), np.asarray([2, 53, 1024], np.int64)),
            'Original consumer flatten shape differs')
    require([n.name for n in model.graph.node if PREFIX + 'qkv_reshape' in n.input] == [PREFIX + 'qkv_transpose']
            and [n.name for n in model.graph.node if PREFIX + 'qkv_transpose' in n.input]
            == [PREFIX + 'query', PREFIX + 'key', PREFIX + 'value'],
            'Removed original QKV intermediates have external consumers')
    for suffix in ROW_SUFFIXES:
        require(PREFIX + suffix in infos and dims(infos[PREFIX + suffix]) == [2, 16, 53, 64],
                'Original head-major ValueInfo differs: ' + suffix)
    for suffix in SCORE_SUFFIXES:
        require(dims(infos[PREFIX + suffix]) == [2, 16, 53, 53], 'Original score ValueInfo differs')
    require(dims(infos[PREFIX + 'key_transpose']) == [2, 16, 64, 53]
            and dims(infos[PREFIX + 'attention_values']) == [2, 16, 53, 64],
            'Original attention consumer layout differs')

    layout_proof = generated_layout_check(np)
    candidate = copy.deepcopy(model)
    h = onnx.helper
    new_constants = {
        PREFIX + 'qk_rows_split_sizes': [1024, 1024, 1024],
        PREFIX + 'qk_rows_operand_heads_shape': [2, 53, 16, 64],
        PREFIX + 'qk_rows_operand_shape': [32, 53, 64],
        PREFIX + 'qk_rows_restore_shape': [2, 16, 53, 64]
    }
    require(not set(new_constants) & set(inits), 'New shape initializer name collision')
    for name, shape in new_constants.items():
        candidate.graph.initializer.append(nh.from_array(np.asarray(shape, np.int64), name))

    def make(suffix, op, inputs, outputs=None, **attrs):
        name = PREFIX + suffix
        return h.make_node(op, inputs, outputs or [name], name=name, **attrs)

    split_suffixes = [name + '_qk_rows_split' for name in ('query', 'key', 'value')]
    producer_nodes = [make('qk_rows_split', 'Split', [PREFIX + 'attn.qkv', PREFIX + 'qk_rows_split_sizes'],
                           outputs=[PREFIX + name for name in split_suffixes], axis=-1)]
    added_infos = {}
    for name, split in zip(('query', 'key', 'value'), split_suffixes):
        producer_nodes.extend([
            make(name + '_qk_rows_heads_view', 'Reshape', [PREFIX + split, PREFIX + 'qk_rows_operand_heads_shape']),
            make(name + '_qk_rows_heads_transpose', 'Transpose', [PREFIX + name + '_qk_rows_heads_view'], perm=[0, 2, 1, 3]),
            make(name, 'Reshape', [PREFIX + name + '_qk_rows_heads_transpose', PREFIX + 'qk_rows_operand_shape'])])
        added_infos[PREFIX + split] = [2, 53, 1024]
        added_infos[PREFIX + name + '_qk_rows_heads_view'] = [2, 53, 16, 64]
        added_infos[PREFIX + name + '_qk_rows_heads_transpose'] = [2, 16, 53, 64]
    table_nodes = [make('qk_rows_rope_cosine_table', 'Gather', [PREFIX + 'rope_full_cos', PREFIX + 'query_index'], axis=0),
                   make('qk_rows_rope_sine_table', 'Gather', [PREFIX + 'rope_full_sin', PREFIX + 'query_index'], axis=0)]
    for table in table_nodes:
        added_infos[table.output[0]] = [1, 53, 64]
    changes = {}
    for operand in ('query', 'key'):
        name = PREFIX + operand + '_rope_permuted'
        node = copy.deepcopy(nodes[name])
        require(len(node.attribute) == 1 and node.attribute[0].name == 'axis', 'Original Gather axis field differs')
        node.attribute[0].i = 2
        changes[name] = node
        for suffix, table in (('cosine', 'qk_rows_rope_cosine_table'), ('sine', 'qk_rows_rope_sine_table')):
            name = PREFIX + operand + '_rope_' + suffix
            node = copy.deepcopy(nodes[name])
            node.input[1] = PREFIX + table
            changes[name] = node
    name = PREFIX + 'key_transpose'
    node = copy.deepcopy(nodes[name])
    require(len(node.attribute) == 1 and node.attribute[0].name == 'perm', 'Original key transpose field differs')
    del node.attribute[0].ints[:]
    node.attribute[0].ints.extend([0, 2, 1])
    changes[name] = node
    name = PREFIX + 'attention_values'
    node = copy.deepcopy(nodes[name])
    node.output[0] = PREFIX + 'qk_rows_attention_values'
    changes[name] = node
    added_infos[node.output[0]] = [32, 53, 64]
    restore = make('qk_rows_restore_head_major', 'Reshape',
                   [node.output[0], PREFIX + 'qk_rows_restore_shape'], outputs=[PREFIX + 'attention_values'])

    replacements = []
    for node in model.graph.node:
        if node.name == PREFIX + 'qkv_reshape':
            replacements.extend(producer_nodes)
        elif node.name in {PREFIX + 'qkv_transpose', PREFIX + 'query', PREFIX + 'key', PREFIX + 'value'}:
            continue
        else:
            if node.name == PREFIX + 'attn.q_norm':
                replacements.extend(table_nodes)
            replacements.append(copy.deepcopy(changes.get(node.name, node)))
            if node.name == PREFIX + 'attention_values':
                replacements.append(restore)
    del candidate.graph.node[:]
    candidate.graph.node.extend(replacements)
    new_shapes = {PREFIX + suffix: [32, 53, 64] for suffix in ROW_SUFFIXES}
    new_shapes.update({PREFIX + suffix: [32, 53, 53] for suffix in SCORE_SUFFIXES})
    new_shapes[PREFIX + 'key_transpose'] = [32, 64, 53]
    kept_infos = []
    for value in model.graph.value_info:
        if value.name in {PREFIX + suffix for suffix in REMOVED_SUFFIXES}:
            continue
        result = copy.deepcopy(value)
        if result.name in new_shapes:
            result.type.tensor_type.shape.CopyFrom(h.make_tensor_value_info(
                result.name, onnx.TensorProto.FLOAT, new_shapes[result.name]).type.tensor_type.shape)
        kept_infos.append(result)
    require(not set(added_infos) & set(infos), 'New intermediate ValueInfo name collision')
    kept_infos.extend(h.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
                      for name, shape in added_infos.items())
    del candidate.graph.value_info[:]
    candidate.graph.value_info.extend(kept_infos)

    after_nodes = {n.name: n for n in candidate.graph.node}
    after_infos = {v.name: v for v in candidate.graph.value_info}
    require(len(after_nodes) == len(candidate.graph.node) == 95
            and len(after_infos) == len(candidate.graph.value_info), 'Candidate node/ValueInfo names collide')
    removed = [name for name in nodes if name not in after_nodes]
    modified = [name for name in nodes if name in after_nodes
                and nodes[name].SerializeToString() != after_nodes[name].SerializeToString()]
    added = [name for name in after_nodes if name not in nodes]
    require(removed == [PREFIX + suffix for suffix in REMOVED_SUFFIXES]
            and set(modified) == {PREFIX + suffix for suffix in MODIFIED_SUFFIXES}
            and len(added) == 10, 'Unexpected named node diff')
    require(all(after_nodes[n.name].SerializeToString() == n.SerializeToString()
                for n in model.graph.node if n.name not in set(removed) | set(modified)),
            'Unrelated original node changed')
    require([t.SerializeToString() for t in candidate.graph.initializer[:63]]
            == [t.SerializeToString() for t in model.graph.initializer],
            'An original learned/constant initializer changed or moved')
    require(len(candidate.graph.initializer) == 67 and all(t.data_type == onnx.TensorProto.INT64
            for t in candidate.graph.initializer[63:]), 'Only four new int64 layout constants allowed')
    require([v.SerializeToString() for v in (*candidate.graph.input, *candidate.graph.output)]
            == [v.SerializeToString() for v in (*model.graph.input, *model.graph.output)], 'Public IO changed')
    require(all(after_infos[v.name].SerializeToString() == v.SerializeToString()
                for v in model.graph.value_info if v.name not in new_shapes
                and v.name not in set(removed)), 'Unrelated original ValueInfo changed')
    require(all(after_nodes[n.name].SerializeToString() == n.SerializeToString()
                for n in model.graph.node if n.name.startswith('global0/') or n.name == 'frame_to_global_view'),
            'Original global branch/boundary changed')
    for suffix in ('attn.q_norm', 'attn.k_norm', 'attention_heads_transpose', 'attention_heads_flatten'):
        require(after_nodes[PREFIX + suffix].SerializeToString() == nodes[PREFIX + suffix].SerializeToString(),
                'Original norm or complete consumer transpose/flatten changed')
    # Undo only documented node/ValueInfo edits and appended constants. This
    # establishes protobuf scope without executing any floating-point operator.
    inverse = copy.deepcopy(candidate)
    del inverse.graph.node[:]
    inverse.graph.node.extend(model.graph.node)
    del inverse.graph.value_info[:]
    inverse.graph.value_info.extend(model.graph.value_info)
    del inverse.graph.initializer[63:]
    require(inverse.SerializeToString() == model.SerializeToString(),
            'Restoring only documented scope did not recover original protobuf')
    del inverse
    onnx.checker.check_model(candidate, full_check=True)
    require(protobuf_sha(model) == GRAPH_SHA, 'Original source protobuf mutated')
    diff = dict(removed=[dict(name=name, original_op=nodes[name].op_type,
                             original_protobuf_sha256=protobuf_sha(nodes[name])) for name in removed],
                modified=[dict(name=name, original_op=nodes[name].op_type, candidate_op=after_nodes[name].op_type,
                    original_protobuf_sha256=protobuf_sha(nodes[name]), candidate_protobuf_sha256=protobuf_sha(after_nodes[name]))
                    for name in modified],
                added=[dict(name=name, op=after_nodes[name].op_type, protobuf_sha256=protobuf_sha(after_nodes[name])) for name in added])
    proof = dict(original_parameter_tensors_verified=36, original_parameter_raw_sha256=parameter_hashes,
        original_initializer_count=63, candidate_initializer_count=67,
        original_initializer_protobuf_sha256={t.name: protobuf_sha(t) for t in model.graph.initializer},
        added_INT64_layout_constants={name: shape for name, shape in new_constants.items()},
        original_node_count=87, candidate_node_count=95, named_node_diff=diff,
        changed_value_info=[dict(name=name, original_shape=dims(infos[name]), candidate_shape=shape)
                            for name, shape in new_shapes.items()],
        removed_value_info=[PREFIX + suffix for suffix in REMOVED_SUFFIXES], added_value_info=added_infos,
        all_original_initializers_protobuf_unchanged=True, all_original_parameter_values_byte_equal=True,
        all_unrelated_nodes_protobuf_unchanged=True, all_unrelated_value_info_protobuf_unchanged=True,
        original_IO_protobuf_unchanged=True, global0_protobuf_unchanged=True,
        original_qk_affine_and_LayerNorm_attributes_preserved=True, original_qk_norm_attributes=norm_attributes,
        original_RoPE_values_and_rotation_preserved=True, original_RoPE_table_proof=table_proof,
        all_original_tokens_and_attention_reductions_preserved=True, original_attention_scale=0.125,
        original_dot_K_reduction_order_unchanged=True, original_probability_V_reduction_order_unchanged=True,
        original_consumer_head_major_and_transpose_flatten_preserved=True,
        original_pair_recovered_bytewise_by_restoring_only_documented_edits=True,
        generated_layout_check=layout_proof, learned_parameter_permutation_applied=False,
        LayerNorm_mean_variance_or_affine_formula_changed=False, quantization_or_kernel_changes=False,
        original_heads=16, original_frame_batches=2, original_tokens_per_head=53, original_head_channels=64,
        candidate_FP32_numerical_equivalence_validated=False, performance_validated=False,
        full_model_or_trajectory_validated=False, candidate_accepted=False)
    return candidate, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-manifest', type=Path,
                        default=BASE / 'experiments/vggt-v55-pair-graphs-r1/compiled.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    require(platform.machine() == 'aarch64'
            and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),
            'RK3588S board root operator required before graph import')
    source = Path(__file__).resolve(strict=True)
    manifest = args.parent_manifest.resolve(strict=True)
    require(sha(manifest) == PARENT_SHA, 'Original complete pair manifest changed')
    ancestor = json.loads(manifest.read_text())
    require(ancestor['schema'] == 'v55_original_VGGT_24_frame_global_pair_RKNN_manifest'
            and ancestor['checkpoint'] == CHECKPOINT, 'Original parent schema/checkpoint provenance differs')
    parent = ancestor['blocks'][0]
    require(parent['role'] == 'pair' and parent['index'] == 0 and parent['sha256'] == GRAPH_SHA,
            'Original pair00 manifest row required')
    graph = (manifest.parent / parent['file']).resolve(strict=True)
    require(graph.is_relative_to(manifest.parent) and sha(graph) == GRAPH_SHA
            and graph.stat().st_size == parent['bytes'], 'Original embedded pair00 graph changed')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    pins = {str(path): sha(path) for path in (source, manifest, graph)}
    report = dict(schema=SCHEMA, status='running', script_sha256=pins[str(source)],
        static_proposal_sha256=PROPOSAL_SHA, original_parent_manifest_sha256=PARENT_SHA,
        original_pair_onnx_sha256=GRAPH_SHA, original_checkpoint=ancestor['checkpoint'], parent=parent,
        original_model_only=True, original_input_binding=dict(name='input', shape=[2, 53, 1024], dtype='float32'),
        original_output_bindings=OUTPUT_BINDINGS, candidate_accepted=False, quality_accepted=False,
        actual_model_OR_Torch_OR_ORT_OR_SDK_OR_GPU_executed=False, host_GPU_used=False,
        full_model_or_trajectory_or_realtime_validated=False, source_and_input_hashes=pins,
        required_next_gate='Independent board original ONNX versus candidate FP32 on v53/v56-calib00: both complete outputs, rtol1e-4/atol1e-5',
        required_next_performance='Same original calibration, complete-pair interleaved flags0 A/B; separate flags8 placement/fusion diagnostic')
    def write():
        (output / 'export.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    write()
    try:
        import onnx
        model = onnx.load(str(graph), load_external_data=False)
        candidate, proof = rewrite(model, parent)
        path = output / 'pair00-frame-qk-rows.onnx'
        require(not path.exists(), 'Candidate graph destination must be new')
        onnx.save_model(candidate, str(path), save_as_external_data=False)
        report.update(candidate=artifact(path), proof=proof, ONNX_version=onnx.__version__, status=SUCCESS)
        report['source_and_input_hashes_after'] = {name: sha(Path(name)) for name in pins}
        report['source_and_inputs_unchanged'] = report['source_and_input_hashes_after'] == pins
        require(report['source_and_inputs_unchanged'], 'Frozen original source or inputs changed during rewrite')
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        write()
        raise
    write()
    print(json.dumps(dict(status=report['status'], candidate=report['candidate'],
                         candidate_accepted=False)), flush=True)


if __name__ == '__main__':
    main()
