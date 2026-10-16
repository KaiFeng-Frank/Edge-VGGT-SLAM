#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-2-Clause
"""Portable static control for the original VGGT QKV prefix.

The pure graph transformation is preserved from the executed source pinned in
HISTORICAL_SOURCE. Default mode reads no artifacts and imports no tensor or SDK
library. --execute constructs ONNX only; it performs no inference or calibration.
Exact parent artifacts contain original weights and must be supplied separately
under their own licenses. This program does not download or distribute them.
"""
import argparse
import base64
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct

HISTORICAL_SOURCE = {'bytes': 29930, 'sha256': '6f0cfa49c639cb61a73000150f9a78cca50332211b616fd890f208776cc89ada'}

PARENT_PINS = dict(canonical=(13674604, '4fd773f0e1bf9290bc0e972cd5ec69eeed75dda3bd6a2903df82f72cf1dd60ce'),
    native_half=(13678979, '5d81f4c76f9bc31fdfbfaf843b1afc1a4f3ee86671db2060bbc85f014c368c36'))


PRIOR_STATIC_PIN = (30677, 'f60ec2c4d34b84ff7ce53debe4122c8dd03c9eb707858edfefda134cf32bcfff')


CAPTURE_PIN = (12587, 'e4ffaea2b11df9ac558c119cb4854d490b0030c02b7706eee344b8894cefa686')


SCHEMA = 'original_native518_CF_RoPE_direct_rank4_static'


SUCCESS = 'static_source_inverses_layout_and_shape_checks_passed_CPU_and_NPU_gates_pending'


KEYS = ('direct4_control', 'cf_canonical', 'cf_native')


COUNTS = dict(direct4_control=(38, 26), cf_canonical=(24, 16), cf_native=(40, 27))


N, H, D, ROWS = 2082, 16, 64, 694


INPUT = '__global_input'


INPUT_SHAPE = [1, N, 1024]


FULL = [1, H, N, D]


ORIGINAL_OUTPUTS = ('global0/query_rope', 'global0/key_rope', 'global0/value')


NS = '__original_native_layout/'


CF = '__original_CF_RoPE/'


RANGES = ((0, 694), (694, 1388), (1388, 2082))


ROLES = (('Q0', ROWS), ('Q1', ROWS), ('Q2', ROWS), ('K', N), ('V', N))


LEARNED = ('global0/norm1.weight', 'global0/norm1.bias', 'global0/attn.qkv.weight_transposed',
    'global0/attn.qkv.bias', 'global0/attn.q_norm.weight', 'global0/attn.q_norm.bias',
    'global0/attn.k_norm.weight', 'global0/attn.k_norm.bias')


ROPE_NAMES = tuple('global0/' + role + '_rope' + suffix for role in ('query', 'key')
                  for suffix in ('_permuted', '_rotated', '_cosine', '_sine', ''))


ROPE_CONSTANTS = ('global0/rope_rotation_sign', 'global0/rope_full_cos', 'global0/rope_full_sin')


PROOF_TRUE = (
    'all8_original_learned_TensorProtos_byte_unchanged',
    'original10_non_RoPE_NodeProtos_byte_unchanged',
    'direct4_control_original20_NodeProtos_byte_unchanged',
    'direct4_control_inverse_to_native_half_parent_byte_equal',
    'cf_canonical_inverse_to_canonical_parent_byte_equal',
    'cf_native_inverse_to_canonical_parent_byte_equal',
    'all3_rewritten_RoPE_TensorProtos_inverse_byte_equal',
    'original10_RoPE_operations_and_arithmetic_order_preserved',
    'original_BHSD_LayerNormalization_NodeProtos_unchanged',
    'full2082_position_RoPE_precedes_all_query_partition_slices',
    'all2082_queries_once_and_all_original_KV_retained',
    'direct4_word_order_equals_old_flat_and_native8_consumer',
    'all_added_constants_are_INT64_layout_or_slice_indices',
    'ONNX_checker_shape_and_closed_single_input_checks_passed')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def descriptor(path, pin=None):
    path = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(4 << 20), b''):
            h.update(part)
    row = dict(file=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())
    require(pin is None or (row['bytes'], row['sha256']) == tuple(pin), 'Pinned source differs: ' + str(path))
    return row


def output_contract(key=None):
    direct = [dict(role=role, name=NS + role, shape=[2, rows, D, 8], dtype='float16',
        bytes=2 * rows * D * 8 * 2, consumer_view_shape=[1, 2, rows, D, 8]) for role, rows in ROLES]
    canonical = [dict(role=role, name=name, shape=FULL[:], dtype='float32', bytes=H * N * D * 4)
                 for role, name in zip(('query', 'key', 'value'), ORIGINAL_OUTPUTS)]
    result = dict(direct4_control=direct, cf_canonical=canonical, cf_native=copy.deepcopy(direct))
    require(key is None or key in result, 'Unknown graph key')
    return result if key is None else result[key]


def metadata():
    return dict(schema=SCHEMA, success=SUCCESS, graph_keys=list(KEYS), parent_pins=PARENT_PINS,
        prior_static_pin=PRIOR_STATIC_PIN, capture_pin=CAPTURE_PIN, input_name=INPUT,
        input_shape=INPUT_SHAPE, outputs=output_contract(), expected_counts=COUNTS,
        PROOF_TRUE=list(PROOF_TRUE), default_reads_artifacts_or_imports_frameworks=False,
        static_graph_construction_only=True, candidate_accepted=False)


def transpose_plane_words(data, rows, channels, inverse=False):
    """Exact 32-bit word permutation, without floating arithmetic or casts."""
    require(len(data) == rows * channels * 4, 'Fixed RoPE table word span differs')
    out = bytearray(len(data))
    for token in range(rows):
        for channel in range(channels):
            old, new = (token * channels + channel) * 4, (channel * rows + token) * 4
            source, target = (new, old) if inverse else (old, new)
            out[target:target + 4] = data[source:source + 4]
    return bytes(out)


def generated_checks():
    """Tiny synthetic indices and exact rational RoPE; no captured values."""
    shape_checks = 0
    for rows in (6, 9):
        for head in range(H):
            for token in range(rows):
                for channel in range(D):
                    old = ((head // 8) * rows + token) * 512 + channel * 8 + head % 8
                    direct = (((head // 8) * rows + token) * D + channel) * 8 + head % 8
                    require(old == direct, 'Direct4/control/consumer offset mismatch')
                    # BDHS reshape[D,2,8,T], then transpose[1,3,0,2].
                    bdhs_source = (channel * H + head) * rows + token
                    reconstructed_source = ((channel * 2 + head // 8) * 8 + head % 8) * rows + token
                    require(bdhs_source == reconstructed_source, 'CF packing source coordinate mismatch')
                    shape_checks += 1
    rows, channels = 6, 8
    words = b''.join(struct.pack('<I', i * 17 + 3) for i in range(rows * channels))
    moved = transpose_plane_words(words, rows, channels)
    require(transpose_plane_words(moved, rows, channels, True) == words, 'RoPE word permutation inverse differs')
    for channel in range(channels):
        for token in range(rows):
            require(moved[(channel * rows + token)*4:(channel * rows + token + 1)*4] ==
                    words[(token * channels + channel)*4:(token * channels + channel + 1)*4], 'RoPE word movement differs')
    permutation = list(range(channels // 2, channels)) + list(range(channels // 2))
    signs = [-1] * (channels // 2) + [1] * (channels // 2)
    count = 0
    for role in range(2):
        x = [[[Fraction(role * 7 + head * 3 + token - channel, 17) for channel in range(channels)]
              for token in range(rows)] for head in range(H)]
        bdhs = [[[x[head][token][channel] for token in range(rows)] for head in range(H)]
                for channel in range(channels)]
        for head in range(H):
            for token in range(rows):
                for channel in range(channels):
                    cosine, sine = Fraction(token + channel + 2, 19), Fraction(token - channel - 1, 23)
                    rotated = x[head][token][permutation[channel]] * signs[channel]
                    original = x[head][token][channel] * cosine + rotated * sine
                    rotated_cf = bdhs[permutation[channel]][head][token] * signs[channel]
                    channel_first = bdhs[channel][head][token] * cosine + rotated_cf * sine
                    require(original == channel_first, 'CF RoPE arithmetic or token position differs')
                    count += 1
    require([i for lo, hi in RANGES for i in range(lo, hi)] == list(range(N)), 'Original query coverage differs')
    return dict(synthetic_integer_layout_components=shape_checks, exact_Fraction_RoPE_components=count,
        fixed_word_permutation_roundtrip=True, full2082_query_coverage=True,
        no_original_QKV_weights_positions_or_capture_payload_read=True)


def build(canonical_parent, native_half_parent):
    # ONNX only: no inference engine, tracing, tensor framework or SDK.
    import onnx
    from onnx import TensorProto as T, helper as h
    parents = dict(canonical=canonical_parent, native_half=native_half_parent)
    parent_bytes = {key: model.SerializeToString() for key, model in parents.items()}
    for key, data in parent_bytes.items():
        require((len(data), digest(data)) == PARENT_PINS[key], 'Exact parent serialization differs: ' + key)
    canonical, native = canonical_parent, native_half_parent
    require((len(canonical.graph.node), len(canonical.graph.initializer)) == (20, 16) and
        (len(native.graph.node), len(native.graph.initializer)) == (43, 28), 'Pinned parent graph counts differ')
    require(not canonical.graph.value_info and not native.graph.value_info, 'Unexpected parent intermediate annotations')
    original_nodes = list(canonical.graph.node)
    original_initializers = list(canonical.graph.initializer)
    node_by_name = {node.name: node for node in original_nodes}
    init_by_name = {tensor.name: tensor for tensor in original_initializers}
    require(len(node_by_name) == 20 and len(init_by_name) == 16 and set(ROPE_NAMES) <= set(node_by_name) and
        set(LEARNED) <= set(init_by_name), 'Original prefix identity differs')
    require([n.SerializeToString() for n in native.graph.node[:20]] == [n.SerializeToString() for n in original_nodes],
        'Pinned parents do not share the same original20 nodes')
    require(all(next(t for t in native.graph.initializer if t.name == name).SerializeToString() == tensor.SerializeToString()
        for name, tensor in init_by_name.items()), 'Pinned parents do not share original16 initializers')
    require(not any(CF in x for node in native.graph.node for x in [node.name, *node.input, *node.output]),
        'New CF namespace collision')

    def attrs(node, name):
        found = [a for a in node.attribute if a.name == name]
        require(len(found) == 1, 'Expected exact named attribute: ' + node.name + '/' + name)
        return found[0]

    def set_perm(node, expected, replacement):
        require(node.op_type == 'Transpose' and list(attrs(node, 'perm').ints) == expected, 'Transpose permutation differs')
        attrs(node, 'perm').ints[:] = replacement

    def i64(tensor):
        require(tensor.data_type == T.INT64 and len(tensor.dims) == 1 and not tensor.external_data,
            'Only inline INT64 shape tensors permitted')
        if tensor.raw_data:
            require(not tensor.int64_data and len(tensor.raw_data) == tensor.dims[0] * 8, 'INT64 raw span differs')
            return list(struct.unpack('<' + 'q' * tensor.dims[0], tensor.raw_data))
        require(len(tensor.int64_data) == tensor.dims[0], 'INT64 shape data differs')
        return list(tensor.int64_data)

    def with_i64(tensor, values):
        result = copy.deepcopy(tensor)
        original_raw = bool(result.raw_data)
        del result.dims[:]; result.dims.extend([len(values)])
        if original_raw:
            result.raw_data = struct.pack('<' + 'q' * len(values), *values)
        else:
            del result.int64_data[:]; result.int64_data.extend(values)
        return result

    def rope_tensor(tensor, inverse=False):
        result = copy.deepcopy(tensor)
        require(tensor.name in ROPE_CONSTANTS and tensor.data_type == T.FLOAT and tensor.raw_data and
            not tensor.external_data and not tensor.float_data, 'Exact inline F32 RoPE storage required')
        if tensor.name.endswith('rotation_sign'):
            require(list(tensor.dims) == ([1,D,1,1] if inverse else [D]) and len(tensor.raw_data) == D*4,
                'Original sign initializer is rank1[64], transformed broadcast is rank4')
            dims = [D] if inverse else [1,D,1,1]
        else:
            require(list(tensor.dims) == ([1,D,1,N] if inverse else [1,1,N,D]), 'Full-position RoPE shape differs')
            result.raw_data = transpose_plane_words(tensor.raw_data, N, D, inverse)
            dims = [1,1,N,D] if inverse else [1,D,1,N]
        del result.dims[:]; result.dims.extend(dims)
        return result

    def vi(name, dtype, shape):
        return h.make_tensor_value_info(name, dtype, shape)

    def model(base, name, nodes, inits, outputs):
        result = copy.deepcopy(base)
        result.graph.name = name
        del result.graph.node[:]; result.graph.node.extend(nodes)
        del result.graph.initializer[:]; result.graph.initializer.extend(inits)
        del result.graph.output[:]; result.graph.output.extend(outputs)
        return result

    def public_info(key):
        return [vi(row['name'], T.FLOAT if row['dtype'] == 'float32' else T.FLOAT16, row['shape'])
                for row in output_contract(key)]

    # Remove only the old added flatten views. Their deleted protos are retained
    # below for an exact inverse; no original arithmetic node is removed.
    flat_names = {NS + role + '/flat_view' for role, _ in ROLES}
    old_added = {node.name: node for node in native.graph.node[20:]}
    old_inits = {tensor.name: tensor for tensor in native.graph.initializer}
    shape_names = {NS + 'split_heads_' + str(rows) for rows in (ROWS, N)}
    flat_init_names = {NS + 'flat_native_' + str(rows) for rows in (ROWS, N)}
    require(flat_names <= set(old_added) and shape_names | flat_init_names <= set(old_inits), 'Old layout additions differ')
    direct_nodes = []
    for old in native.graph.node:
        if old.name in flat_names:
            require(old.op_type == 'Reshape', 'Only old final flatten Reshapes can be removed')
            continue
        node = copy.deepcopy(old)
        for role, _ in ROLES:
            if node.name == NS + role + '/native_order':
                set_perm(node, [0,1,3,4,2], [0,2,3,1])
            if node.name == NS + role + '/half':
                require(node.op_type == 'Cast' and list(node.input) == [NS + role + '/flat'] and
                    attrs(node, 'to').i == T.FLOAT16, 'Old output HALF cast differs')
                node.input[0] = NS + role + '/native'
        direct_nodes.append(node)
    direct_inits = []
    for tensor in native.graph.initializer:
        if tensor.name in flat_init_names:
            continue
        if tensor.name in shape_names:
            rows = int(tensor.name.rsplit('_',1)[1])
            require(i64(tensor) == [1,2,8,rows,D], 'Old split-head shape differs')
            tensor = with_i64(tensor, [2,8,rows,D])
        direct_inits.append(copy.deepcopy(tensor))
    direct = model(native, 'original_global0_direct4_control', direct_nodes, direct_inits, public_info('direct4_control'))

    def cf_name(name):
        return CF + 'layout/' + name

    norm_names = ('global0/attn.q_norm', 'global0/attn.k_norm')
    norm_outputs = {node_by_name[name].output[0] for name in norm_names}
    renamed = {name: cf_name(name) for node in original_nodes if node.name in ROPE_NAMES for name in node.output}
    renamed.update({name: cf_name(name) for name in norm_outputs})
    require(len(renamed) == 12, 'Exact ten RoPE intermediates and two norm outputs required')
    cf_nodes = []
    for old in original_nodes:
        node = copy.deepcopy(old)
        if old.name in ROPE_NAMES:
            for i, name in enumerate(node.input): node.input[i] = renamed.get(name, name)
            for i, name in enumerate(node.output): node.output[i] = renamed[name]
            if node.op_type == 'Gather':
                require(attrs(node, 'axis').i == 3 and node.input[1] == 'global0/rope_permutation', 'Original RoPE Gather differs')
                attrs(node, 'axis').i = 1
            else:
                require(node.op_type in ('Mul', 'Add'), 'Unexpected rewritten RoPE operation')
        cf_nodes.append(node)
        if old.name in norm_names:
            require(old.op_type == 'LayerNormalization' and attrs(old, 'axis').i == -1,
                'Original BHSD per-head LayerNormalization must be retained')
            cf_nodes.append(h.make_node('Transpose', [old.output[0]], [renamed[old.output[0]]],
                name=CF + old.name + '/norm_to_BDHS', perm=[0,3,1,2]))
    cf_inits = [rope_tensor(tensor) if tensor.name in ROPE_CONSTANTS else copy.deepcopy(tensor)
                for tensor in original_initializers]
    canonical_nodes = copy.deepcopy(cf_nodes)
    for name in ORIGINAL_OUTPUTS[:2]:
        canonical_nodes.append(h.make_node('Transpose', [renamed[name]], [name],
            name=CF + name + '/restore_BHSD', perm=[0,2,3,1]))
    cf_canonical = model(canonical, 'original_global0_CF_RoPE_canonical', canonical_nodes, cf_inits, public_info('cf_canonical'))

    extra, constants = [], []
    def const(name, values):
        name = CF + name
        constants.append(h.make_tensor(name, T.INT64, [len(values)], values))
        return name
    def add(op, inputs, outputs, name, **attributes):
        extra.append(h.make_node(op, inputs, outputs, name=CF + name, **attributes))
    axis, step = const('query_slice_axis', [3]), const('query_slice_step', [1])
    for index, (lo, hi) in enumerate(RANGES):
        add('Slice', [renamed[ORIGINAL_OUTPUTS[0]], const('query_start_'+str(index), [lo]),
            const('query_end_'+str(index), [hi]), axis, step], [CF+'Q'+str(index)+'/BDHS'], 'Q'+str(index)+'/slice_after_full_RoPE')
    cf_shapes = {rows: const('CF_pack_shape_'+str(rows), [D,2,8,rows]) for rows in (ROWS,N)}
    value_shape = const('V_pack_shape', [2,8,N,D])
    for role, rows in ROLES:
        if role.startswith('Q'):
            source, shape, perm = CF + role + '/BDHS', cf_shapes[rows], [1,3,0,2]
        elif role == 'K':
            source, shape, perm = renamed[ORIGINAL_OUTPUTS[1]], cf_shapes[rows], [1,3,0,2]
        else:
            source, shape, perm = ORIGINAL_OUTPUTS[2], value_shape, [0,2,3,1]
        add('Reshape', [source, shape], [CF+role+'/split'], role+'/split_heads')
        add('Transpose', [CF+role+'/split'], [CF+role+'/direct4'], role+'/direct4_order', perm=perm)
        add('Cast', [CF+role+'/direct4'], [NS+role], role+'/half', to=T.FLOAT16)
    cf_native = model(canonical, 'original_global0_CF_RoPE_direct4_half', cf_nodes + extra,
                      cf_inits + constants, public_info('cf_native'))
    graphs = dict(direct4_control=direct, cf_canonical=cf_canonical, cf_native=cf_native)

    # Inverses use the actual modified nodes/tensors. Deleted metadata-only
    # flat-view protos and public ValueInfo are restored from the pinned parent.
    restored_nodes = []
    for old in direct.graph.node:
        node = copy.deepcopy(old)
        for role, _ in ROLES:
            if node.name == NS+role+'/native_order': set_perm(node, [0,2,3,1], [0,1,3,4,2])
            if node.name == NS+role+'/half':
                require(node.input[0] == NS+role+'/native', 'Direct4 Cast input differs')
                restored_nodes.append(copy.deepcopy(old_added[NS+role+'/flat_view']))
                node.input[0] = NS+role+'/flat'
        restored_nodes.append(node)
    dmap = {tensor.name:tensor for tensor in direct.graph.initializer}
    restored_inits = []
    for original in native.graph.initializer:
        tensor = copy.deepcopy(original if original.name in flat_init_names else dmap[original.name])
        if original.name in shape_names:
            vals = i64(tensor); require(vals[:2] == [2,8], 'Direct4 split layout differs')
            tensor = with_i64(tensor, [1] + vals)
        restored_inits.append(tensor)
    inverse = model(direct, native.graph.name, restored_nodes, restored_inits, list(native.graph.output))
    require(inverse.SerializeToString() == parent_bytes['native_half'], 'Actual direct4 inverse to native_half parent failed')
    inverse_results = {'direct4_control':digest(inverse.SerializeToString())}
    reverse_names = {new:old for old,new in renamed.items()}
    for key in ('cf_canonical','cf_native'):
        actual = graphs[key]
        restored_nodes = []
        for edited in actual.graph.node:
            if edited.name not in node_by_name:
                require(edited.name.startswith(CF), 'Unexpected new CF node namespace')
                continue
            node = copy.deepcopy(edited)
            if node.name in ROPE_NAMES:
                for i,name in enumerate(node.input): node.input[i] = reverse_names.get(name,name)
                for i,name in enumerate(node.output): node.output[i] = reverse_names[name]
                if node.op_type == 'Gather':
                    require(attrs(node,'axis').i == 1, 'CF Gather axis differs');attrs(node,'axis').i = 3
            restored_nodes.append(node)
        imap = {tensor.name:tensor for tensor in actual.graph.initializer}
        restored_inits = [rope_tensor(imap[t.name], True) if t.name in ROPE_CONSTANTS else copy.deepcopy(imap[t.name])
                          for t in original_initializers]
        inverse = model(actual, canonical.graph.name, restored_nodes, restored_inits, list(canonical.graph.output))
        require(inverse.SerializeToString() == parent_bytes['canonical'], 'Actual CF inverse to canonical parent failed: '+key)
        inverse_results[key] = digest(inverse.SerializeToString())

    immutable_names = [node.name for node in original_nodes if node.name not in ROPE_NAMES]
    require(len(immutable_names) == len(ROPE_NAMES) == 10, 'Original non-RoPE/RoPE split differs')
    preservation = {}
    for key, graph in graphs.items():
        require((len(graph.graph.node),len(graph.graph.initializer)) == COUNTS[key], 'Derived graph counts differ: '+key)
        nmap = {node.name:node for node in graph.graph.node}
        imap = {tensor.name:tensor for tensor in graph.graph.initializer}
        require(len(nmap) == len(graph.graph.node) and len(imap) == len(graph.graph.initializer), 'Duplicate graph symbol names')
        require(all(nmap[name].SerializeToString() == node_by_name[name].SerializeToString() for name in immutable_names),
            'An original non-RoPE node changed')
        require(all(imap[name].SerializeToString() == init_by_name[name].SerializeToString() for name in LEARNED),
            'An original learned tensor changed')
        if key == 'direct4_control':
            require([node.SerializeToString() for node in graph.graph.node[:20]] ==
                    [node.SerializeToString() for node in original_nodes], 'Direct4 original20 nodes changed')
        else:
            require([node.name for node in graph.graph.node if node.name in ROPE_NAMES] ==
                    [node.name for node in original_nodes if node.name in ROPE_NAMES], 'Original RoPE arithmetic order changed')
        produced = set()
        available = set(imap) | {INPUT}
        for node in graph.graph.node:
            require(all(not name or name in available for name in node.input), 'Graph not topologically closed at '+node.name)
            require(not set(node.output) & (produced | set(imap) | {INPUT}), 'Graph value name reused')
            produced.update(node.output);available.update(node.output)
        require(set(v.name for v in graph.graph.output) <= available and len(graph.graph.input) == 1 and
                graph.graph.input[0].name == INPUT, 'Only the original single input is permitted')
        require(all(t.data_type == T.INT64 for name,t in imap.items() if name not in init_by_name),
            'Only INT64 layout/slice constants can be added')
        onnx.checker.check_model(graph, full_check=True)
        inferred = onnx.shape_inference.infer_shapes(graph, check_type=True, strict_mode=True, data_prop=True)
        onnx.checker.check_model(inferred, full_check=True)
        require([v.SerializeToString() for v in inferred.graph.output] == [v.SerializeToString() for v in graph.graph.output],
            'Inferred public output type or shape differs')
        preservation[key] = dict(nodes=len(graph.graph.node),initializers=len(graph.graph.initializer),
            original10_non_RoPE_node_proto_sha256s={name:digest(nmap[name].SerializeToString()) for name in immutable_names},
            original8_learned_tensor_proto_sha256s={name:digest(imap[name].SerializeToString()) for name in LEARNED})
    cf_map = {tensor.name:tensor for tensor in cf_inits}
    for name in ROPE_CONSTANTS:
        require(rope_tensor(cf_map[name], True).SerializeToString() == init_by_name[name].SerializeToString(),
                'Actual fixed RoPE tensor inverse differs')
    proof = {name:True for name in PROOF_TRUE}
    proof.update(parent_inverse_sha256s=inverse_results, preservation=preservation,
        rewritten_RoPE_node_names=list(ROPE_NAMES), rewritten_RoPE_constant_metadata={name:dict(
            original_shape=list(init_by_name[name].dims),CF_shape=list(cf_map[name].dims),
            original_proto_sha256=digest(init_by_name[name].SerializeToString()),CF_proto_sha256=digest(cf_map[name].SerializeToString()))
            for name in ROPE_CONSTANTS},
        original20_NodeProtos_all_byte_unchanged_for_CF_graphs=False,
        deleted_direct4_flat_view_protos_base64={name:base64.b64encode(old_added[name].SerializeToString()).decode() for name in sorted(flat_names)},
        deleted_direct4_flat_shape_protos_base64={name:base64.b64encode(old_inits[name].SerializeToString()).decode() for name in sorted(flat_init_names)},
        native_identity='direct[head//8,token,channel,head%8]=HALF(original_BHSD[0,head,token,channel])',
        CF_identity='CF[0,channel,head,token]=original_BHSD[0,head,token,channel]',
        arithmetic_order='Gather(permutation), Mul(rotation_sign), Mul(cosine) and Mul(rotated,sine), Add(cosine,sine); unchanged per-element order',
        query_ranges=[list(r) for r in RANGES],query_slice_axis_for_CF=3,
        original_RoPE_full_position_count=N,direct4_HALF_payload_bytes=sum(row['bytes'] for row in output_contract('cf_native')),
        generated_checks=generated_checks(),
        compiler_CPU_transpose_cancellation_or_output_ABI_or_precision_or_latency_established=False)
    return graphs, proof

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--canonical-parent', type=Path)
    p.add_argument('--native-half-parent', type=Path)
    a = p.parse_args()
    if not a.execute:
        print(json.dumps(dict(metadata(), historical_source=HISTORICAL_SOURCE,
            published_entrypoint='Portable parent paths; new receipts, not a historical device execution.'), indent=2))
        return 0
    require(a.output_dir is not None and not a.output_dir.exists(), 'A fresh output directory is required')
    require(a.canonical_parent and a.native_half_parent, 'Both exact parent paths are required')
    paths = dict(canonical=a.canonical_parent, native_half=a.native_half_parent)
    sources = {key: descriptor(path, PARENT_PINS[key]) for key, path in paths.items()}
    import onnx
    parents = {key: onnx.load(path, load_external_data=False) for key, path in paths.items()}
    graphs, proofs = build(parents['canonical'], parents['native_half'])
    a.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for key, graph in graphs.items():
        path = a.output_dir / (key + '.onnx')
        with path.open('xb') as stream:
            stream.write(graph.SerializeToString())
        restored = onnx.load(path, load_external_data=False)
        onnx.checker.check_model(restored, full_check=True)
        require(restored.SerializeToString() == graph.SerializeToString(), 'Serialized graph roundtrip differs')
        outputs[key] = dict(descriptor(path), nodes=len(graph.graph.node), initializers=len(graph.graph.initializer))
    require(all(descriptor(row['file']) == row for row in sources.values()), 'Parent changed during export')
    report = dict(schema=SCHEMA, status=SUCCESS, graphs=outputs, proofs=proofs,
        exact_parent_artifacts=sources, exporter=descriptor(__file__), historical_source=HISTORICAL_SOURCE,
        public_source_adaptation=True, static_only=True, inference_SDK_calibration_or_GPU_executed=False,
        CPU_NPU_precision_speed_or_full_model_acceptance=False, candidate_accepted=False)
    with (a.output_dir / 'result.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=SUCCESS, graphs=outputs), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
