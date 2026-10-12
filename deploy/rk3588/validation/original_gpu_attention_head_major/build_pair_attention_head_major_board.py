#!/usr/bin/env python3
"""Export the complete original pair with two RKNN head-major custom attention tails.

Only the RK3588 root operator runs this file.  Export does not execute a model,
ORT, SDK or OpenCL.  make_toolkit_custom_op() supplies a conversion-time class
which executes the exact saved original tail ONNX on the board CPU, not a new
attention implementation.  Device callbacks must be registered independently.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import sys

GRAPH_SHA = 'efbbaee8f3d6326565930e5a045eac0d500fdb35453fdf85ddac7481ec2fac65'
PARENT_SHA = '1872f2ed730f4b6d139562d831661126a088b1ac758249d6a1688cd7e9bfae42'
OP_TYPES = {'frame': 'cstOriginalFrameHeadMajorAttentionTail', 'global': 'cstOriginalGlobalHeadMajorAttentionTail'}
CUSTOM_DOMAIN = 'vggt.original'
FLOAT_INPUT_SUFFIXES = ('query', 'key', 'value', 'attn.q_norm.weight',
    'attn.q_norm.bias', 'attn.k_norm.weight', 'attn.k_norm.bias',
    'rope_rotation_sign', 'rope_full_cos', 'rope_full_sin', 'attention_scale')
TAIL_SUFFIXES = ('attn.q_norm', 'query_rope_permuted', 'query_rope_rotated',
    'query_rope_cosine', 'query_rope_sine', 'query_rope', 'attn.k_norm',
    'key_rope_permuted', 'key_rope_rotated', 'key_rope_cosine',
    'key_rope_sine', 'key_rope', 'key_transpose', 'attention_scores',
    'attention_scaled_scores', 'attention_probabilities', 'attention_values',
)
TAIL_OPS = ('LayerNormalization', 'Gather', 'Mul', 'Mul', 'Mul', 'Add',
    'LayerNormalization', 'Gather', 'Mul', 'Mul', 'Mul', 'Add', 'Transpose',
    'MatMul', 'Mul', 'Softmax', 'MatMul')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def artifact(path):
    path = Path(path)
    return dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path))


def raw(np, value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def normalize_conversion_inputs(np, inputs, expected_names, expected_shapes):
    """Only widen IEEE half and normalize singleton dimensions; never permute."""
    observations, normalized = [], []
    require(len(inputs) == len(expected_names) == len(expected_shapes) == 11,
            'Eleven original custom inputs required')
    for value, name, expected in zip(inputs, expected_names, expected_shapes):
        value = np.asarray(value)
        finite = bool(np.isfinite(value).all()) if value.dtype.kind == 'f' else False
        allowed_shape = ([n for n in value.shape if n != 1] == [n for n in expected if n != 1]
                         and value.size == int(np.prod(expected, dtype=np.int64)))
        observation = dict(name=name, actual_dtype=str(value.dtype), actual_shape=list(value.shape),
            expected_shape=expected, finite=finite, equivalent_singleton_shape=allowed_shape,
            widened_half_to_float=value.dtype == np.float16,
            singleton_shape_normalized=list(value.shape) != expected)
        observations.append(observation)
        if value.dtype not in (np.dtype('float32'), np.dtype('float16')) or not finite or not allowed_shape:
            raise ValueError('Original custom conversion input mismatch: ' + json.dumps(observations, allow_nan=False))
        normalized.append(value.astype(np.float32, copy=False).reshape(expected).copy(order='C'))
    return normalized, observations


def rewrite(onnx, np, model, parent):
    """Return candidate, exact original tail fragments, and static lineage proof."""
    h, nh = onnx.helper, onnx.numpy_helper
    nodes = {n.name: n for n in model.graph.node}
    inits = {t.name: t for t in model.graph.initializer}
    infos = {v.name: v for v in model.graph.value_info}
    require(len(nodes) == len(model.graph.node) == 87 and len(inits) == 63,
            'Pinned original pair node/initializer catalog differs')
    parameter_count = 0
    for role in ('frame', 'global'):
        parameters = parent['parameter_raw_sha256'][role]
        require(len(parameters) == 18, 'All original eighteen block parameters required')
        for name, digest in parameters.items():
            transposed = name in ('attn.qkv.weight', 'attn.proj.weight',
                                 'mlp.fc1.weight', 'mlp.fc2.weight')
            key = role + '0/' + name + ('_transposed' if transposed else '')
            value = nh.to_array(inits[key])
            require(value.dtype == np.float32 and np.isfinite(value).all()
                    and raw(np, value.T if transposed else value) == digest,
                    'Original learned parameter changed: ' + key)
            parameter_count += 1
    candidate = copy.deepcopy(model)
    replacements, fragments, removed = {}, {}, set()
    proof = dict(original_parameter_tensors_verified=parameter_count, tails=[])
    for role, batch, tokens in (('frame', 2, 53), ('global', 1, 106)):
        prefix = role + '0/'
        names = [prefix + suffix for suffix in TAIL_SUFFIXES]
        chosen = [nodes[name] for name in names]
        require([n.op_type for n in chosen] == list(TAIL_OPS), 'Original tail operations differ')
        require([n.name for n in model.graph.node if n.name in names] == names,
                'Original tail node order differs')
        layout = [nodes[prefix + suffix] for suffix in ('attention_heads_transpose', 'attention_heads_flatten')]
        require([n.op_type for n in layout] == ['Transpose', 'Reshape']
                and layout[0].input[0] == prefix + 'attention_values'
                and layout[1].input[0] == prefix + 'attention_heads_transpose',
                'Original consumer layout nodes must remain connected')
        produced = {out for n in chosen for out in n.output}
        result = prefix + 'attention_values'
        used = {name for n in chosen for name in n.input if name not in produced}
        inputs = [prefix + suffix for suffix in FLOAT_INPUT_SUFFIXES]
        require(used == set(inputs) | {prefix + 'rope_permutation'},
                'Original tail boundary dependencies differ')
        for n in model.graph.node:
            if n.name not in names:
                require(not (set(n.input) & (produced - {result})),
                        'Original tail intermediate has an external consumer')
        require(not ({v.name for v in model.graph.output} & produced),
                'Original graph output cannot be removed')
        for norm in (chosen[0], chosen[6]):
            attrs = {a.name: h.get_attribute_value(a) for a in norm.attribute}
            require(norm.domain == '' and attrs.keys() == {'axis', 'epsilon', 'stash_type'}
                    and attrs['axis'] == -1 and attrs['stash_type'] == 1,
                    'Original FP32 last-axis LayerNorm contract differs')
        qattrs = {a.name: h.get_attribute_value(a) for a in chosen[0].attribute}
        kattrs = {a.name: h.get_attribute_value(a) for a in chosen[6].attribute}
        require(qattrs == kattrs, 'Q/K normalization attributes differ')
        permutation = nh.to_array(inits[prefix + 'rope_permutation'])
        require(permutation.dtype == np.int64 and permutation.shape == (64,)
                and sorted(permutation.tolist()) == list(range(64)), 'Original RoPE permutation differs')
        op_type = OP_TYPES[role]
        custom = h.make_node(op_type, inputs, [result], name=prefix + op_type,
            domain=CUSTOM_DOMAIN, batch=batch, tokens=tokens, heads=16, head_dim=64,
            epsilon=float(qattrs['epsilon']), rope_permutation=permutation.tolist())
        replacements[names[0]] = custom
        removed.update(names)
        # Constants still travel through the custom node as original FP32 values.
        # Only the original int64 permutation remains an initializer in
        # the exact reference fragment; the other constants are explicit inputs.
        fragment = copy.deepcopy(model)
        input_infos = []
        for name in inputs:
            shape = ([batch, 16, tokens, 64] if name not in inits else list(inits[name].dims))
            if name in inits:
                require(inits[name].data_type == onnx.TensorProto.FLOAT,
                        'Custom compute supports only original FP32 inputs')
            input_infos.append(h.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape))
        fragment.graph.CopyFrom(h.make_graph(chosen, role + '_original_attention_tail',
            input_infos, [copy.deepcopy(infos[result])],
            initializer=[copy.deepcopy(inits[prefix + 'rope_permutation'])],
            value_info=[copy.deepcopy(v) for v in model.graph.value_info
                        if v.name in produced - {result}]))
        onnx.checker.check_model(fragment)
        fragments[role] = fragment
        proof['tails'].append(dict(role=role, original_nodes=names, original_node_count=17,
            custom_op_type=op_type, domain=CUSTOM_DOMAIN, custom_name=custom.name,
            input_names=inputs, input_shapes=[[d.dim_value for d in v.type.tensor_type.shape.dim]
                                            for v in input_infos],
            output_name=result, output_shape=[batch, 16, tokens, 64],
            epsilon=float(qattrs['epsilon']), rope_permutation_raw_sha256=raw(np, permutation),
            original_tail_node_protobuf_sha256=[hashlib.sha256(n.SerializeToString()).hexdigest() for n in chosen],
            reference_tail_original_nodes_bytewise_retained=True))
    new_nodes = []
    for node in model.graph.node:
        if node.name in replacements:
            new_nodes.append(replacements[node.name])
        elif node.name not in removed:
            new_nodes.append(copy.deepcopy(node))
    del candidate.graph.node[:]
    candidate.graph.node.extend(new_nodes)
    obsolete = {name for role in proof['tails'] for name in
                (out for n in fragments[role['role']].graph.node for out in n.output)}
    ends = {role['output_name'] for role in proof['tails']}
    kept_infos = [copy.deepcopy(v) for v in model.graph.value_info if v.name not in obsolete - ends]
    del candidate.graph.value_info[:]
    candidate.graph.value_info.extend(kept_infos)
    require(all(o.domain != CUSTOM_DOMAIN for o in candidate.opset_import), 'Custom domain already imported')
    candidate.opset_import.append(h.make_opsetid(CUSTOM_DOMAIN, 1))
    untouched = {n.name: n for n in candidate.graph.node if n.name not in {x.name for x in replacements.values()}}
    require(all(untouched[n.name].SerializeToString() == n.SerializeToString()
                for n in model.graph.node if n.name not in removed), 'Unrelated original node changed')
    require([x.SerializeToString() for x in candidate.graph.initializer] ==
            [x.SerializeToString() for x in model.graph.initializer], 'Original initializer changed')
    require([x.SerializeToString() for x in (*candidate.graph.input, *candidate.graph.output)] ==
            [x.SerializeToString() for x in (*model.graph.input, *model.graph.output)], 'Original IO changed')
    # This inverse is a static proof of the edit scope, not a numerical execution.
    inverse = copy.deepcopy(candidate)
    del inverse.graph.node[:]
    inverse.graph.node.extend(model.graph.node)
    del inverse.graph.value_info[:]
    inverse.graph.value_info.extend(model.graph.value_info)
    del inverse.opset_import[-1]
    require(inverse.SerializeToString() == model.SerializeToString(), 'Undoing only the two cuts did not recover original graph')
    onnx.checker.check_model(candidate)
    proof.update(original_node_count=87, candidate_node_count=len(new_nodes),
        original_initializer_count=63, all_original_initializers_protobuf_unchanged=True,
        all_unrelated_nodes_protobuf_unchanged=True, original_IO_protobuf_unchanged=True,
        original_pair_recovered_bytewise_by_restoring_only_two_tail_cuts=True,
        all_original_attention_math_delegated_not_omitted=True,
        original_two_consumer_layout_nodes_per_role_retained=True,
        custom_output_is_original_head_major_attention_values=True,
        executed_FP32_equivalence_or_GPU_quality_gate=False)
    return candidate, fragments, proof


def make_toolkit_custom_op(export_directory, role, *, session_factory=None):
    """RKNN.reg_custom_op class; exact original board-ORT reference per shape.

    Toolkit conversion reference only.  This is not the board runtime callback.
    Use only on the RK3588 board and bind both original-tail artifact digests.
    """
    directory = Path(export_directory).resolve(strict=True)
    report = json.loads((directory / 'export.json').read_text())
    require(report['schema'] == 'v84_original_pair_attention_head_major_custom_ONNX_export', 'Custom export schema differs')
    require(role in OP_TYPES, 'Original frame/global custom operator required')
    selected_role = role
    sessions = {}

    class cstOriginalAttentionTail:
        op_type = OP_TYPES[selected_role]
        conversion_compute_attempted = 0
        conversion_compute_completed = 0
        conversion_input_observations = []

        def shape_infer(self, node, in_shapes, in_dtypes):
            require(len(in_shapes) == 11 and len(in_dtypes) == 11, 'Eleven original FP32 custom inputs required')
            tail = next(t for t in report['proof']['tails'] if t['role'] == selected_role)
            require(all([d for d in shape if d != 1] == [d for d in expected if d != 1]
                        for shape, expected in zip(in_shapes, tail['input_shapes'])),
                    'Original custom shape inference cannot permute dimensions')
            b, heads, n, width = tail['input_shapes'][0]
            return [[b, heads, n, width]], [in_dtypes[0]]

        def compute(self, node, inputs):
            import numpy as np
            role = selected_role
            tail = next(t for t in report['proof']['tails'] if t['role'] == role)
            inputs, observations = normalize_conversion_inputs(np, inputs, tail['input_names'], tail['input_shapes'])
            if observations not in cstOriginalAttentionTail.conversion_input_observations:
                cstOriginalAttentionTail.conversion_input_observations.append(observations)
                print('Original custom conversion input normalization: ' + json.dumps(observations), flush=True)
            if role not in sessions:
                row = report['reference_tails'][role]
                path = directory / row['file']
                require(path.parent == directory and path.stat().st_size == row['bytes']
                        and sha(path) == row['sha256'], 'Exact original tail graph changed')
                if session_factory is None:
                    compatible = next((p.read_bytes() for p in (Path('/proc/device-tree/compatible'),
                        Path('/sys/firmware/devicetree/base/compatible')) if p.is_file()), b'')
                    require(sys.platform == 'linux' and platform.machine().lower() in ('aarch64', 'arm64')
                            and b'rk3588' in compatible, 'Actual original-tail ORT is RK3588/S board-only')
                    import onnxruntime as ort
                    options = ort.SessionOptions()
                    options.intra_op_num_threads = options.inter_op_num_threads = 1
                    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                    sessions[role] = ort.InferenceSession(str(path), sess_options=options,
                                                         providers=['CPUExecutionProvider'])
                else:
                    sessions[role] = session_factory(path)
            cstOriginalAttentionTail.conversion_compute_attempted += 1
            output = sessions[role].run(None, dict(zip(tail['input_names'], inputs)))[0]
            require(output.dtype == np.float32 and list(output.shape) == tail['output_shape']
                    and np.isfinite(output).all(), 'Original tail reference output differs')
            cstOriginalAttentionTail.conversion_compute_completed += 1
            return [np.ascontiguousarray(output)]

    return cstOriginalAttentionTail


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    require(sha(a.parent_manifest) == PARENT_SHA, 'Original complete pair manifest changed')
    parent = json.loads(a.parent_manifest.read_text())
    require(parent['schema'] == 'v55_original_VGGT_24_frame_global_pair_RKNN_manifest', 'Original parent schema differs')
    row = parent['blocks'][0]
    require(row['role'] == 'pair' and row['index'] == 0, 'Only original pair0 supported')
    graph = (a.parent_manifest.parent / row['file']).resolve(strict=True)
    require(graph.is_relative_to(a.parent_manifest.parent.resolve()) and sha(graph) == GRAPH_SHA
            and row['sha256'] == GRAPH_SHA and graph.stat().st_size == row['bytes'], 'Original pair0 graph differs')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import onnx
    candidate, fragments, proof = rewrite(onnx, np, onnx.load(graph), row)
    path = a.output_dir / 'pair-attention-head-major-custom.onnx'
    onnx.save(candidate, path)
    tails = {}
    for role, fragment in fragments.items():
        target = a.output_dir / (role + '-original-attention-tail-head-major.onnx')
        onnx.save(fragment, target)
        tails[role] = artifact(target)
    report = dict(schema='v84_original_pair_attention_head_major_custom_ONNX_export',
        status='exported_not_compiled_or_runtime_validated', script_sha256=sha(__file__),
        original_parent_manifest_sha256=PARENT_SHA, original_pair_onnx_sha256=GRAPH_SHA,
        original_model_only=True, parent=row, onnx=artifact(path), reference_tails=tails,
        proof=proof, original_checkpoint=parent['checkpoint'], candidate_accepted=False,
        actual_model_OR_Toolkit_OR_OpenCL_executed=False,
        required_runtime_callback_target='GPU', required_runtime_dtype='float32',
        compute_native_supported=False, required_device_registration='rknn_register_custom_ops')
    require(sha(a.parent_manifest) == PARENT_SHA and sha(graph) == GRAPH_SHA, 'Original source changed during export')
    (a.output_dir / 'export.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], onnx=report['onnx'], reference_tails=tails)), flush=True)


if __name__ == '__main__':
    main()
