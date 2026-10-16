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

HISTORICAL_SOURCE = {'bytes': 19137, 'sha256': '941cf799e6fb9caefa3dd17ea7e288a6ef0eeef721764def39b6f3c997478b81'}

PARENT_STATIC_PIN = (19491, '94fa2b3644ca2660028d1120f48d97f3a4b108a0781220e300afa48676fe69b8')


PARENT_PINS = {
    'cf_canonical': (13675800, '927d20943b900d026e795ba361c45ff30173402bd9950d22cedd0da57945ea7f'),
    'cf_native': (13678828, 'decd625e8db757e06fbeb8eb0bdea780c3fb313916429410ff98a205c29f2b3b')}


PARENT_COUNTS = {'cf_canonical': (24, 16), 'cf_native': (40, 27)}


PARENT_EXPORTER_PIN = (29930, '6f0cfa49c639cb61a73000150f9a78cca50332211b616fd890f208776cc89ada')


CAPTURE_SHA = 'e4ffaea2b11df9ac558c119cb4854d490b0030c02b7706eee344b8894cefa686'


SCHEMA = 'original_native518_CF_RoPE_quarter_arithmetic_static'


SUCCESS = 'full64_norm_then_exact_quarter_RoPE_arithmetic_static_verified_CPU_and_compiler_gates_pending'


NS = '__original_CF_quarter_RoPE/'


PERM_QUARTERS = (1, 0, 3, 2)


CONSTANTS = ('global0/rope_rotation_sign', 'global0/rope_full_cos', 'global0/rope_full_sin')


LEARNED = ('global0/norm1.weight', 'global0/norm1.bias', 'global0/attn.qkv.weight_transposed',
    'global0/attn.qkv.bias', 'global0/attn.q_norm.weight', 'global0/attn.q_norm.bias',
    'global0/attn.k_norm.weight', 'global0/attn.k_norm.bias')


PROOF_TRUE = (
    'all_parent_fixed_and_learned_TensorProtos_byte_unchanged',
    'all8_learned_parameters_byte_unchanged',
    'original64_channel_LayerNorms_and_all_non_RoPE_NodeProtos_byte_unchanged',
    'only_ten_original_RoPE_NodeProtos_replaced',
    'actual_original64_permutation_equals_quarter_wiring',
    'per_scalar_three_Mul_and_final_Add_order_inputs_and_attributes_preserved',
    'all_added_constants_are_INT64_slice_indices_no_new_floating_values',
    'all_fixed_RoPE_coefficients_selected_from_original_full_TensorProtos',
    'full2082_positions_16heads_64channels_and_original_query_partitioning_retained',
    'public_IO_and_output_packing_NodeProtos_byte_unchanged',
    'actual_inverse_to_each_parent_serialization_byte_equal',
    'ONNX_checker_closed_graph_and_inferred_IO_checks_passed')


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def descriptor(path, pin=None):
    p = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    row = dict(file=str(p), bytes=p.stat().st_size, sha256=h.hexdigest())
    require(pin is None or (row['bytes'], row['sha256']) == pin, 'Pin differs: ' + str(p))
    return row


def metadata():
    return dict(schema=SCHEMA, success=SUCCESS, parent_pins=PARENT_PINS,
        graph_keys=['cf_quarter_canonical', 'cf_quarter_native'], slice_axis=1, concat_axis=1,
        quarter_width=16, rotated_quarter_order=PERM_QUARTERS,
        original64_channel_normalization_retained_before_splitting=True,
        node_count_delta=44, initializer_count_delta=7,
        expected_counts={'cf_quarter_canonical': [68,23], 'cf_quarter_native': [84,34]},
        proof_fields=PROOF_TRUE, default_reads_artifacts_or_imports_frameworks=False,
        compiler_ConvAdd_avoidance_is_unproved=True, candidate_accepted=False)


def i64(tensor, T):
    require(tensor.data_type == T.INT64 and len(tensor.dims) == 1 and not tensor.external_data,
        'Inline rank1 INT64 indices required')
    n = tensor.dims[0]
    if tensor.raw_data:
        require(len(tensor.raw_data) == 8*n and not tensor.int64_data, 'INT64 raw span differs')
        return list(struct.unpack('<' + 'q'*n, tensor.raw_data))
    require(len(tensor.int64_data) == n, 'INT64 field span differs')
    return list(tensor.int64_data)


def generated_mapping_proof(permutation, signs):
    comparisons = 0
    for head in range(16):
        for token in range(3):
            xs = [Fraction(3*head-2*token+d, 31) for d in range(64)]
            quarters = [xs[j*16:(j+1)*16] for j in range(4)]
            for d in range(64):
                j, local = divmod(d, 16)
                cosine = Fraction(d+token+1, 67)
                sine = Fraction(d-token-3, 71)
                original_rotated = xs[permutation[d]] * int(signs[d])
                original_cosine = xs[d] * cosine
                original_sine = original_rotated * sine
                original = original_cosine + original_sine
                sliced_rotated = quarters[PERM_QUARTERS[j]][local] * int(signs[j*16+local])
                sliced_cosine = quarters[j][local] * cosine
                sliced_sine = sliced_rotated * sine
                sliced = sliced_cosine + sliced_sine
                require((original_rotated,original_cosine,original_sine,original) ==
                    (sliced_rotated,sliced_cosine,sliced_sine,sliced), 'Per-element arithmetic correspondence differs')
                comparisons += 1
    return dict(exact_rational_components=comparisons, comparisons_per_component=4,
        uses_actual_original_permutation_and_sign=True, captured_activation_payload_read=False,
        floating_precision_CPU_or_compiler_equivalence_established=False)


def build(parent, key):
    import onnx
    from onnx import TensorProto as T, helper as h
    original = parent.SerializeToString()
    require((len(original),digest(original)) == PARENT_PINS[key], 'Pinned original CF parent required')
    require((len(parent.graph.node),len(parent.graph.initializer)) == PARENT_COUNTS[key], 'Parent counts differ')
    inits = {t.name:t for t in parent.graph.initializer}
    nodes = {n.name:n for n in parent.graph.node}
    require(len(inits) == len(parent.graph.initializer) and len(nodes) == len(parent.graph.node), 'Duplicate parent symbol')
    require(set(LEARNED) <= set(inits), 'All original learned tensors required')
    permutation = i64(inits['global0/rope_permutation'], T)
    require(permutation == [PERM_QUARTERS[d//16]*16+d%16 for d in range(64)] and
        sorted(permutation) == list(range(64)), 'Actual original channel permutation differs')
    sign_tensor = inits[CONSTANTS[0]]
    require(sign_tensor.data_type == T.FLOAT and list(sign_tensor.dims) == [1,64,1,1] and
        not sign_tensor.external_data and not sign_tensor.float_data and len(sign_tensor.raw_data) == 256,
        'Original CF sign storage differs')
    signs = list(struct.unpack('<64f', sign_tensor.raw_data))
    require(signs == [-1.]*16+[1.]*16+[-1.]*16+[1.]*16, 'Actual original signs differ')
    for name in CONSTANTS[1:]:
        tensor = inits[name]
        require(tensor.data_type == T.FLOAT and list(tensor.dims) == [1,64,1,2082] and
            not tensor.external_data and not tensor.float_data and len(tensor.raw_data) == 64*2082*4,
            'Original full-position fixed coefficients required')
    require(not any(NS in name for node in parent.graph.node for name in [node.name,*node.input,*node.output]) and
        not any(name.startswith(NS) for name in inits), 'New namespace collision')
    original_blocks = {}
    for role in ('query','key'):
        names = ['global0/'+role+'_rope'+suffix for suffix in ('_permuted','_rotated','_cosine','_sine','')]
        block = [nodes[name] for name in names]
        gather,rotated,cosine,sine,add = block
        norm = '__original_CF_RoPE/layout/global0/attn.'+role[0]+'_norm'
        require(gather.op_type == 'Gather' and not gather.domain and len(gather.attribute) == 1 and
            gather.attribute[0].name == 'axis' and gather.attribute[0].i == 1 and
            list(gather.input) == [norm,'global0/rope_permutation'], 'Exact original channel-axis Gather required')
        expected_inputs = ([gather.output[0],CONSTANTS[0]], [norm,CONSTANTS[1]],
            [rotated.output[0],CONSTANTS[2]], [cosine.output[0],sine.output[0]])
        for node,op,inputs in zip(block[1:],('Mul','Mul','Mul','Add'),expected_inputs):
            require(node.op_type == op and not node.domain and list(node.input) == inputs and
                len(node.output) == 1 and not node.attribute, 'Original scalar operation contract differs')
        indices = [list(parent.graph.node).index(n) for n in block]
        require(indices == list(range(indices[0],indices[0]+5)), 'Original five-op RoPE block is not contiguous')
        original_blocks[role] = block
    new_constants = [h.make_tensor(NS+'bound_'+str(i),T.INT64,[1],[i]) for i in (0,16,32,48,64)]
    new_constants += [h.make_tensor(NS+'axis',T.INT64,[1],[1]), h.make_tensor(NS+'step',T.INT64,[1],[1])]

    def channel_slice(source, j, name):
        return h.make_node('Slice',[source,NS+'bound_'+str(16*j),NS+'bound_'+str(16*(j+1)),NS+'axis',NS+'step'],
            [NS+name],name=NS+name+'/slice')

    constant_slices, fixed_ports = [], {}
    for short, original_name in zip(('sign','cos','sin'),CONSTANTS):
        for j in range(4):
            node = channel_slice(original_name,j,'fixed/'+short+'/'+str(j))
            constant_slices.append(node); fixed_ports[(short,j)] = node.output[0]
    replacements = {}
    role_parts = {}
    for role, block in original_blocks.items():
        norm = block[0].input[0]
        xs = [channel_slice(norm,j,role+'/input/'+str(j)) for j in range(4)]
        operations, parts = list(xs), []
        for j in range(4):
            target_names = [NS+role+'/'+str(j)+'/'+name for name in ('rotated','cosine','sine','result')]
            inputs = ([xs[PERM_QUARTERS[j]].output[0],fixed_ports[('sign',j)]],
                [xs[j].output[0],fixed_ports[('cos',j)]],
                [target_names[0],fixed_ports[('sin',j)]], [target_names[1],target_names[2]])
            for old, name, ins in zip(block[1:],target_names,inputs):
                node = copy.deepcopy(old)
                node.name = name
                del node.input[:]; node.input.extend(ins)
                del node.output[:]; node.output.append(name)
                operations.append(node)
            parts.append(target_names[-1])
        operations.append(h.make_node('Concat',parts,list(block[-1].output),name=NS+role+'/concat_completed_RoPE',axis=1))
        replacements[block[0].name] = operations
        role_parts[role] = operations
    removed = {node.name for block in original_blocks.values() for node in block}
    result = copy.deepcopy(parent)
    result.graph.name = 'original_global0_'+key.replace('cf_','cf_quarter_')
    del result.graph.node[:]; result.graph.node.extend(constant_slices)
    for node in parent.graph.node:
        if node.name in replacements:
            result.graph.node.extend(replacements[node.name])
        elif node.name not in removed:
            result.graph.node.append(copy.deepcopy(node))
    result.graph.initializer.extend(new_constants)
    require((len(result.graph.node),len(result.graph.initializer)) ==
        (PARENT_COUNTS[key][0]+44,PARENT_COUNTS[key][1]+7), 'Derived counts differ')
    newnodes = {n.name:n for n in result.graph.node}
    newinits = {t.name:t for t in result.graph.initializer}
    require(len(newnodes) == len(result.graph.node) and len(newinits) == len(result.graph.initializer), 'Duplicate derived symbol')
    require(all(newnodes[name].SerializeToString() == node.SerializeToString() for name,node in nodes.items() if name not in removed),
        'A normalization, packing or other non-RoPE node changed')
    require(all(newinits[name].SerializeToString() == tensor.SerializeToString() for name,tensor in inits.items()),
        'A parent fixed or learned tensor changed')
    require(all(t.data_type == T.INT64 for t in new_constants), 'No new floating values are permitted')
    for node in constant_slices + [node for sub in role_parts.values() for node in sub]:
        require(newnodes[node.name].SerializeToString() == node.SerializeToString(), 'Actual generated block differs')
    # Reconstruct from actual transformed graph, replacing its first input slice
    # by each pinned five-node RoPE block and deleting only the added nodes/tensors.
    inverse = copy.deepcopy(result); inverse.graph.name = parent.graph.name
    del inverse.graph.node[:]
    anchors = {role_parts[role][0].name: block for role,block in original_blocks.items()}
    for node in result.graph.node:
        if node.name in anchors:
            inverse.graph.node.extend(copy.deepcopy(anchors[node.name]))
        if not node.name.startswith(NS):
            inverse.graph.node.append(copy.deepcopy(node))
    del inverse.graph.initializer[:]
    inverse.graph.initializer.extend(t for t in result.graph.initializer if not t.name.startswith(NS))
    require(inverse.SerializeToString() == original, 'Actual inverse does not reconstruct pinned parent bytes')
    for section in ('input','output','value_info'):
        require([v.SerializeToString() for v in getattr(result.graph,section)] ==
            [v.SerializeToString() for v in getattr(parent.graph,section)], 'Original public/annotated I/O changed')
    available = set(newinits) | {v.name for v in result.graph.input}
    for node in result.graph.node:
        require(all(not name or name in available for name in node.input), 'Nonclosed input: '+node.name)
        require(not set(node.output)&available, 'Duplicate tensor output: '+node.name)
        available.update(node.output)
    require(len(result.graph.input) == 1 and result.graph.input[0].name == '__global_input' and
        all(v.name in available for v in result.graph.output), 'Unexpected graph I/O')
    onnx.checker.check_model(result,full_check=True)
    inferred = onnx.shape_inference.infer_shapes(result,check_type=True,strict_mode=True,data_prop=True)
    onnx.checker.check_model(inferred,full_check=True)
    require([v.SerializeToString() for v in inferred.graph.output] == [v.SerializeToString() for v in result.graph.output],
        'Inferred original output contract differs')
    return result,dict(proof={name:True for name in PROOF_TRUE},
        parent_serialization_inverse_sha256=digest(inverse.SerializeToString()),
        original64_permutation=permutation,rotated_quarter_order=list(PERM_QUARTERS),quarter_width=16,
        original_layer_norm_dimension=64,layer_norm_split_or_approximation=False,
        original_learned_TensorProto_sha256s={name:digest(inits[name].SerializeToString()) for name in LEARNED},
        all_retained_TensorProto_sha256s={name:digest(t.SerializeToString()) for name,t in inits.items()},
        replaced_RoPE_NodeProtos_base64={name:base64.b64encode(nodes[name].SerializeToString()).decode() for name in sorted(removed)},
        added_NodeProto_sha256s={name:digest(node.SerializeToString()) for name,node in newnodes.items() if name.startswith(NS)},
        added_INT64_TensorProto_sha256s={t.name:digest(t.SerializeToString()) for t in new_constants},
        unchanged_non_RoPE_node_count=len(nodes)-10,shared_original_coefficient_Slice_count=12,
        scalar_operation_order='rot=x[paired_channel]*sign[channel]; a=x[channel]*cos[channel,token]; b=rot*sin[channel,token]; y=a+b',
        complete_quarter_results_concatenated_in_original_channel_order=True,
        generated_mapping_proof=generated_mapping_proof(permutation,signs))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--cf-canonical-parent', type=Path)
    p.add_argument('--cf-native-parent', type=Path)
    a = p.parse_args()
    if not a.execute:
        print(json.dumps(dict(metadata(), historical_source=HISTORICAL_SOURCE,
            published_entrypoint='Portable parent paths; new receipts, not a historical device execution.'), indent=2))
        return 0
    require(a.output_dir is not None and not a.output_dir.exists(), 'A fresh output directory is required')
    require(a.cf_canonical_parent and a.cf_native_parent, 'Both exact CF parent paths are required')
    paths = dict(cf_canonical=a.cf_canonical_parent, cf_native=a.cf_native_parent)
    sources = {key: descriptor(path, PARENT_PINS[key]) for key, path in paths.items()}
    import onnx
    graphs, proofs = {}, {}
    for key, path in paths.items():
        graph, proof = build(onnx.load(path, load_external_data=False), key)
        graphs[key.replace('cf_', 'cf_quarter_')] = graph
        proofs[key] = proof
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
