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

HISTORICAL_SOURCE = {'bytes': 14708, 'sha256': '13c34926bc29b00336d508647d015a92a3bf817cfafd8823fe3905c15e95146e'}

PARENT_STATIC_PIN = (19491, '94fa2b3644ca2660028d1120f48d97f3a4b108a0781220e300afa48676fe69b8')


PARENT_PINS = {
    'cf_canonical': (13675800, '927d20943b900d026e795ba361c45ff30173402bd9950d22cedd0da57945ea7f'),
    'cf_native': (13678828, 'decd625e8db757e06fbeb8eb0bdea780c3fb313916429410ff98a205c29f2b3b')}


PARENT_COUNTS = {'cf_canonical': (24, 16), 'cf_native': (40, 27)}


EXPORTER_PIN = (29930, '6f0cfa49c639cb61a73000150f9a78cca50332211b616fd890f208776cc89ada')


CAPTURE_SHA = 'e4ffaea2b11df9ac558c119cb4854d490b0030c02b7706eee344b8894cefa686'


SCHEMA = 'original_native518_CF_RoPE_slice_concat_static'


SUCCESS = 'two_original_CF_Gathers_replaced_by_exact_four_slice_concat_CPU_and_NPU_gates_pending'


NS = '__original_CF_slice_RoPE/'


RANGES = ((16, 32), (0, 16), (48, 64), (32, 48))


PROOF_TRUE = (
    'all_parent_TensorProtos_byte_unchanged_including_all8_learned',
    'only_two_CF_Gather_NodeProtos_replaced',
    'all_other_NodeProtos_and_public_IO_byte_unchanged',
    'all_original_RoPE_Mul_Add_nodes_order_and_inputs_byte_unchanged',
    'actual_original64_permutation_equals_ordered_slice_indices',
    'permutation_has_no_duplicate_or_dropped_channels',
    'all_added_constants_are_INT64_slice_indices',
    'actual_replacement_subgraphs_match_exact_Slice_Concat_contract',
    'actual_inverse_to_both_parent_serializations_byte_equal',
    'all2082_positions_16heads_64channels_and_query_partitions_retained',
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
        graph_keys=['cf_slice_canonical', 'cf_slice_native'], original_slice_ranges=RANGES,
        slice_axis=1, concat_axis=1, added_INT64_constants=7,
        node_count_delta=8, initializer_count_delta=7, proof_fields=PROOF_TRUE,
        change='Only each channel-axis Gather becomes four Slices and one Concat; all Mul/Add and output layouts remain unchanged.',
        compiler_fusion_change_is_hypothesis_not_fact=True,
        default_reads_artifacts_or_imports_frameworks=False,
        candidate_accepted=False, CPU_NPU_precision_or_speed_accepted=False)


def int64_values(tensor, T):
    require(tensor.data_type == T.INT64 and len(tensor.dims) == 1 and not tensor.external_data,
        'Inline rank1 INT64 indices required')
    n = tensor.dims[0]
    if tensor.raw_data:
        require(len(tensor.raw_data) == n * 8 and not tensor.int64_data, 'Raw INT64 span differs')
        return list(struct.unpack('<' + 'q' * n, tensor.raw_data))
    require(len(tensor.int64_data) == n, 'INT64 repeated field differs')
    return list(tensor.int64_data)


def build(parent, key):
    import onnx
    from onnx import TensorProto as T, helper as h
    original = parent.SerializeToString()
    require((len(original), digest(original)) == PARENT_PINS[key], 'Exact parent bytes required')
    require((len(parent.graph.node), len(parent.graph.initializer)) == PARENT_COUNTS[key], 'Parent counts differ')
    initials = {t.name: t for t in parent.graph.initializer}
    nodes = {n.name: n for n in parent.graph.node}
    require(len(initials) == len(parent.graph.initializer) and len(nodes) == len(parent.graph.node), 'Duplicate parent names')
    permutation = int64_values(initials['global0/rope_permutation'], T)
    indices = [i for lo, hi in RANGES for i in range(lo, hi)]
    require(permutation == indices and sorted(indices) == list(range(64)), 'Actual original permutation differs')
    sign = initials['global0/rope_rotation_sign']
    require(sign.data_type == T.FLOAT and list(sign.dims) == [1, 64, 1, 1] and
        not sign.external_data and not sign.float_data and len(sign.raw_data) == 256 and
        list(struct.unpack('<64f', sign.raw_data)) == [-1.] * 16 + [1.] * 16 + [-1.] * 16 + [1.] * 16,
        'Actual original CF sign differs')
    require(not any(NS in name for node in parent.graph.node for name in [node.name, *node.input, *node.output]) and
        not any(name.startswith(NS) for name in initials), 'New namespace collision')
    targets = {role: nodes['global0/' + role + '_rope_permuted'] for role in ('query', 'key')}
    for role, node in targets.items():
        require(node.op_type == 'Gather' and not node.domain and len(node.attribute) == 1 and
            node.attribute[0].name == 'axis' and node.attribute[0].i == 1 and
            list(node.input) == ['__original_CF_RoPE/layout/global0/attn.' + role[0] + '_norm', 'global0/rope_permutation'] and
            list(node.output) == ['__original_CF_RoPE/layout/global0/' + role + '_rope_permuted'],
            'Exact original CF Gather required: ' + role)
    additions = [h.make_tensor(NS + 'bound_' + str(i), T.INT64, [1], [i]) for i in (0, 16, 32, 48, 64)]
    additions += [h.make_tensor(NS + 'axis', T.INT64, [1], [1]), h.make_tensor(NS + 'step', T.INT64, [1], [1])]
    replacements = {}
    for role, old in targets.items():
        slices = [h.make_node('Slice', [old.input[0], NS + 'bound_' + str(lo), NS + 'bound_' + str(hi),
            NS + 'axis', NS + 'step'], [NS + role + '/part_' + str(i)],
            name=NS + role + '/slice_' + str(i)) for i, (lo, hi) in enumerate(RANGES)]
        concat = h.make_node('Concat', [node.output[0] for node in slices], list(old.output), name=old.name, axis=1)
        replacements[old.name] = slices + [concat]
    result = copy.deepcopy(parent)
    result.graph.name = 'original_global0_' + key.replace('cf_', 'cf_slice_')
    del result.graph.node[:]
    for node in parent.graph.node:
        result.graph.node.extend(replacements.get(node.name, [copy.deepcopy(node)]))
    result.graph.initializer.extend(additions)
    require((len(result.graph.node), len(result.graph.initializer)) ==
        (PARENT_COUNTS[key][0] + 8, PARENT_COUNTS[key][1] + 7), 'Replacement graph counts differ')
    newnodes = {n.name: n for n in result.graph.node}
    newinitials = {t.name: t for t in result.graph.initializer}
    require(len(newnodes) == len(result.graph.node) and len(newinitials) == len(result.graph.initializer), 'Duplicate derived symbols')
    require(all(newinitials[name].SerializeToString() == tensor.SerializeToString() for name, tensor in initials.items()),
        'Parent fixed or learned tensor changed')
    require(all(newnodes[name].SerializeToString() == node.SerializeToString() for name, node in nodes.items() if name not in replacements),
        'A non-Gather operation changed')
    require(all(t.data_type == T.INT64 for t in additions), 'Only integer indexing constants may be introduced')
    for name, subgraph in replacements.items():
        require(all(newnodes[n.name].SerializeToString() == n.SerializeToString() for n in subgraph),
            'Actual Slice/Concat contract differs')
    inverse = copy.deepcopy(result)
    inverse.graph.name = parent.graph.name
    del inverse.graph.node[:]
    for node in result.graph.node:
        if node.name.startswith(NS):
            continue
        inverse.graph.node.append(copy.deepcopy(nodes[node.name] if node.name in replacements else node))
    del inverse.graph.initializer[:]
    inverse.graph.initializer.extend(t for t in result.graph.initializer if not t.name.startswith(NS))
    require(inverse.SerializeToString() == original, 'Actual inverse does not reconstruct pinned parent')
    for name in ('input', 'output', 'value_info'):
        require([v.SerializeToString() for v in getattr(parent.graph, name)] ==
            [v.SerializeToString() for v in getattr(result.graph, name)], 'Public/intermediate annotations changed')
    available = set(newinitials) | {v.name for v in result.graph.input}
    for node in result.graph.node:
        require(all(not name or name in available for name in node.input), 'Nonclosed graph input: ' + node.name)
        require(not (set(node.output) & available), 'Duplicate value producer: ' + node.name)
        available.update(node.output)
    require(len(result.graph.input) == 1 and result.graph.input[0].name == '__global_input' and
        all(o.name in available for o in result.graph.output), 'Unexpected graph I/O')
    onnx.checker.check_model(result, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(result, check_type=True, strict_mode=True, data_prop=True)
    onnx.checker.check_model(inferred, full_check=True)
    require([v.SerializeToString() for v in inferred.graph.output] == [v.SerializeToString() for v in result.graph.output],
        'Inferred output differs')
    # Generated words exercise exact bit movement, including signed zeros, F16
    # subnormals and NaN payload patterns. No captured activations are used.
    synthetic_comparisons = 0
    patterns = (0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0xfc00, 0x7e01)
    for head in range(16):
        for token in range(3):
            words = [struct.pack('<HH', patterns[(c + head + token) % len(patterns)], c) for c in range(64)]
            selected = b''.join(words[i] for i in permutation)
            sliced = b''.join(b''.join(words[lo:hi]) for lo, hi in RANGES)
            require(selected == sliced, 'Synthetic exact word movement differs')
            synthetic_comparisons += 64
    return result, dict(parent_serialization_inverse_sha256=digest(inverse.SerializeToString()),
        original64_permutation=permutation, slice_ranges=[list(r) for r in RANGES],
        retained_initializers={name: digest(t.SerializeToString()) for name, t in initials.items()},
        replaced_Gather_protos_base64={name: base64.b64encode(nodes[name].SerializeToString()).decode() for name in replacements},
        replacement_NodeProto_sha256s={name: [digest(n.SerializeToString()) for n in sub] for name, sub in replacements.items()},
        added_INT64_TensorProto_sha256s={t.name: digest(t.SerializeToString()) for t in additions},
        generated_word_components_checked=synthetic_comparisons,
        unchanged_other_nodes=len(parent.graph.node)-2,
        unused_original_permutation_initializer_retained_for_exact_provenance=True,
        proof={name: True for name in PROOF_TRUE})

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
        graphs[key.replace('cf_', 'cf_slice_')] = graph
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
