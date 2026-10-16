#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-2-Clause
"""Extract the pinned original QKV prefix; static ONNX only, no inference.

Default mode reads no artifact. Supply the separately licensed original parent
with --execute to create new local graphs. No weights are distributed here.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

HISTORICAL_SOURCE = {'bytes': 12350, 'sha256': '5477cd721b419a72e45d79d364b703840f62d640961e97b467ec022cf99282ed'}

PARENT_PIN = (102417440, '57b60feceea0edb57ca4cea38200d6ea47d4e4ee51e4a546357267cbc9f676bb')


LINEAGE_PIN = (64520, 'e23c8f7cdf6faa895a8bde34a41806bf2022e630f5361529d0271374742d740e')


CAPTURE_PIN = (12587, 'e4ffaea2b11df9ac558c119cb4854d490b0030c02b7706eee344b8894cefa686')


NS = '__original_native_layout/'


N, H, D, ROWS = 2082, 16, 64, 694


INPUT = '__global_input'


ORIGINAL_OUTPUTS = ['global0/query_rope', 'global0/key_rope', 'global0/value']


RANGES = [(0, 694), (694, 1388), (1388, 2082)]


ROLES = [('Q0', 694), ('Q1', 694), ('Q2', 694), ('K', 2082), ('V', 2082)]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def descriptor(path, pin=None):
    path = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    result = dict(file=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())
    require(pin is None or (result['bytes'], result['sha256']) == pin, 'Pinned source differs: ' + str(path))
    return result


def output_contract():
    return [dict(role=role, name=NS + role, shape=[1, 2, rows, 512],
                 dtype='float16', bytes=2 * rows * 512 * 2,
                 consumer_view_shape=[1, 2, rows, 64, 8]) for role, rows in ROLES]


def build(parent):
    # Import only ONNX: no inference engine or execution of captured values.
    import onnx
    from onnx import TensorProto as T, helper as h
    raw_parent = parent.SerializeToString()
    require((len(raw_parent), hashlib.sha256(raw_parent).hexdigest()) == PARENT_PIN,
            'Original-A serialization must match its exact file')
    nodes = list(parent.graph.node)
    initializers = {t.name: t for t in parent.graph.initializer}
    require(len(nodes) == 143 and len(initializers) == 79, 'Original-A graph counts differ')
    require(nodes[89].name == '__v126_original_frame_global_view' and
            nodes[90].name == 'global0/norm1' and nodes[110].name == 'global0/key_transpose',
            'Original uncentered prefix cut differs')
    selected = [copy.deepcopy(n) for n in nodes[90:110]]
    used = {name for node in selected for name in node.input}
    original_init = [copy.deepcopy(t) for t in parent.graph.initializer if t.name in used]
    require(len(original_init) == 16 and len(selected) == 20, 'Original prefix closure differs')
    produced = {name for node in selected for name in node.output}
    require(used - set(initializers) - produced == {INPUT}, 'Unexpected original prefix input')
    require(set(ORIGINAL_OUTPUTS) <= produced, 'Original post-RoPE Q/K and raw V required')
    require(not any(NS in name for n in nodes for name in list(n.input) + list(n.output)),
            'Added layout namespace collision')
    require(not any('centered' in n.name or 'value_mean' in n.name for n in selected),
            'Only uncentered original QKV allowed')

    def vi(name, dtype, shape):
        return h.make_tensor_value_info(name, dtype, shape)

    def model(name, graph_nodes, inits, outs):
        result = copy.deepcopy(parent)
        result.graph.CopyFrom(h.make_graph(graph_nodes, name,
            [vi(INPUT, T.FLOAT, [1, N, 1024])], outs, inits))
        onnx.checker.check_model(result, full_check=True)
        inferred = onnx.shape_inference.infer_shapes(result, check_type=True, strict_mode=True, data_prop=True)
        onnx.checker.check_model(inferred, full_check=True)
        actual = {v.name: v for v in inferred.graph.output}
        require([v.SerializeToString() for v in outs] ==
                [actual[v.name].SerializeToString() for v in outs], 'Inferred public output contract differs')
        return result

    canonical = model('original_global0_canonical_QKV', selected, original_init,
                      [vi(name, T.FLOAT, [1, H, N, D]) for name in ORIGINAL_OUTPUTS])
    extra, constants = [], []

    def const(name, values):
        name = NS + name
        constants.append(h.make_tensor(name, T.INT64, [len(values)], values))
        return name

    def node(op, ins, outs, name, **attrs):
        extra.append(h.make_node(op, ins, outs, name=NS + name, **attrs))

    slice_axis, slice_step = const('slice_axis', [2]), const('slice_step', [1])
    for index, (start, end) in enumerate(RANGES):
        node('Slice', [ORIGINAL_OUTPUTS[0], const('start_' + str(index), [start]),
                      const('end_' + str(index), [end]), slice_axis, slice_step],
             [NS + 'Q' + str(index) + '/canonical'], 'Q' + str(index) + '/slice')
    shape5 = {rows: const('split_heads_' + str(rows), [1, 2, 8, rows, D]) for rows in (ROWS, N)}
    shape4 = {rows: const('flat_native_' + str(rows), [1, 2, rows, D * 8]) for rows in (ROWS, N)}
    for role, rows in ROLES:
        src = NS + role + '/canonical' if role.startswith('Q') else ORIGINAL_OUTPUTS[1 if role == 'K' else 2]
        node('Reshape', [src, shape5[rows]], [NS + role + '/split'], role + '/split_heads')
        node('Transpose', [NS + role + '/split'], [NS + role + '/native'], role + '/native_order',
             perm=[0, 1, 3, 4, 2])
        node('Reshape', [NS + role + '/native', shape4[rows]], [NS + role + '/flat'], role + '/flat_view')
        node('Cast', [NS + role + '/flat'], [NS + role], role + '/half', to=T.FLOAT16)
    candidate = model('original_global0_native_half_QKV', selected + extra, original_init + constants,
                      [vi(row['name'], T.FLOAT16, row['shape']) for row in output_contract()])
    require(len(extra) == 23 and len(constants) == 12, 'Only declared layout/slice/cast operations permitted')
    require(len(candidate.graph.node) == 43 and len(candidate.graph.initializer) == 28, 'Candidate counts differ')

    # Recover the actual retained prefix from each graph, and then reinsert into
    # the unchanged parent remainder. This proves source preservation, not full
    # model execution or numerical equivalence of compiler-selected kernels.
    for graph in (canonical, candidate):
        require([n.SerializeToString() for n in graph.graph.node[:20]] ==
                [n.SerializeToString() for n in nodes[90:110]], 'Original node bytes changed')
        derived_init = {t.name: t for t in graph.graph.initializer}
        require(all(derived_init[t.name].SerializeToString() == t.SerializeToString()
                    for t in original_init), 'Original parameter or RoPE bytes changed')
        inverse = copy.deepcopy(parent)
        del inverse.graph.node[:]
        inverse.graph.node.extend(nodes[:90] + list(graph.graph.node[:20]) + nodes[110:])
        del inverse.graph.initializer[:]
        inverse.graph.initializer.extend(derived_init.get(t.name, t) for t in parent.graph.initializer)
        require(inverse.SerializeToString() == raw_parent, 'Exact parent inverse failed')
    require([i for start, end in RANGES for i in range(start, end)] == list(range(N)), 'Query coverage differs')
    witnesses = []
    for role, rows in ROLES:
        for head in range(H):
            for token in (0, rows - 1):
                for channel in (0, D - 1):
                    # Flattened [1,2,T,512] storage equals [1,2,T,64,8].
                    public_offset = ((head // 8) * rows + token) * 512 + channel * 8 + head % 8
                    native_offset = (((head // 8) * rows + token) * D + channel) * 8 + head % 8
                    require(public_offset == native_offset, 'Native word coordinate differs')
                    witnesses.append([role, head, token, channel, public_offset])
    proof = dict(original20_NodeProtos_and16_TensorProtos_byte_unchanged=True,
                 whole143_node79_initializer_parent_inverse_byte_equal=True,
                 inverse_scope='Only original global0 prefix is exported. Parent remainder is retained unchanged for this inverse proof, not executed.',
                 original_full_token_QKV_norm_RoPE_and_raw_V_unchanged=True,
                 no_centering_or_alternative_learned_values=True,
                 unique_HALF_payload_bytes=sum(row['bytes'] for row in output_contract()),
                 query_ranges=[list(pair) for pair in RANGES],
                 added_ops={'Slice': 3, 'Reshape': 10, 'Transpose': 5, 'Cast': 5},
                 native_identity='B[0,h//8,t,d,h%8]=FLOAT16(X[0,h,t,d])',
                 boundary_integer_witnesses=witnesses,
                 all_new_constants_are_INT64_shapes_or_slice_indices=True,
                 declared_rank4_storage_to_rank5_consumer_view_changes_shape_only=True,
                 runtime_ABI_or_CPU_NPU_precision_or_latency_established=False)
    return dict(canonical=canonical, native_half=candidate), proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--parent', type=Path)
    p.add_argument('--output-dir', type=Path)
    a = p.parse_args()
    if not a.execute:
        print(json.dumps(dict(schema='original_QKV_prefix_static_source', parent_pin=PARENT_PIN,
            original_capture_pin=CAPTURE_PIN, historical_source=HISTORICAL_SOURCE,
            default_reads_artifacts_or_imports_frameworks=False, candidate_accepted=False), indent=2))
        return 0
    require(a.parent is not None and a.output_dir is not None and not a.output_dir.exists(),
        'An exact parent and fresh output directory are required')
    source = descriptor(a.parent, PARENT_PIN)
    import onnx
    graphs, proof = build(onnx.load(a.parent, load_external_data=False))
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
    require(descriptor(source['file']) == source, 'Parent changed during export')
    report = dict(graphs=outputs, proof=proof, exact_parent_artifact=source,
        exporter=descriptor(__file__), historical_source=HISTORICAL_SOURCE,
        static_only=True, inference_SDK_calibration_or_GPU_executed=False, candidate_accepted=False)
    with (a.output_dir / 'result.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(graphs=outputs), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
