#!/usr/bin/env python3
"""Exact protobuf cuts of original DINO0; no model/ORT/SDK execution."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import sys
import traceback

PARENT_SHA = 'd8ce26c2f9cb3018b9a24da9f722937569067a5cffde650288f0073743e547e8'
PARENT_BYTES = 50397502
ORIGINAL_EXPORT_SHA = 'bd78874ccc7373121696dea581ad88288425cd3878aaadf02a8ae01fd563d532'
ORIGINAL_EXPORTER_SHA = 'abea44db3785d3e169663c9bcb08bb688df2cdfbb0b3ea0318ef2d794b17c33d'
CHECKPOINT = dict(bytes=5026874952, sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')
SCHEMA = 'v120_original_native518_DINO0_exact_cuts_ONNX_export'
SUCCESS = 'exported_original_DINO0_exact_cuts_not_FP32_or_NPU_accepted'
INPUT = (1, 1041, 1024)
QKV = (1, 1041, 3072)
TRANSPOSED = {'attn.qkv.weight', 'attn.proj.weight', 'mlp.fc1.weight', 'mlp.fc2.weight'}
PARAMETERS = ('norm1.weight', 'norm1.bias', 'attn.qkv.weight', 'attn.qkv.bias',
    'attn.proj.weight', 'attn.proj.bias', 'ls1.gamma', 'norm2.weight', 'norm2.bias',
    'mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight', 'mlp.fc2.bias', 'ls2.gamma')
SPECS = (('prefix', ('input',), ('attn.qkv',), 3, 4),
    ('middle', ('attn.qkv',), ('attention_heads_flatten',), 12, 6),
    ('suffix', ('input', 'attention_heads_flatten'), ('output',), 16, 13))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def pin(path, expected, size=None):
    p = Path(path).resolve(strict=True)
    require(sha(p) == expected and (size is None or p.stat().st_size == size), 'Pinned file differs: ' + str(p))
    return p


def board_guard():
    require(platform.machine() == 'aarch64' and b'rockchip,rk3588' in
        Path('/proc/device-tree/compatible').read_bytes(), 'RK3588S board only')


def fresh_output(path):
    p = Path(path).resolve()
    require(str(p).startswith('/tmp/vggt-v120-'), 'Own fresh /tmp/vggt-v120-* output required')
    p.mkdir(parents=True, exist_ok=False)
    return p


def atomic(path, row):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(row, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def parameter_proof(model, original):
    import numpy as np
    from onnx import numpy_helper
    require(original['schema'] == 'v99_native518_original_DINO0_ONNX_export' and
        original['status'] == 'exported_original_native518_DINO0_not_FP32_accepted' and
        original['script_sha256'] == ORIGINAL_EXPORTER_SHA and original['checkpoint'] == CHECKPOINT and
        original['onnx']['sha256'] == PARENT_SHA and original['onnx']['bytes'] == PARENT_BYTES and
        original['all_original_parameter_values_preserved'] and original['parameter_count'] == 14,
        'Exact pinned original DINO0 export provenance required')
    params = original['blocks'][0]['parameters']
    require(set(params) == set(PARAMETERS), 'Exactly 14 original learned tensors required')
    tensors = {t.name: t for t in model.graph.initializer}
    learned = {}
    for name in PARAMETERS:
        tensor_name = name + '_transposed' if name in TRANSPOSED else name
        require(tensor_name in tensors, 'Original learned initializer missing: ' + name)
        a = numpy_helper.to_array(tensors[tensor_name])
        value = np.ascontiguousarray(a.T if name in TRANSPOSED else a)
        meta = dict(shape=list(value.shape), dtype=str(value.dtype), tensor_bytes=value.nbytes,
            raw_sha256=hashlib.sha256(memoryview(value).cast('B')).hexdigest())
        require(value.dtype == np.float32 and np.isfinite(value).all() and meta == params[name],
            'Original learned value changed: ' + name)
        learned[name] = meta
    return learned


def split_graphs(parent, original):
    import onnx
    from onnx import helper as H, TensorProto as T, numpy_helper
    onnx.checker.check_model(parent, full_check=True)
    g = parent.graph
    require(not parent.functions and not parent.training_info and not g.sparse_initializer and
        len(g.node) == 31 and len(g.initializer) == 23 and len(g.input) == len(g.output) == 1 and
        all(not t.external_data for t in g.initializer), 'Exact self-contained original 31-node/23-initializer graph required')
    learned = parameter_proof(parent, original)
    nodes = {n.name: n for n in g.node}
    producers = {v: n.name for n in g.node for v in n.output}
    initializers = {t.name: t for t in g.initializer}
    require(len(nodes) == len(g.node) and len(producers) == sum(len(n.output) for n in g.node), 'Unique original dataflow required')
    for name in ('norm1', 'norm2'):
        n = nodes[name]
        attrs = {a.name: H.get_attribute_value(a) for a in n.attribute}
        require(n.op_type == 'LayerNormalization' and attrs['axis'] == -1 and
            abs(attrs['epsilon'] - 1e-6) <= 1e-12 and attrs['stash_type'] == T.FLOAT,
            'Original DINO LayerNorm semantics required')
    require(nodes['gelu_erf'].op_type == 'Erf' and nodes['attention_probabilities'].op_type == 'Softmax' and
        H.get_attribute_value(next(a for a in nodes['attention_probabilities'].attribute if a.name == 'axis')) == -1 and
        float(numpy_helper.to_array(initializers['attention_scale'])) == .125,
        'Original exact Erf and full-key scaled softmax required')
    inferred = onnx.shape_inference.infer_shapes(parent, strict_mode=True, check_type=True, data_prop=True)
    info = {v.name: v for v in (*inferred.graph.input, *inferred.graph.output, *inferred.graph.value_info)}
    for name, shape in (('input', INPUT), ('output', INPUT), ('attn.qkv', QKV), ('attention_heads_flatten', INPUT)):
        v = info[name].type.tensor_type
        require(v.elem_type == T.FLOAT and all(d.HasField('dim_value') for d in v.shape.dim) and
            tuple(d.dim_value for d in v.shape.dim) == shape, 'Exact original boundary shape differs: ' + name)
    def closure(outputs, boundaries):
        selected, visiting = set(), set()
        def visit(value):
            if not value or value in boundaries or value in initializers:
                return
            require(value in producers, 'Unbound original tensor: ' + value)
            name = producers[value]
            if name in selected:
                return
            require(name not in visiting, 'Original graph cycle')
            visiting.add(name)
            for value in nodes[name].input:
                visit(value)
            visiting.remove(name)
            selected.add(name)
        for value in outputs:
            visit(value)
        return selected
    before = parent.SerializeToString()
    models, rows, union, init_union = {}, [], set(), set()
    for key, inputs, outputs, expected_nodes, expected_inits in SPECS:
        selected = closure(outputs, set(inputs))
        used = {v for n in g.node if n.name in selected for v in n.input if v in initializers}
        require(len(selected) == expected_nodes and len(used) == expected_inits and not selected & union,
            'Original cut membership differs: ' + key)
        union.update(selected)
        init_union.update(used)
        model = copy.deepcopy(parent)
        del model.graph.node[:]
        model.graph.node.extend(n for n in g.node if n.name in selected)
        del model.graph.initializer[:]
        model.graph.initializer.extend(t for t in g.initializer if t.name in used)
        del model.graph.input[:]
        model.graph.input.extend(copy.deepcopy(info[n]) for n in inputs)
        del model.graph.output[:]
        model.graph.output.extend(copy.deepcopy(info[n]) for n in outputs)
        del model.graph.value_info[:]
        model = onnx.shape_inference.infer_shapes(model, strict_mode=True, check_type=True, data_prop=True)
        onnx.checker.check_model(model, full_check=True)
        require([n.SerializeToString() for n in model.graph.node] ==
            [n.SerializeToString() for n in g.node if n.name in selected] and
            [t.SerializeToString() for t in model.graph.initializer] ==
            [t.SerializeToString() for t in g.initializer if t.name in used], 'Original cut protobufs changed')
        models[key] = model
        def bindings(values):
            return [dict(index=i, name=v.name, shape=[d.dim_value for d in v.type.tensor_type.shape.dim], dtype='float32')
                for i, v in enumerate(values)]
        rows.append(dict(key=key, node_count=len(model.graph.node), initializer_count=len(model.graph.initializer),
            original_node_names=[n.name for n in model.graph.node],
            input_bindings=bindings(model.graph.input), output_bindings=bindings(model.graph.output)))
    require(union == set(nodes) and init_union == set(initializers), 'Full original node/initializer union required')
    recovered = copy.deepcopy(parent)
    parts = {n.name: n for m in models.values() for n in m.graph.node}
    init_parts = {}
    for m in models.values():
        for t in m.graph.initializer:
            require(t.name not in init_parts or init_parts[t.name].SerializeToString() == t.SerializeToString(),
                'Shared original initializer bytes differ')
            init_parts[t.name] = t
    del recovered.graph.node[:]
    recovered.graph.node.extend(parts[n.name] for n in g.node)
    del recovered.graph.initializer[:]
    recovered.graph.initializer.extend(init_parts[t.name] for t in g.initializer)
    require(recovered.SerializeToString() == before, 'Reassembled complete original model protobuf differs')
    proof = dict(original_parent_nodes=31, original_parent_initializers=23,
        original_node_union_exactly_once=True, whole_parent_reassembled_protobuf_equal=True,
        full_initializer_union_original_protobuf_equal=True, all14_original_learned_parameters_verified=True,
        original_middle_12_nodes_preserved=True, original_residual_input_preserved_in_suffix=True,
        all1041_queries_keys_16_heads_64_dimensions=True, original_exact_Erf_GELU_preserved=True,
        norm1_epsilon=1e-6, norm2_epsilon=1e-6, attention_scale=.125,
        mathematical_nodes_or_learned_values_changed=False,
        changed_cut_boundaries_may_change_SDK_fusion_and_FP16_rounding=True)
    return models, rows, proof, learned


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-onnx', type=Path, required=True)
    p.add_argument('--parent-sha256', default=PARENT_SHA)
    p.add_argument('--original-export', type=Path, required=True)
    p.add_argument('--original-export-sha256', default=ORIGINAL_EXPORT_SHA)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    board_guard()
    out = fresh_output(a.output_dir)
    r = dict(schema=SCHEMA, status='failed', script_sha256=sha(__file__), original_model_only=True,
        model_ORT_SDK_GPU_execution=False, checkpoint_deserialized=False, candidate_accepted=False, cleanup_errors=[])
    watched = {Path(__file__).resolve(): r['script_sha256']}
    try:
        require(a.parent_sha256 == PARENT_SHA and a.original_export_sha256 == ORIGINAL_EXPORT_SHA, 'Exact original lineage required')
        parent = pin(a.parent_onnx, PARENT_SHA, PARENT_BYTES)
        source = pin(a.original_export, ORIGINAL_EXPORT_SHA)
        watched.update({parent: PARENT_SHA, source: ORIGINAL_EXPORT_SHA})
        original = json.loads(source.read_text())
        import onnx
        models, rows, proof, learned = split_graphs(onnx.load_model(parent), original)
        for row in rows:
            path = out / (row['key'] + '.onnx')
            path.write_bytes(models[row['key']].SerializeToString())
            graph = dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path))
            row.update(graph)
            row['onnx'] = dict(graph)
        r.update(status=SUCCESS, parent=dict(file=str(parent), bytes=PARENT_BYTES, sha256=PARENT_SHA),
            original_export_receipt=dict(file=str(source), sha256=ORIGINAL_EXPORT_SHA), checkpoint=CHECKPOINT,
            original_parameter_metadata=learned, original_source_hashes=original['source_hashes'], models=rows, proof=proof)
    except BaseException as error:
        r.update(error=repr(error), traceback=traceback.format_exc())
    finally:
        for path, digest in watched.items():
            if not path.is_file() or sha(path) != digest:
                r['cleanup_errors'].append('Pinned source/input changed: ' + str(path))
        r['source_and_inputs_unchanged'] = not r['cleanup_errors']
        if r['cleanup_errors']:
            r['status'] = 'failed'
        atomic(out / 'export.json', r)
    print(json.dumps({k: r.get(k) for k in ('status', 'models', 'error')}), flush=True)
    return 0 if r['status'] == SUCCESS else 1


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
