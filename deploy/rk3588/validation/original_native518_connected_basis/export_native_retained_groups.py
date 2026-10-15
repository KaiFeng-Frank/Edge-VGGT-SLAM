#!/usr/bin/env python3
"""Static native518 original aggregator group; no model/ORT/RKNN execution.

Reuse frozen original block/pair construction and reversible retained-group
composition. Export one group per invocation to bound transient memory/storage.
Original B1 frame branches and optional global FC2 K4 need full-group FP32
validation before compilation; this exporter makes no numerical acceptance claim.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
BASE = Path('/mnt/vggt-data/vggt-slam2')
FRAME, GLOBAL, IMAGE = [2, 1041, 1024], [1, 2082, 1024], [2, 3, 392, 518]
RANGES = ((0, 4), (5, 11), (12, 17), (18, 23))
WEIGHTS_SHA = '4b723e499763bb90565f6e11e16c551664634aaf72da46c7cdc493b84208ef67'
GRAPH_SHA = '3ff8b0d0515ff44ea3625bd2c6529d63a0bed3487d409128ee4b443922a14bfd'
COMPOSER_SHA = 'e6fa508e0b03a011a1ccb484f2a46847ffd49fc1bf6ae11ecfeb8bd45f835f60'
SUCCESS = 'static_original_native_group_not_FP32_or_RKNN_accepted'


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(4 << 20), b''): h.update(data)
    return h.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def raw_metadata(value, np):
    value = np.ascontiguousarray(value)
    return dict(shape=list(value.shape), dtype=str(value.dtype), tensor_bytes=value.nbytes,
                raw_sha256=hashlib.sha256(memoryview(value).cast('B')).hexdigest())


def parameter_check(model, weights, G, np, N):
    tensors = {t.name: t for t in model.graph.initializer}
    for name in G.PARAMETER_NAMES:
        stored = name + '_transposed' if name in G.TRANSPOSED_PARAMETERS else name
        actual = N.to_array(tensors[stored]); expected = weights[name].T if name in G.TRANSPOSED_PARAMETERS else weights[name]
        require(raw_metadata(actual, np) == raw_metadata(expected, np), 'Original initializer changed: ' + name)


def frame_batch1(template, np, onnx, H, N, T):
    """Two original independent full-token frames, shared original initializers."""
    require(len(template.graph.input) == len(template.graph.output) == 1 and
            template.graph.input[0].name == 'input' and template.graph.output[0].name == 'output', 'Original B1 template IO differs')
    names = {t.name for t in template.graph.initializer}; extra = [f'__native_batch_index_{i}' for i in range(3)]
    require(not set(extra) & names, 'B1 slice constant collision')
    inits = [copy.deepcopy(t) for t in template.graph.initializer]
    inits.extend(N.from_array(np.array([i], np.int64), name) for i, name in enumerate(extra))
    nodes, outputs, checked = [], [], 0
    for i in range(2):
        prefix = f'batch{i}/'
        nodes.append(H.make_node('Slice', ['input', extra[i], extra[i + 1], extra[0], extra[1]],
                                 [prefix + 'input'], name=prefix + 'input_slice'))
        for original in template.graph.node:
            target = copy.deepcopy(original); target.name = prefix + original.name
            target.input[:] = [v if v in names or not v else prefix + v for v in original.input]
            target.output[:] = [prefix + v for v in original.output]
            inverse = copy.deepcopy(target); inverse.name = original.name
            inverse.input[:] = [v if v in names or not v else v[len(prefix):] for v in target.input]
            inverse.output[:] = [v[len(prefix):] for v in target.output]
            require(inverse.SerializeToString() == original.SerializeToString(), 'B1 branch changed an original node')
            nodes.append(target); checked += 1
        outputs.append(prefix + 'output')
    nodes.append(H.make_node('Concat', outputs, ['output'], name='restore_original_batch', axis=0))
    model = copy.deepcopy(template)
    model.graph.CopyFrom(H.make_graph(nodes, 'original_native_frame_two_B1_branches',
        [H.make_tensor_value_info('input', T.FLOAT, FRAME)], [H.make_tensor_value_info('output', T.FLOAT, FRAME)], initializer=inits))
    onnx.checker.check_model(model, full_check=True)
    actual = {t.name: t.SerializeToString() for t in model.graph.initializer}
    require(all(actual[t.name] == t.SerializeToString() for t in template.graph.initializer), 'Original shared B1 initializer changed')
    return model, dict(layout='two_full_B1_branches', original_18_learned_parameters_stored_once=True,
        original_B1_node_inverse_proto_checks=checked, original_shared_initializer_proto_checks=len(names),
        original_frames_tokens_and_special_token_values_retained=True, added_Slice_count=2, added_Concat_count=1,
        original_per_frame_complete_1041_key_attention=True, original_B2_numerical_equivalence_not_assumed=True)


def global_k4(model, np, onnx, H, N):
    """Scoped full-K FC2 partition; preserve all other original nodes/inits."""
    graph = model.graph; before_nodes = list(graph.node); before_inits = list(graph.initializer)
    name = 'mlp.fc2.weight_transposed'; tensors = {t.name: t for t in before_inits}
    targets = [n for n in before_nodes if n.op_type == 'MatMul' and list(n.input)[-1:] == [name]]
    require(len(targets) == 1, 'Exactly one original global FC2 MatMul required')
    node = targets[0]; weight = np.ascontiguousarray(N.to_array(tensors[name]))
    require(weight.dtype == np.float32 and weight.shape == (4096, 1024) and len(node.input) == 2 and len(node.output) == 1,
            'Original complete global FC2 weight differs')
    bias = next(n for n in before_nodes if n.name == 'mlp.fc2')
    require(bias.op_type == 'Add' and list(bias.input) == [node.output[0], 'mlp.fc2.bias'] and
            [n.name for n in before_nodes if node.output[0] in n.input] == [bias.name], 'Original global FC2 bias must occur once')
    scope = '__native_global_fc2_k4/'; occupied = {t.name for t in before_inits} | {v for n in before_nodes for v in (*n.input, *n.output)}
    require(not any(v.startswith(scope) for v in occupied), 'K4 namespace collision')
    axis, step = scope + 'axis', scope + 'step'
    additions = [N.from_array(np.array([-1], np.int64), axis), N.from_array(np.array([1], np.int64), step)]
    ops, parts, products = [], [], []
    for i in range(4):
        stem = scope + f'part{i}/'; start, end, activation, product, wn = [stem + k for k in ('start', 'end', 'activation', 'product', 'weight')]
        part = N.from_array(np.ascontiguousarray(weight[i * 1024:(i + 1) * 1024]), wn); parts.append(part); products.append(product)
        additions.extend([part, N.from_array(np.array([i * 1024], np.int64), start), N.from_array(np.array([(i + 1) * 1024], np.int64), end)])
        ops.extend([H.make_node('Slice', [node.input[0], start, end, axis, step], [activation], name=activation),
                    H.make_node('MatMul', [activation, wn], [product], name=product)])
    total = products[0]
    for i in range(1, 4):
        output = node.output[0] if i == 3 else scope + f'sum{i}'
        ops.append(H.make_node('Add', [total, products[i]], [output], name=scope + f'add{i}')); total = output
    joined = np.ascontiguousarray(np.concatenate([N.to_array(t) for t in parts], axis=0))
    restored = N.from_array(joined, name)
    require(joined.tobytes() == weight.tobytes() and restored.SerializeToString() == tensors[name].SerializeToString(), 'Original FC2 weight inverse failed')
    candidate = copy.deepcopy(model)
    del candidate.graph.node[:]; candidate.graph.node.extend(x for n in before_nodes for x in (ops if n.name == node.name else [n]))
    del candidate.graph.initializer[:]; candidate.graph.initializer.extend([t for t in before_inits if t.name != name] + additions)
    onnx.checker.check_model(candidate, full_check=True)
    actual_nodes = {n.name: n for n in candidate.graph.node}; actual_inits = {t.name: t for t in candidate.graph.initializer}
    require(all(actual_nodes[n.name].SerializeToString() == n.SerializeToString() for n in before_nodes if n.name != node.name) and
            all(actual_inits[t.name].SerializeToString() == t.SerializeToString() for t in before_inits if t.name != name), 'Unrelated global math/parameter changed')
    inverse = copy.deepcopy(candidate)
    del inverse.graph.node[:]; inverse.graph.node.extend(n if n.name == node.name else actual_nodes[n.name] for n in before_nodes)
    del inverse.graph.initializer[:]; inverse.graph.initializer.extend(restored if t.name == name else actual_inits[t.name] for t in before_inits)
    require(inverse.SerializeToString() == model.SerializeToString(), 'Whole original global graph inverse protobuf failed')
    return candidate, dict(parts=4, complete_K=4096, K_ranges=[[i * 1024, (i + 1) * 1024] for i in range(4)],
        original_weight_concat_byte_equal=True, original_weight_initializer_inverse_proto_equal=True,
        original_global_model_inverse_proto_equal=True, original_bias_added_once=True, floating_reduction_order_changed=True)


def pair_proto_check(pair, parents, index):
    """Verify the frozen generic pair composer reverses every parent protobuf."""
    nodes = {n.name: n for n in pair.graph.node}; inits = {t.name: t for t in pair.graph.initializer}; nc = ic = 0
    for role, model in parents.items():
        prefix = f'{role}{index}/'; overrides = {'input': 'input' if role == 'frame' else 'global_input', 'output': role + '_output'}
        inverse_names = {v: k for k, v in overrides.items()}
        def inverse_name(v): return inverse_names.get(v, v[len(prefix):]) if v else v
        for original in model.graph.node:
            restored = copy.deepcopy(nodes[prefix + original.name]); restored.name = original.name
            restored.input[:] = [inverse_name(v) for v in restored.input]; restored.output[:] = [inverse_name(v) for v in restored.output]
            require(restored.SerializeToString() == original.SerializeToString(), 'Pair composer changed parent node'); nc += 1
        for original in model.graph.initializer:
            restored = copy.deepcopy(inits[prefix + original.name]); restored.name = original.name
            require(restored.SerializeToString() == original.SerializeToString(), 'Pair composer changed parent initializer'); ic += 1
    require(len(pair.graph.node) == nc + 1 and len(pair.graph.initializer) == ic + 1,
            'Only the original frame-to-global Reshape and its shape initializer may be added')
    return dict(parent_node_inverse_proto_checks=nc, parent_initializer_inverse_proto_checks=ic,
        pair_bridge_added_Reshape_count=1, all_parent_nodes_and_initializers_reversibly_preserved=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-capture', type=Path, required=True); p.add_argument('--native-capture-sha256', required=True)
    p.add_argument('--weights-capture', type=Path, default=BASE/'experiments/vggt-v53-capture-r1/capture.json')
    p.add_argument('--source-root', type=Path, default=BASE/'src'); p.add_argument('--group-index', type=int, choices=range(4), required=True)
    p.add_argument('--frame-layout', choices=('batch1', 'batch2'), default='batch1')
    p.add_argument('--global-fc2-parts', type=int, choices=(1, 4), default=4); p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args(); out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    watched = {}; start, end = RANGES[a.group_index]; indices = list(range(start, end + 1))
    report = dict(schema='v100_original_native518_retained_aggregator_group_ONNX_export', status='failed',
        script_sha256=sha(__file__), original_model_only=True, image_size=518, image_shape=IMAGE,
        group_index=a.group_index, indices=indices, all_group_ranges=[list(r) for r in RANGES],
        frame_layout=a.frame_layout, global_fc2_parts=a.global_fc2_parts,
        model_ORT_SDK_GPU_or_calibration_execution=False, candidate_accepted=False,
        FP32_equivalence_validated=False, full_model_or_realtime_validated=False,
        required_next_gate=dict(relative_l2_max=1e-5, max_abs_max=1e-3,
            scope='Complete group outputs versus corresponding native original CPU outputs; include raw global20 for the last group'))
    def pin(path, digest, size=None):
        path = Path(path).resolve(strict=True)
        require(isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest) and
                sha(path) == digest and (size is None or path.stat().st_size == size), 'Pinned input/source changed: ' + str(path))
        watched[path] = digest; return path
    def artifact(root, row):
        relative = Path(row['file']); require(not relative.is_absolute() and '..' not in relative.parts, 'Capture-owned relative artifact required')
        path = (root / relative).resolve(strict=True); require(path.is_relative_to(root), 'Artifact escaped capture directory')
        return pin(path, row['sha256'], row['bytes'])
    try:
        pin(__file__, report['script_sha256']); root = a.source_root.resolve(strict=True)
        gp = pin(root/'deploy/rk3588/validation/original_full_aggregator/original_block_graph.py', GRAPH_SHA)
        cp = pin(root/'deploy/rk3588/validation/original_retained_groups/compose_all_retained_groups.py', COMPOSER_SHA)
        G = load_module(gp, '_v100_original_native_block_static'); C = load_module(cp, '_v100_original_native_retained_static')
        C.FRAME, C.GLOBAL = list(FRAME), list(GLOBAL)  # Isolated static module; frozen source is unchanged.
        import numpy as np
        import onnx
        from onnx import helper as H, numpy_helper as N, TensorProto as T
        native_path = pin(a.native_capture, a.native_capture_sha256); native = json.loads(native_path.read_text())
        require(native['schema'] == 'v98_original_native518_block_reference' and native['status'] == 'completed_original_native518_block_references' and
                native['completed'] is True and native['checkpoint'] == G.CHECKPOINT and native['image_shape'] == IMAGE and native['cleanup_errors'] == [],
                'Clean original v98 native positions/reference capture required')
        weights_path = pin(a.weights_capture, WEIGHTS_SHA); captured = json.loads(weights_path.read_text())
        require(captured['checkpoint'] == G.CHECKPOINT and captured['status'] == 'completed_original_FP32_full_forward_and_48_block_capture', 'Original v53 parameter capture required')
        rows = {(r['role'], r['index']): r for r in captured['blocks']}
        require(len(rows) == 48 and set(rows) == {(role, i) for i in range(24) for role in ('frame', 'global')}, 'All original 48 parameter sets required')
        arrays, metadata = {}, {}
        for key, shape, dtype in [('positions', (2,1041,2), np.int64), ('global_positions', (1,2082,2), np.int64),
                                  ('rope_cos', (38,32), np.float32), ('rope_sin', (38,32), np.float32)]:
            row = native['artifacts'][key]; value = np.load(artifact(native_path.parent, row), allow_pickle=False)
            require(value.dtype == dtype and value.shape == shape and value.flags.c_contiguous and np.isfinite(value).all() and
                    all(raw_metadata(value, np)[k] == row[k] for k in ('shape','dtype','raw_sha256')), 'Original native position/RoPE array differs')
            arrays[key], metadata[key] = value, row
        require(np.array_equal(arrays['positions'], G.expected_frame_positions(tuple(IMAGE))) and
                np.array_equal(arrays['positions'][:1], arrays['positions'][1:]) and
                np.array_equal(arrays['positions'].reshape(1,2082,2), arrays['global_positions']), 'Original native token/position order differs')
        operator = root/'deploy/rk3588/validation/export-v17-aggregator-block-r3.py'
        for name, digest in G.OPERATOR_BUILDERS.items(): pin(operator.with_name(name), digest)
        builder = G.load_operator_builder(operator)
        require(all(native['source_hashes'].get(k) == v for k, v in builder.SOURCE_PINS.items()), 'Original layer source identity differs')
        models, parents = [], []
        for index in indices:
            role_models, role_proofs, parameters, positions = {}, {}, {}, {}
            for role in ('frame', 'global'):
                row = rows[role, index]
                with np.load(artifact(weights_path.parent, row['weights']), allow_pickle=False) as saved:
                    require(set(saved.files) == set(G.PARAMETER_NAMES) == set(row['parameters']), 'Exactly18 original parameters required')
                    weights = {n: np.ascontiguousarray(saved[n]) for n in saved.files}
                require(all(v.dtype == np.float32 and np.isfinite(v).all() and raw_metadata(v, np) == row['parameters'][n] for n, v in weights.items()) and
                        row['heads'] == 16 and row['gelu_approximate'] == 'none' and row['rope_frequency'] == 100 and row['rope_scaling_factor'] == 1.0 and
                        row['epsilons'] == {n:1e-5 for n in G.EPSILON_NAMES}, 'Original learned value or operator contract differs')
                if role == 'frame' and a.frame_layout == 'batch1':
                    template = builder.build_aggregator_onnx(weights, (1,1041,1024), arrays['positions'][:1], arrays['rope_cos'], arrays['rope_sin'], heads=16, epsilons=row['epsilons'])
                    parameter_check(template, weights, G, np, N)
                    model, proof = frame_batch1(template, np, onnx, H, N, T); del template
                else:
                    model = G.build_original_block_onnx(weights, role=role, index=index,
                        positions=arrays['positions' if role == 'frame' else 'global_positions'], rope_cos=arrays['rope_cos'], rope_sin=arrays['rope_sin'],
                        epsilons=row['epsilons'], exporter_path=operator, image_shape=tuple(IMAGE)); proof = dict(layout='unchanged_original_block')
                if role == 'global' and a.global_fc2_parts == 4:
                    model, kproof = global_k4(model, np, onnx, H, N); proof['FC2_K4'] = kproof
                role_models[role], role_proofs[role] = model, proof
                parameters[role] = {n:r['raw_sha256'] for n,r in row['parameters'].items()}
                positions[role] = metadata['positions' if role == 'frame' else 'global_positions']
                del weights
            pair = G.compose_frame_global_pair(role_models['frame'], role_models['global'], index=index)
            pproof = pair_proto_check(pair, role_models, index)
            parents.append(dict(role='pair', index=index, input_name='input', input_shape=FRAME,
                output_bindings=[dict(index=0,name='frame_output',shape=FRAME,dtype='float32'),dict(index=1,name='global_output',shape=GLOBAL,dtype='float32')],
                heads=16, all_36_original_parameter_values_preserved_by_parent_composition=True,
                parent_nodes_plus_one_reshape=True, original_aggregator_concat_must_be_retained=True,
                parameter_raw_sha256=parameters, positions=positions, role_transforms=role_proofs, pair_preservation=pproof))
            models.append(pair); del role_models, model, pair
            print(json.dumps(dict(stage='original_complete_pair_static_composed',index=index)),flush=True)
        group, proof = C.compose_retained_group(models, parents)
        path = out/f'retained-pair-group-{start:02d}-{end:02d}-native518.onnx'; onnx.save_model(group, path)
        graph = dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path))
        group_row = dict(role='retained_pair_group',index=a.group_index,indices=indices,input_name='input',input_shape=FRAME,
            output_bindings=C.output_bindings(indices),heads=16,parents=parents,
            parameter_raw_sha256={str(r['index']):r['parameter_raw_sha256'] for r in parents}, **graph, **proof)
        report.update(status=SUCCESS,checkpoint=G.CHECKPOINT,source_hashes=native['source_hashes'],
            original_weights_capture_sha256=WEIGHTS_SHA,native_capture_sha256=a.native_capture_sha256,
            graph_helper_sha256=GRAPH_SHA,retained_composer_sha256=COMPOSER_SHA,operator_source_sha256=G.OPERATOR_BUILDERS,
            original_parameter_value_count=36*len(indices),selected_block_count=2*len(indices),total_required_blocks=48,
            retained_consumer_indices=[4,11,17,23],original_target_tokens_global_index=20,
            original_target_token_clone_and_norm_must_be_retained=True,boundary_mode='fp32',
            native_position_metadata=metadata,original_24_pairs_required=True,blocks=[group_row],onnx=graph,
            compiled_internal_precision_not_proven=True,FP32_floating_reduction_order_changed=a.global_fc2_parts==4,
            original_token_preparation_all_other_original_modules_and_SL4_must_be_retained=True,
            export_on_host_CPU_is_static_only=True)
    except BaseException as exc:
        report.update(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    finally:
        errors = []
        for path, digest in watched.items():
            try: require(sha(path) == digest, 'Pinned input/source changed: '+str(path))
            except BaseException as exc: errors.append(str(exc))
        report.update(source_and_inputs_unchanged=not errors,verification_errors=errors,
                      source_and_input_hashes={str(p):d for p,d in watched.items()})
        if errors: report['status'] = 'failed'
        tmp=out/'export.json.tmp';tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(out/'export.json')
    print(json.dumps(dict(status=report['status'],onnx=report.get('onnx'),export_sha256=sha(out/'export.json'))),flush=True)
    return 0 if report['status'] == SUCCESS else 1


if __name__ == '__main__':
    raise SystemExit(main())
