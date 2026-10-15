#!/usr/bin/env python3
"""Static board-only export of one complete original retained group in basis R.

Verify all original states, fold full blocks, then apply the frozen global FC2
K4 splitter. Namespace proofs concern folded parents, not checkpoint bytes.
Only group0 adds the initial basis transform. Group3 restores raw global20 on
a separate output branch while pair21 continues to consume its basis value.
No Torch forward, ORT, RKNN, calibration or accelerator execution is requested.
"""
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import sys
import time
import traceback

BASE = Path('/mnt/vggt-data/vggt-slam2')
E = BASE / 'experiments'
FRAME, GLOBAL, IMAGE = [2, 1041, 1024], [1, 2082, 1024], [2, 3, 392, 518]
RANGES = ((0, 4), (5, 11), (12, 17), (18, 23))
V100_SHA = 'b0ac4e8070db7f0cfdd3f2713e2ce43c28fc7847b8da79e64cc6b73108f5c269'
BASIS_SHA = '73affeb0b78a794ed214cadcf0307d164ef74f79d52b779bc579e906a1089b70'
GRAPH_SHA = '3ff8b0d0515ff44ea3625bd2c6529d63a0bed3487d409128ee4b443922a14bfd'
COMPOSER_SHA = 'e6fa508e0b03a011a1ccb484f2a46847ffd49fc1bf6ae11ecfeb8bd45f835f60'
WEIGHTS_SHA = '4b723e499763bb90565f6e11e16c551664634aaf72da46c7cdc493b84208ef67'
NATIVE_SHA = '4ebbc4adaaed5e3c3cf8558b4947ec3fc090d56e82f8c61c9509951eb34dd35d'
CHECKPOINT = dict(bytes=5026874952, sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')
SUCCESS = 'exported_original_basis_group_not_numerically_accepted'


def require(ok, message):
    if not ok: raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''): h.update(block)
    return h.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def proto_digest(values):
    """Stream one protobuf at a time, without a full-group serialized copy."""
    h = hashlib.sha256()
    for value in values: h.update(value.SerializeToString())
    return h.hexdigest()


def fold_full_block(model, state, branch_count, basis, np, onnx, V):
    """Fold an unsplit full-FC2 block; explicit frame2/global1 site counts."""
    require(branch_count in (1, 2) and len(state) == 18, 'Explicit original branch/parameter count required')
    original_tensors = {t.name: t for t in model.graph.initializer}
    require(len(original_tensors) == len(model.graph.initializer) and 'mlp.fc2.weight_transposed' in original_tensors,
        'Fold must precede global K4 splitting')
    updated = copy.deepcopy(model); replacements = {}
    for norm, linear in (('norm1', 'attn.qkv'), ('norm2', 'mlp.fc1')):
        weight = state[linear + '.weight'].T.astype(np.float64)
        scaled = state[norm + '.weight'].astype(np.float64)[:, None] * weight
        replacements[linear + '.weight_transposed'] = np.ascontiguousarray(basis.rotate64(scaled, 0, np), dtype=np.float32)
        replacements[linear + '.bias'] = np.ascontiguousarray(state[norm + '.bias'].astype(np.float64) @ weight + state[linear + '.bias'].astype(np.float64), dtype=np.float32)
    for linear, scale in (('attn.proj', 'ls1.gamma'), ('mlp.fc2', 'ls2.gamma')):
        scaled = state[linear + '.weight'].T.astype(np.float64) * state[scale].astype(np.float64)[None, :]
        replacements[linear + '.weight_transposed'] = np.ascontiguousarray(basis.rotate64(scaled, 1, np), dtype=np.float32)
        bias = state[linear + '.bias'].astype(np.float64) * state[scale].astype(np.float64)
        replacements[linear + '.bias'] = np.ascontiguousarray(basis.rotate64(bias, 0, np), dtype=np.float32)
    require(len(replacements) == 8 and set(replacements) <= set(original_tensors) and all(np.isfinite(v).all() for v in replacements.values()),
        'Complete finite original full-block fold required')
    for tensor in updated.graph.initializer:
        if tensor.name in replacements:
            tensor.CopyFrom(onnx.numpy_helper.from_array(replacements[tensor.name], tensor.name))
    unit, zero = '__v130_bare_norm_unit', '__v130_bare_norm_zero'
    require(unit not in original_tensors and zero not in original_tensors, 'Bare norm namespace collision')
    updated.graph.initializer.extend([onnx.numpy_helper.from_array(np.ones(1024, np.float32), unit),
        onnx.numpy_helper.from_array(np.zeros(1024, np.float32), zero)])
    changed, norms, scales = [], 0, 0
    for node in updated.graph.node:
        if node.op_type == 'LayerNormalization' and list(node.input[1:]) in (['norm1.weight', 'norm1.bias'], ['norm2.weight', 'norm2.bias']):
            attributes = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            require(attributes['axis'] == -1 and abs(attributes['epsilon'] - 1e-5) < 1e-11, 'Original norm axis/epsilon differs')
            node.input[1:] = [unit, zero]; norms += 1; changed.append(node.name)
        if node.op_type == 'Mul' and len(node.input) == 2 and node.input[1] in ('ls1.gamma', 'ls2.gamma'):
            require(not node.attribute, 'Original LayerScale attributes differ')
            node.op_type = 'Identity'; del node.input[1:]; scales += 1; changed.append(node.name)
    require(norms == scales == 2 * branch_count and len(changed) == 4 * branch_count, 'All original norm/scale fold sites required')
    original_nodes = {n.name: n for n in model.graph.node}
    require(len(original_nodes) == len(model.graph.node), 'Duplicate original node')
    require(all(n.SerializeToString() == original_nodes[n.name].SerializeToString() for n in updated.graph.node if n.name not in changed),
        'Attention/QK norms/RoPE/Erf/residual or other original node changed')
    require(all(t.SerializeToString() == original_tensors[t.name].SerializeToString() for t in updated.graph.initializer if t.name in original_tensors and t.name not in replacements),
        'Unrelated original initializer changed')
    # Only one block is rolled back at a time; never duplicate a full group.
    inverse = copy.deepcopy(updated)
    for node in inverse.graph.node:
        if node.name in changed: node.CopyFrom(original_nodes[node.name])
    restored = []
    for tensor in inverse.graph.initializer:
        if tensor.name in (unit, zero): continue
        restored.append(original_tensors[tensor.name] if tensor.name in replacements else tensor)
    del inverse.graph.initializer[:]; inverse.graph.initializer.extend(restored)
    require(inverse.SerializeToString() == model.SerializeToString(), 'Per-block original edit rollback protobuf failed')
    onnx.checker.check_model(updated, full_check=True)
    proof = dict(branch_count=branch_count, original_18_parameter_metadata={n: V.raw_metadata(v, np) for n, v in state.items()},
        derived_initializer_metadata={n: dict(V.raw_metadata(v, np), finite=True) for n, v in replacements.items()},
        changed_node_names=changed, bare_norm_sites=norms, moved_LayerScale_sites=scales,
        all_other_original_nodes_and_initializers_protobuf_unchanged=True,
        inverse_edit_with_original_initializers_recovers_original_block_protobuf=True,
        inverse_is_edit_rollback_not_recovery_of_original_weights_from_rounded_fold=True,
        original_initializer_byte_preservation_claimed=False, finite_precision_equivalence_claimed=False,
        original_full_FC2_present_before_partition=True, unused_original_affine_and_scale_vectors_retained=True)
    del inverse, restored, replacements, weight, scaled, bias
    return updated, proof


def apply_group_boundaries(group, group_index, basis, np, onnx):
    """In-place boundary edits; stream inverse checks instead of copying group."""
    H, N = onnx.helper, onnx.numpy_helper
    before_nodes = proto_digest(group.graph.node)
    before_infos = proto_digest(group.graph.value_info)
    initializer_pins = {t.name: hashlib.sha256(t.SerializeToString()).hexdigest() for t in group.graph.initializer}
    prefix = '__v130_boundary_'
    occupied = {t.name for t in group.graph.initializer} | {v for n in group.graph.node for v in (*n.input, *n.output) if v}
    require(not any(v.startswith(prefix) for v in occupied), 'Boundary namespace collision')
    mappings, added_names = {}, []
    rname = prefix + 'R'; rin = prefix + 'input_basis'; raw20 = prefix + 'global20_basis'
    need_r = group_index in (0, 3)
    if need_r:
        rotation = np.ascontiguousarray(basis.rotate64(np.eye(1024, dtype=np.float64), 1, np), dtype=np.float32)
        require(np.array_equal(rotation, rotation.T) and np.array_equal(rotation.sum(axis=1), np.ones(1024, np.float32)) and
            set(np.unique(rotation)) == {-0.03125, 0.03125}, 'Exact mean-preserving R1024 construction differs')
        group.graph.initializer.append(N.from_array(rotation, rname)); del rotation
    if group_index == 0:
        for node in group.graph.node:
            node.input[:] = [rin if v == 'input' else v for v in node.input]
        old_nodes = list(group.graph.node)
        name = prefix + 'initial_R'
        del group.graph.node[:]
        group.graph.node.extend([H.make_node('MatMul', ['input', rname], [rin], name=name)] + old_nodes)
        mappings[rin] = 'input'; added_names.append(name); del old_nodes
    if group_index == 3:
        bridge = [n for n in group.graph.node if n.name == 'global20_to_frame21_original_view']
        require(len(bridge) == 1 and bridge[0].input[0] == 'target_global_output', 'Original global20-to-pair21 dependency differs')
        require([v.name for v in group.graph.output] == ['frame_output', 'global_output', 'target_global_output'], 'Original ninth output binding differs')
        for node in group.graph.node:
            node.input[:] = [raw20 if v == 'target_global_output' else v for v in node.input]
            node.output[:] = [raw20 if v == 'target_global_output' else v for v in node.output]
        for info in group.graph.value_info:
            if info.name == 'target_global_output': info.name = raw20
        name = prefix + 'canonical_global20'
        group.graph.node.append(H.make_node('MatMul', [raw20, rname], ['target_global_output'], name=name))
        mappings[raw20] = 'target_global_output'; added_names.append(name)
        require(bridge[0].input[0] == raw20 and not any('target_global_output' in n.input for n in group.graph.node),
            'Canonical target output must not feed any next pair')
    recovered = hashlib.sha256()
    for node in group.graph.node:
        if node.name in added_names: continue
        inverse = copy.deepcopy(node)
        inverse.input[:] = [mappings.get(v, v) for v in inverse.input]
        inverse.output[:] = [mappings.get(v, v) for v in inverse.output]
        recovered.update(inverse.SerializeToString())
    recovered_infos = hashlib.sha256()
    for info in group.graph.value_info:
        inverse = copy.deepcopy(info); inverse.name = mappings.get(inverse.name, inverse.name)
        recovered_infos.update(inverse.SerializeToString())
    require(recovered.hexdigest() == before_nodes and recovered_infos.hexdigest() == before_infos and
        all(hashlib.sha256(t.SerializeToString()).hexdigest() == initializer_pins[t.name] for t in group.graph.initializer if t.name != rname) and
        len(group.graph.initializer) == len(initializer_pins) + int(need_r), 'Scoped boundary inverse node/initializer proof failed')
    group.producer_name = 'Edge-VGGT-SLAM/v130-original-retained-residual-basis'
    group.doc_string = 'Original complete selected frame/global pairs with algebraically derived R-basis linear weights. Basis retained exits, canonical raw global20 sidebranch. FP32/runtime/quality not accepted.'
    onnx.checker.check_model(group, full_check=True)
    return dict(initial_basis_operator_count=int(group_index == 0), canonical_target_inverse_operator_count=int(group_index == 3),
        intergroup_inverse_operator_count=0, inverse_node_and_value_info_stream_digests_equal=True,
        all_existing_folded_group_initializer_protobufs_unchanged=True,
        group3_pair21_consumes_basis_global20_not_canonical_target=True if group_index == 3 else None,
        final_frame_and_global_outputs_remain_basis=True, raw_target_is_unnormalized=True if group_index == 3 else None,
        full_group_inverse_copy_or_full_group_serialized_proof_buffer_used=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-exporter-source', type=Path, required=True)
    p.add_argument('--basis-source', type=Path, required=True)
    p.add_argument('--native-capture', type=Path, default=E / 'vggt-v98-reference-r1/capture.json')
    p.add_argument('--native-capture-sha256', default=NATIVE_SHA)
    p.add_argument('--weights-capture', type=Path, default=E / 'vggt-v53-capture-r1/capture.json')
    p.add_argument('--source-root', type=Path, default=BASE / 'src')
    p.add_argument('--group-index', type=int, choices=range(4), required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args(); sys.dont_write_bytecode = True
    require(platform.machine() == 'aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(), 'RK3588 board-only static exporter')
    require(a.native_capture_sha256 == NATIVE_SHA, 'Exact original native positions/capture required')
    out = a.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    start, end = RANGES[a.group_index]; indices = list(range(start, end + 1))
    environment, affinity, handlers, watched = dict(os.environ), set(os.sched_getaffinity(0)), {}, {}
    began = time.monotonic()
    report = dict(schema='v130_original_native518_retained_basis_group_export', status='running', script_sha256=sha(__file__),
        group_index=a.group_index, indices=indices, checkpoint=CHECKPOINT, image_size=518, image_shape=IMAGE,
        input_coordinates='canonical' if a.group_index == 0 else 'basis_R',
        output_coordinates=dict(frame_output='basis_R', global_output='basis_R', **({'target_global_output': 'canonical'} if a.group_index == 3 else {})),
        original_parameter_value_count=36 * len(indices), selected_block_count=2 * len(indices), total_required_blocks=48,
        original_parameter_count_unit='tensors', original_weights_capture_sha256=WEIGHTS_SHA, native_capture_sha256=NATIVE_SHA,
        fixed_gate=dict(relative_l2_max=1e-5, max_abs_max=1e-3), cleanup_errors=[], proofs={},
        candidate_accepted=False, NN_ORT_RKNN_GPU_or_calibration_execution=False, checkpoint_deserialized=False,
        original_initializer_byte_preservation_claimed=False, finite_precision_equivalence_claimed=False,
        all_24_pairs_required_for_full_chain=True, system_or_frequency_writes=False)

    def progress(stage):
        report.update(stage=stage, seconds=time.monotonic() - began)
        tmp = out / 'result.tmp'; tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n'); tmp.replace(out / 'result.json')
        print(stage, flush=True)

    def pin(path, digest, size=None):
        path = Path(path).resolve(strict=True)
        require(len(digest) == 64 and sha(path) == digest and (size is None or path.stat().st_size == size), 'Pinned file differs: ' + str(path))
        require(path not in watched or watched[path] == digest, 'Conflicting pinned file')
        watched[path] = digest; return path

    def artifact(folder, row):
        name = Path(row['file']); require(not name.is_absolute() and '..' not in name.parts, 'Owned relative artifact required')
        path = (folder / name).resolve(strict=True); require(path.is_relative_to(folder.resolve()), 'Artifact escapes capture')
        return pin(path, row['sha256'], row['bytes'])

    def interrupted(number, frame): raise InterruptedError('External CPU controller signal ' + str(number))

    watched[Path(__file__).resolve()] = report['script_sha256']; progress('preflight')
    try:
        for number in (signal.SIGTERM, signal.SIGINT): handlers[number] = signal.signal(number, interrupted)
        require({4, 5, 6, 7} <= affinity, 'A76 CPU affinity required'); os.sched_setaffinity(0, {4, 5, 6, 7})
        os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        V = load_module(pin(a.native_exporter_source, V100_SHA), '_v130_frozen_native_export')
        basis = load_module(pin(a.basis_source, BASIS_SHA), '_v130_frozen_basis_identities')
        root = a.source_root.resolve(strict=True)
        G = load_module(pin(root / 'deploy/rk3588/validation/original_full_aggregator/original_block_graph.py', GRAPH_SHA), '_v130_original_native_blocks')
        C = load_module(pin(root / 'deploy/rk3588/validation/original_retained_groups/compose_all_retained_groups.py', COMPOSER_SHA), '_v130_retained_composition')
        C.FRAME, C.GLOBAL = list(FRAME), list(GLOBAL)
        native_path = pin(a.native_capture, NATIVE_SHA); native = json.loads(native_path.read_text())
        weights_path = pin(a.weights_capture, WEIGHTS_SHA); captured = json.loads(weights_path.read_text())
        require(native['schema'] == 'v98_original_native518_block_reference' and native['status'] == 'completed_original_native518_block_references' and
            native['completed'] and native['checkpoint'] == captured['checkpoint'] == G.CHECKPOINT == CHECKPOINT and native['image_shape'] == IMAGE and
            not native['cleanup_errors'] and captured['status'] == 'completed_original_FP32_full_forward_and_48_block_capture', 'Original native/checkpoint/48-state lineage differs')
        rows = {(r['role'], r['index']): r for r in captured['blocks']}
        require(set(rows) == {(role, i) for i in range(24) for role in ('frame', 'global')} and len(captured['blocks']) == 48, 'All original 48 states required')
        required_bytes = sum(rows[role, i]['weights']['bytes'] for i in indices for role in ('frame', 'global')) + (128 << 20)
        require(shutil.disk_usage(out).free >= required_bytes, 'Free scratch space insufficient for one complete group')
        import numpy as np
        import onnx
        from onnx import helper as H, numpy_helper as N, TensorProto as T
        arrays, position_meta = {}, {}
        for name, shape, dtype in [('positions', (2, 1041, 2), np.int64), ('global_positions', (1, 2082, 2), np.int64),
                ('rope_cos', (38, 32), np.float32), ('rope_sin', (38, 32), np.float32)]:
            row = native['artifacts'][name]; value = np.load(artifact(native_path.parent, row), allow_pickle=False)
            require(value.shape == shape and value.dtype == dtype and value.flags.c_contiguous and np.isfinite(value).all() and
                all(V.raw_metadata(value, np)[k] == row[k] for k in ('shape', 'dtype', 'raw_sha256')), 'Actual native position/RoPE bytes differ')
            arrays[name], position_meta[name] = value, row
        require(np.array_equal(arrays['positions'], G.expected_frame_positions(tuple(IMAGE))) and
            np.array_equal(arrays['positions'][:1], arrays['positions'][1:]) and
            np.array_equal(arrays['positions'].reshape(1, 2082, 2), arrays['global_positions']), 'Original native token order differs')
        operator = root / 'deploy/rk3588/validation/export-v17-aggregator-block-r3.py'
        for name, digest in G.OPERATOR_BUILDERS.items(): pin(operator.with_name(name), digest)
        builder = G.load_operator_builder(operator)
        require(all(native['source_hashes'].get(k) == digest for k, digest in builder.SOURCE_PINS.items()), 'Original layer source pins differ')
        report.update(source_hashes=native['source_hashes'], native_position_metadata=position_meta,
            native_exporter_source_sha256=V100_SHA, basis_source_sha256=BASIS_SHA, graph_helper_sha256=GRAPH_SHA,
            retained_composer_sha256=COMPOSER_SHA, operator_source_sha256=G.OPERATOR_BUILDERS)
        models, parent_rows, block_proofs = [], [], []
        for index in indices:
            role_models, role_proofs, parameters = {}, {}, {}
            for role in ('frame', 'global'):
                row = rows[role, index]
                with np.load(artifact(weights_path.parent, row['weights']), allow_pickle=False) as saved:
                    require(set(saved.files) == set(G.PARAMETER_NAMES) == set(row['parameters']), 'Exactly18 original learned tensors required')
                    state = {n: np.ascontiguousarray(saved[n]) for n in saved.files}
                require(all(v.dtype == np.float32 and np.isfinite(v).all() and V.raw_metadata(v, np) == row['parameters'][n] for n, v in state.items()) and
                    row['heads'] == 16 and row['gelu_approximate'] == 'none' and row['rope_frequency'] == 100 and row['rope_scaling_factor'] == 1.0 and
                    row['epsilons'] == {n: 1e-5 for n in G.EPSILON_NAMES}, 'Original learned value/operator contract differs')
                if role == 'frame':
                    template = builder.build_aggregator_onnx(state, (1, 1041, 1024), arrays['positions'][:1], arrays['rope_cos'], arrays['rope_sin'], heads=16, epsilons=row['epsilons'])
                    V.parameter_check(template, state, G, np, N)
                    original, frame_proof = V.frame_batch1(template, np, onnx, H, N, T); del template
                else:
                    original = G.build_original_block_onnx(state, role=role, index=index, positions=arrays['global_positions'],
                        rope_cos=arrays['rope_cos'], rope_sin=arrays['rope_sin'], epsilons=row['epsilons'], exporter_path=operator, image_shape=tuple(IMAGE))
                    V.parameter_check(original, state, G, np, N); frame_proof = None
                folded, fold_proof = fold_full_block(original, state, 2 if role == 'frame' else 1, basis, np, onnx, V)
                del original, state
                kproof = None
                if role == 'global':
                    partitioned, kproof = V.global_k4(folded, np, onnx, H, N); del folded; folded = partitioned; del partitioned
                role_models[role] = folded
                role_proofs[role] = dict(fold=fold_proof, original_B1_parent_proof=frame_proof,
                    K4_proof_against_folded_full_FC2_parent=kproof)
                parameters[role] = {n: meta['raw_sha256'] for n, meta in row['parameters'].items()}
                del folded
            pair = G.compose_frame_global_pair(role_models['frame'], role_models['global'], index=index)
            pair_proof = V.pair_proto_check(pair, role_models, index)
            # The frozen composer receives already-folded parents. Its raw
            # checkpoint byte claims are not copied into our exported proof.
            parent_rows.append(dict(role='pair', index=index, input_name='input', input_shape=FRAME,
                output_bindings=[dict(index=0, name='frame_output', shape=FRAME, dtype='float32'), dict(index=1, name='global_output', shape=GLOBAL, dtype='float32')],
                heads=16, all_36_original_parameter_values_preserved_by_parent_composition=True, parent_nodes_plus_one_reshape=True,
                original_aggregator_concat_must_be_retained=True, parameter_raw_sha256=parameters))
            models.append(pair)
            block_proofs.append(dict(index=index, original_state_raw_sha256=parameters, roles=role_proofs,
                pair_composition_inverse_proof_against_folded_parents=pair_proof))
            del role_models, pair; gc.collect(); progress('folded_complete_original_pair_' + str(index))
        group, composition = C.compose_retained_group(models, parent_rows)
        del models, parent_rows; gc.collect()
        composed_proof = dict(proof_scope='Namespace copying of folded parent graphs, not original checkpoint graph protobufs',
            per_pair_inverse_proofs=[dict(index=v['index'], folded_parent_node_count=v['original_node_count'],
                folded_parent_initializer_count=v['original_initializer_count'], folded_parent_nodes_sha256=v['original_nodes_sha256'],
                folded_parent_initializers_sha256=v['original_initializers_sha256'], inverse_namespace_protobuf_equal=v['inverse_namespace_recovers_every_node_and_initializer_protobuf_bytewise'])
                for v in composition['parent_preservation_proofs']],
            added_interpair_reshape_count=composition['added_reshape_count'], original_36_tensor_states_verified_per_pair=True,
            checkpoint_initializer_byte_preservation_claimed=False, original_FP32_numeric_equivalence_not_established=True)
        boundaries = apply_group_boundaries(group, a.group_index, basis, np, onnx)
        bindings = C.output_bindings(indices)
        require([v.name for v in group.graph.output] == [v['name'] for v in bindings] and
            len(group.graph.output) == (3 if a.group_index == 3 else 2), 'Complete retained/target ABI differs')
        path = out / f'retained-basis-group-{start:02d}-{end:02d}-native518.onnx'
        onnx.save(group, str(path))
        report.update(status=SUCCESS, onnx=dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path)),
            input_bindings=[dict(index=0, name='input', shape=FRAME, dtype='float32')], output_bindings=bindings,
            proofs=dict(block_folds=block_proofs, folded_parent_composition=composed_proof, boundaries=boundaries),
            basis=dict(R='((J4-2I4)/2) tensor-power5', residual_features=1024, retained_coordinates='diag(R,R)',
                offline_fold_dtype='float64 then float32', no_attention_or_Erf_neuron_rotation=True),
            retained_consumer_indices=[4, 11, 17, 23], original_target_tokens_global_index=20,
            original_target_clone_and_L2_normalization_remain_CPU=True, raw_global20_target_is_canonical_and_unnormalized=a.group_index == 3,
            next_required_gate='Conditional same-input original group FP32 gate; then actual connected four-group/consumer/precision/trajectory/complete-boundary checks')
        del group
    except BaseException as error:
        report.update(status='failed_preserved_available_evidence', error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc()); traceback.print_exc()
    finally:
        try: report['source_and_inputs_unchanged'] = all(p.exists() and sha(p) == digest for p, digest in watched.items())
        except BaseException as error: report['source_and_inputs_unchanged'] = False; report['cleanup_errors'].append('Identity: ' + str(error))
        try: os.sched_setaffinity(0, affinity)
        except BaseException as error: report['cleanup_errors'].append('Affinity: ' + str(error))
        os.environ.clear(); os.environ.update(environment)
        for number, handler in handlers.items(): signal.signal(number, handler)
        report.update(environment_restored=dict(os.environ) == environment, affinity_restored=set(os.sched_getaffinity(0)) == affinity,
            source_and_input_hashes={str(p): digest for p, digest in watched.items()})
        if report['cleanup_errors'] or not all(report[k] for k in ('source_and_inputs_unchanged', 'environment_restored', 'affinity_restored')):
            report['status'] = 'failed_cleanup_or_identity'
        progress('finished')
    return 0 if report['status'] == SUCCESS else 1


if __name__ == '__main__': raise SystemExit(main())
