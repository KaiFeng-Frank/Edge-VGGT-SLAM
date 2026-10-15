#!/usr/bin/env python3
"""Future R129 board-only joint FC1/Erf-GELU/FC2-K4 consumer-chain screen.

Export changes only the global0 MLP in the pinned original R126 pair-A.
Check compares complete canonical outputs with saved same-input R128 A-FP32
references. No SDK, other model, whole-network executor or speed claim.
"""
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time
import traceback

PARENT_ROOT = Path('/tmp/edge-vggt-v126-basis-export-r2')
PARENT_CHECK = Path('/tmp/edge-vggt-v126-basis-check-r2/result.json')
EXPORT_SHA = 'e23c8f7cdf6faa895a8bde34a41806bf2022e630f5361529d0271374742d740e'
CHECK_SHA = '48a7d5034b9eea0afa139e4a69cdb5b393c694dcec2b8e874a1c1f380f5f42b1'
HELPER_SHA = 'b0f4bd489d1e6ec8eb28869f609d7e48469aa9d19b2120d7906cd9284059bc4c'
PARENT_GRAPH_SHA = '57b60feceea0edb57ca4cea38200d6ea47d4e4ee51e4a546357267cbc9f676bb'
FRAME, GLOBAL = [2, 1041, 1024], [1, 2082, 1024]
GATE = dict(relative_l2_max=1e-5, max_abs_max=1e-3, rtol=1e-4, atol=1e-5)
EXPORT_SCHEMA = 'v129_original_native518_joint_MLP_pair_ONNX_export'
EXPORT_STATUS = 'exported_original_joint_MLP_pair_not_FP32_or_NPU_accepted'
CHECK_SCHEMA = 'v129_original_native518_joint_MLP_pair_FP32_check'
CHECK_STATUS = 'checked_original_joint_MLP_pair_fixed_FP32_gate_passed_not_NPU_accepted'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def raw(value):
    return hashlib.sha256(value.tobytes(order='C')).hexdigest()


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def parameter_values(model, parameters, np, onnx, joint=False):
    """Recover every original learned value; no checkpoint deserialization."""
    tensors = {t.name: t for t in model.graph.initializer}
    result = {}
    linears = {'attn.qkv.weight', 'attn.proj.weight', 'mlp.fc1.weight', 'mlp.fc2.weight'}
    for role in ('frame', 'global'):
        prefix = role + '0/'
        result[role] = {}
        for name, metadata in parameters[role].items():
            if joint and role == 'global' and name in ('mlp.fc1.weight', 'mlp.fc1.bias'):
                stem = prefix + 'joint_mlp_k4/part'
                key = 'fc1_weight' if name.endswith('weight') else 'fc1_bias'
                pieces = [onnx.numpy_helper.to_array(tensors[stem + str(i) + '/' + key]) for i in range(4)]
                value = np.concatenate(pieces, axis=1 if name.endswith('weight') else 0)
                if name.endswith('weight'):
                    value = value.T
            elif role == 'global' and name == 'mlp.fc2.weight':
                value = np.concatenate([onnx.numpy_helper.to_array(tensors[prefix + name + '_transposed.part' + str(i)]) for i in range(4)], axis=0).T
            else:
                value = onnx.numpy_helper.to_array(tensors[prefix + name + ('_transposed' if name in linears else '')])
                if name in linears:
                    value = value.T
            require(value.dtype == np.float32 and list(value.shape) == metadata['shape'] and
                    value.nbytes == metadata['tensor_bytes'] and np.isfinite(value).all() and
                    raw(value) == metadata['raw_sha256'], 'Original learned value differs: ' + prefix + name)
            result[role][name] = dict(shape=list(value.shape), dtype='float32', tensor_bytes=value.nbytes,
                raw_sha256=raw(value), finite=True)
    require(sum(len(v) for v in result.values()) == 36, 'Exactly 36 original learned values required')
    return result


def rewrite(parent, authority, np, onnx):
    original = parent.SerializeToString()
    nodes = list(parent.graph.node)
    initializers = list(parent.graph.initializer)
    by_name = {n.name: n for n in nodes}
    tensors = {t.name: t for t in initializers}
    require(len(by_name) == len(nodes) and len(tensors) == len(initializers), 'Unique original names required')
    before_parameters = parameter_values(parent, authority['original_parameters'], np, onnx)
    prefix = 'global0/'
    producer = {}
    consumers = {}
    for node in nodes:
        for name in node.output:
            require(name not in producer, 'Original value has multiple producers')
            producer[name] = node.name
        for name in node.input:
            consumers.setdefault(name, []).append(node.name)
    def expect(name, kind, inputs, outputs=None):
        node = by_name[name]
        require(node.op_type == kind and list(node.input) == inputs and
                list(node.output) == (outputs or [name]) and not node.attribute,
                'Exact original node pattern differs: ' + name)
        return node
    fc1_mm = expect(prefix + 'mlp.fc1.matmul', 'MatMul', [prefix + 'norm2', prefix + 'mlp.fc1.weight_transposed'])
    fc1_add = expect(prefix + 'mlp.fc1', 'Add', [fc1_mm.name, prefix + 'mlp.fc1.bias'])
    gelu = [
        expect(prefix + 'gelu_scaled_input', 'Div', [fc1_add.name, prefix + 'gelu_sqrt_two']),
        expect(prefix + 'gelu_erf', 'Erf', [prefix + 'gelu_scaled_input']),
        expect(prefix + 'gelu_erf_plus_one', 'Add', [prefix + 'gelu_erf', prefix + 'gelu_one']),
        expect(prefix + 'gelu_before_half', 'Mul', [fc1_add.name, prefix + 'gelu_erf_plus_one']),
        expect(prefix + 'gelu', 'Mul', [prefix + 'gelu_before_half', prefix + 'gelu_half'])]
    for name, value in (('gelu_sqrt_two', math.sqrt(2.)), ('gelu_one', 1.), ('gelu_half', .5)):
        constant = onnx.numpy_helper.to_array(tensors[prefix + name])
        require(constant.shape == () and constant.dtype == np.float32 and
                raw(constant) == raw(np.asarray(value, dtype=np.float32)), 'Original exact GELU scalar differs')
    require(consumers[prefix + 'mlp.fc1.weight_transposed'] == [fc1_mm.name] and
            consumers[prefix + 'mlp.fc1.bias'] == [fc1_add.name] and consumers[fc1_mm.name] == [fc1_add.name] and
            set(consumers[fc1_add.name]) == {gelu[0].name, gelu[3].name} and
            consumers[gelu[0].output[0]] == [gelu[1].name] and
            consumers[gelu[1].output[0]] == [gelu[2].name] and
            consumers[gelu[2].output[0]] == [gelu[3].name] and
            consumers[gelu[3].output[0]] == [gelu[4].name], 'Original hidden region has an unexpected consumer')
    slices = []
    removable_constants = set()
    part_outputs = []
    for part in range(4):
        activation = prefix + 'fc2_k_activation_' + str(part)
        names = [activation + '.starts', activation + '.ends', prefix + 'fc2_k_slice_axis', prefix + 'fc2_k_slice_step']
        sliced = expect(activation, 'Slice', [prefix + 'gelu'] + names)
        for name, expected in zip(names, ([part * 1024], [(part + 1) * 1024], [-1], [1])):
            value = onnx.numpy_helper.to_array(tensors[name])
            require(value.dtype == np.int64 and value.tolist() == expected, 'Original K4 range/axis/step differs')
        removable_constants.update(names)
        slices.append(sliced)
        mm_name = prefix + 'mlp.fc2.matmul.part' + str(part)
        expect(mm_name, 'MatMul', [activation, prefix + 'mlp.fc2.weight_transposed.part' + str(part)])
        require(consumers[activation] == [mm_name], 'Original K4 tile has another consumer')
        part_outputs.append(mm_name)
    require(set(consumers[prefix + 'gelu']) == {n.name for n in slices}, 'All four original hidden tiles required')
    summed = part_outputs[0]
    preserved_adds = []
    for part in range(1, 4):
        output = prefix + 'mlp.fc2.matmul' if part == 3 else prefix + 'fc2_k_sum_' + str(part)
        node_name = prefix + 'mlp.fc2.matmul.sum' + str(part)
        preserved_adds.append(expect(node_name, 'Add', [summed, part_outputs[part]], [output]))
        summed = output
    preserved_adds.append(expect(prefix + 'mlp.fc2', 'Add', [summed, prefix + 'mlp.fc2.bias']))
    weight = onnx.numpy_helper.to_array(tensors[prefix + 'mlp.fc1.weight_transposed'])
    bias = onnx.numpy_helper.to_array(tensors[prefix + 'mlp.fc1.bias'])
    require(weight.shape == (1024, 4096) and bias.shape == (4096,), 'Original full FC1 dimensions required')
    occupied = set(by_name) | set(tensors) | set(producer)
    new_nodes, new_tensors, new_names = [], [], set()
    column_pieces, bias_pieces, branches, branch_nodes = [], [], [], []
    for part in range(4):
        stem = prefix + 'joint_mlp_k4/part' + str(part) + '/'
        column_range = [part * 1024, (part + 1) * 1024]
        w = np.ascontiguousarray(weight[:, column_range[0]:column_range[1]])
        b = np.ascontiguousarray(bias[column_range[0]:column_range[1]])
        column_pieces.append(w); bias_pieces.append(b)
        wn, bn, mm, hidden = [stem + name for name in ('fc1_weight', 'fc1_bias', 'fc1_matmul', 'fc1')]
        new_tensors.extend([onnx.numpy_helper.from_array(w, wn), onnx.numpy_helper.from_array(b, bn)])
        mapping = {fc1_add.name: hidden}
        mapping.update({name: stem + name[len(prefix):] for node in gelu for name in node.output})
        mapping[gelu[-1].output[0]] = slices[part].output[0]
        branch = [onnx.helper.make_node('MatMul', [fc1_mm.input[0], wn], [mm], name=mm),
                  onnx.helper.make_node('Add', [mm, bn], [hidden], name=hidden)]
        for old in gelu:
            node = copy.deepcopy(old)
            node.name = stem + old.name[len(prefix):]
            node.input[:] = [mapping.get(name, name) for name in old.input]
            node.output[:] = [mapping[name] for name in old.output]
            branch.append(node)
        names = {wn, bn} | {n.name for n in branch} | {v for n in branch for v in n.output if v != slices[part].output[0]}
        require(not names.intersection(occupied | new_names), 'New branch name collides with original graph')
        new_names.update(names)
        new_nodes.extend(branch)
        branch_nodes.append(branch)
        branches.append(dict(index=part, original_hidden_channel_range=column_range,
            original_FC2_activation=slices[part].output[0], FC1_weight_raw_sha256=raw(w),
            FC1_bias_raw_sha256=raw(b), exact_original_GELU_node_names=[n.name for n in branch[2:]],
            original_FC2_MatMul_name=prefix + 'mlp.fc2.matmul.part' + str(part)))
    require(raw(np.concatenate(column_pieces, axis=1)) == raw(weight) and
            raw(np.concatenate(bias_pieces, axis=0)) == raw(bias), 'All original FC1 columns/bias bytes must reconstruct exactly')
    removed_nodes = {fc1_mm.name, fc1_add.name} | {n.name for n in gelu + slices}
    relocated_nodes = {prefix + 'mlp.fc2.matmul.part' + str(i) for i in range(4)} | {n.name for n in preserved_adds[:3]}
    interleaved = []
    for part, branch in enumerate(branch_nodes):
        interleaved.extend(branch)
        interleaved.append(copy.deepcopy(by_name[prefix + 'mlp.fc2.matmul.part' + str(part)]))
        if part:
            interleaved.append(copy.deepcopy(preserved_adds[part - 1]))
    candidate = copy.deepcopy(parent)
    changed = []
    for node in nodes:
        if node.name == fc1_mm.name:
            changed.extend(interleaved)
        elif node.name not in removed_nodes | relocated_nodes:
            changed.append(copy.deepcopy(node))
    del candidate.graph.node[:]
    candidate.graph.node.extend(changed)
    used = {name for node in changed for name in node.input} | {v.name for v in candidate.graph.output}
    removed_tensors = {prefix + 'mlp.fc1.weight_transposed', prefix + 'mlp.fc1.bias'} | {name for name in removable_constants if name not in used}
    require(all(name not in used for name in removed_tensors), 'A removed original initializer still has consumers')
    del candidate.graph.initializer[:]
    candidate.graph.initializer.extend([copy.deepcopy(t) for t in initializers if t.name not in removed_tensors] + new_tensors)
    del candidate.graph.value_info[:]
    candidate.graph.name += '__joint_global0_MLP_K4'
    candidate.producer_name = 'Edge-VGGT-SLAM/original-joint-MLP-board-v129'
    candidate.doc_string = 'Original pair-A; only global FC2K4 activation partition is moved through matching FC1 columns and the unchanged exact Erf GELU tree. Full original arithmetic; actual gates required.'
    onnx.checker.check_model(candidate, full_check=True)
    candidate = onnx.shape_inference.infer_shapes(candidate, strict_mode=True, check_type=True)
    onnx.checker.check_model(candidate, full_check=True)
    names = [n.name for n in candidate.graph.node]
    start = names.index(interleaved[0].name)
    require(names[start:start + len(interleaved)] == [n.name for n in interleaved], 'Requested source consumer-chain interleaving changed')
    untouched_nodes = {n.name: n.SerializeToString() for n in nodes if n.name not in removed_nodes}
    untouched_tensors = {t.name: t.SerializeToString() for t in initializers if t.name not in removed_tensors}
    require(untouched_nodes == {n.name: n.SerializeToString() for n in candidate.graph.node if n.name in untouched_nodes}, 'An untouched original node changed')
    require(untouched_tensors == {t.name: t.SerializeToString() for t in candidate.graph.initializer if t.name in untouched_tensors}, 'An untouched original initializer changed')
    require([v.SerializeToString() for v in (*parent.graph.input, *parent.graph.output)] ==
            [v.SerializeToString() for v in (*candidate.graph.input, *candidate.graph.output)], 'Original canonical IO changed')
    after_parameters = parameter_values(candidate, authority['original_parameters'], np, onnx, joint=True)
    require(after_parameters == before_parameters, 'Original 36 learned values changed')
    inverse = copy.deepcopy(candidate)
    kept_nodes = {n.name: n for n in candidate.graph.node if n.name in untouched_nodes}
    del inverse.graph.node[:]
    inverse.graph.node.extend([copy.deepcopy(by_name[n.name] if n.name in removed_nodes else kept_nodes[n.name]) for n in nodes])
    kept_tensors = {t.name: t for t in candidate.graph.initializer if t.name in untouched_tensors}
    del inverse.graph.initializer[:]
    inverse.graph.initializer.extend([copy.deepcopy(tensors[t.name] if t.name in removed_tensors else kept_tensors[t.name]) for t in initializers])
    del inverse.graph.value_info[:]
    inverse.graph.value_info.extend([copy.deepcopy(v) for v in parent.graph.value_info])
    inverse.graph.name, inverse.producer_name, inverse.doc_string = parent.graph.name, parent.producer_name, parent.doc_string
    require(inverse.SerializeToString() == original, 'Whole parent ModelProto inverse proof failed')
    proof = dict(original_nodes=len(nodes), candidate_nodes=len(candidate.graph.node), removed_node_count=len(removed_nodes),
        added_node_count=len(new_nodes), original_initializers=len(initializers), candidate_initializers=len(candidate.graph.initializer),
        branches=branches, all4096_original_hidden_neurons_covered_once=True,
        original_FC1_weight_columns_and_bias_reconstructed_bitwise=True, original_FC2_K4_weights_protobuf_unchanged=True,
        original_36_learned_values_verified=after_parameters, all36_original_learned_values_recovered_and_verified=True,
        all_frame0_nodes_and_initializers_protobuf_unchanged=True, all_untouched_nodes_and_initializers_protobuf_unchanged=True,
        original_three_partial_sum_Add_nodes_and_one_bias_Add_node_protobuf_unchanged=True,
        original_sum_order='((part0 + part1) + part2) + part3, then original bias once',
        relocated_original_node_names=sorted(relocated_nodes),
        explicit_source_consumer_chain_order=[n.name for n in interleaved] + [preserved_adds[-1].name],
        source_topological_interleaving_is_not_an_enforced_SDK_schedule=True,
        actual_SDK_schedule_or_memory_lifetime_validated=False,
        original_GELU_operator_and_operand_order_per_branch_unchanged=True, original_LayerScale_residual_norm_and_attention_unchanged=True,
        original_all1041_frame_tokens_and2082_global_tokens_preserved=True, original_canonical_IO_protobuf_unchanged=True,
        removed_nodes=sorted(removed_nodes), removed_initializer_names=sorted(removed_tensors),
        removed_initializers_are_only_FC1_full_weight_bias_and_now_unused_K4_slice_constants=True,
        hidden_Concat_or_full4096_GELU_materialization_node_added=False,
        dense_MACs_reduced=False, finite_precision_or_compiler_lifetime_equivalence_claimed=False,
        whole_parent_inverse_ModelProto_byte_equal=True, inverse_parent_ModelProto_sha256=hashlib.sha256(original).hexdigest())
    return candidate, proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('export', 'check'), required=True)
    p.add_argument('--reference-helper', type=Path, required=True, help='Frozen R128 check_captured_pair_fp32_board.py')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--export-receipt', type=Path)
    p.add_argument('--export-sha256')
    p.add_argument('--capture-receipt', type=Path)
    p.add_argument('--capture-sha256')
    p.add_argument('--fp32-reference-receipt', type=Path)
    p.add_argument('--fp32-reference-sha256')
    args = p.parse_args()
    sys.dont_write_bytecode = True
    require(platform.machine() == 'aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(), 'Board-only export/reference check; host execution forbidden')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    environment, affinity = dict(os.environ), set(os.sched_getaffinity(0))
    watched, handlers, session = {}, {}, None
    r = dict(schema=EXPORT_SCHEMA if args.mode == 'export' else CHECK_SCHEMA, mode=args.mode, status='running',
        script_sha256=sha(__file__), actual_execution_on_board_only=True, original_model_only=True,
        checkpoint_deserialized=False, RKNN_SDK_GPU_or_host_model_executed=False,
        candidate_accepted=False, full_network_trajectory_realtime_or_8GB_accepted=False,
        system_or_frequency_writes=False, fixed_gate=GATE, strict_allclose_is_independent_diagnostic=True,
        caller_owns_180_second_CPU_thermal_controller=True, models=[], checks=[], cleanup_errors=[],
        ORT_sessions_created=0, ORT_sessions_dropped_count=0)
    def progress(stage):
        r.update(stage=stage, elapsed_seconds=time.monotonic() - started,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        atomic(out / 'result.json', r)
        print(stage, flush=True)
    def pin(path, digest, size=None):
        path = Path(path).resolve(strict=True)
        require(isinstance(digest, str) and len(digest) == 64 and sha(path) == digest and
                (size is None or path.stat().st_size == size), 'Pinned source/artifact differs: ' + str(path))
        require(path not in watched or watched[path] == digest, 'Conflicting pin')
        watched[path] = digest
        return path
    def relative(directory, name):
        require(isinstance(name, str) and not Path(name).is_absolute() and '..' not in Path(name).parts, 'Receipt-relative artifact required')
        path = (directory / name).resolve(strict=True)
        require(path.is_relative_to(directory.resolve()), 'Artifact leaves receipt directory')
        return path
    def artifact(directory, row):
        return pin(relative(directory, row['file']), row['sha256'], row['bytes'])
    def tensor(path, row):
        value = np.load(path, allow_pickle=False)
        require(value.dtype == np.float32 and list(value.shape) == row['shape'] and value.nbytes == row['tensor_bytes'] and
                value.flags.c_contiguous and np.isfinite(value).all() and raw(value) == row['raw_sha256'], 'Saved same-input array differs')
        return value
    def compare(actual, reference):
        a, b = actual.astype(np.float64), reference.astype(np.float64)
        norm = float(np.linalg.norm(b.reshape(-1)))
        relative_l2 = float(np.linalg.norm((a - b).reshape(-1)) / norm) if norm > 0 else None
        maximum = float(np.max(np.abs(a - b)))
        outside = int(np.count_nonzero(np.abs(a - b) > GATE['atol'] + GATE['rtol'] * np.abs(b)))
        numeric = relative_l2 is not None and relative_l2 <= GATE['relative_l2_max'] and maximum <= GATE['max_abs_max']
        return dict(relative_l2=relative_l2, reference_norm=norm, zero_reference_norm=norm == 0,
            max_abs=maximum, outside_count=outside, strict_pass=outside == 0, byte_equal=actual.tobytes() == reference.tobytes(),
            fixed_numeric_gate_passed=numeric, fixed_gate_passed=numeric)
    progress('preflight')
    try:
        def interrupted(number, frame):
            raise InterruptedError('Controller termination signal ' + str(number))
        for number in (signal.SIGTERM, signal.SIGINT):
            handlers[number] = signal.signal(number, interrupted)
        pin(__file__, r['script_sha256'])
        helper_path = pin(args.reference_helper, HELPER_SHA)
        spec = importlib.util.spec_from_file_location('_v129_frozen_metadata_helper', helper_path)
        H = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(H)
        original_export = pin(PARENT_ROOT / 'result.json', EXPORT_SHA)
        original_check = pin(PARENT_CHECK, CHECK_SHA)
        authority, prior = json.loads(original_export.read_text()), json.loads(original_check.read_text())
        H.validate_original_pair(authority, prior)
        original_row = authority['models'][0]
        require(original_row['key'] == 'A' and original_row['onnx']['sha256'] == PARENT_GRAPH_SHA, 'Exact original pair-A required')
        parent_file = artifact(original_export.parent, original_row['onnx'])
        r.update(parent_export_receipt=dict(file=str(original_export), sha256=EXPORT_SHA),
            parent_check_receipt=dict(file=str(original_check), sha256=CHECK_SHA),
            parent_graph=copy.deepcopy(original_row['onnx']), checkpoint=authority['checkpoint'],
            source_hashes=authority['source_hashes'], original_parameters=authority['original_parameters'],
            reference_helper_sha256=HELPER_SHA)
        require({4, 5, 6, 7} <= affinity, 'A76 CPUs4..7 required')
        os.sched_setaffinity(0, {4, 5, 6, 7})
        os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
        import numpy as np
        if args.mode == 'export':
            import onnx
            parent = onnx.load(str(parent_file), load_external_data=False)
            require(parent.ir_version == 8 and len(parent.graph.input) == 1 and len(parent.graph.output) == 2 and
                    [(v.name, [d.dim_value for d in v.type.tensor_type.shape.dim], v.type.tensor_type.elem_type) for v in parent.graph.input] == [('input', FRAME, onnx.TensorProto.FLOAT)] and
                    [(v.name, [d.dim_value for d in v.type.tensor_type.shape.dim], v.type.tensor_type.elem_type) for v in parent.graph.output] ==
                    [('frame_output', FRAME, onnx.TensorProto.FLOAT), ('global_output', GLOBAL, onnx.TensorProto.FLOAT)] and
                    all(t.data_location != onnx.TensorProto.EXTERNAL and not t.external_data for t in parent.graph.initializer), 'Original self-contained canonical graph differs')
            candidate, proof = rewrite(parent, authority, np, onnx)
            graph = out / 'pair-joint-mlp.onnx'
            require(candidate.ByteSize() < 524288000, 'Candidate export storage bound exceeded')
            onnx.save_model(candidate, str(graph))
            row = dict(key='joint_mlp', onnx=dict(file=graph.name, bytes=graph.stat().st_size, sha256=sha(graph)),
                input_bindings=copy.deepcopy(original_row['input_bindings']), output_bindings=copy.deepcopy(original_row['output_bindings']))
            watched[graph] = row['onnx']['sha256']
            r.update(models=[row], proof=proof, onnx_version=onnx.__version__, numpy_version=np.__version__,
                ORT_execution=False, FP32_gate_passed=False, status=EXPORT_STATUS)
            del candidate, parent
            gc.collect()
        else:
            require(all(v is not None for v in (args.export_receipt, args.export_sha256, args.capture_receipt, args.capture_sha256,
                    args.fp32_reference_receipt, args.fp32_reference_sha256)), 'Check requires exact export, capture and matched A-FP32 receipt pins')
            export_file = pin(args.export_receipt, args.export_sha256)
            exported = json.loads(export_file.read_text())
            require(exported['schema'] == EXPORT_SCHEMA and exported['status'] == EXPORT_STATUS and
                    exported['script_sha256'] == r['script_sha256'] and exported['parent_export_receipt']['sha256'] == EXPORT_SHA and
                    exported['parent_check_receipt']['sha256'] == CHECK_SHA and exported['parent_graph'] == r['parent_graph'] and
                    exported['original_parameters'] == authority['original_parameters'] and exported['fixed_gate'] == GATE and
                    all(exported[k] for k in ('source_and_inputs_unchanged', 'affinity_restored', 'environment_restored')) and not exported['cleanup_errors'] and
                    all(exported['proof'][k] is True for k in ('all4096_original_hidden_neurons_covered_once',
                        'all36_original_learned_values_recovered_and_verified', 'all_frame0_nodes_and_initializers_protobuf_unchanged',
                        'all_untouched_nodes_and_initializers_protobuf_unchanged', 'whole_parent_inverse_ModelProto_byte_equal')),
                    'Clean complete same-source candidate export required')
            capture_file = pin(args.capture_receipt, args.capture_sha256)
            capture = json.loads(capture_file.read_text())
            H.validate_capture(capture)
            require(capture['source_hashes'] == authority['source_hashes'], 'Captured upstream source differs')
            for path, digest in capture['pinned_files'].items():
                pin(path, digest)
            reference_file = pin(args.fp32_reference_receipt, args.fp32_reference_sha256)
            references = json.loads(reference_file.read_text())
            require(references['schema'] == 'v128_original_native518_captured_pair_FP32_check' and
                    references['status'] == 'checked_captured_pair_fixed_FP32_gate_passed' and references['script_sha256'] == HELPER_SHA and
                    references['capture_receipt']['sha256'] == args.capture_sha256 and references['records'] == capture['records'] and
                    references['export_receipt']['sha256'] == EXPORT_SHA and references['original_check_receipt']['sha256'] == CHECK_SHA and
                    references['batch_index'] == capture['batch_index'] and references['models'] == authority['models'] and
                    references['original_parameters'] == authority['original_parameters'] and references['fixed_gate'] == GATE and
                    references['saved_array_count'] == 12 and references['sessions_created'] == references['sessions_dropped'] == 6 and
                    all(references[k] for k in ('FP32_gate_passed', 'all6_calls_completed', 'ORT_sessions_dropped',
                        'source_and_inputs_unchanged', 'affinity_restored', 'environment_restored')) and not references['cleanup_errors'],
                    'Matched original same-input A-FP32 reference receipt required')
            require(len(exported['models']) == 1 and exported['models'][0]['input_bindings'] == original_row['input_bindings'] and
                    exported['models'][0]['output_bindings'] == original_row['output_bindings'], 'Candidate canonical graph bindings differ')
            model = exported['models'][0]
            candidate_file = artifact(export_file.parent, model['onnx'])
            r.update(export_receipt=dict(file=str(export_file), sha256=args.export_sha256),
                capture_receipt=dict(file=str(capture_file), sha256=args.capture_sha256),
                original_FP32_reference_receipt=dict(file=str(reference_file), sha256=args.fp32_reference_sha256),
                batch_index=capture['batch_index'], models=exported['models'], proof=exported['proof'],
                reference_scope='Complete original pair-A endpoints conditioned on actual captured NPU DINO24 input; no complete Torch or trajectory reference')
            import onnxruntime as ort
            r.update(ort_version=ort.__version__, numpy_version=np.__version__, maximum_live_ORT_sessions=1)
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            for record in capture['records']:
                x = tensor(artifact(capture_file.parent, record['input']), record['input'])
                matches = [v for v in references['checks'] if v['input_id'] == record['id'] and v['variant'] == 'A']
                require(len(matches) == 1 and matches[0]['complete'] and matches[0]['input_unchanged'] and
                        len(matches[0]['outputs']) == 2, 'Original same-input A endpoint reference absent')
                expected = [tensor(artifact(reference_file.parent, v), v) for v in matches[0]['outputs']]
                require([v['name'] for v in matches[0]['outputs']] == ['frame_output', 'global_output'] and
                        [list(v.shape) for v in expected] == [FRAME, GLOBAL], 'Original reference order/shape differs')
                row = dict(input_id=record['id'], split=record['split'], source_pair_indices=record['source_pair_indices'], outputs=[], comparisons_vs_original_A={})
                r['checks'].append(row)
                try:
                    session = ort.InferenceSession(str(candidate_file), sess_options=options, providers=['CPUExecutionProvider'])
                    r['ORT_sessions_created'] += 1
                    require(session.get_providers() == ['CPUExecutionProvider'] and
                            [(v.name, v.shape, v.type) for v in session.get_inputs()] == [('input', FRAME, 'tensor(float)')] and
                            [(v.name, v.shape, v.type) for v in session.get_outputs()] ==
                            [('frame_output', FRAME, 'tensor(float)'), ('global_output', GLOBAL, 'tensor(float)')], 'Original CPU-only canonical session required')
                    begin = time.monotonic()
                    values = session.run(['frame_output', 'global_output'], {'input': x})
                    row['CPU_forward_seconds_not_NPU_performance'] = time.monotonic() - begin
                    require(len(values) == 2, 'Both canonical outputs required')
                    for name, shape, value, baseline in zip(('frame_output', 'global_output'), (FRAME, GLOBAL), values, expected):
                        value = np.array(value, copy=True, order='C')
                        file = out / (record['id'] + '-joint-' + name + '.npy')
                        with file.open('xb') as stream:
                            np.save(stream, value, allow_pickle=False)
                        metadata = dict(name=name, file=file.name, bytes=file.stat().st_size, sha256=sha(file),
                            shape=list(value.shape), dtype=str(value.dtype), tensor_bytes=value.nbytes,
                            raw_sha256=raw(value), finite=bool(np.isfinite(value).all()))
                        row['outputs'].append(metadata)
                        watched[file] = metadata['sha256']
                        require(value.dtype == np.float32 and list(value.shape) == shape and metadata['finite'], 'Candidate endpoint shape/dtype/finite differs')
                        row['comparisons_vs_original_A'][name] = compare(value, baseline)
                    require(raw(x) == record['input']['raw_sha256'], 'Captured input changed')
                    row.update(complete=True, input_unchanged=True,
                        fixed_gate_passed=all(v['fixed_gate_passed'] for v in row['comparisons_vs_original_A'].values()))
                    del values, value
                finally:
                    if session is not None:
                        session = None
                        gc.collect()
                        r['ORT_sessions_dropped_count'] += 1
                del x, expected
                progress(record['id'] + '_complete_saved')
            r.update(all2_calls_completed=len(r['checks']) == 2 and all(v['complete'] for v in r['checks']),
                FP32_gate_passed=all(v['fixed_gate_passed'] for v in r['checks']), ORT_execution=True)
            r['status'] = CHECK_STATUS if r['FP32_gate_passed'] else 'failed_original_joint_MLP_pair_fixed_FP32_gate'
    except BaseException as error:
        r.update(status='failed', error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        if session is not None:
            session = None
            gc.collect()
            r['ORT_sessions_dropped_count'] += 1
        try:
            os.sched_setaffinity(0, affinity)
        except BaseException as error:
            r['cleanup_errors'].append('affinity: ' + str(error))
        os.environ.clear(); os.environ.update(environment)
        for number, handler in handlers.items():
            signal.signal(number, handler)
        try:
            r['source_and_inputs_unchanged'] = all(path.is_file() and sha(path) == digest for path, digest in watched.items())
        except BaseException as error:
            r['source_and_inputs_unchanged'] = False
            r['cleanup_errors'].append('Final identity verification: ' + str(error))
        r.update(affinity_restored=set(os.sched_getaffinity(0)) == affinity, environment_restored=dict(os.environ) == environment,
            ORT_sessions_dropped=session is None and r['ORT_sessions_created'] == r['ORT_sessions_dropped_count'],
            pinned_files={str(path): digest for path, digest in watched.items()})
        if r['cleanup_errors'] or not all(r[k] for k in ('source_and_inputs_unchanged', 'affinity_restored', 'environment_restored', 'ORT_sessions_dropped')):
            r['status'] = 'failed_cleanup_or_identity'
        progress('finished')
    return 0 if r['status'] in (EXPORT_STATUS, CHECK_STATUS) else 1


if __name__ == '__main__':
    raise SystemExit(main())
