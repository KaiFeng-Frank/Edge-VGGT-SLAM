#!/usr/bin/env python3
"""Read the frozen original VGGT ONNX and saved precision/profile metadata.

Uses only a SHA-pinned pure ONNX protobuf definition, NumPy saved-byte views and
Python's standard library. Never import a real model, ORT, RKNN, or compiler.
The full report is private architecture/hash metadata, never initializer values.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import numpy as np

from v50_original_mlp_scope_contracts import (ATOL, CHECKPOINT_SHA, MLP_SUFFIXES,
    PAIR_EXPORT_SHA, PAIR_GRAPH_SHA, PROTOBUF_SOURCE_SHA, ROLES, RTOL,
    pin_graph, require, sha)

COMPILE_AUDIT_SHA = '8362d1e3fc702c733a451aebb87f78c733e076da57152c254fb2b1ebdb5136b7'
PROFILE_AUDIT_SHA = 'f66a22c838c12d725d8663ef45d9e7b5da07dcb842708f2c38874aadabbe9c58'
API_SOURCE_SHA = '63e4506fb851214e79f933f9de0d27dab04ecf93f6f46959c07c5787433fd05e'
EXPECTED_OPS = {'LayerNormalization': 8, 'MatMul': 12, 'Add': 18, 'Reshape': 5,
                'Transpose': 6, 'Gather': 10, 'Mul': 22, 'Softmax': 2, 'Div': 2, 'Erf': 2}
ATTR_GRAPH_TYPES = (5, 10)


def read_pinned(path, expected, pins):
    path = Path(path).resolve(strict=True)
    require(sha(path) == expected, 'Saved source/receipt SHA mismatch: ' + str(path))
    pins[str(path)] = expected
    return path


def load_pb(path, pins):
    path = read_pinned(path, PROTOBUF_SOURCE_SHA, pins)
    spec = importlib.util.spec_from_file_location('_v50_original_pair_pure_proto', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encoded_sha(message):
    return hashlib.sha256(message.SerializeToString()).hexdigest()


def tensor_shape(value):
    tensor = value.type.tensor_type
    require(tensor.elem_type == 1, 'Original activation must be FP32')
    dims = [dim.dim_value for dim in tensor.shape.dim]
    require(dims and all(x > 0 for x in dims), 'Expected static positive activation dimensions')
    return dims


def classify(name):
    if name == 'frame_to_global_view':
        return 'original_frame_to_global_C_order_view'
    role, suffix = name.split('/', 1)
    require(role in ROLES, 'Unexpected namespace')
    if suffix in MLP_SUFFIXES:
        return 'complete_original_MLP'
    if suffix in ('norm1', 'norm2', 'attn.q_norm', 'attn.k_norm'):
        return 'original_normalization_outside_MLP'
    if suffix.startswith(('query_rope', 'key_rope')):
        return 'original_RoPE_outside_MLP'
    if suffix in ('ls1', 'ls2', 'residual1', 'output'):
        return 'original_LayerScale_or_residual_outside_MLP'
    return 'original_attention_or_layout_outside_MLP'


def inspect_graph(model, exported):
    graph = model.graph
    require(model.ir_version == 8 and [(v.domain, v.version) for v in model.opset_import] == [('', 17)],
            'Original graph IR/opset changed')
    require(not model.functions and not model.training_info and not graph.sparse_initializer
            and not graph.quantization_annotation, 'Unexpected graph machinery')
    require(len(graph.node) == 87 and len(graph.initializer) == 63, 'Original graph count mismatch')
    require(Counter(n.op_type for n in graph.node) == EXPECTED_OPS, 'Original operator inventory mismatch')
    require(len(graph.input) == len(graph.output) == 1 and graph.input[0].name == 'input'
            and graph.output[0].name == 'output', 'Original public IO changed')
    require(tensor_shape(graph.input[0]) == [2, 437, 1024] and
            tensor_shape(graph.output[0]) == [1, 874, 1024], 'Original pair shape changed')
    initializers = {v.name: v for v in graph.initializer}
    nodes = {v.name: v for v in graph.node}
    require(len(nodes) == 87 and len(initializers) == 63, 'Duplicate source names')
    producers, consumers = {}, defaultdict(list)
    inventory = []
    available = {'input'} | set(initializers)
    for index, node in enumerate(graph.node):
        require(node.name and not node.domain and not any(a.type in ATTR_GRAPH_TYPES for a in node.attribute),
                'Only original standard non-control-flow nodes expected')
        require(all(name in available for name in node.input), 'Original dependency order broken')
        for name in node.input:
            consumers[name].append(node.name)
        for name in node.output:
            require(name not in available, 'Duplicate source activation')
            producers[name] = node.name
            available.add(name)
        inventory.append({'index': index, 'name': node.name, 'op': node.op_type,
                          'inputs': list(node.input), 'outputs': list(node.output),
                          'category': classify(node.name), 'protobuf_sha256': encoded_sha(node)})
    backward, pending = set(), ['output']
    while pending:
        tensor = pending.pop()
        if tensor not in producers:
            continue
        name = producers[tensor]
        if name in backward:
            continue
        backward.add(name)
        pending.extend(nodes[name].input)
    require(backward == set(nodes), 'Original output no longer depends on every source node')
    shapes = {v.name: tensor_shape(v) for v in
              list(graph.input) + list(graph.output) + list(graph.value_info)}
    initializer_meta = []
    for name, value in initializers.items():
        require(value.data_location == 0 and not value.external_data and value.raw_data,
                'Original embedded initializer expected')
        initializer_meta.append({'name': name, 'dims': list(value.dims),
            'ONNX_dtype': value.data_type, 'raw_bytes': len(value.raw_data),
            'raw_sha256': hashlib.sha256(value.raw_data).hexdigest(),
            'protobuf_sha256': encoded_sha(value)})
    parameter_meta = []
    for role in ROLES:
        bindings = exported['original_parameter_bindings'][role]
        require(len(bindings) == 18, 'Original complete block parameter binding count changed')
        for logical_name, binding in bindings.items():
            value = initializers[binding['initializer']]
            array = np.frombuffer(value.raw_data, dtype='<f4').reshape(tuple(value.dims))
            embedded = binding['embedded']
            require(value.data_type == 1 and list(array.shape) == embedded['shape']
                    and array.nbytes == embedded['bytes']
                    and hashlib.sha256(value.raw_data).hexdigest() == embedded['sha256'],
                    'Original embedded parameter bytes changed')
            restored = np.ascontiguousarray(array.T if binding['transpose_2d'] else array)
            original = binding['original']
            original_sha = hashlib.sha256(memoryview(restored).cast('B')).hexdigest()
            require(list(restored.shape) == original['shape'] and restored.nbytes == original['bytes']
                    and original_sha == original['sha256'], 'Original untransposed parameter changed')
            parameter_meta.append({'role': role, 'logical_name': logical_name,
                                   'initializer': value.name, 'original_shape': list(restored.shape),
                                   'original_raw_sha256': original_sha,
                                   'embedded_raw_sha256': embedded['sha256'],
                                   'inside_MLP': logical_name.startswith('mlp.')})
    require(len(parameter_meta) == 36 and sum(v['inside_MLP'] for v in parameter_meta) == 8,
            'Complete original parameter inventory changed')
    regions = {}
    for role in ROLES:
        names = [role + '/' + suffix for suffix in MLP_SUFFIXES]
        selected = set(names)
        region = [row for row in inventory if row['name'] in selected]
        require([v['name'] for v in region] == names and
                [v['op'] for v in region] == ['MatMul', 'Add', 'Div', 'Erf', 'Add', 'Mul', 'Mul', 'MatMul', 'Add'],
                'Original complete exact-GELU MLP changed')
        activation_inputs = sorted({name for row in region for name in row['inputs']
                                   if name not in initializers and producers.get(name) not in selected})
        outputs = [name for row in region for name in row['outputs']]
        external_outputs = [name for name in outputs if any(v not in selected for v in consumers[name])]
        require(activation_inputs == [role + '/norm2'] and external_outputs == [role + '/mlp.fc2'],
                'Original MLP boundary changed')
        require(consumers[role + '/mlp.fc2'] == [role + '/ls2'], 'Original LayerScale consumer changed')
        init_names = sorted({name for row in region for name in row['inputs'] if name in initializers})
        require(len(init_names) == 7, 'MLP must retain four learned tensors and three exact-GELU constants')
        regions[role] = {'original_node_indices': [v['index'] for v in region],
            'original_nodes': region, 'original_output_tensor_names': outputs,
            'activation_input': activation_inputs[0], 'input_shape': shapes[activation_inputs[0]],
            'activation_output': external_outputs[0], 'output_shape': shapes[external_outputs[0]],
            'intermediate_hidden_shape': shapes[role + '/gelu'],
            'external_input_producer': role + '/norm2', 'external_output_consumers': [role + '/ls2'],
            'unchanged_initializers': init_names,
            'FP16_boundary_requests': [role + '/norm2', role + '/ls2'],
            'norm_LayerScale_residual_attention_RoPE_are_outside': True}
    return {'node_count': 87, 'initializer_count': 63, 'operator_counts': dict(EXPECTED_OPS),
            'all_source_nodes_reachable_from_final_output': True,
            'node_inventory': inventory, 'initializer_hash_inventory': initializer_meta,
            'original_parameter_hash_inventory': parameter_meta,
            'all_36_original_parameters_checked_after_reversing_original_export_transposes': True,
            'MLP_regions': regions, 'category_counts': dict(Counter(v['category'] for v in inventory))}


def saved_precision(compiled, profiled, compiler_log):
    require(compiled['candidate_accepted'] is False and profiled['candidate_accepted'] is False,
            'Historical diagnostic acceptance changed')
    rows = compiled['final_compile_operator_table']
    selected = [v for v in rows if '/mlp.fc' in v['name']]
    require([v['id'] for v in selected] == [29, 30, 49, 50] and
            [v['op'] for v in selected] == ['ConvExGelu', 'ConvAdd', 'ConvExGelu', 'ConvAdd'],
            'Historical original MLP fusion inventory changed')
    text = re.sub(r'\x1b\[[0-9;]*m', '', compiler_log.read_text())
    evidence = []
    for number, line in enumerate(text.splitlines(), 1):
        if ('fuse_exmatmul_mul' in line and '/ls2' in line) or (
                'ConvAdd' in line and '/residual1-rs_tp' in line):
            evidence.append({'line': number, 'text': line})
    require(len(evidence) == 4, 'Expected two LayerScale folds and two residual fusion input records')
    precision = {}
    for role in ('nonquant_fp16', 'ptq_w8a8'):
        implementation = profiled['implementations'][role]
        precision[role] = {
            'internal_target_counts': implementation['internal_target_counts'],
            'internal_data_type_token_counts': implementation['internal_data_type_token_counts'],
            'all_eight_operator_mean_microseconds': implementation['all_eight_operator_mean_microseconds'],
            'six_repeat_operator_mean_microseconds': implementation['six_repeat_operator_mean_microseconds'],
            'original_subregions': implementation['original_subregions'],
            'internal_CPU_operators': [v for v in implementation['profile_named_operators'] if v['target'] == 'CPU'
                                      and v['op'] not in ('InputOperator', 'OutputOperator')]}
    return {'v49_compiled_MLP_named_operator_families': selected,
            'actual_LayerScale_fold_and_residual_input_log_records': evidence,
            'MLP_name_based_profile_bucket_includes_fused_non_MLP_semantics': True,
            'v49_complete_pair_profile': precision,
            'normal_SDK_IO_separate_from_profile': profiled['separate_normal_SDK_IO'],
            'INT8_IO_label_does_not_prove_all_internal_arithmetic_INT8': True,
            'CPU_reshape_is_attention_value_layout_outside_MLP': True,
            'MLP_precision_changes_alone_do_not_prove_CPU_reshape_removal': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('graph', 'export', 'protobuf-source', 'SDK-api-source', 'compile-audit',
                 'profile-audit', 'compiler-log', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--candidate-graph', type=Path)
    args = parser.parse_args()
    pins = {}
    graph = read_pinned(args.graph, PAIR_GRAPH_SHA, pins)
    preservation = pin_graph(graph)
    if args.candidate_graph is not None:
        pin_graph(args.candidate_graph)
        read_pinned(args.candidate_graph, PAIR_GRAPH_SHA, pins)
    exported_path = read_pinned(args.export, PAIR_EXPORT_SHA, pins)
    exported = json.loads(exported_path.read_text())
    require(exported['checkpoint']['sha256'] == CHECKPOINT_SHA,
            'Original pretrained checkpoint identity changed')
    pb = load_pb(args.protobuf_source, pins)
    model = pb.ModelProto()
    model.ParseFromString(graph.read_bytes())
    inspection = inspect_graph(model, exported)
    compiler_path = read_pinned(args.compile_audit, COMPILE_AUDIT_SHA, pins)
    profile_path = read_pinned(args.profile_audit, PROFILE_AUDIT_SHA, pins)
    compiled = json.loads(compiler_path.read_text())
    profiled = json.loads(profile_path.read_text())
    log_path = read_pinned(args.compiler_log, compiled['actual_main_log_sha256'], pins)
    saved = saved_precision(compiled, profiled, log_path)
    api = read_pinned(args.SDK_api_source, API_SOURCE_SHA, pins)
    methods = {node.name: ast.unparse(node.args) for node in ast.walk(ast.parse(api.read_text()))
               if isinstance(node, ast.FunctionDef) and node.name in
               ('hybrid_quantization_step1', 'hybrid_quantization_step2')}
    require(set(methods) == {'hybrid_quantization_step1', 'hybrid_quantization_step2'},
            'Official two-step wrapper signatures missing')
    require(all(sha(path) == expected for path, expected in pins.items()), 'Saved sources changed during audit')
    report = {'schema': 'v50_original_VGGT_complete_MLP_graph_layout_audit',
        'status': 'static_source_scope_and_saved_fusions_checked_not_runtime_validated',
        'script_sha256': sha(__file__), 'scope_contracts_sha256': sha(Path(__file__).with_name('v50_original_mlp_scope_contracts.py')),
        'original_graph_sha256': PAIR_GRAPH_SHA, 'original_export_sha256': PAIR_EXPORT_SHA,
        'original_checkpoint_sha256': CHECKPOINT_SHA, 'preservation_gate': preservation,
        'graph_layout': inspection, 'actual_v49_saved_evidence': saved,
        'official_SDK_wrapper_signatures': methods, 'source_and_saved_evidence_pins': pins,
        'source_and_saved_inputs_unchanged': True,
        'candidate_scope': {'label': 'MLP-focused precision allocation diagnostic',
            'requested_original_MLP_outputs': [role + '/' + suffix for role in ROLES for suffix in MLP_SUFFIXES],
            'all_other_original_activation_outputs_request_FP16': True,
            'exact_MLP_only_precision_proven': False,
            'generated_tensor_catalog_and_actual_fusions_must_be_bound_before_asserting_exact_scope': True,
            'config_edit_scope': 'only custom_quantize_layers; generated parameters/ranges/model/data unchanged',
            'custom_hybrid': None, 'proposal': False, 'graph_rewrite': False,
            'INT8_vs_FP16_boundaries_add_conversion_and_may_break_existing_fusions': True},
        'rtol': RTOL, 'atol': ATOL, 'candidate_accepted': False,
        'original_Torch_FP32_reference_is_sole_numerical_reference': True,
        'historical_FP32_rejections_retained': True, 'real_time_validated': False,
        'whole_model_quality_validated': False,
        'actual_model_ORT_SDK_compiler_calibration_executed_by_auditor': False,
        'board_access_by_auditor': False,
        'limitations': ['No numerical inference or compiler was run by this static auditor',
            'Preserving exact source graph and ranges does not preserve FP32 math after quantization',
            'Actual generated tensor names are required; profile operation names are not tensor selectors',
            'MLP-only precision is not proved by MLP-named fused operators, which historically consume residual inputs',
            'Keep final compile/profile precision and CPU/NPU placement checks separate from fixed Torch FP32 output gates',
            'No whole-model/fresh-pose real-time, trajectory or stability claim is supported']}
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'report_sha256': sha(args.output),
        'source_nodes': 87, 'source_initializers': 63, 'parameters_preserved': 36,
        'original_MLP_nodes': 18, 'exact_MLP_only_precision_proven': False, 'candidate_accepted': False}))


if __name__ == '__main__':
    main()
