#!/usr/bin/env python3
"""Static protobuf-only audit of SDK-generated original VGGT tensor selectors.

This never imports an SDK, executes a model, or deserializes a pickle. The
SDK-generated graph is private and contains original model parameters. Only
tensor names, source identities, boundaries and producer metadata are emitted.
Operand membership is not proof that the SDK accepts a precision request.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

from v50_original_mlp_scope_contracts import (
    PAIR_EXPORT_SHA, PAIR_GRAPH_SHA, PROTOBUF_SOURCE_SHA, require, sha,
)

MODEL_SHA = 'f1c9f2c6cdbd739f7afbfa49c22355f3eb02604b85931561a1a6d95ef7b2c0a9'
CFG_SHA = 'cf93106589a3d33f5b6e1beabb16f8063ace2ee35fd2f85bbddf875396249632'
STEP1_SHA = '7c2a101be488dded5086c6b418480f459bc14a62f9269991c1d88a6892ec8133'
FAILED_STEP2_SHA = '7f4f1ee288e4314cd4bfea979bc8eb564e9b3b0f0d036665dd4dd55efe8c5a44'
INTEGER_SELECTORS = (
    'frame0/mlp.fc1.matmul_mm_tp_sw', 'frame0/gelu_tp-rs',
    'global0/mlp.fc1.matmul_mm_tp_sw', 'global0/gelu_tp-rs',
)


def pinned(path, expected, pins):
    path = Path(path).resolve(strict=True)
    require(path.is_file() and sha(path) == expected, 'Frozen saved source changed: ' + str(path))
    pins[str(path)] = expected
    return path


def json_unique(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate saved JSON field')
            result[key] = value
        return result
    def bad_constant(value):
        raise ValueError('Nonfinite JSON field: ' + value)
    return json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=bad_constant)


def safe_cfg(path):
    from ruamel.yaml import YAML
    from ruamel.yaml.tokens import AliasToken, AnchorToken
    from ruamel.yaml.nodes import MappingNode, ScalarNode, SequenceNode
    text = path.read_text()
    parser = YAML(typ='safe', pure=True)
    parser.allow_duplicate_keys = False
    require(not any(isinstance(t, (AliasToken, AnchorToken)) for t in parser.scan(text)),
            'Aliases/anchors are not allowed')
    def no_merge(node):
        if isinstance(node, MappingNode):
            for key, value in node.value:
                require(isinstance(key, ScalarNode) and key.value != '<<', 'Merge keys are not allowed')
                no_merge(value)
        elif isinstance(node, SequenceNode):
            for value in node.value:
                no_merge(value)
    no_merge(parser.compose(text))
    value = parser.load(text)
    json.dumps(value, allow_nan=False)
    require(type(value) is dict and list(value) == ['custom_quantize_layers', 'quantize_parameters']
            and value['custom_quantize_layers'] == {}, 'Original untouched SDK catalog required')
    require(type(value['quantize_parameters']) is dict and len(value['quantize_parameters']) == 95,
            'Original actual 95-entry catalog required')
    return value['quantize_parameters']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'cfg', 'protobuf-source', 'step1', 'failed-step2', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    pins = {str(Path(__file__).resolve()): sha(__file__)}
    model_path = pinned(args.model, MODEL_SHA, pins)
    require(model_path.stat().st_size == 102164257, 'Generated original model bytes changed')
    proto_path = pinned(args.protobuf_source, PROTOBUF_SOURCE_SHA, pins)
    cfg_path = pinned(args.cfg, CFG_SHA, pins)
    step1_path = pinned(args.step1, STEP1_SHA, pins)
    failed_path = pinned(args.failed_step2, FAILED_STEP2_SHA, pins)
    step1, failed = json_unique(step1_path), json_unique(failed_path)
    require(step1['pair_graph_sha256'] == PAIR_GRAPH_SHA
            and step1['pair_export_sha256'] == PAIR_EXPORT_SHA
            and step1['generated_artifacts']['model']['sha256'] == MODEL_SHA
            and step1['generated_artifacts']['configuration']['sha256'] == CFG_SHA
            and step1['candidate_accepted'] is False, 'Original generated-model provenance changed')
    require(failed['status'] == 'failed' and failed['candidate_accepted'] is False
            and "Invalid operands name 'input' in custom_quantize_layers" in failed.get('error', '')
            and failed.get('output') is None, 'Actual failed selector diagnostic must remain failed with no export')
    spec = importlib.util.spec_from_file_location('_v50_pure_sdk_original_proto', proto_path)
    pb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pb)
    model = pb.ModelProto()
    model.ParseFromString(model_path.read_bytes())
    graph = model.graph
    require(model.producer_name == 'rk' and model.ir_version == 8
            and [(v.domain, v.version) for v in model.opset_import] == [('', 19)],
            'Unexpected SDK-generated protobuf identity')
    require(len(graph.node) == 62 and len(graph.initializer) == 56,
            'Original generated node/initializer count changed')
    require(not model.functions and not graph.sparse_initializer, 'Unexpected generated graph machinery')
    init = {t.name for t in graph.initializer}
    graph_inputs, graph_outputs = {v.name for v in graph.input}, {v.name for v in graph.output}
    external_inputs = graph_inputs - init
    require(external_inputs == {'input'} and graph_outputs == {'output'}
            and init <= graph_inputs, 'Public IO must exclude legacy initializer-as-input declarations')
    producers, consumers = {}, set()
    for index, node in enumerate(graph.node):
        require(not any(a.type in (5, 10) for a in node.attribute), 'Nested graphs require a separate audit')
        consumers.update(node.input)
        for name in node.output:
            require(name and name not in producers, 'Empty or duplicate internal tensor output')
            producers[name] = {'node_index': index, 'node_name': node.name,
                              'operator': node.op_type, 'inputs': list(node.input)}
    catalog = safe_cfg(cfg_path)
    require(dict(Counter(v['dtype'] for v in catalog.values())) ==
            {'float32': 2, 'int8': 85, 'int16': 8}, 'Actual generated catalog dtype counts changed')
    public = external_inputs | graph_outputs
    internal_outputs = set(producers) - public
    require(len(internal_outputs) == 65 and internal_outputs <= set(catalog),
            'Every real internal output must be present in the calibration catalog')
    initializer_catalog = set(catalog) & init
    phantom = set(catalog) - internal_outputs - initializer_catalog - public
    require(len(initializer_catalog) == 24 and len(phantom) == 4
            and not phantom & consumers, 'Unexpected generated catalog-only names')
    expected_phantoms = {role + '/attention_values_2sdpa_' + suffix
                         for role in ('frame0', 'global0') for suffix in ('q_matmul', 'q_softmax')}
    require(phantom == expected_phantoms, 'SDPA catalog-only intermediates changed')
    require(set(INTEGER_SELECTORS) <= internal_outputs, 'Four requested FC1/GELU outputs missing')
    require([producers[k]['operator'] for k in INTEGER_SELECTORS] ==
            ['Conv', 'exGelu', 'Conv', 'exGelu'], 'Original FC1/GELU direct producers changed')
    requested = {k: 'int8' if k in INTEGER_SELECTORS else 'float16' for k in sorted(internal_outputs)}
    raw = cfg_path.read_bytes()
    suffix_start = raw.index(b'quantize_parameters:')
    require(suffix_start == 27, 'Original custom block boundary changed')
    require(all(sha(path) == expected for path, expected in pins.items()), 'Audit sources changed while reading')
    report = {
        'schema': 'v50_original_VGGT_actual_sdk_graph_selector_membership_audit',
        'status': '65_real_internal_outputs_identified_SDK_precision_acceptance_pending',
        'script_sha256': sha(__file__), 'pure_protobuf_source_sha256': PROTOBUF_SOURCE_SHA,
        'original_graph_sha256': PAIR_GRAPH_SHA, 'original_export_sha256': PAIR_EXPORT_SHA,
        'actual_step1_receipt_sha256': STEP1_SHA, 'actual_failed_step2_r1_receipt_sha256': FAILED_STEP2_SHA,
        'actual_generated_original_model_sha256': MODEL_SHA, 'actual_raw_cfg_sha256': CFG_SHA,
        'generated_model_producer': 'rk', 'generated_node_count': 62, 'generated_initializer_count': 56,
        'external_graph_inputs': sorted(external_inputs), 'external_graph_outputs': sorted(graph_outputs),
        'legacy_graph_inputs_include_56_initializers': True,
        'catalog_count': 95, 'catalog_existing_internal_output_count': 65,
        'catalog_initializer_count': 24, 'catalog_public_IO_count': 2, 'catalog_only_SDPA_subkey_count': 4,
        'exact_sorted_internal_output_selectors': sorted(internal_outputs),
        'direct_producer_provenance': {k: producers[k] for k in sorted(internal_outputs)},
        'excluded_initializers': sorted(initializer_catalog), 'excluded_public_IO': sorted(public),
        'excluded_catalog_only_SDPA_names': sorted(phantom),
        'bounded_r2_requested_custom_layers': requested, 'bounded_r2_INT8_count': 4, 'bounded_r2_FP16_count': 61,
        'untouched_raw_quantize_parameters_suffix_bytes': len(raw) - suffix_start,
        'untouched_raw_quantize_parameters_suffix_sha256': hashlib.sha256(raw[suffix_start:]).hexdigest(),
        'complete_original_MLP_INT8_requested': False, 'excluded_constants_or_attention_internal_math_FP16_proven': False,
        'exact_original_MLP_only_precision_proven': False, 'precision_scope_label': 'MLP-focused precision allocation diagnostic',
        'candidate_accepted': False, 'rtol': 1e-4, 'atol': 1e-5,
        'original_Torch_FP32_reference_is_sole_numerical_reference': True,
        'historical_FP32_rejections_retained': True, 'r1_failure_retained': True,
        'real_time_validated': False, 'whole_model_quality_validated': False,
        'source_and_saved_evidence_pins': pins, 'source_and_saved_inputs_unchanged': True,
        'actual_model_OR_SDK_OR_ORT_OR_compiler_OR_calibration_execution_by_auditor': False,
        'board_access_by_auditor': False,
        'limitations': [
            'Catalog membership alone is not an SDK precision-selector contract: r1 rejected the public input.',
            'This proves actual internal output names and producers; actual SDK acceptance and compiled precision remain pending.',
            'The 24 initializer catalog entries and four SDPA pseudo-intermediates are excluded, with their generated ranges unchanged.',
            'SDK graph optimization and dtype requests do not prove FP32 numerical equivalence or full SLAM real time.',
        ],
    }
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'audit_sha256': sha(args.output),
                      'internal_outputs': 65, 'INT8_requests': 4, 'FP16_requests': 61,
                      'actual_execution_by_auditor': False}))


if __name__ == '__main__':
    main()
