"""Bounded r2: 65 actual original SDK graph internal outputs, no SDK imports.

Reuse unchanged r1 parsers/file gates. Static selector membership is separate
from SDK support, external IO precision, initializer precision and adoption.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import hashlib

R1_SOURCE = Path(__file__).with_name('v50_hybrid_step2_contracts.py').resolve()
R1_CONTRACTS_SHA = 'd92103b618fd186605269d151f4cd480429b26570aa5b466c5f0f90ca01075df'
if hashlib.sha256(R1_SOURCE.read_bytes()).hexdigest() != R1_CONTRACTS_SHA:
    raise ValueError('Frozen original r1 parser/artifact gates changed')
_spec = importlib.util.spec_from_file_location('_v50_frozen_r1_gates', R1_SOURCE)
R1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R1)
for _name in ('SCHEMA', 'PAIR_EXPORT_SHA', 'PAIR_GRAPH_SHA', 'PAIR_CONTRACTS_SHA', 'QUANT_CONTRACTS_SHA',
              'CAPTURE_CONTRACTS_SHA', 'CAPTURE_SOURCE_SHA', 'CAPTURE_SHA', 'CHECKPOINT_SHA', 'STEP1_SHA',
              'STEP1_SOURCE_SHA', 'HYBRID_API_SHA', 'SDK_API_SHA', 'LAYOUT_CONTRACTS_SHA', 'DATASET_SHA',
              'CONFIGURATION', 'VERSIONS', 'INPUT_TRANSFORM', 'RAW_ARTIFACTS', 'INTEGER_SELECTORS',
              'CATALOG_SHA', 'UNTOUCHED_SUFFIX_SHA', 'require', 'sha', 'raw_sha', 'canonical', 'valid_sha',
              'unique_json', 'check_operations', 'check_cleanup', 'check_common', 'load_step1',
              'artifact_path', 'parse_cfg', 'edit_cfg'):
    globals()[_name] = getattr(R1, _name)

REVISION = 'r2_actual_internal_node_output_selectors'
FAILED_R1_SHA = '7f4f1ee288e4314cd4bfea979bc8eb564e9b3b0f0d036665dd4dd55efe8c5a44'
SELECTOR_PROOF_SHA = 'c4445e366f7f2bf54dcca3a9e404b40699819a374d41b8a68731f4e009025a25'
PROTOBUF_SCHEMA_SHA = 'eec80d60778be4d49524475b95fb05e5d1f917d67d815785a075dceeb29c1f53'
POLICY_SHA = 'd250f505c7ed4c4764d13ccf1e5b79b9b412a68266cb9f795ef51be8b4db3935'
APPROVED_CONFIGURATION = {'file': 'pair.internal-fc1-gelu.quantization.cfg', 'bytes': 36081,
                          'sha256': '02b58ad85931626157b9373eb5fb8fa562dfd9201e9fbb9964101ed05334117e'}
DIAGNOSTIC_SCOPE = '65 actual internal node-output selectors; four fc1/GELU INT8 and 61 FP16'
INTERNAL_SELECTORS = (
    'frame0/attention_heads_transpose_sw', 'frame0/attention_values_sdpa',
    'frame0/attn.k_norm', 'frame0/attn.k_norm#1_rs', 'frame0/attn.q_norm', 'frame0/attn.q_norm#1_rs',
    'frame0/attn.qkv.matmul_mm_tp_sw', 'frame0/gelu_tp-rs', 'frame0/key', 'frame0/key_rope',
    'frame0/key_rope_cosine', 'frame0/key_rope_permuted', 'frame0/key_rope_sine',
    'frame0/key_transpose_rs', 'frame0/key_transpose_sw', 'frame0/ls1-rs', 'frame0/ls2-rs',
    'frame0/mlp.fc1.matmul_mm_tp_sw', 'frame0/norm1#1', 'frame0/norm2#1', 'frame0/output-rs_tp',
    'frame0/qkv_transpose_swap', 'frame0/query', 'frame0/query_rope', 'frame0/query_rope_cosine',
    'frame0/query_rope_permuted', 'frame0/query_rope_sine', 'frame0/query_rope_tp_rs',
    'frame0/query_rope_tp_sw', 'frame0/residual1-rs_tp', 'frame0/value', 'frame0/value_tp', 'frame0/value_tp_rs',
    'global0/attention_heads_transpose_sw', 'global0/attention_values_sdpa', 'global0/attn.k_norm#1_rs',
    'global0/attn.q_norm#1_rs', 'global0/attn.qkv.matmul_mm_tp_sw_gather', 'global0/gelu_tp-rs',
    'global0/input_rs#1_sw', 'global0/input_tp-rs', 'global0/key_2sl-rs', 'global0/key_rope',
    'global0/key_rope_cosine', 'global0/key_rope_permuted', 'global0/key_rope_sine',
    'global0/ls1-rs', 'global0/ls2-rs', 'global0/mlp.fc1.matmul_mm_tp_sw', 'global0/norm1#1',
    'global0/norm2#1', 'global0/qkv_transpose-rs_tp', 'global0/query_2sl-rs', 'global0/query_rope',
    'global0/query_rope_cosine', 'global0/query_rope_permuted', 'global0/query_rope_sine',
    'global0/residual1-rs_tp', 'global0/value_2sl-rs', 'input_int8', 'input_rs#1', 'input_tp-rs',
    'output-rs', 'output-rs_tp', 'output_int8')


def check_failed_r1(row):
    require(row.get('schema') == SCHEMA and row.get('status') == 'failed'
            and row.get('script_sha256') == 'ebedcab7df5347162ce5c696d8c13095bd3fda0b15c7da5351d4e8b2dfd44cd0'
            and row.get('contracts_sha256') == R1_CONTRACTS_SHA
            and row.get('precision_policy_sha256') == R1.POLICY_SHA,
            'Actual immutable failed r1 provenance required')
    check_common(row); check_cleanup(row)
    operations = row.get('operations', [])
    require(len(operations) == 1 and operations[0].get('name') == 'hybrid_quantization_step2'
            and operations[0].get('status') == 'failed'
            and "Invalid operands name 'input'" in operations[0].get('error', '')
            and 'output' not in row, 'Keep actual r1 rejection before any export')


def load_failed_r1(path, pins=None):
    path = Path(path).resolve(strict=True)
    require(sha(path) == FAILED_R1_SHA, 'Failed r1 receipt bytes changed')
    row = unique_json(path); check_failed_r1(row)
    if pins is not None:
        pins[str(path)] = FAILED_R1_SHA
    return row


def check_policy(policy, raw, edited, layout):
    require(policy.get('schema') == 'v50_original_VGGT_four_selector_diagnostic_precision_policy_r2'
            and policy.get('revision') == REVISION and policy.get('pair_graph_sha256') == PAIR_GRAPH_SHA
            and policy.get('pair_export_sha256') == PAIR_EXPORT_SHA
            and policy.get('checkpoint_sha256') == CHECKPOINT_SHA and policy.get('capture_manifest_sha256') == CAPTURE_SHA
            and policy.get('step1_receipt_sha256') == STEP1_SHA
            and policy.get('failed_r1_receipt_sha256') == FAILED_R1_SHA
            and policy.get('selector_scope_proof_sha256') == SELECTOR_PROOF_SHA
            and policy.get('layout_contracts_sha256') == LAYOUT_CONTRACTS_SHA,
            'Bounded r2 original/failed-r1/static-output provenance changed')
    before, _, _ = parse_cfg(raw)
    requested = policy.get('requested_custom_layers')
    require(type(requested) is dict and set(requested) == set(INTERNAL_SELECTORS)
            and len(requested) == 65 and len(before['quantize_parameters']) == 95,
            'Only exact 65 actual internal outputs may receive requests')
    require(all(requested[key] == ('int8' if key in INTEGER_SELECTORS else 'float16') for key in requested),
            'Only the unchanged four fc1/GELU outputs may request INT8')
    expected, gate = edit_cfg(raw, requested, layout)
    require(expected == edited and canonical(policy.get('precision_edit_gate')) == canonical(gate),
            'Original generated quantization parameters/block-only edits changed')
    require(policy.get('integer_selector_count') == 4 and policy.get('floating_selector_count') == 61
            and policy.get('excluded_initializer_count') == 24 and policy.get('excluded_public_IO') == ['input', 'output']
            and policy.get('excluded_absent_SDPA_estimate_count') == 4
            and policy.get('external_IO_initializer_and_hidden_attention_precision_unproven') is True
            and policy.get('SDK_acceptance_of_static_selectors_unproven') is True
            and policy.get('candidate_accepted') is False
            and policy.get('final_fc1_weight_or_arithmetic_precision_proven') is False,
            'Static selector proof must not claim SDK acceptance or arithmetic precision')
    require(policy.get('approved_configuration') == {'file': APPROVED_CONFIGURATION['file'],
            'bytes': len(edited), 'sha256': raw_sha(edited)}, 'Reviewed r2 config byte identity changed')
    return gate


def load_policy(path, expected_sha, raw_path, layout, pins=None):
    path = Path(path).resolve(strict=True)
    require(expected_sha == POLICY_SHA and sha(path) == expected_sha, 'Root-reviewed bounded r2 policy SHA mismatch')
    policy = unique_json(path)
    edited_path = artifact_path(path.parent, policy['approved_configuration'])
    raw, edited = Path(raw_path).read_bytes(), edited_path.read_bytes()
    require(raw_sha(raw) == RAW_ARTIFACTS['configuration']['sha256'], 'Raw step1 cfg changed')
    gate = check_policy(policy, raw, edited, layout)
    if pins is not None:
        pins[str(path)] = expected_sha
        pins[str(edited_path)] = policy['approved_configuration']['sha256']
    return policy, edited_path, gate


def check_step2_compile(row, expected_script_sha, expected_contracts_sha,
                        expected_policy_sha, expected_step1_sha, expected_model_sha):
    require(row.get('schema') == SCHEMA and row.get('revision') == REVISION
            and row.get('status') == 'compiled_not_runtime_validated', 'Successful bounded original r2 compile required')
    for key, expected in {'script_sha256': expected_script_sha, 'contracts_sha256': expected_contracts_sha,
                          'precision_policy_sha256': expected_policy_sha,
                          'step1_receipt_sha256': expected_step1_sha}.items():
        require(valid_sha(expected) and row.get(key) == expected, 'Bounded r2 receipt binding mismatch: ' + key)
    require(expected_policy_sha == POLICY_SHA and expected_step1_sha == STEP1_SHA
            and row.get('step1_script_sha256') == STEP1_SOURCE_SHA
            and row.get('r1_contracts_sha256') == R1_CONTRACTS_SHA
            and row.get('failed_r1_receipt_sha256') == FAILED_R1_SHA
            and row.get('failed_r1_rejection_retained') is True
            and row.get('selector_scope_proof_sha256') == SELECTOR_PROOF_SHA
            and row.get('pair_contracts_sha256') == PAIR_CONTRACTS_SHA
            and row.get('layout_contracts_sha256') == LAYOUT_CONTRACTS_SHA,
            'Original r1/r2 source and failed history must remain bound')
    check_common(row)
    check_operations(row.get('operations'), ['hybrid_quantization_step2', 'export_rknn'])
    check_cleanup(row)
    required = {expected_script_sha, expected_contracts_sha, expected_policy_sha, R1_CONTRACTS_SHA,
                FAILED_R1_SHA, STEP1_SHA, PAIR_EXPORT_SHA, PAIR_GRAPH_SHA, PAIR_CONTRACTS_SHA,
                CAPTURE_CONTRACTS_SHA, CAPTURE_SHA, QUANT_CONTRACTS_SHA, LAYOUT_CONTRACTS_SHA,
                SDK_API_SHA, HYBRID_API_SHA, APPROVED_CONFIGURATION['sha256']}
    required.update(item['sha256'] for item in RAW_ARTIFACTS.values())
    require(required <= set(row['source_and_input_hashes'].values()), 'Required r2 source/input pins missing')
    require(row.get('precision_policy_approved') is True and row.get('runtime_inference_API_called') is False
            and row.get('calibration_executed_on_board_only') is True
            and row.get('new_calibration_dataset_supplied_to_step2') is False
            and row.get('original_Torch_FP32_reference_is_sole_numerical_reference') is True
            and row.get('external_IO_initializer_and_hidden_attention_precision_unproven') is True,
            'Frozen static requests remain a board diagnostic')
    require(canonical(row.get('raw_generated_artifacts')) == canonical(RAW_ARTIFACTS)
            and row.get('original_checkpoint', {}).get('sha256') == CHECKPOINT_SHA
            and row['original_checkpoint'].get('bytes') == 5026874952, 'Original model/data/checkpoint changed')
    gate = row.get('precision_edit_gate', {})
    require(all(gate.get(k) is True for k in ('only_custom_quantize_layers_changed',
            'quantize_parameters_and_other_fields_unchanged', 'custom_block_only_bytes_changed'))
            and gate.get('untouched_prefix_bytes') == 0 and gate.get('untouched_prefix_sha256') == raw_sha(b'')
            and gate.get('untouched_suffix_bytes') == 33709 and gate.get('untouched_suffix_sha256') == UNTOUCHED_SUFFIX_SHA
            and gate.get('quantize_parameters_catalog_sha256') == CATALOG_SHA,
            'All SDK generated parameters/ranges must remain byte-exact')
    require(gate.get('exact_original_MLP_only_precision_proven') is False and gate.get('candidate_accepted') is False
            and gate.get('runtime_accuracy_or_speed_proven') is False
            and gate.get('scope_label') == 'MLP-focused precision allocation diagnostic'
            and set(gate.get('integer_requests', [])) == set(INTEGER_SELECTORS)
            and len(gate.get('integer_requests', [])) == 4
            and set(gate.get('FP16_requests', [])) == set(INTERNAL_SELECTORS) - set(INTEGER_SELECTORS)
            and len(gate.get('FP16_requests', [])) == 61
            and row.get('diagnostic_requested_scope') == DIAGNOSTIC_SCOPE
            and row.get('final_fc1_weight_or_arithmetic_precision_proven') is False,
            'Only the approved 65-output requested scope is permitted')
    require(row.get('approved_configuration') == APPROVED_CONFIGURATION
            and row.get('approved_configuration_sha256') == APPROVED_CONFIGURATION['sha256']
            and gate.get('edited_configuration_sha256') == APPROVED_CONFIGURATION['sha256']
            and gate.get('raw_configuration_sha256') == RAW_ARTIFACTS['configuration']['sha256'],
            'Root-reviewed original r2 config changed')
    output = row.get('output', {})
    require(output.get('file') == 'pair.rknn' and type(output.get('bytes')) is int and output['bytes'] > 0
            and valid_sha(expected_model_sha) and output.get('sha256') == expected_model_sha
            and row.get('output_unchanged') is True and row.get('output_after') == output,
            'Actual exported diagnostic model byte identity changed')
    return {'compile_provenance_passed': True, 'candidate_accepted': False,
            'precision_configuration_scope_only': True, 'actual_SDK_IO_and_precision_not_yet_proven': True,
            'failed_r1_retained': True, 'revision': REVISION}
