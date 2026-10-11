"""Pure original-pair hybrid provenance/config gates; no SDK/model imports.

YAML parsing is lazy and safe. Config surgery preserves all bytes outside the
one custom precision block. Compiler scope and accuracy require actual review.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

SCHEMA = 'v50_original_VGGT_pair_board_hybrid_step2_compile'
PAIR_EXPORT_SHA = '149450ffc675ce59950c6f89cf84adb41e388cd050eeb731b6d09a6fe16e9fe1'
PAIR_GRAPH_SHA = '2cf1f4de19c91b42b5bf685ab5e949bd98b08a64fc2e78b7f0c8f07992a009fc'
PAIR_CONTRACTS_SHA = '4ec6c0e827ab9669c942b589d2c504f0c9de306f76f4f4be0799af14ae322cd7'
QUANT_CONTRACTS_SHA = '47237b1a1e1614822c75180efd15be3e5c708b610f587f698fe9b37fcf1bc649'
CAPTURE_CONTRACTS_SHA = '6ce1f87f1683e8082a4d567718acb853101e5ee6bbaeef1ced02993a8e3133c8'
CAPTURE_SOURCE_SHA = '7a2279927a65e3740da933c1192c335d15f4f1803440ed84a27a2aebc8f02613'
CAPTURE_SHA = 'bf39dfa65526a4135cbd2d7ee8b927ce2c4a40da75aebf64652ed2cd87a7e7fc'
CHECKPOINT_SHA = 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
STEP1_SHA = '7c2a101be488dded5086c6b418480f459bc14a62f9269991c1d88a6892ec8133'
STEP1_SOURCE_SHA = 'fe7ff44126c1fa7fee7c4ef01c30d6c11781cf6e9604522c503595753d955086'
HYBRID_API_SHA = '9781b778c999ccadb43ee612a88c973e982dcd9f3977b15e526ddcdff3899b3d'
SDK_API_SHA = '63e4506fb851214e79f933f9de0d27dab04ecf93f6f46959c07c5787433fd05e'
LAYOUT_CONTRACTS_SHA = '105e6cfdf912f3e5a6c97eefc2265c4a7c141c97cff5b50632ebab6f7399e52a'
DATASET_SHA = '0593e7306d501cf7afc7b6476aadbb409620e05c55b935ef0e9e63ea7cd8bf24'
POLICY_SHA = '6eb6c909e2aeac6bdaafce680b4f39c33447f1dfbbc2cd3714be43055c9d9ad8'
APPROVED_CONFIGURATION = {'file': 'pair.mlp-focused.quantization.cfg', 'bytes': 37244,
                          'sha256': '18593e0cf4d05354cbb71f7bc48075542e9083d6fb060e5041d340f8d329dab9'}
CATALOG_SHA = 'ee4eb17524d9374d5b7ffb62dc0c19a05a2d597b2b3515155c04174bd009ab09'
UNTOUCHED_SUFFIX_SHA = '09272223433593681fb69b0afd5c942fdd87dcbb1192dae1e982c08f08358e79'
CONFIGURATION = {'target_platform': 'rk3588', 'float_dtype': 'float16', 'optimization_level': 3,
                 'quantized_dtype': 'w8a8', 'quantized_algorithm': 'normal', 'quantized_method': 'channel'}
VERSIONS = {'rknn-toolkit2': '2.3.2', 'torch': '2.2.0', 'numpy': '1.26.4',
            'onnx': '1.16.1', 'onnxruntime': '1.16.0', 'protobuf': '3.20.3'}
INPUT_TRANSFORM = {'mean_values': None, 'std_values': None, 'quant_img_RGB2BGR': False,
                   'transpose': False, 'extra_batch_dimension': False}
RAW_ARTIFACTS = {
    'model': {'file': 'pair.model', 'bytes': 102164257,
              'sha256': 'f1c9f2c6cdbd739f7afbfa49c22355f3eb02604b85931561a1a6d95ef7b2c0a9'},
    'data': {'file': 'pair.data', 'bytes': 1949723,
             'sha256': '6c339d408cb531c9ab72cace8514a4fa2281b88152b1d79062855753405c39fd'},
    'configuration': {'file': 'pair.quantization.cfg', 'bytes': 33736,
                      'sha256': 'cf93106589a3d33f5b6e1beabb16f8063ace2ee35fd2f85bbddf875396249632'}}
INTEGER_SELECTORS = ('frame0/mlp.fc1.matmul_mm_tp_sw', 'frame0/gelu_tp-rs',
                     'global0/mlp.fc1.matmul_mm_tp_sw', 'global0/gelu_tp-rs')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def raw_sha(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def valid_sha(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def unique_json(path):
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, 'Duplicate JSON key')
            output[key] = value
        return output
    def bad_constant(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=bad_constant)


def check_operations(rows, expected):
    require(type(rows) is list and [r.get('name') for r in rows] == expected, 'Unexpected SDK API sequence')
    for row in rows:
        require(row.get('status') == 'completed' and 'code' in row and
                (row['code'] is None or (type(row['code']) is int and row['code'] == 0)),
                'Failed/unsupported SDK operation return')
        seconds = row.get('seconds')
        require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0,
                'Finite operation timing required')


def check_cleanup(row):
    require(row.get('released') is True and row.get('cleanup_errors') == []
            and row.get('source_and_inputs_unchanged') is True
            and row.get('cwd_restored') is True and row.get('environment_restored') is True,
            'Source/input/release/process cleanup failed')
    before = row.get('source_and_input_hashes')
    require(type(before) is dict and before and before == row.get('source_and_input_hashes_after')
            and all(valid_sha(value) for value in before.values()), 'Complete unchanged source/input pins required')
    require('release_code' in row and (row['release_code'] is None or
            (type(row['release_code']) is int and row['release_code'] == 0)), 'SDK release return failed')


def check_common(row):
    for field, expected in {'pair_export_sha256': PAIR_EXPORT_SHA, 'pair_graph_sha256': PAIR_GRAPH_SHA,
                            'quant_contracts_sha256': QUANT_CONTRACTS_SHA,
                            'capture_contracts_sha256': CAPTURE_CONTRACTS_SHA,
                            'capture_manifest_sha256': CAPTURE_SHA,
                            'capture_script_sha256': CAPTURE_SOURCE_SHA,
                            'hybrid_api_sha256': HYBRID_API_SHA,
                            'SDK_API_source_sha256': SDK_API_SHA}.items():
        require(row.get(field) == expected, 'Original provenance binding mismatch: ' + field)
    require(canonical(row.get('configuration')) == canonical(CONFIGURATION)
            and canonical(row.get('versions')) == canonical(VERSIONS)
            and canonical(row.get('input_transform')) == canonical(INPUT_TRANSFORM),
            'Original frozen toolchain/configuration/input transformation changed')
    board = row.get('board', {})
    require(board.get('machine') in ('aarch64', 'arm64') and 'rk3588' in board.get('device_tree_compatible', ''),
            'Original RK3588 board identity required')
    require(row.get('model_inference_on_host') is False and row.get('candidate_accepted') is False
            and row.get('historical_FP32_rejections_retained') is True
            and row.get('whole_model_quality_validated') is False and row.get('real_time_validated') is False,
            'Compilation diagnostic must not be promoted to model/SLAM adoption')
    require(canonical(row.get('fixed_FP32_gate')) == canonical({'rtol': 1e-4, 'atol': 1e-5}),
            'Original strict FP32 tolerances must be retained')


def check_step1(row):
    require(row.get('schema') == 'v50_original_VGGT_pair_board_hybrid_step1'
            and row.get('status') == 'hybrid_intermediates_generated_not_precision_approved'
            and row.get('script_sha256') == STEP1_SOURCE_SHA, 'Frozen successful original step1 required')
    check_common(row)
    check_operations(row.get('operations'), ['config', 'load_onnx', 'hybrid_quantization_step1'])
    check_cleanup(row)
    require(row.get('calibration_on_board_only') is True and row.get('calibration_completed_on_board') is True
            and row.get('calibration_api_invoked') is True and row.get('precision_policy_approved') is False
            and row.get('SDK_runtime_or_inference_API_called') is False,
            'Only unedited original board calibration intermediates are accepted')
    require(canonical(row.get('step1_options')) == canonical({'rknn_batch_size': None, 'proposal': False,
            'proposal_dataset_size': 1, 'custom_hybrid': None}), 'Step1 API options changed')
    require(row.get('checkpoint', {}).get('sha256') == CHECKPOINT_SHA
            and row['checkpoint'].get('bytes') == 5026874952, 'Original checkpoint changed')
    require(row.get('generated_artifacts_unchanged') is True, 'Generated step1 artifacts changed')
    artifacts = row.get('generated_artifacts', {})
    require(set(artifacts) == set(RAW_ARTIFACTS), 'Three raw generated artifacts required')
    for role, expected in RAW_ARTIFACTS.items():
        require({key: artifacts[role].get(key) for key in expected} == expected
                and row.get('generated_artifacts_after', {}).get(role) ==
                {'sha256': expected['sha256'], 'bytes': expected['bytes']}, 'Raw intermediate identity mismatch')
    dataset = row.get('dataset', {})
    require(dataset.get('sha256') == DATASET_SHA and dataset.get('line_count') == 8
            and dataset.get('input_count') == 1 and dataset.get('heldout_samples_excluded') ==
            ['evaluation_00', 'evaluation_01'], 'Eight calibration-only dataset required')


def artifact_path(directory, item):
    directory, name = Path(directory).resolve(), Path(item['file'])
    require(not name.is_absolute() and name.name == str(name) and name.name not in ('', '.', '..'),
            'Generated artifact must use one confined filename')
    candidate = directory / name
    require(not candidate.is_symlink() and candidate.resolve(strict=True).parent == directory
            and candidate.is_file() and candidate.stat().st_size == item['bytes']
            and sha(candidate) == item['sha256'], 'Pinned nonempty regular intermediate identity required')
    return candidate


def load_step1(directory, expected_sha=STEP1_SHA, pins=None):
    require(expected_sha == STEP1_SHA, 'This diagnostic binds the actual frozen step1 receipt')
    receipt = Path(directory).resolve() / 'step1.json'
    require(sha(receipt) == expected_sha, 'Step1 receipt bytes changed')
    report = unique_json(receipt)
    check_step1(report)
    paths = {role: artifact_path(directory, item) for role, item in RAW_ARTIFACTS.items()}
    dataset = artifact_path(directory, report['dataset'])
    require(dataset.read_text() == ''.join(row['path'] + '\n' for row in report['dataset']['rows']),
            'Step1 dataset bytes/rows disagree')
    if pins is not None:
        pins[str(receipt)] = expected_sha
        for role, path in paths.items():
            pins[str(path)] = RAW_ARTIFACTS[role]['sha256']
        pins[str(dataset)] = DATASET_SHA
    return report, paths


def parse_cfg(raw):
    """Return safe parsed cfg and custom block bounds; reject aliases/merges/duplicates."""
    from ruamel.yaml import YAML
    from ruamel.yaml.tokens import AliasToken, AnchorToken
    from ruamel.yaml.nodes import MappingNode, ScalarNode, SequenceNode
    text = raw.decode('utf-8')
    require(not text.startswith('\ufeff'), 'No BOM in generated configuration')
    scanner = YAML(typ='safe', pure=True)
    require(not any(isinstance(token, (AliasToken, AnchorToken)) for token in scanner.scan(text)),
            'Aliases/anchors are not allowed in this deterministic config edit')
    loader = YAML(typ='safe', pure=True)
    loader.allow_duplicate_keys = False
    value = loader.load(text)
    canonical(value)  # Require finite JSON-compatible metadata; safe loader rejects arbitrary Python tags.
    node = YAML(typ='safe', pure=True).compose(text)
    require(type(value) is dict and isinstance(node, MappingNode), 'One root configuration map required')
    for key, _ in node.value:
        require(isinstance(key, ScalarNode) and key.tag == 'tag:yaml.org,2002:str'
                and key.value != '<<' and key.start_mark.column == 0, 'Plain root keys without merges required')
    def no_merge(current):
        if isinstance(current, MappingNode):
            for key, item in current.value:
                require(isinstance(key, ScalarNode) and key.tag == 'tag:yaml.org,2002:str'
                        and key.value != '<<', 'String mapping keys without YAML merges required')
                no_merge(item)
        elif isinstance(current, SequenceNode):
            for item in current.value:
                no_merge(item)
    no_merge(node)
    selected = [i for i, (key, _) in enumerate(node.value) if key.value == 'custom_quantize_layers']
    require(len(selected) == 1 and type(value.get('custom_quantize_layers')) is dict,
            'One custom_quantize_layers mapping required')
    index = selected[0]
    start = node.value[index][0].start_mark.index
    end = node.value[index + 1][0].start_mark.index if index + 1 < len(node.value) else len(text)
    return value, len(text[:start].encode('utf-8')), len(text[:end].encode('utf-8'))


def edit_cfg(raw, requested, layout):
    before, start, end = parse_cfg(raw)
    require(type(requested) is dict and requested and all(type(k) is str for k in requested),
            'Explicit reviewed generated selectors required')
    block = ('custom_quantize_layers:\n' + ''.join(
        '    ' + json.dumps(key, ensure_ascii=True) + ': ' + requested[key] + '\n'
        for key in requested)).encode('utf-8')
    edited = raw[:start] + block + raw[end:]
    after, next_start, next_end = parse_cfg(edited)
    gate = layout.check_cfg_edit(before, after, requested)
    require(raw[:start] == edited[:next_start] and raw[end:] == edited[next_end:],
            'Bytes outside the custom precision block changed')
    gate.update(custom_block_only_bytes_changed=True,
                untouched_prefix_bytes=start, untouched_prefix_sha256=raw_sha(raw[:start]),
                untouched_suffix_bytes=len(raw) - end, untouched_suffix_sha256=raw_sha(raw[end:]),
                raw_configuration_sha256=raw_sha(raw), edited_configuration_sha256=raw_sha(edited),
                quantize_parameters_catalog_sha256=raw_sha(canonical(before['quantize_parameters']).encode()))
    return edited, gate


def check_policy(policy, raw, edited, layout):
    require(policy.get('schema') == 'v50_original_VGGT_four_selector_diagnostic_precision_policy'
            and policy.get('pair_graph_sha256') == PAIR_GRAPH_SHA
            and policy.get('pair_export_sha256') == PAIR_EXPORT_SHA
            and policy.get('checkpoint_sha256') == CHECKPOINT_SHA
            and policy.get('capture_manifest_sha256') == CAPTURE_SHA
            and policy.get('step1_receipt_sha256') == STEP1_SHA
            and policy.get('layout_contracts_sha256') == LAYOUT_CONTRACTS_SHA,
            'Frozen diagnostic policy provenance changed')
    before, _, _ = parse_cfg(raw)
    catalog = before['quantize_parameters']
    requested = policy.get('requested_custom_layers')
    require(type(requested) is dict and set(requested) == set(catalog) and len(catalog) == 95,
            'This one conservative diagnostic explicitly covers all 95 actual selectors')
    require(set(INTEGER_SELECTORS) <= set(catalog)
            and all(requested[key] == ('int8' if key in INTEGER_SELECTORS else 'float16') for key in catalog),
            'Only the four reviewed fc1/GELU selectors may request INT8')
    expected_edited, expected_gate = edit_cfg(raw, requested, layout)
    require(expected_edited == edited and canonical(policy.get('precision_edit_gate')) == canonical(expected_gate),
            'Frozen byte-preserving config/policy evidence differs')
    require(policy.get('integer_selector_count') == 4 and policy.get('floating_selector_count') == 91
            and policy.get('entire_fc2_LayerScale_residual_group_requested_floating') is True
            and policy.get('catalog_contains_linear_weight_selectors') is False
            and policy.get('final_fc1_weight_or_arithmetic_precision_proven') is False
            and policy.get('exact_original_MLP_only_precision_proven') is False
            and policy.get('candidate_accepted') is False,
            'Conservative requested scope must remain separate from actual arithmetic/adoption')
    require(policy.get('approved_configuration') == {'file': 'pair.mlp-focused.quantization.cfg',
            'bytes': len(edited), 'sha256': raw_sha(edited)}, 'Approved config identity changed')
    return expected_gate


def load_policy(path, expected_sha, raw_path, layout, pins=None):
    path = Path(path).resolve(strict=True)
    require(valid_sha(expected_sha) and sha(path) == expected_sha, 'Reviewed diagnostic policy SHA mismatch')
    policy = unique_json(path)
    edited_path = artifact_path(path.parent, policy['approved_configuration'])
    raw, edited = Path(raw_path).read_bytes(), edited_path.read_bytes()
    require(raw_sha(raw) == RAW_ARTIFACTS['configuration']['sha256'], 'Original generated config changed')
    gate = check_policy(policy, raw, edited, layout)
    if pins is not None:
        pins[str(path)] = expected_sha
        pins[str(edited_path)] = policy['approved_configuration']['sha256']
    return policy, edited_path, gate


def check_step2_compile(row, expected_script_sha, expected_contracts_sha,
                        expected_policy_sha, expected_step1_sha, expected_model_sha):
    """Runtime's SDK-free receipt gate; caller pins actual receipt/model/source bytes."""
    require(row.get('schema') == SCHEMA and row.get('status') == 'compiled_not_runtime_validated',
            'Successful unadopted original hybrid step2 compile required')
    for name, expected in {'script_sha256': expected_script_sha, 'contracts_sha256': expected_contracts_sha,
                           'precision_policy_sha256': expected_policy_sha,
                           'step1_receipt_sha256': expected_step1_sha}.items():
        require(valid_sha(expected) and row.get(name) == expected, 'Runtime binding mismatch: ' + name)
    require(expected_step1_sha == STEP1_SHA and row.get('step1_script_sha256') == STEP1_SOURCE_SHA
            and row.get('layout_contracts_sha256') == LAYOUT_CONTRACTS_SHA
            and row.get('pair_contracts_sha256') == PAIR_CONTRACTS_SHA, 'Original compiler/source chain changed')
    require(expected_policy_sha == POLICY_SHA, 'Use the one root-reviewed conservative diagnostic policy')
    check_common(row)
    check_operations(row.get('operations'), ['hybrid_quantization_step2', 'export_rknn'])
    check_cleanup(row)
    require({expected_script_sha, expected_contracts_sha, expected_policy_sha, STEP1_SHA,
             PAIR_EXPORT_SHA, PAIR_GRAPH_SHA, PAIR_CONTRACTS_SHA, CAPTURE_CONTRACTS_SHA,
             CAPTURE_SHA, QUANT_CONTRACTS_SHA, LAYOUT_CONTRACTS_SHA, SDK_API_SHA, HYBRID_API_SHA,
             RAW_ARTIFACTS['model']['sha256'], RAW_ARTIFACTS['data']['sha256'],
             RAW_ARTIFACTS['configuration']['sha256'], APPROVED_CONFIGURATION['sha256']} <=
            set(row['source_and_input_hashes'].values()), 'Required original source/strategy/file pins missing')
    require(row.get('precision_policy_approved') is True and row.get('runtime_inference_API_called') is False
            and row.get('calibration_executed_on_board_only') is True
            and row.get('new_calibration_dataset_supplied_to_step2') is False
            and row.get('original_Torch_FP32_reference_is_sole_numerical_reference') is True,
            'Only frozen board diagnostic precision allocation is permitted')
    require(canonical(row.get('raw_generated_artifacts')) == canonical(RAW_ARTIFACTS),
            'Step2 must consume unchanged actual generated intermediates')
    require(row.get('original_checkpoint', {}).get('sha256') == CHECKPOINT_SHA
            and row['original_checkpoint'].get('bytes') == 5026874952, 'Original checkpoint identity changed')
    gate = row.get('precision_edit_gate', {})
    for field in ('only_custom_quantize_layers_changed', 'quantize_parameters_and_other_fields_unchanged',
                  'custom_block_only_bytes_changed'):
        require(gate.get(field) is True, 'Precision edit scope/byte preservation failed')
    require(gate.get('exact_original_MLP_only_precision_proven') is False
            and gate.get('candidate_accepted') is False and gate.get('runtime_accuracy_or_speed_proven') is False
            and gate.get('scope_label') == 'MLP-focused precision allocation diagnostic',
            'Generated config editing cannot prove fusion arithmetic or accuracy')
    require(gate.get('untouched_prefix_bytes') == 0 and gate.get('untouched_prefix_sha256') == raw_sha(b'')
            and gate.get('untouched_suffix_bytes') == 33709 and gate.get('untouched_suffix_sha256') == UNTOUCHED_SUFFIX_SHA
            and gate.get('quantize_parameters_catalog_sha256') == CATALOG_SHA,
            'SDK-generated ranges/catalog bytes changed')
    require(set(gate.get('integer_requests', [])) == set(INTEGER_SELECTORS)
            and len(gate.get('integer_requests', [])) == 4 and len(gate.get('FP16_requests', [])) == 91
            and row.get('diagnostic_requested_scope') == 'four actual fc1/GELU selectors INT8; entire fc2/LayerScale/residual group FP16'
            and row.get('final_fc1_weight_or_arithmetic_precision_proven') is False,
            'The approved conservative precision policy changed or overclaimed')
    approved = row.get('approved_configuration', {})
    require(approved == APPROVED_CONFIGURATION
            and gate.get('edited_configuration_sha256') == approved['sha256']
            and gate.get('raw_configuration_sha256') == RAW_ARTIFACTS['configuration']['sha256'],
            'Raw/approved configuration file identity mismatch')
    require(row.get('approved_configuration_sha256') == approved['sha256'], 'SDK config byte guard mismatch')
    output = row.get('output', {})
    require(output.get('file') == 'pair.rknn' and type(output.get('bytes')) is int and output['bytes'] > 0
            and valid_sha(expected_model_sha) and output.get('sha256') == expected_model_sha,
            'Actual exported original hybrid pair identity mismatch')
    require(row.get('output_unchanged') is True and row.get('output_after') == output,
            'Exported model bytes changed after release')
    return {'compile_provenance_passed': True, 'candidate_accepted': False,
            'precision_configuration_scope_only': True, 'actual_SDK_IO_and_precision_not_yet_proven': True}
