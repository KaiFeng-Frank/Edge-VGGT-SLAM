"""Nine-shape M16 research catalogue, retaining the frozen eight-model chain.

DPVO, Princeton University (2022), MIT; see adjacent LICENSE. Only B1_D3_M16
has new v42 ancestry. No model, SDK, Torch or ORT execution occurs on import.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import uuid

V41_CONTRACTS_SHA256 = '3533d791978085216c4a28cfae45d2c8ce9e2d62a8862efbf66db02d34be5e21'
V41_CALLBACK_SHA256 = 'c7caef0c76673cd4511aa1a4c8a8a5427bfee691e1964227d782d0d6f26554eb'
V41_COMPILER_SHA256 = '4ddab438d294172380876d82ae1e6fd7a4f52ed797f901248fec492d331f2ff1'
M16_CONTRACTS_SHA256 = '7be6bf750dabb8a3d88464fd1502bde3bf7690596a39791710cf41a9f8cb62fe'
M16_COMPILER_SHA256 = '3ccd9002247bf3ff15e87bf04db2e3462db782c6d5a637f50b98094bcd584294'
PARENT_CONTRACTS_SHA256 = 'aa9e6af938f64ae4a4a82ffc84b487321fff208547703b0d65646c72b96d85c9'
PARENT_COMPILER_SHA256 = 'fb3a6e9eed9a4bc3f8490f0cf61c7ef5d5f4f4eff065bb9d9f057de97fb9cdd1'
EXPORT_MANIFEST_SHA256 = '5a9d6c571b672fbb174cf6adf9f8f578689fdd9e5f58ee4ec327320b55b62bfb'
ADDITION_EXPORTER_SHA256 = '350294c45f2a78f77daa376b3e24cac7443a11d1cb1d83b6a51b6f3bde6669a9'
EOF_EXPORTER_SHA256 = 'ae036a3fdfb0f9dc1c5cf0dd2dae37bd59ab313974904c20611936d42f9eca52'
ADAPTER_SHA256 = '0f427739037e2178dddc4631a32fb602ac0a2ddc259e48e147028604ae6733f1'
ABI_SHA256 = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
HEADER_SHA256 = 'c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a'
LIBRARY_SHA256 = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
LOG_HELPER_SHA256 = 'a5af6d39b2d19471255fc86aa42c9aa9b0247e428715c23ffd4ba79b184ad865'
LOG_PARENT_SHA256 = '322021ade54fb9035b7ce80652ec90169938a417fd2192528d28a805a83ed25e'
SOURCE_BINDING = dict(
    source_run_sha256='78f15e36e5444dfe17a70610eb4410f0605c262eb7102bc2fee39ba5cfdf427b',
    source_failure_sha256='f7fecc2da09ee74da72da4d30af99bb874149a5e6f6961a199a28c29f91cb072',
    source_failed_input_sha256='b0954b8dd84118d9749e7baeca15df83c59985059a0407cf1970521e8acb18ca',
    source_before_termination_sha256='d879eece461d1c20cd52388bcde97ba29f196b468b9398aac49eec6a9a85313e')
INPUT_NAMES = ('net', 'inp', 'corr')
OUTPUT_NAMES = ('net_out', 'delta', 'weight')
OLD_KEYS = tuple(f'B{b}_D{d}_M16' for b in (1, 2) for d in range(4, 8))
NEW_KEYS = ('B1_D3_M16',)
MODEL_KEYS = OLD_KEYS + NEW_KEYS
EXPORT_SCHEMA = 'v42_EOF_M16_D3_single_source_Update_CPU_ONNX_diagnostic'
COMPILE_SCHEMA = 'v42_online_M16_EOF_Update_RKNN_compilation_diagnostic'
INDEX_SCHEMA = 'v42_online_M16_Update_compile_index'
CONFIGURATION = dict(target_platform='rk3588', float_dtype='float16',
                     optimization_level=3, do_quantization=False)
UINT32_MAX = 2**32 - 1
_V41 = None


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_v41_parent(path=None):
    global _V41
    if path is None and _V41 is not None:
        return _V41
    path = Path(path or os.environ.get('V42_V41_CONTRACTS',
        Path(__file__).with_name('v41_online_contracts.py'))).resolve(strict=True)
    require(sha256(path) == V41_CONTRACTS_SHA256, 'Frozen v41 contracts changed')
    name = '_v42_private_v41_' + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value
    try:
        spec.loader.exec_module(value)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    _V41 = value
    return value


def load_m16_parent(path=None):
    return load_v41_parent().load_m16_parent(path)


def load_parent(path=None):
    return load_v41_parent().load_parent(path)


def parent_contracts_path():
    return load_v41_parent().parent_contracts_path()


def pin(path, expected, pins):
    return load_v41_parent().pin(path, expected, pins)


def recheck_pins(pins):
    return load_v41_parent().recheck_pins(pins)


def artifact(directory, row, pins):
    return load_v41_parent().artifact(directory, row, pins)


def bind_attrs(rows, logical, *, output):
    return load_v41_parent().bind_attrs(rows, logical, output=output)


def contracts(inputs, outputs):
    old = load_v41_parent()
    require(isinstance(inputs, list) and [v.get('name') for v in inputs] == list(INPUT_NAMES),
            'Exact input order required')
    if inputs[0].get('shape') != [1, 384, 3, 16]:
        return old.contracts(inputs, outputs)
    require(isinstance(outputs, list) and [v.get('name') for v in outputs] == list(OUTPUT_NAMES),
            'Exact output order required')
    shapes = ([1, 384, 3, 16], [1, 384, 3, 16], [1, 882, 3, 16],
              [1, 384, 3, 16], [1, 2, 3, 16], [1, 2, 3, 16])
    rows = inputs + outputs
    require(all(v.get('shape') == shape and all(type(d) is int for d in v['shape'])
                and v.get('dtype') == 'float32'
                for v, shape in zip(rows, shapes)), 'Fixed complete B1 D3 M16 FP32 CF IO required')
    result = [dict(name=v['name'], shape=tuple(v['shape']), logical_shape=v['shape'],
                   logical_dtype='float32') for v in rows]
    return dict(zip(INPUT_NAMES, result[:3])), dict(zip(OUTPUT_NAMES, result[3:]))


def model_key(inputs, outputs):
    ins, _ = contracts(inputs, outputs)
    b, _, d, m = ins['net']['shape']
    return f'B{b}_D{d}_M{m}'


def _array_descriptor(row, shape, dtype='float32'):
    import math
    return (row.get('shape') == shape and row.get('dtype') == dtype
        and row.get('bytes') == math.prod(shape) * (8 if dtype == 'int64' else 4)
        and isinstance(row.get('tensor_sha256'), str) and len(row['tensor_sha256']) == 64)


def validate_export_manifest(path, expected_sha256, *, parent_manifest=None,
        exporter_sha256=None, allow_fp32_rejected_diagnostic=False, verify_source_files=False):
    """Validate only the new D3 CPU teacher chain; old chains use the v41 API."""
    pins = {}; path = pin(path, expected_sha256, pins); row = json.loads(path.read_text())
    require(parent_manifest is not None and exporter_sha256 == EOF_EXPORTER_SHA256,
            'Explicit new exporter and frozen M16 parent required')
    require(row['schema'] == EXPORT_SCHEMA and row['script_sha256'] == exporter_sha256
        and row['source_M16_manifest_sha256'] == EXPORT_MANIFEST_SHA256
        and row['parent_exporter_sha256'] == load_m16_parent().EXPORTER_SHA256
        and row['v37_exporter_sha256'] == load_m16_parent().PARENT_EXPORTER_SHA256
        and all(row.get(k) == v for k, v in SOURCE_BINDING.items()), 'D3 export ancestry differs')
    accepted = row['strict_FP32_gate_passed'] is True
    require(row['status'] == ('completed_EOF_M16_D3_numerical_accepted_not_adopted' if accepted
        else 'completed_EOF_M16_D3_numerical_rejected')
        and (accepted or allow_fp32_rejected_diagnostic is True), 'Rejected D3 CPU/ORT requires explicit diagnostic flag')
    require(row['candidate_accepted'] is False and row['outputs_finite'] is True
        and row['historical_SDK_strict_FP32_rejected'] is True
        and row['diagnostic_allow_fp32_rejected'] is True and row['terminal_sigmoid'] == 'stable'
        and row['weight_domain_valid'] is True and row['original_parameters_unchanged'] is True
        and row['original_weights_before'] == row['original_weights_after'] == parent_manifest['original_weights_before']
        and row['original_Update_parameters'] == parent_manifest['original_Update_parameters']
        and len(row['original_weights_before']) == 94 and len(row['original_Update_parameters']) == 50
        and row['source_pins'] == row['source_hashes_after'] and row['sources_and_inputs_unchanged'] is True
        and row['cleanup_errors'] == [] and row['ORT_session_references_released'] is True
        and row['affinity_restored'] is True and row['strict_FP32_thresholds'] == dict(rtol=1e-4, atol=1e-5)
        and row['strict_FP32_thresholds_relaxed'] is False and row['output_clipping'] is False
        and row['host_model_inference'] is False and row['NPU_used'] is False
        and row['BA_executed'] is False and row['VO_state_committed'] is False
        and row['original_full_Update_reexecuted'] is True and row['original_full_Update_forward_calls'] == 1
        and row['teacher_reference_kind'] == 'new_original_Torch_on_saved_NPU_feedback_input'
        and row['original_CPU_trajectory_gold_used'] is False
        and row['source_member_padding'] is False and row['neighbor_or_edge_removal'] is False,
        'D3 teacher/weights/gates/cleanup scope differs')
    require(set(row['models']) == set(NEW_KEYS) and len(row['cases']) == 1,
            'Exactly the missing B1 D3 model required')
    teacher = row['full_teacher_artifact']
    require(Path(teacher['file']).name == teacher['file'] and teacher['bytes'] > 0
        and len(teacher['sha256']) == 64
        and set(teacher['arrays']) == {'net', 'delta', 'weight'}
        and all(_array_descriptor(teacher['arrays'][n], [1, 640, c])
                for n, c in (('net', 384), ('delta', 2), ('weight', 2))), 'New full640 teacher descriptor differs')
    model, case = row['models'][NEW_KEYS[0]], row['cases'][0]
    require(model_key(model['inputs'], model['outputs']) == NEW_KEYS[0]
        and model['batch'] == 1 and model['degree'] == 3 and model['patches'] == 16
        and case['model_key'] == NEW_KEYS[0] and case['edges'] == 48
        and case['original_call_index'] == 203 and case['observation_id'] == 191,
        'Predeclared EOF source shape/identity differs')
    binding = model['source_binding']
    require(case['source_binding'] == binding
        and binding['parent_run_sha256'] == SOURCE_BINDING['source_run_sha256']
        and binding['parent_failure_sha256'] == SOURCE_BINDING['source_failure_sha256']
        and binding['before_termination_sha256'] == SOURCE_BINDING['source_before_termination_sha256']
        and binding['failed_update_index'] == 203 and binding['selected_source_slot'] == 11
        and binding['selected_target_slots'] == [9, 10, 11]
        and binding['teacher_reference_kind'] == row['teacher_reference_kind']
        and binding['original_CPU_trajectory_gold_used'] is False
        and _array_descriptor(binding['canonical_edge_indices'], [1, 3, 16], 'int64')
        and case['original_full_inputs_artifact']['sha256'] == SOURCE_BINDING['source_failed_input_sha256']
        and case['full_original_teacher_artifact'] == teacher, 'D3 full source/teacher binding differs')
    covered = {v['parameter'] for entry in model['parameter_coverage']['mapping'] for v in entry['original_matches']}
    require(model['parameter_coverage']['original_parameter_count'] == 50
        and model['parameter_coverage']['all_original_parameters_covered'] is True
        and model['numerical_validation_completed'] is True
        and covered == set(row['original_Update_parameters']), 'All fifty original parameters required')
    gates = []
    for label in ('Torch_vs_original_group_gold', 'ORT_vs_original_group_gold', 'ORT_vs_grouped_Torch'):
        require(set(case[label]) == {'net', 'delta', 'weight'}, 'Three exact output comparisons required')
        for comparison in case[label].values():
            require(comparison['finite'] is True and comparison['rtol'] == 1e-4 and comparison['atol'] == 1e-5,
                    'Original FP32 gate changed')
            passed = comparison['gate_passed'] is True
            require(passed == (comparison['failed_elements'] == 0), 'Element failure count contradicts gate')
            gates.append(passed)
    require(row['strict_comparison_count'] == 9 and accepted == all(gates), 'All D3 comparisons required')
    if verify_source_files:
        for source, digest in row['source_pins'].items():
            pin(source, digest, pins)
    return row, pins


def validate_compilation(path, expected_sha256, manifest, manifest_sha256, key, *,
        compiler_path, compiler_sha256, contracts_path, contracts_sha256):
    require(key in NEW_KEYS, 'Only the new D3 compiler chain belongs to v42')
    pins = {}; path = pin(path, expected_sha256, pins)
    pin(compiler_path, compiler_sha256, pins); pin(contracts_path, contracts_sha256, pins)
    row = json.loads(path.read_text()); selected = manifest['models'][key]
    require(row['schema'] == COMPILE_SCHEMA and row['status'] == 'compiled_group_diagnostic_not_runtime_validated'
        and row['script_sha256'] == compiler_sha256 and row['contracts_sha256'] == contracts_sha256
        and row['manifest_sha256'] == manifest_sha256 and row['model_key'] == key
        and row['parent_compiler_sha256'] == PARENT_COMPILER_SHA256
        and row['parent_contracts_sha256'] == PARENT_CONTRACTS_SHA256
        and row['input']['sha256'] == selected['model']['sha256']
        and row['logical_inputs'] == selected['inputs'] and row['logical_outputs'] == selected['outputs']
        and row['configuration'] == CONFIGURATION, 'D3 compile ancestry/shape/config differs')
    require(row['released'] is True and not row.get('release_error')
        and row['sources_and_inputs_unchanged'] is True and row['source_pins'] == row['source_hashes_after']
        and row['candidate_accepted'] is False and row['model_inference_executed'] is False
        and row['NPU_execution_or_placement_validated'] is False
        and row['export_strict_FP32_gate_passed'] is manifest['strict_FP32_gate_passed'],
        'D3 compile cleanup/rejection/scope differs')
    if manifest['strict_FP32_gate_passed'] is False:
        require(row['diagnostic_allow_fp32_rejected'] is True, 'Rejected CPU lineage requires explicit compile diagnostic')
    return row, artifact(path.parent, row['output'], pins), pins


def validate_compile_index(path, expected_sha256, parent_manifest, additions_manifest, eof_manifest, *,
        old_online_compile_index, old_online_compile_index_sha256, additions_manifest_sha256,
        eof_manifest_sha256, parent_compiler, addition_compiler, addition_compiler_sha256,
        eof_compiler, eof_compiler_sha256, m16_contracts, v41_contracts, online_contracts,
        online_contracts_sha256, verify_artifacts=True):
    """Retain every old index entry exactly, and independently bind the D3 entry."""
    old = load_v41_parent()
    old_row, specs, compiled, pins = old.validate_compile_index(old_online_compile_index,
        old_online_compile_index_sha256, parent_manifest, additions_manifest,
        additions_manifest_sha256=additions_manifest_sha256, parent_compiler=parent_compiler,
        addition_compiler=addition_compiler, addition_compiler_sha256=addition_compiler_sha256,
        m16_contracts=m16_contracts, online_contracts=v41_contracts,
        online_contracts_sha256=V41_CONTRACTS_SHA256, verify_artifacts=verify_artifacts)
    path = pin(path, expected_sha256, pins); row = json.loads(path.read_text())
    require(row['schema'] == INDEX_SCHEMA and row['status'] == 'staged_nine_online_shapes_diagnostic'
        and row['old_online_compile_index_sha256'] == old_online_compile_index_sha256
        and row['parent_manifest_sha256'] == EXPORT_MANIFEST_SHA256
        and row['additions_manifest_sha256'] == additions_manifest_sha256
        and row['eof_manifest_sha256'] == eof_manifest_sha256
        and row['candidate_accepted'] is False and set(row['models']) == set(MODEL_KEYS),
        'Exact old eight plus new D3 catalogue required')
    require(all(row['models'][key] == old_row['models'][key] for key in OLD_KEYS),
            'Original eight catalogue entries must remain unchanged')
    key = NEW_KEYS[0]; entry = row['models'][key]; model = eof_manifest['models'][key]
    require(entry['lineage'] == 'v42' and model_key(model['inputs'], model['outputs']) == key,
            'New D3 model lineage/shape differs')
    directory = Path(entry['compile_dir'])
    require(directory.is_absolute() and entry['model']['file'] == 'block.rknn'
        and type(entry['model']['bytes']) is int and 0 < entry['model']['bytes'] <= UINT32_MAX,
        'Explicit fixed D3 compiled model required')
    if verify_artifacts:
        compiled_row, model_path, extra = validate_compilation(directory/'compile.json', entry['compile_sha256'],
            eof_manifest, eof_manifest_sha256, key, compiler_path=eof_compiler,
            compiler_sha256=eof_compiler_sha256, contracts_path=online_contracts,
            contracts_sha256=online_contracts_sha256)
        require(compiled_row['output'] == entry['model'], 'D3 index model bytes/SHA disagree with compile')
        pins.update(extra); compiled[key] = compiled_row
    else:
        model_path = directory / entry['model']['file']
    specs[key] = dict(inputs=model['inputs'], outputs=model['outputs'], model_bytes=entry['model']['bytes'],
        model_path=str(model_path), model_sha256=entry['model']['sha256'], compile_sha256=entry['compile_sha256'],
        export_strict_FP32_gate_passed=eof_manifest['strict_FP32_gate_passed'], lineage='v42')
    require(sum(v['model_bytes'] + 1 for v in specs.values()) <= 256 << 20,
            'Nine-model host buffer catalogue exceeds 256MiB')
    return row, specs, compiled, pins
