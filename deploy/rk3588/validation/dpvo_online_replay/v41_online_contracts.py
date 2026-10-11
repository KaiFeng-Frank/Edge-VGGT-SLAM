"""Eight-shape M16 online research catalogue; no SDK/model runs on import.

DPVO, Princeton University (2022), MIT. The six reused graphs remain bound to
their original v40 receipts. Two B1 graphs have a separate v41 ancestry; no
manifest is relabelled as an earlier experiment. Original FP32 gates stay fixed.
"""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import uuid

M16_CONTRACTS_SHA256 = '7be6bf750dabb8a3d88464fd1502bde3bf7690596a39791710cf41a9f8cb62fe'
M16_COMPILER_SHA256 = '3ccd9002247bf3ff15e87bf04db2e3462db782c6d5a637f50b98094bcd584294'
PARENT_CONTRACTS_SHA256 = 'aa9e6af938f64ae4a4a82ffc84b487321fff208547703b0d65646c72b96d85c9'
PARENT_COMPILER_SHA256 = 'fb3a6e9eed9a4bc3f8490f0cf61c7ef5d5f4f4eff065bb9d9f057de97fb9cdd1'
EXPORT_MANIFEST_SHA256 = '5a9d6c571b672fbb174cf6adf9f8f578689fdd9e5f58ee4ec327320b55b62bfb'
ADDITION_EXPORTER_SHA256 = '350294c45f2a78f77daa376b3e24cac7443a11d1cb1d83b6a51b6f3bde6669a9'
ADAPTER_SHA256 = '0f427739037e2178dddc4631a32fb602ac0a2ddc259e48e147028604ae6733f1'
ABI_SHA256 = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
HEADER_SHA256 = 'c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a'
LIBRARY_SHA256 = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
LOG_HELPER_SHA256 = 'a5af6d39b2d19471255fc86aa42c9aa9b0247e428715c23ffd4ba79b184ad865'
LOG_PARENT_SHA256 = '322021ade54fb9035b7ce80652ec90169938a417fd2192528d28a805a83ed25e'
INPUT_NAMES = ('net', 'inp', 'corr')
OUTPUT_NAMES = ('net_out', 'delta', 'weight')
MODEL_KEYS = tuple(f'B{b}_D{d}_M16' for b in (1, 2) for d in range(4, 8))
NEW_KEYS = ('B1_D4_M16', 'B1_D6_M16')
REUSED_KEYS = tuple(k for k in MODEL_KEYS if k not in NEW_KEYS)
EXPORT_SCHEMA = 'v41_missing_M16_single_source_Update_CPU_ONNX_diagnostic'
COMPILE_SCHEMA = 'v41_online_M16_Update_RKNN_compilation_diagnostic'
INDEX_SCHEMA = 'v41_online_M16_Update_compile_index'
CONFIGURATION = dict(target_platform='rk3588', float_dtype='float16',
                     optimization_level=3, do_quantization=False)
UINT32_MAX = 2**32-1
_M16 = None


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_m16_parent(path=None):
    global _M16
    if path is None and _M16 is not None:
        return _M16
    path = Path(path or os.environ.get('V41_M16_CONTRACTS',
                Path(__file__).with_name('v40_m16_grouped_contracts.py'))).resolve(strict=True)
    require(sha256(path) == M16_CONTRACTS_SHA256, 'Frozen v40 M16 contracts changed')
    name = '_v41_private_M16_parent_'+uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    try:
        spec.loader.exec_module(value)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    _M16 = value
    return value


def load_parent(path=None):
    return load_m16_parent().load_parent(path)


def parent_contracts_path():
    return load_m16_parent().parent_contracts_path()


def pin(path, expected, pins):
    return load_m16_parent().pin(path, expected, pins)


def recheck_pins(pins):
    return load_m16_parent().recheck_pins(pins)


def artifact(directory, row, pins):
    return load_m16_parent().artifact(directory, row, pins)


def bind_attrs(rows, logical, *, output):
    return load_m16_parent().bind_attrs(rows, logical, output=output)


def contracts(inputs, outputs):
    def read(rows, names):
        require(isinstance(rows, list) and [r['name'] for r in rows] == list(names), 'Exact named IO order required')
        result = {}
        for row in rows:
            shape = tuple(row['shape'])
            require(row['dtype'] == 'float32' and len(shape) == 4
                and all(type(v) is int and v > 0 for v in shape)
                and math.prod(shape)*4 <= UINT32_MAX, 'Fixed FP32 CF rank4 required')
            result[row['name']] = dict(name=row['name'], shape=shape,
                logical_shape=list(shape), logical_dtype='float32')
        return result
    ins, outs = read(inputs, INPUT_NAMES), read(outputs, OUTPUT_NAMES)
    b, c, d, m = ins['net']['shape']
    require(b in (1, 2) and 4 <= d <= 7 and c == 384 and m == 16,
            'Only complete B1/B2 D4..7 M16 online groups supported; D1 is separate')
    require(ins['inp']['shape'] == (b, 384, d, m) and ins['corr']['shape'] == (b, 882, d, m), 'Input dimensions differ')
    require(outs['net_out']['shape'] == (b, 384, d, m)
            and outs['delta']['shape'] == outs['weight']['shape'] == (b, 2, d, m), 'Output dimensions differ')
    return ins, outs


def model_key(inputs, outputs):
    ins, _ = contracts(inputs, outputs)
    b, _, d, m = ins['net']['shape']
    return f'B{b}_D{d}_M{m}'


def validate_export_manifest(path, expected_sha256, *, parent_manifest=None,
                             exporter_sha256=None, allow_fp32_rejected_diagnostic=False,
                             verify_source_files=False):
    if expected_sha256 == EXPORT_MANIFEST_SHA256:
        return load_m16_parent().validate_export_manifest(path, expected_sha256,
            verify_source_files=verify_source_files)
    require(parent_manifest is not None and exporter_sha256 == ADDITION_EXPORTER_SHA256,
            'Explicit v41 exporter pin and frozen parent manifest required')
    pins = {}; path = pin(path, expected_sha256, pins)
    row = json.loads(path.read_text())
    require(row['schema'] == EXPORT_SCHEMA and row['script_sha256'] == exporter_sha256
        and row['source_M16_manifest_sha256'] == EXPORT_MANIFEST_SHA256
        and row['parent_exporter_sha256'] == load_m16_parent().EXPORTER_SHA256
        and row['v37_exporter_sha256'] == load_m16_parent().PARENT_EXPORTER_SHA256
        and row['original_run_sha256'] == parent_manifest['original_run_sha256'], 'New export ancestry differs')
    accepted = row['strict_FP32_gate_passed'] is True
    require(row['status'] == ('completed_missing_M16_source_numerical_accepted_not_adopted' if accepted
        else 'completed_missing_M16_source_numerical_rejected') and (accepted or allow_fp32_rejected_diagnostic is True),
        'Rejected new CPU/ORT ancestry requires explicit diagnostic flag')
    require(row['candidate_accepted'] is False and row['outputs_finite'] is True
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
        and row['original_full_Update_reexecuted'] is False
        and row['source_member_padding'] is False and row['neighbor_or_edge_removal'] is False,
        'New export weights/gates/cleanup differ')
    require(set(row['models']) == set(NEW_KEYS) and len(row['cases']) == 2, 'Exactly two missing online models required')
    parent_case = next(c for c in parent_manifest['cases'] if c['observation_id'] == 34)
    gates = []
    for key in NEW_KEYS:
        model = row['models'][key]
        require(model_key(model['inputs'], model['outputs']) == key, 'New model IO differs')
        d = model['degree']
        require(model['batch'] == 1 and model['patches'] == 16 and d in (4, 6), 'New complete source shape differs')
        group_index, group = next((i, g) for i, g in enumerate(parent_case['groups']) if g['model_key'] == f'B2_D{d}_M16')
        binding = model['source_binding']
        require(binding['parent_manifest_sha256'] == EXPORT_MANIFEST_SHA256
            and binding['parent_model_key'] == group['model_key'] and binding['observation_id'] == 34
            and binding['original_call_index'] == 34 and binding['parent_group_index'] == group_index
            and binding['selected_source_slot'] == group['topology']['source_slots'][0]
            and binding['selected_target_slots'] == group['topology']['target_slots'][0],
            'New graph is not the predeclared first complete source')
        case = next(c for c in row['cases'] if c['model_key'] == key)
        require(case['source_binding'] == binding and case['edges'] == d*16,
                'Case/model source binding differs')
        covered = {v['parameter'] for entry in model['parameter_coverage']['mapping'] for v in entry['original_matches']}
        require(model['parameter_coverage']['original_parameter_count'] == 50
            and model['parameter_coverage']['all_original_parameters_covered'] is True
            and model['numerical_validation_completed'] is True
            and covered == set(row['original_Update_parameters']), 'All fifty original parameters required')
        for label in ('Torch_vs_original_group_gold', 'ORT_vs_original_group_gold', 'ORT_vs_grouped_Torch'):
            require(set(case[label]) == {'net', 'delta', 'weight'}, 'Three exact output comparisons required')
            for comparison in case[label].values():
                require(comparison['finite'] is True and comparison['rtol'] == 1e-4 and comparison['atol'] == 1e-5,
                        'Original FP32 numerical gate changed')
                if accepted:
                    require(comparison['gate_passed'] is True and comparison['failed_elements'] == 0,
                            'New accepted declaration contradicts element gate')
                gates.append(comparison['gate_passed'] is True)
    require(row['strict_comparison_count'] == 18 and accepted == all(gates), 'All new CPU/ORT comparisons required')
    if verify_source_files:
        for source, digest in row['source_pins'].items():
            pin(source, digest, pins)
    return row, pins


def validate_compilation(path, expected_sha256, manifest, manifest_sha256, key, *,
                         compiler_path, compiler_sha256, contracts_path, contracts_sha256):
    if key in REUSED_KEYS:
        require(manifest_sha256 == EXPORT_MANIFEST_SHA256 and compiler_sha256 == M16_COMPILER_SHA256,
                'Reused graph must retain its original v40 compiler/manifest')
        return load_m16_parent().validate_compilation(path, expected_sha256, manifest, manifest_sha256, key,
            compiler_path=compiler_path, compiler_sha256=compiler_sha256,
            contracts_path=contracts_path, contracts_sha256=contracts_sha256)
    require(key in NEW_KEYS, 'Unknown online model')
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
        and row['configuration'] == CONFIGURATION, 'New compile ancestry/shape/config differs')
    require(row['released'] is True and not row.get('release_error')
        and row['sources_and_inputs_unchanged'] is True and row['source_pins'] == row['source_hashes_after']
        and row['candidate_accepted'] is False and row['model_inference_executed'] is False
        and row['NPU_execution_or_placement_validated'] is False
        and row['export_strict_FP32_gate_passed'] is manifest['strict_FP32_gate_passed'],
        'New compiler cleanup/rejection/scope differs')
    if manifest['strict_FP32_gate_passed'] is False:
        require(row['diagnostic_allow_fp32_rejected'] is True, 'Rejected CPU lineage requires explicit compiler diagnostic')
    return row, artifact(path.parent, row['output'], pins), pins


def validate_compile_index(path, expected_sha256, parent_manifest, additions_manifest, *,
        additions_manifest_sha256, parent_compiler, addition_compiler, addition_compiler_sha256,
        m16_contracts, online_contracts, online_contracts_sha256, verify_artifacts=True):
    """Exact online eight-model catalogue. Metadata-only preflight never runs SDK."""
    pins = {}; path = pin(path, expected_sha256, pins); row = json.loads(path.read_text())
    require(row['schema'] == INDEX_SCHEMA and row['status'] == 'staged_eight_online_shapes_diagnostic'
        and row['parent_manifest_sha256'] == EXPORT_MANIFEST_SHA256
        and row['additions_manifest_sha256'] == additions_manifest_sha256
        and row['candidate_accepted'] is False and set(row['models']) == set(MODEL_KEYS),
        'Exact six reused plus two new online models required')
    specs, compiled_rows = {}, {}
    for key in MODEL_KEYS:
        entry = row['models'][key]; old = key in REUSED_KEYS
        require(entry['lineage'] == ('v40' if old else 'v41'), 'Per-model original lineage differs')
        manifest = parent_manifest if old else additions_manifest
        manifest_sha = EXPORT_MANIFEST_SHA256 if old else additions_manifest_sha256
        model = manifest['models'][key]
        require(model_key(model['inputs'], model['outputs']) == key, 'Index model shape differs')
        directory = Path(entry['compile_dir'])
        require(directory.is_absolute() and entry['model']['file'] == 'block.rknn'
            and type(entry['model']['bytes']) is int and 0 < entry['model']['bytes'] <= UINT32_MAX,
            'Explicit fixed compiled model descriptor required')
        if verify_artifacts:
            compiled, model_path, extra = validate_compilation(directory/'compile.json', entry['compile_sha256'],
                manifest, manifest_sha, key, compiler_path=parent_compiler if old else addition_compiler,
                compiler_sha256=M16_COMPILER_SHA256 if old else addition_compiler_sha256,
                contracts_path=m16_contracts if old else online_contracts,
                contracts_sha256=M16_CONTRACTS_SHA256 if old else online_contracts_sha256)
            require(compiled['output'] == entry['model'], 'Index model bytes/SHA disagree with compile')
            pins.update(extra); compiled_rows[key] = compiled
        else:
            model_path = directory/entry['model']['file']
        specs[key] = dict(inputs=model['inputs'], outputs=model['outputs'], model_bytes=entry['model']['bytes'],
            model_path=str(model_path), model_sha256=entry['model']['sha256'], compile_sha256=entry['compile_sha256'],
            export_strict_FP32_gate_passed=manifest['strict_FP32_gate_passed'], lineage=entry['lineage'])
    require(sum(v['model_bytes']+1 for v in specs.values()) <= 256 << 20, 'Online model buffer catalogue exceeds 256MiB')
    return row, specs, compiled_rows, pins
