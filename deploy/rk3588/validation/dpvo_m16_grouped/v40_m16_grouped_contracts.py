"""Research-only contract for the eight actual v39 M16 complete-source graphs.

DPVO: Princeton-VL/DPVO 0ac95b656d1fda91c271d2a106460d19ad966fc7,
MIT Copyright (c) 2022 Princeton University. No model/compiler/SDK execution.
Generic byte pinning and native tensor layouts reuse the exact frozen v37
contract. M48 export acceptance is never substituted for the v39 M16 result.
"""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True

PARENT_CONTRACTS_SHA256 = 'aa9e6af938f64ae4a4a82ffc84b487321fff208547703b0d65646c72b96d85c9'
PARENT_COMPILER_SHA256 = 'fb3a6e9eed9a4bc3f8490f0cf61c7ef5d5f4f4eff065bb9d9f057de97fb9cdd1'
EXPORT_MANIFEST_SHA256 = '5a9d6c571b672fbb174cf6adf9f8f578689fdd9e5f58ee4ec327320b55b62bfb'
EXPORTER_SHA256 = '0dc3d9e49645cd3e472fa344434bca0aa9360b62d6d37fc3e4d71fefa67d0842'
PARENT_EXPORTER_SHA256 = '473bd9441c983c08aa3ab3a97a7f14b82da8f3d508690403b08468ba25d2dc72'
ORIGINAL_RUN_SHA256 = 'dd9f37524b999c1e586182a92c226481fcfc7f4da851d5e88d62d82be1029ecd'
POLICY_SHA256 = 'cc6a31f89bdcff0b43c0d51266ad62c48d0e297346845a98d42bad5c8874ef4a'
INPUT_MANIFEST_SHA256 = 'ad2c7eadef84cb4ecc550e2b4dc0f27611cb116bcefaf7eb697620646bf4bb10'
INPUT_NAMES = ('net', 'inp', 'corr')
OUTPUT_NAMES = ('net_out', 'delta', 'weight')
EXPORT_SCHEMA = 'v39_full_M16_grouped_Update_CPU_ONNX_diagnostic'
COMPILE_SCHEMA = 'v40_M16_grouped_Update_RKNN_compilation_diagnostic'
CONFIGURATION = dict(target_platform='rk3588', float_dtype='float16',
                     optimization_level=3, do_quantization=False)
ABI_SHA256 = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
HEADER_SHA256 = 'c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a'
LIBRARY_SHA256 = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
LOG_HELPER_SHA256 = 'a5af6d39b2d19471255fc86aa42c9aa9b0247e428715c23ffd4ba79b184ad865'
LOG_PARENT_SHA256 = '322021ade54fb9035b7ce80652ec90169938a417fd2192528d28a805a83ed25e'
ADAPTER_SHA256 = '0f427739037e2178dddc4631a32fb602ac0a2ddc259e48e147028604ae6733f1'
UINT32_MAX = 2**32-1
# This is the saved, independently verified catalogue, not a predicted shape set.
ACTUAL_SHAPES = ((2, 4), (2, 5), (2, 6), (2, 7), (3, 6), (1, 5), (5, 6), (1, 7))
MODEL_KEYS = tuple('B%d_D%d_M16' % shape for shape in ACTUAL_SHAPES)
CASE_IDENTITY = ((34, 34, 704), (127, 138, 800), (191, 202, 800))
_PARENT = None
_PARENT_PATH = None


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_parent(path=None):
    """Bind the frozen generic v37 functions, without changing their globals.

    Runtime callers may supply the ancestor path explicitly. Default deployment
    keeps the original v37 contract adjacent; no private absolute path is needed.
    """
    global _PARENT, _PARENT_PATH
    if path is None and _PARENT is not None:
        return _PARENT
    path = Path(path or os.environ.get('V40_PARENT_CONTRACTS',
                Path(__file__).with_name('v37_grouped_contracts.py'))).resolve(strict=True)
    require(sha256(path) == PARENT_CONTRACTS_SHA256, 'Frozen v37 generic contract SHA differs')
    name = '_v40_M16_generic_parent_' + PARENT_CONTRACTS_SHA256
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
    _PARENT, _PARENT_PATH = sys.modules[name], path
    return _PARENT


def parent_contracts_path():
    load_parent()
    return _PARENT_PATH


def pin(path, expected, pins):
    return load_parent().pin(path, expected, pins)


def recheck_pins(pins):
    return load_parent().recheck_pins(pins)


def artifact(directory, row, pins):
    return load_parent().artifact(directory, row, pins)


def bind_attrs(rows, logical, *, output):
    # Same explicit CF/NHWC mappings, float types, neutral metadata and output-
    # only stride allowance as actually tested v37. No new layout assumption.
    return load_parent().bind_attrs(rows, logical, output=output)


def contracts(inputs, outputs):
    def read(rows, names):
        require(isinstance(rows, list) and [row['name'] for row in rows] == list(names), 'Exact named IO order required')
        result = {}
        for row in rows:
            shape = tuple(row['shape'])
            require(row['dtype'] == 'float32' and len(shape) == 4
                    and all(type(v) is int and v > 0 for v in shape)
                    and math.prod(shape)*4 <= UINT32_MAX, 'Positive fixed FP32 CF rank4 required')
            result[row['name']] = dict(name=row['name'], shape=shape, logical_shape=list(shape), logical_dtype='float32')
        return result
    ins, outs = read(inputs, INPUT_NAMES), read(outputs, OUTPUT_NAMES)
    b, c, d, m = ins['net']['shape']
    require((b, d) in ACTUAL_SHAPES and c == 384 and m == 16, 'Only eight actual M16 true-B groups supported')
    require(ins['inp']['shape'] == (b, 384, d, m) and ins['corr']['shape'] == (b, 882, d, m), 'Input group dimensions differ')
    require(outs['net_out']['shape'] == (b, 384, d, m)
            and outs['delta']['shape'] == outs['weight']['shape'] == (b, 2, d, m), 'Output group dimensions differ')
    return ins, outs


def model_key(inputs, outputs):
    ins, _ = contracts(inputs, outputs)
    b, _, d, m = ins['net']['shape']
    return f'B{b}_D{d}_M{m}'


def _fixed_comparisons(rows):
    require(set(rows) == {'net', 'delta', 'weight'}, 'All three canonical comparisons required')
    for row in rows.values():
        require(row['finite'] is True and row['gate_passed'] is True and row['failed_elements'] == 0
                and row['rtol'] == 1e-4 and row['atol'] == 1e-5, 'Original fixed FP32 component gate differs')


def validate_export_manifest(path, expected_sha256, *, allow_fp32_rejected_diagnostic=False,
                             verify_source_files=False):
    """Exact v39 source export, host-safe with board absolute pins declared.

    allow_fp32_rejected_diagnostic preserves the old call signature, but cannot
    authorize another manifest: only the pinned accepted v39 bytes are allowed.
    """
    require(expected_sha256 == EXPORT_MANIFEST_SHA256, 'Only actual v39 M16 export manifest accepted')
    pins = {}
    path = pin(path, expected_sha256, pins)
    report = json.loads(path.read_text())
    require(report['schema'] == EXPORT_SCHEMA and report['script_sha256'] == EXPORTER_SHA256
            and report['parent_exporter_sha256'] == PARENT_EXPORTER_SHA256
            and report['original_run_sha256'] == ORIGINAL_RUN_SHA256
            and report['policy_sha256'] == POLICY_SHA256
            and report['input_manifest_sha256'] == INPUT_MANIFEST_SHA256, 'M16 exporter/capture lineage differs')
    require(report['status'] == 'completed_grouped_Update_numerical_accepted_not_adopted'
            and report['strict_FP32_gate_passed'] is True and report['candidate_accepted'] is False
            and report['strict_FP32_thresholds'] == dict(rtol=1e-4, atol=1e-5)
            and report['strict_FP32_thresholds_relaxed'] is False and report['output_clipping'] is False,
            'Actual CPU/ORT acceptance without adoption required')
    weights, parameters = report['original_weights_before'], report['original_Update_parameters']
    require(report['original_parameters_unchanged'] is True and len(weights) == 94
            and weights == report['original_weights_after'] and len(parameters) == 50
            and {'update.'+key: value for key, value in parameters.items()}
            == {key: value for key, value in weights.items() if key.startswith('update.')},
            'Original 94 state / all 50 Update parameter identities differ')
    require(report['outputs_finite'] is True and report['weight_domain_valid'] is True
            and report['cleanup_errors'] == [] and report['affinity_restored'] is True
            and report['ORT_session_references_released'] is True and report['patches_per_frame'] == 16
            and report['source_member_padding'] is False and report['neighbor_or_edge_removal'] is False,
            'M16 finite/domain/lifecycle/fullgraph contract failed')
    sources = report['source_pins']
    require(isinstance(sources, dict) and sources and sources == report['source_hashes_after']
            and report['sources_and_inputs_unchanged'] is True, 'Export source provenance differs')
    for source, digest in sources.items():
        require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest), 'Declared source SHA invalid')
        if verify_source_files:
            source = Path(source)
            pin(source if source.is_absolute() else path.parent/source, digest, pins)
    models = report['models']
    require(isinstance(models, dict) and set(models) == set(MODEL_KEYS), 'Exact eight actual M16 graph catalogue required')
    for key, row in models.items():
        ins, _ = contracts(row['inputs'], row['outputs'])
        b, _, d, m = ins['net']['shape']
        require(model_key(row['inputs'], row['outputs']) == key
                and (row['batch'], row['degree'], row['patches']) == (b, d, m)
                and row['padding'] is False and row['external_data_used'] is False
                and row['numerical_validation_completed'] is True, 'Actual model/layout/reference differs')
        coverage = row['parameter_coverage']
        require(coverage['original_parameter_count'] == 50 and coverage['all_original_parameters_covered'] is True,
                'Original 50 parameter byte coverage required')
    require(len(report['cases']) == 3, 'All actual three captured fullgraphs required')
    comparisons = 0
    for case, identity in zip(report['cases'], CASE_IDENTITY):
        observation, call_index, edges = identity
        require((case['observation_id'], case['frame_index'], case['original_call_index'], case['edges'])
                == (observation, observation, call_index, edges), 'Actual M16 fullgraph capture selection differs')
        require(case['Torch_domain'] == case['ORT_domain'] == dict(finite=True, weight_domain_valid=True),
                'Case finite/domain differs')
        plan, groups = case['degree_plan'], case['groups']
        require(plan['edges'] == edges and plan['patches'] == 16
                and plan['all_edges_exactly_once'] is True and plan['original_edge_order_restored'] is True
                and plan['padding'] is False and len(groups) == plan['neural_calls'] == 4,
                'Incomplete or padded M16 source graph')
        for label in ('Torch_vs_original_full_gold', 'ORT_vs_original_full_gold', 'ORT_vs_grouped_Torch'):
            _fixed_comparisons(case[label]); comparisons += 3
        restored_edges = 0
        for index, group in enumerate(groups):
            require(group['index'] == index and group['model_key'] in models, 'Actual complete group model missing')
            model, topo = models[group['model_key']], group['topology']
            b, d, m = model['batch'], model['degree'], model['patches']
            require(topo['sources'] == b and topo['degree'] == d and topo['patches'] == m
                    and topo['edges'] == b*d*m and topo['padding'] is False
                    and topo['dependency_groups_split'] is False, 'True source group dependency differs')
            restored_edges += topo['edges']
            for label in ('Torch_vs_original_group_gold', 'ORT_vs_original_group_gold', 'ORT_vs_grouped_Torch'):
                _fixed_comparisons(group[label]); comparisons += 3
        require(restored_edges == edges, 'All actual fullgraph edges required')
    require(comparisons == report['strict_comparison_count'] == 135, 'All original full/group fixed gates required')
    return report, pins


def validate_compilation(path, expected_sha256, manifest, manifest_sha256, key, *,
                         compiler_path, compiler_sha256, contracts_path, contracts_sha256):
    pins = {}
    path = pin(path, expected_sha256, pins)
    pin(compiler_path, compiler_sha256, pins)
    pin(contracts_path, contracts_sha256, pins)
    pin(parent_contracts_path(), PARENT_CONTRACTS_SHA256, pins)
    require(manifest_sha256 == EXPORT_MANIFEST_SHA256 and key in manifest['models'], 'Actual M16 export/key required')
    row = json.loads(path.read_text())
    require(row['schema'] == COMPILE_SCHEMA and row['status'] == 'compiled_group_diagnostic_not_runtime_validated',
            'Unexpected M16 compilation status')
    model = manifest['models'][key]
    require(row['script_sha256'] == compiler_sha256 and row['contracts_sha256'] == contracts_sha256
            and row['parent_compiler_sha256'] == PARENT_COMPILER_SHA256
            and row['parent_contracts_sha256'] == PARENT_CONTRACTS_SHA256
            and row['manifest_sha256'] == manifest_sha256 and row['model_key'] == key
            and row['input']['sha256'] == model['model']['sha256']
            and row['logical_inputs'] == model['inputs'] and row['logical_outputs'] == model['outputs'],
            'M16 compile/export/source/layout chain differs')
    require(row['configuration'] == CONFIGURATION and row['released'] is True
            and not row.get('release_error') and row['sources_and_inputs_unchanged'] is True
            and row['source_pins'] == row['source_hashes_after'] and row['candidate_accepted'] is False
            and row['model_inference_executed'] is False
            and row['NPU_execution_or_placement_validated'] is False
            and row['export_strict_FP32_gate_passed'] is True,
            'M16 compiler configuration/cleanup/provenance/research scope differs')
    output = artifact(path.parent, row['output'], pins)
    return row, output, pins
