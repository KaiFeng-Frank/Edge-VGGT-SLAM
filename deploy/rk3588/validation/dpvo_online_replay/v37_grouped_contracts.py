"""Research-only lineage/layout contract for complete M48 DPVO Update groups.

No Torch, compiler or SDK import. DPVO topology attribution: Princeton-VL/DPVO
0ac95b656d1fda91c271d2a106460d19ad966fc7, adjacent MIT LICENSE.
The existing CPU/ORT FP32 rejection is retained; no deployment is accepted.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import re

INPUT_NAMES = ('net', 'inp', 'corr')
OUTPUT_NAMES = ('net_out', 'delta', 'weight')
EXPORTER_SHA256 = '473bd9441c983c08aa3ab3a97a7f14b82da8f3d508690403b08468ba25d2dc72'
EXPORT_SCHEMA = 'v36_full_M48_grouped_Update_CPU_ONNX_diagnostic'
COMPILE_SCHEMA = 'v37_grouped_Update_RKNN_compilation_diagnostic'
CONFIGURATION = dict(target_platform='rk3588', float_dtype='float16',
                     optimization_level=3, do_quantization=False)
ABI_SHA256 = '37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608'
HEADER_SHA256 = 'c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a'
LIBRARY_SHA256 = 'd31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'
LOG_HELPER_SHA256 = 'a5af6d39b2d19471255fc86aa42c9aa9b0247e428715c23ffd4ba79b184ad865'
LOG_PARENT_SHA256 = '322021ade54fb9035b7ce80652ec90169938a417fd2192528d28a805a83ed25e'
ADAPTER_SHA256 = '0f427739037e2178dddc4631a32fb602ac0a2ddc259e48e147028604ae6733f1'
UINT32_MAX = 2**32-1


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def pin(path, expected, pins):
    path = Path(path).resolve(strict=True)
    require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected), 'Required SHA is invalid')
    require(sha256(path) == expected, 'Source/model SHA differs: '+str(path))
    pins[str(path)] = expected
    return path


def recheck_pins(pins):
    after = {path: sha256(path) for path in pins}
    require(after == pins, 'Pinned source/input bytes changed')
    return after


def artifact(directory, row, pins):
    directory = Path(directory).resolve(strict=True)
    relative = Path(row['file'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'Safe relative artifact path required')
    path = pin(directory/relative, row['sha256'], pins)
    require(path.is_relative_to(directory) and type(row['bytes']) is int
            and path.stat().st_size == row['bytes'] > 0, 'Artifact path/bytes differ')
    return path


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
    require(1 <= b <= 64 and c == 384 and 1 <= d <= 25 and m == 48, 'Complete original M48 true-B group required')
    require(ins['inp']['shape'] == (b, 384, d, m) and ins['corr']['shape'] == (b, 882, d, m), 'Input group dimensions differ')
    require(outs['net_out']['shape'] == (b, 384, d, m)
            and outs['delta']['shape'] == outs['weight']['shape'] == (b, 2, d, m), 'Output group dimensions differ')
    return ins, outs


def model_key(inputs, outputs):
    ins, _ = contracts(inputs, outputs)
    b, _, d, m = ins['net']['shape']
    return f'B{b}_D{d}_M{m}'


def validate_export_manifest(path, expected_sha256, *, allow_fp32_rejected_diagnostic=False,
                             verify_source_files=False):
    pins = {}
    path = pin(path, expected_sha256, pins)
    report = json.loads(path.read_text())
    require(report['schema'] == EXPORT_SCHEMA and report['script_sha256'] == EXPORTER_SHA256,
            'Frozen complete-group exporter required')
    strict = report['strict_FP32_gate_passed']
    require(type(strict) is bool, 'Explicit export FP32 result required')
    expected_status = ('completed_grouped_Update_numerical_accepted_not_adopted' if strict
                       else 'completed_grouped_Update_numerical_rejected')
    require(report['status'] == expected_status and (strict or allow_fp32_rejected_diagnostic),
            'CPU/ORT rejection requires explicit diagnostic-allow-fp32-rejected')
    require(report['candidate_accepted'] is False and report['original_parameters_unchanged'] is True
            and report['original_weights_before'] == report['original_weights_after']
            and len(report['original_Update_parameters']) == 50
            and report['outputs_finite'] is True and report['weight_domain_valid'] is True
            and report['cleanup_errors'] == [] and report['affinity_restored'] is True,
            'Original parameter/finite/domain/cleanup contract failed')
    sources = report['source_pins']
    require(isinstance(sources, dict) and sources and sources == report['source_hashes_after']
            and report['sources_and_inputs_unchanged'] is True, 'Export source provenance differs')
    for source, digest in sources.items():
        require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest), 'Declared source SHA invalid')
        if verify_source_files:
            source = Path(source)
            pin(source if source.is_absolute() else path.parent/source, digest, pins)
    require([row['observation_id'] for row in report['cases']] == [13, 127, 191], 'All predeclared full-graph cases required')
    models = report['models']
    require(isinstance(models, dict) and models, 'No actual shape graphs')
    for key, row in models.items():
        require(model_key(row['inputs'], row['outputs']) == key and row['padding'] is False
                and row['external_data_used'] is False and row['numerical_validation_completed'] is True,
                'Model key/layout/reference contract differs')
        coverage = row['parameter_coverage']
        require(coverage['original_parameter_count'] == 50 and coverage['all_original_parameters_covered'] is True,
                'All original 50 parameter bytes required')
    for case in report['cases']:
        require(case['Torch_domain'] == case['ORT_domain'] == dict(finite=True, weight_domain_valid=True),
                'Case finite/domain result differs')
        require(case['degree_plan']['all_edges_exactly_once'] is True
                and case['degree_plan']['original_edge_order_restored'] is True
                and case['degree_plan']['padding'] is False, 'Incomplete or padded source topology')
        for group in case['groups']:
            require(group['model_key'] in models, 'Missing actual group model')
    return report, pins


def validate_compilation(path, expected_sha256, manifest, manifest_sha256, key, *,
                         compiler_path, compiler_sha256, contracts_path, contracts_sha256):
    pins = {}
    path = pin(path, expected_sha256, pins)
    pin(compiler_path, compiler_sha256, pins)
    pin(contracts_path, contracts_sha256, pins)
    row = json.loads(path.read_text())
    require(row['schema'] == COMPILE_SCHEMA and row['status'] == 'compiled_group_diagnostic_not_runtime_validated',
            'Unexpected grouped compilation status')
    require(row['script_sha256'] == compiler_sha256 and row['contracts_sha256'] == contracts_sha256
            and row['manifest_sha256'] == manifest_sha256 and row['model_key'] == key
            and row['input']['sha256'] == manifest['models'][key]['model']['sha256'], 'Compile/export source chain differs')
    require(row['configuration'] == CONFIGURATION and row['released'] is True
            and not row.get('release_error') and row['sources_and_inputs_unchanged'] is True
            and row['source_pins'] == row['source_hashes_after'] and row['candidate_accepted'] is False,
            'Compiler configuration/cleanup/source contract differs')
    require(row['export_strict_FP32_gate_passed'] is manifest['strict_FP32_gate_passed']
            and (manifest['strict_FP32_gate_passed'] or row['diagnostic_allow_fp32_rejected'] is True),
            'Inherited numerical rejection was not retained')
    model = artifact(path.parent, row['output'], pins)
    return row, model, pins


def bind_attrs(rows, logical, *, output):
    require(len(rows) == 3 and {row['name'] for row in rows} == set(logical), 'SDK operand names/count differ')
    result = []
    for index, row in enumerate(rows):
        require(type(row['index']) is int and row['index'] == index, 'SDK operand indices differ')
        shape = logical[row['name']]['shape']
        native, fmt = tuple(row['shape']), row['format']
        if fmt in (0, 3):
            require(native == shape, 'SDK original CF shape differs')
            permutation = None
        elif fmt == 1:
            require(native == (shape[0], shape[2], shape[3], shape[1]), 'SDK NHWC shape differs')
            permutation = (0, 2, 3, 1)
        else:
            raise ValueError('Unsupported SDK packed layout')
        dtype = row['type']
        require(type(dtype) is int and dtype in (0, 1), 'Only native FP32/FP16 supported')
        quant = row['quantization_type']
        neutral = quant == 2 and row['scale'] == 1 and row['zero_point'] == 0 and row['fractional_length'] == 0
        require(quant == 0 or neutral, 'Nonidentity quantization rejected')
        elems = math.prod(shape)
        require(row['n_elems'] == elems and row['size'] == elems*(4 if dtype == 0 else 2), 'SDK logical elements/bytes differ')
        stride = row.get('size_with_stride', 0)
        require(type(stride) is int and 0 <= stride <= UINT32_MAX, 'Invalid native stride footprint')
        require((stride == 0 or stride >= row['size']) if output else stride in (0, row['size']),
                'Undersized output stride or unsupported padded input')
        result.append(dict(copy.deepcopy(logical[row['name']]), index=index,
            sdk_shape=list(native), sdk_format=fmt, sdk_type=dtype, sdk_native_bytes=row['size'],
            sdk_size_with_stride=stride, sdk_quantization_type=quant, neutral_affine_metadata=bool(neutral),
            native_output_padding_bytes=max(0, stride-row['size']) if output else 0,
            input_permutation=permutation if not output else None,
            output_permutation=(0, 3, 1, 2) if output and fmt == 1 else None,
            transport_format=fmt, transport_bytes=elems*4, returned_FP32_bytes=elems*4,
            pass_through=0, want_float=1))
    return result
