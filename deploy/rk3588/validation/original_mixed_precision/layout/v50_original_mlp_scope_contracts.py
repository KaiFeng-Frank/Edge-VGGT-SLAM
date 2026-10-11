"""Pure source/config gates for the complete original VGGT pair.

No model, compiler, calibration, SDK, or ORT executor is imported or invoked.
Configuration checks prove edit scope, never numerical equivalence or the
precision inside a compiler fusion. Read actual compiled/profile evidence too.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PAIR_GRAPH_SHA = '2cf1f4de19c91b42b5bf685ab5e949bd98b08a64fc2e78b7f0c8f07992a009fc'
PAIR_EXPORT_SHA = '149450ffc675ce59950c6f89cf84adb41e388cd050eeb731b6d09a6fe16e9fe1'
CHECKPOINT_SHA = 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
PROTOBUF_SOURCE_SHA = 'eec80d60778be4d49524475b95fb05e5d1f917d67d815785a075dceeb29c1f53'
MLP_SUFFIXES = ('mlp.fc1.matmul', 'mlp.fc1', 'gelu_scaled_input', 'gelu_erf',
                'gelu_erf_plus_one', 'gelu_before_half', 'gelu',
                'mlp.fc2.matmul', 'mlp.fc2')
ROLES = ('frame0', 'global0')
RTOL, ATOL = 1e-4, 1e-5


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


def pin_graph(path):
    """Strong whole-file source gate, including all nodes and all initializer bytes."""
    require(sha(path) == PAIR_GRAPH_SHA, 'Original pair ONNX changed; no graph surgery is authorized')
    require(Path(path).stat().st_size == 101700877, 'Original pair byte count changed')
    return {'whole_original_graph_SHA_and_bytes_match': True,
            'all_source_nodes_weights_and_layers_preserved_by_whole_file_pin': True,
            'candidate_accepted': False,
            'compiler_precision_or_numerical_equivalence_proven': False}


def check_cfg_edit(before, after, requested_custom_layers):
    """Check a parsed, actual generated cfg edit without inventing quantization ranges.

The caller must SHA-pin both input files and use a YAML loader that rejects
duplicate mapping keys. This function never edits a cfg. All SDK-generated
parameters and every other top-level setting remain semantically exact.
Only existing generated tensor catalog keys may receive a precision request.
"""
    require(type(before) is dict and type(after) is dict, 'Configuration mappings required')
    require(set(before) == set(after), 'Generated configuration top-level fields changed')
    require('custom_quantize_layers' in before and type(before['custom_quantize_layers']) is dict,
            'Generated custom_quantize_layers section required')
    require(type(requested_custom_layers) is dict and requested_custom_layers,
            'An explicit nonempty precision map is required')
    require(type(before.get('quantize_parameters')) is dict and before['quantize_parameters'],
            'Actual SDK generated tensor catalog/ranges required')
    require(canonical({k: v for k, v in before.items() if k != 'custom_quantize_layers'}) ==
            canonical({k: v for k, v in after.items() if k != 'custom_quantize_layers'}),
            'Generated parameters/ranges/structure were modified')
    require(canonical(after['custom_quantize_layers']) == canonical(requested_custom_layers),
            'Actual precision map differs from the reviewed requested map')
    catalog = before['quantize_parameters']
    for name, dtype in requested_custom_layers.items():
        require(type(name) is str and name in catalog, 'Unknown generated tensor selector: ' + str(name))
        require(dtype in ('int8', 'float16'), 'This diagnostic permits only INT8/FP16 precision requests')
    return {'only_custom_quantize_layers_changed': True,
            'quantize_parameters_and_other_fields_unchanged': True,
            'requested_existing_catalog_keys': list(requested_custom_layers),
            'integer_requests': [k for k, v in requested_custom_layers.items() if v == 'int8'],
            'FP16_requests': [k for k, v in requested_custom_layers.items() if v == 'float16'],
            'exact_original_MLP_only_precision_proven': False,
            'scope_label': 'MLP-focused precision allocation diagnostic',
            'candidate_accepted': False,
            'runtime_accuracy_or_speed_proven': False}


def check_diagnostic_declarations(row):
    """Prevent source/config success from being promoted to numerical acceptance."""
    require(type(row) is dict, 'Diagnostic record mapping required')
    require(row.get('candidate_accepted') is False, 'Precision diagnostic must remain unadopted')
    require(row.get('real_time_validated') is False, 'Component diagnostic cannot prove fresh-pose real time')
    require(row.get('whole_model_quality_validated') is False, 'Component diagnostic cannot prove whole-model quality')
    require(canonical(row.get('rtol')) == canonical(RTOL) and
            canonical(row.get('atol')) == canonical(ATOL), 'Original fixed FP32 tolerance must not be changed')
    require(row.get('original_Torch_FP32_reference_is_sole_numerical_reference') is True,
            'Original heldout Torch output is the sole numerical reference')
    require(row.get('historical_FP32_rejections_retained') is True,
            'Historical strict numerical rejections must stay visible')
    return {'diagnostic_scope_only': True, 'candidate_accepted': False,
            'original_fixed_FP32_thresholds_preserved': True}
