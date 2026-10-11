#!/usr/bin/env python3
"""Namespace and connect two frozen original VGGT ONNX blocks; no inference.

The only new computation is the original aggregator's C-order frame-to-global
Reshape. All parent nodes, FP32 parameters, RoPE coefficients and exact GELU
remain unchanged after reversing the namespace. Checker success is structural;
the new sequential graph still needs its own board CPU and RKNN comparisons.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
PROFILE_SHA = '6c3d2cb51af75410188c062e83a26eae4d4c5d69c632a6eeae95993eba004820'
CHECKPOINT_SHA = 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
R3_EXPORTER_SHA = '6724aba806806ec36d8ba097f0c2076f3703e106f23c83d4272ae2e3a3fc2874'
PARENTS = {
    'frame0': {
        'export_sha256': '07d139bc51dc5ab244920020e7728e774ee62f055a464c107da292664ae24eaa',
        'onnx_sha256': '20912c4580eec7d3935900437cbd2733013bd1f92229befac48bad5d860cb1ce',
        'ort_sha256': 'e65ea97ef477709bc0a346658fdf7adeca1ff6e4c8efcbeaaffca3076889b3ca',
        'input_shape': [2, 437, 1024],
    },
    'global0': {
        'export_sha256': 'f851432a0dd8e9c2d66b4dfee2a76dfb674836bfb710a2ccbea09972f5e3f33d',
        'onnx_sha256': 'f67c2210f6a53a7ff88865e4607c6d1760cf23c221941339c204c7414cda2778',
        'ort_sha256': '1929248412b695bffc5e12d89efdca478413c94a80b3ed68b31354df00fbfa87',
        'input_shape': [1, 874, 1024],
    },
}
WEIGHT_NAMES = (
    'norm1.weight', 'norm1.bias', 'attn.qkv.weight', 'attn.qkv.bias',
    'attn.q_norm.weight', 'attn.q_norm.bias', 'attn.k_norm.weight',
    'attn.k_norm.bias', 'attn.proj.weight', 'attn.proj.bias', 'ls1.gamma',
    'norm2.weight', 'norm2.bias', 'mlp.fc1.weight', 'mlp.fc1.bias',
    'mlp.fc2.weight', 'mlp.fc2.bias', 'ls2.gamma',
)
TRANSPOSED = {'attn.qkv.weight', 'attn.proj.weight', 'mlp.fc1.weight', 'mlp.fc2.weight'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(path, expected, pins):
    if sha(path) != expected:
        raise ValueError(f'Frozen SHA mismatch: {path}')
    pins[str(Path(path).resolve())] = expected


def array_metadata(value):
    if not value.flags.c_contiguous or not value.size or not np.isfinite(value).all():
        raise ValueError('Finite nonempty C-contiguous array required')
    return {'shape': list(value.shape), 'dtype': str(value.dtype), 'bytes': value.nbytes,
            'sha256': hashlib.sha256(memoryview(value).cast('B')).hexdigest()}


def _shape(value):
    from onnx import TensorProto
    tensor = value.type.tensor_type
    result = [d.dim_value for d in tensor.shape.dim]
    if tensor.elem_type != TensorProto.FLOAT or len(result) != 3 or any(d <= 0 for d in result):
        raise ValueError('Positive static rank-three FP32 IO required')
    return result


def _validate_parent_model(model):
    import onnx
    if (len(model.graph.input) != 1 or len(model.graph.output) != 1
            or model.functions or model.training_info or model.graph.sparse_initializer
            or model.graph.quantization_annotation):
        raise ValueError('Single-input/output inference graph without external functions required')
    for item in model.graph.initializer:
        if item.data_location == onnx.TensorProto.EXTERNAL or item.external_data:
            raise ValueError('Embedded initializers required')
    names = [item.name for item in model.graph.initializer]
    if len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError('Unique nonempty initializer names required')
    if model.graph.input[0].name in names:
        raise ValueError('Runtime input must not be an initializer')
    node_names = [node.name for node in model.graph.node]
    if len(set(node_names)) != len(node_names) or any(not name for name in node_names):
        raise ValueError('Unique nonempty node names required')
    for node in model.graph.node:
        if node.domain or any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                              for a in node.attribute):
            raise ValueError('Only standard non-control-flow nodes are supported')
    if _shape(model.graph.input[0]) != _shape(model.graph.output[0]):
        raise ValueError('Original block must preserve its rank-three shape')
    onnx.checker.check_model(model, full_check=True)


def _namespaced_graph(model, prefix, *, public_input=False, public_output=False):
    import onnx
    graph = onnx.GraphProto()
    graph.CopyFrom(model.graph)
    overrides = {}
    if public_input:
        overrides[model.graph.input[0].name] = 'input'
    if public_output:
        overrides[model.graph.output[0].name] = 'output'

    def rename(name):
        return overrides.get(name, prefix + name) if name else name

    graph.name = prefix + graph.name
    for node in graph.node:
        node.name = prefix + node.name
        node.input[:] = [rename(name) for name in node.input]
        node.output[:] = [rename(name) for name in node.output]
    for value in list(graph.input) + list(graph.output) + list(graph.value_info) + list(graph.initializer):
        value.name = rename(value.name)
    return graph


def compose_pair(frame_model, global_model):
    """Generic static composer, also usable with dimension-small generated tests."""
    import onnx
    from onnx import helper, numpy_helper
    for model in (frame_model, global_model):
        _validate_parent_model(model)
    fs, gs = _shape(frame_model.graph.input[0]), _shape(global_model.graph.input[0])
    if gs != [1, fs[0] * fs[1], fs[2]]:
        raise ValueError('Global layout must be [1, frames * tokens, channels]')
    if (frame_model.ir_version != global_model.ir_version
            or [x.SerializeToString() for x in frame_model.opset_import]
            != [x.SerializeToString() for x in global_model.opset_import]):
        raise ValueError('Parent IR/opsets must match exactly')
    f = _namespaced_graph(frame_model, 'frame0/', public_input=True)
    g = _namespaced_graph(global_model, 'global0/', public_output=True)
    bridge = helper.make_node('Reshape', [f.output[0].name, 'frame_global_shape'],
                              [g.input[0].name], name='frame_to_global_view', allowzero=0)
    constant = numpy_helper.from_array(np.asarray(gs, np.int64), 'frame_global_shape')
    values = list(f.value_info) + list(g.value_info) + [f.output[0], g.input[0]]
    if len({v.name for v in values}) != len(values):
        raise ValueError('Unexpected parent value-info collision')
    paired = onnx.ModelProto()
    paired.CopyFrom(frame_model)
    paired.graph.CopyFrom(helper.make_graph(
        list(f.node) + [bridge] + list(g.node), 'original_VGGT_frame0_then_global0',
        list(f.input), list(g.output), initializer=list(f.initializer) + list(g.initializer) + [constant],
        value_info=values))
    paired.producer_name = 'VGGT-SLAM2/v47-static-frame-global-pair'
    paired.doc_string = ('Two original complete blocks; parent nodes and weights are namespaced only. '
                         'One original C-order view connects frame output to global input. '
                         'Static composition is not a new numerical or runtime acceptance.')
    onnx.checker.check_model(paired, full_check=True)
    proof = validate_preservation(frame_model, global_model, paired)
    return paired, proof


def validate_preservation(frame_model, global_model, paired):
    """Check every parent node and initializer byte, allowing only namespaces."""
    from onnx import helper, numpy_helper
    fs, gs = _shape(frame_model.graph.input[0]), _shape(global_model.graph.input[0])
    f = _namespaced_graph(frame_model, 'frame0/', public_input=True)
    g = _namespaced_graph(global_model, 'global0/', public_output=True)
    bridge = helper.make_node('Reshape', [f.output[0].name, 'frame_global_shape'],
                              [g.input[0].name], name='frame_to_global_view', allowzero=0)
    expected_nodes = list(f.node) + [bridge] + list(g.node)
    expected_initializers = list(f.initializer) + list(g.initializer) + [
        numpy_helper.from_array(np.asarray(gs, np.int64), 'frame_global_shape')]
    for actual, expected, label in ((paired.graph.node, expected_nodes, 'node'),
                                     (paired.graph.initializer, expected_initializers, 'initializer')):
        if [x.SerializeToString() for x in actual] != [x.SerializeToString() for x in expected]:
            raise ValueError(f'Pair changed a parent {label} beyond namespaces or the single view')
    if ([x.SerializeToString() for x in paired.graph.input] != [x.SerializeToString() for x in f.input]
            or [x.SerializeToString() for x in paired.graph.output] != [x.SerializeToString() for x in g.output]
            or paired.ir_version != frame_model.ir_version
            or [x.SerializeToString() for x in paired.opset_import]
            != [x.SerializeToString() for x in frame_model.opset_import]):
        raise ValueError('Pair changed its declared IO or original IR/opsets')
    return {'parent_nodes_preserved_except_namespaces': True,
            'parent_initializer_protobuf_preserved_except_names': True,
            'original_parent_node_count': len(f.node) + len(g.node),
            'original_parent_initializer_count': len(f.initializer) + len(g.initializer),
            'new_node_count': 1, 'new_nonlearned_initializer_count': 1,
            'new_node': 'frame_to_global_view', 'frame_input_shape': fs,
            'global_output_shape': gs, 'frame_intermediate_exported': False,
            'ir_version': paired.ir_version,
            'opsets': {x.domain: x.version for x in paired.opset_import}}


def _load_parent(block, export_dir, ort_dir, capture_dir, profile, pins):
    import onnx
    from onnx import numpy_helper
    fixed = PARENTS[block]
    export_dir, ort_dir, capture_dir = map(Path, (export_dir, ort_dir, capture_dir))
    require_sha(export_dir / 'export.json', fixed['export_sha256'], pins)
    require_sha(ort_dir / 'check.json', fixed['ort_sha256'], pins)
    report = json.loads((export_dir / 'export.json').read_text())
    ort = json.loads((ort_dir / 'check.json').read_text())
    if (report.get('block') != block or report.get('script_sha256') != R3_EXPORTER_SHA
            or report.get('profile_sha256') != PROFILE_SHA or report.get('original_weights') is not True
            or report.get('input_shape') != fixed['input_shape'] or report.get('parameter_count') != 18
            or report.get('onnx', {}).get('sha256') != fixed['onnx_sha256']):
        raise ValueError('Frozen R3 parent identity mismatch')
    if (ort.get('status') != 'passed' or ort.get('block') != block
            or ort.get('export_sha256') != fixed['export_sha256']
            or ort.get('all_runs_within_tolerance') is not True
            or ort.get('inference_on_board_only') is not True
            or ort.get('session_released') is not True
            or ort.get('successful_session_run_count') != 4
            or ort.get('tolerance', {}).get('rtol') != 1e-4
            or ort.get('tolerance', {}).get('atol') != 1e-5):
        raise ValueError('Completed original board FP32 parent gate is required')
    path = export_dir / report['onnx']['file']
    require_sha(path, fixed['onnx_sha256'], pins)
    if path.stat().st_size != report['onnx']['bytes']:
        raise ValueError('Parent ONNX byte count mismatch')
    model = onnx.load(path, load_external_data=False)
    if _shape(model.graph.input[0]) != fixed['input_shape']:
        raise ValueError('Parent ONNX shape mismatch')
    captured = profile['blocks'][block]
    arrays, weights = {}, {}
    for kind, target in (('tensors', arrays), ('weights', weights)):
        record = captured['artifacts'][kind]
        if record['file'] != f'{block}-{kind}.npz' or record != report['capture_artifacts'][kind]:
            raise ValueError('Parent capture artifact identity mismatch')
        path = capture_dir / record['file']
        require_sha(path, record['sha256'], pins)
        if path.stat().st_size != record['bytes']:
            raise ValueError('Capture artifact size mismatch')
        metadata = captured['arrays' if kind == 'tensors' else 'weights']
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(metadata):
                raise ValueError('Captured keys differ from metadata')
            for key in saved.files:
                value = saved[key]
                if array_metadata(value) != metadata[key]:
                    raise ValueError(f'Captured raw metadata mismatch: {block}/{key}')
                target[key] = value
    if set(weights) != set(WEIGHT_NAMES) or set(arrays) != {'input', 'output', 'pos', 'rope_cos', 'rope_sin'}:
        raise ValueError('Exact original parameter and captured tensor keys required')
    if (report['captured_parameter_metadata'] != captured['weights']
            or report['captured_array_metadata'] != captured['arrays']
            or report['selected_pair'] != profile['selected_pair']
            or report['checkpoint'] != profile['checkpoint']
            or len(report['source_pins']) != 6
            or any(profile['source_hashes'].get(key) != value
                   for key, value in report['source_pins'].items())):
        raise ValueError('Original source/checkpoint/array provenance differs')
    initializer_map = {item.name: item for item in model.graph.initializer}
    bindings = {}
    for key, value in weights.items():
        name = key + '_transposed' if key in TRANSPOSED else key
        actual = numpy_helper.to_array(initializer_map[name])
        expected = np.ascontiguousarray(value.T) if key in TRANSPOSED else value
        if array_metadata(actual) != array_metadata(expected):
            raise ValueError(f'Original parameter not exactly embedded: {block}/{key}')
        bindings[key] = {'initializer': f'{block}/{name}', 'transpose_2d': key in TRANSPOSED,
                         'original': array_metadata(value), 'embedded': array_metadata(actual)}
    return model, report, arrays, bindings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-dir', required=True)
    parser.add_argument('--frame-export-dir', required=True)
    parser.add_argument('--global-export-dir', required=True)
    parser.add_argument('--frame-ort-dir', required=True)
    parser.add_argument('--global-ort-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'v47_original_VGGT_frame0_global0_static_pair',
              'status': 'failed_static_preparation', 'script_sha256': sha(__file__),
              'host_model_inference_executed': False, 'host_ORT_executed': False,
              'board_operations': False, 'compiler_executed': False,
              'candidate_accepted': False, 'paired_board_FP32_gate_passed': None,
              'paired_RKNN_FP32_gate_passed': None, 'real_time_validated': False,
              'whole_model_quality_validated': False, 'production_changed': False,
              'historical_RKNN_FP32_rejections_retained': True}
    pins = {str(Path(__file__).resolve()): report['script_sha256']}
    error = None
    try:
        capture_dir = Path(args.capture_dir)
        require_sha(capture_dir / 'profile.json', PROFILE_SHA, pins)
        profile = json.loads((capture_dir / 'profile.json').read_text())
        if (profile.get('kind') != 'original_VGGT_rectified_stage_and_block_capture'
                or profile.get('status') != 'completed_original_CPU_profile_and_capture'
                or profile.get('checkpoint', {}).get('sha256') != CHECKPOINT_SHA
                or profile.get('original_weights') is not True or profile.get('architecture_changed') is not False
                or profile.get('hooks_removed') is not True or profile.get('cleanup_errors')
                or profile.get('image_size') != 336):
            raise ValueError('Frozen original two-frame336 CPU capture is required')
        frame, fr, fa, fb = _load_parent('frame0', args.frame_export_dir, args.frame_ort_dir,
                                        capture_dir, profile, pins)
        glob, gr, ga, gb = _load_parent('global0', args.global_export_dir, args.global_ort_dir,
                                       capture_dir, profile, pins)
        if (fa['output'].dtype != ga['input'].dtype
                or fa['output'].reshape(1, 874, 1024).tobytes() != ga['input'].tobytes()
                or fa['pos'].reshape(1, 874, 2).tobytes() != ga['pos'].tobytes()
                or any(fa[key].tobytes() != ga[key].tobytes() for key in ('rope_cos', 'rope_sin'))):
            raise ValueError('Captured original frame-global view/positions/RoPE tables differ')
        paired, proof = compose_pair(frame, glob)
        import onnx
        onnx.save_model(paired, out / 'pair.onnx', save_as_external_data=False)
        onnx.checker.check_model(str(out / 'pair.onnx'), full_check=True)
        artifacts = {}
        for filename, value, origin in (
                ('input.npy', fa['input'], 'original frame0 input'),
                ('positions.npy', fa['pos'], 'original frame0 integer RoPE positions; global is C-order view'),
                ('frame-reference.npy', fa['output'], 'original CPU frame0 output'),
                ('output.npy', ga['output'], 'original CPU global0 final output')):
            np.save(out / filename, value, allow_pickle=False)
            metadata = array_metadata(value)
            artifacts[filename] = {'file': filename, 'bytes': (out / filename).stat().st_size,
                                   'sha256': sha(out / filename), 'shape': metadata['shape'],
                                   'dtype': metadata['dtype'], 'tensor_bytes': metadata['bytes'],
                                   'raw_sha256': metadata['sha256'], 'origin': origin}
        parameter_bytes = sum(x['original']['bytes'] for x in list(fb.values()) + list(gb.values()))
        report.update(status='exported_and_structurally_checked_not_numerically_validated',
                      onnx={'file': 'pair.onnx', 'bytes': (out / 'pair.onnx').stat().st_size,
                            'sha256': sha(out / 'pair.onnx')},
                      artifacts=artifacts, graph_preservation=proof,
                      op_counts=dict(Counter(node.op_type for node in paired.graph.node)),
                      node_count=len(paired.graph.node), initializer_count=len(paired.graph.initializer),
                      input_name='input', output_name='output', input_shape=[2, 437, 1024],
                      output_shape=[1, 874, 1024], original_parameter_tensors=36,
                      original_parameter_FP32_bytes=parameter_bytes,
                      original_parameter_bindings={'frame0': fb, 'global0': gb},
                      original_weights_preserved=True, source_pins=profile['source_hashes'],
                      checkpoint=profile['checkpoint'], selected_pair=profile['selected_pair'],
                      architecture=fr['architecture'], parent_graphs=PARENTS,
                      parent_board_FP32_gates_passed=True, profile_sha256=PROFILE_SHA,
                      captured_connection={'frame_output_equals_global_input_after_view': True,
                                           'position_order_equals_after_view': True,
                                           'RoPE_tables_equal': True,
                                           'raw_sha256': array_metadata(ga['input'])['sha256']})
    except Exception as exc:
        error = exc
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        report['input_pins'] = pins
        after = {path: sha(path) if Path(path).is_file() else None for path in pins}
        report['input_pins_after'] = after
        report['source_and_inputs_unchanged'] = after == pins
        (out / 'export.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    if error is not None:
        raise error
    if not report['source_and_inputs_unchanged']:
        raise ValueError('Source/input changed during static composition')
    print(json.dumps({'status': report['status'], 'onnx': report['onnx'],
                      'export_sha256': sha(out / 'export.json')}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
