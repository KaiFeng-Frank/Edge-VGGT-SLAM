"""Static ONNX builders for every original VGGT frame/global block.

No Torch/model/ORT/RKNN execution occurs here. The caller supplies original
FP32 parameters, the original integer positions and board-produced RoPE tables.
The validated v17-r3 operator builder is reused verbatim. The main export emits
all 24 frame plus all 24 global graphs; the original Aggregator.forward remains
responsible for concatenation, retained intermediates and target tokens.

Usage by the board-side owner of the original model::

    export_all_original_blocks(
        state_provider, metadata_provider, output_directory,
        exporter_path=PINNED_EXPORTER, frame_positions=pos,
        rope_cos=cos, rope_sin=sin, checkpoint=verified_checkpoint,
        source_hashes=original_source_hashes)

state_provider(role, index) returns the 18 original parameter NumPy arrays.
metadata_provider(role, index) returns heads, epsilons, gelu_approximate,
rope_frequency and rope_scaling_factor. Providers are called one layer at a
time, so exporting does not require an extra copy of all 48 states in memory.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

CHECKPOINT = {'sha256': 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0',
              'bytes': 5026874952}
IMAGE_SHAPE = (2, 3, 84, 112)
DEPTH, HEADS, DIM, PATCH_SIZE, SPECIAL_TOKENS = 24, 16, 1024, 14, 5
EPSILON_NAMES = ('norm1', 'norm2', 'attn.q_norm', 'attn.k_norm')
PARAMETER_NAMES = (
    'norm1.weight', 'norm1.bias', 'attn.qkv.weight', 'attn.qkv.bias',
    'attn.q_norm.weight', 'attn.q_norm.bias', 'attn.k_norm.weight', 'attn.k_norm.bias',
    'attn.proj.weight', 'attn.proj.bias', 'ls1.gamma', 'norm2.weight', 'norm2.bias',
    'mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight', 'mlp.fc2.bias', 'ls2.gamma')
TRANSPOSED_PARAMETERS = frozenset(('attn.qkv.weight', 'attn.proj.weight',
                                  'mlp.fc1.weight', 'mlp.fc2.weight'))
OPERATOR_BUILDERS = {
    'export-v17-aggregator-block-r3.py': '6724aba806806ec36d8ba097f0c2076f3703e106f23c83d4272ae2e3a3fc2874',
    'export-v17-aggregator-block-r2.py': '17083d3b111aa90807fe1b442f6ab7391f26598dbfa230cb08c1b9ae3b19d156',
    'export-v17-aggregator-block.py': '13bd46a6b6ee6e4700ddb3742059c1ba80c0d517b874c3cc2c56028220618ff8',
    'export-v12-dino-block.py': '2759fce1551b2a64da140a7be0b16c9d233241e0bb427b4d52c58d2243d6bf07',
    'export-v12-dino-block-gelu.py': '59f5ea1be93812571947f7afb71ea9d908057cf277c3cbc2cfb9bbaaf10e5a78'}
_EXPORTERS = {}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            digest.update(part)
    return digest.hexdigest()


def raw_sha256(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()


def load_operator_builder(exporter_path):
    """Import only the unchanged static graph sources, never a model loader."""
    path = Path(exporter_path).resolve(strict=True)
    if path.name != 'export-v17-aggregator-block-r3.py':
        raise ValueError('The validated original v17-r3 operator builder is required')
    for name, expected in OPERATOR_BUILDERS.items():
        if sha256(path.with_name(name)) != expected:
            raise ValueError('Original static operator builder changed: ' + name)
    if str(path) not in _EXPORTERS:
        spec = importlib.util.spec_from_file_location('_v53_original_static_r3', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _EXPORTERS[str(path)] = module
    return _EXPORTERS[str(path)]


def block_shapes(image_shape=IMAGE_SHAPE, dim=DIM):
    """B=1, S=2; generated smaller dimensions are for static smoke checks."""
    if (len(image_shape) != 4 or tuple(image_shape[:2]) != (2, 3)
            or any(type(n) is not int or n <= 0 for n in image_shape)
            or any(n % PATCH_SIZE for n in image_shape[-2:])
            or type(dim) is not int or dim <= 0):
        raise ValueError('Two RGB frames on the original 14-pixel patch grid are required')
    tokens = SPECIAL_TOKENS + (image_shape[2] // PATCH_SIZE) * (image_shape[3] // PATCH_SIZE)
    return {'frame': (2, tokens, dim), 'global': (1, 2 * tokens, dim)}


def expected_frame_positions(image_shape=IMAGE_SHAPE):
    """Integer grid check only; trigonometric tables are never synthesized."""
    block_shapes(image_shape)
    height, width = image_shape[2] // PATCH_SIZE, image_shape[3] // PATCH_SIZE
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    grid = np.column_stack((yy.ravel(), xx.ravel())).astype(np.int64) + 1
    one_frame = np.concatenate((np.zeros((SPECIAL_TOKENS, 2), np.int64), grid), axis=0)
    return np.ascontiguousarray(np.stack((one_frame, one_frame)))


def build_original_block_onnx(weights, *, role, index, positions, rope_cos, rope_sin,
                              epsilons, exporter_path, image_shape=IMAGE_SHAPE,
                              dim=DIM, heads=HEADS):
    """Static complete block graph: exact attention, GELU, LayerScale, residuals."""
    if role not in ('frame', 'global') or type(index) is not int or not 0 <= index < DEPTH:
        raise ValueError('Original frame/global role and layer index 0..23 required')
    if set(weights) != set(PARAMETER_NAMES):
        raise ValueError('Exactly the 18 original block parameters are required')
    shapes = block_shapes(image_shape, dim)
    frame_pos = expected_frame_positions(image_shape)
    expected_pos = frame_pos if role == 'frame' else frame_pos.reshape(1, -1, 2)
    if (not isinstance(positions, np.ndarray) or positions.dtype != np.int64
            or positions.shape != expected_pos.shape or not np.array_equal(positions, expected_pos)):
        raise ValueError('Original special-token/grid/frame RoPE position order changed')
    if (set(epsilons) != set(EPSILON_NAMES)
            or any(not np.isfinite(v) or v <= 0 for v in epsilons.values())):
        raise ValueError('Provide all four actual original LayerNorm epsilons')
    builder = load_operator_builder(exporter_path)
    graph = builder.build_aggregator_onnx(weights, shapes[role], positions, rope_cos, rope_sin,
                                          heads=heads, epsilons=epsilons)
    # The builder transposes four Linear weights. Confirm it retained every raw
    # learned tensor, not merely its shape, without running an executor.
    from onnx import numpy_helper
    tensors = {value.name: value for value in graph.graph.initializer}
    for key in PARAMETER_NAMES:
        name = key + '_transposed' if key in TRANSPOSED_PARAMETERS else key
        expected = weights[key].T if key in TRANSPOSED_PARAMETERS else weights[key]
        actual = numpy_helper.to_array(tensors[name])
        if actual.dtype != np.float32 or actual.shape != expected.shape or raw_sha256(actual) != raw_sha256(expected):
            raise ValueError('Original learned tensor changed during static construction: ' + key)
    graph.graph.name = f'original_VGGT_{role}_block_{index:02d}'
    graph.producer_name = 'Edge-VGGT-SLAM/original-v53-static-blocks'
    graph.doc_string += f' Original aggregator.{role}_blocks.{index}; no omitted original operations.'
    return graph


def _save_positions(directory, filename, array):
    path = directory / filename
    with path.open('xb') as stream:
        np.save(stream, array, allow_pickle=False)
    return {'file': filename, 'bytes': path.stat().st_size, 'sha256': sha256(path),
            'shape': list(array.shape), 'dtype': str(array.dtype), 'tensor_bytes': array.nbytes,
            'raw_sha256': raw_sha256(array)}


def export_all_original_blocks(state_provider, metadata_provider, output_directory, *,
                               exporter_path, frame_positions, rope_cos, rope_sin,
                               checkpoint, source_hashes):
    """Export exactly 48 complete original blocks, sequentially and privately.

The caller must have verified the full original checkpoint and capture source
identity before supplying parameters. This function proves their preservation
in graphs; it does not infer their origin from numeric values or run inference.
Root may append each compiled ``rknn={file,bytes,sha256}`` and set the runtime
schema to ``v53_original_VGGT_48_block_RKNN_manifest`` only after all builds.
"""
    if not callable(state_provider) or not callable(metadata_provider) or checkpoint != CHECKPOINT:
        raise ValueError('Original checkpoint identity and per-layer providers required')
    if (not isinstance(source_hashes, dict) or not source_hashes
            or any(not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v)
                   for v in source_hashes.values())):
        raise ValueError('Original capture source SHA map is required')
    shapes = block_shapes()
    if (not isinstance(frame_positions, np.ndarray) or frame_positions.dtype != np.int64
            or not np.array_equal(frame_positions, expected_frame_positions())):
        raise ValueError('Original [2,53,2] captured positions required')
    directory = Path(output_directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    positions = {'frame': np.ascontiguousarray(frame_positions),
                 'global': np.ascontiguousarray(frame_positions.reshape(1, 106, 2))}
    position_records = {role: _save_positions(directory, role + '-positions.npy', values)
                        for role, values in positions.items()}
    manifest = {'schema': 'v53_original_VGGT_48_block_ONNX_manifest',
        'status': 'static_export_only_not_runtime_validated', 'script_sha256': sha256(__file__),
        'checkpoint': dict(checkpoint), 'source_hashes': dict(source_hashes),
        'operator_builder_sources': dict(OPERATOR_BUILDERS), 'image_shape': list(IMAGE_SHAPE),
        'aa_order': ['frame', 'global'], 'aa_block_size': 1, 'depth': DEPTH,
        'heads': HEADS, 'frames': 2, 'tokens': 53, 'dim': DIM, 'patch_start_idx': SPECIAL_TOKENS,
        'blocks': [], 'calibration_or_ORT_or_SDK_or_model_execution_by_builder': False,
        'candidate_accepted': False, 'full_NPU_model_or_full_model_quality_or_realtime_validated': False}
    for index in range(DEPTH):
        for role in ('frame', 'global'):
            weights = state_provider(role, index)
            info = metadata_provider(role, index)
            if (info.get('heads') != HEADS or info.get('gelu_approximate') != 'none'
                    or info.get('rope_frequency') != 100 or info.get('rope_scaling_factor') != 1.0):
                raise ValueError('Original block attention/GELU/RoPE metadata mismatch')
            graph = build_original_block_onnx(weights, role=role, index=index, positions=positions[role],
                rope_cos=rope_cos, rope_sin=rope_sin, epsilons=info['epsilons'], exporter_path=exporter_path)
            filename = f'{role}-{index:02d}.onnx'
            path = directory / filename
            with path.open('xb') as stream:
                stream.write(graph.SerializeToString())
            manifest['blocks'].append({'role': role, 'index': index, 'file': filename,
                'bytes': path.stat().st_size, 'sha256': sha256(path),
                'input_name': 'input', 'output_name': 'output',
                'input_shape': list(shapes[role]), 'output_shape': list(shapes[role]),
                'positions': dict(position_records[role]), 'heads': HEADS,
                'epsilons': dict(info['epsilons']), 'gelu_approximate': 'none',
                'rope_frequency': 100, 'rope_scaling_factor': 1.0,
                'parameter_raw_sha256': {name: raw_sha256(weights[name]) for name in PARAMETER_NAMES},
                'all_18_caller_parameter_values_preserved_in_graph': True})
            del graph, weights
    # A partial export never receives a completed manifest.
    if [(row['role'], row['index']) for row in manifest['blocks']] != [
            (role, index) for index in range(DEPTH) for role in ('frame', 'global')]:
        raise ValueError('All 48 original blocks must be present exactly once')
    with (directory / 'export.json').open('x') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False); stream.write('\n')
    return manifest


def compose_frame_global_pair(frame_model, global_model, *, index):
    """Optional pure composer with two outputs, for a future paired adapter.

``frame_output`` is [2,P,C]; ``global_output`` is [1,2P,C]. Their original
C-order views [1,2,P,C] can be concatenated along C exactly as upstream.
This is not needed by the 48-context adapter, which keeps Aggregator.forward.
"""
    import onnx
    from onnx import helper, numpy_helper
    if type(index) is not int or not 0 <= index < DEPTH:
        raise ValueError('Original layer index required')
    def shape(value):
        dims = value.type.tensor_type.shape.dim
        if len(dims) != 3 or any(not d.HasField('dim_value') or d.dim_value <= 0 for d in dims):
            raise ValueError('Static rank-three parent tensors required')
        return [d.dim_value for d in dims]
    for model in (frame_model, global_model):
        if len(model.graph.input) != 1 or len(model.graph.output) != 1:
            raise ValueError('Single-input/output complete parent blocks required')
        if shape(model.graph.input[0]) != shape(model.graph.output[0]):
            raise ValueError('Original parent token shape changed')
    fs, gs = shape(frame_model.graph.input[0]), shape(global_model.graph.input[0])
    if (fs[0] != 2 or gs != [1, fs[0] * fs[1], fs[2]] or frame_model.ir_version != global_model.ir_version
            or [p.SerializeToString() for p in frame_model.opset_import]
            != [p.SerializeToString() for p in global_model.opset_import]):
        raise ValueError('Original paired token layout/IR/opsets mismatch')
    def namespaced(model, prefix, input_name, output_name):
        graph = onnx.GraphProto(); graph.CopyFrom(model.graph)
        overrides = {model.graph.input[0].name: input_name, model.graph.output[0].name: output_name}
        def name(value):
            return overrides.get(value, prefix + value) if value else value
        for node in graph.node:
            node.name = prefix + node.name
            node.input[:] = [name(v) for v in node.input]; node.output[:] = [name(v) for v in node.output]
        for value in list(graph.input) + list(graph.output) + list(graph.value_info) + list(graph.initializer):
            value.name = name(value.name)
        return graph
    f = namespaced(frame_model, f'frame{index}/', 'input', 'frame_output')
    g = namespaced(global_model, f'global{index}/', 'global_input', 'global_output')
    bridge = helper.make_node('Reshape', ['frame_output', 'frame_global_shape'], ['global_input'],
                              name='frame_to_global_view', allowzero=0)
    result = onnx.ModelProto(); result.CopyFrom(frame_model)
    # Public frame_output need not be duplicated in value_info.
    infos = [v for v in list(f.value_info) + list(g.value_info) if v.name not in ('frame_output', 'global_output')]
    infos.append(g.input[0])
    result.graph.CopyFrom(helper.make_graph(list(f.node) + [bridge] + list(g.node),
        f'original_frame_global_pair_{index:02d}', list(f.input), [f.output[0], g.output[0]],
        initializer=list(f.initializer) + list(g.initializer) + [
            numpy_helper.from_array(np.asarray(gs, np.int64), 'frame_global_shape')], value_info=infos))
    result.producer_name = 'Edge-VGGT-SLAM/original-v53-static-pair-two-outputs'
    onnx.checker.check_model(result, full_check=True)
    return result
