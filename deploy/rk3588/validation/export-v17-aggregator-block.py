#!/usr/bin/env python3
"""Build a fixed original aggregator block graph, without model inference.

Only the board-captured frame0/global0 FP32 tensors and 18 original parameters
are accepted. RoPE coefficients are indexed from the captured PyTorch tables;
no host trigonometry, image processing, Torch or ORT execution is performed.
Structural ONNX checks do not establish numerical accuracy or NPU coverage.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PROFILE_SCRIPT_SHA = '518088ae41c39a4b824b3ee7a6b7a2a9f86a16e3f77b9cec168b082e2b8f7279'
CAPTURE_HELPERS = {
    'capture-v17-resolution-pairs-board.py': '3706712dd3a7451fcbce98e079d4018989eca6f64eb98704a81b8f4a9acc03cb',
    'profile-v14-small-vggt-board.py': 'ec69e48e2cb5a953d8e134934af7ebf1f45a88dfa0ec5b2cf5967bb4027e8d22',
}
GRAPH_HELPERS = {
    'export-v12-dino-block.py': '2759fce1551b2a64da140a7be0b16c9d233241e0bb427b4d52c58d2243d6bf07',
    'export-v12-dino-block-gelu.py': '59f5ea1be93812571947f7afb71ea9d908057cf277c3cbc2cfb9bbaaf10e5a78',
}
SOURCE_PINS = {
    'third_party/vggt/vggt/models/aggregator.py': '623a943847a3259292bfd5c8675d8b80b09abc845eefadba7622e6a901cb9cfd',
    'third_party/vggt/vggt/layers/block.py': 'd8b8cb1bfe5f084d33810b9cc96626fa494a4beea2c3c175da197841070bbf17',
    'third_party/vggt/vggt/layers/attention.py': '9d80ff6c6f562efc29d806125fe47bd9b504a6356ce2d9b5d61e8ee4ea770a00',
    'third_party/vggt/vggt/layers/rope.py': 'c56b073724f4d56cc531e381835a2c3dc348f192aefd9e855e3e59ab3c7d6641',
    'third_party/vggt/vggt/layers/mlp.py': '255825c73b60a916dd00eb1e38aacbcdbf316e40d6a005efb46e245b7edb43aa',
    'third_party/vggt/vggt/layers/layer_scale.py': '83586eaa81ef11828b1001f3dd85747ba362df23ddb57c8454b2eba279f5a558',
}
CHECKPOINT_SHA = 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'
BASE_KEYS = (
    'norm1.weight', 'norm1.bias', 'attn.qkv.weight', 'attn.qkv.bias',
    'attn.proj.weight', 'attn.proj.bias', 'ls1.gamma', 'norm2.weight',
    'norm2.bias', 'mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight',
    'mlp.fc2.bias', 'ls2.gamma',
)
QK_KEYS = ('attn.q_norm.weight', 'attn.q_norm.bias', 'attn.k_norm.weight', 'attn.k_norm.bias')
STATE_KEYS = BASE_KEYS + QK_KEYS
ARRAY_KEYS = ('input', 'output', 'pos', 'rope_cos', 'rope_sin')
EPS_FIELDS = {'norm1': 'norm1_eps', 'norm2': 'norm2_eps',
              'attn.q_norm': 'q_norm_eps', 'attn.k_norm': 'k_norm_eps'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def require_sha(path, expected):
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in '0123456789abcdef' for c in expected) or sha(path) != expected):
        raise ValueError(f'SHA256 mismatch or malformed required SHA for {path}')
    return expected


def array_record(array):
    if not array.flags.c_contiguous or not array.size or not np.isfinite(array).all():
        raise ValueError('Expected finite nonempty C-contiguous array')
    return dict(shape=list(array.shape), dtype=str(array.dtype), bytes=array.nbytes,
                sha256=hashlib.sha256(memoryview(array).cast('B')).hexdigest())


def validate_state(weights, dim, heads):
    if set(weights) != set(STATE_KEYS) or dim % heads or (dim // heads) % 4:
        raise ValueError('Exact 18 parameters and RoPE-compatible head dimension required')
    shapes = {key: (dim,) for key in BASE_KEYS if key.startswith('norm') or key.endswith('gamma')}
    shapes.update({'attn.qkv.weight': (3 * dim, dim), 'attn.qkv.bias': (3 * dim,),
                   'attn.proj.weight': (dim, dim), 'attn.proj.bias': (dim,),
                   'mlp.fc1.weight': (4 * dim, dim), 'mlp.fc1.bias': (4 * dim,),
                   'mlp.fc2.weight': (dim, 4 * dim), 'mlp.fc2.bias': (dim,)})
    shapes.update({key: (dim // heads,) for key in QK_KEYS})
    for key, shape in shapes.items():
        value = weights[key]
        if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f'Invalid original FP32 parameter: {key}, expected {shape}')


def validate_positions(pos, cosine, sine, shape, heads):
    if (pos.dtype != np.int64 or pos.shape != (*shape[:2], 2)
            or not pos.size or int(pos.min()) < 0):
        raise ValueError('Invalid captured integer RoPE positions')
    expected = (int(pos.max()) + 1, shape[2] // heads // 2)
    for value in (cosine, sine):
        if value.dtype != np.float32 or value.shape != expected or not np.isfinite(value).all():
            raise ValueError('Invalid original FP32 RoPE lookup table')
    if not np.array_equal(cosine[0], np.ones(expected[1], np.float32)) or not np.array_equal(sine[0], np.zeros(expected[1], np.float32)):
        raise ValueError('RoPE zero-position identity mismatch')


def load_capture(directory, expected_profile_sha, block):
    """Strict saved-array loader, usable by ARM runtimes without importing ONNX."""
    if block not in ('frame0', 'global0'):
        raise ValueError('Only original first frame/global blocks are supported')
    directory = Path(directory).resolve()
    receipt_path = directory / 'profile.json'
    require_sha(receipt_path, expected_profile_sha)
    profile = json.loads(receipt_path.read_text())
    required = {'kind': 'original_VGGT_rectified_stage_and_block_capture',
                'status': 'completed_original_CPU_profile_and_capture',
                'script_sha256': PROFILE_SCRIPT_SHA, 'original_weights': True,
                'architecture_changed': False, 'ORB_used': False, 'npu_used': False,
                'inference_on_board_only': True, 'hooks_removed': True}
    if any(profile.get(key) != value for key, value in required.items()) or profile.get('cleanup_errors'):
        raise ValueError('Completed frozen original CPU capture is required')
    if profile.get('helper_hashes') != CAPTURE_HELPERS or profile.get('image_size') not in (224, 336):
        raise ValueError('Capture helper or resolution mismatch')
    for key in ('unhooked_vs_prior_capture_array_equal', 'hooked_vs_unhooked'):
        records = profile.get(key, {})
        if set(records) != {'pose_enc', 'depth', 'depth_conf'} or not all(
                value is True if key.startswith('unhooked') else value.get('array_equal') is True
                for value in records.values()):
            raise ValueError('Original hooked/unhooked numerical reference not verified')
    if (profile.get('checkpoint', {}).get('sha256') != CHECKPOINT_SHA
            or profile['checkpoint'].get('bytes') != 5026874952):
        raise ValueError('Original full checkpoint identity mismatch')
    pins = {str(receipt_path): expected_profile_sha, str(Path(__file__).resolve()): sha(__file__)}
    for relative, expected in SOURCE_PINS.items():
        if profile.get('source_hashes', {}).get(relative) != expected:
            raise ValueError(f'Captured original source mismatch: {relative}')
        path = REPO / relative
        require_sha(path, expected)
        pins[str(path)] = expected
    pair = profile.get('selected_pair', {})
    ids, images = pair.get('frame_ids', []), pair.get('images', [])
    if (pair.get('revision') != 1 or len(ids) != 2 or len(images) != 2
            or not np.isfinite(ids).all() or not ids[0] < ids[1]
            or any(item.get('samefile_validated') is not True or item.get('frame_id') != frame_id
                   for item, frame_id in zip(images, ids))):
        raise ValueError('Exact first two-frame source identity is required')
    raw_shape = profile.get('input_tensor', {}).get('shape', [])
    if (len(raw_shape) != 4 or raw_shape[:2] != [2, 3]
            or raw_shape[3] != profile['image_size'] or any(n <= 0 or n % 14 for n in raw_shape[2:])):
        raise ValueError('Invalid original two-frame preprocessing shape')
    height, width = raw_shape[2] // 14, raw_shape[3] // 14
    tokens = 5 + height * width
    expected_shape = (2, tokens, 1024) if block == 'frame0' else (1, 2 * tokens, 1024)
    row = profile.get('blocks', {}).get(block, {})
    architecture = {'module_class': 'Block', 'num_heads': 16, 'head_dim': 64,
                    'norm1_eps': 1e-5, 'norm2_eps': 1e-5, 'q_norm_eps': 1e-5,
                    'k_norm_eps': 1e-5, 'gelu_approximate': 'none',
                    'rope_frequency': 100, 'rope_scaling_factor': 1.0}
    if any(row.get(key) != value for key, value in architecture.items()):
        raise ValueError('Original aggregator QK normalization/RoPE architecture mismatch')
    loaded = {}
    for kind, names in (('tensors', ARRAY_KEYS), ('weights', STATE_KEYS)):
        entry = row.get('artifacts', {}).get(kind, {})
        filename = f'{block}-{kind}.npz'
        if entry.get('file') != filename:
            raise ValueError('Captured NPZ filename/identity mismatch')
        path = directory / filename
        require_sha(path, entry.get('sha256'))
        if path.stat().st_size != entry.get('bytes'):
            raise ValueError('Captured NPZ byte count mismatch')
        pins[str(path)] = entry['sha256']
        metadata = row.get('arrays' if kind == 'tensors' else 'weights', {})
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(names) or set(metadata) != set(names):
                raise ValueError('Exact captured tensor/weight keys are required')
            loaded[kind] = {key: saved[key] for key in names}
        for key, array in loaded[kind].items():
            if array_record(array) != metadata[key]:
                raise ValueError(f'Captured raw tensor SHA/type/shape mismatch: {key}')
    arrays, weights = loaded['tensors'], loaded['weights']
    if any(arrays[key].dtype != np.float32 or arrays[key].shape != expected_shape for key in ('input', 'output')):
        raise ValueError('Actual captured block input/output shape mismatch')
    validate_state(weights, 1024, 16)
    validate_positions(arrays['pos'], arrays['rope_cos'], arrays['rope_sin'], expected_shape, 16)
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    patch_pos = np.column_stack((yy.ravel(), xx.ravel())).astype(np.int64) + 1
    original = np.concatenate((np.zeros((5, 2), np.int64), patch_pos), axis=0)
    expected_pos = np.stack((original, original)) if block == 'frame0' else np.concatenate((original, original))[None]
    if not np.array_equal(arrays['pos'], expected_pos):
        raise ValueError('Original special-token/grid/frame ordering mismatch')
    return profile, row, arrays, weights, pins


def load_graph_helper(filename):
    path = HERE / filename
    require_sha(path, GRAPH_HELPERS[filename])
    spec = importlib.util.spec_from_file_location('_v17_' + filename.replace('-', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_aggregator_onnx(weights, input_shape, positions, rope_cos, rope_sin, *, heads=16, epsilons=None):
    """Pure graph construction; small synthetic dimensions are allowed in tests."""
    import onnx
    from onnx import checker, helper, numpy_helper, shape_inference

    if len(input_shape) != 3 or any(type(n) is not int or n <= 0 for n in input_shape):
        raise ValueError('Static positive three-dimensional input shape required')
    if type(heads) is not int or heads <= 0:
        raise ValueError('Positive integer head count required')
    validate_state(weights, input_shape[2], heads)
    validate_positions(positions, rope_cos, rope_sin, input_shape, heads)
    epsilons = dict(epsilons or {name: 1e-5 for name in EPS_FIELDS})
    if set(epsilons) != set(EPS_FIELDS) or any(not np.isfinite(v) or v <= 0 for v in epsilons.values()):
        raise ValueError('Exact positive finite LayerNorm epsilons required')
    base = load_graph_helper('export-v12-dino-block.py')
    gelu = load_graph_helper('export-v12-dino-block-gelu.py')
    model = gelu.rewrite_gelu(base.build_block_onnx({key: weights[key] for key in BASE_KEYS}, input_shape, heads=heads))
    for node in model.graph.node:
        if node.name in ('norm1', 'norm2'):
            if node.op_type != 'LayerNormalization':
                raise ValueError('Frozen builder LayerNorm pattern mismatch')
            for attribute in node.attribute:
                if attribute.name == 'epsilon':
                    attribute.f = epsilons[node.name]
    injected = []

    def constant(name, array):
        model.graph.initializer.append(numpy_helper.from_array(np.array(array, copy=True, order='C'), name))
        return name

    def node(kind, inputs, output, **attributes):
        injected.append(helper.make_node(kind, inputs, [output], name=output, **attributes))
        return output

    axis_dim = input_shape[2] // heads // 2
    axis_sizes = constant('rope_axis_sizes', np.array([axis_dim, axis_dim], np.int64))
    rotate_sizes = constant('rope_rotate_sizes', np.array([axis_dim // 2, axis_dim // 2], np.int64))
    for coordinate, label in enumerate(('y', 'x')):
        for table, trig in ((rope_cos, 'cos'), (rope_sin, 'sin')):
            constant(f'rope_{label}_{trig}', table[positions[..., coordinate]][:, None])
    for source, name in (('query', 'attn.q_norm'), ('key', 'attn.k_norm')):
        weight, bias = (constant(f'{name}.{kind}', weights[f'{name}.{kind}']) for kind in ('weight', 'bias'))
        normalized = node('LayerNormalization', [source, weight, bias], name,
                          axis=-1, epsilon=epsilons[name], stash_type=onnx.TensorProto.FLOAT)
        axes = [f'{source}_vertical', f'{source}_horizontal']
        injected.append(helper.make_node('Split', [normalized, axis_sizes], axes, axis=-1, name=f'{source}_rope_split'))
        results = []
        for tensor, label in zip(axes, ('y', 'x')):
            halves = [tensor + '_first', tensor + '_second']
            injected.append(helper.make_node('Split', [tensor, rotate_sizes], halves, axis=-1, name=tensor + '_split'))
            rotated = node('Concat', [node('Neg', [halves[1]], tensor + '_negative'), halves[0]], tensor + '_rotated', axis=-1)
            results.append(node('Add', [node('Mul', [tensor, f'rope_{label}_cos'], tensor + '_cosine'),
                                        node('Mul', [rotated, f'rope_{label}_sin'], tensor + '_sine')], tensor + '_rope'))
        node('Concat', results, source + '_rope', axis=-1)
    nodes = list(model.graph.node)
    index = next(i for i, value in enumerate(nodes) if value.name == 'key_transpose')
    if nodes[index].op_type != 'Transpose' or list(nodes[index].input) != ['key']:
        raise ValueError('Frozen key transpose pattern mismatch')
    nodes[index].input[0] = 'key_rope'
    scores = next(value for value in nodes if value.name == 'attention_scores')
    if scores.op_type != 'MatMul' or list(scores.input) != ['query', 'key_transpose']:
        raise ValueError('Frozen SDPA score pattern mismatch')
    scores.input[0] = 'query_rope'
    del model.graph.node[:]
    model.graph.node.extend(nodes[:index] + injected + nodes[index:])
    del model.graph.value_info[:]
    model.graph.name = 'original_captured_VGGT_aggregator_block'
    model.producer_name = 'vggt-slam2-rk3588s/v17-aggregator-pure-graph'
    model.doc_string = ('Original 18-parameter aggregator block with Q/K normalization and static captured 2D RoPE. '
                        'Exact Erf GELU; explicit attention/operation order still requires board numerical validation. '
                        'No original weights, iterations or layers are removed; NPU coverage is not established by export.')
    checker.check_model(model, full_check=True)
    model = shape_inference.infer_shapes(model, strict_mode=True, check_type=True, data_prop=True)
    checker.check_model(model, full_check=True)
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--profile-sha256', required=True)
    parser.add_argument('--block', choices=('frame0', 'global0'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    profile, row, arrays, weights, pins = load_capture(args.capture, args.profile_sha256, args.block)
    shape = tuple(arrays['input'].shape)
    graph = build_aggregator_onnx(weights, shape, arrays['pos'], arrays['rope_cos'], arrays['rope_sin'],
                                heads=row['num_heads'], epsilons={name: row[field] for name, field in EPS_FIELDS.items()})
    for name, expected in GRAPH_HELPERS.items():
        pins[str(HERE / name)] = expected
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    onnx_path = output / 'block.onnx'
    with onnx_path.open('xb') as stream:
        stream.write(graph.SerializeToString())
    artifacts = {}
    for key in ('input', 'output'):
        path = output / f'{key}.npy'
        with path.open('xb') as stream:
            np.save(stream, arrays[key], allow_pickle=False)
        metadata = array_record(arrays[key])
        artifacts[path.name] = dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path),
                                    shape=metadata['shape'], dtype=metadata['dtype'], raw_sha256=metadata['sha256'])
    for path, expected in pins.items():
        require_sha(path, expected)
    report = dict(status='exported_and_structurally_checked', script_sha256=sha(__file__),
                  profile_sha256=args.profile_sha256, block=args.block, input_shape=list(shape),
                  input_name='input', output_name='output',
                  onnx=dict(file=onnx_path.name, bytes=onnx_path.stat().st_size, sha256=sha(onnx_path)),
                  capture_directory=str(args.capture.resolve()), capture_artifacts=row['artifacts'],
                  artifacts=artifacts, architecture={key: row[key] for key in ('num_heads', 'head_dim',
                    'norm1_eps', 'norm2_eps', 'q_norm_eps', 'k_norm_eps', 'gelu_approximate',
                    'rope_frequency', 'rope_scaling_factor')},
                  captured_array_metadata=row['arrays'], captured_parameter_metadata=row['weights'],
                  parameter_count=18, original_weights=True, source_pins=SOURCE_PINS,
                  frozen_helpers=GRAPH_HELPERS, input_pins=pins, selected_pair=profile['selected_pair'],
                  checkpoint=profile['checkpoint'], node_count=len(graph.graph.node),
                  op_counts=dict(Counter(value.op_type for value in graph.graph.node)),
                  position_and_tables_fixed=True, host_trigonometry_recomputed=False,
                  host_model_inference_executed=False, host_ORT_executed=False, external_data=False,
                  external_data_used=False,
                  production_changed=False, all_operators_on_npu_verified=False,
                  real_time_validated=False, whole_model_quality_validated=False)
    report_path = output / 'export.json'
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], onnx=report['onnx'], receipt=str(report_path), sha256=sha(report_path))))
    return 0


if __name__ == '__main__':
    sys.exit(main())
