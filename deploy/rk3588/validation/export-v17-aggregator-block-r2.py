#!/usr/bin/env python3
"""V17 R2 pure ONNX export: equivalent RoPE with fixed Gather/sign constants.

The original R1 exporter, capture contract, weights and GELU remain frozen.
This avoids RKNN 2.3.2's failing nested Split/Concat/Neg fusion. A valid ONNX
graph is not proof of RKNN support, numerical quality or whole-model speed.
No Torch, host original-model inference, ORT or trigonometry is executed.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
BASE_SHA = '13bd46a6b6ee6e4700ddb3742059c1ba80c0d517b874c3cc2c56028220618ff8'
BASE_PATH = Path(__file__).with_name('export-v17-aggregator-block.py')
_BASE = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def base_parts():
    global _BASE
    if _BASE is None:
        if sha(BASE_PATH) != BASE_SHA:
            raise ValueError('Frozen R1 aggregator exporter SHA mismatch')
        spec = importlib.util.spec_from_file_location('_v17_r2_private_r1', BASE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _BASE = (module, module.build_aggregator_onnx, module.load_capture)
    return _BASE


# Public provenance dictionaries preserve the R1 checker/runtime interface.
SOURCE_PINS = dict(base_parts()[0].SOURCE_PINS)
GRAPH_HELPERS = {**base_parts()[0].GRAPH_HELPERS, BASE_PATH.name: BASE_SHA}
PROFILE_SCRIPT_SHA = base_parts()[0].PROFILE_SCRIPT_SHA


def load_capture(directory, expected_profile_sha, block):
    """Unchanged strict R1 capture contract; imports NumPy/standard library only."""
    _, _, loader = base_parts()
    values = loader(directory, expected_profile_sha, block)
    values[-1][str(BASE_PATH.resolve())] = BASE_SHA
    values[-1][str(Path(__file__).resolve())] = sha(__file__)
    return values


def rotation_constants(head_dim):
    """Rotate halves within each y/x component, matching original _rotate_features."""
    if type(head_dim) is not int or head_dim <= 0 or head_dim % 4:
        raise ValueError('Positive head dimension divisible by four required')
    quarter = head_dim // 4
    indices = np.arange(head_dim, dtype=np.int64).reshape(2, 2, quarter)[:, ::-1].reshape(-1).copy()
    signs = np.tile(np.concatenate((-np.ones(quarter, np.float32), np.ones(quarter, np.float32))), 2)
    return indices, signs


def rewrite_rope_as_gather(model):
    import onnx
    from onnx import checker, helper, numpy_helper, shape_inference

    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    head_dim = initializers['attn.q_norm.weight'].size
    indices, signs = rotation_constants(head_dim)
    constants = {'rope_permutation': indices, 'rope_rotation_sign': signs}
    for name in ('cos', 'sin'):
        constants[f'rope_full_{name}'] = np.concatenate(
            (initializers[f'rope_y_{name}'], initializers[f'rope_x_{name}']), axis=-1)
    remove_constants = {'rope_axis_sizes', 'rope_rotate_sizes', 'rope_y_cos', 'rope_y_sin', 'rope_x_cos', 'rope_x_sin'}
    if not remove_constants.issubset(initializers):
        raise ValueError('Frozen R1 RoPE constant pattern mismatch')
    remove_nodes = {node.name for node in model.graph.node if
                    node.name.startswith(('query_vertical', 'query_horizontal', 'key_vertical', 'key_horizontal'))
                    or node.name in ('query_rope_split', 'key_rope_split', 'query_rope', 'key_rope')}
    deleted_outputs = {output for node in model.graph.node if node.name in remove_nodes for output in node.output}
    for node in model.graph.node:
        if node.name not in remove_nodes and set(node.input) & (deleted_outputs - {'query_rope', 'key_rope'}):
            raise ValueError('R1 RoPE intermediate has an unexpected consumer')
    nodes, visited = [], set()
    for node in model.graph.node:
        if node.name in remove_nodes:
            continue
        nodes.append(node)
        if node.name not in ('attn.q_norm', 'attn.k_norm'):
            continue
        source = 'query' if node.name == 'attn.q_norm' else 'key'
        if node.op_type != 'LayerNormalization':
            raise ValueError('Frozen Q/K normalization pattern mismatch')
        visited.add(source)
        def append(kind, inputs, name, **attributes):
            nodes.append(helper.make_node(kind, inputs, [name], name=name, **attributes))
            return name
        permuted = append('Gather', [node.output[0], 'rope_permutation'], source + '_rope_permuted', axis=-1)
        rotated = append('Mul', [permuted, 'rope_rotation_sign'], source + '_rope_rotated')
        cosine = append('Mul', [node.output[0], 'rope_full_cos'], source + '_rope_cosine')
        sine = append('Mul', [rotated, 'rope_full_sin'], source + '_rope_sine')
        append('Add', [cosine, sine], source + '_rope')
    if visited != {'query', 'key'}:
        raise ValueError('Exactly one original Q/K normalization is required')
    kept = [item for item in model.graph.initializer if item.name not in remove_constants]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    model.graph.initializer.extend(numpy_helper.from_array(np.ascontiguousarray(value), name)
                                  for name, value in constants.items())
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.value_info[:]
    model.producer_name = 'vggt-slam2-rk3588s/v17-r2-aggregator-pure-graph'
    model.doc_string += (' R2 encodes the identical per-axis half rotation as one full-head Gather and +/-1 Mul; '
                         'capture and all parameters are unchanged. RKNN targets/numerics require actual board validation.')
    checker.check_model(model, full_check=True)
    model = shape_inference.infer_shapes(model, strict_mode=True, check_type=True, data_prop=True)
    checker.check_model(model, full_check=True)
    return model


def build_aggregator_onnx(*args, **kwargs):
    _, builder, _ = base_parts()
    return rewrite_rope_as_gather(builder(*args, **kwargs))


def main(argv=None):
    # Specialize only this fresh private module namespace, keeping the frozen
    # file and its original CLI/metadata/capture validation implementation intact.
    base, original_builder, original_loader = base_parts()
    original_file, original_helpers = base.__file__, base.GRAPH_HELPERS
    try:
        base.__file__ = __file__
        base.GRAPH_HELPERS = GRAPH_HELPERS
        base.build_aggregator_onnx = build_aggregator_onnx
        base.load_capture = load_capture
        return base.main(argv)
    finally:
        base.__file__, base.GRAPH_HELPERS = original_file, original_helpers
        base.build_aggregator_onnx, base.load_capture = original_builder, original_loader


if __name__ == '__main__':
    sys.exit(main())
