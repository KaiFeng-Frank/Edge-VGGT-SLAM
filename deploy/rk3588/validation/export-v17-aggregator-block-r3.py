#!/usr/bin/env python3
"""V17 R3 pure ONNX export: canonical positive axes for rank-four RoPE Gather.

The fixed S=2 inputs, positions, original 18 parameters and R2 computation are
unchanged. This avoids RKNN 2.3.2's transpose optimization failing to normalize
Gather axis=-1. Frozen R1/R2 files remain intact. No model or ORT execution.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode = True
R2_PATH = Path(__file__).with_name('export-v17-aggregator-block-r2.py')
R2_SHA = '17083d3b111aa90807fe1b442f6ab7391f26598dbfa230cb08c1b9ae3b19d156'
_R2 = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def r2_module():
    global _R2
    if _R2 is None:
        if sha(R2_PATH) != R2_SHA:
            raise ValueError('Frozen R2 aggregator exporter SHA mismatch')
        spec = importlib.util.spec_from_file_location('_v17_r3_private_r2', R2_PATH)
        _R2 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_R2)
    return _R2


SOURCE_PINS = dict(r2_module().SOURCE_PINS)
GRAPH_HELPERS = {**r2_module().GRAPH_HELPERS, R2_PATH.name: R2_SHA}
PROFILE_SCRIPT_SHA = r2_module().PROFILE_SCRIPT_SHA


def load_capture(directory, expected_profile_sha, block):
    values = r2_module().load_capture(directory, expected_profile_sha, block)
    values[-1][str(R2_PATH.resolve())] = R2_SHA
    values[-1][str(Path(__file__).resolve())] = sha(__file__)
    return values


def normalize_rope_gather_axes(model):
    from onnx import checker, shape_inference

    names = {'query_rope_permuted', 'key_rope_permuted'}
    original_parameters = [item.SerializeToString() for item in model.graph.initializer]
    original_nodes = {node.name: node.SerializeToString() for node in model.graph.node if node.name not in names}
    shapes = {value.name: value.type.tensor_type.shape for value in
              list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)}
    changed = set()
    for node in model.graph.node:
        if node.name not in names:
            continue
        if node.op_type != 'Gather' or len(shapes[node.input[0]].dim) != 4:
            raise ValueError('Exact rank-four R2 RoPE Gather required')
        axes = [attribute for attribute in node.attribute if attribute.name == 'axis']
        if len(axes) != 1 or axes[0].i != -1:
            raise ValueError('Expected original R2 negative last-axis Gather')
        axes[0].i = 3
        changed.add(node.name)
    if changed != names:
        raise ValueError('Exactly two original RoPE Gather axes must be normalized')
    checker.check_model(model, full_check=True)
    model = shape_inference.infer_shapes(model, strict_mode=True, check_type=True, data_prop=True)
    checker.check_model(model, full_check=True)
    if original_parameters != [item.SerializeToString() for item in model.graph.initializer]:
        raise ValueError('Axis normalization changed an initializer')
    if original_nodes != {node.name: node.SerializeToString() for node in model.graph.node if node.name not in names}:
        raise ValueError('Axis normalization changed an unrelated node')
    model.producer_name = 'vggt-slam2-rk3588s/v17-r3-aggregator-pure-graph'
    model.doc_string += ' R3 only canonicalizes the two rank-four RoPE Gather axes from -1 to 3.'
    return model


def build_aggregator_onnx(*args, **kwargs):
    return normalize_rope_gather_axes(r2_module().build_aggregator_onnx(*args, **kwargs))


def main(argv=None):
    base, original_builder, original_loader = r2_module().base_parts()
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
