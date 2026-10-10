"""Tiny synthetic axis-equivalence tests, without host real-model/ORT execution."""
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / 'deploy/rk3588/validation/export-v17-aggregator-block-r3.py'
spec = importlib.util.spec_from_file_location('v17_r3_test', EXPORT)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)
spec = importlib.util.spec_from_file_location('v17_r3_synthetic_fixture', Path(__file__).with_name('test_v17_aggregator_export.py'))
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def test_normalized_full_tiny_graph_is_bitwise_r2_and_parameters_unchanged():
    pytest.importorskip('onnx')
    from onnx.reference import ReferenceEvaluator
    weights, shape, pos, cos, sin, heads, inputs = fixture.fixture()
    r2 = exporter.r2_module().build_aggregator_onnx(weights, shape, pos, cos, sin, heads=heads)
    r3 = exporter.normalize_rope_gather_axes(copy.deepcopy(r2))
    a = ReferenceEvaluator(r2).run(None, {'input': inputs})[0]
    b = ReferenceEvaluator(r3).run(None, {'input': inputs})[0]
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    assert [a.SerializeToString() for a in r2.graph.initializer] == [a.SerializeToString() for a in r3.graph.initializer]
    for node in r3.graph.node:
        if node.name in ('query_rope_permuted', 'key_rope_permuted'):
            assert next(attribute.i for attribute in node.attribute if attribute.name == 'axis') == 3


def test_unexpected_existing_axis_rejected():
    pytest.importorskip('onnx')
    weights, shape, pos, cos, sin, heads, _ = fixture.fixture()
    model = exporter.r2_module().build_aggregator_onnx(weights, shape, pos, cos, sin, heads=heads)
    node = next(node for node in model.graph.node if node.name == 'query_rope_permuted')
    next(attribute for attribute in node.attribute if attribute.name == 'axis').i = 2
    with pytest.raises(ValueError, match='negative last-axis'):
        exporter.normalize_rope_gather_axes(model)


def test_r3_loader_does_not_import_torch_onnx_or_ort():
    program = ('import sys,importlib.util; s=importlib.util.spec_from_file_location("v17r3",sys.argv[1]); '
               'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
               'assert all(k not in sys.modules for k in ("torch","onnx","onnxruntime"))')
    subprocess.run([sys.executable, '-B', '-c', program, str(EXPORT)], check=True)
