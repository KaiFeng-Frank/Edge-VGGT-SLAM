"""Synthetic R1/R2 equivalence only; no real VGGT model or host ORT execution."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / 'deploy/rk3588/validation/export-v17-aggregator-block-r2.py'
spec = importlib.util.spec_from_file_location('v17_r2_test', EXPORT)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)
spec = importlib.util.spec_from_file_location('v17_r1_synthetic_fixture', Path(__file__).with_name('test_v17_aggregator_export.py'))
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)


@pytest.mark.parametrize('dim', [4, 8, 64])
def test_gather_sign_rotation_is_bitwise_original_including_signed_zero(dim):
    values = np.resize(np.array([0., -0., 1e-40, -1e-40, 3., -5., np.finfo(np.float32).max], np.float32), dim)
    vertical, horizontal = np.split(values, 2)
    expected = np.concatenate(tuple(np.concatenate((-part[len(part)//2:], part[:len(part)//2])) for part in (vertical, horizontal)))
    indices, signs = exporter.rotation_constants(dim)
    actual = values[indices] * signs
    assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
    assert np.array_equal(np.sort(indices), np.arange(dim))


def test_tiny_full_graph_r1_r2_are_elementwise_identical():
    onnx = pytest.importorskip('onnx')
    from onnx.reference import ReferenceEvaluator
    weights, shape, pos, cos, sin, heads, inputs = fixture_module.fixture()
    _, builder, _ = exporter.base_parts()
    old = builder(weights, shape, pos, cos, sin, heads=heads)
    new = exporter.build_aggregator_onnx(weights, shape, pos, cos, sin, heads=heads)
    actual = ReferenceEvaluator(new).run(None, {'input': inputs})[0]
    expected = ReferenceEvaluator(old).run(None, {'input': inputs})[0]
    assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
    assert all(n.op_type not in ('Split', 'Concat', 'Neg') for n in new.graph.node)
    np.testing.assert_allclose(actual, fixture_module.independent_numpy_block(weights, inputs, pos, cos, sin, heads), atol=2e-6, rtol=2e-6)
    old_parameters = {i.name: i.SerializeToString() for i in old.graph.initializer if not i.name.startswith('rope_')}
    new_parameters = {i.name: i.SerializeToString() for i in new.graph.initializer if not i.name.startswith('rope_')}
    assert old_parameters == new_parameters


def test_r2_loader_does_not_import_torch_onnx_or_ort():
    program = ('import sys,importlib.util; s=importlib.util.spec_from_file_location("v17r2",sys.argv[1]); '
               'm=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.base_parts(); '
               'assert all(k not in sys.modules for k in ("torch","onnx","onnxruntime"))')
    subprocess.run([sys.executable, '-B', '-c', program, str(EXPORT)], check=True)


@pytest.mark.parametrize('dim', [0, 2, -4, 4.0])
def test_invalid_rotation_dimension_rejected(dim):
    with pytest.raises(ValueError):
        exporter.rotation_constants(dim)
