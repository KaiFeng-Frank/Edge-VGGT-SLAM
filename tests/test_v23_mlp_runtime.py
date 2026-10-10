"""Tiny synthetic reduction/ownership tests; never use a real checkpoint."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def load_runner(version):
    name = 'run-v23-mlp-board.py' if version == 'r1' else 'run-v23-mlp-board-r2.py'
    path = Path(__file__).resolve().parents[1]/'deploy/rk3588/validation'/name
    spec = importlib.util.spec_from_file_location('v23_mlp_runner_tests_'+version, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=['r1', 'r2'])
def runner(request):
    return load_runner(request.param)


def test_partial_bias_once_ordered_sum_crop_and_output_ownership(runner):
    arrays = [np.full((1, 5, 2), x, dtype=np.float32) for x in (2, 3, 5)]
    bias = np.array([11, -7], dtype=np.float32)
    result = runner.combine(arrays, bias, rows=3, partial=True)
    np.testing.assert_array_equal(result, np.broadcast_to([21, 3], (1, 3, 2)))
    assert result.dtype == np.float32 and result.flags.c_contiguous
    for array in arrays:
        array.fill(100)
    bias.fill(100)
    np.testing.assert_array_equal(result, np.broadcast_to([21, 3], (1, 3, 2)))


def test_full_graph_does_not_add_bias_twice(runner):
    array = np.array([[[4, 5], [6, 7]]], dtype=np.float32)
    result = runner.combine([array], np.array([100, 200], dtype=np.float32), rows=2, partial=False)
    np.testing.assert_array_equal(result, array)
    assert not np.shares_memory(result, array)


def test_fp32_reduction_order_is_explicit(runner):
    arrays = [np.array([[[x]]], dtype=np.float32) for x in (1e8, -1e8, 3)]
    result = runner.combine(arrays, np.array([2], dtype=np.float32), 1, True)
    np.testing.assert_array_equal(result, [[[5]]])


@pytest.mark.parametrize('count,partial', [(1, True), (2, True), (3, False), (0, False)])
def test_missing_or_extra_branch_rejected(runner, count, partial):
    with pytest.raises(ValueError):
        runner.combine([np.zeros((1, 2, 4), dtype=np.float32)]*count,
                       np.zeros(4, dtype=np.float32), 2, partial)


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_partial_is_not_hidden_by_crop(runner, bad):
    array = np.zeros((1, 3, 2), dtype=np.float32)
    array[:, -1] = bad
    with pytest.raises(ValueError):
        runner.combine([array]*3, np.zeros(2, dtype=np.float32), 2, True)


def test_padding_preserves_original_rows_and_owns_zero_tail(runner):
    value = np.arange(24, dtype=np.float32).reshape(1, 3, 8)
    result = runner.padded_input(value, 5)
    np.testing.assert_array_equal(result[:, :3], value)
    assert not np.any(result[:, 3:])
    value.fill(-1)
    assert result[0, 0, 0] == 0


def test_no_silent_truncation_or_dtype_conversion(runner):
    with pytest.raises(ValueError):
        runner.padded_input(np.zeros((1, 5, 3), dtype=np.float32), 4)
    with pytest.raises(ValueError):
        runner.padded_input(np.zeros((1, 3, 3), dtype=np.float16), 4)


def test_board_guard_rejects_host_before_reading_inputs(runner, monkeypatch):
    runtime = runner.runtime_module()
    monkeypatch.setattr(runtime.platform, 'machine', lambda: 'x86_64')
    with pytest.raises(RuntimeError, match='Board-only'):
        runtime.require_board()


def test_quality_gate_is_separate_from_small_relative_error(runner):
    runtime = runner.runtime_module()
    expected = np.ones((1, 3, 2), dtype=np.float32)
    actual = expected.copy()
    actual[0, 0, 0] += .001
    row = runner.comparison(runtime, actual, expected)
    assert row['relative_l2'] < .001
    assert row['failed_elements'] == 1
    assert row['elementwise_gate_passed'] is False


def synthetic_inputs(runner, folder, monkeypatch):
    """Synthetic values only, with test-specific capture identity, never inference."""
    norm2 = np.zeros((1, 874, 1024), dtype=np.float32)
    state = {name: np.zeros((2,), dtype=np.float32) for name in
             ('mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight', 'mlp.fc2.bias')}
    np.savez(folder/'activations.npz', norm2=norm2)
    np.savez(folder/'mlp-state.npz', **state)
    capture = dict(status='captured_original_FP32_stages_bitwise_bound',
                   arrays={'norm2': runner.array_record(norm2)},
                   global0_state={name: runner.array_record(value) for name, value in state.items()},
                   artifacts={name: dict(bytes=(folder/name).stat().st_size, sha256=runner.sha(folder/name))
                              for name in ('activations.npz', 'mlp-state.npz')})
    (folder/'capture.json').write_text(json.dumps(capture))
    monkeypatch.setattr(runner, 'CAPTURE_SHA', runner.sha(folder/'capture.json'))
    exports, hashes = [], []
    for index, bounds in enumerate(runner.SLICES):
        directory = folder/f'export-{index}'
        directory.mkdir()
        (directory/'mlp.onnx').write_bytes(b'synthetic metadata-only; not an executable graph')
        report = dict(capture_sha256=runner.CAPTURE_SHA, original_rows=874, input_name='input', output_name='output',
                      hidden_slice=list(bounds), includes_fc2_bias=index == 0,
                      input_shape=[1, 874, 1024], bucket=874,
                      original_parameter_metadata=capture['global0_state'],
                      script_sha256=runner.sha(runner.HERE/'export-v23-mlp-r2.py'),
                      exporter_name='export-v23-mlp-r2.py',
                      parent_exporter=dict(file='export-v23-mlp.py',
                          sha256='56476c724cde537fd35466be0b41d300592821be59f2a840519be6c117c5146e'),
                      gelu_pattern_source=dict(file='export-v12-dino-block-gelu.py',
                          sha256='59f5ea1be93812571947f7afb71ea9d908057cf277c3cbc2cfb9bbaaf10e5a78'),
                      onnx=dict(file='mlp.onnx', bytes=(directory/'mlp.onnx').stat().st_size,
                                sha256=runner.sha(directory/'mlp.onnx')))
        (directory/'export.json').write_text(json.dumps(report))
        exports.append(directory)
        hashes.append(runner.sha(directory/'export.json'))
    return SimpleNamespace(capture_dir=folder, exports=exports, export_sha256=hashes)


def test_r2_metadata_loader_pins_parent_pattern_and_entrypoint(tmp_path, monkeypatch):
    runner = load_runner('r2')
    args = synthetic_inputs(runner, tmp_path, monkeypatch)
    _, _, specs, pins = runner.load_inputs(args)
    assert len(specs) == 4 and [row['core_mask'] for row in specs] == [7, 1, 2, 4]
    assert str(runner.HERE/'export-v23-mlp.py') in pins
    assert str(runner.HERE/'export-v23-mlp-r2.py') in pins
    assert str(runner.HERE/'export-v12-dino-block-gelu.py') in pins


@pytest.mark.parametrize('mutation', ['parent_metadata', 'parent_source'])
def test_r2_rejects_frozen_parent_provenance_changes(tmp_path, monkeypatch, mutation):
    runner = load_runner('r2')
    args = synthetic_inputs(runner, tmp_path, monkeypatch)
    if mutation == 'parent_metadata':
        receipt = args.exports[0]/'export.json'
        data = json.loads(receipt.read_text())
        data['parent_exporter']['sha256'] = '0'*64
        receipt.write_text(json.dumps(data))
        args.export_sha256[0] = runner.sha(receipt)
    else:
        source_dir = tmp_path/'changed-source'
        source_dir.mkdir()
        (source_dir/'export-v23-mlp.py').write_text('changed synthetic parent source')
        monkeypatch.setattr(runner, 'HERE', source_dir)
    with pytest.raises(ValueError, match='provenance|SHA256'):
        runner.load_inputs(args)
