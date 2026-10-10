"""Synthetic attention operands only; no captured model, real images or ORT."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def capture_helper():
    path = ROOT / 'deploy/rk3588/validation/capture-v19-attention-board.py'
    spec = importlib.util.spec_from_file_location('v19_attention_capture_synthetic', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_sdpa(q, k, v, *, dropout_p):
    assert dropout_p == 0.0
    scores = np.matmul(q, k.swapaxes(-2, -1)) * np.float32(1 / np.sqrt(q.shape[-1]))
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return np.matmul(probabilities, v)


def test_capture_preserves_objects_result_and_restores_original(capture_helper):
    rng = np.random.default_rng(6)
    operands = [rng.standard_normal((2, 3, 5, 64)).astype(np.float32) for _ in range(3)]
    seen, original_calls = {}, []
    expected = synthetic_sdpa(*operands, dropout_p=0.0)

    def original(q, k, v, **kwargs):
        assert all(given is wanted for given, wanted in zip((q, k, v), operands))
        original_calls.append(kwargs)
        return expected

    functional = SimpleNamespace(scaled_dot_product_attention=original)
    observer = capture_helper.SDPAInterceptor(functional, lambda name, value: seen.update({name: value.copy()}))
    with observer:
        result = functional.scaled_dot_product_attention(*operands, dropout_p=0.0)
        assert result is expected
    assert observer.calls == 1 and observer.restored
    assert functional.scaled_dot_product_attention is original
    assert original_calls == [dict(dropout_p=0.0)]
    assert tuple(seen) == capture_helper.ARRAY_KEYS
    assert np.array_equal(seen['output'], expected)
    operands[0][0, 0, 0, 0] = 99
    assert seen['query'][0, 0, 0, 0] != 99


@pytest.mark.parametrize('failure', ['snapshot', 'original', 'duplicate'])
def test_capture_restores_on_observer_model_or_duplicate_failure(capture_helper, failure):
    def original(*args, **kwargs):
        if failure == 'original':
            raise RuntimeError('original failure')
        return args[0]

    def snapshot(name, value):
        if failure == 'snapshot':
            raise RuntimeError('observer failure')

    functional = SimpleNamespace(scaled_dot_product_attention=original)
    observer = capture_helper.SDPAInterceptor(functional, snapshot)
    with pytest.raises((RuntimeError, ValueError)):
        with observer:
            functional.scaled_dot_product_attention(1, 2, 3, dropout_p=0.0)
            if failure == 'duplicate':
                functional.scaled_dot_product_attention(1, 2, 3, dropout_p=0.0)
    assert observer.restored and functional.scaled_dot_product_attention is original


@pytest.mark.parametrize('kwargs', [{}, {'dropout_p': 0.1}, {'dropout_p': 0.0, 'attn_mask': None},
                                    {'dropout_p': 0.0, 'is_causal': True}, {'dropout_p': 0.0, 'scale': 0.125}])
def test_capture_rejects_changed_pinned_call_contract(capture_helper, kwargs):
    functional = SimpleNamespace(scaled_dot_product_attention=lambda *_args, **_kwargs: None)
    original = functional.scaled_dot_product_attention
    observer = capture_helper.SDPAInterceptor(functional, lambda *_: pytest.fail('must not copy'))
    with pytest.raises(ValueError, match='signature changed'):
        with observer:
            functional.scaled_dot_product_attention(1, 2, 3, **kwargs)
    assert observer.calls == 0 and observer.restored
    assert functional.scaled_dot_product_attention is original


def test_reference_gate_reports_bitwise_separately_and_never_relaxes(capture_helper):
    reference = np.array([0., 1., -2.], np.float32)
    assert capture_helper.compare(reference.copy(), reference, np)['array_equal']
    approximate = reference + np.float32(1e-6)
    result = capture_helper.compare(approximate, reference, np)
    assert result['elementwise_gate_passed'] and not result['array_equal']
    bad = reference.copy()
    bad[0] = np.float32(2e-5)
    result = capture_helper.compare(bad, reference, np)
    assert not result['elementwise_gate_passed'] and result['outside_tolerance'] == 1
    assert result['rtol'] == 1e-4 and result['atol'] == 1e-5
    with pytest.raises(ValueError, match='Finite'):
        capture_helper.compare(np.array([np.nan], np.float32), np.zeros(1, np.float32), np)


@pytest.fixture
def board_paths(capture_helper, monkeypatch, tmp_path):
    base = tmp_path / 'board'
    source = base / 'src'
    capture = base / 'experiments/frozen'
    capture.mkdir(parents=True)
    monkeypatch.chdir(base)
    monkeypatch.setattr(capture_helper, 'BASE', base)
    monkeypatch.setattr(capture_helper, 'SOURCE', source)
    monkeypatch.setattr(capture_helper, '__file__', str(source / 'deploy/rk3588/validation/capture.py'))
    monkeypatch.setattr(capture_helper.platform, 'machine', lambda: 'aarch64')
    return SimpleNamespace(capture_dir=capture, output=base / 'experiments/new')


def test_capture_board_guard_accepts_new_isolated_output_only(capture_helper, board_paths):
    assert capture_helper.require_board(board_paths) == (board_paths.capture_dir, board_paths.output)
    board_paths.output.mkdir()
    with pytest.raises(ValueError, match='existing'):
        capture_helper.require_board(board_paths)


def test_capture_refuses_dangling_output_and_host_machine(capture_helper, board_paths, monkeypatch):
    board_paths.output.symlink_to(board_paths.output.parent / 'missing')
    with pytest.raises(ValueError, match='existing'):
        capture_helper.require_board(board_paths)
    monkeypatch.setattr(capture_helper.platform, 'machine', lambda: 'x86_64')
    with pytest.raises(ValueError, match='only on the board'):
        capture_helper.require_board(board_paths)


def test_capture_refuses_nonexperiment_output_before_any_inference(capture_helper, board_paths):
    board_paths.output = board_paths.output.parents[1] / 'other'
    with pytest.raises(ValueError, match='strictly below'):
        capture_helper.require_board(board_paths)


@pytest.fixture
def exporter():
    path = ROOT / 'deploy/rk3588/validation/export-v19-attention.py'
    spec = importlib.util.spec_from_file_location('v19_attention_export_synthetic', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('shape,sizes', [((2, 16, 437, 64), [11, 11, 10]),
                                       ((1, 16, 874, 64), [6, 5, 5]),
                                       ((2, 5, 7, 64), [4, 3, 3])])
def test_real_layout_headers_and_small_synthetic_plan_are_complete_ordered(exporter, shape, sizes):
    plan = exporter.head_plan(shape)
    assert [row['independent_heads'] for row in plan] == sizes
    assert [index for row in plan for index in range(row['begin'], row['end'])] == list(range(shape[0] * shape[1]))
    assert [pair for row in plan for pair in row['original_head_indices']] == [
        dict(batch=batch, head=head) for batch in range(shape[0]) for head in range(shape[1])]
    assert all(row['input_shape'][2:] == list(shape[2:]) for row in plan)


def test_small_synthetic_head_partition_is_equivalent_and_reversible(exporter):
    rng = np.random.default_rng(19)
    shape = (2, 5, 7, 64)
    operands = [rng.standard_normal(shape).astype(np.float32) / 4 for _ in range(3)]
    original = synthetic_sdpa(*operands, dropout_p=0.0)
    flat = [exporter.flatten_heads(value) for value in operands]
    plan = exporter.head_plan(shape)
    outputs = [synthetic_sdpa(*(value[:, row['begin']:row['end']] for value in flat), dropout_p=0.0)
               for row in plan]
    restored = exporter.restore_heads(outputs, shape)
    np.testing.assert_array_equal(restored, original)
    roundtrip = exporter.restore_heads([flat[0][:, row['begin']:row['end']].copy() for row in plan], shape)
    np.testing.assert_array_equal(roundtrip, operands[0])


@pytest.mark.parametrize('shape', [(1, 2, 7, 64), (2, 3, 7, 32), (0, 3, 7, 64), (2, 3, 0, 64), (2, 3, 7)])
def test_head_plan_refuses_nonoriginal_dim_empty_or_unpartitionable_layout(exporter, shape):
    with pytest.raises(ValueError):
        exporter.head_plan(shape)


def test_inverse_rejects_lost_shard_and_incorrect_shape(exporter):
    shape = (2, 5, 7, 64)
    flat = np.zeros((1, 10, 7, 64), np.float32)
    shards = [flat[:, row['begin']:row['end']].copy() for row in exporter.head_plan(shape)]
    with pytest.raises(ValueError, match='three'):
        exporter.restore_heads(shards[:-1], shape)
    shards[0] = shards[0][:, :-1].copy()
    with pytest.raises(ValueError, match='shape'):
        exporter.restore_heads(shards, shape)


def test_graph_is_only_exact_complete_key_attention(exporter):
    onnx = pytest.importorskip('onnx')
    from onnx import numpy_helper
    model = exporter.build_attention_onnx(5, 7)
    onnx.checker.check_model(model, full_check=True)
    assert [node.op_type for node in model.graph.node] == ['Transpose', 'MatMul', 'Mul', 'Softmax', 'MatMul']
    assert [value.name for value in model.graph.input] == ['query', 'key', 'value']
    assert [value.name for value in model.graph.output] == ['output']
    assert len(model.graph.initializer) == 1
    assert float(numpy_helper.to_array(model.graph.initializer[0])) == 0.125
    softmax = model.graph.node[3]
    assert {attr.name: onnx.helper.get_attribute_value(attr) for attr in softmax.attribute} == {'axis': 3}
    for value in [*model.graph.input, *model.graph.output]:
        assert [dim.dim_value for dim in value.type.tensor_type.shape.dim] == [1, 5, 7, 64]
    scores = next(value for value in model.graph.value_info if value.name == 'attention_probabilities')
    assert [dim.dim_value for dim in scores.type.tensor_type.shape.dim] == [1, 5, 7, 7]


def test_only_synthetic_graph_execution_matches_full_and_three_shards(exporter):
    pytest.importorskip('onnx')
    from onnx.reference import ReferenceEvaluator
    rng = np.random.default_rng(1901)
    shape = (2, 5, 7, 64)
    operands = [rng.standard_normal(shape).astype(np.float32) / 3 for _ in range(3)]
    flat = [exporter.flatten_heads(value) for value in operands]
    full = ReferenceEvaluator(exporter.build_attention_onnx(10, 7)).run(
        ['output'], dict(zip(('query', 'key', 'value'), flat)))[0]
    shards = []
    for row in exporter.head_plan(shape):
        model = exporter.build_attention_onnx(row['independent_heads'], 7)
        result = ReferenceEvaluator(model).run(['output'], {
            name: value[:, row['begin']:row['end']] for name, value in zip(('query', 'key', 'value'), flat)})[0]
        shards.append(result)
    np.testing.assert_array_equal(np.concatenate(shards, axis=1), full)
    np.testing.assert_allclose(exporter.restore_heads(shards, shape), synthetic_sdpa(*operands, dropout_p=0.),
                               rtol=1e-6, atol=1e-7)


def test_export_writes_exact_operand_slices_and_failed_receipt(exporter, monkeypatch, tmp_path):
    pytest.importorskip('onnx')
    shape = (2, 5, 7, 64)
    rng = np.random.default_rng(91)
    operands = [rng.standard_normal(shape).astype(np.float32) / 4 for _ in range(3)]
    arrays = dict(zip(exporter.ARRAY_KEYS, [*operands, synthetic_sdpa(*operands, dropout_p=0.)]))
    identity = dict(block='synthetic', flattening_order='synthetic C-order', attention_semantics={'scale': 0.125},
                    checkpoint={}, selected_pair={}, profile_sha256='a'*64,
                    original_block_parameter_metadata={}, original_block_array_metadata={}, source_pins={})
    for key in ('unhooked_vs_frozen_reference', 'captured_vs_unhooked', 'captured_vs_frozen_reference'):
        identity[key] = dict(array_equal=True)
    monkeypatch.setattr(exporter, 'load_operands', lambda *_: (identity, arrays, {}))
    output = tmp_path / 'export'
    output.mkdir()
    report = exporter.export(tmp_path / 'synthetic-capture', 'b'*64, output)
    assert report['status'] == 'exported_and_structurally_checked'
    assert len(report['models']) == 4 and report['saved_reference_partition_inverse_bitwise_equal']
    assert not report['host_model_inference_executed'] and not report['host_ORT_executed']
    for row in report['models']:
        assert row['complete_key_count'] == 7 and not row['token_or_key_partitioned']
        assert row['input_names'] == ['query', 'key', 'value'] and row['output_names'] == ['output']
        for name in exporter.ARRAY_KEYS:
            saved = np.load(output / row['artifacts'][name+'.npy']['file'], allow_pickle=False)
            wanted = exporter.flatten_heads(arrays[name])[:, row['begin']:row['end']]
            np.testing.assert_array_equal(saved, wanted)
    failed = tmp_path / 'failed'
    failed.mkdir()
    monkeypatch.setattr(exporter, 'load_operands', lambda *_: (_ for _ in ()).throw(ValueError('source mismatch')))
    report = exporter.export(tmp_path / 'synthetic-capture', 'b'*64, failed)
    assert report['status'] == 'failed' and 'source mismatch' in report['error']
    assert (failed / 'export.json').is_file() and report['models'] == []
