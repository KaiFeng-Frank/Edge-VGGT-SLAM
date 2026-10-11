"""Tiny generated topology/fake graph tests; no original weights or forwards."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
DRIVER = HERE / 'export-v36-grouped-update-board.py'
if not DRIVER.is_file():
    DRIVER = HERE.parent / 'deploy/rk3588/validation/dpvo_grouped_update/export-v36-grouped-update-board.py'
spec = importlib.util.spec_from_file_location('_v36_grouped_export_test_driver', DRIVER)
driver = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = driver
spec.loader.exec_module(driver)


@pytest.fixture
def receipts():
    capture_sha, parent_sha, manifest_sha = 'a'*64, 'b'*64, 'c'*64
    capture = dict(schema='v36_full_M48_original_Update_capture',
        status='completed_original_M48_capture_not_performance_or_candidate',
        script_sha256=driver.CAPTURE_HELPER_SHA, parent_runner_sha256=driver.PARENT_SHA,
        capture_observations=[13, 127, 191], captured=3,
        records=[dict(observation_id=value) for value in (13, 127, 191)],
        cleanup_errors=[], original_loader_and_stream_class_restored=True,
        parent_runner_source_unchanged=True, original_Update_replaced=False,
        original_weights_or_RNG_changed=False,
        original_parent_receipt=dict(run_sha256=parent_sha, input_manifest_sha256=manifest_sha),
        original_call_attempts=2, original_call_successes=2)
    parent = dict(schema='v35_DPVO_original_FAST_dense_spatial_RKNN_encoder_diagnostic_replay',
        status='completed_diagnostic_rejected_not_adopted', script_sha256=driver.PARENT_SHA,
        cleanup_errors=[], original_weights_unchanged=True, config=dict(patches_per_frame=48),
        original_weights_before={'tiny_weight': 1}, original_weights_after={'tiny_weight': 1},
        sources_and_inputs_unchanged=True, source_pins={'source': 'd'*64}, source_hashes_after={'source': 'd'*64},
        input_manifest_sha256=manifest_sha, update_calls=[dict(index=0), dict(index=1)])
    audit = dict(schema='v36_saved_M48_capture_independent_audit',
        status='passed_saved_capture_identity_not_neural_equivalence', auditor_sha256=driver.CAPTURE_AUDITOR_SHA,
        input_bindings=dict(capture_sha256=capture_sha, parent_run_sha256=parent_sha,
                            parent_input_manifest_sha256=manifest_sha,
                            adapter_sha256=driver.UPDATE_PINS['source_frame_batches.py']),
        checks=[dict(name='fake_saved_identity', passed=True)])
    return capture, parent, audit, dict(capture_sha=capture_sha, parent_sha=parent_sha, manifest_sha=manifest_sha)


def test_saved_identity_status_allows_parent_encoder_rejection_without_claiming_neural_equivalence(receipts):
    capture, parent, audit, hashes = receipts
    assert driver.validate_receipts(capture, parent, audit, **hashes) is None
    assert 'not_neural_equivalence' in audit['status']


@pytest.mark.parametrize('section,key,value', [
    ('audit', 'status', 'passed_neural_equivalence'),
    ('audit', 'auditor_sha256', '0'*64),
    ('capture', 'original_Update_replaced', True),
    ('capture', 'original_weights_or_RNG_changed', True),
    ('capture', 'captured', 2),
    ('parent', 'sources_and_inputs_unchanged', False),
    ('parent', 'cleanup_errors', ['failed cleanup']),
    ('capture', 'original_call_successes', 1),
])
def test_receipt_provenance_failure_is_not_numerical_acceptance(receipts, section, key, value):
    capture, parent, audit, hashes = receipts
    {'capture': capture, 'parent': parent, 'audit': audit}[section][key] = value
    with pytest.raises(ValueError):
        driver.validate_receipts(capture, parent, audit, **hashes)


def test_audit_input_binding_and_failed_check_are_rejected(receipts):
    capture, parent, audit, hashes = receipts
    changed = copy.deepcopy(audit)
    changed['input_bindings']['parent_input_manifest_sha256'] = 'e'*64
    with pytest.raises(ValueError, match='capture audit'):
        driver.validate_receipts(capture, parent, changed, **hashes)
    audit['checks'][0]['passed'] = False
    with pytest.raises(ValueError, match='capture audit'):
        driver.validate_receipts(capture, parent, audit, **hashes)


def test_predeclared_observation_order_is_required(receipts):
    capture, parent, audit, hashes = receipts
    capture['records'].reverse()
    with pytest.raises(ValueError, match='observations'):
        driver.validate_receipts(capture, parent, audit, **hashes)


def test_board_guard_precedes_reference_reads_output_or_model_import(tmp_path, monkeypatch):
    def reject():
        raise ValueError('synthetic host rejection')
    monkeypatch.setattr(driver, 'board_guard', reject)
    def unexpected_read(*args, **kwargs):
        raise AssertionError('Reference was read before board guard')
    monkeypatch.setattr(driver, 'sha', unexpected_read)
    output = tmp_path/'never-created'
    arguments = []
    for name in ('capture-dir', 'parent-run-dir', 'capture-audit', 'update-source-dir', 'v26-source-dir', 'checkpoint'):
        arguments.extend(('--'+name, str(tmp_path/name)))
    arguments.extend(('--output-dir', str(output), '--capture-sha256', 'a'*64, '--capture-audit-sha256', 'b'*64))
    with pytest.raises(ValueError, match='host rejection'):
        driver.main(arguments)
    assert not output.exists()


def test_strict_gate_boundary_never_uses_relative_l2_as_acceptance():
    reference = np.zeros((1, 2, 2), dtype=np.float32)
    actual = reference.copy()
    actual.flat[0] = np.nextafter(np.float32(1e-5), np.float32(0))
    assert driver.compare(actual, reference)['gate_passed']
    actual.flat[0] = np.nextafter(np.float32(1e-5), np.float32(np.inf))
    report = driver.compare(actual, reference)
    assert report['failed_elements'] == 1 and not report['gate_passed']
    assert report['relative_L2'] is None and report['rtol'] == 1e-4 and report['atol'] == 1e-5


def test_nonfinite_comparison_remains_json_safe_rejected():
    reference = np.zeros((1, 2, 2), dtype=np.float32)
    actual = reference.copy()
    actual.flat[0] = np.nan
    report = driver.compare(actual, reference)
    assert report['finite'] is False and report['gate_passed'] is False
    assert report['max_abs'] is None and report['relative_L2'] is None
    driver.json.dumps(report, allow_nan=False)


def test_fullgraph_domain_does_not_clip_micro_negative_weight():
    outputs = {name: np.ones((1, 3, channels), dtype=np.float32)
               for name, channels in zip(driver.CANONICAL_OUTPUTS, (384, 2, 2))}
    outputs['weight'].flat[0] = np.float32(-1e-7)
    domain = driver.validate_full_outputs(outputs, 3)
    assert domain == dict(finite=True, weight_domain_valid=False)
    assert outputs['weight'].flat[0] < 0


def test_contracts_use_actual_B_D_M_and_output_names():
    batch = NS(source_slots=(0, 2, 5), degree=7, patches=48,
               shape=lambda c: (3, c, 7, 48))
    assert driver.model_key(batch) == 'B3_D7_M48'
    inputs, outputs = driver.contracts(batch)
    assert [item['name'] for item in inputs] == ['net', 'inp', 'corr']
    assert inputs[2]['shape'] == [3, 882, 7, 48]
    assert outputs == [dict(name=name, dtype='float32', shape=[3, c, 7, 48])
                       for name, c in zip(('net_out', 'delta', 'weight'), (384, 2, 2))]


@pytest.fixture
def tiny_grouped():
    source = HERE.parent/'update/source_frame_batches.py'
    if not source.is_file():
        source = DRIVER.with_name('source_frame_batches.py')
    assert driver.sha(source) == driver.UPDATE_PINS[source.name]
    adapter = driver.module(source, '_v36_tiny_frozen_degree_adapter')
    graph = [(source, target, patch) for source, targets in ((0, (0, 4)), (2, (1, 3)), (5, (0, 2, 4)))
             for target in targets for patch in range(2)]
    order = np.random.default_rng(36).permutation(len(graph))
    graph = np.array(graph, dtype=np.int64)[order]
    ii, jj, kk = graph[:, 0], graph[:, 1], graph[:, 0]*2+graph[:, 2]
    plan = adapter.build_degree_plan(ii, jj, kk, patches_per_frame=2)
    edges = len(graph)
    arrays = dict(net=np.arange(edges*4, dtype=np.float32).reshape(1, edges, 4),
                  inp=np.full((1, edges, 4), 0.5, dtype=np.float32),
                  corr=np.arange(edges*6, dtype=np.float32).reshape(1, edges, 6))
    return adapter, plan, arrays, adapter.SnapshotIdentity(1, 2, 3, 4)


def test_grouped_run_restores_unsorted_full_edges_and_preserves_true_batches(tiny_grouped):
    adapter, plan, arrays, identity = tiny_grouped
    before = {key: value.tobytes() for key, value in arrays.items()}
    def execute(batch, feeds):
        assert batch.shape(4)[0] == len(batch.source_slots)
        return dict(net=feeds['net']+feeds['inp'], delta=feeds['corr'][:, :2].copy(),
                    weight=np.full(batch.shape(2), 0.25, dtype=np.float32))
    output, calls, seconds = driver.grouped_run(adapter, plan, arrays, execute, identity)
    assert [len(item['batch'].source_slots) for item in calls] == [2, 1]
    assert np.array_equal(output['net'], arrays['net']+arrays['inp'])
    assert np.array_equal(output['delta'], arrays['corr'][:, :, :2])
    assert np.array_equal(output['weight'], np.full(output['weight'].shape, 0.25, dtype=np.float32))
    assert {key: value.tobytes() for key, value in arrays.items()} == before
    assert seconds >= sum(item['callback_seconds'] for item in calls) >= 0
    for item in calls:
        for value in item['feeds'].values():
            assert value.flags.c_contiguous and value.flags.owndata


def test_partial_group_callback_failure_does_not_return_or_mutate_fullgraph(tiny_grouped):
    adapter, plan, arrays, identity = tiny_grouped
    before = {key: value.tobytes() for key, value in arrays.items()}
    called = []
    def execute(batch, feeds):
        called.append(batch.degree)
        if len(called) == 2:
            raise RuntimeError('synthetic second group failed')
        return dict(net=feeds['net'].copy(), delta=feeds['corr'][:, :2].copy(),
                    weight=np.full(batch.shape(2), 0.5, dtype=np.float32))
    with pytest.raises(RuntimeError, match='second group failed'):
        driver.grouped_run(adapter, plan, arrays, execute, identity)
    assert called == [2, 3]
    assert {key: value.tobytes() for key, value in arrays.items()} == before


@pytest.fixture
def fake_graph():
    parameters = {f'p{i:02d}': np.full((2, 3) if i % 2 else (3,), i+0.25, dtype=np.float32)
                  for i in range(50)}
    tensors = []
    for key, value in parameters.items():
        reshaped = value.reshape(*value.shape, 1, 1) if value.ndim == 2 else value.reshape(1, -1, 1, 1)
        tensors.append(NS(name=key+'_reshaped', external_data=[], data_location=0, value=reshaped))
    onnx = NS(TensorProto=NS(EXTERNAL=1), numpy_helper=NS(to_array=lambda tensor: tensor.value))
    return NS(graph=NS(initializer=tensors)), parameters, onnx


def test_all50_raw_parameters_cover_only_exact_reshape_initializers(fake_graph):
    graph, parameters, onnx = fake_graph
    report = driver.initializer_coverage(graph, parameters, onnx)
    assert report['all_original_parameters_covered'] and report['original_parameter_count'] == 50
    assert len(report['mapping']) == 50
    assert {item['original_matches'][0]['transform'] for item in report['mapping']} == {'conv1x1_reshape', 'channel_affine_reshape'}


@pytest.mark.parametrize('failure', ['missing', 'changed', 'external'])
def test_initializer_missing_changed_and_external_data_are_rejected(fake_graph, failure):
    graph, parameters, onnx = fake_graph
    if failure == 'missing':
        graph.graph.initializer.pop()
    elif failure == 'changed':
        graph.graph.initializer[0].value = graph.graph.initializer[0].value.copy()
        graph.graph.initializer[0].value.flat[0] += 1
    else:
        graph.graph.initializer[0].external_data = [NS(key='location', value='weights.bin')]
    with pytest.raises(ValueError):
        driver.initializer_coverage(graph, parameters, onnx)


def test_graph_contract_rejects_symbolic_shape_and_integer_type():
    def value(element_type=1, symbolic=False):
        dims = [NS(dim_value=n, HasField=lambda _key, exists=not symbolic: exists) for n in (2, 384, 8, 48)]
        return NS(name='net', type=NS(tensor_type=NS(elem_type=element_type, shape=NS(dim=dims))))
    assert driver.graph_contract(value()) == dict(name='net', dtype='float32', shape=[2, 384, 8, 48])
    with pytest.raises(ValueError, match='symbolic'):
        driver.graph_contract(value(symbolic=True))
    with pytest.raises(ValueError, match='FP32'):
        driver.graph_contract(value(element_type=7))
