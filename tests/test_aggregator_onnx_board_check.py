"""Synthetic arrays/fake ORT only; no real images, captured tensors or models."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def checker():
    path = ROOT / 'deploy/rk3588/validation/check-v17-aggregator-onnx-board.py'
    spec = importlib.util.spec_from_file_location('aggregator_ort_check_synthetic', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metadata(checker, a):
    return dict(shape=list(a.shape), dtype=str(a.dtype), bytes=a.nbytes, sha256=checker.raw_sha(a))


@pytest.fixture
def saved(checker, monkeypatch, tmp_path):
    capture = tmp_path / 'capture'
    export_dir = tmp_path / 'export'
    helpers = tmp_path / 'helpers'
    for path in (capture, export_dir, helpers):
        path.mkdir()
    (capture / 'profile.json').write_text('{"synthetic":true}')
    for name in checker.EXPORTERS:
        (helpers / name).write_text(f'# synthetic {name} loader, never executed\n')
    (helpers / 'synthetic_graph_helper.py').write_text('# synthetic graph builder\n')
    monkeypatch.setattr(checker, 'HERE', helpers)
    a = np.arange(56, dtype=np.float32).reshape(2, 7, 4) / 10
    arrays = dict(input=a.copy(), output=(a + .25).copy())
    row = dict(arrays={key: metadata(checker, value) for key, value in arrays.items()},
               artifacts=dict(tensors=dict(file='synthetic-tensors.npz', sha256='1' * 64),
                              weights=dict(file='synthetic-weights.npz', sha256='2' * 64)),
               weights=dict(synthetic_parameter=dict(sha256='3' * 64)))
    architecture = dict(num_heads=2, head_dim=2, norm1_eps=1e-5, norm2_eps=1e-5,
                        q_norm_eps=1e-5, k_norm_eps=1e-5, gelu_approximate='none',
                        rope_frequency=100, rope_scaling_factor=1.)
    row.update(architecture)
    profile = dict(checkpoint=dict(sha256='4' * 64, bytes=123),
                   selected_pair=dict(frame_ids=[1., 2.], revision=1))
    pins = {str(capture / 'profile.json'): checker.sha(capture / 'profile.json')}
    exporter = SimpleNamespace(
        SOURCE_PINS={'synthetic/source.py': '5' * 64},
        GRAPH_HELPERS={'synthetic_graph_helper.py': checker.sha(helpers / 'synthetic_graph_helper.py')},
        load_capture=lambda *_: (profile, row, arrays, {}, pins))
    monkeypatch.setattr(checker, 'load_exporter', lambda *_: exporter)
    args = SimpleNamespace(capture_dir=capture, export_receipt=export_dir / 'export.json',
                           exporter_name=checker.EXPORTERS[0],
                           exporter_sha256=checker.sha(helpers / checker.EXPORTERS[0]),
                           profile_sha256=pins[str(capture / 'profile.json')], block='frame0')
    artifacts = {}
    for key, value in arrays.items():
        path = export_dir / (key + '.npy')
        np.save(path, value, allow_pickle=False)
        artifacts[path.name] = dict(file=path.name, bytes=path.stat().st_size,
            sha256=checker.sha(path), shape=list(value.shape), dtype='float32', raw_sha256=checker.raw_sha(value))
    model = export_dir / 'block.onnx'
    model.write_bytes(b'SYNTHETIC not an ONNX model')
    export = dict(status='exported_and_structurally_checked', script_sha256=args.exporter_sha256,
        profile_sha256=args.profile_sha256, block='frame0', input_shape=list(a.shape),
        input_name='input', output_name='output', original_weights=True, parameter_count=18,
        position_and_tables_fixed=True, host_trigonometry_recomputed=False,
        host_model_inference_executed=False, host_ORT_executed=False,
        external_data=False, external_data_used=False,
        capture_artifacts=row['artifacts'], captured_array_metadata=row['arrays'],
        captured_parameter_metadata=row['weights'], source_pins=exporter.SOURCE_PINS,
        frozen_helpers=exporter.GRAPH_HELPERS, architecture=architecture,
        selected_pair=profile['selected_pair'], checkpoint=profile['checkpoint'], artifacts=artifacts,
        onnx=dict(file=model.name, bytes=model.stat().st_size, sha256=checker.sha(model)))

    def write_export():
        args.export_receipt.write_text(json.dumps(export))
        args.export_sha256 = checker.sha(args.export_receipt)

    write_export()
    return args, profile, row, arrays, exporter, export, write_export


@pytest.mark.parametrize('exporter_name', ['export-v17-aggregator-block-r2.py',
                                         'export-v17-aggregator-block-r3.py'])
def test_exact_original_array_binding_and_versioned_name_are_recorded(checker, saved, exporter_name):
    args, _, _, arrays, _, export, write_export = saved
    tokens, reference, blob, pins, identity = checker.prepare_inputs(args, np)
    assert np.array_equal(tokens, arrays['input'])
    assert np.array_equal(reference, arrays['output'])
    assert blob == b'SYNTHETIC not an ONNX model'
    assert identity['original_input_and_reference_exact'] and identity['no_external_data']
    assert all(checker.sha(path) == expected for path, expected in pins.items())
    args.exporter_name = exporter_name
    args.exporter_sha256 = checker.sha(checker.HERE / args.exporter_name)
    export['script_sha256'] = args.exporter_sha256
    write_export()
    _, _, _, pins, identity = checker.prepare_inputs(args, np)
    assert identity['exporter_name'] == exporter_name
    assert pins[str((checker.HERE / args.exporter_name).resolve())] == args.exporter_sha256


@pytest.mark.parametrize('field', ['script_sha256', 'profile_sha256', 'block', 'external_data_used',
                                  'position_and_tables_fixed', 'architecture', 'selected_pair',
                                  'captured_parameter_metadata', 'source_pins', 'frozen_helpers'])
def test_export_schema_and_original_source_mismatches_are_rejected(checker, saved, field):
    args, _, _, _, _, export, write_export = saved
    export[field] = True if field == 'external_data_used' else None
    write_export()
    with pytest.raises(ValueError, match='Export'):
        checker.prepare_inputs(args, np)


def test_different_reference_cannot_be_authorized_by_its_own_updated_file_sha(checker, saved):
    args, _, _, _, _, export, write_export = saved
    path = args.export_receipt.parent / 'output.npy'
    altered = np.load(path, allow_pickle=False)
    altered[0, 0, 0] += 1
    np.save(path, altered, allow_pickle=False)
    export['artifacts']['output.npy']['sha256'] = checker.sha(path)
    write_export()
    with pytest.raises(ValueError, match='values differ'):
        checker.prepare_inputs(args, np)


def test_corrupted_onnx_is_rejected_before_a_session(checker, saved):
    args = saved[0]
    (args.export_receipt.parent / 'block.onnx').write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA mismatch'):
        checker.prepare_inputs(args, np)


class FakeSession:
    def __init__(self, reference, *, shift=0., second_shift=None):
        self.reference = reference
        self.inputs = [SimpleNamespace(name='input', type='tensor(float)', shape=list(reference.shape))]
        self.outputs = [SimpleNamespace(name='output', type='tensor(float)', shape=list(reference.shape))]
        self.providers = ['CPUExecutionProvider']
        self.calls = 0
        self.shift, self.second_shift = shift, second_shift
        self.fallback_disabled = False

    def disable_fallback(self): self.fallback_disabled = True
    def get_providers(self): return self.providers
    def get_inputs(self): return self.inputs
    def get_outputs(self): return self.outputs

    def run(self, outputs, feeds):
        assert outputs == ['output'] and set(feeds) == {'input'}
        value = self.reference.copy()
        value += self.shift if self.calls == 0 or self.second_shift is None else self.second_shift
        self.calls += 1
        return [value]


@pytest.mark.parametrize('bad', ['provider', 'extra_input', 'extra_output', 'name', 'type', 'symbolic', 'shape'])
def test_runtime_contract_is_one_cpu_fixed_fp32_input_and_output(checker, bad):
    shape = (2, 7, 4)
    session = FakeSession(np.ones(shape, np.float32))
    if bad == 'provider': session.providers.append('CUDAExecutionProvider')
    if bad == 'extra_input': session.inputs.append(session.inputs[0])
    if bad == 'extra_output': session.outputs.append(session.outputs[0])
    if bad == 'name': session.inputs[0].name = 'tokens'
    if bad == 'type': session.outputs[0].type = 'tensor(float16)'
    if bad == 'symbolic': session.inputs[0].shape[0] = 'batch'
    if bad == 'shape': session.outputs[0].shape[1] += 1
    with pytest.raises(ValueError): checker.check_session(session, shape)


def test_four_calls_save_outputs_outside_timing_and_compare_original_reference(checker, tmp_path):
    reference = np.arange(56, dtype=np.float32).reshape(2, 7, 4)
    session = FakeSession(reference)
    receipt = dict(successful_session_run_count=0, runs=[])
    assert checker.check_session(session, reference.shape)['output']['name'] == 'output'
    checker.execute_runs(session, reference, reference, tmp_path, receipt, np)
    assert session.calls == checker.CALLS == 4
    assert session.fallback_disabled and receipt['all_runs_within_tolerance']
    assert receipt['repeated_ORT_elementwise_equal'] and len(receipt['warm_seconds']) == 3
    for index, row in enumerate(receipt['runs']):
        path = tmp_path / row['actual']['file']
        assert np.array_equal(np.load(path, allow_pickle=False), reference)
        assert checker.sha(path) == row['actual']['sha256']
        assert row['phase'] == ('cold_first_run' if index == 0 else 'warm')
        assert row['comparison']['outside_tolerance_elements'] == 0


def test_allclose_failure_preserves_actual_arrays_and_is_not_a_success(checker, tmp_path):
    reference = np.ones((1, 3, 4), np.float32)
    receipt = dict(successful_session_run_count=0, runs=[])
    with pytest.raises(ValueError, match='predefined FP32 gate'):
        checker.execute_runs(FakeSession(reference, shift=.001), reference, reference, tmp_path, receipt, np)
    assert receipt['successful_session_run_count'] == 4 and not receipt['all_runs_within_tolerance']
    assert len(list(tmp_path.glob('actual_*.npy'))) == 4


def test_nonfinite_actual_is_saved_but_cannot_pass(checker, tmp_path):
    reference = np.ones((1, 3, 4), np.float32)
    receipt = dict(successful_session_run_count=0, runs=[])
    with pytest.raises(ValueError, match='finite'):
        checker.execute_runs(FakeSession(reference, shift=np.nan), reference, reference, tmp_path, receipt, np)
    assert (tmp_path / 'actual_00.npy').is_file()
    assert not np.isfinite(np.load(tmp_path / 'actual_00.npy', allow_pickle=False)).all()


def test_repeat_instability_fails_even_when_every_call_meets_gate(checker, tmp_path):
    reference = np.ones((1, 3, 4), np.float32)
    receipt = dict(successful_session_run_count=0, runs=[])
    with pytest.raises(ValueError, match='elementwise identical'):
        checker.execute_runs(FakeSession(reference, second_shift=1e-6), reference, reference, tmp_path, receipt, np)
    assert receipt['all_runs_within_tolerance'] and not receipt['repeated_ORT_elementwise_equal']


@pytest.mark.parametrize('bad', [None, 'wrong_output', 'mutated_input'])
def test_complete_fake_runtime_has_separate_timings_and_final_identity_gate(checker, saved, monkeypatch, tmp_path, bad):
    args, _, _, arrays, _, _, _ = saved
    args.output = tmp_path / 'actual_check'
    args.output.mkdir()
    monkeypatch.setattr(checker, 'require_board', lambda: None)
    monkeypatch.setattr(checker, 'configure_cpu', lambda: ({}, {'OPENBLAS_NUM_THREADS': '1'}))
    monkeypatch.setattr(checker.os, 'sched_getaffinity', lambda _: {6, 7})
    session = FakeSession(arrays['output'], shift=.001 if bad == 'wrong_output' else 0.)
    ordinary_run = session.run

    def run(outputs, feeds):
        result = ordinary_run(outputs, feeds)
        if bad == 'mutated_input': feeds['input'][0, 0, 0] += 1
        return result

    session.run = run

    def create(blob, *, sess_options, providers):
        assert blob == b'SYNTHETIC not an ONNX model'
        assert providers == ['CPUExecutionProvider']
        assert sess_options.intra_op_num_threads == 2 and sess_options.inter_op_num_threads == 1
        assert sess_options.execution_mode == 'sequential'
        return session

    fake = SimpleNamespace(__version__='synthetic_no_runtime', SessionOptions=SimpleNamespace,
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL='sequential'),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL='all'), InferenceSession=create)
    monkeypatch.setitem(sys.modules, 'onnxruntime', fake)
    receipt = checker.run(args)
    assert receipt['successful_session_run_count'] == 4 and receipt['session_released']
    assert receipt['status'] == ('passed' if bad is None else 'failed')
    if bad is None:
        assert receipt['input_tensor_unchanged'] and receipt['input_files_unchanged']
        assert receipt['identity']['original_input_and_reference_exact']
    else:
        assert ('predefined FP32' if bad == 'wrong_output' else 'changed during') in receipt['error']
    assert len(list(args.output.glob('actual_*.npy'))) == 4
    assert json.loads((args.output / 'check.json').read_text()) == receipt


def cli(checker, tmp_path):
    return ['--capture-dir', str(tmp_path / 'capture'), '--export-receipt', str(tmp_path / 'export.json'),
            '--profile-sha256', '1' * 64, '--export-sha256', '2' * 64,
            '--exporter-sha256', '3' * 64, '--block', 'frame0', '--output', str(tmp_path / 'output')]


@pytest.mark.parametrize('name', ['export-v17-aggregator-block.py', 'export-v17-aggregator-block-r2.py',
                                 'export-v17-aggregator-block-r3.py'])
def test_recorded_exporter_names_reach_board_guard_without_host_inference(checker, monkeypatch, tmp_path, capsys, name):
    def guard():
        raise ValueError('synthetic board guard after accepted exporter choice')
    monkeypatch.setattr(checker, 'require_board', guard)
    with pytest.raises(SystemExit) as exc:
        checker.main(cli(checker, tmp_path) + ['--exporter-name', name])
    assert exc.value.code == 2
    assert 'synthetic board guard after accepted exporter choice' in capsys.readouterr().err
    assert not (tmp_path / 'output').exists()


def test_host_is_rejected_before_importing_runtime_or_creating_output(checker, monkeypatch, tmp_path):
    monkeypatch.setattr(checker.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(checker, 'run', lambda _: pytest.fail('No real host runtime'))
    with pytest.raises(SystemExit) as exc: checker.main(cli(checker, tmp_path))
    assert exc.value.code == 2 and not (tmp_path / 'output').exists()


def test_existing_broken_symlink_output_is_not_followed_or_overwritten(checker, monkeypatch, tmp_path):
    monkeypatch.setattr(checker, 'require_board', lambda: None)
    target = tmp_path / 'must-not-be-created'
    (tmp_path / 'output').symlink_to(target)
    with pytest.raises(SystemExit) as exc: checker.main(cli(checker, tmp_path))
    assert exc.value.code == 2 and not target.exists()


def test_runtime_failure_writes_new_failure_receipt(checker, monkeypatch, tmp_path):
    args = SimpleNamespace(block='frame0', exporter_name=checker.EXPORTERS[0],
        profile_sha256='1' * 64, export_sha256='2' * 64, exporter_sha256='3' * 64, output=tmp_path)
    monkeypatch.setattr(checker, 'require_board', lambda: None)
    monkeypatch.setattr(checker, 'configure_cpu', lambda: ({}, {}))
    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(__version__='synthetic'))
    monkeypatch.setattr(checker.os, 'sched_getaffinity', lambda _: {6, 7})
    monkeypatch.setattr(checker, 'prepare_inputs', lambda *_: (_ for _ in ()).throw(ValueError('synthetic corrupt reference')))
    receipt = checker.run(args)
    assert receipt['status'] == 'failed' and 'synthetic corrupt reference' in receipt['error']
    assert receipt['successful_session_run_count'] == 0 and receipt['session_released']
    assert json.loads((tmp_path / 'check.json').read_text()) == receipt
    assert not receipt['npu_used'] and not receipt['Torch_model_executed']


@pytest.mark.parametrize('bad', ['x' * 64, 'A' * 64, '123'])
def test_explicit_sha_format_is_strict(checker, bad):
    with pytest.raises(argparse.ArgumentTypeError): checker.sha_argument(bad)
