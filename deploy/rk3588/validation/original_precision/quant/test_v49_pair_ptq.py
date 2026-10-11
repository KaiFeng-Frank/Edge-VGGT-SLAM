"""Tiny generated NPY files/fake compiler only; no SDK import or calibration."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def module(name):
    path = HERE / name
    spec = importlib.util.spec_from_file_location(path.stem.replace('-', '_'), path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


C = module('v49_quant_contracts.py')
Q = module('compile-v49-pair-ptq-board.py')


def write_array(directory, relative, array):
    path = directory / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=False)
    return {'file': relative, 'bytes': path.stat().st_size, 'sha256': C.sha(path),
            'shape': list(array.shape), 'dtype': str(array.dtype), 'tensor_bytes': array.nbytes,
            'raw_sha256': hashlib.sha256(memoryview(array).cast('B')).hexdigest()}


@pytest.fixture
def saved_inputs(tmp_path, monkeypatch):
    # Only twenty-four floating elements per sample. Production source retains
    # its fixed original [2,437,1024] shape; these substitutions are test-only.
    monkeypatch.setattr(C, 'INPUT_SHAPE', (2, 3, 4))
    monkeypatch.setattr(C, 'POSITIONS_SHAPE', (2, 3, 2))
    directory = tmp_path / 'capture'
    directory.mkdir()
    output = tmp_path / 'out'
    output.mkdir()
    positions = np.arange(12, dtype=np.int64).reshape(C.POSITIONS_SHAPE)
    position_raw = hashlib.sha256(memoryview(positions).cast('B')).hexdigest()
    original_raw = 'a' * 64
    pair = {'artifacts': {'input.npy': {'raw_sha256': original_raw},
                          'positions.npy': {'raw_sha256': position_raw}},
            'selected_pair': {'images': [{'sha256': 'b' * 64}, {'sha256': 'c' * 64}]}}
    capture = {'frozen_pair': {'export_sha256': C.PAIR_EXPORT_SHA, 'onnx_sha256': C.PAIR_GRAPH_SHA,
                              'original_input_raw_sha256': original_raw,
                              'original_positions_raw_sha256': position_raw},
               'checkpoint': {'sha256': C.CHECKPOINT_SHA, 'bytes': 5026874952}, 'samples': []}
    for index in range(10):
        split = 'calibration' if index < 8 else 'evaluation'
        sample_id = split + '_' + str(index)
        prefix = 'samples/' + sample_id + '/'
        input_array = (np.arange(24, dtype=np.float32) + np.float32(index * 30)).reshape(C.INPUT_SHAPE)
        artifacts = {'input.npy': write_array(directory, prefix + 'input.npy', input_array),
                     'positions.npy': write_array(directory, prefix + 'positions.npy', positions)}
        if split == 'evaluation':
            artifacts['frame-reference.npy'] = write_array(directory, prefix + 'frame-reference.npy', input_array + 1)
            artifacts['output.npy'] = write_array(directory, prefix + 'output.npy', (input_array + 2).reshape(1, 6, 4))
        images = [{'sha256': hashlib.sha256(f'tiny-generated-image-{index}-{i}'.encode()).hexdigest()}
                  for i in range(2)]
        capture['samples'].append({'sample_id': sample_id, 'split': split,
                                   'images': images, 'artifacts': artifacts})
    return directory, capture, pair, output


def test_dataset_has_eight_original_inputs_and_excludes_heldout_refs_positions(saved_inputs):
    directory, capture, pair, output = saved_inputs
    pins = {}
    path, receipt = C.prepare_dataset(directory, capture, pair, output, pins)
    lines = path.read_text().splitlines()
    assert len(lines) == 8 and len(set(lines)) == 8
    assert all(line.endswith('/input.npy') and '/calibration_' in line for line in lines)
    assert all('/evaluation_' not in line and 'positions' not in line for line in lines)
    assert receipt['line_count'] == 8 and receipt['input_count'] == 1
    assert len(receipt['heldout_samples_excluded']) == 2
    assert pins[str(path)] == C.sha(path)
    assert C.recheck(pins)[1]
    with pytest.raises(FileExistsError):
        C.prepare_dataset(directory, capture, pair, output, {})


@pytest.mark.parametrize('failure', ['same_calib_tensor', 'calib_eval_tensor', 'calib_eval_image',
                                     'within_split_image', 'frozen_image', 'frozen_input',
                                     'calib_extra_reference', 'eval_missing_reference', 'same_sample_id',
                                     'missing_calib', 'unknown_split'])
def test_dataset_leakage_or_wrong_split_rejected_before_build(saved_inputs, failure):
    directory, capture, pair, output = saved_inputs
    samples = capture['samples']
    if failure == 'same_calib_tensor':
        samples[1]['artifacts']['input.npy'] = copy.deepcopy(samples[0]['artifacts']['input.npy'])
    elif failure == 'calib_eval_tensor':
        samples[8]['artifacts']['input.npy'] = copy.deepcopy(samples[0]['artifacts']['input.npy'])
    elif failure == 'calib_eval_image':
        samples[8]['images'][0] = copy.deepcopy(samples[0]['images'][0])
    elif failure == 'within_split_image':
        samples[1]['images'][0] = copy.deepcopy(samples[0]['images'][0])
    elif failure == 'frozen_image':
        samples[0]['images'][0]['sha256'] = pair['selected_pair']['images'][0]['sha256']
    elif failure == 'frozen_input':
        samples[0]['artifacts']['input.npy']['raw_sha256'] = capture['frozen_pair']['original_input_raw_sha256']
    elif failure == 'calib_extra_reference':
        samples[0]['artifacts']['output.npy'] = copy.deepcopy(samples[8]['artifacts']['output.npy'])
    elif failure == 'eval_missing_reference':
        del samples[8]['artifacts']['output.npy']
    elif failure == 'same_sample_id':
        samples[8]['sample_id'] = samples[0]['sample_id']
    elif failure == 'unknown_split':
        samples[0]['split'] = 'unknown'
    else:
        samples.pop(0)
    with pytest.raises(ValueError):
        C.prepare_dataset(directory, capture, pair, output, {})
    assert not (output / 'dataset.txt').exists()


@pytest.mark.parametrize('failure', ['raw_sha', 'position_order', 'tensor_bytes', 'shape', 'actual_fp16',
                                     'actual_nonfinite', 'artifact_escapes', 'whitespace'])
def test_actual_npy_and_confined_sdk_path_checked(saved_inputs, failure):
    directory, capture, pair, output = saved_inputs
    sample = capture['samples'][0]
    item = sample['artifacts']['input.npy']
    if failure == 'raw_sha':
        item['raw_sha256'] = 'f' * 64
    elif failure == 'position_order':
        sample['artifacts']['positions.npy']['raw_sha256'] = 'f' * 64
    elif failure == 'tensor_bytes':
        item['tensor_bytes'] -= 4
    elif failure == 'shape':
        item['shape'] = [1, 6, 4]
    elif failure in ('actual_fp16', 'actual_nonfinite'):
        actual = np.ones(C.INPUT_SHAPE, dtype=np.float16 if failure == 'actual_fp16' else np.float32)
        if failure == 'actual_nonfinite':
            actual.flat[0] = np.inf
        changed = write_array(directory, item['file'], actual)
        sample['artifacts']['input.npy'] = changed
    elif failure == 'artifact_escapes':
        item['file'] = '../outside.npy'
    else:
        item['file'] = 'samples/path with whitespace.npy'
    with pytest.raises(ValueError):
        C.prepare_dataset(directory, capture, pair, output, {})
    assert not (output / 'dataset.txt').exists()


class FakeToolkit:
    def __init__(self, fail=None, release_error=False):
        self.calls, self.fail, self.release_error = [], fail, release_error

    def config(self, **kw):
        self.calls.append(('config', kw))
        return -7 if self.fail == 'config' else 0

    def load_onnx(self, **kw):
        self.calls.append(('load_onnx', kw))
        return -7 if self.fail == 'load_onnx' else 0

    def build(self, **kw):
        self.calls.append(('build', kw))
        return -7 if self.fail == 'build' else 0

    def export_rknn(self, path):
        self.calls.append(('export_rknn', path))
        return -7 if self.fail == 'export_rknn' else 0

    def release(self):
        self.calls.append(('release', None))
        if self.release_error:
            raise RuntimeError('generated fake release failure')


def fake_contract():
    return NS(CONFIGURATION=C.CONFIGURATION, require_board=lambda: 'fake-only-rk3588')


def empty_report():
    return {'operations': [], 'calibration_executed': False, 'released': None, 'cleanup_errors': []}


def test_fake_compiler_uses_explicit_board_ptq_and_identity_input_transform(tmp_path):
    fake, report = FakeToolkit(), empty_report()
    Q.run_compile_operations(fake, tmp_path / 'graph.onnx', tmp_path / 'dataset.txt',
                             tmp_path / 'candidate.rknn', report, fake_contract())
    Q.release_toolkit(fake, report)
    assert [call[0] for call in fake.calls] == ['config', 'load_onnx', 'build', 'export_rknn', 'release']
    config = fake.calls[0][1]
    assert config['quantized_dtype'] == 'w8a8' and config['quantized_algorithm'] == 'normal'
    assert config['quantized_method'] == 'channel'
    assert config['mean_values'] is None and config['std_values'] is None
    assert config['quant_img_RGB2BGR'] is False
    build = fake.calls[2][1]
    assert build == {'do_quantization': True, 'dataset': str((tmp_path / 'dataset.txt').resolve()), 'auto_hybrid': False}
    assert report['calibration_executed'] is True and report['released'] is True
    assert all(row['status'] == 'completed' for row in report['operations'])


@pytest.mark.parametrize('fail', ['config', 'load_onnx', 'build', 'export_rknn'])
def test_fake_compiler_failure_stops_calls_and_release_still_recorded(tmp_path, fail):
    fake, report = FakeToolkit(fail=fail), empty_report()
    try:
        with pytest.raises(RuntimeError, match='returned -7'):
            Q.run_compile_operations(fake, tmp_path / 'graph.onnx', tmp_path / 'dataset.txt',
                                     tmp_path / 'candidate.rknn', report, fake_contract())
    finally:
        Q.release_toolkit(fake, report)
    assert fake.calls[-1][0] == 'release' and report['released'] is True
    assert report['operations'][-1]['name'] == fail
    assert report['operations'][-1]['status'] == 'failed'
    assert report['calibration_executed'] is (fail == 'export_rknn')


def test_failed_release_cannot_claim_cleanup():
    report = empty_report()
    Q.release_toolkit(FakeToolkit(release_error=True), report)
    assert report['released'] is False and report['cleanup_errors']


def test_boolean_return_codes_do_not_count_as_integer_success(tmp_path):
    fake, report = FakeToolkit(), empty_report()
    fake.config = lambda **kw: False
    with pytest.raises(RuntimeError, match='returned False'):
        Q.run_compile_operations(fake, tmp_path / 'graph.onnx', tmp_path / 'dataset.txt',
                                 tmp_path / 'candidate.rknn', report, fake_contract())
    fake.release = lambda: False
    Q.release_toolkit(fake, report)
    assert report['released'] is False and report['cleanup_errors']


def test_board_guard_rejects_host_before_any_sdk(monkeypatch):
    monkeypatch.setattr(C.platform, 'machine', lambda: 'x86_64')
    with pytest.raises(ValueError, match='ARM64'):
        C.require_board()


def versions():
    return {**C.EXPECTED_VERSIONS, 'protobuf': '3.20.3'}


@pytest.mark.parametrize('bad', ['production_torch', 'wrong_rknn', 'wrong_ort', 'wrong_prefix',
                                 'production_prefix', 'wrong_python'])
def test_selected_isolated_versions_are_required(tmp_path, bad):
    values, prefix, selected, python = versions(), tmp_path / 'isolated', tmp_path / 'isolated', (3, 11, 1)
    C.check_versions(values, python, prefix, selected)
    if bad == 'production_torch':
        values['torch'] = '2.3.1'
    elif bad == 'wrong_rknn':
        values['rknn-toolkit2'] = '2.3.0'
    elif bad == 'wrong_ort':
        values['onnxruntime'] = '1.20.1'
    elif bad == 'wrong_prefix':
        selected = tmp_path / 'another'
    elif bad == 'production_prefix':
        selected = prefix = C.BASE / 'venv'
    else:
        python = (3, 10, 1)
    with pytest.raises(ValueError):
        C.check_versions(values, python, prefix, selected)


def passed_ort():
    return dict(schema='original_VGGT_pair_CPU_ORT_check_v47', script_sha256=C.BOARD_ORT_SOURCE_SHA,
                pair_export_sha256=C.PAIR_EXPORT_SHA, status='passed', strict_FP32_gate_passed=True,
                successful_session_run_count=4, session_released=True, cleanup_errors=[],
                source_and_inputs_unchanged=True, input_tensor_unchanged=True,
                NPU_used=False, inference_on_board_only=True, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('field,value', [('status', 'failed'), ('strict_FP32_gate_passed', False),
                                       ('successful_session_run_count', 3), ('rtol', .1)])
def test_previous_board_fp32_gate_cannot_be_relaxed(field, value):
    row = passed_ort()
    C.check_board_ort(row)
    row[field] = value
    with pytest.raises(ValueError):
        C.check_board_ort(row)
