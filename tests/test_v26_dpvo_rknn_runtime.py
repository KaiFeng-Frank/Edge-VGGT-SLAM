"""Tiny arrays and a fake C ABI only; no checkpoint, image decode or inference."""
import ctypes as ct
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


HERE = Path(__file__).resolve().parent
ENTRY = HERE / 'run-v26-dpvo-rknn-board.py'
if not ENTRY.exists():
    ENTRY = HERE.parent / 'deploy/rk3588/validation/run-v26-dpvo-rknn-board.py'
runtime = load(ENTRY, '_v26_dpvo_sdk_test')
ABI_PATH = Path(os.environ.get('RKNN_ABI_HELPER', str(ENTRY.with_name('run-v12-rknn-block-board.py')))).expanduser()
if not ABI_PATH.is_file():
    pytest.skip('requires pinned RKNN ABI helper; set RKNN_ABI_HELPER for isolated tests', allow_module_level=True)
assert runtime.sha(ABI_PATH) == runtime.ABI_SHA, 'Pinned RKNN ABI helper SHA mismatch'
abi = load(ABI_PATH, '_v26_test_frozen_abi')


@pytest.fixture(autouse=True)
def tiny(monkeypatch):
    monkeypatch.setattr(runtime, 'INPUT_SHAPE', (1, 3, 2, 4))
    monkeypatch.setattr(runtime, 'OUTPUTS', (('fmap', (1, 2, 1, 2)), ('imap', (1, 4, 1, 2))))


def attr(index, name, shape, fmt=0, dtype=1):
    value = abi.TensorAttr(index=index, n_dims=len(shape), fmt=fmt, type=dtype, qnt_type=0, scale=1)
    value.name = name.encode()
    for i, dim in enumerate(shape):
        value.dims[i] = dim
    value.n_elems = int(np.prod(shape))
    value.size = value.n_elems * (4 if dtype == 0 else 2)
    return value


class FakeSDK:
    def __init__(self, *, failure=None, release_code=0):
        self.failure, self.release_code = failure, release_code
        self.events, self.buffers = [], []
        self.values = {name: np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + .25
                       for name, shape in runtime.OUTPUTS}

    def rknn_init(self, context, *unused):
        ct.cast(context, ct.POINTER(abi.CONTEXT))[0] = abi.CONTEXT(9)
        self.events.append('init')
        return -9 if self.failure == 'init' else 0

    def rknn_set_core_mask(self, *unused):
        self.events.append('set_mask')
        return 0

    def rknn_inputs_set(self, context, count, inputs):
        self.events.append('inputs_set')
        assert count == 1 and inputs[0].pass_through == 0 and inputs[0].type == 0
        shape = runtime.INPUT_SHAPE if inputs[0].fmt == 0 else (1, 2, 4, 3)
        self.packed = np.ctypeslib.as_array(ct.cast(inputs[0].buf, ct.POINTER(ct.c_float)),
                                          shape=(int(np.prod(shape)),)).copy().reshape(shape)
        return -5 if self.failure == 'set' else 0

    def rknn_run(self, context, extension):
        self.events.append('run')
        ct.cast(extension, ct.POINTER(abi.RunExtend)).contents.frame_id = 7
        return -6 if self.failure == 'run' else 0

    def rknn_outputs_get(self, context, count, outputs, received):
        self.events.append('outputs_get')
        for i, (name, shape) in enumerate(runtime.OUTPUTS):
            assert outputs[i].want_float == 1 and outputs[i].is_prealloc == 0 and outputs[i].index == i
            value = self.values[name]
            buffer = ct.create_string_buffer(value.tobytes())
            self.buffers.append(buffer)
            outputs[i].buf = ct.addressof(buffer)
            outputs[i].size = value.nbytes
            if self.failure == 'partial_get':
                return -3
        if self.failure == 'bad_size':
            outputs[1].size -= 4
        if self.failure == 'null_pointer':
            outputs[1].buf = None
        ct.cast(received, ct.POINTER(abi.OutputExtend)).contents.frame_id = 8 if self.failure == 'identity' else 7
        return 0

    def rknn_outputs_release(self, context, count, outputs):
        self.events.append('release')
        for buffer in self.buffers:
            ct.memset(ct.addressof(buffer), 0, ct.sizeof(buffer))
        return self.release_code

    def rknn_destroy(self, context):
        self.events.append('destroy')
        return -10 if self.failure == 'destroy' else 0

    def rknn_query(self, context, command, pointer, size):
        if command == 0:
            value = abi.InputOutputNum(1, 2)
        elif command == 1:
            value = attr(0, 'normalized_bgr', runtime.INPUT_SHAPE)
        elif command == 2:
            requested = ct.cast(pointer, ct.POINTER(abi.TensorAttr)).contents.index
            name, shape = runtime.OUTPUTS[requested]
            value = attr(requested, name, shape)
        elif command == 5:
            value = abi.SDKVersion()
            value.api_version, value.drv_version = b'2.3.2', b'test-driver'
        elif command == 6:
            value = abi.MemSize()
        elif command == 4:
            value = abi.PerfRun(123)
        else:
            raise AssertionError(command)
        ct.memmove(pointer, ct.byref(value), ct.sizeof(value))
        return 0


def run_call(sdk, fmt=0):
    images = np.linspace(-.5, 1.5, np.prod(runtime.INPUT_SHAPE), dtype=np.float32).reshape(runtime.INPUT_SHAPE)
    errors, activity = [], {}
    actual, row = runtime.execute_call(sdk, abi, abi.CONTEXT(9), images, fmt, 60000, errors, activity)
    return actual, row, errors, activity, images


@pytest.mark.parametrize('fmt', [0, 1])
def test_owned_copy_release_and_native_input_packing(fmt):
    sdk = FakeSDK()
    actual, row, errors, activity, images = run_call(sdk, fmt)
    assert sdk.events == ['inputs_set', 'run', 'outputs_get', 'release']
    assert errors == [] and activity == {'sdk_run_api_calls': 1}
    assert row['output_buffers_released'] is True and row['synchronous'] is True
    assert row['sdk_io_seconds'] >= sum(row[k] for k in (
        'input_packing_seconds', 'input_set_seconds', 'run_api_seconds',
        'outputs_get_seconds', 'output_copy_seconds', 'outputs_release_seconds')) - 1e-6
    expected = images if fmt == 0 else images.transpose(0, 2, 3, 1)
    assert sdk.packed.tobytes() == expected.tobytes(order='C')
    for name in sdk.values:
        assert actual[name].tobytes() == sdk.values[name].tobytes()
    assert all(set(buffer.raw) == {0} for buffer in sdk.buffers)
    canonical = runtime.canonical_outputs(actual)
    assert [v.shape for v in canonical.values()] == [(1, 1, 2, 1, 2), (1, 1, 4, 1, 2)]


@pytest.mark.parametrize('failure', ['partial_get', 'bad_size', 'null_pointer', 'identity'])
def test_partial_or_invalid_get_always_releases(failure):
    sdk = FakeSDK(failure=failure)
    with pytest.raises((RuntimeError, ValueError)):
        run_call(sdk)
    assert sdk.events[-1] == 'release'


@pytest.mark.parametrize('failure', ['set', 'run'])
def test_preallocation_failure_does_not_release_nonexistent_buffers(failure):
    sdk = FakeSDK(failure=failure)
    with pytest.raises(RuntimeError, match='returned'):
        run_call(sdk)
    assert 'release' not in sdk.events


def test_release_error_does_not_mask_get_primary_error():
    sdk = FakeSDK(failure='partial_get', release_code=-99)
    errors = []
    with pytest.raises(RuntimeError, match='rknn_outputs_get returned -3'):
        runtime.execute_call(sdk, abi, abi.CONTEXT(9), np.ones(runtime.INPUT_SHAPE, np.float32), 0,
                             1, errors, {})
    assert errors == ['rknn_outputs_release returned -99']


def test_release_failure_alone_is_fatal():
    with pytest.raises(RuntimeError, match='rknn_outputs_release returned -99'):
        run_call(FakeSDK(release_code=-99))


@pytest.mark.parametrize('change', ['format', 'shape', 'index', 'name', 'size', 'type', 'scale', 'quantization'])
def test_feature_attributes_require_exact_nchw_identity(change):
    value = attr(0, 'fmap', runtime.OUTPUTS[0][1])
    if change == 'format':
        value.fmt = 1
    elif change == 'shape':
        value.dims[1], value.dims[3] = value.dims[3], value.dims[1]
        value.dims[2], value.dims[3] = value.dims[3], value.dims[2]
    elif change == 'index':
        value.index = 1
    elif change == 'name':
        value.name = b'imap'
    elif change == 'size':
        value.size += 2
    elif change == 'type':
        value.type = 2
    elif change == 'scale':
        value.scale = float('nan')
    else:
        value.qnt_type, value.scale = 2, .5
    with pytest.raises(ValueError):
        runtime.validate_attribute(abi, value, 0, 'fmap', runtime.OUTPUTS[0][1], is_input=False)


@pytest.mark.parametrize('fmt', [0, 1])
def test_input_attribute_supports_explicit_native_layout(fmt):
    shape = runtime.INPUT_SHAPE if fmt == 0 else (1, 2, 4, 3)
    value = attr(0, 'normalized_bgr', shape, fmt)
    assert runtime.validate_attribute(abi, value, 0, 'normalized_bgr', runtime.INPUT_SHAPE, is_input=True)['format'] == fmt


@pytest.mark.parametrize('invalid', ['zero', 'nan', 'shape'])
def test_invalid_output_cannot_be_completed(invalid):
    values = FakeSDK().values
    values['imap'] = (np.zeros_like(values['imap']) if invalid == 'zero' else
                      np.full_like(values['imap'], np.nan) if invalid == 'nan' else values['imap'][:, :, :, :1])
    with pytest.raises(ValueError):
        runtime.canonical_outputs(values)


def compile_fixture():
    return dict(script_sha256=runtime.COMPILER_SHA, status='compiled_not_runtime_validated', released=True,
        configuration=dict(target_platform='rk3588', float_dtype='float16', optimization_level=3, do_quantization=False),
        operations=[dict(name=name, code=0) for name in ('config', 'load_onnx', 'build', 'export_rknn')],
        versions={'rknn-toolkit2': '2.3.2'}, input={'sha256': 'a' * 64})


@pytest.mark.parametrize('invalid', ['source', 'rules', 'quantization', 'graph', 'failed', 'release'])
def test_compiler_contract_rejects_changes(invalid):
    row = compile_fixture()
    exported = {'onnx': {'sha256': 'a' * 64}}
    runtime.validate_compilation(row, exported)
    if invalid == 'source':
        row['script_sha256'] = '0' * 64
    elif invalid == 'rules':
        row['configuration']['disable_rules'] = ['gather_to_conv']
    elif invalid == 'quantization':
        row['configuration']['do_quantization'] = True
    elif invalid == 'graph':
        row['input']['sha256'] = 'b' * 64
    elif invalid == 'failed':
        row['operations'][2]['code'] = -1
    else:
        row['released'] = False
    with pytest.raises(ValueError):
        runtime.validate_compilation(row, exported)


def test_fp16_numeric_gate_failure_is_recorded_not_raised():
    baseline = np.ones((1, 2), np.float32)
    row = runtime.comparison(baseline * 1.02, baseline)
    assert row['gate_passed'] is False and row['failed_elements'] == 2
    assert row['relative_l2'] == pytest.approx(.02, abs=1e-6)


def test_byte_gate_detects_signed_zero():
    a, b = np.array([0., 1.], np.float32), np.array([-0., 1.], np.float32)
    row = runtime.comparison(a, b)
    assert row['array_equal'] and row['gate_passed'] and not row['bitwise_equal']


@pytest.mark.parametrize('failure', [None, 'init', 'destroy'])
def test_main_context_and_affinity_cleanup(tmp_path, monkeypatch, failure):
    sdk = FakeSDK(failure=failure)
    model = tmp_path / 'model.rknn'
    model.write_bytes(b'fake-model')
    rows = [dict(seed=dict(source={'sha256': 'a' * 64}, input={}, array={}, frame_id=f'{10 + i}.123'),
                 images=np.ones(runtime.INPUT_SHAPE, np.float32),
                 reference={name: value[:, None] for name, value in sdk.values.items()},
                 ort_reference={name: value[:, None] for name, value in sdk.values.items()}) for i in range(3)]
    captured = dict(preprocessing={}, loader={}, input_manifest_sha256='b' * 64)
    monkeypatch.setattr(runtime, 'BASE', tmp_path)
    monkeypatch.setattr(runtime, 'load_abi', lambda *a: abi)
    monkeypatch.setattr(abi, 'require_board', lambda: 'rockchip,rk3588s')
    monkeypatch.setattr(abi, 'bind_api', lambda value: value)
    monkeypatch.setattr(runtime.ct, 'CDLL', lambda *a: sdk)
    monkeypatch.setattr(runtime, 'pin', lambda path, expected, pins: Path(path))
    monkeypatch.setattr(runtime, 'load_inputs', lambda *a: (model, captured, {}, {'configuration': {}}, rows))
    affinity = {4, 5, 6, 7}
    changes = []
    monkeypatch.setattr(runtime.os, 'sched_getaffinity', lambda *a: affinity)
    monkeypatch.setattr(runtime.os, 'sched_setaffinity', lambda pid, value: changes.append(set(value)))
    argv = []
    for name in ('capture-dir', 'export-dir', 'ort-dir', 'compile-dir', 'abi-helper', 'header'):
        argv += ['--' + name, str(tmp_path / name)]
    for name in ('capture', 'export', 'ort', 'compile'):
        argv += ['--' + name + '-sha256', 'a' * 64]
    output = tmp_path / 'experiments' / 'test-output'
    argv += ['--output-dir', str(output)]
    status = runtime.main(argv)
    row = runtime.read_json(output / 'run.json')
    assert sdk.events[-1] == 'destroy' and row['affinity_restored'] is True
    assert changes == [affinity, affinity]
    if failure is None:
        assert status == 0 and row['successful_synchronous_call_count'] == 6
        assert [c['frame_index'] for c in row['calls']] == [0, 0, 0, 0, 1, 2]
        assert row['context_destroyed'] is True and row['cleanup_errors'] == []
    else:
        assert status == 1 and row['status'] == 'failed'
        if failure == 'init':
            assert row['error']['message'] == 'rknn_init returned -9' and row['context_destroyed'] is True
        else:
            assert row['context_destroyed'] is False and 'destroy' in row['cleanup_errors'][0]


def test_normal_profile_schedules_and_transfer_bytes():
    assert runtime.call_schedule(False) == (0, 0, 0, 0, 1, 2)
    assert runtime.call_schedule(True) == (0, 1, 2)
    assert (128 + 384) * 116 * 152 * 4 == 36110336
