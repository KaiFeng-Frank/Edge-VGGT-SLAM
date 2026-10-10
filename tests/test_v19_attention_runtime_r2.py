"""Synthetic observed SDK layouts; no real capture, model, images or NPU."""
import ctypes as ct
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'deploy/rk3588/validation/run-v19-attention-board-r2.py'
spec = importlib.util.spec_from_file_location('attention_runtime_r2_synthetic', PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
protocol_spec = importlib.util.spec_from_file_location('attention_r1_synthetic_protocol',
                                                     Path(__file__).with_name('test_v19_attention_runtime.py'))
protocol = importlib.util.module_from_spec(protocol_spec)
protocol_spec.loader.exec_module(protocol)


@pytest.fixture
def base():
    return m.load_base()


@pytest.fixture
def runtime(base):
    return base.load_helper('run-v12-rknn-block-board.py', base.RUNTIME_SHA)


class NativeNHWCSDK(protocol.FakeSDK):
    """Declare NHWC internally but consume the explicitly NCHW user buffer."""
    def rknn_query(self, context, command, pointer, size):
        result = super().rknn_query(context, command, pointer, size)
        if command in (1, 2):
            attr = ct.cast(pointer, ct.POINTER(self.runtime.TensorAttr)).contents
            logical = self.states[self.context_id(context)]['shape']
            attr.fmt = 1 if command == 1 else 0
            declared = m.descriptor_shape(logical, attr.fmt, command == 1)
            for i, value in enumerate(declared):
                attr.dims[i] = value
        return result

    def rknn_inputs_set(self, context, count, request):
        cid = self.context_id(context)
        state = self.states[cid]
        assert count == 3
        values = {}
        for i, name in enumerate(m.INPUT_NAMES):
            row = request[i]
            assert row.index == i and row.type == 0 and row.pass_through == 0
            assert row.fmt == 0, 'The actual C-order NCHW buffer must never be labelled NHWC'
            assert row.size == int(np.prod(state['shape'])) * 4
            view = np.ctypeslib.as_array(ct.cast(row.buf, ct.POINTER(ct.c_float)),
                                       shape=(int(np.prod(state['shape'])),))
            values[name] = view.copy().reshape(state['shape'])
        self.requests.append((cid, values))
        state['input'] = values
        return 0


def sessions_for(base, runtime, shapes):
    sdk = NativeNHWCSDK(runtime, shapes)
    sessions, offset = [], 0
    for i, shape in enumerate(shapes):
        spec = dict(shape=shape, name=f'synthetic-{i}', model=Path(f'model-{i}.rknn'),
                    core_mask=7 if len(shapes) == 1 else 1 << i,
                    begin=offset, end=offset+shape[1])
        offset += shape[1]
        base.create_session(sdk, runtime, spec, 0, sessions, {})
    return sdk, sessions


@pytest.mark.parametrize('heads', [16, 6, 5])
def test_actual_observed_input_nhwc_shape_is_strictly_mapped(base, runtime, heads):
    logical = (1, heads, 874, 64)
    actual = (1, 874, 64, heads)
    for i, name in enumerate(m.INPUT_NAMES):
        attr = protocol.tensor_attr(runtime, actual, name, i)
        attr.fmt = 1
        row = base.validate_attr(runtime, attr, logical, name, i)
        assert row['shape'] == list(actual) and row['format'] == 1
        assert row['n_elems'] == np.prod(logical) and row['size'] == np.prod(logical)*2
    output = protocol.tensor_attr(runtime, logical, 'output')
    output.fmt = 0
    assert base.validate_attr(runtime, output, logical, 'output', 0)['shape'] == list(logical)


@pytest.mark.parametrize('axis', [0, 1, 2, 3])
def test_wrong_dimension_in_mapped_input_is_rejected(base, runtime, axis):
    logical = (1, 16, 874, 64)
    attr = protocol.tensor_attr(runtime, (1, 874, 64, 16), 'query')
    attr.fmt = 1
    attr.dims[axis] += 1
    with pytest.raises(ValueError, match='identity/shape'):
        base.validate_attr(runtime, attr, logical, 'query', 0)


def test_nhwc_label_with_unmapped_nchw_dimensions_rejected(base, runtime):
    logical = (1, 16, 874, 64)
    attr = protocol.tensor_attr(runtime, logical, 'query')
    attr.fmt = 1
    with pytest.raises(ValueError):
        base.validate_attr(runtime, attr, logical, 'query', 0)


@pytest.mark.parametrize('fmt', [1, 2, 3])
def test_unimplemented_output_layout_rejected(base, runtime, fmt):
    logical = (1, 5, 7, 64)
    attr = protocol.tensor_attr(runtime, logical, 'output')
    attr.fmt = fmt
    with pytest.raises(ValueError, match='Unimplemented'):
        base.validate_attr(runtime, attr, logical, 'output', 0)


@pytest.mark.parametrize('field,value', [('scale', .5), ('zp', 1), ('type', 2),
                                       ('qnt_type', 1), ('size', 1), ('name', b'wrong'), ('index', 1)])
def test_layout_fix_never_relaxes_dtype_quantization_bytes_or_identity(base, runtime, field, value):
    attr = protocol.tensor_attr(runtime, (1, 7, 64, 5), 'query')
    attr.fmt = 1
    setattr(attr, field, value)
    with pytest.raises(ValueError):
        base.validate_attr(runtime, attr, (1, 5, 7, 64), 'query', 0)


def test_nchw_declared_input_keeps_original_shape(base, runtime):
    shape = (1, 5, 7, 64)
    attr = protocol.tensor_attr(runtime, shape, 'query')
    attr.fmt = 0
    assert base.validate_attr(runtime, attr, shape, 'query', 0)['format'] == 0


def test_transport_requests_nchw_without_reordering_bytes_or_mutating_raw_attrs(base, runtime):
    shape = (1, 2, 3, 64)
    sdk, sessions = sessions_for(base, runtime, [shape])
    inputs = {name: (np.arange(np.prod(shape), dtype=np.float32).reshape(shape)+i*1000)
              for i, name in enumerate(m.INPUT_NAMES)}
    row = {}
    actual = base.execute_call(sdk, runtime, sessions[0], inputs, 30000, row)
    wanted = inputs['query'] + np.float32(2)*inputs['key'] + np.float32(3)*inputs['value']
    np.testing.assert_array_equal(actual, wanted)
    for name in m.INPUT_NAMES:
        np.testing.assert_array_equal(sdk.requests[0][1][name], inputs[name])
    assert row['input_native_formats'] == [1, 1, 1] and row['input_api_format'] == 0
    assert row['input_buffer_order'] == 'C-order NCHW' and row['input_layout_conversion_requested']
    assert [int(attr.fmt) for attr in sessions[0]['attrs']] == [1, 1, 1]
    report = sessions[0]['report']
    assert [r['format'] for r in report['input_attrs']] == [1, 1, 1]
    assert report['layout_policy']['input_api_format'] == 0
    assert report['layout_policy']['R1_implementation_sha256'] == m.R1_SHA
    assert report['layout_policy']['entrypoint_sha256'] == m.sha(PATH)
    errors = []
    base.destroy_sessions(sdk, runtime, sessions, errors)
    assert not errors and sdk.destroyed == [1] and sdk.releases == [1]


def test_three_contexts_keep_original_order_despite_nhwc_sdk_descriptors(base, runtime):
    shape = (1, 10, 7, 64)
    rng = np.random.default_rng(19)
    flat = {name: rng.normal(size=shape).astype(np.float32) for name in m.INPUT_NAMES}
    expected = flat['query'] + np.float32(2)*flat['key'] + np.float32(3)*flat['value']
    sdk, sessions = sessions_for(base, runtime, [(1, 4, 7, 64), (1, 3, 7, 64), (1, 3, 7, 64)])
    row = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        actual = base.execute_round(sdk, runtime, sessions, flat, shape, 30000, row, pool)
    np.testing.assert_array_equal(actual, expected)
    assert sdk.max_active == 3 and row['run_api_overlap']['max_simultaneous'] == 3
    assert all(call['input_api_format'] == 0 and call['input_native_formats'] == [1, 1, 1]
               for call in row['calls'])
    errors = []
    base.destroy_sessions(sdk, runtime, sessions, errors)
    assert not errors and sdk.destroyed == [3, 2, 1] and sorted(sdk.releases) == [1, 2, 3]


def test_entrypoint_identity_and_frozen_base_guard(base, monkeypatch, tmp_path):
    assert Path(base.__file__) == PATH
    assert m.sha(PATH.with_name(m.R1_NAME)) == m.R1_SHA
    (tmp_path/m.R1_NAME).write_bytes(b'not the frozen implementation')
    monkeypatch.setattr(m, 'HERE', tmp_path)
    with pytest.raises(ValueError, match='implementation SHA256'):
        m.load_base()


def test_host_guard_remains_before_output_creation(tmp_path):
    args = ['--mode', 'parallel', '--capture-dir', str(tmp_path/'capture'),
            '--export-dir', str(tmp_path/'export'), '--output', str(tmp_path/'out'),
            '--capture-sha256', 'a'*64, '--export-sha256', 'b'*64,
            '--models', 'synthetic.rknn', '--model-sha256', 'c'*64,
            '--compile-receipts', 'synthetic.json', '--compile-sha256', 'd'*64]
    with pytest.raises(RuntimeError, match='Board-only'):
        m.main(args)
    assert not (tmp_path/'out').exists()
