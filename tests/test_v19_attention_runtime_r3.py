"""Synthetic native NHWC transport only; no images, real model, ORT or NPU."""
import ctypes as ct
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'deploy/rk3588/validation/run-v19-attention-board-r3.py'
spec = importlib.util.spec_from_file_location('attention_runtime_r3_synthetic', PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
protocol_spec = importlib.util.spec_from_file_location('attention_r2_synthetic_protocol',
                                                     Path(__file__).with_name('test_v19_attention_runtime_r2.py'))
r2 = importlib.util.module_from_spec(protocol_spec)
protocol_spec.loader.exec_module(r2)


@pytest.fixture
def base():
    return m.load_base()


@pytest.fixture
def runtime(base):
    return base.load_helper('run-v12-rknn-block-board.py', base.RUNTIME_SHA)


class PackedNHWCSDK(r2.NativeNHWCSDK):
    """Read the actual pointer as NHWC, then reconstruct logical NCHW."""
    def rknn_inputs_set(self, context, count, request):
        cid = self.context_id(context)
        state = self.states[cid]
        logical = state['shape']
        native = (logical[0], logical[2], logical[3], logical[1])
        assert count == 3
        values = {}
        for i, name in enumerate(m.INPUT_NAMES):
            row = request[i]
            assert row.index == i and row.type == 0 and row.pass_through == 0
            assert row.fmt == 1, 'Supplied bytes must actually be NHWC, not just relabelled NCHW'
            assert row.size == int(np.prod(logical))*4
            view = np.ctypeslib.as_array(ct.cast(row.buf, ct.POINTER(ct.c_float)),
                                       shape=(int(np.prod(logical)),))
            packed = view.copy().reshape(native)
            values[name] = np.array(packed.transpose(0, 3, 1, 2), order='C', copy=True)
        with self.lock:
            self.requests.append((cid, values))
        state['input'] = values
        return -2 if self.failure == 'input' and cid == self.bad_context else 0


def sessions_for(base, runtime, shapes, failure=None):
    sdk = PackedNHWCSDK(runtime, shapes, failure)
    sessions, offset = [], 0
    for i, shape in enumerate(shapes):
        item = dict(shape=shape, name=f'synthetic-{i}', model=Path(f'model-{i}.rknn'),
                    core_mask=7 if len(shapes) == 1 else 1 << i,
                    begin=offset, end=offset+shape[1])
        offset += shape[1]
        base.create_session(sdk, runtime, item, 0, sessions, {})
    return sdk, sessions


def operands(shape):
    return {name: np.arange(np.prod(shape), dtype=np.float32).reshape(shape)+i*1000
            for i, name in enumerate(m.INPUT_NAMES)}


@pytest.mark.parametrize('heads', [16, 6, 5])
def test_actual_nhwc_packing_reorders_bytes_and_owns_each_input(heads):
    shape = (1, heads, 7, 64)
    inputs = operands(shape)
    packed = m.pack_nhwc(inputs, shape)
    for name in m.INPUT_NAMES:
        original = inputs[name].copy()
        value = packed[name]
        assert value.shape == (1, 7, 64, heads) and value.flags.c_contiguous
        assert not np.shares_memory(value, inputs[name])
        np.testing.assert_array_equal(value.ravel(), inputs[name].transpose(0, 2, 3, 1).ravel())
        assert not np.array_equal(value.ravel(), inputs[name].ravel())
        value.fill(-42)
        np.testing.assert_array_equal(inputs[name], original)


@pytest.mark.parametrize('parallel', [False, True])
def test_round_real_positional_signatures_nhwc_requests_nchw_output_and_cleanup(base, runtime, parallel):
    original_shape, shape = (2, 5, 7, 64), (1, 10, 7, 64)
    inputs = operands(shape)
    snapshots = {key: value.copy() for key, value in inputs.items()}
    shapes = [(1, 4, 7, 64), (1, 3, 7, 64), (1, 3, 7, 64)] if parallel else [shape]
    sdk, sessions = sessions_for(base, runtime, shapes)
    group = {}
    if parallel:
        with ThreadPoolExecutor(max_workers=3) as pool:
            # Eight positional arguments: group is index 6, executor is index 7.
            actual = base.execute_round(sdk, runtime, sessions, inputs, original_shape, 30000, group, pool)
    else:
        # Seven positional arguments: no executor exists.
        actual = base.execute_round(sdk, runtime, sessions, inputs, original_shape, 30000, group)
    expected = (inputs['query'] + np.float32(2)*inputs['key'] + np.float32(3)*inputs['value']).reshape(original_shape)
    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.c_contiguous
    assert group['additional_NHWC_packing_included_in_end_to_end']
    assert group['nhwc_worker_packing_seconds_sum'] == sum(row['nhwc_packing_seconds'] for row in group['calls'])
    assert group['end_to_end_seconds'] >= group['packing_seconds']+group['concat_seconds']
    assert sdk.max_active == (3 if parallel else 1)
    if parallel:
        assert group['run_api_overlap']['max_simultaneous'] == 3
        assert group['run_api_overlap']['all_three_overlap_seconds'] > 0
    for name in m.INPUT_NAMES:
        np.testing.assert_array_equal(inputs[name], snapshots[name])
    for row, session in zip(group['calls'], sessions):
        assert row['input_api_format'] == 1 and row['input_native_formats'] == [1, 1, 1]
        assert row['input_buffer_order'] == 'C-order NHWC' and not row['input_layout_conversion_requested']
        assert row['output_copied_before_release'] and row['output_buffer_released']
        assert group['begin_monotonic'] <= row['nhwc_pack_begin_monotonic'] <= row['nhwc_pack_end_monotonic']
        assert row['nhwc_pack_end_monotonic'] <= row['io_begin_monotonic'] <= row['io_end_monotonic'] <= group['end_monotonic']
        policy = session['report']['layout_policy']
        assert policy['entrypoint_sha256'] == m.sha(PATH)
        assert policy['R2_adapter_sha256'] == m.R2_SHA
        assert policy['R1_implementation_sha256'] == r2.m.R1_SHA
        assert policy['input_api_format'] == 1
        assert session['report']['output_attr']['format'] == 0
    errors = []
    base.destroy_sessions(sdk, runtime, sessions, errors)
    assert not errors and sdk.active == 0 and len(sdk.destroyed) == len(shapes)
    assert len(sdk.releases) == len(shapes)


@pytest.mark.parametrize('failure,release_count', [('input', 0), ('run', 0), ('frame', 1),
                                                 ('size', 1), ('get_partial', 1), ('release', 1)])
def test_r3_owned_output_error_paths_release_and_destroy(base, runtime, failure, release_count):
    shape = (1, 2, 4, 64)
    sdk, sessions = sessions_for(base, runtime, [shape], failure)
    row = {}
    with pytest.raises((ValueError, RuntimeError)):
        base.execute_call(sdk, runtime, sessions[0], operands(shape), 30000, row)
    assert len(sdk.releases) == release_count
    errors = []
    base.destroy_sessions(sdk, runtime, sessions, errors)
    assert not errors and sdk.destroyed == [1]


@pytest.mark.parametrize('name,fmt', [('query', 0), ('output', 1)])
def test_unimplemented_input_or_output_layout_rejected(base, runtime, name, fmt):
    shape = (1, 2, 7, 64)
    attr = r2.protocol.tensor_attr(runtime, shape, name)
    attr.fmt = fmt
    with pytest.raises(ValueError):
        base.validate_attr(runtime, attr, shape, name, 0)


def test_r3_identity_and_frozen_r2_sha_guard(base, monkeypatch, tmp_path):
    assert Path(base.__file__) == PATH
    assert m.sha(PATH.with_name(m.R2_NAME)) == m.R2_SHA
    (tmp_path/m.R2_NAME).write_bytes(b'not the frozen adapter')
    monkeypatch.setattr(m, 'HERE', tmp_path)
    with pytest.raises(ValueError, match='Frozen R2 adapter SHA256'):
        m.load_base()


def test_zero_output_is_explicit_failure_of_unchanged_strict_fp32_gate(base, runtime):
    row = base.comparison(runtime, np.zeros((1, 2, 3, 64), np.float32), np.ones((1, 2, 3, 64), np.float32))
    assert row['output_all_zero'] and not row['reference_all_zero']
    assert not row['elementwise_gate_passed'] and row['relative_l2'] == 1.
