"""Generated tiny arrays and a fake C API only; never model/ORT/SDK inference."""
from __future__ import annotations

import ctypes as ct
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent


def module(path):
    spec = importlib.util.spec_from_file_location(path.stem.replace('-', '_'), path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


C = module(HERE / 'v48_ab_contracts.py')
AB = module(HERE / 'run-v48-frame-global-ab-board.py')
R = module(HERE.parent / 'run-v12-rknn-block-board.py')


class FakeAPI:
    """Only own generated numpy storage; mutating release proves copy lifetime."""
    def __init__(self, value, *, wrong_size=False, mismatch=False, get_code=0, release_code=0):
        self.value = value.copy()
        self.wrong_size = wrong_size
        self.mismatch = mismatch
        self.get_code = get_code
        self.release_code = release_code
        self.events = []

    def rknn_inputs_set(self, context, count, request):
        self.events.append('inputs_set')
        assert count == 1 and request[0].type == 0 and request[0].pass_through == 0
        assert request[0].fmt == 3
        self.input_pointer = int(request[0].buf)
        return 0

    def rknn_run(self, context, run):
        self.events.append('run')
        actual = ct.cast(run, ct.POINTER(R.RunExtend)).contents
        assert actual.non_block == 0 and actual.fence_fd == -1
        actual.frame_id = 71
        return 0

    def rknn_outputs_get(self, context, count, outputs, received):
        self.events.append('get')
        assert outputs[0].want_float == 1 and outputs[0].is_prealloc == 0
        outputs[0].buf = self.value.ctypes.data
        outputs[0].size = self.value.nbytes - (4 if self.wrong_size else 0)
        ct.cast(received, ct.POINTER(R.OutputExtend)).contents.frame_id = 72 if self.mismatch else 71
        return self.get_code

    def rknn_outputs_release(self, context, count, outputs):
        self.events.append('release')
        self.value.fill(-900)
        return self.release_code


def execute(api):
    inputs = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    errors, ledger = [], []
    output, row = AB.execute_sdk_call(R, api, R.CONTEXT(1), inputs, 3, (1, 4, 3),
                                      30000, errors, ledger, 'fused')
    return output, row, errors, ledger


def test_actual_fp32_copy_survives_release_and_distinct_rank_three_shapes():
    expected = np.arange(12, dtype=np.float32).reshape(1, 4, 3)
    api = FakeAPI(expected)
    actual, row, errors, ledger = execute(api)
    assert np.array_equal(actual, expected)
    assert api.events == ['inputs_set', 'run', 'get', 'release']
    assert not errors and ledger == [row]
    assert row['status'] == 'completed' and row['output_buffer_released']
    assert row['input_shape'] == [2, 2, 3] and row['output_shape'] == [1, 4, 3]
    assert row['io_end_monotonic_ns'] >= row['io_begin_monotonic_ns']


@pytest.mark.parametrize('option,exception', [('wrong_size', ValueError), ('mismatch', ValueError),
                                            ('get_code', RuntimeError), ('release_code', RuntimeError)])
def test_partial_output_or_validation_failure_always_releases(option, exception):
    api = FakeAPI(np.ones((1, 4, 3), np.float32), **{option: 4 if 'code' in option else True})
    errors, ledger = [], []
    with pytest.raises(exception):
        AB.execute_sdk_call(R, api, R.CONTEXT(1), np.ones((2, 2, 3), np.float32),
                            3, (1, 4, 3), 30000, errors, ledger, 'fused')
    assert api.events.count('release') == 1
    assert ledger[0]['status'] == 'failed'
    assert bool(errors) is (option == 'release_code')


def test_separate_route_global_consumes_this_npu_frame_output_without_cpu_reset():
    small = NS(FRAME_SHAPE=(2, 2, 3), GLOBAL_SHAPE=(1, 4, 3), raw_sha=C.raw_sha)
    original = np.arange(12, dtype=np.float32).reshape(small.FRAME_SHAPE)
    context = {'frame0': 10, 'global0': 20, 'fused': 30}
    called = []
    held_frame = None

    def invoke(role, ctx, actual_input, output_shape):
        nonlocal held_frame
        called.append(role)
        if role == 'frame0':
            assert actual_input is original and ctx == 10
            held_frame = original + np.float32(100)
            actual = held_frame
        else:
            assert role == 'global0' and ctx == 20
            assert np.shares_memory(actual_input, held_frame)
            assert np.array_equal(actual_input, (original + np.float32(100)).reshape(small.GLOBAL_SHAPE))
            actual = actual_input + np.float32(1000)
        return actual, {'sdk_call_index': len(called) - 1, 'io_total_seconds': .001,
                        'input_pointer': int(actual_input.ctypes.data)}

    actual, frame, row = AB.run_route('separate', original, context, invoke, small)
    assert called == ['frame0', 'global0']
    assert np.array_equal(actual, (original + np.float32(1100)).reshape(small.GLOBAL_SHAPE))
    assert row['bridge']['input_raw_sha256_before'] == C.raw_sha(frame)
    assert row['bridge']['input_unchanged']
    assert row['sdk_call_indices'] == [0, 1] and row['sdk_io_sum_seconds'] == .002


def test_fused_route_calls_once_and_no_intermediate_reference_can_enter():
    small = NS(FRAME_SHAPE=(2, 2, 3), GLOBAL_SHAPE=(1, 4, 3), raw_sha=C.raw_sha)
    inputs = np.ones(small.FRAME_SHAPE, np.float32)
    called = []

    def invoke(role, context, value, shape):
        called.append(role)
        assert value is inputs and role == 'fused' and context == 30
        return value.reshape(shape).copy(), {'sdk_call_index': 0, 'io_total_seconds': .004}

    actual, frame, row = AB.run_route('fused', inputs, {'fused': 30}, invoke, small)
    assert called == ['fused'] and frame is None and row['bridge'] is None
    assert row['sdk_io_sum_seconds'] == .004


def test_separate_detects_sdk_input_mutation():
    small = NS(FRAME_SHAPE=(2, 2, 3), GLOBAL_SHAPE=(1, 4, 3), raw_sha=C.raw_sha)
    inputs = np.ones(small.FRAME_SHAPE, np.float32)

    def invoke(role, context, value, shape):
        if role == 'global0':
            value.flat[0] = np.float32(99)
        return value.copy().reshape(shape), {'sdk_call_index': 0, 'io_total_seconds': .001,
                                           'input_pointer': int(value.ctypes.data)}

    with pytest.raises(ValueError, match='changed'):
        AB.run_route('separate', inputs, {'frame0': 10, 'global0': 20}, invoke, small)


@pytest.mark.parametrize('wrong', ['nonfinite', 'wrong_shape', 'fp16', 'noncontiguous'])
def test_bridge_rejects_unknown_data_layout(wrong):
    value = np.ones((2, 2, 3), np.float32)
    if wrong == 'nonfinite':
        value.flat[0] = np.inf
    elif wrong == 'wrong_shape':
        value = value.reshape(1, 4, 3)
    elif wrong == 'fp16':
        value = value.astype(np.float16)
    else:
        value = value[:, :, ::-1]
    with pytest.raises(ValueError):
        AB.bridge_frame_output(value, (2, 2, 3), (1, 4, 3))


def descriptor(role, direction):
    shape = C.FRAME_SHAPE if role == 'frame0' or (role == 'fused' and direction == 'input') else C.GLOBAL_SHAPE
    return dict(index=0, name=direction, shape=list(shape), n_elems=int(np.prod(shape)),
                type=1, quantization_type=2, scale=1., zero_point=0, format=3,
                pass_through=0, size=2 * int(np.prod(shape)))


@pytest.mark.parametrize('role', ['frame0', 'global0', 'fused'])
def test_sdk_each_role_requires_exact_named_shapes(role):
    for direction in ('input', 'output'):
        assert C.validate_sdk_attr(descriptor(role, direction), role, direction)
    wrong = descriptor(role, 'output')
    wrong['shape'] = list(C.GLOBAL_SHAPE if role == 'frame0' else C.FRAME_SHAPE)
    with pytest.raises(ValueError):
        C.validate_sdk_attr(wrong, role, 'output')


@pytest.mark.parametrize('field,value', [('format', 1), ('type', 2), ('scale', .9),
                                       ('zero_point', 1), ('size', 4), ('name', 'x'),
                                       ('index', 1), ('pass_through', 1)])
def test_unknown_typed_sdk_descriptor_rejected(field, value):
    row = descriptor('fused', 'input')
    row[field] = value
    with pytest.raises(ValueError):
        C.validate_sdk_attr(row, 'fused', 'input')


def test_fixed_schedule_balances_first_second_positions_and_bounded_calls():
    assert len(AB.SCHEDULE) == 4
    assert AB.SCHEDULE == (('separate', 'fused'), ('fused', 'separate'),
                           ('separate', 'fused'), ('fused', 'separate'))
    for route in ('separate', 'fused'):
        assert sum(order[0] == route for order in AB.SCHEDULE) == 2
        assert sum(route in order for order in AB.SCHEDULE) == 4
    assert sum(2 if route == 'separate' else 1 for order in AB.SCHEDULE for route in order) == 12


def test_output_file_creation_exclusive_and_saved_full_array(tmp_path):
    data = np.arange(12, dtype=np.float32).reshape(1, 4, 3)
    saved = AB.save_array(tmp_path, 'actual.npy', data, C)
    assert saved['raw_sha256'] == C.raw_sha(data)
    assert np.array_equal(np.load(tmp_path / saved['file'], allow_pickle=False), data)
    with pytest.raises(FileExistsError):
        AB.save_array(tmp_path, 'actual.npy', data, C)


def test_source_pin_rejects_mutation(tmp_path):
    source = tmp_path / 'source.txt'
    source.write_text('initial')
    pins = {}
    C.pin(source, C.sha(source), pins)
    assert C.recheck(pins)[1]
    source.write_text('changed')
    assert not C.recheck(pins)[1]


def fused_ort():
    return dict(schema='original_VGGT_pair_CPU_ORT_check_v47', script_sha256=C.PAIR_ORT_SOURCE_SHA,
                pair_export_sha256=C.PAIR_EXPORT_SHA, status='passed', strict_FP32_gate_passed=True,
                successful_session_run_count=4, session_released=True, cleanup_errors=[],
                source_and_inputs_unchanged=True, input_tensor_unchanged=True, NPU_used=False,
                inference_on_board_only=True, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('field,value', [('status', 'failed'), ('strict_FP32_gate_passed', False),
                                       ('successful_session_run_count', 3), ('input_tensor_unchanged', False),
                                       ('NPU_used', True), ('rtol', .1)])
def test_fused_actual_cpu_ort_cannot_be_bypassed_for_controlled_ab(field, value):
    row = fused_ort()
    compiled = {'ORT_receipt_sha256': 'frozen', 'board_ORT_gate_passed': True}
    C.check_fused_ort(row, 'frozen', compiled)
    row[field] = value
    with pytest.raises(ValueError):
        C.check_fused_ort(row, 'frozen', compiled)


def test_summary_retains_npu_original_fp32_rejection():
    rows = [{'route': route, 'route_total_seconds': .004, 'sdk_io_sum_seconds': .002,
             'versus_original_global_reference': {'strict_FP32_gate_passed': False}}
            for _ in range(4) for route in ('separate', 'fused')]
    saved = AB.summary(rows)
    for route in ('separate', 'fused'):
        assert saved[route]['completed_call_count'] == 4
        assert saved[route]['strict_original_FP32_gate_passed'] is False
        assert saved[route]['route_total_seconds']['subsequent_mean'] == .004
