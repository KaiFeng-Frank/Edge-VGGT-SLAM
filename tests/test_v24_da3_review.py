"""Synthetic-only checks for the new saved-array and multi-output probes."""
import ctypes as ct
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent
VALIDATION = HERE.parent / 'deploy/rk3588/validation'
LOCAL = HERE if (HERE / 'run-v24-da3-rknn-board.py').is_file() else VALIDATION
# The source checkout has deploy/.../validation siblings. Before staging, the
# isolated draft lives in tmpfs beside its tests and references the frozen tree.
DEPENDENCIES = VALIDATION if (VALIDATION / 'run-v12-rknn-block-board.py').is_file() else Path(
    '/dev/shm/vggt-v15-public-snapshot/deploy/rk3588/validation')
RUN = load(LOCAL / 'run-v24-da3-rknn-board.py', '_synthetic_v24_runtime')
CMP = load(LOCAL / 'compare-v24-da3-pairs.py', '_synthetic_v24_comparison')
ABI = load(DEPENDENCIES / 'run-v12-rknn-block-board.py', '_synthetic_v24_abi')


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(RUN, 'SHAPE', (2, 3, 4, 6))
    monkeypatch.setattr(RUN, 'OUTPUTS', [dict(name='depth', reference_key='depth', shape=[2, 4, 6]),
        dict(name='depth_conf', reference_key='conf', shape=[2, 4, 6]),
        dict(name='pose_enc', reference_key='pose', shape=[2, 9])])
    return np.arange(144, dtype=np.float32).reshape(2, 3, 4, 6)


class FakeAPI:
    def __init__(self, *, get_error=0, release_error=0, bad_size=False):
        self.get_error, self.release_error, self.bad_size = get_error, release_error, bad_size
        self.buffers = [np.full(item['shape'], i + 1, np.float32) for i, item in enumerate(RUN.OUTPUTS)]
        self.events = []
        self.released = False

    def rknn_inputs_set(self, context, count, inputs):
        assert count == 1
        self.input_format = inputs[0].fmt
        self.input = np.ctypeslib.as_array(ct.cast(inputs[0].buf, ct.POINTER(ct.c_float)),
                                          shape=(inputs[0].size // 4,)).copy()
        self.events.append('set')
        return 0

    def rknn_run(self, context, extension):
        self.events.append('run')
        return 0

    def rknn_outputs_get(self, context, count, outputs, extension):
        assert count == 3
        self.events.append('get')
        for i, a in enumerate(self.buffers):
            outputs[i].buf, outputs[i].size = a.ctypes.data, a.nbytes
        if self.bad_size:
            outputs[1].size -= 4
        return self.get_error

    def rknn_outputs_release(self, context, count, outputs):
        assert count == 3
        self.events.append('release')
        self.released = True
        for a in self.buffers:
            a.fill(-99)
        return self.release_error


@pytest.mark.parametrize('fmt', [0, 1])
def test_native_packing_copies_correct_channels_and_views(small, fmt):
    packed = RUN.pack_input(small, fmt)
    expected = small if fmt == 0 else small.transpose(0, 2, 3, 1)
    assert packed.flags.c_contiguous and not np.shares_memory(packed, small)
    assert packed.tobytes() == expected.tobytes(order='C')


def test_three_outputs_owned_before_sdk_release(small):
    api, cleanup, activity = FakeAPI(), [], {}
    owned, receipt = RUN.execute_multi(api, ABI, ABI.CONTEXT(7), small, 1, 60000, cleanup, activity)
    assert api.events == ['set', 'run', 'get', 'release']
    assert api.input_format == 1
    assert api.input.tobytes() == small.transpose(0, 2, 3, 1).tobytes(order='C')
    assert all(np.all(value == i + 1) for i, value in enumerate(owned))
    assert all(not np.shares_memory(a, b) for a, b in zip(owned, api.buffers))
    assert receipt['output_buffers_released'] and receipt['sdk_io_seconds'] > 0
    assert activity['sdk_run_api_calls'] == 1 and not cleanup


def test_partial_get_allocation_released_without_masking_primary_error(small):
    api, cleanup = FakeAPI(get_error=-7, release_error=-8), []
    with pytest.raises(RuntimeError, match='outputs_get.*-7'):
        RUN.execute_multi(api, ABI, ABI.CONTEXT(7), small, 0, 60000, cleanup, {})
    assert api.released and cleanup == ['rknn_outputs_release returned -8']


def test_bad_output_byte_count_releases_group(small):
    api = FakeAPI(bad_size=True)
    with pytest.raises(ValueError, match='buffer size'):
        RUN.execute_multi(api, ABI, ABI.CONTEXT(7), small, 0, 60000, [], {})
    assert api.released


def test_release_failure_is_not_reported_success(small):
    api = FakeAPI(release_error=-8)
    with pytest.raises(RuntimeError, match='outputs_release.*-8'):
        RUN.execute_multi(api, ABI, ABI.CONTEXT(7), small, 0, 60000, [], {})


@pytest.mark.parametrize('problem', ['zero', 'nonfinite', 'quaternion'])
def test_bad_outputs_rejected_even_after_api_success(small, problem):
    actual = [a.copy() for a in FakeAPI().buffers]
    if problem == 'zero':
        actual[0].fill(0)
    elif problem == 'nonfinite':
        actual[1][0, 0, 0] = np.nan
    else:
        actual[2][:, 3:7] = 0
    with pytest.raises(ValueError):
        RUN.validate_outputs(actual)


def test_conflicting_tensor_sha_rejected():
    a = np.ones((2, 3), np.float32)
    info = RUN.metadata(a)
    info['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='raw SHA'):
        RUN.validate_array(a, info, a.shape)


def test_signed_zero_is_not_bitwise_equal():
    actual, expected = np.array([0.], np.float32), np.array([-0.], np.float32)
    stats = RUN.comparison(actual, expected)
    assert stats['array_equal'] and not stats['C_order_bytes_equal']


@pytest.mark.parametrize('mutation', ['wrong_name', 'shape', 'layout', 'quantization'])
def test_tensor_attribute_guards(small, mutation):
    attr = ABI.TensorAttr(index=0, name=b'images', n_dims=4, n_elems=144, size=576,
                          fmt=1, type=0, qnt_type=0, scale=1.)
    attr.dims[:4] = (2, 4, 6, 3)
    RUN.validate_attribute(ABI, attr, 'images', RUN.SHAPE, is_input=True)
    if mutation == 'wrong_name':
        attr.name = b'depth'
    elif mutation == 'shape':
        attr.dims[:4] = RUN.SHAPE
    elif mutation == 'layout':
        attr.fmt = 2
    else:
        attr.qnt_type = 2
    with pytest.raises(ValueError):
        RUN.validate_attribute(ABI, attr, 'images', RUN.SHAPE, is_input=True)


def test_saved_depth_uses_one_scale_across_both_views(tmp_path):
    helper = CMP.numerical_helper(DEPENDENCIES)
    def row(values):
        a = np.asarray(values, np.float32).reshape(2, 2, 2)
        return dict(arrays={'depth': a[None, ..., None]}, A=np.eye(3), record={'frame_ids': [1., 2.]})
    actual = row(np.ones(8))
    expected = row([2] * 4 + [4] * 4)
    stats = CMP.depth_comparison(actual, expected, 'reference', 0, tmp_path, helper)
    assert stats['positive_scale_DA3_to_reference'] == 3.
    assert stats['scale_aligned']['rmse'] == 1.
    assert stats['holdout_used'] is False and stats['scale_fit_and_score_use_same_pixels'] is True
    assert sum(r['matched_pixels'] for r in stats['frames']) == 8
    with np.load(tmp_path / stats['mask']['file'], allow_pickle=False) as saved:
        assert saved['positive_intersection'].all()


def test_output_host_guard_prevents_any_execution_or_result_creation(tmp_path):
    with pytest.raises(RuntimeError, match='Board-only'):
        ABI.require_board()
    assert not list(tmp_path.iterdir())
