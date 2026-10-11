"""Generated NumPy buffers and fake SDK only; no model, library or board run."""
import copy
import ctypes as ct
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import uuid

import numpy as np
import pytest

TEST_DIR = Path(__file__).resolve().parent
HERE = Path(os.environ.get('V43_TRANSPORT_HELPERS', TEST_DIR))
if not (HERE/'v43_native_fp16_grouped.py').is_file():
    HERE = TEST_DIR.parent/'deploy/rk3588/validation/dpvo_native_transport'
VALIDATION = Path(os.environ['V43_ANCESTOR_VALIDATION']) if 'V43_ANCESTOR_VALIDATION' in os.environ else next(
    p for p in (HERE.parent/'public-snapshot/deploy/rk3588/validation', HERE.parent,
                TEST_DIR.parent/'deploy/rk3588/validation') if (p/'dpvo_grouped_update/v37_grouped_rknn.py').is_file())


def load(path):
    name = '_v43_test_' + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value; spec.loader.exec_module(value)
    return value


T = load(HERE/'v43_native_fp16_grouped.py')


def io(b=2, d=6):
    return ([dict(name=n, dtype='float32', shape=[b, c, d, 16]) for n, c in zip(T.NAMES, (384, 384, 882))],
            [dict(name=n, dtype='float32', shape=[b, c, d, 16]) for n, c in zip(('net_out', 'delta', 'weight'), (384, 2, 2))])


def logical_attrs(rows, *, output=False):
    values = []
    for index, row in enumerate(rows):
        b, c, d, m = row['shape']; count = b*c*d*m
        values.append(dict(index=index, name=row['name'], shape=[b, c, d, m] if output else [b, d, m, c],
            format=0 if output else 1, type=1, quantization_type=2, fractional_length=0,
            zero_point=0, scale=1.0, n_elems=count, size=count*2, pass_through=0,
            size_with_stride=count*2+(64 if output else 0), w_stride=m, h_stride=0))
    return values


def native_attrs(rows):
    result = []
    for index, row in enumerate(rows):
        b, c, d, m = row['shape']; shape = [b, (c+7)//8, d, m, 8]; count = int(np.prod(shape))
        result.append(dict(index=index, name=row['name'], shape=shape, n_dims=5, dims=shape+[0]*11,
            fmt=2, type=1, qnt_type=2, fl=0, zp=0, scale=1.0, scale_finite=True,
            n_elems=count, size=count*2, size_with_stride=count*2, w_stride=16, h_stride=0, pass_through=0))
    return result


def query_rows(rows):
    return [dict(command=cmd, input_index=i, return_code=0, attribute=copy.deepcopy(attr))
            for cmd in (8, 10) for i, attr in enumerate(native_attrs(rows))]


def queries_for(rows, index):
    return [v for v in query_rows(rows) if v['input_index'] == index]


def test_observed_B2D6_native_bytes_and_corr_tail():
    inputs, _ = io()
    for index, row in enumerate(inputs):
        actual = T.validate_native_pair(dict(row, index=index), queries_for(inputs, index))
        assert actual['shape'] == [2, 111 if index == 2 else 48, 6, 16, 8]
        assert actual['size'] == (340992 if index == 2 else 147456)


@pytest.mark.parametrize('field,value', [('fmt', 1), ('type', 0), ('qnt_type', 0), ('scale', .5),
    ('fl', 1), ('zp', 1), ('w_stride', 17), ('size_with_stride', 147458), ('pass_through', 1),
    ('dims', [2, 48, 6, 16, 8]+[1]+[0]*10)])
def test_unimplemented_or_nonidentity_native_attr_rejected(field, value):
    inputs, _ = io(); records = queries_for(inputs, 0)
    records[0]['attribute'][field] = value
    with pytest.raises(ValueError): T.validate_native_pair(dict(inputs[0], index=0), records)


def test_both_commands_required_and_must_match_frozen_model():
    inputs, _ = io(); records = queries_for(inputs, 0)
    with pytest.raises(ValueError): T.validate_native_pair(dict(inputs[0], index=0), records[:1])
    records[1]['return_code'] = -1
    with pytest.raises(ValueError): T.validate_native_pair(dict(inputs[0], index=0), records)
    records = queries_for(inputs, 0); ancestor = copy.deepcopy(records); ancestor[1]['attribute']['size'] += 2
    with pytest.raises(ValueError): T.validate_native_pair(dict(inputs[0], index=0), records, ancestor)


class FakePacker:
    @staticmethod
    def fpcr_record(library):
        return dict(supported=True, FPCR_modified_by_packer=False, raw=0)
    @staticmethod
    def pack_blocked_fp16(value, attribute, library):
        assert value.dtype == np.float32 and value.flags.c_contiguous and np.isfinite(value).all()
        if np.abs(value).max() > 65504: raise ValueError('overflow')
        out = np.zeros(attribute['shape'], dtype=np.uint16)
        channels = value.shape[1]
        cf = np.zeros((value.shape[0], out.shape[1]*8, value.shape[2], value.shape[3]), dtype=np.float16)
        cf[:, :channels] = value
        out[...] = cf.reshape(value.shape[0], out.shape[1], 8, value.shape[2], value.shape[3]).transpose(0, 1, 3, 4, 2).view(np.uint16)
        return out


@pytest.fixture
def setup():
    group = VALIDATION/'dpvo_grouped_update'
    m16 = VALIDATION/'dpvo_m16_grouped'
    bridge = load(m16/'v40_m16_grouped_rknn.py')
    parent = bridge.load_runtime(group/'v37_grouped_rknn.py', m16/'v40_m16_grouped_contracts.py',
        T.sha256(m16/'v40_m16_grouped_contracts.py'), group/'v37_grouped_contracts.py')
    inputs, outputs = io()
    query_models = {'B2_D6_M16': dict(queries=query_rows(inputs), model=dict(sha256='a'*64),
                                    logical_inputs=inputs, logical_outputs=outputs)}
    candidate = T._bind_transport(parent, FakePacker, lambda: object(), {}, query_models)
    abi = load(VALIDATION/'run-v12-rknn-block-board.py')
    return candidate, parent, abi, inputs, outputs


class FakeSDK:
    def __init__(self, abi, inputs, outputs, failure=None):
        self.abi, self.inputs, self.outputs, self.failure = abi, inputs, outputs, failure
        self.queries, self.release_count, self.destroy_count, self.run_count = [], 0, 0, 0
        self.buffers, self.requests = [], []
        self.expected = {row['name']: np.full(row['shape'], .5 if row['name'] == 'weight' else i+1, np.float32)
                         for i, row in enumerate(outputs)}
    def rknn_query(self, ctx, command, pointer, size):
        assert size == ct.sizeof(self.abi.TensorAttr)
        attr = ct.cast(pointer, ct.POINTER(self.abi.TensorAttr)).contents
        index = attr.index; self.queries.append((command, index))
        source = native_attrs(self.inputs)[index]
        for key in ('index', 'n_dims', 'fmt', 'type', 'qnt_type', 'fl', 'zp', 'scale',
                    'n_elems', 'size', 'size_with_stride', 'w_stride', 'h_stride', 'pass_through'):
            setattr(attr, key, source[key])
        attr.name = source['name'].encode(); attr.dims[:] = source['dims']
        if self.failure == 'query': return -8
        if self.failure == 'layout': attr.fmt = 1
        return 0
    def rknn_inputs_set(self, ctx, count, requests):
        assert count == 3
        for i in range(count):
            request = requests[i]; assert (request.index, request.type, request.fmt, request.pass_through) == (i, 1, 2, 1)
            arr = np.ctypeslib.as_array(ct.cast(request.buf, ct.POINTER(ct.c_uint16)), shape=(request.size//2,)).copy()
            self.requests.append((i, request.size, arr))
        return -3 if self.failure == 'set' else 0
    def rknn_run(self, ctx, pointer):
        self.run_count += 1; ct.cast(pointer, ct.POINTER(self.abi.RunExtend)).contents.frame_id = 42; return 0
    def rknn_outputs_get(self, ctx, count, outputs, received):
        for i, row in enumerate(self.outputs):
            value = self.expected[row['name']].copy(); self.buffers.append(value)
            outputs[i].buf, outputs[i].size = value.ctypes.data, value.nbytes
            assert outputs[i].want_float == 1 and outputs[i].is_prealloc == 0
            if self.failure == 'partial_get': return -2
        ct.cast(received, ct.POINTER(self.abi.OutputExtend)).contents.frame_id = 42
        if self.failure == 'bytes': outputs[0].size -= 4
        return 0
    def rknn_outputs_release(self, ctx, count, outputs):
        self.release_count += 1
        for value in self.buffers: value.fill(999)
        return 0
    def rknn_destroy(self, ctx): self.destroy_count += 1; return 0


def fake_session(setup, failure=None):
    candidate, _, abi, inputs, outputs = setup
    sdk = FakeSDK(abi, inputs, outputs, failure)
    session = candidate.GroupedSession(abi, sdk, abi.CONTEXT(9), inputs, outputs,
        logical_attrs(inputs), logical_attrs(outputs, output=True))
    feed = {row['name']: np.full(row['shape'], (i+1)/8, np.float32) for i, row in enumerate(inputs)}
    return session, sdk, feed


def test_pack_query_flags_pointer_tail_and_owned_output_release(setup):
    candidate, _, abi, inputs, _ = setup
    session, sdk, feed = fake_session(setup)
    session.configure_native_inputs('B2_D6_M16')
    assert sdk.queries == [(cmd, i) for cmd in (8, 10) for i in range(3)]
    packed = candidate.pack_inputs(abi, feed, session.input_bindings)
    for request, buffer in zip(packed.requests, packed.owned_buffers):
        assert request.buf == buffer.ctypes.data and request.size == buffer.nbytes
        assert buffer.flags.owndata and buffer.flags.c_contiguous
    corr = packed.owned_buffers[2].view(np.float16).transpose(0, 1, 4, 2, 3).reshape(2, 888, 6, 16)
    assert np.array_equal(corr[:, :882], feed['corr']) and not corr[:, 882:].view(np.uint16).any()
    actual, record = session.execute(feed, sdk.expected)
    assert record['execution_status'] == 'completed' and record['strict_component_gate_passed']
    assert record['candidate_accepted'] is False and record['transport_equivalence_validated'] is False
    assert record['timing']['forward_seconds'] >= record['timing']['pack_seconds'] >= 0
    assert record['timing']['forward_seconds'] >= record['timing']['outputs_release_seconds'] >= 0
    assert all(np.array_equal(actual[n], v) and actual[n].flags.owndata for n, v in sdk.expected.items())
    assert all((value == 999).all() for value in sdk.buffers)
    session.close(); session.close()
    metrics = session.metrics()
    assert metrics['npu_used'] is False and metrics['sdk_run_api_calls'] == metrics['SDK_outputs_releases'] == 1
    assert metrics['closed'] and metrics['context_destroyed'] and sdk.destroy_count == 1


@pytest.mark.parametrize('failure', ['query', 'layout'])
def test_bad_query_preserves_raw_closes_and_never_runs(setup, failure):
    session, sdk, _ = fake_session(setup, failure)
    with pytest.raises((ValueError, RuntimeError)): session.configure_native_inputs('B2_D6_M16')
    metrics = session.metrics()
    assert metrics['native_input_queries'] and metrics['closed'] and sdk.destroy_count == 1 and sdk.run_count == 0
    assert metrics['native_inputs_validated'] is False


@pytest.mark.parametrize('failure', ['set', 'partial_get', 'bytes'])
def test_frozen_execute_failure_cleanup_retained(setup, failure):
    session, sdk, feed = fake_session(setup, failure)
    session.configure_native_inputs('B2_D6_M16')
    _, report = session.execute(feed)
    assert report['execution_status'] == 'failed' and session.metrics()['closed'] and sdk.destroy_count == 1
    assert sdk.release_count == (0 if failure == 'set' else 1)


def test_unconfigured_session_never_silently_uses_old_pack(setup):
    session, sdk, feed = fake_session(setup)
    _, report = session.execute(feed)
    assert report['execution_status'] == 'failed' and 'not queried' in report['error']
    assert sdk.run_count == 0 and sdk.destroy_count == 1


def test_diagnostic_flag_required_before_files_and_no_shared_pack_mutation(setup):
    candidate, _, _, inputs, outputs = setup
    with pytest.raises(ValueError, match='diagnostic_allow_native_fp16'):
        candidate.GroupedSession.open_board('missing', 'a'*64, inputs, outputs)
    ancestor = load(VALIDATION/'dpvo_grouped_update/v37_grouped_rknn.py')
    assert ancestor.pack_inputs is not candidate.pack_inputs
    assert ancestor.pack_inputs.__module__ == ancestor.__name__
    with pytest.raises(ValueError, match='diagnostic_allow_native_fp16'):
        T.load_runtime('missing', 'missing', 'a'*64, 'missing', runtime_bridge='missing',
            packer_path='missing', packer_sha256='a'*64, packing_library_path='missing',
            packing_library_sha256='a'*64, packing_source_pins={}, native_query_receipt='missing', native_query_sha256='a'*64)
