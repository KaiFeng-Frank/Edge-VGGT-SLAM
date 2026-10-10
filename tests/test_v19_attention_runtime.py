"""Synthetic ctypes/SDK protocol only; no model, images, ORT or real NPU."""
import copy
import ctypes as ct
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'deploy/rk3588/validation/run-v19-attention-board.py'
spec = importlib.util.spec_from_file_location('v19_attention_runtime_synthetic', PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def runtime():
    return m.load_helper('run-v12-rknn-block-board.py', m.RUNTIME_SHA)


@pytest.fixture
def exporter():
    return m.load_helper('export-v19-attention.py', m.EXPORTER_SHA)


def tensor_attr(runtime, shape, name, index=0, kind=1, quantization=2):
    attr = runtime.TensorAttr(index=index, n_dims=len(shape), name=name.encode(),
                              n_elems=int(np.prod(shape)), size=int(np.prod(shape)) * (2 if kind == 1 else 4),
                              fmt=3, type=kind, qnt_type=quantization, zp=0, scale=1.0)
    for i, value in enumerate(shape):
        attr.dims[i] = value
    return attr


class FakeSDK:
    """Own allocations and mutate them at release to catch stale SDK views."""
    def __init__(self, runtime, shapes, failure=None, bad_context=1):
        self.runtime, self.shapes = runtime, shapes
        self.failure, self.bad_context = failure, bad_context
        self.states, self.requests, self.releases, self.destroyed = {}, [], [], []
        self.lock = threading.Lock()
        self.active = self.max_active = 0

    @staticmethod
    def context_id(context):
        return context.value

    def rknn_init(self, context, filename, size, flags, extend):
        with self.lock:
            index = len(self.states) + 1
            self.states[index] = dict(shape=tuple(self.shapes[index-1]), input=None, output=None,
                                      run_id=100+index, flags=flags)
        ct.cast(context, ct.POINTER(self.runtime.CONTEXT)).contents.value = index
        return -1 if self.failure == 'init' and index == self.bad_context else 0

    def rknn_set_core_mask(self, context, mask):
        self.states[self.context_id(context)]['core_mask'] = mask
        return 0

    def rknn_query(self, context, command, pointer, size):
        state = self.states[self.context_id(context)]
        r = self.runtime
        if command == 0:
            value = r.InputOutputNum(3, 1)
        elif command in (1, 2):
            old = ct.cast(pointer, ct.POINTER(r.TensorAttr)).contents
            name = m.INPUT_NAMES[old.index] if command == 1 else 'output'
            if self.failure == 'attr' and command == 1 and old.index == 2:
                name = 'wrong-value'
            value = tensor_attr(r, state['shape'], name, int(old.index))
        elif command == 5:
            value = r.SDKVersion(api_version=b'2.3.2-synthetic', drv_version=b'fake-no-hardware')
        elif command == 6:
            value = r.MemSize(total_weight_size=0, total_internal_size=1024, total_dma_allocated_size=2048)
        elif command == 4:
            value = r.PerfRun(run_duration=1000)
        else:
            raise AssertionError(f'Unexpected synthetic query {command}')
        assert size == ct.sizeof(value)
        ct.memmove(pointer, ct.byref(value), size)
        return 0

    def rknn_inputs_set(self, context, count, request):
        cid = self.context_id(context)
        state = self.states[cid]
        assert count == 3
        saved = {}
        for i, name in enumerate(m.INPUT_NAMES):
            row = request[i]
            assert row.index == i and row.type == 0 and row.pass_through == 0 and row.fmt == 3
            assert row.size == int(np.prod(state['shape'])) * 4
            view = np.ctypeslib.as_array(ct.cast(row.buf, ct.POINTER(ct.c_float)),
                                       shape=(int(np.prod(state['shape'])),))
            saved[name] = view.copy().reshape(state['shape'])
        with self.lock:
            self.requests.append((cid, saved))
        state['input'] = saved
        return -2 if self.failure == 'input' and cid == self.bad_context else 0

    def rknn_run(self, context, extend):
        cid = self.context_id(context)
        run = ct.cast(extend, ct.POINTER(self.runtime.RunExtend)).contents
        assert run.non_block == 0 and run.timeout_ms == 30000 and run.fence_fd == -1
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(.004)  # Synthetic blocking SDK call; no hardware/model operation.
            run.frame_id = self.states[cid]['run_id']
            return -3 if self.failure == 'run' and cid == self.bad_context else 0
        finally:
            with self.lock:
                self.active -= 1

    def rknn_outputs_get(self, context, count, outputs, extend):
        cid = self.context_id(context)
        state = self.states[cid]
        assert count == 1 and outputs[0].want_float == 1 and outputs[0].is_prealloc == 0
        inputs = state['input']
        state['output'] = inputs['query'] + np.float32(2) * inputs['key'] + np.float32(3) * inputs['value']
        outputs[0].buf = state['output'].ctypes.data
        outputs[0].size = state['output'].nbytes
        received = ct.cast(extend, ct.POINTER(self.runtime.OutputExtend)).contents
        received.frame_id = state['run_id']
        if cid == self.bad_context:
            if self.failure == 'frame':
                received.frame_id += 1
            elif self.failure == 'size':
                outputs[0].size -= 4
            elif self.failure == 'get_partial':
                return -4  # Deliberately return an allocated buffer on failure.
        return 0

    def rknn_outputs_release(self, context, count, outputs):
        cid = self.context_id(context)
        assert count == 1 and outputs[0].buf
        with self.lock:
            self.releases.append(cid)
        self.states[cid]['output'].fill(-1234.)
        return -5 if self.failure == 'release' and cid == self.bad_context else 0

    def rknn_destroy(self, context):
        cid = self.context_id(context)
        assert self.active == 0, 'Destroy must happen after all workers have joined'
        self.destroyed.append(cid)
        return -6 if self.failure == 'destroy' and cid == self.bad_context else 0


def create_sessions(runtime, shapes, failure=None, masks=None):
    sdk = FakeSDK(runtime, shapes, failure)
    sessions = []
    masks = masks or [7] * len(shapes)
    offset = 0
    for i, (shape, mask) in enumerate(zip(shapes, masks)):
        spec = dict(shape=tuple(shape), name=f'synthetic-{i}', model=Path(f'model-{i}.rknn'),
                    core_mask=mask, begin=offset, end=offset+shape[1])
        offset += shape[1]
        m.create_session(sdk, runtime, spec, 0, sessions, {})
    return sdk, sessions


@pytest.mark.parametrize('kind,quantization', [(0, 0), (1, 0), (1, 2)])
def test_float_descriptor_and_exact_identity_affine_accepted(runtime, kind, quantization):
    shape = (1, 2, 5, 64)
    attr = tensor_attr(runtime, shape, 'query', kind=kind, quantization=quantization)
    assert m.validate_attr(runtime, attr, shape, 'query', 0)['type'] == kind


@pytest.mark.parametrize('field,value', [('scale', .5), ('zp', 1), ('type', 2), ('qnt_type', 1),
                                       ('size', 1), ('fmt', 2), ('n_elems', 1), ('name', b'key'),
                                       ('index', 1), ('scale', float('nan'))])
def test_descriptor_integer_quantization_identity_and_layout_fail_hard(runtime, field, value):
    shape = (1, 2, 5, 64)
    attr = tensor_attr(runtime, shape, 'query')
    setattr(attr, field, value)
    with pytest.raises(ValueError):
        m.validate_attr(runtime, attr, shape, 'query', 0)


def test_fp32_affine_not_silently_accepted(runtime):
    shape = (1, 2, 5, 64)
    with pytest.raises(ValueError, match='Only unquantized'):
        m.validate_attr(runtime, tensor_attr(runtime, shape, 'query', kind=0), shape, 'query', 0)


def test_three_fp32_inputs_owned_copy_frame_identity_release_and_no_global_shape(runtime):
    shape = (1, 3, 4, 64)
    sdk, sessions = create_sessions(runtime, [shape])
    inputs = {name: np.full(shape, i+1, np.float32) for i, name in enumerate(m.INPUT_NAMES)}
    original_global_shape = runtime.SHAPE
    row = {}
    actual = m.execute_call(sdk, runtime, sessions[0], inputs, 30000, row)
    np.testing.assert_array_equal(actual, np.full(shape, 14, np.float32))
    assert sdk.states[1]['output'].flat[0] == -1234  # SDK allocation no longer contains our result.
    assert row['runtime_frame_id'] == row['received_frame_id'] == 101
    assert row['output_copied_before_release'] and row['output_buffer_released']
    assert sdk.releases == [1] and row['run_api_called'] and row['run_api_succeeded']
    assert runtime.SHAPE == original_global_shape
    assert row['io_begin_monotonic'] <= row['run_begin_monotonic'] <= row['run_end_monotonic'] <= row['io_end_monotonic']
    errors = []
    m.destroy_sessions(sdk, runtime, sessions, errors)
    assert errors == [] and sdk.destroyed == [1]


@pytest.mark.parametrize('failure,release_count', [('input', 0), ('run', 0), ('frame', 1), ('size', 1),
                                                 ('get_partial', 1), ('release', 1)])
def test_failures_release_allocated_outputs_and_keep_context_destroyable(runtime, failure, release_count):
    shape = (1, 2, 4, 64)
    sdk, sessions = create_sessions(runtime, [shape], failure)
    row = {}
    with pytest.raises((RuntimeError, ValueError)):
        m.execute_call(sdk, runtime, sessions[0], {name: np.ones(shape, np.float32) for name in m.INPUT_NAMES},
                       30000, row)
    assert len(sdk.releases) == release_count
    errors = []
    m.destroy_sessions(sdk, runtime, sessions, errors)
    assert errors == [] and sdk.destroyed == [1]
    if failure == 'release':
        assert row['outputs_release_code'] == -5 and not row['output_buffer_released']


def test_parallel_barrier_actual_overlap_and_ordered_concat_match_baseline(runtime):
    original_shape = (2, 5, 7, 64)
    rng = np.random.default_rng(19)
    operands = {name: rng.normal(size=original_shape).astype(np.float32) for name in m.INPUT_NAMES}
    flat = {name: value.reshape(1, 10, 7, 64) for name, value in operands.items()}
    baseline, full = create_sessions(runtime, [(1, 10, 7, 64)])
    full_row = {}
    reference = m.execute_round(baseline, runtime, full, flat, original_shape, 30000, full_row)
    parallel, shards = create_sessions(runtime, [(1, 4, 7, 64), (1, 3, 7, 64), (1, 3, 7, 64)], masks=[1, 2, 4])
    parallel_row = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        actual = m.execute_round(parallel, runtime, shards, flat, original_shape, 30000, parallel_row, executor)
    np.testing.assert_array_equal(actual, reference)
    assert parallel.max_active == 3 and parallel.active == 0
    assert [parallel.states[i]['core_mask'] for i in (1, 2, 3)] == [1, 2, 4]
    assert parallel_row['run_api_overlap']['max_simultaneous'] == 3
    assert parallel_row['run_api_overlap']['all_three_overlap_seconds'] > 0
    assert parallel_row['packed_input_bytes'] == sum(value.nbytes for value in operands.values())
    assert len({row['worker_thread_id'] for row in parallel_row['calls']}) == 3
    assert parallel_row['end_to_end_seconds'] >= parallel_row['packing_seconds'] + parallel_row['concat_seconds']
    assert parallel_row['completed_synchronous_calls'] == 3 and sorted(parallel.releases) == [1, 2, 3]
    for sdk, sessions in ((baseline, full), (parallel, shards)):
        errors = []
        m.destroy_sessions(sdk, runtime, sessions, errors)
        assert errors == [] and len(sdk.destroyed) == len(sessions)


def test_parallel_failure_joins_other_workers_before_all_contexts_destroy(runtime):
    sdk, sessions = create_sessions(runtime, [(1, 2, 4, 64)] * 3, 'run', masks=[1, 2, 4])
    flat = {name: np.ones((1, 6, 4, 64), np.float32) for name in m.INPUT_NAMES}
    row = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        with pytest.raises(RuntimeError, match='rknn_run'):
            m.execute_round(sdk, runtime, sessions, flat, (1, 6, 4, 64), 30000, row, executor)
    assert sdk.active == 0 and sorted(sdk.releases) == [2, 3]
    assert row['worker_errors'] and len(row['calls']) == 3
    errors = []
    m.destroy_sessions(sdk, runtime, sessions, errors)
    assert errors == [] and sdk.destroyed == [3, 2, 1]


@pytest.mark.parametrize('failure', ['init', 'attr'])
def test_init_partial_context_and_raw_descriptor_failure_are_reclaimed(runtime, failure):
    shape = (1, 2, 4, 64)
    sdk = FakeSDK(runtime, [shape], failure)
    sessions, row = [], {}
    spec = dict(shape=shape, name='synthetic', model=Path('not-a-real-model'), core_mask=7, begin=0, end=2)
    with pytest.raises((RuntimeError, ValueError)):
        m.create_session(sdk, runtime, spec, 0, sessions, row)
    if failure == 'attr':
        assert len(row['input_attrs']) == 3 and row['output_attr']['name'] == 'output'
        assert row['tensor_attributes_validated'] is False
    errors = []
    m.destroy_sessions(sdk, runtime, sessions, errors)
    assert len(sessions) == 1 and sdk.destroyed == [1] and errors == []


def test_destroy_failure_does_not_prevent_other_owned_contexts_cleanup(runtime):
    sdk, sessions = create_sessions(runtime, [(1, 2, 4, 64)] * 3, 'destroy')
    errors = []
    m.destroy_sessions(sdk, runtime, sessions, errors)
    assert sdk.destroyed == [3, 2, 1] and len(errors) == 1
    assert sessions[0]['report']['context_destroyed'] is False
    assert sessions[1]['report']['context_destroyed'] is True


def test_fp32_strict_element_gate_reports_failure_separately(runtime):
    expected = np.array([0., 1., -2.], np.float32)
    actual = expected.copy()
    actual[0] = np.float32(2e-5)
    row = m.comparison(runtime, actual, expected)
    assert row['finite'] and not row['array_equal'] and not row['elementwise_gate_passed']
    assert row['outside_tolerance'] == 1 and row['rtol'] == 1e-4 and row['atol'] == 1e-5
    assert row['relative_l2'] > 0 and row['elements'] == 3


def test_overlap_keeps_unknown_and_distinguishes_touching_from_concurrent():
    assert m.interval_overlap([{}], 'a', 'b')['max_simultaneous'] is None
    rows = [dict(a=0., b=1.), dict(a=1., b=2.), dict(a=2., b=3.)]
    result = m.interval_overlap(rows, 'a', 'b')
    assert result['max_simultaneous'] == 1 and result['all_three_overlap_seconds'] == 0
    with pytest.raises(ValueError):
        m.interval_overlap([dict(a=2., b=1.)], 'a', 'b')


def valid_compile():
    return dict(status='compiled_not_runtime_validated', script_sha256=m.COMPILER_SHA,
                model_inference_executed=False, input={'sha256': 'a'*64}, output={'sha256': 'b'*64},
                released=True, configuration=dict(target_platform='rk3588', float_dtype='float16',
                                                  optimization_level=3, do_quantization=False),
                operations=[dict(name=name, code=0) for name in ('config', 'load_onnx', 'build', 'export_rknn')])


@pytest.mark.parametrize('alter', [None, 'model_sha', 'onnx_sha', 'quantization', 'operation', 'released'])
def test_compile_chain_and_configuration_fail_closed(alter):
    receipt = valid_compile()
    if alter == 'model_sha':
        receipt['output']['sha256'] = 'c'*64
    elif alter == 'onnx_sha':
        receipt['input']['sha256'] = 'c'*64
    elif alter == 'quantization':
        receipt['configuration']['do_quantization'] = True
    elif alter == 'operation':
        receipt['operations'][2]['code'] = 1
    elif alter == 'released':
        receipt['released'] = False
    if alter:
        with pytest.raises(ValueError):
            m.check_compile(receipt, 'a'*64, 'b'*64)
    else:
        m.check_compile(receipt, 'a'*64, 'b'*64)


def valid_export(exporter):
    shape = (2, 5, 7, 64)
    arrays = {'query': np.zeros(shape, np.float32)}
    capture = dict(block='synthetic', checkpoint={}, selected_pair={}, profile_sha256='a'*64,
                   source_pins={}, original_block_parameter_metadata={}, original_block_array_metadata={},
                   attention_semantics={'scale': .125}, flattening_order='C-order synthetic')
    plan = exporter.head_plan(shape)
    receipt = dict(kind='original_VGGT_independent_head_attention_ONNX_export',
                   status='exported_and_structurally_checked', script_sha256=m.EXPORTER_SHA,
                   capture_sha256='b'*64, original_shape=list(shape), flattened_shape=[1, 10, 7, 64],
                   head_shard_plan=plan, host_model_inference_executed=False, host_ORT_executed=False,
                   production_changed=False, npu_used=False, learned_parameter_count=0,
                   graph_inference_executed=False, source_and_inputs_unchanged=True,
                   saved_reference_partition_inverse_bitwise_equal=True,
                   original_parameter_metadata={}, original_array_metadata={}, models=[], **{
                       key: capture[key] for key in ('block', 'checkpoint', 'selected_pair', 'profile_sha256',
                                                    'source_pins', 'attention_semantics', 'flattening_order')})
    ranges = [(0, 10), *[(row['begin'], row['end']) for row in plan]]
    for name, (begin, end) in zip(m.MODEL_NAMES, ranges):
        want = [1, end-begin, 7, 64]
        receipt['models'].append(dict(name=name, begin=begin, end=end, independent_heads=end-begin,
                                     input_shape=want, output_shape=want, input_names=list(m.INPUT_NAMES),
                                     output_names=['output'], complete_key_count=7, token_or_key_partitioned=False,
                                     external_data_used=False, artifacts={key+'.npy': {} for key in (*m.INPUT_NAMES, 'output')}))
    return receipt, capture, arrays


@pytest.mark.parametrize('alter', [None, 'ordering', 'range', 'names', 'keys', 'token_partition', 'external', 'capture'])
def test_export_chain_requires_original_head_order_full_key_and_capture(exporter, alter):
    receipt, capture, arrays = valid_export(exporter)
    row = receipt['models'][1]
    if alter == 'ordering':
        receipt['models'][1:] = reversed(receipt['models'][1:])
    elif alter == 'range':
        row['begin'] += 1
    elif alter == 'names':
        row['input_names'].reverse()
    elif alter == 'keys':
        del row['artifacts']['output.npy']
    elif alter == 'token_partition':
        row['token_or_key_partitioned'] = True
    elif alter == 'external':
        row['external_data_used'] = True
    elif alter == 'capture':
        receipt['capture_sha256'] = 'c'*64
    if alter:
        with pytest.raises(ValueError):
            m.check_export(receipt, 'b'*64, capture, arrays, exporter)
    else:
        assert len(m.check_export(receipt, 'b'*64, capture, arrays, exporter)) == 4


def test_safe_artifact_hash_path_and_size_gate(tmp_path):
    root = tmp_path / 'export'
    root.mkdir()
    path = root / 'graph.onnx'
    path.write_bytes(b'synthetic graph bytes, not executable')
    entry = dict(file=path.name, bytes=path.stat().st_size, sha256=m.sha(path))
    assert m.safe_artifact(root, entry) == path
    for change in [dict(bytes=1), dict(sha256='0'*64), dict(file=str(path)), dict(file='../outside')]:
        bad = dict(entry, **change)
        with pytest.raises((ValueError, FileNotFoundError)):
            m.safe_artifact(root, bad)


def test_host_guard_before_output_creation_or_sdk(tmp_path):
    args = ['--mode', 'parallel', '--capture-dir', str(tmp_path/'capture'),
            '--export-dir', str(tmp_path/'export'), '--output', str(tmp_path/'out'),
            '--capture-sha256', 'a'*64, '--export-sha256', 'b'*64,
            '--models', 'synthetic.rknn', '--model-sha256', 'c'*64,
            '--compile-receipts', 'synthetic.json', '--compile-sha256', 'd'*64]
    with pytest.raises(RuntimeError, match='Board-only'):
        m.main(args)
    assert not (tmp_path/'out').exists()
