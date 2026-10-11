"""Tiny generated CF buffers, fake SDK and metadata; no actual neural execution."""
import copy
import ctypes as ct
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from contextlib import contextmanager

import numpy as np
import pytest

TEST_DIRECTORY = Path(__file__).resolve().parent
HERE = (Path(os.environ['V37_GROUPED_HELPERS']).resolve() if 'V37_GROUPED_HELPERS' in os.environ else
        TEST_DIRECTORY if (TEST_DIRECTORY/'v37_grouped_rknn.py').exists() else
        TEST_DIRECTORY.parent/'deploy/rk3588/validation/dpvo_grouped_update')

def source_path(environment, public_path, historical_path):
    if environment in os.environ:
        return Path(os.environ[environment]).resolve()
    return public_path if public_path.exists() else Path(historical_path)
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value; spec.loader.exec_module(value)
    return value
runtime = load(HERE/'v37_grouped_rknn.py', '_v37_test_runtime')
C = runtime.C
compiler = load(HERE/'compile-v37-grouped-update.py', '_v37_test_compiler')


def io_contract(b=2, d=2):
    return ([dict(name=name, dtype='float32', shape=[b, c, d, 48]) for name, c in zip(C.INPUT_NAMES, (384, 384, 882))],
            [dict(name=name, dtype='float32', shape=[b, c, d, 48]) for name, c in zip(C.OUTPUT_NAMES, (384, 2, 2))])


def attrs(rows, fmt=0, padding=0):
    result = []
    for i, row in enumerate(rows):
        b, c, d, m = row['shape']; n = b*c*d*m
        result.append(dict(index=i, name=row['name'], shape=[b, d, m, c] if fmt == 1 else list(row['shape']),
            format=fmt, type=1, n_elems=n, size=n*2, size_with_stride=n*2+padding,
            quantization_type=2, scale=1., zero_point=0, fractional_length=0, pass_through=0, w_stride=0, h_stride=0))
    return result


@pytest.fixture
def abi():
    path = source_path('V37_ABI_HELPER', HERE.parent/'run-v12-rknn-block-board.py',
        '/run/user/1000/edge-vggt-v36/public-snapshot/deploy/rk3588/validation/run-v12-rknn-block-board.py')
    assert C.sha256(path) == C.ABI_SHA256
    return load(path, '_v37_test_frozen_ABI')  # Definitions only; no header/library/board access.


@pytest.fixture
def rejected_manifest(tmp_path):
    inputs, outputs = io_contract(8, 8); key = 'B8_D8_M48'
    model = tmp_path/'tiny.onnx'; model.write_bytes(b'FAKE_GRAPH_NOT_INFERENCE')
    models = {key: dict(inputs=inputs, outputs=outputs, padding=False, external_data_used=False,
        numerical_validation_completed=True, parameter_coverage=dict(original_parameter_count=50, all_original_parameters_covered=True),
        model=dict(file=model.name, bytes=model.stat().st_size, sha256=C.sha256(model)))}
    report = dict(schema=C.EXPORT_SCHEMA, script_sha256=C.EXPORTER_SHA256,
        status='completed_grouped_Update_numerical_rejected', strict_FP32_gate_passed=False,
        candidate_accepted=False, original_parameters_unchanged=True,
        original_weights_before={'fake': 'stable'}, original_weights_after={'fake': 'stable'},
        original_Update_parameters={str(i): {} for i in range(50)},
        outputs_finite=True, weight_domain_valid=True, cleanup_errors=[], affinity_restored=True,
        source_pins={'/board/nonexistent/source.py': 'a'*64}, source_hashes_after={'/board/nonexistent/source.py': 'a'*64},
        sources_and_inputs_unchanged=True, models=models, cases=[])
    for observation in (13, 127, 191):
        report['cases'].append(dict(observation_id=observation, Torch_domain=dict(finite=True, weight_domain_valid=True),
            ORT_domain=dict(finite=True, weight_domain_valid=True), degree_plan=dict(all_edges_exactly_once=True,
                original_edge_order_restored=True, padding=False), groups=[dict(model_key=key)]))
    path = tmp_path/'manifest.json'; path.write_text(json.dumps(report))
    return path, report


def test_cpu_rejection_requires_flag_and_host_does_not_require_board_files(rejected_manifest):
    path, report = rejected_manifest
    with pytest.raises(ValueError, match='diagnostic-allow'):
        C.validate_export_manifest(path, C.sha256(path))
    actual, _ = C.validate_export_manifest(path, C.sha256(path), allow_fp32_rejected_diagnostic=True)
    assert actual['strict_FP32_gate_passed'] is False and actual['candidate_accepted'] is False
    with pytest.raises(FileNotFoundError):
        C.validate_export_manifest(path, C.sha256(path), allow_fp32_rejected_diagnostic=True, verify_source_files=True)


@pytest.mark.parametrize('key,value', [('candidate_accepted', True), ('outputs_finite', False), ('original_parameters_unchanged', False)])
def test_diagnostic_flag_does_not_bypass_finite_parameter_or_adoption_contract(rejected_manifest, key, value):
    path, report = rejected_manifest; report[key] = value; path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        C.validate_export_manifest(path, C.sha256(path), allow_fp32_rejected_diagnostic=True)


def test_hardware_entry_requires_diagnostic_flag_before_files_or_SDK():
    inputs, outputs = io_contract()
    with pytest.raises(ValueError, match='ancestry'):
        runtime.GroupedSession.open_board('missing', 'a'*64, inputs, outputs,
            abi_path='missing', header_path='missing', library_path='missing', log_helper_path='missing',
            export_strict_FP32_gate_passed=False)


def test_fake_compiler_sequence_keeps_exact_nonquantized_configuration(tmp_path):
    class Fake:
        def __init__(self): self.calls = []
        def config(self, **kw): self.calls.append(('config', kw)); return 0
        def load_onnx(self, **kw): self.calls.append(('load', kw)); return 0
        def build(self, **kw): self.calls.append(('build', kw)); return 0
        def export_rknn(self, path): self.calls.append(('export', path)); return 0
    fake = Fake(); operations = []
    compiler.compile_operations(fake, tmp_path/'fake.onnx', tmp_path/'fake.rknn', operations)
    assert fake.calls[0] == ('config', dict(target_platform='rk3588', float_dtype='float16', optimization_level=3))
    assert fake.calls[2] == ('build', dict(do_quantization=False))
    assert [row['name'] for row in operations] == ['config', 'load_onnx', 'build', 'export_rknn']
    fake.config = lambda **kw: -9
    with pytest.raises(RuntimeError, match='config returned -9'):
        compiler.compile_operations(fake, 'not_read', 'not_written', [])


def test_true_B_NHWC_pack_preserves_pixels_and_request_fmt1(abi):
    inputs, outputs = io_contract(8, 8); ins, _ = C.contracts(inputs, outputs)
    bindings = C.bind_attrs(attrs(inputs, fmt=1), ins, output=False)
    feed = {row['name']: np.arange(np.prod(row['shape']), dtype=np.float32).reshape(row['shape']) for row in inputs}
    packed = runtime.pack_inputs(abi, feed, bindings)
    for row, buffer in zip(bindings, packed.owned_buffers):
        assert np.array_equal(buffer, feed[row['name']].transpose(0, 2, 3, 1))
        assert buffer.flags.c_contiguous and buffer.flags.owndata
        request = packed.requests[row['index']]
        assert request.fmt == 1 and request.type == 0 and request.pass_through == 0
        assert request.buf == buffer.ctypes.data and request.size == buffer.nbytes


@pytest.mark.parametrize('failure', ['mapping', 'dtype', 'quant', 'input_padding', 'stride_underflow'])
def test_unknown_layout_quantization_or_stride_rejected(failure):
    inputs, outputs = io_contract(); ins, outs = C.contracts(inputs, outputs)
    raw = attrs(outputs if failure == 'stride_underflow' else inputs, fmt=1)
    if failure == 'mapping': raw[0]['shape'][1] += 1
    elif failure == 'dtype': raw[0]['type'] = 8
    elif failure == 'quant': raw[0]['scale'] = .5
    elif failure == 'input_padding': raw[0]['size_with_stride'] += 16
    else: raw[0]['size_with_stride'] -= 2
    with pytest.raises(ValueError): C.bind_attrs(raw, outs if failure == 'stride_underflow' else ins, output=failure == 'stride_underflow')


class FakeSDK:
    def __init__(self, abi, rows, fmt=0, failure=None):
        self.abi, self.rows, self.fmt, self.failure = abi, rows, fmt, failure
        self.release_count = self.destroy_count = self.run_count = 0
        self.buffers = []
        self.expected = {row['name']: np.full(row['shape'], .5 if row['name'] == 'weight' else i+1, dtype=np.float32)
                         for i, row in enumerate(rows)}
        if failure == 'nan': self.expected['net_out'].flat[0] = np.nan
        if failure == 'domain': self.expected['weight'].flat[0] = -1e-7
    def rknn_inputs_set(self, ctx, count, requests):
        assert count == 3
        assert all(requests[i].type == 0 and requests[i].pass_through == 0 for i in range(3)); return 0
    def rknn_run(self, ctx, ext):
        self.run_count += 1
        ct.cast(ext, ct.POINTER(self.abi.RunExtend)).contents.frame_id = 37
        return 0
    def rknn_outputs_get(self, ctx, count, outputs, received):
        for i, row in enumerate(self.rows):
            value = self.expected[row['name']]
            if self.fmt == 1: value = value.transpose(0, 2, 3, 1)
            owned = np.array(value, order='C', copy=True); self.buffers.append(owned)
            outputs[i].buf, outputs[i].size = owned.ctypes.data, owned.nbytes
            assert outputs[i].want_float == 1 and outputs[i].is_prealloc == 0
            if self.failure == 'partial_get': return -2
        ct.cast(received, ct.POINTER(self.abi.OutputExtend)).contents.frame_id = 38 if self.failure == 'frame' else 37
        if self.failure == 'bytes': outputs[0].size -= 4
        return 0
    def rknn_outputs_release(self, ctx, count, outputs):
        self.release_count += 1
        for value in self.buffers: value.fill(999)
        return -3 if self.failure == 'release' else 0
    def rknn_destroy(self, ctx): self.destroy_count += 1; return -4 if self.failure == 'destroy' else 0


def session(abi, *, fmt=0, failure=None, log=None):
    inputs, outputs = io_contract()
    api = FakeSDK(abi, outputs, fmt=fmt, failure=failure)
    value = runtime.GroupedSession(abi, api, abi.CONTEXT(7), inputs, outputs,
        attrs(inputs, fmt=1), attrs(outputs, fmt=fmt, padding=128), log_helper=log)
    feed = {row['name']: np.zeros(row['shape'], dtype=np.float32) for row in inputs}
    return value, api, feed


@pytest.mark.parametrize('fmt', [0, 1, 3])
def test_sdk_owned_output_before_poison_release_and_CF_restore(abi, fmt):
    value, api, feed = session(abi, fmt=fmt)
    outputs, report = value.execute(feed, api.expected)
    assert report['execution_status'] == 'completed' and report['strict_component_gate_passed']
    assert report['candidate_accepted'] is False and api.release_count == 1
    for name, array in outputs.items():
        assert np.array_equal(array, api.expected[name]) and array.flags.c_contiguous and array.flags.owndata
    assert all((buffer == 999).all() for buffer in api.buffers)
    assert report['timing']['forward_seconds'] >= report['timing']['outputs_release_seconds'] >= 0
    value.close(); value.close()
    assert api.destroy_count == 1 and value.metrics()['npu_used'] is False


@pytest.mark.parametrize('failure', ['frame', 'bytes', 'partial_get', 'release'])
def test_failure_releases_partial_or_complete_buffers_and_destroys(abi, failure):
    value, api, feed = session(abi, failure=failure)
    raw, report = value.execute(feed)
    assert report['execution_status'] == 'failed' and api.release_count == api.destroy_count == 1
    assert value.metrics()['closed'] and report['candidate_accepted'] is False


@pytest.mark.parametrize('failure', ['nan', 'domain'])
def test_invalid_raw_arrays_preserved_without_acceptance_or_NaN_JSON(abi, failure):
    value, api, feed = session(abi, failure=failure)
    outputs, report = value.execute(feed, api.expected)
    assert report['execution_status'] == 'completed' and not report['strict_component_gate_passed']
    assert not report['outputs_finite'] if failure == 'nan' else not report['weight_domain_valid']
    assert np.isnan(outputs['net_out'].flat[0]) if failure == 'nan' else outputs['weight'].flat[0] < 0
    json.dumps(report, allow_nan=False)
    value.close()


def test_internal_E_RKNN_is_failure_even_when_all_API_return_zero(abi):
    @contextmanager
    def log(directory):
        yield dict(truncated=False, RKNN_error_detected=True, raw_text='E RKNN normalize error', bytes=0, raw_sha256='a'*64)
    value, api, feed = session(abi, log=NS(_native_log=log))
    outputs, report = value.execute(feed)
    assert outputs and report['execution_status'] == 'failed' and value.metrics()['context_destroyed']
    assert api.release_count == api.destroy_count == 1


class DummySession:
    def __init__(self, spec, dma=30, destroy_fail=False):
        self.spec, self.destroy_fail = spec, destroy_fail
        self.closed = self.destroyed = False; self.calls = 0
        self.host, self.dma = spec['model_bytes']+1, dma
    def execute(self, feed, reference=None, timeout_ms=60000):
        self.calls += 1
        shape = self.spec['outputs'][0]['shape']
        values = {name: np.ones((shape[0], c, shape[2], shape[3]), dtype=np.float32)
                  for name, c in zip(C.OUTPUT_NAMES, (384, 2, 2))}
        return values, dict(execution_status='completed', outputs_finite=True, weight_domain_valid=True,
                            all_outputs_zero=False, candidate_accepted=False)
    def close(self):
        if self.closed: return
        self.closed = True; self.destroyed = not self.destroy_fail
        if self.destroyed: self.host = self.dma = 0
    def metrics(self):
        return dict(closed=self.closed, context_destroyed=self.destroyed, current_model_host_bytes=self.host,
                    current_DMA_bytes=self.dma, cleanup_errors=['destroy failed'] if self.closed and self.destroy_fail else [])


def specs():
    result = {}
    for b in (1, 2):
        inputs, outputs = io_contract(b, 2)
        result[f'B{b}_D2_M48'] = dict(inputs=inputs, outputs=outputs, model_bytes=10, model_sha256='a'*64)
    return result


def test_bounded_LRU_reuses_one_context_per_shape_and_closes_all():
    made = []
    def factory(spec): value = DummySession(spec); made.append(value); return value
    cache = runtime.BoundedContextCache(specs(), factory, max_contexts=1, max_model_bytes=22, max_dma_bytes=60)
    for key in ('B1_D2_M48', 'B1_D2_M48', 'B2_D2_M48'): cache.execute(key, {})
    metrics = cache.metrics()
    assert metrics['created'] == 2 and metrics['hits'] == metrics['evicted'] == 1
    assert metrics['resident_contexts'] == metrics['peak_resident_contexts'] == 1
    cache.close(); cache.close()
    assert cache.metrics()['current_DMA_bytes'] == cache.metrics()['current_model_host_bytes'] == 0
    assert all(value.destroyed for value in made)
    with pytest.raises(ValueError, match='closed'): cache.execute('B1_D2_M48', {})


def test_actual_DMA_over_budget_closes_new_context_and_records_transient_peak():
    made = []
    def factory(spec): value = DummySession(spec, dma=31); made.append(value); return value
    cache = runtime.BoundedContextCache(specs(), factory, max_dma_bytes=30)
    with pytest.raises(ValueError, match='allocation'): cache.execute('B1_D2_M48', {})
    assert made[0].destroyed and cache.metrics()['resident_contexts'] == 0
    assert cache.metrics()['peak_transient_init_DMA_bytes'] == 31
    cache.close()


def test_destroy_failure_does_not_falsely_zero_memory():
    cache = runtime.BoundedContextCache(specs(), lambda spec: DummySession(spec, destroy_fail=True))
    cache.execute('B1_D2_M48', {}); cache.close()
    report = cache.metrics()
    assert report['cleanup_errors'] and report['current_DMA_bytes'] == 30 and report['resident_contexts'] == 1


def test_fixed_FP32_boundary_and_nonfinite_metrics_are_json_safe():
    reference = np.zeros((2, 2, 2, 48), dtype=np.float32); actual = reference.copy()
    actual.flat[0] = np.nextafter(np.float32(1e-5), np.float32(np.inf))
    report = runtime.compare(actual, reference)
    assert report['failed_elements'] == 1 and not report['gate_passed']
    actual.flat[0] = np.nan; report = runtime.compare(actual, reference)
    assert report['finite'] is False and report['max_abs'] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('tamper', [None, 'rejection_flag', 'graph_SHA', 'configuration'])
def test_compiled_lineage_preserves_original_rejection(rejected_manifest, tmp_path, tamper):
    path, manifest = rejected_manifest
    contracts_file = HERE/'v37_grouped_contracts.py'
    compiler_file = HERE/'compile-v37-grouped-update.py'
    output = tmp_path/'block.rknn'; output.write_bytes(b'FAKE_SDK_MODEL_NO_NEURAL_EXECUTION')
    report = dict(schema=C.COMPILE_SCHEMA, status='compiled_group_diagnostic_not_runtime_validated',
        script_sha256=C.sha256(compiler_file), contracts_sha256=C.sha256(contracts_file),
        manifest_sha256=C.sha256(path), model_key='B8_D8_M48',
        input=dict(sha256=manifest['models']['B8_D8_M48']['model']['sha256']),
        configuration=copy.deepcopy(C.CONFIGURATION), released=True,
        sources_and_inputs_unchanged=True, source_pins={'declaration': 'a'*64},
        source_hashes_after={'declaration': 'a'*64}, candidate_accepted=False,
        export_strict_FP32_gate_passed=False, diagnostic_allow_fp32_rejected=True,
        output=dict(file=output.name, bytes=output.stat().st_size, sha256=C.sha256(output)))
    if tamper == 'rejection_flag': report['diagnostic_allow_fp32_rejected'] = False
    elif tamper == 'graph_SHA': report['input']['sha256'] = 'b'*64
    elif tamper == 'configuration': report['configuration']['do_quantization'] = True
    compiled = tmp_path/'compile.json'; compiled.write_text(json.dumps(report))
    kwargs = dict(compiler_path=compiler_file, compiler_sha256=C.sha256(compiler_file),
                  contracts_path=contracts_file, contracts_sha256=C.sha256(contracts_file))
    if tamper:
        with pytest.raises(ValueError):
            C.validate_compilation(compiled, C.sha256(compiled), manifest, C.sha256(path), 'B8_D8_M48', **kwargs)
    else:
        result, model, _ = C.validate_compilation(compiled, C.sha256(compiled), manifest, C.sha256(path), 'B8_D8_M48', **kwargs)
        assert model == output and result['candidate_accepted'] is False
        assert result['export_strict_FP32_gate_passed'] is False


def synthetic_ragged_graph():
    adapter_file = source_path('V37_GROUPED_ADAPTER', HERE/'source_frame_batches.py',
        '/run/user/1000/edge-vggt-v36/update/source_frame_batches.py')
    assert C.sha256(adapter_file) == C.ADAPTER_SHA256
    adapter = load(adapter_file, '_v37_test_frozen_group_adapter')
    edges = [(source, target, source*48+patch)
             for source, targets in ((0, (0,)), (1, (0, 1)))
             for target in targets for patch in range(48)]
    ordered = np.array(edges, dtype=np.int64)[np.random.default_rng(7).permutation(len(edges))]
    ii, jj, kk = (np.array(ordered[:, index], copy=True) for index in range(3))
    plan = adapter.build_degree_plan(ii, jj, kk)
    net = np.broadcast_to(np.arange(len(edges), dtype=np.float32)[None, :, None], (1, len(edges), 384)).copy()
    values = dict(net=net, inp=np.full_like(net, .25), corr=np.zeros((1, len(edges), 882), dtype=np.float32))
    identity = adapter.SnapshotIdentity(1, 2, 3, 4)
    return adapter, plan, values, identity


@pytest.mark.parametrize('failure', [None, 'late_group', 'stale_after_all_groups'])
def test_full_graph_restores_shuffled_edges_and_never_returns_partial(failure):
    adapter, plan, inputs, identity = synthetic_ragged_graph()
    before = {key: value.copy() for key, value in inputs.items()}
    current, calls = [identity], []
    class Cache:
        def execute(self, key, feed, reference=None, *, timeout_ms):
            calls.append(key)
            values = dict(net_out=feed['net']+feed['inp'],
                          delta=np.array(feed['net'][:, :2], copy=True),
                          weight=np.full_like(feed['net'][:, :2], .5))
            record = dict(execution_status='completed', outputs_finite=True,
                          weight_domain_valid=True, all_outputs_zero=False)
            if failure == 'late_group' and len(calls) == 2:
                record['execution_status'] = 'failed'
            if failure == 'stale_after_all_groups' and len(calls) == 2:
                current[0] = adapter.SnapshotIdentity(1, 2, 3, 5)
            return values, record
    if failure is None:
        result, receipt = runtime.execute_full_graph(adapter, plan, inputs, Cache(),
            snapshot_identity=identity, current_identity=lambda: current[0])
        assert calls == ['B1_D1_M48', 'B1_D2_M48']
        assert np.array_equal(result['net'], inputs['net']+inputs['inp'])
        assert np.array_equal(result['delta'], inputs['net'][:, :, :2])
        assert (result['weight'] == .5).all()
        assert receipt['full_graph_returned'] and receipt['sequential_groups'] == 2
        assert receipt['candidate_accepted'] is False and receipt['VO_state_committed'] is False
        # Root single-group driver's same inverse-layout advanced-index scatter,
        # with deliberately shuffled original edges rather than naturally sorted IDs.
        one = adapter.build_degree_plan(np.zeros(96, dtype=np.int64), np.repeat([0, 1], 48)[::-1].copy(),
            np.tile(np.arange(48, dtype=np.int64), 2)[::-1].copy())
        original = np.arange(96*2, dtype=np.float32).reshape(1, 96, 2)
        batch = one.batches[0]; cf = adapter.pack_batch(original, batch)
        restored = np.empty_like(original)
        restored[0, batch.canonical_edge_indices, :] = cf.transpose(0, 2, 3, 1)
        assert np.array_equal(restored, original)
    else:
        expected = runtime.GroupExecutionError if failure == 'late_group' else RuntimeError
        with pytest.raises(expected):
            runtime.execute_full_graph(adapter, plan, inputs, Cache(),
                snapshot_identity=identity, current_identity=lambda: current[0])
        assert len(calls) == 2
    assert all(np.array_equal(inputs[key], value) for key, value in before.items())
