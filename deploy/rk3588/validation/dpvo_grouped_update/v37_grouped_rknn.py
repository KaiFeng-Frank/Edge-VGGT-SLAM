"""Research-only true-B DPVO grouped Update SDK transport and bounded cache.

No hardware/compiler/Torch imports at module load. Outputs own dense CF FP32
memory before SDK release. Original group topology and all parameter bytes live
in the separately pinned exporter/adapter. No VO state or candidate adoption.
MIT DPVO attribution: Princeton University, pinned 0ac95b656d1fda91c271d2a106460d19ad966fc7.
"""
from collections import OrderedDict
import copy
import ctypes as ct
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import sys
import threading
import time

import numpy as np

CONTRACTS_SHA256 = 'aa9e6af938f64ae4a4a82ffc84b487321fff208547703b0d65646c72b96d85c9'


def load_module(path, name):
    if name in sys.modules and Path(sys.modules[name].__file__).resolve() == Path(path).resolve():
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def load_contracts(path=None):
    import hashlib
    path = Path(path) if path is not None else Path(__file__).with_name('v37_grouped_contracts.py')
    if hashlib.sha256(path.read_bytes()).hexdigest() != CONTRACTS_SHA256:
        raise ValueError('Frozen grouped contracts SHA differs')
    return load_module(path, '_v37_grouped_contracts')


C = load_contracts()
INPUT_NAMES, OUTPUT_NAMES = C.INPUT_NAMES, C.OUTPUT_NAMES
validate_export_manifest, validate_compilation = C.validate_export_manifest, C.validate_compilation


@dataclass(frozen=True)
class PackedInputs:
    requests: object
    owned_buffers: tuple
    records: tuple


def pack_inputs(abi, feed, bindings):
    C.require(set(feed) == set(INPUT_NAMES) and len(bindings) == 3, 'Exactly three named CF arrays required')
    C.require([row['index'] for row in bindings] == [0, 1, 2]
              and {row['name'] for row in bindings} == set(INPUT_NAMES), 'SDK bindings differ')
    requests = (abi.Input*3)()
    buffers, records = [], []
    for row in bindings:
        value = feed[row['name']]
        C.require(isinstance(value, np.ndarray) and value.dtype == np.float32
                  and value.shape == row['shape'] and np.isfinite(value).all(), 'Finite FP32 logical group input required: '+row['name'])
        view = value.transpose(row['input_permutation']) if row['input_permutation'] else value
        owned = np.array(view, dtype=np.float32, order='C', copy=True)
        C.require(owned.shape == tuple(row['sdk_shape']) and owned.nbytes == row['transport_bytes'], 'Packed native shape/bytes differ')
        requests[row['index']] = abi.Input(row['index'], owned.ctypes.data, owned.nbytes, 0, 0, row['transport_format'])
        buffers.append(owned)
        records.append(dict(index=row['index'], name=row['name'], logical_shape=list(value.shape),
            buffer_shape=list(owned.shape), native_attribute_format=row['sdk_format'], request_format=row['transport_format'],
            source_dtype='float32', transport_dtype='float32', pass_through=0, bytes=owned.nbytes, owned=True,
            explicit_NHWC_pack=row['input_permutation'] is not None))
    return PackedInputs(requests, tuple(buffers), tuple(records))


def compare(actual, reference):
    C.require(actual.shape == reference.shape and actual.dtype == reference.dtype == np.float32, 'Same FP32 reference shape required')
    finite = bool(np.isfinite(actual).all() and np.isfinite(reference).all())
    if not finite:
        return dict(finite=False, gate_passed=False, failed_elements=None, relative_L2=None, max_abs=None,
                    nonfinite_elements=int((~np.isfinite(actual)).sum()), rtol=1e-4, atol=1e-5)
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    error = a-b
    failures = int(np.count_nonzero(np.abs(error) > 1e-5+1e-4*np.abs(b)))
    norm = float(np.linalg.norm(b.ravel()))
    return dict(finite=True, gate_passed=failures == 0, failed_elements=failures,
        relative_L2=float(np.linalg.norm(error.ravel())/norm) if norm else None,
        max_abs=float(np.abs(error).max()), nonfinite_elements=0, rtol=1e-4, atol=1e-5)


class StartupError(RuntimeError):
    def __init__(self, record):
        self.record = record
        super().__init__(record['error'])


class GroupExecutionError(RuntimeError):
    """Owned raw arrays can be saved by the owner; partial fullgraph is never returned."""
    def __init__(self, record, values):
        self.record, self.raw_outputs = record, values
        super().__init__(record.get('error', 'Invalid group output; fullgraph not returned'))


class GroupedSession:
    """One fixed-shape context. Fake construction is explicit and never uses NPU.

    Real construction is open_board(); no affinity, OMP or thread settings are
    changed here. Normal/profile use separate instances and SDK init flags.
    Native output padding is not read directly: outputs_get must return exactly
    logical FP32 4*N bytes, even when native size_with_stride is larger.
    """
    def __init__(self, abi, api, context, inputs, outputs, input_attrs, output_attrs, *,
                 pins=None, log_helper=None, log_directory=None, collect_perf_detail=False):
        ins, outs = C.contracts(inputs, outputs)
        self.input_bindings = C.bind_attrs(input_attrs, ins, output=False)
        self.output_bindings = C.bind_attrs(output_attrs, outs, output=True)
        self.abi, self.api, self.context = abi, api, context
        self._pins, self._log, self._log_directory = dict(pins or {}), log_helper, log_directory
        self._profile, self._hardware, self._model_buffer = bool(collect_perf_detail), False, None
        self._lock = threading.RLock()
        self._metrics = dict(synthetic=True, closed=False, context_destroyed=False, cleanup_errors=[],
            npu_used=False, sdk_run_api_calls=0, completed_calls=0, SDK_outputs_release_attempts=0,
            SDK_outputs_releases=0, model_host_buffer_bytes=0, collect_perf_detail=self._profile,
            input_attributes=copy.deepcopy(input_attrs), output_attributes=copy.deepcopy(output_attrs),
            input_bindings=copy.deepcopy(self.input_bindings), output_bindings=copy.deepcopy(self.output_bindings),
            memory=dict(total_dma_allocated_size=0, total_weight_size=0, total_internal_size=0),
            current_DMA_bytes=0, current_model_host_bytes=0,
            source_verification_scope='Exact pins before initialization and upon close; no per-call file hashing. Owner rechecks large provenance before/after experiment.')

    @classmethod
    def open_board(cls, model_path, model_sha256, inputs, outputs, *, abi_path, header_path,
                   library_path, log_helper_path, export_strict_FP32_gate_passed,
                   diagnostic_allow_fp32_rejected=False, pins=None, collect_perf_detail=False, log_directory=None):
        C.require(type(export_strict_FP32_gate_passed) is bool
                  and (export_strict_FP32_gate_passed or diagnostic_allow_fp32_rejected is True),
                  'Rejected CPU/ORT ancestry requires explicit diagnostic_allow_fp32_rejected')
        sources = dict(pins or {})
        abi_file = C.pin(abi_path, C.ABI_SHA256, sources)
        abi = load_module(abi_file, '_v37_frozen_v12_ABI')
        abi.require_board()  # Before native library or context initialization.
        C.pin(header_path, C.HEADER_SHA256, sources)
        C.pin(library_path, C.LIBRARY_SHA256, sources)
        C.pin(log_helper_path, C.LOG_HELPER_SHA256, sources)
        C.pin(Path(log_helper_path).with_name('v33_rknn_source_frame.py'), C.LOG_PARENT_SHA256, sources)
        C.pin(__file__, C.sha256(__file__), sources)
        C.pin(Path(__file__).with_name('v37_grouped_contracts.py'), CONTRACTS_SHA256, sources)
        model = C.pin(model_path, model_sha256, sources)
        C.contracts(inputs, outputs)
        C.recheck_pins(sources)
        C.require(log_directory is not None and Path(log_directory).is_dir(), 'Existing diagnostic log directory required')
        log = load_module(log_helper_path, '_v37_frozen_native_fd_capture')
        api = abi.bind_api(ct.CDLL(str(library_path)))
        ctx = abi.CONTEXT()
        instance, raw = None, dict(candidate_accepted=False, input_attributes=[], output_attributes=[], cleanup_errors=[])
        try:
            payload = model.read_bytes()
            C.require(0 < len(payload) <= C.UINT32_MAX, 'RKNN model exceeds ABI size')
            buffer = ct.create_string_buffer(payload)
            with log._native_log(log_directory) as native:
                abi.check_call(api.rknn_init(ct.byref(ctx), ct.cast(buffer, ct.c_void_p), len(payload),
                                           8 if collect_perf_detail else 0, None), 'rknn_init')
                abi.check_call(api.rknn_set_core_mask(ctx, 7), 'rknn_set_core_mask')
                version = abi.query(api, ctx, 5, abi.SDKVersion())
                raw['SDK_version'], raw['driver_version'] = bytes(version.api_version).decode(), bytes(version.drv_version).decode()
                C.require(raw['SDK_version'].startswith('2.3.2'), 'SDK version differs')
                counts = abi.query(api, ctx, 0, abi.InputOutputNum())
                C.require(counts.n_input == counts.n_output == 3, 'Actual SDK operand count differs')
                raw['input_attributes'] = [abi.attr_record(abi.query(api, ctx, 1, abi.TensorAttr(index=i))) for i in range(3)]
                raw['output_attributes'] = [abi.attr_record(abi.query(api, ctx, 2, abi.TensorAttr(index=i))) for i in range(3)]
                memory = abi.query(api, ctx, 6, abi.MemSize())
                raw['memory'] = {name: int(getattr(memory, name)) for name in
                    ('total_dma_allocated_size', 'total_weight_size', 'total_internal_size')}
            raw['native_log'] = native
            C.require(not native['truncated'] and not native['RKNN_error_detected'], 'Native SDK initialization emitted E RKNN')
            instance = cls(abi, api, ctx, inputs, outputs, raw['input_attributes'], raw['output_attributes'],
                           pins=sources, log_helper=log, log_directory=log_directory, collect_perf_detail=collect_perf_detail)
            instance._hardware, instance._model_buffer = True, buffer
            instance._metrics.update(synthetic=False, model_host_buffer_bytes=len(buffer), current_model_host_bytes=len(buffer),
                SDK_version=raw['SDK_version'], driver_version=raw['driver_version'], abi=abi.abi_record(),
                core_mask=7, init_flags=8 if collect_perf_detail else 0, memory=raw['memory'],
                current_DMA_bytes=raw['memory']['total_dma_allocated_size'], initialization_native_log=native,
                source_pins=sources, export_strict_FP32_gate_passed=export_strict_FP32_gate_passed,
                diagnostic_allow_fp32_rejected=diagnostic_allow_fp32_rejected, candidate_accepted=False)
            return instance
        except BaseException as error:
            if 'native' in locals():
                raw['native_log'] = native
            raw['error'] = repr(error)
            if instance is not None:
                instance.close(); raw['cleanup'] = instance.metrics()
            elif ctx.value:
                try:
                    code = int(api.rknn_destroy(ctx)); raw['context_destroyed'] = code == 0
                    if code: raw['cleanup_errors'].append('partial rknn_destroy '+str(code))
                except Exception as cleanup: raw['cleanup_errors'].append(repr(cleanup))
            raise StartupError(raw) from error

    def execute(self, feed, reference=None, *, timeout_ms=60000):
        with self._lock:
            start = time.monotonic()
            C.require(not self._metrics['closed'], 'Context is closed')
            C.require(type(timeout_ms) is int and 1 <= timeout_ms < 2**31, 'Invalid SDK timeout')
            values, row = {}, dict(candidate_accepted=False, execution_status='starting', mode='profile' if self._profile else 'normal')
            def run():
                begin = time.monotonic()
                packed = pack_inputs(self.abi, feed, self.input_bindings)
                packed_at = time.monotonic()
                outputs = (self.abi.Output*3)()
                for index in range(3):
                    outputs[index].index, outputs[index].want_float, outputs[index].is_prealloc = index, 1, 0
                extension = self.abi.RunExtend(non_block=0, timeout_ms=timeout_ms, fence_fd=-1)
                received = self.abi.OutputExtend()
                got = False
                try:
                    self.abi.check_call(self.api.rknn_inputs_set(self.context, 3, packed.requests), 'rknn_inputs_set')
                    set_at = time.monotonic()
                    self._metrics['sdk_run_api_calls'] += 1
                    self._metrics['npu_used'] = self._hardware
                    self.abi.check_call(self.api.rknn_run(self.context, ct.byref(extension)), 'rknn_run')
                    run_at = time.monotonic()
                    self.abi.check_call(self.api.rknn_outputs_get(self.context, 3, outputs, ct.byref(received)), 'rknn_outputs_get')
                    got = True; get_at = time.monotonic()
                    C.require(extension.frame_id == received.frame_id, 'Synchronous SDK frame ID mismatch')
                    for binding in self.output_bindings:
                        output = outputs[binding['index']]
                        C.require(bool(output.buf) and output.size == binding['returned_FP32_bytes'], 'Returned output pointer/logical FP32 bytes differ')
                        raw = np.ctypeslib.as_array(ct.cast(output.buf, ct.POINTER(ct.c_float)), shape=(math.prod(binding['shape']),))
                        native = raw.reshape(binding['sdk_shape'])
                        logical = native.transpose(binding['output_permutation']) if binding['output_permutation'] else native
                        values[binding['name']] = np.array(logical, dtype=np.float32, copy=True, order='C')
                    copy_at = time.monotonic()
                    row['timing'] = dict(pack_seconds=packed_at-begin, input_set_seconds=set_at-packed_at,
                        run_api_seconds=run_at-set_at, outputs_get_seconds=get_at-run_at, owned_copy_seconds=copy_at-get_at,
                        frame_id=int(extension.frame_id), input_records=list(packed.records), synchronous=True)
                finally:
                    if got or any(output.buf for output in outputs):
                        release_begin = time.monotonic()
                        self._metrics['SDK_outputs_release_attempts'] += 1
                        code = int(self.api.rknn_outputs_release(self.context, 3, outputs))
                        row.setdefault('timing', {})['outputs_release_seconds'] = time.monotonic()-release_begin
                        row['outputs_released'] = code == 0
                        if code == 0: self._metrics['SDK_outputs_releases'] += 1
                        else:
                            self._metrics['cleanup_errors'].append('rknn_outputs_release '+str(code))
                            raise RuntimeError('rknn_outputs_release returned '+str(code))
                    row.setdefault('timing', {})['forward_seconds'] = time.monotonic()-begin
            try:
                if self._log is not None:
                    try:
                        with self._log._native_log(self._log_directory) as native:
                            run()
                    finally:
                        row['native_log'] = native
                    C.require(not native['truncated'] and not native['RKNN_error_detected'], 'Native SDK emitted E RKNN/oversized log')
                else:
                    C.require(not self._hardware, 'Hardware log capture is mandatory'); run()
                row.update(execution_status='completed', outputs_finite=bool(all(np.isfinite(value).all() for value in values.values())),
                    weight_domain_valid=bool(np.isfinite(values['weight']).all() and ((values['weight'] >= 0) & (values['weight'] <= 1)).all()),
                    all_outputs_zero=bool(all(not np.any(value) for value in values.values())))
                if reference is not None:
                    C.require(set(reference) == set(OUTPUT_NAMES), 'Three named reference outputs required')
                    row['versus_reference'] = {name: compare(value, reference[name]) for name, value in values.items()}
                    row['strict_FP32_gate_passed'] = all(record['gate_passed'] for record in row['versus_reference'].values())
                else: row['strict_FP32_gate_passed'] = None
                row['strict_component_gate_passed'] = bool(row['outputs_finite'] and row['weight_domain_valid']
                    and not row['all_outputs_zero'] and row['strict_FP32_gate_passed'] is True)
                row['numerical_status'] = 'strict_component_passed_not_adopted' if row['strict_component_gate_passed'] else 'numerical_rejected_or_reference_missing'
                if self._profile:
                    row['profile'] = self.abi.perf_detail(self.api, self.context)
                self._metrics['completed_calls'] += 1
            except BaseException as error:
                row.update(execution_status='failed', error=repr(error)); self.close()
            row['total_execute_seconds'] = time.monotonic()-start
            row['timing_scope'] = 'forward_seconds includes finite input validation, owned NHWC/CF pack, SDK set/run/get, owned CF output copy and release; total_execute_seconds also includes log handling, numerical/reference checks, profiling or failure cleanup. No per-call source hashing.'
            return values, row

    def close(self):
        with self._lock:
            if self._metrics['closed']: return
            self._metrics['closed'] = True
            try:
                code = int(self.api.rknn_destroy(self.context))
                self._metrics['context_destroyed'] = code == 0
                if code: self._metrics['cleanup_errors'].append('rknn_destroy '+str(code))
                else:
                    self._model_buffer = None
                    self._metrics['current_DMA_bytes'] = self._metrics['current_model_host_bytes'] = 0
            except Exception as error: self._metrics['cleanup_errors'].append(repr(error))
            try:
                self._metrics['source_hashes_after'] = C.recheck_pins(self._pins)
                self._metrics['sources_and_inputs_unchanged'] = True
            except Exception as error:
                self._metrics['sources_and_inputs_unchanged'] = False
                self._metrics['cleanup_errors'].append(repr(error))

    def metrics(self):
        with self._lock: return copy.deepcopy(self._metrics)


class BoundedContextCache:
    """Serialized LRU over exact shape/model contracts; never substitutes a shape.

    Factory(spec) returns one validated session. Resident contexts/model buffers
    are bounded before init by count and declared RKNN bytes+1. Actual DMA is only
    knowable after initialization: it is checked immediately, and old entries
    evicted or the new context rejected. Transient init DMA can exceed resident
    budget and is separately recorded; this is not a process-RSS guarantee.
    """
    def __init__(self, specs, factory, *, max_contexts=4, max_model_bytes=128 << 20, max_dma_bytes=512 << 20):
        C.require(all(type(value) is int and value > 0 for value in (max_contexts, max_model_bytes, max_dma_bytes)), 'Positive explicit cache bounds required')
        C.require(callable(factory) and isinstance(specs, dict) and specs, 'Fixed model specs and factory required')
        self.specs, self.factory = copy.deepcopy(specs), factory
        self.max_contexts, self.max_model_bytes, self.max_dma_bytes = max_contexts, max_model_bytes, max_dma_bytes
        for key, spec in self.specs.items():
            C.require(C.model_key(spec['inputs'], spec['outputs']) == key and type(spec['model_bytes']) is int and spec['model_bytes'] > 0, 'Cache fixed shape/model specification differs')
        self._live, self._history, self._closed, self._lock = OrderedDict(), [], False, threading.RLock()
        self._metrics = dict(created=0, hits=0, evicted=0, peak_resident_contexts=0, peak_resident_model_bytes=0,
            peak_resident_DMA_bytes=0, peak_transient_init_DMA_bytes=0, cleanup_errors=[], candidate_accepted=False)

    def _usage(self):
        metrics = [session.metrics() for session in self._live.values()]
        return sum(row['current_model_host_bytes'] for row in metrics), sum(row['current_DMA_bytes'] for row in metrics)

    def _evict(self, key):
        session = self._live[key]; session.close(); record = session.metrics()
        self._history.append(dict(model_key=key, metrics=record))
        if not record['context_destroyed'] or record['cleanup_errors']:
            self._metrics['cleanup_errors'].extend(record['cleanup_errors'] or ['Context destroy failed'])
            raise RuntimeError('Cache context cleanup failed; no replacement initialized')
        del self._live[key]; self._metrics['evicted'] += 1

    def _get(self, key):
        C.require(not self._closed and not self._metrics['cleanup_errors'], 'Context cache is closed or failed')
        C.require(key in self.specs, 'No compiled context for this actual shape')
        if key in self._live:
            C.require(not self._live[key].metrics()['closed'], 'Failed context cannot be reused')
            self._live.move_to_end(key); self._metrics['hits'] += 1
            return self._live[key]
        spec = self.specs[key]
        estimated = spec['model_bytes']+1
        C.require(estimated <= self.max_model_bytes, 'Individual model host buffer exceeds cache budget')
        while self._live and (len(self._live) >= self.max_contexts or self._usage()[0]+estimated > self.max_model_bytes):
            self._evict(next(iter(self._live)))
        session = self.factory(copy.deepcopy(spec)); self._metrics['created'] += 1
        try:
            record = session.metrics()
            C.require(not record['closed'], 'Factory returned closed context')
            host, dma = record['current_model_host_bytes'], record['current_DMA_bytes']
            self._metrics['peak_transient_init_DMA_bytes'] = max(self._metrics['peak_transient_init_DMA_bytes'], self._usage()[1]+dma)
            C.require(host == estimated and dma <= self.max_dma_bytes, 'Actual context/model allocation exceeds declared budget')
            while self._live and (self._usage()[0]+host > self.max_model_bytes or self._usage()[1]+dma > self.max_dma_bytes):
                self._evict(next(iter(self._live)))
            self._live[key] = session
        except BaseException:
            session.close(); self._history.append(dict(model_key=key, metrics=session.metrics()))
            raise
        host_total, dma_total = self._usage()
        for name, value in (('peak_resident_contexts', len(self._live)), ('peak_resident_model_bytes', host_total), ('peak_resident_DMA_bytes', dma_total)):
            self._metrics[name] = max(self._metrics[name], value)
        return session

    def execute(self, key, feed, reference=None, *, timeout_ms=60000):
        with self._lock:
            begin = time.monotonic()
            session = self._get(key)
            values, record = session.execute(feed, reference, timeout_ms=timeout_ms)
            record['cache_call_seconds'] = time.monotonic()-begin
            record['model_key'] = key
            return values, record

    def close(self):
        with self._lock:
            if self._closed: return
            self._closed = True
            for key, session in list(self._live.items()):
                session.close(); record = session.metrics()
                self._history.append(dict(model_key=key, metrics=record))
                if record['context_destroyed'] and not record['cleanup_errors']: del self._live[key]
                else: self._metrics['cleanup_errors'].extend(record['cleanup_errors'] or ['Context destroy failed'])

    def clear(self):
        with self._lock:
            C.require(not self._closed, 'Cache is closed')
            for key in list(self._live): self._evict(key)

    def metrics(self):
        with self._lock:
            host, dma = self._usage()
            return copy.deepcopy(dict(self._metrics, closed=self._closed, resident_contexts=len(self._live),
                current_model_host_bytes=host, current_DMA_bytes=dma, resident=list(self._live),
                histories=self._history, sessions={key: value.metrics() for key, value in self._live.items()},
                max_contexts=self.max_contexts, max_model_bytes=self.max_model_bytes, max_dma_bytes=self.max_dma_bytes))


def execute_full_graph(adapter, plan, arrays, cache, *, snapshot_identity, current_identity, reference=None, timeout_ms=60000):
    """Sequential actual degree groups; failure/stale identity returns no fullgraph.

    The caller must pin the frozen adapter before import. Numeric FP32 rejection
    returns diagnostic full arrays, never commits them into recurrent VO state.
    All group raw values remain available through GroupExecutionError on failure.
    """
    records = []
    def execute(batch, feeds):
        key = f'B{len(batch.source_slots)}_D{batch.degree}_M{batch.patches}'
        expected = None
        if reference is not None:
            expected = {sdk: adapter.pack_batch(reference[canonical], batch) for sdk, canonical in zip(OUTPUT_NAMES, ('net', 'delta', 'weight'))}
        values, record = cache.execute(key, feeds, expected, timeout_ms=timeout_ms)
        records.append(record)
        if record['execution_status'] != 'completed' or not record['outputs_finite'] or not record['weight_domain_valid'] or record['all_outputs_zero']:
            raise GroupExecutionError(record, values)
        return {canonical: values[sdk] for sdk, canonical in zip(OUTPUT_NAMES, ('net', 'delta', 'weight'))}
    begin = time.monotonic()
    outputs = adapter.execute_grouped_update(plan, *(arrays[key] for key in INPUT_NAMES), execute,
        snapshot_identity=snapshot_identity, current_identity=current_identity)
    elapsed = time.monotonic()-begin
    comparisons = {key: compare(value, reference[key]) for key, value in outputs.items()} if reference is not None else None
    return outputs, dict(candidate_accepted=False, VO_state_committed=False, full_graph_returned=True,
        sequential_groups=len(records), groups=records, full_graph_seconds=elapsed,
        strict_FP32_gate_passed=all(row['gate_passed'] for row in comparisons.values()) if comparisons else None,
        versus_reference=comparisons, padding=False, dropped_edges=False,
        timing_scope='Complete staged graph includes immutable full inputs, validation, packing, LRU initialization/eviction when needed, SDK/log handling, release, and scatter; group/cache times nested. Reference comparisons are outside individual SDK forward_seconds, but inside full_graph_seconds when supplied.')
