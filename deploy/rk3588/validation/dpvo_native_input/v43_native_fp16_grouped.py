"""Explicit research-only native FP16 input transport for frozen DPVO Update.

The isolated v37 executor retains synchronous output get/copy/release, locks,
logs, counters, bounded contexts and complete-edge scatter. Only its private
pack_inputs binding changes. Each context queries commands 8 and 10 before any
forward; both must actually return the measured NC1HWC2 C2=8 layout. No Torch,
ORT, model, SDK or packing library is loaded on import. No candidate adoption.
DPVO attribution: Copyright (c) 2022 Princeton University, MIT.
"""
import copy
import ctypes as ct
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import threading
import types
import uuid

import numpy as np

BRIDGE_SHA256 = 'd4cdd1f9f21b04e44b6369c8ad7aea1fe4fbd43855b13993c6fdf4550f7bf0d7'
PARENT_RUNTIME_SHA256 = '55d7b04e782a687fedb6475b8388a12d52361cc56b1a7f260537bd3e1d679d1d'
PARENT_CONTRACTS_SHA256 = 'aa9e6af938f64ae4a4a82ffc84b487321fff208547703b0d65646c72b96d85c9'
NATIVE_QUERY_SHA256 = '89f0cead99df024b25ed3d2638e412a075f91d6736ecdb2b35ace2304eef0642'
PACKER_SHA256 = 'b3d2503d1bb27ec0352cfe54d759efdcee070764ad266841847005fa9b13f8ce'
PACK_C_SHA256 = '78b46cb0abb2a32b2a8da2d4a15beccab725cc5b897297304699ec3bb535443f'
PACK_H_SHA256 = 'e2fe575b74908d73869fc4c6d2dd00294a0f176f89b5a28cb7af2fcdfc9011dc'
NAMES = ('net', 'inp', 'corr')
MODEL_KEYS = frozenset([f'B{b}_D{d}_M16' for b in (1, 2) for d in range(4, 8)] + ['B1_D3_M16'])
TRANSPORT = 'native_FP16_NC1HWC2_C2_8_pass_through_1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _pin(path, expected, pins):
    path = Path(path).resolve(strict=True)
    require(isinstance(expected, str) and len(expected) == 64
            and all(c in '0123456789abcdef' for c in expected)
            and sha256(path) == expected, 'Exact native transport source SHA differs: ' + path.name)
    require(str(path) not in pins or pins[str(path)] == expected, 'Conflicting native source pin')
    pins[str(path)] = expected
    return path


def _load(path, expected, label):
    path = _pin(path, expected, {})
    name = '_v43_private_' + label + '_' + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def native_attribute_record(attr):
    """Record the queried ABI fields before validating them; never guess layout."""
    rank, scale = int(attr.n_dims), float(attr.scale)
    return dict(index=int(attr.index), n_dims=rank, dims=list(attr.dims),
        shape=list(attr.dims[:rank]) if 1 <= rank <= 16 else None,
        name=bytes(attr.name).decode('utf-8', errors='replace'), n_elems=int(attr.n_elems),
        size=int(attr.size), fmt=int(attr.fmt), type=int(attr.type), qnt_type=int(attr.qnt_type),
        fl=int(attr.fl), zp=int(attr.zp), scale=scale if math.isfinite(scale) else repr(scale),
        scale_finite=math.isfinite(scale), w_stride=int(attr.w_stride), h_stride=int(attr.h_stride),
        size_with_stride=int(attr.size_with_stride), pass_through=int(attr.pass_through))


def validate_native_pair(binding, records, expected_records=None):
    """Narrow measured contract; command10's name does not imply its result fmt."""
    shape = tuple(binding['shape'])
    require(len(shape) == 4 and all(type(v) is int for v in shape), 'Fixed logical CF rank4 required')
    b, c, d, m = shape
    require(b in (1, 2) and c in (384, 882) and d in range(3, 8) and m == 16,
            'Only measured original M16 dimensions supported')
    require((binding['index'], binding['name'], c) in ((0, 'net', 384), (1, 'inp', 384), (2, 'corr', 882)),
            'Native name/index/channel binding differs')
    native_shape = [b, (c + 7) // 8, d, m, 8]
    count = math.prod(native_shape)
    fields = dict(index=binding['index'], name=binding['name'], n_dims=5,
        dims=native_shape + [0]*11, shape=native_shape, n_elems=count, size=count*2,
        fmt=2, type=1, qnt_type=2, fl=0, zp=0, scale=1.0, scale_finite=True,
        w_stride=16, h_stride=0, size_with_stride=count*2, pass_through=0)
    require(len(records) == 2 and [v['command'] for v in records] == [8, 10],
            'Both actual native input commands required')
    for record in records:
        require(record['input_index'] == binding['index'] and type(record['return_code']) is int
                and record['return_code'] == 0, 'Native query failed or index differs')
        attr = record['attribute']
        for name, value in fields.items():
            require(attr.get(name) == value and (type(attr[name]) is type(value)),
                    'Native FP16 NC1HWC2 attribute differs: ' + name)
    require(all(records[0]['attribute'][k] == records[1]['attribute'][k] for k in fields),
            'Native commands 8 and 10 disagree')
    if expected_records is not None:
        require(len(expected_records) == 2, 'Measured ancestor queries missing')
        for actual, expected in zip(records, expected_records):
            require(actual['command'] == expected['command'] and expected['return_code'] == 0
                    and all(actual['attribute'][k] == expected['attribute'][k] for k in fields),
                    'Actual context native attrs differ from frozen model query')
    return copy.deepcopy(records[0]['attribute'])


def validate_query_ancestor(report):
    require(report['schema'] == 'v42_nine_shape_logical_and_native_input_query'
            and report['status'] == 'queried_all_nine_shapes_inputs_not_inference_or_transport_validated'
            and report['candidate_accepted'] is False and report['transport_equivalence_validated'] is False
            and report['sdk_run_calls'] == 0 and report['model_inference_executed'] is False
            and report['cleanup_errors'] == [] and report['sources_and_inputs_unchanged'] is True,
            'Frozen query-only ancestry required, not transport acceptance')
    models = report['models']
    require(len(models) == 9 and {v['model_key'] for v in models} == MODEL_KEYS,
            'Exact nine queried model keys required')
    result = {}
    for row in models:
        require(row['initialized'] is True and row['context_destroyed'] is True
                and row['all_input_attr_queries_succeeded'] is True and row['sdk_run_calls'] == 0
                and row['cleanup_errors'] == [], 'Ancestor context query/lifecycle failed')
        require([v['name'] for v in row['logical_inputs']] == list(NAMES), 'Ancestor input order differs')
        for index, logical in enumerate(row['logical_inputs']):
            records = [v for v in row['queries'] if v['command'] in (8, 10) and v['input_index'] == index]
            validate_native_pair(dict(logical, index=index), records)
        result[row['model_key']] = copy.deepcopy(row)
    return result


def _bind_transport(runtime, packer, library_provider, source_pins, query_models):
    """Internal composition; fake sessions use generated SDKs, never NPU flags."""
    parent, c = runtime.parent, runtime.C
    require(parent.C is c and parent.INPUT_NAMES == NAMES, 'An isolated original runtime binding required')
    original = runtime.GroupedSession
    library_lock, library_owner = threading.Lock(), []
    def library():
        with library_lock:
            if not library_owner:
                library_owner.append(library_provider())
            return library_owner[0]

    def pack_inputs(abi, feed, bindings):
        require(set(feed) == set(NAMES) and len(bindings) == 3
                and [v['index'] for v in bindings] == [0, 1, 2], 'Exact three native input bindings required')
        requests = (abi.Input*3)()
        buffers, records = [], []
        for binding in bindings:
            require(binding.get('native_query_commands') == [8, 10]
                    and binding.get('input_transport') == TRANSPORT, 'Native context not queried/validated')
            value = feed[binding['name']]
            require(isinstance(value, np.ndarray) and value.dtype == np.float32
                    and value.shape == binding['shape'], 'Original FP32 CF input shape/dtype differs')
            # Pinned packer checks contiguous/aligned CF input and scans finite/
            # FP16 overflow in the C conversion itself; no duplicate array scan.
            owned = packer.pack_blocked_fp16(value, binding['native_input_attribute'], library())
            require(isinstance(owned, np.ndarray) and owned.dtype == np.uint16
                    and owned.flags.c_contiguous and owned.flags.owndata and owned.flags.aligned
                    and owned.shape == tuple(binding['native_input_attribute']['shape'])
                    and owned.nbytes == binding['native_input_attribute']['size_with_stride'],
                    'Owned native FP16 bits/layout/bytes differ')
            requests[binding['index']] = abi.Input(binding['index'], owned.ctypes.data, owned.nbytes, 1, 1, 2)
            buffers.append(owned)
            records.append(dict(index=binding['index'], name=binding['name'], logical_shape=list(value.shape),
                buffer_shape=list(owned.shape), logical_attribute_format=binding['sdk_format'],
                native_attribute_format=2, native_query_commands=[8, 10], request_format=2,
                request_type=1, source_dtype='float32', transport_dtype='float16', buffer_storage_dtype='uint16',
                pass_through=1, bytes=owned.nbytes, owned=True, input_transport=TRANSPORT,
                padded_channels=owned.shape[1]*8-value.shape[1], tail_padding_value='FP16_positive_zero',
                SDK_input_conversion_requested=False))
        return parent.PackedInputs(requests, tuple(buffers), tuple(records))

    class GroupedSession(original):
        def configure_native_inputs(self, model_key):
            """Query this exact context; initialization failures retain raw logs."""
            with self._lock:
                require(not self._metrics['closed'] and self._metrics['sdk_run_api_calls'] == 0,
                        'Native inputs must be configured before any forward')
                require(model_key in query_models, 'No frozen native query for this exact model')
                require('native_input_queries' not in self._metrics, 'Native initialization already attempted')
                queries = []
                self._metrics.update(native_input_queries=queries, native_inputs_validated=False,
                    input_transport=TRANSPORT, native_query_ancestor_sha256=NATIVE_QUERY_SHA256,
                    diagnostic_allow_native_fp16=True, transport_equivalence_validated=False,
                    candidate_accepted=False)
                def query():
                    for command in (8, 10):
                        for index in range(3):
                            attr = self.abi.TensorAttr(index=index)
                            code = int(self.api.rknn_query(self.context, command, ct.byref(attr), ct.sizeof(attr)))
                            queries.append(dict(command=command, input_index=index, return_code=code,
                                                attribute=native_attribute_record(attr)))
                            self.abi.check_call(code, 'rknn_query_native_input_' + str(command))
                try:
                    if self._log is not None:
                        try:
                            with self._log._native_log(self._log_directory) as native:
                                query()
                        finally:
                            self._metrics['native_input_query_log'] = native
                        require(not native['truncated'] and not native['RKNN_error_detected'], 'Native attr query emitted E RKNN')
                    else:
                        require(not self._hardware, 'Hardware native query log capture required')
                        query()
                    measured = query_models[model_key]
                    for binding in self.input_bindings:
                        rows = [v for v in queries if v['input_index'] == binding['index']]
                        expected = [v for v in measured['queries'] if v['command'] in (8, 10)
                                    and v['input_index'] == binding['index']]
                        attribute = validate_native_pair(binding, rows, expected)
                        binding.update(native_input_attribute=attribute, native_query_commands=[8, 10],
                            input_transport=TRANSPORT, transport_dtype='float16', transport_format=2,
                            transport_bytes=attribute['size'], pass_through=1)
                    fpcr = packer.fpcr_record(library())
                    require(fpcr['supported'] is True and fpcr['FPCR_modified_by_packer'] is False,
                            'Packing FPCR state unsupported; environment is not changed')
                    self._metrics.update(native_inputs_validated=True, input_bindings=copy.deepcopy(self.input_bindings),
                                         packing_FPCR_at_initialization=copy.deepcopy(fpcr))
                except BaseException as error:
                    self._metrics['native_input_configuration_error'] = repr(error)
                    self.close()
                    raise

        @classmethod
        def open_board(cls, model_path, model_sha256, inputs, outputs, *,
                       diagnostic_allow_native_fp16=False, pins=None, **kwargs):
            require(diagnostic_allow_native_fp16 is True, 'Explicit diagnostic_allow_native_fp16 required')
            key = c.model_key(inputs, outputs)
            require(key in query_models and query_models[key]['model']['sha256'] == model_sha256
                    and query_models[key]['logical_inputs'] == inputs and query_models[key]['logical_outputs'] == outputs,
                    'Exact measured compiled model/input/output ancestry required')
            sources = dict(pins or {})
            for path, expected in source_pins.items():
                c.pin(path, expected, sources)
            c.recheck_pins(sources)
            session = super().open_board(model_path, model_sha256, inputs, outputs, pins=sources, **kwargs)
            try:
                session.configure_native_inputs(key)
                return session
            except BaseException as error:
                session.close()
                raise parent.StartupError(dict(error=repr(error), candidate_accepted=False,
                    input_transport=TRANSPORT, native_input_queries=session.metrics().get('native_input_queries', []),
                    cleanup=session.metrics())) from error

        def execute(self, *args, **kwargs):
            values, record = super().execute(*args, **kwargs)
            record.update(input_transport=TRANSPORT, diagnostic_allow_native_fp16=True,
                          transport_equivalence_validated=False, candidate_accepted=False)
            record['timing_scope'] = ('forward_seconds includes native attribute/input validation, owned NC1HWC2 FP16 '
                'allocation/packing, SDK set/run/get, original owned CF FP32 output copy and release. '
                'total_execute_seconds also includes native logs, numerical/reference checks, profile or failure cleanup. '
                'Native input query occurs once during context initialization; no per-call source hashing.')
            return values, record

    parent.pack_inputs = pack_inputs  # Only this private ancestor instance changes.
    return types.SimpleNamespace(C=c, GroupedSession=GroupedSession,
        BoundedContextCache=runtime.BoundedContextCache, execute_full_graph=runtime.execute_full_graph,
        pack_inputs=pack_inputs, compare=runtime.compare, StartupError=runtime.StartupError,
        GroupExecutionError=runtime.GroupExecutionError, parent=parent,
        source_pins=dict(source_pins), native_query_ancestor_sha256=NATIVE_QUERY_SHA256,
        input_transport=TRANSPORT, candidate_accepted=False)


def load_runtime(parent_runtime, contracts_path, contracts_sha256, parent_contracts, *,
        runtime_bridge, packer_path, packer_sha256, packing_library_path, packing_library_sha256,
        packing_source_pins, native_query_receipt, native_query_sha256, diagnostic_allow_native_fp16=False):
    """Compose a private runtime. Loading performs hash/metadata checks only.

    The original bridge initializes the requested pinned contract dependencies;
    they must be staged exactly as before. Packing CDLL is loaded only after the
    original board/SDK initialization guard. Both loader and open_board require
    the explicit diagnostic flag. Existing compiled model bytes are unchanged.
    """
    require(diagnostic_allow_native_fp16 is True, 'Explicit diagnostic_allow_native_fp16 required')
    require(native_query_sha256 == NATIVE_QUERY_SHA256 and packer_sha256 == PACKER_SHA256,
            'Fixed native query and packer ancestor pins required')
    pins = dict(packing_source_pins)
    _pin(__file__, sha256(__file__), pins)
    bridge_path = _pin(runtime_bridge, BRIDGE_SHA256, pins)
    _pin(parent_runtime, PARENT_RUNTIME_SHA256, pins)
    _pin(parent_contracts, PARENT_CONTRACTS_SHA256, pins)
    _pin(contracts_path, contracts_sha256, pins)
    packer_path = _pin(packer_path, PACKER_SHA256, pins)
    _pin(packer_path.with_suffix('.c'), PACK_C_SHA256, pins)
    _pin(packer_path.with_suffix('.h'), PACK_H_SHA256, pins)
    library_path = _pin(packing_library_path, packing_library_sha256, pins)
    query_path = _pin(native_query_receipt, NATIVE_QUERY_SHA256, pins)
    for path, expected in list(pins.items()):
        _pin(path, expected, pins)
    query_models = validate_query_ancestor(json.loads(query_path.read_text()))
    bridge = _load(bridge_path, BRIDGE_SHA256, 'bridge')
    runtime = bridge.load_runtime(parent_runtime, contracts_path, contracts_sha256, parent_contracts)
    pins.update(runtime.source_pins)
    packer = _load(packer_path, PACKER_SHA256, 'packer')
    require(packer.SOURCE_QUERY_SHA256 == NATIVE_QUERY_SHA256, 'Packer queried-layout provenance differs')
    return _bind_transport(runtime, packer,
        lambda: packer.load_native_library(library_path, packing_library_sha256), pins, query_models)
