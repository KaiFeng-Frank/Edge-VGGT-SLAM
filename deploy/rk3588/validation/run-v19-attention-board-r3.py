#!/usr/bin/env python3
"""R3 diagnostic: explicitly pack FP32 Q/K/V as SDK-native C-order NHWC.

R2's NCHW normalizer returned API success while stderr reported unsupported
source layout and all outputs were zero. This variant does the transpose/copy
itself and submits fmt=NHWC, pass_through=0. Original output remains NCHW.
The frozen round also prepares NCHW slices: both copies are honestly included
in whole-round time. This diagnoses transport, not production performance.
"""
from __future__ import annotations

import ctypes as ct
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import sys
import threading
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
R2_NAME = 'run-v19-attention-board-r2.py'
R2_SHA = 'bcf1f45d10891e7df81e878f26e2e5ad0ff5eabf726a306e177dbc1ca205509d'
INPUT_NAMES = ('query', 'key', 'value')
INPUT_API_FORMAT = 1  # Actual supplied buffer is now NHWC.


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            digest.update(part)
    return digest.hexdigest()


def pack_nhwc(inputs, logical_shape):
    import numpy as np
    if set(inputs) != set(INPUT_NAMES):
        raise ValueError('Exactly original logical query/key/value buffers required')
    for value in inputs.values():
        if value.dtype != np.float32 or tuple(value.shape) != tuple(logical_shape) or not value.flags.c_contiguous:
            raise ValueError('Original C-contiguous FP32 NCHW logical shape required')
    return {name: np.array(inputs[name].transpose(0, 2, 3, 1), dtype=np.float32, order='C', copy=True)
            for name in INPUT_NAMES}


def execute_call(api, runtime, session, inputs, timeout_ms, row, barrier=None):
    import numpy as np
    spec, context = session['spec'], session['context']
    tick = time.perf_counter()
    row['nhwc_pack_begin_monotonic'] = time.monotonic()
    native = pack_nhwc(inputs, spec['shape'])
    row.update(nhwc_packing_seconds=time.perf_counter()-tick, nhwc_pack_end_monotonic=time.monotonic(),
               packed_native_input_bytes=sum(value.nbytes for value in native.values()),
               native_input_shapes={name: list(value.shape) for name, value in native.items()},
               input_native_formats=[int(attr.fmt) for attr in session['attrs']],
               input_api_format=INPUT_API_FORMAT, input_buffer_order='C-order NHWC',
               input_layout_conversion_requested=False, input_type='FP32', input_pass_through=0,
               want_float=1, is_prealloc=0, synchronous=True, run_timeout_ms=timeout_ms,
               worker_thread_id=threading.get_native_id(), worker_cpu_affinity=sorted(os.sched_getaffinity(0)),
               worker_enter_monotonic=time.monotonic())
    request = (runtime.Input * 3)()
    for i, name in enumerate(INPUT_NAMES):
        value = native[name]
        request[i] = runtime.Input(i, value.ctypes.data, value.nbytes, 0, 0, INPUT_API_FORMAT)
    outputs = (runtime.Output * 1)()
    outputs[0].want_float, outputs[0].is_prealloc, outputs[0].index = 1, 0, 0
    run = runtime.RunExtend(0, 0, timeout_ms, -1)
    received = runtime.OutputExtend()
    if barrier is not None:
        barrier.wait(timeout=10.)
    begin = time.perf_counter()
    row['io_begin_monotonic'] = time.monotonic()
    got = False
    try:
        runtime.check_call(api.rknn_inputs_set(context, 3, request), 'rknn_inputs_set')
        set_end = time.perf_counter()
        row.update(run_api_called=True, run_begin_monotonic=time.monotonic())
        code = api.rknn_run(context, ct.byref(run))
        row['run_end_monotonic'] = time.monotonic()
        runtime.check_call(code, 'rknn_run')
        row['run_api_succeeded'] = True
        run_end = time.perf_counter()
        runtime.check_call(api.rknn_outputs_get(context, 1, outputs, ct.byref(received)), 'rknn_outputs_get')
        got = True
        get_end = time.perf_counter()
        if run.frame_id != received.frame_id:
            raise ValueError('Synchronous run/output frame identity mismatch')
        if not outputs[0].buf or outputs[0].size != math.prod(spec['shape'])*4:
            raise ValueError('Invalid original NCHW FP32 output pointer/byte size')
        # Output fmt=0 was strictly validated; do not transpose its byte order.
        view = np.ctypeslib.as_array(ct.cast(outputs[0].buf, ct.POINTER(ct.c_float)), shape=(math.prod(spec['shape']),))
        actual = view.copy().reshape(spec['shape'])
        copy_end = time.perf_counter()
        row.update(input_set_seconds=set_end-begin, run_api_seconds=run_end-set_end,
                   outputs_get_seconds=get_end-run_end, output_copy_seconds=copy_end-get_end,
                   runtime_frame_id=int(run.frame_id), received_frame_id=int(received.frame_id),
                   output_bytes=int(outputs[0].size), output_copied_before_release=True)
    except Exception as exc:
        row['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if got or outputs[0].buf:
            tick = time.perf_counter()
            code = api.rknn_outputs_release(context, 1, outputs)
            row.update(outputs_release_seconds=time.perf_counter()-tick,
                       output_buffer_released=code == 0, outputs_release_code=int(code))
            runtime.check_call(code, 'rknn_outputs_release')
        row.update(io_end_monotonic=time.monotonic(), io_total_seconds=time.perf_counter()-begin)
    return actual


def load_base():
    path = HERE / R2_NAME
    if sha(path) != R2_SHA:
        raise ValueError('Frozen R2 adapter SHA256 mismatch')
    spec = importlib.util.spec_from_file_location('_v19_attention_r3_frozen_r2', path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    base = adapter.load_base()
    validate, create, load_inputs, round_call, compare = (
        base.validate_attr, base.create_session, base.load_inputs, base.execute_round, base.comparison)

    def validate_attr(runtime, attr, shape, name, index):
        if name in INPUT_NAMES and int(attr.fmt) != 1:
            raise ValueError('This diagnostic implements SDK-native NHWC inputs only')
        return validate(runtime, attr, shape, name, index)

    def create_session(api, runtime, spec, flags, sessions, report):
        item = create(api, runtime, spec, flags, sessions, report)
        report['layout_policy'].update(
            entrypoint_sha256=sha(__file__), R2_adapter_sha256=R2_SHA,
            logical_buffer_order='Original capture NCHW; explicit transpose(0,2,3,1)/FP32 C-order NHWC copy',
            input_api_format=INPUT_API_FORMAT, input_pass_through=0,
            extra_NHWC_packing_in_whole_round_time=True,
            R2_failure_reason='SDK stderr unsupported NCHW source normalizer, despite inputs_set return0; not precision drift')
        return item

    def inputs_with_pins(*args, **kwargs):
        result = load_inputs(*args, **kwargs)
        result[-1].update({str(path): R2_SHA, str(Path(__file__).resolve()): sha(__file__)})
        return result

    def execute_round(*args, **kwargs):
        group = args[6] if len(args) > 6 else kwargs['group']
        try:
            return round_call(*args, **kwargs)
        finally:
            group.update(additional_NHWC_packing_included_in_end_to_end=True,
                         nhwc_worker_packing_seconds_sum=sum(row.get('nhwc_packing_seconds', 0.)
                                                              for row in group.get('calls', [])),
                         preparation_cost='Original NCHW slice copies plus worker NHWC transpose/copies; both included')

    def comparison(runtime, actual, expected):
        import numpy as np
        row = compare(runtime, actual, expected)
        row.update(output_all_zero=bool(not np.any(actual)), reference_all_zero=bool(not np.any(expected)))
        return row

    base.validate_attr, base.create_session, base.load_inputs = validate_attr, create_session, inputs_with_pins
    base.execute_call, base.execute_round, base.comparison = execute_call, execute_round, comparison
    base.__file__ = str(Path(__file__).resolve())
    return base


def main(argv=None):
    return load_base().main(argv)


if __name__ == '__main__':
    sys.exit(main())
