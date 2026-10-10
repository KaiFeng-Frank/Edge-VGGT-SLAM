#!/usr/bin/env python3
"""R2 attention probe: NCHW user buffers with explicit SDK layout conversion.

Reuses the frozen R1 synchronous ABI, timing, concurrency and cleanup. Only
input fmt=NHWC maps the logical [N,C,H,W] shape to SDK [N,H,W,C]. All submitted
FP32 buffers stay C-order NCHW and use rknn_input.fmt=NCHW, pass_through=0.
The only implemented output layout is NCHW in the original logical shape.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
R1_NAME = 'run-v19-attention-board.py'
R1_SHA = '277d7f22154ac2c187710a121aee94ead1d416c920c2d9fa023e71f257472ebc'
INPUT_NAMES = ('query', 'key', 'value')
INPUT_API_FORMAT = 0  # RKNN_TENSOR_NCHW, describes the actual user buffer.


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            digest.update(part)
    return digest.hexdigest()


def descriptor_shape(logical_shape, native_format, is_input):
    logical = tuple(logical_shape)
    if len(logical) != 4 or any(type(n) is not int or n <= 0 for n in logical):
        raise ValueError('Static four-dimensional NCHW logical shape required')
    if native_format == 0:
        return logical
    if is_input and native_format == 1:
        n, c, h, w = logical
        return n, h, w, c
    raise ValueError('Unimplemented SDK input/output tensor layout')


def load_base():
    path = HERE / R1_NAME
    if sha(path) != R1_SHA:
        raise ValueError('Frozen R1 implementation SHA256 mismatch')
    spec = importlib.util.spec_from_file_location('_v19_attention_r2_frozen_r1', path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    original_validate = base.validate_attr
    original_execute = base.execute_call
    original_create = base.create_session
    original_load_inputs = base.load_inputs

    def validate_attr(runtime, attr, shape, name, index):
        if name not in (*INPUT_NAMES, 'output'):
            raise ValueError('Exact Q/K/V/output tensor names required')
        declared_shape = descriptor_shape(shape, int(attr.fmt), name in INPUT_NAMES)
        return original_validate(runtime, attr, declared_shape, name, index)

    def create_session(api, runtime, spec, flags, sessions, report):
        report['layout_policy'] = dict(
            entrypoint_sha256=sha(__file__), R1_implementation_sha256=R1_SHA,
            header_sha256=runtime.HEADER_SHA256, logical_buffer_order='C-order NCHW [N,C,H,W]',
            input_api_format=INPUT_API_FORMAT, input_pass_through=0,
            native_NHWC_input_shape_mapping='[N,C,H,W] -> [N,H,W,C]',
            output_layout='NCHW only; no unsupported output permutation',
            API_basis='rknn_input.fmt describes input buf; pass_through=0 converts according to type/fmt')
        return original_create(api, runtime, spec, flags, sessions, report)

    def execute_call(api, runtime, session, inputs, timeout_ms, row, barrier=None):
        # Preserve the SDK descriptors in session/report. R1 reads only .fmt
        # from this separate transport view when constructing rknn_input.
        row.update(input_native_formats=[int(attr.fmt) for attr in session['attrs']],
                   input_api_format=INPUT_API_FORMAT, input_buffer_order='C-order NCHW',
                   input_layout_conversion_requested=any(int(attr.fmt) == 1 for attr in session['attrs']))
        transport = dict(session, attrs=[SimpleNamespace(fmt=INPUT_API_FORMAT) for _ in INPUT_NAMES])
        return original_execute(api, runtime, transport, inputs, timeout_ms, row, barrier)

    def load_inputs(*args, **kwargs):
        capture, exported, flat, specs, pins = original_load_inputs(*args, **kwargs)
        # Both implementations participate in the before/after hash chain.
        pins[str(path)] = R1_SHA
        pins[str(Path(__file__).resolve())] = sha(__file__)
        return capture, exported, flat, specs, pins

    base.validate_attr, base.create_session = validate_attr, create_session
    base.execute_call, base.load_inputs = execute_call, load_inputs
    # R1 main computes entrypoint script_sha256 and input pins from __file__.
    # Keep HERE untouched so frozen ABI/export/compiler helpers retain their paths.
    base.__file__ = str(Path(__file__).resolve())
    return base


def main(argv=None):
    return load_base().main(argv)


if __name__ == '__main__':
    sys.exit(main())
