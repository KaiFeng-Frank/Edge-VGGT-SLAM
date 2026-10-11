"""Typed FP16 input experiment; SDK conversion remains enabled (pass_through=0).

This does not use zero-copy/native packed memory or change any output path.
Compare actual results against the original SDK FP32 submission before adoption.
"""
import numpy as np


def make_packer(runtime):
    c = runtime.C
    def pack(abi, feed, bindings):
        c.require(set(feed) == set(runtime.INPUT_NAMES) and len(bindings) == 3,
                  'Exactly three original FP32 logical group arrays required')
        c.require([row['index'] for row in bindings] == [0, 1, 2]
                  and {row['name'] for row in bindings} == set(runtime.INPUT_NAMES), 'SDK bindings differ')
        requests = (abi.Input*3)()
        buffers, records = [], []
        for row in bindings:
            source = feed[row['name']]
            c.require(isinstance(source, np.ndarray) and source.dtype == np.float32
                      and source.shape == row['shape'] and np.isfinite(source).all(), 'Original finite CF FP32 source required')
            c.require(row['sdk_type'] == 1 and row['sdk_quantization_type'] in (0, 2)
                      and row['sdk_size_with_stride'] in (0, row['sdk_native_bytes']), 'Unpadded native FP16 descriptor required')
            view = source.transpose(row['input_permutation']) if row['input_permutation'] else source
            with np.errstate(over='ignore', invalid='ignore'):
                owned = np.array(view, dtype=np.float16, order='C', copy=True)
            c.require(np.isfinite(owned).all(), 'Input values overflowed FP16; no SDK call')
            c.require(tuple(owned.shape) == tuple(row['sdk_shape'])
                      and owned.nbytes == row['sdk_native_bytes'] == row['transport_bytes']//2,
                      'Explicit typed FP16 shape/bytes differ')
            requests[row['index']] = abi.Input(row['index'], owned.ctypes.data, owned.nbytes, 0, 1, row['transport_format'])
            buffers.append(owned)
            records.append(dict(index=row['index'], name=row['name'], logical_shape=list(source.shape),
                buffer_shape=list(owned.shape), native_attribute_format=row['sdk_format'], request_format=row['transport_format'],
                source_dtype='float32', transport_dtype='float16', pass_through=0, bytes=owned.nbytes, owned=True,
                explicit_NHWC_pack=row['input_permutation'] is not None,
                note='NumPy casts original FP32 to FP16 before SDK; identity equivalence must be measured.'))
        return runtime.PackedInputs(requests, tuple(buffers), tuple(records))
    return pack
