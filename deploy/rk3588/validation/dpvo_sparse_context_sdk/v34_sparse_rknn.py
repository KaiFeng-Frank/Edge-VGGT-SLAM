"""Exact typed SDK transport and owned outputs; NumPy/stdlib only at import.

No model loading or board execution occurs on import. DPVO reference is MIT.
"""
import ctypes as ct
import math
import os
import sys
import time

import numpy as np

SPARSE_OUTPUTS = (("fmap", (1, 128, 116, 152)), ("context", (1, 96, 384)))


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_index_attr(row, count=96):
    require(row["index"] == 1 and row["name"] == "patch_indices" and row["shape"] == [count], "Exact ordered index port required")
    require(row["type"] == 8, "Initial version requires actual INT64 index attribute; no implicit INT32/float fallback")
    require(row["n_elems"] == count and row["size"] == count * 8 and row["format"] in (0, 1, 3), "Exact INT64 index elements/bytes/format; fmt3 is UNDEFINED")
    require(row["quantization_type"] == 0 or row["quantization_type"] == 2 and row["zero_point"] == 0 and row["scale"] == 1.0,
            "Unquantized or exact neutral integer descriptor required")
    return row


def validate_context_attr(row, shape=(1, 96, 384)):
    require(row["index"] == 1 and row["name"] == "context" and row["shape"] == list(shape) and row["format"] in (0, 3),
            "Exact context dimensions and unambiguous logical layout required")
    require(row["type"] in (0, 1) and row["n_elems"] == math.prod(shape) and
            row["size"] == row["n_elems"] * (4 if row["type"] == 0 else 2), "FP32/FP16 context elements/bytes")
    require(row["quantization_type"] == 0 or row["type"] == 1 and row["quantization_type"] == 2 and
            row["zero_point"] == 0 and row["scale"] == 1.0, "Unquantized or identity-affine FP16 context descriptor required")
    return row


def validate_indices(indices, count=96, hw=(116, 152)):
    require(isinstance(indices, np.ndarray) and indices.dtype == np.int64 and indices.shape == (count,), "Source indices remain INT64")
    require(np.all(indices >= 0) and np.all(indices < math.prod(hw)), "Index outside original dense feature domain")
    y, x = np.divmod(indices, hw[1])
    require(np.all(x >= 1) and np.all(x <= hw[1] - 2) and np.all(y >= 1) and np.all(y <= hw[0] - 2), "Original patch-center interior required")


def execute_sparse(api, abi, base, context, images, indices, image_format, index_attr,
                   cleanup, activity, *, outputs=SPARSE_OUTPUTS, timeout_ms=60000):
    validate_index_attr(index_attr, count=indices.size)
    validate_indices(indices, count=index_attr["n_elems"], hw=(images.shape[-2] // 4, images.shape[-1] // 4))
    tick = time.perf_counter()
    image = base.pack_input(images, image_format)
    integer = np.array(indices, dtype=np.int64, copy=True, order="C")
    packed_at = time.perf_counter()
    requests = (abi.Input * 2)(abi.Input(0, image.ctypes.data, image.nbytes, 0, 0, image_format),
        abi.Input(1, integer.ctypes.data, integer.nbytes, 0, 8, index_attr["format"]))
    buffers = (abi.Output * len(outputs))()
    for i, row in enumerate(buffers):
        row.index, row.want_float, row.is_prealloc = i, 1, 0
    extension, received = abi.RunExtend(0, 0, timeout_ms, -1), abi.OutputExtend()
    timing = dict(input_packing_seconds=packed_at - tick, input_api_types=[0, 8],
        input_api_formats=[image_format, index_attr["format"]], input_pass_through=[0, 0],
        input_logical_dtypes=["float32", "int64"], input_buffer_shapes=[list(image.shape), list(integer.shape)],
        input_buffer_bytes=[image.nbytes, integer.nbytes], output_want_float=1, output_is_prealloc=0,
        output_size_policy="FP32 logical bytes only; never size_with_stride", synchronous=True)
    activity["last_sdk_attempt"] = timing
    got, actual = False, {}
    try:
        abi.check_call(api.rknn_inputs_set(context, 2, requests), "rknn_inputs_set")
        set_at = time.perf_counter()
        activity["sdk_run_api_calls"] = activity.get("sdk_run_api_calls", 0) + 1
        abi.check_call(api.rknn_run(context, ct.byref(extension)), "rknn_run")
        run_at = time.perf_counter()
        abi.check_call(api.rknn_outputs_get(context, len(buffers), buffers, ct.byref(received)), "rknn_outputs_get")
        got = True
        get_at = time.perf_counter()
        require(extension.frame_id == received.frame_id, "SDK run/get frame IDs disagree")
        for row, (name, shape) in zip(buffers, outputs):
            elements = math.prod(shape)
            require(bool(row.buf) and row.size == elements * 4, "Returned SDK FP32 logical byte count differs: " + name)
            raw = np.ctypeslib.as_array(ct.cast(row.buf, ct.POINTER(ct.c_float)), shape=(elements,))
            actual[name] = raw.copy().reshape(shape)
        copied_at = time.perf_counter()
        timing.update(input_set_seconds=set_at - packed_at, run_api_seconds=run_at - set_at,
            outputs_get_seconds=get_at - run_at, output_copy_seconds=copied_at - get_at,
            output_bytes=[int(row.size) for row in buffers], runtime_frame_id=int(extension.frame_id))
    finally:
        if got or any(row.buf for row in buffers):
            primary = sys.exc_info()[0] is not None
            release_at = time.perf_counter()
            try:
                code = api.rknn_outputs_release(context, len(buffers), buffers)
                timing.update(outputs_release_seconds=time.perf_counter() - release_at,
                    output_buffers_released=code == 0, sdk_io_seconds=time.perf_counter() - tick)
                if code != 0:
                    cleanup.append("rknn_outputs_release returned " + str(code))
                    if not primary:
                        abi.check_call(code, "rknn_outputs_release")
            except Exception as exc:
                if not cleanup or cleanup[-1] != "rknn_outputs_release returned " + str(locals().get("code")):
                    cleanup.append("rknn_outputs_release: " + repr(exc))
                if not primary:
                    raise
    return actual, timing


class Context:
    def __init__(self, base, abi, api, model, sparse, profile, cleanup, record):
        self.base, self.abi, self.api = base, abi, api
        self.sparse, self.cleanup, self.record = sparse, cleanup, record
        self.context, self.closed = abi.CONTEXT(), False
        try:
            filename = ct.create_string_buffer(os.fsencode(model))
            begin = time.perf_counter()
            abi.check_call(api.rknn_init(ct.byref(self.context), ct.cast(filename, ct.c_void_p), 0, 8 if profile else 0, None), "rknn_init")
            record["init_seconds"] = time.perf_counter() - begin
            abi.check_call(api.rknn_set_core_mask(self.context, 7), "rknn_set_core_mask")
            version = abi.query(api, self.context, 5, abi.SDKVersion())
            record.update(sdk_version=bytes(version.api_version).decode(errors="replace"), core_mask=7)
            require(record["sdk_version"].startswith("2.3.2"), "RKNN SDK 2.3.2 required")
            count = abi.query(api, self.context, 0, abi.InputOutputNum())
            require(count.n_input == (2 if sparse else 1) and count.n_output == 2, "Exact SDK input/output count required")
            input_attrs = [abi.query(api, self.context, 1, abi.TensorAttr(index=i)) for i in range(count.n_input)]
            output_attrs = [abi.query(api, self.context, 2, abi.TensorAttr(index=i)) for i in range(count.n_output)]
            # Keep all actual query records even when a guard rejects their dtype/layout.
            record["input_attrs"] = [abi.attr_record(a) for a in input_attrs]
            record["output_attrs"] = [abi.attr_record(a) for a in output_attrs]
            base.validate_attribute(abi, input_attrs[0], 0, "normalized_bgr", base.INPUT_SHAPE, is_input=True)
            base.validate_attribute(abi, output_attrs[0], 0, "fmap", base.OUTPUTS[0][1], is_input=False)
            if sparse:
                self.index_attr = validate_index_attr(record["input_attrs"][1])
                validate_context_attr(record["output_attrs"][1])
            else:
                base.validate_attribute(abi, output_attrs[1], 1, "imap", base.OUTPUTS[1][1], is_input=False)
            self.image_format = int(input_attrs[0].fmt)
            memory = abi.query(api, self.context, 6, abi.MemSize())
            record["model_memory_bytes"] = {name: int(getattr(memory, name)) for name, _ in memory._fields_ if name != "reserved"}
            require(memory.total_dma_allocated_size <= 1024**3, "Context DMA exceeds isolated 1GiB budget")
        except Exception:
            self.close()
            raise

    def execute(self, image, indices, activity):
        require(not self.closed, "Closed RKNN context")
        if self.sparse:
            return execute_sparse(self.api, self.abi, self.base, self.context, image, indices,
                                  self.image_format, self.index_attr, self.cleanup, activity)
        return self.base.execute_call(self.api, self.abi, self.context, image, self.image_format, 60000, self.cleanup, activity)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.context.value:
            try:
                self.abi.check_call(self.api.rknn_destroy(self.context), "rknn_destroy")
                self.record["context_destroyed"] = True
            except Exception as exc:
                self.record["context_destroyed"] = False
                self.cleanup.append("rknn_destroy: " + repr(exc))
        else:
            self.record["context_destroyed"] = None
