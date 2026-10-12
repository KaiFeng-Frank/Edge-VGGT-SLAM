"""Private RKNN 2.3.2 GPU custom-op registration for two original tails.

No SDK/OpenCL library loads on import.  Close this handle BEFORE destroying its
RKNN context.  The SDK owns its GPU context/queue/initial kernel; the borrowed
bridge owns only additional kernels, temporary buffers and per-call imports.
"""
import ctypes as ct
import copy
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace

OP_TYPES = {'frame': 'cstOriginalFrameAttentionTail', 'global': 'cstOriginalGlobalAttentionTail'}
EXPORT_SCHEMA = 'v83_original_pair_attention_custom_ONNX_export'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_abi(base):
    class TensorMem(ct.Structure):
        _fields_ = [('virt_addr', ct.c_void_p), ('phys_addr', ct.c_uint64),
            ('fd', ct.c_int32), ('offset', ct.c_int32), ('size', ct.c_uint32),
            ('flags', ct.c_uint32), ('priv_data', ct.c_void_p)]

    class GPUContext(ct.Structure):
        _fields_ = [('cl_context', ct.c_void_p), ('cl_command_queue', ct.c_void_p), ('cl_kernel', ct.c_void_p)]

    class CustomContext(ct.Structure):
        _fields_ = [('target', ct.c_int), ('internal_ctx', ct.c_uint64),
                    ('gpu_ctx', GPUContext), ('priv_data', ct.c_void_p)]

    class CustomTensor(ct.Structure):
        _fields_ = [('attr', base.TensorAttr), ('mem', TensorMem)]

    class CustomAttr(ct.Structure):
        _fields_ = [('name', ct.c_char * 256), ('dtype', ct.c_int),
                    ('n_elems', ct.c_uint32), ('data', ct.c_void_p)]

    Callback = ct.CFUNCTYPE(ct.c_int, ct.POINTER(CustomContext), ct.POINTER(CustomTensor),
                            ct.c_uint32, ct.POINTER(CustomTensor), ct.c_uint32)
    Destroy = ct.CFUNCTYPE(ct.c_int, ct.POINTER(CustomContext))

    class CustomOp(ct.Structure):
        _fields_ = [('version', ct.c_uint32), ('target', ct.c_int),
            ('op_type', ct.c_char * 256), ('cl_kernel_name', ct.c_char * 256),
            ('cl_kernel_source', ct.c_char_p), ('cl_source_size', ct.c_uint64),
            ('cl_build_options', ct.c_char * 256), ('init', Callback),
            ('prepare', Callback), ('compute', Callback), ('compute_native', Callback), ('destroy', Destroy)]

    return SimpleNamespace(TensorMem=TensorMem, GPUContext=GPUContext, CustomContext=CustomContext,
        CustomTensor=CustomTensor, CustomAttr=CustomAttr, Callback=Callback, Destroy=Destroy, CustomOp=CustomOp)


def check_abi(base, abi, expected):
    types = {'rknn_custom_op': abi.CustomOp, 'rknn_custom_op_context': abi.CustomContext,
        'rknn_custom_op_tensor': abi.CustomTensor, 'rknn_custom_op_attr': abi.CustomAttr,
        'rknn_gpu_op_context': abi.GPUContext, 'rknn_tensor_attr': base.TensorAttr,
        'rknn_tensor_mem': abi.TensorMem}
    require(ct.sizeof(ct.c_void_p) == 8, 'Only verified ARM64 pointer ABI allowed')
    for name, value in expected.items():
        type_name, _, member = name.partition('.')
        actual = getattr(types[type_name], member).offset if member else ct.sizeof(types[type_name])
        require(actual == value, 'Official SDK custom-op ABI mismatch: ' + name)


class OriginalAttentionCustomHandle:
    def __init__(self, api, context, base, export_path, export_sha256, *, borrowed_module,
                 legacy_module, component_module, opencl_library, abi_path, abi_sha256):
        self.api, self.context, self.base = api, context, base
        self._lock, self._states = threading.RLock(), {}
        self._owner, self._closed, self._active = threading.get_ident(), False, 0
        self._references = []
        self._bridge_metrics = {}
        self._stats = dict(init_attempted=0, init_completed=0, compute_attempted=0,
            compute_completed=0, destroy_attempted=0, destroy_completed=0, registered=False,
            errors=[], cleanup_errors=[], roles={}, callback_seconds=0.,
            original_model_only=True, target='SDK_GPU', compute_dtype='float32',
            compute_native_used=False, public_owner_context_used_for_mem_sync=True,
            first_compute_initializes_constant_buffers=True,
            NPU_INT8_internal_arithmetic_equivalence_claimed=False)
        base.require_board()
        require(bool(context.value), 'A live owner RKNN context is required')
        require(sha(export_path) == export_sha256 and sha(abi_path) == abi_sha256,
                'Frozen export/board custom ABI receipt changed')
        self.export = json.loads(Path(export_path).read_text())
        require(self.export['schema'] == EXPORT_SCHEMA
                and self.export['proof']['original_parameter_tensors_verified'] == 36
                and self.export['proof']['all_original_initializers_protobuf_unchanged'] is True,
                'Complete original custom export required')
        self.abi = make_abi(base)
        check_abi(base, self.abi, json.loads(Path(abi_path).read_text()))
        self._factory = borrowed_module.make_borrowed_bridge_class(legacy_module, component_module)
        self._library = str(opencl_library)
        self._kernel_source = borrowed_module.kernel_source(component_module).encode()
        api.rknn_register_custom_ops.argtypes = [base.CONTEXT, ct.POINTER(self.abi.CustomOp), ct.c_uint32]
        api.rknn_register_custom_ops.restype = ct.c_int
        api.rknn_custom_op_get_op_attr.argtypes = [ct.POINTER(self.abi.CustomContext), ct.c_char_p,
                                                   ct.POINTER(self.abi.CustomAttr)]
        api.rknn_custom_op_get_op_attr.restype = None
        api.rknn_mem_sync.argtypes = [base.CONTEXT, ct.POINTER(self.abi.TensorMem), ct.c_int]
        api.rknn_mem_sync.restype = ct.c_int
        ops = (self.abi.CustomOp * 2)()
        for index, role in enumerate(('frame', 'global')):
            tail = next(t for t in self.export['proof']['tails'] if t['role'] == role)
            b, _, s, _ = tail['input_shapes'][0]
            require(tail['custom_op_type'] == OP_TYPES[role], 'Registered custom type differs')
            init = self.abi.Callback(lambda c, i, ni, o, no, role=role: self._init(role, c, i, ni, o, no))
            compute = self.abi.Callback(lambda c, i, ni, o, no, role=role: self._compute(role, c, i, ni, o, no))
            destroy = self.abi.Destroy(self._destroy)
            self._references.extend((init, compute, destroy))
            op = ops[index]
            op.version, op.target, op.op_type = 1, 2, OP_TYPES[role].encode()
            op.cl_kernel_name = ('original_qk_norm_rope_' + role).encode()
            source = self._kernel_source.replace(b'original_qk_norm_rope', op.cl_kernel_name)
            source = (f'#define BATCH {b}\n#define TOKENS {s}\n#define LOCAL_KEYS {64 if s == 53 else 128}\n').encode() + source
            self._references.append(source)
            op.cl_kernel_source, op.cl_source_size = source, len(source)
            op.cl_build_options = f'-cl-std=CL1.2 -DBATCH={b} -DTOKENS={s} -DLOCAL_KEYS={64 if s == 53 else 128}'.encode()
            op.init, op.compute, op.destroy = init, compute, destroy
        self._references.extend((ops, self._kernel_source))
        code = api.rknn_register_custom_ops(context, ops, 2)
        self._stats['registration_code'] = int(code)
        base.check_call(code, 'rknn_register_custom_ops(original attention)')
        require(not self._stats['errors'], 'Custom init callback failed: ' + repr(self._stats['errors']))
        self._stats['registered'] = True

    def _error(self, phase, role, exc):
        self._stats['errors'].append(dict(phase=phase, role=role, error=f'{type(exc).__name__}: {exc}'))
        print('Original attention custom callback failed: ' + json.dumps(self._stats['errors'][-1]), flush=True)
        return -1

    def _record(self, tensor, shape, dma_required=True):
        r = self.base.attr_record(tensor.attr)
        r['memory'] = dict(virt_addr_present=bool(tensor.mem.virt_addr), fd=int(tensor.mem.fd),
                           offset=int(tensor.mem.offset), size=int(tensor.mem.size), flags=int(tensor.mem.flags))
        expected = math.prod(shape)
        require(r['type'] == 0 and r['format'] in (0, 3) and r['n_elems'] == expected
                and r['size'] == expected * 4, 'Custom compute must expose contiguous original FP32 operands: ' + repr(r))
        require([n for n in r['shape'] if n != 1] == [n for n in shape if n != 1],
                'Custom compute operand layout changed: ' + repr(r))
        require(r['quantization_type'] == 0 or (r['quantization_type'] == 2
                and r['zero_point'] == 0 and r['scale'] == 1), 'FP32 custom operand has nonidentity quantization')
        require(tensor.mem.virt_addr and (not dma_required or tensor.mem.fd >= 0) and tensor.mem.offset >= 0
                and tensor.mem.size >= expected * 4, 'Custom tensor memory span is invalid: ' + repr(r))
        require(tensor.attr.size_with_stride in (0, tensor.attr.size)
                and tensor.attr.w_stride in (0, r['shape'][-1] if r['shape'] else 1),
                'Padded custom compute layout unsupported: ' + repr(r))
        return r

    def _validate(self, role, inputs, n_inputs, outputs, n_outputs):
        require(n_inputs == 11 and n_outputs == 1, 'Original custom tail ABI is 11 inputs/1 output')
        tail = next(t for t in self.export['proof']['tails'] if t['role'] == role)
        records = [self._record(inputs[i], shape, dma_required=i < 3)
                   for i, shape in enumerate(tail['input_shapes'])]
        records.append(self._record(outputs[0], tail['output_shape']))
        return tail, records

    def _init(self, role, op, inputs, n_inputs, outputs, n_outputs):
        with self._lock:
            self._stats['init_attempted'] += 1
            try:
                require(not self._closed and op.contents.target == 2, 'SDK GPU custom context required')
                tail, records = self._validate(role, inputs, n_inputs, outputs, n_outputs)
                gpu = op.contents.gpu_ctx
                require(gpu.cl_context and gpu.cl_command_queue and gpu.cl_kernel, 'SDK GPU handles missing')
                key = len(self._states) + 1
                require(not op.contents.priv_data, 'SDK custom private slot must start empty')
                self._states[key] = dict(role=role, tail=tail, gpu=(gpu.cl_context, gpu.cl_command_queue, gpu.cl_kernel),
                                         bridge=None, bridge_closed=False, calls=0)
                op.contents.priv_data = key
                self._stats['roles'][role] = dict(descriptors=records, calls=0, callback_seconds=0.)
                self._stats['init_completed'] += 1
                return 0
            except BaseException as exc:
                return self._error('init', role, exc)

    def _owned(self, tensor, shape):
        import numpy as np
        address = int(tensor.mem.virt_addr) + int(tensor.mem.offset)
        value = np.ctypeslib.as_array((ct.c_float * math.prod(shape)).from_address(address)).copy().reshape(shape)
        require(np.isfinite(value).all(), 'Original custom constant contains nonfinite values')
        return value

    def _attribute(self, op, name, dtype, count):
        row = self.abi.CustomAttr()
        self.api.rknn_custom_op_get_op_attr(op, name.encode(), ct.byref(row))
        require(row.dtype == dtype and row.n_elems == count and row.data, 'Original custom attribute differs: ' + name)
        return row

    def _start_bridge(self, state, op, inputs):
        import numpy as np
        b, _, s, _ = state['tail']['input_shapes'][0]
        parameters = [self._owned(inputs[i], (64,)) for i in range(3, 7)]
        sign = self._owned(inputs[7], (64,))
        require(np.array_equal(sign, np.tile(np.r_[-np.ones(16), np.ones(16)].astype(np.float32), 2)),
                'Original two-axis RoPE rotation signs differ')
        row = self._attribute(op, 'rope_permutation', 8, 64)
        permutation = np.ctypeslib.as_array((ct.c_int64 * 64).from_address(row.data)).copy()
        expected = np.r_[np.arange(16, 32), np.arange(0, 16), np.arange(48, 64), np.arange(32, 48)]
        require(np.array_equal(permutation, expected), 'Original two-axis rotate_half permutation differs')
        cosine = self._owned(inputs[8], (b, 1, s, 64))
        sine = self._owned(inputs[9], (b, 1, s, 64))
        scale = float(self._owned(inputs[10], ()).item())
        require(scale == .125, 'Original SDPA scale differs')
        epsilon = state['tail']['epsilon']
        row = self._attribute(op, 'epsilon', 0, 1)
        require(float(ct.cast(row.data, ct.POINTER(ct.c_float))[0]) == epsilon, 'Original Q/K norm epsilon differs')
        tables = [cosine[..., :32].copy(), sine[..., :32].copy(),
                  cosine[..., 32:].copy(), sine[..., 32:].copy()]
        state['bridge'] = self._factory(context=state['gpu'][0], queue=state['gpu'][1], kernel=state['gpu'][2],
            shape=(b, 16, s, 64), parameters=parameters, tables=tables, epsilon=epsilon, scale=scale,
            library=self._library)
        self._stats['roles'][state['role']]['constant_raw_sha256'] = [
            hashlib.sha256(x.tobytes()).hexdigest() for x in parameters + [sign, cosine, sine]]

    def _sync(self, tensor, direction):
        self.base.check_call(self.api.rknn_mem_sync(self.context, ct.byref(tensor.mem), direction),
                             'Custom operand mem_sync ' + str(direction))

    def _compute(self, role, op, inputs, n_inputs, outputs, n_outputs):
        with self._lock:
            begin = time.perf_counter()
            self._stats['compute_attempted'] += 1
            self._active += 1
            try:
                require(not self._closed, 'Custom callback handle is closed')
                self._validate(role, inputs, n_inputs, outputs, n_outputs)
                state = self._states[int(op.contents.priv_data or 0)]
                require(state['role'] == role and not state['bridge_closed'], 'Custom node state differs')
                gpu = op.contents.gpu_ctx
                require(state['gpu'] == (gpu.cl_context, gpu.cl_command_queue, gpu.cl_kernel),
                        'SDK GPU context/queue/kernel changed')
                # SDK GPU preparation can still be queued when compute is entered.
                gpu_library = ct.CDLL(self._library)
                gpu_library.clFinish.argtypes = [ct.c_void_p]
                gpu_library.clFinish.restype = ct.c_int
                self.base.check_call(gpu_library.clFinish(gpu.cl_command_queue), 'Wait SDK GPU input preparation')
                for i in range(3):
                    self._sync(inputs[i], 2)
                if state['bridge'] is None:
                    self._start_bridge(state, op, inputs)

                def before_input():
                    for i in range(3):
                        self._sync(inputs[i], 2)

                def before_output():
                    self._sync(outputs[0], 1)

                def after_output():
                    self._sync(outputs[0], 2)
                    import numpy as np
                    size = math.prod(state['tail']['output_shape'])
                    address = int(outputs[0].mem.virt_addr) + int(outputs[0].mem.offset)
                    view = np.ctypeslib.as_array((ct.c_float * size).from_address(address))
                    require(np.isfinite(view).all(), 'Custom GPU projection input is nonfinite')
                    self._sync(outputs[0], 1)

                state['bridge'].run(inputs[0].mem, inputs[1].mem, inputs[2].mem, outputs[0].mem,
                    before_input=before_input, before_output=before_output, after_output=after_output)
                state['calls'] += 1
                self._stats['roles'][role]['calls'] += 1
                self._stats['compute_completed'] += 1
                return 0
            except BaseException as exc:
                return self._error('compute', role, exc)
            finally:
                self._active -= 1
                elapsed = time.perf_counter() - begin
                self._stats['callback_seconds'] += elapsed
                if role in self._stats['roles']:
                    self._stats['roles'][role]['callback_seconds'] += elapsed

    def _destroy(self, op):
        with self._lock:
            self._stats['destroy_attempted'] += 1
            key = int(op.contents.priv_data or 0)
            try:
                if key in self._states:
                    state = self._states[key]
                    if state['bridge'] is not None and not state['bridge_closed']:
                        state['bridge'].close()
                        require(state['bridge'].metrics()['fully_released'], 'GPU bridge release incomplete')
                        state['bridge_closed'] = True
                    if state['bridge'] is not None:
                        self._bridge_metrics[state['role']] = state['bridge'].metrics()
                    del self._states[key]
                op.contents.priv_data = None
                self._stats['destroy_completed'] += 1
                return 0
            except BaseException as exc:
                return self._error('destroy', self._states.get(key, {}).get('role'), exc)

    def close(self):
        require(threading.get_ident() == self._owner and self._active == 0,
                'Only the idle owner may close custom GPU resources')
        with self._lock:
            if self._closed:
                return
            errors = []
            for state in reversed(list(self._states.values())):
                if state['bridge'] is not None and not state['bridge_closed']:
                    try:
                        state['bridge'].close()
                        require(state['bridge'].metrics()['fully_released'], 'GPU bridge release incomplete')
                        state['bridge_closed'] = True
                        self._bridge_metrics[state['role']] = state['bridge'].metrics()
                    except BaseException as exc:
                        errors.append(str(exc))
            self._stats['cleanup_errors'].extend(errors)
            require(not errors, 'Do not free RKNN memory still imported by GPU: ' + repr(errors))
            self._closed = True

    def metrics(self):
        with self._lock:
            bridges = dict(self._bridge_metrics)
            bridges.update({state['role']: state['bridge'].metrics() for state in self._states.values()
                            if state['bridge'] is not None})
            return copy.deepcopy(dict(self._stats, closed=self._closed, active_compute=self._active,
                states_waiting_for_SDK_destroy=len(self._states), GPU_bridges=bridges,
                live_GPU_bridges=sum(state['bridge'] is not None and not state['bridge_closed']
                                     for state in self._states.values())))


def register_original_attention_custom(*args, **kwargs):
    handle = OriginalAttentionCustomHandle.__new__(OriginalAttentionCustomHandle)
    try:
        handle.__init__(*args, **kwargs)
        return handle
    except BaseException as exc:
        exc.custom_handle = handle
        raise
