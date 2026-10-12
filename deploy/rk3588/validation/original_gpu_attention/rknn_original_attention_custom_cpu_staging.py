"""Private whole-pair control: SDK CPU callback runs original tails on GPU.

The graph, original parameters and NPU remainder are unchanged. Every callback
owns FP32 CPU operands, performs GPU staging, copies the owned GPU result into
the SDK CPU output and syncs it TO_DEVICE. No silent CPU math fallback exists.
All transfer/finish/copy/sync cost belongs to the full-pair call. The GPU contexts
are private to each role and are closed before the enclosing RKNN context.
"""
import ctypes as ct
import hashlib
import importlib.util
from pathlib import Path
import time

BASE_SOURCE_SHA256 = 'e3dbb1d2ed763ee41ab475332f65b15d02bfef6ba718a5738e257317fed39ddd'
BASE_SOURCE = Path(__file__).with_name('rknn_original_attention_custom.py')
if hashlib.sha256(BASE_SOURCE.read_bytes()).hexdigest() != BASE_SOURCE_SHA256:
    raise RuntimeError('Frozen original custom runtime source changed')
_spec = importlib.util.spec_from_file_location('_v83_custom_cpu_base', BASE_SOURCE)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


class _CPURegistrationAPI:
    """Only this owner's two custom registrations change target; no global patch."""
    def __init__(self, api):
        self._api = api
        def register(context, ops, count):
            for i in range(count):
                ops[i].target = 1
                ops[i].cl_kernel_name = b''
                ops[i].cl_kernel_source = None
                ops[i].cl_source_size = 0
                ops[i].cl_build_options = b''
            api.rknn_register_custom_ops.argtypes = register.argtypes
            api.rknn_register_custom_ops.restype = register.restype
            return api.rknn_register_custom_ops(context, ops, count)
        self.rknn_register_custom_ops = register

    def __getattr__(self, name):
        return getattr(self._api, name)


class CPUStagingOriginalAttentionCustomHandle(_base.OriginalAttentionCustomHandle):
    def __init__(self, api, *args, component_module, legacy_module, **kwargs):
        self._staging_factory = component_module.make_bridge_class(legacy_module)
        super().__init__(_CPURegistrationAPI(api), *args,
                         component_module=component_module, legacy_module=legacy_module, **kwargs)
        def create(**values):
            bridge = self._staging_factory.__new__(self._staging_factory)
            try:
                _base.require(ct.c_float(values['epsilon']).value == ct.c_float(1e-5).value,
                              'Original serialized FP32 epsilon differs')
                bridge.__init__(shape=values['shape'], parameters=values['parameters'],
                    tables=values['tables'], q_eps=1e-5, k_eps=1e-5,
                    scale=values['scale'], library=values['library'])
                return bridge
            except BaseException as exc:
                exc.borrowed_bridge = bridge
                raise
        self._factory = create
        self._stats.update(target='SDK_CPU_callback_private_GPU_staging',
            compute_dtype='float32', all_attention_tail_compute_on_board_GPU=True,
            SDK_owned_GPU_handles_used=False, GPU_input_output_staging=True,
            DMA_imported=False, normal_performance_includes_all_staging=True,
            input_CPU_ready_assumption_applies_only_to_SDK_CPU_target=True)

    def _validate(self, role, inputs, n_inputs, outputs, n_outputs):
        _base.require(n_inputs == 11 and n_outputs == 1, 'Original custom tail ABI is 11 inputs/1 output')
        tail = next(t for t in self.export['proof']['tails'] if t['role'] == role)
        records = [self._record(inputs[i], shape, dma_required=False)
                   for i,shape in enumerate(tail['input_shapes'])]
        records.append(self._record(outputs[0], tail['output_shape'], dma_required=False))
        return tail, records

    def _init(self, role, op, inputs, n_inputs, outputs, n_outputs):
        with self._lock:
            self._stats['init_attempted'] += 1
            try:
                _base.require(not self._closed and op.contents.target == 1, 'SDK CPU callback target required')
                tail,records = self._validate(role,inputs,n_inputs,outputs,n_outputs)
                key = len(self._states) + 1
                _base.require(not op.contents.priv_data, 'SDK private slot must start empty')
                self._states[key] = dict(role=role,tail=tail,gpu=(0,0,0),
                                        bridge=None,bridge_closed=False,calls=0)
                op.contents.priv_data = key
                self._stats['roles'][role] = dict(descriptors=records,calls=0,callback_seconds=0.)
                self._stats['init_completed'] += 1
                return 0
            except BaseException as exc:
                return self._error('CPU_init',role,exc)

    def _start_bridge(self, state, op, inputs):
        try:
            super()._start_bridge(state,op,inputs)
        except BaseException as exc:
            if getattr(exc,'borrowed_bridge',None) is not None:
                state['bridge'] = exc.borrowed_bridge
            raise

    def _compute(self, role, op, inputs, n_inputs, outputs, n_outputs):
        with self._lock:
            begin = time.perf_counter()
            self._stats['compute_attempted'] += 1
            self._active += 1
            try:
                _base.require(not self._closed and op.contents.target == 1, 'Live SDK CPU callback required')
                tail,records = self._validate(role,inputs,n_inputs,outputs,n_outputs)
                state = self._states[int(op.contents.priv_data or 0)]
                _base.require(state['role'] == role and not state['bridge_closed'], 'Custom role/state differs')
                if state['bridge'] is None:
                    self._start_bridge(state,op,inputs)
                # SDK CPU target provides ready CPU data; do not invalidate it as GPU target data.
                owned = [self._owned(inputs[i],tail['input_shapes'][i]) for i in range(3)]
                value,_,timing = state['bridge'].run(*owned,capture_intermediates=False)
                import numpy as np
                _base.require(value.dtype == np.float32 and list(value.shape) == tail['output_shape']
                    and value.flags.owndata and value.flags.c_contiguous and np.isfinite(value).all(),
                    'Original owned finite GPU projection input required')
                address = int(outputs[0].mem.virt_addr) + int(outputs[0].mem.offset)
                ct.memmove(address, value.ctypes.data, value.nbytes)
                self._sync(outputs[0],1)
                self._stats['roles'][role].update(last_GPU_staging_timing=timing,
                    last_input_raw_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in owned],
                    last_output_raw_sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                    last_descriptors=records)
                state['calls'] += 1
                self._stats['roles'][role]['calls'] += 1
                self._stats['compute_completed'] += 1
                return 0
            except BaseException as exc:
                return self._error('CPU_compute_private_GPU',role,exc)
            finally:
                self._active -= 1
                elapsed = time.perf_counter() - begin
                self._stats['callback_seconds'] += elapsed
                if role in self._stats['roles']:
                    self._stats['roles'][role]['callback_seconds'] += elapsed


def register_original_attention_custom(*args, **kwargs):
    handle = CPUStagingOriginalAttentionCustomHandle.__new__(CPUStagingOriginalAttentionCustomHandle)
    try:
        handle.__init__(*args,**kwargs)
        return handle
    except BaseException as exc:
        exc.custom_handle = handle
        raise
