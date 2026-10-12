"""Private bounded callback capture; all timings here include diagnostics.

Keep the frozen distinct-source registration/math path. Capture only the first
two calls for each original attention role, before returning to RKNN. No SDK,
OpenCL or model runs occur on import. This is a diagnosis, not a speed test.
"""
import ctypes as ct
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import time

BASE_SOURCE_SHA256 = '68f77690d05353921f12a67cb1975c361ca996afaf0f35aa15bdb8ebeb25a80b'
BASE_SOURCE = Path(__file__).with_name('rknn_original_attention_custom_distinct.py')
if hashlib.sha256(BASE_SOURCE.read_bytes()).hexdigest() != BASE_SOURCE_SHA256:
    raise RuntimeError('Frozen distinct custom runtime source changed')
_spec = importlib.util.spec_from_file_location('_v83_frozen_distinct_custom', BASE_SOURCE)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


class _DiagnosticBridge:
    def __init__(self, owner, bridge):
        self.owner, self.bridge = owner, bridge

    def run(self, *args, **kwargs):
        current = self.owner._diagnostic_current
        if current is not None:
            def consume(values):
                import numpy as np
                for name, value in values.items():
                    _base.require(isinstance(value, np.ndarray) and value.dtype == np.float32
                                  and np.isfinite(value).all(), 'Invalid GPU readback: ' + name)
                    current['arrays'][name] = value.copy(order='C')
                current['row']['GPU_readback_completed'] = True
            kwargs['diagnostic_consumer'] = consume
        return self.bridge.run(*args, **kwargs)

    def close(self):
        return self.bridge.close()

    def metrics(self):
        return self.bridge.metrics()


class DiagnosticOriginalAttentionCustomHandle(_base.OriginalAttentionCustomHandle):
    def __init__(self, *args, diagnostic_directory, **kwargs):
        self._diagnostic_root = Path(diagnostic_directory).resolve()
        self._diagnostic_root.mkdir(parents=True, exist_ok=False)
        self._diagnostic_attempts = {'frame': 0, 'global': 0}
        self._diagnostic_current = None
        self._diagnostic_report = dict(
            schema='v83_original_attention_custom_callback_diagnostic', status='running',
            diagnostic_source_sha256=_base.sha(__file__),
            base_runtime_source_sha256=BASE_SOURCE_SHA256,
            original_model_only=True, diagnostic_calls_per_role_limit=2,
            diagnostic_capture_in_callback_timing=True,
            normal_performance_claimed=False, DMA_coherence_established=False,
            GPU_program_cache_cause_established=False, captures=[])
        super().__init__(*args, **kwargs)

    def _start_bridge(self, state, op, inputs):
        try:
            super()._start_bridge(state, op, inputs)
        except BaseException as exc:
            # Retain a partially initialized owner even when its first close fails.
            if getattr(exc, 'borrowed_bridge', None) is not None:
                state['bridge'] = exc.borrowed_bridge
            raise
        state['bridge'] = _DiagnosticBridge(self, state['bridge'])

    def _sync(self, tensor, direction):
        begin = time.perf_counter()
        code = self.api.rknn_mem_sync(self.context, ct.byref(tensor.mem), direction)
        if self._diagnostic_current is not None:
            self._diagnostic_current['row']['sync_calls'].append(dict(
                direction=direction, code=int(code), seconds=time.perf_counter() - begin,
                fd=int(tensor.mem.fd), offset=int(tensor.mem.offset), size=int(tensor.mem.size),
                virtual_address=int(tensor.mem.virt_addr or 0)))
        self.base.check_call(code, 'Custom operand mem_sync ' + str(direction))

    def _persist(self, row, arrays):
        import numpy as np
        name = row['role'] + '-' + str(row['role_callback_index']).zfill(2)
        path = self._diagnostic_root / (name + '.npz')
        np.savez(path, **arrays)
        row['artifact'] = dict(file=path.name, bytes=path.stat().st_size,
                               sha256=_base.sha(path), arrays={})
        for key, value in arrays.items():
            row['artifact']['arrays'][key] = dict(shape=list(value.shape), dtype=str(value.dtype),
                finite=bool(np.isfinite(value).all()), bytes=value.nbytes,
                raw_sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())
        self._diagnostic_report['captures'].append(row)
        self._diagnostic_report['status'] = ('captures_saved_no_numerical_acceptance'
            if all(r['status'] == 'returned' for r in self._diagnostic_report['captures'])
            else 'callback_or_capture_failed')
        self._diagnostic_report['runtime_metrics_at_last_capture'] = super().metrics()
        temporary = self._diagnostic_root / 'capture.json.tmp'
        temporary.write_text(json.dumps(self._diagnostic_report, indent=2, allow_nan=False) + '\n')
        temporary.replace(self._diagnostic_root / 'capture.json')

    def _compute(self, role, op, inputs, n_inputs, outputs, n_outputs):
        with self._lock:
            if self._diagnostic_attempts[role] >= 2:
                return super()._compute(role, op, inputs, n_inputs, outputs, n_outputs)
            index = self._diagnostic_attempts[role]
            self._diagnostic_attempts[role] += 1
            begin = time.perf_counter()
            arrays, row = {}, dict(role=role, role_callback_index=index, status='attempted',
                started_monotonic_seconds=begin, sync_calls=[], GPU_readback_completed=False,
                CPU_output_after_FROM_DEVICE_saved=False)
            self._diagnostic_current = dict(row=row, arrays=arrays)
            code = -1
            try:
                tail, records = self._validate(role, inputs, n_inputs, outputs, n_outputs)
                row['descriptors'] = records
                row['input_names'] = tail['input_names']
                row['output_name'] = tail['output_name']
                row['epsilon'] = tail['epsilon']
                row['scale'] = .125
                row['SDK_custom_context'] = dict(target=int(op.contents.target),
                    internal_context=int(op.contents.internal_ctx),
                    private_slot=int(op.contents.priv_data or 0),
                    GPU_context=int(op.contents.gpu_ctx.cl_context or 0),
                    GPU_queue=int(op.contents.gpu_ctx.cl_command_queue or 0),
                    GPU_kernel=int(op.contents.gpu_ctx.cl_kernel or 0))
                for i, shape in enumerate(tail['input_shapes']):
                    arrays['CPU_input_' + str(i).zfill(2)] = self._owned(inputs[i], shape)
                row['CPU_inputs_before_TO_DEVICE_saved'] = True
                code = super()._compute(role, op, inputs, n_inputs, outputs, n_outputs)
                row['SDK_compute_return_code'] = int(code)
                if code == 0:
                    arrays['CPU_output_after_FROM_DEVICE'] = self._owned(outputs[0], tail['output_shape'])
                    row['CPU_output_after_FROM_DEVICE_saved'] = True
                    row['status'] = 'returned'
                else:
                    row['status'] = 'callback_failed'
                state = self._states.get(int(op.contents.priv_data or 0))
                if state is not None and state['bridge'] is not None:
                    row['GPU_bridge_metrics'] = state['bridge'].metrics()
            except BaseException as exc:
                row['status'] = 'capture_or_callback_failed'
                row['error'] = f'{type(exc).__name__}: {exc}'
                code = self._error('diagnostic', role, exc)
            finally:
                self._diagnostic_current = None
                row['callback_and_snapshot_seconds_before_save'] = time.perf_counter() - begin
                try:
                    self._persist(row, arrays)
                except BaseException as exc:
                    code = self._error('diagnostic_save', role, exc)
                row['total_callback_including_capture_save_seconds'] = time.perf_counter() - begin
            return code

    def metrics(self):
        result = super().metrics()
        result['diagnostic'] = dict(capture_attempts=copy.deepcopy(self._diagnostic_attempts),
            captures_saved=len(self._diagnostic_report['captures']),
            captured_roles=[r['role'] for r in self._diagnostic_report['captures']],
            normal_performance_claimed=False, capture_limit_per_role=2)
        return result


def register_original_attention_custom(*args, **kwargs):
    handle = DiagnosticOriginalAttentionCustomHandle.__new__(DiagnosticOriginalAttentionCustomHandle)
    try:
        handle.__init__(*args, **kwargs)
        return handle
    except BaseException as exc:
        exc.custom_handle = handle
        raise
