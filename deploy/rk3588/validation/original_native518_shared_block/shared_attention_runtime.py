"""Original full1041 attention using frozen v118 MatMul and v119 DMA bridge.

This module imports no neural framework or device SDK. The caller loads pinned
board-only implementations, owns process policy, and supplies actual prefix QKV.
The two native products stay owned until imported GPU storage is fully released.
"""
import copy
import ctypes as ct
import time

N, PAD, H, D = 1041, 1056, 16, 64


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


class SharedAttention:
    def __init__(self, parent, gpu, abi, counted, np, neon, *, legacy, mali, kernel):
        self.parent, self.abi, self.api, self.np, self.neon = parent, abi, counted, np, neon
        self.contexts, self.bridge, self.last_timings = [], None, {}
        self.closed = self.failed = False
        self.calls_attempted = self.calls_completed = 0
        try:
            neon.f32_to_f16.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t]
            neon.f32_to_f16.restype = None
            for k, n in ((D, PAD), (PAD, D)):
                product = parent.Product(abi, counted, np, k, n)
                self.contexts.append(product)
                product.open()
            self.qk, self.pv = self.contexts
            self.bridge = gpu.NativeAttentionSoftmaxBridge.__new__(gpu.NativeAttentionSoftmaxBridge)
            self.bridge.__init__(self.qk.memories['C'], self.pv.memories['A'],
                legacy_source_path=legacy, library_path=mali, kernel_source_path=kernel)
            self.initial_bridge_metrics = copy.deepcopy(self.bridge.metrics())
        except BaseException as error:
            self.failed = True
            error.shared_attention = self
            try:
                self.close()
            except BaseException as cleanup:
                error.add_note('Shared attention cleanup: ' + str(cleanup))
            raise

    def half(self, value):
        np = self.np
        value = np.ascontiguousarray(value, dtype=np.float32)
        result = np.empty(value.shape, np.float16)
        self.neon.f32_to_f16(value.ctypes.data, result.ctypes.data, value.size)
        require(np.isfinite(result).all(), 'Half overflow')
        return result

    def _upload(self, product, name, value):
        if name == 'A':
            packed = value.reshape(N, product.k // 8, 8).transpose(1, 0, 2)
        else:
            packed = value.reshape(product.k // 32, 32, product.n // 16, 16).transpose(2, 0, 3, 1)
        self.np.copyto(product.views[name], packed, casting='no')
        product.sync(name, self.abi.RKNN_MEMORY_SYNC_TO_DEVICE)
        product.bind(name)

    def _head(self, query, key, value):
        qk, pv, abi, counted, np, bridge = self.qk, self.pv, self.abi, self.api, self.np, self.bridge
        tick = time.perf_counter()
        self._upload(qk, 'A', query)
        self._upload(qk, 'B', key.T)
        now = time.perf_counter(); upload_qk = now - tick
        abi._check(counted.rknn_matmul_run(qk.ctx), 'QK run')
        end = time.perf_counter(); qk_run = end - now
        shared = bridge.run(before_input=lambda: qk.sync('C', abi.RKNN_MEMORY_SYNC_FROM_DEVICE),
            before_output=lambda: pv.sync('A', abi.RKNN_MEMORY_SYNC_TO_DEVICE),
            after_output=lambda: pv.sync('A', abi.RKNN_MEMORY_SYNC_FROM_DEVICE))
        tick = time.perf_counter()
        pv.sync('A', abi.RKNN_MEMORY_SYNC_TO_DEVICE); pv.bind('A')
        self._upload(pv, 'B', value)
        now = time.perf_counter(); upload_pv = now - tick
        abi._check(counted.rknn_matmul_run(pv.ctx), 'PV run')
        end = time.perf_counter(); pv_run = end - now
        pv.sync('C', abi.RKNN_MEMORY_SYNC_FROM_DEVICE)
        y = np.array(pv.views['C'].transpose(1, 0, 2), copy=True, order='C').reshape(N, D)
        require(np.isfinite(y).all(), 'Nonfinite PV output')
        return y, dict(QK_pack_upload_bind_seconds=upload_qk, QK_SDK_run_seconds=qk_run,
            shared_softmax=shared, PV_upload_bind_seconds=upload_pv, PV_SDK_run_seconds=pv_run,
            PV_sync_owned_finite_seconds=time.perf_counter() - end)

    def run(self, qkv):
        require(not self.closed and not self.failed, 'Closed or failed shared attention')
        np = self.np
        self.calls_attempted += 1
        started = time.perf_counter()
        try:
            require(qkv.shape == (1, N, H * D * 3) and qkv.dtype == np.float32
                and qkv.flags.c_contiguous and np.isfinite(qkv).all(), 'Complete actual prefix QKV required')
            values = self.half(qkv).reshape(1, N, 3, H, D).transpose(2, 0, 3, 1, 4)
            q, k, v = values[:, 0]
            key = np.zeros((PAD, D), np.float16); value = np.zeros((PAD, D), np.float16)
            output = np.empty((H, N, D), np.float32); heads = []
            for head in range(H):
                key[:N] = k[head]; value[:N] = v[head]
                output[head], timing = self._head(q[head], key, value)
                heads.append(dict(head=head, **timing))
            result = np.ascontiguousarray(output.transpose(1, 0, 2).reshape(1, N, H * D))
            require(np.isfinite(result).all(), 'Nonfinite complete attention')
            self.last_timings = dict(complete_attention_seconds=time.perf_counter() - started, heads=heads)
            self.calls_completed += 1
            return result
        except BaseException:
            self.failed = True
            raise

    def close(self):
        self.closed = True
        safe, errors, resources = self.bridge is None, [], []
        if self.bridge is not None:
            try:
                self.bridge.close()
            except BaseException as error:
                errors.append('GPU close: ' + str(error))
            try:
                self.final_bridge_metrics = copy.deepcopy(self.bridge.metrics())
                g = self.final_bridge_metrics
                safe = g['fully_released'] and g['queue_idle'] and g['live_events'] == 0
                errors.extend(g.get('cleanup_errors', []))
            except BaseException as error:
                errors.append('GPU ownership unverified: ' + str(error))
        if safe:
            for product in reversed(self.contexts):
                try:
                    row = product.close(); resources.append(row); errors.extend(row['cleanup_errors'])
                except BaseException as error:
                    errors.append('NPU close: ' + str(error))
        else:
            errors.append('Retained NPU allocations: imported GPU storage not fully released')
        self.cleanup = dict(native_close_permitted_after_GPU_release=safe, resources=resources,
            cleanup_errors=errors, calls_attempted=self.calls_attempted, calls_completed=self.calls_completed)
        if errors:
            self.failed = True
            raise RuntimeError('; '.join(errors))
        return copy.deepcopy(self.cleanup)
