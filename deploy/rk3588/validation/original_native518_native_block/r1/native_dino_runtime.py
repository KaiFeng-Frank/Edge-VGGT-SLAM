"""Complete original1041-token DINO0 with six persistent native products.

Four original static-weight linears and two reused per-head attention products.
All neural computation is on RK3588S; callers pin the source, parameters and ABI.
No saved intermediate replaces computation. Initialization is measured separately.
"""
import copy
import ctypes as ct
import time

N = 1041


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


class NativeDino:
    def __init__(self, parent, gpu, abi, api, np, neon, parameters, *, legacy, mali, kernel, softmax):
        self.parent, self.abi, self.api, self.np = parent, abi, api, np
        self.products, self.bridge = {}, None
        self.closed = self.failed = False
        self.calls_attempted = self.calls_completed = 0
        self.last_timings, self.cleanup = {}, {}
        try:
            neon.f32_to_f16.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t]
            neon.f32_to_f16.restype = None
            specifications = [('qkv',1024,3072,'attn.qkv.weight'),
                ('qk',64,1056,None),('pv',1056,64,None),
                ('proj',1024,1024,'attn.proj.weight'),
                ('fc1',1024,4096,'mlp.fc1.weight'),('fc2',4096,1024,'mlp.fc2.weight')]
            for name, k, n, key in specifications:
                product = parent.Product(abi, api, np, k, n)
                self.products[name] = product  # Own before fallible creation.
                product.open()
                if key:
                    weight = parameters[key]
                    require(weight.shape == (n,k) and weight.dtype == np.float32, 'Original linear weight shape differs')
                    half = np.empty(weight.shape, np.float16)
                    neon.f32_to_f16(weight.ctypes.data, half.ctypes.data, weight.size)
                    require(np.isfinite(half).all(), 'Static Half weight overflow')
                    packed = half.T.reshape(k//32,32,n//16,16).transpose(2,0,3,1)
                    np.copyto(product.views['B'], packed, casting='no')
                    product.sync('B', abi.RKNN_MEMORY_SYNC_TO_DEVICE)
                    product.bind('B')
            memory = {name + field: product.memories[field]
                for name, product in self.products.items()
                for field in ('ABC' if name in ('qk','pv') else 'AC')}
            self.bridge = gpu.NativeDinoBridge.__new__(gpu.NativeDinoBridge)
            self.bridge.__init__(memory, parameters, legacy_source_path=legacy,
                library_path=mali, kernel_source_path=kernel, softmax_source_path=softmax)
            self.initial_gpu_metrics = copy.deepcopy(self.bridge.metrics())
        except BaseException as error:
            self.failed = True
            error.native_dino = self
            try:
                self.close()
            except BaseException as cleanup:
                error.add_note('Native DINO cleanup: ' + str(cleanup))
            raise

    def _prewrite(self, fields):
        for name, field in fields:
            self.products[name].sync(field, self.abi.RKNN_MEMORY_SYNC_TO_DEVICE)

    def _postwrite(self, fields):
        for name, field in fields:
            self.products[name].sync(field, self.abi.RKNN_MEMORY_SYNC_FROM_DEVICE)

    def _gpu(self, method, fields, *args):
        started = time.perf_counter()
        self._prewrite(fields)
        tick = time.perf_counter()
        getattr(self.bridge, method)(*args)
        end = time.perf_counter()
        details = copy.deepcopy(self.bridge.last_timings)
        self._postwrite(fields)
        return dict(complete_seconds=time.perf_counter()-started, prewrite_seconds=tick-started,
            bridge_wall_seconds=end-tick, postwrite_seconds=time.perf_counter()-end, GPU=details)

    def _run(self, name, dynamic_b=False):
        product, abi = self.products[name], self.abi
        started = time.perf_counter()
        for field in ('AB' if dynamic_b else 'A'):
            product.sync(field, abi.RKNN_MEMORY_SYNC_TO_DEVICE)
            product.bind(field)
        tick = time.perf_counter()
        abi._check(self.api.rknn_matmul_run(product.ctx), 'original ' + name)
        end = time.perf_counter()
        product.sync('C', abi.RKNN_MEMORY_SYNC_FROM_DEVICE)
        return dict(complete_seconds=time.perf_counter()-started,
            input_sync_bind_seconds=tick-started, SDK_run_seconds=end-tick,
            output_sync_seconds=time.perf_counter()-end)

    def run(self, x):
        require(not self.closed and not self.failed, 'Closed or failed native DINO')
        np = self.np
        self.calls_attempted += 1
        started = time.perf_counter()
        try:
            require(x.shape == (1,N,1024) and x.dtype == np.float32
                and x.flags.c_contiguous and np.isfinite(x).all(), 'Original input shape or storage differs')
            tick = time.perf_counter()
            self.bridge.upload_input(x)
            stages = dict(input_upload_seconds=time.perf_counter()-tick)
            stages['norm1'] = self._gpu('norm1', [('qkv','A')])
            stages['qkv'] = self._run('qkv')
            heads = []
            for head in range(16):
                row = dict(head=head)
                row['pack'] = self._gpu('pack_head', [('qk','A'),('qk','B'),('pv','B')], head)
                row['QK'] = self._run('qk', dynamic_b=True)
                row['softmax'] = self._gpu('softmax', [('pv','A')])
                row['PV'] = self._run('pv', dynamic_b=True)
                row['merge'] = self._gpu('merge_head', [('proj','A')], head)
                heads.append(row)
            stages['heads'] = heads
            stages['proj'] = self._run('proj')
            stages['proj_residual_norm'] = self._gpu('proj_norm', [('fc1','A')])
            stages['fc1'] = self._run('fc1')
            stages['erf_gelu'] = self._gpu('gelu', [('fc2','A')])
            stages['fc2'] = self._run('fc2')
            stages['final_residual'] = self._gpu('final', [])
            tick = time.perf_counter()
            y = self.bridge.read_output(np)
            require(y.shape == x.shape and y.dtype == np.float32 and np.isfinite(y).all(), 'Nonfinite or incomplete block output')
            stages['owned_output_and_finite_seconds'] = time.perf_counter()-tick
            self.last_timings = dict(complete_native_block_seconds=time.perf_counter()-started, stages=stages)
            self.calls_completed += 1
            return y
        except BaseException:
            self.failed = True
            raise

    def diagnostics(self):
        """Owned actual native buffers, outside timing; never inputs to next call."""
        result = {}
        for name, fields in [('qkv','AC'),('proj','AC'),('fc1','AC'),('fc2','AC')]:
            for field in fields:
                product = self.products[name]
                product.sync(field, self.abi.RKNN_MEMORY_SYNC_FROM_DEVICE)
                result[name+field] = self.np.array(product.views[field], copy=True, order='C')
        return result

    def close(self):
        self.closed = True
        safe, errors, resources = self.bridge is None, [], {}
        if self.bridge is not None:
            try:
                self.bridge.close()
            except BaseException as error:
                errors.append('GPU close: ' + str(error))
            try:
                self.final_gpu_metrics = copy.deepcopy(self.bridge.metrics())
                g = self.final_gpu_metrics
                safe = g['fully_released'] and g['queue_idle'] and g['live_events'] == 0
                errors.extend(g.get('cleanup_errors', []))
            except BaseException as error:
                errors.append('Cannot verify GPU ownership: ' + str(error))
        if safe:
            for name, product in reversed(list(self.products.items())):
                try:
                    row = product.close()
                    resources[name] = row
                    errors.extend(row['cleanup_errors'])
                    require(not row['live_context'] and not row['live_memories'], 'Live native allocation after close')
                except BaseException as error:
                    errors.append(name + ' close: ' + str(error))
        else:
            errors.append('NPU allocations retained because GPU imports remain live')
        self.cleanup = dict(native_close_permitted_after_GPU_release=safe, resources=resources,
            cleanup_errors=errors, calls_attempted=self.calls_attempted, calls_completed=self.calls_completed)
        if errors:
            self.failed = True
            raise RuntimeError('; '.join(errors))
        return copy.deepcopy(self.cleanup)
