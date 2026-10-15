"""Original1041-token DINO0: three fixed native SDK owners, one GPU caller.

All K reductions and learned values remain complete. Output-channel shards use
normal allocated C buffers; the GPU consumes their original logical columns.
No imported FD subrange, saved intermediate, or combined MatMul core mask.
"""
import copy
import ctypes as ct
import hashlib
import math
import queue
import threading
import time
from concurrent.futures import Future, wait

N = 1041
WAIT_SECONDS = 45.
LINEARS = (
    ('qkv', 1024, 3072, 'attn.qkv.weight', (0, 1024, 2048, 3072)),
    ('proj', 1024, 1024, 'attn.proj.weight', (0, 352, 704, 1024)),
    ('fc1', 1024, 4096, 'mlp.fc1.weight', (0, 1376, 2752, 4096)),
    ('fc2', 4096, 1024, 'mlp.fc2.weight', (0, 352, 704, 1024)),
)
HEAD_RANGES = ((0, 6), (6, 11), (11, 16))


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


class _Counted:
    """Lock counters only; never hold the lock during a blocking SDK call."""
    def __init__(self, supplied, lock, owner):
        self.owner, self.lock = owner, lock
        self.aggregate = supplied if all(hasattr(supplied, k) for k in ('api', 'calls', 'successes')) else None
        self.api = self.aggregate.api if self.aggregate is not None else supplied
        self.calls, self.successes = {}, {}

    def __getattr__(self, name):
        function = getattr(self.api, name)
        def call(*args):
            self.owner.check_thread()
            with self.lock:
                self.calls[name] = self.calls.get(name, 0) + 1
                if self.aggregate is not None:
                    counts = self.aggregate.calls
                    counts[name] = counts.get(name, 0) + 1
            result = function(*args)
            success = bool(result) if name == 'rknn_create_mem' else result == 0
            if success:
                with self.lock:
                    self.successes[name] = self.successes.get(name, 0) + 1
                    if self.aggregate is not None:
                        counts = self.aggregate.successes
                        counts[name] = counts.get(name, 0) + 1
            return result
        return call


def _open_product(product, mask):
    """The frozen R118 native descriptors/allocation ABI, with one fixed core."""
    abi, api, np = product.abi, product.api, product.np
    info = abi.RKNNMatmulInfo(M=N, K=product.k, N=product.n,
        type=abi.RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32, B_layout=1, AC_layout=1)
    abi._check(api.rknn_matmul_create(ct.byref(product.ctx), ct.byref(info), ct.byref(product.attrs)), 'create')
    require(product.ctx.value != 0, 'Null successful native context')
    abi._check(api.rknn_matmul_set_core_mask(product.ctx, mask), 'single core ' + str(mask))
    dims = dict(A=(product.k//8, N, 8), B=(product.n//16, product.k//32, 16, 32), C=(product.n//4, N, 4))
    for field in 'ABC':
        attr = getattr(product.attrs, field)
        shape = tuple(attr.dims[:attr.n_dims])
        dtype = np.float32 if field == 'C' else np.float16
        require(shape == dims[field] and attr.type == (0 if field == 'C' else 1)
            and attr.size >= math.prod(shape)*np.dtype(dtype).itemsize,
            'Unexpected original native ' + field + ' descriptor: ' + str(shape))
        product.descriptors[field] = dict(shape=list(shape), type=int(attr.type), bytes=int(attr.size))
        pointer = api.rknn_create_mem(product.ctx, attr.size)
        require(bool(pointer), 'Allocation failed: ' + field)
        product.memories[field] = pointer
        require(pointer.contents.virt_addr and pointer.contents.size >= attr.size, 'Invalid allocation: ' + field)
        if field != 'C':
            ct.memset(pointer.contents.virt_addr, 0, attr.size)
        raw = (ct.c_uint8*attr.size).from_address(pointer.contents.virt_addr)
        product.views[field] = np.frombuffer(raw, dtype=dtype, count=math.prod(shape)).reshape(shape)
        product.bind(field)


class _Worker:
    def __init__(self, runtime, lane):
        self.runtime, self.lane, self.mask = runtime, lane, 1 << lane
        self.products, self.history, self.snapshot = {}, [], {}
        self.lock, self.jobs, self.pending = threading.Lock(), queue.Queue(), 0
        self.ident, self.started, self.stopped = None, False, False
        self.api = _Counted(runtime.api, runtime.counter_lock, self)
        self.thread = threading.Thread(target=self._loop, name='original-dino-core-'+str(self.mask), daemon=False)

    def start(self):
        self.thread.start()
        self.started = True

    def check_thread(self):
        require(threading.get_ident() == self.ident, 'Native SDK operation left its fixed owner thread')

    def _refresh(self):
        self.snapshot = dict(lane=self.lane, core_mask=self.mask, owner_thread_id=self.ident,
            resources={name:dict(K=p.k, output_N=p.n, descriptors=copy.deepcopy(p.descriptors),
                live_context=int(bool(p.ctx.value)), live_memories=len(p.memories)) for name,p in self.products.items()},
            SDK_API=dict(attempts=dict(self.api.calls), successes=dict(self.api.successes)),
            cleanup_history=list(self.history))

    def _loop(self):
        self.ident = threading.get_ident()
        while True:
            job = self.jobs.get()
            if job is None:
                return
            future, function, arguments = job
            error, result = None, None
            try:
                if future.set_running_or_notify_cancel():
                    result = function(*arguments)
            except BaseException as caught:
                error = caught
            finally:
                with self.lock:
                    self._refresh()
                    self.pending -= 1
            if not future.cancelled():
                if error is None:
                    future.set_result(result)
                else:
                    future.set_exception(error)

    def submit(self, function, *arguments):
        require(self.started and not self.stopped and self.thread.is_alive(), 'SDK owner is not available')
        future = Future()
        with self.lock:
            self.pending += 1
        self.jobs.put((future, function, arguments))
        return future

    def open(self, half_weights):
        self.check_thread()
        parent, abi, np = self.runtime.parent, self.runtime.abi, self.runtime.np
        specifications = [(name+str(self.lane), k, edges[self.lane+1]-edges[self.lane], name, edges[self.lane])
            for name,k,_,_,edges in LINEARS]
        lo,hi = HEAD_RANGES[self.lane]
        specifications += [(kind+str(head), k, n, None, 0) for head in range(lo,hi)
            for kind,k,n in (('qk',64,1056), ('pv',1056,64))]
        for name,k,n,weighted,start in specifications:
            product = parent.Product(abi, self.api, np, k, n)
            self.products[name] = product  # Partial context/memory owners stay reachable.
            _open_product(product, self.mask)
            if weighted is not None:
                half = half_weights[weighted][start:start+n]
                packed = half.T.reshape(k//32,32,n//16,16).transpose(2,0,3,1)
                np.copyto(product.views['B'], packed, casting='no')
                product.sync('B', abi.RKNN_MEMORY_SYNC_TO_DEVICE)
                product.bind('B')

    def sync_fields(self, fields, direction):
        self.check_thread()
        for name,field in fields:
            self.products[name].sync(field, direction)

    def bind_shared(self, shared):
        self.check_thread()
        for name,_,_,_,_ in LINEARS:
            product = self.products[name+str(self.lane)]
            self.runtime.abi._check(self.api.rknn_matmul_set_io_mem(product.ctx, shared[name],
                ct.byref(product.attrs.A)), 'original shared-A binding ' + name)

    def run_product(self, name, dynamic_b=False, shared_a=None):
        self.check_thread()
        product,abi = self.products[name],self.runtime.abi
        started = time.perf_counter()
        if shared_a is not None:
            abi._check(self.api.rknn_matmul_set_io_mem(product.ctx, shared_a,
                ct.byref(product.attrs.A)), 'original shared-A binding ' + name)
        else:
            for field in ('AB' if dynamic_b else 'A'):
                product.sync(field, abi.RKNN_MEMORY_SYNC_TO_DEVICE)
                product.bind(field)
        before_run = time.perf_counter()
        abi._check(self.api.rknn_matmul_run(product.ctx), 'original ' + name)
        after_run = time.perf_counter()
        product.sync('C', abi.RKNN_MEMORY_SYNC_FROM_DEVICE)
        return dict(product=name, core_mask=self.mask, owner_thread_id=self.ident,
            complete_seconds=time.perf_counter()-started, input_sync_bind_seconds=before_run-started,
            SDK_run_seconds=after_run-before_run, output_sync_seconds=time.perf_counter()-after_run,
            worker_start_monotonic=started, SDK_run_start_monotonic=before_run, SDK_run_end_monotonic=after_run)

    def run_heads(self, kind):
        lo,hi = HEAD_RANGES[self.lane]
        return [dict(head=head, timing=self.run_product(kind+str(head), dynamic_b=True)) for head in range(lo,hi)]

    def owned_diagnostics(self):
        self.check_thread()
        result = {}
        for name,_,_,_,_ in LINEARS:
            product = self.products[name+str(self.lane)]
            for field in ('AC' if self.lane == 0 else 'C'):
                product.sync(field, self.runtime.abi.RKNN_MEMORY_SYNC_FROM_DEVICE)
                result[name+field] = self.runtime.np.array(product.views[field],copy=True,order='C')
        return result

    def close_products(self):
        self.check_thread()
        resources,errors = {},[]
        for name,product in reversed(list(self.products.items())):
            if product.cleanup_errors:
                self.history.extend(name+': '+e for e in product.cleanup_errors)
                product.cleanup_errors.clear()  # Preserve history, allow a real release retry.
            try:
                row = product.close()
                resources[name] = row
                errors.extend(name+': '+e for e in row['cleanup_errors'])
                if row['live_context'] or row['live_memories']:
                    errors.append('Native owner remains live: '+name)
            except BaseException as error:
                errors.append(name+' close: '+str(error))
        return dict(resources=resources,cleanup_errors=errors)

    def terminal_snapshot(self):
        with self.lock:
            require(not self.pending, 'Busy native owner retained; terminal ledger unavailable')
            return copy.deepcopy(self.snapshot)

    def stop(self):
        with self.lock:
            require(not self.pending, 'Cannot stop a busy native owner')
        if self.started and not self.stopped:
            self.jobs.put(None)
            self.thread.join(5.)
            require(not self.thread.is_alive(), 'Native owner thread remains live')
            self.stopped = True


class NativeDino:
    def __init__(self, parent, gpu, abi, api, np, neon, parameters, *, legacy, mali, kernel, softmax, batch_kernel):
        self.parent,self.abi,self.api,self.np = parent,abi,api,np
        self.main_ident = threading.get_ident()
        self.counter_lock,self.workers,self.inflight = threading.Lock(),[],[]
        self.products,self.shared_A,self.bridge = {},{},None
        self.closed = self.failed = False
        self.calls_attempted = self.calls_completed = 0
        self.last_timings,self.cleanup = {},{}
        self._half_weights,self.original_weight_partitions = {},{}
        try:
            neon.f32_to_f16.argtypes = [ct.c_void_p,ct.c_void_p,ct.c_size_t]
            neon.f32_to_f16.restype = None
            for name,k,n,key,edges in LINEARS:
                weight = parameters[key]
                require(weight.shape == (n,k) and weight.dtype == np.float32 and weight.flags.c_contiguous
                    and np.isfinite(weight).all(), 'Original dense weight differs: '+key)
                half = np.empty(weight.shape,np.float16)
                neon.f32_to_f16(weight.ctypes.data,half.ctypes.data,weight.size)
                require(np.isfinite(half).all(), 'Original static Half weight overflow')
                joined = hashlib.sha256()
                for start,end in zip(edges,edges[1:]):
                    joined.update(memoryview(half[start:end]).cast('B'))
                original_sha = hashlib.sha256(memoryview(half).cast('B')).hexdigest()
                require(joined.hexdigest() == original_sha, 'N partitions changed original Half learned storage')
                self.original_weight_partitions[name] = dict(original_weight=key,K=k,full_N=n,
                    output_channel_ranges=[list(x) for x in zip(edges,edges[1:])],
                    Half_weight_raw_sha256=original_sha,slices_reconstruct_original_Half_storage=True,
                    K_reduction_partitioned=False,weight_values_duplicated=False)
                half.flags.writeable = False
                self._half_weights[name] = half
            for lane in range(3):
                owner = _Worker(self,lane)
                self.workers.append(owner)
                owner.start()
            self._join([w.submit(w.open,self._half_weights) for w in self.workers])
            self.products = {name:p for w in self.workers for name,p in w.products.items()}
            for name,_,_,_,_ in LINEARS:
                leader = self.products[name+'0']
                for lane in (1,2):
                    part = self.products[name+str(lane)]
                    require(part.descriptors['A'] == leader.descriptors['A'], 'Actual shared native-A descriptor differs')
                self.shared_A[name] = leader.memories['A']
            self._join([w.submit(w.bind_shared,self.shared_A) for w in self.workers])
            memory = {name+'A':pointer for name,pointer in self.shared_A.items()}
            for name,product in self.products.items():
                if any(name.startswith(linear+str(lane)) for linear,_,_,_,_ in LINEARS for lane in range(3)):
                    memory[name+'C'] = product.memories['C']
                else:
                    for field in 'ABC':
                        memory[name+field] = product.memories[field]
            require(len(memory) == 112, 'Incomplete original native GPU boundaries')
            self.bridge = gpu.NativeDinoBridge.__new__(gpu.NativeDinoBridge)
            self.bridge.__init__(memory,parameters,legacy_source_path=legacy,library_path=mali,
                kernel_source_path=kernel,softmax_source_path=softmax,batch_source_path=batch_kernel)
            self.initial_gpu_metrics = copy.deepcopy(self.bridge.metrics())
            self._half_weights.clear()
        except BaseException as error:
            self.failed = True
            error.native_dino = self
            try:
                self.close()
            except BaseException as cleanup:
                error.add_note('Native DINO owner retained after cleanup: '+str(cleanup))
            raise

    def _main(self):
        require(threading.get_ident() == self.main_ident, 'GPU and orchestration must remain on the constructing thread')

    def _join(self,futures):
        self.inflight.extend(futures)
        done,pending = wait(futures,timeout=WAIT_SECONDS)
        if pending:
            self.failed = True
            raise TimeoutError('Native SDK owner did not finish; all owners retained')
        self.inflight = [f for f in self.inflight if not f.done()]
        errors = [f.exception() for f in futures if f.exception() is not None]
        if errors:
            self.failed = True
            raise RuntimeError('Native owner operation failed: '+'; '.join(str(e) for e in errors)) from errors[0]
        return [f.result() for f in futures]

    def _sync(self,fields,direction):
        groups = [[] for _ in self.workers]
        for name,field in fields:
            if name in self.shared_A:
                groups[0].append((name+'0',field))
            else:
                head = int(name[2:])
                lane = next(i for i,(lo,hi) in enumerate(HEAD_RANGES) if lo <= head < hi)
                groups[lane].append((name,field))
        self._join([w.submit(w.sync_fields,group,direction) for w,group in zip(self.workers,groups) if group])

    def _gpu(self,method,fields):
        started = time.perf_counter()
        self._sync(fields,self.abi.RKNN_MEMORY_SYNC_TO_DEVICE)
        tick = time.perf_counter()
        getattr(self.bridge,method)()
        end = time.perf_counter()
        details = copy.deepcopy(self.bridge.last_timings)
        self._sync(fields,self.abi.RKNN_MEMORY_SYNC_FROM_DEVICE)
        return dict(complete_seconds=time.perf_counter()-started,prewrite_seconds=tick-started,
            bridge_wall_seconds=end-tick,postwrite_seconds=time.perf_counter()-end,GPU=details)

    def _weighted(self,name):
        started = time.perf_counter()
        self._sync([(name,'A')],self.abi.RKNN_MEMORY_SYNC_TO_DEVICE)
        rows = self._join([w.submit(w.run_product,name+str(w.lane),False,self.shared_A[name]) for w in self.workers])
        return dict(complete_seconds=time.perf_counter()-started,parts=rows,
            measured_blocking_SDK_run_span_seconds=max(r['SDK_run_end_monotonic'] for r in rows)-min(r['SDK_run_start_monotonic'] for r in rows),
            overlapping_worker_durations_are_not_summed_as_chain_latency=True)

    def _heads(self,kind):
        started = time.perf_counter()
        groups = self._join([w.submit(w.run_heads,kind) for w in self.workers])
        rows = [row for group in groups for row in group]
        return rows,dict(complete_seconds=time.perf_counter()-started,owner_groups=[list(x) for x in HEAD_RANGES])

    def run(self,x):
        self._main()
        require(not self.closed and not self.failed,'Closed or failed native DINO')
        np = self.np
        self.calls_attempted += 1
        started = time.perf_counter()
        try:
            require(x.shape == (1,N,1024) and x.dtype == np.float32 and x.flags.c_contiguous
                and np.isfinite(x).all(),'Original input shape or storage differs')
            tick = time.perf_counter()
            self.bridge.upload_input(x)
            stages = dict(input_upload_seconds=time.perf_counter()-tick)
            stages['norm1'] = self._gpu('norm1',[('qkv','A')])
            stages['qkv'] = self._weighted('qkv')
            fields = [(kind+str(head),field) for head in range(16)
                for kind,field in (('qk','A'),('qk','B'),('pv','B'))]
            stages['pack_all'] = self._gpu('pack_all',fields)
            qk,stages['QK_parallel'] = self._heads('qk')
            stages['softmax_all'] = self._gpu('softmax_all',[('pv'+str(h),'A') for h in range(16)])
            pv,stages['PV_parallel'] = self._heads('pv')
            stages['merge_all'] = self._gpu('merge_all',[('proj','A')])
            stages['heads'] = [dict(head=q['head'],QK=q['timing'],PV=p['timing']) for q,p in zip(qk,pv)]
            stages['proj'] = self._weighted('proj')
            stages['proj_residual_norm'] = self._gpu('proj_norm',[('fc1','A')])
            stages['fc1'] = self._weighted('fc1')
            stages['erf_gelu'] = self._gpu('gelu',[('fc2','A')])
            stages['fc2'] = self._weighted('fc2')
            stages['final_residual'] = self._gpu('final',[])
            tick = time.perf_counter()
            y = self.bridge.read_output(np)
            require(y.shape == x.shape and y.dtype == np.float32 and np.isfinite(y).all(),'Nonfinite or incomplete block output')
            stages['owned_output_and_finite_seconds'] = time.perf_counter()-tick
            self.last_timings = dict(complete_native_block_seconds=time.perf_counter()-started,stages=stages,
                all_NPU_stages_joined_before_GPU_consumption=True)
            self.calls_completed += 1
            return y
        except BaseException:
            self.failed = True
            raise

    def diagnostics(self):
        """Original eight owned A/C native views, outside performance timers."""
        self._main()
        require(not self.closed and not self.failed,'Native diagnostics require a completed live candidate')
        parts = self._join([w.submit(w.owned_diagnostics) for w in self.workers])
        result = {}
        for name,_,_,_,_ in LINEARS:
            result[name+'A'] = parts[0][name+'A']
            result[name+'C'] = self.np.concatenate([p[name+'C'] for p in parts],axis=0)
        return result

    def metrics(self):
        self._main()
        rows = [w.terminal_snapshot() for w in self.workers]
        live_contexts = sum(p['live_context'] for w in rows for p in w.get('resources',{}).values())
        live_memories = sum(p['live_memories'] for w in rows for p in w.get('resources',{}).values())
        return copy.deepcopy(dict(core_masks=[1,2,4],fixed_owner_workers=rows,
            live_owner_threads=sum(w.thread.is_alive() for w in self.workers),
            live_owner_workers=sum(w.thread.is_alive() for w in self.workers),
            live_native_contexts=live_contexts,live_native_memories=live_memories,
            live_contexts=live_contexts,live_memories=live_memories,
            original_weight_partitions=self.original_weight_partitions,head_ranges=[list(x) for x in HEAD_RANGES],
            closed=self.closed,failed=self.failed,calls_attempted=self.calls_attempted,calls_completed=self.calls_completed))

    def worker_metrics(self):
        return self.metrics()['fixed_owner_workers']

    def close(self):
        self._main()
        if self.closed and self.cleanup and not self.cleanup.get('cleanup_errors') and all(not w.started or w.stopped for w in self.workers):
            return copy.deepcopy(self.cleanup)
        self.closed = True
        errors,resources = [],{}
        try:
            pending = [f for f in self.inflight if not f.done()]
            if pending:
                _,remaining = wait(pending,timeout=WAIT_SECONDS)
                require(not remaining,'SDK jobs still running; GPU and native owners retained')
            require(all(not w.pending for w in self.workers),'Native worker busy; all owners retained')
        except BaseException as error:
            self.failed = True
            self.cleanup = dict(cleanup_errors=[str(error)],owners_retained=True)
            error.native_dino = self
            raise
        safe = self.bridge is None
        if self.bridge is not None:
            try:
                self.bridge.close()
            except BaseException as error:
                errors.append('GPU close: '+str(error))
            try:
                self.final_gpu_metrics = copy.deepcopy(self.bridge.metrics())
                g = self.final_gpu_metrics
                safe = g['fully_released'] and g['queue_idle'] and g['live_events'] == 0
            except BaseException as error:
                errors.append('Cannot verify GPU ownership: '+str(error))
        if safe:
            # Borrowers must be completely destroyed before leader/shared-A owner0.
            borrowers = [w for w in self.workers if w.lane != 0 and w.started and not w.stopped]
            try:
                rows = self._join([w.submit(w.close_products) for w in borrowers])
                for row in rows:
                    resources.update(row['resources']);errors.extend(row['cleanup_errors'])
            except BaseException as error:
                errors.append('Borrower close: '+str(error))
            borrowers_gone = all(not any(p.ctx.value or p.memories for p in w.products.values()) for w in self.workers[1:])
            if borrowers_gone and not errors:
                leader = self.workers[0] if self.workers else None
                if leader is not None and leader.started and not leader.stopped:
                    try:
                        row = self._join([leader.submit(leader.close_products)])[0]
                        resources.update(row['resources']);errors.extend(row['cleanup_errors'])
                    except BaseException as error:
                        errors.append('Shared-A leader close: '+str(error))
            else:
                errors.append('Shared-A leader retained until every borrower is closed')
            if not errors:
                for owner in self.workers:
                    try:
                        owner.stop()
                    except BaseException as error:
                        errors.append('Owner join: '+str(error))
        else:
            errors.append('Native allocations retained because GPU imports remain live')
        self.cleanup = dict(native_close_permitted_after_GPU_release=safe,resources=resources,
            cleanup_errors=errors,calls_attempted=self.calls_attempted,calls_completed=self.calls_completed,
            shared_A_leader_destroyed_only_after_borrower_contexts=True,
            unused_part_owned_A_allocations_retained_until_close=True,
            live_owner_threads=sum(w.thread.is_alive() for w in self.workers))
        try:
            terminal = self.metrics()
            self.cleanup.update(owner_metrics=terminal,live_owner_workers=terminal['live_owner_workers'],
                live_contexts=terminal['live_contexts'],live_memories=terminal['live_memories'],
                borrowed_A_bindings_released=all(not any(p['live_context'] for p in w.get('resources',{}).values())
                    for w in terminal['fixed_owner_workers'] if w['lane'] != 0))
        except BaseException as error:
            self.cleanup['terminal_ledger_unavailable'] = str(error)
            errors.append(str(error))
        if errors:
            self.failed = True
            error = RuntimeError('; '.join(errors))
            error.native_dino = self
            raise error
        self._half_weights.clear()
        self.shared_A.clear()
        return copy.deepcopy(self.cleanup)
