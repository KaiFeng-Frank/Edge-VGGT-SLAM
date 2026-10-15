#!/usr/bin/env python3
"""Original native518 groups with one next-group preload, at most two contexts.

Constructing this object creates no SDK context or worker. Every init/run/close
belongs to one dedicated worker; runs are serial and consume actual owned prior
global output. Ordinary frozen OwnedContext IO, graphs and consumers are reused.
"""
import copy
from concurrent.futures import Future
import hashlib
import importlib.util
import os
from pathlib import Path
import queue
import re
import threading
import time

BASE = Path('/mnt/vggt-data/vggt-slam2')
FROZEN = BASE/'experiments/vggt-v100-source-r1'
SEQUENTIAL_SHA = 'f4b8eaeeae772b9095b0b0b3e93ce81944441b831ef5193077d380687a2198b7'
FRAME, GLOBAL = (2,1041,1024), (1,2082,1024)
RETAINED = (4,11,17,23)
COUNTERS = ('init_attempts','contexts_created','destroy_attempts','contexts_destroyed',
    'input_set_attempts','input_set_successes','run_attempts','runs_completed',
    'output_get_attempts','output_get_successes','output_buffers_acquired',
    'output_release_attempts','output_buffers_released')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for value in iter(lambda:stream.read(4<<20),b''):
            h.update(value)
    return h.hexdigest()


def contract():
    path=FROZEN/'native_sequential_runtime.py'
    require(sha(path)==SEQUENTIAL_SHA,'Frozen sequential contract changed')
    spec=importlib.util.spec_from_file_location('_v123_frozen_sequential_contract',path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def load_compilations(paths,digests,compiler_source,compiler_sha256):
    return contract().load_compilations(paths,digests,compiler_source,compiler_sha256)


def system_memory():
    """Read-only whole-system snapshots, not unique SDK or 8GB measurements."""
    result={}
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            key,value=line.split(':',1)
            if key in ('MemTotal','MemAvailable','CmaTotal','CmaFree'):
                result[key+'_bytes']=int(value.split()[0])*1024
        zone=None;page=os.sysconf('SC_PAGE_SIZE')
        for line in Path('/proc/zoneinfo').read_text().splitlines():
            match=re.match(r'Node\s+\d+,\s+zone\s+(\S+)',line)
            if match:zone=match.group(1)
            match=re.match(r'\s+pages free\s+(\d+)',line)
            if match and zone in ('DMA','DMA32'):
                key=zone+'_free_bytes';result[key]=result.get(key,0)+int(match.group(1))*page
    except (OSError,ValueError) as error:
        result['snapshot_error']=str(error)
    return result


class _Worker:
    """One original owner thread; failed live contexts keep the worker alive."""
    def __init__(self,controller,row):
        self.controller,self.row=controller,copy.deepcopy(row)
        self.context=None;self.last_metrics=None;self.closed_safely=False
        self.jobs=queue.Queue();self.pending=[];self.run_started=threading.Event()
        self.thread=threading.Thread(target=self._loop,name='v123-agg-'+str(row['index']),daemon=False)
        self.thread.start()

    def submit(self,method,*args):
        future=Future()
        with self.controller._lock:self.pending.append(future)
        self.jobs.put((future,method,args))
        return future

    def _loop(self):
        while True:
            job=self.jobs.get()
            if job is None:return
            future,method,args=job
            try:
                if future.set_running_or_notify_cancel():
                    try:future.set_result(getattr(self,method)(*args))
                    except BaseException as error:future.set_exception(error)
            finally:
                with self.controller._lock:self.pending.remove(future)
                job=future=method=args=None

    def snapshot(self):
        if self.context is not None:
            value=self.context.metrics()  # Owner-thread only, after SDK operation.
            with self.controller._lock:self.last_metrics=copy.deepcopy(value)
        return self.last_metrics

    def open(self):
        c=self.controller;started=time.perf_counter()
        c._reserve(self);c._event(self,'init_begin')
        try:
            require(Path(self.row['file']).stat().st_size==self.row['bytes'],'Original model byte size differs')
            try:self.context=c._factory(self.row)
            except BaseException as error:
                self.context=getattr(error,'owned_context',None)
                raise
            self.snapshot();c._event(self,'init_end')
            return dict(file_pin_and_context_init_seconds=time.perf_counter()-started)
        except BaseException as error:
            c._fatal.set();c._event(self,'init_failed',error=repr(error))
            try:self.close()
            except BaseException as cleanup:error.add_note('Owner cleanup: '+repr(cleanup))
            raise

    def run_close(self,x):
        c=self.controller;times={}
        error=None;ys=None
        try:
            require(not c._fatal.is_set() and self.context is not None,'Preload failed; no later SDK run permitted')
            with c._run_lock:
                c._event(self,'run_begin');self.run_started.set()
                ys=self.context.run([x]);times.update(copy.deepcopy(self.context.last_timings))
                self.snapshot();c._event(self,'run_end')
        except BaseException as caught:
            error=caught;c._fatal.set();c._event(self,'run_failed',error=repr(caught))
        finally:
            tick=time.perf_counter()
            try:self.close()
            except BaseException as cleanup:
                if error is None:error=cleanup
                else:error.add_note('Owner cleanup: '+repr(cleanup))
            times['context_destroy_seconds']=time.perf_counter()-tick
        if error is not None:raise error
        return ys,times

    def close(self):
        c=self.controller;c._event(self,'close_begin')
        try:
            if self.context is not None:
                self.context.close();ledger=self.snapshot()
                require(ledger['closed'] and not ledger['live_contexts'] and not ledger['live_output_buffers'],
                        'Owned SDK resources remain after close')
            self.closed_safely=True;c._release(self);c._event(self,'close_end')
        except BaseException as error:
            c._fatal.set()
            self.snapshot();c._event(self,'close_failed',error=repr(error))
            raise


class NativeSequentialRetainedGroups:
    def __init__(self,model_rows,*,runtime_helper=BASE/'experiments/vggt-v99-source-r1/run_native_block_r1.py',
            abi_source=BASE/'src/deploy/rk3588/validation/run-v12-rknn-block-board.py',
            library_path='/usr/lib/librknnrt.so',core_mask=7,timeout_ms=30000,
            expected_npu_hz=1000000000,source_root=BASE/'src'):
        self._rolling_setup()
        try:
            self._frozen=contract()
            self._frozen.NativeSequentialRetainedGroups.__init__(self,model_rows,
                runtime_helper=runtime_helper,abi_source=abi_source,library_path=library_path,
                core_mask=core_mask,timeout_ms=timeout_ms,expected_npu_hz=expected_npu_hz,source_root=source_root)
            require(core_mask==7,'R123 is the original core7 schedule screen')
            self._wait_seconds=min(60.,timeout_ms/1000.+15.)
        except BaseException as error:
            self.failed=True;error.native_retained_owner=self
            raise

    def _rolling_setup(self):
        self.owner=threading.get_ident();self.closed=self.failed=self.active=False
        self.cleanup_errors=[];self.pins_unchanged=None;self.completed_calls=0
        self.last_timings={};self.last_group_outputs=None
        self._lock=threading.RLock();self._run_lock=threading.Lock();self._fatal=threading.Event()
        self._workers=[];self._reserved=set();self._maximum_reserved=0;self._timeline=[]
        self._wait_seconds=45.

    def _factory(self,row):
        return self.owned.OwnedContext(row['file'],row['sha256'],input_shapes=[FRAME],
            output_shapes=[tuple(b['shape']) for b in row['output_bindings']],input_names=['input'],
            output_names=[b['name'] for b in row['output_bindings']],**self.options)

    def observe(self):
        return self._frozen.NativeSequentialRetainedGroups.observe(self)

    def _reserve(self,worker):
        with self._lock:
            require(not self._fatal.is_set() and len(self._reserved)<2,'Failed preload or two-context limit')
            self._reserved.add(worker);self._maximum_reserved=max(self._maximum_reserved,len(self._reserved))

    def _release(self,worker):
        with self._lock:self._reserved.discard(worker)

    def _event(self,worker,phase,**extra):
        memory=system_memory()
        with self._lock:
            self._timeline.append(dict(group=worker.row['index'],phase=phase,monotonic_seconds=time.perf_counter(),
                worker_thread=threading.get_ident(),reserved_or_initializing_contexts=len(self._reserved),
                system_memory=memory,SDK_memory=copy.deepcopy((worker.last_metrics or {}).get('SDK_memory')),**extra))

    def _wait(self,future):
        return future.result(timeout=self._wait_seconds)

    def _wait_started(self,worker,future):
        deadline=time.perf_counter()+self._wait_seconds
        while not worker.run_started.is_set():
            if future.done():
                self._wait(future)  # Surface failed job immediately, no false run_begin.
                raise RuntimeError('Owner job completed without starting a run')
            remaining=deadline-time.perf_counter()
            require(remaining>0,'Current owner did not start')
            worker.run_started.wait(min(.01,remaining))

    def _start(self,row):
        require(not self._fatal.is_set(),'Failure prevents opening following group')
        worker=_Worker(self,row);self._workers.append(worker)
        return worker,worker.submit('open')

    def _consume(self,row,ys,retained):
        torch=self.torch
        f=torch.from_numpy(ys[0]).view(1,2,1041,1024);g=torch.from_numpy(ys[1]).view(1,2,1041,1024)
        retained[row['indices'][-1]]=torch.cat([f,g],dim=-1).numpy()
        target=None
        if row['index']==3:
            target=torch.from_numpy(ys[2]).view(2,1041,1024).clone()
            target=(target/(target.norm(dim=-1,keepdim=True)+1e-8)).numpy()
        return ys[1].reshape(FRAME),target

    def _finish_workers(self):
        errors=[]
        for worker in self._workers:
            with self._lock:
                for future in worker.pending:future.cancel()
            if not worker.thread.is_alive():continue
            try:self._wait(worker.submit('close'))
            except BaseException as error:
                errors.append(repr(error));continue  # Keep failed owner's live thread/queue reachable.
            worker.jobs.put(None);worker.thread.join(timeout=self._wait_seconds)
            if worker.thread.is_alive():errors.append('Owner worker did not join: '+str(worker.row['index']))
        self.cleanup_errors.extend(errors)
        require(not errors,'Rolling owner cleanup/join failed: '+'; '.join(errors))

    def run(self,x,*,capture=False):
        require(threading.get_ident()==self.owner and not self.closed and not self.failed and not self.active,
                'Rolling runtime requires idle original caller')
        self.active=True;started=time.perf_counter();self.last_timings={'groups':[]};self.last_group_outputs=None
        retained={};raw_groups=[];target=None;result=None;error=None
        try:
            np,torch=self.np,self.torch
            require(isinstance(x,np.ndarray) and x.dtype==np.float32 and x.shape==FRAME and x.flags.c_contiguous
                and np.isfinite(x).all(),'Actual original native FP32 input required')
            input_hash=hashlib.sha256(x.tobytes()).hexdigest();current=x
            worker,opening=self._start(self.rows[0])
            with torch.inference_mode():
                for index,row in enumerate(self.rows):
                    self.observe();group_start=time.perf_counter();tick=group_start
                    times=self._wait(opening);times['preload_wait_seconds']=time.perf_counter()-tick
                    require(not self._fatal.is_set(),'Failed preload prevents further SDK work')
                    running=worker.submit('run_close',current)
                    self._wait_started(worker,running)
                    following=None
                    if index+1<len(self.rows):following=self._start(self.rows[index+1])
                    tick=time.perf_counter();ys,run_times=self._wait(running)
                    times['run_and_destroy_future_wait_seconds']=time.perf_counter()-tick;times.update(run_times)
                    require(worker.closed_safely and not self._fatal.is_set(),'Failed current close prevents following preload')
                    tick=time.perf_counter();current,last_target=self._consume(row,ys,retained)
                    if last_target is not None:target=last_target
                    if capture:raw_groups.append(tuple(ys))
                    times['original_CPU_consumer_and_bridge_seconds']=time.perf_counter()-tick
                    times['complete_group_with_lifetime_seconds']=time.perf_counter()-group_start
                    self.last_timings['groups'].append(dict(index=index,**times));self.observe()
                    if following is not None:worker,opening=following
            require(tuple(retained)==RETAINED and target is not None and target.shape==FRAME and np.isfinite(target).all()
                and all(np.isfinite(v).all() and v.flags.c_contiguous for v in retained.values())
                and hashlib.sha256(x.tobytes()).hexdigest()==input_hash,'Original retained/target/input invariants failed')
            self.last_group_outputs=tuple(raw_groups) if capture else None
            aggregated=[None]*24
            for index in RETAINED:aggregated[index]=torch.from_numpy(retained[index])
            result=dict(retained_outputs=retained,retained_list=[retained[i] for i in RETAINED],target_tokens=target,
                aggregated_tokens_list=aggregated,diagnostic_outputs=tuple(v for group in raw_groups for v in group) if capture else None)
        except BaseException as caught:
            error=caught;self.failed=True;self._fatal.set()
        finally:
            tick=time.perf_counter()
            try:self._finish_workers()
            except BaseException as cleanup:
                self.failed=True
                if error is None:error=cleanup
                else:error.add_note('Rolling final cleanup: '+repr(cleanup))
            self.last_timings['all_owner_cleanup_and_join_seconds']=time.perf_counter()-tick
            self.last_timings['complete_chain_boundary_seconds']=time.perf_counter()-started
            self.active=False
        if error is not None:
            error.native_retained_owner=self;raise error
        self.completed_calls+=1;result['timings']=self.last_timings
        return result

    def close(self):
        require(threading.get_ident()==self.owner and not self.active,'Close requires idle original caller')
        if self.closed:return
        self._finish_workers()
        self.closed=not self._reserved and all(not w.thread.is_alive() for w in self._workers)
        try:
            require(sha(FROZEN/'native_sequential_runtime.py')==SEQUENTIAL_SHA and
                sha(FROZEN/'run_native_retained_groups.py')==self._frozen.ORIGINAL_RUNTIME_SHA and
                sha(FROZEN/'native_heads_runtime.py')==self._frozen.OWNED_RUNTIME_SHA,'Frozen original dependency changed')
            self.pins_unchanged=True
        except BaseException:
            self.failed=True;self.pins_unchanged=False;raise
        require(self.closed,'Live rolling SDK owner or worker remains')

    def metrics(self):
        with self._lock:
            # Frozen v110 retains an owner when metrics raises. A cached zero
            # ledger cannot attest release while init/run/close is outstanding.
            if not self.active and any(w.thread.is_alive() for w in self._workers):
                error=RuntimeError('Rolling owner workers not terminal; SDK counters/release unavailable')
                error.native_retained_owner=self
                raise error
            rows=[copy.deepcopy(w.last_metrics) for w in self._workers if w.last_metrics is not None]
            result={key:sum(row[key] for row in rows) for key in COUNTERS}
            live=sum(row['live_contexts'] for row in rows);pending=sum(row['live_output_buffers'] for row in rows)
            result.update(closed=self.closed,failed=self.failed,active=self.active,live_contexts=live,
                live_output_buffers=pending,cleanup_errors=list(self.cleanup_errors)+[e for row in rows for e in row['cleanup_errors']],
                pins_unchanged=self.pins_unchanged,completed_chain_calls=self.completed_calls,
                completed_original_pairs=24*self.completed_calls,completed_original_blocks=48*self.completed_calls,
                requested_core_mask=getattr(self,'core_mask',None),per_group_closed_contexts=rows,
                resources_balanced=not self._reserved and not pending and result['contexts_created']==result['contexts_destroyed']
                    and result['output_buffers_acquired']==result['output_buffers_released'],
                configured_live_context_limit=2,maximum_simultaneously_live_SDK_contexts=self._maximum_reserved,
                maximum_simultaneously_live_SDK_contexts_scope='Conservative reservation upper bound, includes initializing slots; not a count of unique physical allocations',
                maximum_reserved_or_initializing_contexts_observed=self._maximum_reserved,
                live_reserved_or_initializing_contexts=len(self._reserved),live_owner_workers=sum(w.thread.is_alive() for w in self._workers),
                owner_metrics_scope='Worker-copied snapshots after operations; terminal once all workers joined',
                timeline=copy.deepcopy(self._timeline),persistent_NPU_contexts=False,init_payload_hash_destroy_in_complete_timer=True,
                SDK_memory_scope='Per-context SDK fields and dynamic system snapshots; not unique physical memory or8GB acceptance',
                precision_or_quality_accepted=False,original_model_only=True,teacher_forcing=False)
            return copy.deepcopy(result)


NativeRetainedGroups=NativeSequentialRetainedGroups
NativeRetained=NativeSequentialRetainedGroups
NativeSequentialGroups=NativeSequentialRetainedGroups
NativeRollingGroups=NativeSequentialRetainedGroups
