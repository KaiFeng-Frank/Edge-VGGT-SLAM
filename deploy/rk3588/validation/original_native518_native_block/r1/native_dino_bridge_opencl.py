"""Board-only OpenCL glue between original DINO0 native RKNN MatMuls.

The caller owns and synchronizes all RKNN buffers. This bridge owns one OpenCL
context/queue, duplicated DMA fds/imports, ten original FP32 parameter vectors,
three FP32 row buffers and the softmax status scalar. It performs no SDK calls.
Close must succeed before the caller releases any imported RKNN allocation.
"""
from __future__ import annotations

import copy
import ctypes as ct
import hashlib
import importlib.util
from pathlib import Path
import threading
import time

LEGACY_SHA = 'f06486299101fb233c68abc9908e8c2629c4cb1f7dca6241133e6e222634acc2'
LIBRARY_SHA = 'd57e53240b5078eee5cfbc79e103544c090a7825c09fdf3877fee45084897dda'
SOFTMAX_SHA = '5cbb36e03e09c38db9e6769f067e3168e1cf7f67d719009601c4456e9f2b150a'
BUILD_OPTIONS = b'-cl-std=CL1.2'
ROWS, DIM, HEADS, HEAD_DIM, PAD, WG = 1041, 1024, 16, 64, 1056, 256
ROW_BYTES = ROWS * DIM * 4
VECTOR_LENGTHS = {'norm1.weight': DIM, 'norm1.bias': DIM, 'attn.qkv.bias': 3*DIM,
    'attn.proj.bias': DIM, 'ls1.gamma': DIM, 'norm2.weight': DIM, 'norm2.bias': DIM,
    'mlp.fc1.bias': 4*DIM, 'mlp.fc2.bias': DIM, 'ls2.gamma': DIM}
WEIGHT_SHAPES = {'attn.qkv.weight': (3*DIM,DIM), 'attn.proj.weight': (DIM,DIM),
    'mlp.fc1.weight': (4*DIM,DIM), 'mlp.fc2.weight': (DIM,4*DIM)}
# elem-bytes, number of elements, GPU access (READ_ONLY=4/WRITE_ONLY=2).
IMPORT_SPECS = {'qkvA': (2,ROWS*DIM,2), 'qkvC': (4,ROWS*3*DIM,4),
    'qkA': (2,ROWS*HEAD_DIM,2), 'qkB': (2,PAD*HEAD_DIM,2), 'qkC': (4,ROWS*PAD,4),
    'pvA': (2,ROWS*PAD,2), 'pvB': (2,PAD*HEAD_DIM,2), 'pvC': (4,ROWS*HEAD_DIM,4),
    'projA': (2,ROWS*DIM,2), 'projC': (4,ROWS*DIM,4),
    'fc1A': (2,ROWS*DIM,2), 'fc1C': (4,ROWS*4*DIM,4),
    'fc2A': (2,ROWS*4*DIM,2), 'fc2C': (4,ROWS*DIM,4)}


class NativeDinoBridgeError(RuntimeError):
    pass


def _require(ok, message):
    if not ok:
        raise NativeDinoBridgeError(message)


def _check(code, operation):
    if code:
        raise NativeDinoBridgeError(operation + ' returned ' + str(int(code)))


def _pin(path, expected):
    path = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    _require(expected is not None and h.hexdigest() == expected, 'Pinned source/library differs: ' + str(path))
    return path


def _aligned(n):
    return ((n + WG - 1) // WG) * WG


class NativeDinoBridge:
    def __init__(self, memories, parameters, *, legacy_source_path, library_path,
            kernel_source_path, softmax_source_path):
        self._lock = threading.RLock()
        self._owners = dict(memories)
        self._fds, self._memory, self._objects, self._events, self._host_io = [], [], [], [], []
        self._context = self._queue = self._program = None
        self._lib = self._legacy_class = None
        self._kernels, self._imports, self._fields, self._buffers, self._vectors = {}, {}, {}, {}, {}
        self._status = ct.c_uint32(0)
        self._closed, self._failed, self._queue_idle = False, False, True
        self.last_timings = {}
        self._stats = dict(calls_attempted=0, calls_completed=0, events_created=0, events_released=0,
            objects_created={}, objects_released={}, cleanup_errors=[], methods={},
            input_uploads_completed=0, owned_buffer_reads_completed=0, original_model_only=True,
            rows=ROWS, heads=HEADS, head_dim=HEAD_DIM, actual_keys=ROWS, padded_keys=PAD,
            build_options=BUILD_OPTIONS.decode(), original_scale=.125, norm_epsilon=1e-6,
            original_Erf_GELU=True, FP_CONTRACT_OFF=True, managed_host_consistency=True,
            RKNN_sync_owned_by_caller=True, GPU_matrix_staging_copies=False,
            SDK_GPU_callback=False, frequency_or_driver_writes=False,
            model_or_trajectory_or_performance_accepted=False,
            finite_validation='Softmax finite flag and owned final output; no new per-kernel status arguments')
        started = time.perf_counter()
        try:
            legacy_path = _pin(legacy_source_path, LEGACY_SHA)
            spec = importlib.util.spec_from_file_location('_v121_frozen_v81_opencl_abi', legacy_path)
            _require(spec is not None and spec.loader is not None, 'Cannot load pinned OpenCL source')
            legacy = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(legacy)
            legacy._board_guard()
            self._legacy_class = legacy.NativeErfGeluBridge
            for name in ('_bind_api', '_select_handles', '_string', '_device_string', '_build_log', '_import_function'):
                setattr(self, name, getattr(self._legacy_class, name).__get__(self, type(self)))
            import numpy as np
            self._np = np
            _require(set(memories) == set(IMPORT_SPECS), 'Exactly14 native DINO buffer boundaries required')
            _require(set(parameters) == set(VECTOR_LENGTHS) | set(WEIGHT_SHAPES), 'Exactly14 original learned arrays required')
            for name, shape in {**{k:(v,) for k,v in VECTOR_LENGTHS.items()}, **WEIGHT_SHAPES}.items():
                v = parameters[name]
                _require(isinstance(v,np.ndarray) and v.shape == shape and v.dtype == np.float32 and
                    v.flags.c_contiguous and np.isfinite(v).all(), 'Original FP32 parameter differs: ' + name)
            # Linear weights stay in caller-owned native B. Copy only immutable
            # FP32 vectors used by GPU math, preserving original element order.
            self._vectors = {name: np.array(parameters[name],copy=True,order='C') for name in VECTOR_LENGTHS}
            for v in self._vectors.values():
                v.flags.writeable = False
            for name, (itemsize, count, flags) in IMPORT_SPECS.items():
                fields = legacy._mem_fields(memories[name])
                _require(fields['size'] >= count*itemsize and fields['offset'] % itemsize == 0,
                    'Native allocation undersized/misaligned: ' + name)
                self._fields[name] = fields
            # The root driver pins the new kernel and bridge as one bundle.
            # Record exactly the source bytes compiled; frozen dependencies
            # below retain their independent fixed pins.
            kernel = Path(kernel_source_path).resolve(strict=True)
            kernel_bytes = kernel.read_bytes()
            kernel_sha = hashlib.sha256(kernel_bytes).hexdigest()
            softmax = _pin(softmax_source_path, SOFTMAX_SHA)
            library = _pin(library_path, LIBRARY_SHA)
            self._stats.update(legacy_source_sha256=LEGACY_SHA, library_sha256=LIBRARY_SHA,
                kernel_source_sha256=kernel_sha, softmax_source_sha256=SOFTMAX_SHA,
                vectors_uploaded=10, dense_weights_uploaded_by_bridge=0,
                parameter_vector_raw_sha256={k:hashlib.sha256(memoryview(v).cast('B')).hexdigest() for k,v in self._vectors.items()})
            self._lib = ct.CDLL(str(library))
            self._bind_api()
            ptr, uint, size = ct.c_void_p, ct.c_uint32, ct.c_size_t
            for name in ('clEnqueueReadBuffer', 'clEnqueueWriteBuffer'):
                f = getattr(self._lib,name)
                f.argtypes = [ptr,ptr,uint,size,size,ptr,uint,ct.POINTER(ptr),ct.POINTER(ptr)]
                f.restype = ct.c_int32
            self._lib.clGetKernelWorkGroupInfo.argtypes = [ptr,ptr,uint,size,ptr,ct.POINTER(size)]
            self._lib.clGetKernelWorkGroupInfo.restype = ct.c_int32
            self._lib.clGetKernelInfo.argtypes = [ptr,uint,size,ptr,ct.POINTER(size)]
            self._lib.clGetKernelInfo.restype = ct.c_int32
            self._platform = self._select_handles('platform',0)
            self._device = self._select_handles('device',0)
            required = {'cl_khr_fp16','cl_arm_import_memory','cl_arm_import_memory_dma_buf'}
            _require(required <= set(self._device_string(0x1030).split()), 'Required Mali FP16/DMA extensions missing')
            self._stats.update(device_name=self._device_string(0x102B),driver_version=self._device_string(0x102D),required_extensions=sorted(required))
            self._devices = (ptr*1)(self._device)
            properties = (ct.c_ssize_t*3)(0x1084,self._platform,0)
            self._context = self._create('clCreateContext',properties,1,self._devices,None,None)
            self._queue = self._create('clCreateCommandQueue',self._context,self._device,2)
            source_bytes = kernel_bytes + b'\n' + softmax.read_bytes()
            self._stats['combined_source_sha256'] = hashlib.sha256(source_bytes).hexdigest()
            sources = (ct.c_char_p*1)(source_bytes); lengths = (size*1)(len(source_bytes))
            self._program = self._create('clCreateProgramWithSource',self._context,1,sources,lengths)
            code = self._lib.clBuildProgram(self._program,1,self._devices,BUILD_OPTIONS,None,None)
            if code:
                raise NativeDinoBridgeError('clBuildProgram returned ' + str(code) + ': ' + self._build_log())
            options = self._string(self._lib.clGetProgramBuildInfo,(self._program,self._device,0x1182))
            _require(options.split() == BUILD_OPTIONS.decode().split(), 'Unexpected OpenCL options: ' + options)
            self._stats['actual_build_options'] = options
            self._import = self._import_function()
            for name, (itemsize, count, flags) in IMPORT_SPECS.items():
                self._imports[name] = self._import_dma(self._fields[name],flags,True,name)
                self._stats[name].update(element_bytes=itemsize,minimum_elements=count,GPU_access_flags=flags)
            for name,v in self._vectors.items():
                self._buffers[name] = self._create('clCreateBuffer',self._context,4|32,v.nbytes,ct.c_void_p(v.ctypes.data))
                self._memory.append(self._buffers[name])
            for name in ('x','residual1','out'):
                self._buffers[name] = self._create('clCreateBuffer',self._context,1,ROW_BYTES,None)
                self._memory.append(self._buffers[name])
            self._flag_mem = self._create('clCreateBuffer',self._context,1,ct.sizeof(self._status),None)
            self._memory.append(self._flag_mem)
            def internal(name): return [ptr(self._buffers[name])]
            def imported(name):
                return [ptr(self._imports[name]),ct.c_uint64(self._fields[name]['offset']//IMPORT_SPECS[name][0])]
            specs = {
                'norm1': ('norm1_pack',ROWS*WG,internal('x')+internal('norm1.weight')+internal('norm1.bias')+imported('qkvA')),
                'pack_head': ('qkv_head_pack',_aligned(PAD*HEAD_DIM),imported('qkvC')+internal('attn.qkv.bias')+
                    imported('qkA')+imported('qkB')+imported('pvB')+[ct.c_int32(0)]),
                'softmax': ('original_softmax_c4_to_half_a8',ROWS*WG,imported('qkC')+imported('pvA')+[ptr(self._flag_mem)]),
                'merge_head': ('pv_head_merge',_aligned(ROWS*HEAD_DIM),imported('pvC')+imported('projA')+[ct.c_int32(0)]),
                'proj_norm': ('proj_residual_norm',ROWS*WG,imported('projC')+internal('attn.proj.bias')+internal('ls1.gamma')+
                    internal('x')+internal('norm2.weight')+internal('norm2.bias')+internal('residual1')+imported('fc1A')),
                'gelu': ('gelu_pack',_aligned(ROWS*4*DIM),imported('fc1C')+internal('mlp.fc1.bias')+imported('fc2A')),
                'final': ('final_residual',_aligned(ROWS*DIM),imported('fc2C')+internal('mlp.fc2.bias')+
                    internal('ls2.gamma')+internal('residual1')+internal('out'))}
            device_local = ct.c_uint64()
            _check(self._lib.clGetDeviceInfo(self._device,0x1023,ct.sizeof(device_local),ct.byref(device_local),None),'device local memory')
            self._work = {}
            for method,(name,total,args) in specs.items():
                handle = self._create('clCreateKernel',self._program,name.encode())
                self._kernels[method] = handle
                argc,maxgroup,local = ct.c_uint32(),ct.c_size_t(),ct.c_uint64()
                _check(self._lib.clGetKernelInfo(handle,0x1191,ct.sizeof(argc),ct.byref(argc),None),'kernel argument count')
                _require(argc.value == len(args), 'Kernel argument arity differs: ' + name)
                _check(self._lib.clGetKernelWorkGroupInfo(handle,self._device,0x11B0,ct.sizeof(maxgroup),ct.byref(maxgroup),None),'kernel workgroup')
                _check(self._lib.clGetKernelWorkGroupInfo(handle,self._device,0x11B2,ct.sizeof(local),ct.byref(local),None),'kernel local memory')
                _require(maxgroup.value >= WG and local.value <= device_local.value, 'Kernel exceeds reported device limits: ' + name)
                for index,v in enumerate(args):
                    _check(self._lib.clSetKernelArg(handle,index,ct.sizeof(v),ct.byref(v)),'kernel argument ' + name + '/' + str(index))
                self._work[method] = ((size*1)(total),(size*1)(WG))
                self._stats['methods'][method] = dict(kernel=name,global_size=[total],local_size=[WG],dimensions=1,
                    kernel_arguments=argc.value,local_memory_bytes=local.value,max_workgroup_size=maxgroup.value,
                    calls_attempted=0,calls_completed=0)
            self._stats.update(device_local_memory_bytes=device_local.value,
                owned_row_buffers_bytes=3*ROW_BYTES,owned_parameter_vectors_bytes=sum(v.nbytes for v in self._vectors.values()),
                initialization_seconds=time.perf_counter()-started)
        except BaseException as error:
            self._failed = True
            error.native_dino_bridge = self
            try:
                self.close()
            except BaseException as cleanup:
                error.add_note('Native DINO bridge cleanup: ' + str(cleanup))
            raise

    def _remember(self, before):
        for suffix,handle in self._objects[before:]:
            counts = self._stats['objects_created']; counts[suffix] = counts.get(suffix,0)+1

    def _create(self, name, *args):
        before = len(self._objects)
        try:
            return self._legacy_class._create(self,name,*args)
        finally:
            self._remember(before)

    def _import_dma(self, *args):
        before = len(self._objects)
        try:
            return self._legacy_class._import_dma(self,*args)
        finally:
            self._remember(before)

    def _finish(self, operation):
        _check(self._lib.clFinish(self._queue),operation)
        self._queue_idle = True
        self._host_io.clear()

    def _release_events(self):
        remaining = []
        for event in reversed(self._events):
            code = self._lib.clReleaseEvent(event)
            if code:
                remaining.append(event)
                self._stats['cleanup_errors'].append('clReleaseEvent returned ' + str(code))
            else:
                self._stats['events_released'] += 1
        self._events = list(reversed(remaining))

    def _idle(self):
        _require(not self._closed and not self._failed and not self._events,'Bridge closed/failed or event still live')

    def _execute(self, method, head=None):
        with self._lock:
            self._idle()
            stats = self._stats['methods'][method]
            stats['calls_attempted'] += 1; self._stats['calls_attempted'] += 1
            started = last = time.perf_counter(); timing = {}; event = ct.c_void_p()
            try:
                kernel = self._kernels[method]
                if head is not None:
                    _require(type(head) is int and 0 <= head < HEADS,'Original head index must be0..15')
                    index = 9 if method == 'pack_head' else 4
                    value = ct.c_int32(head)
                    _check(self._lib.clSetKernelArg(kernel,index,ct.sizeof(value),ct.byref(value)),'head kernel argument')
                if method == 'softmax':
                    self._status.value = 0; self._queue_idle = False
                    _check(self._lib.clEnqueueWriteBuffer(self._queue,self._flag_mem,1,0,ct.sizeof(self._status),
                        ct.byref(self._status),0,None,None),'reset softmax finite flag')
                    now=time.perf_counter();timing['finite_flag_reset_seconds']=now-last;last=now
                self._queue_idle = False
                glob,loc = self._work[method]
                code = self._lib.clEnqueueNDRangeKernel(self._queue,kernel,1,None,glob,loc,0,None,ct.byref(event))
                if event.value:
                    self._events.append(event.value);self._stats['events_created'] += 1
                _check(code,'enqueue ' + method);_require(event.value is not None,'Kernel returned no event')
                now=time.perf_counter();timing['kernel_enqueue_seconds']=now-last;last=now
                self._finish('finish ' + method)
                now=time.perf_counter();timing['kernel_finish_seconds']=now-last;last=now
                if method == 'softmax':
                    self._queue_idle = False
                    _check(self._lib.clEnqueueReadBuffer(self._queue,self._flag_mem,1,0,ct.sizeof(self._status),
                        ct.byref(self._status),0,None,None),'read softmax finite flag')
                    self._finish('finish softmax flag read')
                    timing['finite_error_bits'] = self._status.value
                    _require(self._status.value == 0,'GPU softmax error bits: ' + str(self._status.value))
                    now=time.perf_counter();timing['finite_flag_read_check_seconds']=now-last;last=now
                clocks = {}
                for name,key in (('queued',0x1280),('submit',0x1281),('start',0x1282),('end',0x1283)):
                    value = ct.c_uint64()
                    _check(self._lib.clGetEventProfilingInfo(event,key,ct.sizeof(value),ct.byref(value),None),'event '+name)
                    clocks[name] = value.value
                _require(clocks['queued'] <= clocks['submit'] <= clocks['start'] <= clocks['end'],'Profiling clock order differs')
                timing.update(method=method,kernel=stats['kernel'],global_size=list(stats['global_size']),local_size=[WG],
                    GPU_event_clocks_ns=clocks,GPU_kernel_seconds=(clocks['end']-clocks['start'])/1e9)
                if head is not None:
                    timing['head'] = head
            except BaseException:
                self._failed = True
                raise
            finally:
                if self._queue_idle:
                    self._release_events()
                timing['profiling_and_event_release_seconds'] = time.perf_counter()-last
                timing['complete_bridge_seconds'] = time.perf_counter()-started
                self.last_timings = copy.deepcopy(timing)
            if self._events or self._stats['cleanup_errors']:
                self._failed = True
                raise NativeDinoBridgeError('OpenCL event cleanup incomplete')
            stats['calls_completed'] += 1; self._stats['calls_completed'] += 1
            return copy.deepcopy(timing)

    def upload_input(self, x):
        with self._lock:
            self._idle(); started=time.perf_counter()
            np=self._np
            _require(isinstance(x,np.ndarray) and x.shape == (1,ROWS,DIM) and x.dtype == np.float32 and
                x.flags.c_contiguous and np.isfinite(x).all(),'Complete finite original FP32 input required')
            self._host_io.append(x)
            try:
                self._queue_idle=False
                _check(self._lib.clEnqueueWriteBuffer(self._queue,self._buffers['x'],1,0,ROW_BYTES,
                    ct.c_void_p(x.ctypes.data),0,None,None),'upload original input')
                self._finish('finish original input upload')
                self._stats['input_uploads_completed'] += 1
                self.last_timings=dict(method='upload_input',complete_bridge_seconds=time.perf_counter()-started)
                return copy.deepcopy(self.last_timings)
            except BaseException:
                self._failed=True
                raise

    def norm1(self): return self._execute('norm1')
    def pack_head(self, head): return self._execute('pack_head',head)
    def softmax(self): return self._execute('softmax')
    def merge_head(self, head): return self._execute('merge_head',head)
    def proj_norm(self): return self._execute('proj_norm')
    def gelu(self): return self._execute('gelu')
    def final(self): return self._execute('final')

    def read_buffer(self, name, np):
        """Owned FP32 x/residual1/out diagnostic; reads are not kernel enqueues."""
        with self._lock:
            self._idle(); _require(name in ('x','residual1','out'),'Only owned FP32 row buffers may be read')
            started=time.perf_counter();v=np.empty((1,ROWS,DIM),np.float32)
            self._host_io.append(v)
            try:
                self._queue_idle=False
                _check(self._lib.clEnqueueReadBuffer(self._queue,self._buffers[name],1,0,ROW_BYTES,
                    ct.c_void_p(v.ctypes.data),0,None,None),'read owned '+name)
                self._finish('finish owned '+name+' read')
                _require(np.isfinite(v).all(),'Nonfinite owned '+name)
                self._stats['owned_buffer_reads_completed'] += 1
                self.last_timings=dict(method='read_'+name,complete_bridge_seconds=time.perf_counter()-started)
                return v
            except BaseException:
                self._failed=True
                raise

    def read_output(self, np): return self.read_buffer('out',np)

    def metrics(self):
        with self._lock:
            released=not self._objects and not self._fds and not self._events and self._queue_idle
            return copy.deepcopy({**self._stats,'closed':self._closed,'failed':self._failed,
                'queue_idle':self._queue_idle,'live_OpenCL_objects':len(self._objects),
                'live_owned_DMA_fds':len(self._fds),'live_events':len(self._events),
                'live_host_IO_owners':len(self._host_io),'fully_released':released})

    def close(self):
        with self._lock:
            if self._closed and self.metrics()['fully_released']:
                return
            self._closed=True
            if self._queue:
                try:
                    self._finish('finish before imported storage release')
                except BaseException as error:
                    self._queue_idle=False;self._failed=True
                    self._stats['cleanup_errors'].append(str(error))
                    raise NativeDinoBridgeError('Queue finish failed; imports/fds/NPU owners retained') from error
            self._release_events()
            if self._events:
                self._failed=True
                raise NativeDinoBridgeError('Events remain live; imported storage retained')
            errors,remaining=[],[]
            for suffix,handle in reversed(self._objects):
                code=getattr(self._lib,'clRelease'+suffix)(handle)
                if code:
                    errors.append('clRelease'+suffix+' returned '+str(code));remaining.append((suffix,handle))
                else:
                    counts=self._stats['objects_released'];counts[suffix]=counts.get(suffix,0)+1
            self._objects=list(reversed(remaining))
            if ('CommandQueue',self._queue) not in self._objects:
                self._queue=None
            if not self._objects:
                import os
                self._memory.clear();remaining_fds=[]
                for fd in reversed(self._fds):
                    try:
                        os.close(fd)
                    except OSError as error:
                        errors.append('close DMA fd: '+str(error));remaining_fds.append(fd)
                self._fds=list(reversed(remaining_fds))
            if not self._objects and not self._fds and not self._events and self._queue_idle:
                self._owners.clear();self._vectors.clear();self._host_io.clear()
            self._stats['cleanup_errors'].extend(errors)
            if errors:
                self._failed=True
                raise NativeDinoBridgeError('; '.join(errors))
