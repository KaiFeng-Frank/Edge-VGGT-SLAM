#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-2-Clause
"""Source-identical typed-output owner used in the R212 diagnostic.

Library only: no SDK is loaded and no artifact is read at import. The caller
must supply the pinned typed ABI, loaded API and NumPy, and enforce board,
artifact-hash, memory/storage, thermal and restoration guards before initialize.
This is a diagnostic owner, not a deployment backend or an inference CLI.
"""
import copy
import ctypes as ct
import hashlib
import json
import math
import os
import threading
import time

ROLES = ('Q0', 'Q1', 'Q2', 'K', 'V')
NAMES = tuple('__original_native_layout/' + role for role in ROLES)
INPUT_NAME = '__global_input'
INPUT_SHAPE = [1, 2082, 1024]
OUTPUT_SHAPES = [[2, 694, 64, 8]] * 3 + [[2, 2082, 64, 8]] * 2
STORAGE = dict(failure_NPY_max_bytes=(64 << 20) + 128)
TYPED_ABI_SOURCE = dict(file='deploy/rk3588/validation/run-v12-rknn-block-board.py',
    bytes=24823, sha256='37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608')
HISTORICAL_SOURCE = {'bytes': 54775, 'sha256': '3abfdbd6e50316eefbdb5849c508800bc3d238d39134f25d726dabd7d9688f82'}

def require(ok,message):
    if not ok:raise RuntimeError(message)


def floating_identity(a):
    return (a['quantization_type']==0 or (a['type']==1 and a['quantization_type']==2 and
        a['zero_point']==0 and a['scale']==1.)) and math.isfinite(a['scale'])


def join_roles(rows,names):
    require(len(rows)==len(names) and len({r['name'] for r in rows})==len(rows) and {r['name'] for r in rows}==set(names),
        'Actual descriptor names must uniquely match all original role names')
    return [next(r for r in rows if r['name']==n) for n in names]


class ProducerOutputOwner:
    """Ordinary typed outputs; every native release/destroy is attempted once.

    Unknown run/get/release/destroy retains the context, buffers and input owner.
    This class does not inherit the frozen helper's retrying run/close methods.
    """
    def __init__(self,abi,api,np):
        self.b,self.api,self.np=abi,api,np;self.context=ct.c_uint64(0);self.thread=threading.get_ident()
        self.active=False;self.closed=False;self.failed=False;self.unknown_native_operation=None
        self.pending=None;self.close_attempted=False;self.retained_inputs=[];self.last_failed_outputs=[];self.last_failed_output_records=[]
        self.query_records=[];self.last_output_requests=[];self.last_input_requests=[];self.last_timings={}
        self.stats=dict(init_attempts=0,contexts_created=0,destroy_attempts=0,contexts_destroyed=0,input_set_attempts=0,
            input_set_successes=0,run_attempts=0,runs_completed=0,output_get_attempts=0,output_get_successes=0,
            output_buffers_acquired=0,output_release_attempts=0,output_buffers_released=0,cleanup_errors=[],
            input_request_history=[],output_request_history=[],core_mask=7,init_flags=0)

    def _query(self,command,index):
        b=self.b;value=b.TensorAttr(index=index)
        code=self.api.rknn_query(self.context,command,ct.byref(value),ct.sizeof(value))
        rec=dict(command=command,index=index,returncode=int(code),raw_struct_hex=ct.string_at(ct.byref(value),ct.sizeof(value)).hex())
        self.query_records.append(rec);b.check_call(code,'rknn_query('+str(command)+')')
        rec['attribute']=b.attr_record(value);return rec['attribute']

    def initialize(self,path):
        require(threading.get_ident()==self.thread and not self.context.value and not self.closed,'Fresh original owner thread required')
        b=self.b;self.filename_owner=ct.create_string_buffer(os.fsencode(path));self.stats['init_attempts']+=1
        try:
            try:code=self.api.rknn_init(ct.byref(self.context),ct.cast(self.filename_owner,ct.c_void_p),0,0,None)
            except BaseException:self.unknown_native_operation='rknn_init exception';raise
            finally:self.stats['contexts_created']+=int(bool(self.context.value))
            b.check_call(code,'rknn_init');require(self.context.value,'Nonempty context after init required')
            b.check_call(self.api.rknn_set_core_mask(self.context,7),'rknn_set_core_mask no fallback')
            ver=b.query(self.api,self.context,5,b.SDKVersion())
            self.stats['api_version']=bytes(ver.api_version).decode(errors='replace')
            require(self.stats['api_version'].startswith('2.3.2'),'Pinned typed ABI requires actual SDK2.3.2')
            number=b.query(self.api,self.context,0,b.InputOutputNum());require((number.n_input,number.n_output)==(1,5),'Original producer IO counts differ')
            # Record every actual public/native descriptor before validating geometry.
            inputs=[self._query(1,0)]
            outputs=[self._query(2,i) for i in range(5)]
            native_inputs=[self._query(8,0)]
            native_outputs=[self._query(9,i) for i in range(5)]
            self.inputs=join_roles(inputs,[INPUT_NAME]);self.outputs=join_roles(outputs,NAMES)
            self.native_inputs=join_roles(native_inputs,[INPUT_NAME]);self.native_outputs=join_roles(native_outputs,NAMES)
            a=self.inputs[0]
            require(a['index']==0 and a['shape']==INPUT_SHAPE and a['n_elems']==math.prod(INPUT_SHAPE) and
                a['type'] in (0,1) and a['format'] in (0,3) and a['pass_through']==0 and floating_identity(a) and
                a['size']==math.prod(INPUT_SHAPE)*(4 if a['type']==0 else 2),'Unsupported ordinary original X descriptor')
            require({a['index'] for a in outputs}==set(range(5)),'Actual logical output index set differs')
            require({a['index'] for a in native_outputs}==set(range(5)),'Actual native9 output index set differs')
            for a,shape in zip(self.outputs,OUTPUT_SHAPES):
                require(a['shape']==shape and a['n_elems']==math.prod(shape) and a['type']==1 and
                    a['format'] in (0,3) and a['pass_through']==0 and floating_identity(a) and a['size']==2*math.prod(shape),
                    'Declared ordinary output must actually be exact logical FLOAT16, not relabeled F32/native9')
            for a in self.native_inputs+self.native_outputs:
                require(a['n_elems']==math.prod(a['shape']) and a['type'] in (0,1) and floating_identity(a) and
                    a['size']>0 and a['size_with_stride']>=a['size'],'Actual native IO size/identity metadata invalid')
            mem=b.query(self.api,self.context,6,b.MemSize())
            self.stats['SDK_memory']=dict(weight_bytes=int(mem.total_weight_size),internal_bytes=int(mem.total_internal_size),
                dma_bytes=int(mem.total_dma_allocated_size),scope='SDK fields, not unique physical allocation totals')
            self.stats['role_to_logical_index']={r:a['index'] for r,a in zip(ROLES,self.outputs)}
            self.stats['logical_input_descriptors']=copy.deepcopy(self.inputs);self.stats['logical_output_descriptors']=copy.deepcopy(self.outputs)
            self.stats['native8_input_descriptors']=copy.deepcopy(self.native_inputs);self.stats['native9_output_descriptors']=copy.deepcopy(self.native_outputs)
        except BaseException:
            self.failed=True;raise

    def _release_once(self):
        entry=self.pending;require(entry is not None and not entry['release_attempted'],'Output release must be attempted once')
        entry['release_attempted']=True;self.stats['output_release_attempts']+=1
        try:code=self.api.rknn_outputs_release(self.context,5,entry['outputs'])
        except BaseException:self.unknown_native_operation='rknn_outputs_release exception';raise
        entry['release_returncode']=int(code)
        if code!=0:self.unknown_native_operation='rknn_outputs_release returned '+str(code)
        self.b.check_call(code,'rknn_outputs_release single attempt')
        self.stats['output_buffers_released']+=entry['acquired'];self.pending=None

    def run(self,x,want_float):
        require(threading.get_ident()==self.thread and self.context.value and not self.active and not self.failed and
            not self.closed and self.pending is None and self.unknown_native_operation is None,'Idle known original owner thread required')
        require(want_float in (0,1),'Only ordinary declared-half or requested-float outputs are permitted')
        self.active=True;start=time.perf_counter();times={};b,api,np=self.b,self.api,self.np;ys=[];get_known_complete=False
        outputs=(b.Output*5)();self.last_failed_outputs=[];self.last_failed_output_records=[];self.last_output_requests=[]
        try:
            tick=time.perf_counter()
            require(isinstance(x,np.ndarray) and x.dtype==np.float32 and list(x.shape)==INPUT_SHAPE and
                x.flags.c_contiguous and not x.flags.writeable and np.isfinite(x).all(),'Readonly finite exact original X required')
            self.retained_inputs=[x]
            inp=(b.Input*1)(b.Input(index=self.inputs[0]['index'],buf=int(x.ctypes.data),size=x.nbytes,
                pass_through=0,type=0,fmt=self.inputs[0]['format']))
            self.last_input_requests=[dict(index=int(inp[0].index),size=int(inp[0].size),type=int(inp[0].type),fmt=int(inp[0].fmt),
                pass_through=int(inp[0].pass_through),shape=list(x.shape),dtype=str(x.dtype))]
            self.stats['input_request_history'].append(dict(requests=copy.deepcopy(self.last_input_requests),SDK_input_set_return_code=None))
            for o,a in zip(outputs,self.outputs):o.index=a['index'];o.want_float=want_float;o.is_prealloc=0
            times['input_validation_and_output_request_seconds']=time.perf_counter()-tick
            tick=time.perf_counter();self.stats['input_set_attempts']+=1
            try:code=api.rknn_inputs_set(self.context,1,inp)
            except BaseException:self.unknown_native_operation='rknn_inputs_set exception';raise
            self.stats['input_request_history'][-1]['SDK_input_set_return_code']=int(code)
            b.check_call(code,'rknn_inputs_set original X');self.stats['input_set_successes']+=1
            times['inputs_set_seconds']=time.perf_counter()-tick
            run=b.RunExtend(non_block=0,timeout_ms=30000,fence_fd=-1);received=b.OutputExtend()
            tick=time.perf_counter();self.stats['run_attempts']+=1
            try:code=api.rknn_run(self.context,ct.byref(run))
            except BaseException:self.unknown_native_operation='rknn_run exception';raise
            if code!=0:self.unknown_native_operation='rknn_run returned '+str(code)
            b.check_call(code,'rknn_run no retry');self.stats['runs_completed']+=1;times['SDK_run_seconds']=time.perf_counter()-tick
            tick=time.perf_counter();self.stats['output_get_attempts']+=1
            try:code=api.rknn_outputs_get(self.context,5,outputs,ct.byref(received))
            except BaseException:self.unknown_native_operation='rknn_outputs_get exception';raise
            finally:
                acquired=sum(bool(o.buf) for o in outputs)
                if acquired:
                    self.pending=dict(outputs=outputs,acquired=acquired,release_attempted=False)
                    self.stats['output_buffers_acquired']+=acquired
                times['outputs_get_seconds']=time.perf_counter()-tick
                self.last_output_requests=[dict(role=r,name=a['name'],logical_index=a['index'],index=int(o.index),want_float=int(o.want_float),
                    is_prealloc=int(o.is_prealloc),returned_size=int(o.size),pointer_nonnull=bool(o.buf),
                    expected_size=math.prod(s)*(4 if want_float else 2),declared_type=a['type'],copy_dtype='float32' if want_float else 'float16',
                    shape=list(s)) for r,a,s,o in zip(ROLES,self.outputs,OUTPUT_SHAPES,outputs)]
                self.stats['output_request_history'].append(copy.deepcopy(self.last_output_requests))
            if code!=0:self.unknown_native_operation='rknn_outputs_get returned '+str(code)
            b.check_call(code,'rknn_outputs_get no retry');self.stats['output_get_successes']+=1;get_known_complete=True
            # Validate every actual span before any typed pointer dereference. A bounded
            # unexpected span is copied only as uninterpreted bytes, never as HALF/F32.
            for role,o,a,s in zip(ROLES,outputs,self.outputs,OUTPUT_SHAPES):
                valid=(o.index==a['index'] and bool(o.buf) and o.want_float==want_float and o.is_prealloc==0 and
                    o.size==math.prod(s)*(4 if want_float else 2))
                if not valid and o.buf and 0<int(o.size)<=STORAGE['failure_NPY_max_bytes']-128:
                    v=np.ctypeslib.as_array(ct.cast(o.buf,ct.POINTER(ct.c_uint8)),shape=(int(o.size),))
                    self.last_failed_outputs=[v.copy()];del v
                    self.last_failed_output_records=[dict(role=role,output_name=a['name'],actual_index=int(o.index),
                        actual_returned_size=int(o.size),interpretation='uninterpreted actual SDK returned bytes; no floating relabel')]
                require(valid,'Actual ordinary output index/type-selected byte span differs; bounded raw bytes preserved, no reinterpretation')
            tick=time.perf_counter()
            for o,s in zip(outputs,OUTPUT_SHAPES):
                kind=ct.c_float if want_float else ct.c_uint16
                v=np.ctypeslib.as_array(ct.cast(o.buf,ct.POINTER(kind)),shape=(math.prod(s),))
                owned=v.copy();del v
                if not want_float:owned=owned.view(np.float16)
                ys.append(owned.reshape(s))
                self.last_failed_outputs=ys
                self.last_failed_output_records=[dict(role=r,output_name=n,interpretation='typed actual SDK owned output') for r,n in zip(ROLES,NAMES)][:len(ys)]
            self.last_failed_outputs=ys
            times['all_owned_typed_copies_seconds']=time.perf_counter()-tick
            require(run.frame_id==received.frame_id,'Synchronous frame ID differs; copied raw outputs preserved')
            tick=time.perf_counter();self._release_once();times['outputs_release_seconds']=time.perf_counter()-tick
            self.retained_inputs=[];return ys
        except BaseException:
            self.failed=True;raise
        finally:
            if get_known_complete and self.pending is not None and not self.pending['release_attempted'] and self.unknown_native_operation is None:
                try:self._release_once()
                except BaseException:self.failed=True;raise
            self.active=False;times['complete_boundary_seconds']=time.perf_counter()-start;self.last_timings=times

    def known_idle(self):
        return not self.active and self.pending is None and self.unknown_native_operation is None and self.stats['run_attempts']==self.stats['runs_completed'] and self.stats['output_buffers_acquired']==self.stats['output_buffers_released']

    def close(self):
        require(threading.get_ident()==self.thread,'Close requires original calling thread')
        if self.closed:return
        require(not self.close_attempted,'No second close/destroy attempt is allowed')
        self.close_attempted=True
        require(self.known_idle(),'Unknown native work or pending outputs retain the whole owner; no close or release retry')
        if self.context.value:
            self.stats['destroy_attempts']+=1
            try:code=self.api.rknn_destroy(self.context)
            except BaseException:self.unknown_native_operation='rknn_destroy exception';raise
            if code!=0:self.unknown_native_operation='rknn_destroy returned '+str(code)
            self.b.check_call(code,'rknn_destroy single attempt');self.context.value=0;self.stats['contexts_destroyed']+=1
        self.closed=True

    def metrics(self):
        return copy.deepcopy(dict(self.stats,closed=self.closed,failed=self.failed,active=self.active,close_attempted=self.close_attempted,
            unknown_native_operation=self.unknown_native_operation,live_contexts=int(bool(self.context.value)),
            live_output_buffers=self.pending['acquired'] if self.pending else 0,
            output_release_pending_attempted=self.pending['release_attempted'] if self.pending else False,
            calling_thread_ident=self.thread,query_records=self.query_records,
            resource_counts_balanced=self.stats['contexts_created']==self.stats['contexts_destroyed'] and
            self.stats['output_buffers_acquired']==self.stats['output_buffers_released']))


def inverse_roles(np,values):
    """Independent per-head/channel inverse; no graph/exporter packing helper."""
    require(len(values)==5,'Five name-joined role buffers required');parts=[]
    for value,shape in zip(values,OUTPUT_SHAPES):
        require(list(value.shape)==shape and value.dtype in (np.float16,np.float32),'Actual direct4 role shape/type differs')
        y=np.empty((1,16,shape[1],64),dtype=np.float32)
        for h in range(16):
            for d in range(64):y[0,h,:,d]=value[h//8,:,d,h%8]
        parts.append(y)
    return [np.concatenate(parts[:3],axis=2),parts[3],parts[4]]


if __name__ == '__main__':
    print(json.dumps(dict(schema='original_QKV_single_attempt_output_owner_source',
        historical_source=HISTORICAL_SOURCE, typed_ABI=TYPED_ABI_SOURCE,
        SDK_loaded=False, artifacts_read=False, numerical_or_runtime_acceptance=False), indent=2))
