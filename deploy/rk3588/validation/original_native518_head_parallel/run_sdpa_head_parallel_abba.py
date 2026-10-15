#!/usr/bin/env python3
"""Board flags0 ABBA: original16-head SDPA versus concurrent6/5/5 heads.

Actual original post-RoPE Q/K and original V are unchanged. Every head keeps
all2082 keys/queries. Dedicated B workers own construction/run/close on cores
1/2/4. Complete timers include copies, SDK IO/release, joins and original
consumer reassembly. Numerical gates are fixed diagnostics, never relaxed.
"""
import argparse
import concurrent.futures as futures
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import threading
import time
import traceback
import types

BASE=Path('/mnt/vggt-data/vggt-slam2');E=BASE/'experiments';SRC=BASE/'src'
OWNED_SHA='0bfcb295ad5490e8196e8fcc1244e11a99b07bfb3ba7d5c350cb8c81d3ff3be4'
HELPER_SHA='56ce45635a5d010e862bf59db2f9fcc6e8ec642b2b023c8c40cad0bd5808050f'
API_HEADER_SHA='c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a'
FAILED_R1_SHA='1785466d9d60b49589fdac27ed6c3ec5ef6fea02a3a3976b65a90d7d8b6bc38d'
EXPORTER_SHA='8ebf9183d8b04cf3769f5de0e873ef912551b88ce26dac56ab66db883329573c'
DRIVER_SHA='6426825d57a9751a22ece3b74e3e50fa1ad1efa6898a36905837862ead73290c'
CAPTURE_SHA='e4ffaea2b11df9ac558c119cb4854d490b0030c02b7706eee344b8894cefa686'
CAPTURE_SOURCE_SHA='985d0c1daa7bcacd2d0ea6b184f396c35e6c9c3c500c8ee73642db001722a31c'
PARENT_SHA='c44094793c3b590c6066cbdd634cf0c4d8bd5b75987c8b491db7b17b2201f55d'
CHECKPOINT=dict(bytes=5026874952,sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')
FIXED_GATE=dict(relative_l2_max=1e-5,max_abs_max=1e-3,rtol=1e-4,atol=1e-5)
CONFIG=dict(target_platform='rk3588',float_dtype='float16',optimization_level=3,
    mean_values=None,std_values=None,quant_img_RGB2BGR=False,model_pruning=False,
    remove_reshape=False,quantize_weight=False,enable_flash_attention=False)
BUILD=dict(do_quantization=False,dataset=None,auto_hybrid=False)
PARTS=[dict(index=0,start=0,end=6,heads=6,model_key='heads6',core_mask=1),
       dict(index=1,start=6,end=11,heads=5,model_key='heads5',core_mask=2),
       dict(index=2,start=11,end=16,heads=5,model_key='heads5',core_mask=4)]
BHSD=(1,16,2082,64);PROJ=(1,2082,1024)
INPUT_NAMES=('query_rope','key_rope','value');OUTPUT_NAMES=('attention_values',)
COUNTERS=('contexts_created','contexts_destroyed','runs_completed','output_buffers_acquired','output_buffers_released')
SUCCESS='completed_original_SDPA_head_parallel_ABBA_not_quality_or_integration_accepted'


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(4<<20),b''):h.update(part)
    return h.hexdigest()


def raw_sha(value):return hashlib.sha256(value.tobytes(order='C')).hexdigest()


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result);return result


def atomic(path,row):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def logical_API_shape(semantic_shape,fmt,*,is_input):
    shape=tuple(semantic_shape)
    require(len(shape)==4 and shape[0]==1 and shape[1] in (5,6,16) and shape[2:]==(2082,64),
            'Exact original BHSD semantic shape required')
    require(fmt==1 if is_input else fmt in (0,1),
            'Inputs require observed logical NHWC; outputs require explicit logical NCHW/NHWC')
    return (shape[0],shape[2],shape[3],shape[1]) if fmt==1 else shape


def to_logical_API(np,value,fmt):
    return np.array(value.transpose(0,2,3,1) if fmt==1 else value,copy=True,order='C')


def from_logical_API(np,value,fmt):
    return np.array(value.transpose(0,3,1,2) if fmt==1 else value,copy=True,order='C')


def explicit_layout_context(parent):
    """Private format adapter; frozen owned SDK calls/releases remain unchanged.

    SDK query1/2 report logical attrs. For NCHW graph BHSD, NHWC API memory
    is BSDH. The output struct has want_float but no format override; restore
    using its exact queried logical format. Native packed queries are unused.
    """
    class ExplicitLogicalContext(parent):
        def _descriptors(self,command,shapes,names):
            b=self.base
            rows=[b.attr_record(b.query(self.api,self.context,command,b.TensorAttr(index=i))) for i in range(len(shapes))]
            require([row['index'] for row in rows]==list(range(len(shapes))) and names is not None
                and len(names)==len(rows) and len(set(names))==len(names) and {row['name'] for row in rows}==set(names),
                'Original ordered SDK descriptor indices/names required')
            rows=[next(row for row in rows if row['name']==name) for name in names]
            semantic=[tuple(shape) for shape in shapes];api_shapes=[]
            for row,shape in zip(rows,semantic):
                api_shape=logical_API_shape(shape,row['format'],is_input=command==1)
                identity=(row['type']==1 and row['quantization_type']==2 and row['zero_point']==0 and row['scale']==1.)
                require(row['shape']==list(api_shape) and row['n_elems']==math.prod(shape) and row['type'] in (0,1)
                    and (row['quantization_type']==0 or identity) and row['pass_through']==0
                    and row['size']==math.prod(shape)*(4 if row['type']==0 else 2) and math.isfinite(row['scale']),
                    'Exact logical floating NHWC/NCHW descriptor required: '+str(row))
                row['original_BHSD_shape']=list(shape);row['logical_API_shape']=list(api_shape)
                row['logical_API_layout']='NHWC_BSDH' if row['format']==1 else 'NCHW_BHSD'
                api_shapes.append(api_shape)
            if command==1:self.semantic_input_shapes=semantic;self.input_shapes=api_shapes
            else:self.semantic_output_shapes=semantic;self.output_shapes=api_shapes
            return rows
        def run(self,inputs):
            start=time.perf_counter();np=self.np
            require(len(inputs)==len(self.semantic_input_shapes),'All three original QKV inputs required')
            for value,shape in zip(inputs,self.semantic_input_shapes):
                require(isinstance(value,np.ndarray) and value.dtype==np.float32 and value.shape==shape and np.isfinite(value).all(),
                        'Finite original semantic BHSD input required')
            api_inputs=[to_logical_API(np,value,row['format']) for value,row in zip(inputs,self.inputs)]
            prepared=time.perf_counter();api_outputs=super().run(api_inputs);got=time.perf_counter()
            times=copy.deepcopy(self.last_timings)
            outputs=[from_logical_API(np,value,row['format']) for value,row in zip(api_outputs,self.outputs)]
            require(all(value.dtype==np.float32 and value.shape==shape and value.flags.c_contiguous and np.isfinite(value).all()
                for value,shape in zip(outputs,self.semantic_output_shapes)),
                'Owned logical output must restore exact original BHSD shape')
            times.update(original_BHSD_to_logical_API_owned_copy_seconds=prepared-start,
                frozen_owned_API_boundary_seconds=times['complete_boundary_seconds'],
                logical_API_to_original_BHSD_owned_copy_finite_seconds=time.perf_counter()-got,
                complete_boundary_seconds=time.perf_counter()-start)
            self.last_timings=times;return outputs
    return ExplicitLogicalContext


class DedicatedWorker:
    """One single-worker pool keeps every SDK operation on its owner thread."""
    def __init__(self,O,row,part,kwargs):
        self.owner=None;self.thread_id=None;self.run_thread_ids=[];self.close_thread_ids=[]
        self.pool=futures.ThreadPoolExecutor(max_workers=1,thread_name_prefix='original_SDPA_part'+str(part['index']))
        self.shutdown=False;self.part=copy.deepcopy(part)
        self.init_future=self.pool.submit(self._create,O,row,kwargs)
    def _create(self,O,row,kwargs):
        self.thread_id=threading.get_ident();tick=time.perf_counter()
        try:
            self.owner=O.OwnedContext(row['model_path'],row['rknn']['sha256'],input_shapes=[[1,self.part['heads'],2082,64]]*3,
                output_shapes=[[1,self.part['heads'],2082,64]],input_names=INPUT_NAMES,output_names=OUTPUT_NAMES,
                core_mask=self.part['core_mask'],flags=0,**kwargs)
        except BaseException as exc:
            self.owner=getattr(exc,'owned_context',self.owner);raise
        return dict(initialization_seconds=time.perf_counter()-tick,owner_thread_id=self.thread_id,resources=self.owner.metrics())
    def submit(self,inputs):return self.pool.submit(self._run,inputs)
    def _run(self,inputs):
        require(threading.get_ident()==self.thread_id,'Dedicated SDK owner thread changed')
        begin=time.perf_counter();ys=self.owner.run(inputs);end=time.perf_counter()
        self.run_thread_ids.append(threading.get_ident())
        require(len(ys)==1,'Complete original head-range output required')
        return ys[0],dict(timings=copy.deepcopy(self.owner.last_timings),worker_boundary_start=begin,
            worker_boundary_end=end,owner_thread_id=self.thread_id)
    def _close(self):
        require(threading.get_ident()==self.thread_id,'Close must execute on the original SDK owner thread')
        self.close_thread_ids.append(threading.get_ident())
        if self.owner is not None:self.owner.close()
    def close(self,timeout):
        if self.shutdown:return
        self.pool.submit(self._close).result(timeout=timeout)
        self.pool.shutdown(wait=True);self.shutdown=True
    def metrics(self):
        result=self.owner.metrics() if self.owner is not None else dict(closed=True,live_contexts=0,live_output_buffers=0,
            resource_counts_balanced=True,cleanup_errors=[],**{key:0 for key in COUNTERS})
        return dict(result,part=copy.deepcopy(self.part),owner_thread_id=self.thread_id,
            run_thread_ids=list(self.run_thread_ids),close_thread_ids=list(self.close_thread_ids),worker_pool_shutdown=self.shutdown,
            SDK_thread_ownership_preserved=all(i==self.thread_id for i in self.run_thread_ids+self.close_thread_ids))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('export','gate','compile','capture'):
        p.add_argument('--'+name,type=Path,required=True);p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--owned-runtime',type=Path,default=E/'vggt-v100-source-r1/native_heads_runtime.py')
    p.add_argument('--runtime-helper',type=Path,default=E/'vggt-v99-source-r1/run_native_block_r1.py')
    p.add_argument('--source-root',type=Path,default=SRC)
    p.add_argument('--library',type=Path,default=Path('/usr/lib/librknnrt.so'))
    p.add_argument('--baseline-core-mask',type=int,choices=(1,7),default=7)
    p.add_argument('--timeout-seconds',type=float,default=170.)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();sys.dont_write_bytecode=True
    require(platform.machine()=='aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),
            'Actual SDK execution is restricted to RK3588/S')
    require(0<a.timeout_seconds<=170,'Use the existing180s/75C/1GHz-restoring controller with cleanup reserve')
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);receipt=out/'result.json'
    started=time.monotonic();deadline=started+a.timeout_seconds;cleanup_deadline=started+180.
    env,affinity=dict(os.environ),os.sched_getaffinity(0);handlers,pins={},{};owners=[];worker_owners=[];active=None
    r=dict(schema='v112_original_native518_SDPA_head_parallel_ABBA',runtime_revision=2,status='running',script_sha256=sha(__file__),
        checkpoint=CHECKPOINT,original_model_only=True,component_only=True,schedule=list('ABBA'),
        init_flags=0,baseline_core_mask=a.baseline_core_mask,parts=PARTS,expected_NPU_hz=1000000000,
        original_QKV_source='Untouched original CPU global0 post-QK-normalization/post-RoPE capture; not claimed to equal full-model NPU intermediates.',
        head_shape=list(BHSD),consumer_shape=list(PROJ),phase_calls=4,hidden_warmups=0,maximum_live_contexts=3,
        fixed_gate=FIXED_GATE,explicit_CPU_attention_fallback=False,SDK_internal_operator_mapping_profiled=False,
        all2082keys_all2082queries_all16heads_preserved=True,no_per_layer_reference_reset=True,
        candidate_accepted=False,full_model_SL4_realtime_or_8GB_accepted=False,system_or_frequency_writes=False,
        timing_scope='Full component boundary: owned input copies/slices, dispatch, input conversion/set, synchronous SDK runs, outputs get/convert/owned copy/release, all worker joins, ordered head concat, original proj-input transpose/reshape/owned copy and finite checks.',
        outside_timer='Pinned model/source/receipt/array hashes, construction and teardown, SDK memory queries, platform reads, numerical comparisons and unique NPY writes.',
        SDK_thread_contract='B construction/run/close occur on one dedicated worker per part; A is owned by the main thread.',
        failed_r1_source_sha256=FAILED_R1_SHA,API_header_sha256=API_HEADER_SHA,
        logical_format_adapter='Explicit original BHSD to queried logical NHWC BSDH, and queried NCHW/NHWC output back to BHSD. No packed native format or guessed reshape.',
        output_storage_policy='Save first A/B head and consumer outputs plus every later distinct rawSHA; no unexpected distinct array is discarded.',
        phases=[],calls=[],cleanup_errors=[])
    atomic(receipt,r)

    def pin(path,digest,size=None):
        path=Path(path).resolve(strict=True)
        require(len(digest)==64 and all(c in '0123456789abcdef' for c in digest) and
            (size is None or path.stat().st_size==size) and sha(path)==digest,'Pinned original source/artifact differs: '+str(path))
        pins[path]=digest;return path

    def await_job(job,cleanup=False):
        limit=cleanup_deadline if cleanup else deadline
        remaining=limit-time.monotonic();require(remaining>0,'Bounded SDK worker deadline reached')
        return job.result(timeout=remaining)

    def observe():
        require(time.monotonic()<deadline,'Bounded head-parallel deadline reached');v=H.sample()
        require(v['npu_hz']==1000000000 and v['temperatures_c'] and max(v['temperatures_c'].values())<75.,'Clock/thermal bound failed')
        return v

    try:
        def interrupted(number,frame):raise InterruptedError('External bounded controller stopped original SDPA ABBA')
        for number in (signal.SIGTERM,signal.SIGINT):handlers[number]=signal.signal(number,interrupted)
        os.sched_setaffinity(0,{4,5,6,7});os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
        pin(__file__,r['script_sha256']);O=module(pin(a.owned_runtime,OWNED_SHA),'_v112_frozen_owned_context')
        H=module(pin(a.runtime_helper,HELPER_SHA),'_v112_frozen_normal_IO')
        IO=types.SimpleNamespace(OwnedContext=explicit_layout_context(O.OwnedContext))
        pin(a.library,O.SDK_SHA);pin(a.source_root/'deploy/rk3588/validation/run-v12-rknn-block-board.py',O.ABI_SHA)
        ep=pin(a.export,a.export_sha256);gp=pin(a.gate,a.gate_sha256);cp=pin(a.compile,a.compile_sha256)
        exported,gated,compiled=[json.loads(path.read_text()) for path in (ep,gp,cp)]
        require(exported['schema']=='v112_original_native518_SDPA_head_parts_ONNX_export'
            and exported['status']=='exported_original_SDPA_parts_not_FP32_or_NPU_accepted' and exported['script_sha256']==EXPORTER_SHA
            and exported['checkpoint']==CHECKPOINT and exported['parent']['sha256']==PARENT_SHA
            and exported['capture']['sha256']==CAPTURE_SHA and exported['parts']==PARTS
            and exported['source_and_inputs_unchanged'] and not exported['cleanup_errors']
            and all(exported['proof'][k] for k in ('original_attention_node_protobufs_unchanged',
                'original_scale_initializer_protobuf_unchanged','all18_parent_learned_parameters_match_original_capture'))
            and exported['proof']['learned_parameters_inside_SDPA']==0 and exported['proof']['complete_key_rows']==2082
            and exported['proof']['complete_query_rows']==2082 and exported['proof']['heads']==16
            and exported['proof']['head_dim']==64 and exported['proof']['scale']==.125 and exported['proof']['softmax_axis']==-1,
            'Exact original full-key independent-head extraction proof required')
        for d,schema,status in ((gated,'v112_original_native518_SDPA_head_parts_FP32_check','checked_original_SDPA_parts_not_NPU_quality_accepted'),
            (compiled,'v112_original_native518_SDPA_head_parts_FP16_compile','compiled_original_SDPA_parts_not_quality_accepted')):
            require(d['schema']==schema and d['status']==status and d['script_sha256']==DRIVER_SHA and d['parts']==PARTS
                and d['checkpoint']==CHECKPOINT and d['export_receipt']['sha256']==a.export_sha256
                and d['capture']['sha256']==CAPTURE_SHA and d['fixed_gate']==FIXED_GATE and d['original_FP32_gate_passed']
                and d['source_and_inputs_unchanged'] and d['affinity_restored'] and d['environment_restored'] and d['cwd_restored']
                and not d['cleanup_errors'] and all(v['fixed_gate_passed'] for v in d['reassembly_checks'].values())
                and [v['id'] for v in d['checks']]==['A16','part0','part1','part2']
                and all(v['comparison']['fixed_gate_passed'] for v in d['checks']),
                'Clean frozen original FP32 gate and compile linkage required')
        require(gated['mode']=='check' and gated['models']==exported['models']
            and compiled['mode']=='compile' and compiled['check_receipt']['sha256']==a.gate_sha256
            and compiled['configuration']==CONFIG and compiled['build_options']==BUILD and compiled['sdk_released']
            and len(compiled['calls'])==12 and all(v['returncode'] in (None,0) for v in compiled['calls'])
            and len(compiled['release_codes'])==3 and all(v['returncode'] in (None,0) for v in compiled['release_codes']),
            'Same unquantized FP16 graph build and all three clean releases required')
        original_models={row['key']:row for row in exported['models']};models={}
        require(len(original_models)==3 and set(original_models)=={'heads16','heads6','heads5'},'Exactly three original SDPA shapes required')
        for row in compiled['models']:
            key=row['key'];require(key in original_models and {k:v for k,v in row.items() if k!='rknn'}==original_models[key],
                'Compiled graph/bindings differ from gated original extraction')
            heads=int(key[5:]);shape=[1,heads,2082,64]
            require(row['input_bindings']==[dict(index=i,name=name,shape=shape,dtype='float32') for i,name in enumerate(INPUT_NAMES)]
                and row['output_bindings']==[dict(index=0,name='attention_values',shape=shape,dtype='float32')]
                and key not in models and Path(row['rknn']['file']).name==row['rknn']['file'],'Original IO/model identity differs')
            row=copy.deepcopy(row);row['model_path']=str(pin(cp.parent/row['rknn']['file'],row['rknn']['sha256'],row['rknn']['bytes']));models[key]=row
        require(set(models)==set(original_models),'All three compiled shapes required')
        require(a.capture_sha256==CAPTURE_SHA,'Genuine pinned original post-RoPE capture required')
        capture_path=pin(a.capture,CAPTURE_SHA);captured=json.loads(capture_path.read_text())
        require(captured['schema']=='v103_original_native518_global0_attention_capture' and captured['status']=='completed_original_native518_global0_attention_capture'
            and captured['script_sha256']==CAPTURE_SOURCE_SHA and captured['checkpoint']==CHECKPOINT and captured['original_parameters_verified']==18
            and captured['original_whole_block_comparison']['byte_equal'] and captured['original_SDPA_wrapper_forwards_unchanged']
            and captured['model_state_unchanged'] and captured['input_and_parameters_unchanged'] and captured['SDPA_function_restored']
            and captured['hooks_removed'] and captured['source_and_inputs_unchanged'] and not captured['cleanup_errors'],
            'Untouched original CPU attention reference required')
        import numpy as np
        arrays={}
        for name,shape in [('q_post_rope',BHSD),('k_post_rope',BHSD),('v',BHSD),('attention_tail',BHSD),('proj_input',PROJ)]:
            meta=captured['artifacts'][name];require(Path(meta['file']).name==meta['file'],'Direct captured array filename required')
            path=pin(capture_path.parent/meta['file'],meta['sha256'],meta['bytes']);value=np.load(path,allow_pickle=False)
            require(value.shape==shape and value.dtype==np.float32 and value.flags.c_contiguous and np.isfinite(value).all()
                and list(shape)==meta['shape'] and meta['dtype']=='float32' and value.nbytes==meta['tensor_bytes']
                and raw_sha(value)==meta['raw_sha256'] and meta['finite'],'Captured original array differs: '+name)
            arrays[name]=value
        inputs=[arrays[name] for name in ('q_post_rope','k_post_rope','v')];refs=[arrays['attention_tail'],arrays['proj_input']]
        require(np.array_equal(np.ascontiguousarray(refs[0].transpose(0,2,1,3).reshape(PROJ)),refs[1]),'Original consumer restoration differs')
        original_inputs_raw=[raw_sha(value) for value in inputs]
        r.update(export_receipt_sha256=a.export_sha256,gate_receipt_sha256=a.gate_sha256,compile_receipt_sha256=a.compile_sha256,
            capture_sha256=CAPTURE_SHA,original_parameter_preservation_proof=exported['proof'],models=models,
            original_input_raw_sha256=original_inputs_raw,original_reference_outputs=[captured['artifacts'][name] for name in ('attention_tail','proj_input')],
            original_FP32_graph_gate_checks=gated['checks'],original_FP32_reassembly_checks=gated['reassembly_checks'],
            owned_runtime_sha256=OWNED_SHA,runtime_helper_sha256=HELPER_SHA)
        kwargs=dict(source_root=a.source_root,library=a.library,runtime_helper=a.runtime_helper,timeout_ms=30000)
        unique={};first={'A':None,'B':None};steady={'A':[],'B':[]}

        def compare(actual,reference):
            z=H.comparison(np,actual,reference)
            z['fixed_gate_passed']=bool(z['strict_pass'] and z['relative_L2'] is not None
                and z['relative_L2']<=FIXED_GATE['relative_l2_max'] and z['max_abs']<=FIXED_GATE['max_abs_max'])
            return z

        def save(route,role,value):
            raw=raw_sha(value);key=(route,role,raw)
            if key in unique:return copy.deepcopy(unique[key])
            path=out/(route+'-'+role+'-unique'+str(sum(k[:2]==key[:2] for k in unique))+'.npy')
            with path.open('xb') as stream:np.save(stream,value,allow_pickle=False)
            z=dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path),raw_sha256=raw,shape=list(value.shape),
                dtype='float32',tensor_bytes=value.nbytes,finite=True,owned_copy_before_SDK_release=True)
            unique[key]=z;return copy.deepcopy(z)

        for phase_index,route in enumerate('ABBA'):
            phase=dict(index=phase_index,route=route,platform_before=observe(),calls=[]);r['phases'].append(phase);atomic(receipt,r)
            workers=[];tick=time.perf_counter()
            try:
                if route=='A':
                    row=models['heads16']
                    try:active=IO.OwnedContext(row['model_path'],row['rknn']['sha256'],input_shapes=[BHSD]*3,output_shapes=[BHSD],
                        input_names=INPUT_NAMES,output_names=OUTPUT_NAMES,core_mask=a.baseline_core_mask,flags=0,**kwargs)
                    except BaseException as exc:
                        active=getattr(exc,'owned_context',None)
                        if active is not None:owners.append(active)
                        raise
                    owners.append(active);phase['resources_after_init']=[active.metrics()]
                else:
                    for part in PARTS:
                        worker=DedicatedWorker(IO,models[part['model_key']],part,kwargs);workers.append(worker);worker_owners.append(worker)
                        await_job(worker.init_future)
                    phase['resources_after_init']=[worker.metrics() for worker in workers]
                phase['initialization_seconds']=time.perf_counter()-tick
                for ordinal in range(4):
                    call=dict(phase=phase_index,route=route,index=ordinal,recorded_first=ordinal==0,platform_before=observe())
                    begin=time.perf_counter();local={};tick=time.perf_counter()
                    if route=='A':
                        copied=[np.array(value,copy=True,order='C') for value in inputs]
                        local['owned_input_copy_seconds']=time.perf_counter()-tick
                        head=active.run(copied)[0];local['SDK_context_timings']=[copy.deepcopy(active.last_timings)]
                    else:
                        sliced=[[np.array(value[:,part['start']:part['end']],copy=True,order='C') for value in inputs] for part in PARTS]
                        local['owned_input_slice_copy_seconds']=time.perf_counter()-tick;tick=time.perf_counter()
                        jobs=[worker.submit(values) for worker,values in zip(workers,sliced)]
                        local['three_worker_dispatch_seconds']=time.perf_counter()-tick;tick=time.perf_counter()
                        responses=[await_job(job) for job in jobs]
                        local['all_worker_join_seconds']=time.perf_counter()-tick;tick=time.perf_counter()
                        head=np.concatenate([response[0] for response in responses],axis=1)
                        local['ordered_head_concat_seconds']=time.perf_counter()-tick
                        local['SDK_context_timings']=[response[1] for response in responses]
                    tick=time.perf_counter();proj=np.array(head.transpose(0,2,1,3).reshape(PROJ),copy=True,order='C')
                    require(head.shape==BHSD and head.dtype==np.float32 and head.flags.c_contiguous and np.isfinite(head).all()
                        and proj.shape==PROJ and proj.dtype==np.float32 and proj.flags.c_contiguous and np.isfinite(proj).all(),
                        'All16 owned finite heads and original consumer output required')
                    local['original_consumer_owned_reformat_and_finite_seconds']=time.perf_counter()-tick
                    elapsed=time.perf_counter()-begin;ys=[head,proj]
                    if first[route] is None:first[route]=[value.copy() for value in ys]
                    call.update(complete_boundary_seconds=elapsed,timings=local,platform_after=observe(),
                        outputs=[save(route,role,value) for role,value in zip(('attention_values','proj_input'),ys)],
                        vs_original_CPU_FP32=[compare(value,reference) for value,reference in zip(ys,refs)],
                        vs_first_actual_A_FP16=[compare(value,reference) for value,reference in zip(ys,first['A'])],
                        repeat_first_same_route=[compare(value,reference) for value,reference in zip(ys,first[route])])
                    ledgers=[active.metrics()] if route=='A' else [worker.metrics() for worker in workers]
                    require(all(not z['live_output_buffers'] for z in ledgers),'Every SDK output must be released within each full call')
                    call['SDK_output_buffers_released_before_return']=True
                    r['calls'].append(call);phase['calls'].append(len(r['calls'])-1)
                    if ordinal:steady[route].append(elapsed)
                    atomic(receipt,r);print(json.dumps(dict(stage='original_SDPA_ABBA',phase=phase_index,route=route,index=ordinal,seconds=elapsed)),flush=True)
                    del ys,head,proj
                    if route=='A':del copied
                    else:del sliced,jobs,responses
            finally:
                tick=time.perf_counter()
                if active is not None:active.close();phase['resources_after_close']=[active.metrics()];active=None
                if workers:
                    errors=[]
                    for worker in workers:
                        try:worker.close(max(.001,cleanup_deadline-time.monotonic()))
                        except BaseException as exc:errors.append(repr(exc))
                    phase['resources_after_close']=[worker.metrics() for worker in workers]
                    require(not errors,'Dedicated worker cleanup failed: '+str(errors))
                phase['destruction_and_worker_shutdown_seconds']=time.perf_counter()-tick
                atomic(receipt,r)
        require([raw_sha(value) for value in inputs]==original_inputs_raw,'Original source QKV mutated')
        r.update(status=SUCCESS,input_unchanged=True,all_same_route_repeats_byte_equal=all(all(z['byte_equal'] for z in call['repeat_first_same_route']) for call in r['calls']),
            all_actual_A_vs_B_byte_equal=all(all(z['byte_equal'] for z in call['vs_first_actual_A_FP16']) for call in r['calls'] if call['route']=='B'),
            all_original_FP32_runtime_gates_passed=all(all(z['fixed_gate_passed'] for z in call['vs_original_CPU_FP32']) for call in r['calls']),
            summary={route:dict(steady_count=len(values),mean_seconds=sum(values)/len(values),min_seconds=min(values),max_seconds=max(values)) for route,values in steady.items()},
            saved_unique_output_files=len(unique),saved_unique_output_bytes=sum(v['bytes'] for v in unique.values()))
        r['B_to_A_steady_mean_ratio']=r['summary']['B']['mean_seconds']/r['summary']['A']['mean_seconds']
    except BaseException as exc:r.update(status='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        for owner in reversed(owners):
            try:owner.close()
            except BaseException as exc:r['cleanup_errors'].append('A owner close: '+repr(exc))
        for worker in reversed(worker_owners):
            try:worker.close(max(.001,cleanup_deadline-time.monotonic()))
            except BaseException as exc:
                r['cleanup_errors'].append('B worker-owned close: '+repr(exc))
                worker.pool.shutdown(wait=False,cancel_futures=False)
        r['resources']=[owner.metrics() for owner in owners]+[worker.metrics() for worker in worker_owners]
        totals={key:sum(row[key] for row in r['resources']) for key in COUNTERS};r['SDK_totals']=totals
        r['resources_balanced']=all(row['closed'] and row['resource_counts_balanced'] and not row['live_contexts']
            and not row['live_output_buffers'] and not row['cleanup_errors'] and row.get('SDK_thread_ownership_preserved',True)
            and row.get('worker_pool_shutdown',True) for row in r['resources'])
        r['complete_expected_execution']=totals==dict(contexts_created=8,contexts_destroyed=8,runs_completed=32,
            output_buffers_acquired=32,output_buffers_released=32) and len(r['calls'])==16
        try:os.sched_setaffinity(0,affinity);r['affinity_restored']=os.sched_getaffinity(0)==affinity
        except BaseException as exc:r['cleanup_errors'].append('Affinity restore: '+repr(exc))
        os.environ.clear();os.environ.update(env);r['environment_restored']=dict(os.environ)==env
        for number,handler in handlers.items():signal.signal(number,handler)
        errors=[]
        for path,digest in pins.items():
            try:require(sha(path)==digest,'Pinned source/artifact changed: '+str(path))
            except BaseException as exc:errors.append(str(exc))
        r.update(source_and_inputs_unchanged=not errors,pin_errors=errors,source_and_input_sha256={str(path):digest for path,digest in pins.items()},
            elapsed_seconds=time.monotonic()-started,peak_process_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        if (errors or r['cleanup_errors'] or not r['resources_balanced'] or not r.get('affinity_restored') or not r['environment_restored']
            or (r['status']==SUCCESS and not r['complete_expected_execution'])):r['status']='failed_cleanup'
        atomic(receipt,r);print(json.dumps({k:r.get(k) for k in ('status','summary','B_to_A_steady_mean_ratio','cleanup_errors')}),flush=True)
    return 0 if r['status']==SUCCESS else 1


if __name__=='__main__':raise SystemExit(main())
