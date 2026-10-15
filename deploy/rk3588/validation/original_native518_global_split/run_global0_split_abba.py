#!/usr/bin/env python3
"""Board-only complete global0 lifecycle ABBA; genuine prefix→SDPA→suffix.

All SDK construction/hash/IO/layout/release/destruction and worker shutdown
are timed for both routes. No reference QKV/attention is substituted. Only
one weighted context may live at once; head parts have no learned weights.
"""
import argparse
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
import time
import traceback
import types

BASE=Path('/mnt/vggt-data/vggt-slam2');E=BASE/'experiments';SRC=BASE/'src'
HEAD_RUNTIME_SHA='e74ad2969551f90ef7d6f6eabebd3a1aa67bbb4dd819738f161a37189ee429f6'
OWNED_SHA='0bfcb295ad5490e8196e8fcc1244e11a99b07bfb3ba7d5c350cb8c81d3ff3be4'
HELPER_SHA='56ce45635a5d010e862bf59db2f9fcc6e8ec642b2b023c8c40cad0bd5808050f'
EXPORT_SHA='0b1bb2df87b6b115e43677625adcdf9a5b9233a6cb3921591fbe6909881fa81e'
SPLIT_DRIVER_SHA='990b4e97cdfc1552285f08df1a1154b9ff2ae8a9a63190a7c939cc1e7eae3178'
BASE_COMPILE_SHA='711ee6dcc45134b4695ee8b24ec47ee0c7f43356b9ce047c09051e1ed108a33c'
BASE_MODEL_SHA='8f9ca55e5f4f4b8646b6a5dbf041a590c168b0e1e6d390e4ce9d594dddd5654d'
HEAD_COMPILE_SHA='27ce62e53665801a821efe9cf47ba8359cbcce0171cb7d0c2fc142f935807e0a'
HEAD_EXPORT_SHA='e09acb577228c66d3bbae9a14ce5f22115f8c1f4ea7b128cafbbc61709dcaa6e'
REFERENCE_SHA='4ebbc4adaaed5e3c3cf8558b4947ec3fc090d56e82f8c61c9509951eb34dd35d'
CAPTURE_SHA='e4ffaea2b11df9ac558c119cb4854d490b0030c02b7706eee344b8894cefa686'
PARENT_SHA='c44094793c3b590c6066cbdd634cf0c4d8bd5b75987c8b491db7b17b2201f55d'
CHECKPOINT=dict(bytes=5026874952,sha256='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0')
GLOBAL=(1,2082,1024);BHSD=(1,16,2082,64)
GATE=dict(relative_l2_max=1e-5,max_abs_max=1e-3,rtol=1e-4,atol=1e-5)
CONFIG=dict(target_platform='rk3588',float_dtype='float16',optimization_level=3,mean_values=None,std_values=None,
    quant_img_RGB2BGR=False,model_pruning=False,remove_reshape=False,quantize_weight=False,enable_flash_attention=False)
BUILD=dict(do_quantization=False,dataset=None,auto_hybrid=False)
COUNTERS=('contexts_created','contexts_destroyed','runs_completed','output_buffers_acquired','output_buffers_released')
SUCCESS='completed_original_global0_split_ABBA_not_quality_or_integration_accepted'


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(4<<20),b''):h.update(part)
    return h.hexdigest()


def raw_sha(value):return hashlib.sha256(value.tobytes(order='C')).hexdigest()


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def atomic(path,row):
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def mixed_rank_context(parent):
    """Exact logical rank3 identity and rank4 NCHW/NHWC axis conversion."""
    class MixedLogicalContext(parent):
        def _descriptors(self,command,shapes,names):
            b=self.base;rows=[b.attr_record(b.query(self.api,self.context,command,b.TensorAttr(index=i))) for i in range(len(shapes))]
            require([r['index'] for r in rows]==list(range(len(shapes))) and names is not None and len(names)==len(rows) and len(set(names))==len(rows)
                and {r['name'] for r in rows}==set(names),'Exact logical descriptor index/name mapping required')
            rows=[next(r for r in rows if r['name']==name) for name in names];semantic=[tuple(shape) for shape in shapes];api_shapes=[]
            for row,shape in zip(rows,semantic):
                if len(shape)==3:
                    require(shape==GLOBAL and row['format'] in (0,3),'Exact rank3 original logical layout required');api_shape=shape
                else:
                    require(len(shape)==4 and shape[0]==1 and shape[1] in (5,6,16) and shape[2:]==(2082,64)
                        and row['format'] in (0,1),'Exact rank4 NCHW/NHWC logical format required')
                    api_shape=(1,2082,64,shape[1]) if row['format']==1 else shape
                identity=row['type']==1 and row['quantization_type']==2 and row['zero_point']==0 and row['scale']==1.
                require(row['shape']==list(api_shape) and row['n_elems']==math.prod(shape) and row['type'] in (0,1)
                    and (row['quantization_type']==0 or identity) and row['pass_through']==0
                    and row['size']==math.prod(shape)*(4 if row['type']==0 else 2) and math.isfinite(row['scale']),
                    'Exact original floating logical descriptor required: '+str(row))
                row['original_semantic_shape']=list(shape);row['logical_API_shape']=list(api_shape);api_shapes.append(api_shape)
            if command==1:self.semantic_inputs=semantic;self.input_shapes=api_shapes
            else:self.semantic_outputs=semantic;self.output_shapes=api_shapes
            return rows
        def run(self,inputs):
            start=time.perf_counter();np=self.np
            require(len(inputs)==len(self.semantic_inputs),'All original stage inputs required')
            for value,shape in zip(inputs,self.semantic_inputs):
                require(isinstance(value,np.ndarray) and value.dtype==np.float32 and value.shape==shape and np.isfinite(value).all(),
                        'Finite unchanged original semantic input required')
            values=[np.array(value.transpose(0,2,3,1) if value.ndim==4 and row['format']==1 else value,copy=True,order='C')
                for value,row in zip(inputs,self.inputs)]
            prepared=time.perf_counter();native=super().run(values);got=time.perf_counter();times=copy.deepcopy(self.last_timings)
            result=[np.array(value.transpose(0,3,1,2) if value.ndim==4 and row['format']==1 else value,copy=True,order='C')
                for value,row in zip(native,self.outputs)]
            require(all(value.shape==shape and value.dtype==np.float32 and value.flags.c_contiguous and np.isfinite(value).all()
                for value,shape in zip(result,self.semantic_outputs)),'Complete owned original stage outputs required')
            times.update(semantic_to_logical_API_owned_copy_seconds=prepared-start,
                frozen_owned_API_boundary_seconds=times['complete_boundary_seconds'],
                logical_API_to_semantic_owned_copy_finite_seconds=time.perf_counter()-got,
                complete_boundary_seconds=time.perf_counter()-start)
            self.last_timings=times;return result
    return MixedLogicalContext


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('split-export','split-gate','split-compile'):
        p.add_argument('--'+name,type=Path,required=True);p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--baseline-compile',type=Path,default=E/'vggt-v98-global-fc2k4-compile-r1/result.json')
    p.add_argument('--head-compile',type=Path,default=E/'vggt-v112-compile-r1/result.json')
    p.add_argument('--native-reference',type=Path,default=E/'vggt-v98-reference-r1/capture.json')
    p.add_argument('--capture',type=Path,default=E/'vggt-v103-global-attention-capture-r1/capture.json')
    p.add_argument('--head-runtime',type=Path,default=E/'vggt-v112-source-r1/run_sdpa_head_parallel_abba.py')
    p.add_argument('--owned-runtime',type=Path,default=E/'vggt-v100-source-r1/native_heads_runtime.py')
    p.add_argument('--runtime-helper',type=Path,default=E/'vggt-v99-source-r1/run_native_block_r1.py')
    p.add_argument('--source-root',type=Path,default=SRC);p.add_argument('--library',type=Path,default=Path('/usr/lib/librknnrt.so'))
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--timeout-seconds',type=float,default=170.)
    a=p.parse_args();sys.dont_write_bytecode=True
    require(platform.machine()=='aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),'Actual SDK execution is RK3588/S only')
    require(0<a.timeout_seconds<=170,'Existing180s/75C/1GHz controller with cleanup reserve required')
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);receipt=out/'result.json'
    started=time.monotonic();deadline=started+a.timeout_seconds;cleanup_deadline=started+180.
    env,affinity=dict(os.environ),os.sched_getaffinity(0);handlers,pins={},{};owners=[];worker_owners=[]
    r=dict(schema='v114_original_native518_complete_global0_split_ABBA',status='running',script_sha256=sha(__file__),
        original_model_only=True,component_only=True,checkpoint=CHECKPOINT,parent_graph_sha256=PARENT_SHA,schedule=list('ABBA'),
        phase_calls=2,total_calls=8,init_flags=0,weighted_core_mask=7,expected_NPU_hz=1000000000,
        maximum_live_weighted_contexts=1,maximum_live_total_contexts=3,no_reference_QKV_attention_or_suffix_input_reset=True,
        suffix_residual_is_original_input=True,CPU_attention_fallback=False,compiled_internal_mapping_profiled=False,
        candidate_accepted=False,fullmodel_SL4_trajectory_realtime_accepted=False,system_or_frequency_writes=False,fixed_gate=GATE,
        timing_scope='Complete fresh-context block lifecycle: constructor checks/model/helper/library hashes, init/query, input/residual copies and residual-invariant hash, all SDK IO/layout/owned-copy/release, B worker dispatch/join/headconcat, all destroys/thread shutdown and finite checks.',
        outside_timer='Startup provenance/capture verification, platform reads, numerical comparisons, saved-output/repeat raw hashes and unique output writes.',
        sample_policy='Report first invocation of each route separately; summary includes remaining3 route calls. Every call initializes/destroys all contexts, no resident-context or hidden-warmup claim.',
        hash_accounting_scope='Logical file bytes consumed by frozen constructor hashing calls; not physical storage reads, DDR traffic or cold-page-cache measurement.',
        phases=[],calls=[],cleanup_errors=[])
    atomic(receipt,r)

    def pin(path,digest,size=None):
        path=Path(path).resolve(strict=True)
        require(len(digest)==64 and all(c in '0123456789abcdef' for c in digest) and
            (size is None or path.stat().st_size==size) and sha(path)==digest,'Pinned original artifact differs: '+str(path))
        pins[path]=digest;return path

    def await_job(job,cleanup=False):
        remaining=(cleanup_deadline if cleanup else deadline)-time.monotonic();require(remaining>0,'Bounded stage deadline reached')
        return job.result(timeout=remaining)

    def observe():
        require(time.monotonic()<deadline,'Complete block diagnostic deadline reached');z=H.sample()
        require(z['npu_hz']==1000000000 and z['temperatures_c'] and max(z['temperatures_c'].values())<75.,'Clock/thermal bound failed');return z

    try:
        def interrupted(n,frame):raise InterruptedError('External bounded controller stopped global0 lifecycle ABBA')
        for number in (signal.SIGTERM,signal.SIGINT):handlers[number]=signal.signal(number,interrupted)
        os.sched_setaffinity(0,{4,5,6,7});os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
        pin(__file__,r['script_sha256']);HP=module(pin(a.head_runtime,HEAD_RUNTIME_SHA),'_v114_frozen_head_workers')
        O=module(pin(a.owned_runtime,OWNED_SHA),'_v114_frozen_owned_IO');H=module(pin(a.runtime_helper,HELPER_SHA),'_v114_frozen_normal_utilities')
        Mixed=mixed_rank_context(O.OwnedContext);PartIO=types.SimpleNamespace(OwnedContext=HP.explicit_layout_context(O.OwnedContext))
        lib=pin(a.library,O.SDK_SHA);abi=pin(a.source_root/'deploy/rk3588/validation/run-v12-rknn-block-board.py',O.ABI_SHA)
        ep=pin(a.split_export,a.split_export_sha256);gp=pin(a.split_gate,a.split_gate_sha256);cp=pin(a.split_compile,a.split_compile_sha256)
        exported,gated,compiled=[json.loads(path.read_text()) for path in (ep,gp,cp)]
        require(a.split_export_sha256==EXPORT_SHA and exported['schema']=='v114_original_native518_global0_prefix_suffix_ONNX_export'
            and exported['status']=='exported_original_global0_exact_cuts_not_FP32_or_NPU_accepted' and exported['checkpoint']==CHECKPOINT
            and exported['parent']['sha256']==PARENT_SHA and exported['capture']['sha256']==CAPTURE_SHA
            and exported['head_export']['sha256']==HEAD_EXPORT_SHA and exported['source_and_inputs_unchanged'] and not exported['cleanup_errors'],
            'Exact actual53-node/44-initializer/18-parameter original export required')
        proof=exported['proof'];require(all(proof[k] for k in ('original_node_union_exactly_once','whole_parent_reassembled_protobuf_equal',
            'full_initializer_union_including_original_SDPA_protobuf_equal','all18_original_learned_parameters_verified',
            'original_residual_input_preserved_in_suffix','original_suffix_transpose_projection_norm_GELU_FC2K4_and_residuals_preserved')),
            'Original full block/residual/suffix preservation proof required')
        for d,schema,status in ((gated,'v114_original_native518_global0_split_FP32_check','checked_original_global0_split_not_NPU_quality_accepted'),
            (compiled,'v114_original_native518_global0_split_FP16_compile','compiled_original_global0_split_not_quality_accepted')):
            require(d['schema']==schema and d['status']==status and d['script_sha256']==SPLIT_DRIVER_SHA and d['export_receipt']['sha256']==EXPORT_SHA
                and d['capture']['sha256']==CAPTURE_SHA and d['head_export']['sha256']==HEAD_EXPORT_SHA
                and d['fixed_gate']==GATE and d['original_FP32_gate_passed'] and d['source_and_inputs_unchanged']
                and d['affinity_restored'] and d['environment_restored'] and d['cwd_restored'] and not d['cleanup_errors']
                and all(v['fixed_gate_passed'] for v in d['checks'].values())
                and len(d['head_part_checks'])==3 and all(v['comparison']['fixed_gate_passed'] for v in d['head_part_checks']),
                'Linked actual complete original split FP32 gate/compile required')
        require(gated['mode']=='check' and compiled['mode']=='compile' and gated['models']==exported['models']
            and compiled['check_receipt']['sha256']==a.split_gate_sha256 and compiled['configuration']==CONFIG
            and compiled['build_options']==BUILD and compiled['sdk_released']
            and len(compiled['calls'])==8 and all(v['returncode'] in (None,0) for v in compiled['calls'])
            and len(compiled['release_codes'])==2 and all(v['returncode'] in (None,0) for v in compiled['release_codes']),
            'Same frozen nonquantized FP16 builder/gate and clean SDK releases required')
        baseline_path=pin(a.baseline_compile,BASE_COMPILE_SHA);baseline=json.loads(baseline_path.read_text())
        require(baseline['schema']=='original_native518_single_block_FP16_compile' and baseline['status']=='compiled_not_quality_accepted'
            and baseline['onnx']['sha256']==PARENT_SHA and baseline['rknn']['sha256']==BASE_MODEL_SHA and baseline['sdk_released']
            and baseline['configuration']==CONFIG and baseline['build_options']==BUILD and not baseline['cleanup_errors'],
            'Exact original complete global0 baseline required')
        head_path=pin(a.head_compile,HEAD_COMPILE_SHA);heads=json.loads(head_path.read_text())
        require(heads['schema']=='v112_original_native518_SDPA_head_parts_FP16_compile'
            and heads['status']=='compiled_original_SDPA_parts_not_quality_accepted' and heads['configuration']==CONFIG
            and heads['build_options']==BUILD and heads['original_FP32_gate_passed'] and heads['sdk_released'] and not heads['cleanup_errors'],
            'Exact already-tested complete-key head graphs required')
        def model_rows(receipt_path,rows):
            result={}
            for row in rows:
                require(row['key'] not in result and Path(row['rknn']['file']).name==row['rknn']['file'],'Direct distinct compiled model files required')
                value=copy.deepcopy(row);value['model_path']=str(pin(receipt_path.parent/row['rknn']['file'],row['rknn']['sha256'],row['rknn']['bytes']));result[row['key']]=value
            return result
        cuts=model_rows(cp,compiled['models']);head_models=model_rows(head_path,heads['models'])
        originals={row['key']:row for row in exported['models']}
        require(set(cuts)=={'prefix','suffix'} and all({k:v for k,v in row.items() if k not in ('rknn','model_path')}==originals[key]
            for key,row in cuts.items()),'Compiled exact-cut graph/bindings differ')
        baseline_model=pin(baseline_path.parent/baseline['rknn']['file'],BASE_MODEL_SHA,29798710)
        expected=[('prefix',['input'],[GLOBAL],['query_rope','key_rope','value'],[BHSD]*3),
            ('suffix',['input','attention_values'],[GLOBAL,BHSD],['output'],[GLOBAL])]
        for key,inn,ins,outn,outs in expected:
            require(cuts[key]['input_bindings']==[dict(index=i,name=name,shape=list(shape),dtype='float32') for i,(name,shape) in enumerate(zip(inn,ins))]
                and cuts[key]['output_bindings']==[dict(index=i,name=name,shape=list(shape),dtype='float32') for i,(name,shape) in enumerate(zip(outn,outs))],
                'Exact complete original cut ABI required')
        ref_path=pin(a.native_reference,REFERENCE_SHA);native=json.loads(ref_path.read_text());capture=json.loads(pin(a.capture,CAPTURE_SHA).read_text())
        require(native['status']=='completed_original_native518_block_references' and native['checkpoint']==CHECKPOINT
            and capture['original_whole_block_comparison']['byte_equal'] and capture['original_native_reference_sha256']==REFERENCE_SHA,
            'Actual untouched original native input/full-output reference required')
        import numpy as np
        def load(meta):
            require(Path(meta['file']).name==meta['file'],'Direct native capture file required');path=pin(ref_path.parent/meta['file'],meta['sha256'],meta['bytes'])
            value=np.load(path,allow_pickle=False);require(value.shape==GLOBAL and value.dtype==np.float32 and np.isfinite(value).all()
                and value.flags.c_contiguous and raw_sha(value)==meta['raw_sha256'],'Original native capture tensor differs');return value
        x=load(native['artifacts']['global0_input']);reference=load(native['artifacts']['global0_output']);input_raw=raw_sha(x)
        r.update(split_export_sha256=EXPORT_SHA,split_gate_sha256=a.split_gate_sha256,split_compile_sha256=a.split_compile_sha256,
            baseline_compile_sha256=BASE_COMPILE_SHA,head_compile_sha256=HEAD_COMPILE_SHA,original_reference_sha256=REFERENCE_SHA,
            original_input=native['artifacts']['global0_input'],original_output_reference=native['artifacts']['global0_output'],
            original_parameter_preservation_proof=proof,original_input_raw_sha256=input_raw,cut_models=cuts,head_models=head_models)
        kwargs=dict(source_root=a.source_root,library=a.library,runtime_helper=a.runtime_helper,timeout_ms=30000)
        hash_fixed=a.runtime_helper.stat().st_size+2*abi.stat().st_size+lib.stat().st_size
        r['constructor_hash_files']=dict(runtime_helper=dict(bytes=a.runtime_helper.stat().st_size,hash_calls=1),
            typed_ABI=dict(bytes=abi.stat().st_size,hash_calls=2),runtime_library=dict(bytes=lib.stat().st_size,hash_calls=1),model_hash_calls=1)

        def construct(row,owner_list):
            try:
                owner=Mixed(row['model_path'],row['rknn']['sha256'],input_shapes=[v['shape'] for v in row['input_bindings']],
                    output_shapes=[v['shape'] for v in row['output_bindings']],input_names=[v['name'] for v in row['input_bindings']],
                    output_names=[v['name'] for v in row['output_bindings']],core_mask=7,flags=0,**kwargs)
            except BaseException as exc:
                owner=getattr(exc,'owned_context',None)
                if owner is not None:owner_list.append(owner);owners.append(owner)
                raise
            owner_list.append(owner);owners.append(owner);return owner
        base_row=dict(baseline,model_path=str(baseline_model))

        def lifecycle(route):
            local=[];workers=[];times={};ctx=None;constructor_bytes=0;result=None
            try:
                tick=time.perf_counter();residual=np.array(x,copy=True,order='C');times['original_input_residual_owned_copy_seconds']=time.perf_counter()-tick
                if route=='A':
                    tick=time.perf_counter();ctx=construct(base_row,local);times['complete_block_initialization_seconds']=time.perf_counter()-tick
                    constructor_bytes+=hash_fixed+base_row['rknn']['bytes'];result=ctx.run([residual])[0];times['complete_block_IO']=copy.deepcopy(ctx.last_timings)
                    tick=time.perf_counter();ctx.close();times['complete_block_destruction_seconds']=time.perf_counter()-tick;ctx=None
                else:
                    tick=time.perf_counter();ctx=construct(cuts['prefix'],local);times['prefix_initialization_seconds']=time.perf_counter()-tick
                    constructor_bytes+=hash_fixed+cuts['prefix']['rknn']['bytes'];qkv=ctx.run([residual]);times['prefix_IO']=copy.deepcopy(ctx.last_timings)
                    require(len(qkv)==3 and all(y.shape==BHSD for y in qkv),'All three actual NPU prefix outputs required')
                    tick=time.perf_counter();ctx.close();times['prefix_destruction_seconds']=time.perf_counter()-tick;ctx=None
                    require(all(owner.metrics()['closed'] for owner in local),'Prefix must close before head contexts')
                    tick=time.perf_counter()
                    for part in HP.PARTS:
                        worker=HP.DedicatedWorker(PartIO,head_models[part['model_key']],part,kwargs);workers.append(worker);worker_owners.append(worker)
                        await_job(worker.init_future);constructor_bytes+=hash_fixed+head_models[part['model_key']]['rknn']['bytes']
                    times['head_worker_initialization_seconds']=time.perf_counter()-tick;tick=time.perf_counter()
                    slices=[[np.array(value[:,part['start']:part['end']],copy=True,order='C') for value in qkv] for part in HP.PARTS]
                    times['actual_prefix_QKV_slice_copy_seconds']=time.perf_counter()-tick;tick=time.perf_counter()
                    jobs=[worker.submit(values) for worker,values in zip(workers,slices)];responses=[await_job(job) for job in jobs]
                    times['head_dispatch_and_join_seconds']=time.perf_counter()-tick;times['head_IO']=[value[1] for value in responses]
                    tick=time.perf_counter();attention=np.concatenate([value[0] for value in responses],axis=1)
                    require(attention.shape==BHSD and attention.dtype==np.float32 and np.isfinite(attention).all(),'Complete actual16-head attention required')
                    times['actual_attention_concat_and_finite_seconds']=time.perf_counter()-tick;tick=time.perf_counter()
                    errors=[]
                    for worker in workers:
                        try:worker.close(max(.001,cleanup_deadline-time.monotonic()))
                        except BaseException as exc:errors.append(repr(exc))
                    require(not errors,'Part context close/join failed: '+str(errors));times['head_worker_destruction_and_shutdown_seconds']=time.perf_counter()-tick
                    require(all(worker.metrics()['closed'] and worker.shutdown for worker in workers),'All unweighted heads must close before suffix')
                    del qkv,slices,jobs,responses
                    tick=time.perf_counter();ctx=construct(cuts['suffix'],local);times['suffix_initialization_seconds']=time.perf_counter()-tick
                    constructor_bytes+=hash_fixed+cuts['suffix']['rknn']['bytes'];result=ctx.run([residual,attention])[0];times['suffix_IO']=copy.deepcopy(ctx.last_timings)
                    tick=time.perf_counter();ctx.close();times['suffix_destruction_seconds']=time.perf_counter()-tick;ctx=None
                require(raw_sha(residual)==input_raw and result.shape==GLOBAL and result.dtype==np.float32 and result.flags.c_contiguous
                    and np.isfinite(result).all(),'Original residual and complete owned finite block output required')
                return result,dict(stage_timings=times,logical_constructor_hashed_bytes=constructor_bytes,
                    weighted_contexts=[owner.metrics() for owner in local],head_contexts=[worker.metrics() for worker in workers])
            finally:
                for owner in reversed(local):owner.close()
                errors=[]
                for worker in workers:
                    try:worker.close(max(.001,cleanup_deadline-time.monotonic()))
                    except BaseException as exc:errors.append(repr(exc))
                require(not errors,'Final dedicated worker cleanup incomplete: '+str(errors))

        first={};steady={'A':[],'B':[]};unique={};route_ordinals={'A':0,'B':0}
        def compare(actual,expected):
            z=H.comparison(np,actual,expected);z['fixed_gate_passed']=bool(z['strict_pass'] and z['relative_L2'] is not None
                and z['relative_L2']<=1e-5 and z['max_abs']<=1e-3);return z
        def save(route,value):
            raw=raw_sha(value);key=(route,raw)
            if key not in unique:
                path=out/(route+'-output-unique'+str(sum(k[0]==route for k in unique))+'.npy')
                with path.open('xb') as stream:np.save(stream,value,allow_pickle=False)
                unique[key]=dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path),raw_sha256=raw,
                    shape=list(value.shape),dtype='float32',tensor_bytes=value.nbytes,finite=True)
            return copy.deepcopy(unique[key])
        for phase_index,route in enumerate('ABBA'):
            phase=dict(index=phase_index,route=route,calls=[]);r['phases'].append(phase)
            for ordinal in range(2):
                call=dict(phase=phase_index,index=ordinal,route=route,route_ordinal=route_ordinals[route],
                    first_invocation_of_route=route_ordinals[route]==0,platform_before=observe())
                tick=time.perf_counter();value,detail=lifecycle(route);elapsed=time.perf_counter()-tick
                if route not in first:first[route]=value.copy()
                call.update(complete_lifecycle_boundary_seconds=elapsed,execution=detail,output=save(route,value),platform_after=observe(),
                    vs_original_CPU_FP32=compare(value,reference),vs_first_actual_A_FP16=compare(value,first['A']),repeat_same_route=compare(value,first[route]))
                if route_ordinals[route]:steady[route].append(elapsed)
                route_ordinals[route]+=1;r['calls'].append(call);phase['calls'].append(len(r['calls'])-1);atomic(receipt,r)
                print(json.dumps(dict(stage='complete_global0_lifecycle_ABBA',phase=phase_index,route=route,index=ordinal,seconds=elapsed)),flush=True)
        require(raw_sha(x)==input_raw,'Original native input mutated')
        r.update(status=SUCCESS,input_unchanged=True,all_same_route_repeats_byte_equal=all(c['repeat_same_route']['byte_equal'] for c in r['calls']),
            all_actual_A_B_byte_equal=all(c['vs_first_actual_A_FP16']['byte_equal'] for c in r['calls'] if c['route']=='B'),
            all_original_FP32_runtime_gates_passed=all(c['vs_original_CPU_FP32']['fixed_gate_passed'] for c in r['calls']),
            summary={route:dict(subsequent_lifecycle_count=len(v),mean_seconds=sum(v)/len(v),min_seconds=min(v),max_seconds=max(v)) for route,v in steady.items()},
            constructor_hash_summary={route:dict(calls=4,logical_hashed_bytes=sum(c['execution']['logical_constructor_hashed_bytes'] for c in r['calls'] if c['route']==route),
                per_call_logical_bytes=[c['execution']['logical_constructor_hashed_bytes'] for c in r['calls'] if c['route']==route]) for route in 'AB'},
            saved_unique_output_files=len(unique),saved_unique_output_bytes=sum(v['bytes'] for v in unique.values()))
        r['B_to_A_subsequent_mean_ratio']=r['summary']['B']['mean_seconds']/r['summary']['A']['mean_seconds']
    except BaseException as exc:r.update(status='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        for owner in reversed(owners):
            try:owner.close()
            except BaseException as exc:r['cleanup_errors'].append('Weighted owner close: '+repr(exc))
        for worker in reversed(worker_owners):
            try:worker.close(max(.001,cleanup_deadline-time.monotonic()))
            except BaseException as exc:r['cleanup_errors'].append('Worker close: '+repr(exc));worker.pool.shutdown(wait=False,cancel_futures=False)
        r['resources']=[owner.metrics() for owner in owners]+[worker.metrics() for worker in worker_owners]
        totals={key:sum(row[key] for row in r['resources']) for key in COUNTERS};r['SDK_totals']=totals
        r['resources_balanced']=all(row['closed'] and row['resource_counts_balanced'] and not row['live_contexts'] and not row['live_output_buffers']
            and not row['cleanup_errors'] and row.get('worker_pool_shutdown',True) and row.get('SDK_thread_ownership_preserved',True) for row in r['resources'])
        r['complete_expected_execution']=totals==dict(contexts_created=24,contexts_destroyed=24,runs_completed=24,output_buffers_acquired=32,output_buffers_released=32) and len(r['calls'])==8
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
        atomic(receipt,r);print(json.dumps({key:r.get(key) for key in ('status','summary','B_to_A_subsequent_mean_ratio','cleanup_errors')}),flush=True)
    return 0 if r['status']==SUCCESS else 1


if __name__=='__main__':raise SystemExit(main())
