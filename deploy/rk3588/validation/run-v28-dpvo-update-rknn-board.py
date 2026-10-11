#!/usr/bin/env python3
"""Full Update RKNN research: mixed INT64/FP32 inputs, owned FP32 outputs."""
import argparse,ctypes as ct,hashlib,importlib.util,json,math,os,platform,resource,time,traceback
from pathlib import Path
BASE=Path('/mnt/vggt-data/vggt-slam2');EXP=BASE/'experiments'
ABI=BASE/'src/deploy/rk3588/validation/run-v12-rknn-block-board.py'
HEADER=BASE/'src/deploy/rk3588/reference/rknn_api_v2.3.2.h';LIB=Path('/usr/lib/librknnrt.so')
SDK_PINS={str(ABI):'37e4a59c2c5191e9cb75d6e6081b0760d943a36bb9f18e545890193fa5db9608',str(HEADER):'c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a',str(LIB):'d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8'}
CAPTURE_SHA='87e18137a828496d623721f7748e288d02f9d8d6879278da1e5d3d9731cfa0a1'
EXPORT_SHA='56f96b7bf80bde596de6a4008da2858a869a9f19a3162abb8db8ab1af0156cac'
COMPILER_SHA='a40bf09e53ef9b378db4781be18f9e53f824afcc809c9009e0a65f4f0f91b4e5'
def require(ok,message):
    if not ok:raise ValueError(message)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def meta(a):return dict(shape=list(a.shape),dtype=str(a.dtype),bytes=a.nbytes,tensor_sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())
def module(path,name):
    import sys
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def compare(a,b):
    import numpy as np
    require(a.shape==b.shape and a.dtype==b.dtype==np.float32 and np.isfinite(a).all() and np.isfinite(b).all(),'Matching finite FP32 required')
    d=a.astype(np.float64)-b.astype(np.float64);n=int(np.count_nonzero(np.abs(d)>1e-5+1e-4*np.abs(b.astype(np.float64))))
    return dict(rtol=1e-4,atol=1e-5,failed_elements=n,gate_passed=n==0,max_abs=float(np.abs(d).max()),relative_l2=float(np.linalg.norm(d.ravel())/max(np.linalg.norm(b.astype(np.float64).ravel()),1e-30)),bitwise_equal=a.tobytes()==b.tobytes())
def execute(api,context,abi,packing,feed,bindings,output_contracts,cleanup):
    import numpy as np
    start=time.monotonic();packed=packing.pack_inputs(abi,feed,bindings);pack_end=time.monotonic()
    outputs=(abi.Output*len(output_contracts))()
    for i in range(len(outputs)):outputs[i]=abi.Output(1,0,i,None,0)
    run=abi.RunExtend(0,0,30000,-1);received=abi.OutputExtend();got=False;owned={};r={}
    try:
        abi.check_call(api.rknn_inputs_set(context,len(feed),packed.requests),'rknn_inputs_set');set_end=time.monotonic()
        abi.check_call(api.rknn_run(context,ct.byref(run)),'rknn_run');run_end=time.monotonic()
        abi.check_call(api.rknn_outputs_get(context,len(outputs),outputs,ct.byref(received)),'rknn_outputs_get');got=True;get_end=time.monotonic()
        require(run.frame_id==received.frame_id,'Synchronous frame identity differs')
        for i,(item,key) in enumerate(zip(output_contracts,('net','delta','weight'))):
            shape=tuple(item['shape']);count=math.prod(shape)
            require(outputs[i].buf and outputs[i].size==count*4,'Unexpected owned FP32 output size')
            owned[key]=np.ctypeslib.as_array(ct.cast(outputs[i].buf,ct.POINTER(ct.c_float)),shape=(count,)).copy().reshape(shape)
        copy_end=time.monotonic()
        r.update(pack_seconds=pack_end-start,input_set_seconds=set_end-pack_end,run_api_seconds=run_end-set_end,outputs_get_seconds=get_end-run_end,owned_copy_seconds=copy_end-get_end,output_bytes=sum(int(x.size) for x in outputs),frame_id=int(run.frame_id),input_records=packed.records)
    finally:
        if got or any(x.buf for x in outputs):
            t=time.monotonic();code=int(api.rknn_outputs_release(context,len(outputs),outputs));end=time.monotonic()
            r.update(release_seconds=end-t,output_buffers_released=code==0,forward_seconds=end-start)
            if code:cleanup.append('rknn_outputs_release '+str(code))
            abi.check_call(code,'rknn_outputs_release')
    require(all(np.isfinite(x).all() for x in owned.values()),'Nonfinite SDK output')
    require(((owned['weight']>=0)&(owned['weight']<=1)).all(),'Invalid sigmoid weight range')
    return owned,r
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('capture-dir','export-dir','ort-dir','compile-dir','packing-helper','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('ort','compile','packing-helper'):p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--model-key',choices=('E96','E864'),required=True);p.add_argument('--collect-perf-detail',action='store_true');a=p.parse_args(argv)
    require(platform.machine()=='aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),'RK3588 board only')
    out=a.output_dir.resolve();require(out.is_relative_to(EXP),'Isolated output required');out.mkdir(exist_ok=False)
    prior=os.sched_getaffinity(0);pins={};context=None;api=None
    r=dict(schema='v28_DPVO_full_Update_RKNN',status='failed',script_sha256=sha(__file__),actual_board_execution=True,production_changed=False,ORB_used=False,CPU_model_fallback=False,BA_executed=False,valid_pose_output=False,realtime_validated=False,model_key=a.model_key,collect_perf_detail=a.collect_perf_detail,init_flags=8 if a.collect_perf_detail else 0,calls=[],cleanup_errors=[])
    def pin(path,expected):
        require(sha(path)==expected,'Pinned file differs: '+str(path));pins[str(path.resolve())]=expected;return path
    def artifact(folder,item):
        name=item['file'];require(Path(name).name==name,'Artifact basename required')
        path=pin(folder/name,item['sha256']);require(path.stat().st_size==item['bytes'],'Artifact bytes differ')
        with np.load(path,allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
        require({k:meta(v) for k,v in values.items()}==item['arrays'],'Artifact metadata differs');return values
    try:
        for path,digest in SDK_PINS.items():pin(Path(path),digest)
        pin(Path(__file__),r['script_sha256']);pin(a.packing_helper,a.packing_helper_sha256)
        capture=json.loads(pin(a.capture_dir/'capture.json',CAPTURE_SHA).read_text());exported=json.loads(pin(a.export_dir/'export.json',EXPORT_SHA).read_text());ort=json.loads(pin(a.ort_dir/'ort.json',a.ort_sha256).read_text());compiled=json.loads(pin(a.compile_dir/'compile.json',a.compile_sha256).read_text())
        require(capture['status']=='captured_original_Update_states_not_VO' and capture['original_weights_unchanged'] and exported['status']=='exported_Update_board_ORT_pending' and exported['exact_wrapper_FP32_gate_passed'] and exported['original_weights_unchanged'] and exported['capture_sha256']==CAPTURE_SHA,'Validated original/export chain required')
        require(ort['status']=='completed_Update_ORT_probe_not_VO' and ort['exact_original_FP32_gate_passed'] and ort['capture_sha256']==CAPTURE_SHA and ort['export_sha256']==EXPORT_SHA,'Original FP32 CPU ORT gate required')
        for receipt in (capture,exported,ort):
            require(receipt['sources_and_inputs_unchanged'] and receipt['source_pins']==receipt['source_hashes_after'] and not receipt['cleanup_errors'],'Ancestor source/cleanup verification failed')
        model=exported['models'][a.model_key]
        require(compiled['status']=='compiled_not_runtime_validated' and compiled['script_sha256']==COMPILER_SHA and compiled['input']['sha256']==model['sha256'] and compiled['released'] and not compiled.get('release_error') and compiled['configuration']==dict(target_platform='rk3588',float_dtype='float16',optimization_level=3,do_quantization=False),'Exact generic compiler/graph/config required')
        require([x['name'] for x in compiled['operations']]==['config','load_onnx','build','export_rknn'] and all(x['code'] in (None,0) for x in compiled['operations']),'Compiler stage result differs')
        model_path=pin(a.compile_dir/Path(compiled['output']['path']).name,compiled['output']['sha256']);require(model_path.stat().st_size==compiled['output']['size'],'Compiled bytes differ')
        pin(a.export_dir/model['file'],model['sha256'])
        os.sched_setaffinity(0,{4,5,6,7})
        import numpy as np
        abi=module(ABI,'_v28_full_update_abi');packing=module(a.packing_helper,'_v28_mixed_inputs')
        records=[];reference_mode=next(m for m in ort['modes'] if m['name']=='disabled')
        for case in capture['scenarios']:
            rewritten=next(x for x in exported['scenarios'] if x['name']==case['name'])
            if rewritten['model_key']!=a.model_key:continue
            topo=artifact(a.export_dir,rewritten['topology']);ocase=next(x for x in reference_mode['scenarios'] if x['name']==case['name'])
            for state in case['states']:
                inputs=artifact(a.capture_dir,state['inputs']);golden=artifact(a.capture_dir,state['outputs']);oreference=artifact(a.ort_dir,next(c for c in ocase['calls'] if c['step']==state['step'] and c['repetition']==0)['outputs'])
                records.append(dict(name=case['name'],step=state['step'],feed=dict(net=inputs['net'],inp=inputs['ctx'],corr=inputs['corr'],**topo),golden=golden,ort=oreference))
        require(len(records)==8,'Two graph scenarios with four states each required')
        api=abi.bind_api(ct.CDLL(str(LIB)));context=abi.CONTEXT();data=model_path.read_bytes();buffer=ct.create_string_buffer(data)
        begin=time.monotonic();abi.check_call(api.rknn_init(ct.byref(context),ct.cast(buffer,ct.c_void_p),len(data),r['init_flags'],None),'rknn_init');r['init_seconds']=time.monotonic()-begin
        abi.check_call(api.rknn_set_core_mask(context,7),'rknn_set_core_mask');v=abi.query(api,context,5,abi.SDKVersion());count=abi.query(api,context,0,abi.InputOutputNum())
        require(count.n_input==13 and count.n_output==3,'Full Update input/output count differs')
        attrs=[abi.attr_record(abi.query(api,context,1,abi.TensorAttr(index=i))) for i in range(13)];outputs=[abi.attr_record(abi.query(api,context,2,abi.TensorAttr(index=i))) for i in range(3)]
        bindings=packing.validate_input_attrs(model['inputs'],attrs);packing.validate_output_attrs(model['outputs'],outputs)
        mem=abi.query(api,context,6,abi.MemSize())
        r.update(capture_sha256=CAPTURE_SHA,export_sha256=EXPORT_SHA,ort_sha256=a.ort_sha256,compile_sha256=a.compile_sha256,packing_helper_sha256=a.packing_helper_sha256,model_sha256=compiled['output']['sha256'],input_attributes=attrs,output_attributes=outputs,abi=abi.abi_record(),versions=dict(api=bytes(v.api_version).decode(),driver=bytes(v.drv_version).decode()),core_mask=7,cpu_affinity=sorted(os.sched_getaffinity(0)),memory=dict(total_dma_allocated_size=int(mem.total_dma_allocated_size),total_weight_size=int(mem.total_weight_size),total_internal_size=int(mem.total_internal_size)))
        schedule=[records[0],records[3],records[4]] if a.collect_perf_detail else [records[0]]*4+records[1:]
        hashes={}
        for index,source in enumerate(schedule):
            values,timing=execute(api,context,abi,packing,source['feed'],bindings,model['outputs'],r['cleanup_errors']);metadata={k:meta(v) for k,v in values.items()};key=(source['name'],source['step'])
            if key in hashes:require(hashes[key]==metadata,'Repeated SDK output changed')
            hashes[key]=metadata;path=out/f'call-{index:02d}.npz';np.savez(path,**values)
            row=dict(index=index,case=source['name'],step=source['step'],timing_role='profile' if a.collect_perf_detail else 'cold' if index==0 else 'warm_same_state' if index<4 else 'other_state_single',timing=timing,versus_original={k:compare(x,source['golden'][k]) for k,x in values.items()},versus_ORT={k:compare(x,source['ort'][k]) for k,x in values.items()},outputs=dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path),arrays=metadata),repeat_bitwise_equal=True)
            if a.collect_perf_detail:row['profile']=abi.perf_detail(api,context)
            r['calls'].append(row);print(json.dumps(dict(index=index,case=source['name'],step=source['step'],forward_seconds=timing['forward_seconds'])),flush=True)
        r.update(status='completed_mixed_input_Update_RKNN_probe_not_VO',exact_original_FP32_gate_passed=all(x['gate_passed'] for c in r['calls'] for x in c['versus_original'].values()),timing_scope='Includes owned mixed input packing/cast and guards, SDK input set, synchronous run/get, owned FP32 output copies and release. Metadata/reference checks, disk I/O, initialization and topology generation excluded. Profile is a separate context/run.')
        if not a.collect_perf_detail:r['first_state_warm_mean_seconds']=sum(x['timing']['forward_seconds'] for x in r['calls'][1:4])/3
    except Exception as e:r.update(status='failed',error=repr(e),traceback=traceback.format_exc())
    finally:
        if context is not None and context.value:
            try:
                code=int(api.rknn_destroy(context));r['destroy_code']=code
                if code:r['cleanup_errors'].append('rknn_destroy '+str(code));r['status']='failed'
            except Exception as e:r['cleanup_errors'].append(repr(e));r['status']='failed'
        try:
            r['source_hashes_after']={path:sha(path) for path in pins};r['sources_and_inputs_unchanged']=r['source_hashes_after']==pins;require(r['sources_and_inputs_unchanged'],'Source or input changed')
            os.sched_setaffinity(0,prior);r['affinity_restored']=set(os.sched_getaffinity(0))==set(prior)
        except Exception as e:r['cleanup_errors'].append(repr(e));r['status']='failed'
        r.update(source_pins=pins,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024);(out/'run.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:r.get(k) for k in ('status','error','exact_original_FP32_gate_passed','first_state_warm_mean_seconds','cleanup_errors','peak_rss_mib')}),flush=True)
    return 0 if r['status']=='completed_mixed_input_Update_RKNN_probe_not_VO' else 1
if __name__=='__main__':raise SystemExit(main())
