#!/usr/bin/env python3
"""RK3588-only original/spatial-norm RKNN encoder diagnostic comparison.

Historical FP32 rejection is retained. Ordinary AB/BA/AB timing and separate
candidate profile calls are research, never accepted VO or realtime. Uses the
frozen v26 ABI, IO ownership, original input and reference validation helpers.
"""
import argparse,ctypes as ct,hashlib,importlib.util,json,os,platform,sys,time,traceback
from pathlib import Path
import numpy as np

BASE=Path('/mnt/vggt-data/vggt-slam2/experiments')
PARENT_SHA='895ac2902a66c68048c03e98f353300c7065e7568047479db4fee2f9a80a33d3'
REWRITE_SHA='66eb8d135165f7538c97733d0c83198fa5a674e0cd2bdf64eb190c24819c5109'
FAILED_ORT_SHA='a517f822a2e4afd01aff87df303183a3dc5850313134548891dc13c1e36b896b'
CANDIDATE_ONNX_SHA='c3c4048c6d8792b09f0d6cbec8228f99d59c60d7503456eea2e369d7b8dbd826'

def require(ok,why):
    if not ok:raise ValueError(why)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(1<<20),b''):h.update(x)
    return h.hexdigest()
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def board_guard():
    require(platform.system()=='Linux' and platform.machine()=='aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),'Actual model execution is RK3588 board-only')
def schedule(profile):
    if profile:return [(i,'spatial','profile') for i in range(3)]
    return [(0,'original','first'),(0,'spatial','first'),(0,'spatial','warm'),(0,'original','warm'),
            (0,'original','warm'),(0,'spatial','warm'),(1,'spatial','other_frame'),(1,'original','other_frame'),
            (2,'original','other_frame'),(2,'spatial','other_frame')]
def timing_summary(calls):
    out={}
    for method in sorted({r['method'] for r in calls}):
        rows=[r for r in calls if r['method']==method]
        warm=[r['forward_seconds'] for r in rows if r['timing_role']=='warm']
        out[method]=dict(calls=len(rows),warm_seconds=warm,warm_mean_seconds=float(np.mean(warm)) if warm else None,
                         first_seconds=next((r['forward_seconds'] for r in rows if r['timing_role']=='first'),None))
    return out

class Context:
    def __init__(self,base,abi,api,model,flags,cleanup,record):
        self.base=base;self.abi=abi;self.api=api;self.context=abi.CONTEXT();self.cleanup=cleanup;self.record=record;self.closed=False
        try:
            filename=ct.create_string_buffer(os.fsencode(model));start=time.perf_counter()
            abi.check_call(api.rknn_init(ct.byref(self.context),ct.cast(filename,ct.c_void_p),0,flags,None),'rknn_init')
            record['init_seconds']=time.perf_counter()-start
            abi.check_call(api.rknn_set_core_mask(self.context,7),'rknn_set_core_mask')
            version=abi.query(api,self.context,5,abi.SDKVersion())
            record.update(sdk_version=bytes(version.api_version).decode(errors='replace'),driver_version=bytes(version.drv_version).decode(errors='replace'),core_mask=7,init_flags=flags)
            require(record['sdk_version'].startswith('2.3.2'),'Pinned runtime2.3.2 required')
            counts=abi.query(api,self.context,0,abi.InputOutputNum());require(counts.n_input==1 and counts.n_output==2,'Original one input/two outputs required')
            input_attr=abi.query(api,self.context,1,abi.TensorAttr(index=0))
            record['input_attr']=base.validate_attribute(abi,input_attr,0,'normalized_bgr',base.INPUT_SHAPE,is_input=True)
            record['output_attrs']=[base.validate_attribute(abi,abi.query(api,self.context,2,abi.TensorAttr(index=i)),i,name,shape,is_input=False) for i,(name,shape) in enumerate(base.OUTPUTS)]
            self.native_format=int(input_attr.fmt)
            memory=abi.query(api,self.context,6,abi.MemSize());record['model_memory_bytes']={name:int(getattr(memory,name)) for name,_ in memory._fields_ if name!='reserved'}
            require(memory.total_dma_allocated_size<=1024**3,'Per-context DMA exceeds1GiB')
        except Exception:
            self.close();raise
    def execute(self,images,activity):
        require(not self.closed,'Context closed')
        return self.base.execute_call(self.api,self.abi,self.context,images,self.native_format,60000,self.cleanup,activity)
    def profile(self):return self.abi.perf_detail(self.api,self.context)
    def close(self):
        if self.closed:return
        self.closed=True
        if self.context.value:
            try:self.abi.check_call(self.api.rknn_destroy(self.context),'rknn_destroy');self.record['context_destroyed']=True
            except Exception as e:self.record['context_destroyed']=False;self.cleanup.append('rknn_destroy: '+repr(e))
        else:self.record['context_destroyed']=None

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent-source','compiler-source','capture-dir','export-dir','ort-dir','compile-dir','candidate-compile-dir','rewrite-dir','failed-ort-dir','abi-helper','header','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('capture','export','ort','compile','candidate-compile','compiler-source'):p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--library',type=Path,default=Path('/usr/lib/librknnrt.so'))
    p.add_argument('--collect-perf-detail',action='store_true')
    a=p.parse_args(argv);board_guard()
    out=a.output_dir.resolve();require(out.is_relative_to(BASE) and out!=BASE,'New isolated board experiment required')
    protected=[v.resolve() for k,v in vars(a).items() if k!='output_dir' and isinstance(v,Path)]
    require(not any(out==v or out.is_relative_to(v) or v.is_relative_to(out) for v in protected),'Output overlaps source/input')
    out.mkdir(exist_ok=False);started=time.monotonic();affinity=os.sched_getaffinity(0);pins={};contexts={};abi=None
    report=dict(schema='v32_DPVO_encoder_RKNN_layout_diagnostic',status='failed',script_sha256=sha(__file__),actual_inference_on_RK3588S=True,
                historical_FP32_gate_passed=False,FP32_thresholds_relaxed=False,output_clipping=False,candidate_accepted=False,
                original_VGGT_default_unchanged=True,different_model_from_VGGT=True,ORB_used=False,added_LK_PnP_frontend=False,
                encoder_only=True,NPU_used=False,all_operators_on_npu_verified=False,complete_VO_executed=False,realtime_validated=False,trajectory_accuracy_validated=False,
                collect_perf_detail=a.collect_perf_detail,calls=[],contexts={},cleanup_errors=[],sdk_run_api_calls=0,
                ordinary_schedule='Frame0 AB/BA/AB, then frame1 BA and frame2 AB; A original, B spatial. Every call includes packing/set/run/get/copied outputs/release.',
                scope='Fixed three original TUM frames, unchanged parameters. No preprocessing, Update, correlation, BA or trajectory. Separate profile process; no clock/governor/cooling control.')
    def pin(path,digest):
        path=Path(path).resolve();require(sha(path)==digest,'SHA mismatch: '+path.name);pins[str(path)]=digest;return path
    def artifact(name,arrays):
        path=out/name
        with path.open('xb') as f:np.savez(f,**arrays)
        return dict(file=name,bytes=path.stat().st_size,sha256=sha(path))
    try:
        pin(__file__,report['script_sha256']);base=module(pin(a.parent_source,PARENT_SHA),'_v32_original_RKNN_runtime')
        compiler=module(pin(a.compiler_source,a.compiler_source_sha256),'_v32_diagnostic_compiler_contract')
        abi=base.load_abi(a.abi_helper,pins);abi.require_board();pin(a.header,base.HEADER_SHA);pin(a.library,base.LIBRARY_SHA)
        require({4,5,6,7}<=affinity,'Four A76 CPUs required');os.sched_setaffinity(0,{4,5,6,7});report['cpu_affinity']=sorted(os.sched_getaffinity(0))
        original,captured,exported,compiled,rows=base.load_inputs(a,pins)
        rewrite=json.loads(pin(a.rewrite_dir/'rewrite.json',REWRITE_SHA).read_text())
        failed=json.loads(pin(a.failed_ort_dir/'ort.json',FAILED_ORT_SHA).read_text())
        ancestor=compiler.validate_lineage(rewrite,failed)
        candidate_receipt=json.loads(pin(a.candidate_compile_dir/'compile.json',a.candidate_compile_sha256).read_text())
        compiler.validate_compilation(candidate_receipt,ancestor,expected_script_sha256=a.compiler_source_sha256)
        candidate_onnx=pin(a.rewrite_dir/'encoders.onnx',CANDIDATE_ONNX_SHA)
        require(candidate_receipt['input']['sha256']==sha(candidate_onnx),'Compiled rewritten ONNX differs')
        item=candidate_receipt['output'];candidate=pin(a.candidate_compile_dir/Path(item['path']).name,item['sha256'])
        require(candidate.stat().st_size==item['size'] and 0<candidate.stat().st_size<=128*1024**2,'Candidate RKNN size differs')
        report.update(preprocessing=captured['preprocessing'],inputs=[row['seed'] for row in rows],original_parameters=captured['state_before'],
                      original_loader_provenance=captured['loader'],candidate_compile_sha256=a.candidate_compile_sha256,
                      candidate_ancestor=ancestor,configuration=candidate_receipt['configuration'],memory_before_sdk=abi.process_memory(),numpy_version=np.__version__)
        api=abi.bind_api(ct.CDLL(str(a.library.resolve())))
        methods={'spatial':candidate} if a.collect_perf_detail else {'original':original,'spatial':candidate}
        for method,path in methods.items():
            record=dict(model_sha256=sha(path),model_bytes=path.stat().st_size);report['contexts'][method]=record
            contexts[method]=Context(base,abi,api,path,8 if a.collect_perf_detail else 0,report['cleanup_errors'],record)
        fingerprints={}
        for index,(frame,method,role) in enumerate(schedule(a.collect_perf_detail)):
            row=rows[frame];actual,timing=contexts[method].execute(row['images'],report)
            report['NPU_used']=True
            canonical={name:value[:,None] for name,value in actual.items()}
            arrays={name:base.metadata(value) for name,value in canonical.items()}
            key=(frame,method);valid=all(np.isfinite(x).all() and np.any(x!=0) for x in canonical.values())
            record=dict(index=index,method=method,frame_index=frame,frame_id=row['seed']['frame_id'],timing_role=role,
                        input_array=row['seed']['array'],timing=timing,forward_seconds=timing['sdk_io_seconds'],arrays=arrays,finite_nonzero=bool(valid))
            report['calls'].append(record)
            if key not in fingerprints or not valid:
                record['artifact']=artifact('call-%02d-%s-frame-%d.npz'%(index,method,frame),canonical)
            record['repeat_raw_hash_equal']=None if key not in fingerprints else arrays==fingerprints[key]
            fingerprints.setdefault(key,arrays)
            base.canonical_outputs(actual)  # Existing finite/shape/nonzero checks remain fatal.
            record['versus_CPU_reference']={name:base.comparison(value,row['reference'][name]) for name,value in canonical.items()}
            record['versus_CPU_ORT']={name:base.comparison(value,row['ort_reference'][name]) for name,value in canonical.items()}
            if a.collect_perf_detail:record['perf_detail']=contexts[method].profile()
            print(json.dumps(dict(event='encoder_call_completed',index=index,method=method,frame=frame,seconds=record['forward_seconds'],strict_FP32=all(x['gate_passed'] for x in record['versus_CPU_reference'].values()))),flush=True)
        require(len(report['calls'])==len(schedule(a.collect_perf_detail)),'Prescribed call schedule incomplete')
        report.update(status='completed_encoder_diagnostic_FP32_rejection_retained_not_VO',timing=timing_summary(report['calls']),
                      actual_strict_FP32_gate_passed=all(x['gate_passed'] for r in report['calls'] for x in r['versus_CPU_reference'].values()))
    except Exception as e:report.update(error=repr(e),traceback=traceback.format_exc())
    finally:
        for context in contexts.values():context.close()
        try:
            after={path:sha(path) for path in pins};report.update(source_pins=pins,source_hashes_after=after,sources_and_inputs_unchanged=after==pins)
            require(after==pins,'Source/artifact changed')
        except Exception as e:report['cleanup_errors'].append('source recheck: '+repr(e))
        try:os.sched_setaffinity(0,affinity);require(os.sched_getaffinity(0)==affinity,'Affinity restore failed');report['affinity_restored']=True
        except Exception as e:report['cleanup_errors'].append('affinity: '+repr(e))
        if abi is not None:report['memory_final']=abi.process_memory()
        if report['cleanup_errors']:report['status']='failed'
        report['actual_driver_seconds']=time.monotonic()-started
        (out/'run.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report.get(k) for k in ('status','error','actual_driver_seconds','timing','cleanup_errors')}),flush=True)
    return 0 if report['status']=='completed_encoder_diagnostic_FP32_rejection_retained_not_VO' else 1
if __name__=='__main__':raise SystemExit(main())
