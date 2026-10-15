#!/usr/bin/env python3
"""Board-only original global0 split FP32 gate or noncalibrating FP16 compile.

The original complete parent is checked first, then prefix QKV, all three
full-key head parts and the residual-preserving suffix. Compilation uses only
a pinned successful gate; it imports no ORT and performs no inference/calibration.
"""
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import signal
import time
import traceback
import types

EXPORTER_SHA='ac3d9e8a7f3ff2d628f843a374c9eba7c5ab6a35a99e8465a39f84cb491d486c'
HEAD_EXPORTER_SHA='8ebf9183d8b04cf3769f5de0e873ef912551b88ce26dac56ab66db883329573c'
HEAD_DRIVER_SHA='6426825d57a9751a22ece3b74e3e50fa1ad1efa6898a36905837862ead73290c'
PARENT_COMPILER_SHA='ade8f932c267bdc2a9ff5854e305d87456984630025aa34c0c448499f5a6a81b'
HEAD_EXPORT_SHA='e09acb577228c66d3bbae9a14ce5f22115f8c1f4ea7b128cafbbc61709dcaa6e'
CHECK_SCHEMA='v114_original_native518_global0_split_FP32_check'
COMPILE_SCHEMA='v114_original_native518_global0_split_FP16_compile'
CHECK_SUCCESS='checked_original_global0_split_not_NPU_quality_accepted'
COMPILE_SUCCESS='compiled_original_global0_split_not_quality_accepted'
EXPECTED_CONFIG=dict(target_platform='rk3588',float_dtype='float16',optimization_level=3,
    mean_values=None,std_values=None,quant_img_RGB2BGR=False,model_pruning=False,
    remove_reshape=False,quantize_weight=False,enable_flash_attention=False)
BUILD=dict(do_quantization=False,dataset=None,auto_hybrid=False)
THREAD_ENV=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','CUDA_VISIBLE_DEVICES','TMPDIR')


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
    return h.hexdigest()


def atomic(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('check','compile'),required=True)
    p.add_argument('--export',type=Path,required=True);p.add_argument('--export-sha256',required=True)
    p.add_argument('--exporter-source',type=Path,required=True);p.add_argument('--exporter-sha256',required=True)
    p.add_argument('--head-exporter-source',type=Path,required=True);p.add_argument('--head-exporter-sha256',required=True)
    p.add_argument('--head-driver-source',type=Path,required=True);p.add_argument('--head-driver-sha256',required=True)
    p.add_argument('--head-export',type=Path,required=True);p.add_argument('--head-export-sha256',required=True)
    p.add_argument('--capture',type=Path,required=True);p.add_argument('--capture-sha256',required=True)
    p.add_argument('--global-input',type=Path,required=True);p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--check-receipt',type=Path);p.add_argument('--check-sha256')
    p.add_argument('--parent-compiler-source',type=Path);p.add_argument('--parent-compiler-sha256')
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--cpu-affinity',type=int,nargs='+',default=[4,6])
    a=p.parse_args()
    require(platform.machine()=='aarch64' and b'rockchip,rk3588' in Path('/proc/device-tree/compatible').read_bytes(),
        'All ORT/build execution in this driver is restricted to RK3588/S')
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    watched={str(Path(__file__).resolve()):sha(__file__)}
    r=dict(schema=CHECK_SCHEMA if a.mode=='check' else COMPILE_SCHEMA,mode=a.mode,status='failed',
        script_sha256=watched[str(Path(__file__).resolve())],original_model_only=True,component_only=True,
        candidate_accepted=False,full_model_quality_realtime_or_8GB_accepted=False,GPU_execution=False,
        inference_execution=a.mode=='check',SDK_build_only=a.mode=='compile',system_or_frequency_writes=False,
        cleanup_errors=[],fixed_gate=dict(relative_l2_max=1e-5,max_abs_max=1e-3,rtol=1e-4,atol=1e-5),
        caller_gate_timeout_seconds=120,caller_compile_timeout_seconds=180,
        gate_peak_memory_estimate_MiB=1536,estimate_is_not_measured_or_a_memory_guarantee=True,
        saved_output_budget_bytes=48*1024*1024,estimated_saved_output_bytes=5*(8527872+128))
    affinity,oldcwd=os.sched_getaffinity(0),os.getcwd();environment={k:os.environ.get(k) for k in THREAD_ENV}
    handlers={};sdk=None;began=time.perf_counter()
    def pin(path,digest,size=None):
        path=Path(path).resolve(strict=True)
        require(len(digest)==64 and sha(path)==digest and (size is None or path.stat().st_size==size),'Pinned file differs: '+str(path))
        watched[str(path)]=digest;return path
    def source(path,digest,name):
        path=pin(path,digest);m=types.ModuleType(name);m.__file__=str(path)
        exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m
    def interrupted(number,frame):raise InterruptedError('External bounded controller signal '+str(number))
    try:
        require(a.exporter_sha256==EXPORTER_SHA and a.head_exporter_sha256==HEAD_EXPORTER_SHA and
            a.head_driver_sha256==HEAD_DRIVER_SHA and a.head_export_sha256==HEAD_EXPORT_SHA,'Frozen source/head export required')
        S=source(a.exporter_source,EXPORTER_SHA,'v114_static_splitter')
        E=source(a.head_exporter_source,HEAD_EXPORTER_SHA,'v114_original_capture_and_head_contract')
        D=source(a.head_driver_source,HEAD_DRIVER_SHA,'v114_frozen_original_comparison')
        ep=pin(a.export,a.export_sha256);e=json.loads(ep.read_text())
        hp=pin(a.head_export,HEAD_EXPORT_SHA);he=json.loads(hp.read_text())
        require(e['schema']==S.SCHEMA and e['status']==S.SUCCESS and e['script_sha256']==EXPORTER_SHA and
            e['checkpoint']==E.CHECKPOINT and e['source_and_inputs_unchanged'] and not e['cleanup_errors'] and
            e['parent']['sha256']==S.PARENT_SHA and e['head_export']['sha256']==HEAD_EXPORT_SHA and
            e['parts']==E.PARTS and e['proof']['original_residual_input_preserved_in_suffix'] and
            e['proof']['whole_parent_reassembled_protobuf_equal'] and e['proof']['all18_original_learned_parameters_verified'],
            'Exact original residual-preserving graph cuts required')
        require(he['schema']==E.SCHEMA and he['status']==E.SUCCESS and he['script_sha256']==HEAD_EXPORTER_SHA and
            he['parts']==E.PARTS and he['source_and_inputs_unchanged'] and not he['cleanup_errors'], 'Original head graph source required')
        require(a.capture_sha256==E.CAPTURE_SHA,'Genuine original native capture required')
        cp,c=E.capture_receipt(a.capture);watched[str(cp)]=E.CAPTURE_SHA
        require(e['capture']['sha256']==E.CAPTURE_SHA and e['original_parameter_metadata']==c['parameters'], 'Original learned-value lineage differs')
        for name,digest in c['source_hashes'].items():
            if name.startswith('third_party/vggt/'):pin(a.source_root/name,digest)
        parent_path=pin(e['parent']['file'],S.PARENT_SHA,S.PARENT_BYTES)
        import onnx
        parent=onnx.load_model(parent_path)
        regenerated,rows,proof=S.split_graphs(parent,c,E)
        require(proof==e['proof'],'Actual partition proof does not recompute')
        graph_rows={row['key']:row for row in e['models']};graphs={}
        require(set(graph_rows)=={'prefix','suffix'},'Exactly two weighted cuts required')
        for row in rows:
            actual=graph_rows[row['key']]
            require({k:v for k,v in actual.items() if k not in ('file','bytes','sha256')}==row and
                Path(actual['file']).name==actual['file'],'Original cut IO/node identity differs')
            file=pin(ep.parent/actual['file'],actual['sha256'],actual['bytes'])
            require(regenerated[row['key']].SerializeToString()==file.read_bytes(),'Actual cut graph is not the reconstructed exact original')
            graphs[row['key']]=file
        del regenerated,parent;gc.collect()
        head_graphs={}
        for row in he['models']:
            require(row['key'] in ('heads16','heads6','heads5') and Path(row['file']).name==row['file'],'Head graph identity differs')
            head_graphs[row['key']]=pin(hp.parent/row['file'],row['sha256'],row['bytes'])
        require(set(head_graphs)=={'heads16','heads6','heads5'},'All original head shapes required')
        arrays,ap=E.load_capture_arrays(cp,c);watched.update(ap)
        import numpy as np
        input_meta=c['input_and_position_artifacts']['global0_input'];ip=pin(a.global_input,input_meta['sha256'],input_meta['bytes'])
        x=np.load(ip,allow_pickle=False)
        require(x.dtype==np.float32 and list(x.shape)==input_meta['shape']==list(S.FRAME) and np.isfinite(x).all() and
            hashlib.sha256(x.tobytes()).hexdigest()==input_meta['raw_sha256'],'Actual original native global0 input differs')
        meta=c['artifacts']['block_output'];ref_path=pin(cp.parent/meta['file'],meta['sha256'],meta['bytes'])
        reference=np.load(ref_path,allow_pickle=False)
        require(reference.dtype==np.float32 and list(reference.shape)==meta['shape']==list(S.FRAME) and np.isfinite(reference).all() and
            hashlib.sha256(reference.tobytes()).hexdigest()==meta['raw_sha256'],'Original full block output reference differs')
        require(a.cpu_affinity and set(a.cpu_affinity).issubset({4,5,6,7}),'A76-local affinity required')
        os.sched_setaffinity(0,set(a.cpu_affinity));os.chdir(out)
        for name in THREAD_ENV:os.environ[name]='' if name=='CUDA_VISIBLE_DEVICES' else (str(out) if name=='TMPDIR' else '1')
        for number in (signal.SIGTERM,signal.SIGINT):handlers[number]=signal.signal(number,interrupted)
        r.update(export_receipt=dict(file=str(ep),sha256=a.export_sha256),exporter_source_sha256=EXPORTER_SHA,
            capture=dict(file=str(cp),sha256=E.CAPTURE_SHA),head_export=dict(file=str(hp),sha256=HEAD_EXPORT_SHA),
            head_exporter_source_sha256=HEAD_EXPORTER_SHA,head_driver_source_sha256=HEAD_DRIVER_SHA,
            parent=e['parent'],global_input=dict(file=str(ip),**{k:v for k,v in input_meta.items() if k!='file'}),
            checkpoint=E.CHECKPOINT,parts=e['parts'],proof=e['proof'],original_SDPA_arguments=c['SDPA_arguments'],
            original_source_hashes=c['source_hashes'],models=e['models'])
        def checked(name,value,ref):
            comparison=D.compare(value,ref)
            require(comparison['fixed_gate_passed'],'Unchanged original FP32 gate failed: '+name)
            return comparison
        if a.mode=='check':
            import onnxruntime as ort
            options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
            options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            options.enable_cpu_mem_arena=False;options.enable_mem_pattern=False
            def run(file,inputs,outputs):
                session=ort.InferenceSession(str(file),sess_options=options,providers=['CPUExecutionProvider'])
                session.disable_fallback()
                require(session.get_providers()==['CPUExecutionProvider'] and [v.name for v in session.get_inputs()]==list(inputs) and
                    [v.name for v in session.get_outputs()]==list(outputs),'Original static graph IO/providing order differs')
                result=[np.array(v,copy=True,order='C') for v in session.run(list(outputs),inputs)]
                del session;gc.collect();return result
            saved={};checks={};saved_bytes=0
            def save(name,value):
                nonlocal saved_bytes
                require(value.dtype==np.float32 and value.flags.c_contiguous and np.isfinite(value).all(),'Owned finite FP32 output required')
                file=out/(name+'.npy');np.save(file,value,allow_pickle=False);saved_bytes+=file.stat().st_size
                require(saved_bytes<=r['saved_output_budget_bytes'],'Bounded saved output budget exceeded')
                saved[name]=dict(file=file.name,bytes=file.stat().st_size,sha256=sha(file),shape=list(value.shape),dtype='float32',
                    raw_sha256=hashlib.sha256(value.tobytes()).hexdigest())
            baseline=run(parent_path,{'input':x},('output',))[0]
            checks['parent_vs_original']=checked('parent_vs_original',baseline,reference);save('parent_output',baseline)
            qkv=run(graphs['prefix'],{'input':x},S.PREFIX_OUTPUTS)
            for name,value,refname in zip(S.PREFIX_OUTPUTS,qkv,('q_post_rope','k_post_rope','v')):
                checks['prefix_'+name+'_vs_original']=checked('prefix_'+name,value,arrays[refname]);save('prefix_'+name,value)
            part_outputs=[];part_checks=[]
            for part in E.PARTS:
                inputs={name:np.ascontiguousarray(value[:,part['start']:part['end']]) for name,value in zip(E.INPUT_NAMES,qkv)}
                value=run(head_graphs[part['model_key']],inputs,('attention_values',))[0]
                part_checks.append(dict(index=part['index'],model_key=part['model_key'],start=part['start'],end=part['end'],
                    comparison=checked('headpart'+str(part['index']),value,np.ascontiguousarray(arrays['attention_tail'][:,part['start']:part['end']]))))
                part_outputs.append(value)
            attention=np.ascontiguousarray(np.concatenate(part_outputs,axis=1));del part_outputs
            checks['joined_attention_vs_original']=checked('joined_attention',attention,arrays['attention_tail'])
            proj=np.ascontiguousarray(attention.transpose(0,2,1,3).reshape(1,2082,1024))
            checks['joined_consumer_vs_original']=checked('joined_consumer',proj,arrays['proj_input']);del proj
            composed=run(graphs['suffix'],{'input':x,'attention_values':attention},('output',))[0]
            checks['complete_composed_vs_original']=checked('complete_composed',composed,reference)
            checks['complete_composed_vs_parent']=checked('complete_composed_vs_parent',composed,baseline);save('composed_output',composed)
            r.update(status=CHECK_SUCCESS,original_FP32_gate_passed=True,checks=checks,head_part_checks=part_checks,
                artifacts=saved,saved_output_bytes=saved_bytes,SDK_released_not_applicable=True,
                versions=dict(onnxruntime=ort.__version__,numpy=np.__version__,onnx=onnx.__version__))
        else:
            require(a.check_receipt is not None and a.check_sha256 and a.parent_compiler_source is not None and
                a.parent_compiler_sha256==PARENT_COMPILER_SHA,'Pinned prior board FP32 gate and original compiler required')
            gp=pin(a.check_receipt,a.check_sha256);gate=json.loads(gp.read_text())
            require(gate['schema']==CHECK_SCHEMA and gate['mode']=='check' and gate['status']==CHECK_SUCCESS and
                gate['script_sha256']==r['script_sha256'] and gate['export_receipt']['sha256']==a.export_sha256 and
                gate['capture']['sha256']==E.CAPTURE_SHA and gate['head_export']['sha256']==HEAD_EXPORT_SHA and
                gate['global_input']==r['global_input'] and gate['models']==e['models'] and gate['fixed_gate']==D.FIXED_GATE and
                gate['original_FP32_gate_passed'] and gate['source_and_inputs_unchanged'] and not gate['cleanup_errors'] and
                gate['affinity_restored'] and gate['environment_restored'] and gate['cwd_restored'] and
                all(v['fixed_gate_passed'] for v in gate['checks'].values()) and
                [v['index'] for v in gate['head_part_checks']]==[0,1,2] and all(v['comparison']['fixed_gate_passed'] for v in gate['head_part_checks']),
                'Clean same-source complete original FP32 gate required before compiler import')
            expected=('parent_output','prefix_query_rope','prefix_key_rope','prefix_value','composed_output')
            require(set(gate['artifacts'])==set(expected),'Exactly five bounded FP32 outputs required')
            saved={}
            for name in expected:
                row=gate['artifacts'][name];require(Path(row['file']).name==row['file'],'Direct saved graph output required')
                file=pin(gp.parent/row['file'],row['sha256'],row['bytes']);value=np.load(file,allow_pickle=False)
                require(value.dtype==np.float32 and list(value.shape)==row['shape'] and np.isfinite(value).all() and
                    hashlib.sha256(value.tobytes()).hexdigest()==row['raw_sha256'],'Saved FP32 output identity differs')
                saved[name]=value
            require(checked('parent',saved['parent_output'],reference)==gate['checks']['parent_vs_original'] and
                checked('composed',saved['composed_output'],reference)==gate['checks']['complete_composed_vs_original'] and
                checked('composed_parent',saved['composed_output'],saved['parent_output'])==gate['checks']['complete_composed_vs_parent'],
                'Saved full-output numerical gates do not recompute')
            for name,refname in zip(S.PREFIX_OUTPUTS,('q_post_rope','k_post_rope','v')):
                require(checked(name,saved['prefix_'+name],arrays[refname])==gate['checks']['prefix_'+name+'_vs_original'],
                    'Saved prefix QKV gate does not recompute')
            H=source(a.parent_compiler_source,PARENT_COMPILER_SHA,'v114_frozen_build_configuration')
            require(H.CONFIG==EXPECTED_CONFIG,'Original nonquantizing compiler configuration differs')
            require(importlib.metadata.version('rknn-toolkit2').split('+')[0].startswith('2.3.2'),'Frozen toolkit2.3.2 required')
            r.update(check_receipt=dict(file=str(gp),sha256=a.check_sha256),original_FP32_gate_passed=True,
                checks=gate['checks'],head_part_checks=gate['head_part_checks'],FP32_artifacts_owned_by_check_directory=gate['artifacts'],
                parent_compiler_source_sha256=PARENT_COMPILER_SHA,configuration=copy.deepcopy(H.CONFIG),build_options=BUILD,
                calls=[],models=[],release_codes=[])
            from rknn.api import RKNN
            for key,file in graphs.items():
                sdk=RKNN(verbose=False,verbose_file=str(out/(key+'-sdk.log')));target=out/(key+'.rknn')
                try:
                    for operation,fn in [('config',lambda:sdk.config(**H.CONFIG)),('load_onnx',lambda:sdk.load_onnx(model=str(file))),
                        ('build',lambda:sdk.build(**BUILD)),('export_rknn',lambda:sdk.export_rknn(str(target)))]:
                        tick=time.perf_counter();code=fn();r['calls'].append(dict(model_key=key,operation=operation,returncode=code,seconds=time.perf_counter()-tick))
                        atomic(out/'result.json',r);require(code in (None,0),key+' '+operation+' failed')
                    row=copy.deepcopy(graph_rows[key]);row['rknn']=dict(file=target.name,bytes=target.stat().st_size,sha256=sha(target));r['models'].append(row)
                finally:
                    code=sdk.release();r['release_codes'].append(dict(model_key=key,returncode=code));require(code in (None,0),'SDK release failed');sdk=None
            r.update(status=COMPILE_SUCCESS,sdk_released=True,versions={name:importlib.metadata.version(name) for name in ('rknn-toolkit2','onnx','numpy')})
    except BaseException as error:r.update(status='failed',error=repr(error),traceback=traceback.format_exc())
    finally:
        if sdk is not None:
            try:require(sdk.release() in (None,0),'Retry SDK release failed')
            except BaseException as error:r['cleanup_errors'].append('SDK release: '+repr(error))
        try:
            os.chdir(oldcwd);r['cwd_restored']=os.getcwd()==oldcwd
            os.sched_setaffinity(0,affinity);r['affinity_restored']=os.sched_getaffinity(0)==affinity
            for name,value in environment.items():
                if value is None:os.environ.pop(name,None)
                else:os.environ[name]=value
            r['environment_restored']=all(os.environ.get(name)==value for name,value in environment.items())
            for number,handler in handlers.items():signal.signal(number,handler)
        except BaseException as error:r['cleanup_errors'].append('Process restore: '+repr(error))
        pin_errors=[]
        for file,digest in watched.items():
            try:require(sha(file)==digest,'Pinned original file changed: '+file)
            except BaseException as error:pin_errors.append(str(error))
        r['source_and_inputs_unchanged']=not pin_errors;r['cleanup_errors'].extend(pin_errors)
        if r['cleanup_errors']:r['status']='failed_cleanup'
        r.update(elapsed_seconds=time.perf_counter()-began,peak_process_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        atomic(out/'result.json',r)
    print(json.dumps({k:r.get(k) for k in ('mode','status','models','error')}),flush=True)
    return 0 if r['status'] in (CHECK_SUCCESS,COMPILE_SUCCESS) else 1


if __name__=='__main__':raise SystemExit(main())
